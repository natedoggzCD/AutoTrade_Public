"""
Hourly Backtest Runner - Fast hourly-level strategy testing using DuckDB.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
import math

import duckdb
import pandas as pd

from autotrade.data_ingestion.paths import get_ingestion_paths
from autotrade.backtesting.duckdb_backtester import BacktestResult

logger = logging.getLogger('AutoTrade.HourlyBacktestRunner')

def _scoring_profile_sql(profile: str) -> str:
    """Return a SQL expression for signal scoring based on strategy archetype."""
    profile = str(profile or "default").lower()
    if profile == "momentum":
        return (
            "COALESCE(weekly_return, 0) * 1.5"
            " + COALESCE(roc_5, 0) * 0.5"
            " + COALESCE(vol_ratio, 0) * 2.0"
        )
    elif profile == "mean_reversion":
        return (
            "(100.0 - COALESCE(rsi, 50)) * 0.3"
            " + COALESCE(vol_ratio, 1.0) * 1.5"
        )
    elif profile == "breakout":
        return (
            "COALESCE(vol_ratio, 0) * 3.0"
            " + COALESCE(weekly_return, 0) * 0.8"
        )
    else:  # default
        return (
            "COALESCE(weekly_return, 0)"
            " + COALESCE(vol_ratio, 0) * 2.0"
        )

class HourlyBacktestRunner:
    """
    Runs backtests on hourly bars using DuckDB for speed.
    """
    
    def __init__(self, parquet_path: Optional[Path] = None):
        paths = get_ingestion_paths()
        self.parquet_path = parquet_path or paths.hourly_prices_parquet
        self.conn = duckdb.connect(database=':memory:')
        
        if not self.parquet_path.exists():
            logger.warning(f"Hourly prices parquet not found at {self.parquet_path}")

    def backtest_hourly_setup(
        self,
        signal_criteria: dict,
        start_date: str,
        end_date: str,
        hold_hours: int = 24,
        top_n: int = 5
    ) -> BacktestResult:
        """
        Run an hourly-level backtest.
        """
        if not self.parquet_path.exists():
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        _rt_cost_pct = float(signal_criteria.get("_round_trip_cost_pct", 0.11))
        
        # 1. Build WHERE clause
        filter_sql = ""
        if "rsi_min" in signal_criteria:
            filter_sql += f" AND rsi >= {float(signal_criteria['rsi_min'])}"
        if "rsi_max" in signal_criteria:
            filter_sql += f" AND rsi <= {float(signal_criteria['rsi_max'])}"
        
        vol_ratio_thresh = float(signal_criteria.get("min_volume_ratio", 1.0))
        filter_sql += f" AND vol_ratio >= {vol_ratio_thresh}"

        query = f"""
        WITH raw_data AS (
            SELECT 
                ticker,
                CAST(Date AS TIMESTAMP) as date,
                Open as open,
                High as high,
                Low as low,
                Close as close,
                Volume as volume
            FROM read_parquet('{self.parquet_path.as_posix()}')
            WHERE CAST(Date AS DATE) >= '{start_date}' 
              AND CAST(Date AS DATE) <= '{end_date}'
        ),
        with_indicators AS (
            SELECT *,
                -- Simplified hourly indicators
                100.0 - (100.0 / (1.0 + NULLIF(
                    AVG(CASE WHEN close > LAG(close) OVER (PARTITION BY ticker ORDER BY date) THEN close - LAG(close) OVER (PARTITION BY ticker ORDER BY date) ELSE 0 END) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 14 PRECEDING AND CURRENT ROW)
                    / NULLIF(AVG(CASE WHEN close < LAG(close) OVER (PARTITION BY ticker ORDER BY date) THEN LAG(close) OVER (PARTITION BY ticker ORDER BY date) - close ELSE 0 END) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 14 PRECEDING AND CURRENT ROW), 0)
                , 0))) as rsi,
                volume / NULLIF(AVG(volume) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING), 0) as vol_ratio,
                (close - LAG(close, 7) OVER (PARTITION BY ticker ORDER BY date)) / NULLIF(LAG(close, 7) OVER (PARTITION BY ticker ORDER BY date), 0) * 100 as weekly_return
            FROM raw_data
        ),
        signals AS (
            SELECT *,
                {_scoring_profile_sql(signal_criteria.get("scoring_profile", "default"))} as score
            FROM with_indicators
            WHERE TRUE {filter_sql}
        ),
        ranked_signals AS (
            SELECT *,
                ROW_NUMBER() OVER (PARTITION BY date ORDER BY score DESC) as rank
            FROM signals
        ),
        trades AS (
            SELECT 
                s.ticker,
                s.date as entry_time,
                s.close as entry_price,
                LEAD(s.close, {hold_hours}) OVER (PARTITION BY s.ticker ORDER BY s.date) as exit_price,
                (LEAD(s.close, {hold_hours}) OVER (PARTITION BY s.ticker ORDER BY s.date) - s.close) / s.close * 100.0 - {_rt_cost_pct} as pnl_pct
            FROM ranked_signals s
            WHERE s.rank <= {top_n}
        )
        SELECT 
            COUNT(*) as total_trades,
            AVG(pnl_pct) as avg_return,
            SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0) as win_rate,
            SUM(CASE WHEN pnl_pct > 0 THEN pnl_pct ELSE 0 END) as gross_profit,
            SUM(CASE WHEN pnl_pct < 0 THEN ABS(pnl_pct) ELSE 0 END) as gross_loss,
            STDDEV(pnl_pct) as std_dev,
            MIN(pnl_pct) as max_drawdown
        FROM trades
        WHERE exit_price IS NOT NULL
        """
        
        try:
            df = self.conn.execute(query).df()
            if df.empty or df.iloc[0]['total_trades'] == 0:
                return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            
            row = df.iloc[0]
            gross_profit = float(row['gross_profit'] or 0)
            gross_loss = float(row['gross_loss'] or 0)
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
            
            # Simplified Sharpe
            avg_ret = float(row['avg_return'] or 0)
            std_dev = float(row['std_dev'] or 1)
            sharpe = (avg_ret / std_dev) * math.sqrt(252 * 7) if std_dev > 0 else 0
            
            return BacktestResult(
                total_trades=int(row['total_trades']),
                win_rate=float(row['win_rate'] or 0),
                avg_return_1d=avg_ret,
                avg_return_3d=0,
                avg_return_5d=0,
                avg_max_gain=0,
                avg_max_loss=0,
                sharpe_ratio=sharpe,
                profit_factor=profit_factor,
                max_drawdown=float(row['max_drawdown'] or 0)
            )
        except Exception as e:
            logger.error(f"Hourly backtest failed: {e}")
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
