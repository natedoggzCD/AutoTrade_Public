"""
Strategy Validator
==================
Lightweight strategy-level backtesting used by the PM workflow before
signal generation. Evaluates whether the current strategy parameters are
performing acceptably on recent data and suggests a better parameter set
via a small grid search.

This module intentionally keeps runtime under a few seconds by using
DuckDB window functions directly against the daily_features parquet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import duckdb

from config.config_loader import get_config

logger = logging.getLogger(__name__)


@dataclass
class StrategyHealth:
    """Simple health snapshot for the current strategy."""
    status: str
    recent_win_rate: float
    recent_avg_return: float
    recent_profit_factor: float
    sample_size: int
    lookback_days: int
    recommendation: str


class StrategyValidator:
    """Validate current strategy parameters against recent history."""

    def __init__(self, parquet_path: Optional[Path] = None, parent_logger: Optional[logging.Logger] = None):
        self.config = get_config()
        self.logger = parent_logger or logger
        self.parquet_path = Path(parquet_path) if parquet_path else self._resolve_parquet_path()
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Daily features parquet not found: {self.parquet_path}")

    def _resolve_parquet_path(self) -> Path:
        data_cfg = self.config.data
        base = Path(data_cfg.downday_root)
        rel = Path(data_cfg.daily_features_parquet)
        return rel if rel.is_absolute() else base / rel

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect()

    def validate_current_strategy(self, lookback_days: int = 20) -> StrategyHealth:
        """Evaluate current strategy thresholds over a recent window."""
        cfg = self.config
        strat_cfg = cfg.strategy
        uni_cfg = cfg.universe_scanner

        query = f"""
        WITH base AS (
            SELECT
                ticker,
                Date AS dt,
                close,
                RSI_14 AS rsi,
                CASE WHEN close > 0 THEN (atr_14 / close) * 100 ELSE NULL END AS atr_pct,
                volume_ratio,
                LEAD(close, 3) OVER w AS close_3d,
                LEAD(close, 5) OVER w AS close_5d
            FROM parquet_scan('{self.parquet_path.as_posix()}')
            WHERE Date >= CURRENT_DATE - INTERVAL '{lookback_days} days'
              AND close BETWEEN {uni_cfg.min_price} AND {uni_cfg.max_price}
              AND volume >= {uni_cfg.min_avg_volume}
            WINDOW w AS (PARTITION BY ticker ORDER BY Date)
        ),
        filtered AS (
            SELECT *,
                CASE WHEN close > 0 AND close_3d IS NOT NULL
                     THEN ((close_3d - close) / close) * 100 ELSE NULL END AS ret_3d,
                CASE WHEN close > 0 AND close_5d IS NOT NULL
                     THEN ((close_5d - close) / close) * 100 ELSE NULL END AS ret_5d
            FROM base
            WHERE atr_pct >= {uni_cfg.min_atr_percent}
              AND rsi BETWEEN {cfg.screener_v2.rsi_min} AND {cfg.screener_v2.rsi_max}
        )
        SELECT
            COUNT(*) AS n_trades,
            AVG(CASE WHEN ret_3d >= 2.0 THEN 1.0 ELSE 0.0 END) AS win_rate,
            AVG(ret_3d) AS avg_ret,
            SUM(CASE WHEN ret_3d > 0 THEN ret_3d ELSE 0 END) AS gross_profit,
            SUM(CASE WHEN ret_3d < 0 THEN ABS(ret_3d) ELSE 0 END) AS gross_loss
        FROM filtered
        WHERE ret_3d IS NOT NULL
        """

        with self._connect() as con:
            row = con.execute(query).fetchone()

        n_trades = int(row[0]) if row and row[0] is not None else 0
        win_rate = float(row[1]) if row and row[1] is not None else 0.0
        avg_ret = float(row[2]) if row and row[2] is not None else 0.0
        gross_profit = float(row[3]) if row and row[3] is not None else 0.0
        gross_loss = float(row[4]) if row and row[4] is not None else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

        status, recommendation = self._classify_health(win_rate, profit_factor, n_trades, lookback_days)

        health = StrategyHealth(
            status=status,
            recent_win_rate=win_rate,
            recent_avg_return=avg_ret,
            recent_profit_factor=profit_factor,
            sample_size=n_trades,
            lookback_days=lookback_days,
            recommendation=recommendation,
        )

        self._log_health(health)
        return health

    def validate_from_strategy_lab(self) -> StrategyHealth:
        """Validate using Strategy Lab backtest metrics instead of universe scan.

        Loads validated strategies from the Strategy Lab and computes a
        weighted-average win_rate and profit_factor (weighted by total_trades
        so high-sample strategies count more).  Falls back to the universe
        scan if no validated strategies exist.
        """
        from autotrade.signals.strategy_pool import load_validated_strategies

        strategies = load_validated_strategies()
        if not strategies:
            self.logger.warning("[VALIDATOR] No validated strategies found, falling back to universe scan")
            return self.validate_current_strategy()

        total_weight = 0
        weighted_wr = 0.0
        weighted_pf = 0.0
        for s in strategies:
            metrics = s.get("metrics", {})
            trades = metrics.get("total_trades", 0)
            wr = metrics.get("win_rate", 0)
            pf = metrics.get("profit_factor", 0)
            if trades > 0 and wr > 0:
                weighted_wr += wr * trades
                weighted_pf += pf * trades
                total_weight += trades

        if total_weight == 0:
            self.logger.warning("[VALIDATOR] Validated strategies have no trade data, falling back to universe scan")
            return self.validate_current_strategy()

        avg_wr = weighted_wr / total_weight
        avg_pf = weighted_pf / total_weight
        status, recommendation = self._classify_health(avg_wr, avg_pf, total_weight, 0)
        recommendation = f"Strategy Lab: {len(strategies)} strategies, {total_weight} backtest trades"

        health = StrategyHealth(
            status=status,
            recent_win_rate=avg_wr,
            recent_avg_return=0.0,
            recent_profit_factor=avg_pf,
            sample_size=total_weight,
            lookback_days=0,
            recommendation=recommendation,
        )

        self._log_health(health)
        return health

    def find_optimal_parameters(self, lookback_days: int = 60) -> Dict:
        """Small grid search to suggest better parameters."""
        uni_cfg = self.config.universe_scanner

        grids = {
            "rsi_range": [(30, 60), (35, 65), (40, 70), (45, 75)],
            "atr_min": [1.0, 1.5, 2.0, 2.5, 3.0],
            "vol_ratio_min": [1.0, 1.5, 2.0],
        }

        best = None
        best_key = None

        with self._connect() as con:
            base_sql = f"""
            SELECT
                ticker,
                Date AS dt,
                close,
                RSI_14 AS rsi,
                CASE WHEN close > 0 THEN (atr_14 / close) * 100 ELSE NULL END AS atr_pct,
                volume_ratio,
                LEAD(close, 3) OVER w AS close_3d
            FROM parquet_scan('{self.parquet_path.as_posix()}')
            WHERE Date >= CURRENT_DATE - INTERVAL '{lookback_days} days'
              AND close BETWEEN {uni_cfg.min_price} AND {uni_cfg.max_price}
              AND volume >= {uni_cfg.min_avg_volume}
            WINDOW w AS (PARTITION BY ticker ORDER BY Date)
            """
            con.execute(f"CREATE OR REPLACE TEMP VIEW base AS {base_sql}")

            for rsi_min, rsi_max in grids["rsi_range"]:
                for atr_min in grids["atr_min"]:
                    for vol_min in grids["vol_ratio_min"]:
                        query = f"""
                        SELECT
                            COUNT(*) AS n_trades,
                            AVG(CASE WHEN ret_3d >= 2.0 THEN 1.0 ELSE 0.0 END) AS win_rate,
                            AVG(ret_3d) AS avg_ret
                        FROM (
                            SELECT *,
                                CASE WHEN close > 0 AND close_3d IS NOT NULL
                                     THEN ((close_3d - close) / close) * 100 ELSE NULL END AS ret_3d
                            FROM base
                            WHERE rsi BETWEEN {rsi_min} AND {rsi_max}
                              AND atr_pct >= {atr_min}
                              AND (volume_ratio IS NULL OR volume_ratio >= {vol_min})
                        )
                        WHERE ret_3d IS NOT NULL
                        """
                        n_trades, win_rate, avg_ret = con.execute(query).fetchone()
                        score = (win_rate or 0) * 0.7 + max(0.0, avg_ret or 0) * 0.02
                        key = {
                            "rsi_min": rsi_min,
                            "rsi_max": rsi_max,
                            "atr_min": atr_min,
                            "vol_ratio_min": vol_min,
                            "win_rate": win_rate or 0.0,
                            "avg_ret": avg_ret or 0.0,
                            "sample": n_trades or 0,
                            "score": score,
                        }
                        if best is None or key["score"] > best["score"]:
                            best = key
                            best_key = key

        if not best:
            return {"status": "NO_DATA", "message": "No trades found in lookback window"}

        self.logger.info(
            f"[STRATEGY OPTIMIZER] Best params: RSI {best_key['rsi_min']}-{best_key['rsi_max']}, "
            f"ATR%>={best_key['atr_min']} vol_ratio>={best_key['vol_ratio_min']} | "
            f"win_rate={best_key['win_rate']:.1%}, avg_ret={best_key['avg_ret']:+.2f}% (n={best_key['sample']})"
        )
        return best_key

    @staticmethod
    def _classify_health(win_rate: float, profit_factor: float, sample_size: int, lookback_days: int) -> tuple[str, str]:
        """Classify strategy health with graduated thresholds.

        Win rate here is measured as pct of universe-wide filtered stocks that gain
        >=2% in 3 days -- this is a BROAD market health metric, NOT our actual trade
        win rate.  Thresholds must be realistic for this definition: even in bull
        markets, only ~35-45% of filtered stocks meet this bar.

        Classification should be conservative about triggering FAILING because the
        failsafe's FAILING profile severely restricts trading.  We should only
        declare FAILING when conditions are truly awful.
        """
        if sample_size < 20:
            return "INSUFFICIENT_DATA", "Not enough recent trades to evaluate"
        if win_rate >= 0.40 and profit_factor >= 1.1:
            return "HEALTHY", "Strategy is working. Continue."
        if win_rate >= 0.30 and profit_factor >= 0.7:
            return "DEGRADING", "Performance softening; consider loosening RSI or ATR filters"
        if win_rate >= 0.20 and profit_factor >= 0.4:
            return "FAILING", "Strategy failing; tighten risk or reduce size until metrics recover"
        return "FAILING", "Strategy critically underperforming"

    def _log_health(self, health: StrategyHealth) -> None:
        self.logger.info("\n[STRATEGY VALIDATION]")
        self.logger.info("-" * 50)
        self.logger.info(
            f"   Status: {health.status} | lookback={health.lookback_days}d | "
            f"win_rate={health.recent_win_rate:.1%} | avg_ret={health.recent_avg_return:+.2f}% | "
            f"PF={health.recent_profit_factor:.2f} | n={health.sample_size}"
        )
        self.logger.info(f"   Recommendation: {health.recommendation}")
