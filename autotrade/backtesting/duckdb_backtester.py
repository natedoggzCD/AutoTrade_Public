"""
DuckDB-Powered Backtester - Test signal strategies at scale.

Uses DuckDB for sub-second queries across 4,450 tickers and years of data.
No pandas loops - everything is SQL-based for speed.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any

import duckdb
import pandas as pd

from config.config_loader import get_config
from autotrade.feature_engineering.adapters import DuckDBBacktestFeatureAdapter


def _scoring_profile_sql(profile: str) -> str:
    """Return a SQL expression for signal scoring based on strategy archetype."""
    profile = str(profile or "default").lower()
    if profile == "momentum":
        return (
            "COALESCE(weekly_return, 0) * 1.5"
            " + COALESCE(roc_5, 0) * 0.5"
            " + COALESCE(vol_ratio, 0) * 2.0"
            " + COALESCE(adx, 0) * 0.1"
        )
    elif profile == "mean_reversion":
        return (
            "(100.0 - COALESCE(rsi, 50)) * 0.3"
            " + (1.0 - COALESCE(price_pos_bb, 0.5)) * 20.0"
            " + COALESCE(vol_ratio, 0) * 1.5"
            " + COALESCE(atr_pct, 0) * 0.3"
        )
    elif profile == "breakout":
        return (
            "COALESCE(vol_ratio, 0) * 3.0"
            " + COALESCE(atr_pct, 0) * 0.5"
            " + CASE WHEN COALESCE(bb_squeeze_flag, 999) < 1.0 THEN 10.0 ELSE 0 END"
            " + COALESCE(weekly_return, 0) * 0.8"
        )
    elif profile == "trend":
        return (
            "COALESCE(adx, 0) * 0.3"
            " + CASE WHEN COALESCE(macd_hist, 0) > 0 THEN 5.0 ELSE 0 END"
            " + CASE WHEN COALESCE(ema_10, 0) > COALESCE(ema_20, 0) THEN 5.0 ELSE 0 END"
            " + COALESCE(weekly_return, 0) * 0.6"
            " + COALESCE(vol_ratio, 0) * 1.0"
        )
    elif profile == "compression":
        return (
            "(1.0 - COALESCE(bb_width_percentile, 1.0)) * 30.0"
            " + CASE WHEN COALESCE(is_nr7, FALSE) THEN 20.0 ELSE 0.0 END"
            " + (1.0 - ABS(COALESCE(vol_ratio, 1.0) - 1.0)) * 15.0"
            " + CASE WHEN close > COALESCE(sma20, close) THEN 10.0 ELSE 0.0 END"
            " + COALESCE(atr_pct, 0) * 0.3"
        )
    else:  # default
        return (
            "COALESCE(weekly_return, 0)"
            " + COALESCE(vol_ratio, 0) * 2.0"
            " + (COALESCE(atr_pct, 0) * 0.2)"
        )


@dataclass
class BacktestResult:
    total_trades: int
    win_rate: float  # % of trades that hit target
    avg_return_1d: float
    avg_return_3d: float
    avg_return_5d: float
    avg_max_gain: float  # Average best possible gain within hold period
    avg_max_loss: float
    sharpe_ratio: float
    profit_factor: float
    max_drawdown: float
    best_trade: dict = field(default_factory=dict)
    worst_trade: dict = field(default_factory=dict)
    sector_breakdown: dict = field(default_factory=dict)  # Win rate by sector
    scan_type_breakdown: dict = field(
        default_factory=dict
    )  # Win rate by scan type (if applicable)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return asdict(self)


class DuckDBBacktester:
    def __init__(
        self,
        parquet_path: str,
        *,
        skip_enrichment: bool = False,
        preloaded_df: "pd.DataFrame | None" = None,
    ):
        """Initialize with path to daily_features.parquet.

        Args:
            parquet_path: Path to the parquet file on disk.
            skip_enrichment: If True, skip the per-ticker feature enrichment
                loop (saves ~90s per call when the parquet already contains
                all needed indicators, e.g. RSI_14, atr_14, SMA_20).
            preloaded_df: If provided, use this DataFrame instead of reading
                from disk.  The caller is responsible for pre-filtering to the
                target symbols and date range.
        """
        self.parquet_path = parquet_path.replace("\\", "/")
        self.conn = duckdb.connect(":memory:")
        self._preloaded_df = preloaded_df
        self._feature_adapter = None
        self._feature_enabled = False
        if not skip_enrichment:
            try:
                feature_cfg = get_config().feature_engineering
                self._feature_enabled = bool(feature_cfg.enabled)
                if self._feature_enabled:
                    self._feature_adapter = DuckDBBacktestFeatureAdapter(
                        config=feature_cfg.model_dump()
                    )
            except Exception:
                self._feature_enabled = False
                self._feature_adapter = None

        # Verify file exists (only when not using preloaded data)
        if preloaded_df is None and not Path(parquet_path).exists():
            print(f"Warning: Parquet file not found at {parquet_path}")

    def backtest_signal_strategy(
        self,
        signal_criteria: dict,  # e.g. {"rsi_max": 70, "min_volume_ratio": 1.5, "min_atr_pct": 2.0}
        entry_rule: str,  # "next_day_open"
        exit_rule: str,  # "hold_3d" | "target_2pct" | "target_stop_2_3"
        start_date: str,
        end_date: str,
        top_n: int = 25,  # Take top N signals per day by score
    ) -> BacktestResult:
        """
        Backtest a signal strategy entirely in DuckDB SQL.

        Transaction costs are deducted from returns to match the strict
        portfolio simulator used in walk-forward / final validation.
        """
        # Round-trip cost applied to each trade return (percentage points).
        # Matches ExecutionModel: commission (entry+exit) + slippage + spread.
        _rt_cost_pct = float(
            signal_criteria.get(
                "_round_trip_cost_pct",
                0.11,  # default: $0 commission + 0.03% slip×2 + 0.025% spread×2 ≈ 0.11%
            )
        )

        # 1. Build WHERE clause from signal_criteria
        filter_sql = ""

        rsi_min = signal_criteria.get("rsi_min", None)
        rsi_max = signal_criteria.get("rsi_max", None)
        if rsi_min is not None:
            filter_sql += f" AND rsi >= {float(rsi_min)}"
        if rsi_max is not None:
            filter_sql += f" AND rsi <= {float(rsi_max)}"

        atr_min = float(signal_criteria.get("min_atr_pct", 1.5))
        atr_max = signal_criteria.get("max_atr_pct", None)
        filter_sql += f" AND atr_pct >= {atr_min}"
        if atr_max is not None:
            filter_sql += f" AND atr_pct <= {float(atr_max)}"

        vol_ratio_thresh = float(signal_criteria.get("min_volume_ratio", 1.0))
        filter_sql += f" AND vol_ratio >= {vol_ratio_thresh}"

        if bool(signal_criteria.get("require_above_sma20", False)):
            filter_sql += " AND close > COALESCE(sma20, close)"

        if bool(signal_criteria.get("require_sma5_curl_positive", False)):
            filter_sql += " AND sma5 > sma5_prev"

        # --- Extended feature filters ---
        min_adx = signal_criteria.get("min_adx", 0.0)
        if float(min_adx) > 0:
            filter_sql += f" AND COALESCE(adx, 0) >= {float(min_adx)}"

        if bool(signal_criteria.get("require_positive_macd_hist", False)):
            filter_sql += " AND COALESCE(macd_hist, 0) > 0"

        if bool(signal_criteria.get("require_ema_10_above_20", False)):
            filter_sql += " AND COALESCE(ema_10, 0) > COALESCE(ema_20, 0)"

        if bool(signal_criteria.get("require_bb_squeeze", False)):
            filter_sql += " AND COALESCE(bb_squeeze_flag, 999) < 1.0"

        if bool(signal_criteria.get("require_bb_width_compression", False)):
            max_pct = float(signal_criteria.get("max_bb_width_percentile", 0.10))
            filter_sql += f" AND COALESCE(bb_width_percentile, 1.0) <= {max_pct}"
        bb_width_lookback = max(
            1, int(signal_criteria.get("bb_width_percentile_lookback", 60) or 60)
        )

        if bool(signal_criteria.get("require_nr7", False)):
            filter_sql += " AND COALESCE(is_nr7, FALSE) = TRUE"

        volume_deviation = float(signal_criteria.get("volume_avg_deviation_pct", 0.0))
        if volume_deviation > 0.0:
            lower = max(0.0, 1.0 - volume_deviation / 100.0)
            upper = 1.0 + volume_deviation / 100.0
            filter_sql += f" AND COALESCE(vol_ratio, 999.0) BETWEEN {lower} AND {upper}"

        max_bb_pos = float(signal_criteria.get("max_price_position_bb", 999.0))
        if max_bb_pos < 999.0:
            filter_sql += f" AND COALESCE(price_pos_bb, 0.5) <= {max_bb_pos}"
        min_bb_pos = float(signal_criteria.get("min_price_position_bb", -999.0))
        if min_bb_pos > -999.0:
            filter_sql += f" AND COALESCE(price_pos_bb, 0.5) >= {min_bb_pos}"

        min_roc_5 = float(signal_criteria.get("min_roc_5", -999.0))
        if min_roc_5 > -999.0:
            filter_sql += f" AND COALESCE(roc_5, 0) >= {min_roc_5}"
        min_roc_10 = float(signal_criteria.get("min_roc_10", -999.0))
        if min_roc_10 > -999.0:
            filter_sql += f" AND COALESCE(roc_10, 0) >= {min_roc_10}"

        max_stoch_k = float(signal_criteria.get("max_stoch_k", 100.0))
        if max_stoch_k < 100.0:
            filter_sql += f" AND COALESCE(stoch_k, 50) <= {max_stoch_k}"
        min_stoch_k = float(signal_criteria.get("min_stoch_k", 0.0))
        if min_stoch_k > 0.0:
            filter_sql += f" AND COALESCE(stoch_k, 50) >= {min_stoch_k}"

        min_cci = float(signal_criteria.get("min_cci", -999.0))
        if min_cci > -999.0:
            filter_sql += f" AND COALESCE(cci_val, 0) >= {min_cci}"
        max_cci = float(signal_criteria.get("max_cci", 999.0))
        if max_cci < 999.0:
            filter_sql += f" AND COALESCE(cci_val, 0) <= {max_cci}"

        if bool(signal_criteria.get("require_above_ichimoku_cloud", False)):
            filter_sql += " AND close > GREATEST(COALESCE(ichi_senkou_a, 0), COALESCE(ichi_senkou_b, 0))"

        if bool(signal_criteria.get("require_tenkan_above_kijun", False)):
            filter_sql += " AND COALESCE(ichi_tenkan, 0) > COALESCE(ichi_kijun, 0)"

        max_dist_sma20 = float(signal_criteria.get("max_distance_from_sma20", 999.0))
        if max_dist_sma20 < 999.0:
            filter_sql += f" AND ABS(COALESCE(dist_sma20, 0)) <= {max_dist_sma20}"

        min_atr_ratio = float(signal_criteria.get("min_atr_ratio_5_14", 0.0))
        if min_atr_ratio > 0.0:
            filter_sql += f" AND COALESCE(atr_ratio, 0) >= {min_atr_ratio}"

        hold_days = int(signal_criteria.get("hold_days", 5))
        hold_days = max(1, min(10, hold_days))
        stop_atr_mult = float(signal_criteria.get("stop_atr_mult", 2.0))
        target_atr_mult = float(signal_criteria.get("target_atr_mult", 3.0))
        target_entry_range_mult = float(
            signal_criteria.get("target_entry_range_mult", 0.0)
        )
        target_distance_sql = (
            f"CASE WHEN {target_entry_range_mult} > 0.0 "
            f"AND COALESCE(day_range, 0) > 0 "
            f"THEN day_range * {target_entry_range_mult} "
            f"ELSE atr_14 * {target_atr_mult} END"
        )

        symbols = signal_criteria.get("symbols") or []
        symbol_filter_sql = ""
        if symbols:
            safe_symbols = []
            for sym in symbols:
                s = str(sym).upper().strip()
                if s:
                    safe_symbols.append(s.replace("'", "''"))
            if safe_symbols:
                ticker_list = ", ".join([f"'{s}'" for s in safe_symbols])
                symbol_filter_sql = f" AND UPPER(ticker) IN ({ticker_list})"

        base_query = f"""
            SELECT
                UPPER(ticker) AS ticker,
                CAST(Date AS DATE) AS date,
                CAST("Open" AS DOUBLE) AS open,
                CAST("High" AS DOUBLE) AS high,
                CAST("Low" AS DOUBLE) AS low,
                CAST("Close" AS DOUBLE) AS close,
                CAST(Volume AS DOUBLE) AS volume,
                CAST(RSI_14 AS DOUBLE) AS rsi,
                CAST(atr_14 AS DOUBLE) AS atr_14,
                CAST(SMA_20 AS DOUBLE) AS sma20,
                -- Extended features
                CAST(ADX AS DOUBLE) AS adx,
                CAST(DI_plus AS DOUBLE) AS di_plus,
                CAST(DI_minus AS DOUBLE) AS di_minus,
                CAST(MACD AS DOUBLE) AS macd,
                CAST(MACD_signal AS DOUBLE) AS macd_signal,
                CAST(MACD_hist AS DOUBLE) AS macd_hist,
                CAST(EMA_10 AS DOUBLE) AS ema_10,
                CAST(EMA_20 AS DOUBLE) AS ema_20,
                CAST(SMA_10 AS DOUBLE) AS sma_10,
                CAST(ROC_5 AS DOUBLE) AS roc_5,
                CAST(ROC_10 AS DOUBLE) AS roc_10,
                CAST(ROC_20 AS DOUBLE) AS roc_20,
                CAST(BB_upper AS DOUBLE) AS bb_upper,
                CAST(BB_mid AS DOUBLE) AS bb_mid,
                CAST(BB_lower AS DOUBLE) AS bb_lower,
                CAST(bb_width AS DOUBLE) AS bb_width,
                CAST(bb_squeeze AS DOUBLE) AS bb_squeeze_flag,
                CAST(price_position_bb AS DOUBLE) AS price_pos_bb,
                CAST(distance_from_sma20 AS DOUBLE) AS dist_sma20,
                CAST(atr_5 AS DOUBLE) AS atr_5,
                CAST(atr_ratio_5_14 AS DOUBLE) AS atr_ratio,
                CAST(OBV AS DOUBLE) AS obv,
                CAST(volume_ratio AS DOUBLE) AS volume_ratio_raw,
                CAST(volume_sma_20 AS DOUBLE) AS volume_sma_20,
                CAST(volume_roc_5 AS DOUBLE) AS volume_roc_5,
                CAST("Stoch_%K" AS DOUBLE) AS stoch_k,
                CAST("Stoch_%D" AS DOUBLE) AS stoch_d,
                CAST(CCI AS DOUBLE) AS cci_val,
                CAST(Ichimoku_tenkan AS DOUBLE) AS ichi_tenkan,
                CAST(Ichimoku_kijun AS DOUBLE) AS ichi_kijun,
                CAST(Ichimoku_senkou_A AS DOUBLE) AS ichi_senkou_a,
                CAST(Ichimoku_senkou_B AS DOUBLE) AS ichi_senkou_b
            FROM read_parquet('{self.parquet_path}')
            WHERE CAST(Date AS DATE) >= '{start_date}' AND CAST(Date AS DATE) <= '{end_date}'
              {symbol_filter_sql}
        """

        try:
            if self._preloaded_df is not None:
                # Use pre-loaded DataFrame — apply symbol + date filter in pandas
                raw_df = self._preloaded_df
                if symbol_filter_sql:
                    safe_set = set()
                    for sym in symbols:
                        s = str(sym).upper().strip()
                        if s:
                            safe_set.add(s)
                    if safe_set:
                        raw_df = raw_df[raw_df["ticker"].isin(safe_set)]
                # Date filter
                raw_df = raw_df[
                    (raw_df["date"] >= start_date) & (raw_df["date"] <= end_date)
                ].copy()
            else:
                raw_df = self.conn.execute(base_query).df()
        except Exception as e:
            print(f"DuckDB Error loading base data: {e}")
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        if raw_df.empty:
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        if self._feature_enabled and self._feature_adapter is not None:
            try:
                enriched_parts = []
                raw_df["date"] = pd.to_datetime(raw_df["date"])
                for _, group in raw_df.groupby("ticker", sort=False):
                    group = group.sort_values("date").copy()
                    enriched = self._feature_adapter.enrich_duckdb_results(
                        group, as_of_column="date"
                    )

                    if "rsi_14" in enriched.columns:
                        enriched["rsi"] = enriched.get(
                            "rsi", pd.Series(index=enriched.index)
                        ).fillna(enriched["rsi_14"])
                    if "atr_14" in enriched.columns:
                        enriched["atr_14"] = enriched["atr_14"].fillna(group["atr_14"])
                    if "sma20" in enriched.columns:
                        enriched["sma20"] = enriched["sma20"].fillna(group["sma20"])

                    enriched_parts.append(enriched)

                if enriched_parts:
                    raw_df = pd.concat(enriched_parts, ignore_index=True)
            except Exception:
                # Backward-compatible fallback: use the legacy DuckDB-only features.
                pass

        self.conn.register("raw_input", raw_df)

        query = f"""
        WITH raw_data AS (
            SELECT * FROM raw_input
        ),
        with_indicators AS (
            SELECT *,
                CASE WHEN close > 0 THEN (atr_14 / close) * 100 ELSE NULL END AS atr_pct,
                -- Volume Ratio
                volume / NULLIF(AVG(volume) OVER (
                    PARTITION BY ticker ORDER BY date
                    ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING), 0) AS vol_ratio,
                -- Weekly Return (Momentum)
                (close - LAG(close, 5) OVER (PARTITION BY ticker ORDER BY date))
                    / NULLIF(LAG(close, 5) OVER (PARTITION BY ticker ORDER BY date), 0) * 100 AS weekly_return,
                AVG(close) OVER (
                    PARTITION BY ticker ORDER BY date
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                ) AS sma5,
                -- SMA 100: long-term trend filter to exclude dead/downtrending stocks
                AVG(close) OVER (
                    PARTITION BY ticker ORDER BY date
                    ROWS BETWEEN 99 PRECEDING AND CURRENT ROW
                ) AS sma100,
                ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date) AS row_num,
                high - low AS day_range,
                MIN(high - low) OVER (
                    PARTITION BY ticker ORDER BY date
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) AS nr7_min_range,
                    
                -- Future Returns (calculated indiscriminately on all rows)
                LEAD(open) OVER (PARTITION BY ticker ORDER BY date) AS entry_price,
                LEAD(date) OVER (PARTITION BY ticker ORDER BY date) AS entry_date,
                
                LEAD(close, 1) OVER (PARTITION BY ticker ORDER BY date) AS price_local_1d,
                LEAD(close, 3) OVER (PARTITION BY ticker ORDER BY date) AS price_local_3d,
                LEAD(close, 5) OVER (PARTITION BY ticker ORDER BY date) AS price_local_5d,
                LEAD(close, {hold_days}) OVER (PARTITION BY ticker ORDER BY date) AS price_local_hold,
                
                MAX(high) OVER (
                    PARTITION BY ticker ORDER BY date
                    ROWS BETWEEN 1 FOLLOWING AND {hold_days} FOLLOWING
                ) AS future_high_max,
                MIN(low) OVER (
                    PARTITION BY ticker ORDER BY date
                    ROWS BETWEEN 1 FOLLOWING AND {hold_days} FOLLOWING
                ) AS future_low_min
                
            FROM raw_data
        ),
        with_trend AS (
            SELECT
                wi.*,
                LAG(sma5) OVER (PARTITION BY ticker ORDER BY date) AS sma5_prev,
                day_range <= nr7_min_range AS is_nr7,
                (
                    SELECT AVG(
                        CASE
                            WHEN wi2.bb_width <= wi.bb_width THEN 1.0
                            ELSE 0.0
                        END
                    )
                    FROM with_indicators wi2
                    WHERE wi2.ticker = wi.ticker
                      AND wi2.row_num BETWEEN wi.row_num - {bb_width_lookback - 1} AND wi.row_num
                      AND wi2.bb_width IS NOT NULL
                ) AS bb_width_percentile
            FROM with_indicators wi
        ),
        signals AS (
            SELECT *,
                (
                    TRUE
                    AND close >= 5.0               -- Min price filter: no penny stocks
                    AND COALESCE(volume_sma_20, 0) >= 100000  -- Min avg volume: no illiquid stocks
                    AND close > COALESCE(sma100, 0)  -- Hard filter: price must be above SMA100
                    {filter_sql}
                ) as is_signal
            FROM with_trend
        ),
        ranked_signals AS (
            SELECT *,
                {_scoring_profile_sql(signal_criteria.get("scoring_profile", "default"))} AS signal_score,
                ROW_NUMBER() OVER (
                    PARTITION BY date
                    ORDER BY
                        {_scoring_profile_sql(signal_criteria.get("scoring_profile", "default"))} DESC
                ) AS daily_rank
            FROM signals
            WHERE is_signal = TRUE
        ),
        trades AS (
            SELECT
                *,
                -- 1-day return
                (price_local_1d - entry_price) / NULLIF(entry_price, 0) * 100 AS return_1d,
                -- 3-day return
                (price_local_3d - entry_price) / NULLIF(entry_price, 0) * 100 AS return_3d,
                -- 5-day return
                (price_local_5d - entry_price) / NULLIF(entry_price, 0) * 100 AS return_5d,
                -- hold return (raw, no stops)
                (price_local_hold - entry_price) / NULLIF(entry_price, 0) * 100 AS return_hold_raw,

                -- Max potential excursion within hold window
                (future_high_max - entry_price) / NULLIF(entry_price, 0) * 100 AS max_gain_1d,
                (future_low_min - entry_price) / NULLIF(entry_price, 0) * 100 AS max_loss_1d,

                -- Stop-aware realized return WITH transaction costs
                CASE
                    WHEN atr_14 > 0 AND entry_price > 0 THEN
                        CASE
                            -- Stop hit (low went below stop level)
                            WHEN future_low_min <= entry_price - (atr_14 * {stop_atr_mult})
                                 AND future_high_max < entry_price + ({target_distance_sql})
                            THEN -(atr_14 * {stop_atr_mult}) / entry_price * 100.0 - {_rt_cost_pct}
                            -- Target hit (high went above target level)
                            WHEN future_high_max >= entry_price + ({target_distance_sql})
                                 AND future_low_min > entry_price - (atr_14 * {stop_atr_mult})
                            THEN ({target_distance_sql}) / entry_price * 100.0 - {_rt_cost_pct}
                            -- Both hit same window: pessimistically assume stop hit first
                            -- (eliminates look-ahead bias from close-based tiebreaker)
                            WHEN future_low_min <= entry_price - (atr_14 * {stop_atr_mult})
                                 AND future_high_max >= entry_price + ({target_distance_sql})
                            THEN -(atr_14 * {stop_atr_mult}) / entry_price * 100.0 - {_rt_cost_pct}
                            -- Neither hit: use raw hold return minus costs
                            ELSE (price_local_hold - entry_price) / NULLIF(entry_price, 0) * 100 - {_rt_cost_pct}
                        END
                    ELSE (price_local_hold - entry_price) / NULLIF(entry_price, 0) * 100 - {_rt_cost_pct}
                END AS return_hold

            FROM ranked_signals
            WHERE daily_rank <= {top_n}
              AND entry_price IS NOT NULL
              AND price_local_hold IS NOT NULL
        )
        SELECT
            COUNT(*) as total_trades,
            AVG(return_1d) as avg_return_1d,
            AVG(return_3d) as avg_return_3d,
            AVG(return_5d) as avg_return_5d,
            AVG(return_hold) as avg_return_hold,

            AVG(max_gain_1d) as avg_max_gain,
            AVG(max_loss_1d) as avg_max_loss,

            -- Win Rate uses stop-aware return
            SUM(CASE WHEN return_hold > 0 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0) as win_rate,
            SUM(CASE WHEN return_hold > 0 THEN return_hold ELSE 0 END) as gross_profit,
            SUM(CASE WHEN return_hold < 0 THEN ABS(return_hold) ELSE 0 END) as gross_loss,

            STDDEV(return_hold) as std_dev_hold,
            MIN(return_hold) as max_drawdown_trade

        FROM trades
        """

        # Execute
        try:
            df = self.conn.execute(query).df()
            if df.empty or df.iloc[0]["total_trades"] == 0:
                return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

            row = df.iloc[0]

            # Derived metrics
            avg_ret = (
                row["avg_return_hold"] if pd.notnull(row["avg_return_hold"]) else 0
            )
            std_dev = row["std_dev_hold"] if pd.notnull(row["std_dev_hold"]) else 1
            # Annualize Sharpe: per-trade Sharpe × sqrt(trades_per_year)
            # Assume ~252 trading days / hold_days = trades_per_year
            import math

            trades_per_year = 252.0 / max(1, hold_days)
            raw_sharpe = (avg_ret / std_dev) if std_dev != 0 else 0
            sharpe = raw_sharpe * math.sqrt(trades_per_year)
            gross_profit = (
                float(row["gross_profit"]) if pd.notnull(row["gross_profit"]) else 0.0
            )
            gross_loss = (
                float(row["gross_loss"]) if pd.notnull(row["gross_loss"]) else 0.0
            )
            profit_factor = (
                (gross_profit / gross_loss)
                if gross_loss > 0
                else (gross_profit if gross_profit > 0 else 0.0)
            )

            return BacktestResult(
                total_trades=int(row["total_trades"]),
                win_rate=float(row["win_rate"]) if pd.notnull(row["win_rate"]) else 0.0,
                avg_return_1d=float(row["avg_return_1d"])
                if pd.notnull(row["avg_return_1d"])
                else 0.0,
                avg_return_3d=float(row["avg_return_3d"])
                if pd.notnull(row["avg_return_3d"])
                else 0.0,
                avg_return_5d=float(row["avg_return_5d"])
                if pd.notnull(row["avg_return_5d"])
                else 0.0,
                avg_max_gain=float(row["avg_max_gain"])
                if pd.notnull(row["avg_max_gain"])
                else 0.0,
                avg_max_loss=float(row["avg_max_loss"])
                if pd.notnull(row["avg_max_loss"])
                else 0.0,
                sharpe_ratio=float(sharpe),
                profit_factor=float(profit_factor),
                max_drawdown=float(row["max_drawdown_trade"])
                if pd.notnull(row["max_drawdown_trade"])
                else 0.0,
            )

        except Exception as e:
            print(f"DuckDB Error: {e}")
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def compare_strategies(
        self,
        strategies: list[dict],  # List of strategy configs
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Run multiple strategies and return comparison table.
        """
        results = []
        for strat in strategies:
            res = self.backtest_signal_strategy(
                strat, "next_day_open", "hold_5d", start_date, end_date
            )
            results.append(
                {
                    "strategy": str(strat),
                    "win_rate": res.win_rate,
                    "avg_return": res.avg_return_5d,
                    "trades": res.total_trades,
                    "sharpe": res.sharpe_ratio,
                }
            )
        return pd.DataFrame(results)

    def parameter_sweep(
        self,
        base_strategy: dict,
        param_name: str,  # e.g. "rsi_max"
        param_range: list,  # e.g. [60, 65, 70, 75, 80]
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Sweep a single parameter across a range.
        """
        results = []
        for val in param_range:
            strat = base_strategy.copy()
            strat[param_name] = val
            res = self.backtest_signal_strategy(
                strat, "next_day_open", "hold_5d", start_date, end_date
            )
            results.append(
                {
                    param_name: val,
                    "win_rate": res.win_rate,
                    "avg_return": res.avg_return_5d,
                    "trades": res.total_trades,
                }
            )
        return pd.DataFrame(results)
