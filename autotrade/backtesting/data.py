from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from config.config_loader import get_config
from autotrade.data_ingestion.bootstrap import ensure_core_market_data_ready
from autotrade.data_ingestion.paths import get_primary_parquet_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestPaths:
    daily_features_path: Path


def resolve_backtest_paths() -> BacktestPaths:
    daily = get_primary_parquet_path()
    return BacktestPaths(daily_features_path=daily)


def _sanitize_symbols(symbols: Optional[Sequence[str]]) -> Optional[list[str]]:
    if not symbols:
        return None
    cleaned = []
    for s in symbols:
        if not s:
            continue
        cleaned.append(str(s).upper().strip())
    return sorted(set(cleaned))


def _to_date(value) -> date:
    if value is None:
        raise ValueError("Date value is required")
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return pd.to_datetime(value).date()


def get_latest_prediction_date(daily_features_path: Path) -> Optional[date]:
    """
    Return latest market date from local parquet.

    The legacy predictions SQLite workflow is removed; this now anchors lookbacks
    to the latest available market bar date.
    """
    if not daily_features_path or not daily_features_path.exists():
        return None

    try:
        import duckdb

        conn = duckdb.connect(database=":memory:")
        query = f"""
        SELECT MAX(CAST(Date AS DATE)) AS latest_date
        FROM read_parquet('{daily_features_path.as_posix()}')
        """
        row = conn.execute(query).fetchone()
        conn.close()
        if not row or row[0] is None:
            return None
        return pd.to_datetime(row[0]).date()
    except Exception as exc:
        logger.warning(f"Failed to read latest market date from parquet: {exc}")
        return None


def load_predictions(
    daily_features_path: Optional[Path] = None,
    start_date=None,
    end_date=None,
    symbols: Optional[Sequence[str]] = None,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build backtest candidate signals from local market bars only.

    `db_path` is accepted for backward compatibility but ignored.
    """
    _ = db_path
    symbols = _sanitize_symbols(symbols)
    start_d = _to_date(start_date)
    end_d = _to_date(end_date)

    daily_path = daily_features_path or get_primary_parquet_path()
    history_start = start_d - timedelta(days=45)
    bars = load_daily_bars(
        daily_features_path=daily_path,
        start_date=history_start,
        end_date=end_d,
        symbols=symbols,
    )
    if bars.empty:
        return bars

    bars = bars.sort_values(["ticker", "date"]).copy()
    bars["ticker"] = bars["ticker"].astype(str).str.upper()

    grp = bars.groupby("ticker", group_keys=False)

    prev_close = grp["close"].shift(1)
    tr = pd.concat(
        [
            (bars["high"] - bars["low"]).abs(),
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    bars["atr_14"] = tr.groupby(bars["ticker"]).transform(
        lambda s: s.rolling(14, min_periods=5).mean()
    )
    fallback_atr = (bars["close"].abs() * 0.03).clip(lower=0.05)
    # Avoid pandas Series.fillna(Series) bug in this environment that can raise
    # NameError("DataFrame is not defined") inside pandas internals.
    bars["atr_14"] = pd.Series(
        np.where(bars["atr_14"].isna(), fallback_atr, bars["atr_14"]),
        index=bars.index,
    )

    bars["sma_20"] = grp["close"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    bars["vol_sma_20"] = grp["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    bars["weekly_return"] = grp["close"].transform(lambda s: (s / s.shift(5) - 1.0) * 100.0)
    bars["volume_ratio_20"] = bars["volume"] / bars["vol_sma_20"].replace(0, np.nan)
    bars["volume_ratio_20"] = bars["volume_ratio_20"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    bars["s1_price"] = grp["low"].transform(lambda s: s.rolling(10, min_periods=3).min().shift(1))
    bars["s2_price"] = grp["low"].transform(lambda s: s.rolling(20, min_periods=5).min().shift(1))
    bars["r1_price"] = grp["high"].transform(lambda s: s.rolling(10, min_periods=3).max().shift(1))
    bars["r2_price"] = grp["high"].transform(lambda s: s.rolling(20, min_periods=5).max().shift(1))

    # Fix: use np.where for Series.fillna(Series) to bypass namespace bug
    bars["s1_price"] = pd.Series(
        np.where(bars["s1_price"].isna(), bars["close"] - bars["atr_14"] * 1.5, bars["s1_price"]),
        index=bars.index,
    )
    bars["s2_price"] = pd.Series(
        np.where(bars["s2_price"].isna(), bars["close"] - bars["atr_14"] * 2.5, bars["s2_price"]),
        index=bars.index,
    )
    bars["r1_price"] = pd.Series(
        np.where(bars["r1_price"].isna(), bars["close"] + bars["atr_14"] * 2.0, bars["r1_price"]),
        index=bars.index,
    )
    bars["r2_price"] = pd.Series(
        np.where(bars["r2_price"].isna(), bars["close"] + bars["atr_14"] * 3.0, bars["r2_price"]),
        index=bars.index,
    )

    close_safe = bars["close"].replace(0, np.nan)
    s1_dist_pct = ((bars["close"] - bars["s1_price"]) / close_safe * 100.0).abs().fillna(0.0)
    r1_dist_pct = ((bars["r1_price"] - bars["close"]) / close_safe * 100.0).abs().fillna(0.0)
    bars["s1_strength"] = (100.0 - s1_dist_pct * 15.0).clip(20.0, 95.0)
    bars["s2_strength"] = (100.0 - (s1_dist_pct + 1.5) * 15.0).clip(20.0, 90.0)
    bars["r1_strength"] = (100.0 - r1_dist_pct * 15.0).clip(20.0, 95.0)
    bars["r2_strength"] = (100.0 - (r1_dist_pct + 1.5) * 15.0).clip(20.0, 90.0)

    trend_pct = ((bars["close"] / bars["sma_20"].replace(0, np.nan)) - 1.0) * 100.0
    bullish_raw = (
        5.0
        + 0.40 * bars["weekly_return"].fillna(0.0)
        + 0.15 * trend_pct.fillna(0.0)
        + 1.50 * (bars["volume_ratio_20"].fillna(1.0) - 1.0)
    )
    bars["bullish_score"] = bullish_raw.clip(0.0, 10.0)

    bars["atr_percent"] = (bars["atr_14"] / close_safe * 100.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    bars["distance_to_s1_pct"] = ((bars["close"] - bars["s1_price"]) / close_safe * 100.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    bars["distance_to_r1_pct"] = ((bars["r1_price"] - bars["close"]) / close_safe * 100.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    bars["stop_loss"] = bars["close"] - bars["atr_14"] * 2.0
    bars["target1"] = bars["close"] + bars["atr_14"] * 2.5
    bars["target2"] = bars["close"] + bars["atr_14"] * 3.5
    risk = bars["close"] - bars["stop_loss"]
    reward = bars["target1"] - bars["close"]
    bars["rr_ratio"] = np.where(risk > 0, reward / risk, np.nan)
    bars["rr_ratio"] = pd.to_numeric(bars["rr_ratio"], errors="coerce").fillna(0.0)

    bars["momentum_score"] = bars["weekly_return"].fillna(0.0).clip(-15.0, 15.0)
    bars["poc_distance_pct"] = trend_pct.fillna(0.0).clip(-25.0, 25.0)
    bars["in_value_area"] = bars["poc_distance_pct"].abs() <= 2.5
    bars["spread_pct"] = bars["atr_percent"].mul(0.05).clip(lower=0.02, upper=1.25)

    bars["regime"] = np.where(
        bars["weekly_return"].fillna(0.0) >= 3.0,
        "MOMENTUM",
        np.where(bars["weekly_return"].fillna(0.0) <= -3.0, "MEAN_REVERSION", "NEUTRAL"),
    )
    bars["action_plan"] = np.where(
        bars["bullish_score"] >= 6.0,
        "bullish_momentum",
        np.where(bars["bullish_score"] >= 4.0, "watch_pullback", "hold"),
    )

    # Compatibility fields used by legacy research backtesters.
    bars["actual_open"] = bars["open"]
    bars["actual_high"] = bars["high"]
    bars["actual_low"] = bars["low"]
    bars["actual_close"] = bars["close"]
    bars["s1_tested"] = bars["actual_low"] <= bars["s1_price"]
    bars["s1_held"] = bars["actual_close"] >= bars["s1_price"]
    bars["r1_tested"] = bars["actual_high"] >= bars["r1_price"]
    bars["r1_rejected"] = bars["actual_close"] < bars["r1_price"]
    bars["r2_tested"] = bars["actual_high"] >= bars["r2_price"]
    bars["r2_rejected"] = bars["actual_close"] < bars["r2_price"]

    bars["prediction_date"] = pd.to_datetime(bars["date"]).dt.date
    bars["close_price"] = bars["close"].astype(float)

    out = bars[(bars["prediction_date"] >= start_d) & (bars["prediction_date"] <= end_d)].copy()
    if out.empty:
        return out

    fields = [
        "ticker",
        "prediction_date",
        "close_price",
        "regime",
        "bullish_score",
        "rr_ratio",
        "s1_price",
        "s1_strength",
        "s2_price",
        "s2_strength",
        "r1_price",
        "r1_strength",
        "r2_price",
        "r2_strength",
        "atr_14",
        "atr_percent",
        "distance_to_s1_pct",
        "distance_to_r1_pct",
        "spread_pct",
        "momentum_score",
        "poc_distance_pct",
        "in_value_area",
        "action_plan",
        "stop_loss",
        "target1",
        "target2",
        "actual_open",
        "actual_high",
        "actual_low",
        "actual_close",
        "s1_tested",
        "s1_held",
        "r1_tested",
        "r1_rejected",
        "r2_tested",
        "r2_rejected",
    ]
    return out[fields].sort_values(["prediction_date", "ticker"]).reset_index(drop=True)


def load_daily_bars(
    daily_features_path: Path,
    start_date,
    end_date,
    symbols: Optional[Sequence[str]] = None,
    enforce_freshness_gate: bool = True,
) -> pd.DataFrame:
    cfg = get_config().data
    if enforce_freshness_gate and daily_features_path == get_primary_parquet_path():
        ensure_core_market_data_ready(
            max_staleness_days=cfg.max_staleness_days,
            bootstrap_from_h5_enabled=cfg.bootstrap_from_h5_enabled,
            fail_fast=cfg.fail_fast_on_missing_core_data,
            parquet_path=daily_features_path,
        )

    symbols = _sanitize_symbols(symbols)
    start_d = _to_date(start_date)
    end_d = _to_date(end_date)

    if not daily_features_path.exists():
        logger.warning(f"Daily features parquet not found: {daily_features_path}")
        return pd.DataFrame()

    try:
        import duckdb

        where = "date >= ? AND date <= ?"
        if symbols:
            tickers = ", ".join([f"'{s}'" for s in symbols])
            where += f" AND ticker IN ({tickers})"

        query = f"""
        WITH base AS (
            SELECT
                UPPER(ticker) AS ticker,
                CAST(Date AS DATE) AS date,
                "Open" AS open,
                "High" AS high,
                "Low" AS low,
                "Close" AS close,
                "Adj Close" AS adj_close,
                Volume AS volume
            FROM read_parquet('{daily_features_path.as_posix()}')
        )
        SELECT *
        FROM base
        WHERE {where}
        ORDER BY ticker, date
        """

        conn = duckdb.connect(database=":memory:")
        df = conn.execute(query, [str(start_d), str(end_d)]).df()
        conn.close()
    except Exception as exc:
        logger.warning(f"DuckDB load failed ({exc}); falling back to pandas read_parquet (slow)")
        cols = ["ticker", "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        df = pd.read_parquet(daily_features_path, columns=cols)
        df = df.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df[(df["date"] >= start_d) & (df["date"] <= end_d)]
        if symbols:
            df = df[df["ticker"].isin(set(symbols))]
        df = df.sort_values(["ticker", "date"])

    if df.empty:
        return df

    if "adj_close" in df.columns and df["adj_close"].notna().any():
        # Fix: use np.where for Series.fillna(Series) to bypass namespace bug
        df["close"] = pd.Series(
            np.where(df["adj_close"].isna(), df["close"], df["adj_close"]),
            index=df.index,
        )

    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def load_benchmark_close(
    daily_features_path: Path,
    benchmark_symbol: str,
    start_date,
    end_date,
) -> pd.Series:
    df = load_daily_bars(
        daily_features_path=daily_features_path,
        start_date=start_date,
        end_date=end_date,
        symbols=[benchmark_symbol],
    )
    if df.empty:
        return pd.Series(dtype=float)
    return df.set_index("date")["close"].astype(float).sort_index()
