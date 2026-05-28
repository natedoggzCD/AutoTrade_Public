"""
Alpha: Volume-Price Divergence
==============================
Detects bullish and bearish divergence between price trend and OBV trend.

Bullish divergence: price making lower lows but OBV making higher lows
  -> accumulation underway, potential reversal up
Bearish divergence: price making higher highs but OBV making lower highs
  -> distribution underway, potential reversal down

Uses existing DownDay parquet columns (Close, Volume) so no new data needed.
Maps to SignalFamily.MEAN_REVERSION.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_LOOKBACK = 20       # bars for slope regression
DEFAULT_MIN_DIVERGENCE = 0.5  # min absolute divergence z-score to generate signal
DEFAULT_VOLUME_CONFIRM_RATIO = 1.2  # recent volume must be >= this * avg


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def compute_slope(series: pd.Series, window: int) -> pd.Series:
    """Rolling OLS slope (normalized by mean of series in window)."""
    def _slope(arr):
        if len(arr) < window or np.isnan(arr).any():
            return np.nan
        x = np.arange(len(arr))
        mean_val = np.abs(arr).mean()
        if mean_val == 0:
            return 0.0
        slope = np.polyfit(x, arr, 1)[0]
        return slope / mean_val  # normalize

    return series.rolling(window, min_periods=window).apply(_slope, raw=True)


def score_divergence(
    df: pd.DataFrame,
    lookback: int = DEFAULT_LOOKBACK,
    min_divergence: float = DEFAULT_MIN_DIVERGENCE,
    volume_confirm_ratio: float = DEFAULT_VOLUME_CONFIRM_RATIO,
) -> pd.DataFrame:
    """
    Score volume-price divergence for each ticker in a multi-ticker DataFrame.

    Expects columns: ticker, date, close, volume (and optionally avg_volume).
    Returns DataFrame with added columns:
      - obv: On-Balance Volume
      - price_slope: normalized rolling price slope
      - obv_slope: normalized rolling OBV slope
      - divergence_raw: obv_slope - price_slope (positive = bullish divergence)
      - divergence_zscore: cross-sectional z-score of divergence
      - volume_confirmed: bool, recent volume >= threshold
      - divergence_score: 0-100 composite score (higher = stronger bullish divergence)
      - divergence_signal: 'bullish', 'bearish', or 'none'
    """
    if df.empty:
        return df

    base = df.sort_values(["ticker", "date"]).copy()
    close = pd.to_numeric(base.get("close"), errors="coerce")
    volume = pd.to_numeric(base.get("volume"), errors="coerce").fillna(0.0)
    ticker = base["ticker"]
    dates = base["date"]

    direction = np.sign(close.groupby(ticker).diff()).fillna(0.0)
    obv = (direction * volume).groupby(ticker).cumsum()
    price_slope = close.groupby(ticker).transform(lambda s: compute_slope(s, lookback))
    obv_slope = obv.groupby(ticker).transform(lambda s: compute_slope(s, lookback))
    divergence_raw = obv_slope - price_slope

    div_mean = divergence_raw.groupby(dates).transform("mean")
    div_std = divergence_raw.groupby(dates).transform("std").replace(0, np.nan)
    divergence_zscore = ((divergence_raw - div_mean) / div_std).fillna(0.0)

    if "avg_volume" in base.columns:
        avg_vol = pd.to_numeric(base["avg_volume"], errors="coerce")
    else:
        avg_vol = volume.groupby(ticker).transform(
            lambda s: s.rolling(lookback, min_periods=5).mean()
        )
    volume_confirmed = volume >= (avg_vol * float(volume_confirm_ratio))

    z_clipped = divergence_zscore.clip(-3, 3)
    base_score = ((z_clipped + 3) / 6) * 100
    vol_bonus = np.where(volume_confirmed, 10.0, 0.0)
    classic_bullish_bonus = np.where((price_slope < 0) & (obv_slope > 0), 15.0, 0.0)
    divergence_score = (base_score + vol_bonus + classic_bullish_bonus).clip(0, 100)
    divergence_signal = np.where(
        divergence_zscore >= min_divergence,
        "bullish",
        np.where(divergence_zscore <= -min_divergence, "bearish", "none"),
    )

    derived = pd.DataFrame(
        {
            "obv": obv,
            "price_slope": price_slope,
            "obv_slope": obv_slope,
            "divergence_raw": divergence_raw,
            "divergence_zscore": divergence_zscore,
            "volume_confirmed": volume_confirmed,
            "divergence_score": divergence_score,
            "divergence_signal": divergence_signal,
        },
        index=base.index,
    )
    return pd.concat([base, derived], axis=1)


def get_divergence_candidates(
    price_df: pd.DataFrame,
    lookback: int = DEFAULT_LOOKBACK,
    min_score: float = 65.0,
    max_candidates: int = 50,
) -> List[Dict]:
    """
    Run divergence scoring on price data and return top bullish divergence candidates.

    Args:
        price_df: DataFrame with ticker, date, close, volume columns
        lookback: slope lookback window
        min_score: minimum divergence_score to include
        max_candidates: max results to return

    Returns:
        List of candidate dicts with divergence metadata attached
    """
    if price_df.empty:
        return []

    scored = score_divergence(price_df, lookback=lookback)
    if scored.empty:
        return []

    # Take latest row per ticker
    latest_idx = scored.groupby("ticker")["date"].idxmax()
    latest = scored.loc[latest_idx].copy()

    # Filter to bullish divergence with minimum score
    bullish = latest[
        (latest["divergence_signal"] == "bullish") &
        (latest["divergence_score"] >= min_score)
    ].sort_values("divergence_score", ascending=False).head(max_candidates)

    if bullish.empty:
        logger.info("[VolDivergence] No bullish divergence candidates above score %.1f", min_score)
        return []

    results = []
    for _, row in bullish.iterrows():
        results.append({
            "ticker": row["ticker"],
            "divergence_score": float(row["divergence_score"]),
            "divergence_zscore": float(row["divergence_zscore"]),
            "price_slope": float(row.get("price_slope", 0)),
            "obv_slope": float(row.get("obv_slope", 0)),
            "volume_confirmed": bool(row.get("volume_confirmed", False)),
            "divergence_signal": str(row.get("divergence_signal", "none")),
            "alpha_source": "volume_divergence",
            "signal_family": "mean_reversion",
        })

    logger.info("[VolDivergence] Found %d bullish divergence candidates (top score: %.1f)",
                len(results), results[0]["divergence_score"] if results else 0)
    return results
