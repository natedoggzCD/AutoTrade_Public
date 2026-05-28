"""
Mean Reversion / Reversal Features Module
========================================

Pure, side-effect-free mean reversion transforms.
Includes RSI, Bollinger deviation, and capitulation signals.
"""

from typing import Dict, List
import numpy as np
import pandas as pd

from autotrade.feature_engineering.schemas import FeatureFamily, FeatureMetadata
from autotrade.feature_engineering.interfaces import FeatureBuilder
from autotrade.feature_engineering.technical import (
    calculate_rsi,
    calculate_bollinger_bands,
)


def calculate_rsi_reversal(close: pd.Series, window: int = 2) -> pd.Series:
    """Calculate short-term RSI for reversal signals."""
    return calculate_rsi(close, window)


def calculate_bollinger_deviation(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.Series:
    """Calculate deviation from Bollinger Bands (normalized position)."""
    bb = calculate_bollinger_bands(close, window, num_std)
    position = bb["bb_position"]
    deviation = (position - 0.5) * 2
    return deviation


def calculate_atr_stretch(
    close: pd.Series, high: pd.Series, low: pd.Series, window: int = 14
) -> pd.Series:
    """Calculate short-term stretch vs ATR (potential reversal candidate)."""
    from autotrade.feature_engineering.volatility_volume import calculate_atr

    atr = calculate_atr(high, low, close, window)
    stretch = (close - close.shift(1)) / atr
    return stretch


def calculate_capitulation_score(
    close: pd.Series, volume: pd.Series, lookback: int = 20
) -> pd.DataFrame:
    """Calculate capitulation proxy (large down move + relative volume)."""
    returns = close.pct_change()
    volume_ma = volume.rolling(window=lookback, min_periods=lookback).mean()
    vol_ratio = volume / volume_ma

    large_drop = returns < returns.rolling(
        window=lookback, min_periods=lookback
    ).quantile(0.1)
    high_volume = vol_ratio > 1.5

    capitulation = (large_drop & high_volume).astype(int)
    capitulation_strength = (returns.abs() * vol_ratio).rolling(window=lookback).mean()

    return pd.DataFrame(
        {
            "capitulation_signal": capitulation,
            "capitulation_strength": capitulation_strength,
        }
    )


def calculate_mean_reversion_zscore(close: pd.Series, window: int = 20) -> pd.Series:
    """Calculate z-score of price relative to moving average."""
    ma = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std()
    zscore = (close - ma) / std.replace(0, np.nan)
    return zscore


def calculate_gap_reversal(
    open_price: pd.Series, close: pd.Series, prev_close: pd.Series
) -> pd.Series:
    """Calculate gap reversal potential."""
    gap = (open_price - prev_close) / prev_close
    intraday_return = (close - open_price) / open_price
    reversal_potential = -gap * intraday_return
    return reversal_potential


def calculate_support_resistance_proximity(
    close: pd.Series, low: pd.Series, high: pd.Series, window: int = 20
) -> pd.Series:
    """Calculate proximity to support (lows) vs resistance (highs)."""
    rolling_low = low.rolling(window=window, min_periods=window).min()
    rolling_high = high.rolling(window=window, min_periods=window).max()

    dist_to_support = (close - rolling_low) / (rolling_high - rolling_low)
    return dist_to_support


class ReversalFeatureBuilder:
    """Feature builder for mean reversion indicators."""

    @property
    def family(self) -> FeatureFamily:
        return FeatureFamily.MEAN_REVERSION

    @property
    def required_columns(self) -> List[str]:
        return ["close"]

    def can_compute(self, df: pd.DataFrame) -> bool:
        return all(col in df.columns for col in self.required_columns)

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        result = df.copy()

        rsi_windows = params.get("rsi_short_windows", [2, 3])
        for window in rsi_windows:
            if window <= len(df):
                result[f"rsi_{window}"] = calculate_rsi_reversal(df["close"], window)

        if len(df) >= 20:
            bb_window = params.get("bb_window", 20)
            result["bollinger_deviation"] = calculate_bollinger_deviation(
                df["close"], bb_window
            )

        if all(col in df.columns for col in ["close", "high", "low"]):
            atr_window = params.get("atr_window", 14)
            result["atr_stretch"] = calculate_atr_stretch(
                df["close"], df["high"], df["low"], atr_window
            )

            result["sr_proximity"] = calculate_support_resistance_proximity(
                df["close"], df["low"], df["high"], params.get("sr_window", 20)
            )

        if "volume" in df.columns:
            capitulation = calculate_capitulation_score(
                df["close"], df["volume"], params.get("capitulation_lookback", 20)
            )
            for col in capitulation.columns:
                result[col] = capitulation[col]

        if len(df) >= 20:
            result["mean_reversion_zscore"] = calculate_mean_reversion_zscore(
                df["close"], params.get("zscore_window", 20)
            )

        if all(col in df.columns for col in ["open", "close"]) and len(df) > 1:
            result["gap_reversal"] = calculate_gap_reversal(
                df["open"], df["close"], df["close"].shift(1)
            )

        return result

    def get_metadata(self) -> Dict[str, FeatureMetadata]:
        return {
            "rsi_2": FeatureMetadata(
                name="rsi_2",
                family=FeatureFamily.MEAN_REVERSION,
                description="Relative Strength Index 2 periods",
                unit="index",
                is_leakage_risk=False,
                required_columns=["close"],
                window=2,
            ),
            "bollinger_deviation": FeatureMetadata(
                name="bollinger_deviation",
                family=FeatureFamily.MEAN_REVERSION,
                description="Bollinger Band deviation from center",
                unit="score",
                is_leakage_risk=False,
                required_columns=["close"],
                window=20,
            ),
            "capitulation_score": FeatureMetadata(
                name="capitulation_score",
                family=FeatureFamily.MEAN_REVERSION,
                description="Capitulation signal (large drop + volume)",
                unit="score",
                is_leakage_risk=False,
                required_columns=["close", "volume"],
                window=20,
            ),
        }


__all__ = [
    "calculate_rsi_reversal",
    "calculate_bollinger_deviation",
    "calculate_atr_stretch",
    "calculate_capitulation_score",
    "calculate_mean_reversion_zscore",
    "calculate_gap_reversal",
    "calculate_support_resistance_proximity",
    "ReversalFeatureBuilder",
]
