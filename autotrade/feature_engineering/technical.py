"""
Technical Features Module
=========================

Pure, side-effect-free technical indicator transforms.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from autotrade.feature_engineering.schemas import FeatureFamily, FeatureMetadata
from autotrade.feature_engineering.interfaces import FeatureBuilder


def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return series.rolling(window=window, min_periods=window).mean()


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_fast = calculate_ema(series, fast_period)
    ema_slow = calculate_ema(series, slow_period)

    macd = ema_fast - ema_slow
    signal = calculate_ema(macd, signal_period)
    hist = macd - signal

    return pd.DataFrame({"macd": macd, "macd_signal": signal, "macd_hist": hist})


def calculate_bollinger_bands(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Calculate Bollinger Bands."""
    sma = calculate_sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()

    upper = sma + (std * num_std)
    lower = sma - (std * num_std)

    bandwidth = (upper - lower) / sma
    position = (series - lower) / (upper - lower)

    return pd.DataFrame(
        {
            "bb_upper": upper,
            "bb_middle": sma,
            "bb_lower": lower,
            "bb_width": bandwidth,
            "bb_position": position,
        }
    )


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """Calculate Stochastic Oscillator."""
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()

    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period, min_periods=d_period).mean()

    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


def calculate_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """Calculate Volume Weighted Average Price."""
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap


def calculate_ma_slope(
    close: pd.Series, ma_window: int = 20, slope_window: int = 5
) -> pd.Series:
    """Calculate slope of moving average."""
    ma = calculate_sma(close, ma_window)
    slope = ma.pct_change(periods=slope_window)
    return slope


class TechnicalFeatureBuilder:
    """Feature builder for technical indicators."""

    @property
    def family(self) -> FeatureFamily:
        return FeatureFamily.TREND

    @property
    def required_columns(self) -> List[str]:
        return ["close"]

    def can_compute(self, df: pd.DataFrame) -> bool:
        return all(col in df.columns for col in self.required_columns)

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        result = df.copy()

        sma_windows = params.get("sma_windows", [5, 20, 50, 200])
        for window in sma_windows:
            if window <= len(df):
                result[f"sma{window}"] = calculate_sma(df["close"], window)

        ema_windows = params.get("ema_windows", [20, 50])
        for window in ema_windows:
            if window <= len(df):
                result[f"ema{window}"] = calculate_ema(df["close"], window)

        rsi_windows = params.get("rsi_windows", [2, 3, 14])
        for window in rsi_windows:
            if window <= len(df):
                result[f"rsi_{window}"] = calculate_rsi(df["close"], window)

        # MACD and Bollinger Bands only need close.
        macd_result = calculate_macd(df["close"])
        for col in macd_result.columns:
            result[col] = macd_result[col]

        bb_result = calculate_bollinger_bands(df["close"])
        for col in bb_result.columns:
            result[col] = bb_result[col]

        if "high" in df.columns and "low" in df.columns:
            stoch_result = calculate_stochastic(df["high"], df["low"], df["close"])
            for col in stoch_result.columns:
                result[col] = stoch_result[col]

        if all(col in df.columns for col in ["high", "low", "close", "volume"]):
            result["vwap"] = calculate_vwap(
                df["high"], df["low"], df["close"], df["volume"]
            )

        ma_slope_windows = params.get("ma_slope_windows", [5, 20, 50, 200])
        for window in ma_slope_windows:
            if window <= len(df):
                result[f"sma{window}_slope"] = calculate_ma_slope(
                    df["close"], window, params.get("slope_window", 5)
                )

        return result

    def get_metadata(self) -> Dict[str, FeatureMetadata]:
        return {
            f"sma{w}": FeatureMetadata(
                name=f"sma{w}",
                family=FeatureFamily.TREND,
                description=f"Simple moving average {w} periods",
                unit="price",
                is_leakage_risk=False,
                required_columns=["close"],
                window=w,
            )
            for w in [5, 20, 50, 200]
        }


__all__ = [
    "calculate_sma",
    "calculate_ema",
    "calculate_rsi",
    "calculate_macd",
    "calculate_bollinger_bands",
    "calculate_stochastic",
    "calculate_vwap",
    "calculate_ma_slope",
    "TechnicalFeatureBuilder",
]
