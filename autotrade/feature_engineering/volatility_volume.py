"""
Volatility and Volume Features Module
=====================================

Pure, side-effect-free volatility and volume transforms.
"""

from typing import Dict, List
import numpy as np
import pandas as pd

from autotrade.feature_engineering.schemas import FeatureFamily, FeatureMetadata
from autotrade.feature_engineering.interfaces import FeatureBuilder


def calculate_true_range(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Calculate True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def calculate_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Calculate Average True Range."""
    tr = calculate_true_range(high, low, close)
    return tr.rolling(window=window, min_periods=window).mean()


def calculate_atr_percent(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Calculate ATR as percentage of close price."""
    atr = calculate_atr(high, low, close, window)
    return (atr / close) * 100


def calculate_volume_ma(volume: pd.Series, window: int = 20) -> pd.Series:
    """Calculate volume moving average."""
    return volume.rolling(window=window, min_periods=window).mean()


def calculate_volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Calculate volume relative to moving average."""
    vol_ma = calculate_volume_ma(volume, window)
    return volume / vol_ma


def calculate_volume_percentile(volume: pd.Series, window: int = 60) -> pd.Series:
    """Calculate volume percentile over lookback window."""
    return volume.rolling(window=window, min_periods=window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume."""
    direction = np.sign(close.diff())
    obv = (direction * volume).cumsum()
    return obv


def calculate_vwap_volatility(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Calculate VWAP-based volatility."""
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return (
        typical_price.rolling(window=window).std() / vwap.rolling(window=window).mean()
    )


def calculate_price_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Calculate rolling price volatility (standard deviation of returns)."""
    returns = close.pct_change()
    return returns.rolling(window=window, min_periods=window).std() * np.sqrt(252)


class VolatilityVolumeFeatureBuilder:
    """Feature builder for volatility and volume indicators."""

    @property
    def family(self) -> FeatureFamily:
        return FeatureFamily.VOLATILITY

    @property
    def required_columns(self) -> List[str]:
        return ["close"]

    def can_compute(self, df: pd.DataFrame) -> bool:
        return all(col in df.columns for col in self.required_columns)

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        result = df.copy()

        atr_window = params.get("atr_window", 14)
        if all(col in df.columns for col in ["high", "low", "close"]):
            result["atr_14"] = calculate_atr(
                df["high"], df["low"], df["close"], atr_window
            )
            result["atr_percent"] = calculate_atr_percent(
                df["high"], df["low"], df["close"], atr_window
            )

        vol_window = params.get("volume_ma_window", 20)
        if "volume" in df.columns:
            result["volume_ma_20"] = calculate_volume_ma(df["volume"], vol_window)
            result["volume_ratio"] = calculate_volume_ratio(df["volume"], vol_window)
            result["volume_percentile"] = calculate_volume_percentile(
                df["volume"], params.get("volume_percentile_window", 60)
            )

        if all(col in df.columns for col in ["close", "volume"]):
            result["obv"] = calculate_obv(df["close"], df["volume"])

        if all(col in df.columns for col in ["high", "low", "close", "volume"]):
            result["vwap_volatility"] = calculate_vwap_volatility(
                df["high"],
                df["low"],
                df["close"],
                df["volume"],
                params.get("vwap_vol_window", 20),
            )

        result["price_volatility"] = calculate_price_volatility(
            df["close"], params.get("price_vol_window", 20)
        )

        return result

    def get_metadata(self) -> Dict[str, FeatureMetadata]:
        return {
            "atr_14": FeatureMetadata(
                name="atr_14",
                family=FeatureFamily.VOLATILITY,
                description="Average True Range 14 periods",
                unit="price",
                is_leakage_risk=False,
                required_columns=["high", "low", "close"],
                window=14,
            ),
            "atr_percent": FeatureMetadata(
                name="atr_percent",
                family=FeatureFamily.VOLATILITY,
                description="ATR as percentage of close price",
                unit="percent",
                is_leakage_risk=False,
                required_columns=["high", "low", "close"],
                window=14,
            ),
            "volume_ratio": FeatureMetadata(
                name="volume_ratio",
                family=FeatureFamily.VOLUME,
                description="Volume relative to 20-period average",
                unit="ratio",
                is_leakage_risk=False,
                required_columns=["volume"],
                window=20,
            ),
        }


__all__ = [
    "calculate_true_range",
    "calculate_atr",
    "calculate_atr_percent",
    "calculate_volume_ma",
    "calculate_volume_ratio",
    "calculate_volume_percentile",
    "calculate_obv",
    "calculate_vwap_volatility",
    "calculate_price_volatility",
    "VolatilityVolumeFeatureBuilder",
]
