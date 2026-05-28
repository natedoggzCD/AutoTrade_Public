"""
Regime Detection Features Module
=================================

Pure, side-effect-free market regime transforms.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from autotrade.feature_engineering.schemas import FeatureFamily, FeatureMetadata
from autotrade.feature_engineering.interfaces import FeatureBuilder


def calculate_volatility_regime(
    close: pd.Series,
    window: int = 20,
    low_threshold: float = 0.02,
    high_threshold: float = 0.05,
) -> pd.Series:
    """Calculate volatility regime (low/normal/high)."""
    returns = close.pct_change()
    volatility = returns.rolling(window=window, min_periods=window).std() * np.sqrt(252)

    regime = pd.cut(
        volatility,
        bins=[-np.inf, low_threshold, high_threshold, np.inf],
        labels=[0, 1, 2],
    )
    # Keep NaNs for warmup periods instead of forcing invalid integer cast.
    return regime.astype(float)


def calculate_trend_regime(
    close: pd.Series, short_window: int = 20, long_window: int = 50
) -> pd.Series:
    """Calculate trend regime based on MA relationship."""
    sma_short = close.rolling(window=short_window, min_periods=short_window).mean()
    sma_long = close.rolling(window=long_window, min_periods=long_window).mean()

    trend = np.where(sma_short > sma_long, 1, -1)
    return pd.Series(trend, index=close.index)


def calculate_range_regime(close: pd.Series, window: int = 20) -> pd.Series:
    """Calculate range-bound vs trending regime."""
    returns = close.pct_change()
    directional = returns.rolling(window=window, min_periods=window).sum()
    volatility = returns.abs().rolling(window=window, min_periods=window).sum()

    ratio = directional.abs() / volatility

    regime = pd.cut(ratio, bins=[-np.inf, 0.3, 0.7, np.inf], labels=[0, 1, 2])
    return regime.astype(float)


def calculate_market_breadth(close: pd.Series, lookback: int = 20) -> pd.DataFrame:
    """Calculate market breadth metrics."""
    returns = close.pct_change()
    advancing = (returns > 0).rolling(window=lookback, min_periods=lookback).sum()
    declining = (returns < 0).rolling(window=lookback, min_periods=lookback).sum()
    total = advancing + declining

    advance_decline_ratio = advancing / total.replace(0, np.nan)

    return pd.DataFrame(
        {
            "advancing": advancing,
            "declining": declining,
            "advance_decline_ratio": advance_decline_ratio,
        }
    )


def calculate_regime_composite(
    close: pd.Series,
    volatility_window: int = 20,
    trend_short: int = 20,
    trend_long: int = 50,
) -> pd.DataFrame:
    """Calculate composite regime indicator."""
    vol_regime = calculate_volatility_regime(close, volatility_window)
    trend_regime = calculate_trend_regime(close, trend_short, trend_long)
    range_regime = calculate_range_regime(close, volatility_window)

    return pd.DataFrame(
        {
            "volatility_regime": vol_regime,
            "trend_regime": trend_regime,
            "range_regime": range_regime,
            "regime_composite": (vol_regime + trend_regime + range_regime) / 3,
        }
    )


class RegimeFeatureBuilder:
    """Feature builder for regime detection."""

    @property
    def family(self) -> FeatureFamily:
        return FeatureFamily.REGIME

    @property
    def required_columns(self) -> List[str]:
        return ["close"]

    def can_compute(self, df: pd.DataFrame) -> bool:
        return all(col in df.columns for col in self.required_columns)

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        result = df.copy()

        vol_window = params.get("volatility_window", 20)
        trend_short = params.get("trend_short_window", 20)
        trend_long = params.get("trend_long_window", 50)

        regime = calculate_regime_composite(
            df["close"], vol_window, trend_short, trend_long
        )
        for col in regime.columns:
            result[col] = regime[col]

        return result

    def get_metadata(self) -> Dict[str, FeatureMetadata]:
        return {
            "regime": FeatureMetadata(
                name="regime",
                family=FeatureFamily.REGIME,
                description="Market regime composite indicator",
                unit="index",
                is_leakage_risk=False,
                required_columns=["close"],
                window=20,
            ),
            "volatility_regime": FeatureMetadata(
                name="volatility_regime",
                family=FeatureFamily.REGIME,
                description="Volatility regime (low/normal/high)",
                unit="category",
                is_leakage_risk=False,
                required_columns=["close"],
                window=20,
            ),
            "trend_regime": FeatureMetadata(
                name="trend_regime",
                family=FeatureFamily.REGIME,
                description="Trend regime (bullish/bearish)",
                unit="category",
                is_leakage_risk=False,
                required_columns=["close"],
                window=50,
            ),
        }


__all__ = [
    "calculate_volatility_regime",
    "calculate_trend_regime",
    "calculate_range_regime",
    "calculate_market_breadth",
    "calculate_regime_composite",
    "RegimeFeatureBuilder",
]
