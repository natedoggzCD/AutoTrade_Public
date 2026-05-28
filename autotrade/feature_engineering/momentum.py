"""
Momentum Features Module
========================

Pure, side-effect-free momentum transforms.
Includes time-series and cross-sectional momentum.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from autotrade.feature_engineering.schemas import FeatureFamily, FeatureMetadata
from autotrade.feature_engineering.interfaces import FeatureBuilder


def calculate_returns(close: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate simple returns over specified periods."""
    return close.pct_change(periods=periods)


def calculate_excluded_returns(
    close: pd.Series, total_periods: int, exclude_recent: int = 21
) -> pd.Series:
    """Calculate returns excluding most recent bars (leakage prevention)."""
    if total_periods <= 0:
        raise ValueError("total_periods must be positive")
    if exclude_recent < 0:
        raise ValueError("exclude_recent must be non-negative")
    if exclude_recent == 0:
        return close.pct_change(periods=total_periods)

    numerator = close.shift(exclude_recent)
    denominator = close.shift(total_periods + exclude_recent)
    return (numerator / denominator) - 1.0


def calculate_ts_momentum(close: pd.Series, exclude_recent: int = 21) -> pd.DataFrame:
    """Calculate time-series momentum features (12-1, 6-1 returns)."""
    return_12_1 = calculate_excluded_returns(close, 12, exclude_recent)
    return_6_1 = calculate_excluded_returns(close, 6, exclude_recent)
    return_3_1 = calculate_excluded_returns(close, 3, exclude_recent)
    return_1_1 = close.pct_change(1)

    return pd.DataFrame(
        {
            "return_12_1": return_12_1,
            "return_6_1": return_6_1,
            "return_3_1": return_3_1,
            "return_1_1": return_1_1,
        }
    )


def calculate_xs_momentum(close: pd.Series, exclude_recent: int = 21) -> pd.DataFrame:
    """Calculate cross-sectional momentum (relative rank in universe)."""
    returns_6 = calculate_excluded_returns(close, 6, exclude_recent)
    returns_12 = calculate_excluded_returns(close, 12, exclude_recent)

    rank_6 = returns_6.rank(pct=True)
    rank_12 = returns_12.rank(pct=True)

    return pd.DataFrame(
        {
            "rank_6_1": rank_6,
            "rank_12_1": rank_12,
            "cross_sectional_momentum": (rank_6 + rank_12) / 2,
        }
    )


def calculate_sector_relative_momentum(
    close: pd.Series, sector_returns: pd.Series
) -> pd.Series:
    """Calculate momentum relative to sector."""
    stock_returns = close.pct_change(20)
    return stock_returns - sector_returns


def calculate_momentum_acceleration(
    close: pd.Series, short_window: int = 5, long_window: int = 20
) -> pd.Series:
    """Calculate momentum acceleration (rate of change of momentum)."""
    mom_short = close.pct_change(short_window)
    mom_long = close.pct_change(long_window)
    return mom_short - mom_long


def calculate_momentum_divergence(
    close: pd.Series, volume: pd.Series, window: int = 20
) -> pd.Series:
    """Calculate momentum vs volume divergence."""
    mom = close.pct_change(window)
    vol = volume.pct_change(window)
    return mom - vol


class MomentumFeatureBuilder:
    """Feature builder for momentum indicators."""

    @property
    def family(self) -> FeatureFamily:
        return FeatureFamily.TS_MOMENTUM

    @property
    def required_columns(self) -> List[str]:
        return ["close"]

    def can_compute(self, df: pd.DataFrame) -> bool:
        return all(col in df.columns for col in self.required_columns)

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        result = df.copy()

        exclude_recent = params.get("exclude_recent_bars", 21)

        ts_mom = calculate_ts_momentum(df["close"], exclude_recent)
        for col in ts_mom.columns:
            result[col] = ts_mom[col]

        if "volume" in df.columns:
            result["momentum_divergence"] = calculate_momentum_divergence(
                df["close"], df["volume"], params.get("divergence_window", 20)
            )

        result["momentum_acceleration"] = calculate_momentum_acceleration(
            df["close"], params.get("short_window", 5), params.get("long_window", 20)
        )

        return result

    def get_metadata(self) -> Dict[str, FeatureMetadata]:
        return {
            "return_12_1": FeatureMetadata(
                name="return_12_1",
                family=FeatureFamily.TS_MOMENTUM,
                description="12-bar return excluding most recent bar",
                unit="percent",
                is_leakage_risk=False,
                required_columns=["close"],
                window=12,
            ),
            "return_6_1": FeatureMetadata(
                name="return_6_1",
                family=FeatureFamily.TS_MOMENTUM,
                description="6-bar return excluding most recent bar",
                unit="percent",
                is_leakage_risk=False,
                required_columns=["close"],
                window=6,
            ),
            "rank_6_1": FeatureMetadata(
                name="rank_6_1",
                family=FeatureFamily.XS_MOMENTUM,
                description="Cross-sectional rank of 6-bar return",
                unit="rank",
                is_leakage_risk=False,
                required_columns=["close"],
            ),
        }


__all__ = [
    "calculate_returns",
    "calculate_excluded_returns",
    "calculate_ts_momentum",
    "calculate_xs_momentum",
    "calculate_sector_relative_momentum",
    "calculate_momentum_acceleration",
    "calculate_momentum_divergence",
    "MomentumFeatureBuilder",
]
