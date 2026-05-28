"""
Pairs Trading Features Module
=============================

Pure, side-effect-free pairs trading transforms.
Includes rolling correlation, spread z-score, and cointegration interface.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from autotrade.feature_engineering.schemas import FeatureFamily, FeatureMetadata
from autotrade.feature_engineering.interfaces import FeatureBuilder


def calculate_rolling_correlation(
    series1: pd.Series, series2: pd.Series, window: int = 60
) -> pd.Series:
    """Calculate rolling correlation between two series."""
    return series1.rolling(window=window, min_periods=window).corr(series2)


def calculate_rolling_spread(
    series1: pd.Series, series2: pd.Series, window: int = 60
) -> pd.Series:
    """Calculate rolling spread (difference) between two series."""
    return series1 - series2


def calculate_spread_zscore(
    series1: pd.Series, series2: pd.Series, window: int = 60
) -> pd.DataFrame:
    """Calculate z-score of the spread."""
    spread = calculate_rolling_spread(series1, series2)
    spread_ma = spread.rolling(window=window, min_periods=window).mean()
    spread_std = spread.rolling(window=window, min_periods=window).std()

    zscore = (spread - spread_ma) / spread_std.replace(0, np.nan)

    return pd.DataFrame(
        {"spread": spread, "spread_ma": spread_ma, "spread_zscore": zscore}
    )


def calculate_hedge_ratio(
    series1: pd.Series, series2: pd.Series, window: int = 60
) -> pd.Series:
    """Calculate rolling hedge ratio using OLS regression."""

    def rolling_hedge_ratio(x, y):
        if len(x) < window or len(y) < window:
            return np.nan
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < window // 2:
            return np.nan
        x_valid = x[mask]
        y_valid = y[mask]
        try:
            slope, _, _, _, _ = stats.linregress(y_valid, x_valid)
            return slope
        except:
            return np.nan

    return pd.Series(
        [
            rolling_hedge_ratio(
                series1.iloc[max(0, i - window) : i + 1].values,
                series2.iloc[max(0, i - window) : i + 1].values,
            )
            for i in range(len(series1))
        ],
        index=series1.index,
    )


def calculate_cointegration_test(
    series1: pd.Series, series2: pd.Series, window: int = 60
) -> pd.DataFrame:
    """Interface for cointegration testing (placeholder for full implementation)."""
    correlation = calculate_rolling_correlation(series1, series2, window)

    spread_zscore = calculate_spread_zscore(series1, series2, window)["spread_zscore"]

    return pd.DataFrame(
        {
            "correlation": correlation,
            "cointegration_score": spread_zscore,
            "cointegration_signal": np.where(
                spread_zscore > 2, -1, np.where(spread_zscore < -2, 1, 0)
            ),
        }
    )


def calculate_pairs_metrics(
    series1: pd.Series,
    series2: pd.Series,
    zscore_window: int = 60,
    min_correlation: float = 0.7,
) -> pd.DataFrame:
    """Calculate comprehensive pairs trading metrics."""
    correlation = calculate_rolling_correlation(series1, series2, zscore_window)
    zscore_df = calculate_spread_zscore(series1, series2, zscore_window)

    return pd.DataFrame(
        {
            "correlation": correlation,
            "spread": zscore_df["spread"],
            "spread_zscore": zscore_df["spread_zscore"],
            "pairs_signal": np.where(
                correlation < min_correlation,
                0,
                np.where(
                    zscore_df["spread_zscore"] > 2,
                    -1,
                    np.where(zscore_df["spread_zscore"] < -2, 1, 0),
                ),
            ),
        }
    )


class PairsFeatureBuilder:
    """Feature builder for pairs trading indicators."""

    @property
    def family(self) -> FeatureFamily:
        return FeatureFamily.PAIRS

    @property
    def required_columns(self) -> List[str]:
        return ["close"]

    def can_compute(self, df: pd.DataFrame) -> bool:
        return all(col in df.columns for col in self.required_columns)

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        return df

    def compute_for_pair(
        self, series1: pd.Series, series2: pd.Series, **params
    ) -> pd.DataFrame:
        """Compute pairs features for two specific series."""
        zscore_window = params.get("zscore_window", 60)
        min_correlation = params.get("min_correlation", 0.7)

        return calculate_pairs_metrics(series1, series2, zscore_window, min_correlation)

    def get_metadata(self) -> Dict[str, FeatureMetadata]:
        return {
            "correlation_60": FeatureMetadata(
                name="correlation_60",
                family=FeatureFamily.PAIRS,
                description="Rolling 60-period correlation",
                unit="correlation",
                is_leakage_risk=False,
                required_columns=["close"],
                window=60,
            ),
            "spread_zscore": FeatureMetadata(
                name="spread_zscore",
                family=FeatureFamily.PAIRS,
                description="Spread z-score",
                unit="zscore",
                is_leakage_risk=False,
                required_columns=["close"],
                window=60,
            ),
        }


__all__ = [
    "calculate_rolling_correlation",
    "calculate_rolling_spread",
    "calculate_spread_zscore",
    "calculate_hedge_ratio",
    "calculate_cointegration_test",
    "calculate_pairs_metrics",
    "PairsFeatureBuilder",
]
