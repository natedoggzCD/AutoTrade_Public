"""
Feature Engineering Interfaces
==============================

Protocol definitions for feature builders, stores, and pipelines.
"""

from typing import Protocol, List, Dict, Any, Optional, runtime_checkable
import pandas as pd

from autotrade.feature_engineering.schemas import (
    FeatureRequest,
    FeatureFrame,
    FeatureMetadata,
    FeaturePipelineReport,
    FeatureFamily,
    CANONICAL_FEATURE_NAMES,
)


@runtime_checkable
class FeatureBuilder(Protocol):
    """Protocol for computing individual features or feature families."""

    @property
    def family(self) -> FeatureFamily:
        """Return the feature family this builder implements."""
        ...

    @property
    def required_columns(self) -> List[str]:
        """Return list of required input columns."""
        ...

    def can_compute(self, df: pd.DataFrame) -> bool:
        """Check if dataframe has required columns for computation."""
        ...

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """Compute features and return dataframe with new columns."""
        ...

    def get_metadata(self) -> Dict[str, FeatureMetadata]:
        """Return metadata for all features this builder produces."""
        ...


@runtime_checkable
class FeatureStore(Protocol):
    """Protocol for storing and retrieving computed features."""

    def save(self, frame: FeatureFrame) -> None:
        """Save feature frame to store."""
        ...

    def load(self, request: FeatureRequest) -> Optional[FeatureFrame]:
        """Load feature frame from store if available."""
        ...

    def exists(self, request: FeatureRequest) -> bool:
        """Check if feature frame exists in store."""
        ...

    def invalidate(self, request: FeatureRequest) -> None:
        """Invalidate cached feature frame."""
        ...


@runtime_checkable
class FeaturePipeline(Protocol):
    """Protocol for orchestrating feature computation workflow."""

    def add_builder(self, builder: FeatureBuilder) -> None:
        """Add a feature builder to the pipeline."""
        ...

    def remove_builder(self, family: FeatureFamily) -> None:
        """Remove a feature builder from the pipeline."""
        ...

    def execute(self, request: FeatureRequest, data: pd.DataFrame) -> FeatureFrame:
        """Execute pipeline and return computed features."""
        ...

    def get_report(self) -> FeaturePipelineReport:
        """Get execution report from last run."""
        ...

    def validate_no_leakage(self, frame: FeatureFrame, as_of_date: str) -> bool:
        """Validate that features don't contain future information."""
        ...


def get_feature_metadata(name: str) -> Optional[FeatureMetadata]:
    """Get metadata for a canonical feature name."""
    metadata_map = {
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
        "sma5": FeatureMetadata(
            name="sma5",
            family=FeatureFamily.TREND,
            description="Simple moving average 5 periods",
            unit="price",
            is_leakage_risk=False,
            required_columns=["close"],
            window=5,
        ),
        "sma20": FeatureMetadata(
            name="sma20",
            family=FeatureFamily.TREND,
            description="Simple moving average 20 periods",
            unit="price",
            is_leakage_risk=False,
            required_columns=["close"],
            window=20,
        ),
        "sma50": FeatureMetadata(
            name="sma50",
            family=FeatureFamily.TREND,
            description="Simple moving average 50 periods",
            unit="price",
            is_leakage_risk=False,
            required_columns=["close"],
            window=50,
        ),
        "sma200": FeatureMetadata(
            name="sma200",
            family=FeatureFamily.TREND,
            description="Simple moving average 200 periods",
            unit="price",
            is_leakage_risk=False,
            required_columns=["close"],
            window=200,
        ),
        "sma5_slope": FeatureMetadata(
            name="sma5_slope",
            family=FeatureFamily.TREND,
            description="Slope of SMA5 over lookback window",
            unit="percent",
            is_leakage_risk=False,
            required_columns=["close"],
            window=20,
        ),
        "adx": FeatureMetadata(
            name="adx",
            family=FeatureFamily.TREND,
            description="Average Directional Index",
            unit="index",
            is_leakage_risk=False,
            required_columns=["high", "low", "close"],
            window=14,
        ),
        "rsi_2": FeatureMetadata(
            name="rsi_2",
            family=FeatureFamily.MEAN_REVERSION,
            description="Relative Strength Index 2 periods",
            unit="index",
            is_leakage_risk=False,
            required_columns=["close"],
            window=2,
        ),
        "rsi_14": FeatureMetadata(
            name="rsi_14",
            family=FeatureFamily.MEAN_REVERSION,
            description="Relative Strength Index 14 periods",
            unit="index",
            is_leakage_risk=False,
            required_columns=["close"],
            window=14,
        ),
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
        "bb_upper": FeatureMetadata(
            name="bb_upper",
            family=FeatureFamily.VOLATILITY,
            description="Bollinger Bands upper band",
            unit="price",
            is_leakage_risk=False,
            required_columns=["close"],
            window=20,
        ),
        "bb_lower": FeatureMetadata(
            name="bb_lower",
            family=FeatureFamily.VOLATILITY,
            description="Bollinger Bands lower band",
            unit="price",
            is_leakage_risk=False,
            required_columns=["close"],
            window=20,
        ),
        "bb_width": FeatureMetadata(
            name="bb_width",
            family=FeatureFamily.BREAKOUT,
            description="Bollinger Band width (bandwidth)",
            unit="percent",
            is_leakage_risk=False,
            required_columns=["close"],
            window=20,
        ),
        "bb_position": FeatureMetadata(
            name="bb_position",
            family=FeatureFamily.MEAN_REVERSION,
            description="Bollinger Band position (0-100)",
            unit="percent",
            is_leakage_risk=False,
            required_columns=["close"],
            window=20,
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
        "rank_6_1": FeatureMetadata(
            name="rank_6_1",
            family=FeatureFamily.XS_MOMENTUM,
            description="Cross-sectional rank of 6-bar return",
            unit="rank",
            is_leakage_risk=False,
            required_columns=["close"],
        ),
        "rank_12_1": FeatureMetadata(
            name="rank_12_1",
            family=FeatureFamily.XS_MOMENTUM,
            description="Cross-sectional rank of 12-bar return",
            unit="rank",
            is_leakage_risk=False,
            required_columns=["close"],
        ),
        "donchian_upper": FeatureMetadata(
            name="donchian_upper",
            family=FeatureFamily.BREAKOUT,
            description="Donchian channel upper band",
            unit="price",
            is_leakage_risk=False,
            required_columns=["high"],
            window=20,
        ),
        "donchian_lower": FeatureMetadata(
            name="donchian_lower",
            family=FeatureFamily.BREAKOUT,
            description="Donchian channel lower band",
            unit="price",
            is_leakage_risk=False,
            required_columns=["low"],
            window=20,
        ),
        "squeeze": FeatureMetadata(
            name="squeeze",
            family=FeatureFamily.BREAKOUT,
            description="Bollinger Band squeeze indicator",
            unit="boolean",
            is_leakage_risk=False,
            required_columns=["close"],
            window=120,
        ),
        "efficiency_ratio": FeatureMetadata(
            name="efficiency_ratio",
            family=FeatureFamily.TREND,
            description="Kaufman Efficiency Ratio",
            unit="ratio",
            is_leakage_risk=False,
            required_columns=["close"],
            window=20,
        ),
        "macd": FeatureMetadata(
            name="macd",
            family=FeatureFamily.TREND,
            description="MACD line",
            unit="price",
            is_leakage_risk=False,
            required_columns=["close"],
            window=26,
        ),
        "macd_signal": FeatureMetadata(
            name="macd_signal",
            family=FeatureFamily.TREND,
            description="MACD signal line",
            unit="price",
            is_leakage_risk=False,
            required_columns=["close"],
            window=26,
        ),
        "macd_hist": FeatureMetadata(
            name="macd_hist",
            family=FeatureFamily.TREND,
            description="MACD histogram",
            unit="price",
            is_leakage_risk=False,
            required_columns=["close"],
            window=26,
        ),
        "vwap": FeatureMetadata(
            name="vwap",
            family=FeatureFamily.TREND,
            description="Volume Weighted Average Price",
            unit="price",
            is_leakage_risk=False,
            required_columns=["high", "low", "close", "volume"],
        ),
        "ema20": FeatureMetadata(
            name="ema20",
            family=FeatureFamily.TREND,
            description="Exponential moving average 20 periods",
            unit="price",
            is_leakage_risk=False,
            required_columns=["close"],
            window=20,
        ),
        "ema50": FeatureMetadata(
            name="ema50",
            family=FeatureFamily.TREND,
            description="Exponential moving average 50 periods",
            unit="price",
            is_leakage_risk=False,
            required_columns=["close"],
            window=50,
        ),
    }
    return metadata_map.get(name)


__all__ = [
    "FeatureBuilder",
    "FeatureStore",
    "FeaturePipeline",
    "get_feature_metadata",
    "CANONICAL_FEATURE_NAMES",
]
