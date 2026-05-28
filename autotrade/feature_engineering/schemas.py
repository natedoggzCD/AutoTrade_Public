"""
Feature Engineering Schemas
============================

Typed dataclasses defining the feature contract layer.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set
import pandas as pd


class FeatureFamily(str, Enum):
    """Alpha family categories for feature classification."""

    TS_MOMENTUM = "ts_momentum"
    XS_MOMENTUM = "xs_momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    PULLBACK = "pullback"
    TREND = "trend"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    REGIME = "regime"


@dataclass
class FeatureRequest:
    """Request specification for feature computation."""

    symbols: List[str]
    start_date: str
    end_date: str
    families: List[FeatureFamily] = field(
        default_factory=lambda: [FeatureFamily.TS_MOMENTUM]
    )
    primary_timeframe: str = "1d"
    secondary_timeframes: List[str] = field(default_factory=lambda: ["1h"])
    lookback_bars: int = 252
    min_history_bars: int = 80
    as_of_date: Optional[str] = None

    def __post_init__(self):
        if not self.symbols:
            raise ValueError("symbols cannot be empty")
        if self.as_of_date and self.as_of_date > self.end_date:
            raise ValueError("as_of_date cannot be after end_date")


@dataclass
class FeatureMetadata:
    """Metadata about a computed feature."""

    name: str
    family: FeatureFamily
    description: str
    unit: str
    is_leakage_risk: bool = False
    required_columns: List[str] = field(default_factory=list)
    window: Optional[int] = None

    def validate_column_requirements(self, available_columns: Set[str]) -> bool:
        """Check if required columns are available for this feature."""
        return all(col in available_columns for col in self.required_columns)


@dataclass
class FeatureFrame:
    """Container for computed features with metadata."""

    df: pd.DataFrame
    symbols: List[str]
    timestamp: datetime
    features: Dict[str, FeatureMetadata] = field(default_factory=dict)
    primary_timeframe: str = "1d"
    secondary_timeframes: List[str] = field(default_factory=list)
    source: str = "unknown"

    def get_feature_columns(self) -> List[str]:
        """Return list of feature column names."""
        return [
            c
            for c in self.df.columns
            if c
            not in (
                "symbol",
                "date",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            )
        ]

    def get_feature(self, name: str) -> Optional[pd.Series]:
        """Get a specific feature series by name."""
        if name in self.df.columns:
            return self.df[name]
        return None

    def filter_by_symbol(self, symbol: str) -> pd.DataFrame:
        """Filter features to a single symbol."""
        return self.df[self.df["symbol"] == symbol].copy()

    def as_of(self, date: str) -> pd.DataFrame:
        """Get feature values as of a specific date (no future leakage)."""
        return self.df[self.df["date"] <= date].copy()


@dataclass
class FeaturePipelineReport:
    """Report summarizing feature pipeline execution."""

    timestamp: datetime
    request: FeatureRequest
    features_computed: List[str]
    features_failed: List[str] = field(default_factory=list)
    symbols_processed: int = 0
    bars_used: int = 0
    computation_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    leakage_check_passed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "request": {
                "symbols": self.request.symbols,
                "families": [f.value for f in self.request.families],
                "start_date": self.request.start_date,
                "end_date": self.request.end_date,
            },
            "features_computed": self.features_computed,
            "features_failed": self.features_failed,
            "symbols_processed": self.symbols_processed,
            "bars_used": self.bars_used,
            "computation_time_ms": self.computation_time_ms,
            "warnings": self.warnings,
            "leakage_check_passed": self.leakage_check_passed,
        }


CANONICAL_FEATURE_NAMES: Dict[str, str] = {
    "return_12_1": "return_12_1",
    "return_6_1": "return_6_1",
    "return_1_1": "return_1_1",
    "sma5": "sma5",
    "sma20": "sma20",
    "sma50": "sma50",
    "sma200": "sma200",
    "sma5_slope": "sma5_slope",
    "sma20_slope": "sma20_slope",
    "sma50_slope": "sma50_slope",
    "sma200_slope": "sma200_slope",
    "ema20": "ema20",
    "ema50": "ema50",
    "vwap": "vwap",
    "adx": "adx",
    "plus_di": "plus_di",
    "minus_di": "minus_di",
    "rsi_2": "rsi_2",
    "rsi_3": "rsi_3",
    "rsi_14": "rsi_14",
    "bb_upper": "bb_upper",
    "bb_middle": "bb_middle",
    "bb_lower": "bb_lower",
    "bb_width": "bb_width",
    "bb_position": "bb_position",
    "atr_14": "atr_14",
    "atr_percent": "atr_percent",
    "macd": "macd",
    "macd_signal": "macd_signal",
    "macd_hist": "macd_hist",
    "volume_ratio": "volume_ratio",
    "volume_ma_20": "volume_ma_20",
    "donchian_upper": "donchian_upper",
    "donchian_lower": "donchian_lower",
    "donchian_width": "donchian_width",
    "squeeze": "squeeze",
    "efficiency_ratio": "efficiency_ratio",
    "rank_6_1": "rank_6_1",
    "rank_12_1": "rank_12_1",
    "sector_rank_6_1": "sector_rank_6_1",
    "cross_sectional_momentum": "cross_sectional_momentum",
    "bollinger_deviation": "bollinger_deviation",
    "capitulation_score": "capitulation_score",
    "pullback_recovery": "pullback_recovery",
    "trend_strength": "trend_strength",
    "regime": "regime",
    "correlation_60": "correlation_60",
    "spread_zscore": "spread_zscore",
}

ALIAS_MAP: Dict[str, str] = {
    "sma_5": "sma5",
    "sma_20": "sma20",
    "sma_50": "sma50",
    "sma_200": "sma200",
    "sma5_slope": "sma5_slope",
    "sma20_slope": "sma20_slope",
    "sma50_slope": "sma50_slope",
    "sma200_slope": "sma200_slope",
    "rsi": "rsi_14",
    "rsi2": "rsi_2",
    "rsi3": "rsi_3",
    "atr": "atr_14",
    "atr_pct": "atr_percent",
    "bb_upper_band": "bb_upper",
    "bb_lower_band": "bb_lower",
    "bb_middle_band": "bb_middle",
    "bb_bandwidth": "bb_width",
    "bollinger_band_width": "bb_width",
    "bollinger_band_position": "bb_position",
    "dmi_adx": "adx",
    "dmi_plus": "plus_di",
    "dmi_minus": "minus_di",
    "volume_ratio_20": "volume_ratio",
    "ma_20_volume": "volume_ma_20",
    "donchian_channel_upper": "donchian_upper",
    "donchian_channel_lower": "donchian_lower",
    "donchian_channel_width": "donchian_width",
    "efficiency_ratio": "efficiency_ratio",
    "er": "efficiency_ratio",
    "rank_return_6_1": "rank_6_1",
    "rank_return_12_1": "rank_12_1",
    "cross_sectional_return": "cross_sectional_momentum",
    "xs_momentum": "cross_sectional_momentum",
    "bb_deviation": "bollinger_deviation",
    "volume_surge": "capitulation_score",
    "ema20_recovery": "pullback_recovery",
    "vwap_recovery": "pullback_recovery",
    "trend_score": "trend_strength",
    "market_regime": "regime",
    "rolling_correlation": "correlation_60",
    "spread_z_score": "spread_zscore",
}


def canonical_name(name: str) -> str:
    """Convert feature alias to canonical snake_case name."""
    normalized = name.lower().strip()
    if normalized in CANONICAL_FEATURE_NAMES:
        return CANONICAL_FEATURE_NAMES[normalized]
    if normalized in ALIAS_MAP:
        return ALIAS_MAP[normalized]
    return normalized
