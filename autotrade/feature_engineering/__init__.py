"""
Feature Engineering Module
===========================

Deterministic, leakage-safe transforms shared by scanners, signal generation, and backtesting.

Submodules:
- schemas: Typed dataclasses for feature contracts
- interfaces: Protocols for feature builders, stores, and pipelines
- technical: Pure technical indicator transforms
- volatility_volume: Volatility and volume-based features
- trend: Trend detection and measurement
- regime: Market regime features
- momentum: Time-series and cross-sectional momentum
- reversal: Mean reversion features
- breakout: Breakout and volatility expansion features
- pipeline: Multi-timeframe feature assembly
"""

from autotrade.feature_engineering.schemas import (
    FeatureRequest,
    FeatureFrame,
    FeatureMetadata,
    FeaturePipelineReport,
    FeatureFamily,
)
from autotrade.feature_engineering.interfaces import (
    FeatureBuilder,
    FeatureStore,
    FeaturePipeline,
    CANONICAL_FEATURE_NAMES,
)

from autotrade.feature_engineering.technical import (
    TechnicalFeatureBuilder,
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_stochastic,
    calculate_vwap,
    calculate_ma_slope,
)
from autotrade.feature_engineering.volatility_volume import (
    VolatilityVolumeFeatureBuilder,
    calculate_atr,
    calculate_atr_percent,
    calculate_volume_ma,
    calculate_volume_ratio,
    calculate_obv,
)
from autotrade.feature_engineering.trend import (
    TrendFeatureBuilder,
    calculate_ma_slope_trend,
    calculate_adx_trend,
    calculate_efficiency_ratio,
    calculate_trend_strength_composite,
    calculate_ma_stacking,
    calculate_ema_alignment,
)
from autotrade.feature_engineering.momentum import (
    MomentumFeatureBuilder,
    calculate_ts_momentum,
    calculate_xs_momentum,
    calculate_momentum_acceleration,
)
from autotrade.feature_engineering.reversal import (
    ReversalFeatureBuilder,
    calculate_rsi_reversal,
    calculate_bollinger_deviation,
    calculate_capitulation_score,
    calculate_mean_reversion_zscore,
)
from autotrade.feature_engineering.breakout import (
    BreakoutFeatureBuilder,
    calculate_donchian_channels,
    calculate_bollinger_squeeze,
    calculate_volume_expansion,
)
from autotrade.feature_engineering.regime import (
    RegimeFeatureBuilder,
    calculate_volatility_regime,
    calculate_trend_regime,
    calculate_regime_composite,
)
from autotrade.feature_engineering.pipeline import (
    FeaturePipeline,
    InMemoryFeatureStore,
    align_hourly_to_daily,
)
from autotrade.feature_engineering.adapters import (
    UniverseScannerFeatureAdapter,
    ScreenerV2FeatureAdapter,
    BacktestFeatureAdapter,
    DuckDBBacktestFeatureAdapter,
    get_universe_scanner_adapter,
    get_screener_v2_adapter,
    get_backtest_adapter,
)

__all__ = [
    "FeatureRequest",
    "FeatureFrame",
    "FeatureMetadata",
    "FeaturePipelineReport",
    "FeatureFamily",
    "FeatureBuilder",
    "FeatureStore",
    "FeaturePipeline",
    "CANONICAL_FEATURE_NAMES",
    "TechnicalFeatureBuilder",
    "VolatilityVolumeFeatureBuilder",
    "TrendFeatureBuilder",
    "MomentumFeatureBuilder",
    "ReversalFeatureBuilder",
    "BreakoutFeatureBuilder",
    "RegimeFeatureBuilder",
    "InMemoryFeatureStore",
    "align_hourly_to_daily",
    "calculate_sma",
    "calculate_ema",
    "calculate_rsi",
    "calculate_macd",
    "calculate_bollinger_bands",
    "calculate_atr",
    "calculate_atr_percent",
    "calculate_volume_ma",
    "calculate_volume_ratio",
    "calculate_obv",
    "calculate_ma_slope_trend",
    "calculate_adx_trend",
    "calculate_efficiency_ratio",
    "calculate_trend_strength_composite",
    "calculate_ma_stacking",
    "calculate_ema_alignment",
    "calculate_ts_momentum",
    "calculate_xs_momentum",
    "calculate_momentum_acceleration",
    "calculate_rsi_reversal",
    "calculate_bollinger_deviation",
    "calculate_capitulation_score",
    "calculate_mean_reversion_zscore",
    "calculate_donchian_channels",
    "calculate_bollinger_squeeze",
    "calculate_volume_expansion",
    "calculate_volatility_regime",
    "calculate_trend_regime",
    "calculate_regime_composite",
]
