"""
Tests for Feature Engineering Module (Stage 2).
"""

import numpy as np
import pandas as pd

from autotrade.feature_engineering.breakout import (
    BreakoutFeatureBuilder,
    calculate_bollinger_squeeze,
)
from autotrade.feature_engineering.momentum import (
    MomentumFeatureBuilder,
    calculate_excluded_returns,
)
from autotrade.feature_engineering.pairs import PairsFeatureBuilder
from autotrade.feature_engineering.regime import RegimeFeatureBuilder
from autotrade.feature_engineering.reversal import ReversalFeatureBuilder
from autotrade.feature_engineering.technical import TechnicalFeatureBuilder
from autotrade.feature_engineering.trend import (
    TrendFeatureBuilder,
    calculate_adx_trend,
    calculate_efficiency_ratio,
    calculate_ma_slope_trend,
)
from autotrade.feature_engineering.volatility_volume import VolatilityVolumeFeatureBuilder


def _sample_ohlcv(n: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(12345)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = pd.Series(100 + np.cumsum(rng.normal(0.1, 1.0, size=n)), index=idx)
    high = close + rng.uniform(0.1, 1.5, size=n)
    low = close - rng.uniform(0.1, 1.5, size=n)
    open_ = close + rng.normal(0, 0.4, size=n)
    volume = pd.Series(rng.integers(100_000, 800_000, size=n), index=idx)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def test_phase2_builders_compute_expected_columns():
    df = _sample_ohlcv()
    builders = [
        TechnicalFeatureBuilder(),
        VolatilityVolumeFeatureBuilder(),
        TrendFeatureBuilder(),
        RegimeFeatureBuilder(),
        MomentumFeatureBuilder(),
        ReversalFeatureBuilder(),
        BreakoutFeatureBuilder(),
    ]

    result = df.copy()
    for builder in builders:
        result = builder.compute(result)

    expected = {
        "sma50",
        "ema20",
        "bb_width",
        "atr_14",
        "volume_ratio",
        "adx",
        "efficiency_ratio",
        "trend_strength",
        "return_12_1",
        "return_6_1",
        "rsi_2",
        "bollinger_deviation",
        "donchian_upper",
        "squeeze",
    }
    missing = sorted(expected - set(result.columns))
    assert not missing, f"Missing expected feature columns: {missing}"


def test_phase2_trend_orthogonal_methods_exist_and_are_parameterized():
    df = _sample_ohlcv()

    ma_slope = calculate_ma_slope_trend(df["close"], ma_window=30, slope_window=7)
    adx = calculate_adx_trend(df["high"], df["low"], df["close"], window=10)
    er = calculate_efficiency_ratio(df["close"], window=25)

    assert ma_slope.notna().sum() > 0
    assert adx.notna().sum() > 0
    assert er.notna().sum() > 0

    ma_slope_default = calculate_ma_slope_trend(df["close"])
    assert not ma_slope.equals(ma_slope_default)


def test_momentum_excluded_returns_matches_12_1_definition():
    close = pd.Series(np.arange(1.0, 101.0))
    r = calculate_excluded_returns(close, total_periods=12, exclude_recent=21)

    t = 60
    expected = (close.iloc[t - 21] / close.iloc[t - 33]) - 1.0
    assert np.isclose(r.iloc[t], expected, equal_nan=False)


def test_breakout_squeeze_is_parameterized():
    df = _sample_ohlcv(n=300)
    sq_60 = calculate_bollinger_squeeze(df["close"], percentile_window=60)
    sq_120 = calculate_bollinger_squeeze(df["close"], percentile_window=120)

    assert "squeeze_percentile" in sq_60.columns
    assert "squeeze_percentile" in sq_120.columns
    assert not sq_60["squeeze_percentile"].equals(sq_120["squeeze_percentile"])


def test_pairs_builder_interface_returns_core_metrics():
    df = _sample_ohlcv()
    series1 = df["close"]
    series2 = df["close"] * 1.01

    builder = PairsFeatureBuilder()
    out = builder.compute_for_pair(series1, series2, zscore_window=40, min_correlation=0.5)

    assert {"correlation", "spread", "spread_zscore", "pairs_signal"}.issubset(out.columns)
