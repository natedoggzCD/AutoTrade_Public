"""
Phase 5 completion tests for feature engineering:
- Unit coverage for pure transform functions across all feature modules.
- Deterministic golden snapshot lock for shared feature assembly.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autotrade.feature_engineering.breakout import (
    BreakoutFeatureBuilder,
    calculate_bollinger_squeeze,
    calculate_donchian_breakout,
    calculate_donchian_channels,
    calculate_high_low_breakout,
    calculate_price_momentum_breakout,
    calculate_volume_expansion,
)
from autotrade.feature_engineering.momentum import (
    MomentumFeatureBuilder,
    calculate_excluded_returns,
    calculate_momentum_acceleration,
    calculate_momentum_divergence,
    calculate_returns,
    calculate_sector_relative_momentum,
    calculate_ts_momentum,
    calculate_xs_momentum,
)
from autotrade.feature_engineering.pairs import (
    calculate_cointegration_test,
    calculate_hedge_ratio,
    calculate_pairs_metrics,
    calculate_rolling_correlation,
    calculate_rolling_spread,
    calculate_spread_zscore,
)
from autotrade.feature_engineering.regime import (
    RegimeFeatureBuilder,
    calculate_market_breadth,
    calculate_range_regime,
    calculate_regime_composite,
    calculate_trend_regime,
    calculate_volatility_regime,
)
from autotrade.feature_engineering.reversal import (
    ReversalFeatureBuilder,
    calculate_atr_stretch,
    calculate_bollinger_deviation,
    calculate_capitulation_score,
    calculate_gap_reversal,
    calculate_mean_reversion_zscore,
    calculate_rsi_reversal,
    calculate_support_resistance_proximity,
)
from autotrade.feature_engineering.technical import (
    TechnicalFeatureBuilder,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_ma_slope,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
    calculate_stochastic,
    calculate_vwap,
)
from autotrade.feature_engineering.trend import (
    TrendFeatureBuilder,
    calculate_adx_trend,
    calculate_directional_movement,
    calculate_efficiency_ratio,
    calculate_ema_alignment,
    calculate_ma_slope_trend,
    calculate_ma_stacking,
    calculate_trend_strength_composite,
)
from autotrade.feature_engineering.volatility_volume import (
    VolatilityVolumeFeatureBuilder,
    calculate_atr,
    calculate_atr_percent,
    calculate_obv,
    calculate_price_volatility,
    calculate_true_range,
    calculate_volume_ma,
    calculate_volume_percentile,
    calculate_volume_ratio,
    calculate_vwap_volatility,
)


def _sample_ohlcv(n: int = 260, seed: int = 12345) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
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


def test_pure_transforms_technical():
    df = _sample_ohlcv()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    sma = calculate_sma(close, 20)
    ema = calculate_ema(close, 20)
    rsi = calculate_rsi(close, 14)
    macd = calculate_macd(close)
    bb = calculate_bollinger_bands(close)
    stoch = calculate_stochastic(high, low, close)
    vwap = calculate_vwap(high, low, close, volume)
    slope = calculate_ma_slope(close, ma_window=20, slope_window=5)

    assert np.isclose(sma.iloc[19], close.iloc[:20].mean())
    assert ema.notna().sum() > 0
    assert rsi.notna().sum() > 0
    assert {"macd", "macd_signal", "macd_hist"} == set(macd.columns)
    assert {"bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_position"} == set(
        bb.columns
    )
    assert {"stoch_k", "stoch_d"} == set(stoch.columns)
    assert vwap.notna().sum() > 0
    assert slope.notna().sum() > 0


def test_pure_transforms_volatility_volume():
    df = _sample_ohlcv()
    tr = calculate_true_range(df["high"], df["low"], df["close"])
    atr = calculate_atr(df["high"], df["low"], df["close"], window=14)
    atr_pct = calculate_atr_percent(df["high"], df["low"], df["close"], window=14)
    vol_ma = calculate_volume_ma(df["volume"], window=20)
    vol_ratio = calculate_volume_ratio(df["volume"], window=20)
    vol_pct = calculate_volume_percentile(df["volume"], window=60)
    obv = calculate_obv(df["close"], df["volume"])
    vwap_vol = calculate_vwap_volatility(
        df["high"], df["low"], df["close"], df["volume"], window=20
    )
    px_vol = calculate_price_volatility(df["close"], window=20)

    assert (tr.dropna() >= 0).all()
    assert atr.notna().sum() > 0
    assert atr_pct.notna().sum() > 0
    assert vol_ma.notna().sum() > 0
    assert vol_ratio.notna().sum() > 0
    assert vol_pct.dropna().between(0, 100).all()
    assert obv.notna().sum() > 0
    assert vwap_vol.notna().sum() > 0
    assert px_vol.notna().sum() > 0


def test_pure_transforms_trend():
    df = _sample_ohlcv()
    ma_slope = calculate_ma_slope_trend(df["close"], ma_window=20, slope_window=5)
    dm = calculate_directional_movement(df["high"], df["low"], df["close"], window=14)
    adx = calculate_adx_trend(df["high"], df["low"], df["close"], window=14)
    er = calculate_efficiency_ratio(df["close"], window=20)
    composite = calculate_trend_strength_composite(df["close"], df["high"], df["low"])
    stack = calculate_ma_stacking(df["close"], windows=[5, 20, 50, 100])
    alignment = calculate_ema_alignment(df["close"], windows=[8, 21, 55])

    assert ma_slope.notna().sum() > 0
    assert {"plus_di", "minus_di", "adx"} == set(dm.columns)
    assert adx.notna().sum() > 0
    assert er.notna().sum() > 0
    assert {
        "ma_slope_trend",
        "adx_trend",
        "efficiency_ratio",
        "trend_strength",
        "trend_direction",
    } == set(composite.columns)
    assert "ma_stack_score" in stack.columns
    assert alignment.dropna().between(0, 1).all()


def test_pure_transforms_momentum():
    df = _sample_ohlcv()
    close = df["close"]
    volume = df["volume"]

    returns = calculate_returns(close, periods=5)
    excluded = calculate_excluded_returns(close, total_periods=12, exclude_recent=21)
    ts = calculate_ts_momentum(close, exclude_recent=21)
    xs = calculate_xs_momentum(close, exclude_recent=21)
    sector = close.pct_change(20).fillna(0.0)
    sector_rel = calculate_sector_relative_momentum(close, sector)
    accel = calculate_momentum_acceleration(close, short_window=5, long_window=20)
    divergence = calculate_momentum_divergence(close, volume, window=20)

    assert returns.notna().sum() > 0
    assert excluded.notna().sum() > 0
    assert {"return_12_1", "return_6_1", "return_3_1", "return_1_1"} == set(ts.columns)
    assert {"rank_6_1", "rank_12_1", "cross_sectional_momentum"} == set(xs.columns)
    assert sector_rel.notna().sum() > 0
    assert accel.notna().sum() > 0
    assert divergence.notna().sum() > 0

    with pytest.raises(ValueError):
        calculate_excluded_returns(close, total_periods=0, exclude_recent=21)


def test_pure_transforms_reversal():
    df = _sample_ohlcv()
    rsi2 = calculate_rsi_reversal(df["close"], window=2)
    bb_dev = calculate_bollinger_deviation(df["close"], window=20, num_std=2.0)
    atr_stretch = calculate_atr_stretch(df["close"], df["high"], df["low"], window=14)
    cap = calculate_capitulation_score(df["close"], df["volume"], lookback=20)
    zscore = calculate_mean_reversion_zscore(df["close"], window=20)
    gap_rev = calculate_gap_reversal(df["open"], df["close"], df["close"].shift(1))
    sr = calculate_support_resistance_proximity(
        df["close"], df["low"], df["high"], window=20
    )

    assert rsi2.notna().sum() > 0
    assert bb_dev.notna().sum() > 0
    assert atr_stretch.notna().sum() > 0
    assert {"capitulation_signal", "capitulation_strength"} == set(cap.columns)
    assert zscore.notna().sum() > 0
    assert gap_rev.notna().sum() > 0
    assert sr.dropna().between(0, 1).all()


def test_pure_transforms_breakout():
    df = _sample_ohlcv()
    dc = calculate_donchian_channels(df["high"], df["low"], window=20)
    dc_breakout = calculate_donchian_breakout(df["close"], df["high"], df["low"], 20)
    squeeze = calculate_bollinger_squeeze(df["close"], percentile_window=120)
    vol_exp = calculate_volume_expansion(df["close"], df["volume"], window=20)
    px_breakout = calculate_price_momentum_breakout(df["close"], window=20, threshold=0.03)
    hl_breakout = calculate_high_low_breakout(df["high"], df["low"], df["close"], 20)

    assert {"donchian_upper", "donchian_middle", "donchian_lower", "donchian_width"} == set(
        dc.columns
    )
    assert {
        "donchian_upper_breakout",
        "donchian_lower_breakout",
        "donchian_breakout_direction",
    } == set(dc_breakout.columns)
    assert {"squeeze", "squeeze_percentile", "bb_width_raw", "kc_width_raw"} == set(
        squeeze.columns
    )
    assert vol_exp.notna().sum() > 0
    assert set(px_breakout.dropna().unique()).issubset({0, 1})
    assert {"breakout_above_20d_high", "breakout_below_20d_low"} == set(
        hl_breakout.columns
    )


def test_pure_transforms_regime():
    close = _sample_ohlcv()["close"]
    vol_regime = calculate_volatility_regime(close, window=20)
    trend_regime = calculate_trend_regime(close, short_window=20, long_window=50)
    range_regime = calculate_range_regime(close, window=20)
    breadth = calculate_market_breadth(close, lookback=20)
    composite = calculate_regime_composite(close, volatility_window=20, trend_short=20, trend_long=50)

    assert set(vol_regime.dropna().unique()).issubset({0.0, 1.0, 2.0})
    assert set(trend_regime.dropna().unique()).issubset({-1, 1})
    assert set(range_regime.dropna().unique()).issubset({0.0, 1.0, 2.0})
    assert {"advancing", "declining", "advance_decline_ratio"} == set(breadth.columns)
    assert {"volatility_regime", "trend_regime", "range_regime", "regime_composite"} == set(
        composite.columns
    )


def test_pure_transforms_pairs():
    df = _sample_ohlcv()
    s1 = df["close"]
    s2 = df["close"] * 1.01

    corr = calculate_rolling_correlation(s1, s2, window=60)
    spread = calculate_rolling_spread(s1, s2, window=60)
    zdf = calculate_spread_zscore(s1, s2, window=60)
    hedge = calculate_hedge_ratio(s1, s2, window=60)
    coint = calculate_cointegration_test(s1, s2, window=60)
    metrics = calculate_pairs_metrics(s1, s2, zscore_window=60, min_correlation=0.7)

    assert corr.notna().sum() > 0
    assert spread.notna().sum() == len(spread)
    assert {"spread", "spread_ma", "spread_zscore"} == set(zdf.columns)
    assert hedge.notna().sum() > 0
    assert {"correlation", "cointegration_score", "cointegration_signal"} == set(coint.columns)
    assert {"correlation", "spread", "spread_zscore", "pairs_signal"} == set(metrics.columns)


def test_golden_snapshot_for_shared_feature_assembly():
    fixture_path = Path("tests/fixtures/feature_engineering/golden_feature_snapshot.json")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

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

    assert len(result) == fixture["row_count"]
    assert len(result.columns) == fixture["column_count"]
    assert str(result.index[-1]) == fixture["last_timestamp"]

    expected_subset = set(fixture["columns_subset"])
    assert expected_subset.issubset(result.columns)

    last = result.iloc[-1]
    for key, expected in fixture["last_row_values"].items():
        actual = None if pd.isna(last[key]) else float(last[key])
        assert actual == pytest.approx(expected, rel=1e-10, abs=1e-10)
