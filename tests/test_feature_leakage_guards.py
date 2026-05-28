"""
Anti-leakage tests for feature transforms.

These tests assert prefix invariance: feature values up to time t must not
change when future bars are appended.
"""

import numpy as np
import pandas as pd
import pandas.testing as pdt

from autotrade.feature_engineering.breakout import calculate_donchian_channels
from autotrade.feature_engineering.momentum import calculate_excluded_returns
from autotrade.feature_engineering.pipeline import FeaturePipeline
from autotrade.feature_engineering.schemas import FeatureFamily, FeatureRequest
from autotrade.feature_engineering.technical import calculate_rsi
from autotrade.feature_engineering.trend import calculate_ma_slope_trend


def _close_series(n: int = 220) -> pd.Series:
    rng = np.random.default_rng(2026)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(100 + np.cumsum(rng.normal(0.0, 1.0, size=n)), index=idx)


def _extend_series(base: pd.Series, extra: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    future_idx = pd.date_range(base.index[-1] + pd.Timedelta(days=1), periods=extra, freq="D")
    future = pd.Series(base.iloc[-1] + np.cumsum(rng.normal(0.0, 1.0, size=extra)), index=future_idx)
    return pd.concat([base, future])


def _ohl_series(n: int = 220):
    close = _close_series(n)
    rng = np.random.default_rng(2027)
    high = close + rng.uniform(0.1, 1.2, size=n)
    low = close - rng.uniform(0.1, 1.2, size=n)
    return high, low, close


def _extend_ohl(high: pd.Series, low: pd.Series, close: pd.Series, extra: int):
    close_ext = _extend_series(close, extra=extra, seed=3030)
    rng = np.random.default_rng(4040)
    add_high = close_ext.iloc[-extra:] + rng.uniform(0.1, 1.2, size=extra)
    add_low = close_ext.iloc[-extra:] - rng.uniform(0.1, 1.2, size=extra)
    high_ext = pd.concat([high, pd.Series(add_high.values, index=close_ext.index[-extra:])])
    low_ext = pd.concat([low, pd.Series(add_low.values, index=close_ext.index[-extra:])])
    return high_ext, low_ext, close_ext


def _assert_prefix_invariant_series(fn, base_args, ext_args, cutoff: int):
    base = fn(*base_args).iloc[:cutoff]
    extended = fn(*ext_args).iloc[:cutoff]
    pdt.assert_series_equal(base, extended, check_names=False, check_dtype=False)


def _assert_prefix_invariant_frame_col(fn, base_args, ext_args, column: str, cutoff: int):
    base = fn(*base_args)[column].iloc[:cutoff]
    extended = fn(*ext_args)[column].iloc[:cutoff]
    pdt.assert_series_equal(base, extended, check_names=False, check_dtype=False)


def test_no_future_leakage_ma_slope_trend():
    close = _close_series(220)
    close_ext = _extend_series(close, extra=40, seed=3001)
    _assert_prefix_invariant_series(
        lambda s: calculate_ma_slope_trend(s, ma_window=20, slope_window=5),
        (close,),
        (close_ext,),
        cutoff=200,
    )


def test_no_future_leakage_rsi():
    close = _close_series(220)
    close_ext = _extend_series(close, extra=40, seed=3002)
    _assert_prefix_invariant_series(
        lambda s: calculate_rsi(s, window=14),
        (close,),
        (close_ext,),
        cutoff=200,
    )


def test_no_future_leakage_excluded_returns():
    close = _close_series(220)
    close_ext = _extend_series(close, extra=40, seed=3003)
    _assert_prefix_invariant_series(
        lambda s: calculate_excluded_returns(s, total_periods=12, exclude_recent=21),
        (close,),
        (close_ext,),
        cutoff=200,
    )


def test_no_future_leakage_donchian_channels():
    high, low, close = _ohl_series(220)
    high_ext, low_ext, close_ext = _extend_ohl(high, low, close, extra=40)
    _assert_prefix_invariant_frame_col(
        lambda h, l: calculate_donchian_channels(h, l, window=20),
        (high, low),
        (high_ext, low_ext),
        column="donchian_upper",
        cutoff=200,
    )
    _assert_prefix_invariant_frame_col(
        lambda h, l: calculate_donchian_channels(h, l, window=20),
        (high, low),
        (high_ext, low_ext),
        column="donchian_lower",
        cutoff=200,
    )


def test_pipeline_merge_asof_is_symbol_safe():
    pipeline = FeaturePipeline(config={"min_history_bars": 1})
    primary = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "date": pd.to_datetime(["2024-01-02 10:00:00", "2024-01-02 10:00:00"]),
            "close": [10.0, 20.0],
        }
    )
    secondary = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "timestamp": pd.to_datetime(["2024-01-02 09:30:00", "2024-01-02 09:30:00"]),
            "hourly_signal": [1.0, -1.0],
        }
    )

    merged = pipeline._merge_as_of(primary, secondary, "date", "timestamp")
    a_signal = float(merged.loc[merged["symbol"] == "AAA", "hourly_signal"].iloc[0])
    b_signal = float(merged.loc[merged["symbol"] == "BBB", "hourly_signal"].iloc[0])
    assert a_signal == 1.0
    assert b_signal == -1.0


def test_pipeline_excludes_future_hourly_bars_from_asof_merge():
    pipeline = FeaturePipeline(config={"min_history_bars": 1, "alpha_families_enabled": []})
    request = FeatureRequest(
        symbols=["AAA"],
        start_date="2024-01-01",
        end_date="2024-01-03 23:59:59",
        as_of_date="2024-01-03 10:00:00",
        families=[FeatureFamily.TREND],
    )
    daily_data = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "open": [9.0, 10.0, 11.0],
            "high": [11.0, 12.0, 13.0],
            "low": [8.0, 9.0, 10.0],
            "close": [10.0, 11.0, 12.0],
            "volume": [100_000, 110_000, 120_000],
        }
    )
    hourly_data = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "timestamp": pd.to_datetime(["2024-01-03 09:30:00", "2024-01-03 12:00:00"]),
            "close": [10.5, 999.0],
            "vwap": [10.4, 1.0],
            "ema20": [10.3, 1.0],
        }
    )

    frame = pipeline.execute_multitimeframe(
        request=request,
        daily_data=daily_data,
        hourly_data=hourly_data,
    )
    last = frame.df.sort_values("date").iloc[-1]
    assert float(last["hourly_vwap_recovery"]) == 1.0
    assert float(last["hourly_ema20_regain"]) == 1.0
