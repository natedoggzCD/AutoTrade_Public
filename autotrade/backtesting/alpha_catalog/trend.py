"""Trend-following family alphas.

Hypothesis (Turtle, Elder, Hurst, et al.): persistent directional moves can be
captured by reacting to breakouts of recent ranges, moving-average crossovers
filtered by trend strength (ADX), and disciplined pullback re-entries.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from autotrade.backtesting.alpha_catalog import register
from autotrade.backtesting.alpha_catalog.base import (
    AlphaContext,
    AlphaDefinition,
    empty_signal_frame,
    require_bar_columns,
)
from autotrade.backtesting.alpha_catalog._indicators import (
    adx,
    atr,
    donchian_channel,
    macd,
)


def _donchian_breakout(period: int, exit_period: int):
    def _generate(bars: pd.DataFrame, ctx: Optional[AlphaContext]) -> pd.DataFrame:
        require_bar_columns(bars)
        df = bars.sort_values("date").reset_index(drop=True)
        if len(df) < period + exit_period + 5:
            return empty_signal_frame()

        upper, _ = donchian_channel(df["high"].astype(float), df["low"].astype(float), period)
        # signal: close breaks above prior bar's N-day high
        breakout = df["close"].astype(float) > upper.shift(1)
        breakout = breakout.fillna(False) & ~breakout.shift(1, fill_value=False)

        _, exit_lower = donchian_channel(df["high"].astype(float), df["low"].astype(float), exit_period)
        exit_signal = df["close"].astype(float) < exit_lower.shift(1)

        rows = []
        in_position = False
        for i in range(len(df)):
            if not in_position and bool(breakout.iloc[i]):
                rows.append(
                    {
                        "date": df.iloc[i]["date"],
                        "entry": True,
                        "exit": False,
                        "side": "long",
                        "score": float(df["close"].iloc[i] / upper.iloc[i] - 1.0) if pd.notna(upper.iloc[i]) and upper.iloc[i] > 0 else 0.0,
                        "note": f"donchian_{period}",
                    }
                )
                in_position = True
            elif in_position and bool(exit_signal.iloc[i]):
                rows.append(
                    {
                        "date": df.iloc[i]["date"],
                        "entry": False,
                        "exit": True,
                        "side": "long",
                        "score": 0.0,
                        "note": f"donchian_exit_{exit_period}",
                    }
                )
                in_position = False
        if in_position:
            rows.append(
                {
                    "date": df.iloc[-1]["date"],
                    "entry": False,
                    "exit": True,
                    "side": "long",
                    "score": 0.0,
                    "note": "donchian_eod_exit",
                }
            )
        if not rows:
            return empty_signal_frame()
        return pd.DataFrame(rows)

    return _generate


register(
    AlphaDefinition(
        alpha_id="donchian_20_10",
        family="trend",
        hypothesis="Turtle short-term: enter on 20d high break, exit on 10d low",
        variant="entry=20,exit=10",
        params={"entry": 20, "exit": 10},
        regime_compatibility=("TREND", "NEUTRAL", "ROTATION"),
        data_requirements=("daily_only",),
        generate=_donchian_breakout(20, 10),
    )
)

register(
    AlphaDefinition(
        alpha_id="donchian_55_20",
        family="trend",
        hypothesis="Turtle long-term: enter on 55d high break, exit on 20d low",
        variant="entry=55,exit=20",
        params={"entry": 55, "exit": 20},
        regime_compatibility=("TREND",),
        data_requirements=("daily_only",),
        generate=_donchian_breakout(55, 20),
    )
)


def _ma_crossover(fast: int, slow: int, adx_floor: float):
    def _generate(bars: pd.DataFrame, ctx: Optional[AlphaContext]) -> pd.DataFrame:
        require_bar_columns(bars)
        df = bars.sort_values("date").reset_index(drop=True)
        if len(df) < slow + 30:
            return empty_signal_frame()
        close = df["close"].astype(float)
        ma_fast = close.rolling(fast, min_periods=fast).mean()
        ma_slow = close.rolling(slow, min_periods=slow).mean()
        adx_v = adx(df["high"].astype(float), df["low"].astype(float), close, 14)

        cross_up = (ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))
        cross_dn = (ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1))
        gate = adx_v >= adx_floor

        rows = []
        in_pos = False
        for i in range(len(df)):
            if not in_pos and bool(cross_up.iloc[i]) and bool(gate.iloc[i]):
                rows.append(
                    {
                        "date": df.iloc[i]["date"],
                        "entry": True,
                        "exit": False,
                        "side": "long",
                        "score": float(adx_v.iloc[i] / 100.0) if pd.notna(adx_v.iloc[i]) else 0.0,
                        "note": f"macross_{fast}_{slow}",
                    }
                )
                in_pos = True
            elif in_pos and bool(cross_dn.iloc[i]):
                rows.append(
                    {
                        "date": df.iloc[i]["date"],
                        "entry": False,
                        "exit": True,
                        "side": "long",
                        "score": 0.0,
                        "note": "macross_exit",
                    }
                )
                in_pos = False
        if in_pos:
            rows.append(
                {
                    "date": df.iloc[-1]["date"],
                    "entry": False,
                    "exit": True,
                    "side": "long",
                    "score": 0.0,
                    "note": "macross_eod_exit",
                }
            )
        if not rows:
            return empty_signal_frame()
        return pd.DataFrame(rows)

    return _generate


register(
    AlphaDefinition(
        alpha_id="ma_cross_20_50_adx25",
        family="trend",
        hypothesis="20/50 SMA cross with ADX>25 trend filter",
        variant="fast=20,slow=50,adx>=25",
        params={"fast": 20, "slow": 50, "adx_floor": 25.0},
        regime_compatibility=("TREND", "NEUTRAL"),
        data_requirements=("daily_only",),
        generate=_ma_crossover(20, 50, 25.0),
    )
)

register(
    AlphaDefinition(
        alpha_id="ma_cross_50_200_adx20",
        family="trend",
        hypothesis="50/200 SMA golden-cross with ADX>20",
        variant="fast=50,slow=200,adx>=20",
        params={"fast": 50, "slow": 200, "adx_floor": 20.0},
        regime_compatibility=("TREND",),
        data_requirements=("daily_only",),
        generate=_ma_crossover(50, 200, 20.0),
    )
)


def _trend_pullback(ma_period: int, hold: int):
    def _generate(bars: pd.DataFrame, ctx: Optional[AlphaContext]) -> pd.DataFrame:
        require_bar_columns(bars)
        df = bars.sort_values("date").reset_index(drop=True)
        if len(df) < ma_period + hold + 5:
            return empty_signal_frame()
        close = df["close"].astype(float)
        ma = close.rolling(ma_period, min_periods=ma_period).mean()
        macd_line, macd_sig, macd_hist = macd(close)
        uptrend = (close > ma) & (ma.diff(20) > 0)
        # pullback: low touches or crosses MA in the past 3 bars, but close above MA today
        touched = (df["low"].astype(float).rolling(3, min_periods=1).min() <= ma) & (close > ma)
        curl_up = (macd_hist > macd_hist.shift(1)) & (macd_hist.shift(1) <= macd_hist.shift(2))
        entry = uptrend & touched & curl_up
        entry = entry.fillna(False) & ~entry.shift(1, fill_value=False)
        rows = []
        for idx in np.flatnonzero(entry.values):
            rows.append(
                {
                    "date": df.iloc[idx]["date"],
                    "entry": True,
                    "exit": False,
                    "side": "long",
                    "score": float(macd_hist.iloc[idx]) if pd.notna(macd_hist.iloc[idx]) else 0.0,
                    "note": f"pullback_ma{ma_period}",
                }
            )
            exit_idx = min(idx + hold, len(df) - 1)
            rows.append(
                {
                    "date": df.iloc[exit_idx]["date"],
                    "entry": False,
                    "exit": True,
                    "side": "long",
                    "score": 0.0,
                    "note": "pullback_exit",
                }
            )
        if not rows:
            return empty_signal_frame()
        return pd.DataFrame(rows)

    return _generate


register(
    AlphaDefinition(
        alpha_id="trend_pullback_50ema",
        family="trend",
        hypothesis="Uptrend pullback to 50-EMA with MACD curl-up trigger",
        variant="ma=50,hold=10",
        params={"ma": 50, "hold": 10},
        regime_compatibility=("TREND", "NEUTRAL"),
        data_requirements=("daily_only",),
        generate=_trend_pullback(50, 10),
    )
)

register(
    AlphaDefinition(
        alpha_id="trend_pullback_20ema",
        family="trend",
        hypothesis="Fast-trend pullback to 20-EMA with MACD curl-up",
        variant="ma=20,hold=5",
        params={"ma": 20, "hold": 5},
        regime_compatibility=("TREND", "NEUTRAL", "VOLATILE"),
        data_requirements=("daily_only",),
        generate=_trend_pullback(20, 5),
    )
)
