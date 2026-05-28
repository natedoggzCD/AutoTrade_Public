"""Pattern-recognition family alphas.

Hypothesis: certain chart formations (inside-bar consolidation, volatility
contraction patterns, base+pivot breakouts) reflect transitions from
accumulation to markup. These are simplified algorithmic versions of the
discretionary patterns IBD and Minervini use.
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
from autotrade.backtesting.alpha_catalog._indicators import atr


def _inside_bar_breakout(hold: int):
    def _generate(bars: pd.DataFrame, ctx: Optional[AlphaContext]) -> pd.DataFrame:
        require_bar_columns(bars)
        df = bars.sort_values("date").reset_index(drop=True)
        if len(df) < hold + 10:
            return empty_signal_frame()
        high, low = df["high"].astype(float), df["low"].astype(float)
        inside = (high < high.shift(1)) & (low > low.shift(1))
        prev_high = high.shift(1)
        # entry on bar that breaks the inside-bar's prior high
        break_up = (df["close"].astype(float) > prev_high) & inside.shift(1, fill_value=False)
        entry = break_up.fillna(False) & ~break_up.shift(1, fill_value=False)
        rows = []
        for idx in np.flatnonzero(entry.values):
            rows.append(
                {
                    "date": df.iloc[idx]["date"],
                    "entry": True,
                    "exit": False,
                    "side": "long",
                    "score": 1.0,
                    "note": "inside_bar_break",
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
                    "note": "inside_bar_exit",
                }
            )
        if not rows:
            return empty_signal_frame()
        return pd.DataFrame(rows)

    return _generate


register(
    AlphaDefinition(
        alpha_id="inside_bar_breakout",
        family="pattern",
        hypothesis="Inside-bar consolidation breakout marks momentum resumption",
        variant="hold=5",
        params={"hold": 5},
        regime_compatibility=("TREND", "NEUTRAL", "CHOP"),
        data_requirements=("daily_only",),
        generate=_inside_bar_breakout(5),
    )
)


def _vcp_minervini(base_lookback: int, contraction_count: int, hold: int):
    """Simplified Minervini-style volatility-contraction pattern.

    Requires the rolling 20-day ATR percent to have made at least
    `contraction_count` lower-low contractions during the prior `base_lookback`
    days, then entry on close above the trailing high.
    """

    def _generate(bars: pd.DataFrame, ctx: Optional[AlphaContext]) -> pd.DataFrame:
        require_bar_columns(bars)
        df = bars.sort_values("date").reset_index(drop=True)
        if len(df) < base_lookback + hold + 30:
            return empty_signal_frame()
        close = df["close"].astype(float)
        atr_v = atr(df["high"].astype(float), df["low"].astype(float), close, 14)
        atr_pct = atr_v / close.replace(0, np.nan)
        # count strict descents in atr_pct over the base window
        descents = (atr_pct < atr_pct.shift(5)).astype(int)
        recent_descents = descents.rolling(base_lookback, min_periods=base_lookback).sum()
        contraction = recent_descents >= contraction_count
        rolling_high = df["high"].astype(float).rolling(base_lookback, min_periods=base_lookback).max()
        breakout = (close > rolling_high.shift(1)) & contraction
        entry = breakout.fillna(False) & ~breakout.shift(1, fill_value=False)
        rows = []
        for idx in np.flatnonzero(entry.values):
            rows.append(
                {
                    "date": df.iloc[idx]["date"],
                    "entry": True,
                    "exit": False,
                    "side": "long",
                    "score": float(recent_descents.iloc[idx] / base_lookback) if pd.notna(recent_descents.iloc[idx]) else 0.0,
                    "note": "vcp_break",
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
                    "note": "vcp_exit",
                }
            )
        if not rows:
            return empty_signal_frame()
        return pd.DataFrame(rows)

    return _generate


register(
    AlphaDefinition(
        alpha_id="vcp_minervini",
        family="pattern",
        hypothesis="Minervini volatility-contraction pattern then base breakout",
        variant="base=30,contractions>=15,hold=20",
        params={"base": 30, "contractions": 15, "hold": 20},
        regime_compatibility=("TREND", "NEUTRAL"),
        data_requirements=("daily_only",),
        generate=_vcp_minervini(30, 15, 20),
    )
)


def _cup_and_handle(min_base: int, max_base: int, hold: int):
    def _generate(bars: pd.DataFrame, ctx: Optional[AlphaContext]) -> pd.DataFrame:
        require_bar_columns(bars)
        df = bars.sort_values("date").reset_index(drop=True)
        if len(df) < max_base + hold + 20:
            return empty_signal_frame()
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        volume = df["volume"].astype(float)
        avg_volume = volume.rolling(20, min_periods=20).mean()
        rows = []
        for idx in range(max_base, len(df)):
            window = close.iloc[idx - max_base : idx + 1]
            trough_pos = int(window.values.argmin())
            if trough_pos < 15 or trough_pos > len(window) - 15:
                continue
            left_high = window.iloc[:trough_pos].max()
            right_high = window.iloc[trough_pos:].max()
            trough = window.iloc[trough_pos]
            if left_high <= 0 or right_high < left_high * 0.90:
                continue
            depth = (left_high - trough) / left_high
            handle_low = close.iloc[max(idx - 15, 0) : idx + 1].min()
            handle_depth = (right_high - handle_low) / max(right_high, 1e-9)
            breakout = close.iloc[idx] > high.iloc[idx - min_base : idx].max()
            if 0.12 <= depth <= 0.45 and handle_depth < depth * 0.50 and breakout and volume.iloc[idx] > avg_volume.iloc[idx] * 1.4:
                if rows and rows[-2]["date"] == df.iloc[idx]["date"]:
                    continue
                rows.append(
                    {
                        "date": df.iloc[idx]["date"],
                        "entry": True,
                        "exit": False,
                        "side": "long",
                        "score": float(depth - handle_depth),
                        "note": "cup_handle_breakout",
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
                        "note": "cup_handle_exit",
                    }
                )
        return pd.DataFrame(rows) if rows else empty_signal_frame()

    return _generate


def _pivot_breakout_ibd(min_base: int, hold: int):
    def _generate(bars: pd.DataFrame, ctx: Optional[AlphaContext]) -> pd.DataFrame:
        require_bar_columns(bars)
        df = bars.sort_values("date").reset_index(drop=True)
        if len(df) < min_base + hold + 30:
            return empty_signal_frame()
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)
        avg_volume = volume.rolling(50, min_periods=20).mean()
        base_high = high.rolling(min_base, min_periods=min_base).max().shift(1)
        base_low = low.rolling(min_base, min_periods=min_base).min().shift(1)
        base_depth = (base_high - base_low) / base_high.replace(0.0, np.nan)
        tight_enough = base_depth.between(0.05, 0.35)
        breakout = (close > base_high) & tight_enough & (volume >= 1.4 * avg_volume)
        breakout = breakout.fillna(False) & ~breakout.shift(1, fill_value=False)
        rows = []
        for idx in np.flatnonzero(breakout.values):
            rows.append(
                {
                    "date": df.iloc[int(idx)]["date"],
                    "entry": True,
                    "exit": False,
                    "side": "long",
                    "score": float(volume.iloc[int(idx)] / avg_volume.iloc[int(idx)]) if avg_volume.iloc[int(idx)] > 0 else 0.0,
                    "note": "ibd_pivot_breakout",
                }
            )
            exit_idx = min(int(idx) + hold, len(df) - 1)
            rows.append(
                {
                    "date": df.iloc[exit_idx]["date"],
                    "entry": False,
                    "exit": True,
                    "side": "long",
                    "score": 0.0,
                    "note": "ibd_pivot_exit",
                }
            )
        return pd.DataFrame(rows) if rows else empty_signal_frame()

    return _generate


register(
    AlphaDefinition(
        alpha_id="cup_and_handle",
        family="pattern",
        hypothesis="U-shaped cup with shallow handle resolves into a volume-backed breakout",
        variant="base=50-150,handle<50%cup_depth,volume>1.4x,hold=20",
        params={"base_min": 50, "base_max": 150, "hold": 20},
        regime_compatibility=("TREND", "NEUTRAL"),
        data_requirements=("daily_only",),
        generate=_cup_and_handle(50, 150, 20),
    )
)

register(
    AlphaDefinition(
        alpha_id="pivot_breakout_ibd",
        family="pattern",
        hypothesis="IBD-style five-week base pivot breakout on at least 40% volume surge",
        variant="base>=25,volume>1.4x,hold=20",
        params={"base": 25, "volume_ratio": 1.4, "hold": 20},
        regime_compatibility=("TREND", "ROTATION"),
        data_requirements=("daily_only",),
        generate=_pivot_breakout_ibd(25, 20),
    )
)
