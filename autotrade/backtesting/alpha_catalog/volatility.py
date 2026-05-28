"""Volatility / breakout family alphas.

Hypothesis (Crabel, Hurst, Connors): periods of unusually compressed volatility
precede directional moves; the direction is taken at the break of recent range.
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
    atr,
    bollinger_bands,
    donchian_channel,
)


def _bb_width_squeeze(period: int, width_window: int, hold: int):
    def _generate(bars: pd.DataFrame, ctx: Optional[AlphaContext]) -> pd.DataFrame:
        require_bar_columns(bars)
        df = bars.sort_values("date").reset_index(drop=True)
        if len(df) < width_window + hold + 5:
            return empty_signal_frame()
        close = df["close"].astype(float)
        upper, mid, lower, sd = bollinger_bands(close, period, 2.0)
        width = (upper - lower) / mid.replace(0, np.nan)
        width_pct = width.rolling(width_window, min_periods=width_window).rank(pct=True)
        squeeze = (width_pct <= 0.20).fillna(False)
        # entry on break of upper band while in squeeze
        break_up = (close > upper.shift(1)) & squeeze.shift(1, fill_value=False)
        entry = break_up & ~break_up.shift(1, fill_value=False)
        rows = []
        for idx in np.flatnonzero(entry.values):
            rows.append(
                {
                    "date": df.iloc[idx]["date"],
                    "entry": True,
                    "exit": False,
                    "side": "long",
                    "score": float(1.0 - width_pct.iloc[idx]) if pd.notna(width_pct.iloc[idx]) else 0.0,
                    "note": f"bb_sqz_{period}",
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
                    "note": "bb_sqz_exit",
                }
            )
        if not rows:
            return empty_signal_frame()
        return pd.DataFrame(rows)

    return _generate


register(
    AlphaDefinition(
        alpha_id="bb_width_squeeze_20_60",
        family="volatility",
        hypothesis="20d BB width in bottom 20% of trailing 60d → directional break",
        variant="period=20,width_window=60,hold=10",
        params={"period": 20, "width_window": 60, "hold": 10},
        regime_compatibility=("CHOP", "NEUTRAL", "TREND"),
        data_requirements=("daily_only",),
        generate=_bb_width_squeeze(20, 60, 10),
    )
)


def _nr_breakout(nr_n: int, hold: int):
    def _generate(bars: pd.DataFrame, ctx: Optional[AlphaContext]) -> pd.DataFrame:
        require_bar_columns(bars)
        df = bars.sort_values("date").reset_index(drop=True)
        if len(df) < nr_n + hold + 5:
            return empty_signal_frame()
        rng = df["high"].astype(float) - df["low"].astype(float)
        is_nr = rng == rng.rolling(nr_n, min_periods=nr_n).min()
        prev_high = df["high"].astype(float).shift(1)
        entry = (df["close"].astype(float) > prev_high) & is_nr.shift(1, fill_value=False)
        entry = entry.fillna(False) & ~entry.shift(1, fill_value=False)
        rows = []
        for idx in np.flatnonzero(entry.values):
            rows.append(
                {
                    "date": df.iloc[idx]["date"],
                    "entry": True,
                    "exit": False,
                    "side": "long",
                    "score": float((df["close"].iloc[idx] - prev_high.iloc[idx]) / prev_high.iloc[idx]) if pd.notna(prev_high.iloc[idx]) and prev_high.iloc[idx] > 0 else 0.0,
                    "note": f"nr{nr_n}_break",
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
                    "note": "nr_exit",
                }
            )
        if not rows:
            return empty_signal_frame()
        return pd.DataFrame(rows)

    return _generate


register(
    AlphaDefinition(
        alpha_id="nr7_breakout",
        family="volatility",
        hypothesis="Crabel NR7 narrow-range bar breakout",
        variant="nr=7,hold=5",
        params={"nr": 7, "hold": 5},
        regime_compatibility=("CHOP", "NEUTRAL", "TREND"),
        data_requirements=("daily_only",),
        generate=_nr_breakout(7, 5),
    )
)

register(
    AlphaDefinition(
        alpha_id="nr4_breakout",
        family="volatility",
        hypothesis="NR4 narrow-range bar breakout (tighter, more signals)",
        variant="nr=4,hold=3",
        params={"nr": 4, "hold": 3},
        regime_compatibility=("CHOP", "NEUTRAL", "VOLATILE"),
        data_requirements=("daily_only",),
        generate=_nr_breakout(4, 3),
    )
)
