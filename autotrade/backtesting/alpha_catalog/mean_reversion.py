"""Mean-reversion family alphas.

Hypothesis (Connors, Larsson, et al.): short-term overshoots away from a
short-term mean tend to revert in chop and sideways regimes. These alphas
are tagged regime_compatibility={CHOP, NEUTRAL, VOLATILE} so the live regime
router disables them in strong trends.
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
    rsi,
    rolling_zscore,
)


def _bollinger_zscore(period: int, z_entry: float, hold: int):
    def _generate(bars: pd.DataFrame, ctx: Optional[AlphaContext]) -> pd.DataFrame:
        require_bar_columns(bars)
        df = bars.sort_values("date").reset_index(drop=True)
        if len(df) < period + hold + 5:
            return empty_signal_frame()
        close = df["close"].astype(float)
        z = rolling_zscore(close, period)
        # entry: z <= -z_entry (oversold); exit after `hold` days or when z >= 0
        entry = (z <= -z_entry).fillna(False)
        entry = entry & ~entry.shift(1, fill_value=False)
        rows = []
        for idx in np.flatnonzero(entry.values):
            entry_date = df.iloc[idx]["date"]
            rows.append(
                {
                    "date": entry_date,
                    "entry": True,
                    "exit": False,
                    "side": "long",
                    "score": float(-z.iloc[idx]) if pd.notna(z.iloc[idx]) else 0.0,
                    "note": f"bollz_{period}_{z_entry:g}",
                }
            )
            # exit at first of: z>=0 or +hold days
            j_end = min(idx + hold, len(df) - 1)
            exit_idx = j_end
            for j in range(idx + 1, j_end + 1):
                if pd.notna(z.iloc[j]) and z.iloc[j] >= 0.0:
                    exit_idx = j
                    break
            rows.append(
                {
                    "date": df.iloc[exit_idx]["date"],
                    "entry": False,
                    "exit": True,
                    "side": "long",
                    "score": 0.0,
                    "note": "bollz_exit",
                }
            )
        if not rows:
            return empty_signal_frame()
        return pd.DataFrame(rows)

    return _generate


register(
    AlphaDefinition(
        alpha_id="bollinger_z_20_2",
        family="mean_reversion",
        hypothesis="20d Bollinger z<-2 oversold reverts toward mean",
        variant="period=20,z=2,hold=5",
        params={"period": 20, "z": 2.0, "hold": 5},
        regime_compatibility=("CHOP", "NEUTRAL", "VOLATILE"),
        data_requirements=("daily_only",),
        generate=_bollinger_zscore(20, 2.0, 5),
    )
)

register(
    AlphaDefinition(
        alpha_id="bollinger_z_50_2",
        family="mean_reversion",
        hypothesis="50d Bollinger z<-2 deep reversion in slower regimes",
        variant="period=50,z=2,hold=10",
        params={"period": 50, "z": 2.0, "hold": 10},
        regime_compatibility=("CHOP", "NEUTRAL"),
        data_requirements=("daily_only",),
        generate=_bollinger_zscore(50, 2.0, 10),
    )
)


def _rsi_extreme(rsi_period: int, low_thresh: float, exit_rsi: float, max_hold: int):
    def _generate(bars: pd.DataFrame, ctx: Optional[AlphaContext]) -> pd.DataFrame:
        require_bar_columns(bars)
        df = bars.sort_values("date").reset_index(drop=True)
        if len(df) < rsi_period + max_hold + 5:
            return empty_signal_frame()
        close = df["close"].astype(float)
        rsi_v = rsi(close, rsi_period)
        # entry: RSI crosses below low_thresh on a positive bar (reversal confirmation)
        entry = (rsi_v <= low_thresh).fillna(False)
        entry = entry & ~entry.shift(1, fill_value=False)
        rows = []
        for idx in np.flatnonzero(entry.values):
            rows.append(
                {
                    "date": df.iloc[idx]["date"],
                    "entry": True,
                    "exit": False,
                    "side": "long",
                    "score": float(low_thresh - rsi_v.iloc[idx]) if pd.notna(rsi_v.iloc[idx]) else 0.0,
                    "note": f"rsi{rsi_period}_lt{low_thresh:g}",
                }
            )
            exit_idx = min(idx + max_hold, len(df) - 1)
            for j in range(idx + 1, exit_idx + 1):
                if pd.notna(rsi_v.iloc[j]) and rsi_v.iloc[j] >= exit_rsi:
                    exit_idx = j
                    break
            rows.append(
                {
                    "date": df.iloc[exit_idx]["date"],
                    "entry": False,
                    "exit": True,
                    "side": "long",
                    "score": 0.0,
                    "note": f"rsi_exit_ge{exit_rsi:g}",
                }
            )
        if not rows:
            return empty_signal_frame()
        return pd.DataFrame(rows)

    return _generate


register(
    AlphaDefinition(
        alpha_id="rsi_2_oversold_connors",
        family="mean_reversion",
        hypothesis="Connors 2-day RSI <10 strongly mean-reverts (RSI(2) strategy)",
        variant="rsi_period=2,low=10,exit=70,max_hold=5",
        params={"rsi_period": 2, "low": 10.0, "exit": 70.0, "max_hold": 5},
        regime_compatibility=("CHOP", "NEUTRAL", "VOLATILE"),
        data_requirements=("daily_only",),
        generate=_rsi_extreme(2, 10.0, 70.0, 5),
    )
)

register(
    AlphaDefinition(
        alpha_id="rsi_14_oversold",
        family="mean_reversion",
        hypothesis="Classic RSI(14)<30 oversold reversion",
        variant="rsi_period=14,low=30,exit=55,max_hold=10",
        params={"rsi_period": 14, "low": 30.0, "exit": 55.0, "max_hold": 10},
        regime_compatibility=("CHOP", "NEUTRAL"),
        data_requirements=("daily_only",),
        generate=_rsi_extreme(14, 30.0, 55.0, 10),
    )
)


def _internal_bar_strength(low_thresh: float, exit_thresh: float, max_hold: int):
    def _generate(bars: pd.DataFrame, ctx: Optional[AlphaContext]) -> pd.DataFrame:
        require_bar_columns(bars)
        df = bars.sort_values("date").reset_index(drop=True)
        if len(df) < max_hold + 10:
            return empty_signal_frame()
        rng = (df["high"].astype(float) - df["low"].astype(float)).replace(0, np.nan)
        ibs = (df["close"].astype(float) - df["low"].astype(float)) / rng
        entry = (ibs <= low_thresh).fillna(False)
        entry = entry & ~entry.shift(1, fill_value=False)
        rows = []
        for idx in np.flatnonzero(entry.values):
            rows.append(
                {
                    "date": df.iloc[idx]["date"],
                    "entry": True,
                    "exit": False,
                    "side": "long",
                    "score": float(low_thresh - ibs.iloc[idx]) if pd.notna(ibs.iloc[idx]) else 0.0,
                    "note": f"ibs<{low_thresh:g}",
                }
            )
            exit_idx = min(idx + max_hold, len(df) - 1)
            for j in range(idx + 1, exit_idx + 1):
                if pd.notna(ibs.iloc[j]) and ibs.iloc[j] >= exit_thresh:
                    exit_idx = j
                    break
            rows.append(
                {
                    "date": df.iloc[exit_idx]["date"],
                    "entry": False,
                    "exit": True,
                    "side": "long",
                    "score": 0.0,
                    "note": "ibs_exit",
                }
            )
        if not rows:
            return empty_signal_frame()
        return pd.DataFrame(rows)

    return _generate


register(
    AlphaDefinition(
        alpha_id="internal_bar_strength",
        family="mean_reversion",
        hypothesis="Internal-bar-strength <0.1 close near low predicts next-day bounce",
        variant="low=0.1,exit=0.7,max_hold=3",
        params={"low": 0.1, "exit": 0.7, "max_hold": 3},
        regime_compatibility=("CHOP", "NEUTRAL", "VOLATILE"),
        data_requirements=("daily_only",),
        generate=_internal_bar_strength(0.1, 0.7, 3),
    )
)


def _keltner_fade(period: int, atr_mult: float, max_hold: int):
    def _generate(bars: pd.DataFrame, ctx: Optional[AlphaContext]) -> pd.DataFrame:
        require_bar_columns(bars)
        df = bars.sort_values("date").reset_index(drop=True)
        if len(df) < period + max_hold + 5:
            return empty_signal_frame()
        close = df["close"].astype(float)
        ema = close.ewm(span=period, adjust=False, min_periods=period).mean()
        atr_v = atr(df["high"].astype(float), df["low"].astype(float), close, period)
        upper = ema + atr_mult * atr_v
        lower = ema - atr_mult * atr_v
        short_entry = close > upper
        long_entry = close < lower
        entry = (short_entry | long_entry).fillna(False)
        entry = entry & ~entry.shift(1, fill_value=False)
        rows = []
        for idx in np.flatnonzero(entry.values):
            is_short = bool(short_entry.iloc[int(idx)])
            side = "short" if is_short else "long"
            score = (close.iloc[int(idx)] - upper.iloc[int(idx)]) / close.iloc[int(idx)] if is_short else (lower.iloc[int(idx)] - close.iloc[int(idx)]) / close.iloc[int(idx)]
            rows.append(
                {
                    "date": df.iloc[int(idx)]["date"],
                    "entry": True,
                    "exit": False,
                    "side": side,
                    "score": float(abs(score)) if pd.notna(score) else 0.0,
                    "note": "keltner_fade_15",
                }
            )
            exit_idx = min(int(idx) + max_hold, len(df) - 1)
            for j in range(int(idx) + 1, exit_idx + 1):
                if (is_short and close.iloc[j] <= ema.iloc[j]) or ((not is_short) and close.iloc[j] >= ema.iloc[j]):
                    exit_idx = j
                    break
            rows.append(
                {
                    "date": df.iloc[exit_idx]["date"],
                    "entry": False,
                    "exit": True,
                    "side": side,
                    "score": 0.0,
                    "note": "keltner_fade_exit",
                }
            )
        return pd.DataFrame(rows) if rows else empty_signal_frame()

    return _generate


register(
    AlphaDefinition(
        alpha_id="keltner_fade_15",
        family="mean_reversion",
        hypothesis="Close outside 1.5 ATR Keltner channel mean-reverts toward EMA",
        variant="period=15,atr_mult=1.5,max_hold=5",
        params={"period": 15, "atr_mult": 1.5, "max_hold": 5},
        regime_compatibility=("CHOP", "NEUTRAL", "VOLATILE"),
        data_requirements=("daily_only",),
        generate=_keltner_fade(15, 1.5, 5),
    )
)
