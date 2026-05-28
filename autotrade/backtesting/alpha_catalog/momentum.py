"""Momentum-family alphas.

Hypothesis (Jegadeesh-Titman 1993, Asness et al.): assets that have outperformed
over the past 3-12 months continue to outperform in the following 1-3 months,
after skipping the most recent month to avoid short-term reversal. We test
this on a per-symbol basis: an entry fires when the symbol's own trailing
return crosses a positive threshold; the position holds for a fixed window.
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


def _ts_momentum_generator(lookback: int, skip: int, hold: int):
    """Build a generator for ts-momentum with given lookback/skip/hold."""

    def _generate(bars: pd.DataFrame, ctx: Optional[AlphaContext]) -> pd.DataFrame:
        require_bar_columns(bars)
        df = bars.sort_values("date").reset_index(drop=True)
        if len(df) < lookback + skip + hold + 5:
            return empty_signal_frame()

        close = df["close"].astype(float)
        # trailing return from (t - lookback - skip) to (t - skip), as a decimal
        past = close.shift(skip)
        anchor = close.shift(skip + lookback)
        trailing_return = past / anchor - 1.0

        # entry when trailing_return clears 0 and price is above its 50-day MA
        ma50 = close.rolling(50, min_periods=50).mean()
        entry_signal = (trailing_return > 0.0) & (close > ma50)

        # de-dupe: only fire entry when previous bar wasn't an entry day
        entry_signal = entry_signal & ~entry_signal.shift(1, fill_value=False)

        out_rows = []
        for idx in np.flatnonzero(entry_signal.fillna(False).values):
            entry_date = df.iloc[idx]["date"]
            out_rows.append(
                {
                    "date": entry_date,
                    "entry": True,
                    "exit": False,
                    "side": "long",
                    "score": float(trailing_return.iloc[idx]) if pd.notna(trailing_return.iloc[idx]) else 0.0,
                    "note": f"ts_mom_{lookback}_{skip}",
                }
            )
            exit_idx = min(idx + hold, len(df) - 1)
            out_rows.append(
                {
                    "date": df.iloc[exit_idx]["date"],
                    "entry": False,
                    "exit": True,
                    "side": "long",
                    "score": 0.0,
                    "note": f"ts_mom_exit_{hold}d",
                }
            )

        if not out_rows:
            return empty_signal_frame()
        return pd.DataFrame(out_rows)

    return _generate


# 12-1 momentum: 252-day lookback, skip 21d, hold ~21d (Jegadeesh-Titman canonical)
register(
    AlphaDefinition(
        alpha_id="ts_momentum_12_1",
        family="momentum",
        hypothesis="Jegadeesh-Titman 12-month momentum skipping last month",
        variant="lookback=252,skip=21,hold=21",
        params={"lookback": 252, "skip": 21, "hold": 21},
        regime_compatibility=("TREND", "NEUTRAL", "ROTATION"),
        data_requirements=("daily_only",),
        generate=_ts_momentum_generator(252, 21, 21),
    )
)

# 6-1 momentum: medium-term variant
register(
    AlphaDefinition(
        alpha_id="ts_momentum_6_1",
        family="momentum",
        hypothesis="6-month time-series momentum skipping last month",
        variant="lookback=126,skip=21,hold=21",
        params={"lookback": 126, "skip": 21, "hold": 21},
        regime_compatibility=("TREND", "NEUTRAL", "ROTATION"),
        data_requirements=("daily_only",),
        generate=_ts_momentum_generator(126, 21, 21),
    )
)

# 3-1 momentum: short-term variant (more responsive in fast markets)
register(
    AlphaDefinition(
        alpha_id="ts_momentum_3_1",
        family="momentum",
        hypothesis="3-month time-series momentum skipping last week",
        variant="lookback=63,skip=5,hold=10",
        params={"lookback": 63, "skip": 5, "hold": 10},
        regime_compatibility=("TREND", "NEUTRAL", "VOLATILE"),
        data_requirements=("daily_only",),
        generate=_ts_momentum_generator(63, 5, 10),
    )
)


def _high_proximity_generator(window: int, threshold: float, hold: int):
    """George-Hwang 52-week-high proximity: enter when close within `threshold`
    of trailing `window`-day high, exit after `hold` days.
    """

    def _generate(bars: pd.DataFrame, ctx: Optional[AlphaContext]) -> pd.DataFrame:
        require_bar_columns(bars)
        df = bars.sort_values("date").reset_index(drop=True)
        if len(df) < window + hold + 5:
            return empty_signal_frame()

        close = df["close"].astype(float)
        high_w = df["high"].astype(float).rolling(window, min_periods=window).max()
        proximity = close / high_w
        entry_signal = proximity >= (1.0 - threshold)
        entry_signal = entry_signal & ~entry_signal.shift(1, fill_value=False)

        rows = []
        for idx in np.flatnonzero(entry_signal.fillna(False).values):
            entry_date = df.iloc[idx]["date"]
            rows.append(
                {
                    "date": entry_date,
                    "entry": True,
                    "exit": False,
                    "side": "long",
                    "score": float(proximity.iloc[idx]) if pd.notna(proximity.iloc[idx]) else 0.0,
                    "note": f"high_prox_{window}",
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
                    "note": "high_prox_exit",
                }
            )
        if not rows:
            return empty_signal_frame()
        return pd.DataFrame(rows)

    return _generate


register(
    AlphaDefinition(
        alpha_id="high52w_proximity",
        family="momentum",
        hypothesis="George-Hwang 52w high proximity predicts continuation",
        variant="window=252,threshold=2%,hold=20",
        params={"window": 252, "threshold": 0.02, "hold": 20},
        regime_compatibility=("TREND", "NEUTRAL"),
        data_requirements=("daily_only",),
        generate=_high_proximity_generator(252, 0.02, 20),
    )
)


def _infer_symbol_from_context(bars: pd.DataFrame, ctx: Optional[AlphaContext]) -> Optional[str]:
    if "symbol" in bars.columns and not bars["symbol"].dropna().empty:
        return str(bars["symbol"].dropna().iloc[0])
    if ctx is None or ctx.universe_bars is None or ctx.universe_bars.empty:
        return None
    universe = ctx.universe_bars.copy()
    if not {"symbol", "date", "close"}.issubset(universe.columns):
        return None
    sample = bars[["date", "close"]].copy()
    sample["date_key"] = pd.to_datetime(sample["date"]).dt.date
    universe["date_key"] = pd.to_datetime(universe["date"]).dt.date
    best_symbol = None
    best_error = np.inf
    for symbol, group in universe.groupby("symbol"):
        merged = sample.tail(80).merge(group[["date_key", "close"]], on="date_key", suffixes=("_bar", "_uni"))
        if len(merged) < 5:
            continue
        error = ((merged["close_bar"] - merged["close_uni"]).abs() / merged["close_bar"].replace(0.0, np.nan)).mean()
        if pd.notna(error) and float(error) < best_error:
            best_symbol = str(symbol)
            best_error = float(error)
    return best_symbol


def _sector_rank_momentum(lookback: int, hold: int):
    def _generate(bars: pd.DataFrame, ctx: Optional[AlphaContext]) -> pd.DataFrame:
        require_bar_columns(bars)
        if ctx is None or not ctx.sector_map:
            return empty_signal_frame()
        # cache OR universe_bars must be available; nothing else to fall back on
        if not ctx.cache and ctx.universe_bars is None:
            return empty_signal_frame()
        symbol = _infer_symbol_from_context(bars, ctx)
        if symbol is None:
            return empty_signal_frame()
        sector = ctx.sector_map.get(symbol) or ctx.sector_map.get(symbol.upper())
        if not sector:
            return empty_signal_frame()

        cache = ctx.cache or {}
        returns = cache.get(f"sector_returns_{sector}_{lookback}")
        ranks = cache.get(f"sector_rank_{sector}_{lookback}")
        if returns is None or ranks is None:
            universe = ctx.universe_bars.copy()
            if not {"symbol", "date", "close"}.issubset(universe.columns):
                return empty_signal_frame()
            universe["symbol"] = universe["symbol"].astype(str)
            sector_symbols = {s for s, sec in ctx.sector_map.items() if sec == sector}
            universe = universe[universe["symbol"].isin(sector_symbols)]
            if symbol not in set(universe["symbol"]):
                return empty_signal_frame()
            universe["date_key"] = pd.to_datetime(universe["date"]).dt.date
            close = universe.pivot_table(
                index="date_key", columns="symbol", values="close", aggfunc="last"
            ).sort_index()
            if len(close) < lookback + hold + 5:
                return empty_signal_frame()
            returns = close.astype(float).pct_change(lookback)
            ranks = returns.rank(axis=1, pct=True)
        else:
            if symbol not in ranks.columns or len(ranks) < lookback + hold + 5:
                return empty_signal_frame()
        signal = ranks[symbol] >= 0.80
        signal = signal.fillna(False) & ~signal.shift(1, fill_value=False)

        df = bars.sort_values("date").reset_index(drop=True).copy()
        df["date_key"] = pd.to_datetime(df["date"]).dt.date
        score = returns[symbol].reindex(df["date_key"]).fillna(0.0)
        rows = []
        for idx in df.index[df["date_key"].isin(signal[signal].index)]:
            rows.append(
                {
                    "date": df.iloc[idx]["date"],
                    "entry": True,
                    "exit": False,
                    "side": "long",
                    "score": float(score.iloc[idx]) if pd.notna(score.iloc[idx]) else 0.0,
                    "note": f"sector_rank_{lookback}",
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
                    "note": "sector_rank_exit",
                }
            )
        return pd.DataFrame(rows) if rows else empty_signal_frame()

    return _generate


def _residual_momentum(lookback: int, hold: int):
    def _generate(bars: pd.DataFrame, ctx: Optional[AlphaContext]) -> pd.DataFrame:
        require_bar_columns(bars)
        if ctx is None or ctx.spy_bars is None:
            return empty_signal_frame()
        df = bars.sort_values("date").reset_index(drop=True).copy()
        spy = ctx.spy_bars.copy()
        if len(df) < lookback + hold + 5 or not {"date", "close"}.issubset(spy.columns):
            return empty_signal_frame()
        df["date_key"] = pd.to_datetime(df["date"]).dt.date
        spy["date_key"] = pd.to_datetime(spy["date"]).dt.date
        ret = df["close"].astype(float).pct_change(lookback)
        spy_ret = spy.set_index("date_key")["close"].astype(float).pct_change(lookback).reindex(df["date_key"])
        residual = ret.to_numpy() - spy_ret.to_numpy()
        residual_series = pd.Series(residual, index=df.index)
        signal = (residual_series > residual_series.rolling(126, min_periods=40).quantile(0.80)).fillna(False)
        signal = signal & ~signal.shift(1, fill_value=False)
        rows = []
        for idx in np.flatnonzero(signal.values):
            rows.append(
                {
                    "date": df.iloc[int(idx)]["date"],
                    "entry": True,
                    "exit": False,
                    "side": "long",
                    "score": float(residual_series.iloc[int(idx)]) if pd.notna(residual_series.iloc[int(idx)]) else 0.0,
                    "note": f"residual_mom_{lookback}",
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
                    "note": "residual_mom_exit",
                }
            )
        return pd.DataFrame(rows) if rows else empty_signal_frame()

    return _generate


register(
    AlphaDefinition(
        alpha_id="xs_momentum_sector_rank_20d",
        family="momentum",
        hypothesis="Sector-relative 20d leaders continue as capital rotates within industries",
        variant="lookback=20,sector_top_quintile,hold=10",
        params={"lookback": 20, "rank": 0.80, "hold": 10},
        regime_compatibility=("TREND", "ROTATION", "NEUTRAL"),
        data_requirements=("needs_universe", "needs_sector_map"),
        generate=_sector_rank_momentum(20, 10),
    )
)

register(
    AlphaDefinition(
        alpha_id="xs_momentum_sector_rank_60d",
        family="momentum",
        hypothesis="Sector-relative 60d leaders identify durable intra-sector winners",
        variant="lookback=60,sector_top_quintile,hold=20",
        params={"lookback": 60, "rank": 0.80, "hold": 20},
        regime_compatibility=("TREND", "ROTATION"),
        data_requirements=("needs_universe", "needs_sector_map"),
        generate=_sector_rank_momentum(60, 20),
    )
)

register(
    AlphaDefinition(
        alpha_id="residual_momentum_6m",
        family="momentum",
        hypothesis="Positive residual 6m momentum versus SPY persists after market beta is removed",
        variant="lookback=126,hold=20",
        params={"lookback": 126, "hold": 20},
        regime_compatibility=("TREND", "NEUTRAL", "ROTATION"),
        data_requirements=("needs_spy",),
        generate=_residual_momentum(126, 20),
    )
)

register(
    AlphaDefinition(
        alpha_id="residual_momentum_12m",
        family="momentum",
        hypothesis="Positive residual 12m momentum captures idiosyncratic leadership",
        variant="lookback=252,hold=21",
        params={"lookback": 252, "hold": 21},
        regime_compatibility=("TREND", "ROTATION"),
        data_requirements=("needs_spy",),
        generate=_residual_momentum(252, 21),
    )
)
