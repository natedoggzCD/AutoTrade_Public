"""Cross-sectional alpha definitions.

These generators need a long-form universe frame in ``AlphaContext`` so the
symbol under evaluation can be compared with peers on the same dates. If that
context is absent, they degrade to an empty signal frame.
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


def _prepare_universe(ctx: Optional[AlphaContext]) -> pd.DataFrame:
    if ctx is None or ctx.universe_bars is None or ctx.universe_bars.empty:
        return pd.DataFrame()
    uni = ctx.universe_bars.copy()
    required = {"symbol", "date", "close"}
    if not required.issubset(uni.columns):
        return pd.DataFrame()
    uni["date_key"] = pd.to_datetime(uni["date"]).dt.date
    uni["symbol"] = uni["symbol"].astype(str)
    uni["close"] = uni["close"].astype(float)
    return uni.sort_values(["symbol", "date_key"]).reset_index(drop=True)


def _infer_symbol(bars: pd.DataFrame, universe: pd.DataFrame) -> Optional[str]:
    if "symbol" in bars.columns and not bars["symbol"].dropna().empty:
        return str(bars["symbol"].dropna().iloc[0])
    if universe.empty:
        return None

    sample = bars[["date", "close"]].copy()
    sample["date_key"] = pd.to_datetime(sample["date"]).dt.date
    sample = sample[["date_key", "close"]].tail(80)
    best_symbol: Optional[str] = None
    best_error = np.inf
    for symbol, group in universe.groupby("symbol", sort=False):
        merged = sample.merge(group[["date_key", "close"]], on="date_key", suffixes=("_bar", "_uni"))
        if len(merged) < max(5, min(20, len(sample) // 3)):
            continue
        denom = merged["close_bar"].abs().replace(0.0, np.nan)
        error = ((merged["close_bar"] - merged["close_uni"]).abs() / denom).mean()
        if pd.notna(error) and float(error) < best_error:
            best_symbol = str(symbol)
            best_error = float(error)
    return best_symbol


def _universe_matrix(ctx: Optional[AlphaContext]) -> tuple[pd.DataFrame, pd.DataFrame]:
    uni = _prepare_universe(ctx)
    if uni.empty:
        return uni, pd.DataFrame()
    close = uni.pivot_table(index="date_key", columns="symbol", values="close", aggfunc="last")
    return uni, close.sort_index()


def _rows_from_entry_dates(
    df: pd.DataFrame,
    entry_dates: pd.Series,
    score_by_date: pd.Series,
    hold: int,
    note: str,
) -> pd.DataFrame:
    if entry_dates.empty:
        return empty_signal_frame()
    date_to_index = {row.date_key: idx for idx, row in df[["date_key"]].iterrows()}
    rows = []
    for date_key in entry_dates:
        idx = date_to_index.get(date_key)
        if idx is None:
            continue
        rows.append(
            {
                "date": df.iloc[idx]["date"],
                "entry": True,
                "exit": False,
                "side": "long",
                "score": float(score_by_date.get(date_key, 0.0) or 0.0),
                "note": note,
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
                "note": f"{note}_exit",
            }
        )
    return pd.DataFrame(rows) if rows else empty_signal_frame()


def _xs_momentum(lookback: int, quantile: float, hold: int):
    def _generate(bars: pd.DataFrame, ctx: Optional[AlphaContext]) -> pd.DataFrame:
        require_bar_columns(bars)
        cache = (ctx.cache if ctx is not None else None) or {}
        returns = cache.get(f"universe_returns_{lookback}")
        ranks = cache.get(f"universe_rank_{lookback}")
        if returns is None or ranks is None:
            uni, close = _universe_matrix(ctx)
            symbol = _infer_symbol(bars, uni)
            if symbol is None or symbol not in close.columns or len(close) < lookback + hold + 5:
                return empty_signal_frame()
            returns = close.pct_change(lookback)
            ranks = returns.rank(axis=1, pct=True)
        else:
            symbol = _infer_symbol(bars, _prepare_universe(ctx))
            if symbol is None or symbol not in ranks.columns or len(ranks) < lookback + hold + 5:
                return empty_signal_frame()
        signal = ranks[symbol] >= quantile
        signal = signal.fillna(False) & ~signal.shift(1, fill_value=False)

        df = bars.sort_values("date").reset_index(drop=True).copy()
        df["date_key"] = pd.to_datetime(df["date"]).dt.date
        score = returns[symbol].reindex(df["date_key"]).fillna(0.0)
        entries = df.loc[df["date_key"].isin(signal[signal].index), "date_key"]
        return _rows_from_entry_dates(df, entries, score, hold, f"xs_mom_{lookback}")

    return _generate


def _relative_strength_leader(lookback: int, hold: int):
    def _generate(bars: pd.DataFrame, ctx: Optional[AlphaContext]) -> pd.DataFrame:
        require_bar_columns(bars)
        cache = (ctx.cache if ctx is not None else None) or {}
        rel = cache.get(f"rel_spy_returns_{lookback}")
        ranks = cache.get(f"rel_spy_rank_{lookback}")
        if rel is None or ranks is None:
            uni, close = _universe_matrix(ctx)
            symbol = _infer_symbol(bars, uni)
            if (
                ctx is None
                or ctx.spy_bars is None
                or symbol is None
                or symbol not in close.columns
                or len(close) < lookback + hold + 5
            ):
                return empty_signal_frame()
            spy = ctx.spy_bars.copy()
            if not {"date", "close"}.issubset(spy.columns):
                return empty_signal_frame()
            spy["date_key"] = pd.to_datetime(spy["date"]).dt.date
            spy_ret = spy.set_index("date_key")["close"].astype(float).pct_change(
                lookback
            )
            rel = close.pct_change(lookback).sub(spy_ret, axis=0)
            ranks = rel.rank(axis=1, pct=True)
        else:
            symbol = _infer_symbol(bars, _prepare_universe(ctx))
            if symbol is None or symbol not in ranks.columns or len(ranks) < lookback + hold + 5:
                return empty_signal_frame()
        signal = ranks[symbol] >= 0.90
        signal = signal.fillna(False) & ~signal.shift(1, fill_value=False)

        df = bars.sort_values("date").reset_index(drop=True).copy()
        df["date_key"] = pd.to_datetime(df["date"]).dt.date
        score = rel[symbol].reindex(df["date_key"]).fillna(0.0)
        entries = df.loc[df["date_key"].isin(signal[signal].index), "date_key"]
        return _rows_from_entry_dates(df, entries, score, hold, f"rs_leader_{lookback}")

    return _generate


def _dual_momentum(lookback: int, hold: int):
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
        sym_ret = df["close"].astype(float).pct_change(lookback)
        spy_ret = spy.set_index("date_key")["close"].astype(float).pct_change(lookback).reindex(df["date_key"])
        signal = (sym_ret > 0.0) & (sym_ret > spy_ret.to_numpy())
        signal = signal.fillna(False) & ~signal.shift(1, fill_value=False)
        score = pd.Series(sym_ret.to_numpy() - spy_ret.to_numpy(), index=df["date_key"]).fillna(0.0)
        return _rows_from_entry_dates(df, df.loc[signal, "date_key"], score, hold, "dual_mom")

    return _generate


def _low_vol_anomaly(vol_window: int, mom_window: int, hold: int):
    def _generate(bars: pd.DataFrame, ctx: Optional[AlphaContext]) -> pd.DataFrame:
        require_bar_columns(bars)
        cache = (ctx.cache if ctx is not None else None) or {}
        vol_rank = cache.get(f"vol_rank_{vol_window}")
        momentum = cache.get(f"universe_returns_{mom_window}")
        if vol_rank is None or momentum is None:
            uni, close = _universe_matrix(ctx)
            symbol = _infer_symbol(bars, uni)
            if symbol is None or symbol not in close.columns or len(close) < mom_window + hold + 5:
                return empty_signal_frame()
            realized_vol = close.pct_change().rolling(
                vol_window, min_periods=vol_window
            ).std()
            vol_rank = realized_vol.rank(axis=1, pct=True)
            momentum = close.pct_change(mom_window)
        else:
            symbol = _infer_symbol(bars, _prepare_universe(ctx))
            if (
                symbol is None
                or symbol not in vol_rank.columns
                or len(vol_rank) < mom_window + hold + 5
            ):
                return empty_signal_frame()
        signal = (vol_rank[symbol] <= 0.20) & (momentum[symbol] > 0.0)
        signal = signal.fillna(False) & ~signal.shift(1, fill_value=False)

        df = bars.sort_values("date").reset_index(drop=True).copy()
        df["date_key"] = pd.to_datetime(df["date"]).dt.date
        score = (1.0 - vol_rank[symbol]).reindex(df["date_key"]).fillna(0.0)
        entries = df.loc[df["date_key"].isin(signal[signal].index), "date_key"]
        return _rows_from_entry_dates(df, entries, score, hold, "low_vol_anom")

    return _generate


register(
    AlphaDefinition(
        alpha_id="xs_momentum_universe_20d",
        family="cross_sectional",
        hypothesis="Top-decile 20d universe momentum persists over the next month",
        variant="lookback=20,top_decile,hold=10",
        params={"lookback": 20, "quantile": 0.90, "hold": 10},
        regime_compatibility=("TREND", "ROTATION", "NEUTRAL"),
        data_requirements=("needs_universe",),
        generate=_xs_momentum(20, 0.90, 10),
    )
)

register(
    AlphaDefinition(
        alpha_id="xs_momentum_universe_60d",
        family="cross_sectional",
        hypothesis="Top-decile 60d universe momentum captures persistent leadership",
        variant="lookback=60,top_decile,hold=20",
        params={"lookback": 60, "quantile": 0.90, "hold": 20},
        regime_compatibility=("TREND", "ROTATION", "NEUTRAL"),
        data_requirements=("needs_universe",),
        generate=_xs_momentum(60, 0.90, 20),
    )
)

register(
    AlphaDefinition(
        alpha_id="relative_strength_leader_20d",
        family="cross_sectional",
        hypothesis="Leaders versus SPY over 20d continue to attract flows",
        variant="lookback=20,top_decile_vs_spy,hold=10",
        params={"lookback": 20, "hold": 10},
        regime_compatibility=("TREND", "ROTATION"),
        data_requirements=("needs_universe", "needs_spy"),
        generate=_relative_strength_leader(20, 10),
    )
)

register(
    AlphaDefinition(
        alpha_id="dual_momentum_antonacci",
        family="cross_sectional",
        hypothesis="Absolute and relative 6m momentum outperform risk assets",
        variant="lookback=126,hold=21",
        params={"lookback": 126, "hold": 21},
        regime_compatibility=("TREND", "NEUTRAL"),
        data_requirements=("needs_spy",),
        generate=_dual_momentum(126, 21),
    )
)

register(
    AlphaDefinition(
        alpha_id="low_vol_anomaly",
        family="cross_sectional",
        hypothesis="Low realized-volatility winners continue with lower drawdown risk",
        variant="vol_window=60,momentum=126,bottom_quintile,hold=20",
        params={"vol_window": 60, "momentum": 126, "hold": 20},
        regime_compatibility=("NEUTRAL", "ROTATION", "CHOP"),
        data_requirements=("needs_universe",),
        generate=_low_vol_anomaly(60, 126, 20),
    )
)
