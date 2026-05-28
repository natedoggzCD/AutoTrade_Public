"""Catalyst and event-driven alpha definitions."""

from __future__ import annotations

import sqlite3
from pathlib import Path
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


FINANCIAL_DB_PATH = Path("data/financial.db")


def _symbol_from_bars(bars: pd.DataFrame) -> Optional[str]:
    if "symbol" in bars.columns and not bars["symbol"].dropna().empty:
        return str(bars["symbol"].dropna().iloc[0]).upper()
    return None


def _event_rows(
    df: pd.DataFrame, entry_indices: list[int], scores: pd.Series, hold: int, note: str
) -> pd.DataFrame:
    rows = []
    for idx in entry_indices:
        rows.append(
            {
                "date": df.iloc[idx]["date"],
                "entry": True,
                "exit": False,
                "side": "long",
                "score": float(scores.iloc[idx]) if pd.notna(scores.iloc[idx]) else 0.0,
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


def _earnings_events_from_ctx(
    ctx: Optional[AlphaContext], symbol: Optional[str]
) -> pd.DataFrame:
    if ctx is None or ctx.earnings_calendar is None or ctx.earnings_calendar.empty:
        return pd.DataFrame()
    ev = ctx.earnings_calendar.copy()
    if "date" not in ev.columns:
        for col in ("earnings_date", "report_date"):
            if col in ev.columns:
                ev = ev.rename(columns={col: "date"})
                break
    if "date" not in ev.columns:
        return pd.DataFrame()
    if symbol and "symbol" in ev.columns:
        ev = ev[ev["symbol"].astype(str).str.upper() == symbol]
    surprise_col = next(
        (
            c
            for c in ("surprise_pct", "eps_surprise_pct", "surprise")
            if c in ev.columns
        ),
        None,
    )
    if surprise_col is None:
        ev["surprise_score"] = 1.0
    else:
        ev["surprise_score"] = pd.to_numeric(ev[surprise_col], errors="coerce").fillna(
            0.0
        )
        ev = ev[ev["surprise_score"] > 0.0]
    ev["date_key"] = pd.to_datetime(ev["date"]).dt.date
    return ev[["date_key", "surprise_score"]].drop_duplicates("date_key", keep="last")


def _analyst_events_from_db(symbol: Optional[str]) -> pd.DataFrame:
    if symbol is None or not FINANCIAL_DB_PATH.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(FINANCIAL_DB_PATH) as conn:
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='analyst_ratings'",
                conn,
            )
            if tables.empty:
                return pd.DataFrame()
            cols = pd.read_sql_query("PRAGMA table_info(analyst_ratings)", conn)
            symbol_col = "symbol" if "symbol" in set(cols["name"]) else "ticker"
            ratings = pd.read_sql_query(
                f"SELECT * FROM analyst_ratings WHERE upper({symbol_col})=?",
                conn,
                params=(symbol,),
            )
    except Exception:
        return pd.DataFrame()
    if ratings.empty or "date" not in ratings.columns:
        return pd.DataFrame()
    if "symbol" not in ratings.columns and "ticker" in ratings.columns:
        ratings = ratings.rename(columns={"ticker": "symbol"})
    agency_col = (
        "agency"
        if "agency" in ratings.columns
        else "firm"
        if "firm" in ratings.columns
        else None
    )
    subset = ["symbol", "date"] + ([agency_col] if agency_col else [])
    ratings = ratings.drop_duplicates(subset=subset, keep="last")
    text = " ".join(
        c
        for c in ("action", "rating", "to_grade", "new_rating")
        if c in ratings.columns
    )
    if text:
        joined = (
            ratings[[c for c in text.split() if c in ratings.columns]]
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
        )
        ratings = ratings[
            joined.str.contains(
                "upgrade|buy|outperform|overweight|positive", regex=True, na=False
            )
        ]
    ratings["date_key"] = pd.to_datetime(ratings["date"]).dt.date
    ratings["score"] = 1.0
    return ratings[["date_key", "score"]].drop_duplicates("date_key", keep="last")


def _pead_drift(hold: int):
    def _generate(bars: pd.DataFrame, ctx: Optional[AlphaContext]) -> pd.DataFrame:
        require_bar_columns(bars)
        df = bars.sort_values("date").reset_index(drop=True).copy()
        symbol = _symbol_from_bars(df)
        events = _earnings_events_from_ctx(ctx, symbol)
        if events.empty:
            return empty_signal_frame()
        df["date_key"] = pd.to_datetime(df["date"]).dt.date
        event_scores = events.set_index("date_key")["surprise_score"]
        entry_indices = []
        for date_key in event_scores.index:
            matches = df.index[df["date_key"] > date_key]
            if len(matches):
                entry_indices.append(int(matches[0]))
        scores = df["date_key"].map(event_scores).fillna(1.0)
        return _event_rows(df, entry_indices, scores, hold, f"pead_{hold}d")

    return _generate


def _gap_continuation(gap_pct: float, hold: int):
    def _generate(bars: pd.DataFrame, ctx: Optional[AlphaContext]) -> pd.DataFrame:
        require_bar_columns(bars)
        df = bars.sort_values("date").reset_index(drop=True)
        if len(df) < hold + 25:
            return empty_signal_frame()
        open_, close = df["open"].astype(float), df["close"].astype(float)
        prev_close = close.shift(1)
        prev_mid = (
            df["high"].astype(float).shift(1) + df["low"].astype(float).shift(1)
        ) / 2.0
        gap = open_ / prev_close.replace(0.0, np.nan) - 1.0
        entry = (gap >= gap_pct) & (prev_close >= prev_mid)
        entry = entry.fillna(False) & ~entry.shift(1, fill_value=False)
        return _event_rows(
            df,
            [int(i) for i in np.flatnonzero(entry.values)],
            gap.abs(),
            hold,
            f"gap_cont_{gap_pct:g}",
        )

    return _generate


def _gap_fade_exhaustion(hold: int):
    def _generate(bars: pd.DataFrame, ctx: Optional[AlphaContext]) -> pd.DataFrame:
        require_bar_columns(bars)
        df = bars.sort_values("date").reset_index(drop=True)
        if len(df) < 30:
            return empty_signal_frame()
        open_, close = df["open"].astype(float), df["close"].astype(float)
        gap = open_ / close.shift(1).replace(0.0, np.nan) - 1.0
        returns = close.pct_change()
        same_dir_move = (
            (returns.shift(1) > 0) & (returns.shift(2) > 0) & (returns.shift(3) > 0)
        )
        vol_surge = (
            df["volume"].astype(float)
            > 1.5 * df["volume"].astype(float).rolling(20, min_periods=20).mean()
        )
        entry = (gap >= 0.05) & same_dir_move & vol_surge
        entry = entry.fillna(False) & ~entry.shift(1, fill_value=False)
        rows = []
        for idx in np.flatnonzero(entry.values):
            rows.append(
                {
                    "date": df.iloc[int(idx)]["date"],
                    "entry": True,
                    "exit": False,
                    "side": "short",
                    "score": float(gap.iloc[int(idx)]),
                    "note": "gap_fade_exhaustion",
                }
            )
            exit_idx = min(int(idx) + hold, len(df) - 1)
            rows.append(
                {
                    "date": df.iloc[exit_idx]["date"],
                    "entry": False,
                    "exit": True,
                    "side": "short",
                    "score": 0.0,
                    "note": "gap_fade_exit",
                }
            )
        return pd.DataFrame(rows) if rows else empty_signal_frame()

    return _generate


def _analyst_upgrade_drift(hold: int):
    def _generate(bars: pd.DataFrame, ctx: Optional[AlphaContext]) -> pd.DataFrame:
        require_bar_columns(bars)
        df = bars.sort_values("date").reset_index(drop=True).copy()
        symbol = _symbol_from_bars(df)
        events = _analyst_events_from_db(symbol)
        if events.empty:
            return empty_signal_frame()
        df["date_key"] = pd.to_datetime(df["date"]).dt.date
        entry_indices = [
            int(i) for i in df.index[df["date_key"].isin(events["date_key"])]
        ]
        return _event_rows(
            df, entry_indices, pd.Series(1.0, index=df.index), hold, "analyst_upgrade"
        )

    return _generate


def _high52w_break_volume(hold: int):
    def _generate(bars: pd.DataFrame, ctx: Optional[AlphaContext]) -> pd.DataFrame:
        require_bar_columns(bars)
        df = bars.sort_values("date").reset_index(drop=True)
        if len(df) < 252 + hold + 5:
            return empty_signal_frame()
        close = df["close"].astype(float)
        prior_high = (
            df["high"].astype(float).rolling(252, min_periods=252).max().shift(1)
        )
        volume = df["volume"].astype(float)
        vol_ratio = volume / volume.rolling(20, min_periods=20).mean().replace(
            0.0, np.nan
        )
        entry = (close > prior_high) & (vol_ratio > 1.5)
        entry = entry.fillna(False) & ~entry.shift(1, fill_value=False)
        return _event_rows(
            df,
            [int(i) for i in np.flatnonzero(entry.values)],
            vol_ratio.fillna(0.0),
            hold,
            "high52w_break_volume",
        )

    return _generate


register(
    AlphaDefinition(
        alpha_id="pead_drift_3d",
        family="catalyst",
        hypothesis="Positive earnings surprise drifts for several sessions after announcement",
        variant="positive_surprise,hold=3",
        params={"hold": 3},
        regime_compatibility=("TREND", "NEUTRAL", "ROTATION"),
        data_requirements=("needs_earnings",),
        generate=_pead_drift(3),
    )
)

register(
    AlphaDefinition(
        alpha_id="pead_drift_10d",
        family="catalyst",
        hypothesis="Positive earnings surprise drift persists over a two-week window",
        variant="positive_surprise,hold=10",
        params={"hold": 10},
        regime_compatibility=("TREND", "NEUTRAL", "ROTATION"),
        data_requirements=("needs_earnings",),
        generate=_pead_drift(10),
    )
)

register(
    AlphaDefinition(
        alpha_id="gap_continuation_3pct",
        family="catalyst",
        hypothesis="Constructive 3% overnight gaps continue when prior close was strong",
        variant="gap>=3%,hold=3",
        params={"gap": 0.03, "hold": 3},
        regime_compatibility=("TREND", "VOLATILE", "NEUTRAL"),
        data_requirements=("daily_only",),
        generate=_gap_continuation(0.03, 3),
    )
)

register(
    AlphaDefinition(
        alpha_id="gap_continuation_5pct",
        family="catalyst",
        hypothesis="Constructive 5% overnight gaps continue after strong prior-day closes",
        variant="gap>=5%,hold=3",
        params={"gap": 0.05, "hold": 3},
        regime_compatibility=("TREND", "VOLATILE"),
        data_requirements=("daily_only",),
        generate=_gap_continuation(0.05, 3),
    )
)

register(
    AlphaDefinition(
        alpha_id="gap_fade_exhaustion",
        family="catalyst",
        hypothesis="Large gap after a three-day same-direction move often exhausts near-term demand",
        variant="gap>=5%,prior_3d_up,volume_surge,hold=1",
        params={"gap": 0.05, "hold": 1},
        regime_compatibility=("VOLATILE", "CHOP"),
        data_requirements=("daily_only",),
        generate=_gap_fade_exhaustion(1),
    )
)

register(
    AlphaDefinition(
        alpha_id="analyst_upgrade_drift",
        family="catalyst",
        hypothesis="Fresh analyst upgrades drift as institutions digest estimate changes",
        variant="upgrade_event,hold=10",
        params={"hold": 10},
        regime_compatibility=("TREND", "NEUTRAL", "ROTATION"),
        data_requirements=("needs_financial_db",),
        generate=_analyst_upgrade_drift(10),
    )
)

register(
    AlphaDefinition(
        alpha_id="high52w_break_volume",
        family="catalyst",
        hypothesis="New 52-week highs on volume confirm institutional accumulation",
        variant="prior_252d_high,volume>1.5x,hold=20",
        params={"hold": 20, "volume_ratio": 1.5},
        regime_compatibility=("TREND", "ROTATION"),
        data_requirements=("daily_only",),
        generate=_high52w_break_volume(20),
    )
)
