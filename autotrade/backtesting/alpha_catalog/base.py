"""Base contract for alpha-catalog modules.

Each alpha module exposes one or more named `AlphaDefinition`s. An alpha is a
theory-grounded systematic signal generator: given a symbol's daily bars (and
optional context like sector membership or SPY series), it emits entry/exit
signals on specific dates. Parameters per variant are fixed by literature or
canonical practice — NOT swept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

import pandas as pd

# Canonical regime tags. These align with what live regime_router uses.
REGIMES = ("TREND", "CHOP", "RISK_OFF_OK", "NEUTRAL", "VOLATILE", "ROTATION")

# Canonical family tags.
FAMILIES = (
    "momentum",
    "trend",
    "mean_reversion",
    "volatility",
    "catalyst",
    "cross_sectional",
    "pattern",
)


@dataclass(frozen=True)
class AlphaSignalRow:
    """One row emitted into the alpha output frame."""

    date: Any  # date or pd.Timestamp
    entry: bool
    exit: bool
    side: str  # 'long' | 'short'
    score: float = 0.0
    note: str = ""


@dataclass(frozen=True)
class AlphaDefinition:
    """Metadata + generator function for one alpha variant."""

    alpha_id: str  # e.g. "ts_momentum_12_1"
    family: str
    hypothesis: str
    variant: str  # human-readable param tag
    params: Dict[str, Any]
    regime_compatibility: tuple
    data_requirements: tuple  # e.g. ("daily_only",), ("needs_earnings",)
    generate: Callable[[pd.DataFrame, Optional["AlphaContext"]], pd.DataFrame]

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family}")
        for r in self.regime_compatibility:
            if r not in REGIMES:
                raise ValueError(f"unknown regime {r}")


@dataclass
class AlphaContext:
    """Optional cross-sectional context passed to alphas that need it.

    For purely single-symbol alphas (momentum, trend, mean-reversion on the
    series itself) context is None. For cross-sectional alphas (sector rank,
    universe RS) the lab populates these frames before evaluation.
    """

    spy_bars: Optional[pd.DataFrame] = None
    sector_map: Dict[str, str] = field(default_factory=dict)
    universe_bars: Optional[pd.DataFrame] = None  # long-form: [symbol,date,close,...]
    earnings_calendar: Optional[pd.DataFrame] = None
    today: Optional[Any] = None
    # Precomputed cross-sectional matrices keyed by descriptive name. Built
    # once in the orchestrator's context-builder so the universe-wide pivot,
    # pct_change, and cross-sectional rank are not redone inside every alpha
    # generator for every symbol. See `lab_orchestrator.build_default_alpha_context`.
    cache: Dict[str, Any] = field(default_factory=dict)


def empty_signal_frame() -> pd.DataFrame:
    """Return a correctly-typed empty signal frame."""
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="object"),
            "entry": pd.Series(dtype="bool"),
            "exit": pd.Series(dtype="bool"),
            "side": pd.Series(dtype="object"),
            "score": pd.Series(dtype="float64"),
            "note": pd.Series(dtype="object"),
        }
    )


def signal_frame_from_rows(rows: List[AlphaSignalRow]) -> pd.DataFrame:
    if not rows:
        return empty_signal_frame()
    return pd.DataFrame(
        {
            "date": [r.date for r in rows],
            "entry": [r.entry for r in rows],
            "exit": [r.exit for r in rows],
            "side": [r.side for r in rows],
            "score": [r.score for r in rows],
            "note": [r.note for r in rows],
        }
    )


REQUIRED_BAR_COLUMNS = ("date", "open", "high", "low", "close", "volume")


def require_bar_columns(bars: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_BAR_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(f"daily_bars missing columns: {missing}")
