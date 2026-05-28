"""
Premarket VWAP utility.

Tracks cumulative premarket VWAP from the first print and emits a short-horizon
state label for open-prep decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping


STATE_STRONG_ABOVE = "STRONG_ABOVE"
STATE_STRONG_BELOW = "STRONG_BELOW"
STATE_MIXED = "MIXED"
STATE_WAIT = "WAIT"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class PremarketVWAPSnapshot:
    """Snapshot returned after each bar update."""

    vwap: float
    last_price: float
    total_volume: float
    bars_seen: int
    state: str
    distance_pct: float


class PremarketVWAPTracker:
    """
    Cumulative VWAP tracker for premarket bars.

    State logic is intentionally simple and robust:
    - `WAIT`: not enough bars/volume yet.
    - `STRONG_ABOVE`: consecutive closes above VWAP with enough separation.
    - `STRONG_BELOW`: consecutive closes below VWAP with enough separation.
    - `MIXED`: everything else.
    """

    def __init__(
        self,
        min_bars_for_signal: int = 3,
        strong_threshold_pct: float = 0.15,
        strong_streak_bars: int = 2,
    ) -> None:
        self.min_bars_for_signal = max(1, int(min_bars_for_signal))
        self.strong_threshold_pct = max(0.01, float(strong_threshold_pct))
        self.strong_streak_bars = max(1, int(strong_streak_bars))
        self.reset()

    def reset(self) -> None:
        self._cum_tpv = 0.0
        self._cum_volume = 0.0
        self._bars_seen = 0
        self._above_streak = 0
        self._below_streak = 0
        self._last_vwap = 0.0
        self._last_price = 0.0
        self._last_state = STATE_WAIT
        self._last_distance_pct = 0.0

    def update(self, bar: Mapping[str, Any]) -> PremarketVWAPSnapshot:
        """
        Update tracker from a single OHLCV-like bar.

        Expected keys: `high`, `low`, `close`, `volume`. Missing keys degrade to
        reasonable defaults.
        """
        close = _to_float(bar.get("close"))
        high = _to_float(bar.get("high"), close)
        low = _to_float(bar.get("low"), close)
        volume = max(0.0, _to_float(bar.get("volume")))

        typical_price = (high + low + close) / 3.0 if close > 0 else 0.0
        if volume > 0 and typical_price > 0:
            self._cum_tpv += typical_price * volume
            self._cum_volume += volume

        self._bars_seen += 1
        self._last_price = close
        self._last_vwap = (
            self._cum_tpv / self._cum_volume if self._cum_volume > 0 else 0.0
        )
        self._last_distance_pct = (
            ((close - self._last_vwap) / self._last_vwap) * 100.0
            if self._last_vwap > 0
            else 0.0
        )

        self._update_streaks(close=close, vwap=self._last_vwap)
        self._last_state = self._resolve_state()

        return PremarketVWAPSnapshot(
            vwap=self._last_vwap,
            last_price=self._last_price,
            total_volume=self._cum_volume,
            bars_seen=self._bars_seen,
            state=self._last_state,
            distance_pct=self._last_distance_pct,
        )

    def update_many(self, bars: Iterable[Mapping[str, Any]]) -> PremarketVWAPSnapshot:
        snapshot = PremarketVWAPSnapshot(
            vwap=0.0,
            last_price=0.0,
            total_volume=0.0,
            bars_seen=0,
            state=STATE_WAIT,
            distance_pct=0.0,
        )
        for bar in bars:
            snapshot = self.update(bar)
        return snapshot

    def as_dict(self) -> Dict[str, Any]:
        return {
            "vwap": self._last_vwap,
            "last_price": self._last_price,
            "total_volume": self._cum_volume,
            "bars_seen": self._bars_seen,
            "state": self._last_state,
            "distance_pct": self._last_distance_pct,
        }

    def _update_streaks(self, close: float, vwap: float) -> None:
        if close <= 0 or vwap <= 0:
            self._above_streak = 0
            self._below_streak = 0
            return

        if close > vwap:
            self._above_streak += 1
            self._below_streak = 0
        elif close < vwap:
            self._below_streak += 1
            self._above_streak = 0
        else:
            self._above_streak = 0
            self._below_streak = 0

    def _resolve_state(self) -> str:
        if self._bars_seen < self.min_bars_for_signal or self._cum_volume <= 0:
            return STATE_WAIT

        if (
            self._above_streak >= self.strong_streak_bars
            and self._last_distance_pct >= self.strong_threshold_pct
        ):
            return STATE_STRONG_ABOVE

        if (
            self._below_streak >= self.strong_streak_bars
            and self._last_distance_pct <= -self.strong_threshold_pct
        ):
            return STATE_STRONG_BELOW

        return STATE_MIXED
