"""
Position Scheduler — Event-Aware Monitoring Cadence
====================================================
Assigns each open position a monitoring tier (CRITICAL / ACTIVE /
STABLE / DORMANT) and tracks when it is next due for LLM advisor
evaluation.  Deterministic checks (hard stops, risk gate) always
run every cycle; only the expensive LLM call is scheduled.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MonitoringTier(Enum):
    """Monitoring cadence tiers — value is default interval in seconds."""

    CRITICAL = 30
    ACTIVE = 60
    STABLE = 180
    DORMANT = 300


# Default tier intervals (overridden by config)
_DEFAULT_INTERVALS: Dict[MonitoringTier, int] = {
    MonitoringTier.CRITICAL: 30,
    MonitoringTier.ACTIVE: 60,
    MonitoringTier.STABLE: 180,
    MonitoringTier.DORMANT: 300,
}


class PositionScheduler:
    """Manages per-symbol monitoring tiers and next-evaluation times."""

    def __init__(
        self,
        tier_intervals: Optional[Dict[str, int]] = None,
        force_eval_price_move_bps: float = 50.0,
        force_eval_pnl_change_pct: float = 1.0,
    ) -> None:
        # Resolve tier intervals from config dict or defaults
        self._intervals: Dict[MonitoringTier, int] = dict(_DEFAULT_INTERVALS)
        if tier_intervals:
            for name, seconds in tier_intervals.items():
                try:
                    tier = MonitoringTier[name.upper()]
                    self._intervals[tier] = max(10, int(seconds))
                except (KeyError, ValueError):
                    pass

        self._force_price_bps = float(force_eval_price_move_bps)
        self._force_pnl_pct = float(force_eval_pnl_change_pct)

        # Per-symbol state
        self._schedule: Dict[str, _SymbolState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_tier(
        self,
        symbol: str,
        pnl_pct: float = 0.0,
        hold_minutes: int = 0,
        hard_stop_pct: float = -8.0,
        atr_distance_stop: float = 999.0,
        atr_distance_target: float = 999.0,
        has_earnings_today: bool = False,
        thesis_last_action: str = "hold",
    ) -> MonitoringTier:
        """Classify a position into a monitoring tier based on current state."""
        # CRITICAL: near stop, brand new, or earnings day
        if pnl_pct <= (hard_stop_pct + 1.0):
            return MonitoringTier.CRITICAL
        if hold_minutes < 15:
            return MonitoringTier.CRITICAL
        if has_earnings_today:
            return MonitoringTier.CRITICAL

        # ACTIVE: still young or moving significantly
        if hold_minutes < 60:
            return MonitoringTier.ACTIVE
        if thesis_last_action not in ("hold", ""):
            return MonitoringTier.ACTIVE

        # DORMANT: well in profit and far from levels
        if pnl_pct > 5.0 and atr_distance_stop > 2.0 and atr_distance_target > 2.0 and hold_minutes > 240:
            return MonitoringTier.DORMANT

        # STABLE: moderate state, thesis intact
        if abs(pnl_pct) < 3.0 and 30 < hold_minutes < 480:
            return MonitoringTier.STABLE

        return MonitoringTier.ACTIVE  # default fallback

    def is_due(self, symbol: str) -> bool:
        """Check whether a position is due for LLM advisor evaluation."""
        key = symbol.upper()
        state = self._schedule.get(key)
        if state is None:
            return True  # never evaluated = due
        return datetime.utcnow() >= state.next_eval_at

    def schedule_next(
        self,
        symbol: str,
        tier: MonitoringTier,
        current_price: float = 0.0,
        pnl_pct: float = 0.0,
    ) -> None:
        """Record evaluation and schedule the next one."""
        key = symbol.upper()
        now = datetime.utcnow()
        interval = self._intervals.get(tier, 60)
        state = self._schedule.get(key)
        if state is None:
            state = _SymbolState()
            self._schedule[key] = state
        state.tier = tier
        state.last_eval_at = now
        state.next_eval_at = now + timedelta(seconds=interval)
        state.last_price = current_price
        state.last_pnl_pct = pnl_pct

    def check_override(
        self,
        symbol: str,
        current_price: float,
        pnl_pct: float,
    ) -> bool:
        """Check if price/PnL movement forces an immediate evaluation."""
        key = symbol.upper()
        state = self._schedule.get(key)
        if state is None:
            return True

        # Price move override
        if state.last_price > 0 and current_price > 0:
            move_bps = abs((current_price - state.last_price) / state.last_price) * 10000.0
            if move_bps >= self._force_price_bps:
                return True

        # PnL change override
        pnl_delta = abs(pnl_pct - state.last_pnl_pct)
        if pnl_delta >= self._force_pnl_pct:
            return True

        return False

    def should_evaluate(
        self,
        symbol: str,
        current_price: float = 0.0,
        pnl_pct: float = 0.0,
    ) -> bool:
        """Combined check: is the position due OR has an override triggered?"""
        return self.is_due(symbol) or self.check_override(symbol, current_price, pnl_pct)

    def get_tier(self, symbol: str) -> Optional[MonitoringTier]:
        state = self._schedule.get(symbol.upper())
        return state.tier if state else None

    def remove(self, symbol: str) -> None:
        self._schedule.pop(symbol.upper(), None)

    def summary(self) -> Dict[str, str]:
        """Return tier summary for logging."""
        return {k: v.tier.name for k, v in self._schedule.items()}


class _SymbolState:
    """Internal per-symbol scheduling state."""

    __slots__ = ("tier", "last_eval_at", "next_eval_at", "last_price", "last_pnl_pct")

    def __init__(self) -> None:
        self.tier: MonitoringTier = MonitoringTier.ACTIVE
        self.last_eval_at: datetime = datetime.min
        self.next_eval_at: datetime = datetime.min
        self.last_price: float = 0.0
        self.last_pnl_pct: float = 0.0
