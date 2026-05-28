"""
Execution accounting helpers shared across order-submission paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _canonical_action(action_type: str) -> str:
    action = str(action_type or "").strip().lower()
    if "trim" in action:
        return "trim"
    if any(token in action for token in ("entry", "buy", "add", "scale")):
        return "entry"
    return "exit"


@dataclass
class ExecutionAccounting:
    """
    Tracks attempted/success/failed broker submissions for a cycle.
    """

    actions_attempted: int = 0
    orders_submitted: int = 0
    orders_failed: int = 0
    entries_submitted: int = 0
    exits_submitted: int = 0
    trims_submitted: int = 0
    failures_by_action: Dict[str, int] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def attempt(self, action_type: str, symbol: str = "", qty: int = 0, context: str = "") -> None:
        canonical = _canonical_action(action_type)
        self.actions_attempted += 1
        self.events.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "attempt",
                "action_type": canonical,
                "symbol": symbol,
                "qty": int(qty or 0),
                "context": context or "",
            }
        )

    def success(
        self,
        action_type: str,
        symbol: str = "",
        qty: int = 0,
        order_id: Optional[str] = None,
        context: str = "",
    ) -> None:
        canonical = _canonical_action(action_type)
        self.orders_submitted += 1
        if canonical == "entry":
            self.entries_submitted += 1
        elif canonical == "trim":
            self.trims_submitted += 1
        else:
            self.exits_submitted += 1
        self.events.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "submitted",
                "action_type": canonical,
                "symbol": symbol,
                "qty": int(qty or 0),
                "order_id": str(order_id or ""),
                "context": context or "",
            }
        )

    def failure(
        self,
        action_type: str,
        symbol: str = "",
        qty: int = 0,
        error: str = "",
        context: str = "",
    ) -> None:
        canonical = _canonical_action(action_type)
        self.orders_failed += 1
        self.failures_by_action[canonical] = int(self.failures_by_action.get(canonical, 0)) + 1
        self.events.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "failed",
                "action_type": canonical,
                "symbol": symbol,
                "qty": int(qty or 0),
                "error": str(error or ""),
                "context": context or "",
            }
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "actions_attempted": int(self.actions_attempted),
            "orders_submitted": int(self.orders_submitted),
            "orders_failed": int(self.orders_failed),
            "entries_submitted": int(self.entries_submitted),
            "exits_submitted": int(self.exits_submitted),
            "trims_submitted": int(self.trims_submitted),
            "failures_by_action": dict(self.failures_by_action),
        }
