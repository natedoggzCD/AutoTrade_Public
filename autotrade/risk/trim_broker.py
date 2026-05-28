"""Central broker for duplicate-trim and aggregate-trim governance."""

import threading
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple


class TrimBroker:
    """Approve or reject trim proposals using one persisted history."""

    def __init__(
        self,
        session_state,
        cooldown_min: int = 240,
        max_per_day: int = 2,
        aggregate_cap_pct: float = 0.50,
        aggregate_cap_red_day_max_pct: float = 0.80,
        aggregate_cap_red_day_loss_scale_pct: float = 5.0,
    ):
        self._state = session_state
        self._cooldown = timedelta(minutes=max(0, int(cooldown_min or 0)))
        self._max_per_day = max(0, int(max_per_day or 0))
        self._aggregate_cap_pct = max(0.0, min(1.0, float(aggregate_cap_pct or 0.0)))
        self._aggregate_cap_red_day_max_pct = max(
            self._aggregate_cap_pct,
            min(1.0, float(aggregate_cap_red_day_max_pct or self._aggregate_cap_pct)),
        )
        self._aggregate_cap_red_day_loss_scale_pct = max(
            0.1, float(aggregate_cap_red_day_loss_scale_pct or 5.0)
        )
        self._lock = threading.RLock()

    def _effective_aggregate_cap_pct(self, book_pnl_pct: float = 0.0) -> float:
        """Raise the trim cap only as the whole book gets meaningfully red."""
        loss_pct = max(0.0, -float(book_pnl_pct or 0.0))
        if loss_pct <= 0.0:
            return self._aggregate_cap_pct
        scaled = self._aggregate_cap_pct + (
            0.15 * (loss_pct / self._aggregate_cap_red_day_loss_scale_pct)
        )
        return min(
            self._aggregate_cap_red_day_max_pct,
            max(self._aggregate_cap_pct, scaled),
        )

    def propose_trim(
        self,
        symbol: str,
        qty: int,
        source: str,
        reason: str,
        urgency: str = "normal",
        current_qty: int = 0,
        book_pnl_pct: float = 0.0,
        position_pnl_pct: float = 0.0,
    ) -> Tuple[bool, str]:
        symbol_key = str(symbol or "").upper().strip()
        trim_qty = int(qty or 0)
        if not symbol_key or trim_qty <= 0:
            return False, "invalid_trim_proposal"

        with self._lock:
            trim_history: Dict[str, Any] = self._state.get("trim_history", {}) or {}
            history = list(trim_history.get(symbol_key, []) or [])
            now = datetime.now()

            if urgency != "critical" and history and self._cooldown.total_seconds() > 0:
                try:
                    last = datetime.fromisoformat(str(history[-1].get("ts")))
                    if now - last < self._cooldown:
                        elapsed = int((now - last).total_seconds() // 60)
                        limit = int(self._cooldown.total_seconds() // 60)
                        return False, f"cooldown_active:{elapsed}m<{limit}m"
                except Exception:
                    pass

            if self._max_per_day > 0 and len(history) >= self._max_per_day:
                return (
                    False,
                    f"max_per_day_exceeded:{len(history)}>={self._max_per_day}",
                )

            qty_before = max(int(current_qty or 0), trim_qty)
            original_qty = (
                int(history[0].get("qty_before") or qty_before)
                if history
                else qty_before
            )
            cumulative_trimmed = sum(int(row.get("qty", 0) or 0) for row in history)
            proposed_total_pct = (cumulative_trimmed + trim_qty) / max(original_qty, 1)
            aggregate_cap_pct = self._effective_aggregate_cap_pct(book_pnl_pct)
            profit_take_override = (
                str(reason or "").lower().startswith("profit_take")
                and float(position_pnl_pct or 0.0) >= 10.0
                and (trim_qty / max(qty_before, 1)) <= 0.75
            )
            if (
                aggregate_cap_pct < 1.0
                and proposed_total_pct > aggregate_cap_pct
                and not profit_take_override
            ):
                return False, (
                    "aggregate_cap_exceeded:"
                    f"{proposed_total_pct:.2f}>{aggregate_cap_pct:.2f}"
                )

            history.append(
                {
                    "ts": now.isoformat(),
                    "qty": trim_qty,
                    "qty_before": qty_before,
                    "source": str(source or "unknown"),
                    "reason": str(reason or ""),
                }
            )
            trim_history[symbol_key] = history
            trim_count = self._state.get("trim_count_today", {}) or {}
            trim_count[symbol_key] = len(history)
            self._state.update_many(
                {"trim_history": trim_history, "trim_count_today": trim_count}
            )
            return True, "approved"
