"""Loss-floor exit policy (H8).

day-manager 2026-05-19 (H8): user-strategic directive — "there is no
sense in holding losing positions; we can always buy back in on a
bounce." This module evaluates each held position once per cycle and
returns a list of positions to exit NOW, decoupled from scoring,
posture, and the replacement engine.

D-questions settled in handoff:
  D1: ladder = -2%@60min, -3.5%@30min, -5%@anytime (configurable)
  D2: re-entry only via strength_reentry path (proven bounce)
  D3: pnl_pct alone fires exit, no score check
  D4: market sell directly via execute_exit(reason="loss_floor")
  D5: no daily circuit-breaker tie-in here — position-level only

This module is active only when `loss_floor.enabled: true` is set in
config/trading_config.yaml. It uses cumulative position P&L, not the
position's daily move.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# D1 default ladder. Each tier: (max_pnl_pct, min_held_minutes).
# A position triggers exit when pnl_pct <= max_pnl_pct AND held_minutes >= min_held_minutes.
DEFAULT_LADDER: List[Tuple[float, float]] = [
    (-5.0, 0.0),    # anytime hard floor
    (-3.5, 30.0),   # 30 min mid floor
    (-2.0, 60.0),   # 60 min soft floor
]


@dataclass
class LossFloorDecision:
    symbol: str
    qty: float
    entry_price: float
    current_price: float
    pnl_pct: float
    held_minutes: float
    decision: str  # "exit" | "hold"
    tier_fired: Optional[str] = None  # e.g. "-2%@60min", "-3.5%@30min", "-5%@anytime"
    reason: str = ""


@dataclass
class LossFloorChecker:
    """Stateless per-cycle evaluator. Wire from DayManager._run_cycle_inner."""

    ladder: List[Tuple[float, float]] = field(default_factory=lambda: list(DEFAULT_LADDER))
    enabled: bool = False  # default OFF until explicitly enabled
    telemetry_path: Optional[Path] = None

    # In-memory re-entry guard: symbol -> ISO timestamp of last loss_floor exit.
    # Day-manager owns persistence (writes to plans/loss_floor_exit_history.json
    # on each evaluate call when telemetry_path is set).
    exit_history: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], *, telemetry_path: Optional[Path] = None) -> "LossFloorChecker":
        """Construct from a config dict (typically `trading_config.yaml['loss_floor']`)."""
        cfg = cfg or {}
        enabled = bool(cfg.get("enabled", False))
        ladder_cfg = cfg.get("ladder") or []
        if ladder_cfg:
            ladder = [
                (float(t.get("max_pnl_pct", -5.0)), float(t.get("min_held_minutes", 0.0)))
                for t in ladder_cfg
                if isinstance(t, dict)
            ]
        else:
            ladder = list(DEFAULT_LADDER)
        # Sort hardest-first so the strictest tier fires first when multiple match.
        ladder.sort(key=lambda t: t[0])
        return cls(ladder=ladder, enabled=enabled, telemetry_path=telemetry_path)

    @staticmethod
    def _format_tier(max_pnl_pct: float, min_held_minutes: float) -> str:
        if min_held_minutes <= 0:
            return f"{max_pnl_pct:.1f}%@anytime"
        return f"{max_pnl_pct:.1f}%@{int(min_held_minutes)}min"

    def _evaluate_one(
        self,
        symbol: str,
        *,
        qty: float,
        entry_price: float,
        current_price: float,
        held_minutes: float,
        pnl_pct: Optional[float] = None,
    ) -> LossFloorDecision:
        if entry_price <= 0 or current_price <= 0 or qty <= 0:
            return LossFloorDecision(
                symbol=symbol,
                qty=qty,
                entry_price=entry_price,
                current_price=current_price,
                pnl_pct=0.0,
                held_minutes=held_minutes,
                decision="hold",
                reason="invalid_inputs",
            )
        if pnl_pct is None:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100.0
        else:
            pnl_pct = float(pnl_pct)

        # Hardest tier first (most negative max_pnl_pct).
        for max_pnl_pct, min_held in self.ladder:
            if pnl_pct <= max_pnl_pct and held_minutes >= min_held:
                tier = self._format_tier(max_pnl_pct, min_held)
                return LossFloorDecision(
                    symbol=symbol,
                    qty=qty,
                    entry_price=entry_price,
                    current_price=current_price,
                    pnl_pct=round(pnl_pct, 3),
                    held_minutes=round(held_minutes, 2),
                    decision="exit",
                    tier_fired=tier,
                    reason=f"loss_floor:{tier}",
                )
        return LossFloorDecision(
            symbol=symbol,
            qty=qty,
            entry_price=entry_price,
            current_price=current_price,
            pnl_pct=round(pnl_pct, 3),
            held_minutes=round(held_minutes, 2),
            decision="hold",
        )

    def evaluate(self, positions: Iterable[Dict[str, Any]]) -> List[LossFloorDecision]:
        """Evaluate every held position; return all decisions (exits + holds).

        Caller is expected to filter for `decision == 'exit'` and feed
        those symbols through `execute_exit(symbol, qty, reason='loss_floor')`.

        Each position dict must carry:
          - symbol (str)
          - qty (float)
          - entry_price OR avg_entry_price (float)
          - current_price OR last_price (float)
          - held_minutes (float)
        """
        decisions: List[LossFloorDecision] = []
        if not self.enabled:
            return decisions

        for pos in positions or []:
            if not isinstance(pos, dict):
                continue
            symbol = str(pos.get("symbol") or pos.get("ticker") or "").upper()
            if not symbol:
                continue
            try:
                qty = float(pos.get("qty") or pos.get("quantity") or 0.0)
                entry_price = float(
                    pos.get("entry_price")
                    or pos.get("avg_entry_price")
                    or pos.get("cost_basis_per_share")
                    or 0.0
                )
                current_price = float(
                    pos.get("current_price")
                    or pos.get("last_price")
                    or pos.get("market_price")
                    or 0.0
                )
                held_minutes = float(pos.get("held_minutes") or 0.0)
                if pos.get("pnl_pct") is not None:
                    pnl_pct = float(pos.get("pnl_pct"))
                elif pos.get("unrealized_plpc") is not None:
                    pnl_pct = float(pos.get("unrealized_plpc")) * 100.0
                else:
                    pnl_pct = None
            except (TypeError, ValueError):
                continue

            decision = self._evaluate_one(
                symbol,
                qty=qty,
                entry_price=entry_price,
                current_price=current_price,
                held_minutes=held_minutes,
                pnl_pct=pnl_pct,
            )
            decisions.append(decision)

        # Record exits in re-entry guard + telemetry.
        now_iso = datetime.now().isoformat()
        for d in decisions:
            if d.decision == "exit":
                self.exit_history[d.symbol] = now_iso

        if self.telemetry_path:
            self._append_telemetry(decisions, now_iso)

        return decisions

    def _append_telemetry(self, decisions: List[LossFloorDecision], ts: str) -> None:
        try:
            self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            with self.telemetry_path.open("a", encoding="utf-8") as f:
                for d in decisions:
                    f.write(
                        json.dumps(
                            {
                                "ts": ts,
                                "symbol": d.symbol,
                                "qty": d.qty,
                                "entry_price": d.entry_price,
                                "current_price": d.current_price,
                                "pnl_pct": d.pnl_pct,
                                "held_minutes": d.held_minutes,
                                "decision": d.decision,
                                "tier_fired": d.tier_fired,
                            }
                        )
                        + "\n"
                    )
        except Exception:
            # Telemetry must never break the decision path.
            pass

    def should_block_reentry(self, symbol: str, *, strength_reentry_confirmed: bool = False) -> bool:
        """D2 re-entry policy: a symbol exited via loss_floor cannot be
        re-entered same session via the normal entry funnel; only the
        strength_reentry path (which requires a proven bounce + volume
        + VWAP reclaim) clears the guard."""
        symbol = (symbol or "").upper()
        if symbol not in self.exit_history:
            return False
        if strength_reentry_confirmed:
            return False
        return True

    def clear_history(self) -> None:
        """Called at session boundary (EOD or PM workflow start)."""
        self.exit_history.clear()
