"""
Position Thesis Cache — Multi-Cycle Memory for the Day Manager
==============================================================
Maintains a structured thesis per open position so LLM advisors
can answer "has anything changed?" instead of re-evaluating from
scratch every cycle.

Persistence: JSON file at data/position_theses.json (saved after
each cycle, loaded on startup). Exited positions are archived to
data/thesis_archive.jsonl for post-market review.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PERSIST_PATH = "data/position_theses.json"
_DEFAULT_ARCHIVE_PATH = "data/thesis_archive.jsonl"
_MAX_HISTORY = 10


@dataclass
class PositionThesis:
    """Structured thesis for a single open position."""

    symbol: str
    side: str = "long"
    entry_reason: str = ""
    bull_case: str = ""
    bear_case: str = ""
    strategy_type: str = ""  # momentum_ride, breakout, mean_reversion, etc.
    key_levels: Dict[str, float] = field(default_factory=dict)
    last_significant_event: str = ""
    last_action: str = "hold"
    last_action_reasoning: str = ""
    consecutive_holds: int = 0
    created_at: str = ""
    updated_at: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        side_key = str(self.side or "long").lower().strip()
        self.side = side_key if side_key in {"long", "short"} else "long"
        now = datetime.utcnow().isoformat(timespec="seconds")
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    # ------------------------------------------------------------------
    def record_action(self, action: str, reasoning: str = "") -> None:
        """Update thesis after an advisor decision."""
        action_lower = (action or "hold").lower().strip()
        self.updated_at = datetime.utcnow().isoformat(timespec="seconds")

        if action_lower == "hold":
            self.consecutive_holds += 1
        else:
            self.consecutive_holds = 0
            self.last_significant_event = f"{action_lower}: {reasoning[:120]}"
            # Push to history (capped)
            self.history.append(
                {
                    "action": action_lower,
                    "reasoning": reasoning[:200],
                    "at": self.updated_at,
                }
            )
            if len(self.history) > _MAX_HISTORY:
                self.history = self.history[-_MAX_HISTORY:]

        self.last_action = action_lower
        self.last_action_reasoning = reasoning[:200]

    def update_from_advisor(self, updates: Dict[str, Any]) -> None:
        """Merge optional thesis updates returned by the advisor/LangGraph."""
        if not isinstance(updates, dict):
            return
        if updates.get("bear_case"):
            self.bear_case = str(updates["bear_case"])[:300]
        if updates.get("bull_case"):
            self.bull_case = str(updates["bull_case"])[:300]
        if updates.get("key_level_update") and isinstance(
            updates["key_level_update"], dict
        ):
            self.key_levels.update(updates["key_level_update"])
        if updates.get("strategy_type"):
            self.strategy_type = str(updates["strategy_type"])

    def to_prompt_context(self) -> str:
        """Render thesis as a text block for LLM prompts."""
        lines = ["PRIOR THESIS:"]
        lines.append(f"  Side: {self.side}")
        if self.entry_reason:
            lines.append(f"  Entry reason: {self.entry_reason}")
        if self.strategy_type:
            lines.append(f"  Strategy: {self.strategy_type}")
        if self.bull_case:
            lines.append(f"  Bull case: {self.bull_case}")
        if self.bear_case:
            lines.append(f"  Bear case: {self.bear_case}")
        if self.key_levels:
            lvl = ", ".join(f"{k}={v:.2f}" for k, v in self.key_levels.items())
            lines.append(f"  Key levels: {lvl}")
        lines.append(
            f"  Last action: {self.last_action}"
            + (f" ({self.last_action_reasoning})" if self.last_action_reasoning else "")
        )
        lines.append(f"  Consecutive holds: {self.consecutive_holds}")
        lines.append(
            "\nQUESTION: Has anything changed vs. this thesis? "
            "Focus on what is DIFFERENT, not what is the same."
        )
        return "\n".join(lines)

    def unrealized_pnl(
        self, entry_price: float, current_price: float, qty: int
    ) -> float:
        """Calculate side-aware unrealized P&L in dollars."""
        entry = float(entry_price or 0.0)
        current = float(current_price or 0.0)
        shares = int(qty or 0)
        if self.side == "short":
            return (entry - current) * shares
        return (current - entry) * shares

    def stop_triggered(self, current_price: float) -> bool:
        """Return true when current price has crossed the thesis stop."""
        if "stop" not in self.key_levels:
            return False
        stop = float(self.key_levels["stop"])
        current = float(current_price or 0.0)
        if self.side == "short":
            return current >= stop
        return current <= stop

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PositionThesis":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# ======================================================================
# Cache Manager
# ======================================================================


class PositionThesisCache:
    """In-memory cache of position theses with JSON persistence."""

    def __init__(
        self,
        persist_path: str = _DEFAULT_PERSIST_PATH,
        archive_path: str = _DEFAULT_ARCHIVE_PATH,
        max_history: int = _MAX_HISTORY,
    ) -> None:
        self._theses: Dict[str, PositionThesis] = {}
        self._persist_path = Path(persist_path)
        self._archive_path = Path(archive_path)
        self._max_history = max_history
        self._session_date: str = date.today().isoformat()
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, symbol: str) -> Optional[PositionThesis]:
        return self._theses.get(str(symbol).upper())

    def get_prompt_context(self, symbol: str) -> str:
        """Return prompt-ready thesis text, or empty string if none."""
        thesis = self.get(symbol)
        return thesis.to_prompt_context() if thesis else ""

    def seed(
        self,
        symbol: str,
        entry_reason: str = "",
        bull_case: str = "",
        bear_case: str = "",
        strategy_type: str = "",
        key_levels: Optional[Dict[str, float]] = None,
        side: str = "long",
    ) -> PositionThesis:
        """Create or update a thesis when a position is first entered."""
        key = str(symbol).upper()
        existing = self._theses.get(key)
        if existing is not None:
            side_key = str(side or "").lower().strip()
            if side_key in {"long", "short"}:
                existing.side = side_key
            # Update fields only if they were empty
            if not existing.entry_reason and entry_reason:
                existing.entry_reason = entry_reason
            if not existing.bull_case and bull_case:
                existing.bull_case = bull_case
            if not existing.bear_case and bear_case:
                existing.bear_case = bear_case
            if not existing.strategy_type and strategy_type:
                existing.strategy_type = strategy_type
            if key_levels:
                existing.key_levels.update(key_levels)
            return existing

        thesis = PositionThesis(
            symbol=key,
            side=side,
            entry_reason=entry_reason,
            bull_case=bull_case,
            bear_case=bear_case,
            strategy_type=strategy_type,
            key_levels=key_levels or {},
        )
        self._theses[key] = thesis
        logger.info(
            "Thesis seeded for %s: strategy=%s", key, strategy_type or "unknown"
        )
        return thesis

    def record_action(self, symbol: str, action: str, reasoning: str = "") -> None:
        """Record an advisor decision for a position."""
        thesis = self.get(str(symbol).upper())
        if thesis is not None:
            thesis.record_action(action, reasoning)

    def update_from_advisor(self, symbol: str, updates: Dict[str, Any]) -> None:
        """Merge optional thesis updates from advisor output."""
        thesis = self.get(str(symbol).upper())
        if thesis is not None:
            thesis.update_from_advisor(updates)

    def remove(self, symbol: str, archive: bool = True) -> Optional[PositionThesis]:
        """Remove a thesis (position exited). Optionally archive it."""
        key = str(symbol).upper()
        thesis = self._theses.pop(key, None)
        if thesis is not None and archive:
            self._archive(thesis)
        return thesis

    def reconcile(self, active_symbols: List[str]) -> List[str]:
        """Remove theses for positions no longer held. Returns archived symbols."""
        active = {str(s).upper() for s in active_symbols}
        stale = [k for k in self._theses if k not in active]
        for sym in stale:
            self.remove(sym, archive=True)
        if stale:
            logger.info("Thesis cache reconciled: archived %s", stale)
        return stale

    def symbols(self) -> List[str]:
        return list(self._theses.keys())

    def save(self) -> None:
        """Persist current theses to JSON."""
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "session_date": self._session_date,
                "theses": {k: v.to_dict() for k, v in self._theses.items()},
            }
            self._persist_path.write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("Failed to persist thesis cache: %s", exc)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load theses from disk if session date matches today."""
        if not self._persist_path.exists():
            return
        try:
            raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
            saved_date = raw.get("session_date", "")
            if saved_date != self._session_date:
                logger.info(
                    "Thesis cache from %s (today=%s) — clearing stale entries",
                    saved_date,
                    self._session_date,
                )
                return
            theses_raw = raw.get("theses", {})
            for key, d in theses_raw.items():
                try:
                    self._theses[key] = PositionThesis.from_dict(d)
                except Exception as inner:
                    logger.warning("Skip malformed thesis for %s: %s", key, inner)
            if self._theses:
                logger.info(
                    "Loaded %d theses from disk: %s",
                    len(self._theses),
                    list(self._theses.keys()),
                )
        except Exception as exc:
            logger.warning("Failed to load thesis cache: %s", exc)

    def _archive(self, thesis: PositionThesis) -> None:
        """Append exited thesis to JSONL archive."""
        try:
            self._archive_path.parent.mkdir(parents=True, exist_ok=True)
            entry = thesis.to_dict()
            entry["archived_at"] = datetime.utcnow().isoformat(timespec="seconds")
            with open(self._archive_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as exc:
            logger.warning("Failed to archive thesis for %s: %s", thesis.symbol, exc)
