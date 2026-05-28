"""
Sequential Shadow Evaluation Schemas and JSONL I/O helpers.

This module is intentionally lightweight and has no trading side effects.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_DIR = Path(
    os.environ.get("AUTOTRADE_ROOT", Path(__file__).resolve().parents[2])
)
LOG_DIR = PROJECT_DIR / "logs"
REPORTS_DIR = PROJECT_DIR / "reports"


def _now_iso() -> str:
    return datetime.now().isoformat()


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _normalize_day_str(day_value: Any) -> str:
    if isinstance(day_value, datetime):
        return day_value.date().isoformat()
    if isinstance(day_value, date):
        return day_value.isoformat()
    text = str(day_value or "").strip()
    if not text:
        return datetime.now().date().isoformat()
    if " " in text:
        text = text.split(" ", 1)[0]
    if "T" in text:
        text = text.split("T", 1)[0]
    return text


@dataclass
class SequentialShadowQueueEvent:
    event_id: str
    symbol: str
    event_type: str  # buy|sell|trim|exit|add
    qty: int
    order_id: str
    submitted_at: str
    due_cycle: int
    fill_confirmed: bool = False
    fill_price: float = 0.0
    fill_qty: int = 0
    fill_time: str = ""
    reason: str = ""
    context: str = ""
    trade_id: str = ""
    processed: bool = False
    expired: bool = False
    attempts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        symbol: str,
        event_type: str,
        qty: int,
        order_id: str = "",
        due_cycle: int = 0,
        reason: str = "",
        context: str = "",
        trade_id: str = "",
        fill_confirmed: bool = False,
        fill_price: float = 0.0,
        fill_qty: int = 0,
        fill_time: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "SequentialShadowQueueEvent":
        sym = str(symbol or "").upper()
        ts = _now_iso()
        event_id = f"{sym}_{event_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        return cls(
            event_id=event_id,
            symbol=sym,
            event_type=str(event_type or "").lower(),
            qty=int(qty or 0),
            order_id=str(order_id or ""),
            submitted_at=ts,
            due_cycle=int(due_cycle or 0),
            fill_confirmed=bool(fill_confirmed),
            fill_price=float(fill_price or 0.0),
            fill_qty=int(fill_qty or 0),
            fill_time=str(fill_time or ""),
            reason=str(reason or ""),
            context=str(context or ""),
            trade_id=str(trade_id or ""),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "SequentialShadowQueueEvent":
        row = dict(payload or {})
        return cls(
            event_id=_as_str(row.get("event_id"), ""),
            symbol=_as_str(row.get("symbol"), "").upper(),
            event_type=_as_str(row.get("event_type"), "").lower(),
            qty=_as_int(row.get("qty"), 0),
            order_id=_as_str(row.get("order_id"), ""),
            submitted_at=_as_str(row.get("submitted_at"), _now_iso()),
            due_cycle=_as_int(row.get("due_cycle"), 0),
            fill_confirmed=_as_bool(row.get("fill_confirmed"), False),
            fill_price=_as_float(row.get("fill_price"), 0.0),
            fill_qty=_as_int(row.get("fill_qty"), 0),
            fill_time=_as_str(row.get("fill_time"), ""),
            reason=_as_str(row.get("reason"), ""),
            context=_as_str(row.get("context"), ""),
            trade_id=_as_str(row.get("trade_id"), ""),
            processed=_as_bool(row.get("processed"), False),
            expired=_as_bool(row.get("expired"), False),
            attempts=_as_int(row.get("attempts"), 0),
            metadata=dict(row.get("metadata", {}) or {}),
        )


@dataclass
class SequentialShadowReadyEvent:
    event_id: str
    symbol: str
    event_type: str
    fill_time: str
    fill_price: float
    fill_qty: int
    reason: str
    baseline_action: str
    context_bundle: Dict[str, Any]
    created_at: str = field(default_factory=_now_iso)
    order_id: str = ""
    trade_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SequentialShadowPrediction:
    event_id: str
    symbol: str
    event_type: str
    recommended_action: str
    confidence: float
    summary_reasoning: str
    thought_count: int = 0
    revision_count: int = 0
    branch_count: int = 0
    latency_ms: int = 0
    timed_out: bool = False
    error: str = ""
    model: str = ""
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SequentialShadowOutcome:
    event_id: str
    symbol: str
    event_type: str
    baseline_action: str
    sequential_action: str
    actual_outcome_score: float
    counterfactual_score: float
    score_delta: float
    sequential_more_accurate: bool
    notes: str = ""
    metric_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    _ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True, default=str))
        f.write("\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                out.append(row)
    return out


def queue_path_for_day(day_str: Any) -> Path:
    day = _normalize_day_str(day_str)
    return LOG_DIR / f"sequential_shadow_queue_{day}.jsonl"


def ready_path_for_day(day_str: Any) -> Path:
    day = _normalize_day_str(day_str)
    return LOG_DIR / f"sequential_shadow_ready_{day}.jsonl"


def prediction_path_for_day(day_str: Any) -> Path:
    day = _normalize_day_str(day_str)
    return LOG_DIR / f"sequential_shadow_predictions_{day}.jsonl"


def report_path_for_day(day_str: Any) -> Path:
    day = _normalize_day_str(day_str)
    return REPORTS_DIR / f"sequential_shadow_accuracy_{day}.json"


def event_ids(rows: Iterable[Dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for row in rows:
        eid = str(row.get("event_id") or "").strip()
        if eid:
            ids.add(eid)
    return ids
