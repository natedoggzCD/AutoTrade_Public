"""Cross-phase synchronization helpers for plans and decision logging."""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

PROJECT_DIR = Path(
    os.environ.get("AUTOTRADE_ROOT", Path(__file__).resolve().parents[2])
)
PLANS_DIR = PROJECT_DIR / "plans"
LOG_DIR = PROJECT_DIR / "logs"
PLANS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


def _date_str(day: Any = None) -> str:
    if day is None:
        return datetime.now().strftime("%Y-%m-%d")
    if isinstance(day, (datetime, date)):
        return day.strftime("%Y-%m-%d")
    return str(day)[:10]


def normalize_watchlist(
    items: Sequence[Any], *, source: str = "", phase: str = ""
) -> List[Dict[str, Any]]:
    """Normalize watchlist-like items to a consistent schema."""
    normalized: List[Dict[str, Any]] = []
    for item in items or []:
        base: Dict[str, Any] = {}
        if isinstance(item, Mapping):
            sym = str(item.get("symbol") or item.get("ticker") or "").strip().upper()
            if not sym:
                continue
            base["symbol"] = sym
            base["ticker"] = sym
            score_raw = item.get(
                "score", item.get("final_score", item.get("confidence", 0))
            )
            try:
                base["score"] = float(score_raw)
            except Exception:
                base["score"] = 0.0
            base["has_catalyst"] = bool(
                item.get("has_catalyst", item.get("catalyst", False))
            )
            base["sector"] = item.get("sector") or ""
            reason = item.get("reason") or item.get("note") or item.get("rationale")
            if reason:
                base["reason"] = str(reason)
            adjust = item.get("adjust_reasons") or item.get("reasons")
            if adjust:
                base["reasons"] = adjust
        else:
            sym = str(item).strip().upper()
            if not sym:
                continue
            base = {"symbol": sym, "ticker": sym, "score": 0.0, "has_catalyst": False}

        base["source"] = source
        base["phase"] = phase
        normalized.append(base)
    return normalized


def update_phase_snapshot(
    day: Any = None, *, plans_dir: Path | None = None, **sections: Any
) -> Path:
    """
    Persist a merged snapshot of cross-phase artifacts for a given day.

    Sections are merged into an existing snapshot if present.
    """
    date_str = _date_str(day)
    target_dir = Path(plans_dir) if plans_dir else PLANS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = target_dir / f"phase_sync_{date_str}.json"

    payload: Dict[str, Any] = {}
    if snapshot_path.exists():
        try:
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}

    payload["date"] = date_str
    payload["updated_at"] = datetime.now().isoformat()

    for key, value in sections.items():
        if value is not None:
            payload[key] = value

    snapshot_path.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    return snapshot_path


def record_decisions(
    decisions: Iterable[Mapping[str, Any] | str],
    *,
    phase: str,
    day: Any = None,
    log_dir: Path | None = None,
) -> Path | None:
    """Append structured decision rows to a daily JSONL log."""
    decisions_list = list(decisions or [])
    if not decisions_list:
        return None

    date_str = _date_str(day)
    target_dir = Path(log_dir) if log_dir else LOG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    log_path = target_dir / f"phase_decisions_{date_str}.jsonl"

    timestamp = datetime.now().isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        for dec in decisions_list:
            entry: Dict[str, Any] = {
                "timestamp": timestamp,
                "date": date_str,
                "phase": phase,
            }
            if isinstance(dec, Mapping):
                entry.update({k: v for k, v in dec.items() if v is not None})
            else:
                sym = str(dec).strip().upper()
                if not sym:
                    continue
                entry["symbol"] = sym
                entry["action"] = "noted"
            f.write(json.dumps(entry, default=str) + "\n")

    return log_path

