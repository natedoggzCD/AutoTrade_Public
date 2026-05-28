"""PnL audit helpers for comparing EOD review and feedback outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from autotrade.utils.safe_logging import get_safe_logger


LOGGER = get_safe_logger("pnl_audit")


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        LOGGER.warning("Failed to read %s: %s", path, exc)
        return None


def _extract_review_total(review: Dict[str, Any]) -> float:
    trades = review.get("trades") or []
    if trades:
        return float(sum(float(t.get("unrealized_pl", 0.0) or 0.0) for t in trades))

    avg_pnl = float(review.get("avg_pnl", 0.0) or 0.0)
    total_trades = int(review.get("total_trades", 0) or 0)
    return avg_pnl * total_trades


def _file_mtime(path: Path) -> Optional[str]:
    try:
        return path.stat().st_mtime
    except Exception:
        return None


def compare_eod_sources(
    date_str: Optional[str] = None,
    data_dir: Path | str = "data",
) -> Dict[str, Any]:
    """
    Compare EOD review outputs with feedback loop totals for a given date.

    Returns status=ok when both files are found, otherwise status=missing.
    """
    base = Path(data_dir)
    date_key = date_str or "latest"
    review_path = base / f"eod_review_{date_key}.json" if date_str else None
    feedback_path = base / "eod_feedback_latest.json"

    missing = []
    review = None
    if review_path and review_path.exists():
        review = _load_json(review_path)
    else:
        missing.append("eod_review")

    if not feedback_path.exists():
        missing.append("eod_feedback")
        feedback = None
    else:
        feedback = _load_json(feedback_path)

    if missing:
        return {
            "status": "missing",
            "date": date_key,
            "missing": missing,
        }

    review_total = _extract_review_total(review or {})
    feedback_total = float((feedback or {}).get("total_pnl", 0.0) or 0.0)
    delta = feedback_total - review_total
    denom = abs(review_total) if abs(review_total) > 1e-9 else 1.0

    return {
        "status": "ok",
        "date": date_key,
        "eod_review_total_pnl": round(review_total, 4),
        "eod_feedback_total_pnl": round(feedback_total, 4),
        "delta": round(delta, 4),
        "delta_pct": round(delta / denom * 100.0, 2),
        "eod_review_mtime": _file_mtime(review_path) if review_path else None,
        "eod_feedback_mtime": _file_mtime(feedback_path),
    }
