"""H1 D2: Overnight-cut watchlist scaffold.

When the T-15 weak-position cull policy trims positions, cut records are
persisted to ``data/overnight_cuts_<date>.json``. The
next morning's signal-generation pipeline reads this file; any cut symbol
that re-appears in tomorrow's signal list is tagged with
``priority=overnight_recovery`` and gets a small rank boost so the
agent gets first crack at re-entering names it already had a thesis on.

Public API:
    record_cut(symbol, exit_price, exit_reason, prior_avg_entry, date=None)
    load_cuts(date) -> List[CutRecord]
    apply_recovery_boost(candidates, date, *, enabled=False, boost=...)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date as _date, datetime
from pathlib import Path
from typing import Any, Iterable, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CUTS_DIR = REPO_ROOT / "data"

RECOVERY_PRIORITY_TAG = "overnight_recovery"
DEFAULT_RECOVERY_BOOST = 5.0


@dataclass
class CutRecord:
    symbol: str
    exit_price: float
    exit_reason: str
    prior_avg_entry: float
    cut_at: str  # ISO timestamp
    qty: float = 0.0
    trim_fraction: float = 0.0
    policy_mode: str = ""
    weak_signal_pct: Optional[float] = None
    order_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CutRecord":
        return cls(
            symbol=str(d.get("symbol", "")).upper(),
            exit_price=float(d.get("exit_price", 0.0) or 0.0),
            exit_reason=str(d.get("exit_reason", "")),
            prior_avg_entry=float(d.get("prior_avg_entry", 0.0) or 0.0),
            cut_at=str(d.get("cut_at", "")),
            qty=float(d.get("qty", 0.0) or 0.0),
            trim_fraction=float(d.get("trim_fraction", 0.0) or 0.0),
            policy_mode=str(d.get("policy_mode", "")),
            weak_signal_pct=(
                float(d["weak_signal_pct"])
                if d.get("weak_signal_pct") is not None
                else None
            ),
            order_id=str(d.get("order_id", "")),
        )


def cuts_path(d: _date) -> Path:
    return CUTS_DIR / f"overnight_cuts_{d.isoformat()}.json"


def record_cut(
    symbol: str,
    exit_price: float,
    exit_reason: str,
    prior_avg_entry: float,
    *,
    when: Optional[_date] = None,
    cut_at: Optional[datetime] = None,
    qty: float = 0.0,
    trim_fraction: float = 0.0,
    policy_mode: str = "",
    weak_signal_pct: Optional[float] = None,
    order_id: str = "",
) -> CutRecord:
    """Append a cut record to data/overnight_cuts_<date>.json.

    File format: JSON array of CutRecord dicts. Idempotent on (symbol)
    within the same file — re-recording the same symbol updates the
    existing row instead of duplicating.
    """
    when = when or datetime.now().date()
    rec = CutRecord(
        symbol=str(symbol or "").upper(),
        exit_price=float(exit_price or 0.0),
        exit_reason=str(exit_reason or ""),
        prior_avg_entry=float(prior_avg_entry or 0.0),
        cut_at=(cut_at or datetime.now()).isoformat(),
        qty=float(qty or 0.0),
        trim_fraction=float(trim_fraction or 0.0),
        policy_mode=str(policy_mode or ""),
        weak_signal_pct=(
            float(weak_signal_pct) if weak_signal_pct is not None else None
        ),
        order_id=str(order_id or ""),
    )
    path = cuts_path(when)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: List[CutRecord] = []
    if path.exists():
        try:
            existing = [CutRecord.from_dict(r) for r in json.loads(path.read_text(encoding="utf-8"))]
        except Exception as e:
            logger.warning("overnight_cuts: failed to load %s: %s", path, e)
            existing = []
    existing = [r for r in existing if r.symbol != rec.symbol]
    existing.append(rec)
    path.write_text(
        json.dumps([r.to_dict() for r in existing], indent=2),
        encoding="utf-8",
    )
    return rec


def load_cuts(when: _date) -> List[CutRecord]:
    path = cuts_path(when)
    if not path.exists():
        return []
    try:
        return [CutRecord.from_dict(r) for r in json.loads(path.read_text(encoding="utf-8"))]
    except Exception as e:
        logger.warning("overnight_cuts: failed to load %s: %s", path, e)
        return []


def apply_recovery_boost(
    candidates: Iterable[Any],
    cuts: List[CutRecord],
    *,
    enabled: bool = False,
    boost: float = DEFAULT_RECOVERY_BOOST,
    score_attr: str = "entry_score",
    symbol_attr: str = "symbol",
    metadata_attr: str = "metadata",
) -> int:
    """Boost entry_score for any candidate whose symbol is in `cuts`.

    Returns the number of candidates boosted. Feature-flagged via `enabled`
    so the wiring is always present but a no-op when policy is OFF.

    Tags the candidate's metadata dict with::

        {"priority": "overnight_recovery", "prior_cut_price": ..., "prior_avg_entry": ...}

    so downstream consumers (DecisionClaw, position-thesis cache) can see
    why the rank changed.
    """
    if not enabled or not cuts:
        return 0
    cut_map = {c.symbol: c for c in cuts}
    boosted = 0
    for cand in candidates:
        symbol = str(getattr(cand, symbol_attr, "") or "").upper()
        if symbol not in cut_map:
            continue
        c = cut_map[symbol]
        try:
            current = float(getattr(cand, score_attr, 0.0) or 0.0)
            setattr(cand, score_attr, current + float(boost))
            meta = getattr(cand, metadata_attr, None)
            if isinstance(meta, dict):
                meta["priority"] = RECOVERY_PRIORITY_TAG
                meta["prior_cut_price"] = c.exit_price
                meta["prior_avg_entry"] = c.prior_avg_entry
                meta["prior_cut_reason"] = c.exit_reason
                meta["prior_cut_at"] = c.cut_at
            boosted += 1
        except Exception as e:
            logger.debug("overnight_cuts: boost failed for %s: %s", symbol, e)
    if boosted:
        logger.info(
            "[OVERNIGHT-RECOVERY] Boosted %d candidate(s) from prior-day cuts (boost=%.1f)",
            boosted,
            boost,
        )
    return boosted


def log_round_trip_completed(symbol: str, cut: CutRecord, re_entry_price: float) -> None:
    """Called by execution path when a cut-and-re-enter cycle completes.

    Emits a structured INFO line so realized round-trip alpha can be
    aggregated post-hoc.
    """
    realized_delta = (cut.exit_price - re_entry_price)
    logger.info(
        "[OVERNIGHT-RECOVERY-CYCLE] symbol=%s cut_price=%.4f reentry_price=%.4f "
        "delta_per_share=%.4f reason=%s",
        symbol,
        cut.exit_price,
        re_entry_price,
        realized_delta,
        cut.exit_reason,
    )
