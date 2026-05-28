"""Helpers for retriggering overnight research when stale."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict


def attempt_retrigger_if_stale(
    freshness: Dict[str, Any],
    *,
    now_et: datetime,
    state: Dict[str, Any],
    retrigger_fn,
    cfg: Any = None,
    logger: Any = None,
) -> bool:
    """
    Attempt a one-time per-day retrigger when research is stale.
    Returns True if retrigger executed successfully, False otherwise.
    """
    if not freshness.get("age_stale", False):
        return False

    enabled = bool(getattr(cfg, "premarket_retrigger_if_stale", True))
    if not enabled:
        return False

    date_key = now_et.strftime("%Y-%m-%d")
    if state.get("stale_retrigger_attempted") == date_key:
        return False

    state["stale_retrigger_attempted"] = date_key
    try:
        ok = bool(retrigger_fn())
        if logger:
            logger.info(
                "[RESEARCH FRESHNESS] Stale retrigger %s",
                "succeeded" if ok else "failed",
            )
        return ok
    except Exception as exc:
        if logger:
            logger.warning("[RESEARCH FRESHNESS] Stale retrigger failed: %s", exc)
        return False
