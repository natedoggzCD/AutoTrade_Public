"""Helpers for overnight research freshness policy checks."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo


ET = ZoneInfo("America/New_York")


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def build_policy(policy_source: Any = None) -> Dict[str, Any]:
    """Build normalized freshness policy from config object or dict."""
    if policy_source is None:
        data: Dict[str, Any] = {}
    elif hasattr(policy_source, "model_dump"):
        data = dict(policy_source.model_dump())
    elif isinstance(policy_source, dict):
        data = dict(policy_source)
    else:
        data = {}

    return {
        "enabled": bool(data.get("enabled", True)),
        "weekday_max_age_hours": _safe_float(data.get("weekday_max_age_hours", 18.0), 18.0),
        "monday_max_age_hours": _safe_float(data.get("monday_max_age_hours", 60.0), 60.0),
        "premarket_max_age_hours": _safe_float(
            data.get("premarket_max_age_hours", data.get("weekday_max_age_hours", 18.0)),
            _safe_float(data.get("weekday_max_age_hours", 18.0), 18.0),
        ),
        "premarket_previous_day_cutoff_hour_et": _safe_float(
            data.get("premarket_previous_day_cutoff_hour_et", 18.5), 18.5
        ),
        "premarket_previous_day_cutoff_override": bool(
            data.get("premarket_previous_day_cutoff_override", True)
        ),
        "strict_weekday_max_age_hours": _safe_float(
            data.get("strict_weekday_max_age_hours", 6.0), 6.0
        ),
        "strict_weekday_start_hour_et": _safe_float(
            data.get("strict_weekday_start_hour_et", 9.5), 9.5
        ),
        "warning_age_hours": _safe_float(data.get("warning_age_hours", 24.0), 24.0),
        "sunday_refresh_hour_et": _safe_int(data.get("sunday_refresh_hour_et", 18), 18),
        "sunday_refresh_min_age_hours": _safe_float(data.get("sunday_refresh_min_age_hours", 48.0), 48.0),
        "stale_penalty_age_hours": _safe_float(data.get("stale_penalty_age_hours", 24.0), 24.0),
        "stale_score_threshold_penalty": _safe_float(data.get("stale_score_threshold_penalty", 5.0), 5.0),
        "stale_position_size_multiplier": _safe_float(data.get("stale_position_size_multiplier", 0.85), 0.85),
        "premarket_block_if_stale": bool(data.get("premarket_block_if_stale", True)),
    }


def _parse_iso(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    if isinstance(raw, datetime):
        parsed = raw
    else:
        try:
            parsed = datetime.fromisoformat(str(raw))
        except Exception:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ET)
    return parsed.astimezone(ET)


def _policy_max_age_hours(now_et: datetime, policy: Dict[str, Any]) -> float:
    if now_et.weekday() == 0:  # Monday
        return _safe_float(policy.get("monday_max_age_hours", 60.0), 60.0)
    weekday_max = _safe_float(policy.get("weekday_max_age_hours", 18.0), 18.0)

    # On regular weekdays, enforce a stricter freshness cap unless we are
    # carrying across a weekend day.
    prev_date = (now_et - timedelta(days=1)).date()
    if _is_weekend_day(prev_date):
        return weekday_max

    strict_start_hour = _safe_float(
        policy.get("strict_weekday_start_hour_et", 9.5), 9.5
    )
    time_decimal = now_et.hour + (now_et.minute / 60.0)
    if time_decimal < strict_start_hour:
        premarket_cap = _safe_float(
            policy.get("premarket_max_age_hours", weekday_max), weekday_max
        )
        return min(weekday_max, premarket_cap)

    strict_weekday_cap = _safe_float(
        policy.get("strict_weekday_max_age_hours", 6.0), 6.0
    )
    return min(weekday_max, strict_weekday_cap)


def _is_weekend_day(check_date) -> bool:
    return getattr(check_date, "weekday", lambda: 0)() >= 5


def _has_recent_morning_game_plan(state_path: Path, now_et: datetime) -> bool:
    """Best-effort check for a nearby morning game plan artifact (legacy compatibility)."""
    try:
        project_dir = state_path.resolve().parents[1]
    except Exception:
        return False
    plans_dir = project_dir / "plans"
    if not plans_dir.exists():
        return False
    for delta_days in (0, 1, -1):
        d = (now_et + timedelta(days=delta_days)).strftime("%Y%m%d")
        plan_path = plans_dir / f"morning_game_plan_{d}.json"
        if plan_path.exists():
            return True
    return False


def _has_previous_day_evening_artifact(
    state_path: Path,
    now_et: datetime,
    policy: Dict[str, Any],
) -> bool:
    """
    Premarket override: if any key artifact exists from previous day after cutoff
    (default 18:30 ET), do not treat research age as stale.
    """
    if not bool(policy.get("premarket_previous_day_cutoff_override", True)):
        return False

    strict_start_hour = _safe_float(policy.get("strict_weekday_start_hour_et", 9.5), 9.5)
    now_decimal = now_et.hour + (now_et.minute / 60.0)
    if now_decimal >= strict_start_hour:
        return False

    prev_day = (now_et - timedelta(days=1)).date()
    cutoff_hour = _safe_float(
        policy.get("premarket_previous_day_cutoff_hour_et", 18.5), 18.5
    )
    cutoff_dt = datetime.combine(prev_day, datetime.min.time(), tzinfo=ET) + timedelta(
        hours=cutoff_hour
    )

    try:
        project_dir = state_path.resolve().parents[1]
    except Exception:
        project_dir = state_path.parent.parent
    plans_dir = project_dir / "plans"
    research_dir = project_dir / "research"

    candidates = [
        state_path,
        plans_dir / f"morning_game_plan_{now_et.strftime('%Y%m%d')}.json",
        plans_dir / f"pm_plan_{prev_day.strftime('%Y-%m-%d')}.json",
    ]
    if research_dir.exists():
        candidates.extend(sorted(research_dir.glob("overnight_report_*.json"))[-10:])

    for candidate in candidates:
        try:
            if not candidate.exists():
                continue
            mtime = datetime.fromtimestamp(candidate.stat().st_mtime, tz=ET)
            if cutoff_dt <= mtime <= now_et:
                return True
        except Exception:
            continue
    return False


def _is_updated_after_previous_day_cutoff(
    updated_dt: datetime,
    now_et: datetime,
    policy: Dict[str, Any],
) -> bool:
    if not bool(policy.get("premarket_previous_day_cutoff_override", True)):
        return False
    strict_start_hour = _safe_float(policy.get("strict_weekday_start_hour_et", 9.5), 9.5)
    now_decimal = now_et.hour + (now_et.minute / 60.0)
    if now_decimal >= strict_start_hour:
        return False

    prev_day = (now_et - timedelta(days=1)).date()
    cutoff_hour = _safe_float(
        policy.get("premarket_previous_day_cutoff_hour_et", 18.5), 18.5
    )
    cutoff_dt = datetime.combine(prev_day, datetime.min.time(), tzinfo=ET) + timedelta(
        hours=cutoff_hour
    )
    return bool(cutoff_dt <= updated_dt <= now_et)


def _workflow_completion_status(state: Dict[str, Any], state_path: Path, now_et: datetime) -> Dict[str, Any]:
    """
    Determine whether overnight workflow is truly complete.

    Completion requires:
    - `research_complete` true
    - non-empty watchlist
    - generated morning game plan marker OR nearby game plan artifact
    """
    if not bool(state.get("research_complete", False)):
        return {"complete": False, "reason": "research_complete_false"}

    watchlist = state.get("watchlist")
    if not isinstance(watchlist, list) or len(watchlist) == 0:
        return {"complete": False, "reason": "watchlist_missing"}

    completion = state.get("workflow_completion")
    if not isinstance(completion, dict):
        completion = {}

    artifact_audit_status = str(
        state.get("artifact_audit_status")
        or completion.get("artifact_audit_status")
        or ""
    ).strip()
    if artifact_audit_status:
        if artifact_audit_status == "ok":
            artifact_gate = {"complete": True, "reason": "artifact_audit_ok"}
        else:
            artifact_gate = {
                "complete": False,
                "reason": f"artifact_audit_{artifact_audit_status}",
            }
    else:
        artifact_gate = {"complete": True, "reason": "artifact_audit_unset"}

    # If explicit stage keys exist, respect them strictly.
    for stage_key in ("watchlist_selected", "game_plan_generated"):
        if stage_key in completion and not bool(completion.get(stage_key)):
            return {"complete": False, "reason": f"{stage_key}_false"}
    # YouTube readiness is advisory for freshness scoring, not a hard completion gate.

    # New flow: explicit completion marker.
    if bool(completion.get("game_plan_generated")):
        if not bool(artifact_gate.get("complete")):
            return artifact_gate
        return {"complete": True, "reason": "workflow_markers"}

    # Legacy flow: infer from nearby plan artifact.
    if _has_recent_morning_game_plan(state_path=state_path, now_et=now_et):
        if not bool(artifact_gate.get("complete")):
            return artifact_gate
        return {"complete": True, "reason": "legacy_plan_artifact"}

    return {"complete": False, "reason": "game_plan_missing"}


def _resolve_target_trade_date(state: Dict[str, Any], now_et: datetime) -> Optional[str]:
    """Infer the trading session date this overnight artifact is intended to support."""
    completion = state.get("workflow_completion")
    if isinstance(completion, dict):
        target = str(completion.get("target_trade_date") or "").strip()
        if target:
            return target

    for key in ("target_trade_date", "trade_date", "date"):
        target = str(state.get(key) or "").strip()
        if target:
            return target

    return None


def _current_trading_session_date(now_et: datetime) -> str:
    """
    Return the trading session date the runtime should currently be preparing for.

    Overnight work after the market close is building the next trading day's plan.
    Premarket and market hours consume the current trading day.
    """
    session_date = now_et.date()
    if now_et.hour >= 17:
        session_date = session_date + timedelta(days=1)
    while _is_weekend_day(session_date):
        session_date = session_date + timedelta(days=1)
    return session_date.isoformat()


def check_research_freshness(
    state_path: Path,
    policy_source: Any = None,
    now_et: Optional[datetime] = None,
    persist_metadata: bool = True,
    logger: Any = None,
) -> Dict[str, Any]:
    """
    Evaluate overnight research freshness and optionally persist metadata.

    Returns dict fields:
    - exists, is_fresh, is_stale, warning, age_hours, max_age_hours
    - stale_penalty_active, updated_at, state_path, policy, reason
    """
    now_et = (now_et or datetime.now(ET)).astimezone(ET)
    policy = build_policy(policy_source)
    result: Dict[str, Any] = {
        "exists": False,
        "is_fresh": False,
        "is_stale": True,
        "warning": True,
        "age_hours": None,
        "max_age_hours": _policy_max_age_hours(now_et, policy),
        "stale_penalty_active": True,
        "workflow_complete": False,
        "workflow_reason": "state_missing",
        "freshness_basis": "state_missing",
        "target_trade_date": None,
        "current_session_trade_date": _current_trading_session_date(now_et),
        "session_aligned": False,
        "premarket_evening_cutoff_override": False,
        "updated_at": None,
        "state_path": str(state_path),
        "policy": policy,
        "reason": "state_missing",
        "checked_at": now_et.isoformat(),
    }

    if not Path(state_path).exists():
        return result

    try:
        with open(state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle) or {}
    except Exception as exc:
        if logger:
            logger.warning(f"[RESEARCH FRESHNESS] Failed reading state: {exc}")
        result["reason"] = "state_read_error"
        return result

    updated_raw = state.get("updated_at") or state.get("updated")
    updated_dt = _parse_iso(updated_raw)
    if updated_dt is None:
        result["exists"] = True
        result["reason"] = "missing_updated_at"
        if persist_metadata:
            state["freshness_checked_at"] = now_et.isoformat()
            state["freshness_status"] = "unknown_stale"
            state["freshness_policy"] = {
                "weekday_max_age_hours": policy["weekday_max_age_hours"],
                "monday_max_age_hours": policy["monday_max_age_hours"],
                "premarket_max_age_hours": policy["premarket_max_age_hours"],
                "premarket_previous_day_cutoff_hour_et": policy["premarket_previous_day_cutoff_hour_et"],
                "premarket_previous_day_cutoff_override": policy["premarket_previous_day_cutoff_override"],
                "warning_age_hours": policy["warning_age_hours"],
            }
            try:
                with open(state_path, "w", encoding="utf-8") as handle:
                    json.dump(state, handle, indent=2, default=str)
            except Exception:
                pass
        return result

    age_hours = max(0.0, (now_et - updated_dt).total_seconds() / 3600.0)
    max_age_hours = _policy_max_age_hours(now_et, policy)
    warning_age = _safe_float(policy.get("warning_age_hours", 24.0), 24.0)
    penalty_age = _safe_float(policy.get("stale_penalty_age_hours", 24.0), 24.0)
    workflow = _workflow_completion_status(state=state, state_path=state_path, now_et=now_et)
    workflow_complete = bool(workflow.get("complete"))
    workflow_reason = str(workflow.get("reason") or "unknown")
    target_trade_date = _resolve_target_trade_date(state=state, now_et=now_et)
    current_session_trade_date = _current_trading_session_date(now_et)
    session_aligned = bool(
        workflow_complete
        and target_trade_date
        and target_trade_date == current_session_trade_date
    )
    evening_override = _is_updated_after_previous_day_cutoff(
        updated_dt=updated_dt,
        now_et=now_et,
        policy=policy,
    ) or _has_previous_day_evening_artifact(
        state_path=state_path,
        now_et=now_et,
        policy=policy,
    )

    age_within_policy = (age_hours <= max_age_hours) or bool(evening_override)
    age_stale = not age_within_policy  # True only when age exceeds policy max
    session_fresh = bool(session_aligned and workflow_complete)
    is_fresh = bool(workflow_complete and (age_within_policy or session_fresh))
    warning = (age_hours > warning_age and not session_fresh) or (not workflow_complete)
    penalty_active = ((age_hours > penalty_age) and not session_fresh) or (
        not workflow_complete
    )
    if session_fresh and not age_within_policy:
        status_reason = "fresh_same_session"
        freshness_basis = "session_alignment"
    elif not age_within_policy:
        status_reason = "stale"
        freshness_basis = "age"
    elif not workflow_complete:
        status_reason = f"incomplete_workflow:{workflow_reason}"
        freshness_basis = "workflow_incomplete"
    else:
        status_reason = "fresh"
        freshness_basis = "age_policy"

    result.update(
        {
            "exists": True,
            "is_fresh": bool(is_fresh),
            "is_stale": not bool(is_fresh),
            "age_stale": bool(age_stale),
            "age_within_policy": bool(age_within_policy),
            "warning": bool(warning),
            "age_hours": round(age_hours, 3),
            "max_age_hours": float(max_age_hours),
            "stale_penalty_active": bool(penalty_active),
            "workflow_complete": workflow_complete,
            "workflow_reason": workflow_reason,
            "freshness_basis": freshness_basis,
            "target_trade_date": target_trade_date,
            "current_session_trade_date": current_session_trade_date,
            "session_aligned": session_aligned,
            "premarket_evening_cutoff_override": bool(evening_override),
            "updated_at": updated_dt.isoformat(),
            "reason": status_reason,
        }
    )

    if persist_metadata:
        state["updated_at"] = updated_dt.isoformat()
        state["updated"] = updated_dt.isoformat()  # Legacy key compatibility.
        state["age_hours"] = round(age_hours, 3)
        state["freshness_checked_at"] = now_et.isoformat()
        state["freshness_status"] = "fresh" if is_fresh else "stale"
        state["freshness_basis"] = freshness_basis
        state["target_trade_date"] = target_trade_date
        state["current_session_trade_date"] = current_session_trade_date
        state["session_aligned"] = session_aligned
        state["workflow_complete"] = workflow_complete
        state["workflow_reason"] = workflow_reason
        state["freshness_policy"] = {
            "weekday_max_age_hours": policy["weekday_max_age_hours"],
            "monday_max_age_hours": policy["monday_max_age_hours"],
            "premarket_max_age_hours": policy["premarket_max_age_hours"],
            "premarket_previous_day_cutoff_hour_et": policy["premarket_previous_day_cutoff_hour_et"],
            "premarket_previous_day_cutoff_override": policy["premarket_previous_day_cutoff_override"],
            "warning_age_hours": policy["warning_age_hours"],
            "active_max_age_hours": max_age_hours,
        }
        try:
            with open(state_path, "w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2, default=str)
        except Exception as exc:
            if logger:
                logger.debug(f"[RESEARCH FRESHNESS] Metadata persist skipped: {exc}")

    return result

