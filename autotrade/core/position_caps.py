"""Shared position-cap resolution and slot-class helpers.

Canonical execution-cap source:
- `resolved_regime.max_positions` is the only execution cap source for live
  day-manager and autonomous-agent flows.
- `strategy_overrides.max_positions` is a hint for upstream scanning only and
  must not be used as an execution cap at market hours.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping, Optional


WEAK_DAY_REGIMES = {
    "BEARISH",
    "DISTRIBUTION",
    "RISK_OFF",
    "SELL_OFF",
    "SELLOFF",
}

HARD_BEARISH_REGIMES = {
    "CAPITULATION",
    "CRASH",
    "CRISIS",
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _as_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "dict"):
        try:
            mapped = value.dict()
            if isinstance(mapped, dict):
                return dict(mapped)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return dict(vars(value))
        except Exception:
            return {}
    return {}


def _stringify_regime_label(regime: Any) -> str:
    if isinstance(regime, str):
        return regime.strip().upper()
    if hasattr(regime, "value"):
        return str(getattr(regime, "value") or "").strip().upper()
    if isinstance(regime, Mapping):
        for key in ("regime", "label", "name", "market_regime"):
            raw = regime.get(key)
            if raw:
                return str(raw).strip().upper()
    return str(regime or "").strip().upper()


def _resolve_regime_dict(regime: Any) -> Dict[str, Any]:
    if isinstance(regime, Mapping):
        return dict(regime)
    return _as_mapping(regime)


def _bool_from_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _portfolio_value(config: Any, key: str, default: Any = None) -> Any:
    portfolio = _config_value(config, "portfolio", None)
    return _config_value(portfolio, key, default)


def _snapshot_value(snapshot: Any, key: str, default: Any = None) -> Any:
    if snapshot is None:
        return default
    if isinstance(snapshot, Mapping):
        return snapshot.get(key, default)
    return getattr(snapshot, key, default)


def _normalize_slot_class(value: Any) -> str:
    slot_class = str(value or "").strip().lower()
    if slot_class in {
        "inverse",
        "inverse_etf",
        "inverse_etf_reserve",
        "hedge",
    }:
        return "inverse_etf_reserve"
    if slot_class in {
        "momentum",
        "momentum_reserve",
        "momentum_scanner",
        "runner",
        "runner_reserve",
        "late_scan",
        "late_scan_reserve",
    }:
        return "momentum_reserve"
    if slot_class == "core":
        return "core"
    return ""


def classify_slot_class(
    *,
    symbol: Any = None,
    candidate: Any = None,
    position: Any = None,
) -> str:
    """Infer the slot class for a candidate or held position."""

    payload: Dict[str, Any] = {}
    if candidate is not None:
        payload.update(_as_mapping(candidate))
    if position is not None:
        payload.update(_as_mapping(position))

    raw_slot_class = payload.get("slot_class") or payload.get("reserved_slot_class")
    normalized = _normalize_slot_class(raw_slot_class)
    if normalized:
        return normalized

    entry_source = str(
        payload.get("entry_source")
        or payload.get("source")
        or payload.get("source_bucket")
        or ""
    ).strip().lower()
    if (
        entry_source == "intraday_momentum"
        or entry_source.startswith("intraday_reserve")
        or entry_source.startswith("momentum_scanner")
        or entry_source in {"power_hour_volume", "watchlist_momentum"}
    ):
        return "momentum_reserve"

    if _bool_from_value(payload.get("intraday_reserve"), False):
        return "momentum_reserve"

    ticker = str(
        symbol
        or payload.get("symbol")
        or payload.get("ticker")
        or getattr(position, "symbol", "")
        or ""
    ).strip().upper()
    if ticker and (
        ticker.startswith("SDS")
        or ticker.startswith("SPXU")
        or ticker.startswith("SH")
        or ticker.startswith("QQQ")
        or ticker.startswith("TZA")
        or ticker.startswith("SPXS")
        or ticker.startswith("UVXY")
        or ticker.startswith("SQQQ")
        or ticker.startswith("TQQQ")
        or ticker.startswith("UPRO")
        or ticker.startswith("SOXL")
        or ticker.startswith("SOXS")
    ):
        # Keep the inverse ETF classification conservative and symbol-driven.
        from autotrade.risk.inverse_etf_manager import is_inverse_etf

        if is_inverse_etf(ticker):
            return "inverse_etf_reserve"

    if ticker and "inverse" in entry_source:
        return "inverse_etf_reserve"

    return "core"


@dataclass(frozen=True)
class PositionCapResolution:
    total_cap: int
    core_cap: int
    reserve_cap: int
    weak_day: bool
    config_total_cap: int
    config_core_cap: int
    config_reserve_cap: int
    failsafe_cap: int
    regime_cap: int
    alpha_router_cap: int
    regime_label: str = "UNKNOWN"
    source: str = "resolver"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_entry_constraints(self) -> Dict[str, Any]:
        payload = self.to_dict()
        return {
            "max_positions": int(self.total_cap),
            "core_max_positions": int(self.core_cap),
            "reserve_max_positions": int(self.reserve_cap),
            "weak_day": bool(self.weak_day),
            "source": str(self.source or "resolver"),
            "regime": str(self.regime_label or "UNKNOWN"),
            "caps": payload,
        }


def resolve_position_caps(
    regime: Any = None,
    failsafe: Any = None,
    config: Any = None,
    *,
    alpha_router: Any = None,
) -> PositionCapResolution:
    """Resolve the authoritative position-cap contract."""

    regime_dict = _resolve_regime_dict(regime)
    regime_label = _stringify_regime_label(
        regime_dict.get("regime")
        or regime_dict.get("label")
        or regime_dict.get("name")
        or regime
    )

    config_total = _safe_int(_portfolio_value(config, "max_positions", 50), 50)
    config_core = _safe_int(_portfolio_value(config, "core_max_positions", 40), 40)
    config_reserve = _safe_int(
        _portfolio_value(
            config,
            "late_scan_reserved_positions",
            _portfolio_value(config, "strategy_reserve_slots", 10),
        ),
        10,
    )
    if config_reserve <= 0:
        config_reserve = 10

    failsafe_cap = _safe_int(_snapshot_value(failsafe, "max_positions", config_core), config_core)
    if failsafe_cap <= 0:
        failsafe_cap = config_core

    regime_cap = _safe_int(regime_dict.get("max_positions"), 0)
    alpha_router_cap = _safe_int(
        _snapshot_value(alpha_router, "max_positions", 0),
        0,
    )

    core_floor = min(config_core, failsafe_cap)
    router_label = _stringify_regime_label(_snapshot_value(alpha_router, "regime", ""))
    weak_day = any(
        token in WEAK_DAY_REGIMES for token in (regime_label, router_label) if token
    ) or not _bool_from_value(regime_dict.get("allow_new_longs"), True)
    hard_bearish = (
        regime_label in HARD_BEARISH_REGIMES
        or router_label in HARD_BEARISH_REGIMES
        or not _bool_from_value(regime_dict.get("allow_new_longs"), True)
    )

    if hard_bearish:
        weak_candidates = [core_floor]
        if regime_cap > 0:
            weak_candidates.append(regime_cap)
        if alpha_router_cap > 0:
            weak_candidates.append(alpha_router_cap)
        core_cap = max(0, min(weak_candidates))
    elif weak_day:
        core_cap = max(0, min(core_floor, 25))
    else:
        core_cap = max(0, core_floor)

    reserve_cap = max(0, config_reserve)
    total_cap = max(0, min(config_total, core_cap + reserve_cap))
    if total_cap < core_cap:
        total_cap = core_cap

    source = str(
        regime_dict.get("plan_source")
        or regime_dict.get("source")
        or regime_dict.get("source_label")
        or "resolver"
    )

    return PositionCapResolution(
        total_cap=total_cap,
        core_cap=core_cap,
        reserve_cap=reserve_cap,
        weak_day=weak_day,
        config_total_cap=config_total,
        config_core_cap=config_core,
        config_reserve_cap=config_reserve,
        failsafe_cap=failsafe_cap,
        regime_cap=regime_cap,
        alpha_router_cap=alpha_router_cap,
        regime_label=regime_label or "UNKNOWN",
        source=source,
    )


def resolve_entry_constraints(
    plan: Any = None,
    *,
    failsafe: Any = None,
    config: Any = None,
    alpha_router: Any = None,
) -> Dict[str, Any]:
    """Resolve the canonical entry-constraint contract for a plan payload."""

    plan_payload = _resolve_regime_dict(plan) if isinstance(plan, Mapping) else {}
    resolved_regime = _resolve_regime_dict(plan_payload.get("resolved_regime"))
    caps = resolve_position_caps(
        regime=resolved_regime,
        failsafe=failsafe,
        config=config,
        alpha_router=alpha_router,
    )
    entry_constraints = caps.to_entry_constraints()
    existing = _as_mapping(plan_payload.get("entry_constraints"))

    source = str(
        existing.get("source")
        or plan_payload.get("generated_at")
        or plan_payload.get("plan_source")
        or caps.source
        or "plan"
    )
    entry_constraints["source"] = source
    if existing:
        for key, value in existing.items():
            if key in {
                "max_positions",
                "core_max_positions",
                "reserve_max_positions",
                "weak_day",
                "source",
                "regime",
                "caps",
            }:
                continue
            entry_constraints.setdefault(key, value)
    return entry_constraints


def build_cap_drift_warning(
    *,
    plan_total: Any,
    agent_total: Any,
    day_manager_total: Any,
    source: str,
    logger: Optional[Any] = None,
    exec_state: Optional[Dict[str, Any]] = None,
) -> str:
    plan_value = _safe_int(plan_total, 0)
    agent_value = _safe_int(agent_total, 0)
    dm_value = _safe_int(day_manager_total, 0)
    if plan_value == agent_value == dm_value:
        return ""
    drift = max(
        abs(plan_value - agent_value),
        abs(plan_value - dm_value),
        abs(agent_value - dm_value),
    )
    severity = "error" if drift > 10 else "warning"
    if exec_state is not None and drift > 10:
        exec_state["cap_drift_detected"] = True
        exec_state["cap_drift_detected_source"] = str(source or "plan")
        exec_state["cap_drift_detected_max_diff"] = int(drift)
    message = (
        "[CAP_DRIFT] "
        f"plan={plan_value} | agent={agent_value} | day_manager={dm_value} | "
        f"source={source}"
    )
    if logger is not None:
        log_fn = getattr(logger, severity, None)
        if callable(log_fn):
            log_fn(message)
        else:
            fallback = getattr(logger, "warning", None)
            if callable(fallback):
                fallback(message)
    return (
        message
    )
