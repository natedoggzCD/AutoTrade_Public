from __future__ import annotations

import math
from typing import Any, Dict, Optional


MIN_REASONABLE_ATR_MULT = 0.25
MAX_REASONABLE_ATR_MULT = 10.0


def sanitize_atr_multiplier(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    if numeric < MIN_REASONABLE_ATR_MULT or numeric > MAX_REASONABLE_ATR_MULT:
        return float(default)
    return numeric


def sanitize_strategy_params(
    strategy_params: Optional[Dict[str, Any]],
    *,
    fallback_stop: float = 2.0,
    fallback_target: float = 3.0,
) -> Dict[str, Any]:
    params = dict(strategy_params or {}) if isinstance(strategy_params, dict) else {}
    params["stop_atr_mult"] = sanitize_atr_multiplier(
        params.get("stop_atr_mult"), fallback_stop
    )
    params["target_atr_mult"] = sanitize_atr_multiplier(
        params.get("target_atr_mult"), fallback_target
    )
    return params
