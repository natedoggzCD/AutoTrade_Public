"""Position cache helpers for Alpaca connectivity fallbacks."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

from autotrade.utils.safe_logging import get_safe_logger


LOGGER = get_safe_logger("position_cache")
DEFAULT_CACHE_PATH = Path("data") / "positions_cache.json"
POSITION_TIMESTAMP_KEYS = (
    "entry_time",
    "entry_at",
    "created_at",
    "submitted_at",
    "filled_at",
)


def _serialize_timestamp(value: Any) -> Any:
    if isinstance(value, datetime):
        dt_value = value
        if dt_value.tzinfo is None:
            dt_value = dt_value.replace(tzinfo=timezone.utc)
        return dt_value.isoformat()
    return value


def _position_to_dict(position: Any) -> Dict[str, Any]:
    if isinstance(position, dict):
        payload = dict(position)
    else:
        payload = {
            "symbol": getattr(position, "symbol", None),
            "qty": getattr(position, "qty", None),
            "avg_entry_price": getattr(position, "avg_entry_price", None),
            "current_price": getattr(position, "current_price", None),
            "unrealized_pl": getattr(position, "unrealized_pl", None),
            "unrealized_plpc": getattr(position, "unrealized_plpc", None),
            "market_value": getattr(position, "market_value", None),
            "cost_basis": getattr(position, "cost_basis", None),
            "change_today": getattr(position, "change_today", None),
        }
        for key in POSITION_TIMESTAMP_KEYS:
            payload[key] = _serialize_timestamp(getattr(position, key, None))

    if payload.get("qty") is not None:
        payload["qty"] = float(payload["qty"])
    for key in (
        "avg_entry_price",
        "current_price",
        "unrealized_pl",
        "unrealized_plpc",
        "market_value",
        "cost_basis",
        "change_today",
    ):
        if payload.get(key) is not None:
            try:
                payload[key] = float(payload[key])
            except Exception:
                pass
    for key in POSITION_TIMESTAMP_KEYS:
        if key in payload:
            payload[key] = _serialize_timestamp(payload.get(key))
    return payload


def save_positions_cache(
    positions: Iterable[Any],
    *,
    cache_path: Path | str = DEFAULT_CACHE_PATH,
    source: str = "alpaca",
) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "source": source,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
        "positions": [_position_to_dict(pos) for pos in positions],
    }
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        LOGGER.warning("Failed to save positions cache: %s", exc)


def load_positions_cache(cache_path: Path | str = DEFAULT_CACHE_PATH) -> Dict[str, Any]:
    path = Path(cache_path)
    if not path.exists():
        return {"meta": {}, "positions": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Failed to load positions cache: %s", exc)
        return {"meta": {}, "positions": []}


def _dicts_to_namespace(items: Iterable[Dict[str, Any]]) -> List[SimpleNamespace]:
    return [SimpleNamespace(**item) for item in items]


def _is_dns_error(exc: Exception) -> bool:
    """Check if exception is related to DNS resolution failure."""
    import socket
    msg = str(exc).lower()
    if "getaddrinfo failed" in msg or "non-recoverable failure" in msg:
        return True
    if isinstance(exc, socket.gaierror):
        return True
    return False

def fetch_positions_with_fallback(
    client: Any,
    *,
    cache_path: Path | str = DEFAULT_CACHE_PATH,
    logger: Any = None,
    retries: int = 5,
    retry_delay_seconds: float = 2.0,
    backoff_multiplier: float = 2.0,
    max_retry_delay_seconds: float = 30.0,
    use_cache: bool = True,
) -> Dict[str, Any]:
    attempts = max(1, int(retries))
    last_error: Optional[Exception] = None
    delay = max(0.0, float(retry_delay_seconds))
    log = logger or LOGGER
    dns_error_count = 0

    for attempt in range(1, attempts + 1):
        try:
            positions = client.get_all_positions()
            if isinstance(positions, dict) and positions.get("error"):
                raise RuntimeError(positions.get("error"))
            if positions is None:
                positions = []
            save_positions_cache(positions, cache_path=cache_path, source="alpaca")
            return {
                "positions": list(positions),
                "used_cache": False,
                "error": None,
            }
        except Exception as exc:
            last_error = exc
            is_dns = _is_dns_error(exc)
            if is_dns:
                dns_error_count += 1
                try:
                    from autotrade.monitoring.alerts import get_alert_manager
                    get_alert_manager().emit_dns_failure(str(exc), dns_error_count)
                except Exception:
                    pass
            
            log.warning(
                "Positions fetch attempt %s/%s failed (DNS=%s): %s", 
                attempt, attempts, is_dns, exc
            )
            
            if attempt < attempts:
                time.sleep(delay)
                # Exponential backoff
                delay = min(delay * max(1.0, float(backoff_multiplier)), float(max_retry_delay_seconds))

    if use_cache:
        payload = load_positions_cache(cache_path)
        cached = payload.get("positions", [])
        if cached:
            log.warning("Using cached positions (%s) due to persistent failure", cache_path)
            return {
                "positions": _dicts_to_namespace(cached),
                "used_cache": True,
                "error": last_error,
                "cache_meta": payload.get("meta", {}),
            }

    return {
        "positions": [],
        "used_cache": False,
        "error": last_error,
    }
