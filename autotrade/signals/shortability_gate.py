"""Alpaca-backed shortability checks with per-session JSON cache."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from autotrade.utils.safe_logging import get_safe_logger

logger = get_safe_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def _cache_path(
    session_date: Optional[str] = None, *, data_dir: Path = DATA_DIR
) -> Path:
    date_key = session_date or date.today().isoformat()
    return data_dir / f"shortability_cache_{date_key}.json"


def _load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        logger.warning("[SHORTABILITY] Ignoring unreadable cache %s: %s", path, exc)
        return {}


def _save_cache(path: Path, payload: Dict[str, Dict[str, Any]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:
        logger.warning("[SHORTABILITY] Failed to persist cache %s: %s", path, exc)


def _resolve_client(trading_client: Optional[Any]) -> Optional[Any]:
    if trading_client is not None:
        return trading_client
    try:
        from autotrade.utils.alpaca_client_factory import create_trading_client

        return create_trading_client()
    except Exception as exc:
        logger.warning("[SHORTABILITY] Trading client unavailable: %s", exc)
        return None


def is_shortable(
    symbol: str,
    trading_client: Optional[Any] = None,
    *,
    session_date: Optional[str] = None,
    refresh: bool = False,
    data_dir: Path = DATA_DIR,
) -> Tuple[bool, str]:
    """Return (allowed, reason) for a stock short entry."""
    ticker = str(symbol or "").upper().strip()
    if not ticker:
        return False, "missing_symbol"

    path = _cache_path(session_date, data_dir=data_dir)
    cache = _load_cache(path)
    if not refresh and ticker in cache:
        row = cache[ticker]
        return bool(row.get("allowed")), str(row.get("reason", "unknown"))

    client = _resolve_client(trading_client)
    if client is None:
        return False, "client_unavailable"

    try:
        asset = client.get_asset(ticker)
    except Exception as exc:
        logger.warning("[SHORTABILITY] Asset lookup failed for %s: %s", ticker, exc)
        return False, "asset_lookup_failed"

    if not bool(getattr(asset, "shortable", False)):
        allowed, reason = False, "not_shortable"
    elif not bool(getattr(asset, "easy_to_borrow", False)):
        allowed, reason = False, "hard_to_borrow_blocked"
    elif not bool(getattr(asset, "fractionable", False)):
        allowed, reason = False, "not_fractionable"
    else:
        allowed, reason = True, "ok"

    cache[ticker] = {
        "allowed": allowed,
        "reason": reason,
        "shortable": bool(getattr(asset, "shortable", False)),
        "easy_to_borrow": bool(getattr(asset, "easy_to_borrow", False)),
        "fractionable": bool(getattr(asset, "fractionable", False)),
    }
    _save_cache(path, cache)
    return allowed, reason


def refresh_shortability_cache(
    symbols: list[str],
    trading_client: Optional[Any] = None,
    *,
    session_date: Optional[str] = None,
    data_dir: Path = DATA_DIR,
) -> Dict[str, Tuple[bool, str]]:
    """Refresh a batch of symbols for premarket use."""
    client = _resolve_client(trading_client)
    results: Dict[str, Tuple[bool, str]] = {}
    for symbol in symbols:
        results[str(symbol).upper()] = is_shortable(
            symbol,
            trading_client=client,
            session_date=session_date,
            refresh=True,
            data_dir=data_dir,
        )
    return results
