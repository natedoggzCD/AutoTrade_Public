"""Gross/net exposure helpers for mixed long/short books."""

from __future__ import annotations

from typing import Any, Dict, Iterable


def _position_notional(position: Any) -> float:
    for attr in ("market_value", "notional", "value"):
        value = getattr(position, attr, None)
        if value is None and isinstance(position, dict):
            value = position.get(attr)
        if value is not None:
            try:
                return abs(float(value))
            except (TypeError, ValueError):
                continue
    qty = getattr(position, "qty", None)
    price = getattr(position, "current_price", None)
    if isinstance(position, dict):
        qty = position.get("qty", qty)
        price = position.get("current_price", position.get("price", price))
    try:
        return abs(float(qty) * float(price))
    except (TypeError, ValueError):
        return 0.0


def _position_side(position: Any) -> str:
    side = getattr(position, "side", None)
    if isinstance(position, dict):
        side = position.get("side", side)
    side_key = str(side or "").lower().strip()
    if side_key in {"short", "sell_short"}:
        return "short"
    if side_key in {"long", "buy"}:
        return "long"
    qty = getattr(position, "qty", None)
    if isinstance(position, dict):
        qty = position.get("qty", qty)
    try:
        return "short" if float(qty) < 0 else "long"
    except (TypeError, ValueError):
        return "long"


def calculate_portfolio_exposure(
    positions: Iterable[Any],
    equity: float,
) -> Dict[str, float | int]:
    """Calculate gross and net exposure percentages."""
    equity_value = max(float(equity or 0.0), 0.0)
    long_notional = 0.0
    short_notional = 0.0
    long_count = 0
    short_count = 0
    for position in positions or []:
        notional = _position_notional(position)
        if _position_side(position) == "short":
            short_notional += notional
            short_count += 1
        else:
            long_notional += notional
            long_count += 1
    gross = long_notional + short_notional
    return {
        "gross_exposure_pct": (gross / equity_value * 100.0) if equity_value else 0.0,
        "net_exposure_pct": ((long_notional - short_notional) / equity_value * 100.0)
        if equity_value
        else 0.0,
        "long_notional": long_notional,
        "short_notional": short_notional,
        "long_count": long_count,
        "short_count": short_count,
    }
