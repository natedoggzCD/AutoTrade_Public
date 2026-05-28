from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional


def _safe_float(value: Optional[float], default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _round_price(value: float, places: str = "0.01") -> float:
    return float(Decimal(str(value)).quantize(Decimal(places), rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class OrderPriceCalculator:
    atr_multiplier: float = 0.10
    min_offset_bps: float = 5.0
    max_offset_bps: float = 100.0
    min_price: float = 0.01

    def compute_offset(self, reference_price: float, atr_14: Optional[float]) -> float:
        reference = max(_safe_float(reference_price, 0.0), self.min_price)
        atr_value = max(_safe_float(atr_14, 0.0), 0.0)

        atr_offset = atr_value * max(self.atr_multiplier, 0.0)
        min_offset = reference * max(self.min_offset_bps, 0.0) / 10000.0
        max_offset = reference * max(self.max_offset_bps, 0.0) / 10000.0

        if max_offset > 0:
            return max(min_offset, min(atr_offset, max_offset))
        return max(min_offset, atr_offset)

    def compute_marketable_limit(
        self,
        side: str,
        reference_price: float,
        atr_14: Optional[float],
        bid_price: Optional[float] = None,
        ask_price: Optional[float] = None,
    ) -> float:
        direction = str(side or "").strip().lower()
        if direction not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")

        reference = max(_safe_float(reference_price, 0.0), self.min_price)
        bid = _safe_float(bid_price, 0.0)
        ask = _safe_float(ask_price, 0.0)

        if bid > 0 and ask > 0 and bid > ask:
            bid = 0.0
            ask = 0.0

        anchor = (
            ask
            if direction == "buy" and ask > 0
            else bid
            if direction == "sell" and bid > 0
            else reference
        )
        offset = self.compute_offset(reference_price=reference, atr_14=atr_14)
        signed_offset = offset if direction == "buy" else -offset

        return _round_price(max(anchor + signed_offset, self.min_price))
