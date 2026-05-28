from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional in some runtime contexts
    pd = None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


@dataclass(frozen=True)
class LiquidityDecision:
    tradable: bool
    reason: str
    spread_pct: float = 0.0
    avg_volume: float = 0.0
    session_volume: float = 0.0
    reference_price: float = 0.0


class LiquidityGate:
    """Simple tradability gate for watchlist pre-scans."""

    def __init__(
        self,
        min_avg_volume: int = 200_000,
        max_spread_pct: float = 0.5,
        min_price: float = 2.0,
        min_session_volume: int = 25_000,
    ) -> None:
        self.min_avg_volume = max(int(min_avg_volume), 0)
        self.max_spread_pct = max(float(max_spread_pct), 0.0)
        self.min_price = max(float(min_price), 0.0)
        self.min_session_volume = max(int(min_session_volume), 0)

    def evaluate(
        self,
        *,
        price: Optional[float],
        bid_price: Optional[float] = None,
        ask_price: Optional[float] = None,
        avg_volume: Optional[float] = None,
        session_volume: Optional[float] = None,
    ) -> LiquidityDecision:
        reference_price = max(_safe_float(price, 0.0), 0.0)
        avg_volume_value = max(_safe_float(avg_volume, 0.0), 0.0)
        session_volume_value = max(_safe_float(session_volume, 0.0), 0.0)
        bid = max(_safe_float(bid_price, 0.0), 0.0)
        ask = max(_safe_float(ask_price, 0.0), 0.0)

        if reference_price < self.min_price:
            return LiquidityDecision(
                tradable=False,
                reason="below_min_price",
                avg_volume=avg_volume_value,
                session_volume=session_volume_value,
                reference_price=reference_price,
            )
        if avg_volume_value < self.min_avg_volume:
            return LiquidityDecision(
                tradable=False,
                reason="below_min_avg_volume",
                avg_volume=avg_volume_value,
                session_volume=session_volume_value,
                reference_price=reference_price,
            )
        if session_volume_value < self.min_session_volume:
            return LiquidityDecision(
                tradable=False,
                reason="below_min_session_volume",
                avg_volume=avg_volume_value,
                session_volume=session_volume_value,
                reference_price=reference_price,
            )

        spread_pct = 0.0
        if bid > 0 and ask > 0 and ask >= bid:
            mid = (bid + ask) / 2.0
            if mid > 0:
                spread_pct = ((ask - bid) / mid) * 100.0
            if spread_pct > self.max_spread_pct:
                return LiquidityDecision(
                    tradable=False,
                    reason="spread_too_wide",
                    spread_pct=spread_pct,
                    avg_volume=avg_volume_value,
                    session_volume=session_volume_value,
                    reference_price=reference_price,
                )

        return LiquidityDecision(
            tradable=True,
            reason="tradable",
            spread_pct=spread_pct,
            avg_volume=avg_volume_value,
            session_volume=session_volume_value,
            reference_price=reference_price,
        )

    def filter_candidates(self, frame):
        if pd is None:
            raise RuntimeError("pandas is required for filter_candidates")
        if frame is None or frame.empty:
            return frame.copy() if frame is not None else frame

        accepted_rows = []
        for _, row in frame.iterrows():
            price = row.get("premarket_price", row.get("price", row.get("close", 0.0)))
            decision = self.evaluate(
                price=price,
                bid_price=row.get("bid_price", row.get("bid", 0.0)),
                ask_price=row.get("ask_price", row.get("ask", 0.0)),
                avg_volume=row.get("avg_volume", 0.0),
                session_volume=row.get(
                    "premarket_volume",
                    row.get("session_volume", row.get("volume", 0.0)),
                ),
            )
            if decision.tradable:
                accepted_rows.append(row.to_dict())
        return pd.DataFrame(accepted_rows)
