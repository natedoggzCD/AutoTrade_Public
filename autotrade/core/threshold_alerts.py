from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class PriceThreshold:
    name: str
    value: float
    direction: str = "either"


@dataclass(frozen=True)
class ThresholdAlert:
    symbol: str
    threshold_name: str
    threshold_value: float
    direction: str
    previous_price: float
    current_price: float
    triggered_at: Optional[datetime]


class ThresholdAlertEngine:
    """Detect price crossings for VWAP and support/resistance thresholds."""

    def evaluate(
        self,
        *,
        symbol: str,
        previous_price: float,
        current_price: float,
        thresholds: Sequence[PriceThreshold],
        triggered_at: Optional[datetime] = None,
    ) -> List[ThresholdAlert]:
        alerts: List[ThresholdAlert] = []
        for threshold in thresholds:
            if self._crossed(
                previous_price=previous_price,
                current_price=current_price,
                threshold_value=threshold.value,
                direction=threshold.direction,
            ):
                alerts.append(
                    ThresholdAlert(
                        symbol=str(symbol).upper(),
                        threshold_name=threshold.name,
                        threshold_value=float(threshold.value),
                        direction=self._resolve_direction(
                            previous_price=previous_price,
                            current_price=current_price,
                            threshold_value=threshold.value,
                        ),
                        previous_price=float(previous_price),
                        current_price=float(current_price),
                        triggered_at=triggered_at,
                    )
                )
        return alerts

    def evaluate_levels(
        self,
        *,
        symbol: str,
        previous_price: float,
        current_price: float,
        vwap: Optional[float] = None,
        support_levels: Optional[Iterable[float]] = None,
        resistance_levels: Optional[Iterable[float]] = None,
        triggered_at: Optional[datetime] = None,
    ) -> List[ThresholdAlert]:
        thresholds: List[PriceThreshold] = []
        if vwap is not None and float(vwap) > 0:
            thresholds.append(PriceThreshold(name="vwap", value=float(vwap), direction="either"))
        for idx, level in enumerate(support_levels or (), start=1):
            if float(level) > 0:
                thresholds.append(PriceThreshold(name=f"support_{idx}", value=float(level), direction="down"))
        for idx, level in enumerate(resistance_levels or (), start=1):
            if float(level) > 0:
                thresholds.append(PriceThreshold(name=f"resistance_{idx}", value=float(level), direction="up"))
        return self.evaluate(
            symbol=symbol,
            previous_price=previous_price,
            current_price=current_price,
            thresholds=thresholds,
            triggered_at=triggered_at,
        )

    @staticmethod
    def _crossed(
        *,
        previous_price: float,
        current_price: float,
        threshold_value: float,
        direction: str,
    ) -> bool:
        previous = float(previous_price)
        current = float(current_price)
        threshold = float(threshold_value)
        if direction == "up":
            return previous < threshold <= current
        if direction == "down":
            return previous > threshold >= current
        return (previous < threshold <= current) or (previous > threshold >= current)

    @staticmethod
    def _resolve_direction(
        *,
        previous_price: float,
        current_price: float,
        threshold_value: float,
    ) -> str:
        if float(previous_price) < float(threshold_value) <= float(current_price):
            return "up"
        return "down"
