from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from autotrade.utils.market_data_cache import MarketDataCache


@dataclass(frozen=True)
class HedgingMonitorSnapshot:
    vix_level: float = 0.0
    breadth_ratio: float = 1.0
    new_highs: int = 0
    new_lows: int = 0
    market_regime: str = "neutral"
    risk_score: float = 0.0
    triggers: tuple[str, ...] = field(default_factory=tuple)
    should_hedge: bool = False


class HedgingTriggerMonitor:
    """Aggregate risk-off trigger inputs for hedge activation decisions."""

    def __init__(
        self,
        *,
        vix_spike_threshold: float = 25.0,
        breadth_ratio_floor: float = 0.85,
        new_low_excess_threshold: int = 25,
        risk_score_trigger: float = 2.0,
        market_data_cache: Optional[MarketDataCache] = None,
    ) -> None:
        self.vix_spike_threshold = float(vix_spike_threshold)
        self.breadth_ratio_floor = float(breadth_ratio_floor)
        self.new_low_excess_threshold = int(new_low_excess_threshold)
        self.risk_score_trigger = float(risk_score_trigger)
        self.market_data_cache = market_data_cache

    def evaluate(
        self,
        *,
        vix_level: Optional[float] = None,
        breadth_ratio: Optional[float] = None,
        new_highs: Optional[int] = None,
        new_lows: Optional[int] = None,
        market_regime: Optional[str] = None,
    ) -> HedgingMonitorSnapshot:
        triggers = []
        risk_score = 0.0

        vix = self._float(vix_level, 0.0)
        breadth = self._float(breadth_ratio, 1.0)
        highs = int(new_highs or 0)
        lows = int(new_lows or 0)
        regime = str(market_regime or "neutral").lower()

        if vix >= self.vix_spike_threshold:
            triggers.append("vix_spike")
            risk_score += 1.0

        if breadth <= self.breadth_ratio_floor:
            triggers.append("breadth_divergence")
            risk_score += 1.0

        if (lows - highs) >= self.new_low_excess_threshold:
            triggers.append("new_lows_expanding")
            risk_score += 1.0

        if regime in {"crisis", "risk_off", "selloff", "capitulation", "crash"}:
            triggers.append("bearish_regime")
            risk_score += 1.0

        return HedgingMonitorSnapshot(
            vix_level=vix,
            breadth_ratio=breadth,
            new_highs=highs,
            new_lows=lows,
            market_regime=regime,
            risk_score=risk_score,
            triggers=tuple(triggers),
            should_hedge=risk_score >= self.risk_score_trigger,
        )

    def check_vix_spike(self, vix_level: Optional[float] = None) -> bool:
        level = self._resolve_vix_level(vix_level)
        return level >= self.vix_spike_threshold

    def get_vix_snapshot(self, vix_level: Optional[float] = None) -> HedgingMonitorSnapshot:
        level = self._resolve_vix_level(vix_level)
        return HedgingMonitorSnapshot(
            vix_level=level,
            triggers=("vix_spike",) if level >= self.vix_spike_threshold else (),
            risk_score=1.0 if level >= self.vix_spike_threshold else 0.0,
            should_hedge=level >= self.vix_spike_threshold and self.risk_score_trigger <= 1.0,
        )

    def check_breadth_divergence(
        self,
        *,
        breadth_ratio: Optional[float] = None,
        new_highs: Optional[int] = None,
        new_lows: Optional[int] = None,
    ) -> bool:
        ratio = self._float(breadth_ratio, 1.0)
        highs = int(new_highs or 0)
        lows = int(new_lows or 0)
        return ratio <= self.breadth_ratio_floor or (lows - highs) >= self.new_low_excess_threshold

    def get_breadth_snapshot(
        self,
        *,
        breadth_ratio: Optional[float] = None,
        new_highs: Optional[int] = None,
        new_lows: Optional[int] = None,
    ) -> HedgingMonitorSnapshot:
        ratio = self._float(breadth_ratio, 1.0)
        highs = int(new_highs or 0)
        lows = int(new_lows or 0)
        triggers = []
        risk_score = 0.0

        if ratio <= self.breadth_ratio_floor:
            triggers.append("breadth_divergence")
            risk_score += 1.0

        if (lows - highs) >= self.new_low_excess_threshold:
            triggers.append("new_lows_expanding")
            risk_score += 1.0

        return HedgingMonitorSnapshot(
            breadth_ratio=ratio,
            new_highs=highs,
            new_lows=lows,
            triggers=tuple(triggers),
            risk_score=risk_score,
            should_hedge=risk_score >= self.risk_score_trigger,
        )

    def _resolve_vix_level(self, vix_level: Optional[float]) -> float:
        if vix_level is not None:
            return self._float(vix_level, 0.0)
        cache = self.market_data_cache
        if cache is None:
            cache = MarketDataCache()
            self.market_data_cache = cache
        try:
            payload = cache.get_vix_level()
            return self._float(payload.get("current"), 0.0)
        except Exception:
            return 0.0

    @staticmethod
    def _float(value: Any, default: float) -> float:
        try:
            return float(value if value is not None else default)
        except (TypeError, ValueError):
            return float(default)
