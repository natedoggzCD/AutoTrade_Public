from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, List, Optional

from autotrade.data_ingestion.stream_bridge import QuoteStreamEvent


@dataclass(frozen=True)
class ImbalanceEvent:
    symbol: str
    imbalance: float
    baseline_imbalance: float
    bid_size: float
    ask_size: float
    spread_bps: float
    event_type: str
    timestamp: Optional[datetime]


class MicrostructureEventDetector:
    """Detect short-horizon quote imbalance spikes from rolling quote windows."""

    def __init__(
        self,
        *,
        window_size: int = 5,
        imbalance_spike_threshold: float = 0.20,
        min_spread_bps: float = 0.0,
    ) -> None:
        self.window_size = max(2, int(window_size))
        self.imbalance_spike_threshold = float(imbalance_spike_threshold)
        self.min_spread_bps = float(min_spread_bps)
        self._quotes: Dict[str, Deque[QuoteStreamEvent]] = {}

    def ingest_quote(self, event: QuoteStreamEvent) -> List[ImbalanceEvent]:
        symbol = str(event.symbol).upper()
        window = self._quotes.setdefault(symbol, deque(maxlen=self.window_size))
        window.append(event)

        if len(window) < self.window_size:
            return []

        current_imbalance = self._imbalance(event.bid_size, event.ask_size)
        prior_imbalances = [
            self._imbalance(item.bid_size, item.ask_size)
            for item in list(window)[:-1]
        ]
        if not prior_imbalances:
            return []
        baseline = sum(prior_imbalances) / len(prior_imbalances)
        delta = current_imbalance - baseline
        spread_bps = self._spread_bps(event.bid_price, event.ask_price)

        if spread_bps < self.min_spread_bps:
            return []
        if delta >= self.imbalance_spike_threshold:
            return [
                ImbalanceEvent(
                    symbol=symbol,
                    imbalance=current_imbalance,
                    baseline_imbalance=baseline,
                    bid_size=float(event.bid_size),
                    ask_size=float(event.ask_size),
                    spread_bps=spread_bps,
                    event_type="bid_imbalance_spike",
                    timestamp=event.timestamp,
                )
            ]
        if delta <= -self.imbalance_spike_threshold:
            return [
                ImbalanceEvent(
                    symbol=symbol,
                    imbalance=current_imbalance,
                    baseline_imbalance=baseline,
                    bid_size=float(event.bid_size),
                    ask_size=float(event.ask_size),
                    spread_bps=spread_bps,
                    event_type="ask_imbalance_spike",
                    timestamp=event.timestamp,
                )
            ]
        return []

    @staticmethod
    def _imbalance(bid_size: float, ask_size: float) -> float:
        bid = max(float(bid_size or 0.0), 0.0)
        ask = max(float(ask_size or 0.0), 0.0)
        total = bid + ask
        if total <= 0:
            return 0.5
        return bid / total

    @staticmethod
    def _spread_bps(bid_price: float, ask_price: float) -> float:
        bid = max(float(bid_price or 0.0), 0.0)
        ask = max(float(ask_price or 0.0), 0.0)
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return 0.0
        return ((ask - bid) / mid) * 10000.0
