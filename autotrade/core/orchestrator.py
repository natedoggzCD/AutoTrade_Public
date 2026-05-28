from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from autotrade.core.microstructure import ImbalanceEvent, MicrostructureEventDetector
from autotrade.core.threshold_alerts import ThresholdAlert, ThresholdAlertEngine
from autotrade.data_ingestion.fast_cache import FastMarketDataCache
from autotrade.data_ingestion.stream_bridge import (
    AlpacaStreamBridge,
    QuoteStreamEvent,
    TradeStreamEvent,
)

AsyncLoop = Callable[[], Awaitable[None]]
OrderCallback = Callable[[Dict[str, Any]], Awaitable[None]]


class AsyncTradingOrchestrator:
    """Run housekeeping and fast-track loops concurrently."""

    def __init__(
        self,
        housekeeping_loop: AsyncLoop,
        fast_track_loop: AsyncLoop,
    ) -> None:
        self.housekeeping_loop = housekeeping_loop
        self.fast_track_loop = fast_track_loop
        self._housekeeping_task: Optional[asyncio.Task] = None
        self._fast_track_task: Optional[asyncio.Task] = None

    async def run_once(self) -> None:
        await asyncio.gather(
            self.housekeeping_loop(),
            self.fast_track_loop(),
        )

    async def start(self) -> None:
        self._housekeeping_task = asyncio.create_task(self.housekeeping_loop())
        self._fast_track_task = asyncio.create_task(self.fast_track_loop())
        await asyncio.gather(self._housekeeping_task, self._fast_track_task)

    def running_tasks(self) -> int:
        return sum(
            1
            for task in (self._housekeeping_task, self._fast_track_task)
            if task is not None and not task.done()
        )


@dataclass
class FastLoopSignalContext:
    symbol: str
    vwap: Optional[float] = None
    support_levels: Sequence[float] = field(default_factory=tuple)
    resistance_levels: Sequence[float] = field(default_factory=tuple)
    last_price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FastLoopRuntime:
    """Wire stream, cache, triggers, and order callback into one fast-loop flow."""

    def __init__(
        self,
        *,
        cache_store: Optional[FastMarketDataCache] = None,
        threshold_engine: Optional[ThresholdAlertEngine] = None,
        microstructure_detector: Optional[MicrostructureEventDetector] = None,
        order_callback: Optional[OrderCallback] = None,
        stream_bridge: Optional[AlpacaStreamBridge] = None,
    ) -> None:
        self.cache_store = cache_store or FastMarketDataCache()
        self.threshold_engine = threshold_engine or ThresholdAlertEngine()
        self.microstructure_detector = microstructure_detector or MicrostructureEventDetector()
        self.order_callback = order_callback
        self.stream_bridge = stream_bridge
        self._signals: Dict[str, FastLoopSignalContext] = {}

    def register_signal(
        self,
        *,
        symbol: str,
        vwap: Optional[float] = None,
        support_levels: Optional[Sequence[float]] = None,
        resistance_levels: Optional[Sequence[float]] = None,
        last_price: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FastLoopSignalContext:
        context = FastLoopSignalContext(
            symbol=str(symbol).upper(),
            vwap=vwap,
            support_levels=tuple(support_levels or ()),
            resistance_levels=tuple(resistance_levels or ()),
            last_price=last_price,
            metadata=dict(metadata or {}),
        )
        self._signals[context.symbol] = context
        if self.stream_bridge is not None:
            self.stream_bridge.subscribe([context.symbol])
        return context

    async def process_quote(self, event: QuoteStreamEvent) -> List[ImbalanceEvent]:
        await self.cache_store.update_quote(event)
        return self.microstructure_detector.ingest_quote(event)

    async def process_trade(self, event: TradeStreamEvent) -> List[ThresholdAlert]:
        await self.cache_store.update_trade(event)
        symbol = str(event.symbol).upper()
        context = self._signals.get(symbol)
        if context is None or context.last_price is None:
            return []

        alerts = self.threshold_engine.evaluate_levels(
            symbol=symbol,
            previous_price=float(context.last_price),
            current_price=float(event.price),
            vwap=context.vwap,
            support_levels=context.support_levels,
            resistance_levels=context.resistance_levels,
            triggered_at=event.timestamp,
        )
        context.last_price = float(event.price)
        if alerts and self.order_callback is not None:
            await self.order_callback(
                {
                    "symbol": symbol,
                    "event_type": "threshold_alert",
                    "alerts": alerts,
                    "trade": event,
                    "signal_context": context,
                }
            )
        return alerts
