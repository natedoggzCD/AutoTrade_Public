from __future__ import annotations

from datetime import datetime, timezone

import pytest

from autotrade.core.orchestrator import FastLoopRuntime
from autotrade.data_ingestion.stream_bridge import QuoteStreamEvent, TradeStreamEvent


@pytest.mark.asyncio
async def test_fast_loop_runtime_signal_stream_trigger_order_flow():
    emitted = []

    async def _order_callback(payload):
        emitted.append(payload)

    runtime = FastLoopRuntime(order_callback=_order_callback)
    runtime.register_signal(
        symbol="AAPL",
        vwap=100.0,
        support_levels=[98.0],
        resistance_levels=[101.0],
        last_price=99.5,
        metadata={"source": "signal"},
    )

    imbalance_events = await runtime.process_quote(
        QuoteStreamEvent(
            symbol="AAPL",
            bid_price=99.9,
            ask_price=100.0,
            bid_size=100,
            ask_size=100,
            timestamp=datetime(2026, 3, 7, 15, 20, 0, tzinfo=timezone.utc),
        )
    )
    assert imbalance_events == []

    alerts = await runtime.process_trade(
        TradeStreamEvent(
            symbol="AAPL",
            price=100.2,
            size=200,
            exchange="V",
            conditions=("@",),
            timestamp=datetime(2026, 3, 7, 15, 20, 1, tzinfo=timezone.utc),
        )
    )

    assert len(alerts) == 1
    assert alerts[0].threshold_name == "vwap"
    assert emitted
    assert emitted[0]["symbol"] == "AAPL"
    assert emitted[0]["alerts"][0].threshold_name == "vwap"


@pytest.mark.asyncio
async def test_fast_loop_runtime_register_signal_subscribes_stream_bridge():
    subscriptions = []

    class _Bridge:
        def subscribe(self, symbols):
            subscriptions.append(tuple(symbols))
            return tuple(symbols)

    runtime = FastLoopRuntime(stream_bridge=_Bridge())
    runtime.register_signal(symbol="msft", vwap=250.0, last_price=249.0)

    assert subscriptions == [("MSFT",)]
