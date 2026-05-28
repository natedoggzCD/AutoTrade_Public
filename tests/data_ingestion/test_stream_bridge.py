from __future__ import annotations

from datetime import datetime, timezone

import pytest

from autotrade.data_ingestion.stream_bridge import (
    AlpacaStreamBridge,
    QuoteStreamEvent,
    TradeStreamEvent,
)
from autotrade.data_ingestion.fast_cache import FastMarketDataCache


class _FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.quote_handler = None
        self.trade_handler = None
        self.quote_symbols = ()
        self.trade_symbols = ()
        self.run_called = False
        self.stop_called = False

    def subscribe_quotes(self, handler, *symbols):
        self.quote_handler = handler
        self.quote_symbols = symbols

    def subscribe_trades(self, handler, *symbols):
        self.trade_handler = handler
        self.trade_symbols = symbols

    def run(self):
        self.run_called = True

    def stop(self):
        self.stop_called = True


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setattr(
        "autotrade.data_ingestion.stream_bridge.resolve_alpaca_credentials",
        lambda **kwargs: type(
            "Creds",
            (),
            {"api_key": "key", "secret_key": "secret", "paper": True},
        )(),
    )
    return AlpacaStreamBridge(symbols=["aapl"], stream_factory=_FakeStream)


def test_parse_quote_event_from_mocked_websocket_payload(bridge):
    event = bridge.parse_quote(
        {
            "symbol": "AAPL",
            "bid_price": 100.1,
            "ask_price": 100.2,
            "bid_size": 5,
            "ask_size": 7,
            "timestamp": "2026-03-07T14:31:00Z",
        }
    )

    assert isinstance(event, QuoteStreamEvent)
    assert event.symbol == "AAPL"
    assert event.bid_price == pytest.approx(100.1)
    assert event.ask_price == pytest.approx(100.2)
    assert event.timestamp == datetime(2026, 3, 7, 14, 31, tzinfo=timezone.utc)


def test_parse_trade_event_from_mocked_websocket_payload(bridge):
    payload = type(
        "TradePayload",
        (),
        {
            "symbol": "MSFT",
            "price": 301.25,
            "size": 100,
            "exchange": "V",
            "conditions": ["@", "I"],
            "timestamp": datetime(2026, 3, 7, 14, 32, tzinfo=timezone.utc),
        },
    )()

    event = bridge.parse_trade(payload)

    assert isinstance(event, TradeStreamEvent)
    assert event.symbol == "MSFT"
    assert event.price == pytest.approx(301.25)
    assert event.size == pytest.approx(100.0)
    assert event.exchange == "V"
    assert event.conditions == ("@", "I")


@pytest.mark.asyncio
async def test_start_subscribes_quotes_and_trades_for_symbols(bridge):
    await bridge.start()

    stream = bridge._stream
    assert stream is not None
    assert stream.run_called is True
    assert stream.quote_symbols == ("AAPL",)
    assert stream.trade_symbols == ("AAPL",)


@pytest.mark.asyncio
async def test_handle_quote_and_trade_forward_internal_events(monkeypatch):
    monkeypatch.setattr(
        "autotrade.data_ingestion.stream_bridge.resolve_alpaca_credentials",
        lambda **kwargs: type(
            "Creds",
            (),
            {"api_key": "key", "secret_key": "secret", "paper": True},
        )(),
    )
    received = []

    async def _handler(event):
        received.append(event)

    bridge = AlpacaStreamBridge(
        symbols=["aapl", "msft"],
        stream_factory=_FakeStream,
        event_handler=_handler,
    )

    quote_event = await bridge.handle_quote({"symbol": "AAPL", "bid_price": 10.0, "ask_price": 10.1})
    trade_event = await bridge.handle_trade({"symbol": "MSFT", "price": 20.5, "size": 200})

    assert received == [quote_event, trade_event]
    assert isinstance(received[0], QuoteStreamEvent)
    assert isinstance(received[1], TradeStreamEvent)


@pytest.mark.asyncio
async def test_handle_quote_and_trade_update_fast_cache(monkeypatch):
    monkeypatch.setattr(
        "autotrade.data_ingestion.stream_bridge.resolve_alpaca_credentials",
        lambda **kwargs: type(
            "Creds",
            (),
            {"api_key": "key", "secret_key": "secret", "paper": True},
        )(),
    )
    cache = FastMarketDataCache()
    bridge = AlpacaStreamBridge(
        symbols=["aapl"],
        stream_factory=_FakeStream,
        cache_store=cache,
    )

    await bridge.handle_quote({"symbol": "AAPL", "bid_price": 10.0, "ask_price": 10.1})
    await bridge.handle_trade({"symbol": "AAPL", "price": 10.05, "size": 100})

    assert (await cache.get_latest_quote("AAPL")) is not None
    assert (await cache.get_latest_trade("AAPL")) is not None
