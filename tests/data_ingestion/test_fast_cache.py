from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from autotrade.data_ingestion.fast_cache import FastMarketDataCache
from autotrade.data_ingestion.stream_bridge import QuoteStreamEvent, TradeStreamEvent


@pytest.mark.asyncio
async def test_fast_market_data_cache_returns_latest_quote_and_trade():
    cache = FastMarketDataCache()

    quote = QuoteStreamEvent(
        symbol="AAPL",
        bid_price=100.0,
        ask_price=100.1,
        bid_size=10,
        ask_size=12,
        timestamp=datetime(2026, 3, 7, 15, 0, tzinfo=timezone.utc),
    )
    trade = TradeStreamEvent(
        symbol="AAPL",
        price=100.05,
        size=200,
        exchange="V",
        conditions=("@",),
        timestamp=datetime(2026, 3, 7, 15, 0, 1, tzinfo=timezone.utc),
    )

    await cache.update_quote(quote)
    await cache.update_trade(trade)

    assert await cache.get_latest_quote("aapl") == quote
    assert await cache.get_latest_trade("AAPL") == trade


@pytest.mark.asyncio
async def test_fast_market_data_cache_handles_concurrent_updates_without_loss():
    cache = FastMarketDataCache()

    async def _write_quote(index: int):
        await cache.update_quote(
            QuoteStreamEvent(
                symbol="MSFT",
                bid_price=250.0 + index,
                ask_price=250.1 + index,
                bid_size=5 + index,
                ask_size=6 + index,
                timestamp=datetime(2026, 3, 7, 15, 0, index, tzinfo=timezone.utc),
            )
        )

    async def _write_trade(index: int):
        await cache.update_trade(
            TradeStreamEvent(
                symbol="MSFT",
                price=250.05 + index,
                size=100 + index,
                exchange="Q",
                conditions=("@",),
                timestamp=datetime(2026, 3, 7, 15, 1, index, tzinfo=timezone.utc),
            )
        )

    await asyncio.gather(*(_write_quote(i) for i in range(5)), *(_write_trade(i) for i in range(5)))

    snapshot = await cache.get_snapshot("MSFT")
    assert snapshot.quote is not None
    assert snapshot.trade is not None
    assert snapshot.quote.bid_price == pytest.approx(254.0)
    assert snapshot.trade.price == pytest.approx(254.05)
    assert snapshot.updated_at == datetime(2026, 3, 7, 15, 1, 4, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_fast_market_data_cache_snapshot_all_includes_all_symbols():
    cache = FastMarketDataCache()
    await cache.update_quote(
        QuoteStreamEvent(
            symbol="AAPL",
            bid_price=100.0,
            ask_price=100.2,
            bid_size=10,
            ask_size=10,
            timestamp=datetime(2026, 3, 7, 15, 2, tzinfo=timezone.utc),
        )
    )
    await cache.update_trade(
        TradeStreamEvent(
            symbol="MSFT",
            price=250.0,
            size=50,
            exchange="V",
            conditions=("@",),
            timestamp=datetime(2026, 3, 7, 15, 3, tzinfo=timezone.utc),
        )
    )

    snapshots = await cache.snapshot_all()

    assert sorted(snapshots) == ["AAPL", "MSFT"]
    assert snapshots["AAPL"].quote is not None
    assert snapshots["AAPL"].trade is None
    assert snapshots["MSFT"].trade is not None
