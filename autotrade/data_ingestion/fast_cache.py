"""
Low-latency in-memory cache for fast-loop market data.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from autotrade.data_ingestion.stream_bridge import QuoteStreamEvent, TradeStreamEvent


@dataclass(frozen=True)
class FastSymbolSnapshot:
    symbol: str
    quote: Optional[QuoteStreamEvent]
    trade: Optional[TradeStreamEvent]
    updated_at: Optional[datetime]


class FastMarketDataCache:
    """Async-safe cache for latest quote and trade events per symbol."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._quotes: Dict[str, QuoteStreamEvent] = {}
        self._trades: Dict[str, TradeStreamEvent] = {}

    async def update_quote(self, event: QuoteStreamEvent) -> QuoteStreamEvent:
        async with self._lock:
            self._quotes[event.symbol] = event
            return self._quotes[event.symbol]

    async def update_trade(self, event: TradeStreamEvent) -> TradeStreamEvent:
        async with self._lock:
            self._trades[event.symbol] = event
            return self._trades[event.symbol]

    async def get_latest_quote(self, symbol: str) -> Optional[QuoteStreamEvent]:
        async with self._lock:
            return self._quotes.get(str(symbol).upper())

    async def get_latest_trade(self, symbol: str) -> Optional[TradeStreamEvent]:
        async with self._lock:
            return self._trades.get(str(symbol).upper())

    async def get_snapshot(self, symbol: str) -> FastSymbolSnapshot:
        async with self._lock:
            symbol_key = str(symbol).upper()
            quote = self._quotes.get(symbol_key)
            trade = self._trades.get(symbol_key)
            timestamps = [item.timestamp for item in (quote, trade) if item is not None and item.timestamp is not None]
            updated_at = max(timestamps) if timestamps else None
            return FastSymbolSnapshot(
                symbol=symbol_key,
                quote=quote,
                trade=trade,
                updated_at=updated_at,
            )

    async def snapshot_all(self) -> Dict[str, FastSymbolSnapshot]:
        async with self._lock:
            symbols = sorted(set(self._quotes) | set(self._trades))
            snapshots: Dict[str, FastSymbolSnapshot] = {}
            for symbol in symbols:
                quote = self._quotes.get(symbol)
                trade = self._trades.get(symbol)
                timestamps = [item.timestamp for item in (quote, trade) if item is not None and item.timestamp is not None]
                updated_at = max(timestamps) if timestamps else None
                snapshots[symbol] = FastSymbolSnapshot(
                    symbol=symbol,
                    quote=quote,
                    trade=trade,
                    updated_at=updated_at,
                )
            return snapshots
