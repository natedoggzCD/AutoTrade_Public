"""
Live streaming bridge for the fast-loop track.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Iterable, Optional, Protocol, Sequence, Union

from alpaca.data.enums import DataFeed
from alpaca.data.live import StockDataStream

from autotrade.utils.alpaca_client_factory import resolve_alpaca_credentials

if TYPE_CHECKING:
    from autotrade.data_ingestion.fast_cache import FastMarketDataCache


StreamPayload = Union[Dict[str, Any], Any]
StreamEventHandler = Callable[[Union["QuoteStreamEvent", "TradeStreamEvent"]], Awaitable[None]]


class SupportsStockDataStream(Protocol):
    def subscribe_quotes(self, handler: Callable[[StreamPayload], Awaitable[None]], *symbols: str) -> None:
        ...

    def subscribe_trades(self, handler: Callable[[StreamPayload], Awaitable[None]], *symbols: str) -> None:
        ...

    def run(self) -> None:
        ...

    def stop(self) -> None:
        ...


@dataclass(frozen=True)
class QuoteStreamEvent:
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    timestamp: Optional[datetime]
    source: str = "alpaca"
    event_type: str = "quote"


@dataclass(frozen=True)
class TradeStreamEvent:
    symbol: str
    price: float
    size: float
    exchange: Optional[str]
    conditions: tuple[str, ...]
    timestamp: Optional[datetime]
    source: str = "alpaca"
    event_type: str = "trade"


class AlpacaStreamBridge:
    """Bridge Alpaca websocket events into internal fast-loop event types."""

    def __init__(
        self,
        symbols: Optional[Sequence[str]] = None,
        *,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: Optional[bool] = None,
        feed: DataFeed = DataFeed.IEX,
        raw_data: bool = False,
        websocket_params: Optional[Dict[str, Any]] = None,
        event_handler: Optional[StreamEventHandler] = None,
        cache_store: Optional["FastMarketDataCache"] = None,
        stream_factory: Optional[Callable[..., SupportsStockDataStream]] = None,
    ) -> None:
        self._symbols: list[str] = self._normalize_symbols(symbols or [])
        self._event_handler = event_handler
        self._cache_store = cache_store
        self._feed = feed
        self._raw_data = bool(raw_data)
        self._websocket_params = dict(websocket_params or {})
        self._stream_factory = stream_factory or StockDataStream
        self._stream: Optional[SupportsStockDataStream] = None

        creds = resolve_alpaca_credentials(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
            require=True,
        )
        self._api_key = creds.api_key
        self._secret_key = creds.secret_key

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self._symbols)

    def subscribe(self, symbols: Sequence[str]) -> tuple[str, ...]:
        merged = self._normalize_symbols([*self._symbols, *symbols])
        self._symbols = merged
        if self._stream is not None and self._symbols:
            self._stream.subscribe_quotes(self.handle_quote, *self._symbols)
            self._stream.subscribe_trades(self.handle_trade, *self._symbols)
        return tuple(self._symbols)

    def build_stream(self) -> SupportsStockDataStream:
        return self._stream_factory(
            api_key=self._api_key,
            secret_key=self._secret_key,
            raw_data=self._raw_data,
            feed=self._feed,
            websocket_params=self._websocket_params,
        )

    async def start(self) -> None:
        self._stream = self.build_stream()
        if self._symbols:
            self._stream.subscribe_quotes(self.handle_quote, *self._symbols)
            self._stream.subscribe_trades(self.handle_trade, *self._symbols)
        await asyncio.to_thread(self._stream.run)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()

    async def handle_quote(self, payload: StreamPayload) -> QuoteStreamEvent:
        event = self.parse_quote(payload)
        if self._cache_store is not None:
            await self._cache_store.update_quote(event)
        if self._event_handler is not None:
            await self._event_handler(event)
        return event

    async def handle_trade(self, payload: StreamPayload) -> TradeStreamEvent:
        event = self.parse_trade(payload)
        if self._cache_store is not None:
            await self._cache_store.update_trade(event)
        if self._event_handler is not None:
            await self._event_handler(event)
        return event

    def parse_quote(self, payload: StreamPayload) -> QuoteStreamEvent:
        symbol = str(self._value(payload, "symbol", "S") or "").upper()
        return QuoteStreamEvent(
            symbol=symbol,
            bid_price=self._float_value(payload, "bid_price", "bp"),
            ask_price=self._float_value(payload, "ask_price", "ap"),
            bid_size=self._float_value(payload, "bid_size", "bs"),
            ask_size=self._float_value(payload, "ask_size", "as"),
            timestamp=self._timestamp_value(payload, "timestamp", "t"),
        )

    def parse_trade(self, payload: StreamPayload) -> TradeStreamEvent:
        conditions = self._value(payload, "conditions", "c") or ()
        if not isinstance(conditions, (list, tuple)):
            conditions = (str(conditions),)
        return TradeStreamEvent(
            symbol=str(self._value(payload, "symbol", "S") or "").upper(),
            price=self._float_value(payload, "price", "p"),
            size=self._float_value(payload, "size", "s"),
            exchange=self._string_value(payload, "exchange", "x"),
            conditions=tuple(str(item) for item in conditions),
            timestamp=self._timestamp_value(payload, "timestamp", "t"),
        )

    @staticmethod
    def _normalize_symbols(symbols: Iterable[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for symbol in symbols:
            value = str(symbol or "").strip().upper()
            if not value or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized

    @staticmethod
    def _value(payload: StreamPayload, *keys: str) -> Any:
        if isinstance(payload, dict):
            for key in keys:
                if key in payload:
                    return payload[key]
            return None
        for key in keys:
            if hasattr(payload, key):
                return getattr(payload, key)
        return None

    @classmethod
    def _float_value(cls, payload: StreamPayload, *keys: str) -> float:
        value = cls._value(payload, *keys)
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _string_value(cls, payload: StreamPayload, *keys: str) -> Optional[str]:
        value = cls._value(payload, *keys)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @classmethod
    def _timestamp_value(cls, payload: StreamPayload, *keys: str) -> Optional[datetime]:
        value = cls._value(payload, *keys)
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            text = str(value).replace("Z", "+00:00")
            return datetime.fromisoformat(text)
        except ValueError:
            return None
