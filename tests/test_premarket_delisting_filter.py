from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from autotrade.core.premarket_manager import (
    ET,
    PremarketManager,
)

class MockAsset:
    def __init__(self, symbol, status, tradable):
        self.symbol = symbol
        self.status = status
        self.tradable = tradable

class MockTradingClient:
    def __init__(self, active_symbols):
        self.active_symbols = active_symbols

    def get_all_assets(self, request):
        return [
            MockAsset(symbol, 'active', True) 
            for symbol in self.active_symbols
        ]

class MockAnalyzer:
    def __init__(self, active_symbols):
        self.active_symbols = active_symbols
        self.api = MockTradingClient(active_symbols)
    
    def get_active_symbols(self):
        # This mimics the implementation in PreMarketAnalyzer
        from alpaca.trading.enums import AssetStatus, AssetClass
        # In mock, we just return the predefined set
        return {s.upper() for s in self.active_symbols}

    def analyze_ticker(self, symbol: str):
        return SimpleNamespace(
            has_data=True,
            gap_pct=0.0,
            volume_ratio=1.0,
            premarket_volume=100_000,
            premarket_trend="flat",
            is_tradable=True,
            liquidity_block_reason="",
            prev_close=10.0,
            premarket_price=10.0,
            premarket_high=10.1,
            premarket_low=9.9,
            premarket_open=10.0,
            gap_direction="flat",
            avg_volume=1_000_000,
            premarket_range_pct=2.0,
            liquidity_score=70.0,
            spread_pct=0.1,
            last_update="2026-03-22T08:00:00Z"
        )

    def get_premarket_bars(self, symbol: str):
        return []

def test_delisted_ticker_filtering(tmp_path):
    active_symbols = ["AAPL", "MSFT", "TSLA"]
    analyzer = MockAnalyzer(active_symbols)
    
    manager = PremarketManager(
        output_dir=tmp_path,
        premarket_analyzer=analyzer,
        max_watchlist_symbols=10
    )
    
    # Watchlist with some delisted tickers
    watchlist = [
        {"symbol": "AAPL"},
        {"symbol": "MSFT"},
        {"symbol": "DELISTED1"},
        {"symbol": "TSLA"},
        {"symbol": "DELISTED2"}
    ]
    
    # Manually call the filter to verify it works
    filtered = manager._filter_delisted_tickers(watchlist)
    
    symbols = [item["symbol"] for item in filtered]
    assert "AAPL" in symbols
    assert "MSFT" in symbols
    assert "TSLA" in symbols
    assert "DELISTED1" not in symbols
    assert "DELISTED2" not in symbols
    assert len(filtered) == 3

def test_run_cycle_applies_filter(tmp_path):
    active_symbols = ["AAPL", "MSFT"]
    analyzer = MockAnalyzer(active_symbols)
    
    # We need to mock news and stocktwits too since they are used in run_cycle -> _evaluate_symbol
    class MockNews:
        def collect(self, symbol):
            return {"available": False, "sentiment_score": 0.0, "has_catalyst": False, "catalyst_score": 0.0, "confidence": 0.0, "coverage": "none"}
    
    class MockStocktwits:
        def fetch(self, symbol):
            return {"available": False, "sentiment_score": 0.0, "confidence": 0.0, "coverage": "none"}

    manager = PremarketManager(
        output_dir=tmp_path,
        premarket_analyzer=analyzer,
        news_aggregator=MockNews(),
        stocktwits_scraper=MockStocktwits(),
        max_watchlist_symbols=10
    )
    
    # Mock some methods to avoid external dependencies in run_cycle
    manager._load_market_intelligence = lambda: None
    manager._collect_market_context = lambda: {"futures_direction": "neutral"}
    
    watchlist = [{"symbol": "AAPL"}, {"symbol": "DELISTED"}]
    
    handoff = manager.run_cycle(
        watchlist=watchlist,
        holdings=[],
        now_et=datetime(2026, 3, 22, 8, 0, tzinfo=ET)
    )
    
    ranked_symbols = [item["symbol"] for item in handoff["ranked_watchlist"]]
    assert "AAPL" in ranked_symbols
    assert "DELISTED" not in ranked_symbols
