from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from autotrade.utils.catalyst_bridge import CatalystTagBridge
from autotrade.utils.financial_db import FinancialDB
from autotrade.utils.news_aggregator import NewsAggregator


class _NewsAnalyzerStub:
    def analyze_ticker(self, symbol: str):
        return {
            "weighted_score": 0.4,
            "top_headlines": [
                {
                    "headline": f"{symbol} beats earnings and raises guidance",
                    "source": "stub",
                },
            ],
        }


class _SearxStub:
    def search_news(self, query: str, max_results: int = 5):
        class _Response:
            results = []

        return _Response()


def test_catalyst_bridge_extracts_headline_and_earnings_metadata(tmp_path: Path):
    upcoming_date = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
    db = FinancialDB(db_path=tmp_path / "financial.db")
    db.upsert_earnings(
        [
            {
                "ticker": "AAPL",
                "earnings_date": upcoming_date,
                "time_of_day": "AMC",
                "eps_estimate": 1.0,
                "eps_actual": None,
                "revenue_estimate": None,
                "revenue_actual": None,
                "surprise_pct": None,
                "updated_at": datetime.utcnow().isoformat(),
            }
        ]
    )
    bridge = CatalystTagBridge(financial_db=db)

    result = bridge.extract(
        "AAPL",
        [{"title": "AAPL beats earnings and raises guidance"}],
    )

    assert result["has_catalyst"] is True
    assert "earnings" in result["catalyst_tags"]
    assert "guidance" in result["catalyst_tags"]
    assert result["catalyst_score"] > 0
    assert result["earnings_context"]["upcoming"]


def test_news_aggregator_attaches_catalyst_fields(tmp_path: Path):
    upcoming_date = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
    db = FinancialDB(db_path=tmp_path / "financial.db")
    db.upsert_earnings(
        [
            {
                "ticker": "NVDA",
                "earnings_date": upcoming_date,
                "time_of_day": "BMO",
                "eps_estimate": 1.0,
                "eps_actual": None,
                "revenue_estimate": None,
                "revenue_actual": None,
                "surprise_pct": None,
                "updated_at": datetime.utcnow().isoformat(),
            }
        ]
    )
    aggregator = NewsAggregator(
        news_analyzer=_NewsAnalyzerStub(),
        searx_client=_SearxStub(),
        catalyst_bridge=CatalystTagBridge(financial_db=db),
    )

    snapshot = aggregator.collect("NVDA")

    assert snapshot["available"] is True
    assert snapshot["has_catalyst"] is True
    assert snapshot["catalyst_score"] > 0
    assert "earnings" in snapshot["catalyst_tags"]
    assert snapshot["catalyst_note"]


class _SearxResult:
    def __init__(
        self,
        *,
        title: str,
        content: str = "",
        url: str = "",
        engine: str = "bing news",
        publishedDate: str | None = "2026-03-15T08:30:00",
    ) -> None:
        self.title = title
        self.content = content
        self.url = url
        self.engine = engine
        self.publishedDate = publishedDate


class _RelevantSearxStub:
    def search_news(self, query: str, max_results: int = 5):
        class _Response:
            results = [
                _SearxResult(
                    title="BMNR wins hosting contract and raises FY2026 guidance",
                    content="BMNR announced a new contract and raised outlook in a press release.",
                    url="https://example.com/bmnr-contract",
                ),
                _SearxResult(
                    title="BitMine Immersion Technologies (BMNR) - Stocks - Robinhood",
                    content="Quote page for BMNR.",
                    url="https://robinhood.com/us/en/stocks/BMNR/",
                    engine="startpage",
                ),
                _SearxResult(
                    title="Why Fastly Stock Skyrocketed Today",
                    content="Fastly shares rose on AI demand.",
                    url="https://www.fool.com/investing/2026/02/12/why-fastly-stock-is-up-today/",
                ),
            ]

        return _Response()


def test_news_aggregator_filters_low_signal_searx_results(monkeypatch):
    cache = _CacheStub(None)
    cache.get_cached_news = lambda symbol, max_age_hours=None: None
    monkeypatch.setattr("autotrade.utils.news_cache.get_cache", lambda: cache)
    monkeypatch.setattr(NewsAggregator, "_build_news_analyzer", lambda self: None)
    aggregator = NewsAggregator(
        news_analyzer=None,
        searx_client=_RelevantSearxStub(),
        catalyst_bridge=None,
    )

    snapshot = aggregator.collect("BMNR")

    assert snapshot["available"] is True
    assert snapshot["headline_count"] == 1
    assert (
        snapshot["headlines"][0]["title"]
        == "BMNR wins hosting contract and raises FY2026 guidance"
    )
    assert snapshot["search_diagnostics"]["searx_kept"] == 1
    assert snapshot["source_status"]["searxng"] == "ok"


class _CacheStub:
    def __init__(self, rows):
        self.rows = rows
        self.cached_writes = []

    def get_cached_news(self, symbol: str, max_age_hours=None):
        return self.rows

    def cache_news(self, symbol: str, articles, source: str = "unknown"):
        self.cached_writes.append((symbol, articles, source))
        return len(articles)


def test_news_aggregator_cached_irrelevant_news_cannot_create_sentiment(monkeypatch):
    cache = _CacheStub(
        [
            {
                "title": "Fed hawks now rule. Will Warsh go along with them?",
                "source": "Barrons",
                "content_preview": "Federal Reserve policy article with no company facts.",
                "sentiment": "positive",
                "sentiment_score": 0.9,
            }
        ]
    )
    monkeypatch.setattr("autotrade.utils.news_cache.get_cache", lambda: cache)
    aggregator = NewsAggregator(
        news_analyzer=None, searx_client=_SearxStub(), catalyst_bridge=None
    )

    snapshot = aggregator.collect("PFE")

    assert snapshot["available"] is False
    assert snapshot["sentiment_score"] == 0.0
    assert snapshot["coverage"] == "none"


def test_news_aggregator_cached_sentiment_uses_only_relevant_rows(monkeypatch):
    cache = _CacheStub(
        [
            {
                "title": "Company News for May 21, 2026",
                "source": "Zacks",
                "content_preview": "AMC Entertainment Holdings, Inc. (AMC) shares jumped after CEO Adam Aron disclosed a stock purchase.",
                "sentiment": "positive",
                "sentiment_score": 0.7,
            },
            {
                "title": "Fed hawks now rule. Will Warsh go along with them?",
                "source": "Barrons",
                "content_preview": "Federal Reserve policy article.",
                "sentiment": "positive",
                "sentiment_score": 0.9,
            },
        ]
    )
    monkeypatch.setattr("autotrade.utils.news_cache.get_cache", lambda: cache)
    aggregator = NewsAggregator(
        news_analyzer=None, searx_client=None, catalyst_bridge=None
    )

    snapshot = aggregator.collect("AMC")

    assert snapshot["available"] is True
    assert snapshot["sentiment_score"] == 0.7
    assert snapshot["headline_count"] == 1
    assert snapshot["headlines"][0]["relevance_label"] == "relevant"
