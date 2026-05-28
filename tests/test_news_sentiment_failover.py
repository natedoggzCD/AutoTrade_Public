from datetime import datetime
from types import SimpleNamespace

import pandas as pd

import autotrade.analysis.news_sentiment as news_sentiment
from langgraph_workflow import nodes


class _BrokenFinBERTError(Exception):
    def __str__(self):
        raise RecursionError("maximum recursion depth exceeded while getting str")


def _sample_news_frame():
    now = datetime(2026, 3, 25, 9, 30)
    return pd.DataFrame(
        [
            {
                "Title": "ABC files shelf offering after rally",
                "Date": now,
                "freshness_weight": 1.0,
                "age_days": 0,
            },
            {
                "Title": "ABC wins new enterprise contract",
                "Date": now,
                "freshness_weight": 0.85,
                "age_days": 1,
            },
        ]
    )


def test_news_sentiment_disables_finbert_after_first_init_failure(monkeypatch):
    attempts = {"count": 0}

    def _raise_shared_init(*args, **kwargs):
        attempts["count"] += 1
        raise _BrokenFinBERTError("cache missing")

    monkeypatch.setattr(
        news_sentiment, "get_shared_finbert_analyzer", _raise_shared_init
    )
    monkeypatch.setattr(
        news_sentiment.NewsSentimentAnalyzer,
        "_shared_finbert",
        None,
        raising=False,
    )
    monkeypatch.setattr(
        news_sentiment.NewsSentimentAnalyzer,
        "_shared_finbert_disabled",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        news_sentiment.NewsSentimentAnalyzer,
        "_shared_finbert_error",
        None,
        raising=False,
    )
    monkeypatch.setattr(
        news_sentiment.NewsSentimentAnalyzer,
        "get_news",
        lambda self, ticker: _sample_news_frame(),
    )
    monkeypatch.setattr(
        news_sentiment.NewsSentimentAnalyzer,
        "get_current_price",
        lambda self, ticker: None,
    )

    first = news_sentiment.NewsSentimentAnalyzer()
    first_result = first.analyze_ticker("ABC")

    second = news_sentiment.NewsSentimentAnalyzer()
    second_result = second.analyze_ticker("XYZ")

    assert attempts["count"] == 1
    assert first_result["sentiment"] == "neutral"
    assert first_result["news_count"] == 2
    assert (
        first_result["headlines"][0]["headline"]
        == "ABC files shelf offering after rally"
    )
    assert first_result["error"].startswith("_BrokenFinBERTError:")
    assert second_result["error"] == first_result["error"]


def test_node_news_sentiment_uses_safe_exception_text(monkeypatch):
    class _RaisingAnalyzer:
        def __init__(self, max_news_age_days=180, cache_ttl_hours=4):
            pass

        def analyze_ticker(self, ticker):
            raise _BrokenFinBERTError("cache missing")

    monkeypatch.setattr(news_sentiment, "NewsSentimentAnalyzer", _RaisingAnalyzer)
    monkeypatch.setattr(
        "config.config_loader.get_config",
        lambda: SimpleNamespace(sentiment=SimpleNamespace(max_news_age_days=7)),
    )

    result = nodes.node_news_sentiment({"position": {"symbol": "ABC"}})

    assert result["news_context"]["overall_sentiment"] == "neutral"
    assert result["errors"]
    assert result["errors"][0].startswith("news_sentiment: _BrokenFinBERTError:")


def test_news_sentiment_skips_irrelevant_yfinance_articles(monkeypatch):
    now = datetime(2026, 5, 22, 9, 30)

    class _FakeFinbert:
        def analyze_text(self, text, return_all_scores=True):
            return {"label": "positive", "score": 0.9, "confidence": "high"}

    monkeypatch.setattr(
        news_sentiment.NewsSentimentAnalyzer,
        "get_news",
        lambda self, ticker: pd.DataFrame(
            [
                {
                    "Title": "Fed hawks now rule. Will Warsh go along with them?",
                    "Summary": "Federal Reserve policy article with no company facts.",
                    "Date": now,
                    "Publisher": "Unit",
                    "freshness_weight": 1.0,
                    "age_days": 0,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        news_sentiment.NewsSentimentAnalyzer,
        "get_current_price",
        lambda self, ticker: None,
    )
    monkeypatch.setattr(
        news_sentiment.NewsSentimentAnalyzer,
        "_ensure_finbert",
        lambda self: True,
    )

    analyzer = news_sentiment.NewsSentimentAnalyzer()
    analyzer.finbert = _FakeFinbert()
    result = analyzer.analyze_ticker("PFE")

    assert result["sentiment"] == "neutral"
    assert result["weighted_score"] == 0.0
    assert result["relevant_news_count"] == 0
    assert result["skipped_relevance_count"] == 1
    assert result["skipped_relevance"][0]["relevance_label"] == "irrelevant"


def test_news_sentiment_scores_relevant_ticker_articles(monkeypatch):
    now = datetime(2026, 5, 22, 9, 30)

    class _FakeFinbert:
        def analyze_text(self, text, return_all_scores=True):
            return {"label": "positive", "score": 0.9, "confidence": "high"}

    monkeypatch.setattr(
        news_sentiment.NewsSentimentAnalyzer,
        "get_news",
        lambda self, ticker: pd.DataFrame(
            [
                {
                    "Title": "AMC shares jumped after CEO disclosed stock purchase",
                    "Summary": "AMC Entertainment CEO disclosed the purchase of shares.",
                    "Date": now,
                    "Publisher": "Unit",
                    "freshness_weight": 1.0,
                    "age_days": 0,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        news_sentiment.NewsSentimentAnalyzer,
        "get_current_price",
        lambda self, ticker: None,
    )
    monkeypatch.setattr(
        news_sentiment.NewsSentimentAnalyzer,
        "_ensure_finbert",
        lambda self: True,
    )

    analyzer = news_sentiment.NewsSentimentAnalyzer()
    analyzer.finbert = _FakeFinbert()
    result = analyzer.analyze_ticker("AMC")

    assert result["sentiment"] == "positive"
    assert result["weighted_score"] > 0
    assert result["relevant_news_count"] == 1
    assert result["top_headlines"][0]["relevance_label"] == "relevant"
