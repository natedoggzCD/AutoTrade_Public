"""Tests for the hybrid NewsScorer (autotrade.analysis.news_scorer).

Covers each tier in isolation, the escalation rule, and the fallback chain
when Ollama or FinBERT is unavailable. All Ollama calls are mocked so the
tests are deterministic and run without network.
"""

from __future__ import annotations

import os
import importlib

import pytest

import autotrade.analysis.news_scorer as scorer


class FakeFinBERT:
    """Minimal FinBERT stand-in. Returns label/score driven by keywords so
    we can write deterministic chunked-aggregate tests without loading a real
    model."""

    def __init__(self, by_text=None, default=("neutral", 0.5, "low")):
        # by_text maps a substring -> (label, score, confidence). First match wins.
        self.by_text = by_text or {}
        self.default = default

    def analyze_text(self, text, return_all_scores=True):
        for needle, (label, score, conf) in self.by_text.items():
            if needle.lower() in (text or "").lower():
                return {"label": label, "score": score, "confidence": conf}
        label, score, conf = self.default
        return {"label": label, "score": score, "confidence": conf}


# ---------- Tier 1: chunked FinBERT ----------


def test_chunked_finbert_positive_body():
    fb = FakeFinBERT(by_text={"beat": ("positive", 0.9, "high")})
    body = "Q3 earnings beat estimates handily. " * 50  # > 400 chars
    result = scorer._score_chunked_finbert(fb, body)
    assert result.label == "positive"
    assert result.score > 0.5
    assert result.method == "chunked_finbert"
    assert result.error is None


def test_chunked_finbert_body_too_short():
    fb = FakeFinBERT()
    result = scorer._score_chunked_finbert(fb, "short body")
    assert result.error == "body_too_short"


def test_chunked_finbert_neutral_signed_average():
    """Negative and positive chunks balance out -> neutral aggregate.

    A FinBERT-like fake that picks the dominant keyword per chunk: when
    the body has equal positive and negative chunks, the length-weighted
    signed average collapses to ~0 -> neutral.
    """

    class DominantKeywordFinBERT:
        def analyze_text(self, text, return_all_scores=True):
            pos = (text or "").count("BULLISH")
            neg = (text or "").count("BEARISH")
            if pos > neg:
                return {"label": "positive", "score": 0.9, "confidence": "high"}
            if neg > pos:
                return {"label": "negative", "score": 0.9, "confidence": "high"}
            return {"label": "neutral", "score": 0.5, "confidence": "low"}

    # Build two distinct chunks (>= 2000 chars each) with opposite tone.
    pos_chunk = "BULLISH " * 300  # ~2400 chars
    neg_chunk = "BEARISH " * 300  # ~2400 chars
    body = pos_chunk + neg_chunk
    result = scorer._score_chunked_finbert(DominantKeywordFinBERT(), body)
    assert result.label == "neutral"


# ---------- Tier 2: LLM classifier (mocked Ollama) ----------


def _mock_ollama_response(content_json: str, status_code: int = 200):
    """Build a mock Ollama HTTP response."""
    class _Resp:
        def __init__(self, code, content):
            self.status_code = code
            self._content = content
        def json(self):
            import json as _json
            return {"message": {"content": self._content}}
    return _Resp(status_code, content_json)


def test_llm_classifier_parses_clean_json(monkeypatch):
    import requests
    def fake_post(url, json=None, timeout=None):
        return _mock_ollama_response(
            '{"label": "negative", "score": 0.85, "confidence": "high", '
            '"key_fact": "equity offering at 30% discount"}'
        )
    monkeypatch.setattr(requests, "post", fake_post)
    out = scorer._score_llm_classifier("AMC", "AMC announces offering", "body" * 200)
    assert out.label == "negative"
    assert out.score == 0.85
    assert out.confidence == "high"
    assert "offering" in out.detail
    assert out.error is None


def test_llm_classifier_handles_wrapped_json(monkeypatch):
    """Some models wrap JSON in prose; parser must recover."""
    import requests
    def fake_post(url, json=None, timeout=None):
        return _mock_ollama_response(
            'Here is the analysis: {"label":"positive","score":0.7,"confidence":"medium","key_fact":"FDA approval"} done.'
        )
    monkeypatch.setattr(requests, "post", fake_post)
    out = scorer._score_llm_classifier("MRK", "FDA nod", "body")
    assert out.label == "positive"
    assert out.score == 0.7


def test_llm_classifier_ollama_down(monkeypatch):
    import requests
    def fake_post(url, json=None, timeout=None):
        raise requests.ConnectionError("connection refused")
    monkeypatch.setattr(requests, "post", fake_post)
    out = scorer._score_llm_classifier("AAPL", "title", "body")
    assert out.error is not None
    assert "ollama_exc" in out.error


def test_llm_classifier_bad_json(monkeypatch):
    import requests
    def fake_post(url, json=None, timeout=None):
        return _mock_ollama_response("not json at all")
    monkeypatch.setattr(requests, "post", fake_post)
    out = scorer._score_llm_classifier("AAPL", "title", "body")
    assert out.error == "json_parse_failed"


# ---------- Tier 3: title-only FinBERT ----------


def test_title_finbert_runs():
    fb = FakeFinBERT(by_text={"merger": ("positive", 0.8, "high")})
    out = scorer._score_title_finbert(fb, "Company X announces merger")
    assert out.label == "positive"
    assert out.method == "title_finbert"


# ---------- Public score_article: mode dispatch + escalation ----------


def test_score_article_title_mode_uses_legacy_path(monkeypatch):
    monkeypatch.setenv("AUTOTRADE_NEWS_SCORER_MODE", "title")
    fb = FakeFinBERT(by_text={"offering": ("negative", 0.7, "high")})
    out = scorer.score_article(fb, "AMC", "AMC offering", body="long body " * 200)
    assert out.method == "title_finbert"
    assert out.label == "negative"


def test_score_article_chunked_mode_skips_llm(monkeypatch):
    monkeypatch.setenv("AUTOTRADE_NEWS_SCORER_MODE", "chunked")
    # Even with a working LLM, chunked mode must NOT call it. We assert by
    # patching requests.post to raise — if chunked called it, the test fails.
    import requests
    def boom(*a, **kw):
        raise AssertionError("chunked mode must not call Ollama")
    monkeypatch.setattr(requests, "post", boom)
    fb = FakeFinBERT(by_text={"beat": ("positive", 0.6, "high")})
    body = "Q3 beat estimates. " * 50
    out = scorer.score_article(fb, "AAPL", "title", body=body)
    assert out.method == "chunked_finbert"


def test_score_article_hybrid_falls_back_to_title_when_body_missing(monkeypatch):
    monkeypatch.setenv("AUTOTRADE_NEWS_SCORER_MODE", "hybrid")
    fb = FakeFinBERT(by_text={"upgrade": ("positive", 0.6, "medium")})
    out = scorer.score_article(fb, "AAPL", "Analyst upgrade", body="")
    assert out.method == "title_finbert"


def test_score_article_hybrid_escalates_when_chunked_uncertain(monkeypatch):
    monkeypatch.setenv("AUTOTRADE_NEWS_SCORER_MODE", "hybrid")
    monkeypatch.setenv("AUTOTRADE_NEWS_SCORER_ESCALATION_THRESHOLD", "0.40")
    importlib.reload(scorer)  # pick up new env

    # FinBERT returns neutral/low (below 0.40 threshold) -> must escalate.
    fb = FakeFinBERT(default=("neutral", 0.1, "low"))

    import requests
    calls = {"n": 0}
    def fake_post(url, json=None, timeout=None):
        calls["n"] += 1
        return _mock_ollama_response(
            '{"label":"negative","score":0.9,"confidence":"high","key_fact":"offering"}'
        )
    monkeypatch.setattr(requests, "post", fake_post)

    body = "x" * 600
    out = scorer.score_article(fb, "AAPL", "title", body=body)
    assert calls["n"] == 1, "LLM must be invoked when chunked is below threshold"
    assert out.method == "llm_classifier"
    assert out.escalated is True
    assert out.label == "negative"


def test_score_article_hybrid_skips_escalation_when_confident(monkeypatch):
    monkeypatch.setenv("AUTOTRADE_NEWS_SCORER_MODE", "hybrid")
    monkeypatch.setenv("AUTOTRADE_NEWS_SCORER_ESCALATION_THRESHOLD", "0.40")
    importlib.reload(scorer)

    fb = FakeFinBERT(default=("negative", 0.9, "high"))

    import requests
    def boom(*a, **kw):
        raise AssertionError("must not escalate when chunked is confident")
    monkeypatch.setattr(requests, "post", boom)

    body = "y" * 600
    out = scorer.score_article(fb, "AAPL", "title", body=body)
    assert out.method == "chunked_finbert"
    assert out.escalated is False
    assert out.label == "negative"


def test_score_article_hybrid_keeps_chunked_when_llm_fails(monkeypatch):
    """Graceful degradation: chunked uncertain + LLM down -> use chunked anyway."""
    monkeypatch.setenv("AUTOTRADE_NEWS_SCORER_MODE", "hybrid")
    monkeypatch.setenv("AUTOTRADE_NEWS_SCORER_ESCALATION_THRESHOLD", "0.40")
    importlib.reload(scorer)

    fb = FakeFinBERT(default=("positive", 0.2, "low"))

    import requests
    def fake_post(*a, **kw):
        raise requests.ConnectionError("ollama down")
    monkeypatch.setattr(requests, "post", fake_post)

    body = "z" * 600
    out = scorer.score_article(fb, "AAPL", "title", body=body)
    # Chunked label is preserved.
    assert out.method == "chunked_finbert"
    assert out.label == "positive"


def teardown_module(module):
    # Reset module to default state so other tests get a clean slate.
    for var in (
        "AUTOTRADE_NEWS_SCORER_MODE",
        "AUTOTRADE_NEWS_SCORER_ESCALATION_THRESHOLD",
    ):
        os.environ.pop(var, None)
    importlib.reload(scorer)
