"""Hybrid news sentiment scorer for production use.

Three-tier scoring with deterministic fallback. The relevance gate
(autotrade.utils.news_relevance) runs BEFORE this scorer; irrelevant
articles never reach here.

Tier 1: chunked FinBERT on article body (138ms typical, ~69% accuracy)
Tier 2: qwen2.5:3b classifier escalation when Tier 1 is uncertain
        (|score| < CHUNKED_ESCALATION_THRESHOLD). 3.8s typical, ~85% accuracy.
Tier 3: title-only FinBERT (the legacy production path, ~23% accuracy)
        used when body is unavailable OR both higher tiers fail.

Audit data: reports/news_sentiment_audit_run20.json (2026-05-23, n=13 relevant).

Mode is controlled by env var AUTOTRADE_NEWS_SCORER_MODE:
  "hybrid"  (default) — tier 1 + escalation
  "chunked" — tier 1 only (cheapest deterministic)
  "title"   — tier 3 only (legacy behavior; for emergency revert)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# ---------- Configuration ----------

OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_MODEL = os.environ.get("AUTOTRADE_NEWS_SCORER_LLM_MODEL", "qwen2.5:3b")
LLM_TIMEOUT_SECONDS = int(os.environ.get("AUTOTRADE_NEWS_SCORER_LLM_TIMEOUT", "30"))

# When chunked FinBERT magnitude is below this, escalate to LLM.
# Tuned on n=13 audit: 38% escalation rate.
CHUNKED_ESCALATION_THRESHOLD = float(
    os.environ.get("AUTOTRADE_NEWS_SCORER_ESCALATION_THRESHOLD", "0.40")
)

# FinBERT effective window. ~2000 chars ≈ 512 tokens.
FINBERT_CHUNK_CHARS = 2000
FINBERT_MAX_CHUNKS = 4

# Magnitude threshold for converting raw signed average -> label.
LABEL_DECISION_THRESHOLD = 0.15

# Length floor below which chunked FinBERT degrades to title-only.
MIN_BODY_CHARS_FOR_CHUNK = 400


def _get_mode() -> str:
    return os.environ.get("AUTOTRADE_NEWS_SCORER_MODE", "hybrid").strip().lower()


# ---------- Result ----------


@dataclass
class ScoredArticle:
    """Result of scoring one article."""

    label: str = "neutral"  # positive | negative | neutral
    score: float = 0.0  # magnitude 0.0-1.0
    confidence: str = "low"  # low | medium | high
    method: str = "title_finbert"  # tier that produced this result
    latency_ms: float = 0.0
    detail: str = ""  # key fact (LLM) or chunk breakdown (FinBERT)
    error: Optional[str] = None
    escalated: bool = False  # True if tier-2 fired

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "score": float(self.score),
            "confidence": self.confidence,
            "method": self.method,
            "latency_ms": float(self.latency_ms),
            "detail": self.detail,
            "error": self.error,
            "escalated": self.escalated,
        }


# ---------- Tier 1: chunked FinBERT ----------


def _score_chunked_finbert(finbert, body: str) -> ScoredArticle:
    """Tier 1: chunk body, score each, length-weighted signed average."""
    t0 = time.perf_counter()
    text = (body or "").strip()
    if len(text) < MIN_BODY_CHARS_FOR_CHUNK:
        return ScoredArticle(
            method="chunked_finbert",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            error="body_too_short",
        )
    try:
        chunks = [
            text[i : i + FINBERT_CHUNK_CHARS]
            for i in range(0, len(text), FINBERT_CHUNK_CHARS)
        ][:FINBERT_MAX_CHUNKS]
        signed: List[float] = []
        weights: List[float] = []
        confs: List[str] = []
        for ch in chunks:
            out = finbert.analyze_text(ch, return_all_scores=True)
            label = out.get("label", "neutral")
            score = float(out.get("score", 0.0))
            sign = 1.0 if label == "positive" else (-1.0 if label == "negative" else 0.0)
            signed.append(sign * score)
            weights.append(float(len(ch)))
            confs.append(out.get("confidence", "low"))
        total_w = sum(weights) or 1.0
        avg = sum(s * w for s, w in zip(signed, weights)) / total_w
        if avg > LABEL_DECISION_THRESHOLD:
            agg_label = "positive"
        elif avg < -LABEL_DECISION_THRESHOLD:
            agg_label = "negative"
        else:
            agg_label = "neutral"
        agg_conf = max(set(confs), key=confs.count) if confs else "low"
        return ScoredArticle(
            label=agg_label,
            score=abs(avg),
            confidence=agg_conf,
            method="chunked_finbert",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            detail=f"{len(chunks)}_chunks",
        )
    except Exception as e:
        logger.debug("chunked FinBERT failed: %s", e)
        return ScoredArticle(
            method="chunked_finbert",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            error=f"finbert_exc:{type(e).__name__}",
        )


# ---------- Tier 2: LLM classifier ----------


_CLASSIFIER_SYSTEM = (
    "You are a financial sentiment classifier. Read the article and respond "
    "with strict JSON only. Focus on impact to the company's stock price. "
    "Do not include any reasoning or preamble before the JSON."
)


def _classifier_prompt(ticker: str, title: str, body: str) -> str:
    cleaned = (body or "")[:6000]
    return (
        f"Classify the sentiment of this article for {ticker}. "
        "Output JSON with EXACTLY these keys:\n"
        '  "label": "positive" | "negative" | "neutral"\n'
        '  "score": number 0.0-1.0 (magnitude; 0=weak, 1=very strong)\n'
        '  "confidence": "low" | "medium" | "high"\n'
        '  "key_fact": "one short sentence naming the most price-relevant fact"\n\n'
        "Material negative: equity offerings, dilution, going concern, lawsuit, "
        "FDA rejection, earnings miss, guidance cut, downgrade, "
        "competitor wins (e.g. 'X beats Y' is negative for Y).\n"
        "Material positive: M&A premium, FDA approval, earnings beat, guidance raise, "
        "upgrade, large contract win, CEO insider buying.\n\n"
        f"TITLE: {title}\n\nARTICLE:\n{cleaned}\n\nJSON:"
    )


def _parse_classifier_json(raw: str) -> Optional[Dict[str, Any]]:
    """Strict JSON, else extract first {...} block."""
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except Exception:
            return None
    return None


def _score_llm_classifier(ticker: str, title: str, body: str) -> ScoredArticle:
    """Tier 2: qwen2.5:3b (or configured model) as direct classifier."""
    t0 = time.perf_counter()
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": _CLASSIFIER_SYSTEM},
            {"role": "user", "content": _classifier_prompt(ticker, title, body)},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 400},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=LLM_TIMEOUT_SECONDS)
    except Exception as e:
        return ScoredArticle(
            method="llm_classifier",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            error=f"ollama_exc:{type(e).__name__}",
        )
    if resp.status_code != 200:
        return ScoredArticle(
            method="llm_classifier",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            error=f"ollama_http_{resp.status_code}",
        )
    try:
        data = resp.json()
    except Exception as e:
        return ScoredArticle(
            method="llm_classifier",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            error=f"json_decode:{type(e).__name__}",
        )
    raw = (
        (data.get("message") or {}).get("content")
        or data.get("response")
        or ""
    ).strip()
    if not raw:
        return ScoredArticle(
            method="llm_classifier",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            error="empty_response",
        )
    parsed = _parse_classifier_json(raw)
    if not isinstance(parsed, dict):
        return ScoredArticle(
            method="llm_classifier",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            error="json_parse_failed",
            detail=raw[:200],
        )
    label = str(parsed.get("label", "neutral")).lower()
    if label not in {"positive", "negative", "neutral"}:
        label = "neutral"
    try:
        score = float(parsed.get("score", 0.0))
    except Exception:
        score = 0.0
    confidence = str(parsed.get("confidence", "low")).lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"
    key_fact = str(parsed.get("key_fact", ""))[:240]
    return ScoredArticle(
        label=label,
        score=max(0.0, min(1.0, score)),
        confidence=confidence,
        method="llm_classifier",
        latency_ms=(time.perf_counter() - t0) * 1000.0,
        detail=key_fact,
    )


# ---------- Tier 3: title-only FinBERT (legacy) ----------


def _score_title_finbert(finbert, title: str) -> ScoredArticle:
    """Tier 3: the current production path. Used as final fallback."""
    t0 = time.perf_counter()
    try:
        out = finbert.analyze_text(title, return_all_scores=True)
        return ScoredArticle(
            label=out.get("label", "neutral"),
            score=float(out.get("score", 0.0)),
            confidence=out.get("confidence", "low"),
            method="title_finbert",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
    except Exception as e:
        return ScoredArticle(
            method="title_finbert",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            error=f"finbert_exc:{type(e).__name__}",
        )


# ---------- Public entrypoint ----------


def score_article(
    finbert,
    ticker: str,
    title: str,
    body: str = "",
) -> ScoredArticle:
    """Score one article using the configured mode.

    The relevance gate must have already approved this article. This function
    assumes ticker is genuinely the subject of the article.
    """
    mode = _get_mode()

    # Emergency-revert mode: only use the legacy title path.
    if mode == "title":
        return _score_title_finbert(finbert, title)

    # Body too short or missing -> chunked is useless; fall to title.
    body_usable = len(body or "") >= MIN_BODY_CHARS_FOR_CHUNK

    if mode == "chunked":
        if not body_usable:
            return _score_title_finbert(finbert, title)
        return _score_chunked_finbert(finbert, body)

    # mode == "hybrid" (default)
    if not body_usable:
        # No body: degrade to title (current production behavior).
        return _score_title_finbert(finbert, title)

    chunked = _score_chunked_finbert(finbert, body)
    if chunked.error:
        # Chunked failed; try LLM then title.
        llm = _score_llm_classifier(ticker, title, body)
        if not llm.error:
            llm.escalated = True
            return llm
        return _score_title_finbert(finbert, title)

    # Chunked succeeded. Escalate to LLM only if magnitude is below threshold.
    if chunked.score >= CHUNKED_ESCALATION_THRESHOLD:
        return chunked

    llm = _score_llm_classifier(ticker, title, body)
    if llm.error:
        # LLM unavailable -> stick with chunked (graceful degradation).
        logger.debug(
            "LLM escalation failed (%s); using chunked result for %s",
            llm.error, ticker,
        )
        return chunked
    llm.escalated = True
    return llm
