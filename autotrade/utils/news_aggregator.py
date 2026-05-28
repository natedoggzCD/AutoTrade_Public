"""
Premarket news aggregation with graceful degradation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from autotrade.utils.catalyst_bridge import CatalystTagBridge
from autotrade.utils.news_relevance import assess_news_relevance

logger = logging.getLogger(__name__)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class NewsSnapshot:
    symbol: str
    available: bool
    sentiment_score: float
    headline_count: int
    coverage: str
    confidence: float
    headlines: List[Dict[str, Any]]
    source_status: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "available": self.available,
            "sentiment_score": self.sentiment_score,
            "headline_count": self.headline_count,
            "coverage": self.coverage,
            "confidence": self.confidence,
            "headlines": self.headlines,
            "source_status": self.source_status,
        }


class NewsAggregator:
    """
    Multi-source news collector.

    Source priority:
    1) `NewsSentimentAnalyzer` (yfinance-backed sentiment + cached headlines)
    2) `SearXNGClient` finance/news search fallback
    """

    _LOW_SIGNAL_TITLE_FRAGMENTS = (
        "stock price",
        "stock quote",
        "price & overview",
        "quote & news",
        "stocks - robinhood",
        "stock price today",
        "chart -",
        "historical data",
    )
    _LOW_SIGNAL_URL_FRAGMENTS = (
        "robinhood.com",
        "marketwatch.com/investing/stock/",
        "stockanalysis.com/stocks/",
        "zacks.com/stock/quote/",
        "marketbeat.com/stocks/",
        "finance.yahoo.com/quote/",
    )
    _SEARX_NEWS_KEYWORDS = (
        "earnings",
        "guidance",
        "contract",
        "order",
        "deal",
        "announces",
        "announcement",
        "analyst",
        "upgrade",
        "downgrade",
        "price target",
        "sec",
        "filing",
        "fda",
        "approval",
        "press release",
        "premarket",
        "after hours",
        "acquisition",
        "merger",
        "partnership",
        "dividend",
    )
    _SEARX_QUERY_TEMPLATES = (
        ("news", '"{symbol}" stock news today'),
        (
            "catalyst",
            '"{symbol}" earnings guidance contract analyst upgrade downgrade press release',
        ),
    )

    def __init__(
        self,
        max_news_age_days: int = 14,
        headline_limit: int = 5,
        news_analyzer: Optional[Any] = None,
        searx_client: Optional[Any] = None,
        catalyst_bridge: Optional[CatalystTagBridge] = None,
    ) -> None:
        self.max_news_age_days = max(1, int(max_news_age_days))
        self.headline_limit = max(1, int(headline_limit))
        self.news_analyzer = news_analyzer or self._build_news_analyzer()
        self.searx_client = searx_client or self._build_searx_client()
        self.catalyst_bridge = catalyst_bridge or self._build_catalyst_bridge()

    def collect(
        self, symbol: str, max_age_hours: Optional[int] = None
    ) -> Dict[str, Any]:
        symbol = str(symbol or "").upper().strip()
        if not symbol:
            return NewsSnapshot(
                symbol="",
                available=False,
                sentiment_score=0.0,
                headline_count=0,
                coverage="none",
                confidence=0.0,
                headlines=[],
                source_status={"input": "missing_symbol"},
            ).to_dict()

        # Check cache first
        from autotrade.utils.news_cache import get_cache

        cache = get_cache()
        cached = cache.get_cached_news(symbol, max_age_hours=max_age_hours)
        if cached is not None:
            relevant_cached: List[Dict[str, Any]] = []
            for row in cached:
                title = str(row.get("title", "") or "")
                preview = str(row.get("content_preview", "") or "")
                label = row.get("relevance_label")
                score = row.get("relevance_score")
                source = row.get("relevance_source")
                if not label:
                    relevance = assess_news_relevance(symbol, title, preview)
                    label = relevance.label
                    score = relevance.score
                    source = f"deterministic:{relevance.reason}"
                enriched = {
                    **row,
                    "relevance_label": label,
                    "relevance_score": score,
                    "relevance_source": source,
                }
                if label == "relevant":
                    relevant_cached.append(enriched)

            if relevant_cached:
                sentiment_vals = [
                    float(h["sentiment_score"])
                    for h in relevant_cached
                    if h.get("sentiment_score") is not None
                ]
                avg_sentiment = (
                    sum(sentiment_vals) / len(sentiment_vals) if sentiment_vals else 0.0
                )
                coverage = "full" if len(relevant_cached) >= 3 else "partial"
                confidence = 0.4 + min(0.4, 0.1 * len(relevant_cached))
                catalyst = (
                    self.catalyst_bridge.extract(symbol, relevant_cached)
                    if self.catalyst_bridge is not None
                    else {
                        "symbol": symbol,
                        "has_catalyst": False,
                        "catalyst_score": 0.0,
                        "catalyst_tags": [],
                        "catalyst_note": "",
                        "earnings_context": {"upcoming": [], "recent_surprise": []},
                    }
                )
                snapshot = {
                    "symbol": symbol,
                    "available": True,
                    "sentiment_score": avg_sentiment,
                    "headline_count": len(relevant_cached),
                    "coverage": coverage,
                    "confidence": confidence,
                    "headlines": relevant_cached[: self.headline_limit],
                    "source_status": {"cache": "hit_relevant"},
                    "cached": True,
                    "search_diagnostics": {
                        "symbol": symbol,
                        "cache_rows": len(cached),
                        "cache_relevant": len(relevant_cached),
                        "cache_rejected": len(cached) - len(relevant_cached),
                    },
                }
                snapshot.update(catalyst)
                return snapshot

            logger.debug(
                "%s: cache hit had no relevant news rows; refreshing fallback sources",
                symbol,
            )

        source_status: Dict[str, str] = {}
        combined_headlines: List[Dict[str, Any]] = []
        sentiment_values: List[float] = []
        search_diagnostics: Dict[str, Any] = {
            "symbol": symbol,
            "yfinance_headlines": 0,
            "searx_queries": [],
            "searx_raw_hits": 0,
            "searx_kept": 0,
            "rejected_titles": [],
        }

        # Source 1: yfinance + FinBERT pipeline
        if self.news_analyzer is None:
            source_status["news_sentiment"] = "unavailable"
        else:
            try:
                # Pass cache_ttl if analyzer supports it
                result = self.news_analyzer.analyze_ticker(symbol)
                # `weighted_score` is typically in [-1, 1].
                weighted = float(result.get("weighted_score", 0.0) or 0.0)
                sentiment_values.append(_clamp(weighted, -1.0, 1.0))
                headlines = (result.get("top_headlines") or [])[: self.headline_limit]
                for h in headlines:
                    title = str(h.get("headline", "")).strip()
                    if not title:
                        continue
                    combined_headlines.append(
                        {
                            "title": title,
                            "source": str(h.get("source", "news_sentiment")),
                            "sentiment": h.get("sentiment"),
                            "sentiment_score": h.get("weighted_score", 0.0),
                            "source_type": "news_sentiment",
                            "quality_score": 4.0,
                        }
                    )
                search_diagnostics["yfinance_headlines"] = len(headlines)
                source_status["news_sentiment"] = "ok"
            except Exception as exc:
                source_status["news_sentiment"] = f"error:{type(exc).__name__}"
                logger.debug("NewsSentimentAnalyzer failed for %s: %s", symbol, exc)

        # Source 2: SearXNG fallback
        if self.searx_client is None:
            source_status["searxng"] = "unavailable"
        else:
            try:
                existing_titles = {
                    str(item.get("title", "")).strip().lower()
                    for item in combined_headlines
                    if str(item.get("title", "")).strip()
                }
                searx_headlines, searx_debug = self._collect_relevant_searx_headlines(
                    symbol=symbol,
                    existing_titles=existing_titles,
                )
                combined_headlines.extend(searx_headlines)
                search_diagnostics.update(searx_debug)
                if searx_headlines:
                    source_status["searxng"] = "ok"
                elif searx_debug.get("raw_hits", 0) > 0:
                    source_status["searxng"] = "no_relevant_results"
                else:
                    source_status["searxng"] = "empty"
            except Exception as exc:
                source_status["searxng"] = f"error:{type(exc).__name__}"
                logger.debug("SearXNG news fallback failed for %s: %s", symbol, exc)

        deduped = self._dedupe_headlines(combined_headlines)[: self.headline_limit]

        # PERSIST TO CACHE (always update fetch log to prevent redundant fetches)
        cache.cache_news(symbol, deduped, source="aggregator")

        sentiment = _clamp(
            sum(sentiment_values) / max(1, len(sentiment_values)),
            -1.0,
            1.0,
        )

        yfinance_kept = sum(
            1
            for item in deduped
            if str(item.get("source_type", "")) == "news_sentiment"
        )
        searx_kept = sum(
            1 for item in deduped if str(item.get("source_type", "")) == "searxng"
        )
        has_reliable_coverage = bool(yfinance_kept > 0 or searx_kept > 0)

        if not deduped or not has_reliable_coverage:
            coverage = "none"
        elif yfinance_kept > 0 and searx_kept > 0:
            coverage = "full"
        else:
            coverage = "partial"

        confidence = 0.0
        if has_reliable_coverage:
            avg_quality = sum(
                float(item.get("quality_score", 0.0) or 0.0) for item in deduped
            ) / max(1, len(deduped))
            confidence = _clamp(
                0.30
                + min(0.30, 0.08 * len(deduped))
                + min(0.20, abs(sentiment) * 0.30)
                + min(0.20, avg_quality / 20.0),
                0.0,
                1.0,
            )

        snapshot = NewsSnapshot(
            symbol=symbol,
            available=bool(has_reliable_coverage),
            sentiment_score=sentiment,
            headline_count=len(deduped),
            coverage=coverage,
            confidence=confidence,
            headlines=deduped,
            source_status=source_status,
        ).to_dict()
        snapshot["search_diagnostics"] = {
            **search_diagnostics,
            "yfinance_kept": yfinance_kept,
            "searx_kept": searx_kept,
            "headline_titles": [str(item.get("title", "")) for item in deduped[:3]],
        }
        catalyst = (
            self.catalyst_bridge.extract(symbol, deduped)
            if self.catalyst_bridge is not None
            else {
                "symbol": symbol,
                "has_catalyst": False,
                "catalyst_score": 0.0,
                "catalyst_tags": [],
                "catalyst_note": "",
                "earnings_context": {"upcoming": [], "recent_surprise": []},
            }
        )
        snapshot.update(catalyst)
        return snapshot

    def refresh_batch(
        self, symbols: List[str], max_age_hours: int = 4, max_workers: int = 5
    ) -> Dict[str, Any]:
        """
        Refresh news and sentiment for a batch of symbols.
        Only fetches news for symbols that are stale in the cache.
        """
        from autotrade.utils.news_cache import get_cache
        import concurrent.futures

        cache = get_cache()

        needs_update = cache.get_uncached_symbols(symbols, max_age_hours=max_age_hours)
        results = {"total": len(symbols), "refreshed": 0, "errors": 0, "details": {}}

        if not needs_update:
            logger.info("All symbols in batch are fresh in cache.")
            return results

        logger.info(
            f"Refreshing news for {len(needs_update)} symbols using {max_workers} workers..."
        )

        def _refresh_one(sym: str) -> tuple[str, bool, Optional[str]]:
            try:
                res = self.collect(sym)
                return sym, bool(res.get("available")), None, res
            except Exception as e:
                return sym, False, str(e), None

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(_refresh_one, sym): sym for sym in needs_update
            }
            for future in concurrent.futures.as_completed(future_to_symbol):
                sym, ok, error, snapshot = future.result()
                if error:
                    results["errors"] += 1
                    results["details"][sym] = f"error: {error}"
                else:
                    if ok:
                        results["refreshed"] += 1
                    results["details"][sym] = {
                        "status": "ok" if ok else "no_news",
                        "headline_count": int(
                            (snapshot or {}).get("headline_count", 0) or 0
                        ),
                        "coverage": str((snapshot or {}).get("coverage", "none")),
                    }

        return results

    def _build_news_analyzer(self) -> Optional[Any]:
        try:
            from autotrade.analysis.news_sentiment import NewsSentimentAnalyzer

            return NewsSentimentAnalyzer(max_news_age_days=self.max_news_age_days)
        except Exception as exc:
            logger.debug("NewsSentimentAnalyzer unavailable: %s", exc)
            return None

    def _build_searx_client(self) -> Optional[Any]:
        try:
            from autotrade.analysis.searxng_client import SearXNGClient

            return SearXNGClient()
        except Exception as exc:
            logger.debug("SearXNGClient unavailable: %s", exc)
            return None

    def _build_catalyst_bridge(self) -> Optional[CatalystTagBridge]:
        try:
            return CatalystTagBridge()
        except Exception as exc:
            logger.debug("CatalystTagBridge unavailable: %s", exc)
            return None

    @classmethod
    def _is_symbol_mentioned(cls, symbol: str, text: str) -> bool:
        symbol_key = str(symbol or "").strip()
        if not symbol_key or not text:
            return False
        pattern = rf"\b{re.escape(symbol_key)}\b"
        return bool(re.search(pattern, text, flags=re.IGNORECASE))

    @classmethod
    def _score_searx_result(cls, symbol: str, item: Any) -> tuple[float, str]:
        title = str(getattr(item, "title", "") or "").strip()
        content = str(getattr(item, "content", "") or "").strip()
        url = str(getattr(item, "url", "") or "").strip().lower()
        engine = str(getattr(item, "engine", "") or "").strip().lower()
        combined = f"{title} {content}"
        combined_lower = combined.lower()

        score = 0.0
        reasons: List[str] = []

        if cls._is_symbol_mentioned(symbol, combined):
            score += 4.0
            reasons.append("symbol_match")
        else:
            score -= 4.0
            reasons.append("missing_symbol")

        keyword_hits = sum(1 for kw in cls._SEARX_NEWS_KEYWORDS if kw in combined_lower)
        if keyword_hits:
            score += min(3.0, float(keyword_hits))
            reasons.append("news_keywords")

        if getattr(item, "publishedDate", None):
            score += 1.0
            reasons.append("dated")

        if "news" in engine:
            score += 1.0
            reasons.append("news_engine")

        if any(
            fragment in combined_lower for fragment in cls._LOW_SIGNAL_TITLE_FRAGMENTS
        ):
            score -= 5.0
            reasons.append("low_signal_title")
        if any(fragment in url for fragment in cls._LOW_SIGNAL_URL_FRAGMENTS):
            score -= 4.0
            reasons.append("low_signal_url")

        return score, ",".join(reasons)

    def _collect_relevant_searx_headlines(
        self,
        *,
        symbol: str,
        existing_titles: Optional[set[str]] = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        accepted: List[Dict[str, Any]] = []
        seen_titles = set(existing_titles or set())
        debug: Dict[str, Any] = {
            "searx_queries": [],
            "raw_hits": 0,
            "rejected_titles": [],
        }

        for query_name, query_template in self._SEARX_QUERY_TEMPLATES:
            if len(accepted) >= self.headline_limit:
                break

            query = query_template.format(symbol=symbol)
            debug["searx_queries"].append(query)
            response = self.searx_client.search_news(
                query,
                max_results=max(self.headline_limit, 3),
            )
            results = list(getattr(response, "results", []) or [])
            debug["raw_hits"] += len(results)

            for item in results:
                if len(accepted) >= self.headline_limit:
                    break

                title = str(getattr(item, "title", "") or "").strip()
                if not title:
                    continue
                title_key = title.lower()
                if title_key in seen_titles:
                    continue

                quality_score, quality_reason = self._score_searx_result(symbol, item)
                if quality_score < 2.0:
                    if len(debug["rejected_titles"]) < 8:
                        debug["rejected_titles"].append(
                            f"{title[:90]} | score={quality_score:.1f} | {quality_reason}"
                        )
                    continue

                accepted.append(
                    {
                        "title": title,
                        "source": str(getattr(item, "engine", "searxng") or "searxng"),
                        "sentiment": None,
                        "url": str(getattr(item, "url", "") or ""),
                        "published_at": getattr(item, "publishedDate", None),
                        "query": query_name,
                        "source_type": "searxng",
                        "quality_score": round(quality_score, 2),
                    }
                )
                seen_titles.add(title_key)

        return accepted, debug

    @staticmethod
    def _dedupe_headlines(headlines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out: List[Dict[str, Any]] = []
        for item in headlines:
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            key = title.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out
