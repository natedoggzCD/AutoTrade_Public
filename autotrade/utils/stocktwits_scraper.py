"""
Stocktwits scraper wrapper for premarket usage.

Keeps scraping bounded and returns normalized sentiment metadata with
graceful fallback behavior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class StocktwitsSnapshot:
    symbol: str
    available: bool
    sentiment_score: float
    bull_bear_ratio: float
    message_velocity: float
    is_trending: bool
    coverage: str
    confidence: float
    source_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "available": self.available,
            "sentiment_score": self.sentiment_score,
            "bull_bear_ratio": self.bull_bear_ratio,
            "message_velocity": self.message_velocity,
            "is_trending": self.is_trending,
            "coverage": self.coverage,
            "confidence": self.confidence,
            "source_status": self.source_status,
        }


class StocktwitsScraper:
    """Bounded Stocktwits sentiment extractor."""

    def __init__(
        self,
        max_messages: int = 30,
        analyzer: Optional[Any] = None,
    ) -> None:
        self.max_messages = max(5, int(max_messages))
        self.analyzer = analyzer or self._build_analyzer()

    def fetch(self, symbol: str) -> Dict[str, Any]:
        symbol = str(symbol or "").upper().strip()
        if not symbol:
            return StocktwitsSnapshot(
                symbol="",
                available=False,
                sentiment_score=0.0,
                bull_bear_ratio=1.0,
                message_velocity=0.0,
                is_trending=False,
                coverage="none",
                confidence=0.0,
                source_status="missing_symbol",
            ).to_dict()

        if self.analyzer is None:
            return StocktwitsSnapshot(
                symbol=symbol,
                available=False,
                sentiment_score=0.0,
                bull_bear_ratio=1.0,
                message_velocity=0.0,
                is_trending=False,
                coverage="none",
                confidence=0.0,
                source_status="unavailable",
            ).to_dict()

        try:
            sentiment = self.analyzer.get_sentiment(symbol)
            total_messages = int(getattr(sentiment, "total_messages", 0) or 0)
            velocity = float(getattr(sentiment, "message_velocity", 0.0) or 0.0)
            bull_bear_ratio = float(getattr(sentiment, "bull_bear_ratio", 1.0) or 1.0)
            is_trending = bool(getattr(sentiment, "is_trending", False))

            # Source library emits -100..100 sentiment scale.
            raw_score = float(getattr(sentiment, "sentiment_score", 0.0) or 0.0)
            normalized_score = _clamp(raw_score / 100.0, -1.0, 1.0)

            available = total_messages > 0
            coverage = "full" if total_messages >= self.max_messages else "partial" if total_messages > 0 else "none"
            confidence = 0.0
            if available:
                confidence = _clamp(
                    0.3 + min(0.35, total_messages / float(self.max_messages) * 0.35) + min(0.35, abs(normalized_score) * 0.35),
                    0.0,
                    1.0,
                )

            return StocktwitsSnapshot(
                symbol=symbol,
                available=available,
                sentiment_score=normalized_score,
                bull_bear_ratio=bull_bear_ratio,
                message_velocity=velocity,
                is_trending=is_trending,
                coverage=coverage,
                confidence=confidence,
                source_status="ok",
            ).to_dict()
        except Exception as exc:
            logger.debug("Stocktwits scrape failed for %s: %s", symbol, exc)
            return StocktwitsSnapshot(
                symbol=symbol,
                available=False,
                sentiment_score=0.0,
                bull_bear_ratio=1.0,
                message_velocity=0.0,
                is_trending=False,
                coverage="none",
                confidence=0.0,
                source_status=f"error:{type(exc).__name__}",
            ).to_dict()

    @staticmethod
    def _build_analyzer() -> Optional[Any]:
        try:
            from autotrade.analysis.stocktwits_sentiment import StocktwitsSentimentAnalyzer

            return StocktwitsSentimentAnalyzer(cache_ttl=300, include_messages=False)
        except Exception as exc:
            logger.debug("StocktwitsSentimentAnalyzer unavailable: %s", exc)
            return None
