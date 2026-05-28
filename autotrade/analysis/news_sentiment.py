"""
News Sentiment Analyzer v2
- Uses yfinance (primary, no rate limits) + FinBERT for sentiment analysis
- Time-weighted news (recent news weighted more, old news still considered)
- Price movement since article (is it priced in?)
- Supports up to 2 quarters (180 days) of history

No API keys required - 100% local/open-source
"""

import os
import sys
from pathlib import Path

# Apply Windows-safe stdio configuration before any FinBERT/transformers load path.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

if sys.platform == "win32":
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import warnings
from autotrade.utils.mcp_client import mcp_fetch
from autotrade.utils.news_relevance import assess_news_relevance
from autotrade.utils.safe_logging import safe_exception_text
from autotrade.analysis.news_scorer import score_article as _score_article

warnings.filterwarnings("ignore")

# Import the existing FinBERT analyzer
from autotrade.analysis.finbert_analyzer import get_shared_finbert_analyzer  # noqa: E402

logger = logging.getLogger(__name__)


class NewsSentimentAnalyzer:
    """
    Fetches news from yfinance (no rate limits!) and analyzes sentiment using FinBERT.

    Features:
    - Time-weighted scoring (recent news matters more)
    - Price movement analysis (is the news priced in?)
    - Up to 2 quarters of news history
    """

    # Freshness weights: How much to weight news based on age
    FRESHNESS_WEIGHTS = {
        "breaking": 1.0,  # < 1 day
        "recent": 0.85,  # 1-3 days
        "this_week": 0.65,  # 3-7 days
        "this_month": 0.40,  # 7-30 days
        "last_month": 0.20,  # 30-60 days
        "last_quarter": 0.10,  # 60-90 days
        "old": 0.05,  # 90-180 days
    }
    _shared_finbert = None
    _shared_finbert_disabled = False
    _shared_finbert_error = None

    # Per-process body cache. Article bodies are immutable once published, so
    # TTL is unnecessary — we just bound size to avoid runaway memory.
    _body_cache: Dict[str, str] = {}
    _body_cache_max_entries = 2000
    _body_fetch_timeout_seconds = 12
    _body_fetch_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    @classmethod
    def _persistent_body_cache(cls):
        """Lazy-resolve the persistent NewsCache singleton."""
        try:
            from autotrade.utils.news_cache import get_cache
            return get_cache()
        except Exception:
            return None

    @classmethod
    def _remember_body(cls, url: str, body: str) -> None:
        """Update in-memory cache with bounded eviction."""
        if len(cls._body_cache) >= cls._body_cache_max_entries:
            try:
                cls._body_cache.pop(next(iter(cls._body_cache)))
            except StopIteration:
                pass
        cls._body_cache[url] = body

    @classmethod
    def _fetch_body(cls, url: str) -> str:
        """Fetch + extract article body via trafilatura.

        Two-level cache: in-memory dict (fast, lost on restart) backed by
        SQLite (article_bodies table, survives restarts). Bodies are
        immutable once published, so successful fetches are cached
        indefinitely. Failures are cached with a short cooldown.

        Returns empty string on any failure so callers can degrade gracefully.
        """
        if not url:
            return ""
        cached = cls._body_cache.get(url)
        if cached is not None:
            return cached

        # SQLite cache check before paying for the network fetch.
        persistent = cls._persistent_body_cache()
        if persistent is not None:
            row = persistent.get_article_body(url)
            if row is not None:
                body = row.get("body", "") or ""
                cls._remember_body(url, body)
                return body

        try:
            import requests
            import trafilatura
        except Exception as exc:
            logger.debug("trafilatura/requests unavailable: %s", exc)
            cls._remember_body(url, "")
            return ""

        status = "success"
        try:
            resp = requests.get(
                url,
                headers=cls._body_fetch_headers,
                timeout=cls._body_fetch_timeout_seconds,
                allow_redirects=True,
            )
            if resp.status_code != 200:
                body = ""
                status = f"http_{resp.status_code}"
            elif not resp.text:
                body = ""
                status = "empty_response"
            else:
                body = trafilatura.extract(
                    resp.text,
                    include_comments=False,
                    include_tables=False,
                    favor_precision=True,
                ) or ""
                body = body.strip()
                if not body:
                    status = "trafilatura_empty"
        except Exception as exc:
            logger.debug("body fetch failed for %s: %s", url[:80], exc)
            body = ""
            status = f"exc:{type(exc).__name__}"

        cls._remember_body(url, body)
        if persistent is not None:
            try:
                persistent.set_article_body(url, body, status=status)
            except Exception as exc:
                logger.debug("persistent body cache write failed: %s", exc)
        return body

    def __init__(self, max_news_age_days: int = 180, cache_ttl_hours: int = 4):
        """
        Initialize the news sentiment analyzer.

        Args:
            max_news_age_days: Maximum age of news to consider (default 180 = 2 quarters)
            cache_ttl_hours: How long to cache news before refreshing (default 4 hours)
        """
        self.max_news_age_days = max_news_age_days
        self.cache_ttl_hours = cache_ttl_hours
        self.finbert = None  # Lazy load to save memory
        self._cache = {}  # Cache results to avoid re-fetching
        self._price_cache = {}  # Cache price data

        # Initialize news cache
        self._news_cache = None
        try:
            from autotrade.utils.news_cache import get_cache

            self._news_cache = get_cache()
            logger.info("NewsSentimentAnalyzer: News cache enabled")
        except ImportError:
            logger.warning("News cache not available - fetching fresh each time")

        # Initialize LocalDataProvider for prices
        self._local_provider = None
        try:
            from autotrade.utils.local_data_provider import get_provider

            self._local_provider = get_provider()
        except ImportError:
            pass

    def _ensure_finbert(self):
        """Lazy load FinBERT model only when needed."""
        if self.finbert is not None:
            return True

        cls = type(self)
        if cls._shared_finbert is not None:
            self.finbert = cls._shared_finbert
            return True

        if cls._shared_finbert_disabled:
            return False

        try:
            self.finbert = get_shared_finbert_analyzer(verbose=False)
            cls._shared_finbert = self.finbert
            return True
        except Exception as exc:
            cls._shared_finbert_disabled = True
            cls._shared_finbert_error = safe_exception_text(exc)
            logger.warning(
                "NewsSentimentAnalyzer: FinBERT unavailable for this process; using headline-only fallback: %s",
                cls._shared_finbert_error,
            )
            return False

    def _build_headline_only_result(
        self, ticker: str, news_df: pd.DataFrame, error: Optional[str]
    ) -> Dict:
        """Return a neutral result using raw headlines when FinBERT is unavailable."""
        if news_df.empty:
            result = {
                "ticker": ticker,
                "news_count": 0,
                "sentiment": "neutral",
                "sentiment_score": 0.0,
                "weighted_score": 0.0,
                "confidence": 0.0,
                "positive_pct": 0.0,
                "negative_pct": 0.0,
                "neutral_pct": 0.0,
                "breaking_news": 0,
                "recent_news": 0,
                "old_news": 0,
                "headlines": [],
                "freshness_summary": "No news available",
                "error": error,
            }
            self._cache[ticker] = result
            return result

        working_df = news_df.copy()
        age_days = pd.to_numeric(working_df.get("age_days", 0), errors="coerce").fillna(
            0
        )
        working_df["age_days"] = age_days

        breaking = int((working_df["age_days"] < 1).sum())
        recent = int(
            ((working_df["age_days"] >= 1) & (working_df["age_days"] < 7)).sum()
        )
        old = int((working_df["age_days"] >= 30).sum())

        if breaking > 0:
            freshness_summary = f"{breaking} breaking news"
        elif recent > 0:
            freshness_summary = f"{recent} recent articles"
        else:
            freshness_summary = f"Mostly older news ({old} articles 30d+)"

        headlines = []
        for _, row in working_df.head(5).iterrows():
            headline = str(row.get("Title", ""))[:100]
            headlines.append(
                {
                    "headline": headline,
                    "sentiment": "unavailable",
                    "age_days": float(row.get("age_days", 0)),
                    "weighted_score": 0.0,
                }
            )

        result = {
            "ticker": ticker,
            "news_count": int(len(working_df)),
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "weighted_score": 0.0,
            "confidence": 0.0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "neutral_pct": 100.0,
            "breaking_news": breaking,
            "recent_news": recent,
            "old_news": old,
            "headlines": headlines,
            "freshness_summary": freshness_summary,
            "error": error,
        }
        self._cache[ticker] = result
        return result

    def _get_freshness_weight(self, article_date: datetime) -> Tuple[float, str]:
        """
        Get weight multiplier based on article age.

        Returns:
            Tuple of (weight, category)
        """
        now = datetime.now()
        age_days = (now - article_date).days

        if age_days < 1:
            return self.FRESHNESS_WEIGHTS["breaking"], "breaking"
        elif age_days < 3:
            return self.FRESHNESS_WEIGHTS["recent"], "recent"
        elif age_days < 7:
            return self.FRESHNESS_WEIGHTS["this_week"], "this_week"
        elif age_days < 30:
            return self.FRESHNESS_WEIGHTS["this_month"], "this_month"
        elif age_days < 60:
            return self.FRESHNESS_WEIGHTS["last_month"], "last_month"
        elif age_days < 90:
            return self.FRESHNESS_WEIGHTS["last_quarter"], "last_quarter"
        else:
            return self.FRESHNESS_WEIGHTS["old"], "old"

    def _get_price_at_date(self, ticker: str, target_date: datetime) -> Optional[float]:
        """Get closing price on or near a specific date."""
        try:
            import yfinance as yf

            cache_key = f"{ticker}_{target_date.date()}"
            if cache_key in self._price_cache:
                return self._price_cache[cache_key]

            # Get price data around that date
            start = target_date - timedelta(days=5)
            end = target_date + timedelta(days=2)

            stock = yf.Ticker(ticker)
            hist = stock.history(start=start, end=end)

            if hist.empty:
                return None

            # Find closest date
            target_ts = pd.Timestamp(target_date.date())
            hist.index = hist.index.tz_localize(None)

            # Get price on or before target date
            valid_dates = hist[hist.index <= target_ts]
            if not valid_dates.empty:
                price = float(valid_dates["Close"].iloc[-1])
            else:
                price = float(hist["Close"].iloc[0])

            self._price_cache[cache_key] = price
            return price

        except Exception:
            return None

    def _calculate_priced_in_factor(
        self, ticker: str, article_date: datetime, sentiment: str, current_price: float
    ) -> float:
        """
        Calculate if news sentiment has already been "priced in".

        Logic:
        - Positive news + stock up significantly since = priced in (reduce weight)
        - Negative news + stock down significantly since = priced in (reduce weight)
        - Positive news + stock DOWN = concerning (increase negative weight)
        - Negative news + stock UP = recovered (reduce negative weight)

        Returns:
            Factor to multiply sentiment score by (0.0 to 1.5)
        """
        price_at_news = self._get_price_at_date(ticker, article_date)

        if not price_at_news or price_at_news <= 0:
            return 1.0  # No adjustment if we can't get price

        price_change_pct = (current_price - price_at_news) / price_at_news * 100

        if sentiment == "positive":
            if price_change_pct > 15:
                # Positive news, stock up a lot - mostly priced in
                return 0.3
            elif price_change_pct > 5:
                # Positive news, stock up moderately - partially priced in
                return 0.6
            elif price_change_pct < -10:
                # Positive news but stock DOWN - concerning, might be wrong
                return 0.2  # Reduce positive impact
            else:
                return 1.0  # Normal weighting

        elif sentiment == "negative":
            if price_change_pct < -15:
                # Negative news, stock down a lot - priced in
                return 0.3
            elif price_change_pct < -5:
                # Negative news, stock down moderately - partially priced in
                return 0.6
            elif price_change_pct > 10:
                # Negative news but stock UP - recovered/resolved
                return 0.4  # Reduce negative impact
            else:
                return 1.0

        return 1.0  # Neutral

    def get_news_yfinance(self, ticker: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch news for a ticker from yfinance (NO RATE LIMITS!).
        Gets up to 2 quarters of history.

        Uses NewsCache to avoid re-fetching the same articles.

        Args:
            ticker: Stock ticker symbol
            use_cache: Whether to use cached news (default True)
        """
        ticker = ticker.upper()

        # Check cache first
        if use_cache and self._news_cache:
            cached = self._news_cache.get_cached_news(
                ticker, max_age_hours=self.cache_ttl_hours
            )
            if cached:
                logger.debug(f"{ticker}: Using cached news ({len(cached)} articles)")
                return self._cached_to_dataframe(cached)

        try:
            import yfinance as yf
            from dateutil import parser as date_parser

            stock = yf.Ticker(ticker)
            news = stock.news

            if not news:
                return pd.DataFrame()

            # Parse yfinance news format (new API structure has nested 'content')
            news_data = []
            raw_articles = []  # For caching
            cutoff_date = datetime.now() - timedelta(days=self.max_news_age_days)

            for item in news:
                # Handle new nested format
                content = item.get("content", item)

                # Get title
                title = content.get("title", "")
                if not title:
                    continue

                # Get publication date
                pub_date = None
                pub_date_str = content.get("pubDate") or content.get("displayTime")
                if pub_date_str:
                    try:
                        pub_date = date_parser.parse(pub_date_str).replace(tzinfo=None)
                    except Exception:
                        pass

                # Fallback to providerPublishTime (old format)
                if not pub_date:
                    pub_time = item.get("providerPublishTime", 0)
                    if pub_time:
                        pub_date = datetime.fromtimestamp(pub_time)

                if pub_date and pub_date >= cutoff_date:
                    # Get publisher
                    provider = content.get("provider", {})
                    publisher = (
                        provider.get("displayName", "")
                        if isinstance(provider, dict)
                        else ""
                    )

                    # Calculate freshness weight
                    freshness_weight, freshness_category = self._get_freshness_weight(
                        pub_date
                    )

                    summary = (
                        content.get("summary", "") if content.get("summary") else ""
                    )
                    relevance = assess_news_relevance(ticker, title, summary)

                    article = {
                        "Title": title,
                        "Date": pub_date,
                        "Publisher": publisher,
                        "Summary": summary,
                        "Link": content.get("link", "") or "",
                        "freshness_weight": freshness_weight,
                        "freshness_category": freshness_category,
                        "age_days": (datetime.now() - pub_date).days,
                        "relevance_label": relevance.label,
                        "relevance_score": relevance.score,
                        "relevance_reason": relevance.reason,
                        "ticker_mentions": relevance.body_ticker_mentions,
                        "company_mentions": relevance.company_mentions,
                    }
                    news_data.append(article)

                    # Prepare for caching
                    raw_articles.append(
                        {
                            "title": title,
                            "source": publisher,
                            "published_at": pub_date.isoformat() if pub_date else None,
                            "url": content.get("link", ""),
                            "content_preview": summary[:500],
                            "relevance_label": relevance.label,
                            "relevance_score": relevance.score,
                            "relevance_source": f"deterministic:{relevance.reason}",
                        }
                    )

            # Cache the fetched articles
            if raw_articles and self._news_cache:
                new_count = self._news_cache.cache_news(
                    ticker, raw_articles, source="yfinance"
                )
                if new_count > 0:
                    logger.info(f"{ticker}: Cached {new_count} new articles")

            if not news_data:
                return pd.DataFrame()

            return pd.DataFrame(news_data)

        except Exception as e:
            logger.warning(f"Failed to fetch news for {ticker}: {e}")
            return pd.DataFrame()

    def _cached_to_dataframe(self, cached_articles: List[Dict]) -> pd.DataFrame:
        """Convert cached articles to DataFrame format."""
        news_data = []
        for article in cached_articles:
            try:
                pub_date = (
                    datetime.fromisoformat(article["published_at"])
                    if article.get("published_at")
                    else None
                )
                if pub_date:
                    freshness_weight, freshness_category = self._get_freshness_weight(
                        pub_date
                    )
                    news_data.append(
                        {
                            "Title": article["title"],
                            "Date": pub_date,
                            "Publisher": article.get("source", ""),
                            "Summary": article.get("content_preview", ""),
                            "Link": article.get("url", "") or "",
                            "freshness_weight": freshness_weight,
                            "freshness_category": freshness_category,
                            "age_days": (datetime.now() - pub_date).days,
                            "cached_sentiment": article.get("sentiment"),
                            "cached_sentiment_score": article.get("sentiment_score"),
                            "relevance_label": article.get("relevance_label"),
                            "relevance_score": article.get("relevance_score"),
                            "relevance_source": article.get("relevance_source"),
                        }
                    )
            except Exception:
                continue

        return pd.DataFrame(news_data) if news_data else pd.DataFrame()

    def get_news(self, ticker: str, use_cache: bool = True) -> pd.DataFrame:
        """Get news using yfinance with caching."""
        return self.get_news_yfinance(ticker, use_cache=use_cache)

    def fetch_news_content(self, url: str, max_length: int = 3000) -> str:
        """Fetch article content as clean markdown via Fetch MCP."""
        if not url:
            return ""
        try:
            logger.debug(f"Fetching content for {url} via MCP")
            content = mcp_fetch("fetch", url=url, max_length=max_length)
            if isinstance(content, dict) and "content" in content:
                return content["content"]
            return str(content) if content else ""
        except Exception as e:
            logger.warning(f"Failed to fetch content via MCP: {e}")
            return ""

    def get_current_price(self, ticker: str) -> Optional[float]:
        """Get current price for a ticker - uses local data first."""
        # Try LocalDataProvider first (fast)
        if self._local_provider:
            try:
                price = self._local_provider.get_current_price(ticker)
                if price:
                    return price
            except Exception:
                pass

        # Fallback to yfinance
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get("currentPrice") or info.get("regularMarketPrice")
        except Exception:
            return None

    def analyze_ticker(self, ticker: str, current_price: float = None) -> Dict:
        """
        Get news and analyze sentiment for a single ticker.

        Features:
        - Time-weighted sentiment (recent news matters more)
        - Price-adjusted sentiment (priced-in factor)

        Args:
            ticker: Stock ticker symbol
            current_price: Current price (if known, saves API call)

        Returns:
            Dict with sentiment analysis results
        """
        # Check cache first
        if ticker in self._cache:
            return self._cache[ticker]

        news_df = self.get_news(ticker)
        finbert_available = self._ensure_finbert()

        # Get current price if not provided
        if current_price is None:
            current_price = self.get_current_price(ticker)

        if news_df.empty:
            result = {
                "ticker": ticker,
                "news_count": 0,
                "sentiment": "neutral",
                "sentiment_score": 0.0,
                "weighted_score": 0.0,
                "confidence": 0.0,
                "positive_pct": 0.0,
                "negative_pct": 0.0,
                "neutral_pct": 0.0,
                "breaking_news": 0,
                "recent_news": 0,
                "old_news": 0,
                "headlines": [],
                "freshness_summary": "No news available",
                "error": None,
            }
            self._cache[ticker] = result
            return result

        if not finbert_available:
            return self._build_headline_only_result(
                ticker,
                news_df,
                type(self)._shared_finbert_error,
            )

        # Map confidence strings to numeric values
        confidence_map = {"very high": 0.9, "high": 0.75, "medium": 0.5, "low": 0.25}

        # Analyze each ticker-relevant headline with freshness weighting.
        # yfinance frequently returns macro, sector, and comparison articles for
        # loosely associated symbols. Those rows remain visible in diagnostics
        # but cannot move live sentiment.
        results = []
        skipped_relevance = []

        for _, row in news_df.iterrows():
            headline = row["Title"]
            summary = str(row.get("Summary", "") or "")
            relevance_label = row.get("relevance_label")
            relevance_score = row.get("relevance_score")
            relevance_source = row.get("relevance_source")
            if not relevance_label:
                relevance = assess_news_relevance(ticker, headline, summary)
                relevance_label = relevance.label
                relevance_score = relevance.score
                relevance_source = f"deterministic:{relevance.reason}"
            if relevance_label != "relevant":
                skipped_relevance.append(
                    {
                        "headline": headline,
                        "relevance_label": relevance_label,
                        "relevance_score": float(relevance_score or 0.0),
                        "relevance_source": relevance_source,
                    }
                )
                continue
            pub_date = row["Date"]
            freshness_weight = row["freshness_weight"]
            age_days = row.get("age_days", 0)

            try:
                # Hybrid scorer: chunked FinBERT on body, escalates to LLM on
                # low confidence, falls back to title-only FinBERT if body
                # unavailable. See autotrade/analysis/news_scorer.py.
                url = str(row.get("Link", "") or "")
                body = self._fetch_body(url) if url else ""
                scored = _score_article(
                    self.finbert, ticker, headline, body=body
                )
                if scored.error and scored.method == "title_finbert":
                    # Final-fallback path also failed; skip this headline.
                    continue
                conf_str = scored.confidence or "low"
                conf_num = confidence_map.get(conf_str, 0.25)

                # Calculate priced-in factor if we have current price
                priced_in_factor = 1.0
                if current_price and age_days > 3:  # Only for older news
                    priced_in_factor = self._calculate_priced_in_factor(
                        ticker, pub_date, scored.label, current_price
                    )

                # Combined weight = freshness * confidence * priced_in_factor
                combined_weight = freshness_weight * conf_num * priced_in_factor

                results.append(
                    {
                        "headline": headline,
                        "source": row.get("Publisher", "news_sentiment"),
                        "date": pub_date,
                        "age_days": age_days,
                        "sentiment": scored.label,
                        "raw_score": float(scored.score),
                        "relevance_label": relevance_label,
                        "relevance_score": float(relevance_score or 0.0),
                        "relevance_source": relevance_source,
                        "scorer_method": scored.method,
                        "scorer_escalated": bool(scored.escalated),
                        "scorer_detail": scored.detail,
                        "scorer_latency_ms": float(scored.latency_ms),
                        "confidence": conf_num,
                        "freshness_weight": freshness_weight,
                        "priced_in_factor": priced_in_factor,
                        "combined_weight": combined_weight,
                        "weighted_score": float(scored.score)
                        * combined_weight
                        * (
                            1
                            if scored.label == "positive"
                            else -1
                            if scored.label == "negative"
                            else 0
                        ),
                    }
                )
            except Exception as _scorer_exc:
                logger.debug(
                    "news_scorer failed for %s/%s: %s",
                    ticker, headline[:60], _scorer_exc,
                )
                continue

        if not results:
            result = {
                "ticker": ticker,
                "news_count": int(len(news_df)),
                "relevant_news_count": 0,
                "skipped_relevance_count": int(len(skipped_relevance)),
                "sentiment": "neutral",
                "sentiment_score": 0.0,
                "weighted_score": 0.0,
                "confidence": 0.0,
                "positive_pct": 0.0,
                "negative_pct": 0.0,
                "neutral_pct": 0.0,
                "breaking_news": 0,
                "recent_news": 0,
                "old_news": 0,
                "headlines": [],
                "top_headlines": [],
                "skipped_relevance": skipped_relevance[:5],
                "freshness_summary": "Analysis failed",
                "error": "No relevant news results",
            }
            self._cache[ticker] = result
            return result

        # Aggregate with weighting
        total = len(results)
        positive_count = sum(1 for r in results if r["sentiment"] == "positive")
        negative_count = sum(1 for r in results if r["sentiment"] == "negative")
        neutral_count = sum(1 for r in results if r["sentiment"] == "neutral")

        # Count by freshness
        breaking = sum(1 for r in results if r["age_days"] < 1)
        recent = sum(1 for r in results if 1 <= r["age_days"] < 7)
        old = sum(1 for r in results if r["age_days"] >= 30)

        # Raw (unweighted) sentiment score
        raw_score = (
            sum(
                r["raw_score"]
                * (
                    1
                    if r["sentiment"] == "positive"
                    else -1
                    if r["sentiment"] == "negative"
                    else 0
                )
                for r in results
            )
            / total
        )

        # Weighted sentiment score (freshness + confidence + priced-in)
        total_weight = sum(r["combined_weight"] for r in results)
        if total_weight > 0:
            weighted_score = sum(r["weighted_score"] for r in results) / total_weight
        else:
            weighted_score = 0.0

        avg_confidence = sum(r["confidence"] for r in results) / total

        # Determine overall sentiment (use weighted score)
        if weighted_score > 0.15:
            overall_sentiment = "positive"
        elif weighted_score < -0.15:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"

        # Create freshness summary
        if breaking > 0:
            freshness_summary = f"{breaking} breaking news"
        elif recent > 0:
            freshness_summary = f"{recent} recent articles"
        else:
            freshness_summary = f"Mostly older news ({old} articles 30d+)"

        # Sort headlines by weighted importance
        sorted_headlines = sorted(
            results, key=lambda x: abs(x["weighted_score"]), reverse=True
        )

        result = {
            "ticker": ticker,
            "news_count": int(len(news_df)),
            "relevant_news_count": total,
            "skipped_relevance_count": int(len(skipped_relevance)),
            "sentiment": overall_sentiment,
            "sentiment_score": round(raw_score, 4),
            "weighted_score": round(weighted_score, 4),  # This is the one to use!
            "confidence": round(avg_confidence, 4),
            "positive_pct": round(positive_count / total * 100, 1),
            "negative_pct": round(negative_count / total * 100, 1),
            "neutral_pct": round(neutral_count / total * 100, 1),
            "breaking_news": breaking,
            "recent_news": recent,
            "old_news": old,
            "headlines": [
                {
                    "headline": h["headline"][:100],
                    "sentiment": h["sentiment"],
                    "age_days": h["age_days"],
                    "relevance_label": h.get("relevance_label"),
                    "relevance_score": round(float(h.get("relevance_score", 0.0)), 3),
                    "weighted_score": round(h["weighted_score"], 3),
                }
                for h in sorted_headlines[:5]
            ],
            "top_headlines": [
                {
                    "headline": h["headline"][:100],
                    "source": h.get("source", "news_sentiment"),
                    "sentiment": h["sentiment"],
                    "age_days": h["age_days"],
                    "relevance_label": h.get("relevance_label"),
                    "relevance_score": round(float(h.get("relevance_score", 0.0)), 3),
                    "weighted_score": round(h["weighted_score"], 3),
                }
                for h in sorted_headlines[:5]
            ],
            "skipped_relevance": skipped_relevance[:5],
            "freshness_summary": freshness_summary,
            "error": None,
        }
        self._cache[ticker] = result
        return result

    def analyze_tickers(self, tickers: List[str]) -> pd.DataFrame:
        """Analyze sentiment for multiple tickers."""
        results = []
        for ticker in tickers:
            result = self.analyze_ticker(ticker)
            results.append(result)

        df = pd.DataFrame(results)
        return df.sort_values("weighted_score", ascending=False)

    def get_sentiment_signal(self, ticker: str) -> Tuple[str, float, str]:
        """
        Get a trading signal based on news sentiment.
        Uses weighted_score which accounts for freshness and priced-in factor.

        Returns:
            Tuple of (signal, score, reason)
        """
        result = self.analyze_ticker(ticker)

        score = result["weighted_score"]
        news_count = result["news_count"]
        confidence = result["confidence"]
        breaking = result.get("breaking_news", 0)

        # Need sufficient news for confidence
        if news_count < 2:
            return ("neutral", score, f"Insufficient news ({news_count} articles)")

        if confidence < 0.35:
            return ("neutral", score, f"Low confidence ({confidence:.1%})")

        # Determine signal - weight breaking news more
        if score > 0.35 and result["positive_pct"] >= 45:
            qualifier = "BREAKING" if breaking > 0 else "Strong"
            return (
                "bullish",
                score,
                f"{qualifier} positive sentiment ({result['positive_pct']:.0f}% pos)",
            )
        elif score < -0.35 and result["negative_pct"] >= 45:
            qualifier = "BREAKING" if breaking > 0 else "Strong"
            return (
                "bearish",
                score,
                f"{qualifier} negative sentiment ({result['negative_pct']:.0f}% neg)",
            )
        elif score > 0.2:
            return (
                "neutral",
                score,
                f"Mildly positive ({result['freshness_summary']})",
            )
        elif score < -0.2:
            return (
                "neutral",
                score,
                f"Mildly negative ({result['freshness_summary']})",
            )
        else:
            return (
                "neutral",
                score,
                f"Mixed sentiment ({result['freshness_summary']})",
            )

    def clear_cache(self):
        """Clear all caches."""
        self._cache = {}
        self._price_cache = {}


# ============================================================================
# Module-level convenience functions
# ============================================================================

_default_analyzer = None


def get_analyzer() -> NewsSentimentAnalyzer:
    """Get or create the default news sentiment analyzer."""
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = NewsSentimentAnalyzer()
    return _default_analyzer


def analyze_news(ticker: str) -> Dict:
    """
    Analyze news sentiment for a ticker.

    This is the main entry point for news analysis.

    Args:
        ticker: Stock symbol (e.g., 'AAPL')

    Returns:
        Dict with sentiment analysis including:
        - signal: 'bullish', 'bearish', or 'neutral'
        - score: Weighted sentiment score (-1 to +1)
        - news_count: Number of news articles analyzed
        - headlines: Top headlines with sentiment
        - reason: Human-readable explanation
    """
    analyzer = get_analyzer()
    result = analyzer.analyze_ticker(ticker)
    signal, score, reason = analyzer.get_sentiment_signal(ticker)

    return {
        "ticker": ticker,
        "signal": signal,
        "score": score,
        "reason": reason,
        "news_count": result.get("news_count", 0),
        "weighted_score": result.get("weighted_score", 0),
        "headlines": result.get("headlines", []),
        "breaking_news": result.get("breaking_news", 0),
        "recent_news": result.get("recent_news", 0),
        "freshness_summary": result.get("freshness_summary", ""),
    }


def demo():
    """Demo the news sentiment analyzer with freshness weighting."""
    logger.info("=" * 70)
    logger.info("News Sentiment Analyzer v2 - Time-Weighted + Priced-In Analysis")
    logger.info("Using: yfinance (news) + FinBERT (sentiment)")
    logger.info("=" * 70)

    analyzer = NewsSentimentAnalyzer(max_news_age_days=90)  # 1 quarter for demo

    test_tickers = ["NVDA", "TSLA", "AAPL", "LMND", "FORM"]

    for ticker in test_tickers:
        logger.info("%s", "=" * 50)
        logger.info("Analyzing %s...", ticker)

        result = analyzer.analyze_ticker(ticker)
        signal, score, reason = analyzer.get_sentiment_signal(ticker)

        logger.info("Signal: %s", signal.upper())
        logger.info(
            "Weighted Score: %+0.3f (raw: %+0.3f)",
            result["weighted_score"],
            result["sentiment_score"],
        )
        logger.info("Reason: %s", reason)
        logger.info(
            "News: %s total | %s breaking | %s recent",
            result["news_count"],
            result["breaking_news"],
            result["recent_news"],
        )
        logger.info("%s", result["freshness_summary"])

        if result["headlines"]:
            logger.info("Top Headlines (by importance):")
            for h in result["headlines"][:3]:
                age_str = f"{h['age_days']}d" if h["age_days"] > 0 else "today"
                tag = (
                    "[+]"
                    if h["sentiment"] == "positive"
                    else "[-]"
                    if h["sentiment"] == "negative"
                    else "[.]"
                )
                logger.info("  %s (%s) %s...", tag, age_str, h["headline"][:60])


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info(
        "news_sentiment.py loaded; use pytest/CLI for tests."
    )
