"""
News Cache - Persistent cache for news articles to avoid redundant API calls.

Features:
- SQLite-backed persistent storage
- Content-based deduplication (hash of title + source)
- TTL-based expiration (default 24 hours for fresh news)
- Symbol-based indexing for fast lookups
- Tracks article age to avoid re-fetching same articles

Usage:
    cache = NewsCache()

    # Check if we have fresh news
    cached = cache.get_cached_news('AAPL', max_age_hours=4)
    if cached:
        return cached

    # Fetch new news and cache
    articles = fetch_news_from_api('AAPL')
    cache.cache_news('AAPL', articles)
"""

import os
import sqlite3
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger("AutoTrade.NewsCache")

# Default cache location
CACHE_DIR = Path(
    os.environ.get("AUTOTRADE_CACHE_DIR", Path(__file__).parent.parent.parent / "data")
)
NEWS_CACHE_DB = CACHE_DIR / "news_cache.db"


@dataclass
class CachedArticle:
    """Represents a cached news article."""

    article_hash: str
    symbol: str
    title: str
    source: str
    url: str
    published_at: str  # ISO format
    content_preview: str  # First 500 chars
    sentiment: Optional[str] = None  # positive/negative/neutral
    sentiment_score: Optional[float] = None
    cached_at: str = None  # When we cached it

    def __post_init__(self):
        if self.cached_at is None:
            self.cached_at = datetime.now().isoformat()


class NewsCache:
    """
    Persistent cache for news articles.

    Avoids re-fetching the same news articles by:
    1. Content-based deduplication (hash of title + source)
    2. Time-based caching (only fetch new articles since last check)
    3. Symbol-based indexing for fast lookups
    """

    def __init__(self, db_path: Optional[Path] = None, ttl_hours: int = 24):
        """
        Initialize news cache.

        Args:
            db_path: Path to SQLite database (default: data/news_cache.db)
            ttl_hours: Time-to-live for cached articles in hours
        """
        self.db_path = db_path or NEWS_CACHE_DB
        self.ttl_hours = ttl_hours

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

        # In-memory cache for hot data
        self._memory_cache: Dict[str, List[Dict]] = {}
        self._last_fetch: Dict[str, datetime] = {}

        logger.info(f"NewsCache initialized at {self.db_path}")

    def _init_db(self):
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                article_hash TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                title TEXT NOT NULL,
                source TEXT,
                url TEXT,
                published_at TEXT,
                content_preview TEXT,
                sentiment TEXT,
                sentiment_score REAL,
                relevance_label TEXT,
                relevance_score REAL,
                relevance_source TEXT,
                cached_at TEXT NOT NULL,
                last_accessed TEXT
            )
        """)

        self._ensure_articles_schema(cursor)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol_cached 
            ON articles(symbol, cached_at)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_published 
            ON articles(published_at)
        """)

        # Track last fetch time per symbol
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fetch_log (
                symbol TEXT PRIMARY KEY,
                last_fetch TEXT NOT NULL,
                article_count INTEGER,
                source TEXT
            )
        """)

        # Persistent article body cache. Bodies are immutable once published,
        # so successful fetches are cached indefinitely. Failures are cached
        # with a short cooldown so we don't hammer broken URLs every cycle.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS article_bodies (
                url TEXT PRIMARY KEY,
                body_content TEXT,
                body_chars INTEGER,
                fetch_status TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                last_used_at TEXT
            )
        """)

        conn.commit()
        conn.close()

    # ---------- Article body persistence ----------

    BODY_FAILURE_COOLDOWN_HOURS = 6

    def get_article_body(self, url: str) -> Optional[Dict[str, Any]]:
        """Look up a previously fetched article body.

        Returns:
            None on miss or stale failure (caller should re-fetch).
            Dict with keys body, status, fetched_at on hit. body is "" when
            the previous fetch failed but is still within cooldown.
        """
        if not url:
            return None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT body_content, fetch_status, fetched_at
                FROM article_bodies
                WHERE url = ?
                """,
                (url,),
            )
            row = cursor.fetchone()
            if not row:
                conn.close()
                return None
            body, status, fetched_at = row[0] or "", row[1], row[2]
            # Touch last_used_at so future eviction can prefer cold rows.
            cursor.execute(
                "UPDATE article_bodies SET last_used_at = ? WHERE url = ?",
                (datetime.now().isoformat(), url),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"article body cache read failed: {e}")
            return None

        if status == "success" and body:
            return {"body": body, "status": status, "fetched_at": fetched_at}
        # Failure: respect cooldown so transient errors are retried but
        # permanent 404s don't get re-fetched every cycle.
        try:
            age = datetime.now() - datetime.fromisoformat(fetched_at)
            if age < timedelta(hours=self.BODY_FAILURE_COOLDOWN_HOURS):
                return {"body": "", "status": status, "fetched_at": fetched_at}
        except Exception:
            pass
        return None  # cooldown elapsed; caller should re-fetch

    def set_article_body(
        self, url: str, body: str, status: str = "success"
    ) -> None:
        """Persist an article body fetch (success or failure)."""
        if not url:
            return
        body = body or ""
        now = datetime.now().isoformat()
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO article_bodies
                    (url, body_content, body_chars, fetch_status, fetched_at, last_used_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    body_content = excluded.body_content,
                    body_chars = excluded.body_chars,
                    fetch_status = excluded.fetch_status,
                    fetched_at = excluded.fetched_at,
                    last_used_at = excluded.last_used_at
                """,
                (url, body, len(body), status, now, now),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"article body cache write failed: {e}")

    def _ensure_articles_schema(self, cursor) -> None:
        """Backfill columns added after older local cache databases were created."""
        cursor.execute("PRAGMA table_info(articles)")
        existing = {str(row[1]) for row in cursor.fetchall()}
        required_columns = {
            "relevance_label": "TEXT",
            "relevance_score": "REAL",
            "relevance_source": "TEXT",
        }
        for column, column_type in required_columns.items():
            if column not in existing:
                cursor.execute(
                    f"ALTER TABLE articles ADD COLUMN {column} {column_type}"
                )

    @staticmethod
    def _hash_article(title: str, source: str, symbol: str = "") -> str:
        """Generate unique hash for an article based on symbol + title + source."""
        if str(symbol).strip():
            content = (
                f"{str(symbol).upper().strip()}|"
                f"{title.lower().strip()}|"
                f"{source.lower().strip()}"
            )
        else:
            content = f"{title.lower().strip()}|{source.lower().strip()}"
        return hashlib.md5(content.encode()).hexdigest()

    def get_cached_news(
        self, symbol: str, max_age_hours: Optional[int] = None, limit: int = 50
    ) -> Optional[List[Dict]]:
        """
        Get cached news for a symbol if fresh enough.

        Args:
            symbol: Stock ticker symbol
            max_age_hours: Maximum age of cached data (default: self.ttl_hours)
            limit: Maximum number of articles to return

        Returns:
            List of cached articles or None if cache miss/stale
        """
        symbol = symbol.upper()
        max_age = max_age_hours or self.ttl_hours

        # Check memory cache first
        if symbol in self._memory_cache:
            last_fetch = self._last_fetch.get(symbol)
            if last_fetch and datetime.now() - last_fetch < timedelta(hours=max_age):
                logger.debug(f"Cache HIT (memory) for {symbol}")
                return self._memory_cache[symbol][:limit]

        # Check SQLite cache
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check last fetch time
            cursor.execute(
                "SELECT last_fetch FROM fetch_log WHERE symbol = ?", (symbol,)
            )
            row = cursor.fetchone()

            if row:
                last_fetch = datetime.fromisoformat(row[0])
                if datetime.now() - last_fetch < timedelta(hours=max_age):
                    # Cache is fresh, get articles
                    cursor.execute(
                        """
                        SELECT title, source, url, published_at,
                               content_preview, sentiment, sentiment_score,
                               relevance_label, relevance_score, relevance_source
                        FROM articles
                        WHERE symbol = ?
                        ORDER BY published_at DESC
                        LIMIT ?
                    """,
                        (symbol, limit),
                    )

                    articles = []
                    for row in cursor.fetchall():
                        articles.append(
                            {
                                "title": row[0],
                                "source": row[1],
                                "url": row[2],
                                "published_at": row[3],
                                "content_preview": row[4],
                                "sentiment": row[5],
                                "sentiment_score": row[6],
                                "relevance_label": row[7],
                                "relevance_score": row[8],
                                "relevance_source": row[9],
                            }
                        )

                    conn.close()

                    # Update memory cache
                    self._memory_cache[symbol] = articles
                    self._last_fetch[symbol] = last_fetch

                    if articles:
                        logger.debug(
                            f"Cache HIT (SQLite) for {symbol}: {len(articles)} articles"
                        )
                    else:
                        logger.debug(
                            f"Cache HIT (SQLite) for {symbol}: 0 articles (verified empty)"
                        )
                    return articles

            conn.close()

        except Exception as e:
            logger.warning(f"Cache read error for {symbol}: {e}")

        logger.debug(f"Cache MISS for {symbol}")
        return None

    def cache_news(
        self, symbol: str, articles: List[Dict], source: str = "unknown"
    ) -> int:
        """
        Cache news articles for a symbol.

        Args:
            symbol: Stock ticker symbol
            articles: List of article dicts with title, source, url, etc.
            source: Data source (yfinance, searxng, etc.)

        Returns:
            Number of new articles cached (excluding duplicates)
        """
        symbol = symbol.upper()
        new_count = 0

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            now = datetime.now().isoformat()

            for article in articles:
                title = article.get("title", "")
                art_source = article.get("source", article.get("publisher", "unknown"))

                if not title:
                    continue

                article_hash = self._hash_article(title, art_source, symbol)

                # Check if already exists
                cursor.execute(
                    """
                    SELECT article_hash FROM articles
                    WHERE article_hash = ?
                       OR (
                           symbol = ?
                           AND LOWER(title) = LOWER(?)
                           AND LOWER(COALESCE(source, '')) = LOWER(?)
                       )
                    """,
                    (article_hash, symbol, title, art_source),
                )

                existing = cursor.fetchone()
                if existing:
                    # Already cached, update last_accessed
                    cursor.execute(
                        """
                        UPDATE articles
                        SET last_accessed = ?,
                            relevance_label = COALESCE(?, relevance_label),
                            relevance_score = COALESCE(?, relevance_score),
                            relevance_source = COALESCE(?, relevance_source),
                            sentiment = COALESCE(?, sentiment),
                            sentiment_score = COALESCE(?, sentiment_score)
                        WHERE article_hash = ?
                        """,
                        (
                            now,
                            article.get("relevance_label"),
                            article.get("relevance_score"),
                            article.get("relevance_source"),
                            article.get("sentiment"),
                            article.get("sentiment_score"),
                            existing[0],
                        ),
                    )
                    continue

                # Insert new article
                cursor.execute(
                    """
                    INSERT INTO articles 
                    (article_hash, symbol, title, source, url, published_at,
                     content_preview, sentiment, sentiment_score, relevance_label,
                     relevance_score, relevance_source, cached_at, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        article_hash,
                        symbol,
                        title,
                        art_source,
                        article.get("url", article.get("link", "")),
                        article.get(
                            "published_at", article.get("providerPublishTime", now)
                        ),
                        article.get("content_preview", article.get("summary", ""))[
                            :500
                        ],
                        article.get("sentiment"),
                        article.get("sentiment_score"),
                        article.get("relevance_label"),
                        article.get("relevance_score"),
                        article.get("relevance_source"),
                        now,
                        now,
                    ),
                )
                new_count += 1

            # Update fetch log
            cursor.execute(
                """
                INSERT OR REPLACE INTO fetch_log (symbol, last_fetch, article_count, source)
                VALUES (?, ?, ?, ?)
            """,
                (symbol, now, len(articles), source),
            )

            conn.commit()
            conn.close()

            # Update memory cache
            self._memory_cache[symbol] = articles
            self._last_fetch[symbol] = datetime.now()

            if new_count > 0:
                logger.info(
                    f"Cached {new_count} new articles for {symbol} (from {source})"
                )
            else:
                logger.debug(f"No new articles for {symbol} (all duplicates)")

            return new_count

        except Exception as e:
            logger.error(f"Cache write error for {symbol}: {e}")
            return 0

    def update_sentiment(
        self, symbol: str, article_hash: str, sentiment: str, score: float
    ):
        """Update sentiment for a cached article."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE articles 
                SET sentiment = ?, sentiment_score = ?
                WHERE article_hash = ?
            """,
                (sentiment, score, article_hash),
            )

            conn.commit()
            conn.close()

            # Clear memory cache to ensure next read gets updated values
            if symbol.upper() in self._memory_cache:
                del self._memory_cache[symbol.upper()]
                del self._last_fetch[symbol.upper()]

        except Exception as e:
            logger.warning(f"Failed to update sentiment: {e}")

    def update_sentiment_by_fields(
        self, title: str, source: str, sentiment: str, score: float
    ):
        """Update sentiment for an article identified by title and source."""
        article_hash = self._hash_article(title, source)
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE articles 
                SET sentiment = ?, sentiment_score = ?
                WHERE article_hash = ?
                   OR (LOWER(title) = LOWER(?) AND LOWER(COALESCE(source, '')) = LOWER(?))
            """,
                (sentiment, score, article_hash, title, source),
            )
            conn.commit()
            conn.close()
            # Clear memory cache since we don't know which symbol this article belongs to
            self._memory_cache.clear()
            self._last_fetch.clear()
        except Exception as e:
            logger.warning(f"Failed to update sentiment by fields: {e}")

    def update_relevance_by_fields(
        self,
        title: str,
        source: str,
        label: str,
        score: float,
        relevance_source: str = "local",
    ):
        """Update relevance for an article identified by title and source."""
        article_hash = self._hash_article(title, source)
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE articles 
                SET relevance_label = ?, relevance_score = ?, relevance_source = ?
                WHERE article_hash = ?
                   OR (LOWER(title) = LOWER(?) AND LOWER(COALESCE(source, '')) = LOWER(?))
            """,
                (label, score, relevance_source, article_hash, title, source),
            )
            conn.commit()
            conn.close()
            # Clear memory cache since we don't know which symbol this article belongs to
            self._memory_cache.clear()
            self._last_fetch.clear()
        except Exception as e:
            logger.warning(f"Failed to update relevance by fields: {e}")

    def get_uncached_symbols(
        self, symbols: List[str], max_age_hours: Optional[int] = None
    ) -> List[str]:
        """
        Get list of symbols that need fresh news fetch.

        Args:
            symbols: List of symbols to check
            max_age_hours: Maximum age for cache validity

        Returns:
            List of symbols that need refresh
        """
        max_age = max_age_hours or self.ttl_hours
        needs_refresh = []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for symbol in symbols:
                symbol = symbol.upper()
                cursor.execute(
                    "SELECT last_fetch FROM fetch_log WHERE symbol = ?", (symbol,)
                )
                row = cursor.fetchone()

                if not row:
                    needs_refresh.append(symbol)
                else:
                    last_fetch = datetime.fromisoformat(row[0])
                    if datetime.now() - last_fetch >= timedelta(hours=max_age):
                        needs_refresh.append(symbol)

            conn.close()

        except Exception as e:
            logger.warning(f"Error checking cache: {e}")
            return symbols  # If error, refresh all

        logger.info(
            f"News cache check: {len(symbols) - len(needs_refresh)}/{len(symbols)} cached, "
            f"{len(needs_refresh)} need refresh"
        )
        return needs_refresh

    def cleanup_old_articles(self, days: int = 30):
        """Remove articles older than specified days."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cutoff = (datetime.now() - timedelta(days=days)).isoformat()

            cursor.execute("DELETE FROM articles WHERE cached_at < ?", (cutoff,))

            deleted = cursor.rowcount

            conn.commit()
            conn.close()

            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old articles (>{days} days)")

            return deleted

        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM articles")
            total_articles = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM articles")
            unique_symbols = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM fetch_log")
            symbols_tracked = cursor.fetchone()[0]

            cursor.execute("""
                SELECT symbol, COUNT(*) as cnt 
                FROM articles 
                GROUP BY symbol 
                ORDER BY cnt DESC 
                LIMIT 5
            """)
            top_symbols = cursor.fetchall()

            conn.close()

            return {
                "total_articles": total_articles,
                "unique_symbols": unique_symbols,
                "symbols_tracked": symbols_tracked,
                "top_symbols": dict(top_symbols),
                "memory_cache_size": len(self._memory_cache),
                "db_path": str(self.db_path),
            }

        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {"error": str(e)}


# ============================================================================
# Module-level convenience functions
# ============================================================================

_cache: Optional[NewsCache] = None


def get_cache() -> NewsCache:
    """Get or create singleton NewsCache instance."""
    global _cache
    if _cache is None:
        _cache = NewsCache()
    return _cache


def get_cached_news(symbol: str, max_age_hours: int = 4) -> Optional[List[Dict]]:
    """Convenience function to get cached news."""
    return get_cache().get_cached_news(symbol, max_age_hours)


def cache_news(symbol: str, articles: List[Dict], source: str = "unknown") -> int:
    """Convenience function to cache news."""
    return get_cache().cache_news(symbol, articles, source)


def needs_refresh(symbols: List[str], max_age_hours: int = 4) -> List[str]:
    """Get symbols that need news refresh."""
    return get_cache().get_uncached_symbols(symbols, max_age_hours)
