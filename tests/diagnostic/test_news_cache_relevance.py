from datetime import datetime
import sqlite3

from autotrade.utils.news_cache import NewsCache


def test_news_cache_relevance_and_sentiment_updates(tmp_path):
    db_path = tmp_path / "news_cache.db"
    cache = NewsCache(db_path=db_path, ttl_hours=24)

    articles = [
        {
            "title": "Test earnings beat",
            "source": "UnitTest",
            "url": "https://example.com",
            "published_at": datetime.now().isoformat(),
            "content_preview": "Company beats earnings guidance.",
        }
    ]

    cache.cache_news("TEST", articles, source="yfinance")
    cache.update_sentiment_by_fields("Test earnings beat", "UnitTest", "positive", 0.6)
    cache.update_relevance_by_fields(
        "Test earnings beat", "UnitTest", "catalyst", 0.9, "local"
    )

    cached = cache.get_cached_news("TEST", max_age_hours=1, limit=5)
    assert cached, "Expected cached articles"
    row = cached[0]
    assert row["sentiment"] == "positive"
    assert abs(float(row["sentiment_score"]) - 0.6) < 1e-6
    assert row["relevance_label"] == "catalyst"
    assert abs(float(row["relevance_score"]) - 0.9) < 1e-6


def test_news_cache_keeps_same_headline_scoped_by_symbol(tmp_path):
    db_path = tmp_path / "news_cache.db"
    cache = NewsCache(db_path=db_path, ttl_hours=24)
    published_at = datetime.now().isoformat()
    article = {
        "title": "Shared headline",
        "source": "UnitTest",
        "url": "https://example.com/shared",
        "published_at": published_at,
        "content_preview": "A general sector headline.",
    }

    assert cache.cache_news("VZ", [article], source="unit") == 1
    assert cache.cache_news("AVAH", [article], source="unit") == 1

    vz_cached = cache.get_cached_news("VZ", max_age_hours=1, limit=5)
    avah_cached = cache.get_cached_news("AVAH", max_age_hours=1, limit=5)
    assert vz_cached and avah_cached
    assert vz_cached[0]["title"] == "Shared headline"
    assert avah_cached[0]["title"] == "Shared headline"

    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM articles WHERE title = ?",
            ("Shared headline",),
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 2


def test_news_cache_migrates_legacy_schema_without_relevance_columns(tmp_path):
    db_path = tmp_path / "legacy_news_cache.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE articles (
            article_hash TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            title TEXT NOT NULL,
            source TEXT,
            url TEXT,
            published_at TEXT,
            content_preview TEXT,
            sentiment TEXT,
            sentiment_score REAL,
            cached_at TEXT NOT NULL,
            last_accessed TEXT
        )
        """
    )
    now = datetime.now().isoformat()
    conn.execute(
        """
        INSERT INTO articles (
            article_hash, symbol, title, source, url, published_at,
            content_preview, sentiment, sentiment_score, cached_at, last_accessed
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            NewsCache._hash_article("Legacy article", "UnitTest"),
            "TEST",
            "Legacy article",
            "UnitTest",
            "https://example.com/legacy",
            now,
            "Legacy cache row",
            "neutral",
            0.0,
            now,
            now,
        ),
    )
    conn.execute(
        """
        CREATE TABLE fetch_log (
            symbol TEXT PRIMARY KEY,
            last_fetch TEXT NOT NULL,
            article_count INTEGER,
            source TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO fetch_log (symbol, last_fetch, article_count, source) VALUES (?, ?, ?, ?)",
        ("TEST", now, 1, "legacy"),
    )
    conn.commit()
    conn.close()

    cache = NewsCache(db_path=db_path, ttl_hours=24)

    cached = cache.get_cached_news("TEST", max_age_hours=1, limit=5)
    assert cached
    row = cached[0]
    assert row["title"] == "Legacy article"
    assert row["relevance_label"] is None
    assert row["relevance_score"] is None
    assert row["relevance_source"] is None
