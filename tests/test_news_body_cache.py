"""Tests for the persistent article body cache.

Covers:
- Schema migration: existing DB without article_bodies table gets it created
- Successful body persists across NewsCache instances (simulates restart)
- Failed fetches respect the cooldown to avoid re-hammering bad URLs
- Cooldown elapsed -> cache returns None so caller re-fetches
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from autotrade.utils.news_cache import NewsCache


@pytest.fixture
def fresh_cache(tmp_path: Path) -> NewsCache:
    db = tmp_path / "news_cache.db"
    return NewsCache(db_path=db)


def test_article_bodies_table_created(fresh_cache: NewsCache):
    conn = sqlite3.connect(fresh_cache.db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='article_bodies'")
    assert cur.fetchone() is not None
    cur.execute("PRAGMA table_info(article_bodies)")
    cols = {row[1] for row in cur.fetchall()}
    assert {"url", "body_content", "fetch_status", "fetched_at"}.issubset(cols)
    conn.close()


def test_set_then_get_returns_body(fresh_cache: NewsCache):
    url = "https://example.com/article-1"
    fresh_cache.set_article_body(url, "the article body" * 50, status="success")
    row = fresh_cache.get_article_body(url)
    assert row is not None
    assert row["status"] == "success"
    assert "article body" in row["body"]


def test_body_survives_restart_simulation(tmp_path: Path):
    """Body persists across NewsCache instances backed by the same DB file."""
    db = tmp_path / "news_cache.db"
    first = NewsCache(db_path=db)
    first.set_article_body("https://example.com/x", "persistent body content", status="success")
    # New instance == process restart.
    second = NewsCache(db_path=db)
    row = second.get_article_body("https://example.com/x")
    assert row is not None
    assert row["body"] == "persistent body content"


def test_failed_fetch_returns_empty_within_cooldown(fresh_cache: NewsCache):
    """A cached failure should short-circuit re-fetch attempts."""
    url = "https://example.com/broken"
    fresh_cache.set_article_body(url, "", status="http_404")
    row = fresh_cache.get_article_body(url)
    assert row is not None
    assert row["body"] == ""
    assert row["status"] == "http_404"


def test_failed_fetch_expires_after_cooldown(fresh_cache: NewsCache):
    """Once cooldown elapses, get returns None so caller re-fetches."""
    url = "https://example.com/transient"
    # Insert with a stale fetched_at timestamp.
    stale = (datetime.now() - timedelta(hours=NewsCache.BODY_FAILURE_COOLDOWN_HOURS + 1)).isoformat()
    conn = sqlite3.connect(fresh_cache.db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO article_bodies (url, body_content, body_chars, fetch_status, fetched_at, last_used_at)
        VALUES (?, '', 0, ?, ?, ?)
        """,
        (url, "timeout", stale, stale),
    )
    conn.commit()
    conn.close()
    row = fresh_cache.get_article_body(url)
    assert row is None  # cooldown elapsed -> re-fetch allowed


def test_get_missing_url_returns_none(fresh_cache: NewsCache):
    assert fresh_cache.get_article_body("https://example.com/never-seen") is None
    assert fresh_cache.get_article_body("") is None


def test_legacy_db_without_article_bodies_gets_migrated(tmp_path: Path):
    """An existing DB created before the bodies table was added must
    pick up the new table when a new NewsCache instance opens it."""
    db = tmp_path / "legacy_cache.db"
    # Build a legacy DB with only the articles + fetch_log tables.
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    # Schema mirrors the pre-relevance, pre-body-cache era: canonical columns
    # only, no relevance_* fields, no article_bodies table.
    cur.execute(
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
    cur.execute(
        """
        CREATE TABLE fetch_log (
            symbol TEXT PRIMARY KEY,
            last_fetch TEXT NOT NULL,
            article_count INTEGER,
            source TEXT
        )
        """
    )
    conn.commit()
    conn.close()

    # Opening with NewsCache should create the bodies table.
    NewsCache(db_path=db)
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='article_bodies'")
    assert cur.fetchone() is not None
    conn.close()
