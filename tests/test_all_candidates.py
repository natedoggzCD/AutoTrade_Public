"""Tests for lessons_screener candidate retrieval."""

from __future__ import annotations

import pytest

import os

from autotrade.signals.lessons_screener import get_entry_candidates


def test_lessons_screener_returns_candidates():
    if os.getenv("AUTOTRADE_RUN_HEAVY_TESTS") != "1":
        pytest.skip("Skipping heavy screener test; set AUTOTRADE_RUN_HEAVY_TESTS=1 to run.")
    candidates = get_entry_candidates(max_candidates=200)
    assert isinstance(candidates, list)
    if not candidates:
        pytest.skip("No candidates returned; data source may be empty or unavailable")
    assert all(
        ("ticker" in c or "symbol" in c) for c in candidates if isinstance(c, dict)
    )
