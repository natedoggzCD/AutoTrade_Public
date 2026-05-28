"""Tests for the intraday_analysis snapshot freshness guard.

day-manager 2026-05-19: 2026-05-19 session ran on a 6.5-hour-stale snapshot
because the writer was silently failing and consumers had no way to detect
staleness. This module pins the freshness-helper behavior so a future
refactor cannot regress it.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest


def _DM():
    """Lazy import so test collection doesn't load the whole module graph."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from autotrade.core.day_manager import DayManager

    return DayManager


def _make_payload(ts_iso):
    return {
        "timestamp": ts_iso,
        "market_context": {
            "spy_pct": -0.5,
            "vix": 17.8,
            "breadth_pct": 53.0,
            "regime_label": "DISPERSION",
        },
    }


def _write_snapshot(tmp_dir: Path, payload: dict) -> Path:
    path = tmp_dir / "data" / "intraday_analysis.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_age_minutes_returns_none_when_file_missing(tmp_path, monkeypatch):
    DayManager = _DM()
    monkeypatch.chdir(tmp_path)
    dm = DayManager.__new__(DayManager)
    assert dm._intraday_analysis_age_minutes() is None


def test_age_minutes_returns_minutes_for_fresh_snapshot(tmp_path, monkeypatch):
    DayManager = _DM()
    monkeypatch.chdir(tmp_path)
    now = datetime.now()
    _write_snapshot(tmp_path, _make_payload(now.isoformat()))
    dm = DayManager.__new__(DayManager)
    age = dm._intraday_analysis_age_minutes()
    assert age is not None
    assert age < 1.0  # less than a minute old


def test_age_minutes_returns_large_value_for_stale_snapshot(tmp_path, monkeypatch):
    DayManager = _DM()
    monkeypatch.chdir(tmp_path)
    ts = (datetime.now() - timedelta(hours=6, minutes=30)).isoformat()
    _write_snapshot(tmp_path, _make_payload(ts))
    dm = DayManager.__new__(DayManager)
    age = dm._intraday_analysis_age_minutes()
    assert age is not None
    assert age > 380  # ~6h30m == 390min, allow drift


def test_age_minutes_returns_none_when_timestamp_missing(tmp_path, monkeypatch):
    DayManager = _DM()
    monkeypatch.chdir(tmp_path)
    _write_snapshot(tmp_path, {"market_context": {}})
    dm = DayManager.__new__(DayManager)
    assert dm._intraday_analysis_age_minutes() is None


def test_age_minutes_returns_none_when_timestamp_malformed(tmp_path, monkeypatch):
    DayManager = _DM()
    monkeypatch.chdir(tmp_path)
    _write_snapshot(tmp_path, _make_payload("not-an-iso-timestamp"))
    dm = DayManager.__new__(DayManager)
    assert dm._intraday_analysis_age_minutes() is None


def test_is_fresh_true_for_recent_snapshot(tmp_path, monkeypatch):
    DayManager = _DM()
    monkeypatch.chdir(tmp_path)
    _write_snapshot(tmp_path, _make_payload(datetime.now().isoformat()))
    dm = DayManager.__new__(DayManager)
    assert dm._intraday_analysis_is_fresh() is True


def test_is_fresh_false_for_stale_snapshot(tmp_path, monkeypatch):
    DayManager = _DM()
    monkeypatch.chdir(tmp_path)
    ts = (datetime.now() - timedelta(minutes=20)).isoformat()
    _write_snapshot(tmp_path, _make_payload(ts))
    dm = DayManager.__new__(DayManager)
    assert dm._intraday_analysis_is_fresh() is False
    assert dm._intraday_analysis_is_fresh(max_age_minutes=30.0) is True


def test_is_fresh_false_when_file_missing(tmp_path, monkeypatch):
    DayManager = _DM()
    monkeypatch.chdir(tmp_path)
    dm = DayManager.__new__(DayManager)
    assert dm._intraday_analysis_is_fresh() is False


def test_is_fresh_false_at_exact_threshold(tmp_path, monkeypatch):
    """Boundary case: a 15-min-old snapshot must read as STALE (default cap).
    The helper uses strict < so equal-to-cap means stale."""
    DayManager = _DM()
    monkeypatch.chdir(tmp_path)
    ts = (datetime.now() - timedelta(minutes=15, seconds=5)).isoformat()
    _write_snapshot(tmp_path, _make_payload(ts))
    dm = DayManager.__new__(DayManager)
    assert dm._intraday_analysis_is_fresh() is False
