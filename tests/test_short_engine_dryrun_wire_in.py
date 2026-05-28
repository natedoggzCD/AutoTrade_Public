"""Tests for the DayManager short-engine dry-run wire-in (v0.28.50 P2).

Verifies that ``DayManager._run_short_engine_dryrun_pass`` wires
``ShortSignalScorer`` into the live cycle, writes telemetry to
``logs/short_dryrun_YYYY-MM-DD.jsonl``, and honors ``flags/disable_shorts.flag``.

The method is exercised via ``types.MethodType`` bound to a minimal namespace
stand-in so we don't pay the full DayManager init cost.
"""

from __future__ import annotations

import json
import types
from datetime import datetime
from types import SimpleNamespace

from autotrade.core.day_manager import DayManager


def _make_stub(signals, positions=None, breadth_pct=38.0, regime="rotation"):
    """Build a minimal stand-in carrying the attributes the method reads."""
    stub = SimpleNamespace()
    stub.signals = list(signals or [])
    stub._last_short_dryrun_pass = None
    stub._load_resolved_regime_context = lambda: {
        "breadth_pct_positive": breadth_pct,
        "sources_degraded": False,
    }
    stub._effective_market_regime = lambda: regime
    # Bind the real method onto the stub so we exercise production logic.
    stub._run_short_engine_dryrun_pass = types.MethodType(
        DayManager._run_short_engine_dryrun_pass, stub
    )
    stub._collect_short_universe = types.MethodType(
        DayManager._collect_short_universe, stub
    )
    stub._load_short_universe_from_daily_features = lambda limit=250: []
    stub._safe_float = DayManager._safe_float
    return stub


def _bearish_signal(symbol="ACME", price=20.0):
    """Build a signal row that the scorer will accept (score >= 55)."""
    return {
        "symbol": symbol,
        "price": price,
        "close": price,
        "ema_20": price * 1.05,
        "ema_50": price * 1.10,
        "vwap": price * 1.02,
        "rsi": 35.0,
        "rsi_14": 35.0,
        "volume_ratio": 1.8,
        "day_change_pct": -1.5,
        "atr_14": price * 0.025,
        "news_sentiment": "downgrade reported",
    }


def test_short_dryrun_writes_telemetry_and_emits_candidates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Bearish signals on names we don't hold.
    stub = _make_stub(
        signals=[_bearish_signal("ACME", 20.0), _bearish_signal("WIDGET", 15.0)],
        positions=[SimpleNamespace(symbol="UNRELATED")],
        breadth_pct=38.0,
    )

    stub._run_short_engine_dryrun_pass([SimpleNamespace(symbol="UNRELATED")])

    log_path = tmp_path / "logs" / f"short_dryrun_{datetime.now():%Y-%m-%d}.jsonl"
    telemetry_path = (
        tmp_path / "logs" / f"short_engine_telemetry_{datetime.now():%Y-%m-%d}.jsonl"
    )
    assert log_path.exists(), "telemetry file should be created"
    assert telemetry_path.exists(), "short-engine telemetry file should be created"
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    telemetry_lines = telemetry_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1, "exactly one telemetry entry per pass"
    assert telemetry_lines == lines
    record = json.loads(lines[0])
    assert record["regime"] == "rotation"
    assert record["breadth_pct_positive"] == 38.0
    assert record["universe_size"] == 2
    assert record["candidates_count"] >= 1, (
        "scorer should accept the bearish-tilted signals"
    )
    assert record["signals_generated"] == record["candidates_count"]
    # Top candidate has the expected schema.
    top = record["top_candidates"][0]
    assert top["side"] == "short"
    assert "ticker" in top and "score" in top


def test_short_dryrun_honors_kill_switch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Set the disable_shorts kill switch.
    flags_dir = tmp_path / "flags"
    flags_dir.mkdir()
    (flags_dir / "disable_shorts.flag").write_text("blocked by test")
    # Point kill_switch at the test-local flags dir by monkeypatching the
    # module-level PROJECT_ROOT-derived FLAGS_DIR.
    from autotrade.utils import kill_switch as ks

    monkeypatch.setattr(ks, "FLAGS_DIR", flags_dir)

    stub = _make_stub(
        signals=[_bearish_signal("ACME", 20.0)],
        positions=[],
        breadth_pct=30.0,
    )
    stub._run_short_engine_dryrun_pass([])

    log_path = tmp_path / "logs" / f"short_dryrun_{datetime.now():%Y-%m-%d}.jsonl"
    assert not log_path.exists(), (
        "kill switch should short-circuit before any telemetry is written"
    )


def test_short_dryrun_skips_held_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    held_position = SimpleNamespace(symbol="ACME")
    stub = _make_stub(
        signals=[_bearish_signal("ACME", 20.0)],
        positions=[held_position],
    )

    stub._run_short_engine_dryrun_pass([held_position])

    log_path = tmp_path / "logs" / f"short_dryrun_{datetime.now():%Y-%m-%d}.jsonl"
    assert log_path.exists(), "zero-candidate telemetry should still be written"
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["universe_size"] == 0
    assert record["signals_generated"] == 0
    assert record["candidates_rejected_by_gate"] == 0


def test_short_dryrun_writes_zero_signal_telemetry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stub = _make_stub(signals=[], positions=[], breadth_pct=54.4, regime="rotation")

    stub._run_short_engine_dryrun_pass([])

    log_path = tmp_path / "logs" / f"short_dryrun_{datetime.now():%Y-%m-%d}.jsonl"
    telemetry_path = (
        tmp_path / "logs" / f"short_engine_telemetry_{datetime.now():%Y-%m-%d}.jsonl"
    )
    assert log_path.exists()
    assert telemetry_path.exists()
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["regime"] == "rotation"
    assert record["breadth_pct_positive"] == 54.4
    assert record["universe_size"] == 0
    assert record["signals_generated"] == 0
    assert record["candidates_passed"] == 0


def test_short_dryrun_logs_per_cycle_summary(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    stub = _make_stub(signals=[], positions=[], breadth_pct=54.4, regime="rotation")

    stub._run_short_engine_dryrun_pass([])

    assert "[SHORT ENGINE] candidates_screened=0 rejected=0 submitted=0" in caplog.text


def test_short_dryrun_uses_independent_short_universe_before_long_signals(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    stub = _make_stub(
        signals=[_bearish_signal("LONGPOOL", 20.0)],
        positions=[],
        breadth_pct=25.0,
        regime="selloff",
    )
    stub.short_universe_candidates = [_bearish_signal("SHORTONLY", 12.0)]

    stub._run_short_engine_dryrun_pass([])

    log_path = tmp_path / "logs" / f"short_dryrun_{datetime.now():%Y-%m-%d}.jsonl"
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["universe_size"] == 1
    assert record["top_candidates"][0]["ticker"] == "SHORTONLY"


def test_short_dryrun_cooldown_prevents_double_invocation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stub = _make_stub(
        signals=[_bearish_signal("ACME", 20.0)],
        positions=[],
    )

    stub._run_short_engine_dryrun_pass([])
    # Second call within the 60s cooldown window should be a no-op (no new line).
    stub._run_short_engine_dryrun_pass([])

    log_path = tmp_path / "logs" / f"short_dryrun_{datetime.now():%Y-%m-%d}.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1, "cooldown must prevent a second telemetry write"
