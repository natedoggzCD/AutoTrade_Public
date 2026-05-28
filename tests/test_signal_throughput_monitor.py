"""Tests for the signal throughput monitor.

day-manager 2026-05-19: pin the alarm/flag/recovery semantics of the
SignalThroughputMonitor so a future refactor cannot regress them.
"""

from __future__ import annotations

import json
import logging

import pytest

from autotrade.core.signal_throughput_monitor import (
    CONSECUTIVE_BAD_CYCLES_FOR_FLAG,
    CONVERSION_FLOOR,
    MIN_ACCEPTED_FOR_ALARM,
    SignalThroughputMonitor,
)


@pytest.fixture
def monitor(tmp_path):
    flag = tmp_path / "flags" / "signal_throughput_broken.flag"
    return SignalThroughputMonitor(flag_path=flag, telemetry_dir=tmp_path / "logs")


def test_constants_match_directive_defaults():
    assert MIN_ACCEPTED_FOR_ALARM == 50
    assert CONVERSION_FLOOR == 0.05
    assert CONSECUTIVE_BAD_CYCLES_FOR_FLAG == 3


def test_below_sample_floor_never_alarms(monitor, caplog):
    """A quiet family (accepted < 50) must not trip the alarm even if zero
    executed."""
    with caplog.at_level(logging.CRITICAL):
        result = monitor.observe(
            "agentic",
            signals_accepted=10,
            signals_executed=0,
            phase="MARKET_HOURS",
        )
    assert result["broken"] is False
    assert result["flag_written"] is False
    assert not any(
        "[SIGNAL THROUGHPUT BROKEN]" in rec.message for rec in caplog.records
    )


def test_broken_log_fires_at_first_bad_cycle(monitor, caplog):
    """One bad cycle (393/0) emits CRITICAL but does NOT write the flag yet."""
    with caplog.at_level(logging.CRITICAL):
        result = monitor.observe(
            "agentic",
            signals_accepted=393,
            signals_executed=0,
            skip_reasons={"bearish_session": 200, "vwap_timing": 150},
            cycle_n=1,
            phase="MARKET_HOURS",
        )
    assert result["broken"] is True
    assert result["flag_written"] is False
    msgs = [rec.message for rec in caplog.records if rec.levelno == logging.CRITICAL]
    assert any("[SIGNAL THROUGHPUT BROKEN]" in m for m in msgs)
    assert any("conversion=0.00%" in m for m in msgs)
    assert any("bearish_session=200" in m for m in msgs)


def test_flag_writes_after_third_consecutive_bad_cycle(monitor, tmp_path):
    """Exactly today's failure mode: keep observing 393/0 and the flag
    file must appear on the third cycle, not before."""
    flag = tmp_path / "flags" / "signal_throughput_broken.flag"

    r1 = monitor.observe(
        "agentic",
        signals_accepted=393,
        signals_executed=0,
        skip_reasons={"bearish_session": 393},
        cycle_n=1,
        phase="MARKET_HOURS",
    )
    assert r1["broken"] is True
    assert r1["flag_written"] is False
    assert not flag.exists()

    r2 = monitor.observe(
        "agentic",
        signals_accepted=393,
        signals_executed=0,
        skip_reasons={"bearish_session": 393},
        cycle_n=2,
        phase="MARKET_HOURS",
    )
    assert r2["broken"] is True
    assert r2["flag_written"] is False
    assert not flag.exists()

    r3 = monitor.observe(
        "agentic",
        signals_accepted=393,
        signals_executed=0,
        skip_reasons={"bearish_session": 393},
        cycle_n=3,
        phase="MARKET_HOURS",
    )
    assert r3["broken"] is True
    assert r3["flag_written"] is True
    assert flag.exists()

    payload = json.loads(flag.read_text(encoding="utf-8"))
    assert payload["family"] == "agentic"
    assert payload["signals_accepted"] == 393
    assert payload["signals_executed"] == 0
    assert payload["consecutive_bad_cycles"] == 3
    assert payload["top_rejection_reasons"] == {"bearish_session": 393}


def test_recovery_clears_flag_and_emits_log(monitor, tmp_path, caplog):
    flag = tmp_path / "flags" / "signal_throughput_broken.flag"

    # Drive 3 consecutive bad cycles to write the flag.
    for cycle in range(1, 4):
        monitor.observe(
            "agentic",
            signals_accepted=393,
            signals_executed=0,
            skip_reasons={"bearish_session": 393},
            cycle_n=cycle,
            phase="MARKET_HOURS",
        )
    assert flag.exists()

    # Now a healthy cycle: conversion 100/20 = 20%.
    with caplog.at_level(logging.WARNING):
        recovered = monitor.observe(
            "agentic",
            signals_accepted=100,
            signals_executed=20,
            cycle_n=4,
            phase="MARKET_HOURS",
        )
    assert recovered["broken"] is False
    assert recovered["recovered"] is True
    assert recovered["flag_removed"] is True
    assert not flag.exists()
    assert any("[SIGNAL THROUGHPUT RECOVERED]" in rec.message for rec in caplog.records)


def test_streak_resets_on_first_good_cycle(monitor, tmp_path):
    """A single good cycle in the middle of bad cycles must reset the
    consecutive counter so the flag only fires on a true 3-in-a-row run."""
    flag = tmp_path / "flags" / "signal_throughput_broken.flag"

    monitor.observe(
        "agentic", signals_accepted=200, signals_executed=0, phase="MARKET_HOURS"
    )  # bad 1
    monitor.observe(
        "agentic", signals_accepted=200, signals_executed=0, phase="MARKET_HOURS"
    )  # bad 2
    monitor.observe(
        "agentic", signals_accepted=200, signals_executed=40, phase="MARKET_HOURS"
    )  # ok, resets
    monitor.observe(
        "agentic", signals_accepted=200, signals_executed=0, phase="MARKET_HOURS"
    )  # bad 1 again
    monitor.observe(
        "agentic", signals_accepted=200, signals_executed=0, phase="MARKET_HOURS"
    )  # bad 2

    # Only 2 consecutive at this point — flag must NOT exist.
    assert not flag.exists()


def test_telemetry_written_per_observation(monitor, tmp_path):
    monitor.observe(
        "agentic",
        signals_accepted=393,
        signals_executed=0,
        skip_reasons={"bearish_session": 393},
        phase="MARKET_HOURS",
    )
    monitor.observe(
        "agentic",
        signals_accepted=100,
        signals_executed=10,
        skip_reasons={"vwap_timing": 90},
        phase="MARKET_HOURS",
    )

    # Find the dated telemetry file
    log_files = list((tmp_path / "logs").glob("signal_throughput_*.jsonl"))
    assert len(log_files) == 1
    rows = [
        json.loads(line)
        for line in log_files[0].read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["family"] == "agentic"
    assert rows[0]["bad"] is True
    assert rows[0]["top_rejection_reasons"] == {"bearish_session": 393}
    assert rows[1]["bad"] is False  # 10/100 = 10% > 5% floor


def test_below_sample_floor_does_not_break_streak(monitor, tmp_path):
    """A cycle with accepted < 50 is noise and should NOT reset the
    consecutive_bad counter — we want the alarm to survive a brief quiet
    window inside an otherwise broken period."""
    flag = tmp_path / "flags" / "signal_throughput_broken.flag"

    monitor.observe(
        "agentic", signals_accepted=200, signals_executed=0, phase="MARKET_HOURS"
    )  # bad 1
    monitor.observe(
        "agentic", signals_accepted=200, signals_executed=0, phase="MARKET_HOURS"
    )  # bad 2
    monitor.observe(
        "agentic", signals_accepted=10, signals_executed=0, phase="MARKET_HOURS"
    )  # noise — skipped
    monitor.observe(
        "agentic", signals_accepted=200, signals_executed=0, phase="MARKET_HOURS"
    )  # bad 3 -> flag

    assert flag.exists()


def test_per_family_isolation(monitor):
    """An ok family's good cycle must not clear another family's bad streak."""
    r1 = monitor.observe(
        "agentic", signals_accepted=200, signals_executed=0, phase="MARKET_HOURS"
    )
    r2 = monitor.observe(
        "overnight", signals_accepted=200, signals_executed=80, phase="MARKET_HOURS"
    )
    r3 = monitor.observe(
        "agentic", signals_accepted=200, signals_executed=0, phase="MARKET_HOURS"
    )
    assert r1["broken"] is True
    assert r2["broken"] is False
    assert r3["broken"] is True


def test_observe_never_raises_on_garbage_input(monitor):
    """Defensive: bad inputs (None counts, missing skip_reasons, etc.) must
    not crash the caller's cycle loop. The monitor is a tap, not a brake."""
    monitor.observe(
        "agentic", signals_accepted=None, signals_executed=None, phase="MARKET_HOURS"
    )
    monitor.observe(
        "agentic", signals_accepted=-5, signals_executed=-10, phase="MARKET_HOURS"
    )
    monitor.observe(
        "agentic",
        signals_accepted=200,
        signals_executed=0,
        skip_reasons=None,
        phase="MARKET_HOURS",
    )


# H7 (2026-05-21): phase-gating. PREMARKET / OVERNIGHT / POST_MARKET cycles
# with executed=0 are expected, not "broken pipeline". Monitor must not
# emit, flag, or bump the consecutive-bad counter during those phases.
@pytest.mark.parametrize(
    "phase",
    [
        "premarket",
        "PREMARKET",
        "early_premarket",
        "overnight",
        "post_market",
        "after_hours",
        "pm_workflow",
    ],
)
def test_phase_gate_silences_non_trading_phases(monitor, caplog, phase):
    with caplog.at_level(logging.CRITICAL):
        result = monitor.observe(
            "agentic",
            signals_accepted=393,
            signals_executed=0,
            skip_reasons={"bearish_session": 200},
            phase=phase,
        )
    assert result["broken"] is False
    assert result["flag_written"] is False
    assert result.get("skipped_phase") == phase.lower()
    assert not any(
        "[SIGNAL THROUGHPUT BROKEN]" in rec.message for rec in caplog.records
    )


def test_phase_gate_persists_streak_across_premarket_observations(monitor):
    """Bad MARKET_HOURS cycle, then a PREMARKET no-op, then bad MARKET_HOURS
    again: the second MARKET_HOURS cycle should land at consecutive_bad=2,
    not be reset by the intervening premarket call."""
    monitor.observe(
        "agentic", signals_accepted=200, signals_executed=0, phase="MARKET_HOURS"
    )
    monitor.observe(
        "agentic", signals_accepted=200, signals_executed=0, phase="PREMARKET"
    )
    state = monitor._state["agentic"]
    assert state.consecutive_bad == 1
    monitor.observe(
        "agentic", signals_accepted=200, signals_executed=0, phase="MARKET_HOURS"
    )
    assert monitor._state["agentic"].consecutive_bad == 2


@pytest.mark.parametrize(
    "phase", ["MARKET_OPEN", "market_hours", "core_trading", "observation"]
)
def test_phase_gate_allows_trading_phases(monitor, caplog, phase):
    with caplog.at_level(logging.CRITICAL):
        result = monitor.observe(
            "agentic",
            signals_accepted=200,
            signals_executed=0,
            skip_reasons={"vwap_timing": 100},
            phase=phase,
        )
    assert result["broken"] is True
