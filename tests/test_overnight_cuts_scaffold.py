"""H1 D2: overnight-cut watchlist scaffold tests.

Pins the read/write contract for data/overnight_cuts_<date>.json and
verifies the recovery-boost helper is a no-op while the policy flag
is OFF (default).
"""

from __future__ import annotations

from datetime import date

import pytest

from autotrade.signals.overnight_cuts import (
    DEFAULT_RECOVERY_BOOST,
    RECOVERY_PRIORITY_TAG,
    CutRecord,
    apply_recovery_boost,
    cuts_path,
    load_cuts,
    record_cut,
)


@pytest.fixture(autouse=True)
def isolate_cuts_dir(tmp_path, monkeypatch):
    from autotrade.signals import overnight_cuts as oc

    monkeypatch.setattr(oc, "CUTS_DIR", tmp_path)
    yield


class _StubCandidate:
    def __init__(self, symbol: str, entry_score: float):
        self.symbol = symbol
        self.entry_score = entry_score
        self.metadata: dict = {}


def test_record_cut_writes_and_load_cuts_reads():
    when = date(2026, 5, 22)
    rec = record_cut(
        symbol="aapl",
        exit_price=180.5,
        exit_reason="t15_weak",
        prior_avg_entry=175.0,
        when=when,
    )
    assert rec.symbol == "AAPL"
    assert cuts_path(when).exists()
    rows = load_cuts(when)
    assert len(rows) == 1
    assert rows[0].exit_price == pytest.approx(180.5)
    assert rows[0].exit_reason == "t15_weak"
    assert rows[0].prior_avg_entry == pytest.approx(175.0)
    assert rows[0].qty == 0.0


def test_record_cut_is_idempotent_on_symbol():
    when = date(2026, 5, 22)
    record_cut("MSFT", 400.0, "t15_weak", 405.0, when=when)
    record_cut("MSFT", 402.0, "t15_weak", 405.0, when=when)
    rows = load_cuts(when)
    assert len(rows) == 1
    assert rows[0].exit_price == pytest.approx(402.0)


def test_recovery_boost_is_noop_when_disabled():
    cands = [_StubCandidate("ABCD", 50.0), _StubCandidate("EFGH", 60.0)]
    cuts = [
        CutRecord(
            symbol="ABCD",
            exit_price=10.0,
            exit_reason="t15_weak",
            prior_avg_entry=9.0,
            cut_at="2026-05-21T15:45:00",
        )
    ]
    boosted = apply_recovery_boost(cands, cuts, enabled=False)
    assert boosted == 0
    assert cands[0].entry_score == 50.0
    assert cands[0].metadata == {}


def test_recovery_boost_when_enabled_marks_priority_and_bumps_score():
    cands = [_StubCandidate("ABCD", 50.0), _StubCandidate("EFGH", 60.0)]
    cuts = [
        CutRecord(
            symbol="ABCD",
            exit_price=10.0,
            exit_reason="t15_weak",
            prior_avg_entry=9.0,
            cut_at="2026-05-21T15:45:00",
        )
    ]
    boosted = apply_recovery_boost(cands, cuts, enabled=True, boost=7.5)
    assert boosted == 1
    assert cands[0].entry_score == pytest.approx(57.5)
    assert cands[0].metadata["priority"] == RECOVERY_PRIORITY_TAG
    assert cands[0].metadata["prior_cut_price"] == 10.0
    assert cands[0].metadata["prior_avg_entry"] == 9.0
    assert cands[1].entry_score == 60.0
    assert cands[1].metadata == {}


def test_load_cuts_missing_file_returns_empty():
    assert load_cuts(date(2026, 5, 1)) == []


def test_default_boost_value_is_stable():
    assert DEFAULT_RECOVERY_BOOST == 5.0


def test_record_cut_persists_d3_metadata():
    when = date(2026, 5, 22)
    record_cut(
        symbol="weak",
        exit_price=12.25,
        exit_reason="t15_weak_force_exit_trim",
        prior_avg_entry=13.0,
        when=when,
        qty=50,
        trim_fraction=0.5,
        policy_mode="weak_only_t15_trim",
        weak_signal_pct=-1.25,
        order_id="ord-123",
    )

    rows = load_cuts(when)

    assert len(rows) == 1
    assert rows[0].symbol == "WEAK"
    assert rows[0].qty == pytest.approx(50)
    assert rows[0].trim_fraction == pytest.approx(0.5)
    assert rows[0].policy_mode == "weak_only_t15_trim"
    assert rows[0].weak_signal_pct == pytest.approx(-1.25)
    assert rows[0].order_id == "ord-123"
