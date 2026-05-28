"""Tests for the producer-side qty-sizing helper used by ENTRY OVERRIDE.

day-manager 2026-05-19 (H7): PM-plan watch entries promoted via
[ENTRY OVERRIDE] arrived at wave_execute with qty=0, and the silent
1-share fallback produced joke positions (HUT $94, AMKR $66). The
producer-side fix is _size_signal_qty_from_notional() — pin its
behavior here so a future refactor cannot regress it.
"""

import sys
from pathlib import Path

import pytest


def _DM():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from autotrade.core.day_manager import DayManager

    return DayManager


def _make_dm():
    DayManager = _DM()
    dm = DayManager.__new__(DayManager)
    dm.position_size_target = 2000.0
    dm.get_current_price = lambda symbol: 0.0
    return dm


def test_sizing_writes_qty_from_entry_price():
    dm = _make_dm()
    sig = {"entry_price": 94.38}
    qty = dm._size_signal_qty_from_notional(sig, "HUT")
    assert qty == 21  # int(2000 / 94.38)
    assert sig["qty"] == 21
    # notional within 10% of target
    assert abs(21 * 94.38 - 2000.0) / 2000.0 < 0.10


def test_sizing_preserves_existing_positive_qty():
    dm = _make_dm()
    sig = {"qty": 50, "entry_price": 10.0}
    qty = dm._size_signal_qty_from_notional(sig, "ABC")
    assert qty == 50
    assert sig["qty"] == 50


def test_sizing_walks_price_fallback_chain():
    """Helper should try entry_price, planned_entry, current_price, price, entry
    in order before asking get_current_price."""
    dm = _make_dm()
    for field in ("planned_entry", "current_price", "price", "entry"):
        sig = {field: 40.0}
        qty = dm._size_signal_qty_from_notional(sig, "X")
        assert qty == 50, f"failed for field={field}"


def test_sizing_falls_back_to_get_current_price():
    dm = _make_dm()
    dm.get_current_price = lambda symbol: 25.0 if symbol == "FOO" else 0.0
    sig = {}
    qty = dm._size_signal_qty_from_notional(sig, "FOO")
    assert qty == 80  # 2000 / 25


def test_sizing_returns_1_share_when_price_unavailable():
    """Last-resort safety: never crash and never return 0; emit WARNING."""
    dm = _make_dm()
    sig = {}
    qty = dm._size_signal_qty_from_notional(sig, "UNKNOWN")
    assert qty == 1
    assert sig["qty"] == 1


def test_sizing_honors_custom_target_notional():
    dm = _make_dm()
    sig = {"entry_price": 10.0}
    qty = dm._size_signal_qty_from_notional(sig, "ABC", target_notional=500.0)
    assert qty == 50


def test_sizing_acceptance_criteria_within_10_percent_of_target():
    """Handoff doc H7 acceptance criteria: qty * planned_entry within
    10% of position_size_target across the PM-plan promotion candidates."""
    dm = _make_dm()
    today_candidates = [
        ("HUT", 94.38),
        ("AMKR", 66.42),
        ("TEAM", 89.30),
        ("ODD", 12.27),
        ("EOSE", 7.35),
        ("BTG", 4.71),
        ("USAS", 5.83),
        ("ORLA", 12.99),
    ]
    for ticker, price in today_candidates:
        sig = {"entry_price": price}
        qty = dm._size_signal_qty_from_notional(sig, ticker)
        notional = qty * price
        deviation = abs(notional - 2000.0) / 2000.0
        # within 10% except where 1 share already exceeds target (none here)
        assert deviation < 0.10, (
            f"{ticker}: qty={qty} notional=${notional:.0f} deviates "
            f"{deviation*100:.1f}% from $2000 target"
        )
