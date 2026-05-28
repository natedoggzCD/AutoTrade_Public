"""Tests for the producer-side bracket-geometry rejection gate.

day-manager 2026-05-19 (H3): the [PLAN RISK REPAIR] guard kept firing
every cycle for RVYL because the upstream producer kept emitting
invalid take_profit. The repair is defense-in-depth; the producer
should reject explicit bad geometry instead of silently repairing.
"""

from autotrade.core.autonomous_agent import _long_bracket_source_geometry_ok


def test_passes_when_geometry_is_clean():
    ok, reason = _long_bracket_source_geometry_ok(
        entry_price=10.0, stop_loss=9.5, take_profit=11.0
    )
    assert ok is True
    assert reason == ""


def test_passes_when_levels_are_missing_zeros_default_path():
    """Missing levels (0.0) must NOT be rejected — they legitimately
    fall through to entry*0.95 / entry*1.10 defaults at the caller."""
    ok, _ = _long_bracket_source_geometry_ok(
        entry_price=10.0, stop_loss=0.0, take_profit=0.0
    )
    assert ok is True


def test_rejects_take_profit_at_or_below_entry_rvyl_pattern():
    """RVYL's daily pattern: source emitted tp <= entry."""
    ok, reason = _long_bracket_source_geometry_ok(
        entry_price=2.50, stop_loss=2.40, take_profit=2.45
    )
    assert ok is False
    assert "take_profit_not_above_entry" in reason


def test_rejects_take_profit_within_min_gap_but_above_entry():
    """A microscopic TP above entry is just as broken — the min_gap_pct
    floor is what makes the geometry executable."""
    ok, reason = _long_bracket_source_geometry_ok(
        entry_price=10.0, stop_loss=9.5, take_profit=10.05, min_gap_pct=0.01
    )
    assert ok is False
    assert "take_profit_not_above_entry" in reason


def test_rejects_stop_loss_at_or_above_entry():
    ok, reason = _long_bracket_source_geometry_ok(
        entry_price=10.0, stop_loss=10.05, take_profit=11.0
    )
    assert ok is False
    assert "stop_loss_not_below_entry" in reason


def test_passes_when_entry_price_is_invalid():
    """If entry itself is unusable, this gate has nothing to validate —
    defer to the downstream sanitizer."""
    ok, _ = _long_bracket_source_geometry_ok(
        entry_price=0.0, stop_loss=10.0, take_profit=5.0
    )
    assert ok is True


def test_min_gap_pct_is_configurable():
    """A wider min_gap should reject smaller wedges."""
    ok_lax, _ = _long_bracket_source_geometry_ok(
        entry_price=10.0, stop_loss=9.5, take_profit=10.10, min_gap_pct=0.005
    )
    ok_strict, _ = _long_bracket_source_geometry_ok(
        entry_price=10.0, stop_loss=9.5, take_profit=10.10, min_gap_pct=0.02
    )
    assert ok_lax is True
    assert ok_strict is False
