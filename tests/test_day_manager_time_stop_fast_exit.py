"""H5 time-stop fast-exit (day_manager._evaluate_time_stop_fast_exits).

day-manager 2026-05-19 (H5): pins the new fast-exit path that prevents losing
time-stops from being chained to the replacement engine (the chain caused 5
announced exits at 09:12 CDT on 2026-05-19 to produce 0 broker SELLs in the
next 34 min). Losers past effective_max_hold now route through execute_exit
BEFORE the replacement loop runs. Winners still use the score-based advisor.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("alpaca.trading.client")

from autotrade.core.day_manager import DayManager


class _Stub:
    """Minimal stand-in for DayManager exposing only the surface the helpers
    touch. Calls the real bound methods via unbound dispatch (DayManager.foo)."""

    def __init__(self, *, signals, regime="NORMAL", hold_minutes_map=None):
        self.signals = signals or []
        self._regime = regime
        self._hold_map = hold_minutes_map or {}

    def _effective_market_regime(self):
        return self._regime

    def _hold_minutes(self, symbol):
        return self._hold_map.get(str(symbol).upper(), 0)

    def _position_float(self, position, attr, default):
        val = getattr(position, attr, default)
        try:
            return float(val)
        except (TypeError, ValueError):
            return float(default)

    def _compute_time_stop_state(self, position, hold_minutes):
        # Delegate to the real bound method so _evaluate_time_stop_fast_exits
        # exercises production code when invoked via unbound dispatch.
        return DayManager._compute_time_stop_state(self, position, hold_minutes)


def _pos(symbol, *, unrealized_plpc=0.0):
    return SimpleNamespace(symbol=symbol, unrealized_plpc=unrealized_plpc)


def _compute(stub, position, hold_minutes):
    return DayManager._compute_time_stop_state(stub, position, hold_minutes)


def _evaluate(stub, positions):
    return DayManager._evaluate_time_stop_fast_exits(stub, positions)


def test_compute_returns_none_when_no_signal_max_hold():
    stub = _Stub(signals=[{"ticker": "AAPL"}])
    assert _compute(stub, _pos("AAPL"), 1000) is None


def test_compute_applies_floor_of_five_days():
    stub = _Stub(signals=[{"ticker": "AAPL", "max_hold_days": 2.0}])
    # 5 trading days * 6.5 hours * 60 minutes = 1950 minutes -> hold_days == max
    state = _compute(stub, _pos("AAPL"), 1950)
    assert state is not None
    hold_days, eff_max = state
    assert eff_max == 5.0  # floored above the signal's 2.0
    assert hold_days == pytest.approx(5.0)


def test_compute_matches_symbol_when_ticker_missing():
    stub = _Stub(signals=[{"symbol": "AAPL", "max_hold_days": 5.0}])
    state = _compute(stub, _pos("AAPL"), 1950)
    assert state == pytest.approx((5.0, 5.0))


def test_compute_downmarket_extends_max_hold_by_1_5x():
    stub = _Stub(
        signals=[{"ticker": "AAPL", "max_hold_days": 10.0}],
        regime="CRASH",
    )
    state = _compute(stub, _pos("AAPL"), 60)
    assert state is not None
    _, eff_max = state
    assert eff_max == pytest.approx(15.0)


def test_evaluate_fires_for_loser_past_max_hold():
    stub = _Stub(
        signals=[{"ticker": "RGTI", "max_hold_days": 5.0}],
        hold_minutes_map={"RGTI": 6 * 6.5 * 60},  # 6 days held
    )
    out = _evaluate(stub, [_pos("RGTI", unrealized_plpc=-0.05)])
    assert len(out) == 1
    pos, hold_days, eff_max, pnl_pct = out[0]
    assert pos.symbol == "RGTI"
    assert hold_days >= eff_max
    assert pnl_pct == pytest.approx(-5.0)


def test_evaluate_skips_winners_past_max_hold():
    """Winners stay on the score-based advisor path — H5 only fast-exits losers."""
    stub = _Stub(
        signals=[{"ticker": "AAPL", "max_hold_days": 5.0}],
        hold_minutes_map={"AAPL": 10 * 6.5 * 60},
    )
    out = _evaluate(stub, [_pos("AAPL", unrealized_plpc=0.04)])
    assert out == []


def test_evaluate_skips_under_max_hold():
    stub = _Stub(
        signals=[{"ticker": "AAPL", "max_hold_days": 10.0}],
        hold_minutes_map={"AAPL": 60},  # 1 hour
    )
    out = _evaluate(stub, [_pos("AAPL", unrealized_plpc=-0.05)])
    assert out == []


def test_evaluate_skips_positions_with_no_signal_match():
    stub = _Stub(
        signals=[{"ticker": "OTHER", "max_hold_days": 5.0}],
        hold_minutes_map={"AAPL": 10 * 6.5 * 60},
    )
    out = _evaluate(stub, [_pos("AAPL", unrealized_plpc=-0.05)])
    assert out == []


def test_evaluate_handles_zero_pnl_as_winner():
    """pnl_pct >= 0 means winner — exactly-zero must NOT fast-exit."""
    stub = _Stub(
        signals=[{"ticker": "FLAT", "max_hold_days": 5.0}],
        hold_minutes_map={"FLAT": 10 * 6.5 * 60},
    )
    out = _evaluate(stub, [_pos("FLAT", unrealized_plpc=0.0)])
    assert out == []


def test_evaluate_today_repro_five_losers():
    """2026-05-19 repro: 5 positions (FWONA/RGTI/RUM/TEVA/VNO) past max_hold
    with pnl<0 -> 5 fast-exits returned for execute_exit dispatch."""
    syms = ["FWONA", "RGTI", "RUM", "TEVA", "VNO"]
    stub = _Stub(
        signals=[{"ticker": s, "max_hold_days": 5.0} for s in syms],
        hold_minutes_map={s: 7 * 6.5 * 60 for s in syms},
    )
    positions = [_pos(s, unrealized_plpc=-0.04) for s in syms]
    out = _evaluate(stub, positions)
    assert sorted(p.symbol for p, *_ in out) == syms
    for _, hold_days, eff_max, pnl_pct in out:
        assert hold_days >= eff_max
        assert pnl_pct < 0


def test_evaluate_handles_empty_and_none_inputs():
    stub = _Stub(signals=[])
    assert _evaluate(stub, []) == []
    assert _evaluate(stub, None) == []
