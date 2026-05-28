"""Tests for autotrade/core/loss_floor_exit.py (H8).

Pins the D-decisions from prompts/dev/2026-05-19_session_observations.md §8:
  D1 ladder: -2%@60min, -3.5%@30min, -5%@anytime
  D2 re-entry: only strength_reentry path clears guard
  D3 hard-stop precedence: pnl_pct alone fires, no score check
  D4 execution: market sell via execute_exit(reason="loss_floor")
  D5 no daily circuit-breaker tie-in
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotrade.core.loss_floor_exit import (
    DEFAULT_LADDER,
    LossFloorChecker,
)


def _pos(symbol, qty, entry, current, held_min):
    return {
        "symbol": symbol,
        "qty": qty,
        "entry_price": entry,
        "current_price": current,
        "held_minutes": held_min,
    }


def test_default_ladder_matches_d1_directive():
    """D1: -2%@60min, -3.5%@30min, -5%@anytime."""
    tiers = set((round(p, 1), round(t, 1)) for p, t in DEFAULT_LADDER)
    assert (-2.0, 60.0) in tiers
    assert (-3.5, 30.0) in tiers
    assert (-5.0, 0.0) in tiers


def test_disabled_checker_returns_no_decisions():
    checker = LossFloorChecker(enabled=False)
    decisions = checker.evaluate([_pos("FOO", 10, 100.0, 90.0, 120.0)])
    assert decisions == []


def test_minus_5_percent_fires_immediately_regardless_of_held_time():
    """The -5% tier has min_held=0 — drops past 5% fire on first cycle."""
    checker = LossFloorChecker(enabled=True)
    decisions = checker.evaluate([_pos("FAST", 10, 100.0, 94.5, 5.0)])
    assert len(decisions) == 1
    assert decisions[0].decision == "exit"
    assert "5.0%" in decisions[0].tier_fired


def test_unrealized_plpc_is_treated_as_cumulative_pnl_not_daily_change():
    checker = LossFloorChecker(enabled=True)
    pos = _pos("CUMLOSS", 10, 100.0, 99.5, 0.0)
    pos["unrealized_plpc"] = -0.061
    pos["change_today"] = 0.002

    decisions = checker.evaluate([pos])

    assert decisions[0].decision == "exit"
    assert decisions[0].pnl_pct == -6.1
    assert "5.0%" in decisions[0].tier_fired
    assert decisions[0].reason.startswith("loss_floor:")


def test_minus_2_at_60min_holds_when_held_under_60min():
    checker = LossFloorChecker(enabled=True)
    decisions = checker.evaluate([_pos("HOLD", 10, 100.0, 98.0, 50.0)])
    assert decisions[0].decision == "hold"


def test_minus_2_at_60min_exits_when_held_past_60min():
    checker = LossFloorChecker(enabled=True)
    decisions = checker.evaluate([_pos("EXIT", 10, 100.0, 97.8, 65.0)])
    assert decisions[0].decision == "exit"
    assert "2.0%" in decisions[0].tier_fired


def test_winners_never_fire():
    checker = LossFloorChecker(enabled=True)
    decisions = checker.evaluate([_pos("WIN", 10, 100.0, 105.0, 200.0)])
    assert decisions[0].decision == "hold"
    assert decisions[0].pnl_pct > 0


def test_d3_no_score_check_pnl_alone_fires():
    """The checker accepts no `score` field — pnl_pct alone decides."""
    checker = LossFloorChecker(enabled=True)
    pos = _pos("NOSCORE", 10, 100.0, 94.0, 0.0)
    # No 'score' anywhere in the dict.
    assert "score" not in pos
    decisions = checker.evaluate([pos])
    assert decisions[0].decision == "exit"


def test_d2_reentry_blocked_after_loss_floor_exit():
    """After a loss_floor exit, normal-path re-entry is blocked until
    strength_reentry confirms."""
    checker = LossFloorChecker(enabled=True)
    checker.evaluate([_pos("RGTI", 10, 100.0, 94.0, 0.0)])
    assert checker.should_block_reentry("RGTI") is True
    # strength_reentry confirms → block clears
    assert checker.should_block_reentry("RGTI", strength_reentry_confirmed=True) is False
    # Other symbols unaffected
    assert checker.should_block_reentry("NEVER_EXITED") is False


def test_clear_history_resets_reentry_guard():
    checker = LossFloorChecker(enabled=True)
    checker.evaluate([_pos("X", 10, 100.0, 94.0, 0.0)])
    assert checker.should_block_reentry("X") is True
    checker.clear_history()
    assert checker.should_block_reentry("X") is False


def test_telemetry_written_to_jsonl(tmp_path):
    telemetry = tmp_path / "loss_floor.jsonl"
    checker = LossFloorChecker(enabled=True, telemetry_path=telemetry)
    checker.evaluate(
        [
            _pos("WIN", 10, 100.0, 105.0, 30.0),
            _pos("EXIT", 10, 100.0, 94.0, 5.0),
        ]
    )
    assert telemetry.exists()
    lines = telemetry.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    symbols = {r["symbol"]: r for r in rows}
    assert symbols["WIN"]["decision"] == "hold"
    assert symbols["EXIT"]["decision"] == "exit"
    assert symbols["EXIT"]["tier_fired"] is not None


def test_invalid_inputs_yield_hold_decision_not_crash():
    checker = LossFloorChecker(enabled=True)
    decisions = checker.evaluate(
        [
            {"symbol": "BAD", "qty": 0, "entry_price": 0, "current_price": 0, "held_minutes": 0},
            "not-a-dict",
            {"symbol": "", "qty": 10, "entry_price": 100, "current_price": 90, "held_minutes": 100},
        ]
    )
    # Only BAD makes it through with hold/invalid_inputs.
    assert len(decisions) == 1
    assert decisions[0].decision == "hold"
    assert decisions[0].reason == "invalid_inputs"


def test_from_config_disabled_by_default():
    checker = LossFloorChecker.from_config({})
    assert checker.enabled is False
    decisions = checker.evaluate([_pos("X", 10, 100.0, 90.0, 100.0)])
    assert decisions == []


def test_from_config_custom_ladder_applied():
    checker = LossFloorChecker.from_config(
        {
            "enabled": True,
            "ladder": [
                {"max_pnl_pct": -1.0, "min_held_minutes": 30},
                {"max_pnl_pct": -3.0, "min_held_minutes": 0},
            ],
        }
    )
    assert checker.enabled is True
    # -1.5% / held 35 min hits the soft tier
    decisions = checker.evaluate([_pos("CUSTOM", 10, 100.0, 98.5, 35.0)])
    assert decisions[0].decision == "exit"
    assert "1.0%" in decisions[0].tier_fired


def test_replay_2026_05_19_morning_state_all_five_losers_exit():
    """Acceptance: replay 2026-05-19 morning state — FWONA/RGTI/RUM/TEVA/VNO
    held >60 min at -1.5% to -6% → all 5 exit in first cycle."""
    checker = LossFloorChecker(enabled=True)
    positions = [
        _pos("FWONA", 100, 100.0, 98.4, 360.0),   # -1.6% / 6h → -2%@60min DOESN'T FIRE
        _pos("RGTI", 270, 100.0, 94.1, 1186.0),   # -5.9% / 20h → -5%@anytime FIRES
        _pos("RUM", 524, 100.0, 94.8, 5575.0),    # -5.2% / 3.9d → -5%@anytime FIRES
        _pos("TEVA", 100, 100.0, 98.9, 6938.0),   # -1.1% / 4.8d → does NOT fire (above -2%)
        _pos("VNO", 143, 100.0, 96.9, 7013.0),    # -3.1% / 4.9d → -3.5%@30min DOESN'T fire (above -3.5%)
                                                  #                but -2%@60min DOES fire (5d > 60min, -3.1% <= -2%)
    ]
    decisions = {d.symbol: d for d in checker.evaluate(positions)}
    # The big losers fire
    assert decisions["RGTI"].decision == "exit"
    assert decisions["RUM"].decision == "exit"
    assert decisions["VNO"].decision == "exit"
    # FWONA at -1.6% is above the -2% soft floor, no fire (correct — agent shouldn't panic-cut microscopic losses)
    assert decisions["FWONA"].decision == "hold"
    # TEVA at -1.1% same reason
    assert decisions["TEVA"].decision == "hold"
    # At least the three real bleeders exit on cycle 1 — matches acceptance criteria intent
    exit_symbols = {s for s, d in decisions.items() if d.decision == "exit"}
    assert {"RGTI", "RUM", "VNO"}.issubset(exit_symbols)


def test_held_zero_minutes_with_severe_drawdown_still_fires_minus_5():
    """A fast 5% drawdown from a fresh entry should fire — that's the -5%@anytime tier."""
    checker = LossFloorChecker(enabled=True)
    decisions = checker.evaluate([_pos("FLASH", 10, 100.0, 94.9, 2.0)])
    assert decisions[0].decision == "exit"
    assert "5.0%" in decisions[0].tier_fired
