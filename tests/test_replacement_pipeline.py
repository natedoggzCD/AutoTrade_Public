import json
import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("alpaca.trading.client")

from autotrade.core import day_manager as day_manager_mod
from tests.test_day_manager_execution_policy import _new_dm_stub


PLAN_20260413 = Path("plans/morning_game_plan_20260413.json")


def _load_plan_20260413() -> dict:
    return json.loads(PLAN_20260413.read_text(encoding="utf-8"))


def _configure_replacement_dm(plan: dict):
    dm = _new_dm_stub()
    dm._entries_blocked_by_core_data = lambda: (False, "")
    dm._entries_blocked_by_regime = lambda symbol: (False, "")
    dm._apply_candidate_backtest_validation = lambda candidates: list(candidates)
    dm.strategy = SimpleNamespace(
        passes_filter=lambda *args, **kwargs: (True, 0.0, [])
    )
    dm.premarket_handoff = {
        "generated_at_et": plan.get("generated_at", ""),
        "phase": "OPEN_PREP",
        "ranked_watchlist": [dict(row) for row in plan.get("actionable_top50", [])],
    }
    dm.signals = [dict(row) for row in plan.get("full_watchlist", [])]
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm._has_open_buy_order = lambda symbol: (False, "")
    dm._build_candidate_validation_report = lambda signal_data, phase=None: {
        "allowed": True,
        "entry_source": signal_data.get("entry_source", "overnight_plan"),
        "normalized_score": float(signal_data.get("score", signal_data.get("confidence", 0.0)) or 0.0),
        "current_score": float(signal_data.get("score", signal_data.get("confidence", 0.0)) or 0.0),
    }
    dm._resolve_entry_authority = lambda signal_data: {
        "eligible": True,
        "entry_score": float(signal_data.get("score", signal_data.get("confidence", 0.0)) or 0.0),
        "reason": "",
        "entry_source": signal_data.get("entry_source", "overnight_plan"),
        "plan_score_source": "pm_plan_2026-04-13.json",
    }
    dm.can_enter_positions = lambda entry_wave=None, ticker=None: (True, "")
    dm._mark_signal_skipped = day_manager_mod.DayManager._mark_signal_skipped.__get__(
        dm, day_manager_mod.DayManager
    )
    dm._force_signal_skip = day_manager_mod.DayManager._force_signal_skip.__get__(
        dm, day_manager_mod.DayManager
    )
    dm._replace_position_with_candidate = (
        day_manager_mod.DayManager._replace_position_with_candidate.__get__(
            dm, day_manager_mod.DayManager
        )
    )
    return dm


def test_replacement_candidates_use_20260413_ranked_watchlist_and_filter_nne_concentration():
    plan = _load_plan_20260413()
    dm = _configure_replacement_dm(plan)
    dm.get_positions = lambda: []
    dm.day_tracker = SimpleNamespace(
        trades=[
            {
                "date": datetime.now().isoformat(),
                "ticker": f"X{i}",
                "side": "sell",
                "shares": 10,
                "price": 20.0,
                "reason": "Weak position, replacing with NNE",
            }
            for i in range(30)
        ]
        + [
            {
                "date": datetime.now().isoformat(),
                "ticker": f"Y{i}",
                "side": "sell",
                "shares": 10,
                "price": 20.0,
                "reason": "Weak position, replacing with SGML",
            }
            for i in range(2)
        ]
    )

    candidates = dm.find_replacement_candidates({"NCNO", "RJET"})
    symbols = [c.get("ticker", "") for c in candidates]

    assert symbols, "expected candidates from the 2026-04-13 ranked watchlist"
    assert "NNE" not in symbols
    assert any(symbol in symbols for symbol in {"RVYL", "MIGI", "NCNO", "RKLB", "SA"})


def test_replacement_pipeline_20260413_tags_live_entry_rejection_for_ncno_to_nne(
    caplog,
):
    plan = _load_plan_20260413()
    dm = _configure_replacement_dm(plan)
    candidate = next(row for row in plan["actionable_top50"] if row["symbol"] == "NNE")
    exit_calls = []
    positions = [SimpleNamespace(symbol="NCNO", qty=78, market_value=1350.0)]
    dm.get_positions = lambda: list(positions)
    dm._position_qty = lambda pos: int(getattr(pos, "qty", 0) or 0)
    dm.get_current_price = lambda symbol: {"NCNO": 17.29, "NNE": 20.53}.get(symbol, 0.0)
    dm.execute_exit = lambda symbol, qty, reason: exit_calls.append((symbol, qty, reason)) or positions.clear() or True
    dm._entry_submission_block_reason = lambda **kwargs: "" if kwargs.get("preflight") else "buy_guard:insufficient_buying_power:2000.00<1200.00"

    with caplog.at_level(logging.INFO):
        replaced = dm._replace_position_with_candidate(
            SimpleNamespace(symbol="NCNO", qty=78),
            {"score": 20.0},
            candidate,
            entry_wave=1,
        )

    assert replaced is False
    assert exit_calls == [("NCNO", 78, "Weak position, replacing with NNE")]
    assert any(
        "[REPLACEMENT REJECTED] NCNO->NNE reason=buy_guard" in record.message
        for record in caplog.records
    )
