import json
from pathlib import Path

import pytest

from autotrade.core import autonomous_agent as autonomous_agent_mod
from autotrade.core.autonomous_agent import AutonomousAgent

pytest.importorskip("alpaca.trading.client")

from autotrade.core import day_manager as day_manager_mod
from autotrade.core.day_manager import DayManager


def test_overnight_state_freshness_snapshot_is_diagnostic_only(tmp_path, monkeypatch):
    readiness = {
        "is_fresh": False,
        "core_data_fresh": False,
        "pm_ready_for_execution": False,
        "primary_date": "2026-05-01",
        "expected_date": "2026-05-04",
        "blocking_reasons": ["core_data_stale:2026-05-01->2026-05-04"],
    }
    monkeypatch.setattr(autonomous_agent_mod, "PROJECT_DIR", Path(tmp_path))
    monkeypatch.setattr(
        autonomous_agent_mod,
        "get_core_market_data_readiness",
        lambda: dict(readiness),
    )

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent._get_strategy_pool_snapshot = lambda: {}
    agent._summarize_backtest_provenance = lambda rows: {}
    agent._build_backtest_provenance_detail = lambda rows: []
    agent._summarize_catalyst_coverage = lambda rows: {}
    agent._build_catalyst_coverage_detail = lambda rows: []

    agent._save_overnight_state(
        {
            "watchlist": [],
            "data_freshness": {"pm_ready_for_execution": True},
        }
    )

    state = json.loads(
        (Path(tmp_path) / "research" / "overnight_state.json").read_text(
            encoding="utf-8"
        )
    )
    assert "data_freshness" not in state
    assert state["data_freshness_at_overnight_run"] == readiness


def test_day_manager_plan_overlay_uses_canonical_runtime_freshness(monkeypatch):
    canonical = {
        "is_fresh": False,
        "core_data_fresh": False,
        "pm_ready_for_execution": False,
        "primary_date": "2026-05-01",
        "expected_date": "2026-05-04",
        "blocking_reasons": ["core_data_stale:2026-05-01->2026-05-04"],
    }
    stale_plan = {
        "core_data_readiness": {
            "is_fresh": True,
            "core_data_fresh": True,
            "pm_ready_for_execution": True,
            "primary_date": "2026-05-04",
            "expected_date": "2026-05-04",
            "blocking_reasons": [],
        },
        "pm_ready_for_execution": True,
        "core_data_fresh": True,
        "core_data_expected_date": "2026-05-04",
        "blocking_reasons": [],
    }
    monkeypatch.setattr(
        day_manager_mod,
        "get_core_market_data_readiness",
        lambda: dict(canonical),
    )

    dm = DayManager.__new__(DayManager)
    dm._core_data_readiness = {
        "is_fresh": True,
        "core_data_fresh": True,
        "pm_ready_for_execution": True,
        "blocking_reasons": [],
    }
    dm._preopen_execution_contract_block_reason = lambda: ""

    dm._update_core_data_readiness_from_plan(stale_plan)
    assert dm._core_data_readiness["pm_ready_for_execution"] is False
    assert dm._core_data_readiness["expected_date"] == "2026-05-04"
    assert dm._core_data_readiness["blocking_reasons"] == canonical["blocking_reasons"]

    blocked, reason = dm._entries_blocked_by_core_data()
    assert blocked is True
    assert reason == "core_data_stale:2026-05-01->2026-05-04"


def test_market_open_feature_pipeline_failure_logs_freshness_blocker(monkeypatch):
    canonical = {
        "is_fresh": False,
        "core_data_fresh": False,
        "pm_ready_for_execution": False,
        "primary_date": "2026-05-01",
        "expected_date": "2026-05-04",
        "blocking_reasons": ["core_data_stale:2026-05-01->2026-05-04"],
    }
    messages = []
    monkeypatch.setattr(
        day_manager_mod,
        "get_core_market_data_readiness",
        lambda: dict(canonical),
    )
    monkeypatch.setattr(
        day_manager_mod.logger,
        "error",
        lambda msg, *args, **kwargs: messages.append(msg % args),
    )

    dm = DayManager.__new__(DayManager)
    dm._feature_pipeline_health = {
        "status": "failed",
        "last_error": "feature_builder_failed",
        "_logged_market_open_failure": False,
    }
    dm._current_phase_value = lambda phase=None: "market_open"

    dm._log_feature_pipeline_market_open_failure()
    dm._log_feature_pipeline_market_open_failure()

    assert len(messages) == 1
    assert "feature_pipeline_status=failed" in messages[0]
    assert "core_data_stale:2026-05-01->2026-05-04" in messages[0]
