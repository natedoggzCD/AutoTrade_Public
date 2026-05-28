from datetime import datetime, timedelta
from types import SimpleNamespace

import autotrade.core.autonomous_agent as autonomous_agent_mod
from autotrade.core.autonomous_agent import AutonomousAgent


class _LoggerStub:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None


def _config_stub():
    return SimpleNamespace(
        portfolio=SimpleNamespace(
            max_positions=50,
            core_max_positions=40,
            late_scan_reserved_positions=10,
        ),
        strategy_failsafe=SimpleNamespace(
            no_trade_eval_minutes=60,
            no_trade_override_enabled=True,
            no_trade_override_cooldown_minutes=60,
            crash_regime_reduced_caps_enabled=True,
        ),
    )


def _build_agent(monkeypatch):
    monkeypatch.setattr(autonomous_agent_mod, "get_config", lambda: _config_stub())
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent.max_positions = 50
    agent.core_max_positions = 40
    agent.reserve_max_positions = 10
    agent.weak_day = False
    agent.position_caps = {}
    agent.resolved_regime_output = {}
    agent.strategy_failsafe_snapshot = SimpleNamespace(
        max_positions=40,
        halt_new_entries=False,
        level="normal",
        validation_status="HEALTHY",
    )
    agent._runtime_entry_caps_override = {}
    agent._no_trade_monitor = {
        "first_actionable_no_submit_at": "",
        "last_submitted_entry_at": "",
        "last_override_at": "",
        "last_override_reason": "",
        "last_override_cycle_phase": "",
    }
    agent.workflow_health = {}
    return agent


def test_entry_cap_contract_prefers_runtime_override(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent._runtime_entry_caps_override = {
        "max_positions": 7,
        "core_max_positions": 5,
        "reserve_max_positions": 2,
    }
    plan = {
        "resolved_regime": {
            "regime": "NEUTRAL",
            "max_positions": 30,
            "allow_new_longs": True,
        },
        "entry_constraints": {
            "max_positions": 20,
            "core_max_positions": 18,
            "reserve_max_positions": 2,
        },
    }

    resolved = agent._apply_plan_entry_constraints(plan, source="unit_test")

    assert resolved["max_positions"] == 7
    assert resolved["core_max_positions"] == 5
    assert resolved["reserve_max_positions"] == 2
    assert agent.max_positions == 7
    assert "runtime_override" in resolved["precedence_applied"]


def test_no_trade_override_triggers_single_rerun(monkeypatch):
    agent = _build_agent(monkeypatch)
    now_ts = datetime(2026, 4, 30, 11, 0, 0)
    agent._no_trade_monitor["first_actionable_no_submit_at"] = (
        now_ts - timedelta(minutes=61)
    ).isoformat()

    rerun_calls = []
    agent._apply_market_hours_failsafe_override = lambda **kwargs: True
    agent._compute_no_trade_runtime_override = lambda **kwargs: {}
    agent.run_day_manager_cycle = (
        lambda **kwargs: rerun_calls.append(dict(kwargs))
        or {"entries": 1, "orders_submitted": 1}
    )

    override_result = agent._maybe_trigger_no_trade_override(
        phase="market_hours",
        now_ts=now_ts,
        entry_constraints={
            "max_positions": 8,
            "core_max_positions": 6,
            "reserve_max_positions": 2,
            "weak_day": False,
        },
        candidate_count=5,
        open_slots=2.0,
        result={"entries": 0, "orders_submitted": 0, "block_reason": ""},
        dry_run=True,
        cycle_kwargs={
            "candidate_universe_rows": [{"symbol": "AAA"}],
            "override_reason": "",
            "reset_signals": False,
            "max_new_entries_override": None,
            "entry_wave_override": None,
            "plan_payload": {"resolved_regime": {"regime": "NEUTRAL"}},
        },
        already_attempted=False,
    )

    assert isinstance(override_result, dict)
    assert override_result["orders_submitted"] == 1
    assert len(rerun_calls) == 1
    assert rerun_calls[0]["_no_trade_override_attempted"] is True
    assert rerun_calls[0]["reset_signals"] is True
    assert rerun_calls[0]["market_phase"] == "market_hours"


def test_no_trade_override_skips_without_capacity(monkeypatch):
    agent = _build_agent(monkeypatch)
    now_ts = datetime(2026, 4, 30, 11, 0, 0)
    agent._no_trade_monitor["first_actionable_no_submit_at"] = (
        now_ts - timedelta(minutes=61)
    ).isoformat()

    rerun_calls = []
    agent.run_day_manager_cycle = lambda **kwargs: rerun_calls.append(kwargs) or {}
    agent._apply_market_hours_failsafe_override = lambda **kwargs: True

    override_result = agent._maybe_trigger_no_trade_override(
        phase="market_hours",
        now_ts=now_ts,
        entry_constraints={"max_positions": 8, "core_max_positions": 6, "reserve_max_positions": 2},
        candidate_count=0,
        open_slots=3.0,
        result={"entries": 0, "orders_submitted": 0, "block_reason": ""},
        dry_run=True,
        cycle_kwargs={"plan_payload": {"resolved_regime": {"regime": "NEUTRAL"}}},
        already_attempted=False,
    )

    assert override_result is None
    assert rerun_calls == []


def test_crash_regime_uses_reduced_runtime_caps(monkeypatch):
    agent = _build_agent(monkeypatch)

    reduced = agent._compute_no_trade_runtime_override(
        entry_constraints={
            "max_positions": 8,
            "core_max_positions": 6,
            "reserve_max_positions": 2,
        },
        plan_payload={"resolved_regime": {"regime": "CRASH"}},
    )

    assert reduced["max_positions"] >= 1
    assert reduced["max_positions"] < 8
    assert reduced["weak_day"] is True
