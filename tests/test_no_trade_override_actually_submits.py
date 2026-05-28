from datetime import datetime, timedelta
from types import SimpleNamespace

from autotrade.core import autonomous_agent as autonomous_agent_mod
from autotrade.core.autonomous_agent import AutonomousAgent


class _LoggerStub:
    def __init__(self):
        self.messages = {"info": [], "warning": [], "error": []}

    def info(self, *args, **kwargs):
        self.messages["info"].append(args[0] % args[1:] if len(args) > 1 else args[0])

    def warning(self, *args, **kwargs):
        self.messages["warning"].append(
            args[0] % args[1:] if len(args) > 1 else args[0]
        )

    def error(self, *args, **kwargs):
        self.messages["error"].append(args[0] % args[1:] if len(args) > 1 else args[0])


def _agent_for_override(monkeypatch, now_ts):
    cfg = SimpleNamespace(
        strategy_failsafe=SimpleNamespace(
            no_trade_override_enabled=True,
            no_trade_eval_minutes=1,
            no_trade_override_cooldown_minutes=1,
        )
    )
    monkeypatch.setattr(autonomous_agent_mod, "get_config", lambda: cfg)

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent.workflow_health = {}
    agent._runtime_entry_caps_override = {}
    agent._no_trade_monitor = {
        "first_actionable_no_submit_at": (now_ts - timedelta(minutes=5)).isoformat(),
        "last_override_at": "",
    }
    agent._parse_iso_timestamp = AutonomousAgent._parse_iso_timestamp
    agent._apply_market_hours_failsafe_override = lambda **kwargs: True
    agent._compute_no_trade_runtime_override = lambda **kwargs: {}
    return agent


def test_no_trade_override_invokes_immediate_recheck(monkeypatch):
    now_ts = datetime(2026, 5, 1, 10, 45)
    agent = _agent_for_override(monkeypatch, now_ts)
    calls = []

    def _run_day_manager_cycle(**kwargs):
        calls.append(kwargs)
        return {
            "entries_submitted": 1,
            "candidate_count": 2,
            "open_slots": 1,
            "entry_audit": {
                "candidate_count": 2,
                "open_slots": 1,
                "submitted": 1,
                "blocked_by_reason": {},
            },
        }

    agent.run_day_manager_cycle = _run_day_manager_cycle

    result = agent._maybe_trigger_no_trade_override(
        phase="market_hours",
        now_ts=now_ts,
        entry_constraints={"max_positions": 10},
        candidate_count=2,
        open_slots=1,
        result={"entries_submitted": 0},
        dry_run=False,
        cycle_kwargs={"override_reason": ""},
        already_attempted=False,
    )

    assert result["entries_submitted"] == 1
    assert calls
    assert calls[0]["_no_trade_override_attempted"] is True
    assert calls[0]["reset_signals"] is True


def test_no_trade_override_ignores_synthetic_no_reason_blocker(monkeypatch):
    now_ts = datetime(2026, 5, 1, 10, 45)
    agent = _agent_for_override(monkeypatch, now_ts)
    calls = []

    def _run_day_manager_cycle(**kwargs):
        calls.append(kwargs)
        return {
            "entries_submitted": 1,
            "candidate_count": 2,
            "open_slots": 1,
            "entry_audit": {
                "candidate_count": 2,
                "open_slots": 1,
                "submitted": 1,
                "blocked_by_reason": {},
            },
        }

    agent.run_day_manager_cycle = _run_day_manager_cycle

    result = agent._maybe_trigger_no_trade_override(
        phase="market_hours",
        now_ts=now_ts,
        entry_constraints={"max_positions": 10},
        candidate_count=2,
        open_slots=1,
        result={
            "entries_submitted": 0,
            "block_reason": "entry_submission_failed_no_reason",
            "entry_audit": {
                "blocked_by_reason": {"entry_submission_failed_no_reason": 2},
                "block_reason": "entry_submission_failed_no_reason",
            },
        },
        dry_run=False,
        cycle_kwargs={},
        already_attempted=False,
    )

    assert result["entries_submitted"] == 1
    assert calls


def test_no_trade_override_recheck_accepts_concrete_block_reasons(monkeypatch):
    now_ts = datetime(2026, 5, 1, 10, 45)
    agent = _agent_for_override(monkeypatch, now_ts)
    audit = {
        "candidate_count": 2,
        "open_slots": 1,
        "submitted": 0,
        "blocked_by_reason": {"pending_buy_order": 2},
        "block_reason": "",
    }
    agent.run_day_manager_cycle = lambda **kwargs: {
        "entries_submitted": 0,
        "candidate_count": 2,
        "open_slots": 1,
        "blocked_by_reason": {"pending_buy_order": 2},
        "entry_audit": audit,
    }

    result = agent._maybe_trigger_no_trade_override(
        phase="market_hours",
        now_ts=now_ts,
        entry_constraints={"max_positions": 10},
        candidate_count=2,
        open_slots=1,
        result={"entries_submitted": 0},
        dry_run=False,
        cycle_kwargs={},
        already_attempted=False,
    )

    assert "override_recheck_no_submit" in result
    assert result.get("diagnostic_failure") is not True
    assert result["override_recheck_no_submit"]["blocked_by_reason"] == {
        "pending_buy_order": 2
    }


def test_no_trade_override_recheck_fails_zero_submit_without_block_reasons(monkeypatch):
    now_ts = datetime(2026, 5, 1, 10, 45)
    agent = _agent_for_override(monkeypatch, now_ts)
    audit = {
        "candidate_count": 2,
        "open_slots": 1,
        "max_new_entries": 1,
        "submitted": 0,
        "blocked_by_reason": {},
        "block_reason": "",
    }
    agent.run_day_manager_cycle = lambda **kwargs: {
        "entries_submitted": 0,
        "candidate_count": 2,
        "open_slots": 1,
        "entry_audit": audit,
    }

    result = agent._maybe_trigger_no_trade_override(
        phase="market_hours",
        now_ts=now_ts,
        entry_constraints={"max_positions": 10},
        candidate_count=2,
        open_slots=1,
        result={"entries_submitted": 0},
        dry_run=False,
        cycle_kwargs={},
        already_attempted=False,
    )

    assert result["error"] == "override_recheck_no_submit"
    assert result["diagnostic_failure"] is True
    assert result["override_recheck_no_submit"]["entry_audit"] == audit
