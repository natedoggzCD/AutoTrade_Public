import autotrade.core.autonomous_agent as agent_mod
from autotrade.core.autonomous_agent import AutonomousAgent


class _Logger:
    def __init__(self):
        self.critical_calls = []
        self.warning_calls = []

    def critical(self, *args, **kwargs):
        self.critical_calls.append((args, kwargs))

    def warning(self, *args, **kwargs):
        self.warning_calls.append((args, kwargs))


def _agent(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "DATA_DIR", tmp_path)
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _Logger()
    agent._premarket_state = {}
    agent._validator_veto_cache_session_end = (
        lambda: agent_mod.datetime.now() + agent_mod.timedelta(hours=1)
    )
    return agent


def test_invalid_stop_validator_veto_is_cached_and_repeat_is_warning(
    tmp_path, monkeypatch
):
    agent = _agent(tmp_path, monkeypatch)
    plan = {
        "entry_orders": [{"symbol": "RGC"}, {"symbol": "SAFE"}],
        "cleanup_orders": [],
        "review_positions": [],
    }
    decision = {
        "verdict": "reject_partial",
        "affected_symbols": ["RGC"],
        "reasons": ["RGC: Invalid stop loss"],
    }

    first = agent._apply_pm_validator_verdict(
        plan, decision, validation_hash="hash-one"
    )

    assert first["removed_symbols"] == ["RGC"]
    assert (tmp_path / "validator_veto_cache.json").exists()
    assert len(agent.logger.critical_calls) == 1

    second_plan = {
        "entry_orders": [{"symbol": "RGC"}, {"symbol": "SAFE"}],
        "cleanup_orders": [],
        "review_positions": [],
    }
    removed = agent._filter_validator_vetoed_entry_orders(second_plan)

    assert removed == ["RGC"]
    assert second_plan["entry_orders"] == [{"symbol": "SAFE"}]

    repeated_plan = {
        "entry_orders": [{"symbol": "RGC"}],
        "cleanup_orders": [],
        "review_positions": [],
    }
    agent._apply_pm_validator_verdict(
        repeated_plan, decision, validation_hash="hash-two"
    )

    assert len(agent.logger.critical_calls) == 1
    assert agent.logger.warning_calls
