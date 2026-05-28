
import pytest
from autotrade.core.autonomous_agent import AutonomousAgent
from autotrade.replay.runtime_session_replay import RuntimeSessionReplay

def test_autonomous_agent_has_resolved_regime_output():
    agent = AutonomousAgent()
    # This should not raise AttributeError
    assert hasattr(agent, "resolved_regime_output")
    assert agent.resolved_regime_output == {}

def test_autonomous_agent_has_regime_strategy_overrides():
    agent = AutonomousAgent()
    # This should not raise AttributeError
    assert hasattr(agent, "regime_strategy_overrides")
    assert agent.regime_strategy_overrides == {}

def test_autonomous_agent_does_not_have_bare_get_current_positions():
    agent = AutonomousAgent()
    # The agent should NOT have this method; it should use self.plan_generator.get_current_positions()
    assert not hasattr(agent, "get_current_positions")

def test_autonomous_agent_resolved_strategy_overrides_no_crash():
    agent = AutonomousAgent()
    # This should work if resolved_regime_output is initialized
    overrides = agent._resolved_strategy_overrides()
    assert isinstance(overrides, dict)

def test_autonomous_agent_gap_hard_cap_pct_no_attr_crash():
    agent = AutonomousAgent()
    result = agent._gap_hard_cap_pct()
    assert isinstance(result, float)

def test_runtime_session_replay_agent_seeds_regime_attrs(tmp_path):
    replay = RuntimeSessionReplay(session_date="2026-03-18", project_dir=tmp_path)
    agent = replay._build_workflow_replay_agent(
        workspace_root=tmp_path,
        artifacts={},
        signals=[],
        decisions=[],
        workflow_journal=[],
        trade_journal=[],
    )
    assert agent.resolved_regime_output == {}
    assert agent.regime_strategy_overrides == {}
    assert agent._effective_market_regime() == "NEUTRAL"
