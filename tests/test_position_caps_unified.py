import logging
from types import SimpleNamespace

import autotrade.core.autonomous_agent as autonomous_agent_mod
from autotrade.core.autonomous_agent import AutonomousAgent
from autotrade.core.day_manager import DayManager
from autotrade.core.position_caps import build_cap_drift_warning


def _portfolio_config():
    return SimpleNamespace(
        portfolio=SimpleNamespace(
            max_positions=50,
            core_max_positions=40,
            late_scan_reserved_positions=10,
        )
    )


def _regime_payload():
    return {
        "regime": "NEUTRAL",
        "max_positions": 50,
        "strategy_overrides": {"max_positions": 20},
        "allow_new_longs": True,
    }


def test_day_manager_and_agent_resolve_same_max_positions(monkeypatch):
    config = _portfolio_config()

    dm = object.__new__(DayManager)
    dm.config = config
    dm.strategy_failsafe_snapshot = SimpleNamespace(max_positions=40)
    dm.regime_router_context = {}
    dm._live_execution_mode = lambda: {"resolved_regime": _regime_payload(), "entries_allowed": True}
    dm._effective_market_regime = lambda: "NEUTRAL"

    agent = object.__new__(AutonomousAgent)
    agent.max_positions = 40
    monkeypatch.setattr(autonomous_agent_mod, "get_config", lambda: config)

    dm_cap = dm._resolve_position_caps().total_cap
    agent_caps = agent._plan_entry_constraints(
        {
            "resolved_regime": _regime_payload(),
            "entry_constraints": {
                "max_positions": 20,
                "core_max_positions": 20,
                "reserve_max_positions": 0,
                "weak_day": False,
                "source": "pm_workflow",
                "regime": "NEUTRAL",
            },
        }
    )

    assert dm_cap == 50
    assert agent_caps["max_positions"] == 50
    assert agent_caps["core_max_positions"] == 40
    assert agent_caps["reserve_max_positions"] == 10


def test_cap_drift_warning_logs_warning_for_small_mismatch(caplog):
    logger = logging.getLogger("autotrade.core.position_caps")
    exec_state = {}

    with caplog.at_level(logging.WARNING, logger="autotrade.core.position_caps"):
        message = build_cap_drift_warning(
            plan_total=50,
            agent_total=48,
            day_manager_total=50,
            source="pm_workflow",
            logger=logger,
            exec_state=exec_state,
        )

    assert message.startswith("[CAP_DRIFT]")
    assert exec_state == {}
    assert any(rec.levelno == logging.WARNING for rec in caplog.records)


def test_cap_drift_warning_flags_serious_mismatch(caplog):
    logger = logging.getLogger("autotrade.core.position_caps")
    exec_state = {}

    with caplog.at_level(logging.ERROR, logger="autotrade.core.position_caps"):
        message = build_cap_drift_warning(
            plan_total=50,
            agent_total=50,
            day_manager_total=65,
            source="pm_workflow",
            logger=logger,
            exec_state=exec_state,
        )

    assert message.startswith("[CAP_DRIFT]")
    assert exec_state["cap_drift_detected"] is True
    assert exec_state["cap_drift_detected_source"] == "pm_workflow"
    assert exec_state["cap_drift_detected_max_diff"] == 15
    assert any(rec.levelno == logging.ERROR for rec in caplog.records)
