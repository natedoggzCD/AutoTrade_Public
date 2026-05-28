from types import SimpleNamespace

from autotrade.core.autonomous_agent import AutonomousAgent
from datetime import datetime


def _make_agent() -> AutonomousAgent:
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
    )
    agent._reflect_done_today = True
    agent._pm_plan_done_today = False
    agent.scheduler = SimpleNamespace(
        get_current_time=lambda: datetime(2026, 4, 20, 5, 30, 0)
    )
    return agent


def test_evening_backfill_skips_pm_when_target_pm_plan_exists():
    agent = _make_agent()
    agent._pm_plan_exists_for_target_date = lambda: True

    def _unexpected_pm(cycle_count: int) -> int:
        raise AssertionError(
            "PM backfill should be skipped when target PM plan exists"
        )

    agent._run_pm_workflow_cycle = _unexpected_pm

    agent._run_evening_backfill_if_needed("EARLY_PREMARKET", cycle_count=215)

    assert agent._pm_plan_done_today is True


def test_evening_backfill_runs_pm_when_target_pm_plan_missing_even_if_morning_plan_exists():
    agent = _make_agent()
    agent._pm_plan_exists_for_target_date = lambda: False
    called = {"count": 0}

    def _run_pm(cycle_count: int) -> int:
        called["count"] += 1
        return 180

    agent._run_pm_workflow_cycle = _run_pm

    agent._run_evening_backfill_if_needed("EARLY_PREMARKET", cycle_count=216)

    assert called["count"] == 1
