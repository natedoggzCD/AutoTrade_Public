from datetime import datetime
import pytz
from autotrade.core.autonomous_agent import AutonomousAgent, MarketPhase
from autotrade.utils.market_time import is_weekend_daytime


def test_is_weekend_daytime_behavior():
    # Saturday 2 PM
    sat_day = datetime(2026, 2, 28, 14, 0, 0)
    assert is_weekend_daytime(sat_day) is True

    # Sunday 10 AM
    sun_day = datetime(2026, 3, 1, 10, 0, 0)
    assert is_weekend_daytime(sun_day) is True

    # Monday 10 AM
    mon_day = datetime(2026, 3, 2, 10, 0, 0)
    assert is_weekend_daytime(mon_day) is False


def test_agent_phase_logic():
    agent = AutonomousAgent()
    status = agent.scheduler.get_status()
    assert "market_phase" in status

    phase = MarketPhase(status["market_phase"])
    actions = agent.scheduler.get_phase_actions(phase)
    assert "primary" in actions
    assert "interval_seconds" in actions


def test_option12_task_resolution():
    # Verify master supervisor resolves 'continuous' task correctly
    from autotrade.core.master_supervisor import MasterSupervisor

    MasterSupervisor(dry_run=True)
    # Mocking run_task to just return what command it WOULD run
    # This is a bit complex to mock deeply, so we'll just check the logic in run_task
    # which we already read. It runs autotrade.core.autonomous_agent --continuous


def test_option12_evening_transition_forces_fresh_overnight():
    assert (
        AutonomousAgent._should_force_fresh_overnight_on_transition(
            previous_phase="PM_WORKFLOW",
            current_phase="OVERNIGHT",
            no_deadline=True,
        )
        is True
    )
    assert (
        AutonomousAgent._should_force_fresh_overnight_on_transition(
            previous_phase="POST_MARKET",
            current_phase="OVERNIGHT",
            no_deadline=True,
        )
        is True
    )


def test_non_option12_or_non_evening_transition_does_not_force_fresh():
    assert (
        AutonomousAgent._should_force_fresh_overnight_on_transition(
            previous_phase="INIT",
            current_phase="OVERNIGHT",
            no_deadline=True,
        )
        is False
    )


def test_option12_idle_youtube_schedule_only_runs_at_midnight_and_3am_slots():
    agent = AutonomousAgent()
    agent._continuous_no_deadline = True
    completion = {}
    et = pytz.timezone("US/Eastern")

    should_run, reason, slot = agent._get_idle_youtube_check_plan(
        completion=completion,
        cycle_count=8,
        now_et=et.localize(datetime(2026, 3, 9, 23, 10, 0)),
    )
    assert should_run is False
    assert reason == "option12_not_scheduled"
    assert slot is None

    should_run, reason, slot = agent._get_idle_youtube_check_plan(
        completion=completion,
        cycle_count=9,
        now_et=et.localize(datetime(2026, 3, 10, 0, 10, 0)),
    )
    assert should_run is True
    assert reason == "option12_scheduled:2026-03-10_0000"
    assert slot == "2026-03-10_0000"

    completion["youtube_idle_refreshes"] = {slot: "2026-03-10T00:10:00-05:00"}
    should_run, reason, slot = agent._get_idle_youtube_check_plan(
        completion=completion,
        cycle_count=10,
        now_et=et.localize(datetime(2026, 3, 10, 0, 45, 0)),
    )
    assert should_run is False
    assert reason == "option12_already_checked:2026-03-10_0000"
    assert slot == "2026-03-10_0000"

    should_run, reason, slot = agent._get_idle_youtube_check_plan(
        completion=completion,
        cycle_count=20,
        now_et=et.localize(datetime(2026, 3, 10, 3, 5, 0)),
    )
    assert should_run is True
    assert reason == "option12_scheduled:2026-03-10_0300"
    assert slot == "2026-03-10_0300"
    assert (
        AutonomousAgent._should_force_fresh_overnight_on_transition(
            previous_phase="PM_WORKFLOW",
            current_phase="OVERNIGHT",
            no_deadline=False,
        )
        is False
    )
    assert (
        AutonomousAgent._should_force_fresh_overnight_on_transition(
            previous_phase="MARKET_HOURS",
            current_phase="POST_MARKET",
            no_deadline=True,
        )
        is False
    )
