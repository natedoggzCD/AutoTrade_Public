import json
from datetime import datetime, timedelta
import sys
from types import SimpleNamespace

import pytest

from autotrade.core.workflow_manager import WorkflowConfig, WorkflowManager, WorkflowPhase
from autotrade.utils.market_time import get_pm_plan_date


def _manager(tmp_path):
    config = WorkflowConfig(project_dir=tmp_path, dry_run=True, validation_mode=True)
    return WorkflowManager(
        agent=SimpleNamespace(),
        scheduler=SimpleNamespace(),
        config=config,
    )


def test_pm_flag_carries_across_midnight_when_recent_pm_success(tmp_path):
    manager = _manager(tmp_path)
    now = datetime.now()
    manager.state.last_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    manager.state.daily_flags = {"pm_workflow_ran": True}
    manager.state.phase_history = [
        {
            "phase": "pm_workflow",
            "success": True,
            "ended_at": (now - timedelta(hours=8)).isoformat(),
        }
    ]

    manager._reset_daily_flags(now.strftime("%Y-%m-%d"))

    assert manager.state.daily_flags["pm_workflow_ran"] is True


def test_pm_flag_not_carried_when_recent_success_is_too_old(tmp_path):
    manager = _manager(tmp_path)
    now = datetime.now()
    manager.state.last_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    manager.state.daily_flags = {"pm_workflow_ran": True}
    manager.state.phase_history = [
        {
            "phase": "pm_workflow",
            "success": True,
            "ended_at": (now - timedelta(hours=30)).isoformat(),
        }
    ]

    manager._reset_daily_flags(now.strftime("%Y-%m-%d"))

    assert manager.state.daily_flags["pm_workflow_ran"] is False


def test_pm_flag_not_carried_when_last_date_skips_more_than_one_day(tmp_path):
    manager = _manager(tmp_path)
    now = datetime.now()
    manager.state.last_date = (now - timedelta(days=2)).strftime("%Y-%m-%d")
    manager.state.daily_flags = {"pm_workflow_ran": True}
    manager.state.phase_history = [
        {
            "phase": "pm_workflow",
            "success": True,
            "ended_at": (now - timedelta(hours=8)).isoformat(),
        }
    ]

    manager._reset_daily_flags(now.strftime("%Y-%m-%d"))

    assert manager.state.daily_flags["pm_workflow_ran"] is False


def test_resume_normalizes_stale_heartbeat_timestamps(tmp_path):
    manager = _manager(tmp_path)
    stale = datetime.now() - timedelta(days=2)
    manager.state.current_phase = "post_market"
    manager.state.last_checkpoint_at = stale.isoformat()
    manager.state.phase_entered_at = stale.isoformat()
    manager._run_loop = lambda: None

    manager.resume()

    normalized_checkpoint = datetime.fromisoformat(manager.state.last_checkpoint_at)
    normalized_phase = datetime.fromisoformat(manager.state.phase_entered_at)
    assert normalized_checkpoint > stale
    assert normalized_phase > stale


def test_resume_keeps_recent_heartbeat_timestamps(tmp_path):
    manager = _manager(tmp_path)
    recent = datetime.now() - timedelta(seconds=30)
    manager.state.current_phase = "post_market"
    manager.state.last_checkpoint_at = recent.isoformat()
    manager.state.phase_entered_at = recent.isoformat()
    manager._run_loop = lambda: None

    manager.resume()

    assert manager.state.last_checkpoint_at == recent.isoformat()
    assert manager.state.phase_entered_at == recent.isoformat()


def test_self_heal_stalled_failed_phase_resets_failure_budget(tmp_path):
    manager = _manager(tmp_path)
    stale = datetime.now() - timedelta(minutes=25)
    manager.state.current_phase = WorkflowPhase.MARKET_OPEN.value
    manager.state.phase_entered_at = stale.isoformat()
    manager.state.phase_failures = {WorkflowPhase.MARKET_OPEN.value: 2}
    manager.state.cooldown_until = (datetime.now() + timedelta(minutes=10)).isoformat()

    manager._self_heal_stalled_phases()

    assert manager.state.phase_failures[WorkflowPhase.MARKET_OPEN.value] == 0
    assert manager.state.cooldown_until is None
    assert datetime.fromisoformat(manager.state.phase_entered_at) > stale


def test_state_store_prunes_stale_issues_on_load(tmp_path):
    manager = _manager(tmp_path)
    stale = (datetime.now() - timedelta(days=20)).isoformat()
    fresh = (datetime.now() - timedelta(hours=2)).isoformat()
    manager.config.state_path.write_text(
        json.dumps(
            {
                "current_phase": "post_market",
                "issues": [
                    {"type": "phase_stalled", "timestamp": stale},
                    {"type": "checkpoint_stalled", "timestamp": fresh},
                ],
            }
        ),
        encoding="utf-8",
    )

    reloaded = WorkflowManager(
        agent=SimpleNamespace(),
        scheduler=SimpleNamespace(),
        config=manager.config,
    )

    assert reloaded.state.issues == [{"type": "checkpoint_stalled", "timestamp": fresh}]


def test_post_market_triggers_pm_workflow_when_pending(tmp_path):
    manager = _manager(tmp_path)
    manager.state.daily_flags = {"pm_workflow_ran": False}
    manager._is_pm_workflow_local_window_open = lambda now=None: True
    calls = {"pm": 0}

    def _fake_pm():
        calls["pm"] += 1
        manager.state.daily_flags["pm_workflow_ran"] = True
        return {}

    manager._run_pm_workflow = _fake_pm

    result = manager._run_post_market_idle()

    assert calls["pm"] == 1
    assert result["pm_workflow_triggered"] is True
    assert result["pm_workflow_error"] is False


def test_post_market_stays_idle_after_pm_workflow_done(tmp_path):
    manager = _manager(tmp_path)
    manager.state.daily_flags = {"pm_workflow_ran": True}
    manager._run_pm_workflow = lambda: pytest.fail(
        "PM workflow should not run when already complete"
    )

    result = manager._run_post_market_idle()

    assert result == {"idle": True}


def test_post_market_waits_until_local_start_when_pending(tmp_path):
    manager = _manager(tmp_path)
    manager.state.daily_flags = {"pm_workflow_ran": False}
    manager._is_pm_workflow_local_window_open = lambda now=None: False
    manager._run_pm_workflow = lambda: pytest.fail(
        "PM workflow should not run before local start time"
    )

    result = manager._run_post_market_idle()

    assert result["idle"] is True
    assert result["pm_workflow_pending"] is True
    assert result["pm_workflow_local_start"] == "18:30"


def test_run_premarket_raises_on_error_payload_and_preserves_flag(tmp_path):
    config = WorkflowConfig(project_dir=tmp_path, dry_run=True, validation_mode=True)
    manager = WorkflowManager(
        agent=SimpleNamespace(run_premarket_scan=lambda dry_run: {"error": "boom"}),
        scheduler=SimpleNamespace(),
        config=config,
    )
    manager.state.daily_flags = {"premarket_ran": False}

    with pytest.raises(RuntimeError, match="premarket_scan returned error"):
        manager._run_premarket()

    assert manager.state.daily_flags["premarket_ran"] is False


def test_run_day_manager_raises_on_error_payload(tmp_path):
    config = WorkflowConfig(project_dir=tmp_path, dry_run=True, validation_mode=True)
    manager = WorkflowManager(
        agent=SimpleNamespace(run_day_manager_cycle=lambda dry_run: {"error": "dm fail"}),
        scheduler=SimpleNamespace(),
        config=config,
    )

    with pytest.raises(RuntimeError, match="day_manager_cycle returned error"):
        manager._run_day_manager()


def test_run_pm_workflow_raises_on_error_payload_and_preserves_flag(
    tmp_path, monkeypatch
):
    config = WorkflowConfig(project_dir=tmp_path, dry_run=True, validation_mode=True)
    manager = WorkflowManager(
        agent=SimpleNamespace(run_pm_workflow=lambda dry_run: {"error": "pm fail"}),
        scheduler=SimpleNamespace(),
        config=config,
    )
    manager.state.daily_flags = {"pm_workflow_ran": False}

    monkeypatch.setitem(
        sys.modules,
        "autotrade.utils.youtube_readiness",
        SimpleNamespace(
            ensure_youtube_ready=lambda: {"post_check": {}, "scan_ran": False},
            format_readiness_log=lambda status: "[YOUTUBE] ready",
        ),
    )

    with pytest.raises(RuntimeError, match="pm_workflow returned error"):
        manager._run_pm_workflow()

    assert manager.state.daily_flags["pm_workflow_ran"] is False


def test_run_overnight_raises_when_pm_catchup_fails_and_plan_missing(tmp_path):
    config = WorkflowConfig(project_dir=tmp_path, dry_run=True, validation_mode=True)
    overnight_engine = SimpleNamespace(
        run_full_overnight_cycle=lambda pm_plan_path: pytest.fail(
            "overnight should not run without a PM plan"
        )
    )
    manager = WorkflowManager(
        agent=SimpleNamespace(overnight_engine=overnight_engine),
        scheduler=SimpleNamespace(),
        config=config,
    )
    manager.state.daily_flags = {"pm_workflow_ran": False}
    manager._run_pm_workflow = lambda: (_ for _ in ()).throw(RuntimeError("pm boom"))

    with pytest.raises(
        RuntimeError,
        match=r"PM workflow catch-up failed and pm_plan_.* is still unusable",
    ):
        manager._run_overnight()


def test_run_overnight_recovers_when_flag_says_done_but_pm_plan_missing(tmp_path):
    config = WorkflowConfig(project_dir=tmp_path, dry_run=True, validation_mode=True)
    calls = {"pm": 0, "overnight": 0}
    pm_plan_path = (
        tmp_path / "data" / "dry_run" / "plans" / f"pm_plan_{get_pm_plan_date().strftime('%Y-%m-%d')}.json"
    )

    def _fake_pm():
        calls["pm"] += 1
        pm_plan_path.parent.mkdir(parents=True, exist_ok=True)
        pm_plan_path.write_text('{"signals": [{"symbol": "ABC"}]}', encoding="utf-8")
        manager.state.daily_flags["pm_workflow_ran"] = True
        return {"signals": [{"symbol": "ABC"}]}

    def _fake_overnight(path):
        calls["overnight"] += 1
        assert path == pm_plan_path
        return {"status": "ok"}

    agent = SimpleNamespace(
        overnight_engine=SimpleNamespace(run_full_overnight_cycle=_fake_overnight),
    )
    manager = WorkflowManager(
        agent=agent,
        scheduler=SimpleNamespace(),
        config=config,
    )
    manager.state.daily_flags = {"pm_workflow_ran": True}
    manager._run_pm_workflow = _fake_pm

    result = manager._run_overnight()

    assert calls["pm"] == 1
    assert calls["overnight"] == 1
    assert result["status"] == "ok"


def test_run_overnight_recovers_when_pm_plan_exists_but_is_empty(tmp_path):
    config = WorkflowConfig(project_dir=tmp_path, dry_run=True, validation_mode=True)
    calls = {"pm": 0, "overnight": 0}
    plan_date = get_pm_plan_date()
    plan_dir = tmp_path / "data" / "dry_run" / "plans"
    pm_plan_path = plan_dir / f"pm_plan_{plan_date.strftime('%Y-%m-%d')}.json"
    morning_plan_path = plan_dir / f"morning_game_plan_{plan_date.strftime('%Y%m%d')}.json"
    plan_dir.mkdir(parents=True, exist_ok=True)
    pm_plan_path.write_text('{"signals": []}', encoding="utf-8")
    morning_plan_path.write_text(
        '{"signals": [{"symbol": "ABC", "score": 81.0}]}',
        encoding="utf-8",
    )

    def _fake_pm():
        calls["pm"] += 1
        pm_plan_path.write_text(
            morning_plan_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        manager.state.daily_flags["pm_workflow_ran"] = True
        return {"signals": [{"symbol": "ABC", "score": 81.0}]}

    def _fake_overnight(path):
        calls["overnight"] += 1
        assert path == pm_plan_path
        return {"status": "ok"}

    agent = SimpleNamespace(
        overnight_engine=SimpleNamespace(run_full_overnight_cycle=_fake_overnight),
    )
    manager = WorkflowManager(
        agent=agent,
        scheduler=SimpleNamespace(),
        config=config,
    )
    manager.state.daily_flags = {"pm_workflow_ran": True}
    manager._run_pm_workflow = _fake_pm

    result = manager._run_overnight()

    assert calls["pm"] == 1
    assert calls["overnight"] == 1
    assert result["status"] == "ok"
