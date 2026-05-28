import json
from datetime import date, datetime
from pathlib import Path

import autotrade.core.agentic_orchestrator as orchestrator


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_pm_validator_uses_previous_day_log_when_after_midnight(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    # PM plan for next market day exists (generated during prior evening PM workflow).
    _write_json(
        tmp_path / "plans" / "pm_plan_2026-02-27.json",
        {
            "generated_at": "2026-02-27T00:15:49.137257",
            "positions": [{"symbol": "ABC", "recommended_action": "hold"}],
            "account": {"equity": 100000},
        },
    )

    # Only prior-day PM log exists; this should still validate successfully.
    _write_text(
        tmp_path / "logs" / "pm_workflow_2026-02-26.log",
        "\n".join(
            [
                "2026-02-27 00:01:58,857 | INFO | pm_workflow | PM WORKFLOW - Post-Market Analysis",
                "2026-02-27 00:03:50,227 | INFO | pm_workflow | [FAILSAFE] level=NORMAL",
                "2026-02-27 00:15:49,144 | INFO | pm_workflow | >>> Plan saved: plans/pm_plan_2026-02-27.json",
            ]
        ),
    )

    monkeypatch.setattr(
        orchestrator, "get_market_now", lambda: datetime(2026, 2, 27, 0, 15, 0)
    )
    monkeypatch.setattr(
        orchestrator, "get_pm_plan_date", lambda now=None: date(2026, 2, 27)
    )

    agent = orchestrator.PMValidatorAgent(auto_execute=False)
    task = orchestrator.Task(
        type=orchestrator.TaskType.PM_WORKFLOW_CHECK,
        description="Validate PM workflow",
        data={"auto_execute": False},
    )
    result = agent.execute(task)

    assert result.success is True
    assert "No PM log found" not in result.message
    assert not any(
        "No PM log found" in issue for issue in result.data.get("issues", [])
    )
    assert result.data["log_analysis"]["log_path"].endswith(
        "pm_workflow_2026-02-26.log"
    )


def test_pm_validator_does_not_treat_failsafe_info_as_error(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_text(
        tmp_path / "logs" / "pm_workflow_2026-02-26.log",
        "\n".join(
            [
                "2026-02-26 21:33:58,500 | INFO | pm_workflow | PM WORKFLOW - Post-Market Analysis",
                "2026-02-26 21:39:21,736 | INFO | pm_workflow | [FAILSAFE] level=NORMAL | halt_entries=False",
                "2026-02-26 22:11:06,939 | INFO | pm_workflow | >>> Plan saved: plans/pm_plan_2026-02-27.json",
            ]
        ),
    )

    agent = orchestrator.PMValidatorAgent(auto_execute=False)
    analysis = agent._analyze_pm_logs(candidate_dates=[date(2026, 2, 26)])

    assert analysis["exists"] is True
    assert analysis["issues"] == []


def test_pm_validator_rejects_only_picks_with_invalid_stops(monkeypatch):
    agent = orchestrator.PMValidatorAgent(auto_execute=False)
    monkeypatch.setattr(
        agent, "_call_ollama", lambda *args, **kwargs: "structured veto"
    )
    task = orchestrator.Task(
        type=orchestrator.TaskType.PICKS_VALIDATION,
        description="Validate premarket picks",
        data={
            "picks": [
                {
                    "symbol": "BAD",
                    "entry_price": 10.0,
                    "stop_loss": 0.0,
                    "confidence": 80,
                },
                {
                    "symbol": "GOOD",
                    "entry_price": 20.0,
                    "stop_loss": 18.0,
                    "confidence": 82,
                },
            ],
            "pick_count": 2,
            "cleanup_count": 0,
            "plan_date": "2026-05-12",
            "market_phase": "PREMARKET",
        },
    )

    result = agent.execute(task)

    assert result.success is False
    assert result.data["verdict"] == "reject_partial"
    assert result.data["affected_symbols"] == ["BAD"]
    assert "BAD: Missing stop loss" in result.data["risk_flags"]


def test_pm_validator_low_confidence_is_warning_only(monkeypatch):
    agent = orchestrator.PMValidatorAgent(auto_execute=False)
    monkeypatch.setattr(agent, "_call_ollama", lambda *args, **kwargs: "warning only")
    task = orchestrator.Task(
        type=orchestrator.TaskType.PICKS_VALIDATION,
        description="Validate premarket picks",
        data={
            "picks": [
                {
                    "symbol": "LOW",
                    "entry_price": 10.0,
                    "stop_loss": 9.0,
                    "confidence": 40,
                }
            ],
            "pick_count": 1,
            "cleanup_count": 0,
            "plan_date": "2026-05-12",
            "market_phase": "PREMARKET",
        },
    )

    result = agent.execute(task)

    assert result.success is True
    assert result.data["verdict"] == "approve_with_warnings"
    assert result.data["affected_symbols"] == []
