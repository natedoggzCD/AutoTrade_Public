import json

from autotrade.core.agentic_orchestrator import ImprovementAgent, Task, TaskType


def test_performance_review_can_skip_llm(monkeypatch, tmp_path):
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    (plans_dir / "pm_plan_2026-05-10.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-05-10T20:00:00",
                "account": {"equity": 1050},
                "positions": [1],
            }
        ),
        encoding="utf-8",
    )
    (plans_dir / "pm_plan_2026-05-09.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-05-09T20:00:00",
                "account": {"equity": 1000},
                "positions": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    agent = ImprovementAgent()
    monkeypatch.setattr(
        agent,
        "_call_ollama",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("LLM should be skipped")
        ),
    )

    result = agent.execute(
        Task(
            type=TaskType.PERFORMANCE_REVIEW,
            data={"days": 2, "use_llm": False},
        )
    )

    assert result.success is True
    assert result.model_used == "disabled"
    assert result.data["equity_change"] == 50
