import os
import sys
from types import SimpleNamespace

import pytest

if os.getenv("AUTOTRADE_RUN_HEAVY_TESTS") != "1":
    pytest.skip(
        "Skipping slow autonomous agent workflow tests; set AUTOTRADE_RUN_HEAVY_TESTS=1 to run.",
        allow_module_level=True,
    )

from autotrade.core.autonomous_agent import AgentState, AutonomousAgent


class _LogCapture:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.errors = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(str(msg) % args if args else str(msg))

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(str(msg) % args if args else str(msg))

    def error(self, msg, *args, **kwargs):
        self.errors.append(str(msg) % args if args else str(msg))


def _make_agent():
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LogCapture()
    agent.state = AgentState.IDLE
    agent.error_diagnoser = SimpleNamespace(
        diagnose_error=lambda exc, ctx: {"diagnosis": {"issue": f"{ctx}: {exc}"}}
    )
    agent._ensure_ollama_for_phase = lambda phase: None
    agent._record_ollama_result = lambda *args, **kwargs: None
    agent._check_ollama_health = lambda require_models=False: {"ok": True}
    return agent


def test_run_pm_workflow_uses_pmworkflow_pipeline(monkeypatch):
    agent = _make_agent()
    called = {"pm": 0, "review": 0}

    class _PMWorkflow:
        def __init__(self, dry_run):
            self.dry_run = dry_run

        def run(self):
            called["pm"] += 1
            return {
                "signals": [
                    {
                        "symbol": "ABC",
                        "strategy_name": "strat_a",
                        "setup_type": "momentum",
                        "symbol_strategy_rank": 1,
                        "symbol_strategy_source": "per_symbol",
                    }
                ]
            }

    class _DailyReview:
        def run(self):
            called["review"] += 1
            return {"day_summary": {"grade": "OK"}}

    monkeypatch.setitem(
        sys.modules,
        "autotrade.core.pm_workflow",
        SimpleNamespace(PMWorkflow=_PMWorkflow),
    )
    monkeypatch.setitem(
        sys.modules,
        "autotrade.core.daily_review",
        SimpleNamespace(DailyReview=_DailyReview),
    )

    result = agent.run_pm_workflow(dry_run=True)

    assert called["pm"] == 1
    assert called["review"] == 1
    assert isinstance(result.get("signals"), list)
    assert result["signals"][0]["symbol"] == "ABC"
    assert result["signals"][0]["symbol_strategy_rank"] == 1
    assert result["signals"][0]["symbol_strategy_source"] == "per_symbol"
    assert "daily_review" in result
    assert agent.state == AgentState.IDLE


def test_run_pm_workflow_keeps_pm_result_when_daily_review_fails(monkeypatch):
    agent = _make_agent()
    called = {"pm": 0}

    class _PMWorkflow:
        def __init__(self, dry_run):
            self.dry_run = dry_run

        def run(self):
            called["pm"] += 1
            return {"signals": [{"symbol": "XYZ"}]}

    class _DailyReview:
        def __init__(self):
            raise RuntimeError("review boom")

    monkeypatch.setitem(
        sys.modules,
        "autotrade.core.pm_workflow",
        SimpleNamespace(PMWorkflow=_PMWorkflow),
    )
    monkeypatch.setitem(
        sys.modules,
        "autotrade.core.daily_review",
        SimpleNamespace(DailyReview=_DailyReview),
    )

    result = agent.run_pm_workflow(dry_run=False)

    assert called["pm"] == 1
    assert result["signals"][0]["symbol"] == "XYZ"
    assert "daily_review" not in result
    assert any("Daily review failed" in msg for msg in agent.logger.warnings)
    assert agent.state == AgentState.IDLE
