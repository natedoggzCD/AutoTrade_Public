import json
from pathlib import Path
from types import SimpleNamespace

from autotrade.replay.decision_claw_historical_eval import (
    DecisionClawHistoricalEvaluator,
    _extract_failure_notes,
    score_decision_claw_result,
)
from autotrade.core.decision_claw import DecisionClawAction, DecisionClawResult


def test_extract_failure_notes_keeps_actionable_lines():
    markdown = """
    # Daily Review

    1. CRITICAL - Trade journal reconciliation gap
    2. WARNING - Missed opportunities because names were not in waves
    - INFO - harmless status line
    - Stale watchlist caused no entries after 10:00
    """

    notes = _extract_failure_notes(markdown)

    assert "CRITICAL - Trade journal reconciliation gap" in notes
    assert "WARNING - Missed opportunities because names were not in waves" in notes
    assert "Stale watchlist caused no entries after 10:00" in notes


def test_historical_evaluator_builds_scenario_from_realistic_artifacts(
    tmp_path: Path, monkeypatch
):
    (tmp_path / "data").mkdir()
    (tmp_path / "reports" / "Claude").mkdir(parents=True)
    (tmp_path / "logs").mkdir()

    (tmp_path / "data" / "eod_review_2026-03-23.json").write_text(
        json.dumps(
            {
                "trades": [
                    {
                        "symbol": "OLD1",
                        "qty": 25,
                        "current_price": 8.0,
                        "unrealized_plpc": -0.08,
                        "signal_entry_source": "baseline",
                    },
                    {
                        "symbol": "OLD2",
                        "qty": 10,
                        "current_price": 20.0,
                        "unrealized_plpc": -0.04,
                        "signal_entry_source": "baseline",
                    },
                ],
                "missed_watchlist_opportunities": [
                    {
                        "symbol": "GLNG",
                        "validated_score": 91.0,
                        "entry_source": "overnight_full_watchlist",
                        "blocking_reason": "wave_capacity",
                        "ranking_position": 2,
                    },
                    {
                        "symbol": "YPF",
                        "validated_score": 84.0,
                        "entry_source": "overnight_full_watchlist",
                        "blocking_reason": "below_min_score",
                        "ranking_position": 5,
                    },
                ],
                "watchlist_causality_summary": {"blocked": 12},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "reports" / "Claude" / "daily_review_2026-03-23.md").write_text(
        "1. CRITICAL - Trade journal reconciliation gap\n2. WARNING - stale watchlist and missed names\n",
        encoding="utf-8",
    )
    (tmp_path / "logs" / "trade_decisions_20260323.json").write_text(
        json.dumps(
            [
                {"symbol": "OLD1", "actually_executed": True},
                {"symbol": "OLD2", "actually_executed": True},
            ]
        ),
        encoding="utf-8",
    )

    class _ReplayStub:
        def __init__(self, *args, **kwargs):
            return None

        def run(self, persist=True):
            return {
                "divergences_count": 7,
                "handoff_diagnostics": {
                    "status": "degraded",
                    "recoverable_candidates_examples": ["GLNG", "YPF"],
                },
                "actual_day_context": {"total_trades": 8, "net_pnl": -123.45},
            }

    monkeypatch.setattr(
        "autotrade.replay.decision_claw_historical_eval.RuntimeSessionReplay",
        _ReplayStub,
    )

    evaluator = DecisionClawHistoricalEvaluator(project_dir=tmp_path)
    scenario = evaluator.build_scenario(session_date="2026-03-23")

    assert scenario.missed_symbols[:2] == ["GLNG", "YPF"]
    assert scenario.weak_symbols[:2] == ["OLD1", "OLD2"]
    assert scenario.payload["blocked_candidates"] == 12
    assert scenario.payload["actual_net_pnl"] == -123.45
    assert scenario.legacy_recommendation["legacy_choice"] == ["OLD1", "OLD2"]
    assert scenario.failure_notes


def test_score_decision_claw_result_rewards_missed_and_weak_hits():
    class _Scenario:
        missed_symbols = ["GLNG", "YPF"]
        weak_symbols = ["OLD1", "OLD2"]

    result = DecisionClawResult(
        phase_agent="market_state",
        trigger="historical",
        decision="override",
        confidence=0.9,
        actions=[
            DecisionClawAction(action_type="promote_symbol", symbol="GLNG"),
            DecisionClawAction(action_type="trim_position", symbol="OLD1"),
        ],
    )

    score = score_decision_claw_result(result, scenario=_Scenario())

    assert score["missed_hits"] == ["GLNG"]
    assert score["weak_hits"] == ["OLD1"]
    assert score["total_score"] > 0.5


def test_historical_evaluator_aggregates_provider_results(tmp_path: Path, monkeypatch):
    (tmp_path / "logs").mkdir()

    evaluator = DecisionClawHistoricalEvaluator(project_dir=tmp_path)

    def _fake_build_scenario(*, session_date: str, phase_agent: str = "market_state"):
        return SimpleNamespace(
            session_date=session_date,
            phase_agent=phase_agent,
            payload={
                "phase": "market_hours",
                "candidates": [{"symbol": "GLNG", "score": 90.0}],
                "weak_positions": [{"symbol": "OLD1", "pnl_pct": -5.0}],
            },
            legacy_recommendation={"legacy_choice": ["OLD1"]},
            failure_notes=["Missed GLNG"],
            missed_symbols=["GLNG"],
            weak_symbols=["OLD1"],
            replay_report={
                "divergences_count": 3,
                "handoff_diagnostics": {
                    "status": "degraded",
                    "recoverable_candidates_examples": ["GLNG"],
                },
                "actual_day_context": {"net_pnl": -50.0},
            },
        )

    monkeypatch.setattr(evaluator, "build_scenario", _fake_build_scenario)
    monkeypatch.setattr(
        evaluator.controller,
        "benchmark_providers",
        lambda **kwargs: [
            {
                "provider": provider["provider"],
                "model": provider["model"],
                "success": True,
                "latency_sec": 2.0,
                "content": json.dumps(
                    {
                        "decision": "override",
                        "confidence": 0.8,
                        "reasoning_summary": "promote and trim",
                        "legacy_comparison": {
                            "legacy_choice": ["OLD1"],
                            "agent_choice": ["GLNG", "OLD1"],
                            "agreement_level": "override",
                            "override_reason": "better setup",
                        },
                        "actions": [
                            {"action_type": "promote_symbol", "symbol": "GLNG", "reason": "missed"},
                            {"action_type": "trim_position", "symbol": "OLD1", "reason": "weak"},
                        ],
                    }
                ),
            }
            for provider in kwargs.get("providers", [])
        ],
    )

    report = evaluator.evaluate_dates(
        dates=["2026-03-23"],
        providers=[
            {"provider": "local", "model": "qwen2.5-coder:7b"},
            {"provider": "local", "model": "phi4:14b-q4_K_M"},
        ],
        persist=False,
    )

    assert report["summary"]["provider_summary"]["local"]["avg_total_score"] > 0.5
    assert (
        report["summary"]["provider_model_summary"]["local:qwen2.5-coder:7b"][
            "avg_total_score"
        ]
        > 0.5
    )
    assert (
        report["summary"]["provider_model_summary"]["local:phi4:14b-q4_K_M"][
            "avg_total_score"
        ]
        > 0.5
    )
    assert report["dates"][0]["consensus"]["active_models"] == 2
    assert report["dates"][0]["consensus"]["promote_majority_symbols"] == ["GLNG"]
    assert report["dates"][0]["consensus"]["manage_majority_symbols"] == ["OLD1"]
    provider_row = report["dates"][0]["provider_results"][0]
    assert provider_row["score"]["missed_hits"] == ["GLNG"]
    assert provider_row["score"]["weak_hits"] == ["OLD1"]
