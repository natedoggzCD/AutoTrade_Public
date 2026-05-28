from __future__ import annotations

import logging

import pandas as pd

from autotrade.signals.agentic_signal_generator import AgenticSignalGenerator
from autotrade.signals.agentic_signal_generator import EntryCandidate
from autotrade.signals.llm_signal_benchmark import compute_benchmark_metrics
from autotrade.signals.llm_signal_benchmark import recommend_benchmark_strategy
from autotrade.signals.llm_signal_overlay import extract_json_payload
from autotrade.signals.llm_signal_overlay import query_local_signal_classifier


def test_extract_json_payload_strips_think_blocks() -> None:
    payload = extract_json_payload(
        '<think>hidden reasoning</think>{"decision":"reject_no_buy","confidence":91,"reasoning":"trap"}'
    )
    assert payload["decision"] == "reject_no_buy"
    assert payload["confidence"] == 91


def test_query_local_signal_classifier_sets_think_false_for_qwen3(monkeypatch) -> None:
    seen = {}

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "message": {
                    "content": '{"decision":"promote_buy","confidence":83,"reasoning":"strong setup"}'
                }
            }

    def fake_post(url, json=None, timeout=None):  # noqa: ANN001
        seen["url"] = url
        seen["json"] = json
        seen["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("autotrade.signals.llm_signal_overlay.requests.post", fake_post)
    result = query_local_signal_classifier(
        signal_packet={"symbol": "AAA", "confidence": 72},
        model="qwen3:14b-q4_K_M",
        ollama_url="http://test.local/api/chat",
        timeout_seconds=10.0,
        num_ctx=2048,
    )

    assert seen["json"].get("think") is False
    assert result.label == "promote_buy"
    assert result.parse_ok is True


def test_compute_benchmark_metrics_scores_buy_and_reject_quality() -> None:
    rows = pd.DataFrame(
        [
            {
                "sig_date": "2026-04-01",
                "rank": 1,
                "confidence": 41.0,
                "profit_proxy": 2.4,
                "open_to_close_pct": 1.1,
                "hit_2pct": True,
                "bad_close": False,
                "trap_risk": False,
                "fade_risk": False,
                "positive_close": True,
                "winner": True,
                "loser": False,
                "ambiguous": False,
            },
            {
                "sig_date": "2026-04-01",
                "rank": 2,
                "confidence": 22.0,
                "profit_proxy": -1.6,
                "open_to_close_pct": -2.4,
                "hit_2pct": False,
                "bad_close": True,
                "trap_risk": True,
                "fade_risk": True,
                "positive_close": False,
                "winner": False,
                "loser": True,
                "ambiguous": False,
            },
            {
                "sig_date": "2026-04-01",
                "rank": 3,
                "confidence": 18.0,
                "profit_proxy": 0.1,
                "open_to_close_pct": 0.0,
                "hit_2pct": False,
                "bad_close": False,
                "trap_risk": False,
                "fade_risk": False,
                "positive_close": False,
                "winner": False,
                "loser": False,
                "ambiguous": True,
            },
        ]
    )
    decisions = [
        {
            "decision": "promote_buy",
            "confidence": 85.0,
            "model_score": 88.0,
            "latency_seconds": 1.5,
            "parse_ok": True,
            "timed_out": False,
        },
        {
            "decision": "reject_no_buy",
            "confidence": 90.0,
            "model_score": 12.0,
            "latency_seconds": 1.6,
            "parse_ok": True,
            "timed_out": False,
        },
        {
            "decision": "keep_watch",
            "confidence": 50.0,
            "model_score": 50.0,
            "latency_seconds": 1.4,
            "parse_ok": True,
            "timed_out": False,
        },
    ]

    metrics = compute_benchmark_metrics(
        rows, decisions, top_n=2, latency_budget_seconds=10.0
    )

    assert metrics["buy_precision"] == 1.0
    assert metrics["reject_precision"] == 1.0
    assert metrics["overall_utility"] > 0.0
    assert metrics["eligibility"]["eligible"] is True


def test_recommend_benchmark_strategy_prefers_router_for_split_specialists() -> None:
    results = [
        {
            "name": "rules_only",
            "metrics": {
                "overall_utility": 42.0,
                "eligibility": {"eligible": True},
            },
        },
        {
            "name": "promoter",
            "metrics": {
                "overall_utility": 51.0,
                "promote_specialist_score": 81.0,
                "reject_specialist_score": 60.0,
                "eligibility": {"eligible": True},
            },
        },
        {
            "name": "rejector",
            "metrics": {
                "overall_utility": 52.0,
                "promote_specialist_score": 61.0,
                "reject_specialist_score": 84.0,
                "eligibility": {"eligible": True},
            },
        },
    ]

    recommendation = recommend_benchmark_strategy(results)

    assert recommendation["strategy"] == "two_model_router"
    assert recommendation["promote_model"] == "promoter"
    assert recommendation["reject_model"] == "rejector"


def test_final_ranking_honors_reject_veto() -> None:
    generator = AgenticSignalGenerator.__new__(AgenticSignalGenerator)
    generator.logger = logging.getLogger("test_signal_llm_overlay")
    candidate = EntryCandidate(
        symbol="TRAP",
        price=10.0,
        momentum_score=85.0,
        entry_score=92.0,
        risk_reward=2.2,
        news_sentiment=0.4,
        action="watch",
        metadata={"llm_overlay_reject_veto": True},
    )

    ranked = AgenticSignalGenerator._final_ranking(generator, [candidate])

    assert ranked[0].final_score > 60.0
    assert ranked[0].action == "watch"


def test_reject_veto_only_mode_ignores_promote_signal() -> None:
    generator = AgenticSignalGenerator.__new__(AgenticSignalGenerator)
    generator.llm_overlay_cfg = type(
        "_Cfg",
        (),
        {"min_confidence_to_override": 70.0},
    )()
    candidate = EntryCandidate(
        symbol="SAFE",
        price=10.0,
        momentum_score=82.0,
        entry_score=88.0,
        risk_reward=2.1,
        news_sentiment=0.3,
        action="watch",
    )

    AgenticSignalGenerator._apply_overlay_decision(
        generator,
        candidate,
        decision={
            "action": "promote_buy",
            "confidence": 95.0,
            "score": 91.0,
            "reasoning": "strong setup",
            "model": "qwen3.5:9b-q4_K_M",
        },
        role="reject_veto_only",
        allow_promotion=False,
    )

    assert candidate.action == "watch"
    assert candidate.metadata.get("llm_overlay_reject_veto") is False
