from __future__ import annotations

import pandas as pd

from autotrade.signals.veto_policy_benchmark import apply_fixed_rule_reject_veto_overlay
from autotrade.signals.veto_policy_benchmark import apply_llm_reject_veto_overlay
from autotrade.signals.veto_policy_benchmark import FixedRuleVetoPolicy
from autotrade.signals.veto_policy_benchmark import recommend_veto_policy
from autotrade.signals.veto_policy_benchmark import split_rows_by_date


def test_split_rows_by_date_uses_last_dates_as_holdout() -> None:
    rows = pd.DataFrame(
        [
            {
                "sig_date": "2026-04-01",
                "winner": True,
                "loser": False,
                "ambiguous": False,
            },
            {
                "sig_date": "2026-04-02",
                "winner": False,
                "loser": True,
                "ambiguous": False,
            },
            {
                "sig_date": "2026-04-03",
                "winner": False,
                "loser": False,
                "ambiguous": True,
            },
            {
                "sig_date": "2026-04-04",
                "winner": True,
                "loser": False,
                "ambiguous": False,
            },
        ]
    )

    split = split_rows_by_date(rows, test_days=2)

    assert split["train_dates"] == ["2026-04-01", "2026-04-02"]
    assert split["test_dates"] == ["2026-04-03", "2026-04-04"]
    assert len(split["train_rows"]) == 2
    assert len(split["test_rows"]) == 2


def test_apply_llm_reject_veto_overlay_only_vetoes_promoted_rows() -> None:
    baseline = [
        {
            "decision": "promote_buy",
            "confidence": 80.0,
            "model_score": 80.0,
            "latency_seconds": 0.0,
            "parse_ok": True,
            "timed_out": False,
            "error": "",
        },
        {
            "decision": "keep_watch",
            "confidence": 50.0,
            "model_score": 50.0,
            "latency_seconds": 0.0,
            "parse_ok": True,
            "timed_out": False,
            "error": "",
        },
    ]
    llm = [
        {
            "decision": "reject_no_buy",
            "confidence": 88.0,
            "model_score": 12.0,
            "latency_seconds": 1.2,
            "parse_ok": True,
            "timed_out": False,
            "error": "",
        },
        {
            "decision": "reject_no_buy",
            "confidence": 95.0,
            "model_score": 5.0,
            "latency_seconds": 1.1,
            "parse_ok": True,
            "timed_out": False,
            "error": "",
        },
    ]

    decisions = apply_llm_reject_veto_overlay(
        baseline,
        llm,
        threshold_confidence=70.0,
    )

    assert decisions[0]["decision"] == "reject_no_buy"
    assert decisions[0]["latency_seconds"] == 1.2
    assert decisions[1]["decision"] == "keep_watch"
    assert decisions[1]["latency_seconds"] == 1.1


def test_apply_fixed_rule_reject_veto_overlay_rejects_high_risk_tail() -> None:
    rows = pd.DataFrame(
        [
            {
                "rank": 9,
                "confidence": 65.0,
                "weekly_return": 14.0,
                "rsi_14": 68.0,
                "volume_ratio": 0.85,
                "risk_reward": 1.4,
                "technical_score": 45.0,
                "overnight_bad_close_prob": 0.0,
                "overnight_trap_risk": 0.0,
                "overnight_fade_risk": 0.0,
                "overnight_expected_profit_proxy": 0.0,
                "overnight_expected_close_loss_pct": 0.0,
            },
            {
                "rank": 2,
                "confidence": 84.0,
                "weekly_return": 4.0,
                "rsi_14": 55.0,
                "volume_ratio": 2.2,
                "risk_reward": 2.1,
                "technical_score": 62.0,
                "overnight_bad_close_prob": 0.0,
                "overnight_trap_risk": 0.0,
                "overnight_fade_risk": 0.0,
                "overnight_expected_profit_proxy": 1.4,
                "overnight_expected_close_loss_pct": 0.2,
            },
        ]
    )
    baseline = [
        {"decision": "promote_buy", "confidence": 80.0, "model_score": 80.0},
        {"decision": "promote_buy", "confidence": 82.0, "model_score": 82.0},
    ]
    policy = FixedRuleVetoPolicy()

    decisions = apply_fixed_rule_reject_veto_overlay(
        rows,
        baseline,
        policy=policy,
    )

    assert decisions[0]["decision"] == "reject_no_buy"
    assert decisions[1]["decision"] == "promote_buy"


def test_recommend_veto_policy_prefers_best_test_winner() -> None:
    results = [
        {
            "name": "rules_only",
            "kind": "baseline",
            "test_metrics": {
                "overall_utility": 31.0,
                "eligibility": {"eligible": True},
            },
        },
        {
            "name": "fixed_rule_reject_veto",
            "kind": "fixed_rules_veto",
            "test_metrics": {
                "overall_utility": 40.0,
                "eligibility": {"eligible": True},
            },
        },
        {
            "name": "llm_reject_veto",
            "kind": "llm_veto",
            "test_metrics": {
                "overall_utility": 47.5,
                "eligibility": {"eligible": True},
            },
        },
    ]

    recommendation = recommend_veto_policy(results, min_test_improvement_margin=3.0)

    assert recommendation["strategy"] == "reject_veto_overlay"
    assert recommendation["winner"] == "llm_reject_veto"
    assert recommendation["best_deterministic_policy"] == "fixed_rule_reject_veto"
