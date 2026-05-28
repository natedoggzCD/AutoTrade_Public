from __future__ import annotations

from autotrade.core.decision_claw import DecisionClaw


def test_candidate_quality_overlay_rewards_actionable_learned_candidates() -> None:
    claw = DecisionClaw.__new__(DecisionClaw)

    strong_score, strong_flags = claw._candidate_quality_overlay(
        {
            "ranking_score": 82.0,
            "risk_reward": 2.3,
            "volume_ratio": 2.0,
            "support_dist_atr": 0.9,
            "resistance_dist_atr": 1.8,
            "overnight_actionability_score": 76.0,
            "overnight_hit_2pct_prob": 0.64,
            "overnight_positive_close_prob": 0.61,
            "overnight_bad_close_prob": 0.18,
            "overnight_trap_risk": 0.16,
            "overnight_fade_risk": 0.21,
            "overnight_expected_profit_proxy": 2.2,
            "overnight_expected_open_to_close_pct": 0.9,
            "overnight_expected_close_loss_pct": 0.2,
            "overnight_stickiness_score": 0.62,
            "overnight_execution_intent": "hold_candidate",
        }
    )
    weak_score, weak_flags = claw._candidate_quality_overlay(
        {
            "ranking_score": 82.0,
            "risk_reward": 1.0,
            "volume_ratio": 0.9,
            "support_dist_atr": 0.3,
            "resistance_dist_atr": 0.7,
            "overnight_actionability_score": 38.0,
            "overnight_hit_2pct_prob": 0.31,
            "overnight_positive_close_prob": 0.34,
            "overnight_bad_close_prob": 0.58,
            "overnight_trap_risk": 0.66,
            "overnight_fade_risk": 0.57,
            "overnight_expected_profit_proxy": 0.4,
            "overnight_expected_open_to_close_pct": -1.1,
            "overnight_expected_close_loss_pct": 1.3,
            "overnight_stickiness_score": 0.08,
            "overnight_execution_intent": "quick_turnover",
        }
    )

    assert strong_score > weak_score
    assert "actionable" in strong_flags
    assert "bad_close_risk" in weak_flags
    assert "overnight_trap_risk" in weak_flags
