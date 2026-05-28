from __future__ import annotations

import pandas as pd

from autotrade.signals.overnight_model_fit import (
    build_historical_edge_context,
    candidate_feature_row,
    fit_models_from_rows,
    predict_candidate_outcomes,
    predict_candidate_probabilities,
)
from autotrade.signals.overnight_signal_rebuild import score_overnight_candidate


def _training_rows() -> pd.DataFrame:
    rows = []
    for idx in range(60):
        strong = idx % 2 == 0
        rows.append(
            {
                "ticker": f"S{idx:03d}",
                "strategy_name": "fast_trend" if strong else "slow_drift",
                "setup_type": "trend_follow" if strong else "weak_breakout",
                "atr_percent": 7.0 if strong else 2.2,
                "volume_ratio": 2.3 if strong else 0.9,
                "vol_trend_ratio": 1.25 if strong else 0.95,
                "weekly_return": 5.0 if strong else 0.5,
                "rsi_14": 49.0 if strong else 68.0,
                "risk_reward": 2.4 if strong else 1.0,
                "support_dist_atr": 0.9 if strong else 0.3,
                "resistance_dist_atr": 1.8 if strong else 0.6,
                "technical_score": 68.0 if strong else 48.0,
                "ml_score": 62.0 if strong else 45.0,
                "sr_score": 60.0 if strong else 42.0,
                "entry_score": 64.0 if strong else 40.0,
                "sentiment_score": 54.0 if strong else 74.0,
                "catalyst_count": 1 if strong else 0,
                "fresh_news": strong,
                "discovery_family_count": 2 if strong else 0,
                "stale_entry_appearance_streak": 0 if strong else 3,
                "fallback_repeat_penalty": 0.0 if strong else 3.5,
                "open_to_high_pct": 4.5 if strong else 1.0,
                "open_to_close_pct": 1.8 if strong else -2.8,
                "profit_proxy": 3.555 if strong else -0.33,
                "hit_2pct": strong,
                "positive_close": strong,
                "bad_close": not strong,
                "trap_risk": not strong,
                "fade_risk": not strong,
                "stickiness_score": 0.72 if strong else 0.08,
            }
        )
    return pd.DataFrame(rows)


def test_fit_models_and_predict_probabilities() -> None:
    rows = _training_rows()
    artifact = fit_models_from_rows(rows)
    assert sorted((artifact.get("models") or {}).keys()) == [
        "bad_close",
        "fade_risk",
        "hit_2pct",
        "positive_close",
        "trap_risk",
    ]
    assert sorted((artifact.get("regression_models") or {}).keys()) == [
        "close_loss_pct",
        "open_to_close_pct",
        "open_to_high_pct",
        "profit_proxy",
        "stickiness_score",
    ]

    strong_candidate = candidate_feature_row(rows.iloc[0].to_dict())
    weak_candidate = candidate_feature_row(rows.iloc[1].to_dict())
    strong_probs = predict_candidate_probabilities(strong_candidate, artifact)
    weak_probs = predict_candidate_probabilities(weak_candidate, artifact)
    strong_outcomes = predict_candidate_outcomes(strong_candidate, artifact)
    weak_outcomes = predict_candidate_outcomes(weak_candidate, artifact)

    assert strong_probs["hit_2pct"] > weak_probs["hit_2pct"]
    assert strong_probs["positive_close"] > weak_probs["positive_close"]
    assert strong_probs["bad_close"] < weak_probs["bad_close"]
    assert strong_probs["trap_risk"] < weak_probs["trap_risk"]
    assert strong_probs["fade_risk"] < weak_probs["fade_risk"]
    assert strong_outcomes["profit_proxy"] > weak_outcomes["profit_proxy"]
    assert strong_outcomes["open_to_close_pct"] > weak_outcomes["open_to_close_pct"]
    assert strong_outcomes["close_loss_pct"] < weak_outcomes["close_loss_pct"]
    assert strong_outcomes["stickiness_score"] > weak_outcomes["stickiness_score"]


def test_score_overnight_candidate_uses_learned_model_and_edge_context() -> None:
    rows = _training_rows()
    artifact = fit_models_from_rows(rows)
    edge_context = build_historical_edge_context(rows)

    candidate = {
        "symbol": "AAA",
        "recommendation": "BUY",
        "entry_price": 10.0,
        "risk_reward": 2.4,
        "atr_percent": 7.0,
        "volume_ratio": 2.3,
        "vol_trend_ratio": 1.25,
        "weekly_return": 5.0,
        "rsi_14": 49.0,
        "technical_score": 68.0,
        "ml_score": 61.0,
        "sr_score": 58.0,
        "entry_score": 64.0,
        "s1_price": 9.6,
        "r1_price": 11.8,
        "strategy_name": "fast_trend",
        "setup_type": "trend_follow",
        "sentiment_score": 52.0,
        "fresh_news": True,
        "catalyst_count": 1,
        "catalyst_score": 0.5,
        "fallback_repeat_penalty": 0.0,
    }

    scored = score_overnight_candidate(
        candidate,
        historical_context=edge_context,
        model_artifact=artifact,
    )

    assert scored["overnight_hit_2pct_prob"] > 0.5
    assert scored["overnight_positive_close_prob"] > 0.5
    assert scored["overnight_bad_close_prob"] < 0.5
    assert scored["overnight_expected_profit_proxy"] > 1.0
    assert scored["overnight_expected_open_to_close_pct"] > 0.0
    assert scored["overnight_expected_close_loss_pct"] < 0.5
    assert scored["overnight_trap_risk"] < 0.5
    assert scored["overnight_fade_risk"] < 0.5
    assert scored["overnight_stickiness_score"] > 0.3
    assert scored["overnight_execution_intent"] == "hold_candidate"
    assert scored["support_break_trim_profile"] in {"balanced", "patient"}
    assert scored["ranking_score"] > 60.0
