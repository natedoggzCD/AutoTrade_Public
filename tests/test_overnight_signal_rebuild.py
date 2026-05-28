from __future__ import annotations

from autotrade.signals.overnight_signal_rebuild import (
    score_overnight_candidate,
    select_constrained_overnight_candidates,
)


def _base_candidate() -> dict:
    return {
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
        "fresh_news": False,
        "catalyst_count": 0,
        "catalyst_score": 0.0,
        "fallback_repeat_penalty": 0.0,
        "backtest_win_rate": 73.0,
        "profit_factor": 2.2,
        "regime": "NEUTRAL",
    }


def test_actionability_beats_sentiment_without_catalyst() -> None:
    history = {"strategy": {}, "setup": {}, "symbol": {}}

    momentum_candidate = _base_candidate()
    weak_sentiment_candidate = {
        **_base_candidate(),
        "symbol": "BBB",
        "risk_reward": 1.0,
        "atr_percent": 2.1,
        "volume_ratio": 0.9,
        "vol_trend_ratio": 0.95,
        "weekly_return": 0.5,
        "rsi_14": 69.0,
        "technical_score": 52.0,
        "ml_score": 48.0,
        "sr_score": 41.0,
        "entry_score": 42.0,
        "s1_price": 8.0,
        "r1_price": 10.5,
        "sentiment_score": 95.0,
        "stocktwits": {"score": 0.95, "trending": True, "bull_bear_ratio": 3.0},
    }

    strong = score_overnight_candidate(
        momentum_candidate,
        historical_context=history,
        model_artifact={},
    )
    weak = score_overnight_candidate(
        weak_sentiment_candidate,
        historical_context=history,
        model_artifact={},
    )

    assert strong["ranking_score"] > weak["ranking_score"]
    assert strong["final_score"] == strong["ranking_score"]
    assert 0 <= strong["final_score"] <= 100
    assert weak["overnight_catalyst_gate"] == 50.0


def test_selector_uses_final_score_as_canonical_score_before_legacy_ranking_score() -> (
    None
):
    rows = [
        {
            "symbol": "SATURATED",
            "final_score": 72.0,
            "ranking_score": 102.0,
            "confidence": 75,
            "overnight_expected_open_to_high_pct": 2.0,
            "overnight_expected_profit_proxy": 1.2,
            "overnight_expected_open_to_close_pct": 0.3,
            "overnight_stickiness_score": 0.4,
            "overnight_hit_2pct_prob": 0.45,
            "overnight_actionability_score": 70,
            "overnight_bad_close_prob": 0.05,
            "overnight_trap_risk": 0.1,
            "overnight_fade_risk": 0.1,
            "overnight_execution_intent": "hold_candidate",
            "setup_type": "pullback_support",
        },
        {
            "symbol": "CANONICAL",
            "final_score": 80.0,
            "ranking_score": 10.0,
            "confidence": 75,
            "overnight_expected_open_to_high_pct": 2.0,
            "overnight_expected_profit_proxy": 1.2,
            "overnight_expected_open_to_close_pct": 0.3,
            "overnight_stickiness_score": 0.4,
            "overnight_hit_2pct_prob": 0.45,
            "overnight_actionability_score": 70,
            "overnight_bad_close_prob": 0.05,
            "overnight_trap_risk": 0.1,
            "overnight_fade_risk": 0.1,
            "overnight_execution_intent": "hold_candidate",
            "setup_type": "pullback_support",
        },
    ]

    selected, _ = select_constrained_overnight_candidates(rows, top_n=2)

    assert [row["symbol"] for row in selected] == ["CANONICAL", "SATURATED"]


def test_catalyst_confirmation_is_secondary_not_primary() -> None:
    history = {"strategy": {}, "setup": {}, "symbol": {}}
    no_catalyst = score_overnight_candidate(
        _base_candidate(),
        historical_context=history,
        model_artifact={},
    )

    with_catalyst = score_overnight_candidate(
        {
            **_base_candidate(),
            "fresh_news": True,
            "catalyst_count": 2,
            "catalyst_score": 0.8,
            "sentiment_score": 80.0,
        },
        historical_context=history,
        model_artifact={},
    )

    assert (
        with_catalyst["overnight_catalyst_gate"]
        > no_catalyst["overnight_catalyst_gate"]
    )
    assert with_catalyst["ranking_score"] > no_catalyst["ranking_score"]
    assert with_catalyst["ranking_score"] - no_catalyst["ranking_score"] < 10.0


def test_historical_strategy_and_symbol_loser_penalties_apply() -> None:
    candidate = _base_candidate()
    history = {
        "strategy": {
            "fast_trend": {"count": 12, "profit_proxy": 2.4, "open_to_close": 0.8}
        },
        "setup": {
            "trend_follow": {"count": 12, "profit_proxy": 1.5, "open_to_close": 0.4}
        },
        "symbol": {"AAA": {"count": 4, "profit_proxy": 0.3, "open_to_close": -2.0}},
    }

    scored = score_overnight_candidate(
        candidate, historical_context=history, model_artifact={}
    )

    assert scored["overnight_strategy_edge"] > 0
    assert scored["overnight_setup_edge"] > 0
    assert scored["overnight_symbol_loser_penalty"] > 0
    assert scored["overnight_symbol_edge"] <= 0
    assert "edge=" in scored["overnight_rank_reason"]


def test_negative_expected_close_is_penalized_even_if_breakout_probability_is_high() -> (
    None
):
    history = {"strategy": {}, "setup": {}, "symbol": {}}
    candidate = _base_candidate()
    bearish_close = {
        "models": {
            "hit_2pct": {
                "feature_names": ["atr_percent"],
                "means": [7.0],
                "scales": [1.0],
                "coefficients": [1.0],
                "intercept": 0.5,
                "sample_count": 100,
                "target_rate": 0.5,
            },
            "positive_close": {
                "feature_names": ["atr_percent"],
                "means": [7.0],
                "scales": [1.0],
                "coefficients": [0.0],
                "intercept": -0.2,
                "sample_count": 100,
                "target_rate": 0.45,
            },
            "bad_close": {
                "feature_names": ["atr_percent"],
                "means": [7.0],
                "scales": [1.0],
                "coefficients": [0.0],
                "intercept": 0.7,
                "sample_count": 100,
                "target_rate": 0.55,
            },
        },
        "regression_models": {
            "open_to_high_pct": {
                "feature_names": ["atr_percent"],
                "means": [7.0],
                "scales": [1.0],
                "coefficients": [0.0],
                "intercept": 3.4,
                "sample_count": 100,
                "target_rate": 3.4,
            },
            "open_to_close_pct": {
                "feature_names": ["atr_percent"],
                "means": [7.0],
                "scales": [1.0],
                "coefficients": [0.0],
                "intercept": -0.9,
                "sample_count": 100,
                "target_rate": -0.9,
            },
            "profit_proxy": {
                "feature_names": ["atr_percent"],
                "means": [7.0],
                "scales": [1.0],
                "coefficients": [0.0],
                "intercept": 1.9,
                "sample_count": 100,
                "target_rate": 1.9,
            },
            "close_loss_pct": {
                "feature_names": ["atr_percent"],
                "means": [7.0],
                "scales": [1.0],
                "coefficients": [0.0],
                "intercept": 1.1,
                "sample_count": 100,
                "target_rate": 1.1,
            },
        },
    }
    benign_close = {
        **bearish_close,
        "models": {
            **bearish_close["models"],
            "positive_close": {
                **bearish_close["models"]["positive_close"],
                "intercept": 0.4,
                "target_rate": 0.6,
            },
            "bad_close": {
                **bearish_close["models"]["bad_close"],
                "intercept": -0.5,
                "target_rate": 0.35,
            },
        },
        "regression_models": {
            **bearish_close["regression_models"],
            "open_to_close_pct": {
                **bearish_close["regression_models"]["open_to_close_pct"],
                "intercept": 0.7,
                "target_rate": 0.7,
            },
            "profit_proxy": {
                **bearish_close["regression_models"]["profit_proxy"],
                "intercept": 2.2,
                "target_rate": 2.2,
            },
            "close_loss_pct": {
                **bearish_close["regression_models"]["close_loss_pct"],
                "intercept": 0.2,
                "target_rate": 0.2,
            },
        },
    }

    risky = score_overnight_candidate(
        candidate, historical_context=history, model_artifact=bearish_close
    )
    safer = score_overnight_candidate(
        candidate, historical_context=history, model_artifact=benign_close
    )

    assert risky["overnight_expected_open_to_close_pct"] < 0.0
    assert (
        safer["overnight_expected_open_to_close_pct"]
        > risky["overnight_expected_open_to_close_pct"]
    )
    assert risky["ranking_score"] < safer["ranking_score"]
    assert risky["overnight_execution_intent"] == "quick_turnover"
    assert risky["support_break_trim_profile"] in {"aggressive", "balanced"}
    assert safer["overnight_stickiness_score"] > risky["overnight_stickiness_score"]


def test_intraday_skewed_positive_close_candidate_uses_quick_turnover_intent() -> None:
    history = {"strategy": {}, "setup": {}, "symbol": {}}
    candidate = _base_candidate()
    intraday_skew = {
        "models": {
            "hit_2pct": {
                "feature_names": ["atr_percent"],
                "means": [7.0],
                "scales": [1.0],
                "coefficients": [0.0],
                "intercept": 0.6,
                "sample_count": 100,
                "target_rate": 0.6,
            },
            "positive_close": {
                "feature_names": ["atr_percent"],
                "means": [7.0],
                "scales": [1.0],
                "coefficients": [0.0],
                "intercept": 0.55,
                "sample_count": 100,
                "target_rate": 0.55,
            },
            "bad_close": {
                "feature_names": ["atr_percent"],
                "means": [7.0],
                "scales": [1.0],
                "coefficients": [0.0],
                "intercept": -0.4,
                "sample_count": 100,
                "target_rate": 0.25,
            },
            "trap_risk": {
                "feature_names": ["atr_percent"],
                "means": [7.0],
                "scales": [1.0],
                "coefficients": [0.0],
                "intercept": -0.35,
                "sample_count": 100,
                "target_rate": 0.41,
            },
            "fade_risk": {
                "feature_names": ["atr_percent"],
                "means": [7.0],
                "scales": [1.0],
                "coefficients": [0.0],
                "intercept": -1.2,
                "sample_count": 100,
                "target_rate": 0.08,
            },
        },
        "regression_models": {
            "open_to_high_pct": {
                "feature_names": ["atr_percent"],
                "means": [7.0],
                "scales": [1.0],
                "coefficients": [0.0],
                "intercept": 3.4,
                "sample_count": 100,
                "target_rate": 3.4,
            },
            "open_to_close_pct": {
                "feature_names": ["atr_percent"],
                "means": [7.0],
                "scales": [1.0],
                "coefficients": [0.0],
                "intercept": 2.35,
                "sample_count": 100,
                "target_rate": 2.35,
            },
            "profit_proxy": {
                "feature_names": ["atr_percent"],
                "means": [7.0],
                "scales": [1.0],
                "coefficients": [0.0],
                "intercept": 3.55,
                "sample_count": 100,
                "target_rate": 3.55,
            },
            "close_loss_pct": {
                "feature_names": ["atr_percent"],
                "means": [7.0],
                "scales": [1.0],
                "coefficients": [0.0],
                "intercept": 0.18,
                "sample_count": 100,
                "target_rate": 0.18,
            },
            "stickiness_score": {
                "feature_names": ["atr_percent"],
                "means": [7.0],
                "scales": [1.0],
                "coefficients": [0.0],
                "intercept": 0.35,
                "sample_count": 100,
                "target_rate": 0.35,
            },
        },
    }

    scored = score_overnight_candidate(
        candidate,
        historical_context=history,
        model_artifact=intraday_skew,
    )

    assert scored["overnight_expected_open_to_high_pct"] > 3.0
    assert scored["overnight_expected_open_to_close_pct"] > 0.35
    assert scored["overnight_stickiness_score"] < 0.4
    assert scored["overnight_execution_intent"] == "quick_turnover"
    assert scored["support_break_trim_profile"] in {"aggressive", "balanced"}


def test_constrained_selector_limits_toxic_profiles() -> None:
    rows = []
    for idx in range(8):
        rows.append(
            {
                "symbol": f"GOOD{idx}",
                "ranking_score": 90 - idx,
                "confidence": 80,
                "setup_type": "pullback_support",
                "overnight_bad_close_prob": 0.08,
                "overnight_trap_risk": 0.12,
                "overnight_fade_risk": 0.18,
                "overnight_expected_open_to_close_pct": 0.6,
                "overnight_expected_open_to_high_pct": 2.8,
                "overnight_expected_profit_proxy": 2.0,
                "overnight_stickiness_score": 0.56,
                "overnight_setup_edge": 1.5,
                "overnight_strategy_edge": 1.0,
                "overnight_execution_intent": "hold_candidate",
                "overnight_actionability_score": 74.0,
            }
        )
    rows.extend(
        [
            {
                "symbol": "MOMO1",
                "ranking_score": 95,
                "confidence": 90,
                "setup_type": "momentum_breakout",
                "overnight_bad_close_prob": 0.35,
                "overnight_trap_risk": 0.62,
                "overnight_fade_risk": 0.66,
                "overnight_expected_open_to_close_pct": 0.05,
                "overnight_expected_open_to_high_pct": 3.1,
                "overnight_expected_profit_proxy": 1.8,
                "overnight_stickiness_score": 0.08,
                "overnight_setup_edge": -2.0,
                "overnight_strategy_edge": -2.0,
                "overnight_execution_intent": "quick_turnover",
                "overnight_actionability_score": 72.0,
            },
            {
                "symbol": "TAIL1",
                "ranking_score": 94,
                "confidence": 90,
                "setup_type": "trend_follow",
                "overnight_bad_close_prob": 0.30,
                "overnight_trap_risk": 0.51,
                "overnight_fade_risk": 0.45,
                "overnight_expected_open_to_close_pct": -0.6,
                "overnight_expected_open_to_high_pct": 2.0,
                "overnight_expected_profit_proxy": 0.7,
                "overnight_stickiness_score": 0.04,
                "overnight_setup_edge": -1.0,
                "overnight_strategy_edge": -1.6,
                "overnight_execution_intent": "quick_turnover",
                "overnight_actionability_score": 61.0,
            },
        ]
    )

    selected, diag = select_constrained_overnight_candidates(rows, top_n=5)
    symbols = [row["symbol"] for row in selected]

    assert "MOMO1" not in symbols
    assert "TAIL1" not in symbols
    assert len(selected) == 5
    assert diag["selected"] == 5


def test_constrained_selector_caps_quick_turnover_in_weak_regime() -> None:
    rows = []
    for idx in range(8):
        rows.append(
            {
                "symbol": f"FAST{idx}",
                "ranking_score": 96 - idx,
                "confidence": 84,
                "setup_type": "trend_follow",
                "overnight_bad_close_prob": 0.18,
                "overnight_trap_risk": 0.32,
                "overnight_fade_risk": 0.34,
                "overnight_expected_open_to_close_pct": 0.18,
                "overnight_expected_open_to_high_pct": 3.3,
                "overnight_expected_profit_proxy": 2.0,
                "overnight_stickiness_score": 0.26,
                "overnight_setup_edge": 0.5,
                "overnight_strategy_edge": 0.6,
                "overnight_execution_intent": "quick_turnover",
                "overnight_actionability_score": 70.0,
            }
        )
    for idx in range(8):
        rows.append(
            {
                "symbol": f"HOLD{idx}",
                "ranking_score": 88 - idx,
                "confidence": 82,
                "setup_type": "pullback_support",
                "overnight_bad_close_prob": 0.09,
                "overnight_trap_risk": 0.14,
                "overnight_fade_risk": 0.12,
                "overnight_expected_open_to_close_pct": 0.55,
                "overnight_expected_open_to_high_pct": 2.4,
                "overnight_expected_profit_proxy": 1.75,
                "overnight_stickiness_score": 0.58,
                "overnight_setup_edge": 1.2,
                "overnight_strategy_edge": 1.0,
                "overnight_execution_intent": "hold_candidate",
                "overnight_actionability_score": 73.0,
            }
        )

    selected, diag = select_constrained_overnight_candidates(
        rows,
        top_n=10,
        regime_label="RISK_OFF",
    )

    quick_count = sum(
        1
        for row in selected
        if row.get("overnight_execution_intent") == "quick_turnover"
    )
    hold_count = sum(
        1
        for row in selected
        if row.get("overnight_execution_intent") == "hold_candidate"
    )

    assert diag["regime_bucket"] == "weak"
    assert quick_count <= diag["max_quick_turnover"]
    assert hold_count >= 4
