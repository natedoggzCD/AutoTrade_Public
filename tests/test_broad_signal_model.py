from __future__ import annotations

import pandas as pd

from autotrade.signals.broad_signal_model import build_day_relative_features
from autotrade.signals.broad_signal_model import fit_family_models
from autotrade.signals.broad_signal_model import make_walk_forward_folds
from autotrade.signals.broad_signal_model import rank_candidates_by_day
from autotrade.signals.broad_signal_model import score_family_frame


def _synthetic_rows() -> pd.DataFrame:
    rows = []
    for day_idx in range(6):
        sig_date = f"2026-04-{day_idx + 1:02d}"
        for row_idx in range(10):
            strong = row_idx % 2 == 0
            rows.append(
                {
                    "sig_date": sig_date,
                    "ticker": f"S{day_idx}{row_idx}",
                    "rank": row_idx + 1,
                    "final_score": 90.0 if strong else 30.0,
                    "ranking_score": 88.0 if strong else 32.0,
                    "technical_score": 82.0 if strong else 41.0,
                    "sentiment_score": 70.0 if strong else 44.0,
                    "risk_reward": 2.4 if strong else 1.0,
                    "volume_ratio": 2.1 if strong else 0.9,
                    "weekly_return": 6.0 if strong else -1.0,
                    "atr_percent": 6.0 if strong else 2.2,
                    "overnight_expected_profit_proxy": 1.6 if strong else -0.4,
                    "confidence": 82.0 if strong else 48.0,
                    "recommendation": "STRONG BUY" if strong else "WATCH",
                    "action": "buy_open" if strong else "hold",
                    "overnight_regime_bucket": "strong" if strong else "neutral",
                    "overnight_execution_intent": "trend_follow" if strong else "hold_candidate",
                    "open_to_high_pct": 4.0 if strong else 0.7,
                    "open_to_close_pct": 1.8 if strong else -2.4,
                    "profit_proxy": 3.2 if strong else -0.3,
                    "hit_2pct": strong,
                    "positive_close": strong,
                    "bad_close": not strong,
                    "trap_risk": not strong,
                    "fade_risk": not strong,
                    "winner": strong,
                    "loser": not strong,
                    "ambiguous": False,
                }
            )
    return pd.DataFrame(rows)


def test_build_day_relative_features_adds_counts_and_percentiles() -> None:
    rows = pd.DataFrame(
        [
            {"sig_date": "2026-04-01", "ticker": "AAA", "rank": 1, "final_score": 9.0},
            {"sig_date": "2026-04-01", "ticker": "BBB", "rank": 2, "final_score": 3.0},
            {"sig_date": "2026-04-02", "ticker": "CCC", "rank": 1, "final_score": 7.0},
        ]
    )

    out = build_day_relative_features(rows, score_columns=["final_score"])

    assert list(out["day_candidate_count"]) == [2, 2, 1]
    assert list(out["rank_pct"].round(2)) == [0.5, 1.0, 1.0]
    assert out.loc[out["ticker"] == "AAA", "final_score_rank_pct"].iloc[0] > out.loc[
        out["ticker"] == "BBB", "final_score_rank_pct"
    ].iloc[0]


def test_make_walk_forward_folds_keeps_chronology_and_holdout() -> None:
    dates = [f"2026-04-{day:02d}" for day in range(1, 13)]

    split = make_walk_forward_folds(
        dates,
        min_train_days=4,
        validation_days=2,
        step_days=2,
        final_test_days=2,
        embargo_days=1,
    )

    assert split["holdout_dates"] == ["2026-04-11", "2026-04-12"]
    assert len(split["folds"]) >= 2
    for fold in split["folds"]:
        assert fold["train_dates"][-1] < fold["validation_dates"][0]


def test_fit_family_models_scores_stronger_candidate_higher() -> None:
    rows = _synthetic_rows()
    artifact = fit_family_models(rows, family="logit")
    assert artifact is not None

    scored = score_family_frame(
        rows.iloc[:2].copy(),
        artifact,
        blend_weight=0.75,
    )
    strong = scored.iloc[0]["model_score"]
    weak = scored.iloc[1]["model_score"]

    assert strong > weak


def test_rank_candidates_by_day_limits_top_n() -> None:
    rows = _synthetic_rows()
    ranked = rank_candidates_by_day(rows, score_column="final_score", top_n=3)

    assert len(ranked) == 18
    assert ranked.groupby("sig_date").size().eq(3).all()
