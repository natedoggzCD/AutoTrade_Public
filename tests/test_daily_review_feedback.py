import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import autotrade.core.daily_review as daily_review_module
from autotrade.core.daily_review import DailyReview, _build_feedback_from_review


def test_build_feedback_from_review_uses_overall_pnl_and_position_totals():
    review = {
        "day_summary": {
            "overall_pnl_dollars": 123.45,
            "today_pnl_dollars": -55.0,
        },
        "position_performance": [
            {"symbol": "AAA", "total_pnl_dollars": 10.0},
            {"symbol": "BBB", "total_pnl_dollars": -5.0},
            {"symbol": "CCC", "total_pnl_dollars": 0.0},
        ],
    }

    feedback = _build_feedback_from_review(review)

    assert feedback["total_pnl"] == 123.45
    assert feedback["win_rate"] == 0.5


def test_analyze_score_buckets_aggregates_watchlist_metadata():
    review = {
        "watchlist_performance": [
            {
                "symbol": "AAA",
                "change_pct": 4.0,
                "missed_opportunity": True,
                "sector": "Healthcare",
                "final_score": 82.0,
                "setup_type": "new_high_breakout",
            },
            {
                "symbol": "BBB",
                "change_pct": -2.0,
                "missed_opportunity": False,
                "sector": "Healthcare",
                "final_score": 70.0,
                "setup_type": "new_high_breakout",
            },
            {
                "symbol": "CCC",
                "change_pct": -1.0,
                "missed_opportunity": False,
                "sector": "Industrials",
                "confidence": 48.0,
                "setup_type": "pullback",
            },
        ]
    }

    daily_review = DailyReview.__new__(DailyReview)
    buckets = daily_review._analyze_score_buckets(review)

    assert buckets["80+"]["count"] == 1
    assert buckets["80+"]["avg_change_pct"] == 4.0
    assert buckets["80+"]["missed_opportunities"] == 1
    assert buckets["80+"]["top_sector"] == "Healthcare"
    assert buckets["80+"]["top_setup_type"] == "new_high_breakout"
    assert buckets["65-79"]["positive_rate"] == 0.0
    assert buckets["35-49"]["count"] == 1
    assert buckets["35-49"]["top_setup_type"] == "pullback"


def test_score_inversion_diagnostics_flags_top_score_underperformance():
    daily_review = DailyReview.__new__(DailyReview)
    rows = [
        {"symbol": f"TOP{i}", "final_score": 90 - i, "change_pct": -3.0}
        for i in range(10)
    ] + [
        {"symbol": f"LOW{i}", "final_score": 60 - i, "change_pct": 1.0}
        for i in range(10)
    ]

    diagnostics = daily_review._analyze_score_inversion(rows)

    assert diagnostics["available"] is True
    assert diagnostics["inverted"] is True
    assert diagnostics["severity"] == "warning"
    assert diagnostics["top_n"] == 10
    assert diagnostics["top_avg_change_pct"] == -3.0


def test_get_underperforming_sectors_ranks_negative_watchlist_sectors():
    review = {
        "watchlist_performance": [
            {"symbol": "AAA", "sector": "Healthcare", "change_pct": -3.0},
            {"symbol": "BBB", "sector": "Healthcare", "change_pct": -1.0},
            {"symbol": "CCC", "sector": "Energy", "change_pct": 2.0},
            {"symbol": "DDD", "sector": "Energy", "change_pct": -0.5},
            {"symbol": "EEE", "sector": "Industrials", "change_pct": -4.0},
        ]
    }

    daily_review = DailyReview.__new__(DailyReview)
    underperformers = daily_review._get_underperforming_sectors(review)

    assert underperformers[0] == "Industrials"
    assert "Healthcare" in underperformers
    assert "Energy" not in underperformers


def test_summarize_signal_families_returns_best_and_weak_groups():
    family_payload = {
        "families": [
            {
                "signal_family": "new_high_breakout",
                "score_adjustment": 5.5,
                "confidence": 0.8,
                "total_count": 7,
                "avg_realized_pct": 3.2,
                "avg_open_pct": 1.1,
                "reason": "strong follow-through",
            },
            {
                "signal_family": "failed_breakdown_reversal",
                "score_adjustment": -4.25,
                "confidence": 0.7,
                "total_count": 6,
                "avg_realized_pct": -2.1,
                "avg_open_pct": -0.6,
                "reason": "weak follow-through",
            },
        ]
    }

    daily_review = DailyReview.__new__(DailyReview)
    best = daily_review._summarize_signal_families(family_payload, positive=True)
    weak = daily_review._summarize_signal_families(family_payload, positive=False)

    assert best == [
        {
            "signal_family": "new_high_breakout",
            "score_adjustment": 5.5,
            "confidence": 0.8,
            "sample_size": 7,
            "avg_realized_pct": 3.2,
            "avg_open_pct": 1.1,
            "reason": "strong follow-through",
        }
    ]
    assert weak == [
        {
            "signal_family": "failed_breakdown_reversal",
            "score_adjustment": -4.25,
            "confidence": 0.7,
            "sample_size": 6,
            "avg_realized_pct": -2.1,
            "avg_open_pct": -0.6,
            "reason": "weak follow-through",
        }
    ]


def test_learn_from_today_writes_enriched_feedback(monkeypatch, tmp_path):
    review = {
        "day_summary": {"overall_pnl_dollars": 42.0},
        "position_performance": [{"symbol": "AAA", "total_pnl_dollars": 5.0}],
        "watchlist_performance": [
            {
                "symbol": "AAA",
                "sector": "Healthcare",
                "change_pct": 3.5,
                "final_score": 82.0,
                "setup_type": "new_high_breakout",
                "missed_opportunity": False,
            },
            {
                "symbol": "BBB",
                "sector": "Industrials",
                "change_pct": -2.5,
                "final_score": 48.0,
                "setup_type": "failed_breakdown_reversal",
                "missed_opportunity": True,
            },
        ],
    }
    family_payload = {
        "generated_at": "2026-03-07T12:00:00",
        "family_count": 2,
        "families": [
            {
                "signal_family": "new_high_breakout",
                "score_adjustment": 4.0,
                "confidence": 0.75,
                "total_count": 6,
                "avg_realized_pct": 2.4,
                "avg_open_pct": 0.8,
                "reason": "strong",
            },
            {
                "signal_family": "failed_breakdown_reversal",
                "score_adjustment": -3.0,
                "confidence": 0.65,
                "total_count": 5,
                "avg_realized_pct": -1.7,
                "avg_open_pct": -0.4,
                "reason": "weak",
            },
        ],
    }

    monkeypatch.setattr(daily_review_module, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(
        daily_review_module.trade_learner,
        "analyze_and_learn",
        lambda: {"lessons": {"status": "ok"}, "signal_families": family_payload},
    )
    monkeypatch.setattr(
        daily_review_module,
        "DailyLessonsAnalyzer",
        None,
        raising=False,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "autotrade.core.daily_lessons_analyzer",
        SimpleNamespace(
            DailyLessonsAnalyzer=lambda: SimpleNamespace(
                run=lambda date_str: {
                    "success": True,
                    "mode": "fallback",
                    "report_path": str(
                        tmp_path / "reports" / f"daily_lessons_{date_str}.md"
                    ),
                }
            )
        ),
    )
    (tmp_path / "reports").mkdir(exist_ok=True)
    (tmp_path / "reports" / "daily_lessons_2026-03-06.md").write_text(
        """
## 3. Missed Opportunities
- **BBB** +5.2% breakout not taken
""",
        encoding="utf-8",
    )
    (tmp_path / "reports" / "post_market_lesson_2026-03-06.json").write_text(
        json.dumps({"date": "2026-03-06"}),
        encoding="utf-8",
    )

    daily_review = DailyReview.__new__(DailyReview)
    daily_review.review_date_str = "2026-03-06"
    daily_review.trade_journal = SimpleNamespace(
        update_outcomes_from_eod=lambda payload: None
    )
    feedback = daily_review.learn_from_today(review)
    saved_feedback = json.loads(
        (tmp_path / "data" / "eod_feedback_latest.json").read_text(encoding="utf-8")
    )
    saved_learning_state = json.loads(
        (tmp_path / "data" / "daily_learning_state.json").read_text(encoding="utf-8")
    )

    assert feedback["date"] == "2026-03-06"
    assert feedback["lessons_updated"] is True
    assert feedback["learning_context"] == {
        "generated_at": "2026-03-07T12:00:00",
        "family_count": 2,
    }
    assert feedback["best_signal_families"][0]["signal_family"] == "new_high_breakout"
    assert (
        feedback["weak_signal_families"][0]["signal_family"]
        == "failed_breakdown_reversal"
    )
    assert saved_feedback["date"] == "2026-03-06"
    assert saved_feedback["score_bucket_performance"]["80+"]["count"] == 1
    assert saved_feedback["underperforming_sectors"] == ["Industrials"]
    assert saved_feedback["lessons_status"]["report_exists"] is True
    assert saved_feedback["lessons_status"]["success"] is True
    assert feedback["learning_state_path"] == "data/daily_learning_state.json"
    assert saved_learning_state["as_of_date"] == "2026-03-06"
    assert saved_learning_state["learning_artifact_status"]["usable_for_bias"] is True
    assert (
        "workflow:lessons_not_updated"
        not in saved_learning_state["learning_digest"]["workflow_flags"]
    )
    assert saved_learning_state["learning_digest"]["weak_signal_family_count"] == 1


def test_resolve_review_date_uses_prior_session_before_market_open():
    now = datetime(2026, 3, 11, 0, 0, 6)
    review_date = daily_review_module._resolve_review_date(now)
    phase = daily_review_module._classify_review_phase(now)

    assert review_date.strftime("%Y-%m-%d") == "2026-03-10"
    assert phase == "overnight_catchup"


def test_save_review_uses_review_session_date(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_review_module, "LOG_DIR", Path(tmp_path))

    daily_review = DailyReview.__new__(DailyReview)
    daily_review.review_date_str = "2026-03-10"

    review = {
        "review_date": "2026-03-10",
        "review_context": {"phase": "overnight_catchup"},
        "day_summary": {"grade": "OK"},
        "position_performance": [],
        "watchlist_performance": [],
        "workflow_status": {"issues_found": [], "all_healthy": False},
        "orders_today": [],
    }

    daily_review._save_review(review)

    saved = json.loads(
        (tmp_path / "daily_review_2026-03-10.json").read_text(encoding="utf-8")
    )
    assert saved["review_date"] == "2026-03-10"
    assert saved["review_context"]["phase"] == "overnight_catchup"


def test_daily_review_loads_short_side_activity(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_review_module, "LOG_DIR", Path(tmp_path))
    (tmp_path / "short_engine_telemetry_2026-03-10.jsonl").write_text(
        '{"signals_generated": 2}\n{"signals_generated": 0}\n',
        encoding="utf-8",
    )
    (tmp_path / "intraday_analysis_2026-03-10.jsonl").write_text(
        '{"execution_diagnostics":{"inverse_etf_screens":1,"inverse_etf_candidates":3}}\n',
        encoding="utf-8",
    )
    daily_review = DailyReview.__new__(DailyReview)
    daily_review.review_date_str = "2026-03-10"

    activity = daily_review._load_short_side_activity(
        [{"side": "sell_short", "status": "filled"}]
    )

    assert activity["inverse_etf_screens"] == 1
    assert activity["inverse_etf_candidates"] == 3
    assert activity["single_name_shorts_generated"] == 2
    assert activity["single_name_shorts_executed"] == 1
    assert activity["warning"] == ""


def test_daily_review_flags_silent_short_side(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_review_module, "LOG_DIR", Path(tmp_path))
    daily_review = DailyReview.__new__(DailyReview)
    daily_review.review_date_str = "2026-03-10"

    activity = daily_review._load_short_side_activity([])

    assert activity["warning"] == "short_side_silent"


def test_daily_review_loads_market_adaptation_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_review_module, "LOG_DIR", Path(tmp_path))
    (tmp_path / "intraday_analysis_2026-03-10.jsonl").write_text(
        json.dumps(
            {
                "market_context": {
                    "regime_label": "DISPERSION",
                    "dispersion_score": 1.2,
                    "sizing_multiplier": 1.15,
                },
                "tape_inference": "High dispersion tape",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    daily_review = DailyReview.__new__(DailyReview)
    daily_review.review_date_str = "2026-03-10"

    summary = daily_review._load_market_adaptation_summary()

    assert summary["regime_label"] == "DISPERSION"
    assert summary["sizing_multiplier"] > 1.0
    assert "system played to its selection edge" in summary["summary_line"]


def test_build_feedback_from_review_prefers_review_date():
    review = {
        "review_date": "2026-03-10",
        "day_summary": {"overall_pnl_dollars": 42.0},
        "position_performance": [{"symbol": "AAA", "total_pnl_dollars": 5.0}],
    }

    feedback = _build_feedback_from_review(review)

    assert feedback["date"] == "2026-03-10"
