import json

from autotrade.signals.agentic_signal_generator import EntryCandidate
from autotrade.utils.daily_learning_state import (
    apply_learning_gates,
    build_learning_state,
)


def test_build_learning_state_promotes_evidence_qualified_rules(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    state_path = tmp_path / "data" / "daily_learning_state.json"

    files = {
        "daily_lessons_2026-03-07.md": """
## 2. Trades Executed & Risk Management
- **AAA** stopped out -4.0%
## 3. Missed Opportunities
- **BBB** +5.0% breakout not taken
## 4. Actionable Lessons & System Improvements
1. Enhance workflow journal reasoning for holds and exits.
""",
        "daily_lessons_2026-03-08.md": """
## 2. Trades Executed & Risk Management
- **AAA** stopped out -3.1%
## 3. Missed Opportunities
- **BBB** +4.2% continuation
- **AAPL** +1.1% mega-cap drift
""",
        "daily_lessons_2026-03-09.md": """
## 2. Trades Executed & Risk Management
- **CCC** stopped out -2.0%
## 3. Missed Opportunities
- **CCC** +3.0% reversal
- **AAPL** +2.4% mega-cap drift
""",
    }
    for name, content in files.items():
        (reports_dir / name).write_text(content, encoding="utf-8")

    (reports_dir / "vl_quality_20260309.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-03-09T22:00:00",
                "vl_reliable": False,
                "quarantine_reason": "schema_failures",
            }
        ),
        encoding="utf-8",
    )
    (reports_dir / "sequential_shadow_accuracy_2026-03-09.json").write_text(
        json.dumps({"summary": {"evaluated_events": 0}}),
        encoding="utf-8",
    )

    feedback = {
        "date": "2026-03-09",
        "lessons_updated": False,
        "weak_signal_families": [
            {
                "signal_family": "pullback",
                "score_adjustment": -3.25,
                "confidence": 0.7,
                "sample_size": 6,
                "reason": "weak continuation",
            }
        ],
        "best_signal_families": [
            {
                "signal_family": "new_high_breakout",
                "score_adjustment": 4.0,
                "confidence": 0.8,
                "sample_size": 8,
                "reason": "strong follow-through",
            }
        ],
        "underperforming_sector_details": [
            {"sector": "Energy", "avg_change_pct": -2.8, "sample_size": 3}
        ],
        "score_bucket_performance": {
            "35-49": {"count": 3, "positive_rate": 0.0, "avg_change_pct": -1.5},
            "50-64": {"count": 1, "positive_rate": 1.0, "avg_change_pct": 1.2},
        },
    }

    state = build_learning_state(
        review={
            "watchlist_performance": [{"symbol": "AAA", "missed_opportunity": True}]
        },
        feedback=feedback,
        reports_dir=reports_dir,
        state_path=state_path,
        as_of_date="2026-03-09",
    )

    assert state["active_rules"]["blocked_symbols"] == ["AAA"]
    assert state["active_rules"]["preferred_symbols"] == ["BBB"]
    assert state["active_rules"]["mixed_symbols"] == ["CCC"]
    assert "AAPL" not in state["learning_digest"]["preferred_symbols"]
    assert state["active_rules"]["blocked_setup_types"] == ["pullback"]
    assert state["active_rules"]["preferred_setup_types"] == ["new_high_breakout"]
    assert state["active_rules"]["blocked_sectors"] == ["Energy"]
    assert state["active_rules"]["min_candidate_score"] == 50.0
    assert state["active_rules"]["boosted_symbols"] == ["CCC"]
    assert "workflow:vl_unreliable" in state["learning_digest"]["workflow_flags"]
    assert "workflow:lessons_not_updated" in state["learning_digest"]["workflow_flags"]
    assert state["learning_artifact_status"]["usable_for_bias"] is False
    assert (
        "learning_status:post_market_lesson_missing"
        in state["learning_artifact_status"]["warnings"]
    )
    assert "pm_news_audit" not in state["source_summary"]["families"]
    assert state_path.exists()


def test_build_learning_state_extracts_latest_lesson_bias_rules(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    state_path = tmp_path / "data" / "daily_learning_state.json"

    (reports_dir / "daily_lessons_2026-03-24.md").write_text(
        """
## 3. Missed Opportunities
- **BMNR** +5.6% breakout not taken
- **AA** +4.2% breakout continuation
## 4. Actionable Lessons & System Improvements
1. Adjust conviction scoring thresholds because the threshold is too strict for breakout setups.
2. Review entry criteria for breakout names that scored well but were skipped.
""",
        encoding="utf-8",
    )
    (reports_dir / "post_market_lesson_2026-03-24.json").write_text(
        json.dumps({"date": "2026-03-24"}),
        encoding="utf-8",
    )

    state = build_learning_state(
        review={},
        feedback={"date": "2026-03-24", "lessons_updated": True},
        reports_dir=reports_dir,
        state_path=state_path,
        as_of_date="2026-03-24",
    )

    assert state["active_rules"]["boosted_symbols"] == ["AA", "BMNR"]
    assert state["active_rules"]["preferred_setup_keywords"] == ["breakout"]
    assert state["learning_artifact_status"]["usable_for_bias"] is True


def test_apply_learning_gates_blocks_symbols_setups_sectors_and_score():
    state = {
        "as_of_date": "2026-03-09",
        "threshold_rules": {"min_candidate_score": 50.0},
        "rule_index": {
            "blocked_symbols": {"AAA": "symbol:block:AAA", "SPY": "symbol:block:SPY"},
            "preferred_symbols": {},
            "blocked_setup_types": {"pullback": "setup:block:pullback"},
            "blocked_sectors": {"Energy": "sector:block:Energy"},
        },
    }
    symbol_blocked = EntryCandidate(symbol="AAA", price=10.0, final_score=90.0)
    setup_blocked = EntryCandidate(symbol="BBB", price=10.0, final_score=45.0)
    setup_blocked.metadata["setup_type"] = "pullback"
    setup_blocked.metadata["sector"] = "Energy"
    allowed = EntryCandidate(symbol="CCC", price=10.0, final_score=88.0)
    allowed.metadata["setup_type"] = "new_high_breakout"
    allowed.metadata["sector"] = "Healthcare"
    market_context = EntryCandidate(symbol="SPY", price=10.0, final_score=90.0)

    filtered, report = apply_learning_gates(
        [symbol_blocked, setup_blocked, allowed, market_context], state
    )

    assert [cand.symbol for cand in filtered] == ["SPY", "CCC"]
    assert len(report["blocked"]) == 2
    assert symbol_blocked.metadata["learning_gate_status"] == "blocked"
    assert setup_blocked.metadata["learning_gate_status"] == "blocked"
    assert "setup:block:pullback" in setup_blocked.metadata["learning_rule_ids"]
    assert "sector:block:Energy" in setup_blocked.metadata["learning_rule_ids"]
    assert (
        "threshold:min_candidate_score" in setup_blocked.metadata["learning_rule_ids"]
    )
    assert allowed.metadata["learning_gate_status"] == "pass"
    assert market_context.metadata["learning_gate_status"] == "pass"


def test_apply_learning_gates_neutralizes_stale_or_missing_lessons():
    state = {
        "as_of_date": "2026-03-09",
        "learning_artifact_status": {
            "status": "degraded",
            "usable_for_bias": False,
            "neutral_reason": "stale_or_missing_learning_artifacts",
            "warnings": ["learning_status:stale_daily_lessons"],
        },
        "threshold_rules": {"min_candidate_score": 80.0},
        "rule_index": {
            "blocked_symbols": {"AAA": "symbol:block:AAA"},
            "preferred_symbols": {},
            "blocked_setup_types": {},
            "blocked_sectors": {},
        },
    }
    blocked = EntryCandidate(symbol="AAA", price=10.0, final_score=90.0)
    lower_score = EntryCandidate(symbol="BBB", price=10.0, final_score=70.0)

    filtered, report = apply_learning_gates([blocked, lower_score], state)

    assert [cand.symbol for cand in filtered] == ["AAA", "BBB"]
    assert report["learning_neutral"] is True
    assert report["blocked"] == []
    assert blocked.metadata["learning_gate_status"] == "neutral"
    assert lower_score.metadata["learning_gate_status"] == "neutral"


def test_apply_learning_gates_boosts_fresh_lesson_priority():
    state = {
        "as_of_date": "2026-03-24",
        "learning_artifact_status": {"status": "fresh", "usable_for_bias": True},
        "threshold_rules": {"min_candidate_score": 0.0},
        "active_rules": {
            "boosted_symbols": ["BMNR"],
            "preferred_symbols": [],
            "preferred_setup_keywords": ["breakout"],
        },
        "bias_rules": {
            "symbol_boost_rules": [
                {
                    "rule_id": "bias:symbol:BMNR",
                    "symbol": "BMNR",
                    "score_boost": 4.0,
                    "reason": "missed opportunity",
                }
            ],
            "setup_keyword_boost_rules": [],
        },
        "rule_index": {
            "blocked_symbols": {},
            "preferred_symbols": {},
            "blocked_setup_types": {},
            "blocked_sectors": {},
        },
    }
    boosted = EntryCandidate(symbol="BMNR", price=10.0, final_score=78.0)
    baseline = EntryCandidate(symbol="ZZZ", price=10.0, final_score=80.0)

    filtered, report = apply_learning_gates([baseline, boosted], state)

    assert [cand.symbol for cand in filtered] == ["BMNR", "ZZZ"]
    assert report["state_loaded"] is True
    assert boosted.final_score == 82.0
    assert boosted.metadata["daily_learning"]["score_boost"] == 4.0
