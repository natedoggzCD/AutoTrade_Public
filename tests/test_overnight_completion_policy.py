from __future__ import annotations

import pandas as pd
from types import SimpleNamespace

from autotrade.core.autonomous_agent import AutonomousAgent


def _noop_logger():
    return SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )


def _mk_watchlist(count: int) -> list[dict]:
    return [
        {"symbol": f"S{i:03d}", "recommendation": "BUY", "confidence": 70}
        for i in range(count)
    ]


def test_strategy_pool_refresh_requires_full_target_watchlist(monkeypatch):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _noop_logger()
    agent.WATCHLIST_SIZE = 200
    agent.MIN_WATCHLIST_SIZE = 200
    agent.OVERNIGHT_COMPLETION_MIN_WATCHLIST_SIZE = 200
    agent.OVERNIGHT_BREADTH_MIN_SYMBOLS = 2000
    agent.OVERNIGHT_NEWS_STAGE_REQUIRED = True

    state = {
        "research_complete": False,
        "watchlist": _mk_watchlist(142),
        "discovery_queue": [],
        "researched": {},
        "all_candidates": _mk_watchlist(142),
        "rejected": [],
        "sectors": {},
        "screener_metadata": {},
        "workflow_completion": {},
    }

    monkeypatch.setattr(agent, "_load_more_candidates", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent, "_parallel_research_batch", lambda batch: {})
    monkeypatch.setattr(agent, "_select_best_watchlist", lambda s: None)
    monkeypatch.setattr(agent, "_generate_morning_game_plan", lambda s: True)
    monkeypatch.setattr(
        agent,
        "_evaluate_overnight_breadth",
        lambda s: {"passed": True, "scanned_count": 2400, "required_count": 2000},
    )
    monkeypatch.setattr(
        agent,
        "_enforce_top_watchlist_news_coverage",
        lambda s: {"passed": True, "covered_count": 142, "required_count": 142},
    )
    monkeypatch.setattr(agent, "_save_overnight_state", lambda s: None)

    result = agent._run_strategy_pool_delta_refresh(
        state=state,
        current_snapshot={"sha256": "new", "mtime_iso": "2026-03-11T00:00:00"},
        source="test",
        max_batches=1,
    )

    assert result["refreshed_plan"] is False
    assert state["research_complete"] is False
    assert state["workflow_completion"]["watchlist_selected"] is False
    assert state["workflow_completion"]["watchlist_target_met"] is False
    assert (
        state["workflow_completion"]["completion_reason"]
        == "strategy_pool_refresh_test_insufficient_watchlist"
    )


def test_strategy_pool_refresh_blocks_completion_when_news_gate_fails(monkeypatch):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _noop_logger()
    agent.WATCHLIST_SIZE = 200
    agent.MIN_WATCHLIST_SIZE = 200
    agent.OVERNIGHT_COMPLETION_MIN_WATCHLIST_SIZE = 200
    agent.OVERNIGHT_BREADTH_MIN_SYMBOLS = 2000
    agent.OVERNIGHT_NEWS_STAGE_REQUIRED = True

    state = {
        "research_complete": False,
        "watchlist": _mk_watchlist(200),
        "discovery_queue": [],
        "researched": {},
        "all_candidates": _mk_watchlist(200),
        "rejected": [],
        "sectors": {},
        "screener_metadata": {},
        "workflow_completion": {},
    }

    monkeypatch.setattr(agent, "_load_more_candidates", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent, "_parallel_research_batch", lambda batch: {})
    monkeypatch.setattr(agent, "_select_best_watchlist", lambda s: None)
    monkeypatch.setattr(agent, "_generate_morning_game_plan", lambda s: True)
    monkeypatch.setattr(
        agent,
        "_evaluate_overnight_breadth",
        lambda s: {"passed": True, "scanned_count": 2400, "required_count": 2000},
    )
    monkeypatch.setattr(
        agent,
        "_enforce_top_watchlist_news_coverage",
        lambda s: {"passed": False, "covered_count": 180, "required_count": 200},
    )
    monkeypatch.setattr(agent, "_save_overnight_state", lambda s: None)

    result = agent._run_strategy_pool_delta_refresh(
        state=state,
        current_snapshot={"sha256": "new", "mtime_iso": "2026-03-11T00:00:00"},
        source="test",
        max_batches=1,
    )

    assert result["refreshed_plan"] is False
    assert state["research_complete"] is False
    assert (
        state["workflow_completion"]["completion_reason"]
        == "strategy_pool_refresh_test_failed_news_coverage"
    )


def test_ensure_minimum_watchlist_uses_recovery_candidates_when_existing_pool_stalls(
    monkeypatch,
):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _noop_logger()
    agent.MIN_WATCHLIST_SIZE = 200

    initial_watchlist = _mk_watchlist(180)
    state = {
        "watchlist": list(initial_watchlist),
        "all_candidates": list(initial_watchlist),
        "researched": {
            row["symbol"]: {"recommendation": row["recommendation"], "confidence": 70}
            for row in initial_watchlist
        },
        "discovery_queue": [],
        "rejected": [],
    }

    monkeypatch.setattr(
        agent,
        "_build_watchlist_target_recovery_candidates",
        lambda s, limit: [
            {"ticker": f"R{i:03d}", "price": 10.0, "composite_score": 55.0}
            for i in range(20)
        ],
    )

    agent._ensure_minimum_watchlist(state, 200)

    assert len(state["watchlist"]) == 200
    assert {row["symbol"] for row in state["watchlist"]}.issuperset(
        {f"R{i:03d}" for i in range(20)}
    )


def test_is_overnight_workflow_complete_requires_watchlist_target_flag():
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent._morning_game_plan_exists = lambda: True

    state = {
        "research_complete": True,
        "watchlist": [{"symbol": "AAA"}],
        "workflow_completion": {
            "watchlist_selected": True,
            "watchlist_target_met": False,
            "game_plan_generated": True,
            "breadth_passed": True,
            "news_coverage_passed": True,
        },
    }

    assert agent._is_overnight_workflow_complete(state) is False


def test_is_overnight_workflow_complete_allows_degraded_news_handoff():
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent._morning_game_plan_exists = lambda: True

    state = {
        "research_complete": True,
        "watchlist": [{"symbol": "AAA"}],
        "workflow_completion": {
            "watchlist_selected": True,
            "watchlist_target_met": True,
            "game_plan_generated": True,
            "breadth_passed": True,
            "news_coverage_passed": False,
            "degraded_news_coverage": True,
            "completion_reason": "full_cycle_complete_with_failed_news_coverage",
            "missing_news_symbols": ["AAA"],
        },
    }

    assert agent._is_overnight_workflow_complete(state) is True


def test_is_overnight_workflow_complete_can_allow_pending_youtube_for_idle_followups():
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent._morning_game_plan_exists = lambda: True

    state = {
        "research_complete": True,
        "watchlist": [{"symbol": "AAA"}],
        "workflow_completion": {
            "watchlist_selected": True,
            "watchlist_target_met": True,
            "game_plan_generated": True,
            "breadth_passed": True,
            "news_coverage_passed": True,
            "youtube_ready": False,
        },
    }

    assert agent._is_overnight_workflow_complete(state) is False
    assert (
        agent._is_overnight_workflow_complete(state, allow_pending_youtube=True)
        is True
    )


def test_resolve_overnight_completion_status_allows_degraded_news_handoff():
    agent = AutonomousAgent.__new__(AutonomousAgent)

    final_complete, completion_reason, degraded_news_coverage = (
        agent._resolve_overnight_completion_status(
            plan_ready=True,
            watchlist_completion_sufficient=True,
            breadth_ok=True,
            news_ok=False,
            success_reason="full_cycle_complete",
            degraded_news_reason="full_cycle_complete_with_failed_news_coverage",
            watchlist_failure_reason="full_cycle_insufficient_watchlist",
            breadth_failure_reason="full_cycle_failed_breadth_gate",
            news_failure_reason="full_cycle_failed_news_coverage",
            plan_failure_reason="full_cycle_missing_game_plan",
        )
    )

    assert final_complete is True
    assert degraded_news_coverage is True
    assert (
        completion_reason == "full_cycle_complete_with_failed_news_coverage"
    )


def test_resolve_overnight_completion_status_still_blocks_breadth_failure():
    agent = AutonomousAgent.__new__(AutonomousAgent)

    final_complete, completion_reason, degraded_news_coverage = (
        agent._resolve_overnight_completion_status(
            plan_ready=True,
            watchlist_completion_sufficient=True,
            breadth_ok=False,
            news_ok=False,
            success_reason="full_cycle_complete",
            degraded_news_reason="full_cycle_complete_with_failed_news_coverage",
            watchlist_failure_reason="full_cycle_insufficient_watchlist",
            breadth_failure_reason="full_cycle_failed_breadth_gate",
            news_failure_reason="full_cycle_failed_news_coverage",
            plan_failure_reason="full_cycle_missing_game_plan",
        )
    )

    assert final_complete is False
    assert degraded_news_coverage is False
    assert completion_reason == "full_cycle_failed_breadth_gate"


def test_recovery_candidates_derive_history_count_from_prepared_history(monkeypatch):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _noop_logger()
    agent.overnight_engine = SimpleNamespace(regime_scoring_mode="balanced")

    rows = []
    dates = pd.date_range("2026-01-01", periods=65, freq="D")
    for symbol in ("AAA", "BBB"):
        for dt in dates:
            rows.append(
                {
                    "ticker": symbol,
                    "date": dt,
                    "close": 25.0,
                    "avg_volume": 250_000.0,
                    "avg_dollar_volume": 6_250_000.0,
                    "volume_ratio": 1.8,
                    "weekly_return": 4.0,
                    "atr_pct": 4.2,
                    "rsi_14": 55.0,
                }
            )

    prepared_df = pd.DataFrame(rows)

    class _FakeScreener:
        def __init__(self, parent_logger=None, scoring_mode=None):
            self.parent_logger = parent_logger
            self.scoring_mode = scoring_mode

        def _get_prepared_history_session(self):
            return SimpleNamespace(data=prepared_df)

    monkeypatch.setattr(
        "autotrade.signals.screener_v2.ScreenerV2",
        _FakeScreener,
    )

    state = {
        "watchlist": [],
        "all_candidates": [],
        "researched": {},
        "discovery_queue": [],
        "rejected": [],
    }

    candidates = agent._build_watchlist_target_recovery_candidates(state, limit=2)

    assert len(candidates) == 2
    assert {candidate["ticker"] for candidate in candidates} == {"AAA", "BBB"}


def test_news_coverage_persists_symbol_diagnostics(monkeypatch):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _noop_logger()
    agent.OVERNIGHT_NEWS_STAGE_REQUIRED = True
    agent.OVERNIGHT_ENFORCE_TOP_NEWS_COUNT = 2
    agent.OVERNIGHT_NEWS_STAGE_MAX_WORKERS = 2
    agent.OVERNIGHT_NEWS_STAGE_TIMEOUT_SECONDS = 30
    agent._get_watchlist_target_size = lambda: 2

    class _Aggregator:
        def __init__(self, headline_limit: int = 3):
            self.headline_limit = headline_limit

        def collect(self, symbol: str):
            if symbol == "AAA":
                return {
                    "symbol": symbol,
                    "available": True,
                    "headline_count": 2,
                    "coverage": "partial",
                    "confidence": 0.72,
                    "has_catalyst": True,
                    "catalyst_note": "AAA raises guidance",
                    "source_status": {"news_sentiment": "ok", "searxng": "ok"},
                    "search_diagnostics": {"searx_kept": 1},
                    "headlines": [{"title": "AAA raises guidance after earnings beat"}],
                }
            return {
                "symbol": symbol,
                "available": False,
                "headline_count": 0,
                "coverage": "none",
                "confidence": 0.0,
                "has_catalyst": False,
                "catalyst_note": "",
                "source_status": {"news_sentiment": "unavailable", "searxng": "empty"},
                "search_diagnostics": {"searx_kept": 0},
                "headlines": [],
            }

    monkeypatch.setattr(
        "autotrade.utils.news_aggregator.NewsAggregator",
        _Aggregator,
    )

    state = {
        "watchlist": [
            {"symbol": "AAA", "recommendation": "BUY"},
            {"symbol": "BBB", "recommendation": "BUY"},
        ],
        "workflow_completion": {},
    }

    metrics = agent._enforce_top_watchlist_news_coverage(state)

    assert metrics["passed"] is False
    assert metrics["covered_count"] == 1
    assert metrics["failed_symbols"] == ["BBB"]
    details = state["workflow_completion"]["news_coverage_details"]
    assert details["AAA"]["available"] is True
    assert details["AAA"]["top_headline"] == "AAA raises guidance after earnings beat"
    assert details["BBB"]["available"] is False
    assert state["watchlist"][0]["news_coverage"] == "partial"
    assert state["watchlist"][0]["has_catalyst"] is True


def test_evaluate_overnight_breadth_prefers_direct_parquet_count(monkeypatch, tmp_path):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _noop_logger()
    agent.overnight_engine = SimpleNamespace(regime_scoring_mode="balanced")
    agent.OVERNIGHT_BREADTH_MIN_SYMBOLS = 2000

    parquet_path = tmp_path / "daily_features.parquet"
    pd.DataFrame({"ticker": ["AAA", "BBB", "AAA", "CCC"]}).to_parquet(parquet_path)

    monkeypatch.setattr(
        "autotrade.core.autonomous_agent.get_config",
        lambda: SimpleNamespace(
            data=SimpleNamespace(
                daily_features_parquet=str(parquet_path),
                downday_root=str(tmp_path),
            )
        ),
    )

    state = {
        "watchlist": _mk_watchlist(200),
        "all_candidates": _mk_watchlist(400),
        "researched": {"AAA": {"recommendation": "BUY"}},
        "workflow_completion": {},
    }

    metrics = agent._evaluate_overnight_breadth(state)

    assert metrics["prefilter_total"] == 3
    assert metrics["scanned_count"] == 400
    assert metrics["all_candidates_total"] == 400
    assert metrics["watchlist_total"] == 200
    assert metrics["passed"] is False


def test_evaluate_overnight_breadth_uses_candidate_pool_when_parquet_probe_fails(
    monkeypatch,
):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _noop_logger()
    agent.overnight_engine = SimpleNamespace(regime_scoring_mode="balanced")
    agent.OVERNIGHT_BREADTH_MIN_SYMBOLS = 200

    monkeypatch.setattr(
        "autotrade.core.autonomous_agent.get_config",
        lambda: SimpleNamespace(
            data=SimpleNamespace(
                daily_features_parquet="missing.parquet",
                downday_root=".",
            )
        ),
    )

    class _BrokenScreener:
        def __init__(self, parent_logger=None, scoring_mode=None):
            self.parent_logger = parent_logger
            self.scoring_mode = scoring_mode

        def _get_prepared_history_session(self):
            raise NameError("DataFrame is not defined")

    monkeypatch.setattr(
        "autotrade.signals.screener_v2.ScreenerV2",
        _BrokenScreener,
    )

    state = {
        "watchlist": _mk_watchlist(200),
        "all_candidates": _mk_watchlist(401),
        "researched": {"AAA": {"recommendation": "BUY"}},
        "workflow_completion": {},
    }

    metrics = agent._evaluate_overnight_breadth(state)

    assert metrics["prefilter_total"] == 0
    assert metrics["scanned_count"] == 401
    assert metrics["passed"] is True
    assert state["workflow_completion"]["breadth_passed"] is True
