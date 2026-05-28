from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import pytest

from autotrade.core.autonomous_agent import AutonomousAgent
from autotrade.core.premarket_manager import ET
from autotrade.core.research_artifacts import ResearchArtifactBundle, save_research_artifact_bundle
from tests.test_premarket_manager import (
    DeterministicPremarketManager,
    DummyAnalyzer,
    DummyNews,
    DummyStocktwits,
)


def test_generate_morning_game_plan_assigns_conviction_priority(monkeypatch, tmp_path: Path):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    agent._youtube_context = {}

    monkeypatch.setattr("autotrade.core.autonomous_agent.PLANS_DIR", tmp_path)

    state = {
        "watchlist": [
            {
                "symbol": "AAA",
                "recommendation": "BUY",
                "confidence": 82,
                "ranking_score": 82.0,
                "sector": "Technology",
                "catalyst_score": 0.0,
            },
            {
                "symbol": "BBB",
                "recommendation": "BUY",
                "confidence": 78,
                "ranking_score": 76.0,
                "sector": "Energy",
                "catalyst_score": 0.8,
                "catalyst_monitor_score_delta": 4.0,
            },
        ],
        "sectors": {"Technology": 1, "Energy": 1},
        "discovery_registry": {
            "items": {
                "BBB:gap": {"symbol": "BBB", "family": "gap"},
                "BBB:volatility": {"symbol": "BBB", "family": "volatility"},
            },
            "by_symbol": {"BBB": ["gap", "volatility"]},
        },
        "secondary_research": {
            "catalyst_monitor": {
                "BBB": {"score_delta": 4.0},
            }
        },
    }

    ok = agent._generate_morning_game_plan(state)

    assert ok is True
    plan_path = next(tmp_path.glob("morning_game_plan_*.json"))
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert payload["signals"][0]["symbol"] == "BBB"
    assert payload["signals"][0]["conviction_priority"] == 1
    assert payload["signals"][0]["conviction_priority_score"] > payload["signals"][1]["conviction_priority_score"]
    assert payload["signals"][0]["target_allocation_pct"] > payload["signals"][1]["target_allocation_pct"]


def test_generate_morning_game_plan_prefers_empirical_mid_band(monkeypatch, tmp_path: Path):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    agent._youtube_context = {}

    monkeypatch.setattr("autotrade.core.autonomous_agent.PLANS_DIR", tmp_path)
    monkeypatch.setattr("autotrade.core.autonomous_agent.PROJECT_DIR", tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "eod_review_2026-03-16.json").write_text(
        json.dumps(
            {
                "date": "2026-03-16",
                "score_buckets": {
                    "65-80": {"count": 12, "win_rate": 0.58, "avg_pnl": 4.2},
                    "80+": {"count": 8, "win_rate": 0.12, "avg_pnl": -8.5},
                },
            }
        ),
        encoding="utf-8",
    )

    state = {
        "watchlist": [
            {
                "symbol": "HOT",
                "recommendation": "BUY",
                "confidence": 91,
                "ranking_score": 88.0,
                "final_score": 88.0,
                "sector": "Technology",
                "catalyst_score": 0.0,
            },
            {
                "symbol": "MID",
                "recommendation": "BUY",
                "confidence": 78,
                "ranking_score": 78.0,
                "final_score": 78.0,
                "sector": "Technology",
                "catalyst_score": 0.0,
            },
        ],
        "sectors": {"Technology": 2},
    }

    ok = agent._generate_morning_game_plan(state)

    assert ok is True
    plan_path = next(tmp_path.glob("morning_game_plan_*.json"))
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert payload["signals"][0]["symbol"] == "MID"
    assert payload["signals"][0]["empirical_band_adjustment"] > 0
    assert payload["signals"][1]["empirical_band_adjustment"] < 0


def test_premarket_manager_applies_overnight_conviction_priority_bias(tmp_path: Path):
    bundle = ResearchArtifactBundle(
        trade_date="2026-02-11",
        generated_at_et="2026-02-11T04:00:00-05:00",
        full_watchlist=[
            {"symbol": "AAA", "conviction_priority": 2, "conviction_priority_score": 68.0},
            {"symbol": "BBB", "conviction_priority": 1, "conviction_priority_score": 98.0},
        ],
        top_picks=[
            {"symbol": "BBB", "conviction_priority": 1, "conviction_priority_score": 98.0}
        ],
    )
    save_research_artifact_bundle(bundle, output_dir=tmp_path)

    manager = DeterministicPremarketManager(
        output_dir=tmp_path,
        premarket_analyzer=DummyAnalyzer(),
        news_aggregator=DummyNews(),
        stocktwits_scraper=DummyStocktwits(),
        max_watchlist_symbols=10,
    )

    handoff = manager.run_cycle(
        watchlist=[{"ticker": "AAA"}],
        holdings=[],
        now_et=datetime(2026, 2, 11, 8, 45, tzinfo=ET),
    )

    ranked = handoff["ranked_watchlist"]
    assert ranked[0]["symbol"] == "BBB"
    assert ranked[0]["conviction_priority"] == 1
    assert "Overnight conviction priority #1" in ranked[0]["rationale"][0]


def test_generate_morning_game_plan_widens_signal_tiers(monkeypatch, tmp_path: Path):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    agent._youtube_context = {}

    monkeypatch.setattr("autotrade.core.autonomous_agent.PLANS_DIR", tmp_path)

    state = {
        "watchlist": [
            {
                "symbol": f"S{i:03d}",
                "recommendation": "BUY",
                "confidence": 95 - (i * 0.1),
                "ranking_score": 95.0 - (i * 0.1),
                "final_score": 95.0 - (i * 0.1),
                "sector": "Technology",
                "catalyst_score": 0.0,
            }
            for i in range(120)
        ],
        "sectors": {"Technology": 120},
    }

    ok = agent._generate_morning_game_plan(state)

    assert ok is True
    plan_path = next(tmp_path.glob("morning_game_plan_*.json"))
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert len(payload["signals"]) == 50
    assert len(payload["actionable_top50"]) == 100
    assert len(payload["overflow_signals"]) == 20


def test_overnight_cycle_already_ready_gate_returns_wait_interval(monkeypatch):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    agent.scheduler = SimpleNamespace(
        get_current_time=lambda: datetime(2026, 2, 11, 3, 30, tzinfo=ET)
    )
    agent._update_workflow_state_flag = lambda *args, **kwargs: None
    agent._load_overnight_state = lambda: {
        "research_complete": True,
        "watchlist": [{"symbol": "AAA"}],
        "workflow_completion": {"youtube_ready": True},
    }
    agent._morning_game_plan_exists = lambda: True
    agent._build_incremental_overnight_research_summary = lambda state: {}
    agent._write_evening_summary = lambda summary: None
    agent._append_jsonl = lambda *args, **kwargs: None
    triggers = []
    agent._run_secondary_post_plan_jobs = lambda **kwargs: triggers.append(kwargs["trigger_reason"])

    wait_seconds = agent._run_overnight_cycle(cycle_count=1, fresh_run=False)

    assert wait_seconds == 300
    assert triggers == ["already_ready_gate"]


def test_overnight_cycle_does_not_short_circuit_when_youtube_still_pending(monkeypatch):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    agent.scheduler = SimpleNamespace(
        get_current_time=lambda: datetime(2026, 2, 11, 3, 30, tzinfo=ET),
    )
    agent._update_workflow_state_flag = lambda *args, **kwargs: None
    agent._load_overnight_state = lambda: {
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
    triggers = []
    agent._run_secondary_post_plan_jobs = lambda **kwargs: triggers.append(
        kwargs["trigger_reason"]
    )

    class _GatePassed(RuntimeError):
        pass

    def _raise_after_gate(source=None):
        raise _GatePassed(source)

    agent._refresh_strategy_failsafe = _raise_after_gate

    with monkeypatch.context():
        with pytest.raises(_GatePassed):
            agent._run_overnight_cycle(cycle_count=1, fresh_run=False)

    assert triggers == []
