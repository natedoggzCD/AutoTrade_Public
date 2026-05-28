from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from autotrade.core.autonomous_agent import AutonomousAgent, MarketPhase


def _noop_logger():
    return SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )


def test_strategy_pool_snapshot_reads_hash_and_count(tmp_path: Path, monkeypatch):
    strategy_path = tmp_path / "data" / "strategy_lab"
    strategy_path.mkdir(parents=True, exist_ok=True)
    payload = [{"strategy_name": "s1"}, {"strategy_name": "s2"}]
    (strategy_path / "validated_strategies.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    monkeypatch.setattr("autotrade.core.autonomous_agent.PROJECT_DIR", tmp_path)
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _noop_logger()

    snapshot = agent._get_strategy_pool_snapshot()

    assert snapshot["path"] == "data/strategy_lab/validated_strategies.json"
    assert snapshot["strategy_count"] == 2
    assert snapshot["sha256"]
    assert snapshot["mtime_iso"]


def test_force_rescan_keeps_existing_research_and_loads_delta_only(
    monkeypatch, tmp_path: Path
):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _noop_logger()
    agent.overnight_engine = SimpleNamespace(
        disable_new_discovery=False,
        current_regime="neutral",
        regime_scoring_mode="balanced",
        regime_scan_types=[],
    )
    monkeypatch.setattr("autotrade.core.autonomous_agent.PLANS_DIR", tmp_path)

    def fake_screener(*args, **kwargs):
        return [
            {"ticker": "AAPL", "price": 100.0},
            {"ticker": "MSFT", "price": 110.0},
            {"ticker": "NVDA", "price": 120.0},
        ]

    monkeypatch.setattr(
        "autotrade.signals.screener_v2.get_entry_candidates",
        fake_screener,
    )
    monkeypatch.setattr(
        "autotrade.signals.screener_v2.get_last_screen_run_diagnostics",
        lambda: {
            "status": "ok",
            "reason": None,
            "blocking_reasons": [],
            "result_count": 3,
        },
    )
    monkeypatch.setattr(
        "autotrade.signals.strategy_pool.load_validated_strategies",
        lambda: [],
    )

    state = {
        "researched": {"AAPL": {"recommendation": "BUY"}},
        "all_candidates": [{"symbol": "MSFT", "recommendation": "BUY"}],
        "watchlist": [{"symbol": "GOOG"}],
        "rejected": [{"symbol": "TSLA", "reason": "NO TRADE"}],
        "discovery_queue": [],
        "screener_cache": [{"ticker": "OLD", "price": 10.0}],
        "screener_cache_exhausted": True,
    }

    agent._load_more_candidates(state, force_rescan=True)

    assert "AAPL" in state["researched"]
    assert any(row.get("symbol") == "MSFT" for row in state["all_candidates"])
    assert state["discovery_queue"] == ["NVDA"]


def test_load_more_candidates_marks_core_data_block_without_exhausting_retries(
    monkeypatch, tmp_path: Path
):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _noop_logger()
    agent.overnight_engine = SimpleNamespace(
        disable_new_discovery=False,
        current_regime="neutral",
        regime_scoring_mode="balanced",
        regime_scan_types=[],
    )
    monkeypatch.setattr("autotrade.core.autonomous_agent.PLANS_DIR", tmp_path)
    monkeypatch.setattr(
        "autotrade.signals.screener_v2.get_entry_candidates",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "autotrade.signals.screener_v2.get_last_screen_run_diagnostics",
        lambda: {
            "status": "blocked",
            "reason": "core_data_not_ready",
            "blocking_reasons": ["core_data_stale:2026-03-11->2026-03-12"],
        },
    )
    monkeypatch.setattr(
        "autotrade.signals.strategy_pool.load_validated_strategies",
        lambda: [],
    )

    state = {
        "researched": {},
        "all_candidates": [],
        "watchlist": [],
        "rejected": [],
        "discovery_queue": [],
        "screener_cache": [],
        "screener_cache_exhausted": False,
        "workflow_completion": {},
    }

    agent._load_more_candidates(state)

    assert state["discovery_queue"] == []
    assert state["discovery_blocked_reason"] == "core_data_not_ready"
    assert state["screener_cache"] == []
    assert state["screener_cache_exhausted"] is False


def test_strategy_pool_delta_refresh_updates_state_and_plan_markers(monkeypatch):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _noop_logger()
    agent.WATCHLIST_SIZE = 2
    agent.MIN_WATCHLIST_SIZE = 2
    agent.OVERNIGHT_COMPLETION_MIN_WATCHLIST_SIZE = 2

    state = {
        "research_complete": True,
        "watchlist": [{"symbol": "MSFT", "recommendation": "BUY", "confidence": 80}],
        "discovery_queue": [],
        "researched": {"MSFT": {"recommendation": "BUY", "confidence": 80}},
        "all_candidates": [
            {"symbol": "MSFT", "recommendation": "BUY", "confidence": 80}
        ],
        "rejected": [],
        "sectors": {"Technology": 1},
        "screener_metadata": {},
        "workflow_completion": {
            "watchlist_selected": True,
            "game_plan_generated": True,
        },
    }

    def fake_load_more_candidates(s, force_rescan=False, min_watchlist_recovery=False):
        if force_rescan:
            s.setdefault("discovery_queue", []).append("NVDA")

    monkeypatch.setattr(agent, "_load_more_candidates", fake_load_more_candidates)
    monkeypatch.setattr(
        agent,
        "_parallel_research_batch",
        lambda batch: {
            "NVDA": {
                "recommendation": "BUY",
                "confidence": 77,
                "final_score": 68,
                "backtest": {"win_rate": 56},
            }
        },
    )
    monkeypatch.setattr(agent, "_get_sector", lambda symbol: "Technology")
    monkeypatch.setattr(
        agent,
        "_select_best_watchlist",
        lambda s: s.update(
            {
                "watchlist": sorted(
                    s.get("all_candidates", []),
                    key=lambda row: row.get("confidence", 0),
                    reverse=True,
                )[:5]
            }
        ),
    )
    monkeypatch.setattr(agent, "_generate_morning_game_plan", lambda s: True)
    monkeypatch.setattr(agent, "_save_overnight_state", lambda s: None)

    snapshot = {
        "path": "data/strategy_lab/validated_strategies.json",
        "mtime_iso": "2026-02-20T04:10:37",
        "sha256": "new-sha",
        "strategy_count": 5,
    }
    result = agent._run_strategy_pool_delta_refresh(
        state, snapshot, source="test", max_batches=2
    )

    assert result["refreshed_plan"] is True
    assert result["delta_symbols"] >= 1
    assert state["research_complete"] is True
    assert state["strategy_pool_snapshot"] == snapshot
    assert state.get("strategy_pool_last_refresh_at")
    assert any(row.get("symbol") == "NVDA" for row in state["all_candidates"])


def test_strategy_pool_delta_refresh_requires_min_watchlist(monkeypatch):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _noop_logger()
    agent.WATCHLIST_SIZE = 3
    agent.MIN_WATCHLIST_SIZE = 3
    agent.OVERNIGHT_COMPLETION_MIN_WATCHLIST_SIZE = 3

    state = {
        "research_complete": False,
        "watchlist": [],
        "discovery_queue": [],
        "researched": {},
        "all_candidates": [],
        "rejected": [],
        "sectors": {},
        "screener_metadata": {},
        "workflow_completion": {},
    }

    def fake_load_more_candidates(s, force_rescan=False, min_watchlist_recovery=False):
        if force_rescan:
            s.setdefault("discovery_queue", []).append("NVDA")

    monkeypatch.setattr(agent, "_load_more_candidates", fake_load_more_candidates)
    monkeypatch.setattr(
        agent,
        "_parallel_research_batch",
        lambda batch: {
            "NVDA": {
                "recommendation": "BUY",
                "confidence": 77,
                "final_score": 68,
                "backtest": {"win_rate": 56},
            }
        },
    )
    monkeypatch.setattr(agent, "_get_sector", lambda symbol: "Technology")
    monkeypatch.setattr(
        agent,
        "_select_best_watchlist",
        lambda s: s.update(
            {
                # Deliberately under target (1 < MIN_WATCHLIST_SIZE=3)
                "watchlist": [
                    {"symbol": "NVDA", "recommendation": "BUY", "confidence": 77}
                ]
            }
        ),
    )
    monkeypatch.setattr(agent, "_generate_morning_game_plan", lambda s: True)
    monkeypatch.setattr(agent, "_save_overnight_state", lambda s: None)

    snapshot = {
        "path": "data/strategy_lab/validated_strategies.json",
        "mtime_iso": "2026-02-20T04:10:37",
        "sha256": "new-sha",
        "strategy_count": 5,
    }
    result = agent._run_strategy_pool_delta_refresh(
        state, snapshot, source="test", max_batches=2
    )

    assert result["refreshed_plan"] is False
    assert state["research_complete"] is False
    assert (
        state["workflow_completion"]["completion_reason"]
        == "strategy_pool_refresh_test_insufficient_watchlist"
    )
    assert state["workflow_completion"]["completed_at"] is None


def test_no_intraday_strategy_hot_refresh(monkeypatch, tmp_path: Path):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _noop_logger()
    now_et = datetime(2026, 2, 20, 10, 0)
    agent.scheduler = SimpleNamespace(
        get_current_time=lambda: now_et,
        get_market_phase=lambda: MarketPhase.MARKET_HOURS,
    )
    agent._premarket_state = {
        "last_plan_hash": None,
        "last_gap_snapshot": {},
        "last_full_scan_time": now_et - timedelta(minutes=1),
        "last_validation_hash": None,
        "last_validation_output": "",
        "strategy_pool_refresh_token": None,
    }
    agent.task_router = None

    monkeypatch.setattr(
        agent,
        "_check_research_freshness",
        lambda persist_metadata=True, now_et=None: {
            "age_hours": 1.0,
            "age_stale": False,
            "workflow_complete": True,
            "workflow_reason": "ok",
            "warning": False,
            "max_age_hours": 18.0,
            "is_fresh": True,
            "checked_at": "2026-02-20T10:00:00",
        },
    )
    monkeypatch.setattr(
        agent,
        "_capture_premarket_gap_snapshot",
        lambda signals, limit=60: {},
    )
    monkeypatch.setattr(
        agent,
        "_get_strategy_pool_snapshot",
        lambda: {
            "path": "data/strategy_lab/validated_strategies.json",
            "mtime_iso": "2026-02-20T04:10:37",
            "sha256": "new",
            "strategy_count": 5,
        },
    )
    monkeypatch.setattr(
        agent,
        "_load_overnight_state",
        lambda: {
            "strategy_pool_snapshot": {
                "path": "data/strategy_lab/validated_strategies.json",
                "mtime_iso": "2026-02-20T01:27:13",
                "sha256": "old",
                "strategy_count": 5,
            }
        },
    )

    called = {"refresh": 0}
    monkeypatch.setattr(
        agent,
        "_run_strategy_pool_delta_refresh",
        lambda state, current_snapshot, source="premarket", max_batches=10: (
            called.update({"refresh": called["refresh"] + 1}) or {}
        ),
    )

    plan = {
        "strategy_pool_snapshot": {
            "path": "data/strategy_lab/validated_strategies.json",
            "mtime_iso": "2026-02-20T01:27:13",
            "sha256": "old",
            "strategy_count": 5,
        },
        "signals": [{"symbol": "AAPL", "entry_price": 100.0, "score": 80.0}],
    }
    fingerprint = [{"symbol": "AAPL", "entry_price": 100.0, "score": 80.0}]
    agent._premarket_state["last_plan_hash"] = agent._hash_payload(fingerprint)

    monkeypatch.setattr("autotrade.core.autonomous_agent.PLANS_DIR", tmp_path)
    (tmp_path / "morning_game_plan_20260220.json").write_text(
        json.dumps(plan), encoding="utf-8"
    )

    wait_seconds = agent._run_premarket_cycle(cycle_count=1)

    assert wait_seconds == 60
    assert called["refresh"] == 0


def test_premarket_session_plan_includes_strategy_pool_snapshot(monkeypatch, tmp_path: Path):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _noop_logger()
    now_et = datetime(2026, 2, 20, 8, 5)
    agent.scheduler = SimpleNamespace(
        get_current_time=lambda: now_et,
        get_market_phase=lambda: MarketPhase.PREMARKET,
    )
    agent.research_freshness_cfg = SimpleNamespace(premarket_block_if_stale=True)
    agent._premarket_state = {
        "last_plan_hash": None,
        "last_gap_snapshot": {},
        "last_full_scan_time": None,
        "last_validation_hash": None,
        "last_validation_output": "",
        "strategy_pool_refresh_token": None,
    }
    agent.task_router = None
    agent._decision_claw_review_morning_plan = lambda **kwargs: None
    agent.plan_generator = SimpleNamespace(
        get_current_positions=lambda: [],
        _coerce_float=lambda value, default=0.0: float(value or default),
        generate_plan=lambda plan_path: {
            "cleanup_orders": [],
            "entry_orders": [],
        },
    )

    monkeypatch.setattr(
        agent,
        "_check_research_freshness",
        lambda persist_metadata=True, now_et=None: {
            "age_hours": 1.0,
            "age_stale": False,
            "workflow_complete": True,
            "workflow_reason": "ok",
            "warning": False,
            "max_age_hours": 18.0,
            "is_fresh": True,
            "checked_at": "2026-02-20T08:05:00",
        },
    )
    snapshot = {
        "path": "data/strategy_lab/validated_strategies.json",
        "mtime_iso": "2026-02-20T04:10:37",
        "sha256": "new-sha",
        "strategy_count": 5,
    }
    monkeypatch.setattr(agent, "_get_strategy_pool_snapshot", lambda: snapshot)
    monkeypatch.setattr(
        agent, "_load_overnight_state", lambda: {"strategy_pool_snapshot": snapshot}
    )
    monkeypatch.setattr(
        agent, "_capture_premarket_gap_snapshot", lambda signals, limit=60: {}
    )
    monkeypatch.setattr(
        agent,
        "_validate_premarket_gap_policy",
        lambda signals: {
            "signals": list(signals),
            "removed_gapup": [],
            "removed_gapdown": [],
            "adjusted_prices": [],
            "summary": {
                "policy_enabled": True,
                "analyzed": len(signals),
                "with_data": len(signals),
                "missing_data": 0,
                "extreme_drops": 0,
                "repriced": 0,
            },
        },
    )

    class DummyPreMarketAgent:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, pool, holdings=None):
            return SimpleNamespace(
                total_analyzed=len(pool),
                total_dropped=0,
                total_upgraded=0,
                earnings_alerts=[],
                drops=[],
            )

        def get_updated_watchlist(self, pool, pm_plan):
            return list(pool)

    monkeypatch.setattr(
        "autotrade.core.premarket_agent.PreMarketAgent",
        DummyPreMarketAgent,
    )

    monkeypatch.setattr("autotrade.core.autonomous_agent.PLANS_DIR", tmp_path)
    plan = {
        "strategy_pool_snapshot": snapshot,
        "signals": [{"symbol": "AAPL", "entry_price": 100.0, "confidence": 80.0}],
    }
    (tmp_path / "morning_game_plan_20260220.json").write_text(
        json.dumps(plan), encoding="utf-8"
    )

    wait_seconds = agent._run_premarket_cycle(cycle_count=1)

    session_plan = tmp_path / "morning_game_plan_20260220.json"
    assert wait_seconds == 60
    assert session_plan.exists()
    payload = json.loads(session_plan.read_text(encoding="utf-8"))
    assert payload.get("strategy_pool_snapshot") == snapshot


def test_premarket_session_plan_preserves_signals_without_hard_regime_gate(
    monkeypatch, tmp_path: Path
):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _noop_logger()
    now_et = datetime(2026, 2, 20, 8, 5)
    agent.scheduler = SimpleNamespace(
        get_current_time=lambda: now_et,
        get_market_phase=lambda: MarketPhase.PREMARKET,
    )
    agent.research_freshness_cfg = SimpleNamespace(premarket_block_if_stale=True)
    agent._premarket_state = {
        "last_plan_hash": None,
        "last_gap_snapshot": {},
        "last_full_scan_time": None,
        "last_validation_hash": None,
        "last_validation_output": "",
        "strategy_pool_refresh_token": None,
    }
    agent.task_router = None
    agent._decision_claw_review_morning_plan = lambda **kwargs: None
    agent.plan_generator = SimpleNamespace(
        get_current_positions=lambda: [],
        _coerce_float=lambda value, default=0.0: float(value or default),
        generate_plan=lambda plan_path: {
            "cleanup_orders": [],
            "entry_orders": [],
        },
    )

    monkeypatch.setattr(
        agent,
        "_check_research_freshness",
        lambda persist_metadata=True, now_et=None: {
            "age_hours": 1.0,
            "age_stale": False,
            "workflow_complete": True,
            "workflow_reason": "ok",
            "warning": False,
            "max_age_hours": 18.0,
            "is_fresh": True,
            "checked_at": "2026-02-20T08:05:00",
        },
    )
    snapshot = {
        "path": "data/strategy_lab/validated_strategies.json",
        "mtime_iso": "2026-02-20T04:10:37",
        "sha256": "new-sha",
        "strategy_count": 5,
    }
    monkeypatch.setattr(agent, "_get_strategy_pool_snapshot", lambda: snapshot)
    monkeypatch.setattr(
        agent, "_load_overnight_state", lambda: {"strategy_pool_snapshot": snapshot}
    )
    monkeypatch.setattr(
        agent, "_capture_premarket_gap_snapshot", lambda signals, limit=60: {}
    )
    monkeypatch.setattr(
        agent,
        "_validate_premarket_gap_policy",
        lambda signals: {
            "signals": list(signals),
            "removed_gapup": [],
            "removed_gapdown": [],
            "adjusted_prices": [],
            "summary": {
                "policy_enabled": True,
                "analyzed": len(signals),
                "with_data": len(signals),
                "missing_data": 0,
                "extreme_drops": 0,
                "repriced": 0,
            },
        },
    )

    class DummyPreMarketAgent:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, pool, holdings=None):
            return SimpleNamespace(
                total_analyzed=len(pool),
                total_dropped=0,
                total_upgraded=0,
                earnings_alerts=[],
                drops=[],
            )

        def get_updated_watchlist(self, pool, pm_plan):
            return list(pool)

    monkeypatch.setattr(
        "autotrade.core.premarket_agent.PreMarketAgent",
        DummyPreMarketAgent,
    )

    monkeypatch.setattr("autotrade.core.autonomous_agent.PLANS_DIR", tmp_path)
    plan = {
        "strategy_pool_snapshot": snapshot,
        "signals": [{"symbol": "AAPL", "entry_price": 100.0, "confidence": 80.0}],
        "resolved_regime": {
            "regime": "SELL_OFF",
            "allow_new_longs": False,
            "plan_source": "morning_game_plan_20260220.json",
        },
    }
    (tmp_path / "morning_game_plan_20260220.json").write_text(
        json.dumps(plan), encoding="utf-8"
    )

    wait_seconds = agent._run_premarket_cycle(cycle_count=1)

    session_plan = tmp_path / "morning_game_plan_20260220.json"
    assert wait_seconds == 60
    assert session_plan.exists()
    payload = json.loads(session_plan.read_text(encoding="utf-8"))
    assert payload["signals"][0]["symbol"] == "AAPL"
    assert payload["premarket_regime_bias"]["bias_only"] is True
    assert payload["resolved_regime"]["allow_new_longs"] is True
