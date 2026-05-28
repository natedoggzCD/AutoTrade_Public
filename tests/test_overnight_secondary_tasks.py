import json
from pathlib import Path
from types import SimpleNamespace

from autotrade.core.autonomous_agent import AutonomousAgent


def _make_agent() -> AutonomousAgent:
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.overnight_secondary_cfg = SimpleNamespace(
        enabled=True,
        run_when_research_complete=True,
        vl_benchmark_enabled=False,
        top_pick_research_enabled=False,
        historical_revisit_enabled=False,
    )
    return agent


def test_secondary_job_due_respects_interval():
    agent = _make_agent()
    state = {}

    assert (
        agent._secondary_job_due(state, "job_a", cycle_count=10, interval_cycles=5)
        is True
    )
    assert (
        agent._secondary_job_due(state, "job_a", cycle_count=12, interval_cycles=5)
        is False
    )
    assert (
        agent._secondary_job_due(state, "job_a", cycle_count=15, interval_cycles=5)
        is True
    )


def test_get_secondary_top_symbols_actionable_filtering():
    agent = _make_agent()
    state = {
        "watchlist": [
            {"symbol": "AAA", "recommendation": "WATCH", "ranking_score": 90},
            {"symbol": "BBB", "recommendation": "BUY", "ranking_score": 80},
            {"symbol": "CCC", "recommendation": "STRONG BUY", "ranking_score": 95},
            {"symbol": "DDD", "recommendation": "WEAK BUY", "ranking_score": 85},
        ]
    }

    picks = agent._get_secondary_top_symbols(state, limit=5, actionable_only=True)
    symbols = [p["symbol"] for p in picks]
    assert symbols == ["CCC", "DDD", "BBB"]


def test_get_secondary_top_symbols_excludes_market_context_symbols():
    agent = _make_agent()
    state = {
        "watchlist": [
            {"symbol": "SPY", "recommendation": "BUY", "ranking_score": 99},
            {"symbol": "QQQ", "recommendation": "BUY", "ranking_score": 98},
            {"symbol": "NVDA", "recommendation": "BUY", "ranking_score": 97},
        ]
    }

    picks = agent._get_secondary_top_symbols(state, limit=5, actionable_only=True)
    symbols = [p["symbol"] for p in picks]
    assert symbols == ["NVDA"]


def test_get_secondary_top_symbols_without_actionable_filter():
    agent = _make_agent()
    state = {
        "watchlist": [
            {"symbol": "AAA", "recommendation": "WATCH", "ranking_score": 90},
            {"symbol": "BBB", "recommendation": "BUY", "ranking_score": 80},
        ]
    }

    picks = agent._get_secondary_top_symbols(state, limit=2, actionable_only=False)
    symbols = [p["symbol"] for p in picks]
    assert symbols == ["AAA", "BBB"]


def test_get_secondary_rotating_symbols_advances_cursor():
    agent = _make_agent()
    state = {}
    picks = [{"symbol": s} for s in ["AAA", "BBB", "CCC", "DDD"]]

    first = agent._get_secondary_rotating_symbols(
        state=state,
        job_name="top_pick_research",
        picks=picks,
        symbols_cap=2,
        rotation_enabled=True,
    )
    second = agent._get_secondary_rotating_symbols(
        state=state,
        job_name="top_pick_research",
        picks=picks,
        symbols_cap=2,
        rotation_enabled=True,
    )
    third = agent._get_secondary_rotating_symbols(
        state=state,
        job_name="top_pick_research",
        picks=picks,
        symbols_cap=2,
        rotation_enabled=True,
    )

    assert [p["symbol"] for p in first] == ["AAA", "BBB"]
    assert [p["symbol"] for p in second] == ["CCC", "DDD"]
    assert [p["symbol"] for p in third] == ["AAA", "BBB"]


def test_top_pick_research_rotates_and_updates_researched_state(tmp_path, monkeypatch):
    agent = _make_agent()
    agent.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    agent.web_researcher = SimpleNamespace(
        research_stock=lambda symbol: {
            "sentiment": "bullish",
            "confidence": 0.8,
            "fresh_news": True,
            "news": [{"title": f"{symbol} catalyst"}],
            "catalysts": ["earnings"],
            "model_used": "test-model",
            "researched_at": "2026-03-10T00:00:00",
        }
    )
    agent._deep_research_symbol_safe = lambda symbol: {
        "symbol": symbol,
        "recommendation": "BUY",
        "confidence": 72,
        "final_score": 74,
        "entry_price": 10.0,
        "target": 11.2,
        "stop_loss": 9.5,
        "backtest": {
            "source": "per_symbol_strategy_pool",
            "win_rate": 58.0,
            "total_trades": 40,
            "profit_factor": 1.3,
            "top_k_used": 5,
        },
    }
    agent._get_random_universe_symbols = lambda **kwargs: []
    agent.overnight_secondary_cfg = SimpleNamespace(
        enabled=True,
        run_when_research_complete=True,
        vl_benchmark_enabled=False,
        top_pick_research_enabled=True,
        top_pick_research_cycle_interval=1,
        top_pick_research_symbols=4,
        top_pick_research_max_runtime_seconds=120,
        top_pick_research_universe_symbols=4,
        top_pick_research_rotation_enabled=True,
        top_pick_research_actionable_only=True,
        top_pick_research_include_deep_dive=True,
        top_pick_research_include_web_research=True,
        top_pick_research_update_researched_state=True,
        historical_revisit_enabled=False,
    )
    monkeypatch.setattr("autotrade.core.autonomous_agent.PROJECT_DIR", tmp_path)
    state = {
        "watchlist": [
            {"symbol": "AAA", "recommendation": "BUY", "ranking_score": 95},
            {"symbol": "BBB", "recommendation": "BUY", "ranking_score": 90},
            {"symbol": "CCC", "recommendation": "BUY", "ranking_score": 85},
            {"symbol": "DDD", "recommendation": "BUY", "ranking_score": 80},
        ],
        "secondary_research": {},
    }

    first = agent._run_secondary_top_pick_research(state=state, cycle_count=1)
    second = agent._run_secondary_top_pick_research(state=state, cycle_count=2)

    assert first["success"] is True
    assert second["success"] is True
    assert Path(first["artifact_path"]).name.startswith("top_pick_deep_research_")
    researched = state.get("researched", {})
    assert set(researched.keys()) == {"AAA", "BBB", "CCC", "DDD"}
    rotation = state.get("secondary_research", {}).get("rotation_cursor_by_job", {})
    assert rotation.get("top_pick_research") == 0


def test_top_pick_research_updates_watchlist_scores_from_catalysts(
    tmp_path, monkeypatch
):
    agent = _make_agent()
    agent.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    agent.web_researcher = SimpleNamespace(
        research_stock=lambda symbol: {
            "symbol": symbol,
            "sentiment": "bullish",
            "confidence": 88,
            "fresh_news": True,
            "news": [{"title": f"{symbol} wins major contract"}],
            "catalysts": [f"{symbol} wins major contract", f"{symbol} raises guidance"],
            "model_used": "test-model",
            "researched_at": "2026-03-10T00:00:00",
        }
    )
    agent._deep_research_symbol_safe = lambda symbol: {
        "symbol": symbol,
        "recommendation": "BUY",
        "confidence": 72,
        "final_score": 74,
        "entry_price": 10.0,
        "target": 11.2,
        "stop_loss": 9.5,
        "backtest": {
            "source": "per_symbol_strategy_pool",
            "win_rate": 58.0,
            "total_trades": 40,
            "profit_factor": 1.3,
            "top_k_used": 5,
        },
    }
    agent.overnight_secondary_cfg = SimpleNamespace(
        enabled=True,
        run_when_research_complete=True,
        vl_benchmark_enabled=False,
        top_pick_research_enabled=True,
        top_pick_research_cycle_interval=1,
        top_pick_research_symbols=1,
        top_pick_research_max_runtime_seconds=120,
        top_pick_research_universe_symbols=2,
        top_pick_research_rotation_enabled=False,
        top_pick_research_actionable_only=True,
        top_pick_research_include_deep_dive=True,
        top_pick_research_include_web_research=True,
        top_pick_research_update_researched_state=True,
        historical_revisit_enabled=False,
    )
    monkeypatch.setattr("autotrade.core.autonomous_agent.PROJECT_DIR", tmp_path)
    state = {
        "watchlist": [
            {"symbol": "AAA", "recommendation": "BUY", "ranking_score": 70.0},
            {"symbol": "BBB", "recommendation": "BUY", "ranking_score": 65.0},
        ],
        "researched": {
            "AAA": {"symbol": "AAA", "recommendation": "BUY", "final_score": 74.0}
        },
        "secondary_research": {},
    }

    result = agent._run_secondary_top_pick_research(state=state, cycle_count=1)

    assert result["success"] is True
    assert result["watchlist_updates"] == 1
    aaa = next(row for row in state["watchlist"] if row["symbol"] == "AAA")
    assert aaa["ranking_score"] > 70.0
    assert aaa["dynamic_conviction_score"] == aaa["ranking_score"]
    assert aaa["has_catalyst"] is True
    assert aaa["catalyst_note"].startswith("AAA wins major contract")
    catalyst_state = state["secondary_research"]["catalyst_monitor"]["AAA"]
    assert catalyst_state["score_delta"] > 0
    assert state["researched"]["AAA"]["final_score"] > 74.0


def test_strategy_discovery_calls_auto_factory_and_merges_watchlist(
    tmp_path, monkeypatch
):
    agent = _make_agent()
    agent.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    agent.overnight_secondary_cfg = SimpleNamespace(
        enabled=True,
        run_when_research_complete=True,
        vl_benchmark_enabled=False,
        top_pick_research_enabled=False,
        historical_revisit_enabled=False,
        strategy_discovery_enabled=True,
        strategy_discovery_cycle_interval=1,
        strategy_discovery_lookback_days=90,
        strategy_discovery_universe_size=50,
        strategy_discovery_top_candidates=3,
        strategy_discovery_max_runtime_seconds=600,
        strategy_discovery_max_additions=20,
        strategy_discovery_min_profit_factor=1.2,
        strategy_discovery_min_win_rate=0.45,
    )

    # Mock auto_factory to return success
    factory_calls = []

    def fake_auto_factory(**kwargs):
        factory_calls.append(kwargs)
        return 0

    monkeypatch.setattr("autotrade.core.autonomous_agent.PROJECT_DIR", tmp_path)

    # Create validated_strategies_by_symbol.json with strong candidates
    import json

    lab_dir = tmp_path / "data" / "strategy_lab"
    lab_dir.mkdir(parents=True, exist_ok=True)
    (lab_dir / "validated_strategies_by_symbol.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-03-15T21:30:22.751708",
                "source_run_id": "auto_factory_winner_test",
                "top_k": 5,
                "symbols": {
                    "ACME": [
                        {
                            "strategy_name": "momentum_fast",
                            "setup_type": "momentum_breakout",
                            "symbol_metrics": {
                                "trades": 12,
                                "profit_factor": 1.5,
                                "win_rate": 0.52,
                            },
                        }
                    ],
                    "WEAK": [
                        {
                            "strategy_name": "pullback",
                            "setup_type": "pullback_support",
                            "symbol_metrics": {
                                "trades": 10,
                                "profit_factor": 0.8,
                                "win_rate": 0.35,
                            },
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("tools.strategy_lab.auto_factory", fake_auto_factory)

    state = {
        "watchlist": [{"symbol": "EXISTING", "recommendation": "BUY"}],
    }
    result = agent._run_secondary_strategy_discovery(state=state, cycle_count=10)

    assert result["success"] is True
    assert result["return_code"] == 0
    assert len(factory_calls) == 1
    assert factory_calls[0]["lookback_days"] == 90
    assert factory_calls[0]["universe_size"] == 50

    # ACME should be added (PF=1.5, WR=52%), WEAK should not (PF=0.8)
    symbols = [e["symbol"] for e in state["watchlist"]]
    assert "ACME" in symbols
    assert "WEAK" not in symbols

    # Check source tag
    acme_entry = next(e for e in state["watchlist"] if e["symbol"] == "ACME")
    assert acme_entry["source"] == "overnight_discovery"
    assert acme_entry["profit_factor"] == 1.5


def test_strategy_discovery_merges_legacy_flat_payload_shape(tmp_path, monkeypatch):
    agent = _make_agent()
    agent.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("autotrade.core.autonomous_agent.PROJECT_DIR", tmp_path)

    import json

    lab_dir = tmp_path / "data" / "strategy_lab"
    lab_dir.mkdir(parents=True, exist_ok=True)
    (lab_dir / "validated_strategies_by_symbol.json").write_text(
        json.dumps(
            {
                "flat": [
                    {
                        "strategy_name": "legacy_flat_payload",
                        "profit_factor": 1.4,
                        "win_rate": 51.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    state = {"watchlist": []}
    additions = agent._merge_strategy_discovery_into_watchlist(
        state=state,
        max_additions=20,
        min_profit_factor=1.2,
        min_win_rate=0.45,
    )

    assert additions == 1
    assert state["watchlist"][0]["symbol"] == "FLAT"
    assert state["watchlist"][0]["strategy_name"] == "legacy_flat_payload"


def test_strategy_discovery_handles_factory_failure(monkeypatch):
    agent = _make_agent()
    agent.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    agent.overnight_secondary_cfg = SimpleNamespace(
        enabled=True,
        run_when_research_complete=True,
        strategy_discovery_enabled=True,
        strategy_discovery_cycle_interval=1,
        strategy_discovery_lookback_days=90,
        strategy_discovery_universe_size=50,
        strategy_discovery_top_candidates=3,
        strategy_discovery_max_runtime_seconds=600,
        strategy_discovery_max_additions=20,
        strategy_discovery_min_profit_factor=1.2,
        strategy_discovery_min_win_rate=0.45,
    )

    monkeypatch.setattr(
        "tools.strategy_lab.auto_factory",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("GPU OOM")),
    )

    state = {"watchlist": []}
    result = agent._run_secondary_strategy_discovery(state=state, cycle_count=5)

    assert result["success"] is False
    assert "GPU OOM" in result["error"]
    # Watchlist unchanged
    assert state["watchlist"] == []


def test_run_secondary_post_plan_jobs_skips_vl_benchmark_even_if_enabled():
    agent = _make_agent()
    agent.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    agent.overnight_secondary_cfg = SimpleNamespace(
        enabled=True,
        run_when_research_complete=True,
        vl_benchmark_enabled=True,
        vl_benchmark_cycle_interval=1,
        top_pick_research_enabled=False,
        historical_revisit_enabled=False,
        signal_mining_enabled=False,
        strategy_discovery_enabled=False,
    )
    agent._secondary_job_due = lambda **kwargs: True
    agent._run_secondary_vl_benchmark_against_signals = lambda **kwargs: (
        _ for _ in ()
    ).throw(AssertionError("vl_benchmark should be skipped"))
    state = {"research_complete": True}

    agent._run_secondary_post_plan_jobs(
        state=state,
        cycle_count=10,
        trigger_reason="test",
    )

    secondary = state.get("secondary_research", {})
    assert secondary.get("history", []) == []


def test_run_secondary_post_plan_jobs_executes_top_pick_monitor_and_persists_state():
    agent = _make_agent()
    agent.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    saved_states = []
    agent._save_overnight_state = lambda state: saved_states.append(dict(state))
    agent.overnight_secondary_cfg = SimpleNamespace(
        enabled=True,
        run_when_research_complete=True,
        vl_benchmark_enabled=False,
        top_pick_research_enabled=True,
        top_pick_research_cycle_interval=1,
        historical_revisit_enabled=False,
        signal_mining_enabled=False,
        strategy_discovery_enabled=False,
    )
    agent._secondary_job_due = lambda **kwargs: True
    agent._run_secondary_top_pick_research = lambda **kwargs: {
        "success": True,
        "job": "top_pick_research",
        "watchlist_updates": 2,
    }
    state = {"research_complete": True, "watchlist": [{"symbol": "AAA"}]}

    agent._run_secondary_post_plan_jobs(
        state=state,
        cycle_count=12,
        trigger_reason="test",
    )

    secondary = state.get("secondary_research", {})
    history = secondary.get("history", [])
    assert len(history) == 1
    assert history[0]["job"] == "top_pick_research"
    assert history[0]["result"]["watchlist_updates"] == 2
    assert secondary["last_result_by_job"]["top_pick_research"]["cycle"] == 12
    assert saved_states


def test_historical_revisit_writes_degraded_artifact_when_no_candidates(
    tmp_path, monkeypatch
):
    plans_dir = Path(tmp_path) / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("autotrade.core.autonomous_agent.PROJECT_DIR", tmp_path)
    monkeypatch.setattr("autotrade.core.autonomous_agent.PLANS_DIR", plans_dir)

    agent = _make_agent()
    agent.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    agent.overnight_secondary_cfg = SimpleNamespace(
        enabled=True,
        run_when_research_complete=True,
        historical_revisit_enabled=True,
        historical_revisit_max_runtime_seconds=60,
        historical_revisit_days=3,
        historical_revisit_symbols=10,
    )

    state = {"watchlist": [{"symbol": "AAA", "recommendation": "BUY"}]}
    result = agent._run_secondary_historical_revisit(state=state, cycle_count=3)

    artifact_path = Path(result["artifact_path"])
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert result["success"] is True
    assert result["degraded_mode"] is True
    assert result["reason"] == "no_candidate_symbols"
    assert payload["degraded_mode"] is True
    assert payload["degraded_reason"] == "no_candidate_symbols"


def test_secondary_post_plan_jobs_retries_failure_marker_and_continues(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("autotrade.core.autonomous_agent.PROJECT_DIR", tmp_path)
    agent = _make_agent()
    agent.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    saved_states = []
    agent._save_overnight_state = lambda state: saved_states.append(dict(state))
    agent._append_jsonl = lambda *args, **kwargs: None
    agent.overnight_secondary_cfg = SimpleNamespace(
        enabled=True,
        run_when_research_complete=True,
        vl_benchmark_enabled=False,
        top_pick_research_enabled=True,
        top_pick_research_cycle_interval=1,
        historical_revisit_enabled=True,
        historical_revisit_cycle_interval=1,
        signal_mining_enabled=False,
        strategy_discovery_enabled=False,
        secondary_job_max_attempts=3,
        secondary_job_retry_backoff_seconds=0.0,
    )
    agent._secondary_job_due = lambda **kwargs: True
    calls = {"top_pick": 0, "historical": 0}

    def _fail_top_pick(**kwargs):
        calls["top_pick"] += 1
        raise RuntimeError("research provider down")

    def _historical(**kwargs):
        calls["historical"] += 1
        return {
            "success": True,
            "job": "historical_revisit",
            "artifact_path": str(tmp_path / "reports" / "historical.json"),
        }

    agent._run_secondary_top_pick_research = _fail_top_pick
    agent._run_secondary_historical_revisit = _historical
    state = {"research_complete": True}

    agent._run_secondary_post_plan_jobs(
        state=state,
        cycle_count=7,
        trigger_reason="test",
    )

    assert calls == {"top_pick": 3, "historical": 1}
    secondary = state["secondary_research"]
    failed = secondary["last_result_by_job"]["top_pick_research"]["result"]
    failed_path = Path(failed["failed_artifact_path"])
    assert failed_path.name.startswith("top_pick_deep_research_FAILED_")
    payload = json.loads(failed_path.read_text(encoding="utf-8"))
    assert payload["job"] == "top_pick_research"
    assert payload["cycle"] == 7
    assert "research provider down" in payload["reason"]
    assert (
        secondary["last_result_by_job"]["historical_revisit"]["result"]["success"]
        is True
    )
    assert saved_states
