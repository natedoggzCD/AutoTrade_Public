import json
from pathlib import Path
from types import SimpleNamespace

from autotrade.core.autonomous_agent import AutonomousAgent
from autotrade.core.signal_mining import DiscoveryRegistry, SignalMiningEngine


def test_signal_mining_engine_identifies_patterns_and_updates_registry():
    engine = SignalMiningEngine(
        gap_threshold_pct=5.0,
        breadth_threshold_pct=65.0,
        volatility_atr_threshold=6.0,
        volatility_range_threshold=8.0,
    )
    registry = DiscoveryRegistry()
    universe_rows = [
        {
            "symbol": "AAA",
            "sector": "Technology",
            "gap_pct": 7.2,
            "atr_percent": 3.1,
            "sector_breadth_pct": 72.0,
        },
        {
            "symbol": "BBB",
            "sector": "Energy",
            "gap_pct": 1.0,
            "atr_percent": 8.4,
            "intraday_range": 9.1,
        },
        {
            "symbol": "QQQ",
            "gap_pct": 9.0,
            "atr_percent": 10.0,
        },
    ]

    discoveries = engine.mine(universe_rows)
    registry.record_many(discoveries, detected_at="2026-03-11T22:15:00")
    state = registry.to_state()

    discovery_keys = {(row["symbol"], row["family"]) for row in discoveries}
    assert ("AAA", "gap") in discovery_keys
    assert ("AAA", "breadth") in discovery_keys
    assert ("BBB", "volatility") in discovery_keys
    assert all(symbol != "QQQ" for symbol, _ in discovery_keys)
    assert "AAA:gap" in state["items"]
    assert state["items"]["AAA:gap"]["validation_status"] == "candidate"
    assert state["by_symbol"]["AAA"] == ["breadth", "gap"]


def test_secondary_signal_mining_updates_state_registry(tmp_path, monkeypatch):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    agent.signal_mining_engine = SignalMiningEngine()
    agent.overnight_secondary_cfg = SimpleNamespace(
        enabled=True,
        run_when_research_complete=True,
        signal_mining_enabled=True,
        signal_mining_cycle_interval=1,
        signal_mining_universe_symbols=1600,
        signal_mining_max_new_registry_entries=10,
        signal_mining_gap_threshold_pct=5.0,
        signal_mining_breadth_threshold_pct=65.0,
        signal_mining_volatility_atr_threshold=6.0,
        signal_mining_volatility_range_threshold=8.0,
    )
    monkeypatch.setattr("autotrade.core.autonomous_agent.PROJECT_DIR", tmp_path)
    state = {
        "research_complete": True,
        "secondary_research": {},
        "mining_universe": [
            {
                "symbol": "AAA",
                "sector": "Technology",
                "gap_pct": 6.8,
                "sector_breadth_pct": 70.0,
            },
            {
                "symbol": "BBB",
                "sector": "Energy",
                "atr_percent": 7.1,
                "intraday_range": 8.8,
            },
        ],
    }

    result = agent._run_secondary_signal_mining(state=state, cycle_count=5)

    assert result["success"] is True
    assert result["discovery_count"] >= 2
    registry = state["discovery_registry"]
    assert "AAA:gap" in registry["items"]
    assert "BBB:volatility" in registry["items"]
    assert result["artifact_path"].endswith(".json")


def test_secondary_post_plan_jobs_runs_signal_mining_and_records_history():
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    agent._save_overnight_state = lambda state: None
    agent._append_jsonl = lambda *args, **kwargs: None
    agent.overnight_secondary_cfg = SimpleNamespace(
        enabled=True,
        run_when_research_complete=True,
        vl_benchmark_enabled=False,
        top_pick_research_enabled=False,
        historical_revisit_enabled=False,
        signal_mining_enabled=True,
        signal_mining_cycle_interval=1,
        strategy_discovery_enabled=False,
    )
    agent._secondary_job_due = lambda **kwargs: True
    agent._run_secondary_signal_mining = lambda **kwargs: {
        "success": True,
        "job": "signal_mining",
        "registry_updates": 3,
    }
    state = {"research_complete": True, "secondary_research": {}}

    agent._run_secondary_post_plan_jobs(
        state=state,
        cycle_count=9,
        trigger_reason="idle_slot",
    )

    history = state["secondary_research"]["history"]
    assert len(history) == 1
    assert history[0]["job"] == "signal_mining"
    assert history[0]["result"]["registry_updates"] == 3


def test_secondary_signal_mining_writes_degraded_artifact_on_engine_error(
    tmp_path, monkeypatch
):
    class _ExplodingEngine:
        gap_threshold_pct = 5.0
        breadth_threshold_pct = 65.0
        volatility_atr_threshold = 6.0
        volatility_range_threshold = 8.0

        @staticmethod
        def mine(_rows):
            raise RuntimeError("minute_bar_join_failed")

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    agent.signal_mining_engine = _ExplodingEngine()
    agent.overnight_secondary_cfg = SimpleNamespace(
        enabled=True,
        run_when_research_complete=True,
        signal_mining_enabled=True,
        signal_mining_cycle_interval=1,
        signal_mining_universe_symbols=1600,
        signal_mining_max_new_registry_entries=10,
        signal_mining_gap_threshold_pct=5.0,
        signal_mining_breadth_threshold_pct=65.0,
        signal_mining_volatility_atr_threshold=6.0,
        signal_mining_volatility_range_threshold=8.0,
    )
    monkeypatch.setattr("autotrade.core.autonomous_agent.PROJECT_DIR", tmp_path)
    state = {
        "research_complete": True,
        "secondary_research": {},
        "mining_universe": [
            {"symbol": "AAA", "sector": "Technology", "gap_pct": 6.8},
        ],
    }

    result = agent._run_secondary_signal_mining(state=state, cycle_count=7)

    artifact_path = result["artifact_path"]
    payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    assert result["success"] is True
    assert result["degraded_mode"] is True
    assert payload["degraded_mode"] is True
    assert "engine_error:minute_bar_join_failed" in payload["degraded_reason"]
