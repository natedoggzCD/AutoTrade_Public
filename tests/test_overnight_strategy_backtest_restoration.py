from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from autotrade.core.autonomous_agent import AutonomousAgent


def _make_agent() -> AutonomousAgent:
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = logging.getLogger("test_overnight_strategy_backtest_restoration")
    return agent


def _backtest_cfg():
    return SimpleNamespace(
        min_trades_for_validity=12,
        gate_win_rate=40.0,
        gate_min_trades=12,
        boost_win_rate=55.0,
        boost_min_trades=20,
    )


def test_aggregate_strategy_backtest_uses_trade_weighted_top_k():
    agent = _make_agent()
    rows = [
        {
            "strategy_name": "alpha",
            "setup_type": "momentum",
            "symbol_metrics": {"trades": 10, "win_rate": 0.60, "profit_factor": 2.0},
            "walk_forward_validated": True,
        },
        {
            "strategy_name": "beta",
            "setup_type": "reversion",
            "symbol_metrics": {"trades": 30, "win_rate": 0.40, "profit_factor": 1.0},
            "walk_forward_validated": False,
        },
        {
            "strategy_name": "gamma",
            "setup_type": "breakout",
            "symbol_metrics": {"trades": 999, "win_rate": 0.99, "profit_factor": 9.0},
            "walk_forward_validated": True,
        },
    ]

    aggregated = agent._aggregate_strategy_backtest(
        symbol="AAPL",
        strategy_rows=rows,
        source="per_symbol_strategy_pool",
        top_k=2,
    )

    assert aggregated is not None
    assert aggregated["source"] == "per_symbol_strategy_pool"
    assert aggregated["top_k_used"] == 2
    assert aggregated["total_trades"] == 40
    assert aggregated["win_rate"] == pytest.approx(45.0)
    assert aggregated["profit_factor"] == pytest.approx(1.25)
    assert aggregated["walk_forward_validated"] is True
    assert aggregated["walk_forward_strategy_count"] == 1
    assert aggregated["strategy_names"] == ["alpha", "beta"]


def test_get_strategy_backtest_for_symbol_uses_global_fallback(monkeypatch):
    agent = _make_agent()
    fallback_rows = [
        {
            "strategy_name": "fallback_a",
            "setup_type": "momentum",
            "symbol_metrics": {"trades": 20, "win_rate": 0.55, "profit_factor": 1.4},
            "walk_forward_validated": True,
        },
        {
            "strategy_name": "fallback_b",
            "setup_type": "breakout",
            "symbol_metrics": {"trades": 10, "win_rate": 0.45, "profit_factor": 1.2},
            "walk_forward_validated": False,
        },
    ]
    monkeypatch.setattr(
        agent, "_load_symbol_strategy_map", lambda: {"*": fallback_rows}
    )
    monkeypatch.setattr(
        "autotrade.signals.strategy_pool.get_top_strategies_for_symbol",
        lambda symbol, symbol_map, fallback_to_global=True: list(symbol_map["*"]),
    )
    monkeypatch.setattr(
        "autotrade.core.autonomous_agent.get_config",
        lambda: SimpleNamespace(strategy_lab=SimpleNamespace(per_symbol_top_k=5)),
    )

    aggregated = agent._get_strategy_backtest_for_symbol("MSFT")

    assert aggregated is not None
    assert aggregated["source"] == "global_strategy_pool_fallback"
    assert aggregated["fallback_used"] is True
    assert aggregated["top_k_used"] == 2
    assert aggregated["total_trades"] == 30
    assert aggregated["win_rate"] == pytest.approx((20 * 0.55 + 10 * 0.45) / 30 * 100.0)


def test_build_overnight_candidate_row_carries_backtest_provenance():
    agent = _make_agent()

    row = agent._build_overnight_candidate_row(
        symbol="MSFT",
        sector="Technology",
        research={
            "recommendation": "BUY",
            "confidence": 81,
            "final_score": 81,
            "risk_reward": 2.1,
            "backtest": {
                "source": "global_strategy_pool_fallback",
                "fallback_used": True,
                "win_rate": 56.0,
                "total_trades": 34,
                "avg_trade": 1.4,
                "profit_factor": 1.6,
                "strategy_count": 5,
                "top_k_used": 5,
                "walk_forward_validated": True,
                "walk_forward_strategy_count": 2,
                "strategy_names": ["alpha", "beta"],
                "setup_types": ["momentum", "breakout"],
            },
        },
    )

    assert row["backtest_source"] == "global_strategy_pool_fallback"
    assert row["backtest_scope"] == "global_fallback"
    assert row["backtest_fallback_used"] is True
    assert row["backtest_per_symbol"] is False
    assert row["backtest_total_trades"] == 34
    assert row["backtest_avg_trade"] == 1.4
    assert row["backtest_strategy_count"] == 5
    assert row["backtest_top_k_used"] == 5
    assert row["backtest_walk_forward_validated"] is True
    assert row["backtest_strategy_names"] == ["alpha", "beta"]


def test_build_overnight_candidate_row_derives_catalyst_fields_from_inputs():
    agent = _make_agent()

    row = agent._build_overnight_candidate_row(
        symbol="ABX",
        sector="Materials",
        research={
            "recommendation": "BUY",
            "confidence": 77,
            "final_score": 77,
            "catalysts": ["Earnings beat and raised guidance"],
            "fresh_news": True,
            "backtest": {"source": "no_strategy_backtest"},
        },
    )

    assert row["catalyst"] == "Earnings beat and raised guidance"
    assert row["catalyst_note"] == "Earnings beat and raised guidance"
    assert row["has_catalyst"] is True
    assert row["fresh_news"] is True
    assert row["catalyst_count"] == 1
    assert row["catalyst_score"] == pytest.approx(0.5)


def test_summarize_catalyst_coverage_counts_rows_with_catalyst_support():
    agent = _make_agent()

    summary = agent._summarize_catalyst_coverage(
        [
            {
                "symbol": "AAA",
                "has_catalyst": True,
                "catalyst_note": "FDA catalyst",
                "fresh_news": True,
                "catalyst_score": 0.8,
            },
            {
                "symbol": "BBB",
                "has_catalyst": False,
                "catalyst_note": "",
                "fresh_news": False,
                "catalyst_score": 0.0,
            },
            {
                "symbol": "CCC",
                "has_catalyst": True,
                "catalyst": "Product launch",
                "fresh_news": False,
                "catalyst_score": 0.4,
            },
        ]
    )

    assert summary["rows_total"] == 3
    assert summary["rows_with_catalyst"] == 2
    assert summary["rows_with_note"] == 2
    assert summary["rows_with_fresh_news"] == 1
    assert summary["avg_catalyst_score"] == pytest.approx(0.6)


def test_build_catalyst_coverage_detail_flags_thin_top_slices():
    agent = _make_agent()

    rows = [
        {"symbol": "AAA", "has_catalyst": True, "fresh_news": True},
        {"symbol": "BBB", "has_catalyst": False, "fresh_news": False},
        {"symbol": "CCC", "has_catalyst": False, "fresh_news": True},
        {"symbol": "DDD", "has_catalyst": False, "fresh_news": False},
    ]

    detail = agent._build_catalyst_coverage_detail(rows)

    assert detail["coverage_health"] == "mixed"
    assert detail["catalyst_ratio"] == pytest.approx(0.25)
    assert detail["fresh_news_ratio"] == pytest.approx(0.5)
    assert detail["top_slice_coverage"]["10"]["catalyst_rows"] == 1
    assert detail["top_slice_coverage"]["10"]["fresh_news_rows"] == 2
    assert detail["top_catalyst_symbols"] == ["AAA"]
    assert detail["top_fresh_news_symbols"] == ["AAA", "CCC"]


def test_catalyst_priority_adjustment_increases_when_session_is_thin():
    agent = _make_agent()
    row = {
        "has_catalyst": True,
        "fresh_news": True,
        "catalyst_score": 0.5,
        "catalyst_count": 1,
    }

    base = agent._compute_catalyst_priority_adjustment(row, {"catalyst_ratio": 0.25})
    thin = agent._compute_catalyst_priority_adjustment(row, {"catalyst_ratio": 0.05})

    assert thin > base
    assert base == pytest.approx(10.0)
    assert thin == pytest.approx(14.0)


def test_summarize_historical_plan_adoption_counts_adjusted_and_decision_overlap():
    agent = _make_agent()
    rows = [
        {"symbol": "AAA", "has_catalyst": True},
        {"symbol": "BBB", "has_catalyst": False},
        {"symbol": "CCC", "has_catalyst": True},
        {"symbol": "DDD", "has_catalyst": False},
    ]

    summary = agent._summarize_historical_plan_adoption(
        rows,
        adjusted_symbols={"AAA", "DDD", "ZZZ"},
        decision_symbols={"CCC"},
        top_n=3,
    )

    assert summary["top_symbols"] == ["AAA", "BBB", "CCC"]
    assert summary["top_catalyst_symbols"] == ["AAA", "CCC"]
    assert summary["adjusted_plan_overlap_count"] == 1
    assert summary["adjusted_plan_overlap_symbols"] == ["AAA"]
    assert summary["decision_overlap_count"] == 1
    assert summary["decision_overlap_symbols"] == ["CCC"]
    assert summary["top_catalyst_adjusted_overlap_symbols"] == ["AAA"]
    assert summary["top_catalyst_decision_overlap_symbols"] == ["CCC"]


def test_summarize_historical_catalyst_outcomes_separates_adopted_vs_missed():
    agent = _make_agent()
    summary = agent._summarize_historical_catalyst_outcomes(
        top_catalyst_symbols=["AAA", "BBB", "CCC"],
        adjusted_overlap_symbols=["AAA"],
        decision_overlap_symbols=["CCC"],
        session_returns={"AAA": 2.0, "BBB": -1.0, "CCC": 3.0},
    )

    assert summary["symbols_with_return_data"] == 3
    assert summary["avg_day_return_pct"] == pytest.approx(1.33)
    assert summary["positive_count"] == 2
    assert summary["negative_count"] == 1
    assert summary["adjusted_overlap_avg_return_pct"] == pytest.approx(2.0)
    assert summary["missed_avg_return_pct"] == pytest.approx(1.0)
    assert summary["decision_overlap_avg_return_pct"] == pytest.approx(3.0)


def test_merge_backtest_provenance_into_row_normalizes_catalyst_fields(monkeypatch):
    agent = _make_agent()
    monkeypatch.setattr(agent, "_get_strategy_backtest_for_symbol", lambda symbol: {})

    row = agent._merge_backtest_provenance_into_row(
        {
            "symbol": "FGI",
            "news": [{"title": "FGI wins new contract"}],
            "fresh_news": True,
        }
    )

    assert row["catalyst"] == "FGI wins new contract"
    assert row["catalyst_note"] == "FGI wins new contract"
    assert row["has_catalyst"] is True
    assert row["catalyst_count"] == 1
    assert row["fresh_news"] is True
    assert row["catalyst_score"] == pytest.approx(0.5)
    assert row["catalyst_priority_adjustment"] == pytest.approx(10.0)


def test_summarize_backtest_provenance_counts_scopes_and_fallbacks():
    agent = _make_agent()

    summary = agent._summarize_backtest_provenance(
        [
            {
                "symbol": "AAA",
                "backtest_scope": "per_symbol",
                "backtest_fallback_used": False,
                "backtest_walk_forward_validated": True,
                "backtest_total_trades": 40,
                "backtest_win_rate": 62.0,
            },
            {
                "symbol": "BBB",
                "backtest_scope": "global_fallback",
                "backtest_fallback_used": True,
                "backtest_walk_forward_validated": False,
                "backtest_total_trades": 20,
                "backtest_win_rate": 54.0,
            },
            {
                "symbol": "CCC",
                "backtest_scope": "no_strategy",
                "backtest_fallback_used": False,
                "backtest_walk_forward_validated": False,
                "backtest_total_trades": 0,
                "backtest_win_rate": 0.0,
            },
        ]
    )

    assert summary["rows_total"] == 3
    assert summary["per_symbol_rows"] == 1
    assert summary["global_fallback_rows"] == 1
    assert summary["no_strategy_rows"] == 1
    assert summary["fallback_rows"] == 1
    assert summary["walk_forward_validated_rows"] == 1
    assert summary["avg_backtest_trades"] == pytest.approx(30.0)
    assert summary["avg_backtest_win_rate"] == pytest.approx(58.0)


def test_build_backtest_provenance_detail_flags_fallback_heavy_top_slices():
    agent = _make_agent()

    rows = [
        {
            "symbol": "AAA",
            "backtest_scope": "per_symbol",
            "backtest_fallback_used": False,
            "backtest_walk_forward_validated": False,
            "backtest_provenance_label": "per_symbol",
        }
    ] + [
        {
            "symbol": f"F{i}",
            "backtest_scope": "global_fallback",
            "backtest_fallback_used": True,
            "backtest_walk_forward_validated": False,
            "backtest_provenance_label": "global_fallback",
        }
        for i in range(1, 10)
    ]

    detail = agent._build_backtest_provenance_detail(rows)

    assert detail["coverage_health"] == "fallback_heavy"
    assert detail["per_symbol_ratio"] == pytest.approx(0.1)
    assert detail["fallback_ratio"] == pytest.approx(0.9)
    assert detail["top_slice_coverage"]["10"]["per_symbol_rows"] == 1
    assert detail["top_slice_coverage"]["10"]["fallback_rows"] == 9
    assert detail["top_per_symbol_symbols"] == ["AAA"]
    assert detail["top_fallback_symbols"][:3] == ["F1", "F2", "F3"]


def test_backtest_provenance_adjustment_prefers_per_symbol_over_fallback():
    agent = _make_agent()

    per_symbol = {
        "backtest_scope": "per_symbol",
        "backtest_fallback_used": False,
        "backtest_walk_forward_validated": True,
        "backtest_strategy_count": 4,
        "backtest_total_trades": 30,
    }
    fallback = {
        "backtest_scope": "global_fallback",
        "backtest_fallback_used": True,
        "backtest_walk_forward_validated": False,
        "backtest_strategy_count": 4,
        "backtest_total_trades": 30,
    }
    no_strategy = {
        "backtest_scope": "no_strategy",
        "backtest_fallback_used": False,
        "backtest_walk_forward_validated": False,
        "backtest_total_trades": 0,
    }

    assert agent._compute_backtest_provenance_adjustment(per_symbol) == 7.0
    assert agent._compute_backtest_provenance_adjustment(fallback) == 0.0
    assert agent._compute_backtest_provenance_adjustment(no_strategy) == 0.0
    assert (
        agent._compute_backtest_provenance_adjustment(per_symbol)
        > agent._compute_backtest_provenance_adjustment(fallback)
    )


def test_fallback_repeat_penalty_only_hits_repeated_unfresh_fallbacks():
    agent = _make_agent()

    stale_fallback = {
        "symbol": "FGI",
        "backtest_scope": "global_fallback",
        "discovery_family_count": 1,
        "catalyst_score": 0.0,
        "catalyst_monitor_score_delta": 0.0,
        "fresh_news": False,
        "catalyst_count": 0,
    }
    fresh_fallback = {
        "symbol": "FGI",
        "backtest_scope": "global_fallback",
        "discovery_family_count": 2,
        "catalyst_score": 0.4,
        "catalyst_monitor_score_delta": 1.0,
        "fresh_news": True,
        "catalyst_count": 2,
    }
    per_symbol = {
        "symbol": "ABX",
        "backtest_scope": "per_symbol",
    }
    recent_counts = {"FGI": 3, "ABX": 3}

    stale_result = agent._evaluate_fallback_repeat_penalty(stale_fallback, recent_counts)
    fresh_result = agent._evaluate_fallback_repeat_penalty(fresh_fallback, recent_counts)
    per_symbol_result = agent._evaluate_fallback_repeat_penalty(per_symbol, recent_counts)

    assert stale_result["count"] == 3
    assert stale_result["penalty"] == 3.5
    assert stale_result["exempted"] is False
    assert fresh_result["penalty"] == 0.0
    assert fresh_result["exempted"] is True
    assert per_symbol_result["penalty"] == 0.0


def test_ranking_score_prefers_per_symbol_backtest_provenance():
    agent = _make_agent()

    common = {
        "confidence": 82,
        "backtest_win_rate": 56.0,
        "profit_factor": 1.5,
        "weekly_return": 3.0,
        "rsi_14": 42.0,
        "volume_ratio": 1.4,
        "vol_trend_ratio": 1.1,
        "vol_increasing_days": 2,
        "volume_confirmed": True,
        "sentiment_score": 66.0,
        "stocktwits": {"score": 0.7, "trending": True, "bull_bear_ratio": 1.7},
        "recommendation": "BUY",
        "backtest_strategy_count": 4,
        "backtest_total_trades": 30,
    }
    per_symbol = dict(
        common,
        backtest_scope="per_symbol",
        backtest_walk_forward_validated=True,
        backtest_fallback_used=False,
    )
    fallback = dict(
        common,
        backtest_scope="global_fallback",
        backtest_walk_forward_validated=False,
        backtest_fallback_used=True,
    )

    assert agent._compute_ranking_score(per_symbol) > agent._compute_ranking_score(
        fallback
    )


def test_merge_backtest_provenance_into_row_applies_adjustment(monkeypatch):
    agent = _make_agent()
    monkeypatch.setattr(
        agent,
        "_get_strategy_backtest_for_symbol",
        lambda symbol: {
            "source": "per_symbol_strategy_pool",
            "fallback_used": False,
            "win_rate": 58.0,
            "total_trades": 24,
            "avg_trade": 1.6,
            "profit_factor": 1.8,
            "strategy_count": 4,
            "top_k_used": 4,
            "walk_forward_validated": True,
            "walk_forward_strategy_count": 2,
            "strategy_names": ["alpha", "beta"],
            "setup_types": ["momentum"],
        },
    )

    row = agent._merge_backtest_provenance_into_row(
        {
            "symbol": "MSFT",
            "ranking_score": 72.0,
            "conviction_priority_score": 75.0,
        }
    )

    assert row["backtest_scope"] == "per_symbol"
    assert row["backtest_provenance_label"] == "per_symbol_walk_forward"
    assert row["backtest_provenance_adjustment"] == 7.0
    assert row["ranking_score"] == 79.0
    assert row["conviction_priority_score"] == 82.0


def test_rebuild_historical_overnight_state_from_morning_plan(tmp_path, monkeypatch):
    agent = _make_agent()
    plans_dir = tmp_path / "plans"
    research_dir = tmp_path / "research"
    plans_dir.mkdir()
    research_dir.mkdir()

    monkeypatch.setattr("autotrade.core.autonomous_agent.PLANS_DIR", plans_dir)
    monkeypatch.setattr("autotrade.core.autonomous_agent.PROJECT_DIR", tmp_path)
    monkeypatch.setattr(
        agent,
        "_get_strategy_pool_snapshot",
        lambda: {"as_of": "2026-04-01", "rows": 3},
    )
    monkeypatch.setattr(
        agent,
        "_get_strategy_backtest_for_symbol",
        lambda symbol: {
            "source": "per_symbol_strategy_pool"
            if symbol == "AAA"
            else "global_strategy_pool_fallback",
            "fallback_used": symbol != "AAA",
            "win_rate": 61.0 if symbol == "AAA" else 52.0,
            "total_trades": 22,
            "avg_trade": 1.3,
            "profit_factor": 1.7 if symbol == "AAA" else 1.2,
            "strategy_count": 4,
            "top_k_used": 4,
            "walk_forward_validated": symbol == "AAA",
            "walk_forward_strategy_count": 1 if symbol == "AAA" else 0,
            "strategy_names": ["alpha"],
            "setup_types": ["momentum"],
        },
    )
    monkeypatch.setattr(
        "autotrade.utils.market_intelligence.load_market_intelligence",
        lambda date_str, fallback_days=0: {
            "date": date_str,
            "market_regime": "CAUTIOUS-RISK-ON",
            "regime_confidence": 70,
            "overnight_directives": ["Be selective"],
            "trading_signals": {
                "sizing_multiplier": 0.8,
                "sector_bias": [{"sector": "Energy", "bias": "overweight"}],
            },
        },
    )
    monkeypatch.setattr(
        agent,
        "_load_recent_plan_symbol_frequency",
        lambda exclude_target_date_compact=None, lookback_plans=5, top_n=20: {
            "BBB": 3
        },
    )

    (plans_dir / "morning_game_plan_20260402.json").write_text(
        json.dumps(
            {
                "full_watchlist": [
                    {
                        "symbol": "BBB",
                        "ranking_score": 80.0,
                        "conviction_priority_score": 81.0,
                        "confidence": 80,
                    },
                    {
                        "symbol": "AAA",
                        "ranking_score": 78.0,
                        "conviction_priority_score": 79.0,
                        "confidence": 79,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = agent.rebuild_historical_overnight_state_from_morning_plan(
        target_trade_date="2026-04-02",
        youtube_session_date="2026-04-01",
    )

    output_path = Path(result["output_path"])
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["target_trade_date"] == "2026-04-02"
    assert payload["date"] == "2026-04-01"
    assert payload["backtest_provenance_summary"]["rows_total"] == 2
    assert payload["backtest_provenance_summary"]["per_symbol_rows"] == 1
    assert payload["backtest_provenance_summary"]["global_fallback_rows"] == 1
    assert payload["backtest_provenance_detail"]["coverage_health"] == "strong"
    assert payload["backtest_provenance_detail"]["top_per_symbol_symbols"] == ["AAA"]
    assert payload["backtest_provenance_detail"]["top_fallback_symbols"] == ["BBB"]
    assert payload["watchlist"][0]["symbol"] == "AAA"
    assert payload["watchlist"][0]["backtest_provenance_label"] == "per_symbol_walk_forward"
    assert payload["watchlist"][1]["backtest_provenance_label"] == "global_fallback"
    assert payload["watchlist"][1]["fallback_repeat_penalty"] == 3.5


def test_analyze_historical_fallback_overlap_reports_shared_symbols(tmp_path, monkeypatch):
    agent = _make_agent()

    payloads = {
        "2026-04-02": {
            "watchlist": [
                {"symbol": "AAA", "backtest_scope": "global_fallback"},
                {"symbol": "BBB", "backtest_scope": "global_fallback"},
                {"symbol": "CCC", "backtest_scope": "per_symbol"},
            ],
            "date": "2026-04-01",
            "backtest_provenance_detail": {"coverage_health": "fallback_heavy"},
        },
        "2026-04-01": {
            "watchlist": [
                {"symbol": "BBB", "backtest_scope": "global_fallback"},
                {"symbol": "DDD", "backtest_scope": "global_fallback"},
            ],
            "date": "2026-03-31",
            "backtest_provenance_detail": {"coverage_health": "fallback_heavy"},
        },
    }
    for trade_date, payload in payloads.items():
        output_path = tmp_path / f"{trade_date}.json"
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        payload["output_path"] = str(output_path)

    def _fake_rebuild(target_trade_date, youtube_session_date=None, output_path=None):
        trade_date = str(target_trade_date)
        payload = payloads[trade_date]
        return {"output_path": payload["output_path"]}

    monkeypatch.setattr(
        agent, "rebuild_historical_overnight_state_from_morning_plan", _fake_rebuild
    )

    result = agent.analyze_historical_fallback_overlap(
        ["2026-04-02", "2026-04-01"], top_n=3
    )

    assert result["sessions"][0]["top_fallback_symbols"] == ["AAA", "BBB"]
    assert result["sessions"][1]["top_fallback_symbols"] == ["BBB", "DDD"]
    assert result["pairwise_overlap"][0]["shared_symbols"] == ["BBB"]
    assert result["repeated_fallback_symbols"][0]["symbol"] == "BBB"


def test_recommendation_does_not_hard_gate_signal_validator_backtest(monkeypatch):
    agent = _make_agent()
    monkeypatch.setattr(
        "autotrade.core.autonomous_agent.get_config",
        lambda: SimpleNamespace(
            backtest=SimpleNamespace(strategy_backtester=_backtest_cfg())
        ),
    )
    research = {
        "symbol": "AKAM",
        "final_score": 72,
        "direction": "BULLISH",
        "sentiment_score": 50,
        "technical_score": 50,
        "news": [],
        "backtest": {
            "source": "signal_validator",
            "win_rate": 23.0,
            "total_trades": 61,
        },
    }

    updated = agent._calculate_recommendation_multimodel(research)

    assert updated["recommendation"] == "BUY"
    assert updated.get("backtest_gated") is not True


def test_recommendation_hard_gates_real_strategy_pool_backtest(monkeypatch):
    agent = _make_agent()
    monkeypatch.setattr(
        "autotrade.core.autonomous_agent.get_config",
        lambda: SimpleNamespace(
            backtest=SimpleNamespace(strategy_backtester=_backtest_cfg())
        ),
    )
    research = {
        "symbol": "AKAM",
        "final_score": 72,
        "direction": "BULLISH",
        "sentiment_score": 50,
        "technical_score": 50,
        "news": [],
        "backtest": {
            "source": "per_symbol_strategy_pool",
            "win_rate": 23.0,
            "total_trades": 61,
            "profit_factor": 0.75,
        },
    }

    updated = agent._calculate_recommendation_multimodel(research)

    assert updated["recommendation"] == "HOLD"
    assert updated["backtest_gated"] is True
    assert updated["validation_gated"] is False


def test_recommendation_soft_downgrades_weak_strategy_pool_backtest(monkeypatch):
    agent = _make_agent()
    monkeypatch.setattr(
        "autotrade.core.autonomous_agent.get_config",
        lambda: SimpleNamespace(
            backtest=SimpleNamespace(strategy_backtester=_backtest_cfg())
        ),
    )
    research = {
        "symbol": "AKAM",
        "final_score": 72,
        "direction": "BULLISH",
        "sentiment_score": 50,
        "technical_score": 50,
        "news": [],
        "backtest": {
            "source": "per_symbol_strategy_pool",
            "win_rate": 31.0,
            "total_trades": 61,
            "profit_factor": 1.15,
            "walk_forward_strategy_count": 1,
        },
    }

    updated = agent._calculate_recommendation_multimodel(research)

    assert updated["recommendation"] == "WEAK BUY"
    assert updated.get("backtest_gated") is not True
    assert updated["confidence"] < 72


def test_recommendation_soft_penalizes_signal_validation_without_hold(monkeypatch):
    agent = _make_agent()
    monkeypatch.setattr(
        "autotrade.core.autonomous_agent.get_config",
        lambda: SimpleNamespace(
            backtest=SimpleNamespace(strategy_backtester=_backtest_cfg())
        ),
    )
    research = {
        "symbol": "AKAM",
        "final_score": 72,
        "direction": "BULLISH",
        "sentiment_score": 50,
        "technical_score": 50,
        "news": [],
        "signal_validation": {
            "historical_win_rate": 0.31,
            "similar_signals_found": 61,
        },
    }

    updated = agent._calculate_recommendation_multimodel(research)

    assert updated["recommendation"] == "WEAK BUY"
    assert updated["validation_gated"] is False
    assert updated["validation_penalized"] is True
    assert updated["confidence"] < 72


def test_recommendation_severe_signal_validation_stays_actionable(monkeypatch):
    agent = _make_agent()
    monkeypatch.setattr(
        "autotrade.core.autonomous_agent.get_config",
        lambda: SimpleNamespace(
            backtest=SimpleNamespace(strategy_backtester=_backtest_cfg())
        ),
    )
    research = {
        "symbol": "AVPT",
        "final_score": 80,
        "direction": "BULLISH",
        "sentiment_score": 50,
        "technical_score": 50,
        "news": [],
        "signal_validation": {
            "historical_win_rate": 0.12,
            "similar_signals_found": 49,
        },
    }

    updated = agent._calculate_recommendation_multimodel(research)

    assert updated["recommendation"] == "WEAK BUY"
    assert updated["validation_gated"] is False
    assert updated["validation_penalized"] is True
    assert updated["confidence"] >= 65
