from types import SimpleNamespace
import json
from datetime import datetime
from datetime import date
import sys

from autotrade.execution import post_market_workflow as post_market_workflow_mod
from autotrade.execution.post_market_workflow import PostMarketWorkflow


def _workflow_stub() -> PostMarketWorkflow:
    workflow = PostMarketWorkflow.__new__(PostMarketWorkflow)
    workflow.current_regime_analysis = None
    workflow.regime_strategy_overrides = {
        "position_size_multiplier": 1.2,
        "max_positions": 25,
        "stop_multiplier": 2.0,
    }
    workflow.strategy_failsafe_snapshot = SimpleNamespace(
        max_positions=20,
        stop_multiplier=2.0,
    )
    workflow.youtube_context = {
        "available": True,
        "regime": "CRISIS",
        "regime_confidence": 0.9,
        "sizing_multiplier": 0.6,
        "avoid_sectors": ["technology"],
        "favor_sectors": ["energy"],
        "smallcap_ok": False,
        "directives": ["NO NEW LONGS"],
    }
    workflow.regime_router_output = {
        "available": True,
        "regime": "TREND",
        "confidence": 0.7,
        "method": "router",
    }
    workflow.resolved_regime_output = {}
    workflow.regime_router = SimpleNamespace(
        resolve_youtube_divergence=lambda **kwargs: {
            "regime": "TREND",
            "confidence": 0.65,
            "method": "quantitative",
            "weights": {"quantitative": 1.0, "youtube": 0.0},
        }
    )
    return workflow


def test_post_market_workflow_class_docstring_is_preserved():
    assert PostMarketWorkflow.__doc__ is not None
    assert "Post-market agentic workflow for position analysis." in PostMarketWorkflow.__doc__


def test_post_market_workflow_get_pm_plan_date_returns_real_date(monkeypatch):
    workflow = PostMarketWorkflow.__new__(PostMarketWorkflow)
    monkeypatch.setattr(
        post_market_workflow_mod,
        "get_pm_plan_date",
        lambda now=None: date(2026, 4, 17),
    )

    assert workflow.get_pm_plan_date(datetime(2026, 4, 16, 18, 30)) == date(2026, 4, 17)


def test_resolve_tomorrow_regime_uses_pm_plan_date_for_youtube_freshness():
    workflow = _workflow_stub()
    workflow.get_pm_plan_date = lambda now=None: date(2026, 4, 17)
    quantitative_regime = {
        "available": True,
        "regime": "NEUTRAL",
        "confidence": 0.82,
        "strategy_overrides": {
            "position_size_multiplier": 1.0,
            "max_positions": 25,
            "stop_multiplier": 2.0,
        },
    }

    workflow.youtube_context["report_date"] = "2026-04-16"
    stale = workflow._resolve_tomorrow_regime(
        quantitative_regime=quantitative_regime,
        regime_router_snapshot=workflow.regime_router_output,
    )
    assert stale["sources_degraded"] is True

    workflow.youtube_context["report_date"] = "2026-04-17"
    fresh = workflow._resolve_tomorrow_regime(
        quantitative_regime=quantitative_regime,
        regime_router_snapshot=workflow.regime_router_output,
    )
    assert fresh["sources_degraded"] is False


def test_resolve_tomorrow_regime_prefers_router_merge_when_not_defensive():
    workflow = _workflow_stub()
    quantitative_regime = {
        "available": True,
        "regime": "TREND",
        "confidence": 0.82,
        "strategy_overrides": {
            "position_size_multiplier": 1.2,
            "max_positions": 25,
            "stop_multiplier": 2.0,
        },
    }

    resolved = workflow._resolve_tomorrow_regime(
        quantitative_regime=quantitative_regime,
        regime_router_snapshot=workflow.regime_router_output,
    )

    assert resolved["regime"] == "TREND"
    assert resolved["allow_new_longs"] is True
    assert resolved["strategy_overrides"]["position_size_multiplier"] == 0.6
    assert resolved["max_positions"] == 50
    assert resolved["divergence_flags"]["rule_applied"].startswith("router_merge")


def test_youtube_bullish_over_quant_neutral():
    workflow = _workflow_stub()
    workflow.youtube_context = {
        "available": True,
        "regime": "RISK-ON",
        "regime_confidence": 0.70,
        "sizing_multiplier": 0.95,
        "vix_level": 19.4,
        "avoid_sectors": ["utilities"],
        "favor_sectors": ["semiconductors"],
        "smallcap_ok": True,
        "directives": ["VIX <20 confirms risk-on"],
    }
    quantitative_regime = {
        "available": True,
        "regime": "NEUTRAL",
        "confidence": 0.42,
        "strategy_overrides": {
            "position_size_multiplier": 1.1,
            "max_positions": 25,
            "stop_multiplier": 2.0,
        },
    }

    resolved = workflow._resolve_tomorrow_regime(
        quantitative_regime=quantitative_regime,
        regime_router_snapshot={"available": True, "regime": "NEUTRAL", "confidence": 0.5, "method": "router"},
    )

    assert resolved["regime"] == "LEAN-BULLISH"
    assert resolved["allow_new_longs"] is True
    assert resolved["sizing_multiplier"] <= 0.8
    assert resolved["strategy_overrides"]["position_size_multiplier"] <= 0.8
    assert resolved["divergence_flags"]["rule_applied"].startswith(
        "youtube_bullish_override"
    )


def test_resolve_tomorrow_regime_keeps_neutral_sizing_and_gap_cap_override():
    workflow = _workflow_stub()
    workflow.current_regime_analysis = SimpleNamespace(
        regime=SimpleNamespace(value="NEUTRAL"),
        confidence=0.76,
        breadth_pct_positive=72.0,
        pattern_detected="broad_advance",
        detection_timestamp=datetime(2026, 4, 17, 10, 0),
        recommended_strategy={"position_size_multiplier": 1.0, "cash_reserve_pct": 20},
    )
    workflow.youtube_context = {
        "available": True,
        "regime": "NEUTRAL",
        "regime_confidence": 0.75,
        "sizing_multiplier": 0.9,
        "avoid_sectors": [],
        "favor_sectors": [],
        "smallcap_ok": True,
        "directives": [],
    }
    workflow.regime_router = SimpleNamespace(
        resolve_youtube_divergence=lambda **kwargs: {
            "regime": "NEUTRAL",
            "confidence": 0.75,
            "method": "aligned_conservative",
            "weights": {"quantitative": 0.5, "youtube": 0.5},
        }
    )
    quantitative_regime = {
        "available": True,
        "regime": "ROTATION",
        "confidence": 0.88,
        "strategy_overrides": {
            "position_size_multiplier": 0.9,
            "max_positions": 20,
            "stop_multiplier": 2.0,
        },
    }

    resolved = workflow._resolve_tomorrow_regime(
        quantitative_regime=quantitative_regime,
        regime_router_snapshot={
            "available": True,
            "regime": "NEUTRAL",
            "confidence": 0.75,
            "method": "router",
        },
    )

    assert resolved["regime"] == "NEUTRAL"
    assert resolved["strategy_overrides"]["position_size_multiplier"] == 1.0
    assert resolved["strategy_overrides"]["gap_hard_cap_pct"] == 13.0


def test_effective_market_regime_prefers_resolved_regime():
    workflow = _workflow_stub()
    workflow.resolved_regime_output = {"regime": "CRISIS"}

    assert workflow._effective_market_regime() == "CRISIS"


def test_filter_plan_signals_drops_inactive_and_missing_assets(monkeypatch):
    workflow = _workflow_stub()
    workflow._asset_eligibility_cache = {}

    class _Client:
        def get_asset(self, symbol):
            if symbol == "ACTIVE":
                return SimpleNamespace(status="active", tradable=True)
            if symbol == "HALTED":
                return SimpleNamespace(status="inactive", tradable=False)
            raise RuntimeError("404 asset not found")

    workflow.client = _Client()
    monkeypatch.setattr(
        post_market_workflow_mod,
        "get_market_now",
        lambda: datetime(2026, 4, 16, 10, 30),
    )

    filtered, dropped = workflow._filter_plan_signals_by_asset_status(
        [{"symbol": "ACTIVE"}, {"symbol": "HALTED"}, {"symbol": "DELISTED"}]
    )

    assert [row["symbol"] for row in filtered] == ["ACTIVE"]
    assert dropped == ["HALTED", "DELISTED"]


def test_filter_plan_signals_accepts_active_enum_style_asset_status():
    workflow = _workflow_stub()
    workflow._asset_eligibility_cache = {}

    class _Client:
        def get_asset(self, symbol):
            return SimpleNamespace(status="AssetStatus.ACTIVE", tradable=True)

    workflow.client = _Client()

    filtered, dropped = workflow._filter_plan_signals_by_asset_status(
        [{"symbol": "ACTIVE"}]
    )

    assert [row["symbol"] for row in filtered] == ["ACTIVE"]
    assert dropped == []


def test_load_overnight_watchlist_context_uses_fresh_ranked_actionable_symbols(
    tmp_path, monkeypatch
):
    workflow = PostMarketWorkflow.__new__(PostMarketWorkflow)
    research_dir = tmp_path / "research"
    research_dir.mkdir()
    state_path = research_dir / "overnight_state.json"
    state_path.write_text(
        json.dumps(
            {
                "date": "2026-04-02",
                "research_complete": True,
                "watchlist": [
                    {
                        "symbol": "watch",
                        "recommendation": "WATCH",
                        "conviction_priority_score": 99.0,
                    },
                    {
                        "symbol": "low",
                        "recommendation": "BUY",
                        "conviction_priority_score": 70.0,
                    },
                    {
                        "symbol": "high",
                        "recommendation": "STRONG BUY",
                        "conviction_priority_score": 90.0,
                    },
                    {
                        "symbol": "gated",
                        "recommendation": "BUY",
                        "conviction_priority_score": 95.0,
                        "validation_gated": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(post_market_workflow_mod, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(
        post_market_workflow_mod,
        "get_pm_plan_date",
        lambda *_args, **_kwargs: datetime(2026, 4, 3, 18, 30).date(),
    )

    context = workflow._load_overnight_watchlist_context(
        open_slots=3,
        log=SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    assert context["used"] is True
    assert context["reason"] == "fresh_overnight_watchlist"
    assert context["symbols_loaded"] == ["HIGH", "LOW"]
    assert context["symbols_considered"] == 2


def test_load_overnight_watchlist_context_rejects_stale_state(
    tmp_path, monkeypatch
):
    workflow = PostMarketWorkflow.__new__(PostMarketWorkflow)
    research_dir = tmp_path / "research"
    research_dir.mkdir()
    state_path = research_dir / "overnight_state.json"
    state_path.write_text(
        json.dumps(
            {
                "date": "2026-03-20",
                "research_complete": True,
                "watchlist": [{"symbol": "OLD", "recommendation": "BUY"}],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(post_market_workflow_mod, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(
        post_market_workflow_mod,
        "get_pm_plan_date",
        lambda *_args, **_kwargs: datetime(2026, 4, 3, 18, 30).date(),
    )

    context = workflow._load_overnight_watchlist_context(
        open_slots=2,
        log=SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    assert context["used"] is False
    assert context["reason"] == "stale_watchlist"


def test_load_overnight_watchlist_context_promotes_deep_research_bridge_symbols(
    tmp_path, monkeypatch
):
    workflow = PostMarketWorkflow.__new__(PostMarketWorkflow)
    research_dir = tmp_path / "research"
    reports_dir = tmp_path / "reports"
    research_dir.mkdir()
    reports_dir.mkdir()

    watchlist = []
    for idx in range(35):
        watchlist.append(
            {
                "symbol": f"BASE{idx:02d}",
                "recommendation": "BUY",
                "conviction_priority_score": 95.0 - idx,
                "ranking_score": 95.0 - idx,
                "confidence": 80.0,
            }
        )
    watchlist.append(
        {
            "symbol": "TPC",
            "recommendation": "STRONG BUY",
            "conviction_priority_score": 55.0,
            "ranking_score": 55.0,
            "confidence": 85.0,
        }
    )

    (research_dir / "overnight_state.json").write_text(
        json.dumps(
            {
                "date": "2026-04-22",
                "research_complete": True,
                "watchlist": watchlist,
            }
        ),
        encoding="utf-8",
    )
    (reports_dir / "top_pick_deep_research_20260422_025839.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "symbol": "TPC",
                        "base_recommendation": "STRONG BUY",
                        "base_score": 55.0,
                        "deep_research": {
                            "recommendation": "BUY",
                            "confidence": 81,
                            "final_score": 75.25,
                            "entry_price": 84.38,
                        },
                        "watchlist_update": {
                            "updated_score": 62.15,
                            "catalyst_note": "earnings transcript",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(post_market_workflow_mod, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(
        post_market_workflow_mod,
        "get_pm_plan_date",
        lambda *_args, **_kwargs: datetime(2026, 4, 22, 18, 30).date(),
    )

    context = workflow._load_overnight_watchlist_context(
        open_slots=5,
        log=SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    assert context["used"] is True
    assert "TPC" in context["symbols_loaded"]
    assert len(context["symbols_loaded"]) >= 25


def test_build_pm_screener_override_disables_bullish_gate_when_longs_allowed():
    workflow = PostMarketWorkflow.__new__(PostMarketWorkflow)
    workflow.resolved_regime_output = {"allow_new_longs": True}

    override = workflow._build_pm_screener_override()

    assert override == {
        "prefer_bullish_regime": False,
        "momentum_roc_min_score": 25.0,
        "rsi_pullback_min_score": 25.0,
    }


def test_build_pm_screener_override_keeps_default_when_longs_blocked():
    workflow = PostMarketWorkflow.__new__(PostMarketWorkflow)
    workflow.resolved_regime_output = {"allow_new_longs": False}

    override = workflow._build_pm_screener_override()

    assert override == {}


def test_rescale_plan_scores_preserves_rank_and_baseline_fields():
    workflow = PostMarketWorkflow.__new__(PostMarketWorkflow)

    rescaled = workflow._rescale_plan_scores(
        [
            {"symbol": "LOW", "score": 30.0},
            {"symbol": "HIGH", "final_score": 50.0},
            {"symbol": "MID", "confidence": 40.0},
        ]
    )

    assert [row["symbol"] for row in rescaled] == ["HIGH", "MID", "LOW"]
    assert [row["baseline_score"] for row in rescaled] == [50.0, 40.0, 30.0]
    assert [row["score"] for row in rescaled] == [90.0, 75.0, 60.0]
    assert [row["final_score"] for row in rescaled] == [90.0, 75.0, 60.0]
    assert [row["confidence"] for row in rescaled] == [90.0, 75.0, 60.0]
    assert [row["normalized_score"] for row in rescaled] == [90.0, 75.0, 60.0]
    assert all(row["score_scale"] == "realtime_rescaled" for row in rescaled)

    singleton = workflow._rescale_plan_scores([{"symbol": "SOLO", "score": 33.0}])
    assert singleton[0]["symbol"] == "SOLO"
    assert singleton[0]["baseline_score"] == 33.0
    assert singleton[0]["score"] == 90.0


def test_run_generates_signals_even_when_portfolio_is_flat(tmp_path, monkeypatch):
    workflow = PostMarketWorkflow.__new__(PostMarketWorkflow)
    workflow.dry_run = True
    workflow.youtube_context = {}
    workflow.current_regime_analysis = None
    workflow.regime_strategy_overrides = {}
    workflow.regime_router_output = {
        "available": False,
        "regime": "NEUTRAL",
        "confidence": 0.0,
        "method": "disabled",
        "error": None,
    }
    workflow.resolved_regime_output = {}
    workflow._last_validated_strategy_count = 0
    workflow._last_merged_candidate_count = 0
    workflow._last_strategy_routing_diagnostics = {}
    workflow._asset_eligibility_cache = {}
    workflow.enable_parallel = False
    workflow.max_workers = 1

    snapshot = SimpleNamespace(
        level="normal",
        halt_new_entries=False,
        max_positions=5,
        stop_multiplier=2.0,
        drawdown_pct=0.0,
        to_dict=lambda: {
            "level": "normal",
            "halt_new_entries": False,
            "max_positions": 5,
            "stop_multiplier": 2.0,
            "drawdown_pct": 0.0,
        },
    )
    workflow.strategy_failsafe_snapshot = snapshot
    workflow.strategy_failsafe = SimpleNamespace(
        update_from_strategy_validation=lambda **_kwargs: snapshot
    )
    workflow.day_tracker = SimpleNamespace(get_remaining=lambda: 3)

    workflow._ensure_youtube_intelligence = lambda: {}
    workflow._refresh_quantitative_regime = lambda: None
    workflow._detect_signal_router_regime = lambda: {
        "available": False,
        "regime": "NEUTRAL",
        "confidence": 0.0,
        "method": "disabled",
        "error": None,
    }
    workflow._quantitative_regime_snapshot = lambda: {
        "available": False,
        "regime": "NEUTRAL",
        "confidence": 0.0,
        "strategy_overrides": {},
    }
    workflow._resolve_tomorrow_regime = lambda **_kwargs: {
        "regime": "NEUTRAL",
        "allow_new_longs": True,
        "smallcap_ok": True,
        "max_positions": 5,
        "position_size_multiplier": 1.0,
        "stop_multiplier": 2.0,
        "strategy_overrides": {},
        "divergence_flags": {},
    }
    workflow._effective_market_regime = lambda: "NEUTRAL"
    workflow._effective_max_positions = lambda: 5
    workflow._regime_override_float = lambda _key, default: default
    workflow._effective_stop_multiplier = lambda fallback=2.0: fallback
    workflow.get_account = lambda: SimpleNamespace(equity=100000.0, buying_power=100000.0)
    workflow.get_positions = lambda: []
    workflow._merge_multi_strategy_signals = (
        lambda candidates, _open_slots, _logger: list(candidates)
    )
    workflow._validate_entry_candidates = (
        lambda candidates, _portfolio_cfg, _logger: (list(candidates), [])
    )
    workflow._filter_plan_signals_by_asset_status = (
        lambda signals, log=None: (list(signals or []), [])
    )
    workflow._update_and_filter_stale_signals = (
        lambda signals, _current_positions, log=None: (list(signals or []), [])
    )
    workflow._generate_summary = lambda positions, exits, adds: {
        "total_positions": len(positions),
        "total_value": 0,
        "exits_planned": len(exits),
        "adds_planned": len(adds),
    }
    workflow._print_plan = lambda _plan: None
    workflow._run_sequential_shadow_eval_batch = lambda: {"ran": False}

    class _StrategyValidator:
        def __init__(self, parent_logger=None):
            self.parent_logger = parent_logger

        def validate_from_strategy_lab(self):
            return None

        def find_optimal_parameters(self, lookback_days=40):
            return None

    class _SignalGen:
        last_init = None

        def __init__(self, **kwargs):
            _SignalGen.last_init = kwargs

        def run(self):
            return [
                {
                    "symbol": "AAA",
                    "ticker": "AAA",
                    "score": 88.0,
                    "final_score": 88.0,
                    "confidence": 88.0,
                }
            ]

        def print_summary(self, _candidates):
            return None

        def save_signals(self, _candidates, source="pm_workflow"):
            return None

    monkeypatch.setattr(post_market_workflow_mod, "StrategyValidator", _StrategyValidator)
    monkeypatch.setattr(
        post_market_workflow_mod,
        "get_core_market_data_readiness",
        lambda: {
            "is_fresh": True,
            "primary_date": "2026-04-02",
            "expected_date": "2026-04-02",
            "blocking_reasons": [],
        },
    )
    monkeypatch.setattr(
        post_market_workflow_mod,
        "get_config",
        lambda: SimpleNamespace(
            portfolio=SimpleNamespace(),
            screener_v2=SimpleNamespace(enabled=True),
            market_data=SimpleNamespace(news_cache_ttl_hours=4),
        ),
    )
    monkeypatch.setattr(
        post_market_workflow_mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="OK", stderr=""),
    )
    monkeypatch.setitem(
        sys.modules,
        "autotrade.utils.news_aggregator",
        SimpleNamespace(NewsAggregator=lambda: SimpleNamespace(refresh_batch=lambda *args, **kwargs: None)),
    )
    monkeypatch.setitem(
        sys.modules,
        "autotrade.utils.data_quality",
        SimpleNamespace(
            validate_data_for_signals=lambda: (
                True,
                SimpleNamespace(
                    latest_date="2026-04-02",
                    expected_date="2026-04-02",
                    quality_score=100.0,
                    ticker_count=100,
                    recommendation="OK",
                    issues=[],
                ),
            ),
            validate_signal_batch=lambda _signals: (
                True,
                SimpleNamespace(stats={"count": 1, "score_min": 88.0, "score_max": 88.0}, issues=[]),
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "autotrade.signals.agentic_signal_generator",
        SimpleNamespace(AgenticSignalGenerator=_SignalGen),
    )

    research_dir = tmp_path / "research"
    research_dir.mkdir()
    (research_dir / "overnight_state.json").write_text(
        json.dumps(
            {
                "date": "2026-04-02",
                "research_complete": True,
                "watchlist": [
                    {
                        "symbol": "AAA",
                        "recommendation": "BUY",
                        "conviction_priority_score": 91.0,
                        "confidence": 85.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(post_market_workflow_mod, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(
        post_market_workflow_mod,
        "get_pm_plan_date",
        lambda *_args, **_kwargs: datetime(2026, 4, 3, 18, 30).date(),
    )

    result = workflow.run()

    assert result["positions"] == []
    assert result["signals"][0]["symbol"] == "AAA"
    assert result["overnight_watchlist_bridge"]["used"] is True
    assert result["signal_pipeline_trace"]["generated_raw"] == 1
    assert result["signal_pipeline_trace"]["final_published"] == 1
    assert result["signal_pipeline_trace"]["asset_dropped_symbols"] == []
    assert result["signals"][0]["priority"] == 1
    assert result["signals"][0]["entry_source"] == "overnight_plan"
    assert result["signals"][0]["source_bucket"] == "watchlist"
    assert result["signals"][0]["plan_score_source"] == "morning_game_plan_20260403.json"
    assert result["pm_screener_override"] == {
        "prefer_bullish_regime": False,
        "momentum_roc_min_score": 25.0,
        "rsi_pullback_min_score": 25.0,
    }
    assert _SignalGen.last_init["screener_config_override"] == {
        "prefer_bullish_regime": False,
        "momentum_roc_min_score": 25.0,
        "rsi_pullback_min_score": 25.0,
    }
    assert _SignalGen.last_init["restrict_tickers"] == ["AAA"]


def test_validate_entry_candidates_preserves_pre_validation_candidates_when_all_rejected(
    monkeypatch,
):
    workflow = _workflow_stub()

    class _Validator:
        def __init__(self, parent_logger=None):
            self.parent_logger = parent_logger

        def validate_and_filter(self, candidates, **_kwargs):
            return [], {"rejected": len(candidates), "heatmap": [{"scan_type": "x"}]}

    monkeypatch.setitem(
        sys.modules,
        "autotrade.backtesting.signal_validator",
        SimpleNamespace(SignalValidator=_Validator),
    )
    monkeypatch.setattr(
        post_market_workflow_mod,
        "get_config",
        lambda: SimpleNamespace(
            signal_validation=SimpleNamespace(
                enabled=True,
                score_weight_signal=0.7,
                score_weight_backtest=0.3,
                min_backtest_score=35,
                min_similar_signals=10,
            )
        ),
    )

    original_candidates = [
        {"symbol": "AAA", "final_score": 81.0},
        {"symbol": "BBB", "final_score": 77.0},
    ]

    kept, heatmap = workflow._validate_entry_candidates(
        original_candidates,
        portfolio_cfg=SimpleNamespace(),
        log=SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
    )

    assert kept == original_candidates
    assert heatmap == [{"scan_type": "x"}]


def test_save_plan_preserves_existing_non_empty_plan_when_new_signals_empty(
    tmp_path, monkeypatch
):
    workflow = PostMarketWorkflow.__new__(PostMarketWorkflow)
    workflow._filter_plan_signals_by_asset_status = (
        lambda signals, log=None: (list(signals or []), [])
    )

    monkeypatch.setattr(post_market_workflow_mod, "PLANS_DIR", tmp_path)
    monkeypatch.setattr(
        post_market_workflow_mod,
        "get_pm_plan_date",
        lambda *_args, **_kwargs: datetime(2026, 3, 24, 18, 30),
    )

    plan_path = tmp_path / "pm_plan_2026-03-24.json"
    existing_plan = {
        "generated_at": "2026-03-24T00:00:00",
        "signals": [{"symbol": "KEEP", "score": 91.0}],
        "entry_candidates": [{"symbol": "KEEP", "score": 91.0}],
        "summary": {},
    }
    plan_path.write_text(json.dumps(existing_plan), encoding="utf-8")

    result = workflow._save_plan(
        {
            "generated_at": "2026-03-24T06:00:00",
            "signals": [],
            "entry_candidates": [],
            "summary": {},
            "positions": [{"symbol": "CAVA"}],
            "core_data_readiness": {
                "is_fresh": True,
                "primary_date": "2026-03-23",
                "expected_date": "2026-03-23",
                "blocking_reasons": [],
            },
        }
    )

    saved = json.loads(plan_path.read_text(encoding="utf-8"))
    assert result == plan_path
    assert len(saved["signals"]) == 1
    assert saved["signals"][0]["symbol"] == "KEEP"


def test_save_plan_promotes_morning_plan_when_pm_plan_is_empty(
    tmp_path, monkeypatch
):
    workflow = PostMarketWorkflow.__new__(PostMarketWorkflow)
    workflow._filter_plan_signals_by_asset_status = (
        lambda signals, log=None: (list(signals or []), [])
    )

    monkeypatch.setattr(post_market_workflow_mod, "PLANS_DIR", tmp_path)
    monkeypatch.setattr(
        post_market_workflow_mod,
        "get_pm_plan_date",
        lambda *_args, **_kwargs: datetime(2026, 3, 24, 18, 30),
    )

    morning_plan_path = tmp_path / "morning_game_plan_20260324.json"
    morning_plan_path.write_text(
        json.dumps({"signals": [{"symbol": "ABC", "score": 72.0}]}),
        encoding="utf-8",
    )

    result = workflow._save_plan(
        {
            "generated_at": "2026-03-24T06:00:00",
            "signals": [],
            "entry_candidates": [],
            "summary": {},
            "positions": [],
            "core_data_readiness": {
                "is_fresh": True,
                "primary_date": "2026-03-23",
                "expected_date": "2026-03-23",
                "blocking_reasons": [],
            },
        }
    )

    saved = json.loads((tmp_path / "pm_plan_2026-03-24.json").read_text(encoding="utf-8"))
    assert result == tmp_path / "pm_plan_2026-03-24.json"
    assert len(saved["signals"]) == 1
    assert saved["signals"][0]["symbol"] == "ABC"
    assert saved["entry_candidates"][0]["symbol"] == "ABC"
    assert saved["buy_signals"][0]["symbol"] == "ABC"
    assert saved["pm_plan_source"] == morning_plan_path.name
    assert saved["pm_plan_repaired_from"] == morning_plan_path.name


def test_save_plan_merge_prefers_higher_baseline_score_for_existing_symbol(
    tmp_path, monkeypatch
):
    workflow = PostMarketWorkflow.__new__(PostMarketWorkflow)
    workflow._filter_plan_signals_by_asset_status = (
        lambda signals, log=None: (list(signals or []), [])
    )

    monkeypatch.setattr(post_market_workflow_mod, "PLANS_DIR", tmp_path)
    monkeypatch.setattr(
        post_market_workflow_mod,
        "get_pm_plan_date",
        lambda *_args, **_kwargs: datetime(2026, 3, 24, 18, 30),
    )

    readiness = {
        "is_fresh": True,
        "primary_date": "2026-03-23",
        "expected_date": "2026-03-23",
        "blocking_reasons": [],
    }
    workflow._save_plan(
        {
            "generated_at": "2026-03-24T06:00:00",
            "signals": [
                {"symbol": "AAA", "score": 40.0},
                {"symbol": "BBB", "score": 30.0},
            ],
            "entry_candidates": [],
            "summary": {},
            "positions": [],
            "core_data_readiness": readiness,
        }
    )

    workflow._save_plan(
        {
            "generated_at": "2026-03-24T07:00:00",
            "signals": [{"symbol": "AAA", "score": 50.0}],
            "entry_candidates": [],
            "summary": {},
            "positions": [],
            "core_data_readiness": readiness,
        }
    )

    saved = json.loads((tmp_path / "pm_plan_2026-03-24.json").read_text(encoding="utf-8"))
    by_symbol = {row["symbol"]: row for row in saved["signals"]}

    assert by_symbol["AAA"]["baseline_score"] == 50.0
    assert by_symbol["AAA"]["score"] == 90.0
    assert by_symbol["BBB"]["baseline_score"] == 30.0
    assert by_symbol["BBB"]["score"] == 60.0


def test_stale_core_data_block_keeps_monitoring_candidates():
    workflow = PostMarketWorkflow.__new__(PostMarketWorkflow)
    plan = {
        "core_data_asof": "2026-04-01",
        "core_data_expected_date": "2026-04-02",
        "blocking_reasons": ["daily_features_stale"],
        "signals": [{"symbol": "AAA", "score": 91.0}],
        "entry_candidates": [{"symbol": "AAA", "score": 91.0}],
        "summary": {},
        "pm_ready_for_execution": False,
    }

    workflow._apply_execution_block(plan)

    assert plan["signals"] == [{"symbol": "AAA", "score": 91.0}]
    assert plan["entry_candidates"] == [{"symbol": "AAA", "score": 91.0}]
    assert plan["summary"]["execution_blocked"] is True
    assert plan["summary"]["monitoring_candidates_preserved"] == 1
