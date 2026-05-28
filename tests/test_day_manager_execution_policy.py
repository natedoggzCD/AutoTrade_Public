import sys
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("alpaca.trading.client")

from autotrade.core import day_manager as day_manager_mod
from autotrade.core.day_manager import DayManager, TradingPhase
from autotrade.core.loss_floor_exit import LossFloorChecker


class _SessionStateStub(dict):
    def update_many(self, values):
        self.update(values)


def _new_dm_stub() -> DayManager:
    dm = DayManager.__new__(DayManager)
    dm.entry_quality_cfg = SimpleNamespace(
        vwap_universe_min_price=2.0,
        vwap_universe_max_price=200.0,
        wave_entry_enabled=False,
        block_duplicate_open_orders=True,
        momentum_gate_enabled=False,
        momentum_insufficient_data_penalty_points=5.0,
        momentum_hard_block=False,
        core_trading_adaptive_min_score=30.0,
        core_trading_adaptive_top_n=12,
        replacement_score_gap=15.0,
        replacement_weak_score_gap=10.0,
        replacement_weak_position_score_ceiling=5.0,
        replacement_min_candidate_score=45.0,
        replacement_prefilter_enabled=True,
        replacement_prefilter_intraday_move_pct=12.0,
        watchlist_prune_tolerance_min_score=35.0,
        watchlist_prune_orb_tolerance_pct=1.0,
        watchlist_prune_vwap_tolerance_pct=1.0,
        no_fill_watchdog_cycles=2,
        no_fill_escalation_cycles=4,
        no_fill_recheck_top_n=5,
        no_fill_override_top_n=3,
        no_fill_override_max_entries=2,
        replacement_min_hold_minutes=120,
        strength_reentry_enabled=True,
        strength_reentry_active_regimes=["CHOP", "SELLOFF", "CRISIS", "RISK_OFF"],
        strength_reentry_scan_interval_cycles=3,
        strength_reentry_first_hour_minutes=60,
        strength_reentry_min_open_to_hour_gain_pct=1.0,
        strength_reentry_min_relative_strength_pct=1.25,
        strength_reentry_held_min_open_to_hour_gain_pct=-1.0,
        strength_reentry_held_min_current_relative_strength_pct=2.0,
        strength_reentry_held_min_day_return_pct=0.25,
        strength_reentry_held_near_reclaim_tolerance_pct=0.2,
        strength_reentry_follow_through_minutes=0,
        strength_reentry_follow_through_hold_above_reclaim_pct=0.0,
        strength_reentry_learning_block_override_enabled=True,
        strength_reentry_max_pullback_pct=3.25,
        strength_reentry_min_reclaim_volume_ratio=1.15,
        claw_long_exception_require_above_vwap=True,
        claw_long_exception_min_volume_ratio=1.15,
        claw_long_exception_min_relative_strength_pct=1.25,
        strength_reentry_new_entry_size_multiplier=0.55,
        strength_reentry_add_size_multiplier=0.5,
        strength_reentry_max_new_entries_per_day=3,
        strength_reentry_max_adds_per_day=3,
        bearish_session_min_deployed_pct=0.5,
        bearish_session_floor_counts_inverse_exposure=True,
        candidate_backtest_validation_enabled=True,
        candidate_backtest_top_n=12,
        candidate_backtest_refresh_minutes=30,
        candidate_backtest_boost_cap=10.0,
        overnight_watchlist_backtest_reject_enabled=False,
        overnight_watchlist_backtest_penalty_cap=8.0,
        conviction_sizing_enabled=False,
        conviction_size_tiers=[],
    )
    dm.signal_generation_cfg = SimpleNamespace(
        enable_regime_router=True,
        inverse_etf_hedging=SimpleNamespace(symbols=["SH", "PSQ", "RWM", "DOG", "SDS"]),
    )
    dm.signal_validation_cfg = SimpleNamespace(
        enabled=True,
        min_backtest_score=40.0,
        min_similar_signals=10,
        score_weight_signal=0.6,
        score_weight_backtest=0.4,
    )
    dm.execution_entry_cfg = SimpleNamespace(
        enabled=True,
        high_urgency_min_score=72.0,
        critical_urgency_min_score=84.0,
        replace_schedule_seconds=[20, 45, 90],
        max_replacements=3,
        min_reprice_gap_pct=0.2,
        stale_after_seconds=300,
        escalate_to_marketable_min_score=85.0,
        escalate_to_marketable_min_dollar_vol=2_500_000.0,
        normal=SimpleNamespace(max_chase_bps=8, max_slippage_bps=12),
        high=SimpleNamespace(max_chase_bps=18, max_slippage_bps=20),
        critical=SimpleNamespace(max_chase_bps=30, max_slippage_bps=30),
    )
    dm.advisor_control_cfg = SimpleNamespace(
        enabled=True,
        min_recheck_seconds=300,
        force_recheck_seconds=900,
        price_move_bps_trigger=35,
        pnl_change_pct_trigger=0.5,
        hold_minutes_trigger=30,
    )
    dm._advisor_eval_cache = {}
    dm._execution_is_history = []
    dm._execution_circuit_breaker_window = 5
    dm._execution_circuit_breaker_threshold_bps = 10.0
    dm._safe_int = DayManager._safe_int
    dm._safe_float = DayManager._safe_float
    dm._now_utc = DayManager._now_utc
    dm._normalize_entry_time = lambda value: (
        value if isinstance(value, datetime) else None
    )
    dm.position_entries = {}
    dm._market_cap_cache = {}
    dm._feature_cache = {}
    dm.current_regime = "NEUTRAL"
    dm.regime_router_context = {"regime": "neutral", "confidence": 0.0}
    dm.regime_strategy_overrides = {}
    dm._effective_market_regime = lambda: "NEUTRAL"
    dm.youtube_context = {}
    dm._load_resolved_regime_context = lambda: {}
    dm.data_client = None
    dm.premarket_data = {}
    dm.research_position_size_multiplier = 1.0
    dm.runtime_risk_gate = None
    dm.r_unit_sizer = None
    dm.last_account_equity = 0.0
    dm.dry_run = False
    dm.signal_status = {}
    dm._hard_stop_blocked_symbols_today = set()
    dm._add_block_after_trim = True
    dm._min_hold_minutes_for_trim = 120
    dm._add_blocked_symbols_today = set()
    dm._watched_universe_tickers = set()
    dm._session_open_cache = {}
    dm._no_fill_watchdog_state = {
        "streak": 0,
        "last_updated": None,
        "last_reason": "",
        "last_intervention_cycle": 0,
        "intervention_requested": False,
        "intervention_symbols": [],
        "top_ranked_unfilled": [],
    }
    dm._watchlist_causality = {}
    dm._candidate_validation_rejections = []
    dm._strength_reentry_state = {}
    dm._strength_reentry_entries_today = 0
    dm._strength_reentry_adds_today = 0
    dm._vl_recheck_schedule = {}
    dm._vl_recheck_reasons = {}
    dm._recent_symbol_quarantine_days = 14
    dm._recent_symbol_quarantine_losses = 2
    dm.momentum_scanner_cfg = SimpleNamespace(
        enabled=True,
        reserved_slot_cap=10,
        dynamic_reserved_slots_enabled=True,
        max_reserved_slot_cap=15,
        empty_artifact_fallback_minutes=15,
        artifact_path="data/momentum_watchlist_live.json",
    )
    dm.config = SimpleNamespace(
        momentum_scanner=dm.momentum_scanner_cfg,
        portfolio=SimpleNamespace(
            target_position_size=1000.0,
            strategy_reserve_slots=10,
        ),
    )
    dm._momentum_watchlist_state = {}
    dm.premarket_handoff = {}
    dm._last_entry_rejection_reason = ""
    dm._intraday_reserved_symbols = set()
    dm._intraday_reserved_entries_today = 0
    dm._intraday_reserved_last_scan_date = None
    dm._intraday_reserved_scan_enabled = False
    dm._position_slot_class_by_symbol = {}
    dm.signals = []
    dm.exit_manager = SimpleNamespace(check_exits=lambda positions: [])
    dm.runtime_risk_gate = None
    dm.client = None
    dm.strategy = SimpleNamespace(passes_filter=lambda *args, **kwargs: (True, 0.0, []))
    dm.lesson_book = SimpleNamespace(
        get_position_size_multiplier=lambda signal_capture: 1.0
    )
    dm.trade_journal = SimpleNamespace(
        signal_capture=SimpleNamespace(capture=lambda *args, **kwargs: {}),
        record_entry=lambda *args, **kwargs: None,
        record_exit=lambda *args, **kwargs: None,
        record_trim=lambda *args, **kwargs: None,
        save=lambda: None,
    )
    dm._daily_drawdown_halt = False
    dm._daily_drawdown_pct = 0.0
    dm._daily_drawdown_limit = 1.5
    dm._daily_drawdown_tiers = [
        {"drawdown_pct": 1.0, "size_multiplier": 0.5},
        {"drawdown_pct": 1.5, "halt_entries": 1.0, "trail_atr": 1.5},
        {"drawdown_pct": 2.0, "trail_atr": 1.2, "trim_bottom_quartile": 1.0},
        {"drawdown_pct": 3.0, "force_close_losers": 1.0, "halt_next_open": 1.0},
    ]
    dm._daily_drawdown_fired_tiers = set()
    dm._daily_drawdown_action_flags = set()
    dm._daily_drawdown_size_multiplier = 1.0
    dm._daily_drawdown_tight_trail_atr = None
    dm._daily_drawdown_halt_next_open = False
    dm._daily_drawdown_halt_next_open_date = ""
    dm._emit_drawdown_tier_alert = lambda *args, **kwargs: None
    dm.strategy_failsafe_snapshot = SimpleNamespace(
        halt_new_entries=False,
        level="normal",
        position_size_multiplier=1.0,
        position_size_pct=0.0,
        max_positions=day_manager_mod.MAX_POSITIONS,
    )
    dm.position_advisor = None
    dm.use_agentic = False
    dm._thesis_cache = SimpleNamespace(
        get_prompt_context=lambda s: None, get=lambda s: None
    )
    dm.position_health = {}
    dm.policy_risk_decisions = {}
    dm.policy_health_report = {}
    dm.get_positions = lambda: []
    dm._session_state = _SessionStateStub({"trim_history": {}})
    dm.MIN_TRIM_NOTIONAL = 400.0
    dm.MIN_TRIM_QTY_PCT = 0.10
    dm.TRIM_COOLDOWN_MINUTES = 60
    dm.TRIM_PNL_STEP = 2.0
    dm.MAX_TRIMS_PER_SYMBOL_PER_DAY = 2
    dm.llm_advisory_escalation_cfg = SimpleNamespace(
        enabled=True,
        min_gap_minutes=30,
        advisor_cooldown_minutes=30,
        ladder_pct=[25, 50, 100],
        confidence_floor=0.70,
        full_exit_reentry_cooldown_minutes=120,
        max_steps_per_symbol_per_day=3,
        allow_risk_gate_profit_take_full_exit=False,
    )
    dm._llm_advisory_escalation_state = {}
    dm._llm_escalation_reentry_lockouts = {}
    dm._book_velocity_state = {}
    dm._last_trim_pnl = {}
    dm._last_trim_time = {}
    dm._trim_count_today = {}
    dm._last_exit_timestamp = {}
    dm._exited_symbols_today = set()
    dm._symbol_state_flips_today = {}
    dm._symbol_intent_state_today = {}
    dm.risk_gate_cfg = SimpleNamespace(
        gap_hard_cap_by_regime={
            "LEAN_BULLISH": 15.0,
            "NEUTRAL": 10.0,
            "RISK_OFF": 5.0,
        },
        re_entry_cooldown_minutes=30,
        max_state_flips_per_symbol=2,
    )
    dm._momentum_scanner_health = {
        "status": "unknown",
        "reason": "",
        "candidate_count": 0,
        "raw_candidate_count": 0,
        "loaded_count": 0,
        "artifact_path": "",
        "generated_at_et": "",
        "generated_at_ct": "",
        "empty_since": None,
        "last_fallback_reason": "",
    }
    dm._intraday_reserve_fallback_enabled = True
    dm._intraday_reserve_recent_days = 10
    dm._intraday_reserve_seed_cap = 60
    dm._get_full_watchlist_rows = lambda: []
    dm._core_data_readiness = {
        "is_fresh": True,
        "core_data_fresh": True,
        "pm_ready_for_execution": True,
        "blocking_reasons": [],
    }
    dm._hedge_decision_state = {
        "session_date": datetime.now().strftime("%Y-%m-%d"),
        "last_action": "",
        "last_symbol": "",
        "last_target_notional": 0.0,
        "last_decision_at": None,
        "cooldown_until": None,
        "open_order_ids": {},
        "decision_source": "",
    }
    dm._entry_authority_state = {
        "session_date": datetime.now().strftime("%Y-%m-%d"),
        "state": "open",
        "reason": "test",
        "updated_at": None,
        "snapshot": {},
        "inverse_fast_entries_taken": 0,
        "safety_reentry_refresh": None,
    }
    dm._benchmark_snapshot_cache = {}
    dm._benchmark_fetch_warn_seen = set()
    dm._inverse_fast_entry_symbols = set()
    dm._last_inverse_fast_screen_at = None
    dm._entry_order_lifecycle = {}
    dm._entry_capacity_snapshot = {}
    dm._hold_minutes = lambda symbol: 999
    dm.prepare_signal_recheck = DayManager.prepare_signal_recheck.__get__(
        dm, DayManager
    )
    dm._mark_signal_pending = DayManager._mark_signal_pending.__get__(dm, DayManager)
    dm._mark_signal_deferred = DayManager._mark_signal_deferred.__get__(dm, DayManager)
    dm._is_soft_signal_skip_reason = DayManager._is_soft_signal_skip_reason.__get__(
        dm, DayManager
    )
    dm._should_exclude_signal_from_future_scan = (
        DayManager._should_exclude_signal_from_future_scan.__get__(dm, DayManager)
    )
    dm._adaptive_core_score_threshold = (
        DayManager._adaptive_core_score_threshold.__get__(dm, DayManager)
    )
    dm._should_tolerate_watchlist_prune = (
        DayManager._should_tolerate_watchlist_prune.__get__(dm, DayManager)
    )
    dm._top_watchlist_candidates_for_watchdog = (
        DayManager._top_watchlist_candidates_for_watchdog.__get__(dm, DayManager)
    )
    dm._maybe_apply_no_fill_watchdog = DayManager._maybe_apply_no_fill_watchdog.__get__(
        dm, DayManager
    )
    dm._update_no_fill_watchdog_state = (
        DayManager._update_no_fill_watchdog_state.__get__(dm, DayManager)
    )
    return dm


def test_failsafe_halt_blocks_ordinary_long_exposure_increase():
    dm = _new_dm_stub()
    dm.strategy_failsafe_snapshot = SimpleNamespace(
        halt_new_entries=True,
        level="critical",
    )

    reason = dm._long_exposure_increase_block_reason(
        "EQNR", context="scale_into_winner"
    )

    assert reason.startswith("failsafe_halt_increase_exposure")
    assert "level=critical" in reason


def test_failsafe_halt_does_not_block_inverse_hedge_entry_lane():
    dm = _new_dm_stub()
    dm.strategy_failsafe_snapshot = SimpleNamespace(
        halt_new_entries=True,
        level="critical",
    )

    assert dm._long_exposure_increase_block_reason("SQQQ", context="inverse_fast") == ""


def test_feature_pipeline_failure_uses_local_feature_fallback():
    dm = _new_dm_stub()
    dm.feature_pipeline_active = True
    dm.feature_pipeline = SimpleNamespace(
        families_enabled=["technical"],
        execute=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("builder failed")),
    )
    dm._feature_pipeline_health = {
        "status": "initialized",
        "cache_hit": False,
        "last_error": None,
        "retry_requested": False,
    }
    dm._feature_cache = {}
    dm._feature_cache_updated_at = None
    dm._feature_cache_ttl = timedelta(minutes=10)
    dm._is_feature_cache_valid = lambda requested: False
    dm._current_phase_value = lambda: "market_open"
    dm._build_feature_pipeline_input = lambda requested: pd.DataFrame(
        [
            {
                "symbol": "SPY",
                "date": "2026-04-29",
                "close": 500.0,
                "volume": 1_000_000,
            }
        ]
    )

    context = DayManager._refresh_feature_context(dm, ["SPY"])

    assert context["SPY"]["close"] == 500.0
    assert dm._feature_pipeline_health["status"] == "degraded_fallback"
    assert dm._feature_pipeline_health["retry_requested"] is True


def _build_strength_reentry_bars(
    *,
    open_price: float = 10.0,
    first_hour_gain_pct: float = 2.0,
    pullback_pct: float = 1.4,
    reclaim_gain_pct: float = 3.0,
    reclaim_volume_multiplier: float = 1.4,
) -> pd.DataFrame:
    first_leg = [
        open_price * (1.0 + (first_hour_gain_pct / 100.0) * (idx + 1) / 60.0)
        for idx in range(60)
    ]
    high_mark = first_leg[-1]
    pullback_low = high_mark * (1.0 - pullback_pct / 100.0)
    pullback_leg = [
        high_mark - ((high_mark - pullback_low) * (idx + 1) / 15.0) for idx in range(15)
    ]
    reclaim_target = open_price * (1.0 + reclaim_gain_pct / 100.0)
    reclaim_leg = [
        pullback_low + ((reclaim_target - pullback_low) * (idx + 1) / 16.0)
        for idx in range(16)
    ]
    closes = first_leg + pullback_leg + reclaim_leg
    opens = [open_price] + closes[:-1]
    highs = [max(o, c) * 1.001 for o, c in zip(opens, closes)]
    lows = [min(o, c) * 0.999 for o, c in zip(opens, closes)]
    volume = [1000] * 60 + [850] * 15 + [int(1000 * reclaim_volume_multiplier)] * 16
    index = pd.date_range("2026-03-27 09:30:00", periods=len(closes), freq="min")
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
        },
        index=index,
    )


class _LifecycleLoggerStub:
    def __init__(self) -> None:
        self._logs = {}

    def log_signal(self, **kwargs):
        self._logs[kwargs["signal_id"]] = [("signal", kwargs)]

    def log_event(self, signal_id, event):
        self._logs.setdefault(signal_id, []).append(("event", event))


def test_compute_entry_limit_price_obeys_tier_chase_cap():
    dm = _new_dm_stub()

    limit_px = dm._compute_entry_limit_price(
        planned_entry=100.0,
        current_price=101.0,
        urgency_tier="high",
    )

    assert limit_px <= 100.19
    assert limit_px >= 100.0


def test_position_add_block_reason_honors_learning_blocked_symbol(monkeypatch):
    dm = _new_dm_stub()
    dm._entries_blocked_by_core_data = lambda: (False, "")
    dm._entries_blocked_by_regime = lambda symbol: (False, "")
    dm.position_health = {}
    monkeypatch.setitem(
        sys.modules,
        "autotrade.utils.daily_learning_state",
        SimpleNamespace(
            load_learning_state=lambda: {"active_rules": {"blocked_symbols": ["FUTU"]}}
        ),
    )

    assert dm._position_add_block_reason("FUTU") == "learning_blocked_symbol"


def test_position_add_block_reason_allows_learning_block_override_for_strength_reentry(
    monkeypatch,
):
    dm = _new_dm_stub()
    dm._entries_blocked_by_core_data = lambda: (False, "")
    dm._entries_blocked_by_regime = lambda symbol: (False, "")
    dm._strength_reentry_learning_override_allowed = lambda symbol, signal_data=None: (
        symbol == "FUTU"
    )
    dm.position_health = {}
    monkeypatch.setitem(
        sys.modules,
        "autotrade.utils.daily_learning_state",
        SimpleNamespace(
            load_learning_state=lambda: {"active_rules": {"blocked_symbols": ["FUTU"]}}
        ),
    )

    assert dm._position_add_block_reason("FUTU") == ""


def test_position_add_block_reason_allows_learning_block_override_during_follow_through(
    monkeypatch,
):
    dm = _new_dm_stub()
    dm._entries_blocked_by_core_data = lambda: (False, "")
    dm._entries_blocked_by_regime = lambda symbol: (False, "")
    dm.position_health = {}
    dm._strength_reentry_current_state = lambda symbol, **kwargs: {
        "ticker": symbol,
        "strength_reentry_ready": False,
        "strength_reentry_phase": "reclaim_follow_through",
        "strength_reentry_reclaim_detected_at": "2026-03-27T13:40:00-05:00",
    }
    monkeypatch.setitem(
        sys.modules,
        "autotrade.utils.daily_learning_state",
        SimpleNamespace(
            load_learning_state=lambda: {"active_rules": {"blocked_symbols": ["FUTU"]}}
        ),
    )

    assert dm._position_add_block_reason("FUTU") == ""


def test_should_defer_stub_cleanup_for_profitable_add_eligible_residual():
    dm = _new_dm_stub()
    dm.runtime_risk_gate = SimpleNamespace(
        config=SimpleNamespace(add_profit_threshold_pct=4.0)
    )
    dm._position_add_block_reason = lambda symbol: ""
    pos = SimpleNamespace(symbol="SEDG", unrealized_plpc=0.17)

    assert dm._should_defer_stub_cleanup(pos) is True


def test_replace_position_with_candidate_requires_preflight_before_exit():
    dm = _new_dm_stub()
    calls = []
    pos = SimpleNamespace(symbol="OLD")
    candidate = {"ticker": "NEW", "score": 80.0}

    dm.should_replace = lambda health, cand: True
    dm._position_qty = lambda position: 10

    def _fake_execute_entry(
        ticker,
        reason,
        candidate_data=None,
        entry_wave=None,
        preflight_only=False,
        replacement_for_symbol=None,
    ):
        calls.append(("entry", ticker, preflight_only))
        return False if preflight_only else True

    dm.execute_entry = _fake_execute_entry
    dm.execute_exit = lambda symbol, qty, reason: (
        calls.append(("exit", symbol, qty)) or True
    )

    replaced = dm._replace_position_with_candidate(
        pos,
        {"score": 20.0},
        candidate,
        entry_wave=1,
    )

    assert replaced is False
    assert calls == [("entry", "NEW", True)]


def test_replace_position_with_candidate_exits_only_after_preflight_passes():
    dm = _new_dm_stub()
    calls = []
    pos = SimpleNamespace(symbol="OLD")
    candidate = {"ticker": "NEW", "score": 80.0}

    dm.should_replace = lambda health, cand: True
    dm._position_qty = lambda position, default=0: 10
    dm.get_positions = lambda: [SimpleNamespace(symbol="OLD", qty=10)]

    def _fake_execute_entry(
        ticker,
        reason,
        candidate_data=None,
        entry_wave=None,
        preflight_only=False,
        replacement_for_symbol=None,
    ):
        calls.append(("entry", ticker, preflight_only))
        return True

    dm.execute_entry = _fake_execute_entry
    dm.execute_exit = lambda symbol, qty, reason: (
        calls.append(("exit", symbol, qty)) or True
    )

    replaced = dm._replace_position_with_candidate(
        pos,
        {"score": 20.0},
        candidate,
        entry_wave=1,
    )

    assert replaced is True
    assert calls == [
        ("entry", "NEW", True),
        ("exit", "OLD", 10),
        ("entry", "NEW", False),
    ]


def test_replace_position_with_candidate_uses_refreshed_qty_after_trims():
    dm = _new_dm_stub()
    calls = []
    pos = SimpleNamespace(symbol="GFI", qty=27)
    candidate = {"ticker": "PAAS", "score": 86.0}

    dm.should_replace = lambda health, cand: True
    dm.get_positions = lambda: [SimpleNamespace(symbol="GFI", qty=19)]

    def _fake_execute_entry(
        ticker,
        reason,
        candidate_data=None,
        entry_wave=None,
        preflight_only=False,
        replacement_for_symbol=None,
    ):
        calls.append(("entry", ticker, preflight_only, replacement_for_symbol))
        return True

    def _fake_execute_exit(symbol, qty, reason):
        calls.append(("exit", symbol, qty))
        return qty == 19

    dm.execute_entry = _fake_execute_entry
    dm.execute_exit = _fake_execute_exit

    replaced = dm._replace_position_with_candidate(
        pos,
        {"score": 20.0},
        candidate,
        entry_wave=2,
    )

    assert replaced is True
    assert calls == [
        ("entry", "PAAS", True, "GFI"),
        ("exit", "GFI", 19),
        ("entry", "PAAS", False, None),
    ]


def test_replace_position_with_candidate_continues_when_position_already_gone():
    dm = _new_dm_stub()
    calls = []
    pos = SimpleNamespace(symbol="GFI", qty=27)
    candidate = {"ticker": "RKT", "score": 82.0}

    dm.should_replace = lambda health, cand: True
    dm.get_positions = lambda: []

    def _fake_execute_entry(
        ticker,
        reason,
        candidate_data=None,
        entry_wave=None,
        preflight_only=False,
        replacement_for_symbol=None,
    ):
        calls.append(("entry", ticker, preflight_only, replacement_for_symbol))
        return True

    dm.execute_entry = _fake_execute_entry
    dm.execute_exit = lambda symbol, qty, reason: (
        calls.append(("exit", symbol, qty)) or False
    )

    replaced = dm._replace_position_with_candidate(
        pos,
        {"score": 20.0},
        candidate,
        entry_wave=2,
    )

    assert replaced is True
    assert calls == [
        ("entry", "RKT", True, "GFI"),
        ("entry", "RKT", False, None),
    ]


def test_should_replace_uses_lower_gap_for_weak_positions():
    dm = _new_dm_stub()

    assert (
        dm.should_replace(
            {"score": 2.0},
            {"ticker": "NEW", "realtime_score": 52.0},
        )
        is True
    )
    assert (
        dm.should_replace(
            {"score": 12.0},
            {"ticker": "NEW", "realtime_score": 24.0},
        )
        is False
    )


def test_soft_skip_reasons_do_not_block_future_scan_exclusion():
    dm = _new_dm_stub()

    assert (
        dm._should_exclude_signal_from_future_scan(
            {"status": "skipped", "reason": "below_min_score"}
        )
        is False
    )
    assert (
        dm._should_exclude_signal_from_future_scan(
            {"status": "skipped", "reason": "watchlist_invalid:orb_breakdown_15m_low"}
        )
        is False
    )
    assert dm._should_exclude_signal_from_future_scan({"status": "executed"}) is True


def test_adaptive_core_threshold_relaxes_for_strong_top_ranked_candidates():
    dm = _new_dm_stub()

    threshold = dm._adaptive_core_score_threshold(
        phase=day_manager_mod.TradingPhase.CORE_TRADING,
        base_threshold=35.0,
        candidates=[
            {
                "ticker": "WBI",
                "ranking_position": 8,
                "realtime_score": 36.6,
                "entry_validation": {"allowed": True},
            }
        ],
        open_slots=2,
        max_new_entries=2,
    )

    assert threshold == 30.0


def test_should_run_advisor_respects_min_recheck_window():
    dm = _new_dm_stub()
    now = dm._now_utc()
    dm._advisor_eval_cache["AAPL"] = {
        "at": now - timedelta(seconds=120),
        "pnl_pct": 1.0,
        "price": 100.0,
        "hold_minutes": 40,
        "health": {"action": "hold", "score": 0},
    }

    should_run = dm._should_run_advisor(
        symbol="AAPL",
        pnl_pct=1.1,
        current_price=100.1,
        hold_minutes=42,
    )

    assert should_run is False


def test_should_run_advisor_triggers_on_price_move_after_min_window():
    dm = _new_dm_stub()
    now = dm._now_utc()
    dm._advisor_eval_cache["MSFT"] = {
        "at": now - timedelta(seconds=360),
        "pnl_pct": 0.5,
        "price": 100.0,
        "hold_minutes": 10,
        "health": {"action": "hold", "score": 0},
    }

    should_run = dm._should_run_advisor(
        symbol="MSFT",
        pnl_pct=0.6,
        current_price=101.0,
        hold_minutes=15,
    )

    assert should_run is True


def test_execute_entry_blocks_same_day_reentry_after_hard_stop_exit():
    dm = _new_dm_stub()
    dm.signal_status = {"CNK": {"status": "executed", "reason": ""}}
    dm._hard_stop_blocked_symbols_today = {"CNK"}
    dm.get_positions = lambda: []
    dm._mark_signal_skipped = DayManager._mark_signal_skipped.__get__(dm, DayManager)

    allowed = dm.execute_entry("CNK", "test hard-stop re-entry")

    assert allowed is False
    assert dm.signal_status["CNK"]["status"] == "skipped"
    assert dm.signal_status["CNK"]["reason"] == "same_day_hard_stop_exit"


def test_execute_entry_blocks_same_day_reentry_after_losing_exit():
    dm = _new_dm_stub()
    dm.signal_status = {"CNK": {"status": "pending", "reason": ""}}
    dm.trade_journal = SimpleNamespace(
        get_same_day_losing_symbols=lambda: {"CNK"},
        get_recent_symbol_quarantine=lambda **kwargs: {},
        save=lambda: None,
    )
    dm._mark_signal_skipped = DayManager._mark_signal_skipped.__get__(dm, DayManager)

    allowed = dm.execute_entry("CNK", "test losing re-entry")

    assert allowed is False
    assert dm.signal_status["CNK"]["reason"] == "same_day_losing_exit"


def test_strength_reentry_does_not_bypass_same_day_hard_exit():
    dm = _new_dm_stub()
    dm.current_regime = "CHOP"
    dm._hard_stop_blocked_symbols_today = {"RBRK"}
    dm._symbol_intent_state_today = {
        "RBRK": {"last_exit_like_intent": "sell", "last_exit_like_hard": True}
    }
    signal_data = {
        "ticker": "RBRK",
        "entry_source": "strength_reentry",
        "strength_reentry_ready": True,
        "score": 82.0,
    }

    assert dm._strength_reentry_candidate_allowed(signal_data) is False
    reasons = dm._entry_guard_reasons("RBRK", 82.0, signal_data)

    assert reasons == ["same_symbol_intent_conflict:sell_before_entry"]


def test_execute_entry_blocks_recent_reentry_after_exit_cooldown():
    dm = _new_dm_stub()
    dm.signal_status = {"CNK": {"status": "executed", "reason": ""}}
    dm._exited_symbols_today = {"CNK"}
    dm._last_exit_timestamp = {"CNK": datetime.now() - timedelta(minutes=5)}
    dm._symbol_state_flips_today = {"CNK": 1}
    dm.get_positions = lambda: []
    dm.get_current_price = lambda ticker: 10.0
    dm.strategy_failsafe_snapshot = SimpleNamespace(
        halt_new_entries=False,
        level="normal",
        position_size_multiplier=1.0,
        position_size_pct=0.0,
    )
    dm.entry_quality_cfg = SimpleNamespace(
        wave_entry_enabled=False,
        wave_max_entries=10,
        momentum_gate_enabled=False,
        momentum_hard_block=False,
    )
    dm.entry_wave = 1
    dm.wave_positions = {}
    dm.can_enter_positions = lambda entry_wave=None: (True, "")
    dm._check_learned_lessons = lambda ticker, price: (False, "")
    dm.signals = [
        {
            "ticker": "CNK",
            "action": "buy_open",
            "score": 75,
            "atr_14": 1.0,
            "entry_source": "overnight_plan",
        }
    ]
    dm._has_open_buy_order = lambda symbol_key: (False, "")
    dm._should_promote_watch_signal = lambda signal_data: (False, "")
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm._is_strategy_window_open = lambda strategy_profile: True
    dm.lesson_book = SimpleNamespace(get_position_size_multiplier=lambda capture: 1.0)
    dm.trade_journal = SimpleNamespace(
        signal_capture=SimpleNamespace(capture=lambda *args, **kwargs: {}),
        save=lambda: None,
    )
    dm._conviction_size_multiplier = lambda score: 1.0
    dm.youtube_context = {}
    dm._regime_override_float = lambda key, fallback=1.0: 1.0
    dm.research_position_size_multiplier = 1.0
    dm.last_account_equity = 100_000.0
    dm._validate_order = lambda ticker, qty, side: (True, "")
    dm._effective_stop_multiplier = lambda fallback=2.0: 2.0
    dm._is_breakout_continuation_setup = lambda signal_data: False
    dm._effective_market_regime = lambda: "NEUTRAL"
    dm.regime_strategy_overrides = {}
    dm._record_wave_entry = lambda ticker, wave=None: None
    dm._record_strategy_profile_entry = lambda signal_data: None
    dm._clear_vl_recheck = lambda ticker: None
    dm.runtime_risk_gate = None
    dm.execution_entry_cfg = SimpleNamespace(
        high_urgency_min_score=72.0,
        critical_urgency_min_score=84.0,
        normal=SimpleNamespace(max_chase_bps=8, max_slippage_bps=12),
        high=SimpleNamespace(max_chase_bps=10, max_slippage_bps=15),
        critical=SimpleNamespace(max_chase_bps=12, max_slippage_bps=20),
    )
    dm.dry_run = True

    allowed = DayManager.execute_entry(dm, "CNK", "test re-entry")

    assert allowed is False
    assert dm.signal_status["CNK"]["reason"].startswith("re_entry_cooldown:")


def test_execute_entry_blocks_reentry_after_state_flip_cap():
    dm = _new_dm_stub()
    dm.signal_status = {"CNK": {"status": "executed", "reason": ""}}
    dm._exited_symbols_today = {"CNK"}
    dm._last_exit_timestamp = {"CNK": datetime.now() - timedelta(hours=2)}
    dm._symbol_state_flips_today = {"CNK": 2}
    dm.get_positions = lambda: []
    dm.get_current_price = lambda ticker: 10.0
    dm.strategy_failsafe_snapshot = SimpleNamespace(
        halt_new_entries=False,
        level="normal",
        position_size_multiplier=1.0,
        position_size_pct=0.0,
    )
    dm.entry_quality_cfg = SimpleNamespace(
        wave_entry_enabled=False,
        wave_max_entries=10,
        momentum_gate_enabled=False,
        momentum_hard_block=False,
    )
    dm.entry_wave = 1
    dm.wave_positions = {}
    dm.can_enter_positions = lambda entry_wave=None: (True, "")
    dm._check_learned_lessons = lambda ticker, price: (False, "")
    dm.signals = [
        {
            "ticker": "CNK",
            "action": "buy_open",
            "score": 75,
            "atr_14": 1.0,
            "entry_source": "overnight_plan",
        }
    ]
    dm._has_open_buy_order = lambda symbol_key: (False, "")
    dm._should_promote_watch_signal = lambda signal_data: (False, "")
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm._is_strategy_window_open = lambda strategy_profile: True
    dm.lesson_book = SimpleNamespace(get_position_size_multiplier=lambda capture: 1.0)
    dm.trade_journal = SimpleNamespace(
        signal_capture=SimpleNamespace(capture=lambda *args, **kwargs: {}),
        save=lambda: None,
    )
    dm._conviction_size_multiplier = lambda score: 1.0
    dm.youtube_context = {}
    dm._regime_override_float = lambda key, fallback=1.0: 1.0
    dm.research_position_size_multiplier = 1.0
    dm.last_account_equity = 100_000.0
    dm._validate_order = lambda ticker, qty, side: (True, "")
    dm._effective_stop_multiplier = lambda fallback=2.0: 2.0
    dm._is_breakout_continuation_setup = lambda signal_data: False
    dm._effective_market_regime = lambda: "NEUTRAL"
    dm.regime_strategy_overrides = {}
    dm._record_wave_entry = lambda ticker, wave=None: None
    dm._record_strategy_profile_entry = lambda signal_data: None
    dm._clear_vl_recheck = lambda ticker: None
    dm.runtime_risk_gate = None
    dm.execution_entry_cfg = SimpleNamespace(
        high_urgency_min_score=72.0,
        critical_urgency_min_score=84.0,
        normal=SimpleNamespace(max_chase_bps=8, max_slippage_bps=12),
        high=SimpleNamespace(max_chase_bps=10, max_slippage_bps=15),
        critical=SimpleNamespace(max_chase_bps=12, max_slippage_bps=20),
    )
    dm.dry_run = True

    allowed = DayManager.execute_entry(dm, "CNK", "test re-entry")

    assert allowed is False
    assert dm.signal_status["CNK"]["reason"] == "state_flip_cap:2>=2"


def test_execute_entry_blocks_adds_on_red_held_position():
    dm = _new_dm_stub()
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm._entries_blocked_by_core_data = lambda: (False, "")
    dm._entries_blocked_by_regime = lambda symbol: (False, "")
    dm._same_day_loser_reentry_reason = lambda symbol: ""
    dm._recent_symbol_quarantine_reason = lambda symbol: ""
    dm._learning_blocked_symbol_reason = lambda symbol: ""
    dm._validate_order = lambda *args, **kwargs: (True, "")
    dm._entry_submission_block_reason = lambda *args, **kwargs: ""
    dm._acquire_order_submission_guard = lambda *args, **kwargs: (True, "", "guard-1")
    dm._release_order_submission_guard = lambda *args, **kwargs: None
    dm._mark_signal_skipped = DayManager._mark_signal_skipped.__get__(dm, DayManager)
    dm.signal_status = {"ACLS": {"status": "pending", "reason": ""}}
    dm.signals = [{"ticker": "ACLS", "score": 10.0, "entry_source": "overnight_plan"}]
    dm.get_positions = lambda: [
        SimpleNamespace(
            symbol="ACLS",
            qty=17,
            current_price=84.08,
            avg_entry_price=115.275556,
            unrealized_plpc=-0.27062,
            market_value=1429.36,
        )
    ]
    dm._validated_positions = lambda positions, context="": positions

    allowed = dm.execute_entry("ACLS", "test red add block")

    assert allowed is False
    assert dm.signal_status["ACLS"]["reason"].startswith("buy_on_held_blocked:")


def test_execute_entry_blocks_recent_quarantined_symbol():
    dm = _new_dm_stub()
    dm.signal_status = {"CNK": {"status": "pending", "reason": ""}}
    dm.trade_journal = SimpleNamespace(
        get_same_day_losing_symbols=lambda: set(),
        get_recent_symbol_quarantine=lambda **kwargs: {"CNK": "recent_repeat_loser"},
    )
    dm._mark_signal_skipped = DayManager._mark_signal_skipped.__get__(dm, DayManager)

    allowed = dm.execute_entry("CNK", "test quarantine")

    assert allowed is False
    assert dm.signal_status["CNK"]["reason"] == "recent_repeat_loser"


def test_execute_entry_blocks_when_core_data_not_ready():
    dm = _new_dm_stub()
    dm.signal_status = {"CNK": {"status": "pending", "reason": ""}}
    dm._core_data_readiness = {
        "is_fresh": False,
        "core_data_fresh": False,
        "pm_ready_for_execution": False,
        "blocking_reasons": ["core_data_stale:2026-03-05->2026-03-06"],
    }
    dm._refresh_core_data_readiness = lambda: dict(dm._core_data_readiness)
    dm._mark_signal_skipped = DayManager._mark_signal_skipped.__get__(dm, DayManager)
    dm.get_current_phase = lambda: TradingPhase.CORE_TRADING

    allowed = dm.execute_entry("CNK", "test stale core data")
    assert allowed is False
    assert dm.signal_status["CNK"]["reason"] == "core_data_stale:2026-03-05->2026-03-06"


def test_execute_entry_blocks_when_projected_capacity_exhausted_even_for_elite():
    dm = _new_dm_stub()
    dm.signal_status = {"CNK": {"status": "pending", "reason": ""}}
    dm.signals = [
        {
            "ticker": "CNK",
            "action": "buy_open",
            "confidence": 91.0,
            "score": 91.0,
            "entry_source": "overnight_plan",
            "plan_score_source": "pm_plan_2026-04-07.json",
        }
    ]
    dm._effective_max_positions = lambda: 1
    dm.get_positions = lambda: [SimpleNamespace(symbol="AAA", market_value=1000.0)]
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm.get_current_price = lambda ticker: 20.0
    dm._mark_signal_skipped = DayManager._mark_signal_skipped.__get__(dm, DayManager)

    allowed = dm.execute_entry("CNK", "test capacity gate")

    assert allowed is False
    assert dm.signal_status["CNK"]["reason"].startswith(
        "projected_max_positions_reached:"
    )


def test_can_enter_positions_respects_core_cap_and_keeps_reserve_headroom():
    dm = _new_dm_stub()
    dm._resolve_position_caps = lambda: SimpleNamespace(
        total_cap=35,
        core_cap=25,
        reserve_cap=10,
        weak_day=True,
        regime_label="SELLOFF",
    )
    dm._effective_max_positions = lambda: 35
    dm._effective_core_max_positions = lambda: 25
    dm._check_execution_circuit_breaker = lambda: False
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm.get_positions = lambda: [
        SimpleNamespace(symbol=f"C{i}", market_value=1000.0) for i in range(25)
    ]

    allowed_core, core_reason = dm.can_enter_positions(ticker="NEW1", slot_class="core")
    allowed_reserve, reserve_reason = dm.can_enter_positions(
        ticker="SQQQ", slot_class="inverse_etf_reserve"
    )

    assert allowed_core is False
    assert core_reason == "core_max_positions_reached"
    assert allowed_reserve is True
    assert reserve_reason.startswith("STRATEGY RESERVE: Inverse ETF SQQQ allowed")


def test_execute_entry_replacement_preflight_can_free_one_for_one_capacity():
    dm = _new_dm_stub()
    dm.dry_run = True
    dm.signal_status = {"CNK": {"status": "pending", "reason": ""}}
    dm.signals = [
        {
            "ticker": "CNK",
            "action": "buy_open",
            "confidence": 70.0,
            "score": 70.0,
            "entry_source": "overnight_plan",
            "plan_score_source": "pm_plan_2026-04-07.json",
        }
    ]
    dm._effective_max_positions = lambda: 1
    dm.get_positions = lambda: [SimpleNamespace(symbol="AAA", market_value=1000.0)]
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm.get_current_price = lambda ticker: 20.0
    dm.can_enter_positions = lambda entry_wave=None: (True, "")
    dm._has_open_buy_order = lambda symbol: (False, "")
    dm._build_candidate_validation_report = lambda signal_data: {
        "allowed": True,
        "entry_source": "overnight_plan",
    }
    dm._resolve_entry_authority = lambda signal_data: {
        "eligible": True,
        "entry_score": 70.0,
    }
    dm._live_execution_mode = lambda: {"resolved_regime": {"regime": "NEUTRAL"}}
    dm._mark_signal_skipped = DayManager._mark_signal_skipped.__get__(dm, DayManager)

    allowed = dm.execute_entry(
        "CNK",
        "replacement preflight",
        preflight_only=True,
        replacement_for_symbol="AAA",
    )

    assert allowed is True


def test_execute_entry_uses_runtime_checks_instead_of_persisted_resolved_regime_veto():
    dm = _new_dm_stub()
    dm._watched_universe_tickers = {"CNK"}
    dm.signals = [
        {
            "ticker": "CNK",
            "action": "buy_open",
            "recommendation": "buy",
            "confidence": 70.0,
            "score": 70.0,
            "normalized_score": 70.0,
            "entry_source": "overnight_plan",
            "plan_score_source": "adjusted_plan_20260314_0829.json",
        }
    ]
    dm.signal_status = {"CNK": {"status": "pending", "reason": ""}}
    dm.youtube_context = {
        "resolved_regime": {
            "regime": "CRASH",
            "allow_new_longs": False,
            "plan_source": "adjusted_plan_20260314_0829.json",
        }
    }
    dm.strategy_failsafe_snapshot = SimpleNamespace(
        halt_new_entries=False, level="normal"
    )
    dm.get_current_price = lambda ticker: 20.0
    dm._check_universe_compliance = lambda *args, **kwargs: (True, "")
    dm.wave_positions = {}
    dm.entry_wave = 1
    dm.get_positions = lambda: []
    dm.can_enter_positions = lambda entry_wave=None: (True, "")
    dm._has_open_buy_order = lambda symbol: (True, "ord-1")
    dm._mark_signal_skipped = DayManager._mark_signal_skipped.__get__(dm, DayManager)

    allowed = dm.execute_entry("CNK", "test resolved regime advisory")

    assert allowed is False
    assert dm.signal_status["CNK"]["reason"] == "entry_authority_state_bull_lock"


def test_market_posture_uses_cautious_selective_longs_for_risk_off():
    dm = _new_dm_stub()

    dm._live_execution_mode = lambda: {
        "resolved_regime": {"regime": "RISK_OFF"},
        "entry_authority_state": "open",
        "entry_authority_snapshot": {},
    }
    dm.get_positions = lambda: []

    posture = DayManager._market_posture(dm, positions=[])

    assert posture["posture"] == "cautious_selective_longs"
    assert posture["reason"] == "bearish_regime_selective_longs"


def test_defensive_long_block_allows_exceptional_hot_sector_candidate_on_bad_day():
    dm = _new_dm_stub()
    dm.live_sector_bias = {"energy": 15.0}
    dm._live_execution_mode = lambda: {
        "resolved_regime": {"regime": "RISK_OFF"},
        "entry_authority_state": "open",
        "entry_authority_snapshot": {},
    }

    reason = DayManager._defensive_long_block_reason(
        dm,
        "XOM",
        signal_data={
            "ticker": "XOM",
            "sector": "energy",
            "score": 78.0,
            "relative_strength": 1.25,
            "risk_reward": 1.6,
            "volume_ratio": 1.15,
            "entry_source": "overnight_plan",
        },
        current_price=105.0,
        positions=[],
    )

    assert reason == ""


def test_defensive_long_block_blocks_unqualified_risk_off_candidate():
    dm = _new_dm_stub()
    dm.signal_status = {"CNK": {"status": "pending", "reason": ""}}
    dm.signals = [
        {
            "ticker": "CNK",
            "action": "buy_open",
            "confidence": 70.0,
            "score": 70.0,
            "entry_source": "overnight_plan",
            "plan_score_source": "pm_plan_2026-04-07.json",
            "sector": "technology",
            "relative_strength": 0.2,
            "risk_reward": 1.1,
            "volume_ratio": 0.9,
        }
    ]
    dm._live_execution_mode = lambda: {
        "resolved_regime": {"regime": "RISK_OFF"},
        "entry_authority_state": "open",
        "entry_authority_snapshot": {},
    }
    dm.get_positions = lambda: []
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm.get_current_price = lambda ticker: 20.0
    dm._entries_blocked_by_core_data = lambda: (False, "")
    dm._mark_signal_skipped = DayManager._mark_signal_skipped.__get__(dm, DayManager)

    allowed = dm.execute_entry("CNK", "risk off long gate")

    assert allowed is False
    assert (
        dm.signal_status["CNK"]["reason"]
        == "bad_day_posture_block:cautious_selective_longs"
    )


def test_cautious_selective_longs_admits_representative_quality_candidates(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(day_manager_mod, "LOG_DIR", tmp_path / "logs")
    dm = _new_dm_stub()
    dm.cycle_count = 7
    dm._live_execution_mode = lambda: {
        "resolved_regime": {"regime": "RISK_OFF"},
        "entry_authority_state": "open",
        "entry_authority_snapshot": {},
    }
    dm.get_positions = lambda: []
    dm.live_sector_bias = {}
    dm._live_benchmark_snapshot = lambda *args, **kwargs: {"recovery_confirmed": False}

    candidates = []
    for idx in range(10):
        qualified = idx < 5
        candidates.append(
            {
                "ticker": f"T{idx}",
                "score": 69.0 if qualified else 64.0,
                "relative_strength_pp": 0.9,
                "risk_reward": 1.3 if qualified else 1.1,
                "volume_ratio": 1.05 if qualified else 0.8,
                "entry_source": "replacement_engine",
            }
        )

    decisions = [
        dm._defensive_long_block_reason(
            row["ticker"],
            signal_data=row,
            current_price=20.0,
            positions=[],
        )
        for row in candidates
    ]

    assert sum(1 for reason in decisions if reason == "") >= 4
    records = [
        json.loads(line)
        for line in (
            tmp_path / "logs" / f"posture_transitions_{datetime.now():%Y-%m-%d}.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
        if '"event": "posture_impact"' in line
    ]
    assert records[-1]["accepted_count"] >= 4
    assert records[-1]["rejected_count"] >= 4


def test_market_posture_stress_count_ignores_trim_and_observed_exit():
    dm = _new_dm_stub()
    today = datetime.now().isoformat()
    dm.trade_journal = SimpleNamespace(
        trades=[
            {
                "symbol": "TRIM",
                "trade_type": "trim",
                "timestamp": today,
                "pnl_percent": -5.0,
            },
            {
                "symbol": "OBS",
                "trade_type": "observed_exit",
                "timestamp": today,
                "pnl_percent": -8.0,
            },
            {
                "symbol": "LOSS",
                "trade_type": "exit",
                "exit_time": today,
                "pnl_percent": -2.0,
            },
        ]
    )

    assert dm._same_day_stress_exit_count() == 1


def test_capital_preservation_allows_high_conviction_after_recovery():
    dm = _new_dm_stub()
    dm._live_execution_mode = lambda: {
        "resolved_regime": {"regime": "RISK_OFF"},
        "entry_authority_state": "open",
        "entry_authority_snapshot": {},
    }
    dm._live_benchmark_snapshot = lambda: {
        "available_count": 3,
        "recovery_confirmed": True,
    }
    dm._same_day_stress_exit_count = lambda **kwargs: 3
    dm.get_positions = lambda: []
    dm.live_sector_bias = {"technology": 15.0}

    reason = dm._defensive_long_block_reason(
        "ABCD",
        signal_data={
            "ticker": "ABCD",
            "score": 86.0,
            "sector": "technology",
            "risk_reward": 2.0,
            "relative_strength_pp": 1.5,
        },
        current_price=10.0,
        positions=[],
    )

    assert reason == ""


def test_capital_preservation_blocks_high_conviction_without_recovery():
    dm = _new_dm_stub()
    dm._live_execution_mode = lambda: {
        "resolved_regime": {"regime": "RISK_OFF"},
        "entry_authority_state": "open",
        "entry_authority_snapshot": {},
    }
    dm._live_benchmark_snapshot = lambda: {
        "available_count": 3,
        "recovery_confirmed": False,
    }
    dm._same_day_stress_exit_count = lambda **kwargs: 3
    dm.get_positions = lambda: []
    dm.live_sector_bias = {"technology": 15.0}

    reason = dm._defensive_long_block_reason(
        "ABCD",
        signal_data={
            "ticker": "ABCD",
            "score": 82.0,
            "sector": "technology",
            "risk_reward": 2.2,
            "relative_strength_pp": 2.0,
        },
        current_price=10.0,
        positions=[],
    )

    assert reason == "bad_day_posture_block:capital_preservation"


def test_same_symbol_intent_guard_blocks_trim_to_add_churn():
    dm = _new_dm_stub()
    dm._record_symbol_intent("CORZ", "trim", reason="profit_take")

    reason = dm._same_symbol_intent_block_reason(
        "CORZ",
        "add",
        {"ticker": "CORZ", "entry_source": "scale_into_winner"},
    )

    assert reason == "same_symbol_intent_conflict:trim_before_add"


def test_same_symbol_intent_guard_allows_explicit_strength_reentry_after_trim():
    dm = _new_dm_stub()
    dm.current_regime = "CHOP"
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm._record_symbol_intent("CORZ", "trim", reason="profit_take")

    reason = dm._same_symbol_intent_block_reason(
        "CORZ",
        "entry",
        {
            "ticker": "CORZ",
            "entry_source": "strength_reentry",
            "strength_reentry_ready": True,
            "score": 82.0,
        },
    )

    assert reason == ""


def test_execute_entry_blocks_when_entry_authority_missing_score():
    dm = _new_dm_stub()
    dm.signals = [
        {
            "ticker": "CNK",
            "action": "buy_open",
            "entry_source": "overnight_plan",
            "source_bucket": "watchlist",
            "plan_score_source": "morning_game_plan_2026-03-14.json",
        }
    ]
    dm.signal_status = {"CNK": {"status": "pending", "reason": ""}}
    dm.strategy_failsafe_snapshot = SimpleNamespace(
        halt_new_entries=False, level="normal", position_size_multiplier=1.0
    )
    dm.get_current_price = lambda ticker: 20.0
    dm._check_universe_compliance = lambda *args, **kwargs: (True, "")
    dm._apply_regime_entry_gate = lambda signal_data, allow_capped=False: (True, "")
    dm.wave_positions = {}
    dm.entry_wave = 1
    dm.get_positions = lambda: []
    dm.can_enter_positions = lambda entry_wave=None: (True, "")
    dm._mark_signal_skipped = DayManager._mark_signal_skipped.__get__(dm, DayManager)

    allowed = dm.execute_entry("CNK", "missing score")

    assert allowed is False
    assert dm.signal_status["CNK"]["reason"] == "entry_authority_missing_score"


def test_position_add_block_reason_ignores_persisted_resolved_regime_veto():
    dm = _new_dm_stub()
    dm.youtube_context = {
        "resolved_regime": {
            "regime": "RISK_OFF",
            "allow_new_longs": False,
        }
    }

    reason = dm._position_add_block_reason("CNK")

    assert reason == "entry_authority_state_bull_lock"


def test_entries_blocked_by_regime_blocks_bull_lock_state_for_longs():
    dm = _new_dm_stub()
    dm._refresh_entry_authority_state = lambda positions=None: {
        "state": "bull_lock",
        "reason": "bearish_bias_pending_live_recovery",
    }

    blocked, reason = dm._entries_blocked_by_regime("CNK")

    assert blocked is True
    assert reason == "entry_authority_state_bull_lock"


def test_refresh_entry_authority_state_keeps_bearish_bias_locked_without_recovery():
    dm = _new_dm_stub()
    dm.youtube_context = {
        "resolved_regime": {
            "regime": "SELL_OFF",
            "allow_new_longs": False,
        }
    }
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm._live_benchmark_snapshot = lambda now=None: {
        "minutes_since_open": 12,
        "crash_confirmed": False,
        "recovery_confirmed": False,
        "red_ratio": 0.5,
        "avg_pct_change": -0.4,
    }

    state = DayManager._refresh_entry_authority_state(dm, positions=[])

    assert state["state"] == "bull_lock"
    assert state["reason"] == "bearish_bias_pending_live_recovery"


def test_refresh_entry_authority_state_opens_inverse_fast_on_broad_weakness():
    dm = _new_dm_stub()
    dm.youtube_context = {
        "resolved_regime": {
            "regime": "SELL_OFF",
            "allow_new_longs": False,
        }
    }
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm._live_benchmark_snapshot = lambda now=None: {
        "minutes_since_open": 18,
        "available_count": 4,
        "crash_confirmed": False,
        "recovery_confirmed": False,
        "red_ratio": 0.75,
        "avg_pct_change": -0.34,
    }

    state = DayManager._refresh_entry_authority_state(dm, positions=[])

    assert state["state"] == "inverse_fast"
    assert state["reason"] == "bearish_bias_broad_weakness"


def test_refresh_entry_authority_state_opens_inverse_fast_on_live_weakness_without_regime_bias():
    dm = _new_dm_stub()
    dm.youtube_context = {}
    dm._load_resolved_regime_context = lambda: {}
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm._live_benchmark_snapshot = lambda now=None: {
        "minutes_since_open": 18,
        "available_count": 4,
        "crash_confirmed": False,
        "recovery_confirmed": False,
        "red_ratio": 0.75,
        "avg_pct_change": -0.34,
    }

    state = DayManager._refresh_entry_authority_state(dm, positions=[])

    assert state["state"] == "inverse_fast"
    assert state["reason"] == "live_broad_weakness_inverse_fast"


def test_get_entry_authority_state_honors_session_date_override():
    dm = _new_dm_stub()
    dm._entry_authority_session_date_override = "2026-03-19"
    dm._entry_authority_state = {"session_date": "2026-03-25", "state": "inverse_fast"}

    state = DayManager._get_entry_authority_state(dm)

    assert state["session_date"] == "2026-03-19"
    assert state["state"] == "open"
    assert state["reason"] == "session_reset"


def test_refresh_entry_authority_state_keeps_bearish_bias_locked_after_first_hour():
    dm = _new_dm_stub()
    dm.youtube_context = {
        "resolved_regime": {
            "regime": "SELL_OFF",
            "allow_new_longs": False,
        }
    }
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm._live_benchmark_snapshot = lambda now=None: {
        "minutes_since_open": 125,
        "crash_confirmed": False,
        "recovery_confirmed": False,
        "red_ratio": 0.48,
        "avg_pct_change": -0.2,
    }

    state = DayManager._refresh_entry_authority_state(dm, positions=[])

    assert state["state"] == "bull_lock"
    assert state["reason"] == "bearish_bias_pending_live_recovery"


def test_refresh_entry_authority_state_requires_broad_weakness_for_inverse_fast():
    dm = _new_dm_stub()
    dm.youtube_context = {
        "resolved_regime": {
            "regime": "SELL_OFF",
            "allow_new_longs": False,
        }
    }
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm._live_benchmark_snapshot = lambda now=None: {
        "minutes_since_open": 52,
        "available_count": 4,
        "crash_confirmed": False,
        "recovery_confirmed": False,
        "red_ratio": 0.5,
        "avg_pct_change": -0.28,
    }

    state = DayManager._refresh_entry_authority_state(dm, positions=[])

    assert state["state"] == "bull_lock"
    assert state["reason"] == "bearish_bias_pending_live_recovery"


def test_entry_authority_bias_active_ignores_stale_bearish_plan_on_strong_green_day():
    dm = _new_dm_stub()
    dm.youtube_context = {
        "resolved_regime": {
            "regime": "SELL_OFF",
            "allow_new_longs": False,
        }
    }
    dm._live_benchmark_snapshot = lambda now=None: {
        "available_count": 3,
        "green_ratio": 1.0,
        "avg_pct_change": 2.7,
    }

    assert DayManager._entry_authority_bias_active(dm) is False


def test_refresh_entry_authority_state_keeps_inverse_fast_during_bearish_bias_with_neutral_router():
    dm = _new_dm_stub()
    dm.youtube_context = {
        "resolved_regime": {
            "regime": "SELL_OFF",
            "allow_new_longs": False,
        }
    }
    dm.regime_router_context = {"regime": "neutral"}
    dm._entry_authority_state["state"] = "inverse_fast"
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm._live_benchmark_snapshot = lambda now=None: {
        "minutes_since_open": 48,
        "crash_confirmed": False,
        "recovery_confirmed": False,
        "red_ratio": 0.46,
        "avg_pct_change": -0.24,
    }

    state = DayManager._refresh_entry_authority_state(dm, positions=[])

    assert state["state"] == "inverse_fast"
    assert state["reason"] == "bearish_bias_inverse_fast_monitoring"


def test_refresh_entry_authority_state_requires_stable_recovery_before_safety_reentry():
    dm = _new_dm_stub()
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm._entry_authority_state["state"] = "inverse_fast"
    dm._entry_authority_state["recovery_confirmed_streak"] = 1
    dm._live_benchmark_snapshot = lambda now=None: {
        "minutes_since_open": 96,
        "crash_confirmed": False,
        "recovery_confirmed": True,
        "red_ratio": 0.67,
        "avg_pct_change": -0.16,
    }
    pos = SimpleNamespace(symbol="SQQQ")

    state = DayManager._refresh_entry_authority_state(dm, positions=[pos])

    assert state["state"] == "recovery_transition"
    assert state["reason"] == "initial_recovery_detected"
    assert state["recovery_confirmed_streak"] == 2


def test_entries_blocked_by_regime_uses_signal_lookup_during_safety_reentry():
    dm = _new_dm_stub()
    dm.signals = [{"ticker": "XOM", "sector": "energy", "score": 72.0}]
    dm.live_sector_bias = {"energy": 1.0}
    dm._refresh_entry_authority_state = lambda positions=None: {
        "state": "safety_reentry",
        "reason": "live_recovery_confirmed",
    }

    blocked, reason = dm._entries_blocked_by_regime("XOM")

    assert blocked is False
    assert reason == ""


def test_entries_blocked_by_regime_keeps_inverse_fast_candidates_in_quality_gate():
    dm = _new_dm_stub()
    dm.regime_router_context = {"regime": "neutral"}
    dm._refresh_entry_authority_state = lambda positions=None: {
        "state": "inverse_fast",
        "reason": "bearish_bias_inverse_fast_monitoring",
    }

    blocked, reason = dm._entries_blocked_by_regime("CNK")

    assert blocked is False
    assert reason == ""


def test_entries_blocked_by_regime_allows_inverse_fast_when_router_neutral():
    dm = _new_dm_stub()
    dm.regime_router_context = {"regime": "neutral"}
    dm._refresh_entry_authority_state = lambda positions=None: {
        "state": "inverse_fast",
        "reason": "crash_open_confirmed",
    }

    blocked, reason = dm._entries_blocked_by_regime("CNK")

    assert blocked is False
    assert reason == ""


def test_live_execution_mode_allows_bullish_entries_during_inverse_fast_when_router_neutral():
    dm = _new_dm_stub()
    dm.regime_router_context = {"regime": "neutral"}
    dm._core_data_readiness = {
        "is_fresh": True,
        "pm_ready_for_execution": True,
        "primary_date": "2026-03-19",
    }
    dm._entry_authority_state = {
        "session_date": datetime.now().strftime("%Y-%m-%d"),
        "state": "inverse_fast",
        "reason": "crash_open_confirmed",
        "updated_at": None,
        "snapshot": {},
    }
    dm._refresh_entry_authority_state = lambda positions=None, now_local=None: (
        dm._entry_authority_state
    )

    contract = DayManager._live_execution_mode(dm)

    assert contract["entries_allowed"] is True
    assert contract["inverse_entries_allowed"] is True
    assert contract["inverse_fast_bullish_override"] is True


def test_resolve_entry_authority_blocks_weak_symbol_during_inverse_fast_override():
    dm = _new_dm_stub()
    dm.regime_router_context = {"regime": "neutral"}
    dm._core_data_readiness = {
        "is_fresh": True,
        "pm_ready_for_execution": True,
        "primary_date": "2026-03-19",
    }
    dm._entry_authority_state = {
        "session_date": datetime.now().strftime("%Y-%m-%d"),
        "state": "inverse_fast",
        "reason": "crash_open_confirmed",
        "updated_at": None,
        "snapshot": {},
    }
    dm._refresh_entry_authority_state = lambda positions=None, now_local=None: (
        dm._entry_authority_state
    )
    dm.get_current_price = lambda ticker: 10.15 if ticker == "CNK" else 10.1
    dm._check_intraday_momentum = lambda ticker, avg_volume=0.0: {
        "pass": False,
        "reason": "below_vwap",
        "volume_ratio": 0.7,
    }

    contract = DayManager._resolve_entry_authority(
        dm,
        {
            "ticker": "CNK",
            "entry_source": "overnight_plan",
            "plan_score_source": "signals_2026-03-19.json",
            "score": 78.0,
            "entry_price": 10.0,
            "risk_reward": 1.8,
            "volume_ratio": 1.2,
            "short_momentum": -0.3,
            "day_return": -0.8,
            "open_to_now_return_pct": -0.5,
        },
    )

    assert contract["eligible"] is False
    assert contract["reason"] == "inverse_fast_symbol_relative_strength_weak"


def test_resolve_entry_authority_allows_strong_symbol_during_inverse_fast_override():
    dm = _new_dm_stub()
    dm.regime_router_context = {"regime": "neutral"}
    dm._core_data_readiness = {
        "is_fresh": True,
        "pm_ready_for_execution": True,
        "primary_date": "2026-03-19",
    }
    dm._entry_authority_state = {
        "session_date": datetime.now().strftime("%Y-%m-%d"),
        "state": "inverse_fast",
        "reason": "crash_open_confirmed",
        "updated_at": None,
        "snapshot": {},
    }
    dm._refresh_entry_authority_state = lambda positions=None, now_local=None: (
        dm._entry_authority_state
    )
    dm.get_current_price = lambda ticker: 10.15 if ticker == "CNK" else 10.15
    dm._check_intraday_momentum = lambda ticker, avg_volume=0.0: {
        "pass": True,
        "reason": "momentum_confirmed",
        "volume_ratio": 1.4,
    }

    contract = DayManager._resolve_entry_authority(
        dm,
        {
            "ticker": "CNK",
            "entry_source": "overnight_plan",
            "plan_score_source": "signals_2026-03-19.json",
            "score": 78.0,
            "entry_price": 10.0,
            "risk_reward": 1.8,
            "volume_ratio": 1.2,
            "short_momentum": 0.4,
            "day_return": 1.2,
            "support_dist_atr": 0.8,
            "resistance_dist_atr": 1.5,
            "setup_type": "new_high_breakout",
        },
    )

    assert contract["eligible"] is True
    assert contract["reason"] == ""


def test_resolve_entry_authority_blocks_non_plan_watchlist_entry_while_plan_candidates_pending():
    dm = _new_dm_stub()
    dm._entry_authority_state["state"] = "open"
    dm._refresh_entry_authority_state = lambda positions=None, now_local=None: (
        dm._entry_authority_state
    )
    dm._active_plan_path = Path("plans/pm_plan_2026-03-19.json")
    dm.signals = [
        {
            "ticker": "AAA",
            "symbol": "AAA",
            "entry_source": "overnight_plan",
            "plan_score_source": "pm_plan_2026-03-19.json",
            "score": 75.0,
        }
    ]
    dm.signal_status = {"AAA": {"status": "pending", "reason": ""}}
    dm.get_positions = lambda: []

    contract = DayManager._resolve_entry_authority(
        dm,
        {
            "ticker": "CNK",
            "symbol": "CNK",
            "entry_source": "watchlist_batch_rotate",
            "score": 79.0,
            "risk_reward": 1.8,
            "volume_ratio": 1.7,
        },
    )

    assert contract["eligible"] is False
    assert contract["reason"].startswith(
        "strict_plan_authority_symbol_not_in_active_plan:"
    )


def test_resolve_entry_authority_blocks_momentum_scanner_while_plan_candidates_pending():
    dm = _new_dm_stub()
    dm._entry_authority_state["state"] = "open"
    dm._refresh_entry_authority_state = lambda positions=None, now_local=None: (
        dm._entry_authority_state
    )
    dm._active_plan_path = Path("plans/pm_plan_2026-03-19.json")
    dm.signals = [
        {
            "ticker": "AAA",
            "symbol": "AAA",
            "entry_source": "overnight_plan",
            "plan_score_source": "pm_plan_2026-03-19.json",
            "score": 75.0,
        }
    ]
    dm.signal_status = {"AAA": {"status": "pending", "reason": ""}}
    dm.get_positions = lambda: []

    contract = DayManager._resolve_entry_authority(
        dm,
        {
            "ticker": "CNK",
            "symbol": "CNK",
            "entry_source": "momentum_scanner",
            "score": 79.0,
            "risk_reward": 1.8,
            "volume_ratio": 1.7,
        },
    )

    assert contract["eligible"] is False
    assert contract["reason"].startswith(
        "strict_plan_authority_symbol_not_in_active_plan:"
    )


def test_can_enter_positions_allows_early_runner_lane_during_wind_down():
    dm = _new_dm_stub()
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.WIND_DOWN
    dm._check_execution_circuit_breaker = lambda: False

    allowed, reason = DayManager.can_enter_positions(dm, entry_wave="early_runner")

    assert allowed is True
    assert reason.startswith("EARLY RUNNER: Late-day lane active")


def test_late_session_entry_block_reason_blocks_last_30_minutes(monkeypatch):
    class _LateWindowDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 22, 14, 41, 0, tzinfo=tz)

    dm = _new_dm_stub()
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.WIND_DOWN
    monkeypatch.setattr(day_manager_mod, "datetime", _LateWindowDateTime)

    reason = DayManager._late_session_entry_block_reason(
        dm,
        candidate_data={"score": 88.0},
    )

    assert reason == "late_session_no_runway:19m<30m"


def test_late_session_entry_block_reason_exempts_wave_entries(monkeypatch):
    class _LateWindowDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 22, 14, 41, 0, tzinfo=tz)

    dm = _new_dm_stub()
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.WIND_DOWN
    monkeypatch.setattr(day_manager_mod, "datetime", _LateWindowDateTime)

    reason = DayManager._late_session_entry_block_reason(
        dm,
        entry_wave=2,
        candidate_data={"score": 88.0},
    )

    assert reason == ""


def test_resolve_entry_authority_allows_validated_strength_reentry_when_plan_pending():
    dm = _new_dm_stub()
    dm._entry_authority_state["state"] = "open"
    dm._refresh_entry_authority_state = lambda positions=None, now_local=None: (
        dm._entry_authority_state
    )
    dm._active_plan_path = Path("plans/pm_plan_2026-03-19.json")
    dm.signals = [
        {
            "ticker": "AAA",
            "symbol": "AAA",
            "entry_source": "overnight_plan",
            "plan_score_source": "pm_plan_2026-03-19.json",
            "score": 75.0,
        }
    ]
    dm.signal_status = {"AAA": {"status": "pending", "reason": ""}}
    dm.get_positions = lambda: []
    dm._strength_reentry_candidate_allowed = lambda signal_data: True

    contract = DayManager._resolve_entry_authority(
        dm,
        {
            "ticker": "CNK",
            "symbol": "CNK",
            "entry_source": "strength_reentry",
            "score": 81.0,
            "risk_reward": 1.9,
            "volume_ratio": 1.6,
        },
    )

    assert contract["eligible"] is True
    assert contract["reason"] == ""


def test_strict_plan_authority_keeps_active_plan_symbols_authoritative_after_pending_timeout():
    dm = _new_dm_stub()
    now_utc = datetime(2026, 3, 19, 15, 0, tzinfo=timezone.utc)
    dm._now_utc = lambda: now_utc
    dm.entry_quality_cfg.strict_plan_authority_enabled = True
    dm.entry_quality_cfg.strict_plan_authority_pending_timeout_minutes = 30
    dm.signals = [
        {
            "ticker": "DECK",
            "symbol": "DECK",
            "entry_source": "overnight_plan",
            "plan_score_source": "pm_plan_2026-03-19.json",
            "score": 75.0,
        }
    ]
    dm.signal_status = {
        "DECK": {
            "status": "pending",
            "reason": "",
            "pending_since": now_utc - timedelta(minutes=45),
        }
    }
    dm.get_positions = lambda: []

    result = DayManager._strict_plan_authority_block_reason(
        dm, {"symbol": "HUT", "entry_source": "momentum_scanner"}
    )

    assert result.startswith("strict_plan_authority_symbol_not_in_active_plan:HUT:")


def test_strict_plan_authority_ignores_watch_only_pending_plan_candidates():
    dm = _new_dm_stub()
    dm.entry_quality_cfg.strict_plan_authority_enabled = True
    dm.signals = [
        {
            "ticker": "AAA",
            "symbol": "AAA",
            "entry_source": "overnight_plan",
            "plan_score_source": "pm_plan_2026-03-19.json",
            "score": 61.0,
            "recommendation": "WATCH",
            "action": "watch",
        }
    ]
    dm.signal_status = {"AAA": {"status": "pending", "reason": ""}}
    dm.get_positions = lambda: []

    result = DayManager._strict_plan_authority_block_reason(
        dm, {"symbol": "HUT", "entry_source": "watchlist_batch_rotate"}
    )

    assert result == ""


def test_resolve_entry_authority_blocks_intraday_momentum_when_plan_pending():
    dm = _new_dm_stub()
    dm.entry_quality_cfg.intraday_momentum_enabled = True
    dm.entry_quality_cfg.intraday_momentum_max_positions = 8
    dm._entry_authority_state["state"] = "open"
    dm._refresh_entry_authority_state = lambda positions=None, now_local=None: (
        dm._entry_authority_state
    )
    dm._active_plan_path = Path("plans/pm_plan_2026-03-19.json")
    dm.signals = [
        {
            "ticker": "AAA",
            "symbol": "AAA",
            "entry_source": "overnight_plan",
            "plan_score_source": "pm_plan_2026-03-19.json",
            "score": 75.0,
        }
    ]
    dm.signal_status = {"AAA": {"status": "pending", "reason": ""}}
    dm.get_positions = lambda: []

    contract = DayManager._resolve_entry_authority(
        dm,
        {
            "ticker": "CNK",
            "symbol": "CNK",
            "entry_source": "intraday_momentum",
            "slot_class": "momentum_reserve",
            "score": 79.0,
            "risk_reward": 1.8,
            "volume_ratio": 1.7,
        },
    )

    assert contract["eligible"] is False
    assert contract["reason"].startswith(
        "strict_plan_authority_symbol_not_in_active_plan:CNK:"
    )


def test_strict_plan_authority_blocks_non_plan_buy_when_only_pm_plan_loaded():
    dm = _new_dm_stub()
    dm.entry_quality_cfg.strict_plan_authority_enabled = True
    dm._active_plan_path = Path("plans/pm_plan_2026-04-28.json")
    dm.signals = [
        {
            "ticker": "CERT",
            "symbol": "CERT",
            "entry_source": "overnight_plan",
            "plan_score_source": "pm_plan_2026-04-28.json",
            "score": 68.0,
            "action": "buy_open",
        }
    ]
    dm.signal_status = {"CERT": {"status": "executed", "reason": ""}}
    dm.get_positions = lambda: []

    result = DayManager._strict_plan_authority_block_reason(
        dm,
        {
            "ticker": "BMNR",
            "symbol": "BMNR",
            "entry_source": "momentum_scanner",
            "score": 82.0,
        },
    )

    assert result == "strict_plan_authority_symbol_not_in_active_plan:BMNR:1"


def test_can_enter_positions_enforces_intraday_momentum_subcap():
    dm = _new_dm_stub()
    dm.entry_quality_cfg.intraday_momentum_enabled = True
    dm.entry_quality_cfg.intraday_momentum_max_positions = 2
    dm._check_execution_circuit_breaker = lambda: False
    dm.get_positions = lambda: [
        SimpleNamespace(
            symbol="MOMO1", market_value=1000.0, entry_source="intraday_momentum"
        ),
        SimpleNamespace(
            symbol="MOMO2", market_value=1000.0, entry_source="intraday_momentum"
        ),
    ]

    allowed, reason = DayManager.can_enter_positions(
        dm,
        ticker="MOMO3",
        candidate_data={
            "ticker": "MOMO3",
            "entry_source": "intraday_momentum",
            "slot_class": "momentum_reserve",
        },
    )

    assert allowed is False
    assert reason == "intraday_momentum_max_positions_reached"


def test_resolve_entry_authority_allows_replay_safe_symbol_when_live_momentum_missing():
    dm = _new_dm_stub()
    dm.regime_router_context = {"regime": "neutral"}
    dm._core_data_readiness = {
        "is_fresh": True,
        "pm_ready_for_execution": True,
        "primary_date": "2026-03-19",
    }
    dm._entry_authority_state = {
        "session_date": datetime.now().strftime("%Y-%m-%d"),
        "state": "inverse_fast",
        "reason": "crash_open_confirmed",
        "updated_at": None,
        "snapshot": {},
    }
    dm._refresh_entry_authority_state = lambda positions=None, now_local=None: (
        dm._entry_authority_state
    )
    dm.get_current_price = lambda ticker: 10.15 if ticker == "CNK" else 10.15
    dm._check_intraday_momentum = lambda ticker, avg_volume=0.0: {
        "pass": False,
        "reason": "insufficient_data",
        "volume_ratio": 0.0,
    }

    contract = DayManager._resolve_entry_authority(
        dm,
        {
            "ticker": "CNK",
            "entry_source": "overnight_plan",
            "plan_score_source": "signals_2026-03-19.json",
            "score": 81.0,
            "entry_price": 10.0,
            "risk_reward": 1.75,
            "volume_ratio": 1.35,
            "support_dist_atr": 0.9,
            "resistance_dist_atr": 1.4,
            "setup_type": "new_high_breakout",
        },
    )

    assert contract["eligible"] is True
    assert contract["reason"] == ""


def test_buy_order_guard_blocks_daily_cap_and_buying_power_shortfall():
    dm = _new_dm_stub()
    dm.client = SimpleNamespace(
        get_account=lambda: SimpleNamespace(cash="50000", buying_power="2000")
    )
    dm.entry_quality_cfg.max_buy_orders_per_day = 30
    dm._buy_order_counter_date = datetime.now().strftime("%Y-%m-%d")
    dm._buy_orders_submitted_today = 30

    cap_reason = DayManager._buy_order_guard_block_reason(
        dm,
        symbol="CNK",
        qty=10,
        reference_price=10.0,
        context="unit_test",
    )
    assert cap_reason.startswith("daily_buy_order_cap_reached:")

    dm._buy_orders_submitted_today = 0
    bp_reason = DayManager._buy_order_guard_block_reason(
        dm,
        symbol="CNK",
        qty=500,
        reference_price=10.0,
        context="unit_test",
    )
    assert bp_reason.startswith("insufficient_buying_power:")


def test_buy_order_guard_allows_margin_buying_power_when_cash_is_below_floor():
    dm = _new_dm_stub()
    dm.client = SimpleNamespace(
        get_account=lambda: SimpleNamespace(cash="500", buying_power="100000")
    )
    dm._buy_order_counter_date = datetime.now().strftime("%Y-%m-%d")
    dm._buy_orders_submitted_today = 0
    dm.entry_quality_cfg.buy_order_min_cash_floor = 1000.0
    dm.entry_quality_cfg.enforce_buying_power_gate = True

    reason = DayManager._buy_order_guard_block_reason(
        dm,
        symbol="CNK",
        qty=100,
        reference_price=10.0,
        context="unit_test",
    )

    assert reason == ""


def test_terminal_runtime_entry_block_is_quarantined_from_watchdog():
    dm = _new_dm_stub()
    dm._watched_universe_tickers = {"SFD", "WRBY"}
    dm.signal_status = {
        "SFD": {"status": "pending", "reason": ""},
        "WRBY": {"status": "pending", "reason": ""},
    }

    dm._record_runtime_entry_block(
        "SFD",
        "entry_gap_too_large",
        planned_price=25.83,
        current_price=29.28,
    )

    selected = DayManager._top_watchlist_candidates_for_watchdog(
        dm,
        [
            {"ticker": "SFD", "realtime_score": 100.0},
            {"ticker": "WRBY", "realtime_score": 95.0},
        ],
    )

    assert selected == ["WRBY"]
    assert dm.signal_status["SFD"]["status"] == "skipped"
    assert dm.signal_status["SFD"]["reason"] == "entry_gap_too_large"


def test_realtime_score_uses_smaller_penalty_for_missing_momentum_data(monkeypatch):
    dm = _new_dm_stub()
    dm.entry_quality_cfg.momentum_gate_enabled = True
    dm.entry_quality_cfg.momentum_penalty_points = 15.0
    dm.entry_quality_cfg.momentum_insufficient_data_penalty_points = 5.0
    dm.entry_quality_cfg.stale_signal_max_age_days = 1
    dm.entry_quality_cfg.stale_signal_penalty_points = 10.0
    dm.youtube_context = {}
    dm.live_sector_bias = {}
    dm.get_current_price = lambda ticker: 10.0
    dm._premarket_handoff_adjustment = lambda ticker: 0.0
    dm._evaluate_intraday_strategy_profile = lambda ticker, signal_data, bars_df: {}

    monkeypatch.setattr(
        "autotrade.utils.intraday_data_provider.get_intraday_bars",
        lambda *args, **kwargs: None,
    )

    signal = {
        "ticker": "MISS",
        "score": 50.0,
        "rsi": 50,
        "weekly_return": 0.0,
        "sentiment_score": 0.0,
        "volume": 100000,
    }

    score = DayManager._calculate_realtime_score(dm, signal)

    assert score == pytest.approx(15.0)
    assert signal["momentum_gate_reason"] == "insufficient_data"
    assert signal["momentum_gate_volume_ratio"] == 0.0


def test_entries_blocked_by_core_data_enforces_preopen_execution_contract():
    dm = _new_dm_stub()
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.OBSERVATION
    dm._active_plan_path = None
    dm.signals = []

    blocked, reason = DayManager._entries_blocked_by_core_data(dm)

    assert blocked is True
    assert reason == "execution_contract_missing_active_plan"


def test_resolve_entry_authority_blocks_symbol_with_tight_resistance_during_inverse_fast_override():
    dm = _new_dm_stub()
    dm.regime_router_context = {"regime": "neutral"}
    dm._core_data_readiness = {
        "is_fresh": True,
        "pm_ready_for_execution": True,
        "primary_date": "2026-03-19",
    }
    dm._entry_authority_state = {
        "session_date": datetime.now().strftime("%Y-%m-%d"),
        "state": "inverse_fast",
        "reason": "crash_open_confirmed",
        "updated_at": None,
        "snapshot": {},
    }
    dm._refresh_entry_authority_state = lambda positions=None, now_local=None: (
        dm._entry_authority_state
    )
    dm.get_current_price = lambda ticker: 10.15 if ticker == "CNK" else 10.15
    dm._check_intraday_momentum = lambda ticker, avg_volume=0.0: {
        "pass": False,
        "reason": "insufficient_data",
        "volume_ratio": 0.0,
    }

    contract = DayManager._resolve_entry_authority(
        dm,
        {
            "ticker": "CNK",
            "entry_source": "overnight_plan",
            "plan_score_source": "signals_2026-03-19.json",
            "score": 81.0,
            "entry_price": 10.0,
            "risk_reward": 1.75,
            "volume_ratio": 1.35,
            "support_dist_atr": 0.8,
            "resistance_dist_atr": 0.5,
            "setup_type": "new_high_breakout",
        },
    )

    assert contract["eligible"] is False
    assert contract["reason"] == "inverse_fast_symbol_resistance_tight<0.90atr"


def test_effective_market_regime_prefers_live_detector_over_resolved_regime():
    dm = _new_dm_stub()
    dm.youtube_context = {"resolved_regime": {"regime": "SELL_OFF"}}
    dm.current_regime = "melt_up"
    dm._effective_market_regime = DayManager._effective_market_regime.__get__(
        dm, DayManager
    )

    assert dm._effective_market_regime() == "MELT_UP"


def test_source_aware_policy_prefers_watchlist_candidates_over_unvalidated_discovery():
    dm = _new_dm_stub()
    dm._watched_universe_tickers = {"CNK"}

    ranked = dm._apply_source_aware_candidate_policy(
        [
            {
                "ticker": "DISC",
                "score": 68.0,
                "realtime_score": 68.0,
                "entry_source": "vwap_universe_scanner",
                "risk_reward": 1.8,
                "volume_ratio": 1.4,
            },
            {
                "ticker": "CNK",
                "score": 64.0,
                "realtime_score": 64.0,
                "entry_source": "overnight_plan",
                "risk_reward": 1.5,
            },
        ],
        phase="research",
    )

    assert [row["ticker"] for row in ranked] == ["CNK"]
    assert dm._candidate_validation_rejections[0]["ticker"] == "DISC"


def test_source_aware_policy_allows_validated_off_watchlist_discovery():
    dm = _new_dm_stub()

    ranked = dm._apply_source_aware_candidate_policy(
        [
            {
                "ticker": "DISC",
                "score": 74.0,
                "realtime_score": 74.0,
                "entry_source": "intraday_reserve_scan",
                "risk_reward": 2.1,
                "volume_ratio": 1.8,
                "backtest_score": 62.0,
                "similar_signals_found": 14,
                "day_return": 1.3,
            }
        ],
        phase="core_trading",
    )

    assert len(ranked) == 1
    assert ranked[0]["ticker"] == "DISC"
    assert ranked[0]["entry_validation"]["allowed"] is True
    assert ranked[0]["entry_validation"]["source_bucket"] == "intraday_discovery"


def test_source_aware_policy_treats_momentum_scanner_as_watchlist():
    dm = _new_dm_stub()

    ranked = dm._apply_source_aware_candidate_policy(
        [
            {
                "ticker": "LMND",
                "score": 76.0,
                "realtime_score": 76.0,
                "entry_source": "momentum_scanner",
                "intraday_reserve": True,
                "risk_reward": 1.8,
                "volume_ratio": 2.0,
                "trend_strength": 0.7,
                "short_momentum": 0.8,
                "pullback_pct": 1.2,
                "momentum_gate_pass": True,
            }
        ],
        phase="core_trading",
    )

    assert len(ranked) == 1
    assert ranked[0]["entry_validation"]["source_bucket"] == "watchlist"
    assert "watchlist_source_bonus" in ranked[0]["entry_validation"]["reasons"]


def test_source_aware_policy_blocks_weak_momentum_scanner_candidate():
    dm = _new_dm_stub()

    ranked = dm._apply_source_aware_candidate_policy(
        [
            {
                "ticker": "LMND",
                "score": 74.0,
                "realtime_score": 74.0,
                "entry_source": "momentum_scanner",
                "intraday_reserve": True,
                "risk_reward": 1.8,
                "volume_ratio": 1.4,
                "trend_strength": 0.54,
                "short_momentum": 0.3,
                "pullback_pct": 3.1,
                "momentum_gate_pass": False,
            }
        ],
        phase="core_trading",
    )

    assert ranked == []
    assert dm._candidate_validation_rejections[0]["ticker"] == "LMND"
    assert dm._candidate_validation_rejections[0]["reason"].startswith(
        "momentum_scanner_"
    )


def test_candidate_entry_score_threshold_relaxes_for_high_conviction_overnight_watchlist():
    dm = _new_dm_stub()

    adjusted = dm._candidate_entry_score_threshold(
        {
            "ticker": "BTSG",
            "final_score": 75.25,
            "confidence": 92,
            "entry_source": "overnight_plan",
            "source_bucket": "watchlist",
            "plan_score_source": "morning_game_plan_20260403.json",
        },
        phase=day_manager_mod.TradingPhase.RESEARCH,
        base_threshold=50.0,
    )

    assert adjusted == 40.0


def test_candidate_entry_score_threshold_keeps_default_for_non_plan_candidates():
    dm = _new_dm_stub()

    adjusted = dm._candidate_entry_score_threshold(
        {
            "ticker": "DISC",
            "score": 88.0,
            "entry_source": "vwap_universe_scanner",
            "source_bucket": "intraday_discovery",
        },
        phase=day_manager_mod.TradingPhase.RESEARCH,
        base_threshold=50.0,
    )

    assert adjusted == 50.0


def test_build_candidate_validation_report_preserves_some_overnight_plan_conviction_in_research():
    dm = _new_dm_stub()
    dm._watched_universe_tickers = {"BTSG"}

    report = dm._build_candidate_validation_report(
        {
            "ticker": "BTSG",
            "realtime_score": 40.0,
            "final_score": 80.0,
            "entry_source": "overnight_plan",
            "source_bucket": "watchlist",
            "plan_score_source": "pm_plan_2026-04-02.json",
        },
        phase=day_manager_mod.TradingPhase.RESEARCH,
    )

    assert report["base_score"] == 40.0
    assert report["normalized_score"] == 80.0
    assert report["validation_carry_bonus"] == pytest.approx(9.0)
    assert report["adjusted_score"] == pytest.approx(53.0)
    assert "overnight_plan_validation_carry" in report["reasons"]


def test_source_aware_policy_blocks_unknown_source_75_to_85_score_band():
    dm = _new_dm_stub()

    ranked = dm._apply_source_aware_candidate_policy(
        [
            {
                "ticker": "BLEED",
                "score": 80.0,
                "realtime_score": 80.0,
                "risk_reward": 2.4,
                "volume_ratio": 2.0,
                "backtest_score": 72.0,
                "similar_signals_found": 18,
                "day_return": 1.5,
            }
        ],
        phase="core_trading",
    )

    assert ranked == []
    rejection = dm._candidate_validation_rejections[0]
    assert rejection["ticker"] == "BLEED"
    assert rejection["reason"].startswith("unknown_source_score_band")


def test_source_aware_policy_inferrs_plan_source_before_unknown_band_gate():
    dm = _new_dm_stub()

    report = dm._build_candidate_validation_report(
        {
            "ticker": "PLAN",
            "score": 80.0,
            "realtime_score": 55.0,
            "risk_reward": 1.6,
            "plan_score_source": "morning_game_plan_20260402.json",
        },
        phase=day_manager_mod.TradingPhase.RESEARCH,
    )

    assert report["allowed"] is True
    assert report["entry_source"] == "overnight_plan"
    assert report["source_bucket"] == "watchlist"
    assert not any(
        str(reason).startswith("unknown_source_score_band")
        for reason in report["reasons"]
    )


def test_overnight_full_watchlist_edge_boost_targets_60_to_75_band():
    dm = _new_dm_stub()
    dm._watched_universe_tickers = {"EDGE"}

    report = dm._build_candidate_validation_report(
        {
            "ticker": "EDGE",
            "score": 68.0,
            "realtime_score": 50.0,
            "entry_source": "overnight_plan_full_watchlist",
            "source_bucket": "watchlist",
            "plan_score_source": "morning_game_plan_20260402.json",
        },
        phase=day_manager_mod.TradingPhase.CORE_TRADING,
    )

    assert report["allowed"] is True
    assert report["adjusted_score"] == pytest.approx(60.0)
    assert "overnight_full_watchlist_edge_boost" in report["reasons"]


def test_high_score_watchlist_candidate_without_catalyst_needs_confirmation():
    dm = _new_dm_stub()
    dm._watched_universe_tickers = {"NOCA"}

    ranked = dm._apply_source_aware_candidate_policy(
        [
            {
                "ticker": "NOCA",
                "score": 78.0,
                "realtime_score": 78.0,
                "entry_source": "watchlist_batch_rotate",
                "risk_reward": 1.8,
                "volume_ratio": 1.7,
            }
        ],
        phase="core_trading",
    )

    assert ranked == []
    assert dm._candidate_validation_rejections[0]["reason"] == (
        "high_score_no_catalyst_confirmation_missing"
    )


def test_high_score_watchlist_candidate_with_catalyst_passes_alpha_gate():
    dm = _new_dm_stub()
    dm._watched_universe_tickers = {"CAT"}

    ranked = dm._apply_source_aware_candidate_policy(
        [
            {
                "ticker": "CAT",
                "score": 78.0,
                "realtime_score": 78.0,
                "entry_source": "watchlist_batch_rotate",
                "risk_reward": 1.8,
                "volume_ratio": 1.7,
                "has_catalyst": True,
                "catalyst_note": "earnings beat",
            }
        ],
        phase="core_trading",
    )

    assert len(ranked) == 1
    assert ranked[0]["ticker"] == "CAT"


def test_load_signals_prefers_pm_plan_over_morning_game_plan(tmp_path, monkeypatch):
    dm = _new_dm_stub()
    dm._load_plan_payload = lambda path: {"loaded_from": path.name}
    dm._update_core_data_readiness_from_plan = lambda payload: None
    dm._parse_signal_file = lambda path: [{"ticker": path.name}]

    fake_module = tmp_path / "autotrade" / "core"
    fake_module.mkdir(parents=True)
    fake_file = fake_module / "day_manager.py"
    fake_file.write_text("# test stub\n", encoding="utf-8")
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    (plans_dir / "pm_plan_2026-04-07.json").write_text("{}", encoding="utf-8")
    (plans_dir / "morning_game_plan_20260407.json").write_text(
        "{}",
        encoding="utf-8",
    )

    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 7, 8, 55, 0, tzinfo=tz)

    monkeypatch.setattr(day_manager_mod, "__file__", str(fake_file))
    monkeypatch.setattr(day_manager_mod, "datetime", _FixedDateTime)

    signals = dm._load_signals()

    # H4 (2026-05-19): _load_signals tags PM rows with plan_source="pm" for
    # downstream provenance — assert the field is set, then strip for the
    # ticker-shape check that this test originally pinned.
    assert len(signals) == 1
    assert signals[0].get("plan_source") == "pm"
    assert signals[0].get("ticker") == "pm_plan_2026-04-07.json"
    assert dm._active_plan_path == Path(plans_dir / "pm_plan_2026-04-07.json")


def test_load_signals_prefers_actionable_morning_plan_over_watch_only_pm_plan(
    tmp_path, monkeypatch
):
    dm = _new_dm_stub()
    dm._update_core_data_readiness_from_plan = lambda payload: None
    dm._parse_signal_file = lambda path: [{"ticker": path.name}]

    fake_module = tmp_path / "autotrade" / "core"
    fake_module.mkdir(parents=True)
    fake_file = fake_module / "day_manager.py"
    fake_file.write_text("# test stub\n", encoding="utf-8")
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    (plans_dir / "pm_plan_2026-04-03.json").write_text("{}", encoding="utf-8")
    (plans_dir / "morning_game_plan_20260403.json").write_text(
        "{}",
        encoding="utf-8",
    )

    def _load_plan_payload(path):
        if path.name.startswith("pm_plan_"):
            return {
                "signals": [
                    {
                        "symbol": "PM1",
                        "entry_price": 12.0,
                        "score": 80.0,
                        "recommendation": "WATCH",
                    }
                ]
            }
        return {
            "signals": [
                {
                    "symbol": "MORN",
                    "entry_price": 10.0,
                    "score": 60.0,
                    "recommendation": "BUY",
                }
            ]
        }

    dm._load_plan_payload = _load_plan_payload

    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 3, 8, 55, 0, tzinfo=tz)

    monkeypatch.setattr(day_manager_mod, "__file__", str(fake_file))
    monkeypatch.setattr(day_manager_mod, "datetime", _FixedDateTime)

    signals = dm._load_signals()

    assert signals == [{"ticker": "morning_game_plan_20260403.json"}]
    assert dm._active_plan_path == Path(plans_dir / "morning_game_plan_20260403.json")


def test_parse_signal_file_defaults_pm_plan_rows_to_overnight_plan(tmp_path):
    dm = DayManager.__new__(DayManager)
    dm.signal_generation_cfg = SimpleNamespace(day_manager_pipeline_enabled=False)

    signal_path = tmp_path / "pm_plan_2026-04-03.json"
    signal_path.write_text(
        '{"signals":[{"symbol":"AAA","final_score":81.0,"recommendation":"WATCH"}]}',
        encoding="utf-8",
    )

    rows = dm._parse_signal_file(signal_path)

    assert rows[0]["entry_source"] == "overnight_plan"
    assert rows[0]["source_bucket"] == "watchlist"


def test_watchlist_causality_path_honors_session_date_override():
    dm = _new_dm_stub()
    dm._artifact_session_date_override = "2026-04-02"

    path = dm._watchlist_causality_path()
    payload = dm._build_watchlist_causality_snapshot(
        phase=day_manager_mod.TradingPhase.RESEARCH
    )

    assert path.name == "watchlist_causality_2026-04-02.json"
    assert payload["date"] == "2026-04-02"


def test_watchlist_causality_snapshot_emits_symbol_alias():
    dm = _new_dm_stub()
    dm._watched_universe_tickers = {"BTSG"}
    dm._watchlist_causality["BTSG"] = {
        "ticker": "BTSG",
        "watchlist_member": True,
        "was_evaluated": True,
        "executed": False,
        "current_status": "blocked",
        "blocking_reason": "entry_gap_too_large",
        "source_bucket": "watchlist",
    }

    payload = dm._build_watchlist_causality_snapshot(
        phase=day_manager_mod.TradingPhase.CORE_TRADING
    )

    assert payload["symbols"][0]["symbol"] == "BTSG"
    assert payload["missed_opportunities"][0]["symbol"] == "BTSG"


def test_update_watchlist_causality_preserves_nonempty_source_metadata():
    dm = _new_dm_stub()
    dm._watched_universe_tickers = {"BTSG"}

    dm._update_watchlist_causality(
        "BTSG",
        state="ranked",
        phase=day_manager_mod.TradingPhase.RESEARCH,
        candidate={
            "ticker": "BTSG",
            "entry_source": "overnight_plan",
            "source_bucket": "watchlist",
            "plan_score_source": "pm_plan_2026-04-02.json",
            "score": 82.0,
        },
    )
    dm._update_watchlist_causality(
        "BTSG",
        state="blocked",
        phase=day_manager_mod.TradingPhase.CORE_TRADING,
        candidate={
            "ticker": "BTSG",
            "score": 100.0,
        },
        reason="entry_gap_too_large",
        blocking_rule="entry_gap_too_large",
    )

    row = dm._watchlist_causality["BTSG"]
    assert row["symbol"] == "BTSG"
    assert row["entry_source"] == "overnight_plan"
    assert row["source_bucket"] == "watchlist"
    assert row["plan_score_source"] == "pm_plan_2026-04-02.json"


def test_update_watchlist_causality_recomputes_threshold_gap_from_score_and_threshold():
    dm = _new_dm_stub()
    dm._watched_universe_tickers = {"CENX"}

    dm._update_watchlist_causality(
        "CENX",
        state="blocked",
        phase=day_manager_mod.TradingPhase.CORE_TRADING,
        candidate={
            "ticker": "CENX",
            "entry_source": "overnight_plan",
            "source_bucket": "watchlist",
            "plan_score_source": "morning_game_plan_20260402.json",
            "current_score": 100.0,
            "entry_threshold": 69.0,
            "score_gap_to_threshold": -69.0,
        },
        reason="entry_gap_too_large",
        blocking_rule="entry_gap_too_large",
    )

    row = dm._watchlist_causality["CENX"]
    assert row["current_score"] == 100.0
    assert row["entry_threshold"] == 69.0
    assert row["score_gap_to_threshold"] == pytest.approx(31.0)


def test_mark_signal_skipped_derives_threshold_context_when_missing():
    dm = _new_dm_stub()
    dm._watched_universe_tickers = {"FGI"}
    dm.signal_status = {"FGI": {"status": "new", "reason": ""}}

    dm._mark_signal_skipped(
        "FGI",
        "watchlist_invalid:vwap_violation_3x5m,orb_breakdown_15m_low",
        candidate={
            "ticker": "FGI",
            "entry_source": "overnight_plan",
            "source_bucket": "watchlist",
            "plan_score_source": "morning_game_plan_20260402.json",
            "current_score": 100.0,
        },
        phase=day_manager_mod.TradingPhase.RESEARCH,
    )

    row = dm._watchlist_causality["FGI"]
    assert row["entry_threshold"] is not None
    assert row["entry_threshold_base"] is not None
    assert row["current_score"] <= 100.0
    assert row["score_gap_to_threshold"] == pytest.approx(
        row["current_score"] - row["entry_threshold"]
    )


def test_mark_signal_skipped_uses_validation_scaled_current_score_for_artifacts():
    dm = _new_dm_stub()
    dm._watched_universe_tickers = {"FGI"}
    dm.signal_status = {"FGI": {"status": "pending", "reason": ""}}

    dm._mark_signal_skipped(
        "FGI",
        "watchlist_invalid:vwap_violation_3x5m,orb_breakdown_15m_low",
        candidate={
            "ticker": "FGI",
            "entry_source": "overnight_plan",
            "source_bucket": "watchlist",
            "plan_score_source": "morning_game_plan_20260402.json",
            "score": 738.4,
            "realtime_score": 738.4,
            "current_score": 738.4,
        },
        phase=day_manager_mod.TradingPhase.RESEARCH,
    )

    row = dm._watchlist_causality["FGI"]
    assert row["current_score"] <= 100.0
    assert row["entry_threshold"] is not None
    assert row["score_gap_to_threshold"] == pytest.approx(
        row["current_score"] - row["entry_threshold"]
    )


def test_source_aware_policy_limits_discovery_queue_to_top_two():
    dm = _new_dm_stub()

    ranked = dm._apply_source_aware_candidate_policy(
        [
            {
                "ticker": "D1",
                "score": 71.0,
                "realtime_score": 71.0,
                "entry_source": "intraday_reserve_scan",
                "risk_reward": 2.1,
                "volume_ratio": 2.5,
                "backtest_score": 65.0,
                "similar_signals_found": 18,
            },
            {
                "ticker": "D2",
                "score": 73.0,
                "realtime_score": 73.0,
                "entry_source": "intraday_reserve_scan",
                "risk_reward": 2.0,
                "volume_ratio": 2.2,
                "backtest_score": 65.0,
                "similar_signals_found": 18,
            },
            {
                "ticker": "D3",
                "score": 72.0,
                "realtime_score": 72.0,
                "entry_source": "intraday_reserve_scan",
                "risk_reward": 2.0,
                "volume_ratio": 1.9,
                "backtest_score": 65.0,
                "similar_signals_found": 18,
            },
            {
                "ticker": "D4",
                "score": 74.0,
                "realtime_score": 74.0,
                "entry_source": "intraday_reserve_scan",
                "risk_reward": 2.0,
                "volume_ratio": 1.2,
                "backtest_score": 65.0,
                "similar_signals_found": 18,
            },
        ],
        phase="core_trading",
    )

    assert [row["ticker"] for row in ranked] == ["D1", "D2"]
    assert dm._candidate_validation_rejections[-1]["reason"] == "discovery_rank_limit"


def test_refresh_momentum_scanner_signals_adds_reserved_watchlist_candidate(
    monkeypatch,
):
    dm = _new_dm_stub()
    dm.signals = []
    dm.signal_status = {}
    dm.momentum_scanner_cfg = SimpleNamespace(enabled=True, artifact_path="ignored")

    monkeypatch.setattr(
        day_manager_mod,
        "load_momentum_watchlist",
        lambda path=None: {
            "loaded": True,
            "artifact_path": "ignored",
            "symbols": [
                {
                    "ticker": "LMND",
                    "score": 83.0,
                    "confidence": 83.0,
                    "entry_price": 66.0,
                    "stop_loss": 63.0,
                    "target": 72.0,
                    "risk_reward": 2.0,
                    "volume_ratio": 2.4,
                    "day_return": 7.5,
                    "short_momentum": 1.1,
                    "pullback_pct": 1.8,
                    "has_catalyst": True,
                    "catalyst_note": "News-led continuation",
                    "atr_14": 3.0,
                    "atr_percent": 4.5,
                }
            ],
        },
    )

    added = dm._refresh_momentum_scanner_signals()

    assert added == 1
    assert dm.signals[0]["ticker"] == "LMND"
    assert dm.signals[0]["entry_source"] == "momentum_scanner"
    assert dm.signals[0]["source_bucket"] == "watchlist"
    assert dm.signals[0]["intraday_reserve"] is True
    assert "LMND" in dm._watched_universe_tickers


def test_momentum_scanner_fallback_reason_detects_missing_or_empty_artifact():
    dm = _new_dm_stub()

    assert dm._momentum_scanner_fallback_reason({"reason": "missing"}) == (
        "fallback_missing_artifact"
    )
    assert (
        dm._momentum_scanner_fallback_reason(
            {
                "status": "ok",
                "reason": "",
                "candidate_count": 0,
                "age_seconds": 60.0 * 20.0,
                "stale": False,
            }
        )
        == "fallback_zero_candidates"
    )


def test_build_intraday_reserve_seed_symbols_uses_recent_trades_before_watchlist():
    dm = _new_dm_stub()
    now = datetime.now()
    dm.trade_journal = SimpleNamespace(
        trades=[
            {
                "symbol": "AAPL",
                "entry_status": "filled",
                "entry_time": (now - timedelta(days=2)).isoformat(),
            },
            {
                "symbol": "MSFT",
                "entry_status": "filled",
                "entry_time": (now - timedelta(days=4)).isoformat(),
            },
        ]
    )
    dm.signals = [{"ticker": "NVDA"}, {"ticker": "AAPL"}]
    dm._initial_watchlist_tickers = {"QQQ"}
    dm.signal_status = {"TSLA": {"status": "executed"}}

    seeds = dm._build_intraday_reserve_seed_symbols(["AMD"])

    assert seeds[:2] == ["MSFT", "AAPL"] or seeds[:2] == ["AAPL", "MSFT"]
    assert "NVDA" in seeds


def test_watchlist_causality_tracks_unselected_ranked_symbols():
    dm = _new_dm_stub()
    dm._watched_universe_tickers = {"CNK", "MARA"}
    dm.signal_status = {
        "CNK": {"status": "pending", "reason": ""},
        "MARA": {"status": "pending", "reason": ""},
        "DISC": {"status": "pending", "reason": ""},
    }

    ranked = dm._apply_source_aware_candidate_policy(
        [
            {
                "ticker": "DISC",
                "score": 90.0,
                "realtime_score": 90.0,
                "entry_source": "intraday_reserve_scan",
                "risk_reward": 2.0,
                "volume_ratio": 1.8,
                "backtest_score": 61.0,
                "similar_signals_found": 18,
            },
            {
                "ticker": "CNK",
                "score": 69.0,
                "realtime_score": 69.0,
                "entry_source": "overnight_plan",
                "risk_reward": 1.7,
            },
            {
                "ticker": "MARA",
                "score": 66.0,
                "realtime_score": 66.0,
                "entry_source": "overnight_full_watchlist",
                "risk_reward": 1.6,
            },
        ],
        phase="core_trading",
    )

    dm._record_unselected_watchlist_candidates(
        ranked,
        selected_tickers=["DISC"],
        phase="core_trading",
        score_key="realtime_score",
        score_threshold=35.0,
        max_new_entries=1,
    )

    assert dm._watchlist_causality["CNK"]["was_evaluated"] is True
    assert dm._watchlist_causality["CNK"]["blocking_reason"] == "not_ranked_high_enough"
    assert dm._watchlist_causality["CNK"]["displaced_by"] == "DISC"
    assert (
        dm._watchlist_causality["MARA"]["blocking_reason"] == "not_ranked_high_enough"
    )


def test_log_candidate_ranking_snapshot_persists_audit_meta(
    monkeypatch, tmp_path: Path
):
    dm = _new_dm_stub()
    dm._candidate_validation_rejections = []
    dm._persist_watchlist_causality_snapshot = lambda phase=None: {
        "missed_opportunities": []
    }
    monkeypatch.setattr(day_manager_mod, "LOG_DIR", tmp_path)

    dm._log_candidate_ranking_snapshot(
        [
            {
                "ticker": "CNK",
                "entry_source": "overnight_plan",
                "source_bucket": "watchlist",
                "raw_realtime_score": 71.0,
                "realtime_score": 74.0,
                "entry_validation": {
                    "risk_reward": 1.8,
                    "reasons": ["watchlist_source_bonus"],
                },
            }
        ],
        phase=day_manager_mod.TradingPhase.CORE_TRADING,
        score_key="realtime_score",
        selected_tickers=["CNK"],
        audit_meta={"open_slots": 6, "max_new_entries": 4, "score_threshold": 35.0},
    )

    path = (
        tmp_path
        / f"entry_candidate_rankings_{datetime.now().strftime('%Y-%m-%d')}.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload[-1]["audit_meta"]["open_slots"] == 6
    assert payload[-1]["audit_meta"]["max_new_entries"] == 4
    assert payload[-1]["audit_meta"]["score_threshold"] == 35.0


def test_entry_audit_records_skipped_candidate_reasons(monkeypatch, tmp_path: Path):
    dm = _new_dm_stub()
    monkeypatch.setattr(day_manager_mod, "LOG_DIR", tmp_path)
    dm.signal_status = {
        "CNK": {"status": "pending", "reason": ""},
        "SFD": {"status": "pending", "reason": ""},
    }
    candidates = [
        {"ticker": "CNK", "realtime_score": 54.0, "entry_source": "watchlist"},
        {"ticker": "SFD", "realtime_score": 32.0, "entry_source": "watchlist"},
    ]

    dm._begin_entry_audit(
        candidates=candidates,
        phase=day_manager_mod.TradingPhase.CORE_TRADING,
        open_slots=2,
        max_new_entries=1,
    )
    dm._mark_signal_skipped("SFD", "below_min_score")
    audit = dm._finalize_entry_audit(
        {
            "candidate_count": 2,
            "open_slots": 2,
            "max_new_entries": 1,
            "entries_submitted": 0,
            "selected_entry_symbols": [],
            "block_reason": "",
            "phase": "core_trading",
        }
    )

    assert audit["blocked_by_reason"]["below_min_score"] == 1
    sfd_row = next(row for row in audit["candidates"] if row["ticker"] == "SFD")
    assert sfd_row["status"] == "skipped"
    assert sfd_row["skip_reason"] == "below_min_score"
    cnk_row = next(row for row in audit["candidates"] if row["ticker"] == "CNK")
    assert cnk_row["status"] == "skipped"
    assert cnk_row["skip_reason"] == "candidate_not_submitted"
    assert "entry_submission_failed_no_reason" not in audit["blocked_by_reason"]
    path = tmp_path / f"entry_audit_{datetime.now().strftime('%Y-%m-%d')}.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved[-1]["candidate_count"] == 2


def test_mark_signal_skipped_records_threshold_diagnostics_for_watchlist_candidate():
    dm = _new_dm_stub()
    dm._watched_universe_tickers = {"BTSG"}
    dm.signal_status = {"BTSG": {"status": "pending", "reason": ""}}

    dm._mark_signal_skipped(
        "BTSG",
        "below_min_score",
        candidate={
            "ticker": "BTSG",
            "entry_source": "overnight_plan",
            "source_bucket": "watchlist",
            "plan_score_source": "pm_plan_2026-04-02.json",
            "normalized_score": 78.0,
            "raw_realtime_score": 38.0,
            "ranking_score": 47.0,
            "realtime_score": 47.0,
            "current_score": 47.0,
            "entry_threshold": 50.0,
            "entry_threshold_base": 50.0,
            "entry_threshold_source": "overnight_watchlist_research_relief",
            "entry_threshold_adjustment": 10.0,
            "score_gap_to_threshold": -3.0,
            "validation_carry_bonus": 5.0,
        },
        phase=day_manager_mod.TradingPhase.RESEARCH,
    )

    row = dm._watchlist_causality["BTSG"]
    assert row["blocking_reason"] == "below_min_score"
    assert row["plan_score_source"] == "pm_plan_2026-04-02.json"
    assert row["normalized_score"] == 78.0
    assert row["current_score"] > 47.0
    assert row["entry_threshold"] == 50.0
    assert row["score_gap_to_threshold"] == pytest.approx(
        row["current_score"] - row["entry_threshold"]
    )
    assert row["validation_carry_bonus"] == 5.0


def test_mark_signal_pending_clears_stale_blocking_reason():
    dm = _new_dm_stub()
    dm._watched_universe_tickers = {"CNK"}
    dm.signal_status = {"CNK": {"status": "skipped", "reason": "below_min_score"}}
    dm._watchlist_causality["CNK"] = {
        "ticker": "CNK",
        "watchlist_member": True,
        "was_evaluated": True,
        "ever_became_candidate": True,
        "selected_for_entry": False,
        "executed": False,
        "validation_passed": None,
        "current_status": "blocked",
        "blocking_reason": "below_min_score",
        "blocking_rule": "below_min_score",
        "entry_source": "overnight_plan",
        "source_bucket": "watchlist",
        "plan_score_source": "pm_plan_2026-04-02.json",
        "normalized_score": 78.0,
        "raw_score": 38.0,
        "validated_score": 47.0,
        "current_score": 47.0,
        "entry_threshold": 50.0,
        "entry_threshold_base": 50.0,
        "entry_threshold_source": "overnight_watchlist_research_relief",
        "entry_threshold_adjustment": 10.0,
        "score_gap_to_threshold": -3.0,
        "validation_carry_bonus": 5.0,
        "ranking_position": None,
        "displaced_by": "",
        "displaced_by_score": None,
        "missing_inputs": [],
        "phases_seen": [],
        "history": [],
        "last_updated_at": "",
    }

    dm._mark_signal_pending("CNK", "overnight_first_hour_recheck")

    row = dm._watchlist_causality["CNK"]
    assert row["current_status"] == "pending"
    assert row["blocking_reason"] == ""
    assert row["blocking_rule"] == ""


def test_apply_failsafe_triage_enforces_degraded_exit_rules():
    dm = _new_dm_stub()
    dm.strategy_failsafe_snapshot = SimpleNamespace(
        level="degraded",
        min_conviction_exit=55.0,
        loser_exit_pnl_pct=-5.0,
        stop_multiplier=2.0,
    )
    dm._hold_minutes = lambda symbol: 180.0

    updated = dm._apply_failsafe_triage(
        SimpleNamespace(symbol="CALM"),
        {"pnl_pct": -2.0, "action": "hold", "signals": []},
        conviction_score=41.0,
    )

    assert updated["action"] == "exit"
    assert updated["failsafe_forced_exit"] is True


def test_execute_entry_no_longer_uses_lessons_as_intraday_veto():
    dm = _new_dm_stub()
    dm._watched_universe_tickers = {"CNK"}
    dm.signals = [
        {
            "ticker": "CNK",
            "action": "buy_open",
            "recommendation": "buy",
            "confidence": 70.0,
            "score": 70.0,
            "normalized_score": 70.0,
            "entry_source": "overnight_plan",
            "plan_score_source": "morning_game_plan_2026-03-14.json",
        }
    ]
    dm.signal_status = {"CNK": {"status": "pending", "reason": ""}}
    dm.strategy_failsafe_snapshot = SimpleNamespace(
        halt_new_entries=False, level="normal"
    )
    dm.get_current_price = lambda ticker: 20.0
    dm._check_universe_compliance = lambda *args, **kwargs: (True, "")
    dm._apply_regime_entry_gate = lambda signal_data, allow_capped=False: (True, "")
    dm.wave_positions = {}
    dm.entry_wave = 1
    dm.get_positions = lambda: []
    dm.can_enter_positions = lambda entry_wave=None: (True, "")
    dm._has_open_buy_order = lambda symbol: (True, "ord-1")
    dm._mark_signal_skipped = DayManager._mark_signal_skipped.__get__(dm, DayManager)
    dm._check_learned_lessons = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("lessons should not be consulted intraday")
    )

    allowed = dm.execute_entry("CNK", "test")

    assert allowed is False
    assert dm.signal_status["CNK"]["reason"] == "pending_buy_order"


def test_execute_entry_uses_candidate_price_fallback_before_no_price_skip():
    dm = _new_dm_stub()
    dm._watched_universe_tickers = {"CNK"}
    dm.signals = [
        {
            "ticker": "CNK",
            "action": "buy_open",
            "recommendation": "buy",
            "confidence": 70.0,
            "score": 70.0,
            "normalized_score": 70.0,
            "entry_source": "overnight_plan",
            "plan_score_source": "morning_game_plan_2026-03-14.json",
        }
    ]
    dm.signal_status = {"CNK": {"status": "pending", "reason": ""}}
    dm.strategy_failsafe_snapshot = SimpleNamespace(
        halt_new_entries=False, level="normal"
    )
    dm.get_current_price = lambda ticker: None
    dm._check_universe_compliance = lambda *args, **kwargs: (True, "")
    dm._apply_regime_entry_gate = lambda signal_data, allow_capped=False: (True, "")
    dm.wave_positions = {}
    dm.entry_wave = 1
    dm.get_positions = lambda: []
    dm.can_enter_positions = lambda entry_wave=None, ticker=None: (True, "")
    dm._has_open_buy_order = lambda symbol: (True, "ord-1")
    dm._mark_signal_skipped = DayManager._mark_signal_skipped.__get__(dm, DayManager)

    allowed = dm.execute_entry(
        "CNK",
        "test",
        candidate_data={"ticker": "CNK", "current_price": 20.0},
    )

    assert allowed is False
    assert dm.signal_status["CNK"]["reason"] == "pending_buy_order"


def test_find_replacement_candidates_no_longer_uses_lessons_as_intraday_veto(
    monkeypatch,
):
    dm = _new_dm_stub()
    dm.signals = [
        {
            "ticker": "CNK",
            "action": "buy_open",
            "recommendation": "buy_open",
            "score": 68.0,
            "priority": 1,
            "bullish_score": 42.0,
            "s1_strength": 0.2,
            "r1_strength": 0.1,
            "atr_percent": 2.5,
        }
    ]
    dm.signal_status = {"CNK": {"status": "pending", "reason": ""}}
    dm.strategy = SimpleNamespace(
        passes_filter=lambda payload: (False, 18.0, ["avoid_repeat_loser"])
    )
    dm._calculate_realtime_score = lambda sig, vwap_data=None: 61.0
    dm._apply_candidate_backtest_validation = lambda candidates: list(candidates)

    monkeypatch.setattr(day_manager_mod, "INTRADAY_PROVIDER_AVAILABLE", False)

    candidates = dm.find_replacement_candidates(set())

    assert [row["ticker"] for row in candidates] == ["CNK"]
    assert candidates[0]["lessons_filter_blocked"] is True
    assert candidates[0]["lesson_score"] == pytest.approx(18.0)
    assert candidates[0]["lesson_rules"] == ["avoid_repeat_loser"]
    assert dm.signal_status["CNK"]["status"] == "pending"


def test_candidate_backtest_validation_penalizes_plan_watchlist_instead_of_rejecting():
    dm = _new_dm_stub()
    dm._watched_universe_tickers = {"CNK"}
    dm.signal_status = {"CNK": {"status": "pending", "reason": ""}}
    dm._candidate_validation_cache = {}
    dm._get_signal_validator = lambda: SimpleNamespace(
        validate_batch=lambda payload: [
            SimpleNamespace(
                backtest_score=22.0,
                historical_win_rate=0.41,
                similar_signals_found=16,
            )
        ]
    )

    candidates = dm._apply_candidate_backtest_validation(
        [
            {
                "ticker": "CNK",
                "score": 84.0,
                "realtime_score": 84.0,
                "normalized_score": 84.0,
                "entry_source": "overnight_plan",
                "source_bucket": "watchlist",
            }
        ]
    )

    assert [row["ticker"] for row in candidates] == ["CNK"]
    assert candidates[0]["backtest_validation_mode"] == "watchlist_penalty_only"
    assert candidates[0]["backtest_penalty"] > 0.0
    assert candidates[0]["realtime_score"] < 84.0
    assert dm.signal_status["CNK"]["status"] == "pending"


def test_candidate_backtest_validation_still_rejects_intraday_discovery():
    dm = _new_dm_stub()
    dm.signal_status = {"DISC": {"status": "pending", "reason": ""}}
    dm._candidate_validation_cache = {}
    dm._get_signal_validator = lambda: SimpleNamespace(
        validate_batch=lambda payload: [
            SimpleNamespace(
                backtest_score=22.0,
                historical_win_rate=0.41,
                similar_signals_found=16,
            )
        ]
    )

    candidates = dm._apply_candidate_backtest_validation(
        [
            {
                "ticker": "DISC",
                "score": 84.0,
                "realtime_score": 84.0,
                "entry_source": "intraday_reserve_scan",
                "source_bucket": "intraday_discovery",
            }
        ]
    )

    assert candidates == []
    assert dm.signal_status["DISC"]["reason"] == "backtest_rejected"


def test_runtime_entry_anchor_reprices_moderate_upside_move_and_accepts_gap_under_cap():
    dm = _new_dm_stub()
    dm.entry_quality_cfg.entry_gap_reject_pct = 7.0
    dm.premarket_gap_cfg = SimpleNamespace(moderate_gap_up_pct=3.0)

    allowed, anchor, reason = dm._resolve_runtime_entry_anchor(
        action="buy_open",
        entry_price=100.0,
        current_price=102.4,
    )

    assert allowed is True
    assert anchor == pytest.approx(102.4)
    assert reason == "planned_entry_repriced"

    allowed, anchor, reason = dm._resolve_runtime_entry_anchor(
        action="buy_open",
        entry_price=100.0,
        current_price=104.1,
    )

    assert allowed is True
    assert anchor == pytest.approx(104.1)
    assert reason == "planned_entry_within_gap_cap"


def test_runtime_entry_anchor_uses_current_price_for_replacement_candidates():
    dm = _new_dm_stub()
    dm.entry_quality_cfg.entry_gap_reject_pct = 7.0
    dm.premarket_gap_cfg = SimpleNamespace(moderate_gap_up_pct=3.0)

    allowed, anchor, reason = dm._resolve_runtime_entry_anchor(
        action="buy_open",
        entry_price=2.59,
        current_price=4.04,
        signal_data={"replacement_for_symbol": "CYTK"},
    )

    assert allowed is True
    assert anchor == pytest.approx(4.04)
    assert reason in {"planned_entry_prev_close_clamped", ""}


def test_runtime_entry_anchor_clamps_stale_prior_close_drift():
    dm = _new_dm_stub()
    dm.entry_quality_cfg.entry_anchor_prev_close_max_drift_pct = 5.0
    dm.premarket_gap_cfg = SimpleNamespace(moderate_gap_up_pct=3.0)

    allowed, anchor, reason = dm._resolve_runtime_entry_anchor(
        action="buy_open",
        entry_price=115.0,
        current_price=105.0,
        prev_close=100.0,
    )

    assert allowed is True
    assert anchor == pytest.approx(105.0)
    assert reason == "planned_entry_prev_close_clamped"


def test_runtime_entry_anchor_blocks_price_above_prior_close_drift_band():
    dm = _new_dm_stub()
    dm.entry_quality_cfg.entry_anchor_prev_close_max_drift_pct = 5.0
    dm.premarket_gap_cfg = SimpleNamespace(moderate_gap_up_pct=3.0)

    allowed, anchor, reason = dm._resolve_runtime_entry_anchor(
        action="buy_open",
        entry_price=115.0,
        current_price=106.0,
        prev_close=100.0,
    )

    assert allowed is False
    assert anchor == pytest.approx(105.0)
    assert reason == "prev_close_entry_drift"


def test_runtime_entry_anchor_allows_high_conviction_gapup_exception_in_favorable_regime():
    dm = _new_dm_stub()
    dm.entry_quality_cfg.entry_gap_reject_pct = 7.0
    dm.premarket_gap_cfg = SimpleNamespace(moderate_gap_up_pct=3.0)
    dm._effective_market_regime = lambda: "NEUTRAL"

    allowed, anchor, reason = dm._resolve_runtime_entry_anchor(
        action="buy_open",
        entry_price=100.0,
        current_price=108.4,
        entry_score=72.0,
        signal_data={
            "risk_reward": 2.2,
            "volume_ratio": 1.7,
            "setup_type": "opening_range_breakout",
        },
    )

    assert allowed is True
    assert anchor == pytest.approx(108.4)
    assert reason == "planned_entry_gapup_exception"


def test_runtime_entry_anchor_keeps_gapup_exception_disabled_in_bearish_regime():
    dm = _new_dm_stub()
    dm.entry_quality_cfg.entry_gap_reject_pct = 7.0
    dm.premarket_gap_cfg = SimpleNamespace(moderate_gap_up_pct=3.0)
    dm._effective_market_regime = lambda: "CRISIS"

    allowed, anchor, reason = dm._resolve_runtime_entry_anchor(
        action="buy_open",
        entry_price=100.0,
        current_price=108.4,
        entry_score=72.0,
        signal_data={
            "risk_reward": 2.2,
            "volume_ratio": 1.7,
            "setup_type": "opening_range_breakout",
        },
    )

    assert allowed is False
    assert anchor == pytest.approx(100.0)
    assert reason == "entry_gap_too_large"


def test_runtime_entry_anchor_allows_plan_watchlist_gapup_with_wider_exception_window():
    dm = _new_dm_stub()
    dm.entry_quality_cfg.entry_gap_reject_pct = 7.0
    dm.entry_quality_cfg.wave_hard_reject_gap_pct = 12.0
    dm.premarket_gap_cfg = SimpleNamespace(moderate_gap_up_pct=3.0)
    dm._effective_market_regime = lambda: "NEUTRAL"

    allowed, anchor, reason = dm._resolve_runtime_entry_anchor(
        action="buy_open",
        entry_price=100.0,
        current_price=111.2,
        entry_score=65.0,
        signal_data={
            "entry_source": "overnight_plan",
            "source_bucket": "watchlist",
            "plan_score_source": "pm_plan_2026-04-02.json",
            "risk_reward": 1.7,
            "volume_ratio": 1.3,
            "setup_type": "opening_range_breakout",
        },
    )

    assert allowed is True
    assert anchor == pytest.approx(111.2)
    assert reason == "planned_entry_gapup_exception"


def test_runtime_entry_anchor_expands_gap_cap_for_lean_bullish_regime():
    dm = _new_dm_stub()
    dm.entry_quality_cfg.entry_gap_reject_pct = 7.0
    dm.premarket_gap_cfg = SimpleNamespace(moderate_gap_up_pct=3.0)
    dm._effective_market_regime = lambda: "LEAN_BULLISH"

    allowed, anchor, reason = dm._resolve_runtime_entry_anchor(
        action="buy_open",
        entry_price=100.0,
        current_price=114.0,
        entry_score=72.0,
        signal_data={
            "risk_reward": 2.2,
            "volume_ratio": 1.7,
            "setup_type": "opening_range_breakout",
        },
    )

    assert allowed is True
    assert anchor == pytest.approx(114.0)
    assert reason == "planned_entry_gapup_exception"


def test_runtime_entry_anchor_allows_quick_turnover_continuation_reanchor():
    dm = _new_dm_stub()
    dm.entry_quality_cfg.entry_gap_reject_pct = 7.0
    dm.entry_quality_cfg.quick_turnover_continuation_enabled = True
    dm.entry_quality_cfg.quick_turnover_continuation_min_score = 66.0
    dm.entry_quality_cfg.quick_turnover_continuation_max_gap_pct = 18.0
    dm.entry_quality_cfg.quick_turnover_continuation_min_volume_ratio = 1.0
    dm.premarket_gap_cfg = SimpleNamespace(moderate_gap_up_pct=3.0)
    dm._effective_market_regime = lambda: "NEUTRAL"

    allowed, anchor, reason = dm._resolve_runtime_entry_anchor(
        action="buy_open",
        entry_price=100.0,
        current_price=116.0,
        entry_score=68.0,
        signal_data={
            "entry_source": "overnight_plan",
            "source_bucket": "watchlist",
            "plan_score_source": "morning_game_plan_20260422.json",
            "overnight_execution_intent": "quick_turnover",
            "overnight_actionability_score": 71.0,
            "risk_reward": 1.45,
            "volume_ratio": 0.9,
            "setup_type": "pullback_support",
        },
    )

    assert allowed is True
    assert anchor == pytest.approx(116.0)
    assert reason == "planned_entry_quick_turnover_continuation"


def test_runtime_entry_anchor_allows_explicit_pm_wave_breakout_rescue():
    dm = _new_dm_stub()
    dm.entry_quality_cfg.entry_gap_reject_pct = 7.0
    dm.entry_quality_cfg.wave_breakout_rescue_min_score = 75.0
    dm.entry_quality_cfg.wave_breakout_rescue_max_gap_pct = 9.0
    dm.premarket_gap_cfg = SimpleNamespace(moderate_gap_up_pct=3.0)
    dm._effective_market_regime = lambda: "NEUTRAL"

    allowed, anchor, reason = dm._resolve_runtime_entry_anchor(
        action="buy_open",
        entry_price=100.0,
        current_price=108.1,
        entry_score=76.0,
        signal_data={
            "entry_source": "overnight_plan",
            "source_bucket": "watchlist",
            "plan_score_source": "pm_plan_2026-05-04.json",
            "runtime_entry_context": "wave_2_entry",
            "pm_wave_entry": True,
            "wave_breakout_rescue": True,
        },
    )

    assert allowed is True
    assert anchor == pytest.approx(108.1)
    assert reason == "planned_entry_wave_breakout_rescue"


def test_runtime_entry_anchor_keeps_hold_candidate_gap_block_for_same_setup():
    dm = _new_dm_stub()
    dm.entry_quality_cfg.entry_gap_reject_pct = 7.0
    dm.entry_quality_cfg.quick_turnover_continuation_enabled = True
    dm.entry_quality_cfg.quick_turnover_continuation_min_score = 66.0
    dm.entry_quality_cfg.quick_turnover_continuation_max_gap_pct = 18.0
    dm.entry_quality_cfg.quick_turnover_continuation_min_volume_ratio = 1.0
    dm.premarket_gap_cfg = SimpleNamespace(moderate_gap_up_pct=3.0)
    dm._effective_market_regime = lambda: "NEUTRAL"

    allowed, anchor, reason = dm._resolve_runtime_entry_anchor(
        action="buy_open",
        entry_price=100.0,
        current_price=116.0,
        entry_score=68.0,
        signal_data={
            "entry_source": "overnight_plan",
            "source_bucket": "watchlist",
            "plan_score_source": "morning_game_plan_20260422.json",
            "overnight_execution_intent": "hold_candidate",
            "overnight_actionability_score": 71.0,
            "risk_reward": 1.45,
            "volume_ratio": 0.9,
            "setup_type": "pullback_support",
        },
    )

    assert allowed is False
    assert anchor == pytest.approx(100.0)
    assert reason == "entry_gap_too_large"


def test_gap_hard_cap_pct_widens_neutral_on_strong_breadth():
    dm = _new_dm_stub()
    dm.current_regime_analysis = SimpleNamespace(breadth_pct_positive=72.0)

    assert dm._gap_hard_cap_pct("NEUTRAL") == pytest.approx(13.0)

    dm.regime_strategy_overrides = {"gap_hard_cap_pct": 14.5}

    assert dm._gap_hard_cap_pct("NEUTRAL") == pytest.approx(14.5)


def test_capital_reserve_keeps_small_buffer_above_and_below_pdt():
    above_pdt = SimpleNamespace(equity="88500")
    below_pdt = SimpleNamespace(equity="24000")

    assert DayManager._capital_reserve_for_account(
        above_pdt, 331000.0
    ) == pytest.approx(1000.0)
    assert DayManager._capital_reserve_for_account(below_pdt, 9000.0) == pytest.approx(
        1000.0
    )


def test_should_tolerate_watchlist_prune_allows_plan_watchlist_labels():
    dm = _new_dm_stub()

    tolerated = dm._should_tolerate_watchlist_prune(
        signal_data={
            "ticker": "FGI",
            "entry_source": "overnight_plan",
            "source_bucket": "watchlist",
            "plan_score_source": "pm_plan_2026-04-02.json",
            "realtime_score": 100.0,
        },
        reasons=["vwap_violation_3x5m", "orb_breakdown_15m_low"],
        indicator_death=False,
        close_price=99.4,
        orb_low=100.0,
        vwap_value=100.0,
    )

    assert tolerated is True


def test_no_fill_watchdog_escalates_bounded_override_for_top_watchlist_candidates():
    dm = _new_dm_stub()
    dm.cycle_count = 12
    dm.signal_status = {
        "AAA": {"status": "pending", "reason": ""},
        "BBB": {"status": "pending", "reason": ""},
        "CCC": {"status": "pending", "reason": ""},
        "ZZZ": {"status": "pending", "reason": ""},
    }
    dm._watched_universe_tickers = {"AAA", "BBB", "CCC"}
    dm._no_fill_watchdog_state["streak"] = 4
    dm._intraday_reserved_scan_enabled = False

    candidates = [
        {"ticker": "ZZZ", "realtime_score": 95.0, "score": 95.0},
        {"ticker": "AAA", "realtime_score": 91.0, "score": 91.0},
        {"ticker": "BBB", "realtime_score": 88.0, "score": 88.0},
        {"ticker": "CCC", "realtime_score": 84.0, "score": 84.0},
    ]
    stats = {}

    updated, local_cap, symbols = dm._maybe_apply_no_fill_watchdog(
        candidates,
        held_tickers=[],
        current_phase=day_manager_mod.TradingPhase.CORE_TRADING,
        open_slots=2,
        stats=stats,
    )

    assert local_cap == 2
    assert symbols == ["AAA", "BBB", "CCC"][:3]
    assert stats["intervention_requested"] is True
    assert updated[0]["ticker"] == "AAA"
    assert updated[1]["ticker"] == "BBB"


def test_no_fill_watchdog_override_does_not_clamp_existing_entry_capacity():
    max_new_entries = 6
    open_slots = 10
    remaining_day_trades = 8
    local_max_entries_override = 2

    if local_max_entries_override is not None:
        if max_new_entries > 0:
            max_new_entries = max(
                max_new_entries,
                min(
                    int(local_max_entries_override),
                    int(max(0, open_slots)),
                    int(max(0, remaining_day_trades)),
                ),
            )

    assert max_new_entries == 6


def test_candidate_slice_tolerates_float_max_new_entries():
    candidates = [{"ticker": "AAA"}] * 10
    max_new_entries = 2.0

    subset = candidates[: int(max(0, max_new_entries)) + 5]

    assert len(subset) == 7


def test_no_fill_watchdog_state_increments_without_fills_and_resets_after_entry():
    dm = _new_dm_stub()
    dm.signal_status = {"AAA": {"status": "pending", "reason": ""}}
    dm._watched_universe_tickers = {"AAA"}

    state = dm._update_no_fill_watchdog_state(
        current_phase=day_manager_mod.TradingPhase.CORE_TRADING,
        open_slots=2,
        entries_opened=0,
        candidates=[{"ticker": "AAA", "realtime_score": 80.0}],
        intervention_symbols=[],
    )

    assert state["streak"] == 1
    assert state["last_reason"] == "open_slots_without_fills"

    state = dm._update_no_fill_watchdog_state(
        current_phase=day_manager_mod.TradingPhase.CORE_TRADING,
        open_slots=2,
        entries_opened=1,
        candidates=[{"ticker": "AAA", "realtime_score": 80.0}],
        intervention_symbols=["AAA"],
    )

    assert state["streak"] == 0
    assert state["last_reason"] == "entries_opened"
    assert state["intervention_requested"] is True


def test_replacement_hold_guard_blocks_fresh_positions():
    dm = _new_dm_stub()
    dm._hold_minutes = lambda symbol: 45

    reason = dm._replacement_hold_guard_reason(
        "AAA", health={"failsafe_forced_exit": False}
    )

    assert reason == "replacement_min_hold:45<120"


def test_replacement_prefilter_skips_runaway_intraday_candidate():
    dm = _new_dm_stub()
    dm.get_current_price = lambda symbol: 11.5
    dm._get_session_open_price = lambda symbol: 10.0

    reason = dm._replacement_candidate_prefilter_reason(
        {"ticker": "RUN", "entry_price": 10.0, "realtime_score": 80.0},
        replacement_for_symbol="WEAK",
    )

    assert reason.startswith("vwap_overextended:open_to_now=15.0%>12.0%")


def test_replacement_prefilter_keeps_stale_plan_gap_when_replacing():
    dm = _new_dm_stub()
    dm.get_current_price = lambda symbol: 4.04
    dm._get_session_open_price = lambda symbol: 4.00

    reason = dm._replacement_candidate_prefilter_reason(
        {"ticker": "MIGI", "entry_price": 2.59, "realtime_score": 80.0},
        replacement_for_symbol="CYTK",
    )

    assert reason == ""


def test_prune_watchlist_keeps_name_with_only_vwap_violation(monkeypatch):
    dm = _new_dm_stub()
    dm.signals = [{"ticker": "OPEN", "setup_type": "new_high_breakout"}]
    dm.signal_status = {"OPEN": {"status": "pending", "reason": ""}}

    monkeypatch.setattr(day_manager_mod, "INTRADAY_PROVIDER_AVAILABLE", True)

    bars = pd.DataFrame(
        {
            "open": [10.0] * 120,
            "high": [10.1] * 120,
            "low": [9.0, 9.0, 9.0] + [9.9] * 117,
            "close": [10.0] * 105 + [9.7, 9.6, 9.5] + [9.55] * 12,
            "volume": [1000] * 120,
        },
        index=pd.date_range("2026-03-17 09:30:00", periods=120, freq="min"),
    )
    monkeypatch.setattr(
        day_manager_mod,
        "get_intraday_bars_batch",
        lambda tickers, data_client, minutes_back=240, max_batch=25: {"OPEN": bars},
    )

    dropped = dm.prune_watchlist([])

    assert dropped == []
    assert "OPEN" not in getattr(dm, "_watchlist_drop_buffer", {})


def test_prune_watchlist_tolerates_mild_orb_and_vwap_breach_for_strong_watchlist_name(
    monkeypatch,
):
    dm = _new_dm_stub()
    dm.signals = [
        {
            "ticker": "GLNG",
            "setup_type": "trend_follow",
            "entry_source": "overnight_full_watchlist",
            "score": 38.5,
            "realtime_score": 38.5,
            "confidence": 90.0,
        }
    ]
    dm.signal_status = {"GLNG": {"status": "pending", "reason": ""}}

    monkeypatch.setattr(day_manager_mod, "INTRADAY_PROVIDER_AVAILABLE", True)

    closes = [10.0] * 105 + [9.92, 9.91, 9.90] + [9.92] * 12
    bars = pd.DataFrame(
        {
            "open": [10.0] * 120,
            "high": [10.05] * 120,
            "low": [10.0] * 15 + [9.89] * 105,
            "close": closes,
            "volume": [1000] * 120,
        },
        index=pd.date_range("2026-03-17 09:30:00", periods=120, freq="min"),
    )
    monkeypatch.setattr(
        day_manager_mod,
        "get_intraday_bars_batch",
        lambda tickers, data_client, minutes_back=240, max_batch=25: {"GLNG": bars},
    )

    dropped = dm.prune_watchlist([])

    assert dropped == []
    assert dm.signal_status["GLNG"]["status"] == "pending"
    assert (
        dm.signal_status["GLNG"]["reason"]
        == "watchlist_deferred:vwap_violation_3x5m,orb_breakdown_15m_low"
    )
    assert "GLNG" not in getattr(dm, "_watchlist_drop_buffer", {})


def test_prune_watchlist_still_drops_large_orb_breach_for_watchlist_name(monkeypatch):
    dm = _new_dm_stub()
    dm.signals = [
        {
            "ticker": "DVN",
            "setup_type": "trend_follow",
            "entry_source": "overnight_full_watchlist",
            "score": 38.0,
            "realtime_score": 38.0,
            "confidence": 88.0,
        }
    ]
    dm.signal_status = {"DVN": {"status": "pending", "reason": ""}}

    monkeypatch.setattr(day_manager_mod, "INTRADAY_PROVIDER_AVAILABLE", True)

    closes = [10.0] * 105 + [9.7, 9.65, 9.6] + [9.6] * 12
    bars = pd.DataFrame(
        {
            "open": [10.0] * 120,
            "high": [10.05] * 120,
            "low": [10.0] * 15 + [9.55] * 105,
            "close": closes,
            "volume": [1000] * 120,
        },
        index=pd.date_range("2026-03-17 09:30:00", periods=120, freq="min"),
    )
    monkeypatch.setattr(
        day_manager_mod,
        "get_intraday_bars_batch",
        lambda tickers, data_client, minutes_back=240, max_batch=25: {"DVN": bars},
    )

    dropped = dm.prune_watchlist([])

    assert dropped
    assert dropped[0]["ticker"] == "DVN"
    assert dm.signal_status["DVN"]["reason"].startswith("watchlist_invalid:")


def test_submit_order_uses_marketable_router_path_for_buy_limits():
    dm = _new_dm_stub()
    dm.lifecycle_logger = _LifecycleLoggerStub()
    dm.client = SimpleNamespace(
        get_account=lambda: SimpleNamespace(cash="50000", buying_power="100000")
    )
    dm.execution_router = SimpleNamespace()
    captured = {}

    def _submit_marketable_limit_order(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            order_id="ord-1",
            status="submitted",
            symbol=kwargs["symbol"],
            side=kwargs["side"],
            requested_qty=kwargs["qty"],
            filled_qty=0,
            avg_fill_price=0.0,
            raw=None,
            urgency_tier=kwargs["urgency_tier"],
            intended_price=kwargs["reference_price"],
            decision_price=kwargs["max_limit_price"],
            slippage_bps=0.0,
            time_to_first_fill_ms=None,
            replace_count=kwargs["replace_count"],
            metadata={"order_type": "limit"},
        )

    dm.execution_router.submit_marketable_limit_order = _submit_marketable_limit_order
    dm.execution_router.submit_order = lambda request: (_ for _ in ()).throw(
        AssertionError("plain submit_order should not be used")
    )
    dm.get_current_price = lambda symbol: 100.0
    dm._capture_nbbo_snapshot = lambda symbol: {
        "bid_price": 99.95,
        "ask_price": 100.05,
        "mid_price": 100.0,
    }

    order = dm._submit_order_via_execution_adapter(
        symbol="AAPL",
        qty=10,
        side="buy",
        context="execute_entry_limit",
        limit_price=100.08,
        order_type_override="limit",
        urgency_tier="high",
        intended_price=100.0,
        decision_price=100.08,
        slippage_budget_bps=20,
        replace_count=0,
        metadata={"atr_14": 0.4},
        record_journal=False,
    )

    assert captured["symbol"] == "AAPL"
    assert captured["reference_price"] == pytest.approx(100.0)
    assert captured["atr_14"] == pytest.approx(0.4)
    assert captured["bid_price"] == pytest.approx(99.95)
    assert captured["ask_price"] == pytest.approx(100.05)
    assert captured["max_limit_price"] == pytest.approx(100.08)
    assert order.id == "ord-1"


def test_submit_order_uses_marketable_router_path_for_sell_limits():
    dm = _new_dm_stub()
    dm.lifecycle_logger = _LifecycleLoggerStub()
    dm.execution_router = SimpleNamespace()
    captured = {}

    def _submit_marketable_limit_order(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            order_id="ord-sell-1",
            status="submitted",
            symbol=kwargs["symbol"],
            side=kwargs["side"],
            requested_qty=kwargs["qty"],
            filled_qty=0,
            avg_fill_price=0.0,
            raw=None,
            urgency_tier=kwargs["urgency_tier"],
            intended_price=kwargs["reference_price"],
            decision_price=kwargs["min_limit_price"],
            slippage_bps=0.0,
            time_to_first_fill_ms=None,
            replace_count=kwargs["replace_count"],
            metadata={"order_type": "limit"},
        )

    dm.execution_router.submit_marketable_limit_order = _submit_marketable_limit_order
    dm.execution_router.submit_order = lambda request: (_ for _ in ()).throw(
        AssertionError("plain submit_order should not be used")
    )
    dm.get_current_price = lambda symbol: 100.0
    dm._capture_nbbo_snapshot = lambda symbol: {
        "bid_price": 99.95,
        "ask_price": 100.05,
        "mid_price": 100.0,
    }

    order = dm._submit_order_via_execution_adapter(
        symbol="AAPL",
        qty=10,
        side="sell",
        context="execute_exit",
        limit_price=99.92,
        order_type_override="limit",
        urgency_tier="high",
        intended_price=100.0,
        decision_price=99.92,
        slippage_budget_bps=20,
        replace_count=0,
        metadata={"atr_14": 0.4},
        record_journal=False,
    )

    assert captured["symbol"] == "AAPL"
    assert captured["side"] == "sell"
    assert captured["reference_price"] == pytest.approx(100.0)
    assert captured["min_limit_price"] == pytest.approx(99.92)
    assert captured["max_limit_price"] is None
    assert order.id == "ord-sell-1"


def test_adjust_pending_limit_orders_uses_limit_chase_manager(monkeypatch):
    dm = _new_dm_stub()
    dm.dry_run = False
    dm.order_optimizer = None
    canceled = []
    dm.client = SimpleNamespace(
        cancel_order_by_id=lambda order_id: canceled.append(order_id)
    )
    dm.limit_chase_manager = None
    dm._entry_order_lifecycle = {
        "ord-1": {
            "planned_entry": 100.0,
            "urgency_tier": "normal",
            "replace_count": 0,
            "created_at": datetime(2026, 3, 7, 9, 59, 0, tzinfo=timezone.utc),
            "last_replace_at": datetime(2026, 3, 7, 9, 59, 0, tzinfo=timezone.utc),
            "escalate_allowed": False,
            "atr_14": 0.4,
        }
    }
    dm.signals = [{"ticker": "AAPL", "score": 60.0, "atr_14": 0.4}]
    dm._get_limit_chase_manager = DayManager._get_limit_chase_manager.__get__(
        dm, DayManager
    )
    dm._entry_replace_schedule_seconds = (
        DayManager._entry_replace_schedule_seconds.__get__(dm, DayManager)
    )
    dm._entry_tier_limits = DayManager._entry_tier_limits.__get__(dm, DayManager)
    dm._capture_nbbo_snapshot = lambda symbol: {
        "bid_price": 100.08,
        "ask_price": 100.12,
        "mid_price": 100.10,
    }
    dm._normalize_entry_time = lambda value: (
        value if isinstance(value, datetime) else None
    )
    dm._promote_entry_order_lifecycle = (
        DayManager._promote_entry_order_lifecycle.__get__(dm, DayManager)
    )
    dm._submit_order_via_execution_adapter = lambda **kwargs: SimpleNamespace(
        id="ord-2",
        replace_count=kwargs["replace_count"],
    )
    dm.get_current_price = lambda symbol: 100.12

    pending_order = SimpleNamespace(
        symbol="AAPL",
        side="buy",
        type="limit",
        id="ord-1",
        limit_price=99.80,
        qty=10,
    )

    class _FixedDateTime:
        @classmethod
        def now(cls, *args, **kwargs):
            tz = args[0] if args else kwargs.get("tz")
            if tz is not None:
                return datetime(2026, 3, 7, 10, 0, 0, tzinfo=tz)
            return datetime(2026, 3, 7, 10, 0, 0)

    monkeypatch.setattr(day_manager_mod, "datetime", _FixedDateTime)

    adjusted = DayManager._adjust_pending_limit_orders(dm, [pending_order])

    assert adjusted == 1
    assert canceled == ["ord-1"]
    assert "ord-1" not in dm._entry_order_lifecycle
    assert dm._entry_order_lifecycle["ord-2"]["replace_count"] == 1


def test_run_watchlist_rotation_scheduler_reloads_rotated_symbols():
    dm = _new_dm_stub()
    dm.signals = [
        {"symbol": "A", "ticker": "A", "opportunity_score": 40.0},
        {"symbol": "B", "ticker": "B", "opportunity_score": 42.0},
    ]
    dm.opportunity_scorer = SimpleNamespace(
        rank_stocks=lambda df: df.assign(
            rvol=df.get("rvol", 1.0),
            catalyst_score=df.get("catalyst_score", 1.0),
            atr_14=df.get("atr_14", 1.0),
            intraday_range=df.get("intraday_range", 1.0),
        )
    )
    dm.watchlist_rotator = SimpleNamespace(
        evaluate_rotation=lambda watchlist, candidates: [
            {"remove": "A", "add": "D", "reason": "OPPORTUNITY_DELTA", "delta": 20.0}
        ],
        perform_swaps=lambda current_symbols, swaps: ["D", "B"],
    )
    dm.universe_scanner = SimpleNamespace(
        combined_scan=lambda max_candidates=50: __import__("pandas").DataFrame(
            [{"symbol": "D", "opportunity_score": 60.0}]
        )
    )
    dm._load_signals_for_symbols = lambda symbols: [
        {"symbol": symbol, "ticker": symbol} for symbol in symbols
    ]
    dm._emit_watchlist_transition_metrics = lambda transitions, context: None

    rotated = DayManager._run_watchlist_rotation_scheduler(dm, held_tickers=[])

    assert rotated == 1
    assert [row["symbol"] for row in dm.signals] == ["D", "B"]


def test_emit_watchlist_transition_metrics_logs_rotation_events(monkeypatch):
    dm = _new_dm_stub()
    dm.cycle_count = 15
    emitted = []

    class _Collector:
        def emit(self, **kwargs):
            emitted.append(kwargs)

    monkeypatch.setattr(day_manager_mod, "MONITORING_AVAILABLE", True)
    monkeypatch.setattr(day_manager_mod, "get_metrics_collector", lambda: _Collector())

    DayManager._emit_watchlist_transition_metrics(
        dm,
        transitions=[
            {
                "remove": "A",
                "add": "D",
                "reason": "OPPORTUNITY_DELTA",
                "delta": 20.0,
            }
        ],
        context="scheduled_rotation",
    )

    assert len(emitted) == 2
    assert emitted[0]["metric_name"] == "watchlist.rotation"
    assert emitted[1]["metric_name"] == "watchlist.transition"
    assert emitted[1]["values"]["remove"] == "A"
    assert emitted[1]["values"]["add"] == "D"


def test_execute_exit_uses_market_protocol_for_hard_stop():
    dm = _new_dm_stub()
    dm.dry_run = False
    dm.client = SimpleNamespace(get_orders=lambda request: [])
    journal_calls = {}
    dm.trade_journal = SimpleNamespace(
        record_exit=lambda *args, **kwargs: (
            journal_calls.update({"args": args, "kwargs": kwargs}) or "trade-1"
        ),
        save=lambda: None,
    )
    dm.position_entries = {}
    dm.day_tracker = SimpleNamespace(record_day_trade=lambda **kwargs: None)
    dm._queue_sequential_shadow_event = lambda **kwargs: None
    dm._record_execution_attempt = lambda *args, **kwargs: None
    dm._record_execution_success = lambda *args, **kwargs: None
    dm._record_execution_failure = lambda *args, **kwargs: None
    dm._validate_order = lambda symbol, qty, side, allow_open_sell=False: (True, "")
    dm._acquire_order_submission_guard = lambda *args, **kwargs: (True, "", "guard-1")
    dm._release_order_submission_guard = lambda *args, **kwargs: None
    dm._validated_positions = lambda positions, context="": positions
    # day-manager 2026-05-19: desync guard added 2026-05-18 aborts the exit
    # when broker reports no position. Stub a matching AAPL position so the
    # guard is satisfied and the test exercises the actual exit-submission
    # path instead of the abort branch.
    dm.get_positions = lambda: [SimpleNamespace(symbol="AAPL", qty="10")]
    dm.signals = [{"ticker": "AAPL", "atr_14": 0.6}]
    dm.get_current_price = lambda symbol: 100.0
    dm._resolve_execution_atr = DayManager._resolve_execution_atr.__get__(
        dm, DayManager
    )
    captured = {}

    def _submit_order_via_execution_adapter(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id="ord-exit-1", status="submitted")

    dm._submit_order_via_execution_adapter = _submit_order_via_execution_adapter

    assert dm.execute_exit("AAPL", 10, "CRITICAL: hard stop")
    assert captured["side"] == "sell"
    assert captured["order_type_override"] == "market"
    assert captured["intended_price"] == pytest.approx(100.0)
    assert captured["metadata"]["atr_14"] == pytest.approx(0.6)
    assert journal_calls["kwargs"]["symbol"] == "AAPL"
    assert journal_calls["kwargs"]["order_id"] == "ord-exit-1"
    assert journal_calls["kwargs"]["execution_status"] == "submitted"


def test_execute_trim_records_immediate_trim_journal_entry():
    dm = _new_dm_stub()
    dm.dry_run = False
    dm.trade_journal = SimpleNamespace(
        record_trim=lambda *args, **kwargs: (
            trim_calls.update({"args": args, "kwargs": kwargs}) or "trim-1"
        ),
        save=lambda: None,
    )
    dm._record_execution_attempt = lambda *args, **kwargs: None
    dm._record_execution_success = lambda *args, **kwargs: None
    dm._record_execution_failure = lambda *args, **kwargs: None
    dm._queue_sequential_shadow_event = lambda **kwargs: None
    dm._acquire_order_submission_guard = lambda *args, **kwargs: (True, "", "guard-1")
    dm._release_order_submission_guard = lambda *args, **kwargs: None
    dm._validate_order = lambda symbol, qty, side, allow_open_sell=False: (True, "")
    dm.get_current_price = lambda symbol: 50.0
    dm.get_positions = lambda: [SimpleNamespace(symbol="AAPL", qty="50")]
    dm._validated_positions = lambda positions, context="": positions
    dm._last_trim_time = {}
    dm._trim_count_today = {}
    trim_calls = {}

    dm._submit_order_via_execution_adapter = lambda **kwargs: SimpleNamespace(
        id="ord-trim-1",
        status="submitted",
    )

    assert dm.execute_trim("AAPL", 10, "rebalance trim")
    assert trim_calls["kwargs"]["symbol"] == "AAPL"
    assert trim_calls["kwargs"]["shares_sold"] == 10
    assert trim_calls["kwargs"]["order_id"] == "ord-trim-1"
    assert trim_calls["kwargs"]["execution_status"] == "submitted"
    assert trim_calls["kwargs"]["original_size"] == pytest.approx(2500.0)
    assert trim_calls["kwargs"]["new_size"] == pytest.approx(2000.0)


def test_execute_trim_blocks_fractional_share_micro_sell():
    dm = _new_dm_stub()
    dm.dry_run = False
    dm.get_current_price = lambda symbol: 50.0
    dm.get_positions = lambda: [SimpleNamespace(symbol="AAPL", qty="20")]
    dm._validated_positions = lambda positions, context="": positions
    dm._last_trim_time = {}
    dm._trim_count_today = {}
    submitted = {"called": False}
    dm._submit_order_via_execution_adapter = lambda **kwargs: submitted.update(
        called=True
    )

    allowed = dm.execute_trim("AAPL", 4, "rebalance trim")

    assert allowed is False
    assert submitted["called"] is False


def test_execute_trim_respects_trim_cooldown():
    dm = _new_dm_stub()
    dm.dry_run = False
    dm.TRIM_COOLDOWN_MINUTES = 60
    dm._last_trim_time = {"AAPL": datetime.now() - timedelta(minutes=10)}
    dm._trim_count_today = {}

    called = {"submit": False}
    dm._submit_order_via_execution_adapter = lambda **kwargs: called.update(submit=True)

    allowed = dm.execute_trim("AAPL", 5, "rebalance trim")

    assert allowed is False
    assert called["submit"] is False


def test_execute_trim_respects_daily_trim_cap():
    dm = _new_dm_stub()
    dm.dry_run = False
    dm.MAX_TRIMS_PER_SYMBOL_PER_DAY = 2
    dm._last_trim_time = {}
    dm._trim_count_today = {"AAPL": 2}

    called = {"submit": False}
    dm._submit_order_via_execution_adapter = lambda **kwargs: called.update(submit=True)

    allowed = dm.execute_trim("AAPL", 5, "rebalance trim")

    assert allowed is False
    assert called["submit"] is False


def test_execute_trim_blocks_noncritical_fresh_entry():
    dm = _new_dm_stub()
    dm.dry_run = False
    dm._hold_minutes = lambda symbol: 20
    dm._last_trim_time = {}
    dm._trim_count_today = {}

    called = {"submit": False}
    dm._submit_order_via_execution_adapter = lambda **kwargs: called.update(submit=True)

    allowed = dm.execute_trim(
        "AAPL", 5, "decision_claw_trim_override:weak_position_fallback"
    )

    assert allowed is False
    assert called["submit"] is False


def test_execute_trim_allows_quick_turnover_support_break_before_min_hold():
    dm = _new_dm_stub()
    dm.dry_run = True
    dm._hold_minutes = lambda symbol: 20
    dm._last_trim_time = {}
    dm._trim_count_today = {}
    dm._last_scale_time = {}
    dm._validate_order = lambda symbol, qty, side, allow_open_sell=False: (True, "")
    dm._validated_positions = lambda positions, context="": positions
    dm.get_positions = lambda: [SimpleNamespace(symbol="AAPL", qty="20")]
    dm.get_current_price = lambda symbol: 9.78
    dm.signals = [
        {
            "ticker": "AAPL",
            "entry_price": 10.0,
            "s1_price": 9.8,
            "overnight_execution_intent": "quick_turnover",
            "support_break_trim_profile": "aggressive",
        }
    ]

    allowed = dm.execute_trim("AAPL", 5, "decision_claw_trim_override:support_break")

    assert allowed is True


def test_execute_trim_keeps_hold_candidate_min_hold_guard():
    dm = _new_dm_stub()
    dm.dry_run = True
    dm._hold_minutes = lambda symbol: 20
    dm._last_trim_time = {}
    dm._trim_count_today = {}
    dm._last_scale_time = {}
    dm._validate_order = lambda symbol, qty, side, allow_open_sell=False: (True, "")
    dm._validated_positions = lambda positions, context="": positions
    dm.get_positions = lambda: [SimpleNamespace(symbol="AAPL", qty="20")]
    dm.get_current_price = lambda symbol: 9.78
    dm.signals = [
        {
            "ticker": "AAPL",
            "entry_price": 10.0,
            "s1_price": 9.8,
            "overnight_execution_intent": "hold_candidate",
            "support_break_trim_profile": "patient",
        }
    ]

    allowed = dm.execute_trim("AAPL", 5, "decision_claw_trim_override:support_break")

    assert allowed is False


def test_execute_trim_blocks_recently_scaled_noncritical_trim():
    dm = _new_dm_stub()
    dm.dry_run = False
    dm._hold_minutes = lambda symbol: 180
    dm._last_trim_time = {}
    dm._trim_count_today = {}
    dm._last_scale_time = {"AAPL": datetime.now() - timedelta(minutes=20)}

    called = {"submit": False}
    dm._submit_order_via_execution_adapter = lambda **kwargs: called.update(submit=True)

    allowed = dm.execute_trim(
        "AAPL", 5, "decision_claw_trim_override:weak_position_fallback"
    )

    assert allowed is False
    assert called["submit"] is False


def test_execute_trim_strength_lock_does_not_count_toward_daily_trim_cap():
    dm = _new_dm_stub()
    dm.dry_run = False
    dm._hold_minutes = lambda symbol: 900
    dm._last_trim_time = {}
    dm._trim_count_today = {"AAPL": 2}
    dm._last_scale_time = {}
    dm._validate_order = lambda symbol, qty, side, allow_open_sell=False: (True, "")
    dm._validated_positions = lambda positions, context="": positions
    dm.get_positions = lambda: [SimpleNamespace(symbol="AAPL", qty="20")]
    dm.get_current_price = lambda symbol: 50.0
    dm.trade_journal = SimpleNamespace(
        record_trim=lambda *args, **kwargs: None,
        save=lambda: None,
    )
    dm._record_execution_attempt = lambda *args, **kwargs: None
    dm._record_execution_success = lambda *args, **kwargs: None
    dm._record_execution_failure = lambda *args, **kwargs: None
    dm._queue_sequential_shadow_event = lambda **kwargs: None
    dm._acquire_order_submission_guard = lambda *args, **kwargs: (True, "", "guard-1")
    dm._release_order_submission_guard = lambda *args, **kwargs: None
    dm._submit_order_via_execution_adapter = lambda **kwargs: SimpleNamespace(
        id="ord-strength-lock-1",
        status="submitted",
    )

    allowed = dm.execute_trim("AAPL", 10, "strength_lock")

    assert allowed is True
    assert dm._trim_count_today["AAPL"] == 2


def test_get_dynamic_wave_cap_uses_slots_multiplier():
    dm = _new_dm_stub()
    dm.entry_quality_cfg.wave_entry_enabled = True
    dm.entry_quality_cfg.wave_max_entries = 10
    dm.entry_quality_cfg.wave_capacity_slots_multiplier = 0.75
    dm.entry_wave = 1
    dm._count_candidates_within_gap = lambda candidates: 20

    cap = dm._get_dynamic_wave_cap([{"ticker": "AAA"}], total_open_slots=8)

    assert cap == 6


def test_execute_exit_blocks_noncritical_fresh_entry():
    dm = _new_dm_stub()
    dm.dry_run = False
    dm.entry_quality_cfg.fresh_entry_exit_cooldown_minutes = 90
    dm.position_entries = {"AAPL": datetime.now(timezone.utc) - timedelta(minutes=20)}
    dm.get_current_price = lambda symbol: 100.0
    dm._acquire_order_submission_guard = lambda *args, **kwargs: (True, "", "guard-1")
    dm._release_order_submission_guard = lambda *args, **kwargs: None

    called = {"submit": False}
    dm._submit_order_via_execution_adapter = lambda **kwargs: called.update(submit=True)

    allowed = dm.execute_exit("AAPL", 10, "advisor_recheck_exit")

    assert allowed is False
    assert called["submit"] is False


def test_execute_exit_blocks_same_day_journal_entry_when_memory_missing():
    now = datetime.now(timezone.utc)
    dm = _new_dm_stub()
    dm.dry_run = False
    dm.entry_quality_cfg.fresh_entry_exit_cooldown_minutes = 120
    dm.position_entries = {}
    dm.trade_journal = SimpleNamespace(
        trades=[
            {
                "symbol": "CLSK",
                "trade_type": "entry",
                "entry_status": "filled",
                "entry_time": now - timedelta(minutes=36),
                "filled_quantity": 349,
                "side": "buy",
                "action": "buy",
            }
        ]
    )
    dm._now_utc = lambda: now
    dm._hold_minutes = lambda symbol: 999
    dm.get_current_price = lambda symbol: 12.50

    called = {"submit": False}
    dm._submit_order_via_execution_adapter = lambda **kwargs: called.update(submit=True)

    allowed = dm.execute_exit("CLSK", 349, "advisor_recheck_exit")

    assert allowed is False
    assert called["submit"] is False


def test_execute_trim_blocks_noncritical_same_day_churn_during_fresh_cooldown():
    dm = _new_dm_stub()
    dm.dry_run = False
    dm.entry_quality_cfg.fresh_entry_exit_cooldown_minutes = 90
    dm.position_entries = {"CLSK": datetime.now(timezone.utc) - timedelta(minutes=12)}
    dm._hold_minutes = lambda symbol: 12
    dm._min_hold_minutes_for_trim = 0
    dm._last_trim_time = {}
    dm._trim_count_today = {}
    dm._last_scale_time = {}
    dm.get_current_price = lambda symbol: 25.0
    dm.get_positions = lambda: [SimpleNamespace(symbol="CLSK", qty="100")]
    dm._validated_positions = lambda positions, context="": positions

    called = {"submit": False}
    dm._submit_order_via_execution_adapter = lambda **kwargs: called.update(submit=True)

    allowed = dm.execute_trim("CLSK", 10, "decision_claw_trim_override:weak_recheck")

    assert allowed is False
    assert called["submit"] is False


def test_execute_trim_uses_journal_entry_age_for_min_hold_when_memory_missing():
    now = datetime.now(timezone.utc)
    dm = _new_dm_stub()
    dm.dry_run = False
    dm.entry_quality_cfg.fresh_entry_exit_cooldown_minutes = 0
    dm.position_entries = {}
    dm.trade_journal = SimpleNamespace(
        trades=[
            {
                "symbol": "CLSK",
                "trade_type": "entry",
                "entry_status": "filled",
                "entry_time": now - timedelta(minutes=36),
                "filled_quantity": 349,
                "side": "buy",
                "action": "buy",
            }
        ]
    )
    dm._now_utc = lambda: now
    dm._hold_minutes = lambda symbol: 999
    dm._min_hold_minutes_for_trim = 120
    dm._last_trim_time = {}
    dm._trim_count_today = {}
    dm._last_scale_time = {}
    dm.get_current_price = lambda symbol: 12.50
    dm.get_positions = lambda: [SimpleNamespace(symbol="CLSK", qty="349")]
    dm._validated_positions = lambda positions, context="": positions
    dm._allow_fast_trim_on_support_break = lambda *args, **kwargs: (False, "fresh")

    called = {"submit": False}
    dm._submit_order_via_execution_adapter = lambda **kwargs: called.update(submit=True)

    allowed = dm.execute_trim("CLSK", 349, "decision_claw_trim_override:weak_recheck")

    assert allowed is False
    assert called["submit"] is False


def test_execute_exit_allows_critical_hard_stop_during_fresh_cooldown():
    dm = _new_dm_stub()
    dm.dry_run = False
    dm.entry_quality_cfg.fresh_entry_exit_cooldown_minutes = 90
    dm.position_entries = {"AAPL": datetime.now(timezone.utc) - timedelta(minutes=12)}
    dm.get_current_price = lambda symbol: 100.0
    dm.client = SimpleNamespace(get_orders=lambda request: [])
    dm.day_tracker = SimpleNamespace(record_day_trade=lambda **kwargs: None)
    dm._queue_sequential_shadow_event = lambda **kwargs: None
    dm._record_execution_attempt = lambda *args, **kwargs: None
    dm._record_execution_success = lambda *args, **kwargs: None
    dm._record_execution_failure = lambda *args, **kwargs: None
    dm._validate_order = lambda symbol, qty, side, allow_open_sell=False: (True, "")
    dm._acquire_order_submission_guard = lambda *args, **kwargs: (True, "", "guard-1")
    dm._release_order_submission_guard = lambda *args, **kwargs: None
    dm._validated_positions = lambda positions, context="": positions
    dm.get_positions = lambda: [SimpleNamespace(symbol="AAPL", qty="10")]
    dm._resolve_execution_atr = lambda **kwargs: 1.0
    dm._submit_order_via_execution_adapter = lambda **kwargs: SimpleNamespace(
        id="ord-hard-stop",
        status="submitted",
    )

    allowed = dm.execute_exit("AAPL", 10, "CRITICAL: hard stop")

    assert allowed is True


def test_execute_exit_blocks_recently_scaled_noncritical_exit():
    dm = _new_dm_stub()
    dm.dry_run = False
    dm.entry_quality_cfg.fresh_entry_exit_cooldown_minutes = 0
    dm._last_scale_time = {"AAPL": datetime.now() - timedelta(minutes=20)}
    dm.get_current_price = lambda symbol: 100.0
    dm._acquire_order_submission_guard = lambda *args, **kwargs: (True, "", "guard-1")
    dm._release_order_submission_guard = lambda *args, **kwargs: None

    called = {"submit": False}
    dm._submit_order_via_execution_adapter = lambda **kwargs: called.update(submit=True)

    allowed = dm.execute_exit("AAPL", 10, "advisor_recheck_exit")

    assert allowed is False


def test_defensive_screen_can_capture_replay_orders_in_dry_run(monkeypatch):
    dm = _new_dm_stub()
    dm.dry_run = True
    dm.inverse_etf_manager = SimpleNamespace(
        get_instrument_profile=lambda ticker: {
            "entry_size_multiplier": 1.0,
            "leverage": 1,
        }
    )
    dm._refresh_entry_authority_state = lambda positions: {
        "state": "inverse_fast",
        "reason": "test",
    }
    dm._effective_market_regime = lambda: "SELL_OFF"
    dm._market_posture = lambda positions=None: {"posture": "capital_preservation"}
    dm._minutes_since_market_open = lambda now: 5
    dm._last_defensive_screen = None
    dm._has_open_buy_order = lambda ticker: (False, "")
    dm._record_hedge_decision = lambda **kwargs: None
    dm._capture_defensive_orders_in_dry_run = True
    captured = []
    dm._submit_order_via_execution_adapter = lambda **kwargs: (
        captured.append(kwargs) or SimpleNamespace(id="dry-run-capture")
    )

    class FakeScreener:
        def __init__(self, *args, **kwargs):
            pass

        def screen_universe(self, **kwargs):
            return [
                {
                    "ticker": "SH",
                    "signal": "ENTRY",
                    "entry_price": 10.0,
                    "composite_score": 90,
                }
            ]

    monkeypatch.setitem(
        sys.modules,
        "autotrade.signals.inverse_etf_screener",
        SimpleNamespace(InverseETFScreener=FakeScreener),
    )
    monkeypatch.setitem(
        sys.modules,
        "autotrade.utils.financial_db",
        SimpleNamespace(FinancialDB=lambda: SimpleNamespace()),
    )
    monkeypatch.setattr(
        day_manager_mod,
        "CONFIG",
        SimpleNamespace(inverse_etf_hedging=SimpleNamespace(position_size=1000.0)),
    )

    dm._run_defensive_screen([])

    assert captured
    assert captured[0]["symbol"] == "SH"
    assert captured[0]["context"] == "inverse_fast_entry"


def test_trim_oversized_position_blocks_fresh_wind_down_reset():
    dm = _new_dm_stub()
    dm.dry_run = False
    dm.entry_quality_cfg.fresh_entry_exit_cooldown_minutes = 90
    dm.position_entries = {"AAPL": datetime.now(timezone.utc) - timedelta(minutes=20)}
    dm._last_trim_time = {}
    dm._last_trim_pnl = {}
    dm._trim_count_today = {}

    called = {"submit": False}
    dm._submit_order_via_execution_adapter = lambda **kwargs: called.update(submit=True)

    allowed = dm.trim_oversized_position(
        {
            "symbol": "AAPL",
            "current_qty": 100,
            "target_qty": 50,
            "trim_qty": 50,
            "market_value": 5_000.0,
            "pnl_pct": 4.0,
            "price": 50.0,
            "reason": "WIND_DOWN oversize reset",
        }
    )

    assert allowed is False
    assert called["submit"] is False


def test_scale_into_winner_blocks_when_position_health_says_trim():
    dm = _new_dm_stub()
    dm.position_health = {"CALM": {"action": "trim"}}

    allowed = dm.scale_into_winner(
        {
            "symbol": "CALM",
            "additional_qty": 5,
            "current_qty": 10,
            "price": 50.0,
            "pnl_pct": 2.5,
            "reason": "winner add",
        }
    )

    assert allowed is False


def test_trim_oversized_position_blocks_recently_scaled_position():
    dm = _new_dm_stub()
    dm.dry_run = False
    dm._hold_minutes = lambda symbol: 180
    dm._last_trim_time = {}
    dm._trim_count_today = {}
    dm._last_scale_time = {"APA": datetime.now() - timedelta(minutes=20)}

    called = {"submit": False}
    dm._submit_order_via_execution_adapter = lambda **kwargs: called.update(submit=True)

    allowed = dm.trim_oversized_position(
        {
            "symbol": "APA",
            "trim_qty": 10,
            "target_qty": 20,
            "current_qty": 30,
            "market_value": 3000.0,
            "price": 100.0,
            "pnl_pct": -1.0,
            "reason": "Oversized loser: $3000 > $2000",
        }
    )

    assert allowed is False
    assert called["submit"] is False


def test_scale_into_winner_blocks_during_wind_down():
    dm = _new_dm_stub()
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.WIND_DOWN

    allowed = dm.scale_into_winner(
        {
            "symbol": "APA",
            "additional_qty": 5,
            "current_qty": 10,
            "price": 50.0,
            "pnl_pct": 3.0,
            "reason": "winner add",
        }
    )

    assert allowed is False


def test_position_add_block_reason_allows_relative_strength_add_during_bull_lock():
    dm = _new_dm_stub()
    dm.youtube_context = {
        "resolved_regime": {
            "regime": "SELL_OFF",
            "allow_new_longs": False,
        }
    }
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm._entry_authority_state["state"] = "bull_lock"
    dm._strength_reentry_state["CF"] = {"strength_reentry_ready": True}

    reason = dm._position_add_block_reason("CF")

    assert reason == ""


def test_position_add_block_reason_keeps_bull_lock_for_weak_relative_symbol():
    dm = _new_dm_stub()
    dm.youtube_context = {
        "resolved_regime": {
            "regime": "SELL_OFF",
            "allow_new_longs": False,
        }
    }
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm._entry_authority_state["state"] = "bull_lock"
    dm._refresh_entry_authority_state = lambda positions=None, now_local=None: (
        dm._entry_authority_state
    )
    dm._alpha_add_override_snapshot = lambda symbol: {
        "relative_strength": 0.0,
        "is_proven_leader": False,
        "is_elite": False,
        "strength_reentry_ready": False,
    }
    dm._bearish_bias_allows_relative_strength_add = lambda symbol: False

    reason = dm._position_add_block_reason("CF")

    assert reason == "entry_authority_state_bull_lock"


def test_scan_strength_reentry_candidates_returns_reclaim_setup_and_caches_held():
    dm = _new_dm_stub()
    dm.signals = [
        {"ticker": "CF", "confidence": 62.0, "score": 62.0, "action": "watch"},
    ]
    dm._watched_universe_tickers = {"CF"}
    dm._entry_authority_state["state"] = "bull_lock"
    dm.youtube_context = {"regime": "SELL_OFF"}
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING

    benchmark_bars = _build_strength_reentry_bars(
        open_price=100.0,
        first_hour_gain_pct=-1.1,
        pullback_pct=0.4,
        reclaim_gain_pct=-0.7,
        reclaim_volume_multiplier=0.9,
    )
    bars_map = {
        "CF": _build_strength_reentry_bars(),
        "MUR": _build_strength_reentry_bars(reclaim_gain_pct=2.6),
        "SPY": benchmark_bars,
        "QQQ": benchmark_bars,
        "IWM": benchmark_bars,
    }
    original_intraday_available = day_manager_mod.INTRADAY_PROVIDER_AVAILABLE
    original_batch = day_manager_mod.get_intraday_bars_batch
    day_manager_mod.INTRADAY_PROVIDER_AVAILABLE = True
    day_manager_mod.get_intraday_bars_batch = lambda *args, **kwargs: bars_map
    try:
        candidates = dm._scan_strength_reentry_candidates(["MUR"])
    finally:
        day_manager_mod.INTRADAY_PROVIDER_AVAILABLE = original_intraday_available
        day_manager_mod.get_intraday_bars_batch = original_batch

    assert len(candidates) == 1
    assert candidates[0]["ticker"] == "CF"
    assert candidates[0]["entry_source"] == "strength_reentry"
    assert candidates[0]["strength_reentry_ready"] is True
    assert dm._strength_reentry_state["MUR"]["held"] is True
    assert dm._strength_reentry_state["MUR"]["strength_reentry_ready"] is True


def test_evaluate_strength_reentry_setup_allows_held_relative_winner_reclaim():
    dm = _new_dm_stub()
    benchmark_ctx = {
        "ready": True,
        "day_return_avg": -1.2,
        "first_hour_avg": -0.4,
        "available_count": 3,
    }
    bars = _build_strength_reentry_bars(
        first_hour_gain_pct=-0.6,
        pullback_pct=1.1,
        reclaim_gain_pct=2.8,
        reclaim_volume_multiplier=1.45,
    )

    held_state = dm._evaluate_strength_reentry_setup(
        symbol="CF",
        signal_data={"ticker": "CF", "confidence": 64.0},
        bars_df=bars,
        benchmark_ctx=benchmark_ctx,
        held=True,
    )
    fresh_state = dm._evaluate_strength_reentry_setup(
        symbol="CF",
        signal_data={"ticker": "CF", "confidence": 64.0},
        bars_df=bars,
        benchmark_ctx=benchmark_ctx,
        held=False,
    )

    assert held_state["strength_reentry_ready"] is True
    assert held_state["strength_reentry_qualification"] == "held_relative_strength"
    assert fresh_state["strength_reentry_ready"] is False
    assert fresh_state["strength_reentry_reason"] == "no_early_leader"


def test_evaluate_strength_reentry_setup_requires_follow_through_for_held_reclaims():
    dm = _new_dm_stub()
    dm.entry_quality_cfg.strength_reentry_follow_through_minutes = 3
    benchmark_ctx = {
        "ready": True,
        "day_return_avg": -1.2,
        "first_hour_avg": -0.4,
        "available_count": 3,
    }
    bars = _build_strength_reentry_bars(
        first_hour_gain_pct=-0.6,
        pullback_pct=1.1,
        reclaim_gain_pct=2.8,
        reclaim_volume_multiplier=1.45,
    )

    pending_state = dm._evaluate_strength_reentry_setup(
        symbol="CF",
        signal_data={"ticker": "CF", "confidence": 64.0},
        bars_df=bars,
        benchmark_ctx=benchmark_ctx,
        held=True,
    )
    ready_state = dm._evaluate_strength_reentry_setup(
        symbol="CF",
        signal_data={"ticker": "CF", "confidence": 64.0},
        bars_df=bars,
        benchmark_ctx=benchmark_ctx,
        held=True,
        prior_state={
            "strength_reentry_reclaim_detected_at": bars.index[-4].isoformat(),
        },
    )

    assert pending_state["strength_reentry_ready"] is False
    assert pending_state["strength_reentry_reason"] == "awaiting_follow_through"
    assert ready_state["strength_reentry_ready"] is True
    assert ready_state["strength_reentry_follow_through_ready_at"]


def test_evaluate_strength_reentry_setup_keeps_held_only_reclaim_path():
    dm = _new_dm_stub()
    benchmark_ctx = {
        "ready": True,
        "day_return_avg": -2.0,
        "first_hour_avg": -0.5,
        "available_count": 3,
    }
    bars = _build_strength_reentry_bars(
        open_price=36.0,
        first_hour_gain_pct=0.5,
        pullback_pct=0.9,
        reclaim_gain_pct=0.75,
        reclaim_volume_multiplier=1.45,
    )
    bars.iloc[-1, bars.columns.get_loc("close")] = 36.35
    bars.iloc[-1, bars.columns.get_loc("high")] = 36.36
    bars.iloc[-1, bars.columns.get_loc("low")] = 36.31
    signal_data = {"ticker": "CTRA", "prior_day_high": 36.42}

    held_state = dm._evaluate_strength_reentry_setup(
        symbol="CTRA",
        signal_data=signal_data,
        bars_df=bars,
        benchmark_ctx=benchmark_ctx,
        held=True,
    )
    fresh_state = dm._evaluate_strength_reentry_setup(
        symbol="CTRA",
        signal_data=signal_data,
        bars_df=bars,
        benchmark_ctx=benchmark_ctx,
        held=False,
    )

    assert held_state["strength_reentry_ready"] is True
    assert held_state["strength_reentry_qualification"] == "held_relative_strength"
    assert fresh_state["strength_reentry_ready"] is False
    assert fresh_state["strength_reentry_reason"] == "no_early_leader"


def test_resolve_entry_authority_allows_strength_reentry_in_bull_lock():
    dm = _new_dm_stub()
    dm._entry_authority_state["state"] = "bull_lock"
    dm.youtube_context = {
        "regime": "SELL_OFF",
        "resolved_regime": {"regime": "SELL_OFF", "allow_new_longs": False},
    }
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm._strength_reentry_state["CF"] = {"strength_reentry_ready": True}

    authority = dm._resolve_entry_authority(
        {"ticker": "CF", "entry_source": "strength_reentry", "confidence": 78.0}
    )

    assert authority["eligible"] is True
    assert authority["strength_reentry_allowed"] is True
    assert authority["reason"] == "strength_reentry_session_override"


def test_resolve_entry_authority_allows_elite_symbol_during_bull_lock():
    dm = _new_dm_stub()
    dm.signals = [
        {
            "ticker": "CENX",
            "score": 96.0,
            "confidence": 96.0,
            "entry_source": "overnight_full_watchlist",
            "plan_score_source": "adjusted_plan_20260331_0829.json",
        }
    ]
    dm._refresh_entry_authority_state = lambda positions=None: {
        "state": "bull_lock",
        "reason": "bearish_bias_pending_live_recovery",
    }
    dm._live_execution_mode = lambda: {
        "entry_authority_state": "bull_lock",
        "entries_allowed": False,
        "strength_reentry_entries_allowed": False,
        "resolved_regime": {"regime": "SELL_OFF", "allow_new_longs": False},
        "inverse_fast_bullish_override": False,
    }

    authority = dm._resolve_entry_authority(
        {
            "ticker": "CENX",
            "score": 96.0,
            "confidence": 96.0,
            "entry_source": "overnight_full_watchlist",
            "plan_score_source": "adjusted_plan_20260331_0829.json",
        }
    )

    assert authority["eligible"] is True
    assert authority["reason"] == ""


def test_check_position_sizes_applies_strength_reentry_add_multiplier():
    dm = _new_dm_stub()
    dm._entries_blocked_by_core_data = lambda: (False, "")
    dm._position_add_block_reason = lambda symbol: ""
    dm._bearish_bias_allows_relative_strength_add = lambda symbol: symbol == "CF"
    dm._strength_reentry_add_size_multiplier = lambda symbol: 0.5
    dm._validated_positions = lambda positions, context="": positions

    positions = [
        SimpleNamespace(
            symbol="CF",
            market_value=1500.0,
            unrealized_plpc=0.12,
            current_price=100.0,
            qty=15,
        )
    ]

    result = dm.check_position_sizes(positions)

    assert len(result["scalable"]) == 1
    assert result["scalable"][0]["additional_qty"] == 2
    assert result["scalable"][0]["entry_context"] == "strength_reentry_add"
    assert "[strength_reentry_add]" in result["scalable"][0]["reason"]


def test_check_position_sizes_downsizes_add_to_position_equity_cap():
    dm = _new_dm_stub()
    dm.last_account_equity = 100_000.0
    dm.config.portfolio.max_position_pct_of_equity = 4.1
    dm._entries_blocked_by_core_data = lambda: (False, "")
    dm._position_add_block_reason = lambda symbol: ""
    dm._bearish_bias_allows_relative_strength_add = lambda symbol: False
    dm._validated_positions = lambda positions, context="": positions
    dm._find_signal_data = lambda symbol: {"score": 71.0}

    positions = [
        SimpleNamespace(
            symbol="ASAN",
            market_value=4_000.0,
            unrealized_plpc=0.12,
            current_price=10.0,
            qty=400,
        )
    ]

    result = dm.check_position_sizes(positions)

    assert len(result["scalable"]) == 1
    assert result["scalable"][0]["additional_qty"] == 10


def test_check_position_sizes_blocks_add_when_position_already_over_equity_cap():
    dm = _new_dm_stub()
    dm.last_account_equity = 100_000.0
    dm.config.portfolio.max_position_pct_of_equity = 4.1
    dm._entries_blocked_by_core_data = lambda: (False, "")
    dm._position_add_block_reason = lambda symbol: ""
    dm._bearish_bias_allows_relative_strength_add = lambda symbol: False
    dm._validated_positions = lambda positions, context="": positions
    dm._find_signal_data = lambda symbol: {"score": 71.0}

    positions = [
        SimpleNamespace(
            symbol="ASAN",
            market_value=4_300.0,
            unrealized_plpc=0.12,
            current_price=10.0,
            qty=430,
        )
    ]

    result = dm.check_position_sizes(positions)

    assert result["scalable"] == []


def test_check_position_sizes_requires_strength_reentry_confirmation_in_bearish_mode():
    dm = _new_dm_stub()
    dm._entries_blocked_by_core_data = lambda: (False, "")
    dm._position_add_block_reason = lambda symbol: ""
    dm._strength_reentry_mode_active = lambda: True
    dm._bearish_bias_allows_relative_strength_add = lambda symbol: False
    dm._validated_positions = lambda positions, context="": positions

    positions = [
        SimpleNamespace(
            symbol="NP",
            market_value=500.0,
            unrealized_plpc=0.08,
            current_price=25.0,
            qty=20,
        )
    ]

    result = dm.check_position_sizes(positions)

    assert result["scalable"] == []


def test_bearish_bias_allows_relative_strength_add_refreshes_state():
    dm = _new_dm_stub()
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm._entry_authority_state["state"] = "bull_lock"
    dm._refresh_entry_authority_state = lambda trigger_research_reset=False: {
        "state": "bull_lock"
    }
    calls = []

    def _state(symbol, *, force_refresh=False, signal_data=None):
        calls.append((symbol, force_refresh))
        return {"strength_reentry_ready": True}

    dm._strength_reentry_current_state = _state

    allowed = dm._bearish_bias_allows_relative_strength_add("CF")

    assert allowed is True
    assert calls == [("CF", False), ("CF", True)]


def test_check_position_sizes_allows_target_zone_strength_reentry_add():
    dm = _new_dm_stub()
    dm._entries_blocked_by_core_data = lambda: (False, "")
    dm._position_add_block_reason = lambda symbol: ""
    dm._strength_reentry_mode_active = lambda: True
    dm._bearish_bias_allows_relative_strength_add = lambda symbol: symbol == "CF"
    dm._strength_reentry_add_size_multiplier = lambda symbol: 0.5
    dm._validated_positions = lambda positions, context="": positions

    positions = [
        SimpleNamespace(
            symbol="CF",
            market_value=1885.0,
            unrealized_plpc=0.047,
            current_price=134.65,
            qty=14,
        )
    ]

    result = dm.check_position_sizes(positions)

    assert len(result["scalable"]) == 1
    assert result["scalable"][0]["entry_context"] == "strength_reentry_add"
    assert result["scalable"][0]["additional_qty"] == 9
    assert "Target-zone winner" in result["scalable"][0]["reason"]


def test_check_position_sizes_trims_wind_down_oversized_winner_without_claw_approval():
    dm = _new_dm_stub()
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.WIND_DOWN
    dm._validated_positions = lambda positions, context="": positions

    positions = [
        SimpleNamespace(
            symbol="APA",
            market_value=4500.0,
            unrealized_plpc=0.05,
            current_price=100.0,
            qty=45,
        )
    ]

    result = dm.check_position_sizes(positions)

    assert result["scalable"] == []
    assert len(result["oversized"]) == 1
    assert result["oversized"][0]["symbol"] == "APA"
    assert result["oversized"][0]["target_qty"] == 20


def test_check_position_sizes_keeps_wind_down_oversized_winner_with_claw_approval():
    dm = _new_dm_stub()
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.WIND_DOWN
    dm._validated_positions = lambda positions, context="": positions
    dm._execution_override_plan = {
        "hold_overnight_map": {
            "APA": {
                "approve_overnight_oversize": True,
                "max_size_multiplier": 2.0,
            }
        }
    }

    positions = [
        SimpleNamespace(
            symbol="APA",
            market_value=4500.0,
            unrealized_plpc=0.05,
            current_price=100.0,
            qty=45,
        )
    ]

    result = dm.check_position_sizes(positions)

    assert result["oversized"] == []
    assert result["scalable"] == []


def test_deployment_floor_status_counts_inverse_exposure():
    dm = _new_dm_stub()
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm._entry_authority_state["state"] = "bull_lock"
    dm.last_account_equity = 10_000.0

    positions = [
        SimpleNamespace(symbol="CF", market_value=2_000.0),
        SimpleNamespace(symbol="SH", market_value=1_500.0),
    ]

    status = dm._deployment_floor_status(positions)

    assert status["active"] is True
    assert status["deployed_pct"] == pytest.approx(0.35)
    assert status["shortfall_value"] == pytest.approx(1500.0)


def test_deployment_floor_status_suppresses_fill_on_strong_green_day():
    dm = _new_dm_stub()
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm._entry_authority_state["state"] = "bull_lock"
    dm.last_account_equity = 10_000.0
    dm._live_benchmark_snapshot = lambda now=None: {
        "available_count": 3,
        "avg_pct_change": 0.8,
    }

    status = dm._deployment_floor_status([])

    assert status["active"] is False
    assert status["suppressed_by_live_market"] is True


def test_enforce_bearish_session_deployment_floor_respects_recent_inverse_trim():
    dm = _new_dm_stub()
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm._entry_authority_state["state"] = "bull_lock"
    dm.last_account_equity = 10_000.0
    dm.dry_run = False
    submitted = []
    dm._submit_order_via_execution_adapter = lambda **kwargs: submitted.append(kwargs)
    dm.get_current_price = lambda symbol: 20.0
    dm._live_benchmark_snapshot = lambda now=None: {
        "available_count": 3,
        "avg_pct_change": -0.5,
    }
    dm.inverse_etf_manager = SimpleNamespace(
        check_hedge_conditions=lambda **kwargs: {
            "should_enter": True,
            "should_exit": False,
            "reason": "risk_off",
            "recommended_etfs": ["RWM"],
        }
    )
    dm._hedge_decision_state = {
        "session_date": datetime.now().strftime("%Y-%m-%d"),
        "last_action": "scale_out",
        "last_symbol": "RWM",
        "last_target_notional": 2000.0,
        "last_decision_at": datetime.now() - timedelta(minutes=5),
    }

    DayManager._enforce_bearish_session_deployment_floor(dm, [])

    assert submitted == []


def test_entry_wave_number_handles_early_runner_sentinel():
    dm = _new_dm_stub()
    dm.entry_wave = "early_runner"

    assert DayManager._entry_wave_number(dm) == 1


def test_manage_inverse_etfs_uses_rebalance_plan_for_entry_size():
    dm = _new_dm_stub()
    dm.dry_run = False
    dm.last_account_equity = 100000.0
    submitted = []
    dm._submit_order_via_execution_adapter = lambda **kwargs: submitted.append(kwargs)
    dm._position_qty = lambda pos: int(getattr(pos, "qty", 0) or 0)
    dm.get_current_price = lambda symbol: 50.0 if symbol == "SH" else 20.0
    dm.inverse_etf_manager = SimpleNamespace(
        check_hedge_conditions=lambda **kwargs: {
            "should_enter": True,
            "should_exit": False,
            "reason": "risk_off",
            "recommended_etfs": ["SH"],
        },
        calculate_current_allocation=lambda positions: 0.0,
        rebalance_hedge=lambda **kwargs: {
            "hedge_symbol": "SH",
            "target_notional": 10000.0,
            "target_allocation_pct": 10.0,
            "current_allocation_pct": 0.0,
            "rebalance_delta_pct": 10.0,
            "action": "increase",
        },
        check_hedge_exit=lambda **kwargs: {"should_exit": False, "reasons": []},
        regime_detector=SimpleNamespace(
            detect_regime=lambda use_cache=True: SimpleNamespace(regime="risk_off")
        ),
    )

    DayManager._manage_inverse_etfs(dm, positions=[])

    assert len(submitted) == 1
    assert submitted[0]["symbol"] == "SH"
    assert submitted[0]["qty"] == 200
    assert submitted[0]["context"] == "hedge_entry"
    assert submitted[0]["order_type_override"] == "market"


def test_manage_inverse_etfs_skips_duplicate_open_buy_order_for_hedge():
    dm = _new_dm_stub()
    dm.dry_run = False
    submitted = []
    dm._submit_order_via_execution_adapter = lambda **kwargs: submitted.append(kwargs)
    dm._position_qty = lambda pos: int(getattr(pos, "qty", 0) or 0)
    dm._has_open_buy_order = lambda symbol: (True, "hedge-buy-1")
    dm.get_current_price = lambda symbol: 50.0
    dm.inverse_etf_manager = SimpleNamespace(
        check_hedge_conditions=lambda **kwargs: {
            "should_enter": True,
            "should_exit": False,
            "reason": "risk_off",
            "recommended_etfs": ["PSQ"],
        },
        calculate_current_allocation=lambda positions: 0.0,
        rebalance_hedge=lambda **kwargs: {
            "hedge_symbol": "PSQ",
            "target_notional": 10000.0,
            "target_allocation_pct": 10.0,
            "current_allocation_pct": 0.0,
            "rebalance_delta_pct": 10.0,
            "action": "increase",
        },
        check_hedge_exit=lambda **kwargs: {"should_exit": False, "reasons": []},
        regime_detector=SimpleNamespace(
            detect_regime=lambda use_cache=True: SimpleNamespace(regime="risk_off")
        ),
    )

    DayManager._manage_inverse_etfs(dm, positions=[])

    assert submitted == []


def test_manage_inverse_etfs_uses_exit_criteria_to_liquidate_positions():
    dm = _new_dm_stub()
    dm.dry_run = False
    submitted = []
    journal_calls = {}
    dm.trade_journal = SimpleNamespace(
        record_exit=lambda *args, **kwargs: (
            journal_calls.update({"args": args, "kwargs": kwargs}) or "trade-1"
        ),
        save=lambda: None,
    )
    dm._submit_order_via_execution_adapter = lambda **kwargs: (
        submitted.append(kwargs)
        or SimpleNamespace(id="hedge-exit-1", status="submitted")
    )
    dm._position_qty = lambda pos: int(getattr(pos, "qty", 0) or 0)
    dm.get_current_price = lambda symbol: 18.0 if symbol == "^VIX" else 500.0
    dm.inverse_etf_manager = SimpleNamespace(
        check_hedge_conditions=lambda **kwargs: {
            "should_enter": False,
            "should_exit": False,
            "reason": "neutral",
            "recommended_etfs": [],
        },
        calculate_current_allocation=lambda positions: 12.0,
        rebalance_hedge=lambda **kwargs: {
            "hedge_symbol": "SH",
            "target_notional": 0.0,
            "target_allocation_pct": 0.0,
            "current_allocation_pct": 12.0,
            "rebalance_delta_pct": -12.0,
            "action": "decrease",
        },
        check_hedge_exit=lambda **kwargs: {
            "should_exit": True,
            "reasons": ["regime_recovery", "volatility_mean_reversion"],
        },
        regime_detector=SimpleNamespace(
            detect_regime=lambda use_cache=True: SimpleNamespace(regime="neutral")
        ),
    )
    pos = SimpleNamespace(symbol="SH", qty=40, market_value=-2000.0, entry_at=None)

    DayManager._manage_inverse_etfs(dm, positions=[pos])

    assert len(submitted) == 1
    assert submitted[0]["symbol"] == "SH"
    assert submitted[0]["side"] == "sell"
    assert submitted[0]["context"] == "hedge_regime_flip"
    assert submitted[0]["order_type_override"] == "market"
    assert journal_calls["kwargs"]["symbol"] == "SH"
    assert journal_calls["kwargs"]["order_id"] == "hedge-exit-1"


def test_manage_inverse_etfs_skips_duplicate_open_sell_order_for_exit():
    dm = _new_dm_stub()
    dm.dry_run = False
    submitted = []
    dm._submit_order_via_execution_adapter = lambda **kwargs: submitted.append(kwargs)
    dm._position_qty = lambda pos: int(getattr(pos, "qty", 0) or 0)
    dm._has_open_sell_order = lambda symbol: (True, "hedge-sell-1")
    dm.get_current_price = lambda symbol: 18.0 if symbol == "^VIX" else 500.0
    dm.inverse_etf_manager = SimpleNamespace(
        check_hedge_conditions=lambda **kwargs: {
            "should_enter": False,
            "should_exit": False,
            "reason": "neutral",
            "recommended_etfs": [],
        },
        calculate_current_allocation=lambda positions: 12.0,
        rebalance_hedge=lambda **kwargs: {
            "hedge_symbol": "SH",
            "target_notional": 0.0,
            "target_allocation_pct": 0.0,
            "current_allocation_pct": 12.0,
            "rebalance_delta_pct": -12.0,
            "action": "decrease",
        },
        check_hedge_exit=lambda **kwargs: {
            "should_exit": True,
            "reasons": ["regime_recovery"],
        },
        regime_detector=SimpleNamespace(
            detect_regime=lambda use_cache=True: SimpleNamespace(regime="neutral")
        ),
    )
    pos = SimpleNamespace(symbol="PSQ", qty=40, market_value=-2000.0, entry_at=None)

    DayManager._manage_inverse_etfs(dm, positions=[pos])

    assert submitted == []


def test_manage_inverse_etfs_scales_out_when_enter_and_exit_conflict():
    dm = _new_dm_stub()
    dm.dry_run = False
    submitted = []
    dm._submit_order_via_execution_adapter = lambda **kwargs: submitted.append(kwargs)
    dm._position_qty = lambda pos: int(getattr(pos, "qty", 0) or 0)
    dm.get_current_price = lambda symbol: 22.0 if symbol == "^VIX" else 31.5
    dm.inverse_etf_manager = SimpleNamespace(
        check_hedge_conditions=lambda **kwargs: {
            "should_enter": True,
            "should_exit": False,
            "reason": "risk_off",
            "recommended_etfs": ["PSQ"],
        },
        calculate_current_allocation=lambda positions: 12.0,
        rebalance_hedge=lambda **kwargs: {
            "hedge_symbol": "PSQ",
            "target_notional": 10000.0,
            "target_allocation_pct": 10.0,
            "current_allocation_pct": 12.0,
            "rebalance_delta_pct": -2.0,
            "action": "decrease",
        },
        check_hedge_exit=lambda **kwargs: {
            "should_exit": True,
            "reasons": ["regime_recovery"],
        },
        regime_detector=SimpleNamespace(
            detect_regime=lambda use_cache=True: SimpleNamespace(regime="neutral")
        ),
        evaluate_intraday_reversal=lambda symbol, bars=None: {
            "should_scale_out": False,
            "should_exit": False,
            "reasons": [],
            "profile": {"scale_out_fraction": 0.33},
        },
    )
    pos = SimpleNamespace(symbol="PSQ", qty=40, market_value=-2000.0, entry_at=None)

    DayManager._manage_inverse_etfs(dm, positions=[pos])

    assert len(submitted) == 1
    assert submitted[0]["symbol"] == "PSQ"
    assert submitted[0]["side"] == "sell"
    assert submitted[0]["context"] == "inverse_recovery_scale_out"


def test_calculate_position_health_holds_hedges_in_single_decision_mode():
    dm = _new_dm_stub()
    dm.signals = []
    dm._is_hedge_symbol = lambda symbol: True
    dm._live_execution_mode = lambda: {"hedge_automation_mode": "single_decision"}

    health = dm.calculate_position_health(
        SimpleNamespace(
            symbol="PSQ",
            avg_entry_price=31.0,
            current_price=31.1,
            qty=10,
            unrealized_plpc=0.001,
        )
    )

    assert health["action"] == "hold"
    assert health["advisor_type"] == "hedge_manager_single_decision"


def test_calculate_position_health_forces_hard_stop_exit():
    dm = _new_dm_stub()
    dm.signals = []
    dm.exit_manager = SimpleNamespace(check_exits=lambda positions: [])
    dm.position_advisor = None
    dm.runtime_risk_gate = None
    dm._hold_minutes = lambda symbol: 180

    health = dm.calculate_position_health(
        SimpleNamespace(
            symbol="ACLS",
            avg_entry_price=115.28,
            current_price=84.08,
            qty=17,
            unrealized_plpc=-0.27062,
        )
    )

    assert health["action"] == "exit"
    assert health["hard_stop_forced_exit"] is True
    assert health["failsafe_forced_exit"] is True
    assert health["hard_stop_pct"] == pytest.approx(-15.0)


def test_run_defensive_screen_allows_inverse_fast_entries_after_first_minute(
    monkeypatch,
):
    dm = _new_dm_stub()
    dm.dry_run = False
    submitted = []
    dm._submit_order_via_execution_adapter = lambda **kwargs: submitted.append(
        SimpleNamespace(id="ord-1", **kwargs)
    )
    dm._has_open_buy_order = lambda symbol: (False, "")
    dm._minutes_since_market_open = lambda now=None: 1
    dm._refresh_entry_authority_state = lambda positions=None: {
        "state": "inverse_fast",
        "reason": "crash_open_confirmed",
        "inverse_fast_entries_taken": 0,
    }
    dm._get_entry_authority_state = lambda: dm._entry_authority_state
    dm._entry_authority_state["state"] = "inverse_fast"
    dm.inverse_etf_manager = SimpleNamespace(
        get_instrument_profile=lambda symbol: {
            "entry_size_multiplier": 0.45,
            "leverage": 3,
        }
    )

    class _FakeScreener:
        def __init__(self, *args, **kwargs):
            pass

        def screen_universe(
            self,
            regime,
            portfolio_holdings=None,
            entry_mode="",
            minutes_since_open=None,
            **kwargs,
        ):
            assert entry_mode == "inverse_fast"
            assert minutes_since_open == 1
            return [
                {
                    "ticker": "SQQQ",
                    "signal": "ENTRY",
                    "entry_price": 20.0,
                    "composite_score": 91,
                },
                {
                    "ticker": "SPXU",
                    "signal": "ENTRY",
                    "entry_price": 25.0,
                    "composite_score": 88,
                },
            ]

    screener_mod = __import__(
        "autotrade.signals.inverse_etf_screener",
        fromlist=["InverseETFScreener"],
    )
    db_mod = __import__(
        "autotrade.utils.financial_db",
        fromlist=["FinancialDB"],
    )
    monkeypatch.setattr(
        screener_mod,
        "InverseETFScreener",
        _FakeScreener,
    )
    monkeypatch.setattr(db_mod, "FinancialDB", lambda: object())

    DayManager._run_defensive_screen(dm, positions=[])

    assert len(submitted) == 2
    assert {order.symbol for order in submitted} == {"SQQQ", "SPXU"}
    assert all(order.context == "inverse_fast_entry" for order in submitted)


def test_run_defensive_screen_skips_selective_bad_day_without_inverse_fast(
    monkeypatch,
):
    dm = _new_dm_stub()
    dm.dry_run = False
    submitted = []
    dm._submit_order_via_execution_adapter = lambda **kwargs: submitted.append(
        SimpleNamespace(id="ord-risk-1", **kwargs)
    )
    dm._has_open_buy_order = lambda symbol: (False, "")
    dm._minutes_since_market_open = lambda now=None: 5
    dm._effective_market_regime = lambda: "RISK_OFF"
    dm._refresh_entry_authority_state = lambda positions=None: {
        "state": "bull_lock",
        "reason": "bearish_bias_pending_live_recovery",
        "inverse_fast_entries_taken": 0,
    }
    dm._market_posture = lambda positions=None: {
        "posture": "cautious_selective_longs",
        "reason": "bearish_regime_selective_longs",
    }
    dm._get_entry_authority_state = lambda: dm._entry_authority_state
    dm._entry_authority_state["state"] = "bull_lock"
    dm.inverse_etf_manager = SimpleNamespace(
        get_instrument_profile=lambda symbol: {
            "entry_size_multiplier": 1.0,
            "leverage": 1,
        }
    )

    class _FakeScreener:
        def __init__(self, *args, **kwargs):
            pass

        def screen_universe(
            self,
            regime,
            portfolio_holdings=None,
            entry_mode="",
            minutes_since_open=None,
            **kwargs,
        ):
            return [
                {
                    "ticker": "RWM",
                    "signal": "ENTRY",
                    "entry_price": 20.0,
                    "composite_score": 78,
                }
            ]

    screener_mod = __import__(
        "autotrade.signals.inverse_etf_screener",
        fromlist=["InverseETFScreener"],
    )
    db_mod = __import__(
        "autotrade.utils.financial_db",
        fromlist=["FinancialDB"],
    )
    monkeypatch.setattr(screener_mod, "InverseETFScreener", _FakeScreener)
    monkeypatch.setattr(db_mod, "FinancialDB", lambda: object())

    DayManager._run_defensive_screen(dm, positions=[])

    assert submitted == []


def test_run_defensive_screen_skips_first_open_minute():
    dm = _new_dm_stub()
    dm.dry_run = False
    submitted = []
    dm._submit_order_via_execution_adapter = lambda **kwargs: submitted.append(kwargs)
    dm._minutes_since_market_open = lambda now=None: 0
    dm._refresh_entry_authority_state = lambda positions=None: {"state": "inverse_fast"}
    dm.inverse_etf_manager = SimpleNamespace(
        get_instrument_profile=lambda symbol: {
            "entry_size_multiplier": 0.45,
            "leverage": 3,
        }
    )

    DayManager._run_defensive_screen(dm, positions=[])

    assert submitted == []


def test_run_defensive_screen_runs_premarket_without_submitting_entries(
    monkeypatch,
):
    dm = _new_dm_stub()
    dm.dry_run = False
    submitted = []
    calls = []
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.PREMARKET
    dm._submit_order_via_execution_adapter = lambda **kwargs: submitted.append(kwargs)
    dm._has_open_buy_order = lambda symbol: (False, "")
    dm._minutes_since_market_open = lambda now=None: -30
    dm._refresh_entry_authority_state = lambda positions=None: {
        "state": "inverse_fast",
        "reason": "premarket_probe",
        "inverse_fast_entries_taken": 0,
    }
    dm.inverse_etf_manager = SimpleNamespace(
        get_instrument_profile=lambda symbol: {
            "entry_size_multiplier": 0.45,
            "leverage": 3,
        }
    )

    class _FakeScreener:
        def __init__(self, *args, **kwargs):
            pass

        def screen_universe(self, *args, **kwargs):
            calls.append(kwargs)
            return [
                {
                    "ticker": "SQQQ",
                    "signal": "ENTRY",
                    "entry_price": 20.0,
                    "composite_score": 91,
                }
            ]

    screener_mod = __import__(
        "autotrade.signals.inverse_etf_screener",
        fromlist=["InverseETFScreener"],
    )
    db_mod = __import__(
        "autotrade.utils.financial_db",
        fromlist=["FinancialDB"],
    )
    monkeypatch.setattr(screener_mod, "InverseETFScreener", _FakeScreener)
    monkeypatch.setattr(db_mod, "FinancialDB", lambda: object())

    DayManager._run_defensive_screen(dm, positions=[])

    assert calls, "premarket pass should invoke the inverse ETF screener"
    assert submitted == []


def test_run_defensive_screen_runs_neutral_premarket_probe(monkeypatch):
    dm = _new_dm_stub()
    dm.dry_run = False
    calls = []
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.PREMARKET
    dm._submit_order_via_execution_adapter = lambda **kwargs: pytest.fail(
        "premarket probe must not submit entries"
    )
    dm._minutes_since_market_open = lambda now=None: -30
    dm._refresh_entry_authority_state = lambda positions=None: {
        "state": "open",
        "reason": "neutral_premarket_probe",
        "inverse_fast_entries_taken": 0,
    }
    dm._market_posture = lambda positions=None: {"posture": "normal_risk"}
    dm._live_benchmark_snapshot = lambda: {}
    dm._load_resolved_regime_context = lambda: {"breadth_pct_positive": 54.4}
    dm.inverse_etf_manager = SimpleNamespace(
        get_instrument_profile=lambda symbol: {
            "entry_size_multiplier": 0.45,
            "leverage": 3,
        }
    )

    class _FakeScreener:
        def __init__(self, *args, **kwargs):
            pass

        def screen_universe(self, *args, **kwargs):
            calls.append(kwargs)
            return []

    screener_mod = __import__(
        "autotrade.signals.inverse_etf_screener",
        fromlist=["InverseETFScreener"],
    )
    db_mod = __import__(
        "autotrade.utils.financial_db",
        fromlist=["FinancialDB"],
    )
    monkeypatch.setattr(screener_mod, "InverseETFScreener", _FakeScreener)
    monkeypatch.setattr(db_mod, "FinancialDB", lambda: object())

    DayManager._run_defensive_screen(dm, positions=[])

    assert calls, "neutral premarket probe should invoke the inverse ETF screener"
    assert dm._last_inverse_etf_screen_summary["results"] == 0


def test_run_defensive_screen_runs_core_low_breadth_probe(monkeypatch):
    dm = _new_dm_stub()
    calls = []
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm._minutes_since_market_open = lambda now=None: 90
    dm._refresh_entry_authority_state = lambda positions=None: {
        "state": "open",
        "reason": "core_low_breadth_probe",
        "inverse_fast_entries_taken": 0,
    }
    dm._market_posture = lambda positions=None: {"posture": "normal_risk"}
    dm._live_benchmark_snapshot = lambda: {
        "avg_pct_change": -0.1,
        "benchmarks": {"SPY": {"pct_change": -0.1}},
        "crash_confirmed": False,
    }
    dm._load_resolved_regime_context = lambda: {"breadth_pct_positive": 38.0}
    dm.inverse_etf_manager = SimpleNamespace(
        get_instrument_profile=lambda symbol: {
            "entry_size_multiplier": 0.45,
            "leverage": 3,
        }
    )

    class _FakeScreener:
        def __init__(self, *args, **kwargs):
            pass

        def screen_universe(self, *args, **kwargs):
            calls.append(kwargs)
            return []

    screener_mod = __import__(
        "autotrade.signals.inverse_etf_screener",
        fromlist=["InverseETFScreener"],
    )
    db_mod = __import__(
        "autotrade.utils.financial_db",
        fromlist=["FinancialDB"],
    )
    monkeypatch.setattr(screener_mod, "InverseETFScreener", _FakeScreener)
    monkeypatch.setattr(db_mod, "FinancialDB", lambda: object())

    DayManager._run_defensive_screen(dm, positions=[])

    assert calls, "low-breadth core session should probe inverse ETF screen"
    assert calls[0]["breadth_pct_positive"] == 38.0


def test_write_intraday_market_analysis_creates_cycle_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dm = _new_dm_stub()
    dm._load_resolved_regime_context = lambda: {"breadth_pct_positive": 54.4}
    dm._effective_market_regime = lambda: "NEUTRAL"
    dm.get_current_price = lambda symbol: 27.1 if symbol == "^VIX" else 100.0
    dm._get_previous_close = lambda symbol: 100.1 if symbol == "SPY" else 100.0
    dm._minutes_since_market_open = lambda now=None: 120
    dm._last_short_engine_telemetry_entry = {"signals_generated": 0}
    dm._last_inverse_etf_screen_summary = {"entry_candidates": 0}
    dm._session_state = {"trim_history": {}}

    DayManager._write_intraday_market_analysis(
        dm,
        [
            SimpleNamespace(
                symbol="ASAN",
                unrealized_pl=100.0,
                unrealized_plpc=0.05,
                market_value=2000.0,
            )
        ],
    )

    path = tmp_path / "data" / "intraday_analysis.json"
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["market_context"]["regime_label"] == "DISPERSION"
    assert payload["execution_diagnostics"]["short_engine_ran"] is True


def test_refresh_entry_authority_state_enters_inverse_fast_after_open_confirmation():
    dm = _new_dm_stub()
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm._live_benchmark_snapshot = lambda now=None: {
        "minutes_since_open": 2,
        "crash_confirmed": True,
        "recovery_confirmed": False,
    }

    state = DayManager._refresh_entry_authority_state(dm, positions=[])

    assert state["state"] == "inverse_fast"
    assert state["reason"] == "crash_open_confirmed"


def test_build_feature_pipeline_input_flattens_multiindex_columns():
    dm = _new_dm_stub()
    dm.feature_engineering_cfg = SimpleNamespace(lookback_bars=10)

    multi = pd.MultiIndex.from_tuples(
        [
            ("price", "Date"),
            ("price", "Open"),
            ("price", "High"),
            ("price", "Low"),
            ("price", "Close"),
            ("price", "Volume"),
        ]
    )
    raw = pd.DataFrame(
        [
            ["2026-03-27", 10.0, 11.0, 9.0, 10.5, 1000],
            ["2026-03-28", 10.5, 11.5, 10.0, 11.0, 1500],
        ],
        columns=multi,
    )
    provider = SimpleNamespace(get_ticker_data=lambda *args, **kwargs: raw)

    with patch(
        "autotrade.utils.local_data_provider.get_provider", return_value=provider
    ):
        frame = DayManager._build_feature_pipeline_input(dm, ["ABC"])

    assert {"symbol", "date", "open", "high", "low", "close", "volume"}.issubset(
        frame.columns
    )
    assert frame["symbol"].tolist() == ["ABC", "ABC"]


def test_build_feature_pipeline_input_restores_symbol_when_normalizer_drops_it():
    dm = _new_dm_stub()
    dm.feature_engineering_cfg = SimpleNamespace(lookback_bars=10)

    multi = pd.MultiIndex.from_tuples(
        [
            ("price", "Date"),
            ("price", "Open"),
            ("price", "High"),
            ("price", "Low"),
            ("price", "Close"),
            ("price", "Volume"),
        ]
    )
    raw = pd.DataFrame(
        [
            ["2026-03-27", 10.0, 11.0, 9.0, 10.5, 1000],
            ["2026-03-28", 10.5, 11.5, 10.0, 11.0, 1500],
        ],
        columns=multi,
    )
    provider = SimpleNamespace(get_ticker_data=lambda *args, **kwargs: raw)

    original_normalizer = DayManager._normalize_feature_input_column_name

    def _dropping_symbol(column):
        normalized = original_normalizer(column)
        return "ticker" if normalized == "symbol" else normalized

    with patch(
        "autotrade.utils.local_data_provider.get_provider", return_value=provider
    ):
        with patch.object(
            DayManager,
            "_normalize_feature_input_column_name",
            side_effect=_dropping_symbol,
        ):
            frame = DayManager._build_feature_pipeline_input(dm, ["ABC"])

    assert {"symbol", "date", "open", "high", "low", "close", "volume"}.issubset(
        frame.columns
    )
    assert frame["symbol"].tolist() == ["ABC", "ABC"]


def test_build_signal_pipeline_price_data_includes_symbol_column():
    dm = _new_dm_stub()
    dm.feature_engineering_cfg = SimpleNamespace(lookback_bars=10)

    raw = pd.DataFrame(
        {
            "Date": ["2026-03-27", "2026-03-28"],
            "Open": [10.0, 10.5],
            "High": [11.0, 11.5],
            "Low": [9.5, 10.0],
            "Close": [10.6, 11.1],
            "Volume": [1000, 1200],
        }
    )
    provider = SimpleNamespace(get_ticker_data=lambda *args, **kwargs: raw)

    with patch(
        "autotrade.utils.local_data_provider.get_provider", return_value=provider
    ):
        frame = DayManager._build_signal_pipeline_price_data(dm, ["ABC"])

    assert frame is not None
    assert {
        "ticker",
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }.issubset(frame.columns)
    assert frame["symbol"].tolist() == ["ABC", "ABC"]


def test_normalize_feature_input_column_name_prefers_field_over_ticker_symbol():
    assert (
        DayManager._normalize_feature_input_column_name(("Adj Close", "NAVN"))
        == "adj_close"
    )
    assert DayManager._normalize_feature_input_column_name(("Close", "NAVN")) == "close"
    assert (
        DayManager._normalize_feature_input_column_name(("Volume", "NAVN")) == "volume"
    )


def test_refresh_entry_authority_state_marks_safety_reentry_transition_time():
    dm = _new_dm_stub()
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm._entry_authority_state["recovery_confirmed_streak"] = 2
    dm._live_benchmark_snapshot = lambda now=None: {
        "minutes_since_open": 45,
        "crash_confirmed": False,
        "recovery_confirmed": True,
    }
    dm._entry_authority_state["state"] = "inverse_fast"
    pos = SimpleNamespace(symbol="SQQQ")

    state = DayManager._refresh_entry_authority_state(dm, positions=[pos])

    assert state["state"] == "safety_reentry"
    assert state["reason"] == "live_recovery_confirmed_stable"
    assert state["safety_reentry_refresh"] is not None


def test_midday_trim_requires_new_deterioration_step_same_window():
    dm = _new_dm_stub()
    dm._midday_trim_window_state = {}
    dm._current_midday_trim_window_key = (
        DayManager._current_midday_trim_window_key.__get__(dm, DayManager)
    )

    allowed, reason = DayManager._should_allow_midday_trim(dm, "ABC", -2.0)
    assert allowed is True
    assert reason == ""

    DayManager._record_midday_trim_state(dm, "ABC", -2.0)

    allowed, reason = DayManager._should_allow_midday_trim(dm, "ABC", -2.4)
    assert allowed is False
    assert "has not deteriorated" in reason

    allowed, reason = DayManager._should_allow_midday_trim(dm, "ABC", -4.1)
    assert allowed is True
    assert reason == ""


def test_manage_inverse_etfs_safety_reentry_scales_out_only_once():
    dm = _new_dm_stub()
    dm.dry_run = False
    submitted = []
    dm._submit_order_via_execution_adapter = lambda **kwargs: submitted.append(kwargs)
    dm._position_qty = lambda pos: int(getattr(pos, "qty", 0) or 0)
    dm.get_current_price = lambda symbol: 18.0 if symbol == "^VIX" else 31.5
    dm._refresh_entry_authority_state = lambda positions=None: {
        "state": "safety_reentry",
        "reason": "live_recovery_confirmed",
        "snapshot": {},
    }
    dm.inverse_etf_manager = SimpleNamespace(
        check_hedge_conditions=lambda **kwargs: {
            "should_enter": True,
            "should_exit": False,
            "reason": "risk_off",
            "recommended_etfs": ["PSQ"],
        },
        calculate_current_allocation=lambda positions: 12.0,
        rebalance_hedge=lambda **kwargs: {
            "hedge_symbol": "PSQ",
            "target_notional": 10000.0,
            "target_allocation_pct": 10.0,
            "current_allocation_pct": 12.0,
            "rebalance_delta_pct": -2.0,
            "action": "decrease",
        },
        check_hedge_exit=lambda **kwargs: {"should_exit": False, "reasons": []},
        regime_detector=SimpleNamespace(
            detect_regime=lambda use_cache=True: SimpleNamespace(regime="neutral")
        ),
        evaluate_intraday_reversal=lambda symbol, bars=None: {
            "should_scale_out": False,
            "should_exit": False,
            "reasons": [],
            "profile": {"scale_out_fraction": 0.5},
        },
    )
    pos = SimpleNamespace(symbol="PSQ", qty=40, market_value=-2000.0, entry_at=None)

    DayManager._manage_inverse_etfs(dm, positions=[pos])

    assert len(submitted) == 1
    assert submitted[0]["symbol"] == "PSQ"
    assert submitted[0]["side"] == "sell"
    assert submitted[0]["context"] == "inverse_recovery_scale_out"


def test_set_candidate_universe_override_normalizes_and_clears():
    dm = _new_dm_stub()

    dm.set_candidate_universe_override(
        [" aapl ", "MSFT", "", None],
        reason="overnight_recheck",
        max_new_entries=3,
    )

    assert dm._candidate_universe_override == {"AAPL", "MSFT"}
    assert dm._candidate_universe_override_reason == "overnight_recheck"
    assert dm._max_new_entries_override == 3

    dm.clear_candidate_universe_override()

    assert dm._candidate_universe_override is None
    assert dm._candidate_universe_override_reason == ""
    assert dm._max_new_entries_override is None


def test_set_execution_override_plan_normalizes_and_clears():
    dm = _new_dm_stub()

    dm.set_execution_override_plan(
        {
            "entry_priority_symbols": [" glng ", "YPF", ""],
            "blocked_entry_symbols": [" weak ", None],
            "trim_map": {"viav": {"trim_fraction": 0.5}, "": {}},
            "hold_overnight_map": {
                " apa ": {
                    "approve_overnight_oversize": True,
                    "max_size_multiplier": 2.0,
                }
            },
            "exit_symbols": [" dvn "],
            "wave_action": "hold_wave",
            "deployment_request": {
                "mode": "long_exception",
                "symbols": [" glng "],
                "max_new_entries": 1,
                "reason": "dead capital",
            },
            "symbol_reviews": {
                " glng ": {"decision": "approve", "accepted": True, "thesis": "clean"}
            },
        },
        reason="market_state_override",
    )

    assert dm._execution_override_plan["entry_priority_symbols"] == ["GLNG", "YPF"]
    assert dm._execution_override_plan["blocked_entry_symbols"] == ["WEAK"]
    assert dm._execution_override_plan["trim_map"] == {"VIAV": {"trim_fraction": 0.5}}
    assert dm._execution_override_plan["hold_overnight_map"]["APA"] == {
        "approve_overnight_oversize": True,
        "max_size_multiplier": 2.0,
    }
    assert dm._execution_override_plan["exit_symbols"] == ["DVN"]
    assert dm._execution_override_plan["wave_action"] == "hold_wave"
    assert dm._execution_override_plan["deployment_request"]["mode"] == "long_exception"
    assert dm._execution_override_plan["deployment_request"]["symbols"] == ["GLNG"]
    assert dm._execution_override_plan["symbol_reviews"]["GLNG"]["accepted"] is True
    assert dm._execution_override_plan["reason"] == "market_state_override"

    dm.clear_execution_override_plan()

    assert dm._execution_override_plan["entry_priority_symbols"] == []
    assert dm._execution_override_plan["blocked_entry_symbols"] == []
    assert dm._execution_override_plan["trim_map"] == {}
    assert dm._execution_override_plan["hold_overnight_map"] == {}
    assert dm._execution_override_plan["exit_symbols"] == []
    assert dm._execution_override_plan["wave_action"] == ""
    assert dm._execution_override_plan["deployment_request"] == {}
    assert dm._execution_override_plan["symbol_reviews"] == {}


def test_entries_blocked_by_regime_allows_valid_decision_claw_long_exception():
    dm = _new_dm_stub()
    dm.youtube_context = {"resolved_regime": {"allow_new_longs": False}}
    dm._load_resolved_regime_context = lambda: {"allow_new_longs": False}
    dm._is_hedge_symbol = lambda symbol: False
    dm._refresh_entry_authority_state = lambda: {
        "state": "bull_lock",
        "snapshot": {"avg_pct_change": -1.2, "available_count": 3},
    }
    dm._execution_override_plan = {
        "deployment_request": {
            "mode": "long_exception",
            "symbols": ["GLNG"],
            "max_new_entries": 1,
            "reason": "dead capital",
        },
        "symbol_reviews": {
            "GLNG": {
                "decision": "approve",
                "accepted": True,
                "thesis": "reclaim holding",
            }
        },
    }
    dm._find_signal_data = lambda symbol: {
        "ticker": symbol,
        "current_price": 10.5,
        "prev_close": 10.0,
        "volume_ratio": 1.6,
    }
    dm.get_current_price = lambda symbol: 10.5
    dm._check_intraday_momentum = lambda symbol, avg_volume=0.0: {
        "above_vwap": True,
        "volume_ratio": 1.6,
    }

    blocked, reason = DayManager._entries_blocked_by_regime(dm, "GLNG")

    assert blocked is False
    assert reason == ""


def test_entries_blocked_by_regime_rejects_weak_decision_claw_long_exception():
    dm = _new_dm_stub()
    dm.youtube_context = {"resolved_regime": {"allow_new_longs": False}}
    dm._load_resolved_regime_context = lambda: {"allow_new_longs": False}
    dm._is_hedge_symbol = lambda symbol: False
    dm._refresh_entry_authority_state = lambda: {
        "state": "bull_lock",
        "snapshot": {"avg_pct_change": -0.8, "available_count": 3},
    }
    dm._execution_override_plan = {
        "deployment_request": {
            "mode": "long_exception",
            "symbols": ["GLNG"],
            "max_new_entries": 1,
            "reason": "dead capital",
        },
        "symbol_reviews": {
            "GLNG": {
                "decision": "deny",
                "accepted": False,
                "reject_reason": "weak_news_and_fade",
                "failure_mode": "weak_news_and_fade",
            }
        },
    }
    dm._find_signal_data = lambda symbol: {
        "ticker": symbol,
        "current_price": 10.1,
        "prev_close": 10.0,
        "volume_ratio": 1.6,
    }
    dm.get_current_price = lambda symbol: 10.1
    dm._check_intraday_momentum = lambda symbol, avg_volume=0.0: {
        "above_vwap": False,
        "volume_ratio": 1.6,
    }

    blocked, reason = DayManager._entries_blocked_by_regime(dm, "GLNG")

    assert blocked is True
    assert reason == "decision_claw_local_reject:weak_news_and_fade"


def test_sync_entry_wave_respects_hold_wave_override():
    dm = _new_dm_stub()
    dm.entry_quality_cfg.wave_entry_enabled = True
    dm.entry_wave = 2
    dm.wave_positions = {}
    dm._get_current_entry_wave = lambda: 4
    dm._execution_override_plan = {"wave_action": "hold_wave"}

    DayManager._sync_entry_wave(dm, [])

    assert dm.entry_wave == 2


def test_prepare_signal_recheck_resets_only_non_executed_symbols():
    dm = _new_dm_stub()
    dm.signal_status = {
        "AAPL": {"status": "skipped", "reason": "old_reason"},
        "MSFT": {"status": "executed", "reason": "already_filled"},
    }
    dm._mark_signal_pending = DayManager._mark_signal_pending.__get__(dm, DayManager)

    refreshed = dm.prepare_signal_recheck(
        ["aapl", "msft", "nvda"], reason="overnight_recheck"
    )

    assert refreshed == 1
    assert dm.signal_status["AAPL"]["status"] == "pending"
    assert dm.signal_status["AAPL"]["reason"] == "overnight_recheck"
    assert dm.signal_status["MSFT"]["status"] == "executed"
    assert dm.signal_status["MSFT"]["reason"] == "already_filled"


def test_direct_buy_submission_blocks_after_same_symbol_trim_intent():
    dm = _new_dm_stub()
    dm._symbol_intent_state_today = {
        "CORZ": {
            "last_exit_like_intent": "trim",
            "last_exit_like_hard": False,
        }
    }

    reason = DayManager._entry_submission_block_reason(
        dm,
        symbol="CORZ",
        qty=10,
        side="buy",
        context="scale_into_winner",
        latest_price=12.0,
    )

    assert reason == "same_symbol_intent_conflict:trim_before_add"


def test_trim_lot_guard_blocks_small_noncritical_partial_trim():
    dm = _new_dm_stub()

    reason = DayManager._trim_lot_block_reason(
        dm,
        "SKE",
        qty=5,
        current_price=30.0,
        current_qty=40,
        reason="mean_reversion_trim",
    )

    assert reason == "trim_lot_too_small:150.00<200.00"


def test_trim_lot_guard_allows_full_exit_cleanup():
    dm = _new_dm_stub()

    reason = DayManager._trim_lot_block_reason(
        dm,
        "SKE",
        qty=5,
        current_price=30.0,
        current_qty=5,
        reason="mean_reversion_trim",
    )

    assert reason == ""


def test_eod_review_force_closes_deep_loser_and_trims_big_winner():
    dm = _new_dm_stub()
    dm._eod_review_session_date = None
    dm._eod_review_actions = {}
    dm._is_eod_review_window = lambda: True
    exits = []
    dm.execute_exit = lambda symbol, qty, reason: (
        exits.append((symbol, qty, reason)) or True
    )

    scored_positions = [
        (
            SimpleNamespace(symbol="LOSE", qty=10),
            {"pnl_pct": -5.5, "action": "hold"},
            50.0,
        ),
        (
            SimpleNamespace(symbol="WIN", qty=10),
            {"pnl_pct": 12.0, "action": "hold"},
            80.0,
        ),
    ]
    stats = {"exits": 0}

    dm._run_eod_review(scored_positions, stats)

    assert ("LOSE", 10, "eod_force_exit_deep_loser") in exits
    assert ("WIN", 5, "eod_force_exit_profit_lock") in exits
    assert stats["exits"] == 2


def test_daily_drawdown_ladder_sets_sizing_halt_and_next_open():
    dm = _new_dm_stub()
    dm._daily_drawdown_pct = 3.1

    dm._apply_daily_drawdown_ladder()

    assert dm._daily_drawdown_size_multiplier == 0.5
    assert dm._daily_drawdown_halt is True
    assert dm._daily_drawdown_tight_trail_atr == 1.2
    assert dm._daily_drawdown_halt_next_open is True
    assert {1.0, 1.5, 2.0, 3.0}.issubset(dm._daily_drawdown_fired_tiers)


def test_daily_drawdown_position_actions_trim_and_force_close_once():
    dm = _new_dm_stub()
    dm._daily_drawdown_fired_tiers = {2.0, 3.0}
    trims = []
    exits = []
    dm.execute_trim = lambda symbol, qty, reason: (
        trims.append((symbol, qty, reason)) or True
    )
    dm.execute_exit = lambda symbol, qty, reason: (
        exits.append((symbol, qty, reason)) or True
    )
    scored_positions = [
        (SimpleNamespace(symbol="A", qty=20), {"pnl_pct": -4.0, "action": "hold"}, 30),
        (SimpleNamespace(symbol="B", qty=20), {"pnl_pct": -3.0, "action": "watch"}, 40),
        (SimpleNamespace(symbol="C", qty=20), {"pnl_pct": -2.0, "action": "trim"}, 50),
        (SimpleNamespace(symbol="D", qty=20), {"pnl_pct": -1.0, "action": "add"}, 20),
    ]
    stats = {"exits": 0}

    dm._apply_daily_drawdown_position_actions(scored_positions, stats)
    dm._apply_daily_drawdown_position_actions(scored_positions, stats)

    assert trims == [("A", 5, "daily_drawdown_tier_2_bottom_quartile_trim")]
    assert exits == [
        ("A", 20, "daily_drawdown_tier_3_force_close_loser"),
        ("B", 20, "daily_drawdown_tier_3_force_close_loser"),
        ("C", 20, "daily_drawdown_tier_3_force_close_loser"),
    ]
    assert stats["exits"] == 4


def test_llm_advisory_trim_votes_advance_escalation_ladder():
    dm = _new_dm_stub()

    h1 = dm._promote_llm_advisory_vote(
        symbol="ABCD",
        action="trim",
        advice={"confidence": 0.80, "size_pct": 100},
        current_qty=100,
        pnl_pct=-0.5,
        score=-25,
        signals=["llm trim"],
        reasoning="trim risk",
    )
    assert h1["action"] == "hold"
    assert h1["llm_escalation"]["step"] == 1
    assert h1["llm_escalation"]["qty"] == 25

    dm._llm_advisory_escalation_state["ABCD"]["last_vote_at"] -= timedelta(minutes=31)
    h2 = dm._promote_llm_advisory_vote(
        symbol="ABCD",
        action="trim",
        advice={"confidence": 0.80},
        current_qty=75,
        pnl_pct=-1.0,
        score=-25,
        signals=["llm trim"],
        reasoning="trim risk again",
    )
    assert h2["llm_escalation"]["step"] == 2
    assert h2["llm_escalation"]["qty"] == 50

    dm._llm_advisory_escalation_state["ABCD"]["last_vote_at"] -= timedelta(minutes=31)
    h3 = dm._promote_llm_advisory_vote(
        symbol="ABCD",
        action="trim",
        advice={"confidence": 0.80},
        current_qty=25,
        pnl_pct=-2.0,
        score=-25,
        signals=["llm trim"],
        reasoning="full risk off",
    )
    assert h3["llm_escalation"]["step"] == 3
    assert h3["llm_escalation"]["full_exit"] is True
    assert h3["llm_escalation"]["qty"] == 25


def test_risk_gate_profit_take_trim_votes_do_not_advance_to_full_exit():
    dm = _new_dm_stub()

    risk_gate_advice = {
        "confidence": 0.80,
        "advisor_source": "risk_gate_fast_path",
        "risk_gate_skip_agents": True,
        "risk_gate_rules": ["profit_take_5.0%"],
    }
    h1 = dm._promote_llm_advisory_vote(
        symbol="FAST",
        action="trim",
        advice=risk_gate_advice,
        current_qty=100,
        pnl_pct=5.5,
        score=-15,
        signals=["profit_take_5.0%"],
        reasoning="Profit 5.5% >= 5.0% level",
    )
    assert h1["llm_escalation"]["step"] == 1
    assert h1["llm_escalation"]["advisor_source"] == "risk_gate_fast_path"

    dm._llm_advisory_escalation_state["FAST"]["last_vote_at"] -= timedelta(minutes=31)
    h2 = dm._promote_llm_advisory_vote(
        symbol="FAST",
        action="trim",
        advice=risk_gate_advice,
        current_qty=75,
        pnl_pct=5.8,
        score=-15,
        signals=["profit_take_5.0%"],
        reasoning="Profit 5.8% >= 5.0% level",
    )
    assert h2["llm_escalation"]["step"] == 2

    dm._llm_advisory_escalation_state["FAST"]["last_vote_at"] -= timedelta(minutes=31)
    h3 = dm._promote_llm_advisory_vote(
        symbol="FAST",
        action="trim",
        advice=risk_gate_advice,
        current_qty=25,
        pnl_pct=6.1,
        score=-15,
        signals=["profit_take_5.0%"],
        reasoning="Profit 6.1% >= 5.0% level",
    )
    assert (
        h3["llm_escalation_skip_reason"]
        == "llm_escalation_risk_gate_profit_take_full_exit_blocked"
    )
    assert dm._llm_advisory_escalation_state["FAST"]["vote_count"] == 2
    assert "FAST" not in dm._llm_escalation_reentry_lockouts


def test_llm_advisory_exit_is_demoted_to_ladder_vote_not_direct_exit():
    dm = _new_dm_stub()

    health = dm._promote_llm_advisory_vote(
        symbol="EXIT",
        action="exit",
        advice={"confidence": 0.90},
        current_qty=40,
        pnl_pct=-3.0,
        score=-35,
        signals=["llm exit"],
        reasoning="exit now",
    )

    assert health["action"] == "hold"
    assert health["llm_escalation"]["source_action"] == "exit"
    assert health["llm_escalation"]["qty"] == 10


def test_llm_advisory_confidence_floor_and_gap_ignore_votes():
    dm = _new_dm_stub()

    low = dm._promote_llm_advisory_vote(
        symbol="LOWC",
        action="trim",
        advice={"confidence": 0.69},
        current_qty=100,
        pnl_pct=-1.0,
        score=-25,
        signals=[],
        reasoning="weak confidence",
    )
    assert low["llm_escalation_skip_reason"].startswith(
        "llm_escalation_confidence_below_floor"
    )
    assert dm._llm_advisory_escalation_state == {}

    first = dm._promote_llm_advisory_vote(
        symbol="GAP",
        action="trim",
        advice={"confidence": 0.90},
        current_qty=100,
        pnl_pct=-1.0,
        score=-25,
        signals=[],
        reasoning="first",
    )
    second = dm._promote_llm_advisory_vote(
        symbol="GAP",
        action="trim",
        advice={"confidence": 0.90},
        current_qty=75,
        pnl_pct=-1.2,
        score=-25,
        signals=[],
        reasoning="too soon",
    )
    assert first["llm_escalation"]["step"] == 1
    assert second["llm_escalation_skip_reason"].startswith("llm_escalation_min_gap")
    assert dm._llm_advisory_escalation_state["GAP"]["vote_count"] == 1


def test_llm_escalation_step3_blocks_reentry_for_two_hours():
    dm = _new_dm_stub()
    dm._llm_escalation_reentry_lockouts["LOCK"] = datetime.now(timezone.utc)

    reason = dm._llm_escalation_reentry_cooldown_reason("LOCK")

    assert reason.startswith("llm_escalation_reentry_cooldown")


def test_llm_escalation_caps_steps_per_symbol_per_session():
    dm = _new_dm_stub()
    now = datetime.now(timezone.utc) - timedelta(minutes=31)
    dm._llm_advisory_escalation_state["CAP"] = {
        "vote_count": 3,
        "last_vote_at": now,
        "original_qty": 100,
        "cumulative_cut_pct": 1.0,
        "step3_at": now,
    }

    health = dm._promote_llm_advisory_vote(
        symbol="CAP",
        action="trim",
        advice={"confidence": 0.90},
        current_qty=10,
        pnl_pct=-3.0,
        score=-25,
        signals=[],
        reasoning="extra vote",
    )

    assert health["llm_escalation_skip_reason"] == "llm_escalation_max_steps:3>=3"


def test_agentic_advisor_skips_during_llm_eval_cooldown():
    dm = _new_dm_stub()
    dm.use_agentic = True
    dm.advisor_control_cfg = SimpleNamespace(enabled=True)
    dm._llm_advisory_escalation_state["COOL"] = {
        "vote_count": 1,
        "last_vote_at": datetime.now(timezone.utc) - timedelta(minutes=20),
        "original_qty": 100,
        "cumulative_cut_pct": 0.25,
        "step3_at": None,
    }

    assert dm._should_run_advisor("COOL", 1.0, 10.0, 120) is False


def test_agentic_advisor_runs_after_llm_eval_cooldown():
    dm = _new_dm_stub()
    dm.use_agentic = True
    dm.advisor_control_cfg = SimpleNamespace(
        enabled=True,
        min_recheck_seconds=300,
        force_recheck_seconds=900,
        price_move_bps_trigger=35,
        pnl_change_pct_trigger=0.5,
        hold_minutes_trigger=30,
    )
    dm._advisor_eval_cache = {}
    dm._llm_advisory_escalation_state["COOL"] = {
        "vote_count": 1,
        "last_vote_at": datetime.now(timezone.utc) - timedelta(minutes=31),
        "original_qty": 100,
        "cumulative_cut_pct": 0.25,
        "step3_at": None,
    }

    assert dm._should_run_advisor("COOL", 1.0, 10.0, 120) is True


def test_trim_pnl_step_deterioration_requests_full_liquidation():
    dm = _new_dm_stub()
    dm._last_trim_pnl["DROP"] = 5.0

    reason = dm._trim_pnl_step_deterioration_exit_reason("DROP", 2.9)

    assert reason.startswith("trim_pnl_step_deterioration_force_exit")


def test_trim_governance_default_cooldown_is_60_minutes():
    from config.config_loader import TrimGovernanceConfig

    assert TrimGovernanceConfig().cooldown_minutes == 60


def test_book_velocity_brake_fires_sweeps_and_pauses_entries():
    dm = _new_dm_stub()
    dm._normalize_entry_time = DayManager._normalize_entry_time
    exits = []
    dm.execute_exit = lambda symbol, qty, reason: (
        exits.append((symbol, qty, reason)) or True
    )
    dm._last_trim_pnl["ROLL"] = 7.0
    stats = {"exits": 0}
    scored = [
        (SimpleNamespace(symbol="LOSE", qty=10), {"pnl_pct": -1.2}, 30.0),
        (SimpleNamespace(symbol="ROLL", qty=8), {"pnl_pct": 4.8}, 40.0),
        (SimpleNamespace(symbol="HOLD", qty=6), {"pnl_pct": 1.0}, 50.0),
    ]

    assert dm._update_book_velocity_brake(
        equity=100_000.0, scored_positions=scored, stats=stats
    ) is False
    assert dm._update_book_velocity_brake(
        equity=99_400.0, scored_positions=scored, stats=stats
    ) is True
    assert [row[0] for row in exits] == ["LOSE", "ROLL"]
    assert stats["book_velocity_brake"]["pause_minutes"] == 30
    assert dm._book_velocity_entry_pause_reason().startswith(
        "book_velocity_entry_pause"
    )


def test_loss_floor_exits_cumulative_loser_in_normal_failsafe_state():
    dm = _new_dm_stub()
    dm.loss_floor_checker = LossFloorChecker(enabled=True)
    dm._positions_from_cache = False
    dm._hold_minutes = lambda symbol: 10
    dm._mark_hard_stop_blocked_symbol = lambda symbol: None
    exits = []
    dm.execute_exit = lambda symbol, qty, reason: (
        exits.append((symbol, qty, reason)) or True
    )
    stats = {"exits": 0}
    positions = [
        SimpleNamespace(
            symbol="PCT",
            qty=366,
            avg_entry_price=12.26,
            current_price=11.23,
            unrealized_plpc=-0.084,
            change_today=0.01,
        )
    ]

    executed = dm._run_loss_floor_exits(positions, stats)

    assert executed == {"PCT"}
    assert stats["exits"] == 1
    assert exits == [("PCT", 366, "loss_floor_force_exit:-5.0%@anytime:pnl=-8.4%:held=10m")]


def test_loss_floor_skips_cached_positions_to_avoid_stale_broker_truth():
    dm = _new_dm_stub()
    dm.loss_floor_checker = LossFloorChecker(enabled=True)
    dm._positions_from_cache = True
    dm._hold_minutes = lambda symbol: 10
    exits = []
    dm.execute_exit = lambda symbol, qty, reason: (
        exits.append((symbol, qty, reason)) or True
    )
    stats = {"exits": 0}
    positions = [
        SimpleNamespace(
            symbol="STALE",
            qty=10,
            avg_entry_price=100.0,
            current_price=90.0,
            unrealized_plpc=-0.10,
        )
    ]

    assert dm._run_loss_floor_exits(positions, stats) == set()
    assert exits == []
    assert stats["exits"] == 0
