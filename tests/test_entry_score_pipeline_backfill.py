from types import SimpleNamespace

from autotrade.core import day_manager as day_manager_mod
from autotrade.core.day_manager import DayManager
from tests.test_day_manager_execution_policy import _new_dm_stub


def test_execute_entry_backfills_plan_score_when_runtime_entry_score_is_placeholder():
    dm = _new_dm_stub()
    dm.dry_run = False
    dm._watched_universe_tickers = {"CNK"}
    dm.signals = [
        {
            "ticker": "CNK",
            "symbol": "CNK",
            "action": "buy_open",
            "recommendation": "buy",
            "entry_score": 0.0,
            "final_score": 76.5,
            "confidence": 76.5,
            "score": 76.5,
            "entry_price": 20.0,
            "current_price": 20.0,
            "position_size": 1000.0,
            "atr_14": 1.0,
            "risk_reward": 2.0,
            "entry_source": "overnight_plan",
            "source_bucket": "watchlist",
            "plan_score_source": "pm_plan_2026-04-21.json",
        }
    ]
    dm.signal_status = {"CNK": {"status": "pending", "reason": ""}}
    dm._feature_pipeline_health = {"status": "failed", "retry_requested": True}
    dm.strategy_failsafe_snapshot = SimpleNamespace(
        halt_new_entries=False,
        level="normal",
        position_size_multiplier=1.0,
        position_size_pct=0.0,
        max_positions=day_manager_mod.MAX_POSITIONS,
    )
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.CORE_TRADING
    dm.get_current_price = lambda ticker: 20.0
    dm._check_universe_compliance = lambda *args, **kwargs: (True, "")
    dm._apply_regime_entry_gate = lambda signal_data, allow_capped=False: (True, "")
    dm.get_positions = lambda: []
    dm.can_enter_positions = lambda **kwargs: (True, "")
    dm._has_open_buy_order = lambda symbol: (False, "")
    dm._build_candidate_validation_report = (
        lambda signal_data: {"allowed": True, "entry_source": "overnight_plan"}
    )
    dm._resolve_entry_authority = lambda signal_data: {
        "eligible": True,
        "entry_score": 76.5,
        "entry_score_source": "final_score",
        "plan_score_source": "pm_plan_2026-04-21.json",
    }
    dm._is_vl_recheck_pending = lambda symbol: (False, None, "")
    dm._vl_confirm_entry = lambda **kwargs: (True, "")
    dm._clear_vl_recheck = lambda symbol: None
    dm._check_execution_circuit_breaker = lambda: False
    dm._conviction_size_multiplier = lambda entry_score: 1.0
    dm._resolve_entry_urgency_tier = lambda entry_score, signal_data=None: "normal"
    dm._entry_tier_limits = lambda urgency_tier: (0.0, 12.0)
    dm._resolve_runtime_entry_anchor = (
        lambda **kwargs: (True, float(kwargs["entry_price"]), "")
    )
    dm._check_intraday_momentum = lambda *args, **kwargs: {"pass": True}
    dm.get_bars = lambda *args, **kwargs: None
    dm._compute_entry_limit_price = (
        lambda planned_entry, current_price, urgency_tier: float(planned_entry)
    )
    dm._entry_submission_block_reason = lambda **kwargs: ""
    dm._force_signal_skip = lambda ticker, reason: None
    dm._record_execution_attempt = lambda *args, **kwargs: None
    dm._record_execution_success = lambda *args, **kwargs: None
    dm._record_execution_failure = lambda *args, **kwargs: None
    dm._note_local_pending_entry = lambda symbol: None
    dm._register_entry_order_lifecycle = lambda **kwargs: None
    dm._queue_sequential_shadow_event = lambda **kwargs: None
    dm._record_wave_entry = lambda *args, **kwargs: None
    dm._record_strategy_profile_entry = lambda *args, **kwargs: None
    dm._mark_signal_executed = lambda ticker: None
    dm._acquire_order_submission_guard = lambda *args, **kwargs: (True, "", "guard-1")
    dm._release_order_submission_guard = lambda *args, **kwargs: None
    dm.position_entries = {}
    dm.position_entry_sources = {}
    dm.position_health = {}
    dm._entry_order_lifecycle = {}
    dm.wave_positions = {}
    dm.entry_wave = 1

    recorded = {}
    dm.trade_journal = SimpleNamespace(
        signal_capture=SimpleNamespace(capture=lambda *args, **kwargs: {}),
        record_entry=lambda **kwargs: recorded.update(kwargs) or "trade-1",
        record_exit=lambda *args, **kwargs: None,
        record_trim=lambda *args, **kwargs: None,
        save=lambda: None,
    )
    dm._submit_order_via_execution_adapter = lambda **kwargs: SimpleNamespace(
        id="ord-1",
        filled_qty=10,
        filled_avg_price=20.0,
        slippage_bps=0.0,
        time_to_first_fill_ms=25,
        replace_count=0,
    )

    allowed = DayManager.execute_entry(dm, "CNK", "test entry score backfill")

    assert allowed is True
    assert recorded["entry_score"] == 76.5
    assert recorded["signal_context"]["entry_score"] == 76.5
    assert recorded["signal_context"]["entry_score_source"] == "final_score"
