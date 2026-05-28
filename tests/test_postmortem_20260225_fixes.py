from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import json
import logging

import pytest

pytest.importorskip("alpaca.trading.client")

from autotrade.core.day_manager import (
    DayManager,
    MAX_EXITS_PER_CYCLE,
    TradingPhase,
)
from autotrade.signals import agentic_signal_generator as signal_gen_module


def _position(symbol: str, qty: int = 10, price: float = 10.0):
    return SimpleNamespace(
        symbol=symbol,
        qty=qty,
        current_price=price,
        market_value=qty * price,
        unrealized_plpc=0.0,
        avg_entry_price=price,
    )


def _build_run_cycle_stub() -> tuple[DayManager, list[str]]:
    dm = DayManager.__new__(DayManager)
    dm.cycle_count = 1  # run_cycle increments first; avoids refresh branches at %10==1
    dm.dry_run = True
    dm.errors_this_session = 0
    dm._sequential_shadow_enabled = False

    dm._feature_pipeline_health = {"status": "ok", "cache_hit": False, "symbol_count": 0}
    dm.regime_router_context = {"regime": "neutral"}
    dm.research_freshness = {}
    dm.research_age_hours = None
    dm.youtube_context = {"available": False}
    dm.regime_detector = None
    dm.regime_strategy_overrides = {}
    dm.current_regime = None
    dm.premarket_handoff = {}
    dm.signals = []
    dm.strategy_profile_entries_today = {}
    dm.position_health = {}
    dm.vwap_universe_scanner = None
    dm.inverse_etf_manager = None
    dm.halt_monitor = None
    dm._intraday_reserved_symbols = set()
    dm._position_slot_class_by_symbol = {}
    dm._position_lock_list = set()
    dm.research_freshness_cfg = SimpleNamespace(stale_penalty_age_hours=24.0)
    dm.research_freshness = {}
    dm.research_age_hours = 0.0
    dm.premarket_handoff = {}
    dm.advisor_control_cfg = SimpleNamespace()
    dm.risk_gate_cfg = SimpleNamespace(
        add_block_after_trim=True,
        midday_revalidation_trim_fraction=0.33,
        min_hold_minutes_for_trim=0,
    )
    dm.execution_entry_cfg = SimpleNamespace()
    dm.feature_engineering_cfg = SimpleNamespace(enabled=False)
    dm.signal_generation_cfg = SimpleNamespace()
    dm.signal_validation_cfg = SimpleNamespace(enabled=False)
    dm._max_new_entries_override = None
    dm.data_client = None
    dm.r_unit_sizer = None
    dm._add_block_after_trim = True
    dm._midday_revalidation_trim_fraction = 0.33
    dm._min_hold_minutes_for_trim = 0
    dm._hard_stop_blocked_symbols_today = set()
    dm._exited_symbols_today = set()
    dm.config = SimpleNamespace(
        portfolio=SimpleNamespace(target_position_size=10000.0),
        entry_quality=SimpleNamespace(
            wave_entry_enabled=False,
            wave_max_entries=10,
            vwap_universe_scan_interval_cycles=5,
        ),
    )
    dm._momentum_scanner_health = {"status": "disabled"}
    dm.regime_detector = None
    dm.regime_router_context = {}
    dm._feature_pipeline_health = {"status": "disabled"}
    dm.current_regime = None
    dm.cycle_count = 0
    dm.position_health = {}
    dm.signal_status = {}
    dm._last_trim_time = {}
    dm._trim_count_today = {}
    dm._last_trim_pnl = {}
    dm.position_entries = {}
    dm._last_scale_time = {}
    dm.SCALE_TRIM_LOCKOUT_MINUTES = 60
    dm.TRIM_PNL_STEP = 2.0
    dm.TRIM_COOLDOWN_MINUTES = 60
    dm.MAX_TRIMS_PER_SYMBOL_PER_DAY = 2
    dm.MIN_POSITION_VALUE = 200.0
    dm.MAX_EXITS_PER_CYCLE = 5
    dm.errors_this_session = 0
    dm.youtube_context = {"available": False}
    dm._execution_override_plan = {}
    dm._strength_reentry_scan_enabled = False
    dm._strength_reentry_scan_interval = 30
    dm.regime_detector = None
    dm.current_regime = None
    dm.regime_strategy_overrides = {}
    dm.live_sector_bias = {}
    dm.signal_pipeline = None
    dm.watchlist_rotator = None
    dm.universe_scanner = None
    dm._watchlist_drop_buffer = []
    dm.rotation_scheduler = None
    dm._candidate_validation_rejections = []
    dm.prune_watchlist = lambda held: []
    dm.batch_rotate = lambda **kwargs: 0
    dm._run_watchlist_rotation_scheduler = lambda held: 0
    dm._log_intraday_signals = lambda *args, **kwargs: None
    dm._check_research_freshness = lambda **kwargs: {"age_hours": 1.0}
    dm._load_latest_premarket_handoff = lambda: {}
    dm._load_youtube_intelligence = lambda: {"available": False}
    dm._refresh_entry_authority_state = lambda: {"state": "open"}
    dm._refresh_momentum_scanner_signals = lambda: 0
    dm._record_execution_attempt = lambda *args, **kwargs: None
    dm._record_execution_success = lambda *args, **kwargs: None
    dm._record_execution_failure = lambda *args, **kwargs: None
    dm._acquire_order_submission_guard = lambda *args, **kwargs: (True, "", "")
    dm._release_order_submission_guard = lambda *args, **kwargs: None
    dm._validate_order = lambda *args, **kwargs: (True, "")
    dm._submit_order_via_execution_adapter = lambda *args, **kwargs: None

    dm._daily_drawdown_pct = 0.0
    dm._daily_drawdown_limit = 99.0
    dm._daily_drawdown_halt = False
    dm._start_of_day_equity = 100_000.0
    dm.last_account_equity = 100_000.0

    dm.strategy_failsafe_snapshot = SimpleNamespace(
        level="normal",
        halt_new_entries=False,
        drawdown_pct=0.0,
        position_size_multiplier=1.0,
        position_size_pct=0.0,
        min_conviction_exit=55.0,
        loser_exit_pnl_pct=-5.0,
    )
    dm.entry_quality_cfg = SimpleNamespace(
        wave_entry_enabled=False,
        wave_max_entries=10,
        vwap_universe_scan_interval_cycles=5,
    )
    dm.entry_wave = 1
    dm.wave_positions = {}

    dm.day_tracker = SimpleNamespace(get_remaining=lambda: 10)
    dm._refresh_strategy_failsafe = lambda account=None: dm.strategy_failsafe_snapshot
    dm._run_policy_engine_cycle = lambda positions, account, remaining: None
    dm._manage_inverse_etfs = lambda positions: None

    dm.get_positions = lambda: [_position(f"S{i}") for i in range(5)]
    dm.get_account = lambda: SimpleNamespace(
        equity="100000",
        last_equity="100000",
        buying_power="250000",
    )
    dm._validated_positions = lambda positions, context=None: positions
    dm._reconcile_submitted_journal_entries = lambda: {}
    dm._validate_positions = lambda positions: {}
    dm._hydrate_position_entries = lambda positions=None, reason="": None
    dm._refresh_feature_context = lambda tickers: None
    dm._refresh_regime_router_context = lambda: None
    dm._maybe_midday_revalidate = lambda positions: []
    dm._effective_max_positions = lambda: 50
    dm.get_current_phase = lambda: TradingPhase.CORE_TRADING
    dm.can_enter_positions = lambda *args, **kwargs: (True, "")
    dm.check_position_sizes = lambda positions: {"oversized": [], "scalable": []}
    dm.portfolio_rotator = None

    dm.calculate_position_health = lambda pos: {
        "action": "exit",
        "score": -40,
        "pnl_pct": -1.0,
        "signals": [],
    }
    dm._estimate_position_conviction = lambda pos, health: 0.0
    dm._apply_failsafe_triage = lambda pos, health, conviction: health
    dm._position_qty = lambda pos: int(getattr(pos, "qty", 0))
    dm._position_float = lambda pos, field, default=0.0, absolute=False: (
        abs(float(getattr(pos, field, default) or default))
        if absolute
        else float(getattr(pos, field, default) or default)
    )

    exit_calls: list[str] = []

    def _execute_exit(symbol, qty, reason):
        exit_calls.append(str(symbol))
        return True

    dm.execute_exit = _execute_exit
    dm.find_replacement_candidates = lambda held_tickers: [
        {"ticker": f"R{i}", "score": 90, "realtime_score": 90, "entry_source": "overnight_plan"} for i in range(20)
    ]
    dm._watched_universe_tickers = {f"R{i}" for i in range(20)}
    dm.should_replace = lambda weak_health, candidate: True
    dm.execute_entry = lambda *args, **kwargs: True
    dm.scan_power_hour_opportunities = lambda: []
    dm.scan_watchlist_movers = lambda held_tickers: []
    dm._is_power_hour = lambda: False
    dm._vwap_mean_reversion_window_active = lambda: False
    dm._is_power_hour_volume_signal_active = lambda: False
    dm._regime_router_entry_threshold_adjustment = lambda: 0.0
    dm.research_score_threshold_penalty = 0.0
    dm._mark_signal_skipped = lambda ticker, reason: None

    return dm, exit_calls


def test_wave_capacity_ignores_exited_wave_symbols():
    dm = DayManager.__new__(DayManager)
    dm.entry_quality_cfg = SimpleNamespace(wave_entry_enabled=True, wave_max_entries=10)
    dm.entry_wave = 1
    dm.wave_positions = {1: ["AAA", "BBB", "CCC"]}
    dm._sync_entry_wave = lambda positions: None

    capacity = DayManager._get_wave_capacity(dm, positions=[])
    assert capacity == 10


def test_execute_entry_allows_reentry_after_exit():
    dm = DayManager.__new__(DayManager)
    dm.signal_status = {"ABC": {"status": "exited", "re_entry_eligible": True}}
    dm._watched_universe_tickers = {"ABC"}
    dm.get_positions = lambda: []  # previously executed but not currently held
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
        {"ticker": "ABC", "action": "buy_open", "score": 75, "atr_14": 1.0, "entry_source": "overnight_plan"}
    ]
    dm._has_open_buy_order = lambda symbol_key: (False, "")
    dm._should_promote_watch_signal = lambda signal_data: (False, "")
    dm.get_current_phase = lambda: TradingPhase.CORE_TRADING
    dm._is_strategy_window_open = lambda strategy_profile: True
    dm.lesson_book = SimpleNamespace(
        get_position_size_multiplier=lambda capture: 1.0,
    )
    dm.trade_journal = SimpleNamespace(
        signal_capture=SimpleNamespace(capture=lambda *args, **kwargs: {}),
        save=lambda: None,
    )
    dm._conviction_size_multiplier = lambda score: 1.0
    dm.youtube_context = {}
    dm._regime_override_float = lambda key, fallback=1.0: 1.0
    dm.research_position_size_multiplier = 1.0
    dm.last_account_equity = 100_000.0
    dm._validate_order = lambda *args, **kwargs: (True, "")
    dm._effective_stop_multiplier = lambda fallback=2.0: 2.0
    dm._is_breakout_continuation_setup = lambda signal_data: False
    dm._effective_market_regime = lambda: "NEUTRAL"
    dm.regime_strategy_overrides = {}
    dm._record_wave_entry = lambda ticker, wave=None: None
    dm._record_strategy_profile_entry = lambda signal_data: None
    dm._clear_vl_recheck = lambda ticker: None
    dm.runtime_risk_gate = None
    dm._intraday_reserved_symbols = set()
    dm._position_slot_class_by_symbol = {}
    dm.execution_entry_cfg = SimpleNamespace(
        high_urgency_min_score=72.0,
        critical_urgency_min_score=84.0,
        normal=SimpleNamespace(max_chase_bps=8, max_slippage_bps=12),
        high=SimpleNamespace(max_chase_bps=10, max_slippage_bps=15),
        critical=SimpleNamespace(max_chase_bps=12, max_slippage_bps=20),
    )
    dm.dry_run = True

    result = DayManager.execute_entry(dm, "ABC", "test re-entry")
    assert result is True
    assert dm.signal_status["ABC"]["status"] == "executed"
    assert dm.signal_status["ABC"].get("re_entry") is True


def test_failsafe_triage_respects_grace_period():
    dm = DayManager.__new__(DayManager)
    dm.strategy_failsafe_snapshot = SimpleNamespace(
        level="failing",
        min_conviction_exit=55.0,
        loser_exit_pnl_pct=-5.0,
    )
    dm._hold_minutes = lambda symbol: 5

    position = SimpleNamespace(symbol="XYZ")
    health = {"action": "hold", "pnl_pct": -1.0, "signals": []}
    updated = DayManager._apply_failsafe_triage(dm, position, health, conviction_score=10.0)

    assert updated["action"] == "hold"
    assert any("GRACE PERIOD" in s for s in updated.get("signals", []))
    assert updated.get("failsafe_forced_exit") is not True


def test_run_cycle_circuit_breaker_caps_non_hard_exits():
    dm, exit_calls = _build_run_cycle_stub()

    stats = DayManager.run_cycle(dm)

    assert len(exit_calls) == MAX_EXITS_PER_CYCLE
    assert stats["exits"] == MAX_EXITS_PER_CYCLE


def test_run_cycle_midday_revalidation_trim_records_journal_entry():
    dm, _exit_calls = _build_run_cycle_stub()
    dm.dry_run = False
    dm.get_positions = lambda: [_position("MID", qty=30, price=10.0)]
    dm.calculate_position_health = lambda pos: {
        "action": "hold",
        "score": 10,
        "pnl_pct": 5.0,
        "signals": [],
    }
    dm._maybe_midday_revalidate = lambda positions: ["MID"]
    dm._should_allow_midday_trim = lambda symbol, pnl_pct: (True, "")
    dm._hold_minutes = lambda symbol: 120
    dm._refresh_entry_capacity_snapshot = lambda **kwargs: None
    dm._materialize_sequential_shadow_events = lambda: {}
    dm._current_phase_value = lambda phase=None: "core_trading"
    dm._refresh_entry_authority_state = lambda *args, **kwargs: {"state": "open"}
    dm._record_midday_trim_state = lambda symbol, pnl_pct: None
    dm._midday_trim_window_state = {}
    dm._position_slot_class_by_symbol = {}
    dm.find_replacement_candidates = lambda held_tickers: []
    trim_calls = {}
    dm.trade_journal = SimpleNamespace(
        record_trim=lambda *args, **kwargs: trim_calls.update(
            {"args": args, "kwargs": kwargs}
        )
        or "trim-1"
    )
    dm._submit_order_via_execution_adapter = lambda **kwargs: SimpleNamespace(
        id="ord-midday-1",
        status="submitted",
    )

    DayManager.run_cycle(dm)

    assert trim_calls["kwargs"]["symbol"] == "MID"
    assert trim_calls["kwargs"]["shares_sold"] == 9
    assert trim_calls["kwargs"]["order_id"] == "ord-midday-1"
    assert trim_calls["kwargs"]["execution_status"] == "submitted"


def test_run_cycle_mean_reversion_trim_records_journal_entry():
    dm, _exit_calls = _build_run_cycle_stub()
    dm.dry_run = False
    dm.get_positions = lambda: [_position("TRM", qty=20, price=25.0)]
    dm.calculate_position_health = lambda pos: {
        "action": "trim",
        "score": -10,
        "pnl_pct": 8.0,
        "signals": ["mean_reversion_trim"],
        "trim_reason": "mean_reversion_trim",
        "trim_fraction": 0.10,
    }
    dm._maybe_midday_revalidate = lambda positions: []
    dm._hold_minutes = lambda symbol: 120
    dm._refresh_entry_capacity_snapshot = lambda **kwargs: None
    dm._materialize_sequential_shadow_events = lambda: {}
    dm._current_phase_value = lambda phase=None: "core_trading"
    dm._refresh_entry_authority_state = lambda *args, **kwargs: {"state": "open"}
    dm._position_slot_class_by_symbol = {}
    dm.find_replacement_candidates = lambda held_tickers: []
    dm.client = SimpleNamespace(
        get_orders=lambda request: [],
        cancel_order_by_id=lambda order_id: None,
    )
    trim_calls = {}
    dm.trade_journal = SimpleNamespace(
        record_trim=lambda *args, **kwargs: trim_calls.update(
            {"args": args, "kwargs": kwargs}
        )
        or "trim-1"
    )
    dm._submit_order_via_execution_adapter = lambda **kwargs: SimpleNamespace(
        id="ord-trim-1",
        status="submitted",
    )

    DayManager.run_cycle(dm)

    assert trim_calls["kwargs"]["symbol"] == "TRM"
    assert trim_calls["kwargs"]["shares_sold"] == 2
    assert trim_calls["kwargs"]["order_id"] == "ord-trim-1"
    assert trim_calls["kwargs"]["execution_status"] == "submitted"


def test_save_signals_uses_next_trading_day_after_6pm(tmp_path, monkeypatch):
    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 2, 25, 18, 30, 0)

    monkeypatch.setattr(signal_gen_module, "datetime", _FixedDateTime)
    monkeypatch.setattr(signal_gen_module, "LOG_DIR", Path(tmp_path))

    gen = signal_gen_module.AgenticSignalGenerator.__new__(
        signal_gen_module.AgenticSignalGenerator
    )
    gen.logger = logging.getLogger("test.agentic_signal_generator")

    out = signal_gen_module.AgenticSignalGenerator.save_signals(
        gen,
        candidates=[{"ticker": "ABC", "symbol": "ABC", "action": "buy_open"}],
        min_count=1,
    )

    assert out.name == "signals_2026-02-26.json"
    assert out.exists()


def test_save_signals_sanitizes_atr_fields_and_can_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(signal_gen_module, "LOG_DIR", Path(tmp_path))

    gen = signal_gen_module.AgenticSignalGenerator.__new__(
        signal_gen_module.AgenticSignalGenerator
    )
    gen.logger = logging.getLogger("test.agentic_signal_generator")

    existing = Path(tmp_path) / "signals_2026-02-26.json"
    existing.write_text('{"signals": [{"ticker": "OLD"}]}', encoding="utf-8")

    out = signal_gen_module.AgenticSignalGenerator.save_signals(
        gen,
        candidates=[
            {
                "ticker": "ABC",
                "symbol": "ABC",
                "action": "buy_open",
                "stop_atr_mult": 99.0,
                "target_atr_mult": 99.0,
                "strategy_params": {
                    "stop_atr_mult": 99.0,
                    "target_atr_mult": 99.0,
                },
            }
        ],
        target_date="2026-02-26",
        allow_overwrite=True,
        min_count=1,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    row = payload["signals"][0]

    assert row["stop_atr_mult"] == 2.0
    assert row["target_atr_mult"] == 3.0
    assert row["strategy_params"]["stop_atr_mult"] == 2.0
    assert row["strategy_params"]["target_atr_mult"] == 3.0


def test_save_signals_backfills_entry_score_from_final_score(tmp_path, monkeypatch):
    monkeypatch.setattr(signal_gen_module, "LOG_DIR", Path(tmp_path))

    gen = signal_gen_module.AgenticSignalGenerator.__new__(
        signal_gen_module.AgenticSignalGenerator
    )
    gen.logger = logging.getLogger("test.agentic_signal_generator")

    out = signal_gen_module.AgenticSignalGenerator.save_signals(
        gen,
        candidates=[
            {
                "ticker": "ABC",
                "symbol": "ABC",
                "action": "buy_open",
                "final_score": 87.5,
            }
        ],
        target_date="2026-02-26",
        allow_overwrite=True,
        min_count=1,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    row = payload["signals"][0]

    assert row["entry_score"] == 87.5
    assert row["final_score"] == 87.5


def test_save_signals_exact_universe_mode_preserves_adjusted_plan_symbols(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(signal_gen_module, "LOG_DIR", Path(tmp_path))

    gen = signal_gen_module.AgenticSignalGenerator.__new__(
        signal_gen_module.AgenticSignalGenerator
    )
    gen.logger = logging.getLogger("test.agentic_signal_generator")

    out = signal_gen_module.AgenticSignalGenerator.save_signals(
        gen,
        candidates=[
            {
                "ticker": "SPY",
                "symbol": "SPY",
                "action": "buy_open",
                "entry_source": "premarket_adjusted",
            },
            {
                "ticker": "ABC",
                "symbol": "ABC",
                "action": "buy_open",
                "market_cap": 500_000_000,
                "entry_source": "premarket_adjusted",
            },
        ],
        target_date="2026-02-26",
        allow_overwrite=True,
        min_count=1,
        enforce_filters=False,
        metadata={"plan_file": "adjusted_plan_20260226_0829.json"},
    )

    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["total_signals"] == 2
    assert [row["ticker"] for row in payload["signals"]] == ["SPY", "ABC"]
    assert payload["signal_manifest"]["source"] == "unknown"
    assert payload["signal_manifest"]["enforce_filters"] is False
    assert payload["signal_manifest"]["plan_file"] == "adjusted_plan_20260226_0829.json"
    assert payload["signal_manifest"]["saved_total"] == 2
