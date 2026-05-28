from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("alpaca.trading.client")

from autotrade.core import day_manager as day_manager_mod
from autotrade.core.day_manager import DayManager
from tests.test_day_manager_execution_policy import _new_dm_stub


def _pm_wave_candidate(pm_wave_entry: bool = True) -> dict:
    return {
        "ticker": "PMW",
        "symbol": "PMW",
        "action": "buy_open",
        "score": 82.0,
        "confidence": 82.0,
        "entry_price": 100.0,
        "current_price": 112.0,
        "prev_close": 100.0,
        "entry_source": "overnight_plan",
        "source_bucket": "watchlist",
        "plan_score_source": "pm_plan_2026-05-04.json",
        "runtime_entry_context": "wave_1_entry" if pm_wave_entry else "",
        "pm_wave_entry": pm_wave_entry,
        "risk_reward": 1.4,
        "volume_ratio": 0.9,
        "atr_14": 2.0,
    }


def _configure_entry_preflight(dm: DayManager, candidate: dict) -> None:
    dm.dry_run = True
    dm.signal_status = {"PMW": {"status": "pending", "reason": ""}}
    dm.signals = [dict(candidate)]
    dm._watched_universe_tickers = {"PMW"}
    dm.premarket_gap_cfg = SimpleNamespace(moderate_gap_up_pct=3.0)
    dm.entry_quality_cfg.entry_gap_reject_pct = 7.0
    dm.entry_quality_cfg.wave_hard_reject_gap_pct = 12.0
    dm.entry_quality_cfg.wave_breakout_rescue_min_score = 75.0
    dm.entry_quality_cfg.wave_breakout_rescue_max_gap_pct = 9.0
    dm.entry_quality_cfg.pm_wave_gap_rescue_enabled = True
    dm.entry_quality_cfg.pm_wave_gap_rescue_min_score = 80.0
    dm.entry_quality_cfg.pm_wave_gap_rescue_max_gap_pct = 18.0
    dm._effective_market_regime = lambda: "NEUTRAL"
    dm._entries_blocked_by_core_data = lambda: (False, "")
    dm.get_current_phase = lambda: day_manager_mod.TradingPhase.RESEARCH
    dm._late_session_entry_block_reason = lambda **kwargs: ""
    dm._entries_blocked_by_regime = lambda ticker: (False, "")
    dm.get_positions = lambda: []
    dm._validated_positions = lambda positions, context=None: list(positions or [])
    dm._recent_symbol_quarantine_reason = lambda symbol: ""
    dm._learning_blocked_symbol_reason = lambda symbol: ""
    dm._recent_exit_reentry_reason = lambda symbol: ""
    dm._entry_guard_reasons = lambda *args, **kwargs: []
    dm._entry_capacity_block_reason = lambda *args, **kwargs: ""
    dm._defensive_long_block_reason = lambda *args, **kwargs: ""
    dm._position_slot_class = lambda **kwargs: "core"
    dm.can_enter_positions = lambda *args, **kwargs: (True, "")
    dm._build_candidate_validation_report = lambda signal_data: {
        "allowed": True,
        "entry_source": "overnight_plan",
    }
    dm._resolve_entry_authority = lambda signal_data: {
        "eligible": True,
        "entry_score": 82.0,
        "entry_source": "overnight_plan",
        "plan_score_source": "pm_plan_2026-05-04.json",
    }
    dm._has_open_buy_order = lambda symbol: (False, "")
    dm._should_promote_watch_signal = lambda signal_data: (False, "")
    dm._is_strategy_window_open = lambda strategy_profile: True
    dm.get_current_price = lambda ticker: 112.0
    dm.get_bars = lambda ticker, limit=100: None
    dm._resolved_journal_entry_score = lambda *args, **kwargs: (None, "")
    dm._conviction_size_multiplier = lambda score: 1.0
    dm._regime_override_float = lambda *args, **kwargs: 1.0
    dm._is_breakout_continuation_setup = lambda signal_data: False
    dm._effective_stop_multiplier = lambda fallback=2.0: fallback
    dm._resolve_entry_urgency_tier = lambda score, signal_data: "normal"
    dm._entry_tier_limits = lambda urgency_tier: (0.0, 0.0)
    dm._compute_entry_limit_price = (
        lambda planned_entry, current_price, urgency_tier: planned_entry
    )
    dm._validate_order = lambda *args, **kwargs: (True, "")
    dm._entry_submission_block_reason = lambda *args, **kwargs: ""
    dm._record_wave_entry = lambda *args, **kwargs: None
    dm._record_strategy_profile_entry = lambda *args, **kwargs: None
    dm._mark_signal_skipped = DayManager._mark_signal_skipped.__get__(dm, DayManager)


def test_pm_wave_gap_up_rescue_clears_gap_vwap_and_l2_preflight(monkeypatch):
    candidate = _pm_wave_candidate(pm_wave_entry=True)
    dm = _new_dm_stub()
    _configure_entry_preflight(dm, candidate)

    bars = pd.DataFrame(
        {
            "high": [110.0, 112.0, 113.0],
            "low": [108.0, 110.0, 111.0],
            "close": [109.0, 111.0, 112.0],
            "volume": [1000, 1200, 1400],
        }
    )

    class _VWAP:
        @staticmethod
        def calculate(frame):
            return {"vwap": 105.0, "std": 5.0}

        @staticmethod
        def get_deviation(current_price, vwap_data):
            return 1.4

    class _OrderBook:
        @staticmethod
        def get_market_imbalance(ticker, data_client):
            return {"imbalance": 0.42, "spread_pct": 0.1}

    monkeypatch.setattr(
        "autotrade.utils.intraday_data_provider.get_intraday_bars",
        lambda *args, **kwargs: bars,
    )
    monkeypatch.setattr(day_manager_mod, "VWAPCalculator", _VWAP)
    monkeypatch.setattr(day_manager_mod, "OrderBookAnalyzer", _OrderBook)

    allowed = dm.execute_entry(
        "PMW",
        "pm wave rescue preflight",
        candidate_data=dict(candidate),
        entry_wave=1,
        preflight_only=True,
    )

    assert allowed is True
    assert dm.signal_status["PMW"]["reason"] not in {
        "entry_gap_too_large",
        "vwap_overextended",
        "l2_bearish_imbalance",
    }


def test_same_gap_without_pm_wave_entry_still_hits_normal_gap_block():
    dm = _new_dm_stub()
    _configure_entry_preflight(dm, _pm_wave_candidate(pm_wave_entry=False))

    allowed, anchor, reason = dm._resolve_runtime_entry_anchor(
        action="buy_open",
        entry_price=100.0,
        current_price=112.0,
        entry_score=82.0,
        signal_data=_pm_wave_candidate(pm_wave_entry=False),
    )

    assert allowed is False
    assert anchor == pytest.approx(100.0)
    assert reason == "entry_gap_too_large"
