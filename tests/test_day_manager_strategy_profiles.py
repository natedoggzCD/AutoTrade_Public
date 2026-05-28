from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("alpaca.trading.client")

from autotrade.core.day_manager import DayManager

def _entry_quality_stub():
    return SimpleNamespace(
        strategy_profiles_enabled=True,
        opening_range_window_minutes=30,
        opening_range_breakout_score_bonus=12.0,
        opening_range_breakout_size_multiplier=1.15,
        vwap_mean_reversion_window_start_ct=630,
        vwap_mean_reversion_window_end_ct=750,
        vwap_mean_reversion_std_threshold=1.6,
        vwap_mean_reversion_score_bonus=16.0,
        vwap_mean_reversion_size_multiplier=1.10,
        vwap_mean_reversion_momentum_penalty_multiplier=0.35,
        gap_fill_window_end_ct=630,
        gap_fill_min_abs_gap_pct=1.5,
        gap_fill_score_bonus=10.0,
        gap_fill_size_multiplier=0.95,
    )

@pytest.fixture
def dm_mock():
    # This fixture is now more robust. It uses the real DayManager constructor
    # but patches out all external network calls and file I/O to isolate the logic.
    with patch("autotrade.core.day_manager.create_trading_client", return_value=MagicMock()), \
         patch("autotrade.core.day_manager.create_data_client", return_value=MagicMock()), \
         patch("autotrade.core.day_manager.get_intraday_bars", return_value=pd.DataFrame()), \
         patch("autotrade.core.day_manager.DayManager._load_signals", return_value=[]):
        
        dm = DayManager(dry_run=True)
        # Manually override settings for test stability
        dm.entry_quality_cfg = _entry_quality_stub()
        dm.universe = ["TEST", "ABC"]
        dm.get_positions = MagicMock(return_value=[])
        dm.premarket_data = {"TEST": {}, "ABC": {}}
        return dm

def _build_vwap_reversion_bars() -> pd.DataFrame:
    close = np.array(
        list(np.linspace(100.0, 100.8, 25))
        + list(np.linspace(100.2, 95.8, 10))
        + [96.0, 96.3, 96.6, 96.9, 97.2],
        dtype=float,
    )
    volume = np.array([22000] * 30 + [18000] * 5 + [12000] * 5, dtype=float)
    opens = np.r_[close[0], close[:-1]]
    highs = np.maximum(opens, close) + 0.2
    lows = np.minimum(opens, close) - 0.2
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": close,
            "volume": volume,
        }
    )

def test_vwap_mean_reversion_setup_activates_in_lunch_window(dm_mock):
    dm = dm_mock
    bars = _build_vwap_reversion_bars()
    now = datetime(2026, 2, 18, 11, 45)  # CT (11:30-13:30 ET window)

    profile = dm._evaluate_vwap_mean_reversion_setup("TEST", bars_df=bars, now=now)

    assert profile["strategy_profile"] == "vwap_mean_reversion"
    assert profile["active"] is True
    assert profile["score_delta"] > 0
    assert profile["momentum_penalty_multiplier"] < 1.0

def test_opening_range_breakout_setup_activates_near_open(dm_mock):
    dm = dm_mock
    close = np.array(
        list(np.linspace(100.0, 100.4, 30))
        + [100.55, 100.62, 100.70, 100.78, 100.85, 100.90, 100.95, 101.0]
    )
    volume = np.array([9000] * 30 + [18000] * 8, dtype=float)
    opens = np.r_[close[0], close[:-1]]
    highs = np.maximum(opens, close) + 0.08
    lows = np.minimum(opens, close) - 0.08
    bars = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": close,
            "volume": volume,
        }
    )
    now = datetime(2026, 2, 18, 8, 45)  # CT open window

    profile = dm._evaluate_opening_range_breakout_setup("TEST", bars_df=bars, now=now)

    assert profile["strategy_profile"] == "opening_range_breakout"
    assert profile["active"] is True
    assert profile["score_delta"] > 0

def test_strategy_profile_overlay_prefers_vwap_mean_reversion(dm_mock):
    dm = dm_mock
    bars = _build_vwap_reversion_bars()
    now = datetime(2026, 2, 18, 11, 45)

    ctx = dm._evaluate_intraday_strategy_profile(
        ticker="TEST",
        signal_data={"ticker": "TEST", "action": "watch"},
        bars_df=bars,
        now=now,
    )

    assert ctx["strategy_profile"] == "vwap_mean_reversion"
    assert ctx["strategy_active"] is True
    assert ctx["strategy_risk_budget_pct"] == pytest.approx(25.0)

def test_find_replacement_candidates_promotes_watch_when_strategy_active(dm_mock):
    dm = dm_mock
    skipped = []

    # Mock required methods surgically to isolate the promotion logic
    dm._mark_signal_skipped = MagicMock()
    dm._should_promote_watch_signal = MagicMock(return_value=(False, ""))
    dm._apply_candidate_backtest_validation = MagicMock(side_effect=lambda cands: cands)
    dm.strategy = MagicMock()
    dm.strategy.passes_filter.return_value = (True, 65.0, ["test_pass"])

    # This is the key: Mock the method that calculates the score that triggers promotion
    def _fake_realtime_score(sig, vwap_data=None):
        sig["strategy_active"] = True
        sig["strategy_profile"] = "vwap_mean_reversion"
        sig["strategy_score_delta"] = 18.0
        return 62.0
    dm._calculate_realtime_score = MagicMock(side_effect=_fake_realtime_score)

    # Prevent the real _init_intraday_data from running and clearing signals
    dm._init_intraday_data = MagicMock()

    with patch("autotrade.core.day_manager.INTRADAY_PROVIDER_AVAILABLE", True):
        # Manually set signals AFTER DM init to prevent overwrite
        dm.signals = [
            {
                "ticker": "ABC",
                "action": "watch",
                "score": 50,
                "strategy_active": True,
                "strategy_profile": "vwap_mean_reversion",
            }
        ]
        dm.signal_status = {"ABC": {"status": "pending"}}
        cands = dm.find_replacement_candidates(held_tickers=set())

    assert len(cands) == 1
    assert cands[0]["ticker"] == "ABC"
    assert cands[0]["action"] == "buy_open"
    assert skipped == []
    dm._mark_signal_skipped.assert_not_called()
