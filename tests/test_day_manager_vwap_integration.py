
import pytest
from unittest.mock import MagicMock, patch
from autotrade.core.day_manager import DayManager
import pandas as pd
from datetime import datetime

def test_vwap_score_adjustment():
    """Test the VWAP score adjustment logic in DayManager."""
    with patch('autotrade.core.day_manager.DayManager._init_clients'), \
         patch('autotrade.core.day_manager.resolve_alpaca_credentials'):
        dm = DayManager(dry_run=True)
        
        # Mock vwap_data
        vwap_data = {
            "vwap": 100.0,
            "std": 2.0
        }
        
        # Test Case 1: Extreme discount (deviation < -1.0)
        # Price = 97.0, deviation = (97-100)/2 = -1.5
        adj = dm._apply_vwap_score_adjustment("TEST", 97.0, vwap_data)
        assert adj == 10.0
        
        # Test Case 2: Slight discount (-1.0 <= deviation < 0)
        # Price = 99.0, deviation = (99-100)/2 = -0.5
        adj = dm._apply_vwap_score_adjustment("TEST", 99.0, vwap_data)
        assert adj == 5.0
        
        # Test Case 3: Near fair value (0 <= deviation < 0.5)
        # Price = 100.5, deviation = (100.5-100)/2 = 0.25
        adj = dm._apply_vwap_score_adjustment("TEST", 100.5, vwap_data)
        assert adj == 2.0
        
        # Test Case 4: Slightly chasing (0.5 <= deviation < 1.0)
        # Price = 101.5, deviation = (101.5-100)/2 = 0.75
        adj = dm._apply_vwap_score_adjustment("TEST", 101.5, vwap_data)
        assert adj == -5.0
        
        # Test Case 5: Heavily chasing (1.0 <= deviation < 2.0)
        # Price = 103.0, deviation = (103-100)/2 = 1.5
        adj = dm._apply_vwap_score_adjustment("TEST", 103.0, vwap_data)
        assert adj == -15.0
        
        # Test Case 6: Extreme chasing (deviation >= 2.0)
        # Price = 105.0, deviation = (105-100)/2 = 2.5
        adj = dm._apply_vwap_score_adjustment("TEST", 105.0, vwap_data)
        assert adj == -25.0

@patch('autotrade.core.day_manager.get_intraday_bars_batch')
def test_find_replacement_candidates_with_vwap(mock_batch_bars):
    """Test that find_replacement_candidates fetches VWAP data and applies it."""
    with patch('autotrade.core.day_manager.DayManager._init_clients'), \
         patch('autotrade.core.day_manager.resolve_alpaca_credentials'):
        dm = DayManager(dry_run=True)
        dm.signals = [
            {"ticker": "AAPL", "score": 50, "rsi": 50},
            {"ticker": "TSLA", "score": 50, "rsi": 50}
        ]
        dm.signal_status = {"AAPL": {"status": "pending"}, "TSLA": {"status": "pending"}}
        
        # Mock batch bars
        aapl_bars = pd.DataFrame({
            "close": [100.0] * 20, "high": [101.0] * 20, "low": [99.0] * 20, "volume": [100] * 20
        }, index=pd.date_range(datetime.now(), periods=20, freq='min'))
        tsla_bars = pd.DataFrame({
            "close": [200.0] * 20, "high": [202.0] * 20, "low": [198.0] * 20, "volume": [100] * 20
        }, index=pd.date_range(datetime.now(), periods=20, freq='min'))
        
        mock_batch_bars.return_value = {"AAPL": aapl_bars, "TSLA": tsla_bars}
        
        # Mock get_current_price
        dm.get_current_price = MagicMock(side_effect=lambda t: 98.0 if t == "AAPL" else 205.0)
        dm.entry_quality_cfg = MagicMock()
        dm.entry_quality_cfg.stale_signal_max_age_days = 7
        dm.entry_quality_cfg.stale_signal_penalty_points = 20
        dm.youtube_context = {"available": False}
        dm.live_sector_bias = {}
        
        candidates = dm.find_replacement_candidates(held_tickers=[])
        
        # AAPL price 98.0 is below VWAP 100.0 -> Should have bonus
        # TSLA price 205.0 is above VWAP 200.0 (deviation (205-200)/std) -> Should have penalty
        
        aapl_cand = next(c for c in candidates if c["ticker"] == "AAPL")
        tsla_cand = next(c for c in candidates if c["ticker"] == "TSLA")
        
        assert aapl_cand["realtime_score"] > tsla_cand["realtime_score"]
        assert "realtime_score" in aapl_cand

def test_power_hour_execution_logic(caplog):
    """Test that DayManager applies enhanced sizing and tighter stops for power hour entries."""
    import logging
    caplog.set_level(logging.INFO)
    
    with patch('autotrade.core.day_manager.DayManager._init_clients'), \
         patch('autotrade.core.day_manager.resolve_alpaca_credentials'):
        dm = DayManager(dry_run=True)
        dm.last_account_equity = 100000
        dm.strategy_failsafe_snapshot = MagicMock()
        dm.strategy_failsafe_snapshot.position_size_multiplier = 1.0
        dm.strategy_failsafe_snapshot.position_size_pct = 0
        dm.strategy_failsafe_snapshot.halt_new_entries = False
        dm.runtime_risk_gate = None

        
        # Mock signal data with is_power_hour=True
        signal_data = {
            "ticker": "AAPL",
            "is_power_hour": True,
            "atr_14": 2.0,
            "position_size": 10000
        }
        
        # We need to mock get_current_price for execute_entry
        dm.get_current_price = MagicMock(return_value=150.0)
        dm.can_enter_positions = MagicMock(return_value=(True, "ok"))
        dm._check_learned_lessons = MagicMock(return_value=(False, ""))
        dm._has_open_buy_order = MagicMock(return_value=(False, None))
        dm._is_strategy_window_open = MagicMock(return_value=True)
        dm.get_current_phase = MagicMock(return_value=MagicMock(value="core"))
        dm._vl_confirm_entry = MagicMock(return_value=(True, "ok"))
        dm._validate_order = MagicMock(return_value=(True, "ok"))
        dm.execution_accounting = MagicMock()
        dm.execution_accounting.as_dict.return_value = {}
        
        # Call execute_entry
        dm.execute_entry("AAPL", "Power Hour Test", candidate_data=signal_data)
        
        # Check for power hour specific logs in caplog
        log_messages = caplog.text
        assert "[POWER-HOUR] Enhanced sizing multiplier: 1.2x" in log_messages
        assert "[POWER-HOUR] Tightening stop multiplier" in log_messages



