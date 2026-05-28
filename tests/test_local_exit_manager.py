from __future__ import annotations
import pytest
from unittest.mock import MagicMock
from datetime import datetime, timedelta
from autotrade.execution.local_exit_manager import LocalExitManager

def test_local_exit_manager_initialization():
    manager = LocalExitManager()
    assert manager is not None

def test_check_stop_loss_hit():
    manager = LocalExitManager()
    
    # Mock position
    pos = MagicMock()
    pos.symbol = "AAPL"
    pos.side = "long"
    pos.qty = 10
    pos.avg_entry_price = 150.0
    pos.current_price = 145.0
    pos.stop_price = 146.0  # Stop price higher than current price
    
    decisions = manager.check_exits([pos])
    assert len(decisions) == 1
    assert decisions[0]["symbol"] == "AAPL"
    assert decisions[0]["action"] == "exit"
    assert "stop_loss" in decisions[0]["reason"]

def test_check_profit_target_hit():
    manager = LocalExitManager()
    
    # Mock position
    pos = MagicMock()
    pos.symbol = "TSLA"
    pos.side = "long"
    pos.qty = 5
    pos.avg_entry_price = 200.0
    pos.current_price = 225.0
    pos.target_price = 220.0  # Target price lower than current price
    pos.stop_price = 190.0
    
    decisions = manager.check_exits([pos])
    assert len(decisions) == 1
    assert decisions[0]["symbol"] == "TSLA"
    assert decisions[0]["action"] == "exit"
    assert "profit_target" in decisions[0]["reason"]

def test_dynamic_atr_trailing_stop():
    manager = LocalExitManager()
    
    # Mock position with ATR info
    pos = MagicMock()
    pos.symbol = "NVDA"
    pos.side = "long"
    pos.qty = 20
    pos.avg_entry_price = 100.0
    pos.current_price = 110.0
    pos.atr_14 = 2.0
    pos.target_price = 0.0
    pos.stop_price = 0.0
    # Trailing stop should be current_price - 2 * ATR = 110 - 4 = 106
    # If price drops to 105, it should trigger.
    
    pos.current_price = 105.0
    # We'll assume the manager computes the trailing stop dynamically or it's stored
    
    # Initial check (sets high water mark)
    manager.check_exits([pos])
    
    # Price moves up
    pos.current_price = 120.0
    manager.check_exits([pos])
    
    # Price drops below trailing stop (120 - 2*2 = 116)
    pos.current_price = 115.0
    decisions = manager.check_exits([pos])
    
    assert len(decisions) == 1
    assert decisions[0]["symbol"] == "NVDA"
    assert decisions[0]["action"] == "exit"
    assert "trailing_stop" in decisions[0]["reason"]
