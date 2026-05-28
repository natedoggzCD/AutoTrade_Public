
import pytest
from unittest.mock import MagicMock
from autotrade.core.day_manager import DayManager

@pytest.fixture
def day_manager():
    # Mock enough of DayManager to run _position_add_block_reason
    manager = MagicMock(spec=DayManager)
    
    # Internal state mocks
    manager.strategy_failsafe_snapshot = MagicMock()
    manager.strategy_failsafe_snapshot.halt_new_entries = True
    manager.strategy_failsafe_snapshot.level = "critical"
    
    manager._hard_stop_blocked_symbols_today = set()
    manager._add_blocked_symbols_today = set()
    manager._add_block_after_trim = False
    
    # Method mocks
    manager._long_exposure_increase_block_reason.return_value = ""
    manager._alpha_add_override_snapshot.return_value = {
        "relative_strength": 80.0,
        "is_proven_leader": True,
        "is_elite": False
    }
    manager._entries_blocked_by_core_data.return_value = (False, "")
    
    # We want to test the actual implementation of _position_add_block_reason
    # while mocking its dependencies.
    # Note: In Python, we can't easily swap out a method on an instance if it's already bound
    # to the class logic we want to test. We'll use the class method directly.
    return manager

def test_position_add_failsafe_halt_enforcement(day_manager):
    """
    Verify that _position_add_block_reason now enforces the failsafe halt.
    """
    symbol = "AAPL"
    
    # Call the actual method logic
    reason = DayManager._position_add_block_reason(day_manager, symbol)
    
    # Correct behavior: returns the failsafe reason
    assert reason == "failsafe_halt_entries_critical"

def test_scale_into_winner_failsafe_halt_enforcement(day_manager):
    """
    Verify that scale_into_winner now skips orders when entries are halted.
    """
    pos_info = {
        "symbol": "AAPL",
        "current_qty": 100,
        "price": 150.0,
        "entry_context": "scale_into_winner"
    }
    
    day_manager.scale_into_winner = DayManager.scale_into_winner.__get__(day_manager, DayManager)
    
    # Use real _position_add_block_reason logic for the test
    day_manager._position_add_block_reason = DayManager._position_add_block_reason.__get__(day_manager, DayManager)
    
    day_manager._submit_order_via_execution_adapter = MagicMock()
    
    result = day_manager.scale_into_winner(pos_info)
    
    # Correct behavior: returns False (skipped)
    assert result is False
    assert not day_manager._submit_order_via_execution_adapter.called
