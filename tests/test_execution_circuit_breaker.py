
import pytest
from unittest.mock import MagicMock, patch
from autotrade.core.day_manager import DayManager

def test_execution_circuit_breaker_halts_entries():
    """Verify that entries are halted when IS avg > 10 bps."""
    dm = DayManager.__new__(DayManager)
    dm.logger = MagicMock()
    
    # Mock IS history (last 5 trades)
    # Average = (15 + 12 + 8 + 11 + 14) / 5 = 12 bps (> 10 bps)
    dm._execution_is_history = [15.0, 12.0, 8.0, 11.0, 14.0]
    dm._execution_circuit_breaker_threshold_bps = 10.0
    dm._execution_circuit_breaker_window = 5
    
    # This method should return False (halt entry)
    halted = dm._check_execution_circuit_breaker()
    assert halted is True
    
def test_execution_circuit_breaker_allows_entries():
    """Verify that entries are allowed when IS avg <= 10 bps."""
    dm = DayManager.__new__(DayManager)
    dm.logger = MagicMock()
    
    # Mock IS history (last 5 trades)
    # Average = (5 + 2 + 8 + 1 + 4) / 5 = 4 bps (<= 10 bps)
    dm._execution_is_history = [5.0, 2.0, 8.0, 1.0, 4.0]
    dm._execution_circuit_breaker_threshold_bps = 10.0
    dm._execution_circuit_breaker_window = 5
    
    halted = dm._check_execution_circuit_breaker()
    assert halted is False
