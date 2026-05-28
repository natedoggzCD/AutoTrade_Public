
import pytest
import pandas as pd
from datetime import datetime, timedelta
from autotrade.analysis.execution_quality_tracker import ExecutionQualityTracker

def test_calculate_slippage_vs_vwap():
    """Test slippage calculation against a benchmark VWAP."""
    tracker = ExecutionQualityTracker()
    
    # Buy trade: execution price > VWAP means positive slippage (bad)
    exec_price = 100.10
    vwap_benchmark = 100.00
    qty = 100
    side = "buy"
    
    slippage_bps = tracker.calculate_slippage_bps(exec_price, vwap_benchmark, side)
    
    # (100.10 - 100.00) / 100.00 = 0.001 = 10 bps
    assert slippage_bps == pytest.approx(10.0)

def test_calculate_implementation_shortfall():
    """Test Implementation Shortfall (IS) calculation."""
    tracker = ExecutionQualityTracker()
    
    # IS = (Execution Price - Decision Price) / Decision Price
    # Decision Price is the price at the time the decision to trade was made.
    decision_price = 100.00
    exec_price = 100.05
    qty = 100
    side = "buy"
    
    is_bps = tracker.calculate_is_bps(exec_price, decision_price, side)
    
    # (100.05 - 100.00) / 100.00 = 0.0005 = 5 bps
    assert is_bps == pytest.approx(5.0)

def test_side_aware_slippage():
    """Test that slippage calculation respects trade side."""
    tracker = ExecutionQualityTracker()
    
    # Sell trade: execution price < VWAP means positive slippage (bad)
    exec_price = 99.90
    vwap_benchmark = 100.00
    side = "sell"
    
    slippage_bps = tracker.calculate_slippage_bps(exec_price, vwap_benchmark, side)
    
    # (100.00 - 99.90) / 100.00 = 0.001 = 10 bps
    assert slippage_bps == pytest.approx(10.0)
