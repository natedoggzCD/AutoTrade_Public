
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from autotrade.utils.vwap_calculator import VWAPCalculator

@pytest.fixture
def mock_bars():
    """Create mock OHLCV bars for a single day."""
    times = [datetime(2026, 2, 27, 9, 30) + timedelta(minutes=i) for i in range(10)]
    data = {
        "timestamp": times,
        "close": [100.0, 101.0, 102.0, 101.5, 103.0, 102.5, 104.0, 103.5, 105.0, 104.5],
        "high": [100.5, 101.5, 102.5, 102.0, 103.5, 103.0, 104.5, 104.0, 105.5, 105.0],
        "low": [99.5, 100.5, 101.5, 101.0, 102.5, 102.0, 103.5, 103.0, 104.5, 104.0],
        "volume": [100, 200, 150, 300, 100, 200, 150, 300, 100, 200]
    }
    return pd.DataFrame(data).set_index("timestamp")

def test_vwap_basic_calculation(mock_bars):
    """Verify basic VWAP calculation."""
    result = VWAPCalculator.calculate(mock_bars)
    
    # Calculate manually
    # Typical Price = (H + L + C) / 3
    tp = (mock_bars['high'] + mock_bars['low'] + mock_bars['close']) / 3.0
    pv = tp * mock_bars['volume']
    expected_vwap = pv.sum() / mock_bars['volume'].sum()
    
    assert result["vwap"] == pytest.approx(expected_vwap)
    assert result["last_price"] == 104.5

def test_vwap_bands(mock_bars):
    """Verify standard deviation bands."""
    result = VWAPCalculator.calculate(mock_bars)
    
    # Current implementation only has 'upper_band' and 'lower_band' (1.0 std)
    # The requirement says "bands" (plural), we should expect upper_band_1, upper_band_2, etc.
    # This test might fail if it's not implemented yet.
    assert "upper_band_1" in result
    assert "upper_band_2" in result
    assert result["upper_band_2"] > result["upper_band_1"]

def test_vwap_deviation(mock_bars):
    """Verify deviation calculation."""
    vwap_data = VWAPCalculator.calculate(mock_bars)
    dev = VWAPCalculator.get_deviation(106.0, vwap_data)
    
    # Expected deviation = (106 - vwap) / std
    expected_dev = (106.0 - vwap_data["vwap"]) / vwap_data["std"]
    assert dev == pytest.approx(expected_dev)

def test_vwap_intraday_reset():
    """Verify VWAP resets for new day data."""
    # Create 2 days of data
    times1 = [datetime(2026, 2, 26, 9, 30) + timedelta(minutes=i) for i in range(10)]
    times2 = [datetime(2026, 2, 27, 9, 30) + timedelta(minutes=i) for i in range(10)]
    
    data = {
        "timestamp": times1 + times2,
        "close": [10.0] * 10 + [100.0] * 10,
        "high": [10.5] * 10 + [100.5] * 10,
        "low": [9.5] * 10 + [99.5] * 10,
        "volume": [100] * 10 + [100] * 10
    }
    bars = pd.DataFrame(data).set_index("timestamp")
    
    # Calculation should only use the LAST day's data (intraday VWAP)
    result = VWAPCalculator.calculate(bars)
    
    # VWAP should be around 100, not around 55
    assert result["vwap"] > 90
