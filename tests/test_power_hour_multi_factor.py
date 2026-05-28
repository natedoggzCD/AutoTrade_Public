
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime, timedelta
from autotrade.core.day_manager import DayManager

@pytest.fixture
def mock_dm():
    dm = DayManager.__new__(DayManager)
    dm.logger = MagicMock()
    dm.data_client = MagicMock()
    dm.entry_quality_cfg = MagicMock()
    dm.entry_quality_cfg.power_hour_volume_enabled = True
    dm.entry_quality_cfg.power_hour_volume_threshold = 2.5
    dm.get_positions = MagicMock(return_value=[])
    dm.get_current_price = MagicMock(return_value=100.0)
    dm._is_power_hour = MagicMock(return_value=True)
    dm._safe_float = lambda x, d: float(x) if x is not None else d
    dm.signals = [{"ticker": "AAPL"}]
    return dm

@patch("autotrade.analysis.order_book_analyzer.OrderBookAnalyzer.get_market_imbalance")
@patch("autotrade.analysis.order_book_analyzer.OrderBookAnalyzer.detect_iceberg")
@patch("autotrade.utils.vwap_calculator.VWAPCalculator.calculate")
def test_scan_power_hour_multi_factor(mock_vwap, mock_iceberg, mock_l2, mock_dm):
    """Test the multi-factor scoring in power hour scanning."""
    # Mock Bars (Ascending order)
    times = [datetime.now() - timedelta(minutes=60-i) for i in range(60)]
    df = pd.DataFrame({
        "close": [100.0] * 30 + [101.0] * 30, # 1% price change (ascending)
        "volume": [100] * 30 + [400] * 30 # 4x surge
    }, index=times)
    mock_bars = MagicMock()
    mock_bars.df = pd.DataFrame(df)
    mock_bars.df.index = pd.MultiIndex.from_product([["AAPL"], times], names=["symbol", "timestamp"])
    mock_dm.data_client.get_stock_bars.return_value = mock_bars
    
    # Mock L2: Bullish imbalance (0.8)
    mock_l2.return_value = {"imbalance": 0.8, "spread_pct": 0.1}
    
    # Mock Iceberg: Detected (+10 bonus)
    mock_iceberg.return_value = {"iceberg_detected": True, "side": "bid", "confidence": 0.8}
    
    # Mock VWAP: Price (100) is above VWAP (99.5) -> +20 bonus
    mock_vwap.return_value = {"vwap": 99.5, "std": 0.2}
    
    # Run scan
    opps = mock_dm.scan_power_hour_opportunities()
    
    assert len(opps) == 1
    aapl = opps[0]
    
    # Check that new factors are in the output
    assert "power_hour_score" in aapl
    assert aapl["l2_imbalance"] == 0.8
    assert aapl["iceberg_detected"] is True
    assert aapl["above_vwap"] is True
    
    # Scoring should be high due to all positive factors
    assert aapl["power_hour_score"] > 80
