"""
Tests for Volume Profile Analysis (VLM) Feature Builder.
"""

import numpy as np
import pandas as pd
import pytest
from autotrade.feature_engineering.vlm import VolumeProfileAnalyzer

def _sample_ohlcv(n: int = 100) -> pd.DataFrame:
    """Generate a controlled OHLCV dataset with a clear volume profile."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=n, freq="min")
    
    # Create a price range [100, 110]
    # Concentrating volume around 105
    prices = np.linspace(100, 110, n)
    close = pd.Series(prices, index=idx)
    high = close + 0.1
    low = close - 0.1
    open_ = close.shift(1).fillna(100)
    
    # Volume peak at 105 (middle of the range)
    # Using a normal distribution for volume centered at index n/2
    x = np.arange(n)
    volume_profile = np.exp(-((x - n/2)**2) / (2 * (n/10)**2))
    volume = pd.Series(1000 + 5000 * volume_profile, index=idx).astype(int)
    
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )

def test_vlm_analyzer_computes_expected_columns():
    """Verify that the VLM analyzer produces POC and Value Area columns."""
    df = _sample_ohlcv()
    analyzer = VolumeProfileAnalyzer()
    
    # This should fail with ModuleNotFoundError initially
    result = analyzer.compute(df)
    
    expected_cols = {
        "vlm_poc",
        "vlm_va_high",
        "vlm_va_low",
        "vlm_poc_dist_pct",
        "vlm_in_va"
    }
    
    for col in expected_cols:
        assert col in result.columns
        assert not result[col].isnull().all()

def test_vlm_poc_calculation():
    """Verify that POC matches the highest volume price level."""
    # Create data with a very specific volume spike at $105
    df = _sample_ohlcv(n=10)
    # n=10, prices are 100, 101.1, ..., 110
    # Let's force index 5 (price ~105.5) to have 10x volume
    df.iloc[5, df.columns.get_loc("volume")] = 1000000
    
    analyzer = VolumeProfileAnalyzer()
    result = analyzer.compute(df)
    
    # POC should be near 105.5 (the close/high/low of index 5)
    expected_poc = df.iloc[5]["close"]
    assert np.isclose(result["vlm_poc"].iloc[-1], expected_poc, atol=0.5)

def test_vlm_va_encapsulation():
    """Verify that Value Area (70%) contains the expected volume."""
    df = _sample_ohlcv(n=100)
    analyzer = VolumeProfileAnalyzer(value_area_pct=0.7)
    result = analyzer.compute(df)
    
    last_row = result.iloc[-1]
    va_low = last_row["vlm_va_low"]
    va_high = last_row["vlm_va_high"]
    
    assert va_low < last_row["vlm_poc"] < va_high
    
    # Check if price is correctly flagged as being in VA
    # In our sample, price goes from 100 to 110. POC is 105.
    # VA should be roughly [102.5, 107.5]
    curr_price = last_row["close"] # 110
    if va_low <= curr_price <= va_high:
        assert last_row["vlm_in_va"] == 1.0
    else:
        assert last_row["vlm_in_va"] == 0.0

def test_vlm_poc_dist_pct():
    """Verify the distance to POC calculation."""
    df = _sample_ohlcv(n=10)
    df.iloc[0, df.columns.get_loc("volume")] = 1000000 # POC at $100
    
    analyzer = VolumeProfileAnalyzer()
    result = analyzer.compute(df)
    
    # Last price is $110, POC is $100. Dist should be 10%
    last_row = result.iloc[-1]
    expected_dist = (last_row["close"] / last_row["vlm_poc"]) - 1.0
    assert np.isclose(last_row["vlm_poc_dist_pct"], expected_dist, atol=0.01)

def test_vlm_volume_delta():
    """Verify that volume delta (buy vs sell) is calculated."""
    df = _sample_ohlcv(n=10)
    # Price is rising in _sample_ohlcv, so most bars should have positive delta
    analyzer = VolumeProfileAnalyzer()
    result = analyzer.compute(df)
    
    assert "vlm_delta" in result.columns
    # With rising prices, we expect positive delta (aggressive buying)
    assert (result["vlm_delta"] >= 0).all()

def test_vlm_divergence():
    """Verify that volume divergence is detected."""
    df = _sample_ohlcv(n=20)
    # Create a divergence: price rises but volume drops significantly
    df.loc[df.index[10:], "volume"] = 10
    
    analyzer = VolumeProfileAnalyzer()
    result = analyzer.compute(df)
    
    assert "vlm_divergence" in result.columns
    # Check if divergence is flagged in the second half
    assert result["vlm_divergence"].iloc[-1] == 1.0
