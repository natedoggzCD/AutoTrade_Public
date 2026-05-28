
import pytest
from unittest.mock import MagicMock
from autotrade.signals.inverse_etf_screener import InverseETFScreener

@pytest.fixture
def screener():
    db = MagicMock()
    # Mock liquid ETFs in DB
    db.get_all_inverse_etfs.return_value = [
        {"ticker": "SQQQ", "avg_daily_volume": 10_000_000, "aum_millions": 1000.0, "category": "index"},
        {"ticker": "SH", "avg_daily_volume": 5_000_000, "aum_millions": 500.0, "category": "index"}
    ]
    
    data_client = MagicMock()
    # Mock bars df
    import pandas as pd
    import numpy as np
    
    def mock_bars(ticker):
        dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='5min')
        df = pd.DataFrame({
            "open": np.linspace(100, 110, 100),
            "high": np.linspace(101, 111, 100),
            "low": np.linspace(99, 109, 100),
            "close": np.linspace(100.5, 110.5, 100),
            "volume": [1000] * 100
        }, index=dates)
        return df

    screener = InverseETFScreener(db, data_client=data_client)
    screener._fetch_intraday_bars = MagicMock(side_effect=mock_bars)
    return screener

def test_inverse_etf_skip_on_healthy_neutral(screener):
    """
    Verify that screener skips on a healthy NEUTRAL day.
    """
    results = screener.screen_universe(
        regime="neutral",
        sources_degraded=False,
        breadth_pct_positive=50.0
    )
    assert len(results) == 0

def test_inverse_etf_fire_on_degraded_bearish_neutral(screener):
    """
    Verify that screener fires on a degraded day with bearish breadth even if regime is NEUTRAL.
    """
    results = screener.screen_universe(
        regime="neutral",
        sources_degraded=True,
        breadth_pct_positive=40.0
    )
    # Should not be empty anymore
    assert len(results) > 0
    assert any(r["ticker"] == "SQQQ" for r in results)

def test_inverse_etf_fire_on_bearish_regime(screener):
    """
    Verify that screener still fires on an explicitly bearish regime.
    """
    results = screener.screen_universe(
        regime="bear",
        sources_degraded=False,
        breadth_pct_positive=50.0
    )
    assert len(results) > 0
