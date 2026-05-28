import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import date

from autotrade.signals.contracts import (
    SignalContext,
    SignalDecision,
    SignalAction,
    SignalFamily,
    RegimeLabel,
)
from autotrade.signals.alpha.fundamental import PEADAlphaSource

class TestPEADAlphaSource:
    def test_name_property(self):
        source = PEADAlphaSource()
        assert source.name == "PEADAlphaSource_v1"

    def test_family_property(self):
        source = PEADAlphaSource()
        assert source.family == SignalFamily.TS_MOMENTUM

    def test_version_property(self):
        source = PEADAlphaSource()
        assert isinstance(source.version, str)

    @patch("autotrade.signals.alpha.fundamental.FinancialDB")
    def test_generate_signals(self, mock_db_class):
        # Setup mock DB
        mock_db = mock_db_class.return_value
        mock_db.get_recent_surprises.return_value = [
            {"ticker": "AAPL", "surprise_pct": 10.5, "earnings_date": "2024-01-01"},
            {"ticker": "MSFT", "surprise_pct": 7.2, "earnings_date": "2024-01-02"},
        ]

        source = PEADAlphaSource()
        
        # Create a dummy context
        context = SignalContext(
            tickers=["AAPL", "MSFT", "GOOGL"],
            price_data=pd.DataFrame(), # Empty for this test
            regime=RegimeLabel.TREND
        )

        decisions = source.generate(context)

        assert len(decisions) == 2
        assert decisions[0].ticker == "AAPL"
        assert "PEAD" in decisions[0].reason
        assert decisions[1].ticker == "MSFT"
        assert decisions[0].strategy_id == "pead_momentum_v1"

    @patch("autotrade.signals.alpha.fundamental.FinancialDB")
    def test_generate_no_surprises(self, mock_db_class):
        mock_db = mock_db_class.return_value
        mock_db.get_recent_surprises.return_value = []

        source = PEADAlphaSource()
        context = SignalContext(tickers=["AAPL"], price_data=pd.DataFrame())
        
        decisions = source.generate(context)
        assert len(decisions) == 0
