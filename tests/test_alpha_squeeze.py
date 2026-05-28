import pytest
from unittest.mock import MagicMock, patch
import pandas as pd

from autotrade.signals.contracts import (
    SignalContext,
    SignalDecision,
    SignalAction,
    SignalFamily,
    RegimeLabel,
)
from autotrade.signals.alpha.fundamental import SqueezeAlphaSource

class TestSqueezeAlphaSource:
    def test_name_property(self):
        source = SqueezeAlphaSource()
        assert source.name == "SqueezeAlphaSource_v1"

    def test_family_property(self):
        source = SqueezeAlphaSource()
        assert source.family == SignalFamily.TS_MOMENTUM

    def test_version_property(self):
        source = SqueezeAlphaSource()
        assert isinstance(source.version, str)

    @patch("autotrade.signals.alpha.fundamental.FinancialDB")
    def test_generate_signals(self, mock_db_class):
        # Setup mock DB
        mock_db = mock_db_class.return_value
        mock_db.get_high_short_interest_tickers.return_value = [
            {"ticker": "GME", "short_pct": 25.0},
            {"ticker": "AMC", "short_pct": 18.5},
        ]

        source = SqueezeAlphaSource()
        
        # Create a dummy context
        context = SignalContext(
            tickers=["GME", "AMC", "TSLA"],
            price_data=pd.DataFrame(),
            regime=RegimeLabel.TREND
        )

        decisions = source.generate(context)

        assert len(decisions) == 2
        assert decisions[0].ticker == "GME"
        assert "SQUEEZE" in decisions[0].reason
        assert decisions[1].ticker == "AMC"
        assert decisions[0].strategy_id == "short_squeeze_v1"

    @patch("autotrade.signals.alpha.fundamental.FinancialDB")
    def test_generate_no_candidates(self, mock_db_class):
        mock_db = mock_db_class.return_value
        mock_db.get_high_short_interest_tickers.return_value = []

        source = SqueezeAlphaSource()
        context = SignalContext(tickers=["AAPL"], price_data=pd.DataFrame())
        
        decisions = source.generate(context)
        assert len(decisions) == 0
