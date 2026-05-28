from __future__ import annotations

import pandas as pd
from unittest.mock import MagicMock, patch

from autotrade.signals.contracts import SignalContext, RegimeLabel
from autotrade.signals.alpha.inverse_etf import InverseETFAlphaSource


def test_hedge_simulation_generates_signals_for_crisis_regime():
    mock_config = MagicMock()
    mock_config.inverse_etf_hedging.enabled = True
    mock_config.inverse_etf_hedging.allowed_regimes = ["crisis"]
    mock_config.inverse_etf_hedging.symbols = ["SH", "PSQ", "DOG"]
    mock_config.inverse_etf_hedging.max_hold_days = 3

    with patch("autotrade.signals.alpha.inverse_etf.get_config", return_value=mock_config):
        source = InverseETFAlphaSource()
        context_crisis = SignalContext(
            tickers=[], price_data=pd.DataFrame(), regime=RegimeLabel.CRISIS
        )
        signals = source.generate(context_crisis)

    assert len(signals) == 3
    assert {s.ticker for s in signals} == {"SH", "PSQ", "DOG"}


def test_hedge_simulation_skips_non_crisis_regimes():
    mock_config = MagicMock()
    mock_config.inverse_etf_hedging.enabled = True
    mock_config.inverse_etf_hedging.allowed_regimes = ["crisis"]
    mock_config.inverse_etf_hedging.symbols = ["SH", "PSQ", "DOG"]
    mock_config.inverse_etf_hedging.max_hold_days = 3

    with patch("autotrade.signals.alpha.inverse_etf.get_config", return_value=mock_config):
        source = InverseETFAlphaSource()
        context_trend = SignalContext(
            tickers=[], price_data=pd.DataFrame(), regime=RegimeLabel.TREND
        )
        signals = source.generate(context_trend)

    assert signals == []
