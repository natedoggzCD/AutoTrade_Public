from unittest.mock import patch

from autotrade.signals.alpha_pead import PEADAlphaSource
from autotrade.signals.alpha_squeeze import SqueezeAlphaSource
from autotrade.signals.contracts import SignalAction


@patch("autotrade.signals.alpha.pead.FinancialDB")
def test_alpha_pead_legacy_wrapper_supports_generate_signals(mock_db_class):
    mock_db = mock_db_class.return_value
    mock_db.get_recent_surprises.return_value = [
        {"ticker": "AAPL", "surprise_pct": 8.5, "earnings_date": "2026-03-20"}
    ]

    source = PEADAlphaSource()
    decisions = source.generate_signals()

    assert len(decisions) == 1
    assert decisions[0].ticker == "AAPL"
    assert decisions[0].action == SignalAction.BUY
    assert decisions[0].strategy_id == "pead_momentum_v1"


@patch("autotrade.signals.alpha.fundamental.FinancialDB")
def test_alpha_squeeze_legacy_wrapper_supports_generate_signals(mock_db_class):
    mock_db = mock_db_class.return_value
    mock_db.get_high_short_interest_tickers.return_value = [
        {"ticker": "GME", "short_pct": 24.0}
    ]

    source = SqueezeAlphaSource()
    decisions = source.generate_signals()

    assert len(decisions) == 1
    assert decisions[0].ticker == "GME"
    assert decisions[0].action == SignalAction.BUY
    assert decisions[0].strategy_id == "short_squeeze_v1"
