from __future__ import annotations
import pytest
from datetime import datetime
from autotrade.utils.execution_logger import OrderLifecycleLogger

def test_order_lifecycle_logger_initialization():
    logger = OrderLifecycleLogger()
    assert logger is not None

def test_log_signal_creation():
    logger = OrderLifecycleLogger()
    signal_id = "sig_123"
    symbol = "AAPL"
    arrival_price = 150.0
    nbbo = (149.95, 150.05)
    
    logger.log_signal(
        signal_id=signal_id,
        symbol=symbol,
        arrival_price=arrival_price,
        nbbo=nbbo
    )
    
    log = logger.get_log(signal_id)
    assert log["symbol"] == symbol
    assert log["arrival_price"] == arrival_price
    assert log["nbbo"] == nbbo
    assert "signal_time" in log

def test_log_event():
    logger = OrderLifecycleLogger()
    signal_id = "sig_123"
    logger.log_signal(signal_id, "AAPL", 150.0)
    
    now = datetime.now()
    logger.log_event(signal_id, "submission", timestamp=now)
    
    log = logger.get_log(signal_id)
    assert "submission" in log["events"]
    assert log["events"]["submission"] == now

def test_get_nonexistent_log():
    logger = OrderLifecycleLogger()
    with pytest.raises(KeyError):
        logger.get_log("nonexistent")

def test_capture_arrival_price_from_context():
    from autotrade.signals.contracts import SignalContext
    import pandas as pd
    import numpy as np
    
    # Create dummy price data
    df = pd.DataFrame({
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.5], # 102.5 is the "current bar close"
        "volume": [1000, 1100]
    }, index=pd.date_range("2026-03-01 09:30:00", periods=2, freq="1min"))
    
    context = SignalContext(tickers=["AAPL"], price_data=df)
    
    logger = OrderLifecycleLogger()
    # This method doesn't exist yet, it's the target of implementation
    arrival_price = logger.capture_arrival_price("AAPL", context)
    
    assert arrival_price == 102.5

def test_log_lifecycle_timestamps():
    logger = OrderLifecycleLogger()
    signal_id = "sig_123"
    logger.log_signal(signal_id, "AAPL", 150.0)
    
    # Submission
    t_sub = datetime.now()
    logger.log_event(signal_id, "submission", timestamp=t_sub)
    
    # Acknowledgement
    t_ack = datetime.now()
    logger.log_event(signal_id, "acknowledgement", timestamp=t_ack)
    
    # Fill
    t_fill = datetime.now()
    logger.log_event(signal_id, "fill", timestamp=t_fill)
    
    log = logger.get_log(signal_id)
    assert log["events"]["submission"] == t_sub
    assert log["events"]["acknowledgement"] == t_ack
    assert log["events"]["fill"] == t_fill

def test_capture_nbbo_logic():
    # This might require mocking the data provider
    from unittest.mock import MagicMock
    
    logger = OrderLifecycleLogger()
    mock_client = MagicMock()
    # Mock Alpaca response for latest quote
    mock_quote = MagicMock()
    mock_quote.ask_price = 150.05
    mock_quote.bid_price = 149.95
    
    mock_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}
    
    nbbo = logger.capture_nbbo("AAPL", mock_client)
    assert nbbo == (149.95, 150.05)

def test_calculate_is_and_decomposition():
    logger = OrderLifecycleLogger()
    
    # Buy Order
    # Arrival: 100.0
    # Submission: 100.10 (Delay: +0.10)
    # Executed: 100.25 (Execution: +0.15)
    # Total IS: 0.25 (100.25 - 100.0)
    
    arrival_price = 100.0
    submission_price = 100.10
    execution_price = 100.25
    side = "buy"
    
    is_total = logger.calculate_is(execution_price, arrival_price, side)
    assert is_total == pytest.approx(0.25)
    
    delay_cost = logger.calculate_delay_cost(submission_price, arrival_price, side)
    assert delay_cost == pytest.approx(0.10)
    
    exec_cost = logger.calculate_execution_cost(execution_price, submission_price, side)
    assert exec_cost == pytest.approx(0.15)
    
    # Sell Order
    # Arrival: 100.0
    # Submission: 99.90 (Delay: -0.10)
    # Executed: 99.70 (Execution: -0.20)
    # Total IS: -0.30 (99.70 - 100.0) * -1 = 0.30
    
    arrival_price = 100.0
    submission_price = 99.90
    execution_price = 99.70
    side = "sell"
    
    is_total = logger.calculate_is(execution_price, arrival_price, side)
    assert is_total == pytest.approx(0.30)

def test_calculate_effective_spread():
    logger = OrderLifecycleLogger()
    
    # Buy Order
    # NBBO: 100.00 / 100.10 (Midpoint: 100.05)
    # Executed: 100.08
    # Eff. Spread = 2 * (100.08 - 100.05) * 1 = 0.06
    
    nbbo = (100.00, 100.10)
    execution_price = 100.08
    side = "buy"
    
    eff_spread = logger.calculate_effective_spread(execution_price, nbbo, side)
    assert eff_spread == pytest.approx(0.06)
    
    # Sell Order
    # NBBO: 100.00 / 100.10 (Midpoint: 100.05)
    # Executed: 100.02
    # Eff. Spread = 2 * (100.02 - 100.05) * -1 = 2 * -0.03 * -1 = 0.06
    
    nbbo = (100.00, 100.10)
    execution_price = 100.02
    side = "sell"
    
    eff_spread = logger.calculate_effective_spread(execution_price, nbbo, side)
    assert eff_spread == pytest.approx(0.06)
