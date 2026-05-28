import logging

import pytest

pytest.importorskip("alpaca.trading.client")

from autotrade.core import day_manager as day_manager_module
from autotrade.core.day_manager import DayManager


def test_get_current_price_skips_alpaca_when_data_client_missing(monkeypatch, caplog):
    dm = object.__new__(DayManager)
    dm.data_client = None

    monkeypatch.setattr(
        day_manager_module,
        "resolve_alpaca_credentials",
        lambda require=False: None,
    )
    monkeypatch.setattr(day_manager_module, "INTRADAY_PROVIDER_AVAILABLE", True)
    monkeypatch.setattr(
        day_manager_module,
        "get_current_price_with_fallback",
        lambda symbol, data_client=None: (123.45, "test_fallback"),
    )

    with caplog.at_level(logging.WARNING):
        price = dm.get_current_price("SPY")

    assert price == 123.45
    assert dm.data_client is None
    assert "NoneType" not in caplog.text
    assert "Alpaca price failed for SPY" not in caplog.text


def test_get_current_price_refreshes_missing_data_client(monkeypatch):
    dm = object.__new__(DayManager)
    dm.data_client = None

    class Creds:
        api_key = "key"
        secret_key = "secret"
        paper = True

    class Quote:
        ask_price = 10.25
        bid_price = 10.20

    class Client:
        def get_stock_latest_quote(self, _request):
            return {"QQQ": Quote()}

    monkeypatch.setattr(
        day_manager_module,
        "resolve_alpaca_credentials",
        lambda require=False: Creds(),
    )
    monkeypatch.setattr(
        day_manager_module,
        "create_data_client",
        lambda **_kwargs: Client(),
    )

    assert dm.get_current_price("QQQ") == 10.25
    assert dm.data_client is not None
