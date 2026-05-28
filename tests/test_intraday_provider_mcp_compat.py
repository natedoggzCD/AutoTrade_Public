from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd

from autotrade.utils import intraday_data_provider as idp
from autotrade.utils.mcp_client import MCPClientProxy


def test_mcp_proxy_serializes_stock_bars_request(monkeypatch):
    captured = {}

    def _fake_call_mcp_tool(server, tool, **kwargs):
        captured["server"] = server
        captured["tool"] = tool
        captured["kwargs"] = kwargs
        return {"ok": True}

    monkeypatch.setattr("autotrade.utils.mcp_client.call_mcp_tool", _fake_call_mcp_tool)

    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    req = StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=TimeFrame.Minute,
        start=datetime.now() - timedelta(minutes=30),
        end=datetime.now(),
    )
    proxy = MCPClientProxy("alpaca")
    proxy.get_stock_bars(req)

    assert captured["server"] == "alpaca"
    assert captured["tool"] == "get_stock_bars"
    assert captured["kwargs"]["symbol"] == "SPY"
    assert captured["kwargs"]["symbol_or_symbols"] == "SPY"
    assert captured["kwargs"]["timeframe"] == "1Min"
    assert isinstance(captured["kwargs"]["start"], str)
    assert isinstance(captured["kwargs"]["end"], str)


def test_get_intraday_bars_uses_5m_fallback_when_1m_unavailable(monkeypatch):
    calls = []

    def _fake_try_alpaca_then_mcp(
        ticker,
        data_client,
        minutes_back,
        alpaca_timeframe=None,
        interval="1m",
    ):
        calls.append((ticker, minutes_back, interval, alpaca_timeframe))
        if interval == "5m":
            idx = pd.date_range("2026-02-26 10:00:00", periods=3, freq="5min")
            return pd.DataFrame(
                {
                    "open": [10.0, 10.1, 10.2],
                    "high": [10.2, 10.3, 10.4],
                    "low": [9.9, 10.0, 10.1],
                    "close": [10.1, 10.2, 10.3],
                    "volume": [1000, 1100, 1200],
                },
                index=idx,
            )
        return None

    monkeypatch.setattr(idp, "_try_alpaca_then_mcp", _fake_try_alpaca_then_mcp)
    monkeypatch.setattr(idp, "_try_yfinance", lambda *args, **kwargs: None)

    out = idp.get_intraday_bars(
        "TEST",
        data_client=object(),
        minutes_back=60,
        interval="1m",
        min_bars=3,
    )

    assert out is not None
    assert len(out) == 3
    assert len(calls) == 2
    assert calls[0][2] == "1m"
    assert calls[1][2] == "5m"


def test_yfinance_fallback_uses_valid_one_day_period_for_two_hour_lookback(
    monkeypatch,
):
    captured = {}

    class _Ticker:
        def __init__(self, ticker):
            self.ticker = ticker

        def history(self, period, interval):
            captured["period"] = period
            captured["interval"] = interval
            idx = pd.date_range(datetime.now(), periods=3, freq="1min")
            return pd.DataFrame(
                {
                    "Open": [10.0, 10.1, 10.2],
                    "High": [10.2, 10.3, 10.4],
                    "Low": [9.9, 10.0, 10.1],
                    "Close": [10.1, 10.2, 10.3],
                    "Volume": [1000, 1100, 1200],
                },
                index=idx,
            )

    monkeypatch.setattr(idp, "_YF_AVAILABLE", True)
    monkeypatch.setattr(idp, "yf", SimpleNamespace(Ticker=_Ticker))

    out = idp._try_yfinance("TEST", minutes_back=120, interval="1m")

    assert out is not None
    assert captured == {"period": "1d", "interval": "1m"}


def test_yfinance_fallback_uses_valid_one_day_period_for_one_hour_lookback(
    monkeypatch,
):
    captured = {}

    class _Ticker:
        def __init__(self, ticker):
            self.ticker = ticker

        def history(self, period, interval):
            captured["period"] = period
            captured["interval"] = interval
            idx = pd.date_range(datetime.now(), periods=3, freq="1min")
            return pd.DataFrame(
                {
                    "Open": [10.0, 10.1, 10.2],
                    "High": [10.2, 10.3, 10.4],
                    "Low": [9.9, 10.0, 10.1],
                    "Close": [10.1, 10.2, 10.3],
                    "Volume": [1000, 1100, 1200],
                },
                index=idx,
            )

    monkeypatch.setattr(idp, "_YF_AVAILABLE", True)
    monkeypatch.setattr(idp, "yf", SimpleNamespace(Ticker=_Ticker))

    out = idp._try_yfinance("TEST", minutes_back=60, interval="1m")

    assert out is not None
    assert captured == {"period": "1d", "interval": "1m"}
