from types import SimpleNamespace

import pandas as pd

from langgraph_workflow import nodes


def test_load_price_history_uses_local_provider_before_yfinance(monkeypatch):
    frame = pd.DataFrame(
        {
            "Open": [10.0, 10.5],
            "High": [10.4, 10.9],
            "Low": [9.8, 10.2],
            "Close": [10.2, 10.7],
            "Volume": [1000, 1200],
        },
        index=pd.to_datetime(["2026-04-10", "2026-04-11"]),
    )

    monkeypatch.setattr(
        "autotrade.utils.local_data_provider.get_provider",
        lambda: SimpleNamespace(
            get_ticker_data=lambda symbol, days=60, include_features=False: frame
        ),
    )

    class _BrokenTicker:
        def history(self, period="60d"):
            raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(nodes.yf, "Ticker", lambda symbol: _BrokenTicker())

    history = nodes._load_price_history("ACLS", days=60)

    assert not history.empty
    assert history["Close"].tolist() == [10.2, 10.7]
