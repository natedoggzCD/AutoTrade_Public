from unittest.mock import MagicMock

import pandas as pd

from autotrade.core.decision_claw import DecisionClaw
from autotrade.signals.contracts import RegimeLabel
from autotrade.signals.inverse_etf_screener import InverseETFScreener
from autotrade.signals.regime_router import RegimeRouter


def _inverse_bars() -> pd.DataFrame:
    rows = 30
    close = [100.0 + (idx * 0.2) for idx in range(rows)]
    volume = [1000] * (rows - 1) + [4000]
    return pd.DataFrame(
        {
            "open": [value - 0.05 for value in close],
            "high": [value + 0.2 for value in close],
            "low": [value - 0.2 for value in close],
            "close": close,
            "volume": volume,
        }
    )


def test_sustained_bearish_regime_reaches_inverse_screener_and_claw(monkeypatch):
    router = RegimeRouter()
    regime = router.get_current_regime(
        price_data={
            "SPY": [{"open": 100.0, "close": 99.0}],
            "market_return_5d": -0.05,
            "breadth_pct_positive": 38.0,
            "realized_volatility": 0.032,
            "realized_vol_p70": 0.025,
        }
    )

    assert regime.regime == RegimeLabel.BEARISH
    assert regime.method == "sustained_bearish_5d"

    db = MagicMock()
    db.get_all_inverse_etfs.return_value = [
        {
            "ticker": "SQQQ",
            "name": "ProShares UltraPro Short QQQ",
            "avg_daily_volume": 10_000_000,
            "aum_millions": 1000.0,
            "category": "index",
            "leverage": 3,
            "underlying": "QQQ",
        }
    ]
    screener = InverseETFScreener(db, data_client=None)
    screener._fetch_intraday_bars = lambda ticker: _inverse_bars()
    monkeypatch.setattr(screener, "_calc_rsi", lambda bars, period=14: 55.0)
    monkeypatch.setattr(
        screener,
        "_calc_vwap",
        lambda bars: float(bars["close"].iloc[-1]),
    )

    candidates = screener.screen_universe(
        regime=regime.regime.value,
        sources_degraded=False,
        breadth_pct_positive=38.0,
    )

    assert candidates
    assert candidates[0]["ticker"] == "SQQQ"
    assert candidates[0]["signal"] == "ENTRY"

    claw = DecisionClaw.__new__(DecisionClaw)
    deployment_request = claw._normalize_deployment_request(
        {
            "mode": "inverse",
            "symbols": ["SQQQ"],
            "max_new_entries": 1,
            "reason": "sustained_bearish_regime",
        }
    )
    assert deployment_request["mode"] == "inverse"
    assert deployment_request["symbols"] == ["SQQQ"]


def test_breadth_and_realized_vol_trigger_bearish_without_crisis():
    router = RegimeRouter()
    regime = router.get_current_regime(
        price_data={
            "SPY": [{"open": 100.0, "close": 99.2}],
            "breadth_negative": True,
            "realized_volatility": 0.033,
            "realized_vol_p70": 0.025,
        }
    )

    assert regime.regime == RegimeLabel.BEARISH
    assert regime.method == "breadth_vol_bearish"
