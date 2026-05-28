from __future__ import annotations

import pandas as pd

from autotrade.monitoring.liquidity_gate import LiquidityGate


def test_liquidity_gate_rejects_wide_spreads():
    gate = LiquidityGate(
        min_avg_volume=200_000,
        max_spread_pct=0.5,
        min_price=2.0,
        min_session_volume=25_000,
    )

    decision = gate.evaluate(
        price=10.0,
        bid_price=9.90,
        ask_price=10.10,
        avg_volume=500_000,
        session_volume=100_000,
    )

    assert decision.tradable is False
    assert decision.reason == "spread_too_wide"
    assert decision.spread_pct > 0.5


def test_liquidity_gate_rejects_low_volume():
    gate = LiquidityGate(min_avg_volume=300_000, min_session_volume=25_000)

    decision = gate.evaluate(
        price=12.0,
        bid_price=11.99,
        ask_price=12.01,
        avg_volume=100_000,
        session_volume=50_000,
    )

    assert decision.tradable is False
    assert decision.reason == "below_min_avg_volume"


def test_liquidity_gate_filters_candidate_frame():
    gate = LiquidityGate(min_avg_volume=200_000, max_spread_pct=0.5)
    frame = pd.DataFrame(
        [
            {
                "ticker": "LIQD",
                "premarket_price": 12.0,
                "bid_price": 11.99,
                "ask_price": 12.01,
                "avg_volume": 500_000,
                "premarket_volume": 40_000,
            },
            {
                "ticker": "WIDE",
                "premarket_price": 10.0,
                "bid_price": 9.90,
                "ask_price": 10.10,
                "avg_volume": 500_000,
                "premarket_volume": 40_000,
            },
            {
                "ticker": "THIN",
                "premarket_price": 8.0,
                "bid_price": 7.99,
                "ask_price": 8.01,
                "avg_volume": 80_000,
                "premarket_volume": 40_000,
            },
        ]
    )

    filtered = gate.filter_candidates(frame)

    assert list(filtered["ticker"]) == ["LIQD"]
