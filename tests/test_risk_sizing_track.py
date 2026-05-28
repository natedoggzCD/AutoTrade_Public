from __future__ import annotations

from datetime import datetime, timezone

from autotrade.risk.contracts import PortfolioSnapshot, PositionSnapshot
from autotrade.risk.policy_engine import FailsafeAdapter, StandardSizingPolicy


def test_stop_distance_uses_max_of_atr_spread_and_liquidity_floor():
    policy = StandardSizingPolicy()

    stop_distance = policy.compute_stop_distance_dollars(
        entry_price=100.0,
        atr_14=1.0,
        bid_ask_spread=0.4,
        atr_multiplier=2.0,
        spread_multiplier=3.0,
        liquidity_floor_bps=50.0,
    )

    assert stop_distance == 2.0


def test_stop_distance_enforces_liquidity_floor_when_atr_and_spread_are_small():
    policy = StandardSizingPolicy()

    stop_distance = policy.compute_stop_distance_dollars(
        entry_price=100.0,
        atr_14=0.1,
        bid_ask_spread=0.05,
        atr_multiplier=1.0,
        spread_multiplier=1.0,
        liquidity_floor_bps=50.0,
    )

    assert stop_distance == 0.5


def test_calculate_size_uses_risk_budget_formula_with_liquidity_cap():
    portfolio = PortfolioSnapshot(
        total_equity=100000.0,
        cash_available=100000.0,
        buying_power=100000.0,
        positions_value=0.0,
    )
    policy = StandardSizingPolicy()

    qty = policy.calculate_size(
        symbol="AAPL",
        entry_price=100.0,
        portfolio=portfolio,
        risk_per_trade=0.02,
        atr_14=1.0,
        bid_ask_spread=0.1,
        equity=100000.0,
        risk_percent=0.02,
        atr_multiplier=2.0,
        spread_multiplier=2.0,
        liquidity_floor_bps=10.0,
        avg_1m_volume=10000.0,
        max_liquidity_take_pct=0.05,
    )

    # Risk budget => 2000 / 2.0 = 1000 shares, liquidity cap => 500.
    assert qty == 500


def test_failsafe_blocks_entry_when_open_risk_budget_is_consumed():
    position = PositionSnapshot(
        symbol="AAPL",
        entry_price=100.0,
        current_price=100.0,
        qty=100,
        entry_time=datetime.now(timezone.utc),
        atr_pct=5.0,
    )
    portfolio = PortfolioSnapshot(
        total_equity=10000.0,
        cash_available=5000.0,
        buying_power=10000.0,
        positions_value=10000.0,
        positions=[position],
    )
    adapter = FailsafeAdapter(
        failsafe_level="normal",
        halt_new_entries=False,
        max_positions=5,
        max_open_risk_r=4.0,
        risk_per_trade=0.02,
    )

    allowed, reason = adapter.check_entry_allowed("MSFT", portfolio, 70.0)

    assert allowed is False
    assert reason is not None
    assert "open risk" in reason.lower()
