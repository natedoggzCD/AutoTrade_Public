from __future__ import annotations

import pytest

from autotrade.monitoring.contracts import ExecutionQualityMetrics


def test_execution_quality_computes_tca_metrics_buy():
    metrics = ExecutionQualityMetrics(
        order_id="o1",
        symbol="AAPL",
        side="buy",
        order_type="limit",
        quantity=100.0,
        fill_quantity=100.0,
        fill_price=100.2,
        expected_price=100.0,
        arrival_price=100.0,
        bid_price_at_arrival=99.9,
        ask_price_at_arrival=100.1,
        execution_latency_ms=35.0,
        status="filled",
    )
    metrics.compute_slippage_metrics()
    assert metrics.implementation_shortfall_bps > 0
    assert metrics.effective_spread_bps > 0
    assert metrics.quoted_spread_bps == pytest.approx(20.0)


def test_execution_quality_tracks_quoted_vs_effective_spread():
    metrics = ExecutionQualityMetrics(
        order_id="o3",
        symbol="AAPL",
        side="buy",
        order_type="limit",
        quantity=100.0,
        fill_quantity=100.0,
        fill_price=100.2,
        arrival_price=100.0,
        bid_price_at_arrival=99.9,
        ask_price_at_arrival=100.1,
        status="filled",
    )
    metrics.compute_slippage_metrics()
    assert metrics.quoted_spread_bps == pytest.approx(20.0)
    assert metrics.effective_spread_bps == pytest.approx(40.0)


def test_execution_quality_opportunity_cost_for_unfilled_limit():
    metrics = ExecutionQualityMetrics(
        order_id="o2",
        symbol="AAPL",
        side="buy",
        order_type="limit",
        quantity=100.0,
        fill_quantity=0.0,
        limit_price=99.5,
        expected_price=100.0,
        arrival_price=100.0,
        status="cannot_fill",
    )
    metrics.compute_slippage_metrics()
    assert metrics.fill_rate == 0.0
    assert metrics.opportunity_cost_bps > 0
