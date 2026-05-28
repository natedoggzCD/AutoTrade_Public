from __future__ import annotations

import tempfile
from pathlib import Path

from autotrade.monitoring.collector import MetricsCollector


def test_daily_tca_summary_aggregates_by_symbol_and_order_type():
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = MetricsCollector(
            output_dir=Path(tmpdir),
            enable_console_output=False,
        )

        collector.emit_execution_quality(
            order_id="o1",
            symbol="AAPL",
            side="buy",
            order_type="limit",
            quantity=10.0,
            fill_quantity=10.0,
            fill_price=100.2,
            expected_price=100.0,
            arrival_price=100.0,
            bid_price_at_arrival=99.9,
            ask_price_at_arrival=100.1,
            status="filled",
        )
        collector.emit_execution_quality(
            order_id="o2",
            symbol="AAPL",
            side="sell",
            order_type="market",
            quantity=5.0,
            fill_quantity=5.0,
            fill_price=99.8,
            expected_price=100.0,
            arrival_price=100.0,
            bid_price_at_arrival=99.9,
            ask_price_at_arrival=100.1,
            status="filled",
        )
        collector.emit_execution_quality(
            order_id="o3",
            symbol="MSFT",
            side="buy",
            order_type="limit",
            quantity=20.0,
            fill_quantity=0.0,
            limit_price=99.5,
            expected_price=100.0,
            arrival_price=100.0,
            status="cannot_fill",
        )

        summary = collector.get_daily_tca_summary(date="2026-02-28")
        assert summary["date"] == "2026-02-28"
        assert summary["total_orders"] == 3
        assert summary["by_symbol"]["AAPL"]["orders"] == 2
        assert summary["by_symbol"]["MSFT"]["orders"] == 1
        assert summary["by_order_type"]["limit"]["orders"] == 2
        assert summary["by_order_type"]["market"]["orders"] == 1
        assert summary["by_symbol"]["AAPL"]["avg_quoted_spread_bps"] > 0
        assert summary["by_order_type"]["limit"]["avg_effective_spread_bps"] > 0
        assert summary["opportunity_cost"]["unfilled_limit_orders"] == 1
        assert summary["opportunity_cost"]["total_bps"] > 0
