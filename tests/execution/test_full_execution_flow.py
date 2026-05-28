from __future__ import annotations

from autotrade.execution.chase_logic import LimitChaseManager
from autotrade.execution.contracts import ExecutionReport
from autotrade.execution.router import build_execution_router, make_order_request


def test_signal_limit_place_nbbo_shift_replace_fill_flow():
    router = build_execution_router(
        mode="sim",
        execution_config={
            "sim": {
                "seed": 7,
                "latency_ms_min": 1,
                "latency_ms_max": 1,
                "partial_fill_enabled": False,
                "cannot_fill_probability": 0.0,
            },
            "cost": {
                "commission_pct": 0.0,
                "slippage_pct": 0.0,
                "spread_pct": 0.0,
            },
            "marketable_limit": {
                "atr_multiplier": 0.10,
                "min_offset_bps": 5.0,
                "max_offset_bps": 100.0,
                "min_price": 0.01,
            },
        },
    )
    chase_manager = LimitChaseManager(
        price_calculator=router.price_calculator,
        min_reprice_bps=1.0,
        max_replacements=3,
    )

    initial_request = make_order_request(
        symbol="AAPL",
        side="buy",
        qty=10,
        action="execute_entry_limit",
        reason="signal",
        order_type="limit",
        limit_price=99.50,
        reference_price=100.00,
        intended_price=100.00,
        decision_price=99.50,
        slippage_budget_bps=20,
        metadata={"atr_14": 0.40},
    )
    initial_report = router.submit_order(initial_request)

    assert initial_report.status == "submitted"

    tracked = chase_manager.register_order(initial_report, initial_request)
    assert tracked.current_limit_price == 99.50

    canceled_reports: list[ExecutionReport] = []

    def _cancel(order_id: str) -> ExecutionReport:
        report = router.cancel_order(order_id)
        canceled_reports.append(report)
        return report

    decision, replacement_state = chase_manager.chase_order(
        order_id=initial_report.order_id,
        bid_price=100.08,
        ask_price=100.12,
        cancel_order=_cancel,
        submit_order=router.submit_order,
    )

    assert decision.action == "replace"
    assert decision.replacement_request is not None
    assert canceled_reports[0].status == "canceled"
    assert replacement_state is not None
    assert replacement_state.order_id != initial_report.order_id
    assert replacement_state.replace_count == 1
    assert replacement_state.state in {"working", "filled"}

    replacement_report = router.sim_adapter._orders[replacement_state.order_id]
    assert replacement_report.status == "filled"
    assert replacement_report.filled_qty == 10
    assert replacement_report.avg_fill_price > 0
