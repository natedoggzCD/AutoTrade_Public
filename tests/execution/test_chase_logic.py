from autotrade.execution.chase_logic import LimitChaseManager
from autotrade.execution.contracts import ExecutionReport
from autotrade.execution.router import make_order_request


def _submitted_report(
    order_id: str = "ord-1", limit_price: float = 100.05
) -> ExecutionReport:
    return ExecutionReport(
        order_id=order_id,
        status="submitted",
        symbol="AAPL",
        side="buy",
        requested_qty=10,
        filled_qty=0,
        avg_fill_price=0.0,
        venue="sim",
        intended_price=100.0,
        decision_price=limit_price,
        replace_count=0,
        metadata={"order_type": "limit", "limit_price": limit_price},
    )


def test_register_order_tracks_submitted_limit():
    manager = LimitChaseManager(min_reprice_bps=1.0)
    request = make_order_request(
        symbol="AAPL",
        side="buy",
        qty=10,
        action="entry",
        reason="test",
        order_type="limit",
        limit_price=100.05,
        reference_price=100.0,
        intended_price=100.0,
        decision_price=100.05,
        metadata={"atr_14": 0.4, "limit_cap_price": 100.20},
    )

    state = manager.register_order(_submitted_report(), request)

    assert state.order_id == "ord-1"
    assert state.state == "working"
    assert state.current_limit_price == 100.05
    assert state.atr_14 == 0.4


def test_evaluate_nbbo_returns_replacement_request_when_ask_moves_away():
    manager = LimitChaseManager(min_reprice_bps=1.0)
    request = make_order_request(
        symbol="AAPL",
        side="buy",
        qty=10,
        action="entry",
        reason="test",
        order_type="limit",
        limit_price=100.05,
        reference_price=100.0,
        intended_price=100.0,
        decision_price=100.05,
        metadata={"atr_14": 0.4, "limit_cap_price": 100.20},
    )
    manager.register_order(_submitted_report(), request)

    decision = manager.evaluate_nbbo("ord-1", bid_price=100.08, ask_price=100.12)

    assert decision.action == "replace"
    assert decision.replacement_request is not None
    assert decision.replacement_request.limit_price == 100.17
    assert decision.replacement_request.replace_count == 1


def test_chase_order_executes_cancel_replace_cycle():
    manager = LimitChaseManager(min_reprice_bps=1.0)
    request = make_order_request(
        symbol="AAPL",
        side="buy",
        qty=10,
        action="entry",
        reason="test",
        order_type="limit",
        limit_price=100.05,
        reference_price=100.0,
        intended_price=100.0,
        decision_price=100.05,
        metadata={"atr_14": 0.4, "limit_cap_price": 100.20},
    )
    manager.register_order(_submitted_report(), request)

    canceled = []

    def _cancel(order_id: str) -> ExecutionReport:
        canceled.append(order_id)
        return ExecutionReport(
            order_id=order_id,
            status="canceled",
            symbol="AAPL",
            side="buy",
            requested_qty=10,
            filled_qty=0,
            avg_fill_price=0.0,
            venue="sim",
        )

    def _submit(new_request):
        return ExecutionReport(
            order_id="ord-2",
            status="submitted",
            symbol=new_request.intent.symbol,
            side=new_request.intent.side,
            requested_qty=new_request.intent.qty,
            filled_qty=0,
            avg_fill_price=0.0,
            venue="sim",
            intended_price=new_request.intended_price,
            decision_price=new_request.limit_price,
            replace_count=new_request.replace_count,
            metadata={
                "order_type": new_request.order_type,
                "limit_price": new_request.limit_price,
            },
        )

    decision, new_state = manager.chase_order(
        "ord-1",
        bid_price=100.08,
        ask_price=100.12,
        cancel_order=_cancel,
        submit_order=_submit,
    )

    assert decision.action == "replace"
    assert canceled == ["ord-1"]
    assert new_state is not None
    assert new_state.order_id == "ord-2"
    assert new_state.state == "working"
    assert new_state.replace_count == 1


def test_evaluate_nbbo_holds_when_move_is_below_threshold():
    manager = LimitChaseManager(min_reprice_bps=15.0)
    request = make_order_request(
        symbol="AAPL",
        side="buy",
        qty=10,
        action="entry",
        reason="test",
        order_type="limit",
        limit_price=100.05,
        reference_price=100.0,
        intended_price=100.0,
        decision_price=100.05,
        metadata={"atr_14": 0.4, "limit_cap_price": 100.20},
    )
    manager.register_order(_submitted_report(), request)

    decision = manager.evaluate_nbbo("ord-1", bid_price=100.06, ask_price=100.07)

    assert decision.action == "hold"


def test_evaluate_nbbo_stops_replacing_after_max_attempts():
    manager = LimitChaseManager(min_reprice_bps=1.0, max_replacements=1)
    request = make_order_request(
        symbol="AAPL",
        side="buy",
        qty=10,
        action="entry",
        reason="test",
        order_type="limit",
        limit_price=100.05,
        reference_price=100.0,
        intended_price=100.0,
        decision_price=100.05,
        replace_count=1,
        metadata={"atr_14": 0.4, "limit_cap_price": 100.20},
    )
    manager.register_order(
        ExecutionReport(
            order_id="ord-1",
            status="submitted",
            symbol="AAPL",
            side="buy",
            requested_qty=10,
            filled_qty=0,
            avg_fill_price=0.0,
            venue="sim",
            intended_price=100.0,
            decision_price=100.05,
            replace_count=1,
            metadata={"order_type": "limit", "limit_price": 100.05},
        ),
        request,
    )

    decision = manager.evaluate_nbbo("ord-1", bid_price=100.08, ask_price=100.12)

    assert decision.action == "hold"
    assert decision.reason == "max_replacements_reached"


def test_evaluate_nbbo_holds_when_budget_cap_blocks_higher_limit():
    manager = LimitChaseManager(min_reprice_bps=1.0, max_replacements=3)
    request = make_order_request(
        symbol="AAPL",
        side="buy",
        qty=10,
        action="entry",
        reason="test",
        order_type="limit",
        limit_price=100.05,
        reference_price=100.0,
        intended_price=100.0,
        decision_price=100.05,
        slippage_budget_bps=10,
        metadata={"atr_14": 0.4},
    )
    manager.register_order(_submitted_report(), request)

    decision = manager.evaluate_nbbo("ord-1", bid_price=100.08, ask_price=100.12)

    assert decision.action == "hold"
    assert decision.reason == "hard_budget_reached"
