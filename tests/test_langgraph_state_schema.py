import importlib
import warnings


def test_langgraph_state_stop_price_alias_keeps_schema_and_drops_local_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import langgraph_workflow.state as state

        importlib.reload(state)

    local_warnings = [
        item
        for item in caught
        if "langgraph_workflow\\state.py" in str(getattr(item, "filename", ""))
    ]
    assert local_warnings == []

    candidate = state.CandidateAction(stop_price=10.0)
    final_action = state.FinalAction(stop_price=10.0)
    order = state.OrderData(
        order_id="ord-1",
        symbol="ABC",
        qty=10,
        side="buy",
        order_type="limit",
        stop_price=10.0,
        submitted_at="2026-03-27T10:00:00",
        planned_entry=11.0,
        planned_stop=10.0,
        planned_target=13.0,
    )

    assert candidate.model_dump(by_alias=True)["stop_price"] == 10.0
    assert final_action.model_dump(by_alias=True)["stop_price"] == 10.0
    assert order.model_dump(by_alias=True)["stop_price"] == 10.0
