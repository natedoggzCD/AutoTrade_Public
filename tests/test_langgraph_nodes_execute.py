from datetime import datetime

from langgraph_workflow.nodes import node_execute


class _TradingClientStub:
    def __init__(self):
        self.submitted = []

    def submit_order(self, order_request):
        self.submitted.append(order_request)
        raise AssertionError("submit_order should not be called for zero-qty trims")


def test_node_execute_skips_zero_qty_trim_orders():
    trading_client = _TradingClientStub()
    state = {
        "position": {
            "symbol": "TEST",
            "qty": 1,
            "current_price": 10.0,
        },
        "final_action": {
            "action": "trim",
            "size_pct": 25,
        },
    }

    result = node_execute(
        state, trading_client=trading_client, allow_live_execute=True
    )

    assert result["execution_result"]["executed"] is False
    assert result["execution_result"]["error"] == "Calculated trim qty is 0"
    assert result["errors"] == ["Calculated trim qty is 0"]
    assert trading_client.submitted == []


def test_node_execute_blocks_same_day_trim_orders():
    trading_client = _TradingClientStub()
    state = {
        "position": {
            "symbol": "TEST",
            "qty": 40,
            "current_price": 10.0,
            "entry_time": datetime.now().isoformat(),
        },
        "final_action": {
            "action": "trim",
            "size_pct": 25,
        },
    }

    result = node_execute(
        state, trading_client=trading_client, allow_live_execute=True
    )

    assert result["execution_result"]["executed"] is False
    assert result["execution_result"]["error"] == "Same-day trim blocked for newly opened position"
    assert trading_client.submitted == []


class _CaptureTradingClient:
    def __init__(self):
        self.submitted = []

    def submit_order(self, order_request):
        self.submitted.append(order_request)
        return type("Order", (), {"id": "ord-1"})()


def test_node_execute_consolidates_micro_trim_size():
    trading_client = _CaptureTradingClient()
    state = {
        "position": {
            "symbol": "TEST",
            "qty": 90,
            "current_price": 10.0,
            "entry_time": "2026-03-24T10:00:00",
        },
        "final_action": {
            "action": "trim",
            "size_pct": 10,
        },
    }

    result = node_execute(
        state, trading_client=trading_client, allow_live_execute=True
    )

    assert result["execution_result"]["executed"] is True
    assert len(trading_client.submitted) == 1
    assert int(trading_client.submitted[0].qty) == 22


def test_node_execute_is_advisory_by_default_even_with_client():
    trading_client = _CaptureTradingClient()
    state = {
        "position": {
            "symbol": "TEST",
            "qty": 90,
            "current_price": 10.0,
            "entry_time": "2026-03-24T10:00:00",
        },
        "final_action": {
            "action": "trim",
            "size_pct": 10,
        },
    }

    result = node_execute(state, trading_client=trading_client)

    assert result["execution_result"]["executed"] is False
    assert (
        result["execution_result"]["error"]
        == "LangGraph live execution disabled - advisory only"
    )
    assert trading_client.submitted == []
