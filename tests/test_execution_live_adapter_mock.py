from types import SimpleNamespace

import pytest

pytest.importorskip("alpaca.trading.requests")

from autotrade.execution.alpaca_adapter import AlpacaExecutionAdapter
from autotrade.execution.router import make_order_request


class MockAlpacaClient:
    def __init__(self):
        self.last_request = None
        self.canceled = []
        self.next_order = SimpleNamespace(
            id="ord-1",
            status="filled",
            filled_qty="10",
            filled_avg_price="101.25",
        )

    def submit_order(self, request):
        self.last_request = request
        return self.next_order

    def cancel_order_by_id(self, order_id):
        self.canceled.append(order_id)


def test_live_adapter_maps_filled_status():
    client = MockAlpacaClient()
    adapter = AlpacaExecutionAdapter(client=client, default_order_type="market")
    req = make_order_request(
        symbol="AAPL",
        side="buy",
        qty=10,
        action="entry",
        reason="test",
        order_type="market",
        reference_price=100.0,
        urgency_tier="critical",
        intended_price=100.0,
        decision_price=100.1,
        replace_count=2,
    )

    report = adapter.submit_order(req)

    assert report.status == "filled"
    assert report.filled_qty == 10
    assert report.avg_fill_price == pytest.approx(101.25)
    assert report.order_id == "ord-1"
    assert report.urgency_tier == "critical"
    assert report.intended_price == 100.0
    assert report.decision_price == pytest.approx(100.1)
    assert report.replace_count == 2
    assert report.slippage_bps is not None


def test_live_adapter_maps_partial_status():
    client = MockAlpacaClient()
    client.next_order = SimpleNamespace(
        id="ord-2",
        status="partially_filled",
        filled_qty="4",
        filled_avg_price="99.75",
    )
    adapter = AlpacaExecutionAdapter(client=client)
    req = make_order_request(
        symbol="NVDA",
        side="sell",
        qty=10,
        action="exit",
        reason="test",
        order_type="market",
        reference_price=100.0,
    )

    report = adapter.submit_order(req)

    assert report.status == "partial"
    assert report.filled_qty == 4


def test_live_adapter_cancel_maps_canceled_status():
    client = MockAlpacaClient()
    adapter = AlpacaExecutionAdapter(client=client)

    report = adapter.cancel_order("ord-3")

    assert report.status == "canceled"
    assert report.order_id == "ord-3"
    assert client.canceled == ["ord-3"]
