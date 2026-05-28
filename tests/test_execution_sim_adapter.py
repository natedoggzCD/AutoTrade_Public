from autotrade.execution.router import make_order_request
from autotrade.execution.sim_adapter import SimAdapterConfig, SimExecutionAdapter


def test_sim_adapter_is_deterministic_for_same_seed():
    cfg = SimAdapterConfig(
        seed=123,
        latency_ms_min=50,
        latency_ms_max=50,
        partial_fill_enabled=True,
        cannot_fill_probability=0.0,
    )
    req = make_order_request(
        symbol="AAPL",
        side="buy",
        qty=100,
        action="entry",
        reason="test",
        order_type="market",
        reference_price=100.0,
    )

    report_1 = SimExecutionAdapter(cfg).submit_order(req)
    report_2 = SimExecutionAdapter(cfg).submit_order(req)

    assert report_1.status == report_2.status
    assert report_1.filled_qty == report_2.filled_qty
    assert report_1.avg_fill_price == report_2.avg_fill_price
    assert report_1.latency_ms == report_2.latency_ms


def test_sim_adapter_cannot_fill_state():
    cfg = SimAdapterConfig(
        seed=42,
        latency_ms_min=10,
        latency_ms_max=10,
        partial_fill_enabled=True,
        cannot_fill_probability=1.0,
    )
    adapter = SimExecutionAdapter(cfg)
    req = make_order_request(
        symbol="TSLA",
        side="buy",
        qty=50,
        action="entry",
        reason="test",
        order_type="market",
        reference_price=200.0,
    )

    report = adapter.submit_order(req)

    assert report.status == "cannot_fill"
    assert report.filled_qty == 0
    assert report.remaining_qty == 50


def test_sim_adapter_limit_submitted_then_canceled():
    cfg = SimAdapterConfig(
        seed=99,
        latency_ms_min=0,
        latency_ms_max=0,
        partial_fill_enabled=True,
        cannot_fill_probability=0.0,
    )
    adapter = SimExecutionAdapter(cfg)
    req = make_order_request(
        symbol="MSFT",
        side="buy",
        qty=25,
        action="entry",
        reason="test",
        order_type="limit",
        limit_price=95.0,
        reference_price=100.0,
    )

    submitted = adapter.submit_order(req)
    canceled = adapter.cancel_order(submitted.order_id)

    assert submitted.status == "submitted"
    assert canceled.status == "canceled"
    assert canceled.order_id == submitted.order_id


def test_sim_adapter_emits_execution_telemetry():
    cfg = SimAdapterConfig(
        seed=5,
        latency_ms_min=25,
        latency_ms_max=25,
        partial_fill_enabled=False,
        cannot_fill_probability=0.0,
    )
    adapter = SimExecutionAdapter(cfg)
    req = make_order_request(
        symbol="AMD",
        side="buy",
        qty=10,
        action="entry",
        reason="telemetry",
        order_type="limit",
        limit_price=100.05,
        reference_price=100.0,
        urgency_tier="high",
        intended_price=100.0,
        decision_price=100.05,
        slippage_budget_bps=20,
        replace_count=1,
    )

    report = adapter.submit_order(req)

    assert report.status in {"filled", "partial"}
    assert report.urgency_tier == "high"
    assert report.intended_price == 100.0
    assert report.decision_price == 100.05
    assert report.replace_count == 1
    assert report.slippage_bps is not None
