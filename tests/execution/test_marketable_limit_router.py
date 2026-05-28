import pytest

from autotrade.execution.router import build_execution_router


def test_router_builds_buy_marketable_limit_request_from_quote():
    router = build_execution_router(
        mode="sim",
        execution_config={
            "marketable_limit": {
                "atr_multiplier": 0.1,
                "min_offset_bps": 1.0,
                "max_offset_bps": 100.0,
            }
        },
    )

    request = router.build_marketable_limit_request(
        symbol="AAPL",
        side="buy",
        qty=10,
        action="entry",
        reason="breakout",
        reference_price=100.0,
        atr_14=0.4,
        bid_price=99.95,
        ask_price=100.05,
        urgency_tier="high",
    )

    assert request.order_type == "limit"
    assert request.limit_price == pytest.approx(100.09)
    assert request.intended_price == pytest.approx(100.0)
    assert request.decision_price == pytest.approx(100.09)
    assert request.urgency_tier == "high"
    assert request.metadata["pricing_mode"] == "marketable_limit"
    assert request.metadata["bid_price"] == pytest.approx(99.95)
    assert request.metadata["ask_price"] == pytest.approx(100.05)


def test_router_submits_marketable_limit_order_via_active_adapter():
    router = build_execution_router(
        mode="sim",
        execution_config={
            "sim": {
                "seed": 5,
                "latency_ms_min": 25,
                "latency_ms_max": 25,
                "partial_fill_enabled": False,
                "cannot_fill_probability": 0.0,
            },
            "marketable_limit": {
                "atr_multiplier": 0.1,
                "min_offset_bps": 1.0,
                "max_offset_bps": 100.0,
            },
        },
    )

    report = router.submit_marketable_limit_order(
        symbol="AMD",
        side="buy",
        qty=10,
        action="entry",
        reason="momentum",
        reference_price=100.0,
        atr_14=0.4,
        bid_price=99.95,
        ask_price=100.05,
        urgency_tier="high",
    )

    assert report.status in {"filled", "partial"}
    assert report.decision_price == pytest.approx(100.09)
    assert report.metadata["order_type"] == "limit"


def test_router_caps_marketable_limit_to_planned_limit():
    router = build_execution_router(
        mode="sim",
        execution_config={
            "marketable_limit": {
                "atr_multiplier": 0.1,
                "min_offset_bps": 1.0,
                "max_offset_bps": 100.0,
            }
        },
    )

    request = router.build_marketable_limit_request(
        symbol="AAPL",
        side="buy",
        qty=10,
        action="entry",
        reason="breakout",
        reference_price=100.0,
        atr_14=0.4,
        bid_price=99.95,
        ask_price=100.05,
        max_limit_price=100.06,
    )

    assert request.limit_price == pytest.approx(100.06)
    assert request.decision_price == pytest.approx(100.06)
    assert request.metadata["limit_cap_price"] == pytest.approx(100.06)
