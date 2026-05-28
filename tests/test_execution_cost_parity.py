import pytest

from autotrade.backtesting.execution import ExecutionModel
from autotrade.execution.router import make_order_request
from autotrade.execution.sim_adapter import SimAdapterConfig, SimExecutionAdapter


def test_execution_costs_match_backtesting_model_buy():
    cfg = SimAdapterConfig(
        seed=7,
        latency_ms_min=0,
        latency_ms_max=0,
        partial_fill_enabled=False,
        cannot_fill_probability=0.0,
        commission_pct=0.001,
        slippage_pct=0.0005,
        spread_pct=0.001,
    )
    adapter = SimExecutionAdapter(cfg)
    model = ExecutionModel(
        commission_pct=cfg.commission_pct,
        slippage_pct=cfg.slippage_pct,
        spread_pct=cfg.spread_pct,
        partial_fill_min=1.0,
        partial_fill_max=1.0,
        seed=cfg.seed,
    )
    req = make_order_request(
        symbol="AAPL",
        side="buy",
        qty=10,
        action="entry",
        reason="parity",
        order_type="market",
        reference_price=100.0,
    )

    report = adapter.submit_order(req)

    expected_price = model.effective_price(100.0, side="buy")
    expected_notional = expected_price * 10
    expected_commission = model.commission(expected_notional)
    expected_spread = abs(expected_notional) * cfg.spread_pct / 2.0
    expected_slippage = abs(expected_notional) * cfg.slippage_pct

    assert report.status == "filled"
    assert report.avg_fill_price == pytest.approx(expected_price)
    assert report.total_fees == pytest.approx(expected_commission)
    assert report.total_spread == pytest.approx(expected_spread)
    assert report.total_slippage == pytest.approx(expected_slippage)


def test_execution_costs_match_backtesting_model_sell():
    cfg = SimAdapterConfig(
        seed=11,
        latency_ms_min=0,
        latency_ms_max=0,
        partial_fill_enabled=False,
        cannot_fill_probability=0.0,
        commission_pct=0.001,
        slippage_pct=0.0005,
        spread_pct=0.001,
    )
    adapter = SimExecutionAdapter(cfg)
    model = ExecutionModel(
        commission_pct=cfg.commission_pct,
        slippage_pct=cfg.slippage_pct,
        spread_pct=cfg.spread_pct,
        partial_fill_min=1.0,
        partial_fill_max=1.0,
        seed=cfg.seed,
    )
    req = make_order_request(
        symbol="MSFT",
        side="sell",
        qty=15,
        action="exit",
        reason="parity",
        order_type="market",
        reference_price=50.0,
    )

    report = adapter.submit_order(req)

    expected_price = model.effective_price(50.0, side="sell")
    expected_notional = expected_price * 15
    expected_commission = model.commission(expected_notional)

    assert report.status == "filled"
    assert report.avg_fill_price == pytest.approx(expected_price)
    assert report.total_fees == pytest.approx(expected_commission)

