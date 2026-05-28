import pytest

from autotrade.execution.price_logic import OrderPriceCalculator


def test_compute_offset_scales_with_atr():
    calculator = OrderPriceCalculator(
        atr_multiplier=0.1, min_offset_bps=5.0, max_offset_bps=100.0
    )

    low_vol = calculator.compute_offset(reference_price=25.0, atr_14=0.2)
    high_vol = calculator.compute_offset(reference_price=25.0, atr_14=1.5)

    assert low_vol == pytest.approx(0.02)
    assert high_vol == pytest.approx(0.15)
    assert high_vol > low_vol


def test_compute_offset_respects_minimum_floor_when_atr_is_small():
    calculator = OrderPriceCalculator(
        atr_multiplier=0.1, min_offset_bps=12.0, max_offset_bps=100.0
    )

    offset = calculator.compute_offset(reference_price=10.0, atr_14=0.01)

    assert offset == pytest.approx(0.012)


def test_compute_buy_limit_uses_ask_plus_offset():
    calculator = OrderPriceCalculator(
        atr_multiplier=0.1, min_offset_bps=1.0, max_offset_bps=100.0
    )

    limit_price = calculator.compute_marketable_limit(
        side="buy",
        reference_price=100.0,
        atr_14=0.4,
        bid_price=99.95,
        ask_price=100.05,
    )

    assert limit_price == pytest.approx(100.09)


def test_compute_sell_limit_uses_bid_minus_offset():
    calculator = OrderPriceCalculator(
        atr_multiplier=0.1, min_offset_bps=1.0, max_offset_bps=100.0
    )

    limit_price = calculator.compute_marketable_limit(
        side="sell",
        reference_price=100.0,
        atr_14=0.4,
        bid_price=99.95,
        ask_price=100.05,
    )

    assert limit_price == pytest.approx(99.91)


def test_compute_marketable_limit_falls_back_to_reference_when_quote_missing():
    calculator = OrderPriceCalculator(
        atr_multiplier=0.1, min_offset_bps=5.0, max_offset_bps=100.0
    )

    limit_price = calculator.compute_marketable_limit(
        side="buy",
        reference_price=42.0,
        atr_14=0.2,
        bid_price=None,
        ask_price=None,
    )

    assert limit_price == pytest.approx(42.02)
