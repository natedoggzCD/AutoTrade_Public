from __future__ import annotations

from datetime import datetime, timezone

from autotrade.core.threshold_alerts import PriceThreshold, ThresholdAlertEngine


def test_threshold_alert_engine_triggers_on_vwap_cross_up():
    engine = ThresholdAlertEngine()

    alerts = engine.evaluate(
        symbol="AAPL",
        previous_price=99.5,
        current_price=100.2,
        thresholds=[PriceThreshold(name="vwap", value=100.0)],
        triggered_at=datetime(2026, 3, 7, 15, 5, tzinfo=timezone.utc),
    )

    assert len(alerts) == 1
    assert alerts[0].threshold_name == "vwap"
    assert alerts[0].direction == "up"


def test_threshold_alert_engine_triggers_on_support_break_down():
    engine = ThresholdAlertEngine()

    alerts = engine.evaluate_levels(
        symbol="MSFT",
        previous_price=251.0,
        current_price=249.8,
        support_levels=[250.0],
    )

    assert len(alerts) == 1
    assert alerts[0].threshold_name == "support_1"
    assert alerts[0].direction == "down"


def test_threshold_alert_engine_triggers_on_resistance_break_out():
    engine = ThresholdAlertEngine()

    alerts = engine.evaluate_levels(
        symbol="NVDA",
        previous_price=119.7,
        current_price=120.4,
        resistance_levels=[120.0],
    )

    assert len(alerts) == 1
    assert alerts[0].threshold_name == "resistance_1"
    assert alerts[0].direction == "up"


def test_threshold_alert_engine_ignores_non_crossing_moves():
    engine = ThresholdAlertEngine()

    alerts = engine.evaluate_levels(
        symbol="AMD",
        previous_price=100.1,
        current_price=100.3,
        vwap=99.5,
        support_levels=[98.0],
        resistance_levels=[101.0],
    )

    assert alerts == []
