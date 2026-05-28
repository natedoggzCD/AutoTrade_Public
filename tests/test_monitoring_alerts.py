import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autotrade.monitoring.alerts import (
    Alert,
    AlertLevel,
    AlertManager,
    AlertRule,
    AlertType,
)


class TestAlertRules:
    def test_alert_rule_evaluation_greater_than(self):
        rule = AlertRule(
            name="test_rule",
            alert_type=AlertType.DRAWDOWN_BREACH,
            threshold=5.0,
            comparator=lambda v, t: v >= t,
        )

        assert rule.evaluate(6.0) is True
        assert rule.evaluate(4.0) is False

    def test_alert_rule_evaluation_less_than(self):
        rule = AlertRule(
            name="test_rule",
            alert_type=AlertType.TRADE_CONVERSION_DEGRADATION,
            threshold=0.05,
            comparator=lambda v, t: v < t,
        )

        assert rule.evaluate(0.03) is True
        assert rule.evaluate(0.10) is False

    def test_alert_rule_create_alert(self):
        rule = AlertRule(
            name="drawdown_alarm",
            alert_type=AlertType.DRAWDOWN_BREACH,
            threshold=5.0,
            comparator=lambda v, t: v >= t,
            level=AlertLevel.ERROR,
            description="Drawdown exceeds threshold",
        )

        alert = rule.create_alert(6.5, {"portfolio": "main"})

        assert alert.alert_type == "drawdown_breach"
        assert alert.level == "error"
        assert alert.actual_value == 6.5
        assert alert.threshold == 5.0
        assert alert.tags["portfolio"] == "main"


class TestAlertManagerRules:
    @pytest.fixture
    def alert_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AlertManager(output_dir=Path(tmpdir), enable_console=False)
            yield manager

    def test_default_rules_exist(self, alert_manager):
        assert AlertType.DRAWDOWN_BREACH in alert_manager._rules
        assert AlertType.TRADE_CONVERSION_DEGRADATION in alert_manager._rules
        assert AlertType.SLIPPAGE_DIVERGENCE in alert_manager._rules
        assert AlertType.DATA_STALENESS in alert_manager._rules
        assert AlertType.ALPHA_DEGRADATION in alert_manager._rules
        assert AlertType.REGIME_DRIFT in alert_manager._rules
        assert AlertType.EXECUTION_FAILURE in alert_manager._rules
        assert AlertType.ORDER_FAILURE in alert_manager._rules

    def test_add_rule(self, alert_manager):
        new_rule = AlertRule(
            name="custom_rule",
            alert_type=AlertType.DATA_STALENESS,
            threshold=48,
            comparator=lambda v, t: v > t,
        )

        alert_manager.add_rule(new_rule)
        assert AlertType.DATA_STALENESS in alert_manager._rules

    def test_remove_rule(self, alert_manager):
        alert_manager.remove_rule(AlertType.DATA_STALENESS)
        assert AlertType.DATA_STALENESS not in alert_manager._rules

    def test_baseline_conversion_rate(self, alert_manager):
        alert_manager.set_baseline_conversion_rate(0.15)
        assert alert_manager._baseline_conversion_rate == 0.15

    def test_update_baseline_conversion_rate(self, alert_manager):
        alert_manager.update_baseline_conversion_rate(0.12)
        alert_manager.update_baseline_conversion_rate(0.10)

        assert len(alert_manager._recent_conversion_rates) == 2
        assert alert_manager._baseline_conversion_rate == 0.12

    def test_baseline_alpha(self, alert_manager):
        alert_manager.set_baseline_alpha("momentum", 5.0)
        assert alert_manager._baseline_alpha["momentum"] == 5.0

    def test_baseline_regime_distribution(self, alert_manager):
        dist = {"trend": 0.5, "chop": 0.3, "crisis": 0.2}
        alert_manager.set_baseline_regime_distribution(dist)
        assert alert_manager._baseline_regime_dist == dist

    def test_record_failure(self, alert_manager):
        alert_manager.record_failure("api_error")
        assert alert_manager._failure_counts["api_error"] == 1

    def test_failure_count_tracking(self, alert_manager):
        for _ in range(5):
            alert_manager.record_failure("network_error")

        assert alert_manager._failure_counts["network_error"] == 5


class TestAlertEmission:
    @pytest.fixture
    def alert_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AlertManager(output_dir=Path(tmpdir), enable_console=False)
            yield manager

    def test_emit_alert_stores_in_history(self, alert_manager):
        alert = Alert(
            alert_type="test_alert",
            level="warning",
            message="Test alert",
        )

        alert_manager.emit(alert)

        assert len(alert_manager._alert_history) == 1
        assert alert_manager._alert_history[0] == alert

    def test_emit_triggers_callbacks(self, alert_manager):
        callback_called = {"count": 0}

        def test_callback(alert):
            callback_called["count"] += 1

        alert_manager.register_callback(test_callback)

        alert = Alert(
            alert_type="test_alert",
            level="info",
            message="Test",
        )
        alert_manager.emit(alert)

        assert callback_called["count"] == 1

    def test_emit_writes_to_file(self, alert_manager):
        alert = Alert(
            alert_type="drawdown_breach",
            level="error",
            message="Drawdown exceeded",
        )

        alert_manager.emit(alert)

        output_files = list(alert_manager.output_dir.glob("alerts_*.jsonl"))
        assert len(output_files) == 1

    def test_get_alerts_since(self, alert_manager):
        now = datetime.now(timezone.utc)
        old_alert = Alert(
            alert_type="test",
            level="info",
            message="Old",
            timestamp=now - timedelta(hours=48),
        )
        new_alert = Alert(
            alert_type="test",
            level="info",
            message="New",
            timestamp=now,
        )

        alert_manager.emit(old_alert)
        alert_manager.emit(new_alert)

        alerts = alert_manager.get_alerts_since(now - timedelta(hours=24))
        assert len(alerts) == 1

    def test_get_alert_summary(self, alert_manager):
        alert1 = Alert(
            alert_type="drawdown_breach",
            level="error",
            message="Test",
        )
        alert2 = Alert(
            alert_type="drawdown_breach",
            level="warning",
            message="Test2",
        )
        alert3 = Alert(
            alert_type="data_staleness",
            level="error",
            message="Test3",
        )

        alert_manager.emit(alert1)
        alert_manager.emit(alert2)
        alert_manager.emit(alert3)

        summary = alert_manager.get_alert_summary()

        assert summary["total_alerts"] == 3
        assert summary["by_level"]["error"] == 2
        assert summary["by_level"]["warning"] == 1
        assert summary["by_type"]["drawdown_breach"] == 2
        assert summary["by_type"]["data_staleness"] == 1


class TestPSIComputation:
    @pytest.fixture
    def alert_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AlertManager(output_dir=Path(tmpdir), enable_console=False)
            yield manager

    def test_psi_identical_distributions(self, alert_manager):
        dist = {"trend": 0.5, "chop": 0.3, "crisis": 0.2}
        psi = alert_manager._compute_psi(dist, dist)
        assert psi == 0.0

    def test_psi_similar_distributions(self, alert_manager):
        dist1 = {"trend": 0.5, "chop": 0.3, "crisis": 0.2}
        dist2 = {"trend": 0.48, "chop": 0.32, "crisis": 0.2}
        psi = alert_manager._compute_psi(dist1, dist2)
        assert psi < 0.1

    def test_psi_different_distributions(self, alert_manager):
        dist1 = {"trend": 0.8, "chop": 0.1, "crisis": 0.1}
        dist2 = {"trend": 0.2, "chop": 0.4, "crisis": 0.4}
        psi = alert_manager._compute_psi(dist1, dist2)
        assert psi > 0.3


class TestAlertLevels:
    def test_alert_level_ordering(self):
        assert AlertLevel.DEBUG.value == "debug"
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.CRITICAL.value == "critical"


class TestAlertTypes:
    def test_all_alert_types_defined(self):
        assert AlertType.DRAWDOWN_BREACH.value == "drawdown_breach"
        assert (
            AlertType.TRADE_CONVERSION_DEGRADATION.value
            == "trade_conversion_degradation"
        )
        assert AlertType.SLIPPAGE_DIVERGENCE.value == "slippage_divergence"
        assert AlertType.ALPHA_DEGRADATION.value == "alpha_degradation"
        assert AlertType.REGIME_DRIFT.value == "regime_drift"
        assert AlertType.DATA_STALENESS.value == "data_staleness"
        assert AlertType.EXECUTION_FAILURE.value == "execution_failure"
        assert AlertType.ORDER_FAILURE.value == "order_failure"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
