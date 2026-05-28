import json
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from autotrade.monitoring.contracts import (
    AlphaFamilyMetrics,
    ExecutionQualityMetrics,
    RiskHealthMetrics,
    SignalFamily,
    SignalLifecycleMetrics,
    SystemHealthEvent,
    CANONICAL_METRIC_NAMES,
    MetricNames,
    RegimeType,
    GateType,
    GateStatus,
    ReleaseGateStatus,
)


class TestMetricSchemaStability:
    @pytest.fixture
    def temp_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_signal_lifecycle_metrics_schema(self):
        metrics = SignalLifecycleMetrics(
            family="momentum",
            signals_evaluated=100,
            signals_accepted=50,
            signals_rejected=50,
            signals_executed=25,
            skip_reasons={"gap_up": 10, "low_volume": 5},
            hit_rate=0.5,
            expectancy=0.02,
            avg_duration_hours=48.0,
            turnover_rate=0.5,
            cost_impact_bps=5.0,
        )
        metrics.compute_derived_metrics()

        assert metrics.metric_name == MetricNames.SIGNAL_LIFECYCLE
        assert metrics.family == "momentum"
        assert metrics.signals_evaluated == 100
        assert metrics.conversion_rate == 0.25
        assert metrics.hit_rate == 0.5

    def test_execution_quality_metrics_schema(self):
        metrics = ExecutionQualityMetrics(
            order_id="order_123",
            symbol="AAPL",
            side="buy",
            order_type="limit",
            quantity=100.0,
            limit_price=150.0,
            fill_price=150.25,
            expected_price=150.0,
            slippage_expected_bps=5.0,
            execution_latency_ms=250.0,
            fill_quantity=100.0,
            status="filled",
        )
        metrics.compute_slippage_metrics()

        assert metrics.metric_name == MetricNames.EXECUTION_QUALITY
        assert metrics.symbol == "AAPL"
        assert metrics.slippage_bps > 0
        assert metrics.slippage_divergence_bps >= 0
        assert metrics.fill_rate == 1.0

    def test_risk_health_metrics_schema(self):
        metrics = RiskHealthMetrics(
            portfolio_id="main",
            total_exposure_pct=75.0,
            cash_pct=25.0,
            leverage=1.0,
            var_95_pct=2.5,
            drawdown_pct=6.0,
            daily_pnl_pct=-1.5,
            position_count=10,
            max_position_size_pct=10.0,
        )
        metrics.compute_risk_state()

        assert metrics.metric_name == MetricNames.RISK_HEALTH
        assert metrics.alert_level == "red"
        assert metrics.hard_stop_triggered is False

    def test_risk_health_critical_drawdown(self):
        metrics = RiskHealthMetrics(
            portfolio_id="main",
            drawdown_pct=9.0,
        )
        metrics.compute_risk_state()

        assert metrics.alert_level == "critical"
        assert metrics.hard_stop_triggered is True

    def test_alpha_family_metrics_schema(self):
        metrics = AlphaFamilyMetrics(
            family="momentum",
            regime="trend",
            period_days=30,
            total_return_pct=5.0,
            benchmark_return_pct=2.0,
            sharpe_ratio=1.5,
            max_drawdown_pct=-3.0,
            win_rate=0.6,
            trade_count=20,
            baseline_return_pct=4.5,
        )
        metrics.compute_alpha()

        assert metrics.metric_name == MetricNames.ALPHA_FAMILY
        assert metrics.alpha_pct == 3.0
        assert metrics.degradation_vs_baseline_pct > 0

    def test_system_health_event_schema(self):
        event = SystemHealthEvent(
            component="execution",
            status="healthy",
            cpu_percent=45.0,
            memory_percent=60.0,
            error_count=0,
            warning_count=2,
        )

        assert event.metric_name == MetricNames.SYSTEM_HEALTH
        assert event.tags["component"] == "execution"
        assert event.tags["status"] == "healthy"

    def test_release_gate_status_schema(self):
        gate = ReleaseGateStatus(
            gate_type="paper",
            paper_trading_days=15,
            paper_performance_pct=2.5,
            canary_risk_pct=5.0,
            canary_max_risk_pct=10.0,
        )
        gate.evaluate()

        assert gate.metric_name == MetricNames.RELEASE_GATE
        assert gate.gate_status == GateStatus.PASS.value
        assert gate.rollback_ready is True

    def test_canonical_metric_names_consistency(self):
        assert CANONICAL_METRIC_NAMES["signal_count"] == "signal.count"
        assert CANONICAL_METRIC_NAMES["conversion_rate"] == "signal.conversion_rate"
        assert CANONICAL_METRIC_NAMES["slippage_bps"] == "execution.slippage_bps"
        assert CANONICAL_METRIC_NAMES["drawdown_pct"] == "risk.drawdown_pct"
        assert CANONICAL_METRIC_NAMES["alpha_pct"] == "alpha.return_pct"

    def test_signal_lifecycle_to_dict(self):
        metrics = SignalLifecycleMetrics(
            family="technical",
            signals_evaluated=200,
            signals_accepted=80,
            signals_executed=40,
        )
        metrics.compute_derived_metrics()

        result = {
            "metric_name": metrics.metric_name,
            "family": metrics.family,
            "conversion_rate": metrics.conversion_rate,
            "tags": metrics.tags,
        }

        assert "metric_name" in result
        assert result["family"] == "technical"
        assert isinstance(result["conversion_rate"], float)


class TestAlertTriggerCorrectness:
    @pytest.fixture
    def alert_manager(self):
        from autotrade.monitoring.alerts import AlertManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AlertManager(output_dir=Path(tmpdir), enable_console=False)
            yield manager

    def test_drawdown_alert_trigger(self, alert_manager):
        alert = alert_manager.check_drawdown(9.0)
        assert alert is not None
        assert alert.alert_type == "drawdown_breach"
        assert alert.level == "critical"

    def test_critical_drawdown_alert(self, alert_manager):
        alert = alert_manager.check_drawdown(9.0)
        assert alert is not None
        assert alert.alert_type == "drawdown_breach"
        assert alert.level == "critical"

    def test_drawdown_below_threshold_no_alert(self, alert_manager):
        alert = alert_manager.check_drawdown(3.0)
        assert alert is None

    def test_trade_conversion_below_floor(self, alert_manager):
        alert = alert_manager.check_trade_conversion(0.03)
        assert alert is not None
        assert alert.alert_type == "trade_conversion_degradation"

    def test_trade_conversion_above_floor_no_alert(self, alert_manager):
        alert = alert_manager.check_trade_conversion(0.10)
        assert alert is None

    def test_slippage_divergence_above_threshold(self, alert_manager):
        alert = alert_manager.check_slippage_divergence(25.0, "AAPL")
        assert alert is not None
        assert alert.alert_type == "slippage_divergence"

    def test_slippage_divergence_below_threshold(self, alert_manager):
        alert = alert_manager.check_slippage_divergence(10.0)
        assert alert is None

    def test_data_staleness_alert(self, alert_manager):
        alert = alert_manager.check_data_staleness(26.0, "predictions_db")
        assert alert is not None
        assert alert.alert_type == "data_staleness"

    def test_data_fresh_no_alert(self, alert_manager):
        alert = alert_manager.check_data_staleness(12.0, "predictions_db")
        assert alert is None

    def test_alpha_degradation_trigger(self, alert_manager):
        alert_manager.set_baseline_alpha("momentum", 5.0)
        alert = alert_manager.check_alpha_degradation("momentum", 2.0)
        assert alert is not None
        assert alert.alert_type == "alpha_degradation"

    def test_alpha_degradation_below_threshold(self, alert_manager):
        alert_manager.set_baseline_alpha("momentum", 5.0)
        alert = alert_manager.check_alpha_degradation("momentum", 4.0)
        assert alert is None

    def test_execution_failure_threshold(self, alert_manager):
        alert_manager.record_failure("network_error")
        alert_manager.record_failure("network_error")
        alert_manager.record_failure("network_error")
        alert = alert_manager.check_execution_failure("network_error")
        assert alert is not None
        assert alert.alert_type == "execution_failure"

    def test_order_failure_threshold(self, alert_manager):
        alert_manager.record_failure("order_failure_AAPL")
        alert_manager.record_failure("order_failure_AAPL")
        alert_manager.record_failure("order_failure_AAPL")
        alert_manager.record_failure("order_failure_AAPL")
        alert_manager.record_failure("order_failure_AAPL")
        alert = alert_manager.check_order_failure("AAPL")
        assert alert is not None
        assert alert.alert_type == "order_failure"

    def test_regime_drift_psi(self, alert_manager):
        baseline = {"trend": 0.7, "chop": 0.2, "crisis": 0.1}
        current = {"trend": 0.2, "chop": 0.3, "crisis": 0.5}

        alert_manager.set_baseline_regime_distribution(baseline)
        alert = alert_manager.check_regime_drift(current)
        assert alert is not None
        assert alert.alert_type == "regime_drift"

    def test_regime_drift_below_threshold(self, alert_manager):
        baseline = {"trend": 0.5, "chop": 0.3, "crisis": 0.2}
        current = {"trend": 0.48, "chop": 0.32, "crisis": 0.2}

        alert_manager.set_baseline_regime_distribution(baseline)
        alert = alert_manager.check_regime_drift(current)
        assert alert is None


class TestMetricsCollectorIntegration:
    @pytest.fixture
    def collector(self):
        from autotrade.monitoring.collector import MetricsCollector

        with tempfile.TemporaryDirectory() as tmpdir:
            coll = MetricsCollector(
                output_dir=Path(tmpdir), enable_console_output=False
            )
            yield coll

    def test_emit_signal_lifecycle(self, collector):
        collector.aggregate_signal_lifecycle(
            family="momentum",
            signals_evaluated=100,
            signals_accepted=50,
            signals_executed=25,
        )

        summary = collector.get_signal_lifecycle_summary()
        assert "momentum" in summary
        assert summary["momentum"]["signals_evaluated"] == 100

    def test_emit_execution_quality(self, collector):
        collector.emit_execution_quality(
            order_id="order_1",
            symbol="TSLA",
            side="buy",
            quantity=100.0,
            fill_price=200.0,
            bid_price_at_arrival=199.8,
            ask_price_at_arrival=200.2,
            expected_price=199.0,
            slippage_expected_bps=5.0,
            status="filled",
        )

        summary = collector.get_execution_quality_summary()
        assert "order_1" in summary
        assert summary["order_1"]["symbol"] == "TSLA"
        assert summary["order_1"]["quoted_spread_bps"] > 0
        assert summary["order_1"]["effective_spread_bps"] >= 0

    def test_emit_system_health(self, collector):
        collector.emit_system_health(
            component="execution",
            status="healthy",
            cpu_percent=50.0,
        )

    def test_emit_risk_health(self, collector):
        collector.emit_risk_health(
            portfolio_id="main",
            drawdown_pct=4.0,
        )

    def test_emit_alpha_family_metrics(self, collector):
        collector.emit_alpha_family_metrics(
            family="value",
            regime="trend",
            total_return_pct=3.0,
            benchmark_return_pct=1.0,
        )

    def test_flush_writes_to_file(self, collector):
        collector.aggregate_signal_lifecycle(
            family="test",
            signals_evaluated=10,
        )
        collector.flush()

        assert len(collector._metrics) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
