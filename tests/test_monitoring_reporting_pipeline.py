import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autotrade.monitoring.contracts import (
    GateStatus,
    GateType,
    ReleaseGateStatus,
)
from autotrade.monitoring.collector import MetricsCollector
from autotrade.monitoring.alerts import AlertManager, Alert, AlertType, AlertLevel
from autotrade.monitoring.reporting import (
    ReportingEngine,
    DailyReport,
    WeeklyRollup,
    ReleaseGateReport,
    DashboardArtifact,
    get_reporting_engine,
    reset_reporting_engine,
)


class TestReportingEngineInitialization:
    def test_initialization_with_defaults(self):
        reset_reporting_engine()
        engine = get_reporting_engine()

        assert engine.output_dir == Path("logs/reports")
        assert engine.collector is None
        assert engine.alert_manager is None

    def test_initialization_with_custom_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = ReportingEngine(output_dir=Path(tmpdir))
            assert engine.output_dir == Path(tmpdir)


class TestDailyReportGeneration:
    @pytest.fixture
    def reporting_engine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = ReportingEngine(output_dir=Path(tmpdir))
            yield engine

    def test_generate_daily_report_default_date(self, reporting_engine):
        report = reporting_engine.generate_daily_report()

        assert report.date == datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert report.generated_at is not None

    def test_generate_daily_report_specific_date(self, reporting_engine):
        report = reporting_engine.generate_daily_report("2026-02-14")

        assert report.date == "2026-02-14"

    def test_daily_report_saved_to_file(self, reporting_engine):
        reporting_engine.generate_daily_report("2026-02-14")

        output_file = reporting_engine.output_dir / "daily_report_2026-02-14.json"
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)
            assert data["date"] == "2026-02-14"

    def test_daily_report_with_collector(self, reporting_engine):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(output_dir=Path(tmpdir))
            collector.aggregate_signal_lifecycle(
                family="momentum",
                signals_evaluated=100,
                signals_accepted=50,
                signals_executed=25,
            )

            engine = ReportingEngine(
                output_dir=reporting_engine.output_dir,
                collector=collector,
            )

            report = engine.generate_daily_report("2026-02-14")
            assert "signal_metrics" in report.__dict__


class TestReleaseGateReport:
    @pytest.fixture
    def reporting_engine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = ReportingEngine(output_dir=Path(tmpdir))
            yield engine

    def test_release_gate_report_all_pass(self, reporting_engine):
        with patch.object(reporting_engine, "_check_test_status", return_value=True):
            with patch.object(
                reporting_engine,
                "_check_backtest_status",
                return_value={"status": "pass", "regression_pct": -2.0},
            ):
                with patch.object(
                    reporting_engine,
                    "_check_paper_trading_status",
                    return_value={"status": "pass", "days": 15, "performance_pct": 3.0},
                ):
                    with patch.object(
                        reporting_engine,
                        "_check_canary_status",
                        return_value={"ready": True, "risk_pct": 5.0},
                    ):
                        report = reporting_engine.generate_release_gate_report()

        assert report.overall_status == GateStatus.PASS.value
        assert report.canary_ready is True

    def test_release_gate_report_test_fails(self, reporting_engine):
        with patch.object(reporting_engine, "_check_test_status", return_value=False):
            with patch.object(
                reporting_engine,
                "_check_backtest_status",
                return_value={"status": "pass", "regression_pct": -2.0},
            ):
                with patch.object(
                    reporting_engine,
                    "_check_paper_trading_status",
                    return_value={"status": "pass", "days": 15, "performance_pct": 3.0},
                ):
                    with patch.object(
                        reporting_engine,
                        "_check_canary_status",
                        return_value={"ready": True, "risk_pct": 5.0},
                    ):
                        report = reporting_engine.generate_release_gate_report()

        assert report.overall_status == GateStatus.FAIL.value

    def test_release_gate_report_pending_has_concrete_reason(self, reporting_engine):
        with patch.object(reporting_engine, "_check_test_status", return_value=True):
            with patch.object(
                reporting_engine,
                "_check_backtest_status",
                return_value={"status": "unknown", "regression_pct": 0.0},
            ):
                with patch.object(
                    reporting_engine,
                    "_check_paper_trading_status",
                    return_value={
                        "status": "unknown",
                        "days": 0,
                        "performance_pct": 0.0,
                    },
                ):
                    report = reporting_engine.generate_release_gate_report()

        assert report.overall_status == GateStatus.PENDING.value
        assert "Backtest gate pending" in report.fail_reason
        assert "Paper trading gate pending" in report.fail_reason
        assert "Unknown failure" not in report.fail_reason

    def test_release_gate_report_saved(self, reporting_engine):
        with patch.object(reporting_engine, "_check_test_status", return_value=True):
            with patch.object(
                reporting_engine,
                "_check_backtest_status",
                return_value={"status": "pass", "regression_pct": -2.0},
            ):
                with patch.object(
                    reporting_engine,
                    "_check_paper_trading_status",
                    return_value={"status": "pass", "days": 15, "performance_pct": 3.0},
                ):
                    with patch.object(
                        reporting_engine,
                        "_check_canary_status",
                        return_value={"ready": True, "risk_pct": 5.0},
                    ):
                        reporting_engine.save_release_gate_report()

        output_file = reporting_engine.output_dir / "release_gate_report.json"
        assert output_file.exists()


class TestWeeklyRollup:
    @pytest.fixture
    def reporting_engine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = ReportingEngine(output_dir=Path(tmpdir))
            yield engine

    @pytest.fixture
    def daily_reports(self, reporting_engine):
        for i in range(3):
            date = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
            report_data = {
                "date": date,
                "signal_metrics": {
                    "total_signals_evaluated": 100 * (i + 1),
                    "total_signals_accepted": 50 * (i + 1),
                    "total_signals_executed": 25 * (i + 1),
                },
                "execution_metrics": {
                    "total_orders": 10 * (i + 1),
                    "avg_slippage_bps": 5.0,
                },
                "risk_metrics": {
                    "drawdown_pct": 2.0 * i,
                },
                "alpha_metrics": {
                    "families": {"momentum": {"alpha_pct": 1.5}},
                },
                "alert_summary": {
                    "by_type": {"drawdown_breach": i},
                },
            }

            output_file = reporting_engine.output_dir / f"daily_report_{date}.json"
            with open(output_file, "w") as f:
                json.dump(report_data, f)

    def test_generate_weekly_rollup(self, reporting_engine, daily_reports):
        rollup = reporting_engine.generate_weekly_rollup()

        assert rollup.week_start is not None
        assert rollup.week_end is not None
        assert "signal_funnel" in rollup.__dict__
        assert "execution_quality" in rollup.__dict__

    def test_signal_funnel_aggregation(self, reporting_engine, daily_reports):
        rollup = reporting_engine.generate_weekly_rollup()

        assert rollup.signal_funnel["total_evaluated"] > 0

    def test_weekly_rollup_saved(self, reporting_engine, daily_reports):
        rollup = reporting_engine.generate_weekly_rollup()

        output_file = (
            reporting_engine.output_dir / f"weekly_rollup_{rollup.week_start}.json"
        )
        assert output_file.exists()


class TestDashboardArtifact:
    @pytest.fixture
    def reporting_engine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = ReportingEngine(output_dir=Path(tmpdir))
            yield engine

    def test_dashboard_artifact_generation(self, reporting_engine):
        artifact = reporting_engine.generate_dashboard_artifact()

        assert artifact.generated_at is not None
        assert "kpis" in artifact.__dict__
        assert "signal_funnel" in artifact.__dict__
        assert "execution_quality" in artifact.__dict__
        assert "alpha_contribution" in artifact.__dict__
        assert "risk_alarms" in artifact.__dict__

    def test_dashboard_artifact_with_collector(self, reporting_engine):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(output_dir=Path(tmpdir))
            collector.aggregate_signal_lifecycle(
                family="momentum",
                signals_evaluated=100,
                signals_accepted=50,
                signals_executed=25,
            )
            collector.emit_execution_quality(
                order_id="order_1",
                symbol="AAPL",
                side="buy",
                quantity=100,
                fill_price=150.0,
                expected_price=149.0,
                slippage_expected_bps=5.0,
                status="filled",
            )

            engine = ReportingEngine(
                output_dir=reporting_engine.output_dir,
                collector=collector,
            )

            artifact = engine.generate_dashboard_artifact()
            assert artifact.kpis["signal_funnel_total"] == 100

    def test_dashboard_kpis_extraction(self, reporting_engine):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(output_dir=Path(tmpdir))
            collector.aggregate_signal_lifecycle(
                family="momentum",
                signals_evaluated=100,
                signals_accepted=50,
                signals_executed=20,
            )

            engine = ReportingEngine(
                output_dir=reporting_engine.output_dir,
                collector=collector,
            )

            artifact = engine.generate_dashboard_artifact()

            assert "trade_conversion_rate" in artifact.kpis
            assert artifact.kpis["trade_conversion_rate"] == 0.2

    def test_dashboard_signal_funnel(self, reporting_engine):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(output_dir=Path(tmpdir))
            collector.aggregate_signal_lifecycle(
                family="momentum",
                signals_evaluated=100,
                signals_accepted=60,
                signals_executed=30,
            )
            collector.aggregate_signal_lifecycle(
                family="value",
                signals_evaluated=80,
                signals_accepted=40,
                signals_executed=20,
            )

            engine = ReportingEngine(
                output_dir=reporting_engine.output_dir,
                collector=collector,
            )

            artifact = engine.generate_dashboard_artifact()

            assert artifact.signal_funnel["evaluated"] == 180
            assert artifact.signal_funnel["accepted"] == 100
            assert artifact.signal_funnel["executed"] == 50

    def test_dashboard_execution_quality(self, reporting_engine):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(output_dir=Path(tmpdir))
            collector.emit_execution_quality(
                order_id="order_1",
                symbol="AAPL",
                side="buy",
                quantity=100,
                fill_price=150.0,
                expected_price=149.0,
                slippage_expected_bps=5.0,
                status="filled",
            )

            engine = ReportingEngine(
                output_dir=reporting_engine.output_dir,
                collector=collector,
            )

            artifact = engine.generate_dashboard_artifact()

            assert artifact.execution_quality["total_orders"] >= 1

    def test_dashboard_risk_alarms(self, reporting_engine):
        with tempfile.TemporaryDirectory() as tmpdir:
            alert_manager = AlertManager(output_dir=Path(tmpdir), enable_console=False)

            alert1 = Alert(
                alert_type="drawdown_breach",
                level="warning",
                message="Drawdown exceeded 5%",
            )
            alert2 = Alert(
                alert_type="data_staleness",
                level="error",
                message="Data is stale",
            )

            alert_manager.emit(alert1)
            alert_manager.emit(alert2)

            engine = ReportingEngine(
                output_dir=reporting_engine.output_dir,
                alert_manager=alert_manager,
            )

            artifact = engine.generate_dashboard_artifact()

            assert artifact.risk_alarms["drawdown_alarm"] is True
            assert artifact.risk_alarms["data_staleness"] is True
            assert artifact.risk_alarms["active_alerts"] == 2

    def test_dashboard_saved_to_file(self, reporting_engine):
        reporting_engine.generate_dashboard_artifact("2026-02-14")

        output_file = reporting_engine.output_dir / "dashboard_2026-02-14.json"
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)
            assert "kpis" in data
            assert "signal_funnel" in data


class TestReportingEngineHelperMethods:
    @pytest.fixture
    def reporting_engine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = ReportingEngine(output_dir=Path(tmpdir))
            yield engine

    def test_avg_calculation(self, reporting_engine):
        assert reporting_engine._avg([1.0, 2.0, 3.0]) == 2.0
        assert reporting_engine._avg([5.0]) == 5.0
        assert reporting_engine._avg([]) == 0.0

    def test_count_by_key(self, reporting_engine):
        data = {
            "a": {"status": "filled"},
            "b": {"status": "filled"},
            "c": {"status": "rejected"},
        }

        result = reporting_engine._count_by_key(data, "status")

        assert result["filled"] == 2
        assert result["rejected"] == 1


class TestIntegrationScenarios:
    def test_full_pipeline_with_mocked_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(output_dir=Path(tmpdir))
            alert_manager = AlertManager(output_dir=Path(tmpdir), enable_console=False)
            engine = ReportingEngine(
                output_dir=Path(tmpdir),
                collector=collector,
                alert_manager=alert_manager,
            )

            collector.aggregate_signal_lifecycle(
                family="momentum",
                signals_evaluated=200,
                signals_accepted=100,
                signals_executed=50,
            )

            collector.emit_execution_quality(
                order_id="order_1",
                symbol="NVDA",
                side="buy",
                quantity=50,
                fill_price=500.0,
                expected_price=498.0,
                slippage_expected_bps=10.0,
                status="filled",
            )

            alert = Alert(
                alert_type="drawdown_breach",
                level="warning",
                message="Test alert",
            )
            alert_manager.emit(alert)

            daily_report = engine.generate_daily_report("2026-02-14")
            dashboard = engine.generate_dashboard_artifact("2026-02-14")
            gate_report = engine.generate_release_gate_report()

            assert daily_report is not None
            assert dashboard is not None
            assert gate_report is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
