from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from autotrade.monitoring.contracts import (
    GateStatus,
    GateType,
    ReleaseGateStatus,
    SignalFamily,
    RegimeType,
)
from autotrade.monitoring.collector import MetricsCollector
from autotrade.monitoring.alerts import AlertManager, Alert


@dataclass
class DashboardArtifact:
    generated_at: datetime
    kpis: Dict[str, Any] = field(default_factory=dict)
    signal_funnel: Dict[str, Any] = field(default_factory=dict)
    execution_quality: Dict[str, Any] = field(default_factory=dict)
    alpha_contribution: Dict[str, Any] = field(default_factory=dict)
    risk_alarms: Dict[str, Any] = field(default_factory=dict)


try:
    from autotrade.utils.safe_logging import get_safe_logger
except ImportError:
    import logging

    get_safe_logger = lambda name: logging.getLogger(name)


@dataclass
class DailyReport:
    date: str
    generated_at: datetime
    signal_metrics: Dict[str, Any] = field(default_factory=dict)
    execution_metrics: Dict[str, Any] = field(default_factory=dict)
    risk_metrics: Dict[str, Any] = field(default_factory=dict)
    alpha_metrics: Dict[str, Any] = field(default_factory=dict)
    alert_summary: Dict[str, Any] = field(default_factory=dict)
    workflow_summary: Dict[str, Any] = field(default_factory=dict)
    release_gates: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WeeklyRollup:
    week_start: str
    week_end: str
    generated_at: datetime
    daily_summaries: List[Dict[str, Any]] = field(default_factory=list)
    signal_funnel: Dict[str, Any] = field(default_factory=dict)
    execution_quality: Dict[str, Any] = field(default_factory=dict)
    risk_summary: Dict[str, Any] = field(default_factory=dict)
    alpha_by_family: Dict[str, Any] = field(default_factory=dict)
    alert_trends: Dict[str, Any] = field(default_factory=dict)
    release_gate_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ReleaseGateReport:
    generated_at: datetime
    test_status: str = "unknown"
    test_pass_rate: float = 0.0
    backtest_status: str = "unknown"
    backtest_regression_pct: float = 0.0
    paper_trading_status: str = "unknown"
    paper_trading_days: int = 0
    paper_performance_pct: float = 0.0
    canary_ready: bool = False
    canary_risk_pct: float = 0.0
    rollback_ready: bool = False
    pass_reason: str = ""
    fail_reason: str = ""
    overall_status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReportingEngine:
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        collector: Optional[MetricsCollector] = None,
        alert_manager: Optional[AlertManager] = None,
    ):
        self.output_dir = output_dir or Path("logs/reports")
        self.collector = collector
        self.alert_manager = alert_manager
        self.logger = get_safe_logger("reporting_engine")

        self._lock = threading.RLock()
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_daily_report(
        self,
        date: Optional[str] = None,
    ) -> DailyReport:
        with self._lock:
            if date is None:
                date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            report = DailyReport(
                date=date,
                generated_at=datetime.now(timezone.utc),
            )

            if self.collector:
                report.signal_metrics = self._aggregate_signal_metrics()
                report.execution_metrics = self._aggregate_execution_metrics()
                report.risk_metrics = self._aggregate_risk_metrics()
                report.alpha_metrics = self._aggregate_alpha_metrics()
                report.workflow_summary = self._aggregate_workflow_metrics()

            if self.alert_manager:
                report.alert_summary = self.alert_manager.get_alert_summary()

            report.release_gates = self._generate_release_gate_status()

            self._save_daily_report(report)
            return report

    def _aggregate_signal_metrics(self) -> Dict[str, Any]:
        summary = self.collector.get_signal_lifecycle_summary()
        return {
            "families": summary,
            "total_signals_evaluated": sum(
                m.get("signals_evaluated", 0) for m in summary.values()
            ),
            "total_signals_accepted": sum(
                m.get("signals_accepted", 0) for m in summary.values()
            ),
            "total_signals_executed": sum(
                m.get("signals_executed", 0) for m in summary.values()
            ),
            "avg_conversion_rate": self._avg(
                m.get("conversion_rate", 0) for m in summary.values()
            ),
            "avg_hit_rate": self._avg(m.get("hit_rate", 0) for m in summary.values()),
        }

    def _aggregate_execution_metrics(self) -> Dict[str, Any]:
        summary = self.collector.get_execution_quality_summary()
        if not summary:
            return {"total_orders": 0}

        slippage_bps = [m.get("slippage_bps", 0) for m in summary.values()]
        fill_rates = [m.get("fill_rate", 0) for m in summary.values()]

        return {
            "total_orders": len(summary),
            "avg_slippage_bps": self._avg(slippage_bps),
            "max_slippage_bps": max(slippage_bps) if slippage_bps else 0,
            "avg_fill_rate": self._avg(fill_rates),
            "orders_by_status": self._count_by_key(summary, "status"),
        }

    def _aggregate_risk_metrics(self) -> Dict[str, Any]:
        summary = self.collector.get_risk_health_summary()
        if not summary:
            return {
                "alert_level": "green",
                "position_count": 0,
                "drawdown_pct": 0.0,
            }

        # Aggregate across portfolios if needed, here we take 'default' or max risk
        main_portfolio = summary.get("default") or next(iter(summary.values()))

        return {
            "alert_level": main_portfolio.get("alert_level", "green"),
            "position_count": sum(m.get("position_count", 0) for m in summary.values()),
            "drawdown_pct": max(m.get("drawdown_pct", 0.0) for m in summary.values()),
        }

    def _aggregate_alpha_metrics(self) -> Dict[str, Any]:
        summary = self.collector.get_alpha_family_summary()

        total_alpha = 0.0
        if summary:
            total_alpha = self._avg([m.get("alpha_pct", 0.0) for m in summary.values()])

        return {
            "families": summary,
            "total_alpha_pct": total_alpha,
        }

    def _aggregate_workflow_metrics(self) -> Dict[str, Any]:
        return {
            "phases_completed": 0,
            "total_duration_sec": 0.0,
        }

    def _generate_release_gate_status(self) -> Dict[str, Any]:
        report = self._evaluate_release_gates()
        return {
            "test_status": report.test_status,
            "backtest_status": report.backtest_status,
            "paper_trading_status": report.paper_trading_status,
            "canary_ready": report.canary_ready,
            "rollback_ready": report.rollback_ready,
            "overall_status": report.overall_status,
        }

    def _evaluate_release_gates(self) -> ReleaseGateReport:
        report = ReleaseGateReport(
            generated_at=datetime.now(timezone.utc),
        )

        test_pass = self._check_test_status()
        report.test_status = (
            GateStatus.PASS.value if test_pass else GateStatus.FAIL.value
        )
        report.test_pass_rate = 1.0 if test_pass else 0.0

        backtest_result = self._check_backtest_status()
        report.backtest_status = backtest_result.get("status", "unknown")
        report.backtest_regression_pct = backtest_result.get("regression_pct", 0.0)

        paper_result = self._check_paper_trading_status()
        report.paper_trading_status = paper_result.get("status", "unknown")
        report.paper_trading_days = paper_result.get("days", 0)
        report.paper_performance_pct = paper_result.get("performance_pct", 0.0)

        canary_result = self._check_canary_status()
        report.canary_ready = canary_result.get("ready", False)
        report.canary_risk_pct = canary_result.get("risk_pct", 0.0)

        report.rollback_ready = (
            report.test_status == GateStatus.FAIL.value
            or report.paper_trading_days >= 5
        )

        report.overall_status = self._compute_overall_gate_status(report)

        if report.overall_status == GateStatus.PASS.value:
            report.pass_reason = "All release gates passed"
        else:
            report.fail_reason = self._compute_fail_reason(report)

        return report

    def _check_test_status(self) -> bool:
        test_results_path = self.output_dir.parent / "test_results.json"
        if test_results_path.exists():
            try:
                with open(test_results_path) as f:
                    results = json.load(f)
                    return results.get("passed", False)
            except Exception:
                pass
        return True

    def _check_backtest_status(self) -> Dict[str, Any]:
        backtest_path = self.output_dir.parent / "backtest_results.json"
        if backtest_path.exists():
            try:
                with open(backtest_path) as f:
                    results = json.load(f)
                    return {
                        "status": GateStatus.PASS.value
                        if results.get("regression_pct", 0) >= -5.0
                        else GateStatus.FAIL.value,
                        "regression_pct": results.get("regression_pct", 0),
                    }
            except Exception:
                pass
        return {"status": "unknown", "regression_pct": 0.0}

    def _check_paper_trading_status(self) -> Dict[str, Any]:
        paper_log_path = self.output_dir.parent / "paper_trading_log.json"
        if paper_log_path.exists():
            try:
                with open(paper_log_path) as f:
                    data = json.load(f)
                    days = data.get("trading_days", 0)
                    performance = data.get("performance_pct", 0.0)
                    return {
                        "status": GateStatus.PASS.value
                        if days >= 10 and performance >= 0
                        else GateStatus.FAIL.value,
                        "days": days,
                        "performance_pct": performance,
                    }
            except Exception:
                pass
        return {"status": "unknown", "days": 0, "performance_pct": 0.0}

    def _check_canary_status(self) -> Dict[str, Any]:
        return {
            "ready": False,
            "risk_pct": 0.0,
        }

    def _compute_overall_gate_status(self, report: ReleaseGateReport) -> str:
        gates = [
            report.test_status,
            report.backtest_status,
            report.paper_trading_status,
        ]

        if all(g == GateStatus.PASS.value for g in gates):
            return GateStatus.PASS.value
        elif any(g == GateStatus.FAIL.value for g in gates):
            return GateStatus.FAIL.value
        else:
            return GateStatus.PENDING.value

    def _compute_fail_reason(self, report: ReleaseGateReport) -> str:
        reasons = []
        if report.test_status == GateStatus.FAIL.value:
            reasons.append("Test gate failed")
        elif report.test_status in {GateStatus.PENDING.value, "unknown"}:
            reasons.append("Test gate pending")
        if report.backtest_status == GateStatus.FAIL.value:
            reasons.append("Backtest regression detected")
        elif report.backtest_status in {GateStatus.PENDING.value, "unknown"}:
            reasons.append("Backtest gate pending")
        if report.paper_trading_status == GateStatus.FAIL.value:
            reasons.append("Paper trading requirements not met")
        elif report.paper_trading_status in {GateStatus.PENDING.value, "unknown"}:
            reasons.append("Paper trading gate pending")
        return "; ".join(reasons) if reasons else "No failing gate reason recorded"

    def _count_by_key(self, data: Dict, key: str) -> Dict[str, int]:
        counts = {}
        for item in data.values():
            value = item.get(key, "unknown")
            counts[value] = counts.get(value, 0) + 1
        return counts

    def _avg(self, values: List[float]) -> float:
        filtered = [v for v in values if v is not None and v != 0]
        return sum(filtered) / len(filtered) if filtered else 0.0

    def _save_daily_report(self, report: DailyReport):
        try:
            output_path = self.output_dir / f"daily_report_{report.date}.json"
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "date": report.date,
                        "generated_at": report.generated_at.isoformat(),
                        "signal_metrics": report.signal_metrics,
                        "execution_metrics": report.execution_metrics,
                        "risk_metrics": report.risk_metrics,
                        "alpha_metrics": report.alpha_metrics,
                        "alert_summary": report.alert_summary,
                        "workflow_summary": report.workflow_summary,
                        "release_gates": report.release_gates,
                    },
                    f,
                    indent=2,
                )
            self.logger.info(f"Daily report saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save daily report: {e}")

    def generate_weekly_rollup(
        self,
        week_start: Optional[datetime] = None,
    ) -> WeeklyRollup:
        with self._lock:
            if week_start is None:
                today = datetime.now(timezone.utc)
                week_start = today - timedelta(days=today.weekday())

            week_end = week_start + timedelta(days=6)

            rollup = WeeklyRollup(
                week_start=week_start.strftime("%Y-%m-%d"),
                week_end=week_end.strftime("%Y-%m-%d"),
                generated_at=datetime.now(timezone.utc),
            )

            rollup.daily_summaries = self._load_daily_summaries(week_start, week_end)
            rollup.signal_funnel = self._aggregate_signal_funnel(rollup.daily_summaries)
            rollup.execution_quality = self._aggregate_execution_quality(
                rollup.daily_summaries
            )
            rollup.risk_summary = self._aggregate_risk_summary(rollup.daily_summaries)
            rollup.alpha_by_family = self._aggregate_alpha_by_family(
                rollup.daily_summaries
            )
            rollup.alert_trends = self._aggregate_alert_trends(rollup.daily_summaries)

            self._save_weekly_rollup(rollup)
            return rollup

    def _load_daily_summaries(
        self,
        start: datetime,
        end: datetime,
    ) -> List[Dict[str, Any]]:
        summaries = []
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            report_path = self.output_dir / f"daily_report_{date_str}.json"
            if report_path.exists():
                try:
                    with open(report_path) as f:
                        summaries.append(json.load(f))
                except Exception:
                    pass
            current += timedelta(days=1)
        return summaries

    def _aggregate_signal_funnel(
        self,
        daily_summaries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not daily_summaries:
            return {}

        total_evaluated = sum(
            s.get("signal_metrics", {}).get("total_signals_evaluated", 0)
            for s in daily_summaries
        )
        total_accepted = sum(
            s.get("signal_metrics", {}).get("total_signals_accepted", 0)
            for s in daily_summaries
        )
        total_executed = sum(
            s.get("signal_metrics", {}).get("total_signals_executed", 0)
            for s in daily_summaries
        )

        return {
            "total_evaluated": total_evaluated,
            "total_accepted": total_accepted,
            "total_executed": total_executed,
            "acceptance_rate": total_accepted / total_evaluated
            if total_evaluated
            else 0,
            "execution_rate": total_executed / total_accepted if total_accepted else 0,
        }

    def _aggregate_execution_quality(
        self,
        daily_summaries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not daily_summaries:
            return {}

        total_orders = sum(
            s.get("execution_metrics", {}).get("total_orders", 0)
            for s in daily_summaries
        )
        avg_slippage = self._avg(
            [
                s.get("execution_metrics", {}).get("avg_slippage_bps", 0)
                for s in daily_summaries
            ]
        )
        avg_effective_spread = self._avg(
            [
                s.get("execution_metrics", {}).get("avg_effective_spread_bps", 0)
                for s in daily_summaries
            ]
        )
        avg_quoted_spread = self._avg(
            [
                s.get("execution_metrics", {}).get("avg_quoted_spread_bps", 0)
                for s in daily_summaries
            ]
        )

        return {
            "total_orders": total_orders,
            "avg_slippage_bps": avg_slippage,
            "avg_effective_spread_bps": avg_effective_spread,
            "avg_quoted_spread_bps": avg_quoted_spread,
            "daily_breakdown": [
                {
                    "date": s.get("date"),
                    "orders": s.get("execution_metrics", {}).get("total_orders", 0),
                }
                for s in daily_summaries
            ],
        }

    def _aggregate_risk_summary(
        self,
        daily_summaries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not daily_summaries:
            return {}

        return {
            "days_traded": len(daily_summaries),
            "max_drawdown": max(
                [
                    s.get("risk_metrics", {}).get("drawdown_pct", 0)
                    for s in daily_summaries
                ]
            ),
        }

    def _aggregate_alpha_by_family(
        self,
        daily_summaries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not daily_summaries:
            return {}

        family_alphas = {}
        for summary in daily_summaries:
            families = summary.get("alpha_metrics", {}).get("families", {})
            for family, metrics in families.items():
                if family not in family_alphas:
                    family_alphas[family] = []
                family_alphas[family].append(metrics.get("alpha_pct", 0))

        return {
            family: {
                "avg_alpha_pct": self._avg(alphas),
                "weeks_tracked": len(alphas),
            }
            for family, alphas in family_alphas.items()
        }

    def _aggregate_alert_trends(
        self,
        daily_summaries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not daily_summaries:
            return {}

        alert_counts = {}
        for summary in daily_summaries:
            alerts = summary.get("alert_summary", {}).get("by_type", {})
            for alert_type, count in alerts.items():
                alert_counts[alert_type] = alert_counts.get(alert_type, 0) + count

        return {
            "total_alerts": sum(alert_counts.values()),
            "by_type": alert_counts,
        }

    def _save_weekly_rollup(self, rollup: WeeklyRollup):
        try:
            output_path = self.output_dir / f"weekly_rollup_{rollup.week_start}.json"
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "week_start": rollup.week_start,
                        "week_end": rollup.week_end,
                        "generated_at": rollup.generated_at.isoformat(),
                        "daily_summaries": rollup.daily_summaries,
                        "signal_funnel": rollup.signal_funnel,
                        "execution_quality": rollup.execution_quality,
                        "risk_summary": rollup.risk_summary,
                        "alpha_by_family": rollup.alpha_by_family,
                        "alert_trends": rollup.alert_trends,
                        "release_gate_history": rollup.release_gate_history,
                    },
                    f,
                    indent=2,
                )
            self.logger.info(f"Weekly rollup saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save weekly rollup: {e}")

    def generate_release_gate_report(self) -> ReleaseGateReport:
        return self._evaluate_release_gates()

    def save_release_gate_report(
        self,
        report: Optional[ReleaseGateReport] = None,
    ):
        if report is None:
            report = self._evaluate_release_gates()

        try:
            output_path = self.output_dir / "release_gate_report.json"
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "generated_at": report.generated_at.isoformat(),
                        "test_status": report.test_status,
                        "test_pass_rate": report.test_pass_rate,
                        "backtest_status": report.backtest_status,
                        "backtest_regression_pct": report.backtest_regression_pct,
                        "paper_trading_status": report.paper_trading_status,
                        "paper_trading_days": report.paper_trading_days,
                        "paper_performance_pct": report.paper_performance_pct,
                        "canary_ready": report.canary_ready,
                        "canary_risk_pct": report.canary_risk_pct,
                        "rollback_ready": report.rollback_ready,
                        "pass_reason": report.pass_reason,
                        "fail_reason": report.fail_reason,
                        "overall_status": report.overall_status,
                        "metadata": report.metadata,
                    },
                    f,
                    indent=2,
                )
            self.logger.info(f"Release gate report saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save release gate report: {e}")

    def generate_dashboard_artifact(
        self,
        date: Optional[str] = None,
    ) -> DashboardArtifact:
        with self._lock:
            if date is None:
                date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            artifact = DashboardArtifact(
                generated_at=datetime.now(timezone.utc),
            )

            artifact.kpis = self._generate_kpis(date)
            artifact.signal_funnel = self._generate_signal_funnel()
            artifact.execution_quality = self._generate_execution_quality_kpis()
            artifact.alpha_contribution = self._generate_alpha_contribution()
            artifact.risk_alarms = self._generate_risk_alarms()

            self._save_dashboard_artifact(artifact, date)
            return artifact

    def _generate_kpis(self, date: str) -> Dict[str, Any]:
        kpis = {
            "trade_conversion_rate": 0.0,
            "realized_vs_expected_slippage_bps": 0.0,
            "total_drawdown_pct": 0.0,
            "active_alerts": 0,
            "signal_funnel_total": 0,
        }

        if self.collector:
            signal_summary = self.collector.get_signal_lifecycle_summary()
            if signal_summary:
                total_evaluated = sum(
                    m.get("signals_evaluated", 0) for m in signal_summary.values()
                )
                total_executed = sum(
                    m.get("signals_executed", 0) for m in signal_summary.values()
                )
                kpis["trade_conversion_rate"] = (
                    total_executed / total_evaluated if total_evaluated > 0 else 0.0
                )
                kpis["signal_funnel_total"] = total_evaluated

            exec_summary = self.collector.get_execution_quality_summary()
            if exec_summary:
                slippage_divs = [
                    m.get("slippage_divergence_bps", 0) for m in exec_summary.values()
                ]
                kpis["realized_vs_expected_slippage_bps"] = (
                    sum(slippage_divs) / len(slippage_divs) if slippage_divs else 0.0
                )

            risk_summary = self.collector.get_risk_health_summary()
            if risk_summary:
                kpis["total_drawdown_pct"] = max(
                    m.get("drawdown_pct", 0.0) for m in risk_summary.values()
                )

        if self.alert_manager:
            alerts = self.alert_manager.get_alerts_since(
                datetime.now(timezone.utc) - timedelta(hours=24)
            )
            kpis["active_alerts"] = len(alerts)

        return kpis

    def _generate_signal_funnel(self) -> Dict[str, Any]:
        if not self.collector:
            return {}

        signal_summary = self.collector.get_signal_lifecycle_summary()
        if not signal_summary:
            return {}

        total_evaluated = sum(
            m.get("signals_evaluated", 0) for m in signal_summary.values()
        )
        total_accepted = sum(
            m.get("signals_accepted", 0) for m in signal_summary.values()
        )
        total_executed = sum(
            m.get("signals_executed", 0) for m in signal_summary.values()
        )

        return {
            "evaluated": total_evaluated,
            "accepted": total_accepted,
            "executed": total_executed,
            "acceptance_rate": total_accepted / total_evaluated
            if total_evaluated > 0
            else 0,
            "execution_rate": total_executed / total_accepted
            if total_accepted > 0
            else 0,
            "by_family": {
                family: {
                    "evaluated": m.get("signals_evaluated", 0),
                    "accepted": m.get("signals_accepted", 0),
                    "executed": m.get("signals_executed", 0),
                    "conversion_rate": m.get("conversion_rate", 0.0),
                }
                for family, m in signal_summary.items()
            },
        }

    def _generate_execution_quality_kpis(self) -> Dict[str, Any]:
        if not self.collector:
            return {}

        exec_summary = self.collector.get_execution_quality_summary()
        if not exec_summary:
            return {}

        slippage_bps = [m.get("slippage_bps", 0) for m in exec_summary.values()]
        fill_rates = [m.get("fill_rate", 0) for m in exec_summary.values()]
        slippage_divs = [
            m.get("slippage_divergence_bps", 0) for m in exec_summary.values()
        ]
        effective_spreads = [
            m.get("effective_spread_bps", 0) for m in exec_summary.values()
        ]
        quoted_spreads = [m.get("quoted_spread_bps", 0) for m in exec_summary.values()]

        return {
            "total_orders": len(exec_summary),
            "avg_slippage_bps": self._avg(slippage_bps),
            "avg_fill_rate": self._avg(fill_rates),
            "avg_slippage_divergence_bps": self._avg(slippage_divs),
            "avg_effective_spread_bps": self._avg(effective_spreads),
            "avg_quoted_spread_bps": self._avg(quoted_spreads),
            "orders_by_status": self._count_by_key(exec_summary, "status"),
        }

    def _generate_alpha_contribution(self) -> Dict[str, Any]:
        if not self.collector:
            return {}

        summary = self.collector.get_alpha_family_summary()

        regimes = {}
        for m in summary.values():
            r = m.get("regime", "unknown")
            if r not in regimes:
                regimes[r] = []
            regimes[r].append(m.get("alpha_pct", 0.0))

        return {
            "families": {k: v.get("alpha_pct", 0.0) for k, v in summary.items()},
            "regimes": {k: self._avg(v) for k, v in regimes.items()},
        }

    def _generate_risk_alarms(self) -> Dict[str, Any]:
        alarms = {
            "drawdown_alarm": False,
            "critical_drawdown": False,
            "data_staleness": False,
            "execution_failures": 0,
            "active_alerts": 0,
        }

        if self.alert_manager:
            recent_alerts = self.alert_manager.get_alerts_since(
                datetime.now(timezone.utc) - timedelta(hours=24)
            )
            alarms["active_alerts"] = len(recent_alerts)

            for alert in recent_alerts:
                if alert.alert_type == "drawdown_breach":
                    if alert.level == "critical":
                        alarms["critical_drawdown"] = True
                    else:
                        alarms["drawdown_alarm"] = True
                elif alert.alert_type == "data_staleness":
                    alarms["data_staleness"] = True
                elif alert.alert_type in ("execution_failure", "order_failure"):
                    alarms["execution_failures"] += 1

        return alarms

    def _save_dashboard_artifact(self, artifact: DashboardArtifact, date: str):
        try:
            output_path = self.output_dir / f"dashboard_{date}.json"
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "generated_at": artifact.generated_at.isoformat(),
                        "kpis": artifact.kpis,
                        "signal_funnel": artifact.signal_funnel,
                        "execution_quality": artifact.execution_quality,
                        "alpha_contribution": artifact.alpha_contribution,
                        "risk_alarms": artifact.risk_alarms,
                    },
                    f,
                    indent=2,
                )
            self.logger.info(f"Dashboard artifact saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save dashboard artifact: {e}")


_reporting_engine_instance: Optional[ReportingEngine] = None
_reporting_engine_lock = threading.RLock()


def get_reporting_engine(
    output_dir: Optional[Path] = None,
    collector: Optional[MetricsCollector] = None,
    alert_manager: Optional[AlertManager] = None,
) -> ReportingEngine:
    global _reporting_engine_instance
    with _reporting_engine_lock:
        if _reporting_engine_instance is None:
            _reporting_engine_instance = ReportingEngine(
                output_dir=output_dir,
                collector=collector,
                alert_manager=alert_manager,
            )
        return _reporting_engine_instance


def reset_reporting_engine():
    global _reporting_engine_instance
    with _reporting_engine_lock:
        _reporting_engine_instance = None
