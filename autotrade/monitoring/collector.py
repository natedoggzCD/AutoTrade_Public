from __future__ import annotations

import json
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from autotrade.monitoring.contracts import (
    AlphaFamilyMetrics,
    ExecutionQualityMetrics,
    RiskHealthMetrics,
    SignalFamily,
    SignalLifecycleMetrics,
    SystemHealthEvent,
    CANONICAL_METRIC_NAMES,
)

try:
    from autotrade.utils.safe_logging import get_safe_logger
except ImportError:
    import logging

    get_safe_logger = lambda name: logging.getLogger(name)


@dataclass
class MetricEvent:
    metric_name: str
    timestamp: datetime
    tags: Dict[str, Any]
    values: Dict[str, Any]
    source: str


class MetricsCollector:
    """
    Centralized metrics collector that aggregates telemetry from all producers.

    Preserves existing logging while emitting normalized metric events for
    monitoring, alerting, and reporting.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        flush_interval_sec: int = 60,
        enable_console_output: bool = True,
    ):
        self.output_dir = output_dir or Path("logs/monitoring")
        self.flush_interval_sec = flush_interval_sec
        self.enable_console_output = enable_console_output

        self._metrics: List[MetricEvent] = []
        self._lock = threading.RLock()
        self._callbacks: List[Callable[[MetricEvent], None]] = []

        self.logger = get_safe_logger("metrics_collector")

        self._signal_lifecycle_cache: Dict[str, SignalLifecycleMetrics] = {}
        self._execution_quality_cache: Dict[str, ExecutionQualityMetrics] = {}
        self._risk_health_cache: Dict[str, RiskHealthMetrics] = {}
        self._alpha_family_cache: Dict[str, AlphaFamilyMetrics] = {}

        self._ensure_output_dir()

    def _ensure_output_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def register_callback(self, callback: Callable[[MetricEvent], None]):
        with self._lock:
            self._callbacks.append(callback)

    def emit(
        self,
        metric_name: str,
        tags: Optional[Dict[str, Any]] = None,
        values: Optional[Dict[str, Any]] = None,
        source: str = "unknown",
    ):
        event = MetricEvent(
            metric_name=metric_name,
            timestamp=datetime.now(timezone.utc),
            tags=tags or {},
            values=values or {},
            source=source,
        )

        with self._lock:
            self._metrics.append(event)
            for callback in self._callbacks:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.warning(f"Callback error: {e}")

        if self.enable_console_output:
            self._log_event(event)

    def _log_event(self, event: MetricEvent):
        tags_str = ", ".join(f"{k}={v}" for k, v in event.tags.items())
        values_str = ", ".join(f"{k}={v}" for k, v in event.values.items())
        self.logger.debug(f"[{event.metric_name}] {tags_str} | {values_str}")

    def flush(self):
        with self._lock:
            if not self._metrics:
                return

            metrics_to_write = self._metrics.copy()
            self._metrics.clear()

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        output_path = self.output_dir / f"metrics_{date_str}.jsonl"

        with open(output_path, "a") as f:
            for event in metrics_to_write:
                record = {
                    "metric_name": event.metric_name,
                    "timestamp": event.timestamp.isoformat(),
                    "tags": event.tags,
                    "values": event.values,
                    "source": event.source,
                }
                f.write(json.dumps(record) + "\n")

        self.logger.info(f"Flushed {len(metrics_to_write)} metrics to {output_path}")

    def get_metrics_since(
        self,
        since: datetime,
        metric_name: Optional[str] = None,
    ) -> List[MetricEvent]:
        with self._lock:
            filtered = [m for m in self._metrics if m.timestamp >= since]
            if metric_name:
                filtered = [m for m in filtered if m.metric_name == metric_name]
            return filtered

    def aggregate_signal_lifecycle(
        self,
        family: str,
        signals_evaluated: int = 0,
        signals_accepted: int = 0,
        signals_rejected: int = 0,
        signals_executed: int = 0,
        skip_reasons: Optional[Dict[str, int]] = None,
        hit_rate: float = 0.0,
        expectancy: float = 0.0,
        avg_duration_hours: float = 0.0,
        turnover_rate: float = 0.0,
        cost_impact_bps: float = 0.0,
    ):
        metrics = SignalLifecycleMetrics(
            family=family,
            signals_evaluated=signals_evaluated,
            signals_accepted=signals_accepted,
            signals_rejected=signals_rejected,
            signals_executed=signals_executed,
            skip_reasons=skip_reasons or {},
            hit_rate=hit_rate,
            expectancy=expectancy,
            avg_duration_hours=avg_duration_hours,
            turnover_rate=turnover_rate,
            cost_impact_bps=cost_impact_bps,
        )
        metrics.compute_derived_metrics()

        self._signal_lifecycle_cache[family] = metrics

        self.emit(
            metric_name=CANONICAL_METRIC_NAMES.get("signal_count", "signal.lifecycle"),
            tags={"family": family},
            values={
                "signals_evaluated": metrics.signals_evaluated,
                "signals_accepted": metrics.signals_accepted,
                "signals_rejected": metrics.signals_rejected,
                "signals_executed": metrics.signals_executed,
                "conversion_rate": metrics.conversion_rate,
                "hit_rate": metrics.hit_rate,
                "expectancy": metrics.expectancy,
                "turnover_rate": metrics.turnover_rate,
                "cost_impact_bps": metrics.cost_impact_bps,
            },
            source="signal_lifecycle",
        )

        if skip_reasons:
            for reason, count in skip_reasons.items():
                self.emit(
                    metric_name="signal.skip_reason",
                    tags={"family": family, "reason": reason},
                    values={"count": count},
                    source="signal_lifecycle",
                )

        # day-manager 2026-05-19: surface zero-execution-rate sessions as
        # CRITICAL log lines + flag file when the gate composition
        # collapses (see prompts/dev/2026-05-19_session_observations.md
        # item #11). Never raises into the caller's loop.
        try:
            from autotrade.core.signal_throughput_monitor import (
                get_signal_throughput_monitor,
            )

            get_signal_throughput_monitor().observe(
                family=family,
                signals_accepted=metrics.signals_accepted,
                signals_executed=metrics.signals_executed,
                skip_reasons=skip_reasons or {},
            )
        except Exception:
            # Monitor must never break the caller's signal-aggregation flow.
            pass

    def emit_execution_quality(
        self,
        order_id: str,
        symbol: str,
        side: str,
        order_type: str = "limit",
        quantity: float = 0.0,
        limit_price: Optional[float] = None,
        fill_price: Optional[float] = None,
        expected_price: Optional[float] = None,
        arrival_price: Optional[float] = None,
        bid_price_at_arrival: Optional[float] = None,
        ask_price_at_arrival: Optional[float] = None,
        mid_price_at_arrival: Optional[float] = None,
        slippage_expected_bps: float = 0.0,
        execution_latency_ms: float = 0.0,
        fill_quantity: float = 0.0,
        rejection_reason: str = "",
        status: str = "pending",
    ):
        metrics = ExecutionQualityMetrics(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            fill_price=fill_price,
            expected_price=expected_price,
            arrival_price=arrival_price,
            bid_price_at_arrival=bid_price_at_arrival,
            ask_price_at_arrival=ask_price_at_arrival,
            mid_price_at_arrival=mid_price_at_arrival,
            slippage_expected_bps=slippage_expected_bps,
            execution_latency_ms=execution_latency_ms,
            fill_quantity=fill_quantity,
            rejection_reason=rejection_reason,
            status=status,
        )
        metrics.compute_slippage_metrics()

        self._execution_quality_cache[order_id] = metrics

        self.emit(
            metric_name=CANONICAL_METRIC_NAMES.get("slippage_bps", "execution.quality"),
            tags={"symbol": symbol, "side": side, "status": status},
            values={
                "order_id": order_id,
                "quantity": quantity,
                "limit_price": limit_price,
                "fill_price": fill_price,
                "expected_price": expected_price,
                "arrival_price": arrival_price,
                "bid_price_at_arrival": bid_price_at_arrival,
                "ask_price_at_arrival": ask_price_at_arrival,
                "mid_price_at_arrival": mid_price_at_arrival,
                "slippage_bps": metrics.slippage_bps,
                "slippage_divergence_bps": metrics.slippage_divergence_bps,
                "implementation_shortfall_bps": metrics.implementation_shortfall_bps,
                "effective_spread_bps": metrics.effective_spread_bps,
                "quoted_spread_bps": metrics.quoted_spread_bps,
                "opportunity_cost_bps": metrics.opportunity_cost_bps,
                "fill_rate": metrics.fill_rate,
                "latency_ms": execution_latency_ms,
            },
            source="execution_quality",
        )

    def emit_system_health(
        self,
        component: str,
        status: str = "healthy",
        cpu_percent: float = 0.0,
        memory_percent: float = 0.0,
        disk_percent: float = 0.0,
        error_count: int = 0,
        warning_count: int = 0,
        message: str = "",
    ):
        event = SystemHealthEvent(
            component=component,
            status=status,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_percent=disk_percent,
            error_count=error_count,
            warning_count=warning_count,
            message=message,
        )

        self.emit(
            metric_name=CANONICAL_METRIC_NAMES.get(
                "signal_count", "system.health"
            ).replace("signal.count", "system.health"),
            tags={"component": component, "status": status},
            values={
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "error_count": error_count,
                "warning_count": warning_count,
                "message": message,
            },
            source="system_health",
        )

    def emit_risk_health(
        self,
        portfolio_id: str = "default",
        total_exposure_pct: float = 0.0,
        cash_pct: float = 0.0,
        leverage: float = 1.0,
        var_95_pct: float = 0.0,
        drawdown_pct: float = 0.0,
        drawdown_peak_pct: float = 0.0,
        daily_pnl_pct: float = 0.0,
        unrealized_pnl_pct: float = 0.0,
        realized_pnl_pct: float = 0.0,
        position_count: int = 0,
        max_position_size_pct: float = 0.0,
        sector_exposures: Optional[Dict[str, float]] = None,
        beta_exposure: float = 0.0,
        pdt_violations_24h: int = 0,
        margin_calls: int = 0,
    ):
        metrics = RiskHealthMetrics(
            portfolio_id=portfolio_id,
            total_exposure_pct=total_exposure_pct,
            cash_pct=cash_pct,
            leverage=leverage,
            var_95_pct=var_95_pct,
            drawdown_pct=drawdown_pct,
            drawdown_peak_pct=drawdown_peak_pct,
            daily_pnl_pct=daily_pnl_pct,
            unrealized_pnl_pct=unrealized_pnl_pct,
            realized_pnl_pct=realized_pnl_pct,
            position_count=position_count,
            max_position_size_pct=max_position_size_pct,
            sector_exposures=sector_exposures or {},
            beta_exposure=beta_exposure,
            pdt_violations_24h=pdt_violations_24h,
            margin_calls=margin_calls,
        )
        metrics.compute_risk_state()

        self._risk_health_cache[portfolio_id] = metrics

        self.emit(
            metric_name=CANONICAL_METRIC_NAMES.get("drawdown_pct", "risk.health"),
            tags={"portfolio_id": portfolio_id, "alert_level": metrics.alert_level},
            values={
                "total_exposure_pct": total_exposure_pct,
                "cash_pct": cash_pct,
                "leverage": leverage,
                "var_95_pct": var_95_pct,
                "drawdown_pct": drawdown_pct,
                "daily_pnl_pct": daily_pnl_pct,
                "position_count": position_count,
                "hard_stop_triggered": metrics.hard_stop_triggered,
            },
            source="risk_health",
        )

    def emit_alpha_family_metrics(
        self,
        family: str,
        regime: str,
        period_days: int = 30,
        total_return_pct: float = 0.0,
        benchmark_return_pct: float = 0.0,
        sharpe_ratio: float = 0.0,
        max_drawdown_pct: float = 0.0,
        win_rate: float = 0.0,
        avg_win_pct: float = 0.0,
        avg_loss_pct: float = 0.0,
        trade_count: int = 0,
        baseline_return_pct: float = 0.0,
    ):
        metrics = AlphaFamilyMetrics(
            family=family,
            regime=regime,
            period_days=period_days,
            total_return_pct=total_return_pct,
            benchmark_return_pct=benchmark_return_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_pct=max_drawdown_pct,
            win_rate=win_rate,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            trade_count=trade_count,
            baseline_return_pct=baseline_return_pct,
        )
        metrics.compute_alpha()

        self._alpha_family_cache[family] = metrics

        self.emit(
            metric_name=CANONICAL_METRIC_NAMES.get("alpha_pct", "alpha.family"),
            tags={"family": family, "regime": regime},
            values={
                "total_return_pct": total_return_pct,
                "benchmark_return_pct": benchmark_return_pct,
                "alpha_pct": metrics.alpha_pct,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown_pct": max_drawdown_pct,
                "win_rate": win_rate,
                "trade_count": trade_count,
                "degradation_vs_baseline_pct": metrics.degradation_vs_baseline_pct,
            },
            source="alpha_family",
        )

    def emit_workflow_phase(
        self,
        phase: str,
        status: str,
        duration_sec: float = 0.0,
        attempt: int = 1,
        outputs: Optional[Dict[str, Any]] = None,
    ):
        self.emit(
            metric_name="workflow.phase",
            tags={"phase": phase, "status": status},
            values={
                "duration_sec": duration_sec,
                "attempt": attempt,
                "output_keys": list(outputs.keys()) if outputs else [],
            },
            source="workflow_manager",
        )

    def emit_data_freshness(
        self,
        source: str,
        age_hours: float,
        is_stale: bool = False,
        record_count: int = 0,
        last_update: Optional[datetime] = None,
    ):
        self.emit(
            metric_name=CANONICAL_METRIC_NAMES.get(
                "data_freshness_hours", "data.freshness"
            ),
            tags={"source": source, "is_stale": is_stale},
            values={
                "age_hours": age_hours,
                "record_count": record_count,
                "last_update": last_update.isoformat() if last_update else None,
            },
            source="data_quality",
        )

    def get_signal_lifecycle_summary(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {
                family: {
                    "signals_evaluated": m.signals_evaluated,
                    "signals_accepted": m.signals_accepted,
                    "signals_executed": m.signals_executed,
                    "conversion_rate": m.conversion_rate,
                    "hit_rate": m.hit_rate,
                    "skip_reasons": m.skip_reasons,
                }
                for family, m in self._signal_lifecycle_cache.items()
            }

    def get_execution_quality_summary(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {
                order_id: {
                    "symbol": m.symbol,
                    "side": m.side,
                    "status": m.status,
                    "slippage_bps": m.slippage_bps,
                    "slippage_divergence_bps": m.slippage_divergence_bps,
                    "effective_spread_bps": m.effective_spread_bps,
                    "quoted_spread_bps": m.quoted_spread_bps,
                    "fill_rate": m.fill_rate,
                }
                for order_id, m in self._execution_quality_cache.items()
            }

    def get_daily_tca_summary(self, date: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            rows = list(self._execution_quality_cache.values())

        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        def _new_bucket() -> Dict[str, Any]:
            return {
                "orders": 0,
                "_slippage_sum": 0.0,
                "_impl_shortfall_sum": 0.0,
                "_effective_spread_sum": 0.0,
                "_quoted_spread_sum": 0.0,
                "total_opportunity_cost_bps": 0.0,
            }

        by_symbol: Dict[str, Dict[str, Any]] = {}
        by_order_type: Dict[str, Dict[str, Any]] = {}
        total_opp_cost_bps = 0.0
        unfilled_limit_orders = 0

        for m in rows:
            symbol = str(m.symbol or "UNKNOWN").upper()
            order_type = str(m.order_type or "unknown").lower()

            sym_bucket = by_symbol.setdefault(symbol, _new_bucket())
            type_bucket = by_order_type.setdefault(order_type, _new_bucket())
            for bucket in (sym_bucket, type_bucket):
                bucket["orders"] += 1
                bucket["_slippage_sum"] += float(m.slippage_bps or 0.0)
                bucket["_impl_shortfall_sum"] += float(
                    m.implementation_shortfall_bps or 0.0
                )
                bucket["_effective_spread_sum"] += float(m.effective_spread_bps or 0.0)
                bucket["_quoted_spread_sum"] += float(m.quoted_spread_bps or 0.0)
                bucket["total_opportunity_cost_bps"] += float(
                    m.opportunity_cost_bps or 0.0
                )

            status = str(m.status or "").lower()
            if (
                order_type == "limit"
                and float(m.fill_quantity or 0.0) <= 0.0
                and status in {"submitted", "canceled", "cannot_fill", "rejected"}
            ):
                unfilled_limit_orders += 1
                total_opp_cost_bps += max(0.0, float(m.opportunity_cost_bps or 0.0))

        def _finalize(buckets: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
            finalized: Dict[str, Dict[str, Any]] = {}
            for key, b in buckets.items():
                orders = int(b.get("orders", 0))
                denom = float(orders) if orders > 0 else 1.0
                finalized[key] = {
                    "orders": orders,
                    "avg_slippage_bps": float(b["_slippage_sum"] / denom),
                    "avg_implementation_shortfall_bps": float(
                        b["_impl_shortfall_sum"] / denom
                    ),
                    "avg_effective_spread_bps": float(b["_effective_spread_sum"] / denom),
                    "avg_quoted_spread_bps": float(b["_quoted_spread_sum"] / denom),
                    "total_opportunity_cost_bps": float(
                        b.get("total_opportunity_cost_bps", 0.0)
                    ),
                }
            return finalized

        return {
            "date": date,
            "total_orders": len(rows),
            "by_symbol": _finalize(by_symbol),
            "by_order_type": _finalize(by_order_type),
            "opportunity_cost": {
                "unfilled_limit_orders": int(unfilled_limit_orders),
                "total_bps": float(total_opp_cost_bps),
            },
        }

    def get_risk_health_summary(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {
                portfolio_id: {
                    "alert_level": m.alert_level,
                    "position_count": m.position_count,
                    "drawdown_pct": m.drawdown_pct,
                    "total_exposure_pct": m.total_exposure_pct,
                    "hard_stop_triggered": m.hard_stop_triggered,
                }
                for portfolio_id, m in self._risk_health_cache.items()
            }

    def get_alpha_family_summary(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {
                family: {
                    "regime": m.regime,
                    "alpha_pct": m.alpha_pct,
                    "sharpe_ratio": m.sharpe_ratio,
                    "max_drawdown_pct": m.max_drawdown_pct,
                    "win_rate": m.win_rate,
                    "trade_count": m.trade_count,
                    "degradation_vs_baseline_pct": m.degradation_vs_baseline_pct,
                }
                for family, m in self._alpha_family_cache.items()
            }


_collector_instance: Optional[MetricsCollector] = None
_collector_lock = threading.RLock()


def get_metrics_collector(
    output_dir: Optional[Path] = None,
    flush_interval_sec: int = 60,
    enable_console_output: bool = True,
) -> MetricsCollector:
    global _collector_instance
    with _collector_lock:
        if _collector_instance is None:
            _collector_instance = MetricsCollector(
                output_dir=output_dir,
                flush_interval_sec=flush_interval_sec,
                enable_console_output=enable_console_output,
            )
        return _collector_instance


def reset_metrics_collector():
    global _collector_instance
    with _collector_lock:
        if _collector_instance:
            _collector_instance.flush()
        _collector_instance = None
