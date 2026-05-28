from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from autotrade.utils.safe_logging import get_safe_logger
except ImportError:
    import logging

    get_safe_logger = lambda name: logging.getLogger(name)


class AlertLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    DRAWDOWN_BREACH = "drawdown_breach"
    TRADE_CONVERSION_DEGRADATION = "trade_conversion_degradation"
    SLIPPAGE_DIVERGENCE = "slippage_divergence"
    ALPHA_DEGRADATION = "alpha_degradation"
    REGIME_DRIFT = "regime_drift"
    DATA_STALENESS = "data_staleness"
    STALE_RESEARCH_CRITICAL = "stale_research_critical"
    EXECUTION_FAILURE = "execution_failure"
    ORDER_FAILURE = "order_failure"
    DNS_RESOLUTION_FAILURE = "dns_resolution_failure"


@dataclass
class Alert:
    alert_type: str
    level: str
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, Any] = field(default_factory=dict)
    metric_values: Dict[str, Any] = field(default_factory=dict)
    threshold: Optional[Any] = None
    actual_value: Optional[Any] = None


class AlertRule:
    def __init__(
        self,
        name: str,
        alert_type: AlertType,
        threshold: Any,
        comparator: Callable[[Any, Any], bool],
        level: AlertLevel = AlertLevel.WARNING,
        description: str = "",
    ):
        self.name = name
        self.alert_type = alert_type
        self.threshold = threshold
        self.comparator = comparator
        self.level = level
        self.description = description

    def evaluate(self, value: Any) -> bool:
        try:
            return self.comparator(value, self.threshold)
        except Exception:
            return False

    def create_alert(
        self, actual_value: Any, tags: Optional[Dict[str, Any]] = None
    ) -> Alert:
        return Alert(
            alert_type=self.alert_type.value,
            level=self.level.value,
            message=self.description
            or f"{self.name}: {actual_value} (threshold: {self.threshold})",
            tags=tags or {},
            metric_values={"actual": actual_value},
            threshold=self.threshold,
            actual_value=actual_value,
        )


class AlertManager:
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        enable_console: bool = True,
    ):
        self.output_dir = output_dir or Path("logs/alerts")
        self.enable_console = enable_console
        self.logger = get_safe_logger("alert_manager")

        self._rules: Dict[AlertType, AlertRule] = {}
        self._alert_history: List[Alert] = []
        self._lock = threading.RLock()
        self._callbacks: List[Callable[[Alert], None]] = []

        self._baseline_conversion_rate: Optional[float] = None
        self._baseline_alpha: Dict[str, float] = {}
        self._baseline_regime_dist: Optional[Dict[str, float]] = None
        self._recent_conversion_rates: List[float] = []
        self._recent_alpha_values: Dict[str, List[float]] = {}
        self._failure_counts: Dict[str, int] = {}

        self._setup_default_rules()
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _setup_default_rules(self):
        self.add_rule(
            AlertRule(
                name="drawdown_alarm",
                alert_type=AlertType.DRAWDOWN_BREACH,
                threshold=5.0,
                comparator=lambda v, t: v >= t,
                level=AlertLevel.ERROR,
                description="Drawdown exceeds alarm threshold",
            )
        )

        self.add_rule(
            AlertRule(
                name="critical_drawdown",
                alert_type=AlertType.DRAWDOWN_BREACH,
                threshold=8.0,
                comparator=lambda v, t: v >= t,
                level=AlertLevel.CRITICAL,
                description="Critical drawdown - hard stop triggered",
            )
        )

        self.add_rule(
            AlertRule(
                name="trade_conversion_rate",
                alert_type=AlertType.TRADE_CONVERSION_DEGRADATION,
                threshold=0.05,
                comparator=lambda v, t: v < t,
                level=AlertLevel.WARNING,
                description="Trade conversion rate below floor",
            )
        )

        self.add_rule(
            AlertRule(
                name="slippage_divergence",
                alert_type=AlertType.SLIPPAGE_DIVERGENCE,
                threshold=20,
                comparator=lambda v, t: v > t,
                level=AlertLevel.WARNING,
                description="Realized vs expected slippage exceeds threshold (bps)",
            )
        )

        self.add_rule(
            AlertRule(
                name="data_staleness",
                alert_type=AlertType.DATA_STALENESS,
                threshold=24,
                comparator=lambda v, t: v > t,
                level=AlertLevel.ERROR,
                description="Data is stale",
            )
        )

        self.add_rule(
            AlertRule(
                name="alpha_degradation",
                alert_type=AlertType.ALPHA_DEGRADATION,
                threshold=30.0,
                comparator=lambda v, t: v > t,
                level=AlertLevel.WARNING,
                description="Alpha family degraded vs baseline",
            )
        )

        self.add_rule(
            AlertRule(
                name="regime_drift",
                alert_type=AlertType.REGIME_DRIFT,
                threshold=0.2,
                comparator=lambda v, t: v > t,
                level=AlertLevel.WARNING,
                description="Regime distribution drift detected (PSI)",
            )
        )

        self.add_rule(
            AlertRule(
                name="execution_failure_limit",
                alert_type=AlertType.EXECUTION_FAILURE,
                threshold=3,
                comparator=lambda v, t: v >= t,
                level=AlertLevel.ERROR,
                description="Repeated execution failures detected",
            )
        )

        self.add_rule(
            AlertRule(
                name="order_failure_limit",
                alert_type=AlertType.ORDER_FAILURE,
                threshold=5,
                comparator=lambda v, t: v >= t,
                level=AlertLevel.ERROR,
                description="Repeated order failures detected",
            )
        )

        self.add_rule(
            AlertRule(
                name="dns_resolution_failure",
                alert_type=AlertType.DNS_RESOLUTION_FAILURE,
                threshold=3,
                comparator=lambda v, t: v >= t,
                level=AlertLevel.CRITICAL,
                description="Repeated DNS resolution failures detected",
            )
        )

        self.add_rule(
            AlertRule(
                name="stale_research_critical",
                alert_type=AlertType.STALE_RESEARCH_CRITICAL,
                threshold=18,
                comparator=lambda v, t: v > t,
                level=AlertLevel.CRITICAL,
                description="Research is critically stale (>18h)",
            )
        )

    def add_rule(self, rule: AlertRule):
        with self._lock:
            self._rules[rule.alert_type] = rule
            self.logger.debug(f"Added alert rule: {rule.name}")

    def remove_rule(self, alert_type: AlertType):
        with self._lock:
            self._rules.pop(alert_type, None)

    def register_callback(self, callback: Callable[[Alert], None]):
        with self._lock:
            self._callbacks.append(callback)

    def set_baseline_conversion_rate(self, rate: float):
        with self._lock:
            self._baseline_conversion_rate = rate

    def update_baseline_conversion_rate(self, rate: float):
        with self._lock:
            self._recent_conversion_rates.append(rate)
            if len(self._recent_conversion_rates) > 10:
                self._recent_conversion_rates.pop(0)
            if self._baseline_conversion_rate is None:
                self._baseline_conversion_rate = rate

    def set_baseline_alpha(self, family: str, alpha: float):
        with self._lock:
            if family not in self._baseline_alpha:
                self._baseline_alpha[family] = alpha
            if family not in self._recent_alpha_values:
                self._recent_alpha_values[family] = []
            self._recent_alpha_values[family].append(alpha)
            if len(self._recent_alpha_values[family]) > 10:
                self._recent_alpha_values[family].pop(0)

    def set_baseline_regime_distribution(self, distribution: Dict[str, float]):
        with self._lock:
            self._baseline_regime_dist = distribution

    def record_failure(self, failure_type: str):
        with self._lock:
            self._failure_counts[failure_type] = (
                self._failure_counts.get(failure_type, 0) + 1
            )
            
            # Check for immediate alert trigger
            if failure_type.startswith("order_failure_"):
                symbol = failure_type.replace("order_failure_", "")
                alert = self.check_order_failure(symbol)
            else:
                alert = self.check_execution_failure(failure_type)
                
            if alert:
                self.emit(alert)

    def check_drawdown(self, drawdown_pct: float) -> Optional[Alert]:
        with self._lock:
            rule = self._rules.get(AlertType.DRAWDOWN_BREACH)
            if rule and rule.evaluate(drawdown_pct):
                return rule.create_alert(drawdown_pct, {"drawdown_pct": drawdown_pct})
            return None

    def check_trade_conversion(
        self,
        conversion_rate: float,
        family: Optional[str] = None,
    ) -> Optional[Alert]:
        with self._lock:
            if self._baseline_conversion_rate is not None:
                degradation = (
                    (self._baseline_conversion_rate - conversion_rate)
                    / self._baseline_conversion_rate
                ) * 100
                if degradation > 30.0:
                    rule = self._rules.get(AlertType.TRADE_CONVERSION_DEGRADATION)
                    if rule:
                        return rule.create_alert(
                            conversion_rate,
                            {
                                "family": family,
                                "baseline": self._baseline_conversion_rate,
                                "degradation_pct": degradation,
                            },
                        )

            rule = self._rules.get(AlertType.TRADE_CONVERSION_DEGRADATION)
            if rule and rule.evaluate(conversion_rate):
                return rule.create_alert(
                    conversion_rate,
                    {"family": family, "baseline": self._baseline_conversion_rate},
                )
            return None

    def check_slippage_divergence(
        self,
        slippage_divergence_bps: float,
        symbol: Optional[str] = None,
    ) -> Optional[Alert]:
        with self._lock:
            rule = self._rules.get(AlertType.SLIPPAGE_DIVERGENCE)
            if rule and rule.evaluate(slippage_divergence_bps):
                return rule.create_alert(
                    slippage_divergence_bps,
                    {"symbol": symbol, "slippage_bps": slippage_divergence_bps},
                )
            return None

    def check_data_staleness(
        self,
        age_hours: float,
        source: str,
    ) -> Optional[Alert]:
        with self._lock:
            rule = self._rules.get(AlertType.DATA_STALENESS)
            if rule and rule.evaluate(age_hours):
                return rule.create_alert(
                    age_hours,
                    {"source": source, "age_hours": age_hours},
                )
            return None

    def check_alpha_degradation(
        self,
        family: str,
        current_alpha: float,
    ) -> Optional[Alert]:
        with self._lock:
            baseline = self._baseline_alpha.get(family)
            if baseline is not None and baseline != 0:
                degradation = ((baseline - current_alpha) / abs(baseline)) * 100
                rule = self._rules.get(AlertType.ALPHA_DEGRADATION)
                if rule and rule.evaluate(degradation):
                    return rule.create_alert(
                        degradation,
                        {
                            "family": family,
                            "current_alpha": current_alpha,
                            "baseline_alpha": baseline,
                            "degradation_pct": degradation,
                        },
                    )
            return None

    def check_regime_drift(
        self,
        current_distribution: Dict[str, float],
    ) -> Optional[Alert]:
        with self._lock:
            if self._baseline_regime_dist is None:
                return None

            psi = self._compute_psi(current_distribution, self._baseline_regime_dist)

            rule = self._rules.get(AlertType.REGIME_DRIFT)
            if rule and rule.evaluate(psi):
                return rule.create_alert(
                    psi,
                    {
                        "current_distribution": current_distribution,
                        "baseline_distribution": self._baseline_regime_dist,
                    },
                )
            return None

    def _compute_psi(
        self,
        actual: Dict[str, float],
        expected: Dict[str, float],
    ) -> float:
        psi = 0.0
        all_keys = set(actual.keys()) | set(expected.keys())

        for key in all_keys:
            actual_pct = actual.get(key, 0.001)
            expected_pct = expected.get(key, 0.001)

            try:
                psi += (actual_pct - expected_pct) * (
                    (actual_pct / expected_pct) if expected_pct > 0 else 0
                )
            except (ValueError, ZeroDivisionError):
                pass

        return abs(psi)

    def check_execution_failure(
        self,
        error_type: str,
        threshold: Optional[int] = None,
    ) -> Optional[Alert]:
        with self._lock:
            # Try to resolve threshold from rules if not provided
            rule = self._rules.get(AlertType.EXECUTION_FAILURE)
            if threshold is None and rule:
                threshold = rule.threshold
            
            eff_threshold = threshold if threshold is not None else 3

            count = self._failure_counts.get(error_type, 0)
            if count >= eff_threshold:
                level = AlertLevel.ERROR.value
                if rule:
                    level = rule.level.value

                return Alert(
                    alert_type=AlertType.EXECUTION_FAILURE.value,
                    level=level,
                    message=f"Repeated execution failures: {error_type} ({count} occurrences)",
                    tags={"error_type": error_type, "count": count},
                    metric_values={"failure_count": count},
                    threshold=eff_threshold,
                    actual_value=count,
                )
            return None

    def check_order_failure(
        self,
        symbol: str,
        threshold: Optional[int] = None,
    ) -> Optional[Alert]:
        key = f"order_failure_{symbol}"
        
        with self._lock:
            # Try to resolve threshold from rules if not provided
            rule = self._rules.get(AlertType.ORDER_FAILURE)
            if threshold is None and rule:
                threshold = rule.threshold
            
            eff_threshold = threshold if threshold is not None else 3
            
            count = self._failure_counts.get(key, 0)
            
            if count >= eff_threshold:
                level = AlertLevel.ERROR.value
                if rule:
                    level = rule.level.value

                return Alert(
                    alert_type=AlertType.ORDER_FAILURE.value,
                    level=level,
                    message=f"Repeated order failures for {symbol}: ({count} occurrences)",
                    tags={"symbol": symbol, "count": count},
                    metric_values={"failure_count": count},
                    threshold=eff_threshold,
                    actual_value=count,
                )
            return None

    def emit_dns_failure(self, error_message: str, count: int):
        """Specifically emit a DNS resolution failure alert."""
        with self._lock:
            rule = self._rules.get(AlertType.DNS_RESOLUTION_FAILURE)
            if rule:
                alert = rule.create_alert(
                    count,
                    {"error": error_message, "count": count}
                )
                self.emit(alert)

    def check_stale_research(self, age_hours: float) -> Optional[Alert]:
        """Check if research is critically stale."""
        with self._lock:
            rule = self._rules.get(AlertType.STALE_RESEARCH_CRITICAL)
            if rule and rule.evaluate(age_hours):
                return rule.create_alert(
                    age_hours,
                    {"age_hours": age_hours}
                )
            return None

    def emit(self, alert: Alert):
        with self._lock:
            self._alert_history.append(alert)

            for callback in self._callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.warning(f"Alert callback error: {e}")

            if self.enable_console:
                self._log_alert(alert)

            self._write_alert(alert)

    def _log_alert(self, alert: Alert):
        level_map = {
            "debug": self.logger.debug,
            "info": self.logger.info,
            "warning": self.logger.warning,
            "error": self.logger.error,
            "critical": self.logger.critical,
        }
        log_func = level_map.get(alert.level, self.logger.info)
        log_func(f"[{alert.alert_type}] {alert.message}")

    def _write_alert(self, alert: Alert):
        try:
            import json

            date_str = alert.timestamp.strftime("%Y-%m-%d")
            output_path = self.output_dir / f"alerts_{date_str}.jsonl"

            with open(output_path, "a") as f:
                record = {
                    "alert_type": alert.alert_type,
                    "level": alert.level,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "tags": alert.tags,
                    "metric_values": alert.metric_values,
                    "threshold": str(alert.threshold) if alert.threshold else None,
                    "actual_value": (
                        str(alert.actual_value)
                        if alert.actual_value is not None
                        else None
                    ),
                }
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write alert: {e}")

    def get_alerts_since(
        self,
        since: datetime,
        alert_type: Optional[str] = None,
    ) -> List[Alert]:
        with self._lock:
            filtered = [a for a in self._alert_history if a.timestamp >= since]
            if alert_type:
                filtered = [a for a in filtered if a.alert_type == alert_type]
            return filtered

    def get_alert_summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_alerts": len(self._alert_history),
                "by_level": self._count_by_level(),
                "by_type": self._count_by_type(),
                "baseline_conversion_rate": self._baseline_conversion_rate,
                "baseline_alpha_families": list(self._baseline_alpha.keys()),
                "failure_counts": dict(self._failure_counts),
            }

    def _count_by_level(self) -> Dict[str, int]:
        counts = {}
        for alert in self._alert_history:
            counts[alert.level] = counts.get(alert.level, 0) + 1
        return counts

    def _count_by_type(self) -> Dict[str, int]:
        counts = {}
        for alert in self._alert_history:
            counts[alert.alert_type] = counts.get(alert.alert_type, 0) + 1
        return counts


_alert_manager_instance: Optional[AlertManager] = None
_alert_manager_lock = threading.RLock()


def get_alert_manager(
    output_dir: Optional[Path] = None,
    enable_console: bool = True,
) -> AlertManager:
    global _alert_manager_instance
    with _alert_manager_lock:
        if _alert_manager_instance is None:
            _alert_manager_instance = AlertManager(
                output_dir=output_dir,
                enable_console=enable_console,
            )
        return _alert_manager_instance


def reset_alert_manager():
    global _alert_manager_instance
    with _alert_manager_lock:
        _alert_manager_instance = None
