from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class MetricNames:
    SYSTEM_HEALTH = "system.health"
    SIGNAL_LIFECYCLE = "signal.lifecycle"
    ALPHA_FAMILY = "alpha.family"
    EXECUTION_QUALITY = "execution.quality"
    RISK_HEALTH = "risk.health"
    RELEASE_GATE = "release.gate"


class SignalFamily(Enum):
    MOMENTUM = "momentum"
    VALUE = "value"
    GROWTH = "growth"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SCREEN = "screen"
    LLM = "llm"


class RegimeType(Enum):
    TREND = "trend"
    CHOP = "chop"
    CRISIS = "crisis"
    NEUTRAL = "neutral"


class GateType(Enum):
    TEST = "test"
    BACKTEST = "backtest"
    PAPER = "paper"
    CANARY = "canary"
    PRODUCTION = "production"


class GateStatus(Enum):
    PENDING = "pending"
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class BaseMetric:
    metric_name: str = field(init=False)
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealthEvent(BaseMetric):
    component: str = ""
    status: str = "healthy"
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    error_count: int = 0
    warning_count: int = 0
    message: str = ""

    def __post_init__(self):
        self.metric_name = MetricNames.SYSTEM_HEALTH
        self.tags = {"component": self.component, "status": self.status}


@dataclass
class SignalLifecycleMetrics(BaseMetric):
    family: str = ""
    signal_count: int = 0
    signals_evaluated: int = 0
    signals_accepted: int = 0
    signals_rejected: int = 0
    signals_executed: int = 0
    skip_reasons: dict = field(default_factory=dict)
    hit_rate: float = 0.0
    expectancy: float = 0.0
    avg_duration_hours: float = 0.0
    turnover_rate: float = 0.0
    cost_impact_bps: float = 0.0
    conversion_rate: float = 0.0

    def __post_init__(self):
        self.metric_name = MetricNames.SIGNAL_LIFECYCLE
        self.tags = {"family": self.family}

    def compute_derived_metrics(self):
        if self.signals_evaluated > 0:
            self.conversion_rate = self.signals_executed / self.signals_evaluated
        if self.signals_accepted > 0:
            self.hit_rate = self.signals_executed / self.signals_accepted
            self.turnover_rate = (
                self.signals_accepted / self.signals_evaluated
                if self.signals_evaluated > 0
                else 0.0
            )


@dataclass
class AlphaFamilyMetrics(BaseMetric):
    family: str = ""
    regime: str = ""
    period_days: int = 30
    total_return_pct: float = 0.0
    benchmark_return_pct: float = 0.0
    alpha_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    trade_count: int = 0
    degradation_vs_baseline_pct: float = 0.0
    baseline_return_pct: float = 0.0

    def __post_init__(self):
        self.metric_name = MetricNames.ALPHA_FAMILY
        self.tags = {"family": self.family, "regime": self.regime}

    def compute_alpha(self):
        self.alpha_pct = self.total_return_pct - self.benchmark_return_pct
        if self.degradation_vs_baseline_pct == 0.0 and self.baseline_return_pct != 0:
            self.degradation_vs_baseline_pct = (
                (self.total_return_pct - self.baseline_return_pct)
                / abs(self.baseline_return_pct)
            ) * 100


@dataclass
class ExecutionQualityMetrics(BaseMetric):
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    order_type: str = ""
    quantity: float = 0.0
    limit_price: Optional[float] = None
    fill_price: Optional[float] = None
    expected_price: Optional[float] = None
    arrival_price: Optional[float] = None
    bid_price_at_arrival: Optional[float] = None
    ask_price_at_arrival: Optional[float] = None
    mid_price_at_arrival: Optional[float] = None
    slippage_bps: float = 0.0
    slippage_expected_bps: float = 0.0
    slippage_divergence_bps: float = 0.0
    implementation_shortfall_bps: float = 0.0
    effective_spread_bps: float = 0.0
    quoted_spread_bps: float = 0.0
    opportunity_cost_bps: float = 0.0
    execution_latency_ms: float = 0.0
    fill_quantity: float = 0.0
    fill_rate: float = 0.0
    rejection_reason: str = ""
    status: str = "pending"

    def __post_init__(self):
        self.metric_name = MetricNames.EXECUTION_QUALITY
        self.tags = {"symbol": self.symbol, "side": self.side, "status": self.status}

    def compute_slippage_metrics(self):
        side_sign = 1.0 if str(self.side).lower() == "buy" else -1.0
        if self.expected_price and self.fill_price and self.expected_price > 0:
            self.slippage_bps = (
                (self.fill_price - self.expected_price) / self.expected_price
            ) * 10000
            self.slippage_divergence_bps = abs(
                self.slippage_bps - self.slippage_expected_bps
            )
        arrival_px = (
            float(self.arrival_price)
            if self.arrival_price is not None
            else (
                float(self.expected_price)
                if self.expected_price is not None
                else 0.0
            )
        )
        if arrival_px > 0 and self.fill_price and self.fill_price > 0:
            self.implementation_shortfall_bps = (
                ((float(self.fill_price) - arrival_px) / arrival_px)
                * 10000.0
                * side_sign
            )
        bid = (
            float(self.bid_price_at_arrival)
            if self.bid_price_at_arrival is not None
            else 0.0
        )
        ask = (
            float(self.ask_price_at_arrival)
            if self.ask_price_at_arrival is not None
            else 0.0
        )
        mid = (
            float(self.mid_price_at_arrival)
            if self.mid_price_at_arrival is not None
            else ((bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0)
        )
        if mid > 0 and bid > 0 and ask > 0 and ask >= bid:
            self.quoted_spread_bps = ((ask - bid) / mid) * 10000.0
        if mid > 0 and self.fill_price and self.fill_price > 0:
            self.effective_spread_bps = (
                abs(float(self.fill_price) - mid) / mid
            ) * 10000.0 * 2.0
        if (
            str(self.order_type).lower() == "limit"
            and str(self.status).lower() in {"submitted", "canceled", "cannot_fill", "rejected"}
            and self.limit_price is not None
            and arrival_px > 0
            and self.fill_quantity <= 0
        ):
            limit_px = float(self.limit_price)
            if str(self.side).lower() == "buy":
                self.opportunity_cost_bps = max(
                    0.0, ((arrival_px - limit_px) / arrival_px) * 10000.0
                )
            else:
                self.opportunity_cost_bps = max(
                    0.0, ((limit_px - arrival_px) / arrival_px) * 10000.0
                )
        if self.quantity > 0:
            self.fill_rate = self.fill_quantity / self.quantity


@dataclass
class RiskHealthMetrics(BaseMetric):
    portfolio_id: str = ""
    total_exposure_pct: float = 0.0
    cash_pct: float = 0.0
    leverage: float = 1.0
    var_95_pct: float = 0.0
    drawdown_pct: float = 0.0
    drawdown_peak_pct: float = 0.0
    daily_pnl_pct: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl_pct: float = 0.0
    position_count: int = 0
    max_position_size_pct: float = 0.0
    sector_exposures: dict = field(default_factory=dict)
    beta_exposure: float = 0.0
    pdt_violations_24h: int = 0
    margin_calls: int = 0
    hard_stop_triggered: bool = False
    alert_level: str = "green"

    def __post_init__(self):
        self.metric_name = MetricNames.RISK_HEALTH
        self.tags = {"portfolio_id": self.portfolio_id, "alert_level": self.alert_level}

    def compute_risk_state(self):
        if self.drawdown_pct >= 8.0:
            self.alert_level = "critical"
            self.hard_stop_triggered = True
        elif self.drawdown_pct >= 5.0:
            self.alert_level = "red"
        elif self.drawdown_pct >= 3.0:
            self.alert_level = "yellow"
        else:
            self.alert_level = "green"


@dataclass
class ReleaseGateStatus(BaseMetric):
    gate_type: str = ""
    gate_status: str = "pending"
    test_results: dict = field(default_factory=dict)
    backtest_status: str = ""
    backtest_regression_pct: float = 0.0
    paper_trading_days: int = 0
    paper_trading_min_days: int = 10
    paper_performance_pct: float = 0.0
    canary_risk_pct: float = 0.0
    canary_max_risk_pct: float = 10.0
    canary_ready: bool = False
    rollback_ready: bool = False
    pass_reason: str = ""
    fail_reason: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.metric_name = MetricNames.RELEASE_GATE
        self.tags = {"gate_type": self.gate_type, "status": self.gate_status}

    def evaluate(self):
        if self.gate_type == GateType.TEST.value:
            self.gate_status = (
                GateStatus.PASS.value
                if self.test_results.get("passed", False)
                else GateStatus.FAIL.value
            )
        elif self.gate_type == GateType.BACKTEST.value:
            self.gate_status = (
                GateStatus.PASS.value
                if self.backtest_regression_pct >= -5.0
                else GateStatus.FAIL.value
            )
        elif self.gate_type == GateType.PAPER.value:
            self.gate_status = (
                GateStatus.PASS.value
                if (
                    self.paper_trading_days >= self.paper_trading_min_days
                    and self.paper_performance_pct >= 0
                )
                else GateStatus.FAIL.value
            )
        elif self.gate_type == GateType.CANARY.value:
            self.canary_ready = self.canary_risk_pct <= self.canary_max_risk_pct
            self.gate_status = (
                GateStatus.PASS.value if self.canary_ready else GateStatus.FAIL.value
            )

        self.rollback_ready = (
            self.gate_status == GateStatus.FAIL.value or self.paper_trading_days >= 5
        )


CANONICAL_METRIC_NAMES = {
    "signal_count": "signal.count",
    "signal_executed": "signal.executed",
    "signal_rejected": "signal.rejected",
    "conversion_rate": "signal.conversion_rate",
    "hit_rate": "signal.hit_rate",
    "expectancy": "signal.expectancy",
    "slippage_bps": "execution.slippage_bps",
    "slippage_divergence": "execution.slippage_divergence_bps",
    "fill_rate": "execution.fill_rate",
    "latency_ms": "execution.latency_ms",
    "drawdown_pct": "risk.drawdown_pct",
    "var_95": "risk.var_95_pct",
    "exposure_pct": "risk.exposure_pct",
    "alpha_pct": "alpha.return_pct",
    "sharpe": "alpha.sharpe_ratio",
    "regime": "market.regime",
    "data_freshness_hours": "data.freshness_hours",
}


CANONICAL_TAGS = {
    "source": "source",
    "family": "signal_family",
    "regime": "regime_type",
    "symbol": "symbol",
    "portfolio": "portfolio_id",
    "phase": "workflow_phase",
    "gate": "gate_type",
}
