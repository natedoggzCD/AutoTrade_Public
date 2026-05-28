"""
Risk/Portfolio Contracts - Data Structures
==========================================
Explicit data contracts for risk and portfolio decisions.
These serve as the typed interface layer between pure decision logic
and side-effecting code (broker calls, state persistence).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any


def _to_percent(amount: float, total: float) -> float:
    """Return `amount / total` as percent, guarded against zero/negative totals."""
    if total <= 0:
        return 0.0
    return (amount / total) * 100.0


class RiskActionType(Enum):
    """Possible risk actions."""

    NONE = "none"
    EXIT = "exit"
    TRIM = "trim"
    PROFIT_TAKE = "profit_take"
    ADD = "add"
    HOLD = "hold"


class PositionPhase(Enum):
    """Position lifecycle phases based on PDT constraints."""

    LOCKED = "locked"
    UNLOCKING = "unlocking"
    UNLOCKED = "unlocked"
    CRITICAL = "critical"


class ConvictionLevel(Enum):
    """Conviction level for positions."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    WEAK = "weak"
    SELL = "sell"
    CRITICAL_EXIT = "critical"


class FailureReason(Enum):
    """Reasons for risk policy failures."""

    OVER_MAX_EXPOSURE = "over_max_exposure"
    DRAWDOWN_BREACH = "drawdown_breach"
    FAILSAFE_CRITICAL = "failsafe_critical"
    INVALID_POSITION_METADATA = "invalid_position_metadata"
    STALE_DATA = "stale_data"
    PDT_LOCKED = "pdt_locked"
    SECTOR_CAP_REACHED = "sector_cap_reached"
    CORRELATION_CAP_REACHED = "correlation_cap_reached"
    FAMILY_BUDGET_EXHAUSTED = "family_budget_exhausted"


@dataclass
class PositionSnapshot:
    """
    Immutable snapshot of a position at a point in time.
    Used as input for risk/portfolio decisions.
    """

    symbol: str
    entry_price: float
    current_price: float
    qty: int
    entry_time: datetime

    # Computed metrics
    pnl_dollars: float = 0.0
    pnl_pct: float = 0.0
    market_value: float = 0.0

    # Risk metrics
    atr_pct: float = 2.0
    high_since_entry: float = 0.0
    low_since_entry: float = 0.0
    hold_minutes: float = 0.0

    # Technical context
    rsi: float = 50.0
    s1_price: float = 0.0
    r1_price: float = 0.0
    sector: str = ""
    signal_family: str = ""

    # Position phase
    phase: PositionPhase = PositionPhase.LOCKED

    # Metadata
    snapshot_time: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.qty != 0 and self.entry_price > 0:
            self.market_value = abs(self.current_price * self.qty)
            self.pnl_dollars = (self.current_price - self.entry_price) * self.qty
            self.pnl_pct = _to_percent(
                (self.current_price - self.entry_price), self.entry_price
            )

    @property
    def is_critical(self) -> bool:
        return self.phase == PositionPhase.CRITICAL

    @property
    def is_unlocked(self) -> bool:
        return self.phase in (PositionPhase.UNLOCKED, PositionPhase.UNLOCKING)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "qty": self.qty,
            "pnl_pct": self.pnl_pct,
            "pnl_dollars": self.pnl_dollars,
            "market_value": self.market_value,
            "atr_pct": self.atr_pct,
            "hold_minutes": self.hold_minutes,
            "phase": self.phase.value,
            "sector": self.sector,
            "signal_family": self.signal_family,
        }


@dataclass
class PortfolioSnapshot:
    """
    Immutable snapshot of portfolio state at a point in time.
    Used as input for portfolio-level risk decisions.
    """

    # Capital
    total_equity: float
    cash_available: float
    buying_power: float
    positions_value: float

    # Position snapshots
    positions: List[PositionSnapshot] = field(default_factory=list)

    # PDT status
    day_trades_remaining: int = 10
    day_trades_used_today: int = 0

    # Limits
    max_positions: int = 7

    # Sector exposure (sector -> percentage of portfolio)
    sector_exposure: Dict[str, float] = field(default_factory=dict)

    # Correlation cluster exposure
    correlation_cluster_exposure: float = 0.0

    # Signal family exposure (family -> percentage of portfolio)
    signal_family_exposure: Dict[str, float] = field(default_factory=dict)

    # Portfolio-level metrics
    total_pnl_dollars: float = 0.0
    total_pnl_pct: float = 0.0

    # Drawdown tracking
    peak_equity: float = 0.0
    current_drawdown_pct: float = 0.0

    # Timestamps
    snapshot_time: datetime = field(default_factory=datetime.now)

    @property
    def position_count(self) -> int:
        return len(self.positions)

    @property
    def open_slots(self) -> int:
        return max(0, self.max_positions - self.position_count)

    @property
    def gross_exposure_pct(self) -> float:
        return _to_percent(self.positions_value, self.total_equity)

    @property
    def net_exposure_pct(self) -> float:
        if self.total_equity <= 0:
            return 0.0
        if self.positions:
            # Signed notional keeps short books negative and long books positive.
            signed_notional = sum((p.current_price * p.qty) for p in self.positions)
            return _to_percent(signed_notional, self.total_equity)
        # Fallback for legacy callers that pass aggregate totals only.
        return self.gross_exposure_pct

    @property
    def can_open_new_position(self) -> bool:
        return (
            self.position_count < self.max_positions
            and self.cash_available > 0
            and self.open_slots > 0
        )

    def get_position(self, symbol: str) -> Optional[PositionSnapshot]:
        for pos in self.positions:
            if pos.symbol == symbol:
                return pos
        return None

    def get_unlocked_positions(self) -> List[PositionSnapshot]:
        return [p for p in self.positions if p.is_unlocked]

    def get_critical_positions(self) -> List[PositionSnapshot]:
        return [p for p in self.positions if p.is_critical]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_equity": self.total_equity,
            "cash_available": self.cash_available,
            "buying_power": self.buying_power,
            "positions_value": self.positions_value,
            "position_count": self.position_count,
            "open_slots": self.open_slots,
            "max_positions": self.max_positions,
            "day_trades_remaining": self.day_trades_remaining,
            "day_trades_used_today": self.day_trades_used_today,
            "total_pnl_pct": self.total_pnl_pct,
            "total_pnl_dollars": self.total_pnl_dollars,
            "gross_exposure_pct": self.gross_exposure_pct,
            "net_exposure_pct": self.net_exposure_pct,
            "peak_equity": self.peak_equity,
            "current_drawdown_pct": self.current_drawdown_pct,
            "correlation_cluster_exposure": self.correlation_cluster_exposure,
            "sector_exposure": self.sector_exposure,
            "signal_family_exposure": self.signal_family_exposure,
            "positions": [p.to_dict() for p in self.positions],
        }


@dataclass
class RiskDecision:
    """
    Result of evaluating a position against risk policies.
    Pure decision output - no side effects.
    """

    action: RiskActionType
    size_delta: int  # Negative = reduce, 0 = hold, positive = add

    # Confidence and reasoning
    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)
    triggered_rules: List[str] = field(default_factory=list)

    # PDT-related
    requires_pdt_burn: bool = False
    hold_phase: PositionPhase = PositionPhase.LOCKED

    # Additional context
    conviction_score: float = 50.0
    skip_agents: bool = False

    # Failure tracking
    failure_reasons: List[FailureReason] = field(default_factory=list)

    @property
    def is_critical_action(self) -> bool:
        """Actions that bypass normal hold constraints."""
        return self.action in (RiskActionType.EXIT, RiskActionType.TRIM) and bool(
            self.triggered_rules
        )

    @property
    def should_execute(self) -> bool:
        """Whether this decision should be acted upon."""
        return self.action != RiskActionType.NONE and len(self.failure_reasons) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "size_delta": self.size_delta,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "triggered_rules": self.triggered_rules,
            "requires_pdt_burn": self.requires_pdt_burn,
            "hold_phase": self.hold_phase.value,
            "conviction_score": self.conviction_score,
            "skip_agents": self.skip_agents,
            "failure_reasons": [f.value for f in self.failure_reasons],
            "should_execute": self.should_execute,
        }


@dataclass
class RotationPlan:
    """
    Plan to rotate from one position to another.
    Represents a proposed capital reallocation.
    """

    # Exit details
    exit_symbol: str
    exit_qty: int
    exit_price: float
    exit_conviction: float
    exit_reasons: List[str] = field(default_factory=list)

    # Entry details
    enter_symbol: str = ""
    enter_price: float = 0.0
    enter_qty: int = 0
    enter_signal_score: float = 0.0
    enter_reasons: List[str] = field(default_factory=list)

    # Assessment
    improvement_score: float = 0.0
    capital_amount: float = 0.0

    # Constraints that must be satisfied
    satisfies_min_improvement: bool = False
    satisfies_sector_limit: bool = True
    satisfies_correlation_limit: bool = True
    satisfies_family_budget: bool = True

    # Failure reasons if any
    failure_reasons: List[FailureReason] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Whether this rotation plan passes all constraints."""
        return (
            self.satisfies_min_improvement
            and self.satisfies_sector_limit
            and self.satisfies_correlation_limit
            and self.satisfies_family_budget
            and len(self.failure_reasons) == 0
        )

    def summary(self) -> str:
        return (
            f"ROTATE: {self.exit_symbol} ({self.exit_conviction:.0f}) -> "
            f"{self.enter_symbol} ({self.enter_signal_score:.0f}) | "
            f"+{self.improvement_score:.0f} | ${self.capital_amount:,.0f}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exit_symbol": self.exit_symbol,
            "exit_qty": self.exit_qty,
            "enter_symbol": self.enter_symbol,
            "enter_qty": self.enter_qty,
            "improvement_score": self.improvement_score,
            "capital_amount": self.capital_amount,
            "is_valid": self.is_valid,
            "failure_reasons": [f.value for f in self.failure_reasons],
        }


@dataclass
class RiskHealthReport:
    """
    Portfolio-level risk health assessment.
    Aggregates checks across all positions and limits.
    """

    # Overall health
    is_healthy: bool = True
    health_score: float = 100.0

    # Exposure checks
    max_position_exposure_pct: float = 0.0
    max_sector_exposure_pct: float = 0.0
    max_correlation_exposure_pct: float = 0.0
    gross_exposure_pct: float = 0.0
    net_exposure_pct: float = 0.0

    # Drawdown
    current_drawdown_pct: float = 0.0
    entry_halt_drawdown_pct: float = 0.0

    # Failures
    failures: List[FailureReason] = field(default_factory=list)
    failure_details: Dict[str, List[str]] = field(default_factory=dict)

    # Position-level issues
    critical_positions: List[str] = field(default_factory=list)
    rotation_candidates: List[str] = field(default_factory=list)
    add_candidates: List[str] = field(default_factory=list)

    # Signal family budgets
    family_budget_status: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Kill switch
    kill_switch_triggered: bool = False
    kill_switch_reason: Optional[str] = None

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Timestamp
    report_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_healthy": self.is_healthy,
            "health_score": self.health_score,
            "max_position_exposure_pct": self.max_position_exposure_pct,
            "max_sector_exposure_pct": self.max_sector_exposure_pct,
            "max_correlation_exposure_pct": self.max_correlation_exposure_pct,
            "current_drawdown_pct": self.current_drawdown_pct,
            "entry_halt_drawdown_pct": self.entry_halt_drawdown_pct,
            "gross_exposure_pct": self.gross_exposure_pct,
            "net_exposure_pct": self.net_exposure_pct,
            "failures": [f.value for f in self.failures],
            "failure_details": self.failure_details,
            "critical_positions": self.critical_positions,
            "rotation_candidates": self.rotation_candidates,
            "add_candidates": self.add_candidates,
            "family_budget_status": self.family_budget_status,
            "kill_switch_triggered": self.kill_switch_triggered,
            "kill_switch_reason": self.kill_switch_reason,
            "recommendations": self.recommendations,
            "report_time": self.report_time.isoformat(),
        }


def position_state_to_snapshot(pos) -> PositionSnapshot:
    """Convert PositionState (from position_state.py) to PositionSnapshot."""
    phase = PositionPhase.LOCKED
    hold_phase = getattr(pos, "hold_phase", None)
    if hasattr(hold_phase, "value"):
        try:
            phase = PositionPhase(hold_phase.value)
        except ValueError:
            phase = PositionPhase.LOCKED
    elif isinstance(hold_phase, str):
        try:
            phase = PositionPhase(hold_phase)
        except ValueError:
            phase = PositionPhase.LOCKED

    return PositionSnapshot(
        symbol=pos.symbol,
        entry_price=pos.entry_price,
        current_price=pos.current_price,
        qty=pos.qty,
        entry_time=pos.entry_time,
        pnl_dollars=pos.pnl_dollars,
        pnl_pct=pos.pnl_pct,
        market_value=pos.current_price * pos.qty
        if pos.qty and pos.current_price
        else 0.0,
        atr_pct=pos.atr_pct,
        high_since_entry=pos.high_since_entry,
        low_since_entry=pos.low_since_entry,
        hold_minutes=pos.hold_minutes,
        rsi=pos.rsi,
        s1_price=pos.s1_price,
        r1_price=pos.r1_price,
        sector=getattr(pos, "sector", ""),
        signal_family=getattr(pos, "signal_family", ""),
        phase=phase,
        snapshot_time=getattr(pos, "last_update", None) or datetime.now(),
    )


def portfolio_state_to_snapshot(portfolio) -> PortfolioSnapshot:
    """Convert PortfolioState (from position_state.py) to PortfolioSnapshot."""
    positions_dict = getattr(portfolio, "positions", {})
    positions = [position_state_to_snapshot(p) for p in positions_dict.values()]

    sector_exposure: Dict[str, float] = {}
    signal_family_exposure: Dict[str, float] = {}

    for pos in positions:
        if pos.sector:
            sector_exposure[pos.sector] = (
                sector_exposure.get(pos.sector, 0.0) + pos.market_value
            )
        if pos.signal_family:
            signal_family_exposure[pos.signal_family] = (
                signal_family_exposure.get(pos.signal_family, 0.0) + pos.market_value
            )

    total_equity = float(getattr(portfolio, "total_equity", 0.0))
    if total_equity > 0:
        for sector in sector_exposure:
            sector_exposure[sector] = _to_percent(sector_exposure[sector], total_equity)
        for family in signal_family_exposure:
            signal_family_exposure[family] = _to_percent(
                signal_family_exposure[family], total_equity
            )

    pdt_status = getattr(portfolio, "pdt_status", None)
    day_trades_remaining = (
        int(getattr(pdt_status, "trades_remaining", 10)) if pdt_status else 10
    )
    day_trades_used_today = (
        int(getattr(pdt_status, "trades_used_today", 0)) if pdt_status else 0
    )
    peak_equity = float(getattr(portfolio, "peak_equity", total_equity))
    current_drawdown_pct = 0.0
    if peak_equity > 0:
        current_drawdown_pct = _to_percent(max(0.0, peak_equity - total_equity), peak_equity)

    return PortfolioSnapshot(
        total_equity=total_equity,
        cash_available=float(getattr(portfolio, "cash_available", 0.0)),
        buying_power=float(getattr(portfolio, "buying_power", 0.0)),
        positions_value=float(getattr(portfolio, "positions_value", 0.0)),
        positions=positions,
        day_trades_remaining=day_trades_remaining,
        day_trades_used_today=day_trades_used_today,
        max_positions=int(getattr(portfolio, "max_positions", 7)),
        sector_exposure=sector_exposure,
        correlation_cluster_exposure=float(
            getattr(portfolio, "correlation_cluster_exposure", 0.0)
        ),
        signal_family_exposure=signal_family_exposure,
        total_pnl_dollars=float(getattr(portfolio, "total_pnl", 0.0)),
        total_pnl_pct=float(getattr(portfolio, "total_pnl_pct", 0.0)),
        peak_equity=peak_equity,
        current_drawdown_pct=current_drawdown_pct,
        snapshot_time=getattr(portfolio, "last_update", None) or datetime.now(),
    )
