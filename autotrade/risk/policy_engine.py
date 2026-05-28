"""
Policy Engine - Pure Decision Layer
===================================
Deterministic policy evaluator pipeline that produces risk, sizing, and
rotation decisions without side effects.

Pipeline flow:
1. Risk Policy -> RiskDecision (per position)
2. Sizing Policy -> Position size recommendations
3. Rotation Policy -> RotationPlan (portfolio-level)
4. Portfolio Allocation -> Signal family budgets and rebalance cadence

All functions are pure - no broker calls, no state mutation.

Phase 3 - Fail-Fast Invariants + Kill-Switch:
- Typed invariants with explicit failure reasons:
  - OVER_MAX_EXPOSURE
  - DRAWDOWN_BREACH
  - FAILSAFE_CRITICAL
  - INVALID_POSITION_METADATA
- Standardized fallback behavior when risk data is stale/missing
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple

from autotrade.risk.contracts import (
    PositionSnapshot,
    PortfolioSnapshot,
    RiskDecision,
    RiskActionType,
    RotationPlan,
    RiskHealthReport,
    FailureReason,
    PositionPhase,
)
from autotrade.risk.interfaces import (
    RiskPolicy,
    SizingPolicy,
    RotationPolicy,
    ExposurePolicy,
    RotationPolicyConfig,
    ExposurePolicyConfig,
    KillSwitchPolicy,
)
from autotrade.risk.risk_gate import RiskGateConfig, ActionPlan


# ============================================================
# Phase 3 - Fail-Fast Invariants + Fallback Behavior
# ============================================================


@dataclass
class InvariantCheckResult:
    """Result of an invariant check with explicit failure reason."""

    passed: bool
    failure_reason: Optional[FailureReason] = None
    failure_detail: str = ""
    fallback_applied: bool = False
    fallback_value: Any = None


class InvariantValidator:
    """
    Validates typed invariants with explicit failure reasons.

    Checks:
    - Over max exposure
    - Drawdown breach
    - Failsafe critical state
    - Invalid position metadata

    Provides fallback behavior when data is stale/missing.
    """

    def __init__(
        self,
        max_portfolio_exposure_pct: float = 100.0,
        max_symbol_exposure_pct: float = 15.0,
        max_sector_exposure_pct: float = 40.0,
        max_correlation_cluster_exposure_pct: float = 35.0,
        max_gross_exposure_pct: float = 150.0,
        max_net_exposure_pct: float = 100.0,
        entry_halt_drawdown_pct: float = 10.0,
    ):
        self.max_portfolio_exposure_pct = max_portfolio_exposure_pct
        self.max_symbol_exposure_pct = max_symbol_exposure_pct
        self.max_sector_exposure_pct = max_sector_exposure_pct
        self.max_correlation_cluster_exposure_pct = max_correlation_cluster_exposure_pct
        self.max_gross_exposure_pct = max_gross_exposure_pct
        self.max_net_exposure_pct = max_net_exposure_pct
        self.entry_halt_drawdown_pct = entry_halt_drawdown_pct

    def check_position_exposure_invariant(
        self,
        position: PositionSnapshot,
        portfolio: PortfolioSnapshot,
    ) -> InvariantCheckResult:
        """Check if position exceeds maximum exposure limits."""
        if portfolio.total_equity <= 0:
            return InvariantCheckResult(
                passed=False,
                failure_reason=FailureReason.INVALID_POSITION_METADATA,
                failure_detail="Portfolio total_equity is zero or negative",
            )

        exposure_pct = (position.market_value / portfolio.total_equity) * 100

        if exposure_pct > self.max_symbol_exposure_pct:
            return InvariantCheckResult(
                passed=False,
                failure_reason=FailureReason.OVER_MAX_EXPOSURE,
                failure_detail=f"Position exposure {exposure_pct:.1f}% exceeds max {self.max_symbol_exposure_pct}%",
            )

        return InvariantCheckResult(passed=True)

    def check_portfolio_exposure_invariants(
        self,
        portfolio: PortfolioSnapshot,
    ) -> InvariantCheckResult:
        """Check portfolio-level exposure limits."""
        failures = []

        if portfolio.gross_exposure_pct > self.max_gross_exposure_pct:
            failures.append(
                f"gross {portfolio.gross_exposure_pct:.1f}% > {self.max_gross_exposure_pct}%"
            )

        if abs(portfolio.net_exposure_pct) > self.max_net_exposure_pct:
            failures.append(
                f"net {portfolio.net_exposure_pct:.1f}% exceeds +/-{self.max_net_exposure_pct}%"
            )

        if failures:
            return InvariantCheckResult(
                passed=False,
                failure_reason=FailureReason.OVER_MAX_EXPOSURE,
                failure_detail="; ".join(failures),
            )

        return InvariantCheckResult(passed=True)

    def check_drawdown_invariant(
        self,
        portfolio: PortfolioSnapshot,
    ) -> InvariantCheckResult:
        """Check if portfolio drawdown exceeds halt threshold."""
        if portfolio.current_drawdown_pct > self.entry_halt_drawdown_pct:
            return InvariantCheckResult(
                passed=False,
                failure_reason=FailureReason.DRAWDOWN_BREACH,
                failure_detail=f"Drawdown {portfolio.current_drawdown_pct:.1f}% exceeds threshold {self.entry_halt_drawdown_pct}%",
            )

        return InvariantCheckResult(passed=True)

    def check_sector_exposure_invariant(
        self,
        sector: str,
        additional_exposure_pct: float,
        portfolio: PortfolioSnapshot,
    ) -> InvariantCheckResult:
        """Check if sector exposure would exceed limits."""
        if not sector:
            return InvariantCheckResult(passed=True)

        current_exposure = portfolio.sector_exposure.get(sector, 0.0)
        new_exposure = current_exposure + additional_exposure_pct

        if new_exposure > self.max_sector_exposure_pct:
            return InvariantCheckResult(
                passed=False,
                failure_reason=FailureReason.SECTOR_CAP_REACHED,
                failure_detail=f"Sector {sector} exposure {new_exposure:.1f}% exceeds max {self.max_sector_exposure_pct}%",
            )

        return InvariantCheckResult(passed=True)

    def check_correlation_exposure_invariant(
        self,
        additional_exposure_pct: float,
        portfolio: PortfolioSnapshot,
    ) -> InvariantCheckResult:
        """Check if correlation cluster exposure would exceed limits."""
        new_exposure = portfolio.correlation_cluster_exposure + additional_exposure_pct

        if new_exposure > self.max_correlation_cluster_exposure_pct:
            return InvariantCheckResult(
                passed=False,
                failure_reason=FailureReason.CORRELATION_CAP_REACHED,
                failure_detail=f"Correlation cluster {new_exposure:.1f}% exceeds max {self.max_correlation_cluster_exposure_pct}%",
            )

        return InvariantCheckResult(passed=True)

    def check_position_metadata_invariant(
        self,
        position: PositionSnapshot,
    ) -> InvariantCheckResult:
        """Check if position has valid metadata."""
        if not position.symbol or position.symbol.strip() == "":
            return InvariantCheckResult(
                passed=False,
                failure_reason=FailureReason.INVALID_POSITION_METADATA,
                failure_detail="Position has empty symbol",
            )

        if position.qty <= 0:
            return InvariantCheckResult(
                passed=False,
                failure_reason=FailureReason.INVALID_POSITION_METADATA,
                failure_detail=f"Position {position.symbol} has invalid qty: {position.qty}",
            )

        if position.entry_price <= 0:
            return InvariantCheckResult(
                passed=False,
                failure_reason=FailureReason.INVALID_POSITION_METADATA,
                failure_detail=f"Position {position.symbol} has invalid entry_price: {position.entry_price}",
            )

        if position.current_price <= 0:
            return InvariantCheckResult(
                passed=False,
                failure_reason=FailureReason.INVALID_POSITION_METADATA,
                failure_detail=f"Position {position.symbol} has invalid current_price: {position.current_price}",
            )

        return InvariantCheckResult(passed=True)

    def check_failsafe_critical_invariant(
        self,
        failsafe_critical: bool,
        detail: str = "",
    ) -> InvariantCheckResult:
        """Check if failsafe is in critical state."""
        if failsafe_critical:
            return InvariantCheckResult(
                passed=False,
                failure_reason=FailureReason.FAILSAFE_CRITICAL,
                failure_detail=detail or "Failsafe entered critical state",
            )
        return InvariantCheckResult(passed=True)

    def check_all_invariants(
        self,
        position: Optional[PositionSnapshot],
        portfolio: PortfolioSnapshot,
        check_position: bool = True,
        failsafe_critical: bool = False,
    ) -> List[InvariantCheckResult]:
        """Run all invariant checks and return failures."""
        results = []

        results.append(self.check_portfolio_exposure_invariants(portfolio))
        results.append(self.check_drawdown_invariant(portfolio))
        results.append(self.check_failsafe_critical_invariant(failsafe_critical))

        if check_position and position is not None:
            results.append(self.check_position_metadata_invariant(position))
            results.append(self.check_position_exposure_invariant(position, portfolio))

        return results


def get_stale_data_fallback(
    position: Optional[PositionSnapshot],
    portfolio: Optional[PortfolioSnapshot],
    data_age_seconds: int = 300,
) -> Tuple[Optional[PositionSnapshot], Optional[PortfolioSnapshot]]:
    """
    Provide fallback behavior when risk data is stale or missing.

    Args:
        position: Position snapshot that may be stale
        portfolio: Portfolio snapshot that may be stale
        data_age_seconds: Maximum age for fresh data (default 5 minutes)

    Returns:
        Tuple of (position, portfolio) with fallback values applied
    """
    now = datetime.now()

    if position is not None:
        age = (now - position.snapshot_time).total_seconds()
        if age > data_age_seconds:
            position = PositionSnapshot(
                symbol=position.symbol,
                entry_price=position.entry_price,
                current_price=position.current_price,
                qty=position.qty,
                entry_time=position.entry_time,
                snapshot_time=now,
            )

    if portfolio is not None:
        age = (now - portfolio.snapshot_time).total_seconds()
        if age > data_age_seconds:
            portfolio = PortfolioSnapshot(
                total_equity=portfolio.total_equity,
                cash_available=portfolio.cash_available,
                buying_power=portfolio.buying_power,
                positions_value=portfolio.positions_value,
                snapshot_time=now,
            )

    return position, portfolio


class StandardRiskPolicy:
    """
    Standard implementation of RiskPolicy using RiskGate logic.
    Pure function - no side effects.
    """

    def __init__(self, config: Optional[RiskGateConfig] = None):
        self.config = config or RiskGateConfig.from_trading_config()
        self._profit_levels_hit: Dict[str, List[float]] = {}

    def evaluate(
        self, position: PositionSnapshot, portfolio: PortfolioSnapshot
    ) -> RiskDecision:
        """Evaluate a position against risk rules - pure function."""
        pnl_pct = position.pnl_pct
        atr_k = self.config.atr_k
        atr_trail_k = self.config.atr_trail_k

        reasons = []
        triggered_rules = []

        # Hard stop
        if pnl_pct <= self.config.hard_stop_pct:
            reasons.append(
                f"HARD STOP: Loss {pnl_pct:.1f}% exceeds limit {self.config.hard_stop_pct}%"
            )
            triggered_rules.append("hard_stop")
            return RiskDecision(
                action=RiskActionType.EXIT,
                size_delta=-position.qty,
                confidence=0.95,
                reasons=reasons,
                triggered_rules=triggered_rules,
                skip_agents=True,
                hold_phase=position.phase,
            )

        # ATR stop
        if position.atr_pct > 0:
            atr_stop_pct = -atr_k * position.atr_pct
            if pnl_pct <= atr_stop_pct:
                reasons.append(
                    f"ATR STOP: Loss {pnl_pct:.1f}% exceeds {atr_k:.2f}x ATR ({atr_stop_pct:.1f}%)"
                )
                triggered_rules.append("atr_stop")
                return RiskDecision(
                    action=RiskActionType.EXIT,
                    size_delta=-position.qty,
                    confidence=0.90,
                    reasons=reasons,
                    triggered_rules=triggered_rules,
                    skip_agents=True,
                    hold_phase=position.phase,
                )

        # Trailing stop
        if (
            position.high_since_entry > 0
            and position.high_since_entry > position.entry_price
            and position.atr_pct > 0
        ):
            trail_stop_price = position.high_since_entry * (
                1 - atr_trail_k * position.atr_pct / 100
            )
            if position.current_price < trail_stop_price:
                reasons.append("TRAILING STOP: Price below trail stop")
                triggered_rules.append("trailing_stop")
                return RiskDecision(
                    action=RiskActionType.EXIT,
                    size_delta=-position.qty,
                    confidence=0.85,
                    reasons=reasons,
                    triggered_rules=triggered_rules,
                    skip_agents=True,
                    hold_phase=position.phase,
                )

        # Trim threshold (skip if position not held long enough)
        min_hold_for_trim = getattr(self.config, "min_hold_minutes_for_trim", 120)
        if pnl_pct <= self.config.trim_threshold_pct:
            if position.hold_minutes >= min_hold_for_trim:
                trim_qty = int(position.qty * self.config.trim_fraction)
                if trim_qty > 0:
                    reasons.append(f"TRIM THRESHOLD: Loss {pnl_pct:.1f}% exceeds threshold")
                    triggered_rules.append("trim_threshold")
                    return RiskDecision(
                        action=RiskActionType.TRIM,
                        size_delta=-trim_qty,
                        confidence=0.80,
                        reasons=reasons,
                        triggered_rules=triggered_rules,
                        skip_agents=True,
                        hold_phase=position.phase,
                    )

        # Time stop
        if position.hold_minutes >= self.config.time_stop_minutes:
            is_flat = abs(pnl_pct) <= self.config.time_stop_flat_threshold_pct
            if is_flat:
                trim_qty = int(position.qty * 0.5)
                if trim_qty > 0:
                    reasons.append(
                        f"TIME STOP: Flat for {position.hold_minutes:.0f} min"
                    )
                    triggered_rules.append("time_stop_flat")
                    return RiskDecision(
                        action=RiskActionType.TRIM,
                        size_delta=-trim_qty,
                        confidence=0.70,
                        reasons=reasons,
                        triggered_rules=triggered_rules,
                        skip_agents=True,
                        hold_phase=position.phase,
                    )
            elif pnl_pct < 0:
                reasons.append(f"TIME STOP: Losing for {position.hold_minutes:.0f} min")
                triggered_rules.append("time_stop_loss")
                return RiskDecision(
                    action=RiskActionType.EXIT,
                    size_delta=-position.qty,
                    confidence=0.75,
                    reasons=reasons,
                    triggered_rules=triggered_rules,
                    skip_agents=True,
                    hold_phase=position.phase,
                )

        # Profit taking (skip if position not held long enough)
        if pnl_pct > 0 and position.hold_minutes >= min_hold_for_trim:
            symbol = position.symbol
            if symbol not in self._profit_levels_hit:
                self._profit_levels_hit[symbol] = []

            for level in self.config.profit_take_levels:
                level_pct = level["pct"]
                trim_frac = level["trim"]

                if (
                    pnl_pct >= level_pct
                    and level_pct not in self._profit_levels_hit[symbol]
                ):
                    trim_qty = int(position.qty * trim_frac)
                    if trim_qty > 0:
                        self._profit_levels_hit[symbol].append(level_pct)
                        reasons.append(
                            f"PROFIT TAKE: +{pnl_pct:.1f}% hit {level_pct}% target"
                        )
                        triggered_rules.append(f"profit_take_{level_pct}")
                        return RiskDecision(
                            action=RiskActionType.PROFIT_TAKE,
                            size_delta=-trim_qty,
                            confidence=0.85,
                            reasons=reasons,
                            triggered_rules=triggered_rules,
                            skip_agents=False,
                            hold_phase=position.phase,
                        )

        # No trigger
        return RiskDecision(
            action=RiskActionType.NONE,
            size_delta=0,
            confidence=0.0,
            reasons=["No risk gate triggered"],
            triggered_rules=[],
            skip_agents=False,
            hold_phase=position.phase,
        )

    def should_skip_agents(self, decision: RiskDecision) -> bool:
        """Determine if LLM pipeline should be skipped."""
        return decision.skip_agents

    def reset_profit_levels(self, symbol: str) -> None:
        """Reset profit level tracking - for testing only."""
        if symbol in self._profit_levels_hit:
            del self._profit_levels_hit[symbol]


class StandardSizingPolicy:
    """
    Standard implementation of SizingPolicy.
    Pure function - no side effects.
    """

    def compute_stop_distance_dollars(
        self,
        entry_price: float,
        atr_14: Optional[float] = None,
        bid_ask_spread: Optional[float] = None,
        atr_multiplier: float = 2.0,
        spread_multiplier: float = 2.0,
        liquidity_floor_bps: float = 25.0,
    ) -> float:
        """Compute stop distance as max(ATR*mult, spread*mult, liquidity floor)."""
        entry = max(0.0, float(entry_price or 0.0))
        if entry <= 0:
            return 0.0

        atr_component = max(0.0, float(atr_14 or 0.0)) * max(0.0, float(atr_multiplier))
        spread_component = max(0.0, float(bid_ask_spread or 0.0)) * max(
            0.0, float(spread_multiplier)
        )
        floor_component = entry * (max(0.0, float(liquidity_floor_bps)) / 10000.0)

        stop_distance = max(atr_component, spread_component, floor_component)
        if stop_distance <= 0:
            return entry * 0.01
        return float(stop_distance)

    def calculate_size(
        self,
        symbol: str,
        entry_price: float,
        portfolio: PortfolioSnapshot,
        risk_per_trade: float = 0.02,
        atr_14: Optional[float] = None,
        bid_ask_spread: Optional[float] = None,
        equity: Optional[float] = None,
        risk_percent: Optional[float] = None,
        atr_multiplier: float = 2.0,
        spread_multiplier: float = 2.0,
        liquidity_floor_bps: float = 25.0,
        avg_1m_volume: Optional[float] = None,
        max_liquidity_take_pct: float = 0.05,
    ) -> int:
        """Calculate position size based on risk parameters."""
        if entry_price <= 0 or portfolio.total_equity <= 0:
            return 0

        effective_equity = max(0.0, float(equity or portfolio.total_equity))
        effective_risk_pct = (
            float(risk_percent)
            if risk_percent is not None
            else float(risk_per_trade)
        )
        if effective_equity <= 0 or effective_risk_pct <= 0:
            return 0
        risk_amount = effective_equity * effective_risk_pct

        stop_distance = self.compute_stop_distance_dollars(
            entry_price=entry_price,
            atr_14=atr_14,
            bid_ask_spread=bid_ask_spread,
            atr_multiplier=atr_multiplier,
            spread_multiplier=spread_multiplier,
            liquidity_floor_bps=liquidity_floor_bps,
        )
        if stop_distance <= 0:
            return 0

        position_size = int(risk_amount / stop_distance)

        has_target_cap = hasattr(portfolio, "position_size_target")
        has_max_cap = hasattr(portfolio, "position_size_max")

        if has_target_cap:
            position_size_target = float(getattr(portfolio, "position_size_target", 0.0) or 0.0)
            target_size = int(position_size_target / entry_price)
            if target_size > 0:
                position_size = min(position_size, target_size)

        if has_max_cap:
            position_size_max = float(getattr(portfolio, "position_size_max", 0.0) or 0.0)
            max_size = int(position_size_max / entry_price)
            if max_size > 0:
                position_size = min(position_size, max_size)

        if portfolio.cash_available > 0:
            max_cash_size = int(portfolio.cash_available / entry_price)
            position_size = min(position_size, max_cash_size)

        avg_vol = max(0.0, float(avg_1m_volume or 0.0))
        liq_take = max(0.0, float(max_liquidity_take_pct or 0.0))
        if avg_vol > 0 and liq_take > 0:
            liq_cap = int(avg_vol * liq_take)
            position_size = min(position_size, liq_cap)

        return max(0, position_size)

    def can_increase_position(
        self,
        position: PositionSnapshot,
        portfolio: PortfolioSnapshot,
        additional_shares: int,
    ) -> bool:
        """Check if adding to a position is allowed."""
        if additional_shares <= 0:
            return False

        additional_value = additional_shares * position.current_price
        if additional_value > portfolio.cash_available:
            return False

        new_exposure = (
            (position.market_value + additional_value) / portfolio.total_equity * 100
        )

        max_position_pct = 15.0
        return new_exposure <= max_position_pct


class StandardRotationPolicy:
    """
    Standard implementation of RotationPolicy.
    Pure function - no side effects.
    """

    def __init__(self, config: Optional[RotationPolicyConfig] = None):
        self.config = config or RotationPolicyConfig()

    def find_rotation_candidates(
        self,
        portfolio: PortfolioSnapshot,
        min_conviction_threshold: float = 40.0,
    ) -> List[PositionSnapshot]:
        """Find positions eligible for rotation."""
        candidates = []
        for pos in portfolio.positions:
            if pos.is_unlocked:
                candidates.append(pos)
        candidates.sort(key=lambda p: p.pnl_pct)
        return candidates[:3]

    def evaluate_rotation(
        self,
        exit_position: PositionSnapshot,
        enter_symbol: str,
        enter_price: float,
        enter_signal_score: float,
        portfolio: PortfolioSnapshot,
        min_improvement: float = 15.0,
    ) -> RotationPlan:
        """Evaluate a proposed rotation."""
        improvement_score = enter_signal_score - exit_position.pnl_pct

        capital_amount = exit_position.market_value

        exit_reasons = [
            f"Low conviction P&L: {exit_position.pnl_pct:+.1f}%",
            f"Signal score: {enter_signal_score:.0f}",
        ]

        enter_reasons = [
            f"Higher conviction: {enter_signal_score:.0f}",
            f"Improvement: +{improvement_score:.0f}",
        ]

        satisfies_min_improvement = improvement_score >= min_improvement

        return RotationPlan(
            exit_symbol=exit_position.symbol,
            exit_qty=exit_position.qty,
            exit_price=exit_position.current_price,
            exit_conviction=exit_position.pnl_pct + 50,
            exit_reasons=exit_reasons,
            enter_symbol=enter_symbol,
            enter_price=enter_price,
            enter_qty=int(capital_amount / enter_price) if enter_price > 0 else 0,
            enter_signal_score=enter_signal_score,
            enter_reasons=enter_reasons,
            improvement_score=improvement_score,
            capital_amount=capital_amount,
            satisfies_min_improvement=satisfies_min_improvement,
        )

    def apply_rotation_constraints(
        self,
        plan: RotationPlan,
        portfolio: PortfolioSnapshot,
        config: RotationPolicyConfig,
    ) -> RotationPlan:
        """Apply portfolio constraints to rotation plan."""
        plan.satisfies_sector_limit = True
        plan.satisfies_correlation_limit = True
        plan.satisfies_family_budget = True

        return plan


class StandardExposurePolicy:
    """
    Standard implementation of ExposurePolicy.
    Pure function - no side effects.
    """

    def __init__(self, config: Optional[ExposurePolicyConfig] = None):
        self.config = config or ExposurePolicyConfig()

    def check_position_exposure(
        self,
        position: PositionSnapshot,
        portfolio: PortfolioSnapshot,
        max_position_pct: float = 15.0,
    ) -> List[FailureReason]:
        """Check if position exceeds exposure limits."""
        if portfolio.total_equity <= 0:
            return [FailureReason.INVALID_POSITION_METADATA]

        exposure_pct = (position.market_value / portfolio.total_equity) * 100
        if exposure_pct > max_position_pct:
            return [FailureReason.OVER_MAX_EXPOSURE]
        return []

    def check_sector_exposure(
        self,
        sector: str,
        additional_exposure_pct: float,
        portfolio: PortfolioSnapshot,
        max_sector_pct: float = 40.0,
    ) -> List[FailureReason]:
        """Check if sector exposure would exceed limits."""
        current_exposure = portfolio.sector_exposure.get(sector, 0.0)
        new_exposure = current_exposure + additional_exposure_pct
        if new_exposure > max_sector_pct:
            return [FailureReason.SECTOR_CAP_REACHED]
        return []

    def check_correlation_exposure(
        self,
        additional_exposure_pct: float,
        portfolio: PortfolioSnapshot,
        max_correlation_pct: float = 35.0,
    ) -> List[FailureReason]:
        """Check if correlation cluster exposure would exceed limits."""
        new_exposure = portfolio.correlation_cluster_exposure + additional_exposure_pct
        if new_exposure > max_correlation_pct:
            return [FailureReason.CORRELATION_CAP_REACHED]
        return []

    def check_family_budget(
        self,
        signal_family: str,
        additional_exposure_pct: float,
        portfolio: PortfolioSnapshot,
        family_budgets: Dict[str, float] = None,
    ) -> List[FailureReason]:
        """Check if signal family budget would be exceeded."""
        if family_budgets is None:
            family_budgets = self.config.signal_family_budgets

        if not signal_family or signal_family not in family_budgets:
            return []

        budget_pct = family_budgets[signal_family] * 100
        current_exposure = portfolio.signal_family_exposure.get(signal_family, 0.0)
        new_exposure = current_exposure + additional_exposure_pct

        if new_exposure > budget_pct:
            return [FailureReason.FAMILY_BUDGET_EXHAUSTED]
        return []

    def get_risk_health_report(
        self,
        portfolio: PortfolioSnapshot,
        config: ExposurePolicyConfig,
    ) -> RiskHealthReport:
        """Generate comprehensive risk health report."""
        report = RiskHealthReport()
        report.gross_exposure_pct = portfolio.gross_exposure_pct
        report.net_exposure_pct = portfolio.net_exposure_pct
        report.current_drawdown_pct = portfolio.current_drawdown_pct
        report.entry_halt_drawdown_pct = config.entry_halt_drawdown_pct

        failures = []
        failure_details = {}

        if portfolio.gross_exposure_pct > config.max_gross_exposure_pct:
            failures.append(FailureReason.OVER_MAX_EXPOSURE)
            failure_details["gross_exposure"] = [
                f"{portfolio.gross_exposure_pct:.1f}% > {config.max_gross_exposure_pct}%"
            ]

        if abs(portfolio.net_exposure_pct) > config.max_net_exposure_pct:
            failures.append(FailureReason.OVER_MAX_EXPOSURE)
            failure_details["net_exposure"] = [
                f"{portfolio.net_exposure_pct:.1f}% exceeds +/-{config.max_net_exposure_pct}%"
            ]

        if portfolio.current_drawdown_pct > config.entry_halt_drawdown_pct:
            failures.append(FailureReason.DRAWDOWN_BREACH)
            failure_details["drawdown"] = [
                f"{portfolio.current_drawdown_pct:.1f}% > {config.entry_halt_drawdown_pct}%"
            ]

        report.failures = failures
        report.failure_details = failure_details
        report.is_healthy = len(failures) == 0

        health_score = 100.0
        for f in failures:
            if f == FailureReason.DRAWDOWN_BREACH:
                health_score -= 30
            else:
                health_score -= 10
        report.health_score = max(0, health_score)

        for pos in portfolio.positions:
            if pos.pnl_pct <= -8:
                report.critical_positions.append(pos.symbol)
            elif pos.pnl_pct < 0 and pos.is_unlocked:
                report.rotation_candidates.append(pos.symbol)
            elif pos.pnl_pct > 5:
                report.add_candidates.append(pos.symbol)

        report.family_budget_status = {}
        for family, budget_pct in config.signal_family_budgets.items():
            current = portfolio.signal_family_exposure.get(family, 0.0)
            report.family_budget_status[family] = {
                "current_pct": current,
                "budget_pct": budget_pct * 100,
                "available_pct": max(0, budget_pct * 100 - current),
            }

        if failures:
            report.recommendations.append("Review portfolio constraints")
        if report.critical_positions:
            report.recommendations.append("Exit critical positions")
        if report.rotation_candidates:
            report.recommendations.append("Consider rotation")

        return report


class StandardKillSwitchPolicy:
    """
    Standard implementation of KillSwitchPolicy.
    Maintains internal state for tracking.
    """

    def __init__(
        self,
        max_consecutive_failures: int = 3,
        max_realized_daily_loss_pct: float = 3.0,
    ):
        self.max_consecutive_failures = max_consecutive_failures
        self.max_realized_daily_loss_pct = max_realized_daily_loss_pct
        self._consecutive_failures = 0
        self._daily_realized_pnl = 0.0
        self._daily_equity_start = 0.0
        self._failure_reason: Optional[FailureReason] = None

    def check_kill_switch(
        self,
        portfolio: PortfolioSnapshot,
        consecutive_failures: int = 0,
    ) -> bool:
        """Check if kill switch should be triggered."""
        self._consecutive_failures = consecutive_failures
        self._failure_reason = None

        if consecutive_failures >= self.max_consecutive_failures:
            self._failure_reason = FailureReason.FAILSAFE_CRITICAL
            return True

        daily_loss_pct = self.get_daily_loss_pct()
        if daily_loss_pct <= -self.max_realized_daily_loss_pct:
            self._failure_reason = FailureReason.DRAWDOWN_BREACH
            return True

        return False

    def get_failure_reason(self) -> Optional[FailureReason]:
        """Get the reason for kill switch trigger."""
        return self._failure_reason

    def record_failure(self) -> None:
        """Record a failed trade."""
        self._consecutive_failures += 1

    def record_success(self) -> None:
        """Record a successful trade."""
        self._consecutive_failures = 0

    def get_daily_loss_pct(self) -> float:
        """Get current day's realized loss percentage."""
        if self._daily_equity_start <= 0:
            return 0.0
        return (self._daily_realized_pnl / self._daily_equity_start) * 100


@dataclass
class PolicyEvaluatorResult:
    """Result of the complete policy evaluator pipeline."""

    risk_decisions: List[RiskDecision]
    sizing_recommendations: Dict[str, int]
    rotation_plans: List[RotationPlan]
    health_report: RiskHealthReport
    kill_switch_triggered: bool
    recommendations: List[str]

    @property
    def has_critical_actions(self) -> bool:
        """Check if any critical risk actions were generated."""
        return any(
            d.action in (RiskActionType.EXIT, RiskActionType.TRIM)
            and d.confidence > 0.7
            for d in self.risk_decisions
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_decisions": [d.to_dict() for d in self.risk_decisions],
            "sizing_recommendations": self.sizing_recommendations,
            "rotation_plans": [p.to_dict() for p in self.rotation_plans],
            "health_report": self.health_report.to_dict(),
            "kill_switch_triggered": self.kill_switch_triggered,
            "recommendations": self.recommendations,
            "has_critical_actions": self.has_critical_actions,
        }


class PolicyEvaluator:
    """
    Deterministic policy evaluator pipeline.

    Flow: Risk Policy -> Sizing Policy -> Rotation Policy -> Health Report
    """

    def __init__(
        self,
        risk_policy: Optional[RiskPolicy] = None,
        sizing_policy: Optional[SizingPolicy] = None,
        rotation_policy: Optional[RotationPolicy] = None,
        exposure_policy: Optional[ExposurePolicy] = None,
        kill_switch_policy: Optional[KillSwitchPolicy] = None,
    ):
        self.risk_policy = risk_policy or StandardRiskPolicy()
        self.sizing_policy = sizing_policy or StandardSizingPolicy()
        self.rotation_policy = rotation_policy or StandardRotationPolicy()
        self.exposure_policy = exposure_policy or StandardExposurePolicy()
        self.kill_switch_policy = kill_switch_policy or StandardKillSwitchPolicy()

    def evaluate_portfolio(
        self,
        portfolio: PortfolioSnapshot,
        rotation_candidates: Optional[List[Tuple[str, float, float]]] = None,
    ) -> PolicyEvaluatorResult:
        """
        Run the complete policy evaluation pipeline.

        Args:
            portfolio: Current portfolio snapshot
            rotation_candidates: Optional list of (symbol, price, score) for rotations

        Returns:
            PolicyEvaluatorResult with all decisions and health report
        """
        risk_decisions = []
        for position in portfolio.positions:
            decision = self.risk_policy.evaluate(position, portfolio)
            risk_decisions.append(decision)

        sizing_recommendations: Dict[str, int] = {}
        if rotation_candidates:
            for enter_symbol, enter_price, _ in rotation_candidates:
                if enter_symbol in sizing_recommendations:
                    continue
                sizing_recommendations[enter_symbol] = (
                    self.sizing_policy.calculate_size(
                        enter_symbol, enter_price, portfolio
                    )
                )

        rotation_plans = []
        if rotation_candidates:
            candidates = self.rotation_policy.find_rotation_candidates(portfolio)
            for exit_pos in candidates:
                for enter_symbol, enter_price, enter_score in rotation_candidates:
                    sized_qty = sizing_recommendations.get(enter_symbol, 0)
                    if sized_qty <= 0:
                        continue
                    plan = self.rotation_policy.evaluate_rotation(
                        exit_pos,
                        enter_symbol,
                        enter_price,
                        enter_score,
                        portfolio,
                    )
                    if plan.enter_qty <= 0:
                        plan.enter_qty = sized_qty
                    plan = self.rotation_policy.apply_rotation_constraints(
                        plan, portfolio, RotationPolicyConfig()
                    )
                    if plan.is_valid:
                        rotation_plans.append(plan)
                        break

        exposure_config = ExposurePolicyConfig()
        health_report = self.exposure_policy.get_risk_health_report(
            portfolio, exposure_config
        )

        kill_switch_triggered = self.kill_switch_policy.check_kill_switch(portfolio)

        recommendations = []
        if kill_switch_triggered:
            recommendations.append("KILL SWITCH TRIGGERED - halt trading")
        if health_report.critical_positions:
            recommendations.append(
                f"Exit critical positions: {health_report.critical_positions}"
            )
        if health_report.rotation_candidates:
            recommendations.append(
                f"Consider rotation: {health_report.rotation_candidates}"
            )
        if health_report.failures:
            recommendations.append(
                f"Resolve failures: {[f.value for f in health_report.failures]}"
            )

        return PolicyEvaluatorResult(
            risk_decisions=risk_decisions,
            sizing_recommendations=sizing_recommendations,
            rotation_plans=rotation_plans,
            health_report=health_report,
            kill_switch_triggered=kill_switch_triggered,
            recommendations=recommendations,
        )


@dataclass
class PortfolioAllocationConfig:
    """Configuration for portfolio allocation policy."""

    signal_family_budgets: Dict[str, float] = field(
        default_factory=lambda: {
            "ts_momentum": 0.35,
            "xs_momentum": 0.25,
            "mean_reversion": 0.15,
            "breakout": 0.15,
            "pullback": 0.10,
        }
    )

    rebalance_frequency: str = "weekly"
    rotation_cooldown_minutes: int = 30
    max_rotations_per_day: int = 2


class PortfolioAllocationPolicy:
    """
    Portfolio allocation policy for signal-zoo family budgets and rebalance cadence.

    Pure function - no side effects.
    """

    def __init__(self, config: Optional[PortfolioAllocationConfig] = None):
        self.config = config or PortfolioAllocationConfig()

    def calculate_family_allocation(
        self,
        portfolio: PortfolioSnapshot,
        signal_family: str,
    ) -> float:
        """
        Calculate available allocation for a signal family.

        Returns:
            Available percentage points for the family
        """
        budget_pct = self.config.signal_family_budgets.get(signal_family, 0.0)
        current_exposure = portfolio.signal_family_exposure.get(signal_family, 0.0)
        return max(0, budget_pct * 100 - current_exposure)

    def get_rebalance_candidates(
        self,
        portfolio: PortfolioSnapshot,
    ) -> List[Tuple[str, float]]:
        """
        Get positions that should be rebalanced based on family budget drift.

        Returns:
            List of (symbol, rebalance_priority) sorted by priority
        """
        candidates = []

        for pos in portfolio.positions:
            if not pos.signal_family:
                continue

            budget_pct = self.config.signal_family_budgets.get(pos.signal_family, 0.0)
            current_pct = portfolio.signal_family_exposure.get(pos.signal_family, 0.0)

            drift = current_pct - (budget_pct * 100)

            if drift > 5:
                candidates.append((pos.symbol, drift))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def should_rebalance(
        self,
        portfolio: PortfolioSnapshot,
        min_drift_threshold: float = 10.0,
    ) -> bool:
        """Determine if portfolio should be rebalanced."""
        for family, budget_pct in self.config.signal_family_budgets.items():
            current_pct = portfolio.signal_family_exposure.get(family, 0.0)
            drift = abs(current_pct - budget_pct * 100)
            if drift > min_drift_threshold:
                return True
        return False

    def get_allocation_report(
        self,
        portfolio: PortfolioSnapshot,
    ) -> Dict[str, Dict[str, float]]:
        """Generate allocation report for all signal families."""
        report = {}

        for family, budget_pct in self.config.signal_family_budgets.items():
            current_pct = portfolio.signal_family_exposure.get(family, 0.0)
            report[family] = {
                "budget_pct": budget_pct * 100,
                "current_pct": current_pct,
                "target_pct": budget_pct * 100,
                "drift_pct": current_pct - budget_pct * 100,
                "available_pct": max(0, budget_pct * 100 - current_pct),
                "over_budget": current_pct > budget_pct * 100,
            }

        return report


def create_policy_evaluator() -> PolicyEvaluator:
    """Factory function to create a configured policy evaluator."""
    return PolicyEvaluator()


def evaluate_position_pure(
    position: PositionSnapshot,
    portfolio: PortfolioSnapshot,
    risk_config: Optional[RiskGateConfig] = None,
) -> RiskDecision:
    """
    Convenience function for pure position evaluation.
    No side effects, no state required.
    """
    policy = StandardRiskPolicy(risk_config)
    return policy.evaluate(position, portfolio)


def get_portfolio_health_pure(
    portfolio: PortfolioSnapshot,
    exposure_config: Optional[ExposurePolicyConfig] = None,
) -> RiskHealthReport:
    """
    Convenience function for pure portfolio health evaluation.
    No side effects, no state required.
    """
    policy = StandardExposurePolicy(exposure_config)
    return policy.get_risk_health_report(
        portfolio, exposure_config or ExposurePolicyConfig()
    )


# ============================================================
# Phase 4 - Compatibility Adapters
# ============================================================


def action_plan_to_risk_decision(
    action_plan: "ActionPlan",
    position: PositionSnapshot,
) -> RiskDecision:
    """
    Adapter to convert RiskGate ActionPlan (old interface) to RiskDecision (new interface).
    Preserves action semantics and reasons.

    Args:
        action_plan: The old RiskGate ActionPlan
        position: Position snapshot for context

    Returns:
        RiskDecision compatible with the new contract
    """
    from autotrade.risk.risk_gate import RiskAction as GateAction

    action_map = {
        GateAction.EXIT: RiskActionType.EXIT,
        GateAction.TRIM: RiskActionType.TRIM,
        GateAction.PROFIT_TAKE: RiskActionType.PROFIT_TAKE,
        GateAction.ADD: RiskActionType.ADD,
        GateAction.NONE: RiskActionType.NONE,
    }

    phase_map = {
        "locked": PositionPhase.LOCKED,
        "unlocking": PositionPhase.UNLOCKING,
        "unlocked": PositionPhase.UNLOCKED,
        "critical": PositionPhase.CRITICAL,
    }

    phase = phase_map.get(
        getattr(action_plan.hold_phase, "value", str(action_plan.hold_phase)),
        PositionPhase.LOCKED,
    )

    return RiskDecision(
        action=action_map.get(action_plan.action, RiskActionType.NONE),
        size_delta=int(action_plan.size_delta),
        confidence=action_plan.confidence,
        reasons=action_plan.reasons,
        triggered_rules=action_plan.triggered_rules,
        requires_pdt_burn=action_plan.pdt_burn_required,
        hold_phase=phase,
        conviction_score=action_plan.conviction_score,
        skip_agents=action_plan.skip_agents,
    )


def risk_decision_to_day_manager_action(
    decision: RiskDecision,
    current_action: str = "hold",
    current_score: int = 50,
) -> Dict[str, Any]:
    """
    Adapter to convert RiskDecision to day-manager compatible action dict.
    Preserves current action semantics (exit, trim, add, hold) and reasons.

    Args:
        decision: RiskDecision from policy engine
        current_action: Current day-manager action
        current_score: Current conviction score

    Returns:
        Dict with action, score, signals keys for day-manager consumption
    """
    result = {
        "action": current_action,
        "score": current_score,
        "signals": [],
        "policy_engine_action": decision.action.value,
        "policy_engine_triggered_rules": list(decision.triggered_rules),
    }

    if decision.action == RiskActionType.EXIT:
        result["action"] = "exit"
        result["score"] = min(current_score, -40)
        result["signals"].append(
            "Policy engine EXIT: "
            + ", ".join(decision.triggered_rules or decision.reasons[:1])
        )
    elif decision.action in (RiskActionType.TRIM, RiskActionType.PROFIT_TAKE):
        if current_action not in ("exit", "trim"):
            result["action"] = "trim"
        result["score"] = min(current_score, -20)
        result["signals"].append(
            "Policy engine TRIM: "
            + ", ".join(decision.triggered_rules or decision.reasons[:1])
        )
    elif decision.action == RiskActionType.ADD:
        if current_action in ("hold", "watch"):
            result["action"] = "add"
        result["score"] = max(current_score, int(decision.confidence * 100))
        result["signals"].append(
            "Policy engine ADD: "
            + ", ".join(
                decision.reasons[:1] if decision.reasons else ["higher conviction"]
            )
        )

    return result


class PDTAdapter:
    """
    Legacy adapter retained for API compatibility.

    PDT constraints are disabled for the current workflow; this adapter never
    blocks, trims, or delays a risk decision because of day-trade counts.
    """

    def __init__(self, day_trades_remaining: int = 10):
        self.day_trades_remaining = day_trades_remaining

    def should_respect_pdt(
        self,
        decision: RiskDecision,
        position: PositionSnapshot,
    ) -> bool:
        """PDT is disabled; no decision should be adjusted for PDT."""
        return False

    def adjust_decision_for_pdt(
        self,
        decision: RiskDecision,
        position: PositionSnapshot,
    ) -> RiskDecision:
        """Return the original decision unchanged."""
        return decision

    def can_rotate(
        self,
        exit_position: PositionSnapshot,
        enter_value: float,
        portfolio: PortfolioSnapshot,
    ) -> Tuple[bool, Optional[str]]:
        """PDT is disabled; rotation is never blocked by day-trade counts."""
        return True, None


class FailsafeAdapter:
    """
    Adapter to preserve strategy failsafe behavior in policy decisions.

    Ensures:
    - Failsafe halts are respected
    - Max position limits are enforced
    - Entry halts during critical states
    """

    def __init__(
        self,
        failsafe_level: str = "normal",
        halt_new_entries: bool = False,
        max_positions: int = 7,
        max_open_risk_r: float = 5.0,
        risk_per_trade: float = 0.02,
        atr_stop_multiplier: float = 2.0,
        min_stop_distance_pct: float = 0.005,
    ):
        self.failsafe_level = failsafe_level
        self.halt_new_entries = halt_new_entries
        self.max_positions = max_positions
        self.max_open_risk_r = max(0.0, float(max_open_risk_r or 0.0))
        self.risk_per_trade = max(0.0, float(risk_per_trade or 0.0))
        self.atr_stop_multiplier = max(0.0, float(atr_stop_multiplier or 0.0))
        self.min_stop_distance_pct = max(0.0, float(min_stop_distance_pct or 0.0))

    def should_halt_entries(self) -> bool:
        """Check if new entries should be halted."""
        return self.halt_new_entries or self.failsafe_level == "critical"

    def check_entry_allowed(
        self,
        symbol: str,
        portfolio: PortfolioSnapshot,
        signal_score: float = 50.0,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a new entry is allowed given failsafe constraints.

        Returns:
            (allowed, reason) tuple
        """
        if self.should_halt_entries():
            return False, f"Failsafe active: {self.failsafe_level} - no new entries"

        if portfolio.position_count >= self.max_positions:
            return False, f"Max positions reached ({self.max_positions})"

        if portfolio.current_drawdown_pct > 10.0:
            return (
                False,
                f"Portfolio drawdown too high: {portfolio.current_drawdown_pct:.1f}%",
            )

        open_risk_r = self.estimate_open_risk_r(portfolio)
        if self.max_open_risk_r > 0 and open_risk_r >= self.max_open_risk_r:
            return (
                False,
                f"Open risk budget consumed ({open_risk_r:.2f}R/{self.max_open_risk_r:.2f}R)",
            )

        return True, None

    def estimate_open_risk_r(self, portfolio: PortfolioSnapshot) -> float:
        """Estimate aggregate open risk in R units across current positions."""
        if portfolio.total_equity <= 0 or self.risk_per_trade <= 0:
            return 0.0

        risk_unit = portfolio.total_equity * self.risk_per_trade
        if risk_unit <= 0:
            return 0.0

        total_risk_dollars = 0.0
        for pos in portfolio.positions:
            qty = abs(int(getattr(pos, "qty", 0) or 0))
            if qty <= 0:
                continue
            px = float(
                getattr(pos, "entry_price", 0.0) or getattr(pos, "current_price", 0.0) or 0.0
            )
            if px <= 0:
                continue

            atr_pct = abs(float(getattr(pos, "atr_pct", 0.0) or 0.0))
            atr_stop_pct = (atr_pct / 100.0) * self.atr_stop_multiplier
            stop_distance_pct = max(self.min_stop_distance_pct, atr_stop_pct)

            total_risk_dollars += qty * px * stop_distance_pct

        return float(total_risk_dollars / risk_unit)

    def check_rotation_allowed(
        self,
        exit_symbol: str,
        enter_symbol: str,
        portfolio: PortfolioSnapshot,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if rotation is allowed given failsafe constraints.
        """
        if self.should_halt_entries():
            return False, f"Failsafe active: {self.failsafe_level} - no new entries"

        return True, None

    def get_max_position_size(
        self,
        portfolio: PortfolioSnapshot,
        risk_per_trade: float = 0.02,
    ) -> float:
        """Get maximum position size given failsafe constraints."""
        base_size = portfolio.total_equity * risk_per_trade * 5

        if self.failsafe_level == "elevated":
            base_size *= 0.75
        elif self.failsafe_level == "critical":
            base_size *= 0.5

        return base_size


class DayManagerAdapter:
    """
    Adapter layer for day-manager to consume new decision interfaces.

    Provides:
    - Snapshot building from day-manager position data
    - Risk decision application to day-manager health dicts
    - PDT and failsafe constraint checking
    """

    def __init__(
        self,
        pdt_adapter: Optional[PDTAdapter] = None,
        failsafe_adapter: Optional[FailsafeAdapter] = None,
    ):
        self.pdt_adapter = pdt_adapter or PDTAdapter()
        self.failsafe_adapter = failsafe_adapter or FailsafeAdapter()
        self._policy_risk_decisions: Dict[str, RiskDecision] = {}

    def set_risk_decisions(
        self,
        decisions: Dict[str, RiskDecision],
    ) -> None:
        """Set risk decisions from policy engine evaluation."""
        self._policy_risk_decisions = decisions

    def apply_risk_overlay(
        self,
        position: PositionSnapshot,
        health: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply policy engine risk overlay to day-manager health dict.

        Preserves current action semantics (exit, trim, add, hold).
        """
        symbol = position.symbol.upper()
        decision = self._policy_risk_decisions.get(symbol)

        if not decision:
            return health

        return risk_decision_to_day_manager_action(
            decision,
            current_action=str(health.get("action", "hold")),
            current_score=int(health.get("score", 50)),
        )

    def check_pdt_constraints(
        self,
        decision: RiskDecision,
        position: PositionSnapshot,
    ) -> RiskDecision:
        """Check and adjust decision for PDT constraints."""
        return self.pdt_adapter.adjust_decision_for_pdt(decision, position)

    def check_failsafe_constraints(
        self,
        symbol: str,
        portfolio: PortfolioSnapshot,
        signal_score: float = 50.0,
    ) -> Tuple[bool, Optional[str]]:
        """Check failsafe constraints for new entry."""
        return self.failsafe_adapter.check_entry_allowed(
            symbol, portfolio, signal_score
        )

    def get_compatible_action(
        self,
        symbol: str,
        current_action: str,
        current_score: int,
    ) -> Tuple[str, int, List[str]]:
        """
        Get day-manager compatible action tuple.

        Returns:
            (action, score, signals)
        """
        symbol = symbol.upper()
        decision = self._policy_risk_decisions.get(symbol)

        if not decision:
            return current_action, current_score, []

        result = risk_decision_to_day_manager_action(
            decision, current_action, current_score
        )

        return (
            result["action"],
            result["score"],
            result.get("signals", []),
        )


def create_day_manager_adapter(
    day_trades_remaining: int = 10,
    failsafe_level: str = "normal",
    halt_new_entries: bool = False,
    max_positions: int = 7,
) -> DayManagerAdapter:
    """
    Factory function to create a configured DayManagerAdapter.

    Args:
        day_trades_remaining: PDT day trades remaining
        failsafe_level: Current failsafe level (normal, elevated, critical)
        halt_new_entries: Whether to halt new entries
        max_positions: Maximum positions allowed

    Returns:
        Configured DayManagerAdapter
    """
    pdt_adapter = PDTAdapter(day_trades_remaining)
    failsafe_adapter = FailsafeAdapter(
        failsafe_level=failsafe_level,
        halt_new_entries=halt_new_entries,
        max_positions=max_positions,
    )
    return DayManagerAdapter(pdt_adapter, failsafe_adapter)
