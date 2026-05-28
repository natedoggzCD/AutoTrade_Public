"""
Risk/Portfolio Interfaces - Protocol Definitions
=================================================
Explicit interfaces (Protocols) for risk and portfolio policies.
These define the contract that policy implementations must fulfill.

Using Protocol (structural subtyping) allows for:
- Easy mocking in tests
- Alternative policy implementations
- Clear separation between interface and implementation
"""

from typing import Protocol, List, Dict, Optional, runtime_checkable
from dataclasses import dataclass

from autotrade.risk.contracts import (
    PositionSnapshot,
    PortfolioSnapshot,
    RiskDecision,
    RotationPlan,
    RiskHealthReport,
    FailureReason,
)


@runtime_checkable
class RiskPolicy(Protocol):
    """
    Interface for position-level risk evaluation.

    Implementations should:
    - Be pure functions where possible (no side effects)
    - Return RiskDecision with confidence and reasons
    - Check all relevant risk rules in priority order
    """

    def evaluate(
        self, position: PositionSnapshot, portfolio: PortfolioSnapshot
    ) -> RiskDecision:
        """
        Evaluate a single position against risk rules.

        Args:
            position: Current position snapshot
            portfolio: Current portfolio snapshot

        Returns:
            RiskDecision with recommended action and reasoning
        """
        ...

    def should_skip_agents(self, decision: RiskDecision) -> bool:
        """
        Determine if the LLM/agent pipeline should be skipped.

        Critical rules (hard stop, ATR stop) should return True.
        """
        ...


@runtime_checkable
class SizingPolicy(Protocol):
    """
    Interface for position sizing and allocation.

    Implementations should:
    - Calculate appropriate position size based on risk parameters
    - Respect portfolio-level constraints
    - Return size in shares/contracts
    """

    def calculate_size(
        self,
        symbol: str,
        entry_price: float,
        portfolio: PortfolioSnapshot,
        risk_per_trade: float = 0.02,
    ) -> int:
        """
        Calculate position size.

        Args:
            symbol: Trading symbol
            entry_price: Expected entry price
            portfolio: Current portfolio snapshot
            risk_per_trade: Fraction of portfolio to risk (default 2%)

        Returns:
            Number of shares/contracts to buy
        """
        ...

    def can_increase_position(
        self,
        position: PositionSnapshot,
        portfolio: PortfolioSnapshot,
        additional_shares: int,
    ) -> bool:
        """
        Check if adding to an existing position is allowed.

        Args:
            position: Current position snapshot
            portfolio: Current portfolio snapshot
            additional_shares: Additional shares to add

        Returns:
            True if increase is allowed
        """
        ...


@runtime_checkable
class RotationPolicy(Protocol):
    """
    Interface for portfolio rotation decisions.

    Implementations should:
    - Identify rotation candidates (low conviction, unlocked)
    - Match exits with entry candidates
    - Apply minimum improvement thresholds
    - Check portfolio constraints
    """

    def find_rotation_candidates(
        self,
        portfolio: PortfolioSnapshot,
        min_conviction_threshold: float = 40.0,
    ) -> List[PositionSnapshot]:
        """
        Find positions eligible for rotation.

        Args:
            portfolio: Current portfolio snapshot
            min_conviction_threshold: Maximum conviction to be a candidate

        Returns:
            List of positions to consider rotating (sorted by conviction ascending)
        """
        ...

    def evaluate_rotation(
        self,
        exit_position: PositionSnapshot,
        enter_symbol: str,
        enter_price: float,
        enter_signal_score: float,
        portfolio: PortfolioSnapshot,
        min_improvement: float = 15.0,
    ) -> RotationPlan:
        """
        Evaluate a proposed rotation.

        Args:
            exit_position: Position to exit
            enter_symbol: Symbol to enter
            enter_price: Current price of entry symbol
            enter_signal_score: Signal score for entry
            portfolio: Current portfolio snapshot
            min_improvement: Minimum improvement score required

        Returns:
            RotationPlan with assessment and any failure reasons
        """
        ...

    def apply_rotation_constraints(
        self,
        plan: RotationPlan,
        portfolio: PortfolioSnapshot,
        config: "RotationPolicyConfig",
    ) -> RotationPlan:
        """
        Apply portfolio constraints to a rotation plan.

        Checks:
        - Sector exposure limits
        - Correlation cluster limits
        - Signal family budget limits

        Modifies the plan with constraint status and failure reasons.
        """
        ...


@dataclass
class RotationPolicyConfig:
    """Configuration for rotation policy."""

    min_improvement_score: float = 15.0
    max_rotations_per_day: int = 2
    max_sector_exposure_pct: float = 40.0
    max_correlation_cluster_exposure_pct: float = 35.0
    signal_family_budgets: Dict[str, float] = None

    def __post_init__(self):
        if self.signal_family_budgets is None:
            self.signal_family_budgets = {
                "ts_momentum": 0.35,
                "xs_momentum": 0.25,
                "mean_reversion": 0.15,
                "breakout": 0.15,
                "pullback": 0.10,
            }


@runtime_checkable
class KillSwitchPolicy(Protocol):
    """
    Interface for kill switch and emergency halt decisions.

    Implementations should:
    - Track consecutive failures
    - Monitor daily realized losses
    - Make go/no-go decisions for trading
    """

    def check_kill_switch(
        self,
        portfolio: PortfolioSnapshot,
        consecutive_failures: int = 0,
    ) -> bool:
        """
        Check if kill switch should be triggered.

        Args:
            portfolio: Current portfolio snapshot
            consecutive_failures: Number of consecutive failed trades

        Returns:
            True if trading should halt
        """
        ...

    def get_failure_reason(self) -> Optional[FailureReason]:
        """
        Get the reason for kill switch trigger.

        Returns:
            FailureReason if kill switch triggered, None otherwise
        """
        ...

    def record_failure(self) -> None:
        """Record a failed trade (increment failure counter)."""
        ...

    def record_success(self) -> None:
        """Record a successful trade (reset failure counter)."""
        ...

    def get_daily_loss_pct(self) -> float:
        """
        Get current day's realized loss percentage.

        Returns:
            Negative percentage (e.g., -2.5 for 2.5% loss)
        """
        ...


@runtime_checkable
class ExposurePolicy(Protocol):
    """
    Interface for portfolio-level exposure limits.

    Implementations should:
    - Check position, sector, correlation, and family exposure
    - Return failure reasons for any breached limits
    - Provide real-time exposure calculations
    """

    def check_position_exposure(
        self,
        position: PositionSnapshot,
        portfolio: PortfolioSnapshot,
        max_position_pct: float = 15.0,
    ) -> List[FailureReason]:
        """
        Check if position exceeds exposure limits.

        Returns:
            List of failure reasons (empty if within limits)
        """
        ...

    def check_sector_exposure(
        self,
        sector: str,
        additional_exposure_pct: float,
        portfolio: PortfolioSnapshot,
        max_sector_pct: float = 40.0,
    ) -> List[FailureReason]:
        """
        Check if sector exposure would exceed limits.

        Returns:
            List of failure reasons (empty if within limits)
        """
        ...

    def check_correlation_exposure(
        self,
        additional_exposure_pct: float,
        portfolio: PortfolioSnapshot,
        max_correlation_pct: float = 35.0,
    ) -> List[FailureReason]:
        """
        Check if correlation cluster exposure would exceed limits.

        Returns:
            List of failure reasons (empty if within limits)
        """
        ...

    def check_family_budget(
        self,
        signal_family: str,
        additional_exposure_pct: float,
        portfolio: PortfolioSnapshot,
        family_budgets: Dict[str, float] = None,
    ) -> List[FailureReason]:
        """
        Check if signal family budget would be exceeded.

        Returns:
            List of failure reasons (empty if within limits)
        """
        ...

    def get_risk_health_report(
        self,
        portfolio: PortfolioSnapshot,
        config: "ExposurePolicyConfig",
    ) -> RiskHealthReport:
        """
        Generate comprehensive risk health report.

        Args:
            portfolio: Current portfolio snapshot
            config: Exposure policy configuration

        Returns:
            RiskHealthReport with all exposure checks and recommendations
        """
        ...


@dataclass
class ExposurePolicyConfig:
    """Configuration for exposure policy."""

    max_portfolio_exposure_pct: float = 100.0
    max_position_exposure_pct: float = 15.0
    max_sector_exposure_pct: float = 40.0
    max_correlation_cluster_exposure_pct: float = 35.0
    max_gross_exposure_pct: float = 150.0
    max_net_exposure_pct: float = 100.0
    entry_halt_drawdown_pct: float = 10.0
    signal_family_budgets: Dict[str, float] = None

    def __post_init__(self):
        if self.signal_family_budgets is None:
            self.signal_family_budgets = {
                "ts_momentum": 0.35,
                "xs_momentum": 0.25,
                "mean_reversion": 0.15,
                "breakout": 0.15,
                "pullback": 0.10,
            }
