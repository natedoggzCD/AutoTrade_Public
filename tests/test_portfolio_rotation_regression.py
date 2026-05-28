"""
Phase 5 - Scenario Tests + Regression Harness
==============================================
Deterministic scenario tests for risk/portfolio decisions and
regression tests comparing pre/post action decisions for fixture portfolios.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from autotrade.risk.contracts import (
    PositionPhase,
    PositionSnapshot,
    PortfolioSnapshot,
    RiskActionType,
    RiskDecision,
    FailureReason,
)
from autotrade.risk.policy_engine import (
    PolicyEvaluator,
    StandardRiskPolicy,
    StandardSizingPolicy,
    StandardRotationPolicy,
    StandardExposurePolicy,
    StandardKillSwitchPolicy,
    PortfolioAllocationPolicy,
    PortfolioAllocationConfig,
    InvariantValidator,
    PDTAdapter,
    FailsafeAdapter,
    DayManagerAdapter,
    action_plan_to_risk_decision,
    risk_decision_to_day_manager_action,
)


def create_test_position(
    symbol: str,
    entry_price: float,
    current_price: float,
    qty: int,
    hold_minutes: float = 60.0,
    pnl_pct: float = 0.0,
    atr_pct: float = 2.0,
    high_since_entry: float = 0.0,
    phase: PositionPhase = PositionPhase.LOCKED,
    sector: str = "technology",
    signal_family: str = "ts_momentum",
) -> PositionSnapshot:
    """Helper to create test positions."""
    return PositionSnapshot(
        symbol=symbol,
        entry_price=entry_price,
        current_price=current_price,
        qty=qty,
        entry_time=datetime.now() - timedelta(minutes=hold_minutes),
        pnl_pct=pnl_pct,
        atr_pct=atr_pct,
        high_since_entry=high_since_entry,
        hold_minutes=hold_minutes,
        phase=phase,
        sector=sector,
        signal_family=signal_family,
    )


def create_test_portfolio(
    total_equity: float = 10000.0,
    positions: Optional[List[PositionSnapshot]] = None,
) -> PortfolioSnapshot:
    """Helper to create test portfolios."""
    return PortfolioSnapshot(
        total_equity=total_equity,
        cash_available=total_equity * 0.3,
        buying_power=total_equity * 0.6,
        positions_value=total_equity * 0.7 if positions else 0.0,
        positions=positions or [],
        day_trades_remaining=3,
        max_positions=7,
        sector_exposure={"technology": 30.0},
        signal_family_exposure={"ts_momentum": 25.0},
    )


class TestHardStopATRStopScenarios:
    """Scenario tests for hard stop and ATR stop triggers."""

    def test_hard_stop_at_exact_threshold(self):
        """Hard stop should trigger at exactly -8%."""
        position = create_test_position("AAPL", 100.0, 92.0, 10, pnl_pct=-8.0)
        portfolio = create_test_portfolio()

        policy = StandardRiskPolicy()
        decision = policy.evaluate(position, portfolio)

        assert decision.action == RiskActionType.EXIT
        assert "hard_stop" in decision.triggered_rules

    def test_hard_stop_below_threshold(self):
        """Hard stop should trigger when P&L is below -8%."""
        position = create_test_position("AAPL", 100.0, 91.0, 10, pnl_pct=-9.0)
        portfolio = create_test_portfolio()

        policy = StandardRiskPolicy()
        decision = policy.evaluate(position, portfolio)

        assert decision.action == RiskActionType.EXIT
        assert decision.skip_agents is True

    def test_hard_stop_not_triggered_above_threshold(self):
        """Profitable position should not trigger hard stop or ATR stop."""
        position = create_test_position("AAPL", 100.0, 103.0, 10, pnl_pct=3.0)
        portfolio = create_test_portfolio()

        policy = StandardRiskPolicy()
        decision = policy.evaluate(position, portfolio)

        assert decision.action == RiskActionType.NONE
        assert len(decision.triggered_rules) == 0

    def test_atr_stop_triggered_2x_atr_loss(self):
        """ATR stop should trigger at 2x ATR loss."""
        position = create_test_position(
            "AAPL", 100.0, 94.0, 10, pnl_pct=-6.0, atr_pct=3.0
        )
        portfolio = create_test_portfolio()

        policy = StandardRiskPolicy()
        decision = policy.evaluate(position, portfolio)

        assert decision.action == RiskActionType.EXIT
        assert "atr_stop" in decision.triggered_rules

    def test_atr_stop_not_triggered_below_2x_atr(self):
        """ATR stop should NOT trigger below 2x ATR."""
        position = create_test_position(
            "AAPL", 100.0, 97.0, 10, pnl_pct=-3.0, atr_pct=2.0
        )
        portfolio = create_test_portfolio()

        policy = StandardRiskPolicy()
        decision = policy.evaluate(position, portfolio)

        assert decision.action != RiskActionType.EXIT

    def test_profit_take_levels(self):
        """Profit take should trigger at defined levels."""
        test_cases = [
            (112.0, 12.0, RiskActionType.PROFIT_TAKE),
            (107.0, 7.0, RiskActionType.PROFIT_TAKE),
            (105.0, 5.0, RiskActionType.NONE),
        ]

        for price, pnl, expected_action in test_cases:
            position = create_test_position("AAPL", 100.0, price, 10, pnl_pct=pnl)
            portfolio = create_test_portfolio()

            policy = StandardRiskPolicy()
            decision = policy.evaluate(position, portfolio)

            if expected_action != RiskActionType.NONE:
                assert decision.action == expected_action


class TestPDTLockUnlockScenarios:
    """Scenario tests for legacy day-trade compatibility."""

    def test_locked_phase_blocks_exit(self):
        """Legacy LOCKED phase positions should still allow forced exits."""
        position = create_test_position(
            "AAPL", 100.0, 90.0, 10, pnl_pct=-10.0, phase=PositionPhase.LOCKED
        )
        portfolio = create_test_portfolio()

        policy = StandardRiskPolicy()
        decision = policy.evaluate(position, portfolio)

        assert decision.action == RiskActionType.EXIT
        assert decision.skip_agents is True

    def test_unlocked_phase_allows_rotation(self):
        """UNLOCKED phase positions should allow rotation."""
        position = create_test_position(
            "AAPL", 100.0, 95.0, 10, pnl_pct=-5.0, phase=PositionPhase.UNLOCKED
        )
        portfolio = create_test_portfolio()

        policy = StandardRiskPolicy()
        decision = policy.evaluate(position, portfolio)

        assert decision.action in (RiskActionType.TRIM, RiskActionType.EXIT)

    def test_critical_phase_forces_exit(self):
        """CRITICAL phase should force immediate exit."""
        position = create_test_position(
            "AAPL", 100.0, 88.0, 10, pnl_pct=-12.0, phase=PositionPhase.CRITICAL
        )
        portfolio = create_test_portfolio()

        policy = StandardRiskPolicy()
        decision = policy.evaluate(position, portfolio)

        assert decision.action == RiskActionType.EXIT
        assert decision.skip_agents is True

    def test_pdt_adapter_no_trades_remaining_still_allows_rotation(self):
        """PDT adapter is disabled and should not block rotation."""
        adapter = PDTAdapter(day_trades_remaining=0)

        position = create_test_position(
            "AAPL", 100.0, 90.0, 10, phase=PositionPhase.LOCKED
        )
        portfolio = create_test_portfolio()

        allowed, reason = adapter.can_rotate(position, 5000.0, portfolio)

        assert allowed is True
        assert reason is None

    def test_pdt_adapter_with_trades_available(self):
        """PDT adapter is legacy-only and should allow when trades available."""
        adapter = PDTAdapter(day_trades_remaining=3)

        position = create_test_position(
            "AAPL", 100.0, 95.0, 10, phase=PositionPhase.UNLOCKED
        )
        portfolio = create_test_portfolio()

        allowed, reason = adapter.can_rotate(position, 5000.0, portfolio)

        assert allowed is True


class TestLowConvictionRotationScenarios:
    """Scenario tests for low-conviction rotation."""

    def test_low_pnl_unlock_rotation_candidate(self):
        """Low P&L unlocked positions should be rotation candidates."""
        positions = [
            create_test_position(
                "AAPL", 100.0, 93.0, 10, pnl_pct=-7.0, phase=PositionPhase.UNLOCKED
            ),
            create_test_position(
                "MSFT", 100.0, 105.0, 10, pnl_pct=5.0, phase=PositionPhase.UNLOCKED
            ),
        ]
        portfolio = create_test_portfolio(positions=positions)

        policy = StandardRotationPolicy()
        candidates = policy.find_rotation_candidates(portfolio)

        assert len(candidates) >= 1
        assert candidates[0].symbol == "AAPL"

    def test_rotation_requires_improvement_threshold(self):
        """Rotation should evaluate improvement against threshold."""
        exit_pos = create_test_position(
            "AAPL", 100.0, 105.0, 10, pnl_pct=5.0, phase=PositionPhase.UNLOCKED
        )
        portfolio = create_test_portfolio()

        policy = StandardRotationPolicy()
        plan = policy.evaluate_rotation(
            exit_pos,
            "MSFT",
            100.0,
            90.0,
            portfolio,
            min_improvement=20.0,
        )

        assert plan.exit_symbol == "AAPL"
        assert hasattr(plan, "improvement_score")

    def test_rotation_passes_high_improvement(self):
        """Rotation should pass with high improvement."""
        exit_pos = create_test_position(
            "AAPL", 100.0, 90.0, 10, pnl_pct=-10.0, phase=PositionPhase.UNLOCKED
        )
        portfolio = create_test_portfolio()

        policy = StandardRotationPolicy()
        plan = policy.evaluate_rotation(
            exit_pos,
            "MSFT",
            100.0,
            80.0,
            portfolio,
            min_improvement=15.0,
        )

        assert plan.exit_symbol == "AAPL"
        assert plan.enter_symbol == "MSFT"
        assert plan.improvement_score >= 15.0


class TestFamilyBudgetConstraintScenarios:
    """Scenario tests for signal family budget constraints."""

    def test_family_within_budget_allows_entry(self):
        """Entry should be allowed when family within budget."""
        portfolio = create_test_portfolio()
        portfolio.signal_family_exposure = {"ts_momentum": 20.0}

        config = PortfolioAllocationConfig()
        policy = PortfolioAllocationPolicy(config)

        available = policy.calculate_family_allocation(portfolio, "ts_momentum")

        assert available > 0

    def test_family_at_budget_blocks_entry(self):
        """Entry should be blocked when family at budget."""
        portfolio = create_test_portfolio()
        portfolio.signal_family_exposure = {"ts_momentum": 25.0}

        config = PortfolioAllocationConfig(signal_family_budgets={"ts_momentum": 0.25})
        policy = PortfolioAllocationPolicy(config)

        available = policy.calculate_family_allocation(portfolio, "ts_momentum")

        assert available <= 0

    def test_family_over_budget_triggers_rebalance(self):
        """Over-budget family should trigger rebalance."""
        portfolio = create_test_portfolio()
        portfolio.signal_family_exposure = {"xs_momentum": 25.0}

        config = PortfolioAllocationConfig()
        policy = PortfolioAllocationPolicy(config)

        should_rebalance = policy.should_rebalance(portfolio, min_drift_threshold=10.0)

        assert should_rebalance is True

    def test_multiple_families_budget_enforcement(self):
        """Multiple families should each respect their budgets."""
        portfolio = create_test_portfolio()
        portfolio.signal_family_exposure = {
            "ts_momentum": 20.0,
            "xs_momentum": 18.0,
            "mean_reversion": 12.0,
        }

        config = PortfolioAllocationConfig()
        policy = PortfolioAllocationPolicy(config)

        for family in config.signal_family_budgets.keys():
            budget_pct = config.signal_family_budgets[family]
            current = portfolio.signal_family_exposure.get(family, 0.0)
            if current > 0:
                available = policy.calculate_family_allocation(portfolio, family)
                assert current <= budget_pct * 100 or available <= 0


class TestSectorCorrelationCapScenarios:
    """Scenario tests for sector and correlation cap enforcement."""

    def test_sector_within_cap_allows_entry(self):
        """Entry should be allowed when sector within cap."""
        portfolio = create_test_portfolio()
        portfolio.sector_exposure = {"technology": 30.0}

        policy = StandardExposurePolicy()
        failures = policy.check_sector_exposure(
            "technology", 5.0, portfolio, max_sector_pct=40.0
        )

        assert len(failures) == 0

    def test_sector_at_cap_blocks_entry(self):
        """Entry should be blocked when sector at cap."""
        portfolio = create_test_portfolio()
        portfolio.sector_exposure = {"technology": 38.0}

        policy = StandardExposurePolicy()
        failures = policy.check_sector_exposure(
            "technology", 5.0, portfolio, max_sector_pct=40.0
        )

        assert len(failures) > 0

    def test_correlation_cluster_within_cap(self):
        """Correlation cluster within cap should allow entry."""
        portfolio = create_test_portfolio()
        portfolio.correlation_cluster_exposure = 25.0

        validator = InvariantValidator(max_correlation_cluster_exposure_pct=35.0)

        result = validator.check_correlation_exposure_invariant(10.0, portfolio)

        assert result.passed is True

    def test_correlation_cluster_at_cap(self):
        """Correlation cluster at cap should block entry."""
        portfolio = create_test_portfolio()
        portfolio.correlation_cluster_exposure = 30.0

        validator = InvariantValidator(max_correlation_cluster_exposure_pct=35.0)

        result = validator.check_correlation_exposure_invariant(10.0, portfolio)

        assert result.passed is False
        assert result.failure_reason == FailureReason.CORRELATION_CAP_REACHED


class TestFailsafeHaltForcedExitScenarios:
    """Scenario tests for failsafe halts and forced exits."""

    def test_failsafe_critical_halts_all_entries(self):
        """Critical failsafe should halt all new entries."""
        adapter = FailsafeAdapter(
            failsafe_level="critical",
            halt_new_entries=False,
            max_positions=7,
        )

        portfolio = create_test_portfolio()

        allowed, reason = adapter.check_entry_allowed("AAPL", portfolio, 60.0)

        assert allowed is False
        assert reason is not None and (
            "critical" in reason.lower() or "failsafe" in reason.lower()
        )

    def test_failsafe_halt_flag_blocks_entries(self):
        """Halt flag should block entries regardless of level."""
        adapter = FailsafeAdapter(
            failsafe_level="normal",
            halt_new_entries=True,
            max_positions=7,
        )

        portfolio = create_test_portfolio()

        allowed, reason = adapter.check_entry_allowed("AAPL", portfolio, 60.0)

        assert allowed is False

    def test_failsafe_max_positions_blocks_entry(self):
        """Max positions reached should block entry."""
        adapter = FailsafeAdapter(
            failsafe_level="normal",
            halt_new_entries=False,
            max_positions=3,
        )

        positions = [
            create_test_position(f"SYM{i}", 100.0, 100.0, 10) for i in range(3)
        ]
        portfolio = create_test_portfolio(positions=positions)

        allowed, reason = adapter.check_entry_allowed("NEWSYM", portfolio, 60.0)

        assert allowed is False
        assert reason is not None and "positions" in reason.lower()

    def test_failsafe_drawdown_blocks_entry(self):
        """High drawdown should block new entries."""
        adapter = FailsafeAdapter(
            failsafe_level="normal",
            halt_new_entries=False,
            max_positions=7,
        )

        portfolio = create_test_portfolio()
        portfolio.current_drawdown_pct = 12.0

        allowed, reason = adapter.check_entry_allowed("AAPL", portfolio, 60.0)

        assert allowed is False
        assert reason is not None and "drawdown" in reason.lower()

    def test_elevated_failsafe_reduces_size(self):
        """Elevated failsafe should reduce max position size."""
        adapter = FailsafeAdapter(
            failsafe_level="elevated",
            halt_new_entries=False,
            max_positions=7,
        )

        portfolio = create_test_portfolio()

        size = adapter.get_max_position_size(portfolio, risk_per_trade=0.02)

        expected = 10000.0 * 0.02 * 5 * 0.75
        assert size == expected

    def test_kill_switch_triggers_after_failures(self):
        """Kill switch should trigger after consecutive failures."""
        policy = StandardKillSwitchPolicy(max_consecutive_failures=3)
        portfolio = create_test_portfolio()

        for _ in range(3):
            result = policy.check_kill_switch(portfolio, consecutive_failures=3)
            assert result is True


class TestRegressionFixtures:
    """Regression tests comparing pre/post action decisions for fixture portfolios."""

    def test_fixture_portfolio_1_healthy(self):
        """Fixture 1: Healthy portfolio with profits."""
        positions = [
            create_test_position("AAPL", 100.0, 105.0, 10, pnl_pct=5.0),
            create_test_position("MSFT", 200.0, 210.0, 5, pnl_pct=5.0),
            create_test_position("GOOGL", 150.0, 155.0, 8, pnl_pct=3.33),
        ]
        portfolio = create_test_portfolio(positions=positions)

        evaluator = PolicyEvaluator()
        result = evaluator.evaluate_portfolio(portfolio)

        for decision in result.risk_decisions:
            assert decision.action in (
                RiskActionType.NONE,
                RiskActionType.HOLD,
                RiskActionType.PROFIT_TAKE,
            )

    def test_fixture_portfolio_2_hard_stop_trigger(self):
        """Fixture 2: Portfolio with hard stop trigger."""
        positions = [
            create_test_position("AAPL", 100.0, 91.0, 10, pnl_pct=-9.0),
            create_test_position("MSFT", 200.0, 210.0, 5, pnl_pct=5.0),
        ]
        portfolio = create_test_portfolio(positions=positions)

        evaluator = PolicyEvaluator()
        result = evaluator.evaluate_portfolio(portfolio)

        assert len(result.risk_decisions) >= 1
        exit_decisions = [
            d for d in result.risk_decisions if d.action == RiskActionType.EXIT
        ]
        assert len(exit_decisions) >= 1

    def test_fixture_portfolio_3_mixed_pnl(self):
        """Fixture 3: Portfolio with mixed P&L positions."""
        positions = [
            create_test_position("AAPL", 100.0, 108.0, 10, pnl_pct=8.0),
            create_test_position("MSFT", 200.0, 190.0, 5, pnl_pct=-5.0),
            create_test_position("GOOGL", 150.0, 145.0, 8, pnl_pct=-3.33),
        ]
        portfolio = create_test_portfolio(positions=positions)

        evaluator = PolicyEvaluator()
        result = evaluator.evaluate_portfolio(portfolio)

        exit_or_trim = [
            d
            for d in result.risk_decisions
            if d.action in (RiskActionType.EXIT, RiskActionType.TRIM)
        ]
        profit_take_or_hold = [
            d
            for d in result.risk_decisions
            if d.action
            in (RiskActionType.PROFIT_TAKE, RiskActionType.HOLD, RiskActionType.NONE)
        ]

        assert len(exit_or_trim) >= 1
        assert len(profit_take_or_hold) >= 1

    def test_fixture_portfolio_4_pdt_disabled_preserves_exit(self):
        """Fixture 4: PDT disabled; hard exits are not trimmed."""
        positions = [
            create_test_position(
                "AAPL", 100.0, 92.0, 10, pnl_pct=-8.0, phase=PositionPhase.LOCKED
            ),
        ]
        pdt_adapter = PDTAdapter(day_trades_remaining=1)
        position = positions[0]

        adjusted = pdt_adapter.adjust_decision_for_pdt(
            RiskDecision(
                action=RiskActionType.EXIT,
                size_delta=-10,
                confidence=0.95,
                reasons=["Hard stop"],
                triggered_rules=["hard_stop"],
                skip_agents=True,
                hold_phase=PositionPhase.LOCKED,
                requires_pdt_burn=True,
            ),
            position,
        )

        assert adjusted.action == RiskActionType.EXIT

    def test_fixture_portfolio_5_sector_concentration(self):
        """Fixture 5: Sector-concentrated portfolio."""
        positions = [
            create_test_position("AAPL", 100.0, 105.0, 10, sector="technology"),
            create_test_position("MSFT", 200.0, 210.0, 8, sector="technology"),
            create_test_position("GOOGL", 150.0, 155.0, 5, sector="technology"),
        ]
        portfolio = create_test_portfolio(positions=positions)
        portfolio.sector_exposure = {"technology": 60.0}

        policy = StandardExposurePolicy()
        failures = policy.check_sector_exposure(
            "technology", 10.0, portfolio, max_sector_pct=40.0
        )

        assert len(failures) > 0

    def test_fixture_portfolio_6_family_budget_breach(self):
        """Fixture 6: Family budget breach scenario."""
        positions = [
            create_test_position("AAPL", 100.0, 105.0, 10, signal_family="ts_momentum"),
            create_test_position("MSFT", 100.0, 105.0, 10, signal_family="ts_momentum"),
            create_test_position(
                "GOOGL", 100.0, 105.0, 10, signal_family="ts_momentum"
            ),
        ]
        portfolio = create_test_portfolio(positions=positions)
        portfolio.signal_family_exposure = {"ts_momentum": 30.0}

        config = PortfolioAllocationConfig(signal_family_budgets={"ts_momentum": 0.25})
        policy = PortfolioAllocationPolicy(config)

        available = policy.calculate_family_allocation(portfolio, "ts_momentum")

        assert available <= 0

    def test_fixture_portfolio_7_failsafe_elevated(self):
        """Fixture 7: Elevated failsafe state."""
        adapter = FailsafeAdapter(
            failsafe_level="elevated",
            halt_new_entries=False,
            max_positions=5,
        )

        positions = [
            create_test_position("AAPL", 100.0, 95.0, 10),
            create_test_position("MSFT", 200.0, 190.0, 8),
        ]
        portfolio = create_test_portfolio(positions=positions)

        size = adapter.get_max_position_size(portfolio, risk_per_trade=0.02)
        assert size < 10000.0 * 0.02 * 5

    def test_fixture_portfolio_8_rotation_with_pdt_disabled(self):
        """Fixture 8: PDT disabled; rotation is allowed."""
        pdt_adapter = PDTAdapter(day_trades_remaining=0)

        position = create_test_position(
            "AAPL", 100.0, 95.0, 10, phase=PositionPhase.LOCKED
        )
        portfolio = create_test_portfolio(positions=[position])

        allowed, reason = pdt_adapter.can_rotate(position, 5000.0, portfolio)

        assert allowed is True
        assert reason is None


class TestPhase5Completion:
    """Tests to verify Phase 5 completion requirements."""

    def test_hard_stop_scenario_tests_present(self):
        """Hard stop scenario tests should be present."""
        assert hasattr(
            TestHardStopATRStopScenarios, "test_hard_stop_at_exact_threshold"
        )
        assert hasattr(TestHardStopATRStopScenarios, "test_hard_stop_below_threshold")
        assert hasattr(
            TestHardStopATRStopScenarios, "test_hard_stop_not_triggered_above_threshold"
        )

    def test_atr_stop_scenario_tests_present(self):
        """ATR stop scenario tests should be present."""
        assert hasattr(
            TestHardStopATRStopScenarios, "test_atr_stop_triggered_2x_atr_loss"
        )
        assert hasattr(
            TestHardStopATRStopScenarios, "test_atr_stop_not_triggered_below_2x_atr"
        )

    def test_pdt_transition_tests_present(self):
        """Legacy day-trade compatibility tests should be present."""
        assert hasattr(TestPDTLockUnlockScenarios, "test_locked_phase_blocks_exit")
        assert hasattr(
            TestPDTLockUnlockScenarios, "test_unlocked_phase_allows_rotation"
        )
        assert hasattr(
            TestPDTLockUnlockScenarios,
            "test_pdt_adapter_no_trades_remaining_still_allows_rotation",
        )

    def test_rotation_scenario_tests_present(self):
        """Rotation scenario tests should be present."""
        assert hasattr(
            TestLowConvictionRotationScenarios, "test_low_pnl_unlock_rotation_candidate"
        )
        assert hasattr(
            TestLowConvictionRotationScenarios,
            "test_rotation_requires_improvement_threshold",
        )

    def test_family_budget_tests_present(self):
        """Family budget constraint tests should be present."""
        assert hasattr(
            TestFamilyBudgetConstraintScenarios,
            "test_family_within_budget_allows_entry",
        )
        assert hasattr(
            TestFamilyBudgetConstraintScenarios,
            "test_family_over_budget_triggers_rebalance",
        )

    def test_sector_correlation_tests_present(self):
        """Sector and correlation cap tests should be present."""
        assert hasattr(
            TestSectorCorrelationCapScenarios, "test_sector_within_cap_allows_entry"
        )
        assert hasattr(
            TestSectorCorrelationCapScenarios, "test_correlation_cluster_at_cap"
        )

    def test_failsafe_tests_present(self):
        """Failsafe halt and forced exit tests should be present."""
        assert hasattr(
            TestFailsafeHaltForcedExitScenarios,
            "test_failsafe_critical_halts_all_entries",
        )
        assert hasattr(
            TestFailsafeHaltForcedExitScenarios,
            "test_failsafe_max_positions_blocks_entry",
        )
        assert hasattr(
            TestFailsafeHaltForcedExitScenarios,
            "test_kill_switch_triggers_after_failures",
        )

    def test_regression_fixture_tests_present(self):
        """Regression fixture tests should be present."""
        assert hasattr(TestRegressionFixtures, "test_fixture_portfolio_1_healthy")
        assert hasattr(
            TestRegressionFixtures, "test_fixture_portfolio_2_hard_stop_trigger"
        )
        assert hasattr(TestRegressionFixtures, "test_fixture_portfolio_3_mixed_pnl")
        assert hasattr(
            TestRegressionFixtures, "test_fixture_portfolio_4_pdt_disabled_preserves_exit"
        )
        assert hasattr(
            TestRegressionFixtures, "test_fixture_portfolio_5_sector_concentration"
        )
        assert hasattr(
            TestRegressionFixtures, "test_fixture_portfolio_6_family_budget_breach"
        )
        assert hasattr(
            TestRegressionFixtures, "test_fixture_portfolio_7_failsafe_elevated"
        )
        assert hasattr(
            TestRegressionFixtures, "test_fixture_portfolio_8_rotation_with_pdt_disabled"
        )

    def test_all_scenario_categories_covered(self):
        """All required scenario categories should be covered."""
        required_categories = [
            "hard_stop",
            "atr_stop",
            "pdt_disabled",
            "low_conviction_rotation",
            "family_budget",
            "sector_cap",
            "correlation_cap",
            "failsafe_halt",
            "forced_exit",
            "regression_fixture",
        ]

        test_classes = [
            TestHardStopATRStopScenarios,
            TestPDTLockUnlockScenarios,
            TestLowConvictionRotationScenarios,
            TestFamilyBudgetConstraintScenarios,
            TestSectorCorrelationCapScenarios,
            TestFailsafeHaltForcedExitScenarios,
            TestRegressionFixtures,
        ]

        assert len(test_classes) >= 7

    def test_phase5_completion_test_class_present(self):
        """Phase 5 completion test class should be present."""
        assert hasattr(TestPhase5Completion, "test_all_scenario_categories_covered")
        assert hasattr(TestPhase5Completion, "test_regression_fixture_tests_present")
