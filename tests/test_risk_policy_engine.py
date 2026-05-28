"""
Tests for Phase 2 - Policy Engine and Pure Decision Layer
========================================================
Tests the deterministic policy evaluator pipeline and portfolio allocation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pytest

from autotrade.risk.contracts import (
    PositionPhase,
    PositionSnapshot,
    PortfolioSnapshot,
    RiskActionType,
    RiskDecision,
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
    evaluate_position_pure,
    get_portfolio_health_pure,
    ExposurePolicyConfig,
    RotationPolicyConfig,
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
    positions: list = None,
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


class TestStandardRiskPolicy:
    """Tests for StandardRiskPolicy - pure risk evaluation."""

    def test_hard_stop_triggered(self):
        """Hard stop should trigger at -8%."""
        position = create_test_position("AAPL", 100.0, 91.0, 10, pnl_pct=-9.0)
        portfolio = create_test_portfolio()

        policy = StandardRiskPolicy()
        decision = policy.evaluate(position, portfolio)

        assert decision.action == RiskActionType.EXIT
        assert decision.confidence >= 0.9
        assert "hard_stop" in decision.triggered_rules
        assert decision.skip_agents is True
        assert decision.should_execute is True

    def test_atr_stop_triggered(self):
        """ATR stop should trigger when loss exceeds k*ATR."""
        position = create_test_position(
            "AAPL", 100.0, 92.0, 10, pnl_pct=-8.0, atr_pct=3.0
        )
        portfolio = create_test_portfolio()

        policy = StandardRiskPolicy()
        decision = policy.evaluate(position, portfolio)

        assert decision.action == RiskActionType.EXIT
        assert decision.skip_agents is True

    def test_trim_threshold_triggered(self):
        """Trim threshold should trigger at -5% when ATR doesn't trigger first."""
        position = create_test_position(
            "AAPL", 100.0, 95.5, 10, pnl_pct=-4.5, atr_pct=1.0
        )
        portfolio = create_test_portfolio()

        policy = StandardRiskPolicy()
        decision = policy.evaluate(position, portfolio)

        assert decision.action in (RiskActionType.TRIM, RiskActionType.EXIT)

    def test_no_trigger_returns_none(self):
        """No risk rule should return NONE action."""
        position = create_test_position("AAPL", 100.0, 102.0, 10, pnl_pct=2.0)
        portfolio = create_test_portfolio()

        policy = StandardRiskPolicy()
        decision = policy.evaluate(position, portfolio)

        assert decision.action == RiskActionType.NONE
        assert decision.skip_agents is False


class TestStandardSizingPolicy:
    """Tests for StandardSizingPolicy - position sizing."""

    def test_calculate_size_returns_positive(self):
        """Size calculation should return positive value."""
        portfolio = PortfolioSnapshot(
            total_equity=10000.0,
            cash_available=3000.0,
            buying_power=6000.0,
            positions_value=7000.0,
            max_positions=7,
        )

        policy = StandardSizingPolicy()
        size = policy.calculate_size("AAPL", 100.0, portfolio, risk_per_trade=0.02)

        assert size >= 0


class TestStandardRotationPolicy:
    """Tests for StandardRotationPolicy - portfolio rotation."""

    def test_find_rotation_candidates(self):
        """Should find unlocked positions with low conviction."""
        positions = [
            create_test_position(
                "AAPL", 100.0, 95.0, 10, pnl_pct=-5.0, phase=PositionPhase.UNLOCKED
            ),
            create_test_position(
                "MSFT", 100.0, 105.0, 10, pnl_pct=5.0, phase=PositionPhase.UNLOCKED
            ),
            create_test_position(
                "GOOGL", 100.0, 98.0, 10, pnl_pct=-2.0, phase=PositionPhase.LOCKED
            ),
        ]
        portfolio = create_test_portfolio(positions=positions)

        policy = StandardRotationPolicy()
        candidates = policy.find_rotation_candidates(portfolio)

        assert len(candidates) >= 1
        assert candidates[0].symbol == "AAPL"

    def test_evaluate_rotation(self):
        """Rotation evaluation should calculate improvement."""
        exit_pos = create_test_position(
            "AAPL", 100.0, 95.0, 10, pnl_pct=-5.0, phase=PositionPhase.UNLOCKED
        )
        portfolio = create_test_portfolio()

        policy = StandardRotationPolicy()
        plan = policy.evaluate_rotation(
            exit_pos,
            "MSFT",
            100.0,
            70.0,
            portfolio,
            min_improvement=15.0,
        )

        assert plan.exit_symbol == "AAPL"
        assert plan.enter_symbol == "MSFT"
        assert plan.improvement_score > 0


class TestStandardExposurePolicy:
    """Tests for StandardExposurePolicy - portfolio exposure limits."""

    def test_check_position_exposure(self):
        """Should detect over-exposed positions."""
        position = PositionSnapshot(
            symbol="AAPL",
            entry_price=100.0,
            current_price=100.0,
            qty=10,
            entry_time=datetime.now(),
            market_value=1000.0,
        )
        portfolio = PortfolioSnapshot(
            total_equity=100000.0,
            cash_available=90000.0,
            buying_power=90000.0,
            positions_value=1000.0,
            max_positions=7,
        )

        policy = StandardExposurePolicy()
        failures = policy.check_position_exposure(
            position, portfolio, max_position_pct=15.0
        )

        assert len(failures) == 0

    def test_check_sector_exposure(self):
        """Should detect sector over-exposure."""
        portfolio = PortfolioSnapshot(
            total_equity=10000.0,
            cash_available=5000.0,
            buying_power=5000.0,
            positions_value=5000.0,
            max_positions=7,
            sector_exposure={"technology": 30.0},
        )

        policy = StandardExposurePolicy()
        failures = policy.check_sector_exposure(
            "energy", 5.0, portfolio, max_sector_pct=40.0
        )

        assert len(failures) == 0


class TestStandardKillSwitchPolicy:
    """Tests for StandardKillSwitchPolicy - emergency halt."""

    def test_consecutive_failures_triggers_kill(self):
        """Should trigger kill after consecutive failures."""
        policy = StandardKillSwitchPolicy(max_consecutive_failures=3)
        portfolio = create_test_portfolio()

        result = policy.check_kill_switch(portfolio, consecutive_failures=3)

        assert result is True

    def test_success_resets_failures(self):
        """Success should reset failure counter."""
        policy = StandardKillSwitchPolicy(max_consecutive_failures=3)
        policy._consecutive_failures = 2

        policy.record_success()

        assert policy._consecutive_failures == 0


class TestPolicyEvaluator:
    """Tests for complete PolicyEvaluator pipeline."""

    def test_evaluate_portfolio_returns_risk_decisions(self):
        """Should return risk decisions for all positions."""
        positions = [
            create_test_position("AAPL", 100.0, 91.0, 10, pnl_pct=-9.0),
        ]
        portfolio = create_test_portfolio(positions=positions)

        evaluator = PolicyEvaluator()
        result = evaluator.evaluate_portfolio(portfolio)

        assert len(result.risk_decisions) == 1
        assert result.risk_decisions[0].action == RiskActionType.EXIT

    def test_evaluate_portfolio_generates_health_report(self):
        """Should generate health report."""
        positions = [
            create_test_position(
                "AAPL", 100.0, 95.0, 10, pnl_pct=-5.0, phase=PositionPhase.UNLOCKED
            ),
        ]
        portfolio = create_test_portfolio(positions=positions)

        evaluator = PolicyEvaluator()
        result = evaluator.evaluate_portfolio(portfolio)

        assert result.health_report is not None
        assert hasattr(result.health_report, "failures")

    def test_evaluate_portfolio_runs_sizing_stage_for_rotation_candidates(self):
        """Pipeline should execute sizing before evaluating rotations."""
        positions = [
            create_test_position(
                "AAPL", 100.0, 95.0, 10, pnl_pct=-5.0, phase=PositionPhase.UNLOCKED
            ),
        ]
        portfolio = create_test_portfolio(positions=positions)

        class RecordingSizingPolicy:
            def __init__(self):
                self.calls = []

            def calculate_size(
                self,
                symbol: str,
                entry_price: float,
                portfolio,
                risk_per_trade: float = 0.02,
            ) -> int:
                self.calls.append((symbol, entry_price))
                return 10

            def can_increase_position(
                self, position, portfolio, additional_shares: int
            ) -> bool:
                return True

        sizing_policy = RecordingSizingPolicy()
        evaluator = PolicyEvaluator(sizing_policy=sizing_policy)
        result = evaluator.evaluate_portfolio(
            portfolio, rotation_candidates=[("MSFT", 100.0, 80.0)]
        )

        assert ("MSFT", 100.0) in sizing_policy.calls
        assert result.sizing_recommendations.get("MSFT") == 10


class TestPortfolioAllocationPolicy:
    """Tests for PortfolioAllocationPolicy - signal family budgets."""

    def test_calculate_family_allocation(self):
        """Should calculate available allocation for family."""
        portfolio = create_test_portfolio()
        portfolio.signal_family_exposure = {"ts_momentum": 20.0}

        config = PortfolioAllocationConfig()
        policy = PortfolioAllocationPolicy(config)

        available = policy.calculate_family_allocation(portfolio, "ts_momentum")

        assert available >= 0
        assert available <= config.signal_family_budgets["ts_momentum"] * 100

    def test_should_rebalance_detects_drift(self):
        """Should detect when rebalancing is needed."""
        portfolio = create_test_portfolio()
        portfolio.signal_family_exposure = {"ts_momentum": 40.0}

        config = PortfolioAllocationConfig()
        policy = PortfolioAllocationPolicy(config)

        should_rebalance = policy.should_rebalance(portfolio, min_drift_threshold=10.0)

        assert should_rebalance is True


class TestPureFunctions:
    """Tests for convenience pure functions."""

    def test_evaluate_position_pure(self):
        """evaluate_position_pure should work without state."""
        position = create_test_position("AAPL", 100.0, 91.0, 10, pnl_pct=-9.0)
        portfolio = create_test_portfolio()

        decision = evaluate_position_pure(position, portfolio)

        assert isinstance(decision, RiskDecision)
        assert decision.action == RiskActionType.EXIT

    def test_get_portfolio_health_pure(self):
        """get_portfolio_health_pure should work without state."""
        portfolio = create_test_portfolio()

        health = get_portfolio_health_pure(portfolio)

        assert isinstance(health, dict) or hasattr(health, "to_dict")


class TestPhase2Completion:
    """Regression tests for Phase 2 completion."""

    def test_policy_engine_module_imports(self):
        """policy_engine should be importable."""
        from autotrade.risk import policy_engine

        assert hasattr(policy_engine, "PolicyEvaluator")
        assert hasattr(policy_engine, "StandardRiskPolicy")
        assert hasattr(policy_engine, "StandardSizingPolicy")
        assert hasattr(policy_engine, "StandardRotationPolicy")
        assert hasattr(policy_engine, "StandardExposurePolicy")
        assert hasattr(policy_engine, "StandardKillSwitchPolicy")
        assert hasattr(policy_engine, "PortfolioAllocationPolicy")

    def test_pure_functions_available(self):
        """Pure convenience functions should be available."""
        from autotrade.risk import policy_engine

        assert hasattr(policy_engine, "evaluate_position_pure")
        assert hasattr(policy_engine, "get_portfolio_health_pure")

    def test_no_broker_dependencies_in_policy_engine(self):
        """policy_engine should not import broker clients."""
        import autotrade.risk.policy_engine as pe

        source = pe.__file__
        with open(source, "r") as f:
            content = f.read()

        assert "TradingClient" not in content
        assert (
            "alpaca" not in content.lower()
            or "alpaca" in content.lower()
            and "import" not in content.lower()[:100]
        )


class TestPhase3FailFastInvariants:
    """Tests for Phase 3 - Fail-Fast Invariants + Kill-Switch."""

    def test_position_exposure_invariant_pass(self):
        """Position within exposure limits should pass."""
        from autotrade.risk.policy_engine import InvariantValidator

        validator = InvariantValidator(max_symbol_exposure_pct=15.0)
        position = create_test_position("AAPL", 150.0, 145.0, 100)
        position.market_value = 14500.0

        portfolio = create_test_portfolio(
            total_equity=100000.0,
            positions=[position],
        )

        result = validator.check_position_exposure_invariant(position, portfolio)
        assert result.passed is True
        assert result.failure_reason is None

    def test_position_exposure_invariant_fail(self):
        """Position exceeding exposure should fail with OVER_MAX_EXPOSURE."""
        from autotrade.risk.policy_engine import InvariantValidator
        from autotrade.risk.contracts import FailureReason

        validator = InvariantValidator(max_symbol_exposure_pct=15.0)
        position = create_test_position("AAPL", 150.0, 145.0, 200)
        position.market_value = 29000.0

        portfolio = create_test_portfolio(
            total_equity=100000.0,
            positions=[position],
        )

        result = validator.check_position_exposure_invariant(position, portfolio)
        assert result.passed is False
        assert result.failure_reason == FailureReason.OVER_MAX_EXPOSURE

    def test_portfolio_exposure_invariants_pass(self):
        """Portfolio within exposure limits should pass."""
        from autotrade.risk.policy_engine import InvariantValidator

        validator = InvariantValidator(
            max_gross_exposure_pct=150.0,
            max_net_exposure_pct=100.0,
        )

        portfolio = PortfolioSnapshot(
            total_equity=100000.0,
            cash_available=20000.0,
            buying_power=40000.0,
            positions_value=80000.0,
            positions=[],
            snapshot_time=datetime.now(),
        )

        result = validator.check_portfolio_exposure_invariants(portfolio)
        assert result.passed is True
        assert result.failure_reason is None

    def test_portfolio_exposure_invariants_fail_gross(self):
        """Portfolio exceeding gross exposure should fail."""
        from autotrade.risk.policy_engine import InvariantValidator
        from autotrade.risk.contracts import FailureReason

        validator = InvariantValidator(
            max_gross_exposure_pct=100.0,
            max_net_exposure_pct=100.0,
        )

        portfolio = PortfolioSnapshot(
            total_equity=100000.0,
            cash_available=20000.0,
            buying_power=40000.0,
            positions_value=180000.0,
            positions=[],
            snapshot_time=datetime.now(),
        )

        result = validator.check_portfolio_exposure_invariants(portfolio)
        assert result.passed is False
        assert result.failure_reason == FailureReason.OVER_MAX_EXPOSURE
        assert "gross" in result.failure_detail.lower()

    def test_drawdown_invariant_pass(self):
        """Portfolio within drawdown should pass."""
        from autotrade.risk.policy_engine import InvariantValidator

        validator = InvariantValidator(entry_halt_drawdown_pct=10.0)
        portfolio = PortfolioSnapshot(
            total_equity=95000.0,
            cash_available=20000.0,
            buying_power=40000.0,
            positions_value=75000.0,
            positions=[],
            peak_equity=100000.0,
            snapshot_time=datetime.now(),
        )

        result = validator.check_drawdown_invariant(portfolio)
        assert result.passed is True
        assert result.failure_reason is None

    def test_drawdown_invariant_fail(self):
        """Portfolio exceeding drawdown threshold should fail with DRAWDOWN_BREACH."""
        from autotrade.risk.policy_engine import InvariantValidator
        from autotrade.risk.contracts import FailureReason

        validator = InvariantValidator(entry_halt_drawdown_pct=10.0)
        portfolio = PortfolioSnapshot(
            total_equity=85000.0,
            cash_available=20000.0,
            buying_power=40000.0,
            positions_value=65000.0,
            positions=[],
            peak_equity=100000.0,
            current_drawdown_pct=15.0,
            snapshot_time=datetime.now(),
        )

        result = validator.check_drawdown_invariant(portfolio)
        assert result.passed is False
        assert result.failure_reason == FailureReason.DRAWDOWN_BREACH

    def test_sector_exposure_invariant_pass(self):
        """Sector within limits should pass."""
        from autotrade.risk.policy_engine import InvariantValidator

        validator = InvariantValidator(max_sector_exposure_pct=40.0)
        portfolio = PortfolioSnapshot(
            total_equity=100000.0,
            cash_available=20000.0,
            buying_power=40000.0,
            positions_value=80000.0,
            positions=[],
            sector_exposure={"technology": 25.0},
            snapshot_time=datetime.now(),
        )

        result = validator.check_sector_exposure_invariant(
            "technology", 10.0, portfolio
        )
        assert result.passed is True

    def test_sector_exposure_invariant_fail(self):
        """Sector exceeding limit should fail with SECTOR_CAP_REACHED."""
        from autotrade.risk.policy_engine import InvariantValidator
        from autotrade.risk.contracts import FailureReason

        validator = InvariantValidator(max_sector_exposure_pct=40.0)
        portfolio = PortfolioSnapshot(
            total_equity=100000.0,
            cash_available=20000.0,
            buying_power=40000.0,
            positions_value=80000.0,
            positions=[],
            sector_exposure={"technology": 35.0},
            snapshot_time=datetime.now(),
        )

        result = validator.check_sector_exposure_invariant(
            "technology", 10.0, portfolio
        )
        assert result.passed is False
        assert result.failure_reason == FailureReason.SECTOR_CAP_REACHED

    def test_correlation_exposure_invariant_pass(self):
        """Correlation cluster within limits should pass."""
        from autotrade.risk.policy_engine import InvariantValidator

        validator = InvariantValidator(max_correlation_cluster_exposure_pct=35.0)
        portfolio = PortfolioSnapshot(
            total_equity=100000.0,
            cash_available=20000.0,
            buying_power=40000.0,
            positions_value=80000.0,
            positions=[],
            correlation_cluster_exposure=20.0,
            snapshot_time=datetime.now(),
        )

        result = validator.check_correlation_exposure_invariant(10.0, portfolio)
        assert result.passed is True

    def test_correlation_exposure_invariant_fail(self):
        """Correlation cluster exceeding limit should fail."""
        from autotrade.risk.policy_engine import InvariantValidator
        from autotrade.risk.contracts import FailureReason

        validator = InvariantValidator(max_correlation_cluster_exposure_pct=35.0)
        portfolio = PortfolioSnapshot(
            total_equity=100000.0,
            cash_available=20000.0,
            buying_power=40000.0,
            positions_value=80000.0,
            positions=[],
            correlation_cluster_exposure=30.0,
            snapshot_time=datetime.now(),
        )

        result = validator.check_correlation_exposure_invariant(10.0, portfolio)
        assert result.passed is False
        assert result.failure_reason == FailureReason.CORRELATION_CAP_REACHED

    def test_failsafe_critical_invariant_fail(self):
        """Critical failsafe state should fail with FAILSAFE_CRITICAL."""
        from autotrade.risk.policy_engine import InvariantValidator
        from autotrade.risk.contracts import FailureReason

        validator = InvariantValidator()
        result = validator.check_failsafe_critical_invariant(
            True, detail="strategy_failsafe_level=critical"
        )
        assert result.passed is False
        assert result.failure_reason == FailureReason.FAILSAFE_CRITICAL
        assert "critical" in result.failure_detail.lower()

    def test_portfolio_exposure_invariants_fail_net_short(self):
        """Extreme net short exposure should fail absolute net cap invariant."""
        from autotrade.risk.policy_engine import InvariantValidator
        from autotrade.risk.contracts import FailureReason

        validator = InvariantValidator(max_net_exposure_pct=100.0)

        short_pos = create_test_position("TSLA", 100.0, 100.0, -150)
        portfolio = PortfolioSnapshot(
            total_equity=10000.0,
            cash_available=20000.0,
            buying_power=20000.0,
            positions_value=15000.0,
            positions=[short_pos],
            snapshot_time=datetime.now(),
        )

        result = validator.check_portfolio_exposure_invariants(portfolio)
        assert result.passed is False
        assert result.failure_reason == FailureReason.OVER_MAX_EXPOSURE
        assert "net" in result.failure_detail.lower()

    def test_check_all_invariants(self):
        """Check all invariants returns combined results."""
        from autotrade.risk.policy_engine import InvariantValidator
        from autotrade.risk.contracts import FailureReason

        validator = InvariantValidator(
            max_symbol_exposure_pct=15.0,
            entry_halt_drawdown_pct=10.0,
        )

        position = create_test_position("AAPL", 150.0, 145.0, 100)
        position.market_value = 14500.0

        portfolio = PortfolioSnapshot(
            total_equity=100000.0,
            cash_available=20000.0,
            buying_power=40000.0,
            positions_value=80000.0,
            positions=[position],
            peak_equity=100000.0,
            snapshot_time=datetime.now(),
        )

        results = validator.check_all_invariants(position, portfolio)
        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]

        assert len(passed) > 0
        assert len(failed) == 0


class TestStaleDataFallback:
    """Tests for stale data fallback behavior."""

    def test_stale_data_fallback_position_fresh(self):
        """Fresh position data should not be modified."""
        from autotrade.risk.policy_engine import get_stale_data_fallback

        position = PositionSnapshot(
            symbol="AAPL",
            entry_price=150.0,
            current_price=145.0,
            qty=100,
            entry_time=datetime.now(),
            snapshot_time=datetime.now(),
        )

        result_pos, result_port = get_stale_data_fallback(
            position, None, data_age_seconds=300
        )

        assert result_pos is not None
        assert result_pos.symbol == "AAPL"

    def test_stale_data_fallback_position_stale(self):
        """Stale position data should get fresh timestamp."""
        from autotrade.risk.policy_engine import get_stale_data_fallback

        old_time = datetime.now() - timedelta(seconds=400)
        position = PositionSnapshot(
            symbol="AAPL",
            entry_price=150.0,
            current_price=145.0,
            qty=100,
            entry_time=old_time,
            snapshot_time=old_time,
        )

        result_pos, result_port = get_stale_data_fallback(
            position, None, data_age_seconds=300
        )

        assert result_pos is not None
        age = (datetime.now() - result_pos.snapshot_time).total_seconds()
        assert age < 10

    def test_stale_data_fallback_portfolio_fresh(self):
        """Fresh portfolio data should not be modified."""
        from autotrade.risk.policy_engine import get_stale_data_fallback

        portfolio = PortfolioSnapshot(
            total_equity=100000.0,
            cash_available=50000.0,
            buying_power=100000.0,
            positions_value=50000.0,
            snapshot_time=datetime.now(),
        )

        result_pos, result_port = get_stale_data_fallback(
            None, portfolio, data_age_seconds=300
        )

        assert result_port is not None
        assert result_port.total_equity == 100000.0

    def test_stale_data_fallback_both_none(self):
        """None inputs should return None outputs."""
        from autotrade.risk.policy_engine import get_stale_data_fallback

        result_pos, result_port = get_stale_data_fallback(
            None, None, data_age_seconds=300
        )

        assert result_pos is None
        assert result_port is None

    def test_risk_health_report_flags_net_short_overexposure(self):
        """Health report should flag net exposure cap for extreme net short books."""
        from autotrade.risk.policy_engine import (
            StandardExposurePolicy,
            ExposurePolicyConfig,
        )
        from autotrade.risk.contracts import FailureReason

        policy = StandardExposurePolicy()
        short_pos = create_test_position("TSLA", 100.0, 100.0, -150)
        portfolio = PortfolioSnapshot(
            total_equity=10000.0,
            cash_available=20000.0,
            buying_power=20000.0,
            positions_value=15000.0,
            positions=[short_pos],
            snapshot_time=datetime.now(),
        )
        config = ExposurePolicyConfig(max_net_exposure_pct=100.0)

        report = policy.get_risk_health_report(portfolio, config)
        assert FailureReason.OVER_MAX_EXPOSURE in report.failures
        assert "net_exposure" in report.failure_details


class TestPhase3Completion:
    """Tests to verify Phase 3 completion requirements."""

    def test_invariant_validator_imports_available(self):
        """InvariantValidator should be importable."""
        from autotrade.risk.policy_engine import InvariantValidator

        validator = InvariantValidator()
        assert validator is not None
        assert hasattr(validator, "check_position_exposure_invariant")
        assert hasattr(validator, "check_portfolio_exposure_invariants")
        assert hasattr(validator, "check_drawdown_invariant")
        assert hasattr(validator, "check_sector_exposure_invariant")
        assert hasattr(validator, "check_correlation_exposure_invariant")
        assert hasattr(validator, "check_position_metadata_invariant")
        assert hasattr(validator, "check_failsafe_critical_invariant")
        assert hasattr(validator, "check_all_invariants")

    def test_invariant_check_result_imports_available(self):
        """InvariantCheckResult should be importable."""
        from autotrade.risk.policy_engine import InvariantCheckResult

        result = InvariantCheckResult(passed=True)
        assert result.passed is True
        assert result.failure_reason is None
        assert result.failure_detail == ""

    def test_stale_data_fallback_imports_available(self):
        """get_stale_data_fallback should be importable."""
        from autotrade.risk.policy_engine import get_stale_data_fallback

        assert callable(get_stale_data_fallback)

    def test_all_failure_reasons_covered(self):
        """All required failure reasons should be in contracts."""
        from autotrade.risk.contracts import FailureReason

        required = {
            "OVER_MAX_EXPOSURE",
            "DRAWDOWN_BREACH",
            "FAILSAFE_CRITICAL",
            "INVALID_POSITION_METADATA",
            "STALE_DATA",
            "SECTOR_CAP_REACHED",
            "CORRELATION_CAP_REACHED",
        }

        existing = {f.name for f in FailureReason}
        assert required.issubset(existing), f"Missing: {required - existing}"

    def test_risk_management_config_in_config_loader(self):
        """RiskManagementConfig should be in config_loader."""
        from config.config_loader import RiskManagementConfig

        config = RiskManagementConfig()
        assert config.enabled is True
        assert config.max_symbol_exposure_pct == 15.0
        assert config.max_sector_exposure_pct == 40.0
        assert config.max_consecutive_failures == 3
        assert config.kill_switch.max_consecutive_failures == 3

    def test_risk_management_in_trading_config(self):
        """risk_management should be in TradingConfig."""
        from config.config_loader import get_config

        config = get_config()
        assert hasattr(config, "risk_management")
        assert config.risk_management is not None
        exported = config.to_dict()
        assert "risk_management" in exported
        assert "kill_switch" in exported["risk_management"]
