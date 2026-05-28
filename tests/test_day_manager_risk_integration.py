"""Phase 2 and Phase 4 integration tests for day-manager policy-engine adapter."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from autotrade.risk.contracts import (
    PositionPhase,
    RiskActionType,
    RiskDecision,
    PositionSnapshot,
    PortfolioSnapshot,
)

pytest.importorskip("alpaca.trading.client")

from autotrade.core.day_manager import DayManager


def _new_day_manager_stub() -> DayManager:
    dm = DayManager.__new__(DayManager)
    dm.signals = []
    dm.position_levels = {}
    dm.position_entries = {}
    dm.policy_risk_decisions = {}
    dm.last_account_equity = 100000.0
    dm._safe_float = DayManager._safe_float
    dm._now_utc = DayManager._now_utc
    dm._normalize_entry_time = DayManager._normalize_entry_time
    dm._hold_minutes = DayManager._hold_minutes.__get__(dm, DayManager)
    dm._effective_max_positions = lambda: 7
    return dm


def test_build_position_snapshot_for_policy_maps_core_fields():
    dm = _new_day_manager_stub()
    dm.signals = [
        {
            "ticker": "AAPL",
            "atr_percent": 2.4,
            "sector": "technology",
            "signal_family": "ts_momentum",
            "rsi": 62.0,
        }
    ]
    dm.position_entries["AAPL"] = datetime.now(timezone.utc) - timedelta(hours=2)

    position = SimpleNamespace(
        symbol="AAPL",
        avg_entry_price="100",
        current_price="95",
        qty="10",
        unrealized_plpc=-0.05,
        market_value="950",
    )

    snapshot = dm._build_position_snapshot_for_policy(position)

    assert snapshot is not None
    assert snapshot.symbol == "AAPL"
    assert snapshot.atr_pct == pytest.approx(2.4)
    assert snapshot.sector == "technology"
    assert snapshot.signal_family == "ts_momentum"
    assert snapshot.phase in (
        PositionPhase.LOCKED,
        PositionPhase.UNLOCKING,
        PositionPhase.UNLOCKED,
    )


def test_apply_policy_risk_overlay_enforces_exit():
    dm = _new_day_manager_stub()
    dm.policy_risk_decisions = {
        "AAPL": RiskDecision(
            action=RiskActionType.EXIT,
            size_delta=-10,
            confidence=0.95,
            reasons=["hard stop"],
            triggered_rules=["hard_stop"],
            skip_agents=True,
            hold_phase=PositionPhase.CRITICAL,
        )
    }

    position = SimpleNamespace(symbol="AAPL")
    health = {
        "score": 5,
        "signals": ["baseline hold"],
        "action": "hold",
        "pnl_pct": -9.0,
    }

    updated = dm._apply_policy_risk_overlay(position, health)

    assert updated["action"] == "exit"
    assert updated["score"] <= -40
    assert any("Policy engine EXIT" in s for s in updated["signals"])


class TestPhase4CompatibilityAdapters:
    """Phase 4 tests for compatibility adapters."""

    def test_action_plan_to_risk_decision_exit(self):
        from autotrade.risk.policy_engine import action_plan_to_risk_decision
        from autotrade.risk.risk_gate import RiskAction, ActionPlan, HoldPhase
        from datetime import datetime, timezone

        action_plan = ActionPlan(
            action=RiskAction.EXIT,
            size_delta=-10,
            confidence=0.95,
            reasons=["Hard stop breach"],
            triggered_rules=["hard_stop"],
            skip_agents=True,
            hold_phase=HoldPhase.CRITICAL,
            pdt_burn_required=False,
            conviction_score=30.0,
        )

        position = PositionSnapshot(
            symbol="AAPL",
            entry_price=100.0,
            current_price=90.0,
            qty=10,
            entry_time=datetime.now(timezone.utc),
        )

        decision = action_plan_to_risk_decision(action_plan, position)

        assert decision.action == RiskActionType.EXIT
        assert decision.size_delta == -10
        assert decision.confidence == 0.95
        assert "Hard stop breach" in decision.reasons
        assert "hard_stop" in decision.triggered_rules
        assert decision.skip_agents is True
        assert decision.hold_phase == PositionPhase.CRITICAL

    def test_action_plan_to_risk_decision_trim(self):
        from autotrade.risk.policy_engine import action_plan_to_risk_decision
        from autotrade.risk.risk_gate import RiskAction, ActionPlan, HoldPhase
        from datetime import datetime, timezone

        action_plan = ActionPlan(
            action=RiskAction.TRIM,
            size_delta=-5,
            confidence=0.80,
            reasons=["Trim threshold"],
            triggered_rules=["trim_threshold"],
            skip_agents=True,
            hold_phase=HoldPhase.LOCKED,
            pdt_burn_required=True,
            conviction_score=40.0,
        )

        position = PositionSnapshot(
            symbol="AAPL",
            entry_price=100.0,
            current_price=96.0,
            qty=10,
            entry_time=datetime.now(timezone.utc),
        )

        decision = action_plan_to_risk_decision(action_plan, position)

        assert decision.action == RiskActionType.TRIM
        assert decision.size_delta == -5
        assert decision.requires_pdt_burn is True

    def test_risk_decision_to_day_manager_action_exit(self):
        from autotrade.risk.policy_engine import risk_decision_to_day_manager_action

        decision = RiskDecision(
            action=RiskActionType.EXIT,
            size_delta=-10,
            confidence=0.95,
            reasons=["Hard stop"],
            triggered_rules=["hard_stop"],
            skip_agents=True,
            hold_phase=PositionPhase.CRITICAL,
        )

        result = risk_decision_to_day_manager_action(
            decision, current_action="hold", current_score=50
        )

        assert result["action"] == "exit"
        assert result["score"] == -40
        assert "Policy engine EXIT" in result["signals"][0]

    def test_risk_decision_to_day_manager_action_trim(self):
        from autotrade.risk.policy_engine import risk_decision_to_day_manager_action

        decision = RiskDecision(
            action=RiskActionType.TRIM,
            size_delta=-5,
            confidence=0.80,
            reasons=["Trim threshold"],
            triggered_rules=["trim_threshold"],
            skip_agents=True,
            hold_phase=PositionPhase.LOCKED,
        )

        result = risk_decision_to_day_manager_action(
            decision, current_action="hold", current_score=50
        )

        assert result["action"] == "trim"
        assert result["score"] == -20

    def test_risk_decision_to_day_manager_action_add(self):
        from autotrade.risk.policy_engine import risk_decision_to_day_manager_action

        decision = RiskDecision(
            action=RiskActionType.ADD,
            size_delta=5,
            confidence=0.85,
            reasons=["Higher conviction"],
            triggered_rules=[],
            skip_agents=False,
            hold_phase=PositionPhase.UNLOCKED,
        )

        result = risk_decision_to_day_manager_action(
            decision, current_action="hold", current_score=40
        )

        assert result["action"] == "add"
        assert result["score"] == 85

    def test_risk_decision_to_day_manager_action_preserves_hold(self):
        from autotrade.risk.policy_engine import risk_decision_to_day_manager_action

        decision = RiskDecision(
            action=RiskActionType.NONE,
            size_delta=0,
            confidence=0.0,
            reasons=["No trigger"],
            triggered_rules=[],
            skip_agents=False,
            hold_phase=PositionPhase.LOCKED,
        )

        result = risk_decision_to_day_manager_action(
            decision, current_action="hold", current_score=50
        )

        assert result["action"] == "hold"
        assert result["score"] == 50


class TestPDTAdapter:
    """Compatibility tests for the disabled PDT adapter."""

    def test_should_not_respect_pdt_locked_position(self):
        from autotrade.risk.policy_engine import PDTAdapter

        adapter = PDTAdapter(day_trades_remaining=5)

        decision = RiskDecision(
            action=RiskActionType.EXIT,
            size_delta=-10,
            confidence=0.95,
            reasons=[],
            triggered_rules=[],
            skip_agents=True,
            hold_phase=PositionPhase.LOCKED,
            requires_pdt_burn=True,
        )

        position = PositionSnapshot(
            symbol="AAPL",
            entry_price=100.0,
            current_price=90.0,
            qty=10,
            entry_time=datetime.now(timezone.utc),
            phase=PositionPhase.LOCKED,
        )

        assert adapter.should_respect_pdt(decision, position) is False

    def test_should_not_respect_pdt_unlocked_position(self):
        from autotrade.risk.policy_engine import PDTAdapter

        adapter = PDTAdapter(day_trades_remaining=5)

        decision = RiskDecision(
            action=RiskActionType.EXIT,
            size_delta=-10,
            confidence=0.95,
            reasons=[],
            triggered_rules=[],
            skip_agents=True,
            hold_phase=PositionPhase.UNLOCKED,
            requires_pdt_burn=True,
        )

        position = PositionSnapshot(
            symbol="AAPL",
            entry_price=100.0,
            current_price=90.0,
            qty=10,
            entry_time=datetime.now(timezone.utc),
            phase=PositionPhase.UNLOCKED,
        )

        assert adapter.should_respect_pdt(decision, position) is False

    def test_should_not_respect_pdt_critical_position(self):
        from autotrade.risk.policy_engine import PDTAdapter

        adapter = PDTAdapter(day_trades_remaining=5)

        decision = RiskDecision(
            action=RiskActionType.EXIT,
            size_delta=-10,
            confidence=0.95,
            reasons=[],
            triggered_rules=[],
            skip_agents=True,
            hold_phase=PositionPhase.CRITICAL,
            requires_pdt_burn=True,
        )

        position = PositionSnapshot(
            symbol="AAPL",
            entry_price=100.0,
            current_price=90.0,
            qty=10,
            entry_time=datetime.now(timezone.utc),
            phase=PositionPhase.CRITICAL,
        )

        assert adapter.should_respect_pdt(decision, position) is False

    def test_adjust_decision_for_pdt_leaves_exit_unchanged(self):
        from autotrade.risk.policy_engine import PDTAdapter

        adapter = PDTAdapter(day_trades_remaining=3)

        decision = RiskDecision(
            action=RiskActionType.EXIT,
            size_delta=-10,
            confidence=0.95,
            reasons=["Hard stop"],
            triggered_rules=["hard_stop"],
            skip_agents=True,
            hold_phase=PositionPhase.LOCKED,
            requires_pdt_burn=True,
        )

        position = PositionSnapshot(
            symbol="AAPL",
            entry_price=100.0,
            current_price=90.0,
            qty=10,
            entry_time=datetime.now(timezone.utc),
            phase=PositionPhase.LOCKED,
        )

        adjusted = adapter.adjust_decision_for_pdt(decision, position)

        assert adjusted.action == RiskActionType.EXIT
        assert adjusted is decision

    def test_can_rotate_unlocked(self):
        from autotrade.risk.policy_engine import PDTAdapter

        adapter = PDTAdapter(day_trades_remaining=5)

        position = PositionSnapshot(
            symbol="AAPL",
            entry_price=100.0,
            current_price=95.0,
            qty=10,
            entry_time=datetime.now(timezone.utc),
            phase=PositionPhase.UNLOCKED,
        )

        portfolio = PortfolioSnapshot(
            total_equity=100000.0,
            cash_available=50000.0,
            buying_power=100000.0,
            positions_value=50000.0,
            positions=[position],
        )

        allowed, reason = adapter.can_rotate(position, 5000.0, portfolio)

        assert allowed is True
        assert reason is None


class TestFailsafeAdapter:
    """Phase 4 tests for failsafe adapter."""

    def test_should_halt_entries_critical_level(self):
        from autotrade.risk.policy_engine import FailsafeAdapter

        adapter = FailsafeAdapter(
            failsafe_level="critical",
            halt_new_entries=False,
            max_positions=7,
        )

        assert adapter.should_halt_entries() is True

    def test_should_halt_entries_halt_flag(self):
        from autotrade.risk.policy_engine import FailsafeAdapter

        adapter = FailsafeAdapter(
            failsafe_level="normal",
            halt_new_entries=True,
            max_positions=7,
        )

        assert adapter.should_halt_entries() is True

    def test_check_entry_allowed_normal(self):
        from autotrade.risk.policy_engine import FailsafeAdapter

        adapter = FailsafeAdapter(
            failsafe_level="normal",
            halt_new_entries=False,
            max_positions=7,
        )

        portfolio = PortfolioSnapshot(
            total_equity=100000.0,
            cash_available=50000.0,
            buying_power=100000.0,
            positions_value=30000.0,
            positions=[],
            current_drawdown_pct=5.0,
        )

        allowed, reason = adapter.check_entry_allowed("AAPL", portfolio, 60.0)

        assert allowed is True
        assert reason is None

    def test_check_entry_allowed_max_positions_reached(self):
        from autotrade.risk.policy_engine import FailsafeAdapter

        adapter = FailsafeAdapter(
            failsafe_level="normal",
            halt_new_entries=False,
            max_positions=3,
        )

        positions = [
            PositionSnapshot(
                symbol=f"SYM{i}",
                entry_price=100.0,
                current_price=100.0,
                qty=10,
                entry_time=datetime.now(timezone.utc),
            )
            for i in range(3)
        ]

        portfolio = PortfolioSnapshot(
            total_equity=100000.0,
            cash_available=50000.0,
            buying_power=100000.0,
            positions_value=30000.0,
            positions=positions,
        )

        allowed, reason = adapter.check_entry_allowed("AAPL", portfolio, 60.0)

        assert allowed is False
        assert "Max positions" in reason

    def test_check_entry_allowed_high_drawdown(self):
        from autotrade.risk.policy_engine import FailsafeAdapter

        adapter = FailsafeAdapter(
            failsafe_level="normal",
            halt_new_entries=False,
            max_positions=7,
        )

        portfolio = PortfolioSnapshot(
            total_equity=100000.0,
            cash_available=50000.0,
            buying_power=100000.0,
            positions_value=30000.0,
            positions=[],
            current_drawdown_pct=15.0,
        )

        allowed, reason = adapter.check_entry_allowed("AAPL", portfolio, 60.0)

        assert allowed is False
        assert "drawdown" in reason.lower()

    def test_get_max_position_size_elevated(self):
        from autotrade.risk.policy_engine import FailsafeAdapter

        adapter = FailsafeAdapter(
            failsafe_level="elevated",
            halt_new_entries=False,
            max_positions=7,
        )

        portfolio = PortfolioSnapshot(
            total_equity=100000.0,
            cash_available=50000.0,
            buying_power=100000.0,
            positions_value=30000.0,
            positions=[],
        )

        size = adapter.get_max_position_size(portfolio, risk_per_trade=0.02)

        assert size == 100000.0 * 0.02 * 5 * 0.75


class TestDayManagerAdapter:
    """Phase 4 tests for DayManager adapter."""

    def test_apply_risk_overlay_exit(self):
        from autotrade.risk.policy_engine import DayManagerAdapter

        adapter = DayManagerAdapter()
        adapter.set_risk_decisions(
            {
                "AAPL": RiskDecision(
                    action=RiskActionType.EXIT,
                    size_delta=-10,
                    confidence=0.95,
                    reasons=["Hard stop"],
                    triggered_rules=["hard_stop"],
                    skip_agents=True,
                    hold_phase=PositionPhase.CRITICAL,
                )
            }
        )

        position = PositionSnapshot(
            symbol="AAPL",
            entry_price=100.0,
            current_price=90.0,
            qty=10,
            entry_time=datetime.now(timezone.utc),
        )

        health = {
            "action": "hold",
            "score": 50,
            "signals": ["baseline"],
        }

        result = adapter.apply_risk_overlay(position, health)

        assert result["action"] == "exit"
        assert result["score"] == -40

    def test_apply_risk_overlay_no_decision(self):
        from autotrade.risk.policy_engine import DayManagerAdapter

        adapter = DayManagerAdapter()
        adapter.set_risk_decisions({})

        position = PositionSnapshot(
            symbol="AAPL",
            entry_price=100.0,
            current_price=100.0,
            qty=10,
            entry_time=datetime.now(timezone.utc),
        )

        health = {
            "action": "hold",
            "score": 50,
            "signals": ["baseline"],
        }

        result = adapter.apply_risk_overlay(position, health)

        assert result["action"] == "hold"
        assert result["score"] == 50

    def test_check_pdt_constraints(self):
        from autotrade.risk.policy_engine import DayManagerAdapter, PDTAdapter

        pdt_adapter = PDTAdapter(day_trades_remaining=3)
        adapter = DayManagerAdapter(pdt_adapter=pdt_adapter)

        decision = RiskDecision(
            action=RiskActionType.EXIT,
            size_delta=-10,
            confidence=0.95,
            reasons=["Hard stop"],
            triggered_rules=["hard_stop"],
            skip_agents=True,
            hold_phase=PositionPhase.LOCKED,
            requires_pdt_burn=True,
        )

        position = PositionSnapshot(
            symbol="AAPL",
            entry_price=100.0,
            current_price=90.0,
            qty=10,
            entry_time=datetime.now(timezone.utc),
            phase=PositionPhase.LOCKED,
        )

        adjusted = adapter.check_pdt_constraints(decision, position)

        assert adjusted.action == RiskActionType.EXIT

    def test_get_compatible_action(self):
        from autotrade.risk.policy_engine import DayManagerAdapter

        adapter = DayManagerAdapter()
        adapter.set_risk_decisions(
            {
                "AAPL": RiskDecision(
                    action=RiskActionType.EXIT,
                    size_delta=-10,
                    confidence=0.95,
                    reasons=["Hard stop"],
                    triggered_rules=["hard_stop"],
                    skip_agents=True,
                    hold_phase=PositionPhase.CRITICAL,
                )
            }
        )

        action, score, signals = adapter.get_compatible_action("AAPL", "hold", 50)

        assert action == "exit"
        assert score == -40
        assert len(signals) > 0

    def test_create_day_manager_adapter_factory(self):
        from autotrade.risk.policy_engine import create_day_manager_adapter

        adapter = create_day_manager_adapter(
            day_trades_remaining=5,
            failsafe_level="normal",
            halt_new_entries=False,
            max_positions=7,
        )

        assert adapter is not None
        assert adapter.pdt_adapter is not None
        assert adapter.failsafe_adapter is not None


class TestPhase4Completion:
    """Phase 4 completion verification tests."""

    def test_action_semantics_preserved_exit(self):
        from autotrade.risk.policy_engine import risk_decision_to_day_manager_action

        decision = RiskDecision(
            action=RiskActionType.EXIT,
            size_delta=-10,
            confidence=0.95,
            reasons=["hard stop"],
            triggered_rules=["hard_stop"],
            skip_agents=True,
            hold_phase=PositionPhase.CRITICAL,
        )

        result = risk_decision_to_day_manager_action(
            decision, current_action="hold", current_score=50
        )

        assert result["action"] == "exit"
        assert result["score"] <= -40
        assert "Policy engine EXIT" in result["signals"][0]

    def test_action_semantics_preserved_trim(self):
        from autotrade.risk.policy_engine import risk_decision_to_day_manager_action

        decision = RiskDecision(
            action=RiskActionType.TRIM,
            size_delta=-5,
            confidence=0.80,
            reasons=["trim threshold"],
            triggered_rules=["trim_threshold"],
            skip_agents=True,
            hold_phase=PositionPhase.LOCKED,
        )

        result = risk_decision_to_day_manager_action(
            decision, current_action="hold", current_score=50
        )

        assert result["action"] == "trim"
        assert result["score"] <= -20

    def test_pdt_behavior_disabled(self):
        from autotrade.risk.policy_engine import PDTAdapter

        adapter = PDTAdapter(day_trades_remaining=0)

        position = PositionSnapshot(
            symbol="AAPL",
            entry_price=100.0,
            current_price=90.0,
            qty=10,
            entry_time=datetime.now(timezone.utc),
            phase=PositionPhase.LOCKED,
        )

        portfolio = PortfolioSnapshot(
            total_equity=100000.0,
            cash_available=50000.0,
            buying_power=100000.0,
            positions_value=50000.0,
            positions=[position],
        )

        allowed, reason = adapter.can_rotate(position, 5000.0, portfolio)

        assert allowed is True
        assert reason is None

    def test_failsafe_behavior_preserved(self):
        from autotrade.risk.policy_engine import FailsafeAdapter

        adapter = FailsafeAdapter(
            failsafe_level="critical",
            halt_new_entries=True,
            max_positions=7,
        )

        portfolio = PortfolioSnapshot(
            total_equity=100000.0,
            cash_available=50000.0,
            buying_power=100000.0,
            positions_value=30000.0,
            positions=[],
        )

        allowed, reason = adapter.check_entry_allowed("AAPL", portfolio, 60.0)

        assert allowed is False
        assert "Failsafe" in reason

    def test_policy_engine_no_alpaca_imports(self):
        import autotrade.risk.policy_engine as pe

        assert not hasattr(pe, "TradingClient")
        assert not hasattr(pe, "alpaca")

    def test_phase4_imports_available(self):
        from autotrade.risk.policy_engine import (
            action_plan_to_risk_decision,
            risk_decision_to_day_manager_action,
            PDTAdapter,
            FailsafeAdapter,
            DayManagerAdapter,
            create_day_manager_adapter,
        )

        assert action_plan_to_risk_decision is not None
        assert risk_decision_to_day_manager_action is not None
        assert PDTAdapter is not None
        assert FailsafeAdapter is not None
        assert DayManagerAdapter is not None
        assert create_day_manager_adapter is not None
