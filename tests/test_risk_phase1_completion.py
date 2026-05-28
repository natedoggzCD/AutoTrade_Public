from dataclasses import is_dataclass
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from autotrade.risk.contracts import (
    PortfolioSnapshot,
    PositionPhase,
    PositionSnapshot,
    RiskDecision,
    RiskHealthReport,
    RotationPlan,
    portfolio_state_to_snapshot,
    position_state_to_snapshot,
)
from autotrade.risk.interfaces import RiskPolicy, RotationPolicy, SizingPolicy
from autotrade.risk.position_state import PortfolioState, create_position_state


def test_phase1_required_dataclasses_exist():
    assert is_dataclass(PositionSnapshot)
    assert is_dataclass(PortfolioSnapshot)
    assert is_dataclass(RiskDecision)
    assert is_dataclass(RotationPlan)
    assert is_dataclass(RiskHealthReport)


def test_phase1_required_protocols_exist():
    assert "evaluate" in RiskPolicy.__dict__
    assert "should_skip_agents" in RiskPolicy.__dict__
    assert "calculate_size" in SizingPolicy.__dict__
    assert "can_increase_position" in SizingPolicy.__dict__
    assert "find_rotation_candidates" in RotationPolicy.__dict__
    assert "evaluate_rotation" in RotationPolicy.__dict__
    assert "apply_rotation_constraints" in RotationPolicy.__dict__


def test_portfolio_snapshot_net_exposure_matches_long_only_book():
    portfolio = PortfolioSnapshot(
        total_equity=10_000.0,
        cash_available=5_000.0,
        buying_power=5_000.0,
        positions_value=5_000.0,
    )

    assert portfolio.gross_exposure_pct == 50.0
    assert portfolio.net_exposure_pct == 50.0


def test_portfolio_state_conversion_uses_equity_denominator_for_exposure():
    entry_time = datetime.now() - timedelta(hours=30)
    pos = create_position_state(
        symbol="AAPL",
        entry_price=100.0,
        current_price=110.0,
        qty=20,
        entry_time=entry_time,
    )
    pos.sector = "technology"
    pos.signal_family = "ts_momentum"

    state = PortfolioState(
        total_equity=10_000.0,
        cash_available=8_000.0,
        buying_power=8_000.0,
        positions_value=2_200.0,
    )
    state.add_position(pos)

    snap = portfolio_state_to_snapshot(state)
    assert snap.sector_exposure["technology"] == pytest.approx(22.0)
    assert snap.signal_family_exposure["ts_momentum"] == pytest.approx(22.0)


def test_position_state_conversion_handles_unknown_hold_phase():
    unknown_phase = SimpleNamespace(value="invalid_phase")
    pos = SimpleNamespace(
        symbol="MSFT",
        entry_price=100.0,
        current_price=99.0,
        qty=10,
        entry_time=datetime.now() - timedelta(hours=2),
        pnl_dollars=-10.0,
        pnl_pct=-1.0,
        atr_pct=2.0,
        high_since_entry=101.0,
        low_since_entry=98.0,
        hold_minutes=120.0,
        rsi=48.0,
        s1_price=97.0,
        r1_price=102.0,
        hold_phase=unknown_phase,
    )

    snap = position_state_to_snapshot(pos)
    assert snap.phase == PositionPhase.LOCKED
