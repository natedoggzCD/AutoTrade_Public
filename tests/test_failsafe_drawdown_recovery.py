import pytest
from datetime import datetime
from unittest.mock import MagicMock

from autotrade.risk.strategy_failsafe import (
    StrategyFailsafeManager,
    StrategyFailsafeSnapshot,
)


@pytest.fixture
def failsafe_manager(tmp_path):
    state_path = tmp_path / "strategy_failsafe_state.json"
    config = MagicMock()
    # Default thresholds from trading_config.yaml
    config.critical_drawdown_pct = 10.0
    config.failing_drawdown_pct = 7.0
    config.degraded_drawdown_pct = 5.0
    config.recovery_days = 2

    # Validation thresholds
    config.critical_win_rate = 0.2
    config.failing_win_rate = 0.25
    config.degraded_win_rate = 0.3
    config.failing_profit_factor = 0.5
    config.degraded_profit_factor = 0.7

    # Critical profile
    critical_profile = MagicMock()
    critical_profile.halt_new_entries = True
    config.critical = critical_profile

    manager = StrategyFailsafeManager(state_path=state_path)
    manager.config = config
    return manager


def test_failsafe_recovery_with_fix(failsafe_manager):
    """
    This test will verify the fix: if validation is healthy and sample size is sufficient,
    the stale peak should be rebased.
    """
    # 1. Setup stale peak state
    snapshot = StrategyFailsafeSnapshot(
        level="critical",
        peak_equity=210000.0,
        current_equity=88000.0,
        drawdown_pct=58.1,
        validation_status="HEALTHY",
        win_rate=0.56,
        profit_factor=1.38,
        sample_size=400,
        updated_at=datetime.now().isoformat(),
    )
    failsafe_manager.save_snapshot(snapshot)

    # 2. Update with healthy validation
    # With the fix, this should trigger a rebase
    new_snapshot = failsafe_manager.update_from_strategy_validation(
        strategy_validation={
            "status": "HEALTHY",
            "win_rate": 0.56,
            "profit_factor": 1.38,
            "sample_size": 400,
        },
        equity=88000.0,
    )

    # Expected behavior after fix:
    # 1. Peak is rebased to current equity
    # 2. Level becomes 'normal' (or degraded if recovery gate triggers)
    # 3. Entries are no longer halted
    assert new_snapshot.peak_equity == 88000.0
    assert new_snapshot.drawdown_pct == 0.0
    assert new_snapshot.level in ("normal", "degraded")
    assert new_snapshot.reason == "stale_drawdown_peak_rebased"
