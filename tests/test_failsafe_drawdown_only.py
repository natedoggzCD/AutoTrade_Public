"""Verify failsafe drawdown gates use a daily session baseline."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import autotrade.risk.strategy_failsafe as strategy_failsafe_mod
from autotrade.risk.strategy_failsafe import StrategyFailsafeManager
from autotrade.risk.strategy_failsafe import StrategyFailsafeSnapshot
from config.config_loader import get_config


def _temp_state(payload):
    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, mode="w", encoding="utf-8"
    ) as f:
        json.dump(payload, f)
        return Path(f.name)


def test_config_still_exposes_drawdown_fields_for_compatibility():
    config = get_config(reload=True)
    assert hasattr(config.strategy_failsafe, "failing_drawdown_pct")
    assert config.strategy_failsafe.failing_drawdown_pct == 7.0


def test_classify_level_uses_validation_health():
    mgr = StrategyFailsafeManager()

    level, reason = mgr._classify_level("HEALTHY", 0.60, 1.20)
    assert level == "normal"
    assert "validation healthy" in reason

    level, reason = mgr._classify_level("WARNING", 0.60, 1.20)
    assert level == "degraded"
    assert "validation degraded" in reason

    level, reason = mgr._classify_level("FAILING", 0.20, 0.40)
    assert level == "failing"
    assert "validation failing" in reason

    level, reason = mgr._classify_level("CRITICAL", 0.10, 0.40)
    assert level == "critical"
    assert "validation critical" in reason


def test_healthy_validation_drawdown_does_not_hard_halt_entries():
    tmp_path = _temp_state(
        {
            "level": "normal",
            "previous_level": "normal",
            "reason": "test",
            "source": "test",
            "current_equity": 90_000.0,
            "peak_equity": 100_000.0,
            "drawdown_pct": 10.0,
            "sample_size": 0,
        }
    )
    try:
        mgr = StrategyFailsafeManager(state_path=tmp_path)
        snap = mgr.update_from_strategy_validation(
            strategy_validation={
                "status": "HEALTHY",
                "recent_win_rate": 0.60,
                "recent_profit_factor": 1.5,
                "sample_size": 100,
            },
            equity=90_000.0,
            source="test",
        )
        assert snap.level == "normal"
        assert snap.halt_new_entries is False
        assert "validation healthy" in snap.reason.lower()
        assert round(snap.drawdown_pct, 2) == 10.0
        assert snap.session_drawdown_pct == 0.0
        assert snap.drawdown_anchor_date == ""
        assert snap.drawdown_anchor_equity == 0.0
    finally:
        tmp_path.unlink(missing_ok=True)


def test_validation_critical_still_hard_halts_with_peak_drawdown():
    tmp_path = _temp_state(
        {
            "level": "normal",
            "previous_level": "normal",
            "reason": "test",
            "source": "test",
            "current_equity": 90_000.0,
            "peak_equity": 100_000.0,
            "drawdown_pct": 10.0,
            "sample_size": 0,
        }
    )
    try:
        mgr = StrategyFailsafeManager(state_path=tmp_path)
        snap = mgr.update_from_strategy_validation(
            strategy_validation={
                "status": "CRITICAL",
                "recent_win_rate": 0.10,
                "recent_profit_factor": 0.4,
                "sample_size": 100,
            },
            equity=90_000.0,
            source="test",
        )
        assert snap.level == "critical"
        assert snap.halt_new_entries is True
        assert "validation critical" in snap.reason.lower()
    finally:
        tmp_path.unlink(missing_ok=True)


def test_update_equity_only_applies_drawdown_gate(monkeypatch):
    class _FrozenDateTime:
        _now = datetime(2026, 5, 1, 10, 30, 0)

        @classmethod
        def now(cls):
            return cls._now

        @classmethod
        def fromisoformat(cls, value):
            return datetime.fromisoformat(value)

    monkeypatch.setattr(strategy_failsafe_mod, "datetime", _FrozenDateTime)
    today_key = "2026-05-01"
    tmp_path = _temp_state(
        {
            "level": "degraded",
            "previous_level": "normal",
            "reason": "validation degraded: status=WARNING wr=0.60 pf=1.20",
            "source": "test",
            "validation_status": "WARNING",
            "win_rate": 0.60,
            "profit_factor": 1.20,
            "sample_size": 50,
            "current_equity": 99_000.0,
            "peak_equity": 100_000.0,
            "drawdown_pct": 1.0,
            "drawdown_anchor_date": today_key,
            "drawdown_anchor_equity": 100_000.0,
        }
    )
    try:
        mgr = StrategyFailsafeManager(state_path=tmp_path)
        snap = mgr.update_equity_only(equity=85_000.0, source="runtime")
        assert snap.level == "critical"
        assert snap.halt_new_entries is True
        assert "drawdown critical" in snap.reason.lower()
        assert round(snap.drawdown_pct, 2) == 15.0
    finally:
        tmp_path.unlink(missing_ok=True)


def test_update_equity_only_resets_drawdown_anchor_on_new_session_day(monkeypatch):
    class _FrozenDateTime:
        _now = datetime(2026, 5, 1, 9, 30, 0)

        @classmethod
        def now(cls):
            return cls._now

        @classmethod
        def fromisoformat(cls, value):
            return datetime.fromisoformat(value)

    monkeypatch.setattr(strategy_failsafe_mod, "datetime", _FrozenDateTime)
    tmp_path = _temp_state(
        {
            "level": "degraded",
            "previous_level": "normal",
            "reason": "validation degraded: status=WARNING wr=0.60 pf=1.20",
            "source": "test",
            "validation_status": "WARNING",
            "win_rate": 0.60,
            "profit_factor": 1.20,
            "sample_size": 50,
            "current_equity": 99_000.0,
            "peak_equity": 100_000.0,
            "drawdown_pct": 1.0,
            "drawdown_anchor_date": "2026-05-01",
            "drawdown_anchor_equity": 100_000.0,
        }
    )
    try:
        mgr = StrategyFailsafeManager(state_path=tmp_path)
        day_one = mgr.update_equity_only(equity=99_000.0, source="runtime")
        assert round(day_one.drawdown_pct, 2) == 1.0

        _FrozenDateTime._now = datetime(2026, 5, 4, 9, 30, 0)
        day_two = mgr.update_equity_only(equity=98_500.0, source="runtime")
        assert day_two.drawdown_anchor_date == "2026-05-04"
        assert day_two.drawdown_anchor_equity == 98_500.0
        assert round(day_two.drawdown_pct, 2) == 1.5
        assert day_two.session_drawdown_pct == 0.0
    finally:
        tmp_path.unlink(missing_ok=True)


def test_load_snapshot_clears_non_trading_session_anchor():
    tmp_path = _temp_state(
        {
            "level": "normal",
            "previous_level": "critical",
            "reason": "validation healthy",
            "source": "pm_workflow",
            "validation_status": "HEALTHY",
            "win_rate": 0.55,
            "profit_factor": 1.8,
            "sample_size": 100,
            "current_equity": 88_900.0,
            "peak_equity": 100_000.0,
            "drawdown_pct": 11.1,
            "session_drawdown_pct": 11.1,
            "drawdown_anchor_date": "2026-05-17",
            "drawdown_anchor_equity": 100_000.0,
        }
    )
    try:
        mgr = StrategyFailsafeManager(state_path=tmp_path)
        snap = mgr.load_snapshot()
        assert snap.level == "normal"
        assert snap.halt_new_entries is False
        assert snap.peak_drawdown_pct == 11.1
        assert snap.drawdown_pct == 11.1
        assert snap.session_drawdown_pct == 0.0
        assert snap.drawdown_anchor_date == ""
        assert snap.drawdown_anchor_equity == 0.0
    finally:
        tmp_path.unlink(missing_ok=True)


def test_load_snapshot_normalizes_legacy_drawdown_reason():
    tmp_path = _temp_state(
        {
            "level": "degraded",
            "previous_level": "normal",
            "reason": "auto-escalated from normal: drawdown 5.3% >= 5.0%",
            "source": "post_market_reflect",
            "validation_status": "HEALTHY",
            "win_rate": 0.53,
            "profit_factor": 1.27,
            "sample_size": 1730,
            "current_equity": 92_034.46,
            "peak_equity": 97_140.54,
            "drawdown_pct": 5.26,
        }
    )
    try:
        mgr = StrategyFailsafeManager(state_path=tmp_path)
        snap = mgr.load_snapshot()
        assert snap.level == "normal"
        assert snap.reason.startswith("validation healthy")
    finally:
        tmp_path.unlink(missing_ok=True)


def test_load_snapshot_resets_stale_legacy_critical_peak_when_validation_is_healthy():
    tmp_path = _temp_state(
        {
            "level": "critical",
            "previous_level": "critical",
            "reason": "drawdown critical: 57.97% >= 10.00%",
            "source": "day_manager",
            "validation_status": "HEALTHY",
            "win_rate": 0.564,
            "profit_factor": 1.38,
            "sample_size": 434,
            "current_equity": 88_669.74,
            "peak_equity": 210_973.89,
            "drawdown_pct": 57.97,
        }
    )
    try:
        mgr = StrategyFailsafeManager(state_path=tmp_path)
        snap = mgr.load_snapshot()

        assert snap.level == "normal"
        assert snap.halt_new_entries is False
        assert snap.max_positions > 0
        assert snap.reason == "legacy_drawdown_peak_reset"
        assert snap.peak_equity == snap.current_equity
        assert snap.drawdown_pct == 0.0
        assert snap.session_drawdown_pct == 0.0

        saved = json.loads(tmp_path.read_text(encoding="utf-8"))
        assert saved["reason"] == "legacy_drawdown_peak_reset"
    finally:
        tmp_path.unlink(missing_ok=True)


def test_update_validation_rebases_stale_drawdown_peak_without_legacy_reason():
    tmp_path = _temp_state(
        {
            "level": "critical",
            "previous_level": "critical",
            "reason": "validation healthy but historical peak stale",
            "source": "day_manager",
            "validation_status": "HEALTHY",
            "win_rate": 0.564,
            "profit_factor": 1.38,
            "sample_size": 434,
            "current_equity": 88_669.74,
            "peak_equity": 210_973.89,
            "drawdown_pct": 57.97,
        }
    )
    try:
        mgr = StrategyFailsafeManager(state_path=tmp_path)
        snap = mgr.update_from_strategy_validation(
            strategy_validation={
                "status": "HEALTHY",
                "recent_win_rate": 0.564,
                "recent_profit_factor": 1.38,
                "sample_size": 434,
            },
            equity=88_669.74,
            source="day_manager",
            as_of_date="2026-04-29",
        )

        assert snap.level != "critical"
        assert snap.halt_new_entries is False
        assert snap.reason == "stale_drawdown_peak_rebased"
        assert snap.peak_equity == snap.current_equity
        assert snap.drawdown_pct == 0.0
        assert snap.session_drawdown_pct == 0.0
    finally:
        tmp_path.unlink(missing_ok=True)


def test_stale_peak_rebase_guard_requires_all_predicates():
    mgr = StrategyFailsafeManager()
    base = StrategyFailsafeSnapshot(
        level="normal",
        validation_status="HEALTHY",
        win_rate=0.55,
        profit_factor=1.25,
        sample_size=30,
        current_equity=80_000.0,
        peak_equity=100_000.0,
        drawdown_pct=20.0,
    )

    assert mgr._should_rebase_stale_drawdown_peak(base, "normal") is True

    low_sample = StrategyFailsafeSnapshot(**{**base.to_dict(), "sample_size": 29})
    assert mgr._should_rebase_stale_drawdown_peak(low_sample, "normal") is False

    no_wr = StrategyFailsafeSnapshot(**{**base.to_dict(), "win_rate": 0.0})
    assert mgr._should_rebase_stale_drawdown_peak(no_wr, "normal") is False

    low_pf = StrategyFailsafeSnapshot(**{**base.to_dict(), "profit_factor": 0.99})
    assert mgr._should_rebase_stale_drawdown_peak(low_pf, "normal") is False

    below_stale_dd = StrategyFailsafeSnapshot(
        **{
            **base.to_dict(),
            "current_equity": 87_800.0,
            "drawdown_pct": 12.2,
        }
    )
    assert mgr._should_rebase_stale_drawdown_peak(below_stale_dd, "normal") is False


def test_operator_override_clears_failing_halt_with_reason_lineage():
    tmp_path = _temp_state(
        {
            "level": "failing",
            "previous_level": "degraded",
            "reason": "validation failing: status=FAILING wr=0.24 pf=0.60",
            "source": "day_manager",
            "validation_status": "FAILING",
            "win_rate": 0.24,
            "profit_factor": 0.6,
            "sample_size": 100,
            "current_equity": 95_000.0,
            "peak_equity": 100_000.0,
            "drawdown_pct": 5.0,
        }
    )
    try:
        mgr = StrategyFailsafeManager(state_path=tmp_path)
        snap = mgr.apply_operator_override(
            reason="no_trade_60m_with_capacity",
            source="autonomous_agent.market_hours",
            session_key="2026-04-30",
        )

        assert snap.operator_override_active is True
        assert snap.halt_new_entries is False
        assert snap.original_halt_reason.startswith("validation failing")
        assert snap.override_reason == "no_trade_60m_with_capacity"
        assert snap.override_session_key == "2026-04-30"
        reloaded = mgr.load_snapshot()
        assert reloaded.halt_new_entries is False
        assert reloaded.operator_override_active is True
    finally:
        tmp_path.unlink(missing_ok=True)


def test_operator_override_does_not_bypass_critical_state():
    tmp_path = _temp_state(
        {
            "level": "critical",
            "previous_level": "failing",
            "reason": "validation critical: status=CRITICAL wr=0.10 pf=0.40",
            "source": "day_manager",
            "validation_status": "CRITICAL",
            "win_rate": 0.10,
            "profit_factor": 0.4,
            "sample_size": 120,
            "current_equity": 80_000.0,
            "peak_equity": 100_000.0,
            "drawdown_pct": 20.0,
        }
    )
    try:
        mgr = StrategyFailsafeManager(state_path=tmp_path)
        snap = mgr.apply_operator_override(
            reason="no_trade_60m_with_capacity",
            source="autonomous_agent.market_hours",
            session_key="2026-04-30",
        )

        assert snap.level == "critical"
        assert snap.halt_new_entries is True
        assert snap.operator_override_active is False
    finally:
        tmp_path.unlink(missing_ok=True)


def test_error_cascade_critical_latch_steps_down_after_quiet_window(monkeypatch):
    class _FrozenDateTime:
        _now = datetime(2026, 4, 30, 10, 0, 0)

        @classmethod
        def now(cls):
            return cls._now

        @classmethod
        def fromisoformat(cls, value):
            return datetime.fromisoformat(value)

    monkeypatch.setattr(strategy_failsafe_mod, "datetime", _FrozenDateTime)

    mgr = StrategyFailsafeManager(state_path=_temp_state({}))
    try:
        snap = StrategyFailsafeSnapshot(
            level="critical",
            previous_level="normal",
            reason="runtime_error_cascade",
            halt_lineage_cause="error_cascade",
            halt_lineage_set_at=_FrozenDateTime.now().isoformat(),
            error_cascade_last_error_at=_FrozenDateTime.now().isoformat(),
            error_cascade_last_stepdown_at=_FrozenDateTime.now().isoformat(),
            drawdown_pct=0.0,
        )

        _FrozenDateTime._now = datetime(2026, 4, 30, 10, 20, 0)
        snap = mgr._apply_error_cascade_recovery(snap)
        assert snap.level == "critical"
        assert snap.error_cascade_quiet_cycles == 1

        _FrozenDateTime._now = datetime(2026, 4, 30, 10, 21, 0)
        snap = mgr._apply_error_cascade_recovery(snap)
        assert snap.level == "failing"
        assert snap.reason == "error_cascade_stepdown:critical_to_failing"

        _FrozenDateTime._now = datetime(2026, 4, 30, 10, 40, 0)
        snap = mgr._apply_error_cascade_recovery(snap)
        assert snap.level == "normal"
        assert snap.reason == "error_cascade_stepdown:failing_to_normal"
        assert snap.halt_lineage_cause == ""
        assert snap.halt_lineage_cleared_at != ""
    finally:
        mgr.state_path.unlink(missing_ok=True)


def test_error_cascade_recovery_never_auto_clears_drawdown_critical():
    mgr = StrategyFailsafeManager(state_path=_temp_state({}))
    try:
        snap = StrategyFailsafeSnapshot(
            level="critical",
            previous_level="failing",
            reason="runtime_error_cascade",
            halt_lineage_cause="error_cascade",
            halt_lineage_set_at=datetime(2026, 4, 30, 10, 0, 0).isoformat(),
            error_cascade_last_error_at=datetime(2026, 4, 30, 10, 0, 0).isoformat(),
            error_cascade_last_stepdown_at=datetime(2026, 4, 30, 10, 0, 0).isoformat(),
            drawdown_pct=25.0,
            session_drawdown_pct=25.0,
        )

        stepped = mgr._apply_error_cascade_recovery(snap)
        assert stepped.level == "critical"
        assert stepped.halt_lineage_cause == "drawdown"
    finally:
        mgr.state_path.unlink(missing_ok=True)


def test_failsafe_snapshot_persists_halt_lineage_fields():
    tmp_path = _temp_state({})
    try:
        mgr = StrategyFailsafeManager(state_path=tmp_path)
        anchor_ts = datetime.now().replace(microsecond=0).isoformat()
        snapshot = StrategyFailsafeSnapshot(
            level="critical",
            reason="runtime_error_cascade",
            halt_lineage_cause="error_cascade",
            halt_lineage_set_at=anchor_ts,
            halt_lineage_cleared_at="2026-04-30T11:00:00",
            halt_lineage_clear_reason="error_cascade_auto_recovered",
            error_cascade_last_error_at=anchor_ts,
            error_cascade_quiet_cycles=2,
            error_cascade_recovery_stage="normal",
            error_cascade_last_stepdown_at=anchor_ts,
        )
        mgr.save_snapshot(snapshot)
        reloaded = mgr.load_snapshot()

        assert reloaded.halt_lineage_cause == "error_cascade"
        assert reloaded.halt_lineage_set_at == anchor_ts
        assert reloaded.halt_lineage_cleared_at == "2026-04-30T11:00:00"
        assert reloaded.error_cascade_last_stepdown_at == anchor_ts
    finally:
        tmp_path.unlink(missing_ok=True)
