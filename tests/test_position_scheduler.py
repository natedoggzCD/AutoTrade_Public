"""Tests for PositionScheduler and MonitoringTier classification."""

import time
from datetime import datetime, timedelta

import pytest

from autotrade.core.position_scheduler import MonitoringTier, PositionScheduler


class TestTierClassification:
    def setup_method(self):
        self.sched = PositionScheduler()

    def test_near_stop_is_critical(self):
        tier = self.sched.classify_tier("ACME", pnl_pct=-7.5, hard_stop_pct=-8.0, hold_minutes=120)
        assert tier == MonitoringTier.CRITICAL

    def test_brand_new_position_is_critical(self):
        tier = self.sched.classify_tier("ACME", pnl_pct=0.5, hold_minutes=5)
        assert tier == MonitoringTier.CRITICAL

    def test_earnings_day_is_critical(self):
        tier = self.sched.classify_tier("ACME", pnl_pct=2.0, hold_minutes=300, has_earnings_today=True)
        assert tier == MonitoringTier.CRITICAL

    def test_young_position_is_active(self):
        tier = self.sched.classify_tier("ACME", pnl_pct=1.0, hold_minutes=30)
        assert tier == MonitoringTier.ACTIVE

    def test_recent_action_is_active(self):
        tier = self.sched.classify_tier("ACME", pnl_pct=1.0, hold_minutes=120, thesis_last_action="trim")
        assert tier == MonitoringTier.ACTIVE

    def test_well_in_profit_far_from_levels_is_dormant(self):
        tier = self.sched.classify_tier(
            "ACME",
            pnl_pct=7.0,
            hold_minutes=300,
            atr_distance_stop=3.0,
            atr_distance_target=3.0,
        )
        assert tier == MonitoringTier.DORMANT

    def test_moderate_state_is_stable(self):
        tier = self.sched.classify_tier("ACME", pnl_pct=1.5, hold_minutes=120)
        assert tier == MonitoringTier.STABLE

    def test_negative_pnl_moderate_is_stable(self):
        tier = self.sched.classify_tier("ACME", pnl_pct=-2.0, hold_minutes=120)
        assert tier == MonitoringTier.STABLE

    def test_large_pnl_not_dormant_if_close_to_stop(self):
        tier = self.sched.classify_tier(
            "ACME",
            pnl_pct=6.0,
            hold_minutes=300,
            atr_distance_stop=1.0,  # too close
            atr_distance_target=3.0,
        )
        assert tier != MonitoringTier.DORMANT


class TestScheduling:
    def test_new_symbol_is_due(self):
        sched = PositionScheduler()
        assert sched.is_due("ACME") is True

    def test_after_schedule_not_due(self):
        sched = PositionScheduler()
        sched.schedule_next("ACME", MonitoringTier.STABLE, current_price=10.0, pnl_pct=1.0)
        assert sched.is_due("ACME") is False

    def test_becomes_due_after_interval(self):
        sched = PositionScheduler(tier_intervals={"stable": 180})
        sched.schedule_next("ACME", MonitoringTier.STABLE, current_price=10.0, pnl_pct=1.0)
        assert sched.is_due("ACME") is False
        # Manually backdate next_eval_at to simulate time passing
        sched._schedule["ACME"].next_eval_at = datetime.utcnow() - timedelta(seconds=1)
        assert sched.is_due("ACME") is True

    def test_price_override(self):
        sched = PositionScheduler(force_eval_price_move_bps=50.0)
        sched.schedule_next("ACME", MonitoringTier.DORMANT, current_price=10.0, pnl_pct=5.0)
        # Small move — no override
        assert sched.check_override("ACME", current_price=10.03, pnl_pct=5.0) is False
        # Large move — override
        assert sched.check_override("ACME", current_price=10.06, pnl_pct=5.0) is True

    def test_pnl_override(self):
        sched = PositionScheduler(force_eval_pnl_change_pct=1.0)
        sched.schedule_next("ACME", MonitoringTier.DORMANT, current_price=10.0, pnl_pct=5.0)
        assert sched.check_override("ACME", current_price=10.0, pnl_pct=5.5) is False
        assert sched.check_override("ACME", current_price=10.0, pnl_pct=6.1) is True

    def test_should_evaluate_combined(self):
        sched = PositionScheduler(force_eval_price_move_bps=50.0)
        sched.schedule_next("ACME", MonitoringTier.DORMANT, current_price=10.0, pnl_pct=5.0)
        # Not due, no override
        assert sched.should_evaluate("ACME", current_price=10.01, pnl_pct=5.0) is False
        # Override triggered
        assert sched.should_evaluate("ACME", current_price=10.10, pnl_pct=5.0) is True

    def test_should_evaluate_new_symbol(self):
        sched = PositionScheduler()
        assert sched.should_evaluate("NEW") is True


class TestSchedulerMisc:
    def test_get_tier(self):
        sched = PositionScheduler()
        assert sched.get_tier("ACME") is None
        sched.schedule_next("ACME", MonitoringTier.STABLE)
        assert sched.get_tier("ACME") == MonitoringTier.STABLE

    def test_remove(self):
        sched = PositionScheduler()
        sched.schedule_next("ACME", MonitoringTier.ACTIVE)
        sched.remove("ACME")
        assert sched.get_tier("ACME") is None
        assert sched.is_due("ACME") is True

    def test_summary(self):
        sched = PositionScheduler()
        sched.schedule_next("ACME", MonitoringTier.CRITICAL)
        sched.schedule_next("BCDE", MonitoringTier.DORMANT)
        s = sched.summary()
        assert s == {"ACME": "CRITICAL", "BCDE": "DORMANT"}

    def test_custom_intervals(self):
        sched = PositionScheduler(tier_intervals={"critical": 15, "dormant": 600})
        assert sched._intervals[MonitoringTier.CRITICAL] == 15
        assert sched._intervals[MonitoringTier.DORMANT] == 600
        assert sched._intervals[MonitoringTier.ACTIVE] == 60  # unchanged default

    def test_case_insensitive(self):
        sched = PositionScheduler()
        sched.schedule_next("acme", MonitoringTier.ACTIVE)
        assert sched.get_tier("ACME") == MonitoringTier.ACTIVE
        assert sched.is_due("Acme") is False
