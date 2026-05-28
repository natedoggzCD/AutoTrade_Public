from datetime import datetime
import os
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from autotrade.core.autonomous_agent import AutonomousAgent
from autotrade.core.autonomous_agent import MarketPhase, MarketScheduler
from autotrade.core.day_manager import DayManager
from autotrade.utils.research_freshness import check_research_freshness


ET = ZoneInfo("America/New_York")


def test_monday_freshness_window_accepts_weekend_state(tmp_path: Path):
    state_path = tmp_path / "overnight_state.json"
    state_path.write_text(
        (
            '{"updated_at":"2026-02-06T20:00:00-05:00","research_complete":true,'
            '"watchlist":[{"symbol":"SPY"}],'
            '"workflow_completion":{"youtube_ready":true,"watchlist_selected":true,"game_plan_generated":true}}'
        ),
        encoding="utf-8",
    )
    now_et = datetime(2026, 2, 9, 7, 0, tzinfo=ET)  # Monday
    policy = {
        "weekday_max_age_hours": 18.0,
        "monday_max_age_hours": 60.0,
        "warning_age_hours": 24.0,
    }

    result = check_research_freshness(
        state_path=state_path,
        policy_source=policy,
        now_et=now_et,
        persist_metadata=False,
    )

    assert result["is_fresh"] is True
    assert result["warning"] is True
    assert result["max_age_hours"] == 60.0
    assert result["age_hours"] > 24.0


def test_weekday_freshness_window_rejects_stale_state(tmp_path: Path):
    state_path = tmp_path / "overnight_state.json"
    state_path.write_text(
        (
            '{"updated_at":"2026-02-10T10:00:00-05:00","research_complete":true,'
            '"watchlist":[{"symbol":"QQQ"}],'
            '"workflow_completion":{"youtube_ready":true,"watchlist_selected":true,"game_plan_generated":true}}'
        ),
        encoding="utf-8",
    )
    now_et = datetime(2026, 2, 11, 7, 0, tzinfo=ET)  # Wednesday
    policy = {
        "weekday_max_age_hours": 18.0,
        "monday_max_age_hours": 60.0,
        "warning_age_hours": 24.0,
    }

    result = check_research_freshness(
        state_path=state_path,
        policy_source=policy,
        now_et=now_et,
        persist_metadata=False,
    )

    assert result["is_stale"] is True
    assert result["is_fresh"] is False
    assert result["max_age_hours"] == 18.0


def test_weekday_market_hours_uses_strict_cap(tmp_path: Path):
    state_path = tmp_path / "overnight_state.json"
    state_path.write_text(
        (
            '{"updated_at":"2026-02-17T06:00:00-05:00","research_complete":true,'
            '"watchlist":[{"symbol":"SPY"}],'
            '"workflow_completion":{"watchlist_selected":true,"game_plan_generated":true}}'
        ),
        encoding="utf-8",
    )
    now_et = datetime(2026, 2, 17, 13, 0, tzinfo=ET)  # Tuesday market hours
    policy = {
        "weekday_max_age_hours": 18.0,
        "monday_max_age_hours": 60.0,
        "warning_age_hours": 24.0,
    }

    result = check_research_freshness(
        state_path=state_path,
        policy_source=policy,
        now_et=now_et,
        persist_metadata=False,
    )

    assert result["max_age_hours"] == 6.0
    assert result["age_hours"] > 6.0
    assert result["is_fresh"] is False


def test_weekday_premarket_uses_extended_cap(tmp_path: Path):
    state_path = tmp_path / "overnight_state.json"
    state_path.write_text(
        (
            '{"updated_at":"2026-02-10T20:30:00-05:00","research_complete":true,'
            '"watchlist":[{"symbol":"QQQ"}],'
            '"workflow_completion":{"watchlist_selected":true,"game_plan_generated":true}}'
        ),
        encoding="utf-8",
    )
    now_et = datetime(2026, 2, 11, 7, 30, tzinfo=ET)  # Wednesday premarket
    policy = {
        "weekday_max_age_hours": 18.0,
        "premarket_max_age_hours": 18.0,
        "strict_weekday_max_age_hours": 6.0,
        "strict_weekday_start_hour_et": 9.5,
    }

    result = check_research_freshness(
        state_path=state_path,
        policy_source=policy,
        now_et=now_et,
        persist_metadata=False,
    )

    assert result["max_age_hours"] == 18.0
    assert result["is_fresh"] is True


def test_premarket_previous_day_cutoff_override_from_state_timestamp(tmp_path: Path):
    state_path = tmp_path / "overnight_state.json"
    state_path.write_text(
        (
            '{"updated_at":"2026-02-10T18:45:00-05:00","research_complete":true,'
            '"watchlist":[{"symbol":"QQQ"}],'
            '"workflow_completion":{"watchlist_selected":true,"game_plan_generated":true}}'
        ),
        encoding="utf-8",
    )
    now_et = datetime(2026, 2, 11, 7, 30, tzinfo=ET)
    policy = {
        "weekday_max_age_hours": 18.0,
        "premarket_max_age_hours": 6.0,
        "strict_weekday_max_age_hours": 6.0,
        "strict_weekday_start_hour_et": 9.5,
        "premarket_previous_day_cutoff_hour_et": 18.5,
        "premarket_previous_day_cutoff_override": True,
    }

    result = check_research_freshness(
        state_path=state_path,
        policy_source=policy,
        now_et=now_et,
        persist_metadata=False,
    )

    assert result["age_hours"] > 6.0
    assert result["age_stale"] is False
    assert result["premarket_evening_cutoff_override"] is True
    assert result["is_fresh"] is True


def test_premarket_previous_day_cutoff_override_from_artifact_mtime(tmp_path: Path):
    research_dir = tmp_path / "research"
    plans_dir = tmp_path / "plans"
    research_dir.mkdir(parents=True, exist_ok=True)
    plans_dir.mkdir(parents=True, exist_ok=True)

    state_path = research_dir / "overnight_state.json"
    state_path.write_text(
        (
            '{"updated_at":"2026-02-10T12:00:00-05:00","research_complete":true,'
            '"watchlist":[{"symbol":"QQQ"}],'
            '"workflow_completion":{"watchlist_selected":true,"game_plan_generated":true}}'
        ),
        encoding="utf-8",
    )
    plan_path = plans_dir / "morning_game_plan_20260211.json"
    plan_path.write_text("{}", encoding="utf-8")

    mtime = datetime(2026, 2, 10, 19, 0, tzinfo=ET).timestamp()
    os.utime(plan_path, (mtime, mtime))

    now_et = datetime(2026, 2, 11, 7, 30, tzinfo=ET)
    policy = {
        "weekday_max_age_hours": 18.0,
        "premarket_max_age_hours": 6.0,
        "strict_weekday_max_age_hours": 6.0,
        "strict_weekday_start_hour_et": 9.5,
        "premarket_previous_day_cutoff_hour_et": 18.5,
        "premarket_previous_day_cutoff_override": True,
    }

    result = check_research_freshness(
        state_path=state_path,
        policy_source=policy,
        now_et=now_et,
        persist_metadata=False,
    )

    assert result["age_stale"] is False
    assert result["premarket_evening_cutoff_override"] is True


def test_recent_state_without_completion_is_not_fresh(tmp_path: Path):
    state_path = tmp_path / "overnight_state.json"
    state_path.write_text(
        '{"updated_at":"2026-02-11T06:30:00-05:00","research_complete":false,"watchlist":[{"symbol":"AAPL"}]}',
        encoding="utf-8",
    )
    now_et = datetime(2026, 2, 11, 7, 0, tzinfo=ET)

    result = check_research_freshness(
        state_path=state_path,
        policy_source={"weekday_max_age_hours": 18.0, "monday_max_age_hours": 60.0},
        now_et=now_et,
        persist_metadata=False,
    )

    assert result["is_fresh"] is False
    assert result["is_stale"] is True
    assert result["workflow_complete"] is False
    assert str(result["reason"]).startswith("incomplete_workflow:")


def test_legacy_complete_state_with_plan_artifact_is_fresh(tmp_path: Path):
    research_dir = tmp_path / "research"
    plans_dir = tmp_path / "plans"
    research_dir.mkdir(parents=True, exist_ok=True)
    plans_dir.mkdir(parents=True, exist_ok=True)

    state_path = research_dir / "overnight_state.json"
    state_path.write_text(
        '{"updated_at":"2026-02-11T06:30:00-05:00","research_complete":true,"watchlist":[{"symbol":"MSFT"}]}',
        encoding="utf-8",
    )
    (plans_dir / "morning_game_plan_20260211.json").write_text("{}", encoding="utf-8")
    now_et = datetime(2026, 2, 11, 7, 0, tzinfo=ET)

    result = check_research_freshness(
        state_path=state_path,
        policy_source={"weekday_max_age_hours": 18.0, "monday_max_age_hours": 60.0},
        now_et=now_et,
        persist_metadata=False,
    )

    assert result["is_fresh"] is True
    assert result["workflow_complete"] is True


def test_gap_policy_attaches_gap_and_reprices_without_network(monkeypatch):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.premarket_gap_cfg = SimpleNamespace(
        enabled=True,
        extreme_gap_up_pct=10.0,
        extreme_gap_down_pct=-5.0,
        moderate_gap_up_pct=3.0,
        moderate_gap_down_pct=-2.0,
        reprice_up_multiplier=0.99,
        reprice_down_multiplier=1.01,
    )
    agent.logger = SimpleNamespace(warning=lambda *args, **kwargs: None)
    agent.plan_generator = SimpleNamespace(
        _coerce_float=lambda value, default=0.0: (
            float(value) if value is not None else float(default)
        )
    )

    def fake_gap_context(self, symbol):
        data = {
            "AAA": {
                "has_data": True,
                "gap_pct": 12.0,
                "pre_price": 11.2,
                "prev_close": 10.0,
            },
            "BBB": {
                "has_data": True,
                "gap_pct": -6.0,
                "pre_price": 9.4,
                "prev_close": 10.0,
            },
            "CCC": {
                "has_data": True,
                "gap_pct": 4.0,
                "pre_price": 10.0,
                "prev_close": 9.6,
            },
            "EEE": {
                "has_data": True,
                "gap_pct": -3.0,
                "pre_price": 7.0,
                "prev_close": 10.0,
            },
            "DDD": {
                "has_data": False,
                "gap_pct": None,
                "pre_price": 0.0,
                "prev_close": 0.0,
            },
        }
        payload = data.get(
            symbol,
            {"has_data": False, "gap_pct": None, "pre_price": 0.0, "prev_close": 0.0},
        )
        payload["symbol"] = symbol
        return payload

    monkeypatch.setattr(AutonomousAgent, "_get_premarket_gap_context", fake_gap_context)

    signals = [
        {"symbol": "AAA", "entry_price": 10.0},
        {"symbol": "BBB", "entry_price": 10.0},
        {"symbol": "CCC", "entry_price": 9.8},
        {"symbol": "DDD", "entry_price": 8.0},
        {"symbol": "EEE", "entry_price": 10.0, "stop_loss": 9.4, "target": 12.0},
    ]
    result = agent._validate_premarket_gap_policy(signals)

    kept = result["signals"]
    by_symbol = {s["symbol"]: s for s in kept if isinstance(s, dict)}

    assert "AAA" in by_symbol
    assert by_symbol["AAA"]["premarket_volatility_watch"] is True
    assert "BBB" not in by_symbol
    assert "CCC" in by_symbol
    assert "DDD" in by_symbol
    assert round(by_symbol["CCC"]["entry_price"], 2) == 9.90
    assert by_symbol["CCC"]["premarket_gap_pct"] == 4.0
    assert by_symbol["DDD"]["premarket_gap_pct"] is None
    assert round(by_symbol["EEE"]["entry_price"], 2) == 7.07
    assert by_symbol["EEE"]["stop_loss"] < by_symbol["EEE"]["entry_price"]
    assert by_symbol["EEE"]["target"] > by_symbol["EEE"]["entry_price"]
    assert "premarket_risk_level_repaired" not in by_symbol["EEE"]
    assert result["summary"]["extreme_drops"] == 2
    assert result["summary"]["repriced"] == 2


def test_gap_policy_reprice_preserves_original_bracket_geometry(monkeypatch):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.premarket_gap_cfg = SimpleNamespace(
        enabled=True,
        extreme_gap_up_pct=10.0,
        extreme_gap_down_pct=-5.0,
        moderate_gap_up_pct=3.0,
        moderate_gap_down_pct=-2.0,
        reprice_up_multiplier=0.99,
        reprice_down_multiplier=1.01,
    )
    agent.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    agent.plan_generator = SimpleNamespace(
        _coerce_float=lambda value, default=0.0: (
            float(value) if value is not None else float(default)
        )
    )

    monkeypatch.setattr(
        AutonomousAgent,
        "_get_premarket_gap_context",
        lambda self, symbol: {
            "symbol": symbol,
            "has_data": True,
            "gap_pct": 4.0,
            "pre_price": 100.0,
            "prev_close": 96.0,
        },
    )

    result = agent._validate_premarket_gap_policy(
        [
            {
                "symbol": "RR",
                "entry_price": 107.79,
                "stop_loss": 102.40,
                "target": 118.57,
            }
        ]
    )

    [repriced] = result["signals"]
    assert repriced["entry_price"] == pytest.approx(99.0)
    original_risk_pct = (107.79 - 102.40) / 107.79
    original_reward_pct = (118.57 - 107.79) / 107.79
    repriced_risk_pct = (repriced["entry_price"] - repriced["stop_loss"]) / repriced[
        "entry_price"
    ]
    repriced_reward_pct = (repriced["target"] - repriced["entry_price"]) / repriced[
        "entry_price"
    ]
    assert repriced_risk_pct == pytest.approx(original_risk_pct, rel=0.1)
    assert repriced_reward_pct == pytest.approx(original_reward_pct, rel=0.1)


def test_scheduler_uses_overnight_phase_sunday_evening(monkeypatch):
    scheduler = MarketScheduler()

    monkeypatch.setattr(
        scheduler,
        "get_current_time",
        lambda: datetime(2026, 2, 8, 18, 30, tzinfo=ET),
    )
    assert scheduler.get_market_phase() == MarketPhase.OVERNIGHT

    monkeypatch.setattr(
        scheduler,
        "get_current_time",
        lambda: datetime(2026, 2, 8, 14, 0, tzinfo=ET),
    )
    assert scheduler.get_market_phase() == MarketPhase.PM_WORKFLOW


def test_day_manager_stale_penalty_activation_without_runtime_clients():
    dm = DayManager.__new__(DayManager)
    dm.research_freshness_cfg = SimpleNamespace(
        stale_penalty_age_hours=24.0,
        stale_score_threshold_penalty=10.0,
        stale_position_size_multiplier=0.7,
    )
    dm.research_freshness = {
        "is_stale": True,
        "age_hours": 30.0,
        "max_age_hours": 18.0,
        "warning": True,
    }
    dm.research_score_threshold_penalty = 0.0
    dm.research_position_size_multiplier = 1.0

    dm._apply_research_freshness_degradation()

    assert dm.research_score_threshold_penalty == 10.0
    assert dm.research_position_size_multiplier == 0.7


def test_generate_morning_game_plan_sets_completion_markers(
    monkeypatch, tmp_path: Path
):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    agent.entry_quality_cfg = SimpleNamespace(
        stale_entry_reprice_streak=2,
        stale_entry_temp_evict_streak=3,
        stale_entry_max_gap_pct=15.0,
    )

    monkeypatch.setattr("autotrade.core.autonomous_agent.PLANS_DIR", tmp_path)
    state = {
        "watchlist": [
            {
                "symbol": "AAPL",
                "recommendation": "BUY",
                "confidence": 80,
                "sector": "Technology",
            }
        ],
        "sectors": {"Technology": 1},
    }

    ok = agent._generate_morning_game_plan(state)

    assert ok is True
    completion = state.get("workflow_completion", {})
    assert completion.get("watchlist_selected") is True
    assert completion.get("game_plan_generated") is True
    assert len(list(tmp_path.glob("morning_game_plan_*.json"))) == 1


def test_wave_limit_price_is_marketable_and_bounded():
    agent = AutonomousAgent.__new__(AutonomousAgent)

    limit_price = agent._compute_wave_limit_price(
        planned_entry=100.0,
        current_price=101.0,
        score=82.0,
        wave=2,
    )
    max_chase_pct = agent._wave_max_chase_pct(score=82.0, wave=2)
    max_allowed = 100.0 * (1.0 + max_chase_pct / 100.0)

    assert limit_price >= 101.0
    assert limit_price <= round(max_allowed, 2)


def test_wave_max_chase_pct_tiers():
    agent = AutonomousAgent.__new__(AutonomousAgent)

    assert agent._wave_max_chase_pct(score=60.0, wave=1) == 2.0
    assert agent._wave_max_chase_pct(score=78.0, wave=1) == 2.5
    assert agent._wave_max_chase_pct(score=90.0, wave=2) == 3.5


def test_wave_breakout_rescue_candidate_rules():
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.entry_quality_cfg = SimpleNamespace(
        wave_breakout_rescue_enabled=True,
        wave_breakout_rescue_min_score=75.0,
        wave_breakout_rescue_max_gap_pct=9.0,
    )
    agent._effective_market_regime = lambda: "NEUTRAL"

    assert (
        agent._is_wave_breakout_rescue_candidate(gap_pct=7.8, score=82.0, wave=2)
        is True
    )
    assert (
        agent._is_wave_breakout_rescue_candidate(gap_pct=2.9, score=82.0, wave=2)
        is False
    )
    assert (
        agent._is_wave_breakout_rescue_candidate(gap_pct=7.8, score=70.0, wave=2)
        is False
    )


def test_wave_breakout_rescue_allows_quick_turnover_continuation():
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.entry_quality_cfg = SimpleNamespace(
        wave_breakout_rescue_enabled=True,
        wave_breakout_rescue_min_score=75.0,
        wave_breakout_rescue_max_gap_pct=9.0,
        quick_turnover_continuation_enabled=True,
        quick_turnover_continuation_min_score=66.0,
        quick_turnover_continuation_min_volume_ratio=1.0,
    )
    agent._effective_market_regime = lambda: "NEUTRAL"
    agent._coerce_float = lambda value, default=0.0: float(value or default)

    assert (
        agent._is_wave_breakout_rescue_candidate(
            gap_pct=5.2,
            score=67.75,
            wave=2,
            signal_data={
                "overnight_execution_intent": "quick_turnover",
                "overnight_actionability_score": 69.16,
                "risk_reward": 1.75,
                "volume_ratio": 1.26,
                "setup_type": "pullback_support",
            },
        )
        is True
    )
    assert (
        agent._is_wave_breakout_rescue_candidate(
            gap_pct=5.2,
            score=67.75,
            wave=2,
            signal_data={
                "overnight_execution_intent": "hold_candidate",
                "overnight_actionability_score": 69.16,
                "risk_reward": 1.75,
                "volume_ratio": 1.26,
                "setup_type": "pullback_support",
            },
        )
        is False
    )


def test_same_session_completed_overnight_run_stays_fresh_intraday(tmp_path: Path):
    state_path = tmp_path / "overnight_state.json"
    state_path.write_text(
        (
            '{"updated_at":"2026-02-12T22:30:00-05:00","research_complete":true,'
            '"watchlist":[{"symbol":"AAA"}],'
            '"workflow_completion":{"youtube_ready":true,"watchlist_selected":true,'
            '"game_plan_generated":true,"target_trade_date":"2026-02-13"}}'
        ),
        encoding="utf-8",
    )
    now_et = datetime(2026, 2, 13, 15, 59, tzinfo=ET)
    policy = {
        "weekday_max_age_hours": 18.0,
        "strict_weekday_max_age_hours": 6.0,
        "strict_weekday_start_hour_et": 9.5,
        "warning_age_hours": 12.0,
        "penalty_age_hours": 16.0,
    }

    result = check_research_freshness(
        state_path=state_path,
        policy_source=policy,
        now_et=now_et,
        persist_metadata=False,
    )

    assert result["is_fresh"] is True
    assert result["session_aligned"] is True
    assert result["freshness_basis"] == "session_alignment"
    assert result["warning"] is False
    assert result["stale_penalty_active"] is False


def test_prior_session_completed_run_is_stale_next_session(tmp_path: Path):
    state_path = tmp_path / "overnight_state.json"
    state_path.write_text(
        (
            '{"updated_at":"2026-02-11T22:30:00-05:00","research_complete":true,'
            '"watchlist":[{"symbol":"AAA"}],'
            '"workflow_completion":{"youtube_ready":true,"watchlist_selected":true,'
            '"game_plan_generated":true,"target_trade_date":"2026-02-12"}}'
        ),
        encoding="utf-8",
    )
    now_et = datetime(2026, 2, 13, 10, 0, tzinfo=ET)
    policy = {
        "weekday_max_age_hours": 18.0,
        "strict_weekday_max_age_hours": 6.0,
        "strict_weekday_start_hour_et": 9.5,
    }

    result = check_research_freshness(
        state_path=state_path,
        policy_source=policy,
        now_et=now_et,
        persist_metadata=False,
    )

    assert result["is_fresh"] is False
    assert result["session_aligned"] is False
    assert result["is_stale"] is True
