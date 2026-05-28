from datetime import datetime, timedelta
from types import SimpleNamespace

from autotrade.core.day_manager import DayManager


def _manager() -> DayManager:
    dm = DayManager.__new__(DayManager)
    dm.posture_cfg = SimpleNamespace(
        stress_window_minutes=90,
        hard_stop_lockout_minutes=90,
    )
    dm.trade_journal = SimpleNamespace(trades=[])
    dm._hard_stop_blocked_symbols_today = {}
    dm._last_market_posture_state = {}
    dm._exited_symbols_today = set()
    dm.live_sector_bias = {}
    dm._live_execution_mode = lambda: {
        "resolved_regime": {"regime": "NEUTRAL"},
        "entry_authority_state": "open",
        "entry_authority_snapshot": {},
    }
    dm._effective_market_regime = lambda: "NEUTRAL"
    dm._validated_positions = lambda positions, context="": positions
    dm.get_positions = lambda: []
    dm._average_unrealized_pnl_pct = lambda positions: 0.0
    dm._live_benchmark_snapshot = lambda: {"recovery_confirmed": False}
    dm._record_market_posture_transition = lambda snapshot: setattr(
        dm, "_last_market_posture_state", dict(snapshot)
    )
    return dm


def test_stress_events_respect_posture_window():
    dm = _manager()
    now = datetime.now()
    dm.trade_journal.trades = [
        {
            "symbol": "AAA",
            "trade_type": "exit",
            "exit_reason": "hard_stop",
            "exit_time": (now - timedelta(minutes=30)).isoformat(),
        },
        {
            "symbol": "BBB",
            "trade_type": "exit",
            "exit_reason": "hard_stop",
            "exit_time": (now - timedelta(minutes=120)).isoformat(),
        },
    ]

    assert dm._same_day_stress_exit_count(window_minutes=90) == 1


def test_posture_stepdown_when_stress_window_empties():
    dm = _manager()
    dm._last_market_posture_state = {
        "posture": "capital_preservation",
        "reason": "bad_day_book_stress",
    }
    dm._live_execution_mode = lambda: {
        "resolved_regime": {"regime": "BEARISH"},
        "entry_authority_state": "open",
        "entry_authority_snapshot": {"red_ratio": 0.2, "avg_pct_change": 0.1},
    }

    posture = dm._market_posture(positions=[])

    assert posture["posture"] == "cautious_selective_longs"
    assert posture["reason"] == "bad_day_book_stress_decayed"


def test_posture_recovers_to_normal_after_cautious_decay_and_positive_book():
    dm = _manager()
    dm._last_market_posture_state = {
        "posture": "cautious_selective_longs",
        "reason": "bad_day_book_stress_decayed",
    }
    dm._average_unrealized_pnl_pct = lambda positions: 0.4

    posture = dm._market_posture(positions=[])

    assert posture["posture"] == "normal_risk"
    assert posture["reason"] == "bad_day_book_stress_recovered"


def test_per_symbol_hard_stop_exclusion_respects_lockout_window():
    dm = _manager()
    now = datetime.now()
    dm._hard_stop_blocked_symbols_today = {
        "FRESH": now - timedelta(minutes=20),
        "STALE": now - timedelta(minutes=120),
    }

    assert dm._is_hard_stop_blocked_symbol("FRESH") is True
    assert dm._is_hard_stop_blocked_symbol("STALE") is False


def test_cross_day_hard_stop_rehydrate_drops_yesterday():
    dm = _manager()
    yesterday = datetime.now() - timedelta(days=1)

    restored = dm._normalize_hard_stop_blocked_symbols(
        {"OLD": yesterday.isoformat()}
    )

    assert restored == {}
