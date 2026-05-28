from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from autotrade.core.day_manager import DayManager
from autotrade.signals.trade_learner import TradeJournal


def _make_journal(now: datetime) -> TradeJournal:
    journal = TradeJournal.__new__(TradeJournal)
    journal.trades = [
        {
            "symbol": "MUR",
            "trade_type": "entry",
            "entry_time": (now - timedelta(minutes=30)).isoformat(),
            "entry_status": "filled",
        },
        {
            "symbol": "MUR",
            "trade_type": "trim",
            "time": (now - timedelta(minutes=15)).isoformat(),
            "entry_status": "filled",
        },
    ]
    return journal


def _make_day_manager(now: datetime) -> DayManager:
    dm = DayManager.__new__(DayManager)
    dm.entry_quality_cfg = SimpleNamespace(
        same_day_reentry_cooldown_minutes=60,
        entry_conviction_floor=60.0,
    )
    dm.risk_gate_cfg = SimpleNamespace(re_entry_cooldown_minutes=30)
    dm.trade_journal = _make_journal(now)
    dm._normalize_entry_source = lambda payload=None: (
        str(((payload or {}).get("entry_source") or "")).strip().lower()
    )
    dm._strength_reentry_candidate_allowed = lambda payload=None: False
    return dm


def test_trade_journal_flags_recent_same_day_sell_activity():
    now = datetime.now(timezone.utc)
    journal = _make_journal(now)

    reason = journal.get_same_day_reentry_cooldown_reason("MUR", cooldown_minutes=60)

    assert reason == "same_day_sell_lockout:symbol_was_sold_or_trimmed_today"


def test_day_manager_blocks_low_conviction_same_day_reentry():
    now = datetime.now(timezone.utc)
    dm = _make_day_manager(now)

    reasons = dm._entry_guard_reasons(
        "MUR",
        50.0,
        {"symbol": "MUR", "entry_source": "pm_plan"},
        current_position=SimpleNamespace(symbol="MUR"),
    )

    assert "same_day_sell_lockout:symbol_was_sold_or_trimmed_today" in reasons
    assert "entry_conviction_floor:50.0<60.0" in reasons


def test_day_manager_strength_reentry_path_is_exempt_from_cooldown_and_floor():
    now = datetime.now(timezone.utc)
    dm = _make_day_manager(now)
    dm._strength_reentry_candidate_allowed = lambda payload=None: True

    reasons = dm._entry_guard_reasons(
        "MUR",
        50.0,
        {"symbol": "MUR", "entry_source": "strength_reentry"},
        current_position=SimpleNamespace(symbol="MUR"),
    )

    assert reasons == []
