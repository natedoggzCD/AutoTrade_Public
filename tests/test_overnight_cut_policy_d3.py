from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pandas as pd

from autotrade.core import day_manager as day_manager_mod
from autotrade.core.day_manager import DayManager
from autotrade.signals import overnight_cuts as cuts_mod


class _FixedDateTime(datetime):
    current_utc = datetime(2026, 5, 22, 19, 50, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        if tz is not None:
            return cls.current_utc.astimezone(tz)
        local = cls.current_utc.astimezone().replace(tzinfo=None)
        return cls(
            local.year,
            local.month,
            local.day,
            local.hour,
            local.minute,
            local.second,
        )


def _policy(enabled: bool = True):
    return SimpleNamespace(
        enabled=enabled,
        recovery_boost=5.0,
        trigger_minutes_before_close=15,
        latest_minutes_before_close=2,
        weak_entry_plus_minutes=30,
        weak_entry_plus_threshold_pct=-1.0,
        trim_fraction=0.5,
        min_hold_conviction_score=90.0,
        min_remaining_position_value=100.0,
        decision_claw_override_mode="shadow_only",
    )


def _make_dm(tmp_path, *, enabled: bool = True) -> DayManager:
    dm = DayManager.__new__(DayManager)
    dm.overnight_cut_policy_cfg = _policy(enabled=enabled)
    dm._overnight_cut_session_date = None
    dm._overnight_cut_actions = {}
    dm.position_entries = {"WEAK": "2026-05-22T14:00:00+00:00"}
    dm.data_client = None
    dm._position_qty = DayManager._position_qty.__get__(dm, DayManager)
    dm._position_float = DayManager._position_float.__get__(dm, DayManager)
    dm._safe_float = lambda value, default=0.0: DayManager._safe_float(value, default)
    dm._is_hedge_symbol = lambda symbol: False
    dm._is_elite_bullish_candidate = lambda symbol: False
    dm._alpha_add_override_snapshot = lambda symbol: {"is_proven_leader": False}
    dm._execution_override_plan = {
        "symbol_reviews": {
            "WEAK": {
                "accepted": True,
                "action": "hold_position",
                "confidence": 0.99,
            }
        }
    }
    dm.execute_trim_calls = []
    dm.execute_trim = (
        lambda symbol, qty, reason: dm.execute_trim_calls.append((symbol, qty, reason))
        is None
        or True
    )
    cuts_mod.CUTS_DIR = tmp_path
    return dm


def _weak_position(symbol: str = "WEAK"):
    return SimpleNamespace(
        symbol=symbol,
        qty="100",
        avg_entry_price="100.0",
        current_price="99.0",
        market_value="9900.0",
        unrealized_plpc="-0.01",
    )


def _bars(close: float = 98.0):
    return pd.DataFrame(
        [{"open": 99.0, "high": 99.5, "low": 97.5, "close": close, "volume": 1000}],
        index=pd.to_datetime(["2026-05-22T14:30:00Z"]),
    )


def test_d3_policy_skips_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(day_manager_mod, "datetime", _FixedDateTime)
    monkeypatch.setattr(day_manager_mod, "get_intraday_bars", lambda *a, **k: _bars())
    dm = _make_dm(tmp_path, enabled=False)

    dm._run_overnight_cut_policy(
        [(_weak_position(), {"action": "hold", "pnl_pct": -1.0}, 50.0)],
        {"exits": 0},
    )

    assert dm.execute_trim_calls == []


def test_d3_policy_skips_outside_t15_window(tmp_path, monkeypatch):
    monkeypatch.setattr(day_manager_mod, "datetime", _FixedDateTime)
    _FixedDateTime.current_utc = datetime(2026, 5, 22, 18, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(day_manager_mod, "get_intraday_bars", lambda *a, **k: _bars())
    dm = _make_dm(tmp_path)

    dm._run_overnight_cut_policy(
        [(_weak_position(), {"action": "hold", "pnl_pct": -1.0}, 50.0)],
        {"exits": 0},
    )

    assert dm.execute_trim_calls == []
    _FixedDateTime.current_utc = datetime(2026, 5, 22, 19, 50, tzinfo=timezone.utc)


def test_d3_policy_trims_weak_entry_plus_signal_and_records_cut(tmp_path, monkeypatch):
    monkeypatch.setattr(day_manager_mod, "datetime", _FixedDateTime)
    monkeypatch.setattr(day_manager_mod, "get_intraday_bars", lambda *a, **k: _bars())
    dm = _make_dm(tmp_path)
    stats = {"exits": 0}

    dm._run_overnight_cut_policy(
        [(_weak_position(), {"action": "hold", "pnl_pct": -1.0}, 50.0)],
        stats,
    )

    assert dm.execute_trim_calls == [
        ("WEAK", 50, "t15_weak_force_exit_trim:entry_plus_30m=-2.00%")
    ]
    assert stats["exits"] == 1
    rows = cuts_mod.load_cuts(date.today())
    assert len(rows) == 1
    assert rows[0].symbol == "WEAK"
    assert rows[0].qty == 50
    assert rows[0].trim_fraction == 0.5
    assert rows[0].policy_mode == "weak_only_t15_trim"
    assert rows[0].weak_signal_pct == -2.0


def test_d3_policy_shadow_claw_cannot_veto_trim(tmp_path, monkeypatch):
    monkeypatch.setattr(day_manager_mod, "datetime", _FixedDateTime)
    monkeypatch.setattr(day_manager_mod, "get_intraday_bars", lambda *a, **k: _bars())
    dm = _make_dm(tmp_path)
    dm._execution_override_plan["symbol_reviews"]["WEAK"]["accepted"] = True
    dm._execution_override_plan["symbol_reviews"]["WEAK"]["action"] = "hold_position"

    dm._run_overnight_cut_policy(
        [(_weak_position(), {"action": "hold", "pnl_pct": -1.0}, 50.0)],
        {"exits": 0},
    )

    assert dm.execute_trim_calls


def test_d3_policy_skips_elite_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(day_manager_mod, "datetime", _FixedDateTime)
    monkeypatch.setattr(day_manager_mod, "get_intraday_bars", lambda *a, **k: _bars())
    dm = _make_dm(tmp_path)
    dm._is_elite_bullish_candidate = lambda symbol: True

    dm._run_overnight_cut_policy(
        [(_weak_position(), {"action": "hold", "pnl_pct": -1.0}, 50.0)],
        {"exits": 0},
    )

    assert dm.execute_trim_calls == []


def test_d3_policy_skips_when_entry_plus_bar_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(day_manager_mod, "datetime", _FixedDateTime)
    monkeypatch.setattr(day_manager_mod, "get_intraday_bars", lambda *a, **k: None)
    dm = _make_dm(tmp_path)

    dm._run_overnight_cut_policy(
        [(_weak_position(), {"action": "hold", "pnl_pct": -1.0}, 50.0)],
        {"exits": 0},
    )

    assert dm.execute_trim_calls == []
