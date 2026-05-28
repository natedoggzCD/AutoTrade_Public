from datetime import datetime
from types import SimpleNamespace

from autotrade.core import day_manager as day_manager_mod
from autotrade.core.day_manager import DayManager


def _make_dm() -> DayManager:
    dm = DayManager.__new__(DayManager)
    dm._position_float = DayManager._position_float.__get__(dm, DayManager)
    dm._position_qty = DayManager._position_qty.__get__(dm, DayManager)
    dm._safe_float = lambda value, default=0.0: DayManager._safe_float(value, default)
    dm._entry_time_for_advisor = lambda symbol: None
    dm._hold_minutes = lambda symbol: 900
    dm._hard_stop_pct = lambda: -15.0
    dm._is_hedge_symbol = lambda symbol: False
    dm._live_execution_mode = lambda: {}
    dm.exit_manager = SimpleNamespace(check_exits=lambda positions: [])
    dm.runtime_risk_gate = None
    dm.position_advisor = None
    dm.signals = [
        {
            "ticker": "BMNR",
            "entry_rsi_14": 74.0,
            "atr_14": 4.0,
            "entry_source": "overnight_plan",
        }
    ]
    dm._get_feature_context = lambda symbol: {"rsi_14": 78.0, "atr_14": 4.0}
    dm._find_signal_data = lambda symbol: dict(dm.signals[0])
    dm._candidate_relative_strength_value = lambda signal_data, current_price: 1.0
    dm._cache_advisor_health = lambda **kwargs: None
    dm._apply_policy_risk_overlay = lambda position, health: health
    dm._apply_hard_stop_override = lambda position, health: health
    dm._effective_market_regime = lambda: "NEUTRAL"
    dm.data_client = None
    dm._thesis_cache = SimpleNamespace(
        get=lambda symbol: None,
        get_prompt_context=lambda symbol: "",
    )
    dm.position_health = {}
    return dm


class _FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        base = cls(2026, 4, 20, 10, 30, 0)
        if tz is None:
            return base
        return tz.localize(base)


def test_calculate_position_health_trims_overbought_overnight_winner(monkeypatch):
    monkeypatch.setattr(
        day_manager_mod,
        "CONFIG",
        SimpleNamespace(
            portfolio=SimpleNamespace(
                strength_lock_enabled=True,
                strength_lock_min_entry_rsi_14=70.0,
                strength_lock_trigger_atr_fraction=0.5,
                strength_lock_trim_fraction=0.33,
                strength_lock_lower_high_retrace_pct=0.25,
                strength_lock_min_minutes_after_open=15,
                strength_lock_min_hold_minutes=360,
            ),
            risk_gate=SimpleNamespace(hard_stop_pct=-15.0),
        ),
    )
    monkeypatch.setattr(day_manager_mod, "datetime", _FixedDateTime)

    dm = _make_dm()
    position = SimpleNamespace(
        symbol="BMNR",
        avg_entry_price="100.0",
        current_price="102.0",
        unrealized_plpc=0.02,
        qty="30",
        high_since_entry="103.0",
    )

    health = dm.calculate_position_health(position)

    assert health["action"] == "trim"
    assert health["trim_reason"] == "strength_lock"
    assert health["trim_fraction"] >= 0.33
    assert "strength_lock:" in health["signals"][0]
