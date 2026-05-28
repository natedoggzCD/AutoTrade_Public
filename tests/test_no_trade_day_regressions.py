from datetime import datetime
from types import SimpleNamespace

from autotrade.core import autonomous_agent as autonomous_agent_mod
from autotrade.core.autonomous_agent import TomorrowsPlanGenerator
from autotrade.core.day_manager import DayManager
from autotrade.signals.lessons_screener import DB_PATH


def _build_signal(symbol: str, entry_price: float, score: float = 70.0):
    return {
        "symbol": symbol,
        "entry_price": entry_price,
        "stop_loss": entry_price * 0.95,
        "target": entry_price * 1.10,
        "final_score": score,
    }


def test_convert_signals_to_plan_backfills_non_held_symbols():
    """
    Regression: conversion must select the top N *eligible* symbols, not a fixed
    prefix slice that may be mostly already-held names.
    """
    held_symbols = [f"H{i:02d}" for i in range(1, 16)]
    incoming_signals = []
    # First 15 ranked signals are already held.
    for i, sym in enumerate(held_symbols):
        incoming_signals.append(_build_signal(sym, entry_price=10 + i, score=90 - i))
    # Next ranked signals are tradeable.
    incoming_signals.extend(
        [
            _build_signal("N1", 20.0, 75.0),
            _build_signal("N2", 21.0, 74.0),
            _build_signal("N3", 22.0, 73.0),
            _build_signal("N4", 23.0, 72.0),
            _build_signal("N5", 24.0, 71.0),
        ]
    )

    generator = TomorrowsPlanGenerator.__new__(TomorrowsPlanGenerator)
    generator.logger = SimpleNamespace(info=lambda *a, **k: None)
    generator.max_positions = 50
    generator.position_size_target = 2000
    generator.get_current_positions = lambda: [{"symbol": s} for s in held_symbols]
    generator.get_account_info = lambda: {"buying_power": 200_000.0}
    generator._coerce_float = TomorrowsPlanGenerator._coerce_float.__get__(
        generator, TomorrowsPlanGenerator
    )

    converted = generator._convert_signals_to_plan({"signals": incoming_signals})
    symbols = [o["symbol"] for o in converted["entry_orders"]]

    assert symbols == ["N1", "N2", "N3", "N4", "N5"]


def test_convert_signals_to_plan_preserves_non_execution_metadata():
    generator = TomorrowsPlanGenerator.__new__(TomorrowsPlanGenerator)
    generator.logger = SimpleNamespace(info=lambda *a, **k: None)
    generator.max_positions = 50
    generator.position_size_target = 2000
    generator.get_current_positions = lambda: []
    generator.get_account_info = lambda: {"buying_power": 200_000.0}
    generator._coerce_float = TomorrowsPlanGenerator._coerce_float.__get__(
        generator, TomorrowsPlanGenerator
    )

    plan_data = {
        "date": "2026-03-26",
        "signals": [_build_signal("N1", 20.0, 75.0)],
        "actionable_top50": [{"symbol": "N1"}, {"symbol": "N2"}],
        "full_watchlist": [{"symbol": "N1"}, {"symbol": "N2"}, {"symbol": "N3"}],
        "overflow_signals": [{"symbol": "N4"}],
    }

    converted = generator._convert_signals_to_plan(plan_data)

    assert converted["date"] == "2026-03-26"
    assert len(converted["actionable_top50"]) == 2
    assert len(converted["full_watchlist"]) == 3
    assert len(converted["overflow_signals"]) == 1
    assert converted["converted_from"] == "buy_signals"
    assert converted["entry_orders"][0]["symbol"] == "N1"


def test_day_manager_get_positions_drops_malformed_rows():
    """Regression: malformed MCP rows should be dropped, not crash cycle logic."""

    class FakeClient:
        @staticmethod
        def get_all_positions():
            return [
                "error",  # malformed row previously causing attribute crashes
                {
                    "symbol": "SND",
                    "qty": "100",
                    "market_value": "513.0",
                    "current_price": "5.13",
                    "avg_entry_price": "5.10",
                    "unrealized_plpc": "0.005",
                },
                SimpleNamespace(
                    symbol="CAKE",
                    qty="10",
                    market_value="640.7",
                    current_price="64.07",
                    avg_entry_price="63.90",
                    unrealized_plpc="0.002",
                ),
            ]

    manager = DayManager.__new__(DayManager)
    manager.client = FakeClient()

    positions = DayManager.get_positions(manager)
    symbols = [getattr(p, "symbol", "") for p in positions]

    assert symbols == ["SND", "CAKE"]
    assert all(hasattr(p, "market_value") for p in positions)


def test_mcp_probe_validation_rejects_error_payloads():
    assert (
        DayManager._is_valid_mcp_payload(
            account_probe={"error": "missing server"},
            positions_probe=[],
        )
        is False
    )


def test_lessons_screener_db_path_points_to_downday():
    assert DB_PATH is not None
    assert "DownDay" in str(DB_PATH)
    assert DB_PATH.exists()


def test_convert_signals_to_plan_guard_alert_on_collapse():
    """Guard should alert when many signals collapse to <=1 executable entries."""
    generator = TomorrowsPlanGenerator.__new__(TomorrowsPlanGenerator)
    generator.logger = SimpleNamespace(
        info=lambda *a, **k: None,
        critical=lambda *a, **k: None,
        warning=lambda *a, **k: None,
    )
    generator.max_positions = 50
    generator.position_size_target = 2000
    generator.get_current_positions = lambda: []
    generator.get_account_info = lambda: {"buying_power": 0.0}
    generator._coerce_float = TomorrowsPlanGenerator._coerce_float.__get__(
        generator, TomorrowsPlanGenerator
    )

    alerts = []
    generator._persist_plan_conversion_alert = lambda payload: alerts.append(payload)

    incoming_signals = [
        _build_signal(f"S{i:02d}", entry_price=10.0 + i, score=60.0) for i in range(30)
    ]
    converted = generator._convert_signals_to_plan({"signals": incoming_signals})

    assert converted["entry_orders"] == []
    assert len(alerts) == 1
    assert alerts[0]["signals"] == 30
    assert alerts[0]["built_entries"] == 0


def test_load_latest_plan_prefers_pm_plan_over_morning_plan(tmp_path, monkeypatch):
    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 8, 9, 40, 0)

    monkeypatch.setattr(autonomous_agent_mod, "datetime", _FixedDateTime)
    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", tmp_path)

    (tmp_path / "morning_game_plan_20260408.json").write_text(
        '{"signals":[{"symbol":"MORN","entry_price":10.0,"score":60.0}]}',
        encoding="utf-8",
    )
    (tmp_path / "pm_plan_2026-04-08.json").write_text(
        '{"signals":[{"symbol":"PM1","entry_price":12.0,"score":80.0}]}',
        encoding="utf-8",
    )

    generator = TomorrowsPlanGenerator.__new__(TomorrowsPlanGenerator)
    generator.logger = SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None
    )
    generator.max_positions = 10
    generator.position_size_target = 2000
    generator.get_current_positions = lambda: []
    generator.get_account_info = lambda: {"buying_power": 100_000.0}
    generator._coerce_float = TomorrowsPlanGenerator._coerce_float.__get__(
        generator, TomorrowsPlanGenerator
    )

    plan = generator._load_latest_plan()

    assert [row["symbol"] for row in plan["entry_orders"]] == ["PM1"]
    assert plan.get("signals", [{}])[0]["symbol"] == "PM1"


def test_load_latest_plan_converts_pm_entry_candidates(tmp_path, monkeypatch):
    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 8, 9, 40, 0)

    monkeypatch.setattr(autonomous_agent_mod, "datetime", _FixedDateTime)
    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", tmp_path)

    (tmp_path / "morning_game_plan_20260408.json").write_text(
        '{"signals":[{"symbol":"MORN","entry_price":10.0,"score":60.0}]}',
        encoding="utf-8",
    )
    (tmp_path / "pm_plan_2026-04-08.json").write_text(
        '{"entry_candidates":[{"symbol":"PM1","entry_price":12.0,"score":80.0}]}',
        encoding="utf-8",
    )

    generator = TomorrowsPlanGenerator.__new__(TomorrowsPlanGenerator)
    generator.logger = SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None
    )
    generator.max_positions = 10
    generator.position_size_target = 2000
    generator.get_current_positions = lambda: []
    generator.get_account_info = lambda: {"buying_power": 100_000.0}
    generator._coerce_float = TomorrowsPlanGenerator._coerce_float.__get__(
        generator, TomorrowsPlanGenerator
    )

    plan = generator._load_latest_plan()

    assert [row["symbol"] for row in plan["entry_orders"]] == ["PM1"]
    assert plan.get("entry_candidates", [{}])[0]["symbol"] == "PM1"


def test_load_latest_plan_prefers_actionable_morning_plan_over_watch_only_pm_plan(
    tmp_path, monkeypatch
):
    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 8, 9, 40, 0)

    monkeypatch.setattr(autonomous_agent_mod, "datetime", _FixedDateTime)
    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", tmp_path)

    (tmp_path / "morning_game_plan_20260408.json").write_text(
        '{"signals":[{"symbol":"MORN","entry_price":10.0,"score":60.0,"recommendation":"BUY"}]}',
        encoding="utf-8",
    )
    (tmp_path / "pm_plan_2026-04-08.json").write_text(
        '{"signals":[{"symbol":"PM1","entry_price":12.0,"score":80.0,"recommendation":"WATCH"}]}',
        encoding="utf-8",
    )

    generator = TomorrowsPlanGenerator.__new__(TomorrowsPlanGenerator)
    generator.logger = SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None
    )
    generator.max_positions = 10
    generator.position_size_target = 2000
    generator.get_current_positions = lambda: []
    generator.get_account_info = lambda: {"buying_power": 100_000.0}
    generator._coerce_float = TomorrowsPlanGenerator._coerce_float.__get__(
        generator, TomorrowsPlanGenerator
    )

    plan = generator._load_latest_plan()

    assert [row["symbol"] for row in plan["entry_orders"]] == ["MORN"]
    assert plan.get("signals", [{}])[0]["symbol"] == "MORN"


def test_generate_plan_marks_near_flat_odd_lot_for_review():
    generator = TomorrowsPlanGenerator.__new__(TomorrowsPlanGenerator)
    generator.logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    generator.alpaca_client = None
    generator.max_positions = 50
    generator.position_size_target = 2000
    generator.get_account_info = lambda: {"buying_power": 10000.0}
    generator.get_current_positions = lambda: []
    generator.detect_odd_lots = lambda min_position_size=5: [
        {
            "symbol": "CCJ",
            "qty": 1,
            "reason": "Position too small (1 shares)",
            "current_value": 109.55,
            "pnl": -0.33,
            "pnl_pct": -0.3,
        }
    ]
    generator.detect_losers_to_cut = lambda max_loss_pct=-5.0: []
    generator._convert_signals_to_plan = lambda pm_plan: {"entry_orders": []}

    plan = TomorrowsPlanGenerator.generate_plan(generator)

    assert plan["cleanup_orders"] == []
    assert plan["review_positions"] == [
        {
            "symbol": "CCJ",
            "qty": 1,
            "action": "REVIEW_AT_OPEN",
            "reason": "Odd lot recovered near flat; re-check conviction before liquidating",
            "current_value": 109.55,
            "pnl_pct": -0.3,
        }
    ]


def test_validated_positions_drops_malformed_rows():
    manager = DayManager.__new__(DayManager)
    positions = [
        "error",
        None,
        SimpleNamespace(symbol="", qty="1"),
        SimpleNamespace(symbol="SND", qty="10"),
    ]
    validated = DayManager._validated_positions(manager, positions, context="test")
    assert len(validated) == 1
    assert validated[0].symbol == "SND"


def test_execute_exit_recheck_uses_wrapped_get_positions():
    class DummyClient:
        @staticmethod
        def get_orders(_request):
            return [SimpleNamespace(id="open-order-1")]

        @staticmethod
        def cancel_order_by_id(_order_id):
            return None

        @staticmethod
        def get_all_positions():
            raise AssertionError(
                "execute_exit should not call client.get_all_positions directly"
            )

    manager = DayManager.__new__(DayManager)
    manager.client = DummyClient()
    manager.dry_run = False
    manager.entry_quality_cfg = SimpleNamespace(fresh_entry_exit_cooldown_minutes=0)
    manager.position_entries = {}
    manager.day_tracker = SimpleNamespace(record_day_trade=lambda **kwargs: None)
    # day-manager 2026-05-19: execute_exit calls trade_journal.save() after
    # record_exit; the stub needs both methods to avoid AttributeError.
    manager.trade_journal = SimpleNamespace(
        record_exit=lambda *args, **kwargs: None,
        save=lambda: None,
    )
    manager.get_current_price = lambda _symbol: 10.0
    manager.get_positions = lambda: [SimpleNamespace(symbol="SND", qty="10")]
    manager._validate_order = lambda symbol, qty, side, allow_open_sell=False: (
        True,
        "",
    )
    manager._acquire_order_submission_guard = lambda symbol, side, context="": (
        True,
        "",
        "guard-key",
    )
    manager._release_order_submission_guard = lambda guard_key, submitted=False: None
    manager._record_execution_attempt = lambda *args, **kwargs: None
    manager._record_execution_success = lambda *args, **kwargs: None
    manager._record_execution_failure = lambda *args, **kwargs: None
    manager._queue_sequential_shadow_event = lambda *args, **kwargs: None
    manager._sequential_shadow_enabled = False
    manager._submit_order_via_execution_adapter = lambda **kwargs: SimpleNamespace(
        id="submitted-1"
    )

    assert DayManager.execute_exit(manager, "SND", 5, "test_exit") is True
