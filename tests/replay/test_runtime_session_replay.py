import json
import threading
import time as wall_time
from pathlib import Path
from datetime import time
from types import SimpleNamespace

import pandas as pd
import pytest

from autotrade.core.day_manager import DayManager
from autotrade.replay.minute_bar_archive import archive_session_minute_bars
from autotrade.replay.runtime_session_replay import (
    DEFAULT_DISCOVERY_VARIANT,
    REPLAY_MODE_AGENT_WORKFLOW,
    REPLAY_MODE_DAYMANAGER_CORE,
    RuntimeSessionReplay,
    _ReplayTradingClient,
    _build_inverse_absence_diagnostic,
    _load_actual_day_context,
    build_replay_entry_diagnostic_gate,
    main,
    replay_runtime_session,
    parse_args,
    wait_for_runtime_replay_benchmark_completion,
    run_runtime_replay_benchmark,
)


def _bars(prices):
    index = pd.date_range(
        "2026-03-18 08:30",
        periods=len(prices),
        freq="min",
        tz="America/Chicago",
    )
    return pd.DataFrame(
        {
            "open": prices,
            "high": [price + 0.1 for price in prices],
            "low": [price - 0.1 for price in prices],
            "close": prices,
            "volume": [1_000_000] * len(prices),
        },
        index=index,
    )


def _bars_for_date(session_date: str, prices):
    index = pd.date_range(
        f"{session_date} 08:30",
        periods=len(prices),
        freq="min",
        tz="America/Chicago",
    )
    return pd.DataFrame(
        {
            "open": prices,
            "high": [price + 0.1 for price in prices],
            "low": [price - 0.1 for price in prices],
            "close": prices,
            "volume": [1_000_000] * len(prices),
        },
        index=index,
    )


def test_replay_regime_analysis_hydrates_sector_momentum():
    replay = RuntimeSessionReplay(session_date="2026-05-05")

    analysis = replay._build_replay_regime_analysis(
        resolved_regime={"regime": "CHOP", "allow_new_longs": True},
        artifacts={
            "plan_payload": {
                "current_regime_analysis": {
                    "breadth_pct_positive": 61.0,
                    "sector_momentum": {"Technology": 0.8, "Energy": -0.4},
                }
            }
        },
    )

    assert analysis.breadth_pct_positive == 61.0
    assert analysis.sector_momentum == {"technology": 0.8, "energy": -0.4}


class _ReplayArchiveClient:
    def __init__(self, minute_bars, previous_closes=None):
        self.minute_bars = {
            str(symbol).upper(): frame.copy()
            for symbol, frame in (minute_bars or {}).items()
        }
        self.previous_closes = {
            str(symbol).upper(): float(value)
            for symbol, value in (previous_closes or {}).items()
        }

    def get_stock_bars(self, request):
        symbol = str(request.symbol_or_symbols).upper()
        timeframe_name = str(getattr(request.timeframe, "value", request.timeframe))
        if "Day" in timeframe_name or timeframe_name == "1Day":
            close_value = self.previous_closes.get(symbol)
            if close_value is None:
                return type("Bars", (), {"df": pd.DataFrame()})()
            index = pd.DatetimeIndex([pd.Timestamp("2026-03-17 00:00:00", tz="UTC")])
            frame = pd.DataFrame(
                {
                    "open": [close_value],
                    "high": [close_value],
                    "low": [close_value],
                    "close": [close_value],
                },
                index=index,
            )
            return type("Bars", (), {"df": frame})()
        frame = self.minute_bars.get(symbol, pd.DataFrame())
        return type("Bars", (), {"df": frame})()


def test_replay_trading_client_supports_trim_order_surface():
    client = _ReplayTradingClient(
        SimpleNamespace(
            get_account_info=lambda: {
                "equity": 100000.0,
                "buying_power": 100000.0,
                "cash": 100000.0,
            }
        )
    )

    assert client.get_orders() == []
    assert client.cancel_order_by_id("abc-123") is None
    assert client.cancelled_order_ids == ["abc-123"]


def test_replay_environment_snapshot_records_active_remediation_policy(tmp_path):
    plan_path = tmp_path / "plans" / "morning_game_plan_20260427.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text("{}", encoding="utf-8")
    replay = RuntimeSessionReplay(session_date="2026-04-27")
    replay.project_dir = tmp_path

    snapshot = replay._build_replay_environment_snapshot(
        {
            "plan_path": plan_path,
            "signals_path": None,
            "decision_path": None,
            "pm_plan_path": None,
            "morning_plan_path": plan_path,
            "adjusted_plan_path": None,
            "workflow_journal_path": None,
            "trade_journal_path": None,
        }
    )

    policies = snapshot["active_remediation_policies"]
    assert snapshot["replay_scope"]["uses_current_python_code"] is True
    assert policies["prev_close_entry_drift_guard"]["enabled"] is True
    assert policies["same_day_sell_trim_lockout"]["enabled"] is True
    assert policies["screener_high_score_trap_veto"]["enabled"] is True
    assert snapshot["source_artifacts"]["plan"]["mtime"]


def test_replay_sell_fill_removes_simulated_position():
    """Inverse-ETF sell drains self._positions (the inverse-only list)."""
    replay = RuntimeSessionReplay(
        session_date="2026-04-24",
        market_bars={
            "SQQQ": _bars_for_date("2026-04-24", [10.0, 10.2, 10.4]),
        },
        previous_closes={"SQQQ": 9.9},
    )
    replay._current_replay_time = pd.Timestamp(
        "2026-04-24 09:35", tz="America/Chicago"
    ).to_pydatetime()
    replay._positions = [
        SimpleNamespace(
            symbol="SQQQ",
            qty=100,
            avg_entry_price=10.0,
            avg_entry=10.0,
            entry_price=10.0,
            current_price=10.0,
            market_value=1000.0,
            cost_basis=1000.0,
            unrealized_pl=0.0,
            unrealized_pnl=0.0,
            unrealized_plpc=0.0,
            pnl_pct=0.0,
            side="long",
        )
    ]

    replay._capture_inverse_order(symbol="SQQQ", qty=100, side="sell", context="exit")

    assert replay._positions == []
    assert replay._workflow_action_log[-1]["action"] == "sell_fill"
    assert replay._workflow_action_log[-1]["remaining_qty"] == 0


def test_replay_sell_fill_reduces_simulated_position():
    replay = RuntimeSessionReplay(
        session_date="2026-04-24",
        market_bars={
            "SQQQ": _bars_for_date("2026-04-24", [10.0, 10.2, 10.4]),
        },
        previous_closes={"SQQQ": 9.9},
    )
    replay._current_replay_time = pd.Timestamp(
        "2026-04-24 09:35", tz="America/Chicago"
    ).to_pydatetime()
    replay._positions = [
        SimpleNamespace(
            symbol="SQQQ",
            qty=100,
            avg_entry_price=10.0,
            avg_entry=10.0,
            entry_price=10.0,
            current_price=10.0,
            market_value=1000.0,
            cost_basis=1000.0,
            unrealized_pl=0.0,
            unrealized_pnl=0.0,
            unrealized_plpc=0.0,
            pnl_pct=0.0,
            side="long",
        )
    ]

    replay._capture_inverse_order(symbol="SQQQ", qty=40, side="sell", context="trim")

    assert len(replay._positions) == 1
    assert replay._positions[0].qty == 60
    assert replay._workflow_action_log[-1]["remaining_qty"] == 60


def test_equity_sell_fill_drains_plan_generator_seed_positions():
    """Regression: equity exits must decrement plan_generator's seed_positions.

    Without this, DM keeps re-issuing the same exit every minute because its
    authoritative position view (plan_generator.get_current_positions) never
    sees the exit fill.
    """
    from autotrade.replay.runtime_session_replay import _ReplayPlanGenerator
    from pathlib import Path as _Path

    replay = RuntimeSessionReplay(
        session_date="2026-04-24",
        market_bars={
            "USAR": _bars_for_date("2026-04-24", [10.0, 10.2, 10.4]),
        },
        previous_closes={"USAR": 9.9},
    )
    replay._current_replay_time = pd.Timestamp(
        "2026-04-24 09:35", tz="America/Chicago"
    ).to_pydatetime()

    plan_gen = _ReplayPlanGenerator(
        replay,
        plan_payload={},
        logs_dir=_Path("."),
        positions=[
            {
                "symbol": "USAR",
                "qty": 100,
                "avg_entry": 10.0,
                "entry_price": 10.0,
                "avg_entry_price": 10.0,
            }
        ],
    )
    replay._workflow_agent = SimpleNamespace(plan_generator=plan_gen)

    replay._capture_inverse_order(
        symbol="USAR", qty=60, side="sell", context="execute_exit"
    )
    positions = plan_gen.get_current_positions()
    assert len(positions) == 1
    assert positions[0]["qty"] == 40
    assert replay._workflow_action_log[-1]["action"] == "equity_sell_fill"

    replay._capture_inverse_order(
        symbol="USAR", qty=40, side="sell", context="execute_exit"
    )
    assert plan_gen.get_current_positions() == []


def test_equity_orders_do_not_pollute_inverse_position_list():
    """Regression: equity buys must NOT be appended to self._positions.

    self._positions is reserved for inverse-ETF positions (which have their own
    evaluation loop). Mixing equity positions in there caused the inverse list
    to balloon and produced bogus "Inverse orders: 556" reports.
    """
    replay = RuntimeSessionReplay(
        session_date="2026-04-24",
        market_bars={
            "USAR": _bars_for_date("2026-04-24", [10.0, 10.2, 10.4]),
        },
        previous_closes={"USAR": 9.9},
    )
    replay._current_replay_time = pd.Timestamp(
        "2026-04-24 09:35", tz="America/Chicago"
    ).to_pydatetime()

    replay._capture_inverse_order(
        symbol="USAR", qty=100, side="buy", context="execute_entry_limit"
    )

    assert replay._positions == []
    bucket_counts = replay._summarize_replay_orders()["by_bucket"]
    assert bucket_counts["equity_entry"] == 1
    assert bucket_counts["inverse_etf"] == 0


def test_no_data_buy_is_rejected_as_data_gap_skip():
    """Regression: DM-issued buys against symbols with no minute-bar data
    must be rejected, not booked as phantom entries.

    Before the fix, the replay would record 100s of "entries" against
    price=0 symbols (no archive coverage), inflating the entry count
    versus what production actually traded. They now go into a separate
    data_gap_skipped bucket.
    """
    replay = RuntimeSessionReplay(
        session_date="2026-04-24",
        market_bars={
            "USAR": _bars_for_date("2026-04-24", [10.0, 10.2, 10.4]),
        },
        previous_closes={"USAR": 9.9},
    )
    replay._current_replay_time = pd.Timestamp(
        "2026-04-24 09:35", tz="America/Chicago"
    ).to_pydatetime()

    replay._capture_inverse_order(
        symbol="GHOST", qty=100, side="buy", context="execute_entry_limit"
    )
    replay._capture_inverse_order(
        symbol="USAR", qty=50, side="buy", context="execute_entry_limit"
    )

    breakdown = replay._summarize_replay_orders()
    assert breakdown["by_bucket"]["equity_entry"] == 1
    assert breakdown["data_gap_skipped_total"] == 1
    assert breakdown["data_gap_skipped_unique_symbols"] == 1
    assert "GHOST" in breakdown["data_gap_skipped_unique_symbols_examples"]
    assert breakdown["data_gap_skipped_by_reason"]["no_price_data"] == 1
    # The phantom order should NOT be in _replay_orders.
    assert all(o["symbol"] != "GHOST" for o in replay._replay_orders)


def test_initial_workflow_position_rows_uses_morning_time_window():
    """Regression: seed positions should include all symbols seen in the
    morning window (first 60 min of journal events), not just the first
    sweep until the first repeat.

    Production journals emit position snapshots in sweeps; if the scheduler
    visits 12 holdings, then re-sweeps and adds 4 more before any symbol
    repeats again, the seed should capture all 16 — not break at the first
    repeat at row 13.
    """
    replay = RuntimeSessionReplay(
        session_date="2026-04-24",
        market_bars={},
        previous_closes={},
    )

    def _row(ts: str, sym: str, qty: int = 100, px: float = 10.0):
        return {
            "timestamp": ts,
            "symbol": sym,
            "position": {"symbol": sym, "qty": qty, "avg_entry": px},
        }

    journal = [
        _row("2026-04-24T08:46:00", "ACLS"),
        _row("2026-04-24T08:47:00", "AS"),
        _row("2026-04-24T08:48:00", "AU"),
        _row("2026-04-24T08:49:00", "CELH"),
        # Re-sweep starts here — old logic would break at first repeat
        _row("2026-04-24T08:58:00", "ACLS"),
        _row("2026-04-24T08:59:00", "AS"),
        # New symbols introduced AFTER first repeat but still in morning window
        _row("2026-04-24T09:02:00", "TBI"),
        _row("2026-04-24T09:05:00", "USAR"),
        # After-window event (>60 min from first row) — must be excluded
        _row("2026-04-24T10:30:00", "LATE_ENTRY"),
    ]

    rows = replay._initial_workflow_position_rows(journal)
    symbols = {r["symbol"] for r in rows}
    assert symbols == {"ACLS", "AS", "AU", "CELH", "TBI", "USAR"}
    assert "LATE_ENTRY" not in symbols


def test_replay_order_classification_buckets():
    replay = RuntimeSessionReplay(
        session_date="2026-04-24",
        market_bars={},
        previous_closes={},
    )
    assert (
        replay._classify_replay_order(
            symbol="SQQQ", side="buy", context="execute_entry_limit"
        )
        == "inverse_etf"
    )
    assert (
        replay._classify_replay_order(
            symbol="QID", side="buy", context="inverse_fast_entry"
        )
        == "inverse_etf"
    )
    assert (
        replay._classify_replay_order(
            symbol="USAR", side="buy", context="execute_entry_limit"
        )
        == "equity_entry"
    )
    assert (
        replay._classify_replay_order(
            symbol="USAR", side="sell", context="execute_exit"
        )
        == "equity_exit"
    )
    assert (
        replay._classify_replay_order(
            symbol="USAR", side="sell", context="trim_oversized_position"
        )
        == "trim"
    )


def test_workflow_selected_symbols_are_diagnostic_without_replay_order():
    replay = RuntimeSessionReplay(
        session_date="2026-04-24",
        mode=REPLAY_MODE_AGENT_WORKFLOW,
        market_bars={
            "LONG1": _bars_for_date("2026-04-24", [10.0, 10.2, 10.4]),
        },
    )
    replay._current_replay_time = pd.Timestamp(
        "2026-04-24 09:35", tz="America/Chicago"
    ).to_pydatetime()
    replay._workflow_dm = SimpleNamespace(
        _find_signal_data=lambda symbol: {
            "symbol": symbol,
            "entry_price": 10.0,
            "qty": 10,
        },
        _deployment_floor_status=lambda positions: {},
        _get_entry_authority_state=lambda: {"state": "open"},
    )
    applied = []
    agent = SimpleNamespace(
        plan_generator=SimpleNamespace(
            apply_replay_entry=lambda **kwargs: applied.append(kwargs),
            get_current_positions=lambda: [],
        ),
        _is_wave_breakout_rescue_candidate=lambda **kwargs: False,
        _wave_hard_reject_gap_pct=lambda: 20.0,
        _wave_max_chase_pct=lambda **kwargs: 20.0,
        entry_quality_cfg=SimpleNamespace(wave_breakout_rescue_size_multiplier=0.4),
    )

    replay._capture_replay_cycle_result(
        agent=agent,
        result={"selected_entry_symbols": ["LONG1"]},
        candidate_universe_rows=[
            {"symbol": "LONG1", "entry_price": 10.0, "qty": 10, "score": 90.0}
        ],
        override_reason="",
    )

    assert applied
    assert replay._summarize_replay_orders()["by_bucket"]["equity_entry"] == 0
    assert replay._summarize_workflow_actions()["selected_symbols_total"] == 1
    accounting = replay._build_replay_order_accounting(
        order_breakdown=replay._summarize_replay_orders(),
        workflow_summary=replay._summarize_workflow_actions(),
    )
    assert accounting["actual_replay_order_submissions"] == 0
    assert accounting["diagnostic_only_candidates"] == 1


def test_runtime_session_replay_flags_recorded_longs_blocked_by_bearish_runtime(
    tmp_path: Path,
):
    logs_dir = tmp_path / "logs"
    plans_dir = tmp_path / "plans"
    logs_dir.mkdir()
    plans_dir.mkdir()

    (logs_dir / "signals_2026-03-18.json").write_text(
        """
        {
          "signals": [
            {
              "ticker": "LONG1",
              "score": 84.0,
              "plan_score_source": "adjusted_plan_20260318_0822.json",
              "sector": "energy"
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )
    (logs_dir / "trade_decisions_20260318.json").write_text(
        """
        [
          {
            "timestamp": "2026-03-18T08:32:10",
            "symbol": "LONG1",
            "score": 84.0,
            "actually_executed": true
          }
        ]
        """.strip(),
        encoding="utf-8",
    )
    (plans_dir / "adjusted_plan_20260318_0822.json").write_text(
        """
        {
          "resolved_regime": {
            "regime": "SELLOFF",
            "allow_new_longs": false
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    def _inverse_provider(replay_time, regime, portfolio_holdings=None):
        if replay_time and replay_time.minute >= 31:
            return [
                {
                    "ticker": "SQQQ",
                    "signal": "ENTRY",
                    "entry_price": 20.0,
                    "composite_score": 91,
                }
            ]
        return []

    replay = RuntimeSessionReplay(
        session_date="2026-03-18",
        project_dir=tmp_path,
        force_modern_crash_protection=True,
        market_bars={
            "SPY": _bars([100.0, 97.0, 96.5]),
            "QQQ": _bars([200.0, 194.0, 193.5]),
            "IWM": _bars([50.0, 48.0, 47.8]),
            "SQQQ": _bars([20.0, 20.4, 21.1]),
        },
        previous_closes={"SPY": 102.0, "QQQ": 205.0, "IWM": 51.0},
        inverse_screen_provider=_inverse_provider,
    )

    report = replay.run(persist=False)

    assert report["summary"]["first_inverse_fast_at"].startswith("2026-03-18T08:31")
    assert report["summary"]["inverse_orders_captured"] == 1
    assert report["summary"]["inverse_trades_closed"] == 1
    assert report["summary"]["inverse_net_pnl"] > 0
    assert report["long_decision_checks"][0]["replay_long_allowed"] is False
    assert report["inverse_trade_results"][0]["symbol"] == "SQQQ"
    assert report["inverse_trade_results"][0]["exit_reason"] == "profit_target_hit"
    assert (
        report["divergences"][0]["type"]
        == "recorded_long_executed_while_replay_blocked"
    )


def test_runtime_session_replay_forces_inverse_eod_exit_when_no_intraday_trigger(
    tmp_path: Path,
):
    logs_dir = tmp_path / "logs"
    plans_dir = tmp_path / "plans"
    logs_dir.mkdir()
    plans_dir.mkdir()

    (logs_dir / "trade_decisions_20260318.json").write_text("[]", encoding="utf-8")
    (plans_dir / "adjusted_plan_20260318_0822.json").write_text(
        """
        {
          "resolved_regime": {
            "regime": "SELLOFF",
            "allow_new_longs": false
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    def _inverse_provider(replay_time, regime, portfolio_holdings=None):
        if replay_time and replay_time.minute == 31:
            return [
                {
                    "ticker": "DOG",
                    "signal": "ENTRY",
                    "entry_price": 30.0,
                    "composite_score": 88,
                }
            ]
        return []

    replay = RuntimeSessionReplay(
        session_date="2026-03-18",
        project_dir=tmp_path,
        force_modern_crash_protection=True,
        market_bars={
            "SPY": _bars([100.0, 99.5, 99.4]),
            "QQQ": _bars([200.0, 199.4, 199.2]),
            "IWM": _bars([50.0, 49.8, 49.7]),
            "DOG": _bars([30.0, 30.05, 30.08]),
        },
        previous_closes={"SPY": 102.0, "QQQ": 205.0, "IWM": 51.0},
        inverse_screen_provider=_inverse_provider,
    )

    report = replay.run(persist=False)

    assert report["summary"]["inverse_orders_captured"] == 1
    assert report["summary"]["inverse_trades_closed"] == 1
    assert report["inverse_trade_results"][0]["symbol"] == "DOG"
    assert report["inverse_trade_results"][0]["exit_reason"] == "eod_no_overnight"


def test_runtime_session_replay_reports_counterfactual_gapup_long_pnl(tmp_path: Path):
    replay = RuntimeSessionReplay(
        session_date="2026-03-18",
        project_dir=tmp_path,
        market_bars={
            "SPY": _bars([100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7]),
            "QQQ": _bars([200.0, 200.1, 200.2, 200.3, 200.4, 200.5, 200.6, 200.7]),
            "IWM": _bars([50.0, 50.1, 50.2, 50.3, 50.4, 50.5, 50.6, 50.7]),
            "BASE": _bars([10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7]),
            "GAPX": _bars([10.3, 10.4, 10.5, 10.6, 10.7, 10.9, 11.2, 11.4]),
        },
        previous_closes={
            "SPY": 99.5,
            "QQQ": 199.5,
            "IWM": 49.8,
            "BASE": 10.0,
            "GAPX": 10.0,
        },
    )

    fake_dm = SimpleNamespace(
        entry_quality_cfg=SimpleNamespace(entry_gap_reject_pct=7.0),
        premarket_gap_cfg=SimpleNamespace(moderate_gap_up_pct=3.0),
        position_entries={},
        _entry_time_overrides={},
        _safe_float=lambda value, default=0.0: (
            float(value) if value not in (None, "") else float(default)
        ),
        _effective_market_regime=lambda: "CHOP",
        _refresh_entry_authority_state=lambda positions: {
            "state": "open",
            "reason": "test",
        },
        _entries_blocked_by_regime=lambda symbol: (False, ""),
        _resolve_entry_authority=lambda signal_data: {
            "eligible": True,
            "reason": "",
            "execution_mode": {
                "entry_authority_state": "open",
                "resolved_regime": {"regime": "CHOP", "allow_new_longs": True},
            },
        },
        calculate_position_health=lambda position: {"action": "hold"},
    )
    fake_dm._resolve_runtime_entry_anchor = lambda **kwargs: (
        DayManager._resolve_runtime_entry_anchor(fake_dm, **kwargs)
    )

    counterfactual = replay._evaluate_counterfactual_longs(
        dm=fake_dm,
        signals=[
            {
                "ticker": "BASE",
                "score": 84.0,
                "entry_price": 10.0,
                "qty": 20,
                "risk_reward": 2.0,
                "volume_ratio": 1.8,
                "setup_type": "trend_follow",
            },
            {
                "ticker": "GAPX",
                "score": 72.0,
                "entry_price": 10.0,
                "qty": 20,
                "risk_reward": 2.1,
                "volume_ratio": 2.0,
                "setup_type": "opening_range_breakout",
            },
        ],
        decisions=[
            {
                "timestamp": "2026-03-18T08:32:10",
                "symbol": "BASE",
                "score": 84.0,
                "actually_executed": True,
            }
        ],
        trade_journal=[],
        actual_day_net_pnl=-42.24,
        capacity_snapshot={
            "available_capital": 250.0,
            "available_slots": 1,
            "source": "unit_test",
        },
    )

    assert counterfactual["eligible_candidates_total"] >= 1
    assert counterfactual["selected_candidates_total"] == 1
    assert counterfactual["net_synthetic_pnl_selected"] > 0
    assert counterfactual["estimated_day_net_pnl_selected"] > -42.24
    top = counterfactual["top_added_candidates"][0]
    assert top["symbol"] == "GAPX"
    assert top["runtime_anchor_reason"] == "planned_entry_gapup_exception"
    assert top["synthetic_pnl"] > 0
    assert top["management_style"] == "daymanager_position_health"
    assert top["exit_reason"] == "eod_close"


def test_runtime_session_replay_blocks_counterfactual_without_planned_size(
    tmp_path: Path,
):
    replay = RuntimeSessionReplay(
        session_date="2026-03-18",
        project_dir=tmp_path,
        market_bars={
            "SPY": _bars([100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7]),
            "QQQ": _bars([200.0, 200.1, 200.2, 200.3, 200.4, 200.5, 200.6, 200.7]),
            "IWM": _bars([50.0, 50.1, 50.2, 50.3, 50.4, 50.5, 50.6, 50.7]),
            "GAPX": _bars([10.3, 10.4, 10.5, 10.6, 10.7, 10.9, 11.2, 11.4]),
        },
        previous_closes={"SPY": 99.5, "QQQ": 199.5, "IWM": 49.8, "GAPX": 10.0},
    )

    fake_dm = SimpleNamespace(
        entry_quality_cfg=SimpleNamespace(entry_gap_reject_pct=7.0),
        premarket_gap_cfg=SimpleNamespace(moderate_gap_up_pct=3.0),
        position_entries={},
        _entry_time_overrides={},
        _safe_float=lambda value, default=0.0: (
            float(value) if value not in (None, "") else float(default)
        ),
        _effective_market_regime=lambda: "CHOP",
        _refresh_entry_authority_state=lambda positions: {
            "state": "open",
            "reason": "test",
        },
        _entries_blocked_by_regime=lambda symbol: (False, ""),
        _resolve_entry_authority=lambda signal_data: {
            "eligible": True,
            "reason": "",
            "execution_mode": {
                "entry_authority_state": "open",
                "resolved_regime": {"regime": "CHOP", "allow_new_longs": True},
            },
        },
        calculate_position_health=lambda position: {"action": "hold"},
    )
    fake_dm._resolve_runtime_entry_anchor = lambda **kwargs: (
        DayManager._resolve_runtime_entry_anchor(fake_dm, **kwargs)
    )

    counterfactual = replay._evaluate_counterfactual_longs(
        dm=fake_dm,
        signals=[
            {
                "ticker": "GAPX",
                "score": 72.0,
                "entry_price": 10.0,
                "risk_reward": 2.1,
                "volume_ratio": 2.0,
                "setup_type": "opening_range_breakout",
            }
        ],
        decisions=[],
        trade_journal=[],
        actual_day_net_pnl=0.0,
        capacity_snapshot={
            "available_capital": 250.0,
            "available_slots": 1,
            "source": "unit_test",
        },
    )

    assert counterfactual["eligible_candidates_total"] == 0
    assert counterfactual["selected_candidates_total"] == 0
    assert counterfactual["blocked_or_skipped_candidates"][0]["reasons"] == [
        "missing_planned_size"
    ]


def test_runtime_session_replay_counterfactual_uses_management_exit(tmp_path: Path):
    replay = RuntimeSessionReplay(
        session_date="2026-03-18",
        project_dir=tmp_path,
        market_bars={
            "SPY": _bars([100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7]),
            "QQQ": _bars([200.0, 200.1, 200.2, 200.3, 200.4, 200.5, 200.6, 200.7]),
            "IWM": _bars([50.0, 50.1, 50.2, 50.3, 50.4, 50.5, 50.6, 50.7]),
            "EXITX": _bars([10.0, 10.0, 10.0, 10.0, 9.8, 9.6, 9.4, 9.3]),
        },
        previous_closes={"SPY": 99.5, "QQQ": 199.5, "IWM": 49.8, "EXITX": 10.0},
    )

    fake_dm = SimpleNamespace(
        entry_quality_cfg=SimpleNamespace(entry_gap_reject_pct=7.0),
        premarket_gap_cfg=SimpleNamespace(moderate_gap_up_pct=3.0),
        position_entries={},
        _entry_time_overrides={},
        _safe_float=lambda value, default=0.0: (
            float(value) if value not in (None, "") else float(default)
        ),
        _effective_market_regime=lambda: "CHOP",
        _refresh_entry_authority_state=lambda positions: {
            "state": "open",
            "reason": "test",
        },
        _entries_blocked_by_regime=lambda symbol: (False, ""),
        _resolve_entry_authority=lambda signal_data: {
            "eligible": True,
            "reason": "",
            "execution_mode": {
                "entry_authority_state": "open",
                "resolved_regime": {"regime": "CHOP", "allow_new_longs": True},
            },
        },
        calculate_position_health=lambda position: {
            "action": "exit"
            if float(getattr(position, "pnl_pct", 0.0) or 0.0) <= -2.0
            else "hold"
        },
    )
    fake_dm._resolve_runtime_entry_anchor = lambda **kwargs: (
        DayManager._resolve_runtime_entry_anchor(fake_dm, **kwargs)
    )

    counterfactual = replay._evaluate_counterfactual_longs(
        dm=fake_dm,
        signals=[
            {
                "ticker": "EXITX",
                "score": 74.0,
                "entry_price": 10.0,
                "qty": 20,
                "risk_reward": 2.0,
                "volume_ratio": 1.4,
                "setup_type": "trend_follow",
            }
        ],
        decisions=[],
        trade_journal=[],
        actual_day_net_pnl=0.0,
        capacity_snapshot={
            "available_capital": 250.0,
            "available_slots": 1,
            "source": "unit_test",
        },
    )

    top = counterfactual["top_added_candidates"][0]
    assert top["first_management_action"] == "exit"
    assert top["exit_reason"] == "management_exit"
    assert top["synthetic_pnl"] < 0


def test_runtime_session_replay_backfills_signal_metadata_from_adjusted_plan(
    tmp_path: Path,
):
    logs_dir = tmp_path / "logs"
    plans_dir = tmp_path / "plans"
    logs_dir.mkdir()
    plans_dir.mkdir()

    (logs_dir / "signals_2026-03-18.json").write_text(
        """
        {
          "signals": [
            {
              "ticker": "GAPX",
              "score": 72.0,
              "entry_price": 10.0,
              "qty": 20
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )
    (plans_dir / "adjusted_plan_20260318_0822.json").write_text(
        """
        {
          "resolved_regime": {
            "regime": "CHOP",
            "allow_new_longs": true
          },
          "signals": [
            {
              "ticker": "GAPX",
              "score": 72.0,
              "entry_price": 10.0,
              "qty": 20,
              "entry_source": "premarket_adjusted",
              "source_bucket": "watchlist",
              "plan_score_source": "adjusted_plan_20260318_0822.json",
              "setup_type": "opening_range_breakout"
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    replay = RuntimeSessionReplay(session_date="2026-03-18", project_dir=tmp_path)
    artifacts = replay._resolve_artifacts()
    raw_signals = replay._load_signals(artifacts)
    enriched = replay._augment_signals_with_plan_metadata(
        raw_signals, artifacts=artifacts
    )

    assert len(enriched) == 1
    assert enriched[0]["ticker"] == "GAPX"
    assert enriched[0]["qty"] == 20
    assert enriched[0]["entry_source"] == "premarket_adjusted"
    assert enriched[0]["source_bucket"] == "watchlist"
    assert enriched[0]["plan_score_source"] == "adjusted_plan_20260318_0822.json"
    assert enriched[0]["setup_type"] == "opening_range_breakout"
    assert enriched[0]["_candidate_source_plan"] == "adjusted_plan"
    assert enriched[0]["_replay_plan_metadata_backfilled"] is True


def test_runtime_session_replay_replaces_stale_zero_capacity_snapshot_from_executed_budget(
    tmp_path: Path,
):
    replay = RuntimeSessionReplay(session_date="2026-03-18", project_dir=tmp_path)
    capacity = replay._resolve_counterfactual_capacity(
        capacity_snapshot={
            "available_capital": 0.0,
            "available_slots": 50,
            "source": "decision_claw_review",
            "phase_agent": "premarket",
            "positions_count": 0,
            "current_positions_count": 0,
            "reference_account_buying_power": 250000.0,
        },
        decisions=[
            {
                "symbol": "BASE",
                "actually_executed": True,
                "planned_price": 10.0,
                "adjusted_qty": 20,
            }
        ],
        signals=[
            {
                "ticker": "BASE",
                "entry_price": 10.0,
                "qty": 20,
            }
        ],
    )

    assert capacity["source"] == "executed_decision_budget_fallback"
    assert capacity["mode"] == "replacement_budget"
    assert capacity["available_capital"] == 200.0
    assert capacity["available_slots"] == 1
    assert capacity["stale_snapshot"] is True
    assert capacity["stale_reason"] == "premarket_open_slots_with_zero_free_capital"
    assert capacity["reference_account_buying_power"] == 250000.0


def test_runtime_session_replay_extracts_pm_plan_buying_power_reference(tmp_path: Path):
    replay = RuntimeSessionReplay(session_date="2026-03-18", project_dir=tmp_path)

    snapshot = replay._extract_counterfactual_capacity_snapshot(
        {
            "decision_claw": {
                "review": {
                    "phase_agent": "premarket",
                    "evidence_summary": {
                        "free_capital": 0.0,
                        "open_slots": 50,
                        "positions_count": 0,
                        "current_positions": [],
                    },
                }
            }
        },
        {
            "account": {
                "buying_power": 352923.38,
            }
        },
    )

    assert snapshot["available_capital"] == 0.0
    assert snapshot["available_slots"] == 50
    assert snapshot["reference_account_buying_power"] == 352923.38


def test_runtime_session_replay_inferrs_candidate_size_from_executed_budget(
    tmp_path: Path,
):
    replay = RuntimeSessionReplay(session_date="2026-03-18", project_dir=tmp_path)

    row = {
        "ticker": "MISS1",
        "entry_price": 20.0,
        "entry_source": "premarket_adjusted",
        "source_bucket": "watchlist",
        "setup_type": "opening_range_breakout",
    }
    qty = replay._infer_counterfactual_qty_from_decisions(
        row,
        decisions=[
            {
                "symbol": "EXEC1",
                "actually_executed": True,
                "planned_price": 10.0,
                "adjusted_qty": 100,
                "entry_source": "premarket_adjusted",
                "source_bucket": "watchlist",
                "setup_type": "opening_range_breakout",
            },
            {
                "symbol": "EXEC2",
                "actually_executed": True,
                "planned_price": 12.0,
                "adjusted_qty": 100,
                "entry_source": "premarket_adjusted",
                "source_bucket": "watchlist",
                "setup_type": "opening_range_breakout",
            },
        ],
    )

    assert qty == 55.0
    assert row["_replay_inferred_notional"] == 1100.0
    assert row["_replay_inferred_size_source"] == "executed_decision_exact_median"


def test_runtime_session_replay_replays_position_management_against_workflow_journal(
    tmp_path: Path,
):
    logs_dir = tmp_path / "logs"
    plans_dir = tmp_path / "plans"
    logs_dir.mkdir()
    plans_dir.mkdir()

    (logs_dir / "trade_decisions_20260318.json").write_text("[]", encoding="utf-8")
    (logs_dir / "workflow_journal_2026-03-18.jsonl").write_text(
        """
{"timestamp":"2026-03-18T10:45:00","symbol":"LONG1","position":{"symbol":"LONG1","entry_price":20.0,"current_price":19.92,"qty":100,"unrealized_plpc":-0.004,"entry_time":"2026-03-18T09:00:00","cost_basis":2000.0,"market_value":1992.0},"final_action":{"action":"hold"},"execution":{"executed":false}}
        """.strip(),
        encoding="utf-8",
    )
    (plans_dir / "adjusted_plan_20260318_0822.json").write_text(
        """
        {
          "resolved_regime": {
            "regime": "SELLOFF",
            "allow_new_longs": false
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    replay = RuntimeSessionReplay(
        session_date="2026-03-18",
        project_dir=tmp_path,
        force_modern_crash_protection=True,
        market_bars={
            "SPY": _bars([100.0, 99.2, 98.9]),
            "QQQ": _bars([200.0, 198.5, 198.0]),
            "IWM": _bars([50.0, 49.2, 49.0]),
        },
        previous_closes={"SPY": 102.0, "QQQ": 205.0, "IWM": 51.0},
    )

    report = replay.run(persist=False)
    management = report["position_management_replay"]

    assert management["summary"]["events_replayed"] == 1
    assert management["summary"]["action_differences"] == 0
    assert management["summary"]["exit_like_signals"] == 0
    assert management["events"][0]["recorded_action"] == "hold"
    assert management["events"][0]["replay_action"] == "hold"
    assert management["first_exit_like_by_symbol"] == []


def test_runtime_session_replay_loads_trade_journal_trims_into_management_replay(
    tmp_path: Path,
):
    logs_dir = tmp_path / "logs"
    plans_dir = tmp_path / "plans"
    logs_dir.mkdir()
    plans_dir.mkdir()
    (logs_dir / "trade_journal.json").write_text(
        """
        {
          "trades": [
            {
              "symbol": "TRIM1",
              "trade_type": "trim",
              "entry_time": "2026-03-17T14:00:00+00:00",
              "exit_time": "2026-03-18T15:00:00+00:00",
              "entry_price": 20.0,
              "exit_price": 22.0,
              "quantity": 10
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )
    replay = RuntimeSessionReplay(session_date="2026-03-18", project_dir=tmp_path)
    trade_journal = replay._load_trade_journal(
        {"trade_journal_path": logs_dir / "trade_journal.json"}
    )
    events = replay._trade_journal_management_events(trade_journal)
    dm = SimpleNamespace(
        _refresh_entry_authority_state=lambda **kwargs: None,
        _entry_time_overrides={},
        position_entries={},
        calculate_position_health=lambda position: {"action": "trim", "signals": []},
    )

    management = replay._replay_position_management(dm=dm, journal_entries=events)

    assert len(trade_journal) == 1
    assert management["summary"]["events_replayed"] == 1
    assert management["summary"]["recorded_trim_events"] == 1
    assert management["summary"]["trim_signals"] == 1
    assert management["events"][0]["source"] == "trade_journal"


def test_agent_workflow_position_seed_uses_initial_journal_sweep_only(tmp_path: Path):
    replay = RuntimeSessionReplay(session_date="2026-03-18", project_dir=tmp_path)
    workflow_journal = [
        {
            "timestamp": "2026-03-18T08:46:00",
            "symbol": "HELD1",
            "position": {"symbol": "HELD1", "qty": 10, "entry_price": 10.0},
        },
        {
            "timestamp": "2026-03-18T08:47:00",
            "symbol": "HELD2",
            "position": {"symbol": "HELD2", "qty": 20, "entry_price": 20.0},
        },
        {
            "timestamp": "2026-03-18T08:58:00",
            "symbol": "HELD1",
            "position": {"symbol": "HELD1", "qty": 10, "entry_price": 10.0},
        },
        {
            "timestamp": "2026-03-18T12:00:00",
            "symbol": "FUTURE",
            "position": {"symbol": "FUTURE", "qty": 30, "entry_price": 30.0},
        },
    ]
    trade_journal = [
        {
            "symbol": "TRIMONLY",
            "trade_type": "trim",
            "entry_time": "2026-03-17T14:00:00+00:00",
            "exit_time": "2026-03-18T15:00:00+00:00",
            "entry_price": 40.0,
            "exit_price": 42.0,
            "quantity": 5,
        }
    ]

    seeded = replay._seed_workflow_positions(
        decisions=[],
        workflow_journal=workflow_journal,
        trade_journal=trade_journal,
    )

    assert [row["symbol"] for row in seeded] == ["HELD1", "HELD2"]


def test_management_replay_coverage_flags_zero_events_with_production_trims(
    tmp_path: Path,
):
    replay = RuntimeSessionReplay(session_date="2026-03-18", project_dir=tmp_path)

    diagnostics = replay._build_management_coverage_diagnostics(
        management_replay={"summary": {"events_replayed": 0}},
        actual_day_context={"trims_journaled": 2, "exits_journaled": 0},
        workflow_journal=[],
        trade_journal_management_events=[],
    )

    assert diagnostics["status"] == "incomplete"
    assert diagnostics["problem"] == "production_management_events_not_replayed"


def test_runtime_session_replay_emits_hard_stop_audit_rows(tmp_path: Path):
    replay = RuntimeSessionReplay(
        session_date="2026-03-18",
        project_dir=tmp_path,
    )
    dm = SimpleNamespace(
        _refresh_entry_authority_state=lambda **kwargs: None,
        _entry_time_overrides={},
        position_entries={},
        calculate_position_health=lambda position: {
            "action": "exit",
            "signals": ["HARD STOP: -27.1% <= -15.0%"],
            "pnl_pct": -27.1,
            "hard_stop_forced_exit": True,
            "hard_stop_pct": -15.0,
        },
    )
    journal_entries = [
        {
            "timestamp": "2026-03-18T10:45:00-05:00",
            "symbol": "ACLS",
            "position": {
                "symbol": "ACLS",
                "entry_price": 115.275556,
                "current_price": 84.08,
                "qty": 17,
                "unrealized_plpc": -0.27062,
                "entry_time": "2026-03-18T09:00:00-05:00",
                "cost_basis": 1959.684452,
                "market_value": 1429.36,
            },
            "final_action": {"action": "hold"},
        }
    ]

    management = replay._replay_position_management(
        dm=dm, journal_entries=journal_entries
    )

    assert management["summary"]["events_replayed"] == 1
    assert management["summary"]["hard_stop_exit_signals"] == 1
    assert management["hard_stop_audit_rows"][0]["symbol"] == "ACLS"
    assert management["hard_stop_audit_rows"][0]["action"] == "hard_stop_exit"
    assert management["hard_stop_audit_rows"][0]["hard_stop_pct"] == -15.0
    assert management["first_hard_stop_by_symbol"][0]["symbol"] == "ACLS"
    assert management["events"][0]["hard_stop_forced_exit"] is True


def test_stop_loss_diagnostics_counts_repeated_unresolved_warnings(tmp_path: Path):
    replay = RuntimeSessionReplay(session_date="2026-03-18", project_dir=tmp_path)

    diagnostics = replay._build_stop_loss_diagnostics(
        {
            "events": [
                {
                    "symbol": "LOSS1",
                    "recorded_action": "stop",
                    "replay_action": "watch",
                    "hard_stop_forced_exit": True,
                },
                {
                    "symbol": "LOSS1",
                    "recorded_action": "stop",
                    "replay_action": "watch",
                    "hard_stop_forced_exit": True,
                },
            ]
        }
    )

    assert diagnostics["unresolved_stop_loss_warnings"] == 2
    assert diagnostics["repeated_unresolved_stop_loss_warnings"] == [
        {"symbol": "LOSS1", "count": 2}
    ]
    assert diagnostics["status"] == "degraded"


def test_runtime_session_replay_allows_missing_decision_artifact(tmp_path: Path):
    logs_dir = tmp_path / "logs"
    plans_dir = tmp_path / "plans"
    logs_dir.mkdir()
    plans_dir.mkdir()

    (plans_dir / "adjusted_plan_20260318_0822.json").write_text(
        """
        {
          "resolved_regime": {
            "regime": "SELLOFF",
            "allow_new_longs": false
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    replay = RuntimeSessionReplay(
        session_date="2026-03-18",
        project_dir=tmp_path,
        market_bars={
            "SPY": _bars([100.0, 99.5, 99.4]),
            "QQQ": _bars([200.0, 199.4, 199.2]),
            "IWM": _bars([50.0, 49.8, 49.7]),
        },
        previous_closes={"SPY": 102.0, "QQQ": 205.0, "IWM": 51.0},
    )

    report = replay.run(persist=False)

    assert report["summary"]["recorded_decisions_total"] == 0
    assert report["summary"]["decision_artifact_missing"] is True
    assert report["artifacts"]["decision_file"] == ""
    assert "Decision artifact missing" in report["replay_notes"][0]


def test_runtime_session_replay_prefers_local_minute_archive(
    tmp_path: Path, monkeypatch
):
    logs_dir = tmp_path / "logs"
    plans_dir = tmp_path / "plans"
    logs_dir.mkdir()
    plans_dir.mkdir()

    (plans_dir / "adjusted_plan_20260318_0822.json").write_text(
        """
        {
          "resolved_regime": {
            "regime": "CHOP",
            "allow_new_longs": true
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    archive_client = _ReplayArchiveClient(
        minute_bars={
            "SPY": _bars([100.0, 100.1, 100.2]),
            "QQQ": _bars([200.0, 200.2, 200.4]),
            "IWM": _bars([50.0, 50.1, 50.2]),
        }
    )
    archive_session_minute_bars(
        db_path=tmp_path / "data" / "replay_minute_bars.duckdb",
        session_date="2026-03-18",
        client=archive_client,
        symbol_sources={
            "SPY": {"benchmark"},
            "QQQ": {"benchmark"},
            "IWM": {"benchmark"},
        },
        preferred_start_et=time(4, 0),
        preferred_end_et=time(16, 0),
        fallback_regular_session_only=True,
    )

    def _unexpected_client(*args, **kwargs):
        raise AssertionError(
            "live fetch should not run when archive coverage is complete"
        )

    monkeypatch.setattr(
        "autotrade.replay.runtime_session_replay.create_data_client",
        _unexpected_client,
    )

    replay = RuntimeSessionReplay(
        session_date="2026-03-18",
        project_dir=tmp_path,
        previous_closes={"SPY": 99.5, "QQQ": 199.5, "IWM": 49.8},
    )

    report = replay.run(persist=False)

    assert report["summary"]["archive_found"] is True
    assert report["summary"]["market_data_coverage_mode"] == "fully_local_archive"
    assert report["summary"]["archive_symbols_loaded"] == 3
    assert report["summary"]["archive_live_fetch_symbols"] == 0


def test_runtime_session_replay_live_fetches_only_archive_gaps(
    tmp_path: Path, monkeypatch
):
    logs_dir = tmp_path / "logs"
    plans_dir = tmp_path / "plans"
    logs_dir.mkdir()
    plans_dir.mkdir()

    (plans_dir / "adjusted_plan_20260318_0822.json").write_text(
        """
        {
          "resolved_regime": {
            "regime": "CHOP",
            "allow_new_longs": true
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    archive_client = _ReplayArchiveClient(
        minute_bars={
            "SPY": _bars([100.0, 100.1, 100.2]),
            "QQQ": _bars([200.0, 200.2, 200.4]),
        }
    )
    archive_session_minute_bars(
        db_path=tmp_path / "data" / "replay_minute_bars.duckdb",
        session_date="2026-03-18",
        client=archive_client,
        symbol_sources={"SPY": {"benchmark"}, "QQQ": {"benchmark"}},
        preferred_start_et=time(4, 0),
        preferred_end_et=time(16, 0),
        fallback_regular_session_only=True,
    )

    live_client = _ReplayArchiveClient(minute_bars={"IWM": _bars([50.0, 50.1, 50.2])})
    monkeypatch.setattr(
        "autotrade.replay.runtime_session_replay.create_data_client",
        lambda *args, **kwargs: live_client,
    )

    replay = RuntimeSessionReplay(
        session_date="2026-03-18",
        project_dir=tmp_path,
        previous_closes={"SPY": 99.5, "QQQ": 199.5, "IWM": 49.8},
    )

    report = replay.run(persist=False)

    assert (
        report["summary"]["market_data_coverage_mode"] == "mixed_archive_and_live_fetch"
    )
    assert sorted(report["market_data_archive"]["loaded_symbols"]) == ["QQQ", "SPY"]
    assert report["market_data_archive"]["live_fetch_symbols"] == ["IWM"]


def test_agent_workflow_replay_runs_market_phases_in_isolated_workspace(tmp_path: Path):
    logs_dir = tmp_path / "logs"
    plans_dir = tmp_path / "plans"
    logs_dir.mkdir()
    plans_dir.mkdir()

    (plans_dir / "adjusted_plan_20260318_0822.json").write_text(
        """
        {
          "resolved_regime": {
            "regime": "CHOP",
            "allow_new_longs": true
          },
          "entry_orders": [
            {
              "symbol": "LONG1",
              "entry_price": 10.0,
              "qty": 25,
              "score": 84.0,
              "entry_source": "adjusted_plan",
              "source_bucket": "watchlist"
            }
          ],
          "signals": [
            {
              "symbol": "LONG1",
              "entry_price": 10.0,
              "qty": 25,
              "score": 84.0
            }
          ],
          "full_watchlist": [
            {
              "symbol": "LONG1",
              "entry_price": 10.0,
              "qty": 25,
              "score": 84.0,
              "deep_research_bridge": true
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    replay = RuntimeSessionReplay(
        session_date="2026-03-18",
        mode=REPLAY_MODE_AGENT_WORKFLOW,
        project_dir=tmp_path,
        market_bars={
            "SPY": _bars([100.0] * 391),
            "QQQ": _bars([200.0] * 391),
            "IWM": _bars([50.0] * 391),
            "LONG1": _bars([10.0 + (idx * 0.01) for idx in range(391)]),
        },
        previous_closes={"SPY": 99.5, "QQQ": 199.5, "IWM": 49.8},
    )

    report = replay.run(persist=False)

    assert report["summary"]["replay_mode"] == REPLAY_MODE_AGENT_WORKFLOW
    assert report["summary"]["workflow_selected_symbols_total"] >= 1
    assert report["summary"]["market_open_cycles"] == 15
    assert report["summary"]["market_hours_cycles"] == 376
    assert not (tmp_path / "plans" / ".execution_state_20260318.json").exists()
    assert not (tmp_path / "logs" / "trade_decisions_20260318.json").exists()


def test_agent_workflow_replay_surfaces_held_strength_adds(tmp_path: Path):
    logs_dir = tmp_path / "logs"
    plans_dir = tmp_path / "plans"
    logs_dir.mkdir()
    plans_dir.mkdir()

    (plans_dir / "adjusted_plan_20260318_0822.json").write_text(
        """
        {
          "resolved_regime": {
            "regime": "SELL_OFF",
            "allow_new_longs": false
          }
        }
        """.strip(),
        encoding="utf-8",
    )
    (logs_dir / "workflow_journal_2026-03-18.jsonl").write_text(
        """
        {"timestamp":"2026-03-18T14:40:00+00:00","symbol":"LONG1","position":{"symbol":"LONG1","qty":5,"entry_price":10.0,"avg_entry":10.0}}
        """.strip(),
        encoding="utf-8",
    )

    long_prices = []
    for idx in range(391):
        if idx < 60:
            long_prices.append(10.0 * (1 - (0.006 * ((idx + 1) / 60.0))))
        elif idx < 75:
            progress = (idx - 59) / 15.0
            long_prices.append(9.94 - ((9.94 - 9.83) * progress))
        elif idx < 91:
            progress = (idx - 74) / 16.0
            long_prices.append(9.83 + ((10.28 - 9.83) * progress))
        else:
            progress = (idx - 90) / 300.0
            long_prices.append(10.28 + (0.42 * progress))

    report = RuntimeSessionReplay(
        session_date="2026-03-18",
        mode=REPLAY_MODE_AGENT_WORKFLOW,
        project_dir=tmp_path,
        market_bars={
            "SPY": _bars([100.0 - (idx * 0.01) for idx in range(391)]),
            "QQQ": _bars([200.0 - (idx * 0.02) for idx in range(391)]),
            "IWM": _bars([50.0 - (idx * 0.01) for idx in range(391)]),
            "LONG1": _bars(long_prices),
        },
        previous_closes={"SPY": 100.5, "QQQ": 201.0, "IWM": 50.5, "LONG1": 9.95},
    ).run(persist=False)

    assert report["summary"]["workflow_add_selected_total"] >= 1
    # workflow_bullish_actions_total used to be the meaningless sum
    # selections + adds; replaced by the unique-symbol counters below.
    assert report["summary"]["replay_adds_executed"] >= 1
    assert report["summary"]["workflow_bullish_mark_to_close_pnl"] > 0
    assert "deployment_floor_final_pct" in report["summary"]
    assert report["workflow_replay"]["deployment_floor_samples"]
    traces = report["workflow_replay"]["held_strength_latest"]
    long1 = next(row for row in traces if row["symbol"] == "LONG1")
    assert long1["status"] == "add_ready"


def test_replay_runtime_session_defaults_daily_replay_to_agent_workflow(tmp_path: Path):
    logs_dir = tmp_path / "logs"
    plans_dir = tmp_path / "plans"
    logs_dir.mkdir()
    plans_dir.mkdir()

    (plans_dir / "adjusted_plan_20260318_0822.json").write_text(
        """
        {
          "resolved_regime": {
            "regime": "CHOP",
            "allow_new_longs": true
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    report = replay_runtime_session(
        "2026-03-18",
        project_dir=tmp_path,
        market_bars={
            "SPY": _bars([100.0, 100.1, 100.2]),
            "QQQ": _bars([200.0, 200.1, 200.2]),
            "IWM": _bars([50.0, 50.1, 50.2]),
        },
        previous_closes={"SPY": 99.5, "QQQ": 199.5, "IWM": 49.8},
    )

    assert report["summary"]["replay_mode"] == REPLAY_MODE_AGENT_WORKFLOW


def test_runtime_session_replay_backfills_signal_metadata_from_trade_journal(
    tmp_path: Path,
):
    logs_dir = tmp_path / "logs"
    plans_dir = tmp_path / "plans"
    logs_dir.mkdir()
    plans_dir.mkdir()

    (logs_dir / "trade_decisions_20260318.json").write_text(
        """
        [
          {
            "timestamp": "2026-03-18T08:32:10",
            "symbol": "LONG1",
            "score": 84.0,
            "actually_executed": true
          }
        ]
        """.strip(),
        encoding="utf-8",
    )
    (logs_dir / "trade_journal.json").write_text(
        """
        [
          {
            "timestamp": "2026-03-18T14:32:11+00:00",
            "symbol": "LONG1",
            "entry_score": 84.0,
            "plan_score_source": "adjusted_plan_20260318_0822.json",
            "entry_source": "premarket_adjusted",
            "origin_entry_source": "overnight_full_watchlist",
            "runtime_entry_context": "overnight_first_hour_recheck",
            "override_reason": "overnight_first_hour_recheck",
            "setup_type": "opening_drive",
            "strategy_name": "opening_drive_factory"
          }
        ]
        """.strip(),
        encoding="utf-8",
    )
    (plans_dir / "adjusted_plan_20260318_0822.json").write_text(
        """
        {
          "resolved_regime": {
            "regime": "CHOP",
            "allow_new_longs": true
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    replay = RuntimeSessionReplay(
        session_date="2026-03-18",
        project_dir=tmp_path,
        market_bars={
            "SPY": _bars([100.0, 100.1, 100.2]),
            "QQQ": _bars([200.0, 200.2, 200.4]),
            "IWM": _bars([50.0, 50.1, 50.2]),
        },
        previous_closes={"SPY": 99.5, "QQQ": 199.5, "IWM": 49.8},
    )

    report = replay.run(persist=False)
    check = report["long_decision_checks"][0]

    assert report["summary"]["metadata_backfilled_checks"] == 1
    assert check["metadata_backfilled"] is True
    assert check["metadata_inferred"] is False
    assert check["replay_authority_reason"] == ""
    assert check["replay_authority_eligible"] is True
    assert check["replay_long_allowed"] is True
    assert check["origin_entry_source"] == "overnight_full_watchlist"
    assert check["runtime_entry_context"] == "overnight_first_hour_recheck"
    assert check["override_reason"] == "overnight_first_hour_recheck"
    assert check["persisted_resolved_regime"] == "CHOP"
    assert check["persisted_allow_new_longs"] is True


def test_runtime_session_replay_reports_handoff_diagnostics_and_actual_day_context(
    tmp_path: Path,
):
    logs_dir = tmp_path / "logs"
    plans_dir = tmp_path / "plans"
    data_dir = tmp_path / "data"
    logs_dir.mkdir()
    plans_dir.mkdir()
    data_dir.mkdir()

    (logs_dir / "signals_2026-03-18.json").write_text(
        """
        {
          "signals": [
            {
              "ticker": "LONG1",
              "score": 84.0,
              "plan_score_source": "adjusted_plan_20260318_0822.json"
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )
    (logs_dir / "trade_decisions_20260318.json").write_text(
        """
        [
          {
            "timestamp": "2026-03-18T08:32:10",
            "symbol": "LONG1",
            "score": 84.0,
            "actually_executed": true
          }
        ]
        """.strip(),
        encoding="utf-8",
    )
    (plans_dir / "pm_plan_2026-03-18.json").write_text(
        """
        {
          "signals": [
            {"ticker": "PM1", "score": 71.0},
            {"ticker": "PM2", "score": 69.0}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )
    (plans_dir / "adjusted_plan_20260318_0822.json").write_text(
        """
        {
          "signals": [
            {"ticker": "LONG1", "score": 84.0}
          ],
          "resolved_regime": {
            "regime": "CHOP",
            "allow_new_longs": true
          }
        }
        """.strip(),
        encoding="utf-8",
    )
    (data_dir / "eod_review_2026-03-18.json").write_text(
        """
        {
          "date": "2026-03-18",
          "total_trades": 2,
          "avg_pnl": 5.0,
          "win_rate": 0.5,
          "score_buckets": {
            "80+": {"count": 2, "pnl_sum": 10.0}
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    replay = RuntimeSessionReplay(
        session_date="2026-03-18",
        project_dir=tmp_path,
        market_bars={
            "SPY": _bars([100.0, 100.1, 100.2]),
            "QQQ": _bars([200.0, 200.2, 200.4]),
            "IWM": _bars([50.0, 50.1, 50.2]),
        },
        previous_closes={"SPY": 99.5, "QQQ": 199.5, "IWM": 49.8},
    )

    report = replay.run(persist=False)

    assert report["actual_day_context"]["net_pnl"] == 10.0
    assert report["summary"]["actual_net_pnl"] == 10.0
    assert report["handoff_diagnostics"]["status"] == "degraded"
    assert (
        report["handoff_diagnostics"]["problem"]
        == "overnight_candidates_dropped_before_day_plan"
    )
    assert report["handoff_diagnostics"]["recoverable_candidates_count"] == 2
    assert report["handoff_diagnostics"]["recoverable_candidates_examples"] == [
        "PM1",
        "PM2",
    ]
    assert report["handoff_diagnostics"]["signal_alignment_status"] == "aligned"
    assert report["summary"]["signal_alignment_status"] == "aligned"


def test_runtime_session_replay_repairs_signal_ledger_drift_from_authoritative_plan(
    tmp_path: Path,
):
    logs_dir = tmp_path / "logs"
    plans_dir = tmp_path / "plans"
    logs_dir.mkdir()
    plans_dir.mkdir()

    (logs_dir / "signals_2026-03-18.json").write_text(
        """
        {
          "signals": [
            {"ticker": "LONG1", "score": 84.0}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )
    (logs_dir / "trade_decisions_20260318.json").write_text(
        """
        [
          {
            "timestamp": "2026-03-18T08:32:10",
            "symbol": "LONG2",
            "score": 82.0,
            "actually_executed": true
          }
        ]
        """.strip(),
        encoding="utf-8",
    )
    (plans_dir / "morning_game_plan_20260318.json").write_text(
        """
        {
          "signals": [
            {"ticker": "LONG1", "score": 84.0},
            {"ticker": "LONG2", "score": 82.0}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )
    (plans_dir / "adjusted_plan_20260318_0822.json").write_text(
        """
        {
          "signals": [
            {"ticker": "LONG1", "score": 84.0},
            {"ticker": "LONG2", "score": 82.0}
          ],
          "resolved_regime": {
            "regime": "CHOP",
            "allow_new_longs": true
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    replay = RuntimeSessionReplay(
        session_date="2026-03-18",
        project_dir=tmp_path,
        market_bars={
            "SPY": _bars([100.0, 100.1, 100.2]),
            "QQQ": _bars([200.0, 200.2, 200.4]),
            "IWM": _bars([50.0, 50.1, 50.2]),
        },
        previous_closes={"SPY": 99.5, "QQQ": 199.5, "IWM": 49.8},
    )

    report = replay.run(persist=False)

    assert report["handoff_diagnostics"]["status"] == "healthy"
    assert report["handoff_diagnostics"]["problem"] == ""
    assert report["handoff_diagnostics"]["signal_alignment_status"] == "aligned"
    assert report["handoff_diagnostics"]["signal_alignment_problem"] == ""
    assert (
        report["handoff_diagnostics"]["decision_symbols_missing_from_signals_examples"]
        == []
    )
    assert report["summary"]["signal_alignment_status"] == "aligned"
    repaired = json.loads(
        (logs_dir / "signals_2026-03-18.json").read_text(encoding="utf-8")
    )
    assert repaired["source"] == "adjusted_plan"
    assert repaired["signal_manifest"]["replay_refresh"] is True
    assert {row["ticker"] for row in repaired["signals"]} == {"LONG1", "LONG2"}
    assert not any(
        "Workflow watchlist causality snapshot persistence failed" in note
        for note in report.get("replay_notes", [])
    )
    assert report["artifacts"]["watchlist_causality_file"].endswith(
        "watchlist_causality_2026-03-18.json"
    )
    assert report["watchlist_causality_snapshot"]["date"] == "2026-03-18"
    assert report["watchlist_causality_snapshot"]["watchlist_count"] == 2


def test_handoff_diagnostics_treats_runtime_augmented_decisions_as_aligned(
    tmp_path: Path,
):
    replay = RuntimeSessionReplay(
        session_date="2026-03-18",
        project_dir=tmp_path,
        market_bars={
            "SPY": _bars([100.0, 100.1, 100.2]),
            "QQQ": _bars([200.0, 200.2, 200.4]),
            "IWM": _bars([50.0, 50.1, 50.2]),
        },
        previous_closes={"SPY": 99.5, "QQQ": 199.5, "IWM": 49.8},
    )

    diagnostics = replay._build_handoff_diagnostics(
        artifacts={
            "signals_path": None,
            "pm_plan_path": None,
            "morning_plan_path": None,
            "adjusted_plan_path": None,
        },
        raw_signals=[
            {
                "ticker": "LONG1",
                "entry_source": "overnight_plan",
                "source_bucket": "watchlist",
                "plan_score_source": "pm_plan_2026-03-18.json",
            }
        ],
        replay_signals=[
            {
                "ticker": "LONG1",
                "entry_source": "overnight_plan",
                "source_bucket": "watchlist",
                "plan_score_source": "pm_plan_2026-03-18.json",
            },
            {
                "ticker": "FAST1",
                "entry_source": "momentum_scanner",
                "source_bucket": "watchlist",
            },
            {
                "ticker": "FAST2",
                "entry_source": "wave_entry",
                "source_bucket": "watchlist",
            },
        ],
        decisions=[
            {"symbol": "FAST1"},
            {"symbol": "FAST2"},
        ],
    )

    assert diagnostics["signal_alignment_status"] == "aligned"
    assert diagnostics["signal_alignment_problem"] == ""
    assert diagnostics["decision_symbols_missing_from_signals_count"] == 0
    assert diagnostics["decision_symbols_runtime_augmented_count"] == 2
    assert set(diagnostics["decision_symbols_runtime_augmented_examples"]) == {
        "FAST1",
        "FAST2",
    }


def test_handoff_diagnostics_explains_filtered_day_plan_when_decisions_align(
    tmp_path: Path,
):
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    adjusted = plans_dir / "adjusted_plan_20260318_0822.json"
    adjusted.write_text(
        """
        {
          "signals": [
            {"ticker": "KEEP", "score": 90.0},
            {"ticker": "FILTERED", "score": 70.0}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )
    replay = RuntimeSessionReplay(session_date="2026-03-18", project_dir=tmp_path)

    diagnostics = replay._build_handoff_diagnostics(
        artifacts={
            "signals_path": tmp_path / "logs" / "signals_2026-03-18.json",
            "pm_plan_path": None,
            "morning_plan_path": None,
            "adjusted_plan_path": adjusted,
        },
        raw_signals=[{"ticker": "KEEP"}],
        replay_signals=[{"ticker": "KEEP"}],
        decisions=[{"symbol": "KEEP"}],
    )

    assert diagnostics["signal_alignment_status"] == "filtered"
    assert diagnostics["status"] == "filtered_expected"
    assert diagnostics["problem"] == ""
    assert (
        diagnostics["interpretation"]
        == "recorded_decisions_aligned_after_expected_filtering"
    )
    assert (
        "adjusted_plan_20260318_0822.json"
        in diagnostics["source_paths"]["adjusted_plan"]
    )


def test_replay_mark_to_market_positions_refreshes_current_price_and_pnl(
    tmp_path: Path,
):
    replay = RuntimeSessionReplay(
        session_date="2026-03-18",
        project_dir=tmp_path,
        market_bars={
            "ABC": _bars([10.0, 10.5, 11.0]),
        },
    )
    replay._positions = [
        SimpleNamespace(
            symbol="ABC",
            qty=10,
            avg_entry_price=10.0,
            avg_entry=10.0,
            entry_price=10.0,
            current_price=10.0,
            market_value=100.0,
            cost_basis=100.0,
            unrealized_pl=0.0,
            unrealized_pnl=0.0,
            unrealized_plpc=0.0,
            pnl_pct=0.0,
            side="long",
        )
    ]
    replay._current_replay_time = pd.Timestamp(
        "2026-03-18 08:32:00", tz="America/Chicago"
    ).to_pydatetime()

    marked = replay._mark_to_market_positions()

    assert len(marked) == 1
    assert marked[0].current_price == pytest.approx(11.0)
    assert marked[0].unrealized_pl == pytest.approx(10.0)
    assert marked[0].unrealized_plpc == pytest.approx(0.10)
    assert marked[0].pnl_pct == pytest.approx(10.0)


def test_intraday_bar_coverage_reports_zero_bar_symbols_by_source(tmp_path: Path):
    replay = RuntimeSessionReplay(
        session_date="2026-03-18",
        project_dir=tmp_path,
        market_bars={"WITHBARS": _bars([10.0, 10.1, 10.2])},
    )

    coverage = replay._build_intraday_bar_coverage(
        signals=[
            {"ticker": "WITHBARS", "source_bucket": "watchlist"},
            {"ticker": "NOBARS", "source_bucket": "overnight"},
        ],
        decisions=[],
        trade_journal=[],
    )

    assert coverage["symbols_scored"] == 2
    assert coverage["symbols_with_bars"] == 1
    assert coverage["symbols_with_zero_bars"] == 1
    assert coverage["actionable_symbols_scored"] == 2
    assert coverage["actionable_symbols_with_zero_bars"] == 1
    assert coverage["diagnostic_only_zero_bar_symbols"] == 0
    assert coverage["top_missing_symbols"][0]["symbol"] == "NOBARS"
    assert coverage["zero_bar_symbols_by_source_bucket"] == {"overnight": 1}


def test_intraday_bar_coverage_separates_diagnostic_workflow_selected_symbols(
    tmp_path: Path,
):
    replay = RuntimeSessionReplay(
        session_date="2026-03-18",
        project_dir=tmp_path,
        market_bars={"KEEP": _bars([10.0, 10.1, 10.2])},
    )

    coverage = replay._build_intraday_bar_coverage(
        signals=[{"ticker": "KEEP", "source_bucket": "signals"}],
        decisions=[],
        trade_journal=[],
        workflow_actions=[
            {"selected_symbols": ["DIAG1", "DIAG2"]},
        ],
    )

    assert coverage["symbols_with_zero_bars"] == 2
    assert coverage["actionable_symbols_with_zero_bars"] == 0
    assert coverage["diagnostic_only_zero_bar_symbols"] == 2
    assert coverage["zero_bar_symbols_by_source_bucket"] == {"workflow_selected": 2}
    assert coverage["interpretation"] == "zero_bar_symbols_are_diagnostic_only"


def test_build_signal_pipeline_audit_summarizes_stage_presence_and_blockers(
    tmp_path: Path,
):
    logs_dir = tmp_path / "logs"
    plans_dir = tmp_path / "plans"
    logs_dir.mkdir()
    plans_dir.mkdir()

    (plans_dir / "pm_plan_2026-03-18.json").write_text(
        """
        {
          "signals": [
            {
              "ticker": "LONG1",
              "entry_source": "overnight_plan",
              "source_bucket": "watchlist",
              "plan_score_source": "morning_game_plan_20260318.json"
            }
          ],
          "overnight_watchlist_bridge": {
            "used": true,
            "ranked_symbols": 1
          },
          "signal_pipeline_trace": {
            "generated_candidates": 1,
            "published_signals": 1
          }
        }
        """.strip(),
        encoding="utf-8",
    )
    (plans_dir / "morning_game_plan_20260318.json").write_text(
        """
        {
          "signals": [
            {
              "ticker": "LONG1",
              "entry_source": "overnight_plan",
              "source_bucket": "watchlist",
              "plan_score_source": "morning_game_plan_20260318.json"
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )
    (plans_dir / "adjusted_plan_20260318_0822.json").write_text(
        """
        {
          "signals": [
            {
              "ticker": "LONG1",
              "entry_source": "overnight_plan",
              "source_bucket": "watchlist",
              "plan_score_source": "morning_game_plan_20260318.json",
              "current_score": 82.5,
              "entry_threshold": 50.0,
              "score_gap_to_threshold": 32.5
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    replay = RuntimeSessionReplay(
        session_date="2026-03-18",
        project_dir=tmp_path,
    )
    artifacts = {
        "pm_plan_path": plans_dir / "pm_plan_2026-03-18.json",
        "pm_plan_payload": json.loads(
            (plans_dir / "pm_plan_2026-03-18.json").read_text(encoding="utf-8")
        ),
        "morning_plan_path": plans_dir / "morning_game_plan_20260318.json",
        "adjusted_plan_path": plans_dir / "adjusted_plan_20260318_0822.json",
    }

    audit = replay._build_signal_pipeline_audit(
        artifacts=artifacts,
        raw_signals=[{"ticker": "LONG1", "score": 82.0}],
        replay_signals=[{"ticker": "LONG1", "score": 82.0}],
        decisions=[
            {
                "symbol": "LONG1",
                "score": 82.0,
                "timestamp": "2026-03-18T08:32:10",
                "actually_executed": True,
            }
        ],
        trade_journal=[
            {
                "symbol": "LONG1",
                "_local_timestamp": "2026-03-18T08:32:15",
            }
        ],
        watchlist_causality_snapshot={
            "symbols": [
                {
                    "symbol": "LONG1",
                    "watchlist_member": True,
                    "current_status": "executed",
                    "blocking_reason": "",
                    "current_score": 82.5,
                    "entry_threshold": 50.0,
                    "score_gap_to_threshold": 32.5,
                },
                {
                    "symbol": "MISS1",
                    "watchlist_member": True,
                    "current_status": "blocked",
                    "blocking_reason": "entry_gap_too_large",
                },
            ]
        },
    )

    assert audit["summary"]["symbols_total"] == 2
    assert audit["summary"]["executed_symbols"] == 1
    assert audit["summary"]["blocked_symbols"] == 1
    assert audit["summary"]["stage_counts"]["pm_plan"] == 1
    assert audit["summary"]["status_counts"]["executed"] == 1
    assert audit["summary"]["status_counts"]["blocked"] == 1
    assert audit["summary"]["blocking_reason_counts"]["entry_gap_too_large"] == 1
    assert audit["pm_bridge"]["used"] is True
    assert audit["pm_signal_pipeline_trace"]["published_signals"] == 1

    rows = {row["symbol"]: row for row in audit["rows"]}
    assert rows["LONG1"]["in_adjusted_plan"] is True
    assert rows["LONG1"]["in_recorded_decisions"] is True
    assert rows["LONG1"]["actually_executed"] is True
    assert rows["LONG1"]["entry_source"] == "overnight_plan"
    assert rows["LONG1"]["source_bucket"] == "watchlist"
    assert rows["MISS1"]["current_status"] == "blocked"
    assert rows["MISS1"]["blocking_reason"] == "entry_gap_too_large"


def test_build_signal_pipeline_audit_backfills_nonempty_source_metadata_from_any_stage(
    tmp_path: Path,
):
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()

    (plans_dir / "morning_game_plan_20260318.json").write_text(
        """
        {
          "signals": [
            {
              "ticker": "FGI",
              "entry_source": "deep_research_bridge",
              "source_bucket": "watchlist",
              "plan_score_source": "morning_game_plan_20260318.json"
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )
    (plans_dir / "adjusted_plan_20260318_0822.json").write_text(
        """
        {
          "signals": [
            {
              "ticker": "FGI",
              "entry_source": "",
              "source_bucket": "",
              "plan_score_source": "",
              "current_score": 100.0
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    replay = RuntimeSessionReplay(
        session_date="2026-03-18",
        project_dir=tmp_path,
    )
    audit = replay._build_signal_pipeline_audit(
        artifacts={
            "morning_plan_path": plans_dir / "morning_game_plan_20260318.json",
            "adjusted_plan_path": plans_dir / "adjusted_plan_20260318_0822.json",
        },
        raw_signals=[{"ticker": "FGI"}],
        replay_signals=[{"ticker": "FGI"}],
        decisions=[],
        trade_journal=[],
        watchlist_causality_snapshot={
            "symbols": [
                {
                    "symbol": "FGI",
                    "watchlist_member": True,
                    "current_status": "blocked",
                    "blocking_reason": "entry_gap_too_large",
                }
            ]
        },
    )

    row = {row["symbol"]: row for row in audit["rows"]}["FGI"]
    assert row["entry_source"] == "deep_research_bridge"
    assert row["source_bucket"] == "watchlist"
    assert row["plan_score_source"] == "morning_game_plan_20260318.json"


def test_run_runtime_replay_benchmark_summarizes_primary_and_secondary_days(
    monkeypatch,
):
    reports = {
        "2026-03-17": {
            "summary": {
                "recorded_decisions_total": 12,
                "recorded_longs_executed": 12,
                "replay_longs_allowed": 12,
                "divergences_count": 0,
                "inverse_orders_captured": 0,
                "inverse_net_pnl": 0.0,
                "management_replay_differences": 2,
                "decision_artifact_missing": False,
                "metadata_backfilled_checks": 12,
                "metadata_inferred_checks": 10,
                "first_inverse_fast_at": "",
                "actual_net_pnl": -12.5,
                "actual_total_trades": 19,
            },
            "actual_day_context": {"eod_review_file": "eod_review_2026-03-17.json"},
            "handoff_diagnostics": {
                "status": "healthy",
                "problem": "",
                "recoverable_candidates_count": 0,
                "recoverable_candidates_examples": [],
                "signals_source": "signals_2026-03-17.json",
            },
            "divergence_summary": {"examples": [], "top_reasons": []},
            "replay_notes": [],
        },
        "2026-03-16": {
            "summary": {
                "recorded_decisions_total": 0,
                "recorded_longs_executed": 0,
                "replay_longs_allowed": 0,
                "divergences_count": 0,
                "inverse_orders_captured": 0,
                "inverse_net_pnl": 0.0,
                "management_replay_differences": 0,
                "decision_artifact_missing": True,
                "metadata_backfilled_checks": 0,
                "metadata_inferred_checks": 0,
                "first_inverse_fast_at": "",
                "actual_net_pnl": 28.09,
                "actual_total_trades": 6,
            },
            "actual_day_context": {"eod_review_file": "eod_review_2026-03-16.json"},
            "handoff_diagnostics": {
                "status": "healthy",
                "problem": "",
                "recoverable_candidates_count": 3,
                "recoverable_candidates_examples": ["ABC", "XYZ"],
                "signals_source": "plan_fallback",
                "signal_alignment_status": "missing",
                "signal_alignment_problem": "no_signals_file",
                "decision_symbols_missing_from_signals_count": 0,
                "decision_symbols_missing_from_signals_examples": [],
            },
            "divergence_summary": {
                "examples": [
                    {
                        "timestamp": "2026-03-16T08:31:00",
                        "symbol": "ABC",
                        "replay_authority_reason": "entry_authority_missing_source",
                        "replay_regime_reason": "",
                    }
                ],
                "top_reasons": ["authority:entry_authority_missing_source (1)"],
            },
            "replay_notes": ["Decision artifact missing for 2026-03-16."],
        },
    }

    def _fake_run(self, persist=True):
        return reports[self.session_date]

    monkeypatch.setattr(RuntimeSessionReplay, "run", _fake_run)

    benchmark = run_runtime_replay_benchmark(
        dates=["2026-03-17", "2026-03-16"],
        benchmark_set="test_set",
        persist=False,
    )

    assert benchmark["summary"]["dates_total"] == 2
    assert benchmark["summary"]["primary_dates"] == 1
    assert benchmark["summary"]["secondary_dates"] == 1
    assert benchmark["summary"]["high_confidence_dates"] == 1
    assert benchmark["summary"]["medium_confidence_dates"] == 1
    assert benchmark["summary"]["actual_net_pnl_total"] == 15.59
    assert benchmark["dates"][0]["gating_tier"] == "primary"
    assert benchmark["dates"][1]["gating_tier"] == "secondary"
    assert benchmark["dates"][1]["confidence"] == "medium"
    assert benchmark["dates"][1]["handoff_problem"] == ""
    assert benchmark["dates"][1]["signal_alignment_problem"] == "no_signals_file"
    assert benchmark["dates"][1]["divergence_top_reasons"] == [
        "authority:entry_authority_missing_source (1)"
    ]


def test_runtime_session_replay_divergence_summary_captures_reasons(tmp_path: Path):
    logs_dir = tmp_path / "logs"
    plans_dir = tmp_path / "plans"
    logs_dir.mkdir()
    plans_dir.mkdir()

    (logs_dir / "signals_2026-03-18.json").write_text(
        """
        {
          "signals": [
            {
              "ticker": "LONG1",
              "score": 84.0,
              "plan_score_source": "adjusted_plan_20260318_0822.json"
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )
    (logs_dir / "trade_decisions_20260318.json").write_text(
        """
        [
          {
            "timestamp": "2026-03-18T08:32:10",
            "symbol": "LONG1",
            "score": 84.0,
            "actually_executed": true
          }
        ]
        """.strip(),
        encoding="utf-8",
    )
    (plans_dir / "adjusted_plan_20260318_0822.json").write_text(
        """
        {
          "signals": [
            {"ticker": "LONG1", "score": 84.0}
          ],
          "resolved_regime": {
            "regime": "SELLOFF",
            "allow_new_longs": false
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    replay = RuntimeSessionReplay(
        session_date="2026-03-18",
        project_dir=tmp_path,
        force_modern_crash_protection=True,
        market_bars={
            "SPY": _bars([100.0, 97.0, 96.5]),
            "QQQ": _bars([200.0, 194.0, 193.5]),
            "IWM": _bars([50.0, 48.0, 47.8]),
        },
        previous_closes={"SPY": 102.0, "QQQ": 205.0, "IWM": 51.0},
    )

    report = replay.run(persist=False)

    assert report["divergence_summary"]["by_type"] == {
        "recorded_long_executed_while_replay_blocked": 1
    }
    assert report["divergence_summary"]["authority_reasons"] == {
        "inverse_fast_symbol_rr<1.40": 1
    }
    assert report["divergence_summary"]["examples"][0]["symbol"] == "LONG1"


def test_runtime_session_replay_discovery_eval_scores_extra_candidates(tmp_path: Path):
    logs_dir = tmp_path / "logs"
    plans_dir = tmp_path / "plans"
    logs_dir.mkdir()
    plans_dir.mkdir()

    (logs_dir / "signals_2026-03-19.json").write_text(
        """
        {
          "signals": [
            {"ticker": "BASE", "score": 82.0, "entry_price": 10.0},
            {
              "ticker": "DISC1",
              "score": 79.0,
              "entry_price": 10.0,
              "stop_price": 9.6,
              "target_price": 10.8,
              "risk_reward": 2.0,
              "relative_volume": 1.4
            },
            {
              "ticker": "DISC2",
              "score": 76.0,
              "entry_price": 12.0,
              "stop_price": 11.4,
              "target_price": 13.2,
              "risk_reward": 2.0,
              "relative_volume": 1.3
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )
    (logs_dir / "trade_decisions_20260319.json").write_text(
        """
        [
          {
            "timestamp": "2026-03-19T08:32:10",
            "symbol": "BASE",
            "score": 82.0,
            "actually_executed": true
          }
        ]
        """.strip(),
        encoding="utf-8",
    )
    (plans_dir / "adjusted_plan_20260319_0822.json").write_text(
        """
        {
          "signals": [
            {"ticker": "BASE", "score": 82.0, "entry_price": 10.0}
          ],
          "resolved_regime": {
            "regime": "CHOP",
            "allow_new_longs": true
          }
        }
        """.strip(),
        encoding="utf-8",
    )
    (plans_dir / "pm_plan_2026-03-19.json").write_text(
        """
        {
          "signals": [
            {
              "ticker": "DISC1",
              "score": 79.0,
              "entry_price": 10.0,
              "stop_price": 9.6,
              "target_price": 10.8,
              "risk_reward": 2.0,
              "relative_volume": 1.4
            },
            {
              "ticker": "DISC2",
              "score": 76.0,
              "entry_price": 12.0,
              "stop_price": 11.4,
              "target_price": 13.2,
              "risk_reward": 2.0,
              "relative_volume": 1.3
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    replay = RuntimeSessionReplay(
        session_date="2026-03-19",
        project_dir=tmp_path,
        strategy_eval="discovery",
        market_bars={
            "SPY": _bars_for_date(
                "2026-03-19", [100.0, 100.1, 100.2, 100.3, 100.4, 100.5]
            ),
            "QQQ": _bars_for_date(
                "2026-03-19", [200.0, 200.1, 200.2, 200.3, 200.4, 200.5]
            ),
            "IWM": _bars_for_date(
                "2026-03-19", [50.0, 50.05, 50.1, 50.15, 50.2, 50.25]
            ),
            "DISC1": _bars_for_date(
                "2026-03-19", [10.0, 10.1, 10.2, 10.5, 10.85, 10.9]
            ),
            "DISC2": _bars_for_date(
                "2026-03-19", [12.0, 11.95, 11.8, 11.6, 11.3, 11.2]
            ),
        },
        previous_closes={"SPY": 99.5, "QQQ": 199.5, "IWM": 49.8},
    )

    report = replay.run(persist=False)
    discovery = report["discovery_strategy_eval"]

    assert discovery["candidate_pool_size"] == 3
    assert discovery["synthetic_candidates_evaluated"] == 2
    assert discovery["added_winners"] == 1
    assert discovery["added_losers"] == 1
    assert discovery["top_added_candidates"][0]["symbol"] == "DISC1"
    assert report["summary"]["discovery_variant"] == DEFAULT_DISCOVERY_VARIANT


def test_run_runtime_replay_benchmark_summarizes_discovery_eval(monkeypatch):
    reports = {
        "2026-03-19": {
            "summary": {
                "recorded_decisions_total": 13,
                "recorded_longs_executed": 13,
                "replay_longs_allowed": 13,
                "divergences_count": 0,
                "inverse_orders_captured": 0,
                "inverse_net_pnl": 0.0,
                "management_replay_differences": 0,
                "decision_artifact_missing": False,
                "metadata_backfilled_checks": 13,
                "metadata_inferred_checks": 0,
                "first_inverse_fast_at": "",
                "actual_net_pnl": 445.49,
                "actual_total_trades": 50,
            },
            "actual_day_context": {"eod_review_file": "eod_review_2026-03-19.json"},
            "handoff_diagnostics": {
                "status": "healthy",
                "problem": "",
                "recoverable_candidates_count": 0,
                "recoverable_candidates_examples": [],
                "signals_source": "signals_2026-03-19.json",
            },
            "divergence_summary": {"examples": [], "top_reasons": []},
            "replay_notes": [],
            "discovery_strategy_eval": {
                "variant": DEFAULT_DISCOVERY_VARIANT,
                "candidate_pool_size": 6,
                "synthetic_candidates_evaluated": 3,
                "added_winners": 2,
                "added_losers": 1,
                "net_synthetic_pnl": 42.5,
                "avg_return_pct": 1.9,
                "slot_fill_delta": 3,
                "top_added_candidates": [{"symbol": "DISC1", "synthetic_pnl": 30.0}],
            },
        }
    }

    def _fake_run(self, persist=True):
        return reports[self.session_date]

    monkeypatch.setattr(RuntimeSessionReplay, "run", _fake_run)

    benchmark = run_runtime_replay_benchmark(
        dates=["2026-03-19"],
        benchmark_set="strict_core_plus_mar19",
        strategy_eval="discovery",
        persist=False,
    )

    assert benchmark["summary"]["discovery_strategy_eval_enabled"] is True
    assert benchmark["summary"]["discovery_candidates_evaluated_total"] == 3
    assert benchmark["summary"]["discovery_net_synthetic_pnl_total"] == 42.5
    assert (
        benchmark["dates"][0]["discovery_top_added_candidates"][0]["symbol"] == "DISC1"
    )


def test_parse_args_accepts_benchmark_set():
    args = parse_args(["--benchmark-set", "strict_core"])

    assert args.benchmark_set == "strict_core"
    assert args.date is None
    assert args.force_modern_crash_protection is False


def test_parse_args_accepts_discovery_strategy_eval():
    args = parse_args(
        ["--benchmark-set", "strict_core_plus_mar19", "--strategy-eval", "discovery"]
    )

    assert args.benchmark_set == "strict_core_plus_mar19"
    assert args.strategy_eval == "discovery"
    assert args.variant == DEFAULT_DISCOVERY_VARIANT


def test_parse_args_accepts_force_modern_crash_protection():
    args = parse_args(["--date", "2026-03-19", "--force-modern-crash-protection"])

    assert args.date == "2026-03-19"
    assert args.force_modern_crash_protection is True


def test_replay_entry_gate_fails_silent_zero_entry_cycle():
    report = {
        "summary": {
            "date": "2026-05-01",
            "divergences_count": 0,
            "replay_orders_by_bucket": {"equity_entry": 0, "inverse_etf": 0},
        },
        "daymanager_cycle_stats": [
            {
                "timestamp": "2026-05-01T10:45:00-05:00",
                "candidate_count": 2,
                "open_slots": 1,
                "orders_submitted": 0,
                "blocked_by_reason": {},
                "block_reason": "",
            }
        ],
    }

    gate = build_replay_entry_diagnostic_gate(report)

    assert gate["status"] == "fail"
    assert gate["incident_acceptance"] == "fail_reproduced_production_failure"
    assert gate["zero_divergence_zero_entry_reproduces_failure"] is True
    assert gate["silent_no_submit_cycles"] == 1


def test_replay_entry_gate_fails_incident_explained_zero_entry_cycle():
    report = {
        "summary": {
            "date": "2026-05-01",
            "divergences_count": 0,
            "replay_orders_by_bucket": {"equity_entry": 0, "inverse_etf": 0},
        },
        "daymanager_cycle_stats": [
            {
                "timestamp": "2026-05-01T10:45:00-05:00",
                "candidate_count": 2,
                "open_slots": 1,
                "orders_submitted": 0,
                "blocked_by_reason": {"pending_buy_order": 2},
                "block_reason": "",
            }
        ],
    }

    gate = build_replay_entry_diagnostic_gate(report)

    assert gate["status"] == "fail"
    assert gate["passed"] is False
    assert gate["reason"] == "reproduced_production_failure_no_entry_explained"
    assert gate["diagnostic_status"] == "no_entry_explained"
    assert gate["diagnostic_passed"] is True
    assert gate["incident_acceptance"] == "fail_reproduced_production_failure"
    assert gate["explained_no_submit_cycles"] == 1


def test_replay_entry_gate_passes_with_watchlist_entry_order():
    report = {
        "summary": {
            "date": "2026-05-01",
            "divergences_count": 0,
            "replay_orders_by_bucket": {"equity_entry": 1, "inverse_etf": 0},
        },
        "daymanager_cycle_stats": [
            {
                "timestamp": "2026-05-01T10:45:00-05:00",
                "candidate_count": 2,
                "open_slots": 1,
                "orders_submitted": 1,
            }
        ],
    }

    gate = build_replay_entry_diagnostic_gate(report)

    assert gate["status"] == "pass"
    assert gate["passed"] is True
    assert gate["incident_acceptance"] == "pass_entry_replayed"


def test_inverse_absence_diagnostic_explains_recovered_selloff():
    diagnostic = _build_inverse_absence_diagnostic(
        [
            {
                "timestamp": "2026-05-05T08:30:00-05:00",
                "state": "recovery_transition",
                "snapshot": {
                    "recovery_confirmed": True,
                    "red_ratio": 0.0,
                    "avg_pct_change": 0.7,
                },
            },
            {
                "timestamp": "2026-05-05T08:31:00-05:00",
                "state": "open",
                "snapshot": {
                    "recovery_confirmed": True,
                    "red_ratio": 0.0,
                    "avg_pct_change": 0.8,
                },
            },
        ],
        {"regime": "SELLOFF"},
    )

    assert diagnostic["reason"] == "recovered_benchmarks_no_inverse_fast"
    assert diagnostic["persisted_regime"] == "SELLOFF"
    assert diagnostic["recovered_snapshots"] == 2


def test_actual_day_context_prefers_broker_day_pnl_over_unrealized_only(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "eod_review_2026-05-01.json").write_text(
        json.dumps(
            {
                "date": "2026-05-01",
                "total_trades": 3,
                "avg_pnl": 13.63,
                "broker_day_pnl": -197.45,
                "realized_day_pnl": -238.35,
                "open_position_unrealized_pnl": 40.90,
            }
        ),
        encoding="utf-8",
    )

    context = _load_actual_day_context(tmp_path, "2026-05-01")

    assert context["net_pnl"] == -197.45
    assert context["realized_day_pnl"] == -238.35
    assert context["open_position_unrealized_pnl"] == 40.90


def test_parse_args_accepts_replay_profile():
    args = parse_args(["--date", "2026-03-19", "--profile", "fast"])

    assert args.date == "2026-03-19"
    assert args.profile == "fast"


def test_main_defaults_daily_replay_to_fast_daymanager_core(monkeypatch, capsys):
    captured = {}

    class _FakeReplay:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.output_path = Path("logs/runtime_replay_20260319.json")

        def run(self, persist=True):
            return {
                "summary": {
                    "signals_total": 0,
                    "recorded_decisions_total": 0,
                    "recorded_longs_executed": 0,
                    "inverse_orders_captured": 0,
                    "inverse_trades_closed": 0,
                    "inverse_net_pnl": 0.0,
                    "inverse_avg_return_pct": 0.0,
                    "inverse_win_rate": 0.0,
                    "divergences_count": 0,
                    "market_data_coverage_mode": "unit_test",
                    "archive_found": False,
                    "archive_symbols_loaded": 0,
                    "archive_symbols_missing": 0,
                    "archive_live_fetch_symbols": 0,
                    "actual_net_pnl": 0.0,
                    "handoff_status": "healthy",
                    "handoff_problem": "",
                    "signal_alignment_status": "aligned",
                    "signal_alignment_problem": "",
                    "replay_profile": "fast",
                    "position_advisor_mode": "skipped",
                }
            }

    monkeypatch.setattr(
        "autotrade.replay.runtime_session_replay.RuntimeSessionReplay",
        _FakeReplay,
    )

    rc = main(["--date", "2026-03-19"])

    assert rc == 0
    assert captured["mode"] == REPLAY_MODE_DAYMANAGER_CORE
    assert captured["profile"] == "fast"
    assert "profile=fast" in capsys.readouterr().out


def test_main_allows_explicit_agent_workflow_in_fast_profile(monkeypatch):
    captured = {}

    class _FakeReplay:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.output_path = Path("logs/runtime_replay_20260319.json")

        def run(self, persist=True):
            return {
                "summary": {
                    "signals_total": 0,
                    "recorded_decisions_total": 0,
                    "recorded_longs_executed": 0,
                    "workflow_generated_decisions_total": 0,
                    "workflow_selected_symbols_total": 0,
                    "workflow_add_selected_total": 0,
                    "workflow_bullish_actions_total": 0,
                    "workflow_bullish_mark_to_close_pnl": 0.0,
                    "deployment_floor_final_pct": 0.0,
                    "deployment_floor_final_shortfall_value": 0.0,
                    "workflow_blocked_symbols_total": 0,
                    "market_open_cycles": 0,
                    "market_hours_cycles": 0,
                    "inverse_orders_captured": 0,
                    "inverse_trades_closed": 0,
                    "inverse_net_pnl": 0.0,
                    "inverse_avg_return_pct": 0.0,
                    "inverse_win_rate": 0.0,
                    "divergences_count": 0,
                    "market_data_coverage_mode": "unit_test",
                    "archive_found": False,
                    "archive_symbols_loaded": 0,
                    "archive_symbols_missing": 0,
                    "archive_live_fetch_symbols": 0,
                    "actual_net_pnl": 0.0,
                    "handoff_status": "healthy",
                    "handoff_problem": "",
                    "signal_alignment_status": "aligned",
                    "signal_alignment_problem": "",
                    "replay_profile": "fast",
                    "position_advisor_mode": "skipped",
                }
            }

    monkeypatch.setattr(
        "autotrade.replay.runtime_session_replay.RuntimeSessionReplay",
        _FakeReplay,
    )

    rc = main(["--date", "2026-03-19", "--profile", "fast", "--mode", "agent_workflow"])

    assert rc == 0
    assert captured["mode"] == REPLAY_MODE_AGENT_WORKFLOW
    assert captured["profile"] == "fast"


def test_runtime_session_replay_sets_counterfactual_modern_crash_override(
    tmp_path: Path,
):
    plans_dir = tmp_path / "plans"
    logs_dir = tmp_path / "logs"
    plans_dir.mkdir()
    logs_dir.mkdir()
    (plans_dir / "morning_game_plan_2026-03-19.json").write_text(
        '{"signals": [], "resolved_regime": {"regime": "CHOP", "allow_new_longs": true}}',
        encoding="utf-8",
    )
    (logs_dir / "trade_decisions_2026-03-19.json").write_text("[]", encoding="utf-8")
    (logs_dir / "workflow_journal_2026-03-19.jsonl").write_text("", encoding="utf-8")

    replay = RuntimeSessionReplay(
        session_date="2026-03-19",
        project_dir=tmp_path,
        force_modern_crash_protection=True,
        market_bars={
            "SPY": _bars_for_date("2026-03-19", [100.0, 99.7, 99.5, 99.4]),
            "QQQ": _bars_for_date("2026-03-19", [200.0, 199.1, 198.9, 198.8]),
            "IWM": _bars_for_date("2026-03-19", [50.0, 49.8, 49.7, 49.6]),
        },
        previous_closes={"SPY": 101.0, "QQQ": 202.0, "IWM": 50.5},
    )

    dm = replay._build_replay_day_manager(
        signals=[],
        resolved_regime={"regime": "CHOP", "allow_new_longs": True},
    )

    assert dm._entry_authority_force_modern_session is True


def test_runtime_session_replay_fast_profile_skips_position_advisor(tmp_path: Path):
    replay = RuntimeSessionReplay(
        session_date="2026-03-19",
        project_dir=tmp_path,
        profile="fast",
    )

    dm = replay._build_replay_day_manager(
        signals=[],
        resolved_regime={"regime": "CHOP", "allow_new_longs": True},
    )

    assert replay.skip_position_advisor is True
    assert dm.position_advisor is None
    assert dm.use_agentic is False
    assert dm._should_run_advisor("ABC", 0.0, 10.0, 5.0) is False


def test_runtime_session_replay_fast_profile_runs_sparse_daymanager_cycles(
    tmp_path: Path,
):
    replay = RuntimeSessionReplay(
        session_date="2026-03-19",
        project_dir=tmp_path,
        profile="fast",
        mode=REPLAY_MODE_DAYMANAGER_CORE,
    )

    assert (
        replay._should_run_replay_daymanager_cycle(
            pd.Timestamp("2026-03-19 08:35", tz="America/Chicago").to_pydatetime()
        )
        is True
    )
    assert (
        replay._should_run_replay_daymanager_cycle(
            pd.Timestamp("2026-03-19 08:36", tz="America/Chicago").to_pydatetime()
        )
        is False
    )
    assert (
        replay._should_run_replay_daymanager_cycle(
            pd.Timestamp("2026-03-19 09:00", tz="America/Chicago").to_pydatetime()
        )
        is True
    )
    assert (
        replay._should_run_replay_daymanager_cycle(
            pd.Timestamp("2026-03-19 11:00", tz="America/Chicago").to_pydatetime()
        )
        is True
    )
    assert (
        replay._should_run_replay_daymanager_cycle(
            pd.Timestamp("2026-03-19 11:05", tz="America/Chicago").to_pydatetime()
        )
        is False
    )


def test_runtime_session_replay_cycle_runner_forces_capture_submission_path(
    tmp_path: Path, monkeypatch
):
    replay = RuntimeSessionReplay(
        session_date="2026-03-19",
        project_dir=tmp_path,
        profile="fast",
        mode=REPLAY_MODE_DAYMANAGER_CORE,
    )
    dm = replay._build_replay_day_manager(
        signals=[],
        resolved_regime={"regime": "CHOP", "allow_new_longs": True},
    )
    adapter = dm._submit_order_via_execution_adapter
    assert getattr(adapter, "__self__", None) is replay
    assert (
        getattr(adapter, "__func__", None)
        is RuntimeSessionReplay._capture_inverse_order
    )
    assert dm.client.get_orders() == []
    assert dm.client.cancel_order_by_id("replay-order") is True

    dm.dry_run = True
    seen = {}

    def _fake_run_cycle(self):
        seen["dry_run_during_call"] = self.dry_run
        return {"candidate_count": 3, "entries": 0, "open_slots": 5}

    monkeypatch.setattr(DayManager, "run_cycle", _fake_run_cycle)
    result = replay._run_replay_daymanager_cycle(
        dm=dm,
        replay_minute=pd.Timestamp(
            "2026-03-19 08:35", tz="America/Chicago"
        ).to_pydatetime(),
    )

    assert seen["dry_run_during_call"] is False
    assert dm.dry_run is True
    assert result["candidate_count"] == 3


def test_run_runtime_replay_benchmark_records_counterfactual_mode(monkeypatch):
    reports = {
        "2026-03-19": {
            "summary": {
                "recorded_decisions_total": 13,
                "recorded_longs_executed": 13,
                "replay_longs_allowed": 9,
                "divergences_count": 4,
                "inverse_orders_captured": 1,
                "inverse_net_pnl": 12.5,
                "management_replay_differences": 0,
                "decision_artifact_missing": False,
                "metadata_backfilled_checks": 13,
                "metadata_inferred_checks": 0,
                "first_inverse_fast_at": "2026-03-19T08:31:00-05:00",
                "actual_net_pnl": 445.49,
                "actual_total_trades": 50,
                "force_modern_crash_protection": True,
            },
            "actual_day_context": {"eod_review_file": "eod_review_2026-03-19.json"},
            "handoff_diagnostics": {
                "status": "healthy",
                "problem": "",
                "recoverable_candidates_count": 0,
                "recoverable_candidates_examples": [],
                "signals_source": "signals_2026-03-19.json",
            },
            "divergence_summary": {"examples": [], "top_reasons": []},
            "replay_notes": [],
            "discovery_strategy_eval": {},
        }
    }

    def _fake_run(self, persist=True):
        return reports[self.session_date]

    monkeypatch.setattr(RuntimeSessionReplay, "run", _fake_run)

    benchmark = run_runtime_replay_benchmark(
        dates=["2026-03-19"],
        benchmark_set="strict_core_plus_mar19",
        force_modern_crash_protection=True,
        persist=False,
    )

    assert benchmark["summary"]["force_modern_crash_protection"] is True
    assert benchmark["dates"][0]["force_modern_crash_protection"] is True


def test_run_runtime_replay_benchmark_writes_completion_status(tmp_path, monkeypatch):
    reports = {
        "2026-03-19": {
            "summary": {
                "recorded_decisions_total": 13,
                "recorded_longs_executed": 13,
                "replay_longs_allowed": 9,
                "divergences_count": 4,
                "inverse_orders_captured": 1,
                "inverse_net_pnl": 12.5,
                "management_replay_differences": 0,
                "decision_artifact_missing": False,
                "metadata_backfilled_checks": 13,
                "metadata_inferred_checks": 0,
                "first_inverse_fast_at": "2026-03-19T08:31:00-05:00",
                "actual_net_pnl": 445.49,
                "actual_total_trades": 50,
                "force_modern_crash_protection": False,
            },
            "actual_day_context": {"eod_review_file": "eod_review_2026-03-19.json"},
            "handoff_diagnostics": {
                "status": "healthy",
                "problem": "",
                "recoverable_candidates_count": 0,
                "recoverable_candidates_examples": [],
                "signals_source": "signals_2026-03-19.json",
            },
            "divergence_summary": {"examples": [], "top_reasons": []},
            "replay_notes": [],
            "discovery_strategy_eval": {},
        }
    }

    def _fake_run(self, persist=True):
        return reports[self.session_date]

    monkeypatch.setattr(RuntimeSessionReplay, "run", _fake_run)

    output_path = tmp_path / "runtime_replay_benchmark_strict_core.json"
    status_path = tmp_path / "runtime_replay_benchmark_strict_core.status.json"
    benchmark = run_runtime_replay_benchmark(
        dates=["2026-03-19"],
        benchmark_set="strict_core",
        output_path=output_path,
        completion_status_path=status_path,
        persist=True,
    )

    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    output_payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert benchmark["status"] == "completed"
    assert benchmark["completion_status_path"] == str(status_path)
    assert benchmark["completed_at"]
    assert status_payload["status"] == "completed"
    assert status_payload["completion_status_path"] == str(status_path)
    assert status_payload["output_path"] == str(output_path)
    assert output_payload["status"] == "completed"
    assert output_payload["completion_status_path"] == str(status_path)


def test_wait_for_runtime_replay_benchmark_completion_returns_after_completed(
    tmp_path,
):
    status_path = tmp_path / "benchmark.status.json"
    status_path.write_text(
        json.dumps(
            {
                "status": "running",
                "started_at": "2026-03-19T08:30:00",
                "completed_dates": 0,
            }
        ),
        encoding="utf-8",
    )

    def _complete():
        wall_time.sleep(0.1)
        status_path.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "started_at": "2026-03-19T08:30:00",
                    "completed_at": "2026-03-19T09:05:00",
                    "output_path": "logs/runtime_replay_benchmark_strict_core.json",
                    "completion_status_path": str(status_path),
                }
            ),
            encoding="utf-8",
        )

    thread = threading.Thread(target=_complete, daemon=True)
    thread.start()

    result = wait_for_runtime_replay_benchmark_completion(
        status_path,
        timeout_seconds=5.0,
        poll_interval_seconds=0.05,
    )
    thread.join(timeout=1.0)

    assert result["status"] == "completed"
    assert result["completion_status_path"] == str(status_path)


def test_run_runtime_replay_benchmark_writes_failed_status_on_error(
    tmp_path, monkeypatch
):
    def _fake_run(self, persist=True):
        raise RuntimeError("boom")

    monkeypatch.setattr(RuntimeSessionReplay, "run", _fake_run)

    output_path = tmp_path / "runtime_replay_benchmark_strict_core.json"
    status_path = tmp_path / "runtime_replay_benchmark_strict_core.status.json"

    with pytest.raises(RuntimeError, match="boom"):
        run_runtime_replay_benchmark(
            dates=["2026-03-19"],
            benchmark_set="strict_core",
            output_path=output_path,
            completion_status_path=status_path,
            persist=True,
        )

    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert status_payload["status"] == "failed"
    assert status_payload["completion_status_path"] == str(status_path)
    assert status_payload["error"] == "boom"
