import json
from datetime import datetime as real_datetime
from pathlib import Path
from types import SimpleNamespace

from autotrade.core import autonomous_agent as autonomous_agent_mod
from autotrade.core.autonomous_agent import AutonomousAgent


class _LoggerStub:
    def info(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


def _build_agent() -> AutonomousAgent:
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent.state = None
    agent.market_cycle_count = 0
    agent._data_gateway = None
    agent._coerce_float = lambda value, default=0.0: float(value or default)
    agent.plan_generator = SimpleNamespace(
        get_account_info=lambda: {"equity": 100000.0, "buying_power": 80000.0},
        get_current_positions=lambda: [],
        _load_latest_plan=lambda: {
            "entry_orders": [
                {"symbol": "AAA", "entry_price": 10.0, "score": 82.0},
                {"symbol": "BBB", "entry_price": 12.0, "score": 76.0},
            ],
            "full_watchlist": [
                {"symbol": "AAA", "score": 82.0},
                {"symbol": "BBB", "score": 76.0},
                {"symbol": "CCC", "score": 71.0},
            ],
        },
        _coerce_float=lambda value, default=0.0: float(value or default),
        _get_current_price_quick=lambda symbol: {"AAA": 10.1, "BBB": 12.2}.get(
            symbol, 0.0
        ),
        detect_odd_lots=lambda min_position_size=5: [],
        detect_losers_to_cut=lambda max_loss_pct=-5.0: [],
    )
    agent.entry_quality_cfg = SimpleNamespace(
        early_runner_enabled=True,
        early_runner_max_positions=10,
        early_runner_post_open_rescan_enabled=True,
        overnight_recheck_after_first_hour_enabled=True,
    )
    agent.config = SimpleNamespace(
        risk_gate=SimpleNamespace(
            gap_hard_cap_by_regime={
                "LEAN_BULLISH": 15.0,
                "NEUTRAL": 10.0,
                "RISK_OFF": 5.0,
            }
        )
    )
    agent.strategy_failsafe = None
    agent.resolved_regime_output = {}
    agent.regime_strategy_overrides = {}
    agent._position_slot_class_by_symbol = {}
    agent._refresh_strategy_failsafe = lambda source=None: SimpleNamespace(
        halt_new_entries=False, level="normal"
    )
    agent._ensure_ollama_for_phase = lambda phase: None
    agent._monitor_ollama_health = lambda *args, **kwargs: None
    agent._record_ollama_result = lambda *args, **kwargs: None
    agent._update_workflow_state_flag = lambda *args, **kwargs: None
    agent.monitor_workflow_health = lambda *args, **kwargs: None
    agent._detect_and_fix_runtime_errors = lambda *args, **kwargs: None
    agent._execute_entry_waves = lambda execute=True: None
    return agent


def test_market_open_cycle_runs_early_runner_lane(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", tmp_path)
    agent = _build_agent()
    agent._build_early_runner_rows = lambda: [
        {"ticker": "AAA", "score": 82.0},
        {"ticker": "BBB", "score": 76.0},
    ]
    agent._remaining_early_runner_capacity = lambda exec_state, **kwargs: 3

    calls = []

    def fake_run_day_manager_cycle(**kwargs):
        calls.append(kwargs)
        return {"entries": 1, "exits": 0, "selected_entry_symbols": ["AAA"]}

    agent.run_day_manager_cycle = fake_run_day_manager_cycle

    wait_seconds = agent._run_market_open_cycle(cycle_count=1, execute=True)

    assert wait_seconds == 30
    assert len(calls) == 1
    assert calls[0]["dry_run"] is False
    assert calls[0]["override_reason"] == "early_runner_market_open"
    assert calls[0]["reset_signals"] is True
    assert calls[0]["max_new_entries_override"] == 3
    assert [row["ticker"] for row in calls[0]["candidate_universe_rows"]] == [
        "AAA",
        "BBB",
    ]

    state_path = tmp_path / f".execution_state_{real_datetime.now().strftime('%Y%m%d')}.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["early_runner_done"] is True
    assert state["early_runner_symbols"] == ["AAA"]
    assert state["early_runner_diagnostics"]["candidate_count"] == 2
    assert state["early_runner_diagnostics"]["selected_count"] == 1
    assert [row["symbol"] for row in state["full_watchlist_rows"]] == [
        "AAA",
        "BBB",
        "CCC",
    ]


def test_market_open_cycle_sets_early_runner_observe_only_after_empty_streak(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", tmp_path)
    previous_state = tmp_path / ".execution_state_20260316.json"
    previous_state.write_text(
        json.dumps(
            {
                "early_runner_diagnostics": {
                    "candidate_count": 0,
                    "selected_count": 0,
                }
            }
        ),
        encoding="utf-8",
    )

    agent = _build_agent()
    agent.entry_quality_cfg.early_runner_observe_only_streak = 2
    agent._build_early_runner_rows = lambda: []
    calls = []
    agent.run_day_manager_cycle = lambda **kwargs: calls.append(kwargs) or {
        "entries": 0,
        "exits": 0,
        "selected_entry_symbols": [],
    }

    agent._run_market_open_cycle(cycle_count=1, execute=True)

    state_path = tmp_path / f".execution_state_{real_datetime.now().strftime('%Y%m%d')}.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert calls == [{"dry_run": False, "market_phase": "market_open"}]
    assert state["early_runner_diagnostics"]["observe_only"] is True
    assert state["early_runner_diagnostics"]["blocked_reason"].startswith(
        "observe_only_after_empty_streak"
    )


def test_upsert_signals_into_dm_preserves_actionable_entry_fields_for_full_watchlist_injection():
    agent = _build_agent()
    dm = SimpleNamespace(
        signals=[
            {
                "symbol": "LUNR",
                "ticker": "LUNR",
                "entry_price": 28.9773,
                "entry": 28.9773,
                "recommendation": "BUY",
                "score": 67.75,
                "final_score": 67.75,
                "confidence": 70.0,
                "plan_score_source": "morning_game_plan_20260422.json",
                "entry_source": "overnight_plan",
                "source_bucket": "watchlist",
            }
        ],
        signal_status={},
        _mark_watchlist_symbol=lambda symbol: None,
    )

    added, touched = agent._upsert_signals_into_dm(
        dm,
        [
            {
                "symbol": "LUNR",
                "ticker": "LUNR",
                "entry_price": 22.920000076293945,
                "recommendation": "BUY",
                "final_score": 67.75,
                "confidence": 70.0,
                "ranking_score": 100.0,
                "catalyst": "metadata_only",
            }
        ],
        default_entry_source="overnight_full_watchlist",
        default_reason="full_watchlist_injected",
        regime="NEUTRAL",
    )

    assert added == 0
    assert touched == ["LUNR"]
    row = dm.signals[0]
    assert row["entry_price"] == 28.9773
    assert row["entry"] == 28.9773
    assert row["entry_source"] == "overnight_plan"
    assert row["plan_score_source"] == "morning_game_plan_20260422.json"
    assert row["catalyst"] == "metadata_only"


def test_upsert_signals_into_dm_preserves_full_watchlist_source_during_recheck():
    agent = _build_agent()
    dm = SimpleNamespace(
        signals=[],
        signal_status={},
        _mark_watchlist_symbol=lambda symbol: None,
    )

    added, touched = agent._upsert_signals_into_dm(
        dm,
        [
            {
                "symbol": "VG",
                "ticker": "VG",
                "score": 100.0,
                "entry_source": "overnight_full_watchlist",
                "plan_score_source": "pm_plan_2026-05-04.json",
                "source_bucket": "watchlist",
            }
        ],
        default_entry_source="overnight_recheck",
        default_reason="candidate_override",
        regime="NEUTRAL",
    )

    assert added == 1
    assert touched == ["VG"]
    row = dm.signals[0]
    assert row["entry_source"] == "overnight_full_watchlist"
    assert row["runtime_override_entry_source"] == "overnight_recheck"
    assert row["runtime_entry_context"] == "candidate_override"


def test_market_hours_cycle_runs_post_open_overnight_rescan(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", tmp_path)
    agent = _build_agent()
    agent._remaining_early_runner_capacity = lambda exec_state, **kwargs: 2

    state_path = tmp_path / f".execution_state_{real_datetime.now().strftime('%Y%m%d')}.json"
    state_path.write_text(
        json.dumps(
            {
                "early_runner_done": True,
                "post_open_rescan_done": False,
                "first_hour_recheck_done": False,
                "early_runner_symbols": ["AAA"],
                "full_watchlist_rows": [
                    {"ticker": "BBB", "score": 76.0},
                    {"ticker": "CCC", "score": 71.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    calls = []

    def fake_run_day_manager_cycle(**kwargs):
        calls.append(kwargs)
        return {"entries": 1, "exits": 0, "selected_entry_symbols": ["BBB"]}

    agent.run_day_manager_cycle = fake_run_day_manager_cycle

    wait_seconds = agent._run_market_hours_cycle(cycle_count=1, execute=True)

    assert wait_seconds == 60
    assert len(calls) == 1
    assert calls[0]["dry_run"] is False
    assert calls[0]["override_reason"] == "overnight_post_open_rescan"
    assert calls[0]["reset_signals"] is True
    assert calls[0]["max_new_entries_override"] == 2
    assert [row["ticker"] for row in calls[0]["candidate_universe_rows"]] == [
        "AAA",
        "BBB",
        "CCC",
    ]

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["post_open_rescan_done"] is True
    assert state["early_runner_symbols"] == ["AAA", "BBB"]


def test_market_hours_cycle_refreshes_stale_watchlist_entry_prices(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", tmp_path)
    agent = _build_agent()
    agent.plan_generator = SimpleNamespace(
        get_account_info=lambda: {"equity": 100000.0, "buying_power": 80000.0},
        get_current_positions=lambda: [],
        _load_latest_plan=lambda: {
            "signals": [
                {
                    "symbol": "LUNR",
                    "ticker": "LUNR",
                    "entry_price": 28.9773,
                    "final_score": 67.75,
                    "confidence": 70.0,
                    "recommendation": "BUY",
                    "plan_score_source": "morning_game_plan_20260422.json",
                }
            ],
            "full_watchlist": [
                {
                    "symbol": "LUNR",
                    "ticker": "LUNR",
                    "entry_price": 22.920000076293945,
                    "final_score": 67.75,
                    "confidence": 70.0,
                    "recommendation": "BUY",
                }
            ],
        },
        _coerce_float=lambda value, default=0.0: float(value or default),
        _get_current_price_quick=lambda symbol: 29.5 if symbol == "LUNR" else 0.0,
        detect_odd_lots=lambda min_position_size=5: [],
        detect_losers_to_cut=lambda max_loss_pct=-5.0: [],
    )
    agent._remaining_early_runner_capacity = lambda exec_state, **kwargs: 1

    state_path = tmp_path / f".execution_state_{real_datetime.now().strftime('%Y%m%d')}.json"
    state_path.write_text(
        json.dumps(
            {
                "early_runner_done": True,
                "post_open_rescan_done": False,
                "first_hour_recheck_done": False,
                "early_runner_symbols": [],
                "full_watchlist_rows": [
                    {
                        "symbol": "LUNR",
                        "ticker": "LUNR",
                        "entry_price": 22.920000076293945,
                        "final_score": 67.75,
                        "confidence": 70.0,
                        "recommendation": "BUY",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    calls = []

    def fake_run_day_manager_cycle(**kwargs):
        calls.append(kwargs)
        return {"entries": 0, "exits": 0, "selected_entry_symbols": []}

    agent.run_day_manager_cycle = fake_run_day_manager_cycle

    wait_seconds = agent._run_market_hours_cycle(cycle_count=1, execute=True)

    assert wait_seconds == 60
    assert len(calls) == 1
    candidate = next(
        row
        for row in calls[0]["candidate_universe_rows"]
        if str(row.get("ticker") or row.get("symbol")) == "LUNR"
    )
    assert candidate["entry_price"] == 28.9773
    assert candidate["plan_score_source"] == "morning_game_plan_20260422.json"

    state = json.loads(state_path.read_text(encoding="utf-8"))
    stored = next(
        row
        for row in state["full_watchlist_rows"]
        if str(row.get("ticker") or row.get("symbol")) == "LUNR"
    )
    assert stored["entry_price"] == 28.9773
    assert stored["plan_score_source"] == "morning_game_plan_20260422.json"


def test_market_hours_cycle_runs_first_hour_overnight_recheck(
    monkeypatch, tmp_path: Path
):
    class _AfterFirstHourDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 3, 17, 9, 35, 0, tzinfo=tz)

    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", tmp_path)
    monkeypatch.setattr(autonomous_agent_mod, "datetime", _AfterFirstHourDateTime)
    agent = _build_agent()
    agent._remaining_early_runner_capacity = lambda exec_state, **kwargs: 0

    state_path = tmp_path / ".execution_state_20260317.json"
    state_path.write_text(
        json.dumps(
            {
                "early_runner_done": True,
                "post_open_rescan_done": True,
                "first_hour_recheck_done": False,
                "early_runner_symbols": ["AAA"],
                "full_watchlist_rows": [
                    {"ticker": "DDD", "score": 73.0},
                    {"ticker": "EEE", "score": 69.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    calls = []

    def fake_run_day_manager_cycle(**kwargs):
        calls.append(kwargs)
        return {"entries": 0, "exits": 0, "selected_entry_symbols": []}

    agent.run_day_manager_cycle = fake_run_day_manager_cycle

    wait_seconds = agent._run_market_hours_cycle(cycle_count=2, execute=True)

    assert wait_seconds == 60
    assert len(calls) == 1
    assert calls[0]["dry_run"] is False
    assert calls[0]["override_reason"] == "overnight_first_hour_recheck"
    assert calls[0]["reset_signals"] is True
    assert "max_new_entries_override" not in calls[0]
    assert [row["ticker"] for row in calls[0]["candidate_universe_rows"]] == [
        "AAA",
        "BBB",
        "DDD",
        "EEE",
    ]

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["first_hour_recheck_done"] is True


def test_market_hours_cycle_closes_late_overnight_recheck_window(
    monkeypatch, tmp_path: Path
):
    class _LateMorningDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 3, 17, 11, 45, 0, tzinfo=tz)

    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", tmp_path)
    monkeypatch.setattr(autonomous_agent_mod, "datetime", _LateMorningDateTime)
    agent = _build_agent()

    state_path = tmp_path / ".execution_state_20260317.json"
    state_path.write_text(
        json.dumps(
            {
                "early_runner_done": True,
                "post_open_rescan_done": True,
                "first_hour_recheck_done": False,
                "full_watchlist_rows": [
                    {"ticker": "DDD", "score": 73.0},
                    {"ticker": "EEE", "score": 69.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    calls = []
    agent.run_day_manager_cycle = lambda **kwargs: calls.append(kwargs) or {
        "entries": 0,
        "exits": 0,
        "selected_entry_symbols": [],
    }

    wait_seconds = agent._run_market_hours_cycle(cycle_count=2, execute=True)

    assert wait_seconds == 60
    assert calls == [{"dry_run": False, "market_phase": "market_hours"}]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["first_hour_recheck_done"] is True
    assert state["overnight_recheck_status"]["reason"] == "overnight_recheck_window_closed"


def test_market_hours_cycle_restores_broader_watchlist_when_state_subset_is_narrow(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", tmp_path)
    agent = _build_agent()
    agent._remaining_early_runner_capacity = lambda exec_state, **kwargs: 2
    agent._get_full_watchlist_rows = lambda: [
        {"ticker": "BBB", "score": 76.0},
        {"ticker": "CCC", "score": 71.0},
        {"ticker": "DDD", "score": 68.0},
    ]

    state_path = tmp_path / f".execution_state_{real_datetime.now().strftime('%Y%m%d')}.json"
    state_path.write_text(
        json.dumps(
            {
                "early_runner_done": True,
                "post_open_rescan_done": False,
                "first_hour_recheck_done": False,
                "full_watchlist_rows": [
                    {"ticker": "BBB", "score": 76.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    calls = []

    def fake_run_day_manager_cycle(**kwargs):
        calls.append(kwargs)
        return {"entries": 0, "exits": 0, "selected_entry_symbols": []}

    agent.run_day_manager_cycle = fake_run_day_manager_cycle

    wait_seconds = agent._run_market_hours_cycle(cycle_count=1, execute=True)

    assert wait_seconds == 60
    assert [row["ticker"] for row in calls[0]["candidate_universe_rows"]] == [
        "AAA",
        "BBB",
        "CCC",
        "DDD",
    ]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert [row["ticker"] for row in state["full_watchlist_rows"]] == [
        "AAA",
        "BBB",
        "CCC",
        "DDD",
    ]
    assert state["watchlist_handoff_status"]["state"] == "watchlist_handoff_restored"


def test_remaining_early_runner_capacity_ignores_off_watchlist_carryovers():
    agent = _build_agent()
    agent._get_current_position_symbols = lambda: {"AAA", "ZZZ"}

    exec_state = {"early_runner_symbols": ["AAA", "ZZZ"]}
    active_rows = [
        {"ticker": "AAA", "score": 82.0},
        {"ticker": "BBB", "score": 76.0},
    ]

    remaining = agent._remaining_early_runner_capacity(
        exec_state, active_rows=active_rows
    )

    assert remaining == 9
    assert exec_state["early_runner_symbols"] == ["AAA"]


def test_market_hours_cycle_prunes_stale_early_runner_carryover_before_rescan(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", tmp_path)
    agent = _build_agent()
    agent._get_current_position_symbols = lambda: {"AAA", "ZZZ"}
    agent.run_day_manager_cycle = lambda **kwargs: {
        "entries": 1,
        "exits": 0,
        "selected_entry_symbols": ["BBB"],
    }

    state_path = tmp_path / f".execution_state_{real_datetime.now().strftime('%Y%m%d')}.json"
    state_path.write_text(
        json.dumps(
            {
                "early_runner_done": True,
                "post_open_rescan_done": False,
                "first_hour_recheck_done": False,
                "early_runner_symbols": ["AAA", "ZZZ"],
                "full_watchlist_rows": [
                    {"ticker": "AAA", "score": 82.0},
                    {"ticker": "BBB", "score": 76.0},
                    {"ticker": "CCC", "score": 71.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    wait_seconds = agent._run_market_hours_cycle(cycle_count=1, execute=True)

    assert wait_seconds == 60
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["post_open_rescan_done"] is True
    assert state["early_runner_symbols"] == ["AAA", "BBB"]


def test_market_open_cycle_logs_traceback_via_logger_exception(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", tmp_path)
    agent = _build_agent()
    logged = []
    healed = []
    workflow_health = []

    class _ExceptionLogger(_LoggerStub):
        def exception(self, *args, **kwargs):
            logged.append(args[0] if args else "")

    agent.logger = _ExceptionLogger()
    agent.plan_generator = SimpleNamespace(
        get_account_info=lambda: (_ for _ in ()).throw(RuntimeError("open boom")),
        get_current_positions=lambda: [],
    )
    agent._detect_and_fix_runtime_errors = lambda *args, **kwargs: healed.append(
        args[0] if args else None
    )
    agent.monitor_workflow_health = lambda phase, payload: workflow_health.append(
        (phase, payload)
    )

    wait_seconds = agent._run_market_open_cycle(cycle_count=1, execute=True)

    assert wait_seconds == 30
    assert logged == ["   [ERROR] Market open observation failed"]
    assert healed == ["open boom"]
    assert workflow_health[-1] == (
        "market_open",
        {"entries": 0, "exits": 0, "errors": 1},
    )


def test_market_hours_cycle_logs_traceback_via_logger_exception(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", tmp_path)
    agent = _build_agent()
    logged = []
    healed = []
    workflow_health = []

    class _ExceptionLogger(_LoggerStub):
        def exception(self, *args, **kwargs):
            logged.append(args[0] if args else "")

    agent.logger = _ExceptionLogger()
    agent.plan_generator = SimpleNamespace(
        detect_odd_lots=lambda min_position_size=5: (_ for _ in ()).throw(
            RuntimeError("hours boom")
        ),
        detect_losers_to_cut=lambda max_loss_pct=-5.0: [],
        get_current_positions=lambda: [],
        _load_latest_plan=lambda: {},
    )
    agent._detect_and_fix_runtime_errors = lambda *args, **kwargs: healed.append(
        args[0] if args else None
    )
    agent.monitor_workflow_health = lambda phase, payload: workflow_health.append(
        (phase, payload)
    )

    wait_seconds = agent._run_market_hours_cycle(cycle_count=1, execute=True)

    assert wait_seconds == 60
    assert logged == ["   [ERROR] Market hours cycle failed"]
    assert healed == ["hours boom"]
    assert workflow_health[-1] == (
        "market_hours",
        {"entries": 0, "exits": 0, "errors": 1},
    )


def test_wave_entry_regime_gate_uses_session_capacity_instead_of_plan_regime():
    agent = _build_agent()
    agent.max_positions = 2

    allowed, reason = agent._wave_entry_regime_gate(
        {
            "resolved_regime": {
                "regime": "CRASH",
                "allow_new_longs": False,
                "max_positions": 0,
                "strategy_overrides": {"max_positions": 0},
            },
            "premarket_regime_gate": {
                "allow_new_longs": False,
                "regime": "CRASH",
            },
        },
        current_positions=["AAA", "BBB"],
    )

    assert allowed is False
    assert reason == "max_positions_reached:2/2:regime=CRASH"


def test_wave_hard_reject_gap_pct_uses_regime_specific_caps():
    agent = _build_agent()
    agent._effective_market_regime = lambda: "LEAN_BULLISH"
    assert agent._wave_hard_reject_gap_pct() == 15.0

    agent._effective_market_regime = lambda: "NEUTRAL"
    assert agent._wave_hard_reject_gap_pct() == 10.0

    agent._effective_market_regime = lambda: "CRISIS"
    assert agent._wave_hard_reject_gap_pct() == 5.0


def test_execute_entry_waves_respects_session_capacity_gate(monkeypatch, tmp_path: Path):
    class _WaveOneDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 3, 20, 10, 0, 0, tzinfo=tz)

    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", tmp_path)
    monkeypatch.setattr(autonomous_agent_mod, "datetime", _WaveOneDateTime)

    state_path = tmp_path / ".execution_state_20260320.json"
    state_path.write_text(json.dumps({"opening_snapshot": {"AAA": {}}}), encoding="utf-8")

    agent = _build_agent()
    agent.max_positions = 1
    agent._execute_entry_waves = AutonomousAgent._execute_entry_waves.__get__(
        agent, AutonomousAgent
    )
    dm_calls = []
    agent._get_day_manager = lambda dry_run=False: SimpleNamespace(
        execute_entry=lambda *args, **kwargs: dm_calls.append((args, kwargs)) or True
    )
    agent._get_current_position_symbols = lambda: {"HOLD1"}
    agent.plan_generator = SimpleNamespace(
        _load_latest_plan=lambda: {
            "resolved_regime": {
                "regime": "CRASH",
                "allow_new_longs": False,
                "max_positions": 0,
                "strategy_overrides": {"max_positions": 0},
            },
            "premarket_regime_gate": {
                "allow_new_longs": False,
                "regime": "CRASH",
            },
            "entry_orders": [
                {"symbol": "AAA", "entry_price": 10.0, "qty": 10, "score": 82.0}
            ],
        },
        _coerce_float=lambda value, default=0.0: float(value or default),
        _coerce_int=lambda value, default=0: int(value or default),
        _get_current_price_quick=lambda symbol: 10.1,
    )

    agent._execute_entry_waves(execute=True)

    assert dm_calls == []
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["wave1_done"] is True
    assert state["wave_block_reason"].startswith("max_positions_reached:1/1")
    assert "AAA" in state["skipped_symbols"]


def test_execute_entry_waves_honors_core_and_reserve_caps(monkeypatch, tmp_path: Path):
    class _WaveOneDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 3, 20, 10, 0, 0, tzinfo=tz)

    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", tmp_path)
    monkeypatch.setattr(autonomous_agent_mod, "datetime", _WaveOneDateTime)

    state_path = tmp_path / ".execution_state_20260320.json"
    state_path.write_text(json.dumps({"opening_snapshot": {"AAA": {}}}), encoding="utf-8")

    dm_calls = []
    agent = _build_agent()
    agent.max_positions = 50
    agent.core_max_positions = 40
    agent.reserve_max_positions = 10
    agent._execute_entry_waves = AutonomousAgent._execute_entry_waves.__get__(
        agent, AutonomousAgent
    )
    agent._get_day_manager = lambda dry_run=False: SimpleNamespace(
        execute_entry=lambda *args, **kwargs: dm_calls.append((args, kwargs)) or True,
        _effective_max_positions=lambda: 50,
    )
    agent.get_current_positions = lambda: [
        SimpleNamespace(symbol=f"C{i}", market_value=1000.0) for i in range(40)
    ]
    agent._wave_hard_reject_gap_pct = lambda: 10.0
    agent._get_current_position_symbols = lambda: {f"C{i}" for i in range(40)}
    agent.plan_generator = SimpleNamespace(
        _load_latest_plan=lambda: {
            "resolved_regime": {"regime": "NEUTRAL"},
            "entry_constraints": {
                "max_positions": 50,
                "core_max_positions": 40,
                "reserve_max_positions": 10,
                "weak_day": False,
                "source": "pm_workflow",
                "regime": "NEUTRAL",
            },
            "entry_orders": [
                {"symbol": "CORE1", "entry_price": 10.0, "qty": 10, "score": 82.0},
                {"symbol": "SQQQ", "entry_price": 10.0, "qty": 10, "score": 81.0},
            ],
        },
        _coerce_float=lambda value, default=0.0: float(value or default),
        _coerce_int=lambda value, default=0: int(value or default),
        _get_current_price_quick=lambda symbol: 10.05,
        get_current_positions=lambda: [],
        _log_trade_decision=lambda *args, **kwargs: None,
    )

    agent._execute_entry_waves(execute=True)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert agent.max_positions == 50
    assert agent.core_max_positions == 40
    assert agent.reserve_max_positions == 10
    assert "CORE1" in state.get("skipped_symbols", [])
    assert "SQQQ" in state.get("executed_symbols", [])
    assert len(dm_calls) == 2
    assert all(call[0][0] == "SQQQ" for call in dm_calls)


def test_execute_entry_waves_uses_configured_wave_entry_limit(monkeypatch, tmp_path: Path):
    class _WaveOneDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 3, 20, 10, 0, 0, tzinfo=tz)

    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", tmp_path)
    monkeypatch.setattr(autonomous_agent_mod, "datetime", _WaveOneDateTime)

    state_path = tmp_path / ".execution_state_20260320.json"
    state_path.write_text(json.dumps({"opening_snapshot": {"AAA": {}}}), encoding="utf-8")

    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
    dm_calls = []
    upserts = []
    agent = _build_agent()
    agent.max_positions = 50
    agent.core_max_positions = 40
    agent.reserve_max_positions = 10
    agent.entry_quality_cfg.wave_max_entries = 30
    agent._execute_entry_waves = AutonomousAgent._execute_entry_waves.__get__(
        agent, AutonomousAgent
    )
    agent._get_day_manager = lambda dry_run=False: SimpleNamespace(
        execute_entry=lambda *args, **kwargs: dm_calls.append((args, kwargs)) or True,
        _effective_max_positions=lambda: 50,
    )
    agent._upsert_signals_into_dm = (
        lambda dm, rows, **kwargs: upserts.append((rows, kwargs)) or (len(rows), [rows[0]["ticker"]])
    )
    agent._wave_hard_reject_gap_pct = lambda: 10.0
    agent._get_current_position_symbols = lambda: set()
    agent.get_current_positions = lambda: []
    agent.plan_generator = SimpleNamespace(
        _load_latest_plan=lambda: {
            "resolved_regime": {"regime": "NEUTRAL"},
            "entry_constraints": {
                "max_positions": 50,
                "core_max_positions": 40,
                "reserve_max_positions": 10,
                "weak_day": False,
                "source": "pm_workflow",
                "regime": "NEUTRAL",
            },
            "entry_orders": [
                {"symbol": symbol, "entry_price": 10.0, "qty": 10, "score": 82.0}
                for symbol in symbols
            ],
        },
        _coerce_float=lambda value, default=0.0: float(value or default),
        _coerce_int=lambda value, default=0: int(value or default),
        _get_current_price_quick=lambda symbol: 10.05,
        get_current_positions=lambda: [],
        _log_trade_decision=lambda *args, **kwargs: None,
    )

    agent._execute_entry_waves(execute=True)

    submitted_symbols = [
        call[0][0] for call in dm_calls if not call[1].get("preflight_only")
    ]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert submitted_symbols == symbols
    assert len(upserts) == len(symbols)
    assert state.get("executed_symbols") == symbols


def test_execute_entry_waves_routes_orders_through_day_manager(monkeypatch, tmp_path: Path):
    class _WaveOneDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 3, 20, 10, 0, 0, tzinfo=tz)

    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", tmp_path)
    monkeypatch.setattr(autonomous_agent_mod, "datetime", _WaveOneDateTime)

    state_path = tmp_path / ".execution_state_20260320.json"
    state_path.write_text(json.dumps({"opening_snapshot": {"AAA": {}}}), encoding="utf-8")

    dm_calls = []
    upserts = []
    agent = _build_agent()
    agent.max_positions = 5
    agent._execute_entry_waves = AutonomousAgent._execute_entry_waves.__get__(
        agent, AutonomousAgent
    )
    agent._get_day_manager = lambda dry_run=False: SimpleNamespace(
        execute_entry=lambda *args, **kwargs: dm_calls.append((args, kwargs)) or True
    )
    agent._upsert_signals_into_dm = (
        lambda dm, rows, **kwargs: upserts.append((rows, kwargs)) or (len(rows), [rows[0]["ticker"]])
    )
    agent._get_current_position_symbols = lambda: set()
    agent.plan_generator = SimpleNamespace(
        _load_latest_plan=lambda: {
            "resolved_regime": {"regime": "ROTATION"},
            "entry_orders": [
                {
                    "symbol": "AAA",
                    "entry_price": 10.0,
                    "qty": 10,
                    "score": 82.0,
                    "plan_score_source": "pm_plan_2026-03-20.json",
                }
            ],
        },
        _coerce_float=lambda value, default=0.0: float(value or default),
        _coerce_int=lambda value, default=0: int(value or default),
        _get_current_price_quick=lambda symbol: 10.05,
        get_current_positions=lambda: [],
        _log_trade_decision=lambda *args, **kwargs: None,
    )

    agent._execute_entry_waves(execute=True)

    assert len(upserts) == 1
    assert upserts[0][0][0]["ticker"] == "AAA"
    assert upserts[0][1]["default_entry_source"] == "overnight_plan"
    assert len(dm_calls) == 2
    assert dm_calls[0][0][0] == "AAA"
    assert dm_calls[0][1]["preflight_only"] is True
    assert dm_calls[1][0][0] == "AAA"
    assert dm_calls[0][1]["entry_wave"] == 1
    assert "preflight_only" not in dm_calls[1][1]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "AAA" in state.get("executed_symbols", [])


def test_execute_entry_waves_allows_quick_turnover_continuation_rescue(
    monkeypatch, tmp_path: Path
):
    class _WaveTwoDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 22, 10, 30, 0, tzinfo=tz)

    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", tmp_path)
    monkeypatch.setattr(autonomous_agent_mod, "datetime", _WaveTwoDateTime)

    state_path = tmp_path / ".execution_state_20260422.json"
    state_path.write_text(
        json.dumps({"opening_snapshot": {"LUNR": {"open_price": 30.44}}}),
        encoding="utf-8",
    )

    dm_calls = []
    agent = _build_agent()
    agent.max_positions = 5
    agent.entry_quality_cfg.wave_breakout_rescue_enabled = True
    agent.entry_quality_cfg.wave_breakout_rescue_min_score = 75.0
    agent.entry_quality_cfg.wave_breakout_rescue_max_gap_pct = 9.0
    agent.entry_quality_cfg.quick_turnover_continuation_enabled = True
    agent.entry_quality_cfg.quick_turnover_continuation_min_score = 66.0
    agent.entry_quality_cfg.quick_turnover_continuation_min_volume_ratio = 1.0
    agent._effective_market_regime = lambda: "NEUTRAL"
    agent._execute_entry_waves = AutonomousAgent._execute_entry_waves.__get__(
        agent, AutonomousAgent
    )
    agent._get_day_manager = lambda dry_run=False: SimpleNamespace(
        execute_entry=lambda *args, **kwargs: dm_calls.append((args, kwargs)) or True
    )
    agent._upsert_signals_into_dm = lambda *args, **kwargs: (1, ["LUNR"])
    agent._get_current_position_symbols = lambda: set()
    agent.get_current_positions = lambda: []
    agent.plan_generator = SimpleNamespace(
        _load_latest_plan=lambda: {
            "resolved_regime": {"regime": "NEUTRAL"},
            "entry_orders": [
                {
                    "symbol": "LUNR",
                    "entry_price": 28.9773,
                    "qty": 10,
                    "score": 67.75,
                    "recommendation": "BUY",
                    "overnight_execution_intent": "quick_turnover",
                    "overnight_actionability_score": 69.16,
                    "risk_reward": 1.75,
                    "volume_ratio": 1.26,
                    "setup_type": "pullback_support",
                    "plan_score_source": "morning_game_plan_20260422.json",
                }
            ],
        },
        _coerce_float=lambda value, default=0.0: float(value or default),
        _coerce_int=lambda value, default=0: int(value or default),
        _get_current_price_quick=lambda symbol: 30.44 if symbol == "LUNR" else 0.0,
        get_current_positions=lambda: [],
        _log_trade_decision=lambda *args, **kwargs: None,
    )

    agent._execute_entry_waves(execute=True)

    assert len(dm_calls) == 2
    assert all(call[0][0] == "LUNR" for call in dm_calls)
    assert dm_calls[0][1]["preflight_only"] is True
    assert dm_calls[0][1]["candidate_data"]["overnight_execution_intent"] == "quick_turnover"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "LUNR" in state.get("executed_symbols", [])
    assert "LUNR" not in state.get("skipped_symbols", [])


def test_build_early_runner_rows_prioritizes_actionable_and_deep_research():
    agent = _build_agent()
    agent._get_scalp_watchlist_symbols = lambda: ["SCALP"]
    agent._get_actionable_entry_rows = lambda: [
        {"ticker": "AAA", "ranking_score": 82.0},
        {"ticker": "STALE", "ranking_score": 90.0, "stale_entry_action": "temp_evicted"},
    ]
    agent._get_full_watchlist_rows = lambda: [
        {"ticker": "SCALP", "ranking_score": 70.0},
        {"ticker": "AAA", "ranking_score": 82.0},
        {"ticker": "TIGO", "ranking_score": 72.75, "deep_research_bridge": True},
        {"ticker": "ZZZ", "ranking_score": 60.0},
    ]

    rows = agent._build_early_runner_rows()

    assert [row["ticker"] for row in rows[:3]] == ["SCALP", "AAA", "TIGO"]
    assert "STALE" not in [row["ticker"] for row in rows]
    assert rows[0]["entry_source"] == "premarket_scalp_watchlist"
    assert rows[1]["entry_source"] == "early_runner_actionable"
