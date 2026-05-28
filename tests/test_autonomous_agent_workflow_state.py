import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest

from autotrade.core import autonomous_agent as autonomous_agent_mod
from autotrade.core.autonomous_agent import (
    AutonomousAgent,
    OvernightResearchEngine,
    TomorrowsPlanGenerator,
)


class _LoggerStub:
    def __init__(self):
        self.messages = {"info": [], "debug": [], "warning": [], "error": []}

    def info(self, *args, **kwargs):
        self.messages["info"].append(args[0] % args[1:] if len(args) > 1 else args[0])

    def debug(self, *args, **kwargs):
        self.messages["debug"].append(args[0] % args[1:] if len(args) > 1 else args[0])

    def warning(self, *args, **kwargs):
        self.messages["warning"].append(
            args[0] % args[1:] if len(args) > 1 else args[0]
        )

    def error(self, *args, **kwargs):
        self.messages["error"].append(args[0] % args[1:] if len(args) > 1 else args[0])


def test_update_workflow_state_tracks_phase_transitions(tmp_path, monkeypatch):
    ws_path = Path(tmp_path) / "workflow_state.json"
    today = datetime.now().strftime("%Y-%m-%d")
    readiness = {
        "is_fresh": True,
        "primary_date": today,
        "expected_date": today,
        "blocking_reasons": [],
    }
    ws_path.write_text(
        json.dumps(
            {
                "last_date": today,
                "current_phase": "premarket",
                "phase_entered_at": (datetime.now() - timedelta(hours=2)).isoformat(),
                "daily_flags": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(autonomous_agent_mod, "DATA_DIR", Path(tmp_path))

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent.scheduler = SimpleNamespace(
        get_market_phase=lambda: SimpleNamespace(value="market_hours")
    )
    monkeypatch.setattr(
        autonomous_agent_mod,
        "get_core_market_data_readiness",
        lambda: dict(readiness),
    )

    agent._update_workflow_state_flag("market_hours_ran")

    state = json.loads(ws_path.read_text(encoding="utf-8"))
    assert state["current_phase"] == "market_hours"
    assert state["last_phase"] == "premarket"
    assert state["daily_flags"]["market_hours_ran"] is True
    assert state["data_readiness"] == readiness


def test_save_overnight_state_writes_diagnostic_data_freshness(tmp_path, monkeypatch):
    readiness = {
        "is_fresh": False,
        "primary_date": "2026-04-30",
        "expected_date": "2026-05-01",
        "blocking_reasons": ["primary_data_stale"],
    }
    monkeypatch.setattr(autonomous_agent_mod, "PROJECT_DIR", Path(tmp_path))
    monkeypatch.setattr(
        autonomous_agent_mod,
        "get_core_market_data_readiness",
        lambda: dict(readiness),
    )

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent._get_strategy_pool_snapshot = lambda: {}
    agent._summarize_backtest_provenance = lambda rows: {}
    agent._build_backtest_provenance_detail = lambda rows: []
    agent._summarize_catalyst_coverage = lambda rows: {}
    agent._build_catalyst_coverage_detail = lambda rows: []

    agent._save_overnight_state({"watchlist": []})

    state_path = Path(tmp_path) / "research" / "overnight_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "data_freshness" not in state
    assert state["data_freshness_at_overnight_run"] == readiness


def test_load_workflow_state_normalizes_stale_phase_timestamp(tmp_path, monkeypatch):
    ws_path = Path(tmp_path) / "workflow_state.json"
    today = datetime.now().strftime("%Y-%m-%d")
    stale = datetime.now() - timedelta(days=2)
    ws_path.write_text(
        json.dumps(
            {
                "last_date": today,
                "current_phase": "post_market",
                "phase_entered_at": stale.isoformat(),
                "daily_flags": {"overnight_ran": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(autonomous_agent_mod, "DATA_DIR", Path(tmp_path))

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent._pm_plan_done_today = False
    agent._reflect_done_today = False
    agent._full_overnight_done_today = False

    agent._load_workflow_state()

    state = json.loads(ws_path.read_text(encoding="utf-8"))
    assert datetime.fromisoformat(state["phase_entered_at"]) > stale
    assert agent._full_overnight_done_today is True


def test_load_workflow_state_prunes_stale_issues(tmp_path, monkeypatch):
    ws_path = Path(tmp_path) / "workflow_state.json"
    today = datetime.now().strftime("%Y-%m-%d")
    ws_path.write_text(
        json.dumps(
            {
                "last_date": today,
                "current_phase": "post_market",
                "phase_entered_at": datetime.now().isoformat(),
                "daily_flags": {},
                "issues": [
                    {
                        "type": "phase_stalled",
                        "timestamp": (datetime.now() - timedelta(days=10)).isoformat(),
                    },
                    {
                        "type": "phase_stalled",
                        "timestamp": datetime.now().isoformat(),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(autonomous_agent_mod, "DATA_DIR", Path(tmp_path))

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent._pm_plan_done_today = False
    agent._reflect_done_today = False
    agent._full_overnight_done_today = False

    agent._load_workflow_state()

    state = json.loads(ws_path.read_text(encoding="utf-8"))
    assert len(state["issues"]) == 1
    assert state["issues"][0]["timestamp"][:10] == today
    assert state["issues_pruned_count"] == 1


def test_build_daily_learning_advisory_marks_held_blocked_symbols(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "autotrade.utils.daily_learning_state",
        SimpleNamespace(
            load_learning_state=lambda: {
                "as_of_date": "2026-03-24",
                "active_rules": {
                    "blocked_symbols": ["FUTU", "CAVA"],
                    "blocked_setup_types": [],
                    "blocked_sectors": [],
                    "boosted_symbols": [],
                    "preferred_setup_keywords": [],
                },
                "learning_digest": {"workflow_flags": ["workflow:lessons_not_updated"]},
            }
        ),
    )

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()

    advisory = agent._build_daily_learning_advisory(
        {"positions": [{"symbol": "FUTU"}, {"symbol": "HPE"}]}
    )

    assert advisory["held_blocked_symbols"] == ["FUTU"]
    assert advisory["blocked_symbols"] == ["FUTU", "CAVA"]


def test_validate_pm_workflow_prefers_non_empty_pm_plan(tmp_path, monkeypatch):
    plans_dir = Path(tmp_path) / "plans"
    logs_dir = Path(tmp_path) / "logs"
    plans_dir.mkdir()
    logs_dir.mkdir()

    (plans_dir / "pm_plan_2026-04-02.json").write_text(
        json.dumps({"signals": [{"symbol": "PM1", "score": 77.0}]}),
        encoding="utf-8",
    )
    (plans_dir / "morning_game_plan_20260402.json").write_text(
        json.dumps({"signals": [{"symbol": "M1", "score": 88.0}]}),
        encoding="utf-8",
    )
    (logs_dir / "signals_2026-04-02.json").write_text(
        json.dumps({"signals": [{"symbol": "PM1"}]}),
        encoding="utf-8",
    )

    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 3, 9, 0, 0, tzinfo=tz)

    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", plans_dir)
    monkeypatch.setattr(autonomous_agent_mod, "LOG_DIR", logs_dir)
    monkeypatch.setattr(autonomous_agent_mod, "datetime", _FixedDateTime)
    monkeypatch.setitem(
        sys.modules,
        "autotrade.utils.market_time",
        SimpleNamespace(get_pm_plan_date=lambda _now=None: _FixedDateTime(2026, 4, 2)),
    )

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent.watchlist_optimizer = SimpleNamespace(
        analyze_current_picks=lambda path: {"score": 80, "path": path.name},
        optimize_with_openai=lambda signals: {},
        update_watchlist=lambda *args, **kwargs: None,
    )
    agent.web_researcher = SimpleNamespace(openai_client=None)
    agent.data_fetcher = SimpleNamespace(
        get_missing_tickers=lambda tickers: [],
        available=False,
        bulk_fetch=lambda tickers: [],
    )

    result = agent.validate_pm_workflow()

    assert result["analysis_plan"] == "pm_plan_2026-04-02.json"


def test_load_latest_plan_merges_pm_with_morning_tail(tmp_path, monkeypatch):
    plans_dir = Path(tmp_path) / "plans"
    plans_dir.mkdir()
    (plans_dir / "pm_plan_2026-04-02.json").write_text(
        json.dumps(
            {
                "entry_orders": [
                    {"symbol": "PM1", "qty": 10},
                    {"symbol": "OVERLAP", "qty": 20},
                ]
            }
        ),
        encoding="utf-8",
    )
    (plans_dir / "morning_game_plan_20260402.json").write_text(
        json.dumps(
            {
                "entry_orders": [
                    {"symbol": "OVERLAP", "qty": 1},
                    {"symbol": "TAIL1", "qty": 5},
                ]
            }
        ),
        encoding="utf-8",
    )

    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 2, 9, 0, 0, tzinfo=tz)

    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", plans_dir)
    monkeypatch.setattr(autonomous_agent_mod, "datetime", _FixedDateTime)
    monkeypatch.setitem(
        sys.modules,
        "autotrade.utils.market_time",
        SimpleNamespace(get_pm_plan_date=lambda _now=None: _FixedDateTime(2026, 4, 2)),
    )

    generator = TomorrowsPlanGenerator.__new__(TomorrowsPlanGenerator)
    generator.logger = _LoggerStub()
    generator._coerce_float = TomorrowsPlanGenerator._coerce_float.__get__(
        generator, TomorrowsPlanGenerator
    )

    plan = TomorrowsPlanGenerator._load_latest_plan(generator)

    rows = plan["entry_orders"]
    assert [row["symbol"] for row in rows] == ["PM1", "OVERLAP", "TAIL1"]
    assert rows[1]["qty"] == 20
    assert [row["plan_source"] for row in rows] == ["pm", "pm", "morning_tail"]
    assert plan["plan_source"] == "pm_plus_morning_tail"
    assert plan["pm_plan_merge"] == {
        "pm_rows": 2,
        "morning_tail_rows": 1,
        "overlap_rows": 1,
        "total_rows": 3,
    }


def test_validate_pm_workflow_refreshes_drifted_signals_ledger(tmp_path, monkeypatch):
    plans_dir = Path(tmp_path) / "plans"
    logs_dir = Path(tmp_path) / "logs"
    plans_dir.mkdir()
    logs_dir.mkdir()

    (plans_dir / "pm_plan_2026-04-02.json").write_text(
        json.dumps(
            {
                "signals": [
                    {"symbol": "PM1", "score": 77.0},
                    {"symbol": "PM2", "score": 70.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    (logs_dir / "signals_2026-04-02.json").write_text(
        json.dumps({"signals": [{"symbol": "OLD1"}]}),
        encoding="utf-8",
    )

    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 3, 9, 0, 0, tzinfo=tz)

    class _SignalGen:
        def save_signals(
            self,
            signals,
            target_date=None,
            source=None,
            allow_overwrite=False,
            min_count=10,
            enforce_filters=True,
            metadata=None,
        ):
            payload = {
                "date": target_date,
                "signals": list(signals),
                "total_signals": len(signals),
                "source": source,
                "signal_manifest": dict(metadata or {}),
            }
            (logs_dir / f"signals_{target_date}.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", plans_dir)
    monkeypatch.setattr(autonomous_agent_mod, "LOG_DIR", logs_dir)
    monkeypatch.setattr(autonomous_agent_mod, "datetime", _FixedDateTime)
    monkeypatch.setitem(
        sys.modules,
        "autotrade.utils.market_time",
        SimpleNamespace(get_pm_plan_date=lambda _now=None: _FixedDateTime(2026, 4, 2)),
    )
    monkeypatch.setitem(
        sys.modules,
        "autotrade.signals.agentic_signal_generator",
        SimpleNamespace(AgenticSignalGenerator=_SignalGen),
    )

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent.watchlist_optimizer = SimpleNamespace(
        analyze_current_picks=lambda path: {"score": 80, "path": path.name},
        optimize_with_openai=lambda signals: {},
        update_watchlist=lambda *args, **kwargs: None,
    )
    agent.web_researcher = SimpleNamespace(openai_client=None)
    agent.data_fetcher = SimpleNamespace(
        get_missing_tickers=lambda tickers: [],
        available=False,
        bulk_fetch=lambda tickers: [],
    )

    result = agent.validate_pm_workflow()

    refreshed = json.loads(
        (logs_dir / "signals_2026-04-02.json").read_text(encoding="utf-8")
    )
    refreshed_symbols = sorted(row["symbol"] for row in refreshed["signals"])

    assert refreshed_symbols == ["PM1", "PM2"]
    assert (
        "Refreshed raw signals ledger from pm_plan_2026-04-02.json"
        in result["actions_taken"]
    )


def test_build_pm_artifact_audit_does_not_report_retired_pm_news_audit(
    tmp_path, monkeypatch
):
    reports_dir = Path(tmp_path) / "reports"
    reports_dir.mkdir()

    target_date = "2026-04-14"
    (reports_dir / f"daily_lessons_{target_date}.md").write_text(
        "## 4. Actionable Lessons & System Improvements\n1. Tighten PM review discipline.\n",
        encoding="utf-8",
    )
    (reports_dir / f"post_market_lesson_{target_date}.json").write_text(
        json.dumps({"date": target_date}),
        encoding="utf-8",
    )
    (reports_dir / f"sequential_shadow_accuracy_{target_date}.json").write_text(
        json.dumps({"success": True, "summary": {"evaluated_events": 3}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(autonomous_agent_mod, "PROJECT_DIR", Path(tmp_path))

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()

    audit = agent._build_pm_artifact_audit(
        pm_result={"sequential_shadow_eval": {"ran": True, "success": True}},
        review_result={"review_date": target_date},
        include_review=True,
        feedback_status={"matched": True},
    )

    assert audit["artifact_audit_status"] == "ok"
    assert "pm_news_audit" not in audit["skipped_artifacts"]
    assert "daily_lessons" in audit["generated_artifacts"]
    assert "post_market_lesson" in audit["generated_artifacts"]


def test_filter_inactive_symbols_drops_delisted_rows():
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent._active_tradable_symbols_cache = {"AAPL", "MSFT"}

    filtered = agent._filter_inactive_symbols(
        [{"symbol": "AAPL"}, {"symbol": "BLBX"}, {"symbol": "MSFT"}],
        context_label="test",
    )

    assert filtered == [{"symbol": "AAPL"}, {"symbol": "MSFT"}]


def test_build_stale_entry_diagnostics_reprices_and_temp_evicts(tmp_path, monkeypatch):
    plans_dir = Path(tmp_path)
    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", plans_dir)

    (plans_dir / "morning_game_plan_20260323.json").write_text(
        json.dumps({"signals": [{"symbol": "AAA", "entry_price": 10.0}]}),
        encoding="utf-8",
    )
    (plans_dir / "morning_game_plan_20260324.json").write_text(
        json.dumps({"signals": [{"symbol": "AAA", "entry_price": 10.2}]}),
        encoding="utf-8",
    )
    (plans_dir / ".execution_state_20260324.json").write_text(
        json.dumps({"opening_snapshot": {"AAA": {"gap_pct": 8.4}}}),
        encoding="utf-8",
    )
    (plans_dir / ".execution_state_20260323.json").write_text(
        json.dumps({"opening_snapshot": {"AAA": {"gap_pct": 7.2}}}),
        encoding="utf-8",
    )

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent.entry_quality_cfg = SimpleNamespace(
        stale_entry_reprice_streak=2,
        stale_entry_gap_reject_streak=2,
        stale_entry_temp_evict_sessions=1,
        entry_gap_reject_pct=7.0,
    )
    agent.plan_generator = SimpleNamespace(
        _coerce_float=lambda value, default=0.0: float(value or default)
    )

    rows, diagnostics = agent._build_stale_entry_diagnostics(
        [{"symbol": "AAA", "entry_price": 10.0, "current_price": 11.25}],
        "20260325",
    )

    assert rows[0]["entry_price"] == 11.25
    assert rows[0]["stale_entry_action"] == "temp_evicted"
    assert diagnostics["repriced_symbols"] == ["AAA"]
    assert diagnostics["temporarily_evicted_symbols"] == ["AAA"]


def test_merge_promoted_deep_research_rows_bridges_strong_buy(tmp_path, monkeypatch):
    reports_dir = Path(tmp_path) / "reports"
    reports_dir.mkdir(parents=True)
    monkeypatch.setattr(autonomous_agent_mod, "PROJECT_DIR", Path(tmp_path))

    (reports_dir / "top_pick_deep_research_20260325_025649.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "symbol": "TIGO",
                        "deep_research": {
                            "recommendation": "STRONG BUY",
                            "final_score": 72.75,
                            "entry_price": 75.07,
                            "target": 82.95,
                            "stop_loss": 70.56,
                        },
                        "backtest": {"win_rate": 56.2, "total_trades": 665},
                        "web_research": {
                            "sentiment": "bullish",
                            "fresh_news": True,
                            "catalyst_count": 4,
                        },
                        "watchlist_update": {
                            "catalyst_note": "Q4 earnings transcript",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()

    merged = agent._merge_promoted_deep_research_rows(
        [{"symbol": "AAA", "confidence": 70.0}]
    )

    by_symbol = {row["symbol"]: row for row in merged}
    assert "TIGO" in by_symbol
    assert by_symbol["TIGO"]["deep_research_bridge"] is True
    assert by_symbol["TIGO"]["recommendation"] == "STRONG BUY"
    assert by_symbol["TIGO"]["entry_price"] == 75.07


def test_run_reflect_phase_uses_daily_review_session_date_for_lessons(
    monkeypatch, tmp_path
):
    class _PlanGenerator:
        @staticmethod
        def get_account_info():
            return {"day_pnl": 0.0, "equity": 100000.0}

    class _DailyReview:
        def run(self, learn=True):
            return {"review_date": "2026-03-10", "day_summary": {"grade": "OK"}}

    class _LessonsAnalyzer:
        def run(self, date_str=None):
            return {"date": date_str}

    monkeypatch.setitem(
        sys.modules,
        "autotrade.core.daily_review",
        SimpleNamespace(DailyReview=_DailyReview),
    )
    monkeypatch.setitem(
        sys.modules,
        "autotrade.core.daily_lessons_analyzer",
        SimpleNamespace(DailyLessonsAnalyzer=_LessonsAnalyzer),
    )
    monkeypatch.setitem(
        sys.modules,
        "autotrade.signals.trade_learner",
        SimpleNamespace(analyze_today=lambda date_str=None: {"date": date_str}),
    )

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent.plan_generator = _PlanGenerator()
    agent.task_router = None
    agent.strategy_failsafe = None
    agent._refresh_strategy_failsafe = lambda source=None: None
    agent._run_self_debugging = lambda day_pnl: None
    agent._update_workflow_state_flag = lambda flag_name: None
    agent._append_jsonl = lambda *args, **kwargs: None
    agent._reflect_done_today = False
    agent.RESEARCH_DIR = Path(tmp_path)
    monkeypatch.setattr(autonomous_agent_mod, "LOG_DIR", Path(tmp_path))

    result = agent._run_reflect_phase(force=True)

    assert result["success"] is True
    assert result["review_session_date"] == "2026-03-10"
    assert result["trade_learner"]["date"] == "2026-03-10"
    assert result["daily_lessons"]["date"] == "2026-03-10"


def test_ensure_feedback_latest_matches_review_rewrites_stale_file(
    monkeypatch, tmp_path
):
    data_dir = Path(tmp_path) / "data"
    data_dir.mkdir(parents=True)
    feedback_path = data_dir / "eod_feedback_latest.json"
    feedback_path.write_text(
        json.dumps({"date": "2026-03-18", "win_rate": 0.5}),
        encoding="utf-8",
    )
    monkeypatch.setattr(autonomous_agent_mod, "PROJECT_DIR", Path(tmp_path))

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()

    result = agent._ensure_feedback_latest_matches_review(
        {
            "review_date": "2026-03-20",
            "learning": {
                "date": "2026-03-20",
                "win_rate": 0.62,
                "lessons_updated": True,
            },
        }
    )

    saved = json.loads(feedback_path.read_text(encoding="utf-8"))
    assert result["matched"] is True
    assert result["rewritten_from_review"] is True
    assert saved["date"] == "2026-03-20"
    assert saved["win_rate"] == 0.62


def test_ensure_feedback_latest_matches_review_writes_fallback_when_learning_missing(
    monkeypatch, tmp_path
):
    data_dir = Path(tmp_path) / "data"
    data_dir.mkdir(parents=True)
    feedback_path = data_dir / "eod_feedback_latest.json"
    feedback_path.write_text(
        json.dumps({"date": "2026-03-18", "win_rate": 0.5}),
        encoding="utf-8",
    )
    monkeypatch.setattr(autonomous_agent_mod, "PROJECT_DIR", Path(tmp_path))

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()

    result = agent._ensure_feedback_latest_matches_review(
        {
            "success": True,
            "review_date": "2026-03-20",
            "total_pnl": 999.0,
            "broker_day_pnl": 123.45,
            "realized_day_pnl": 23.45,
            "open_position_unrealized_pnl": 100.0,
            "learning": {"date": "2026-03-19", "win_rate": 0.62},
        }
    )

    saved = json.loads(feedback_path.read_text(encoding="utf-8"))
    assert result["matched"] is True
    assert result["rewritten_from_review"] is True
    assert result["reason"] == "rewritten_from_review_fallback"
    assert saved["date"] == "2026-03-20"
    assert saved["source"] == "daily_review_fallback"
    assert saved["total_pnl"] == 123.45
    assert saved["realized_day_pnl"] == 23.45
    assert saved["open_position_unrealized_pnl"] == 100.0
    assert saved["lessons_status"]["success"] is False


def test_ensure_feedback_latest_rewrites_same_date_inconsistent_broker_truth(
    monkeypatch, tmp_path
):
    data_dir = Path(tmp_path) / "data"
    data_dir.mkdir(parents=True)
    feedback_path = data_dir / "eod_feedback_latest.json"
    feedback_path.write_text(
        json.dumps({"date": "2026-05-01", "total_pnl": -75.0, "win_rate": 0.5}),
        encoding="utf-8",
    )
    monkeypatch.setattr(autonomous_agent_mod, "PROJECT_DIR", Path(tmp_path))

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()

    result = agent._ensure_feedback_latest_matches_review(
        {
            "success": True,
            "review_date": "2026-05-01",
            "broker_day_pnl": -425.0,
            "realized_day_pnl": -100.0,
            "open_position_unrealized_pnl": -325.0,
            "learning": {"date": "2026-05-01", "win_rate": 0.42},
        }
    )

    saved = json.loads(feedback_path.read_text(encoding="utf-8"))
    assert result["matched"] is True
    assert result["rewritten_from_review"] is True
    assert saved["total_pnl"] == -425.0
    assert saved["realized_day_pnl"] == -100.0
    assert saved["open_position_unrealized_pnl"] == -325.0
    assert saved["broker_day_pnl"] == -425.0


def test_ensure_feedback_latest_matches_review_accepts_persisted_date_key(
    monkeypatch, tmp_path
):
    data_dir = Path(tmp_path) / "data"
    data_dir.mkdir(parents=True)
    feedback_path = data_dir / "eod_feedback_latest.json"
    feedback_path.write_text(
        json.dumps({"date": "2026-04-27", "win_rate": 0.5}),
        encoding="utf-8",
    )
    monkeypatch.setattr(autonomous_agent_mod, "PROJECT_DIR", Path(tmp_path))

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()

    result = agent._ensure_feedback_latest_matches_review(
        {
            "success": True,
            "date": "2026-04-28",
            "total_pnl": -925.45,
        }
    )

    saved = json.loads(feedback_path.read_text(encoding="utf-8"))
    assert result["matched"] is True
    assert result["target_date"] == "2026-04-28"
    assert saved["date"] == "2026-04-28"
    assert saved["source"] == "daily_review_fallback"


def test_build_overnight_artifact_audit_recovers_bundle_and_signals(
    monkeypatch, tmp_path
):
    plans_dir = Path(tmp_path) / "plans"
    logs_dir = Path(tmp_path) / "logs"
    research_dir = Path(tmp_path) / "research"
    reports_dir = Path(tmp_path) / "reports"
    plans_dir.mkdir()
    logs_dir.mkdir()
    research_dir.mkdir()
    reports_dir.mkdir()

    target_date = "2026-03-25"
    plan_path = plans_dir / "morning_game_plan_20260325.json"
    plan_path.write_text(
        json.dumps(
            {
                "date": target_date,
                "signals": [{"symbol": "AAA", "score": 88.0}],
                "full_watchlist": [{"symbol": "AAA", "score": 88.0}],
                "resolved_regime": {"regime": "NEUTRAL"},
            }
        ),
        encoding="utf-8",
    )

    state = {
        "research_complete": True,
        "watchlist": [{"symbol": "AAA", "score": 88.0}],
        "resolved_regime": {"regime": "NEUTRAL"},
        "workflow_completion": {
            "target_trade_date": target_date,
            "game_plan_generated": True,
            "game_plan_path": str(plan_path),
        },
        "secondary_research": {"last_result_by_job": {}},
    }
    (research_dir / "overnight_state.json").write_text(
        json.dumps(state),
        encoding="utf-8",
    )
    (reports_dir / f"daily_lessons_{target_date}.md").write_text(
        "# Lessons\n",
        encoding="utf-8",
    )
    (reports_dir / f"post_market_lesson_{target_date}.json").write_text(
        json.dumps({"date": target_date}),
        encoding="utf-8",
    )

    class _SignalGen:
        def save_signals(
            self,
            signals,
            target_date=None,
            source=None,
            allow_overwrite=False,
            min_count=1,
            enforce_filters=False,
            metadata=None,
        ):
            (logs_dir / f"signals_{target_date}.json").write_text(
                json.dumps(
                    {"signals": signals, "source": source, "metadata": metadata}
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr(autonomous_agent_mod, "PROJECT_DIR", Path(tmp_path))
    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", plans_dir)
    monkeypatch.setattr(autonomous_agent_mod, "LOG_DIR", logs_dir)
    monkeypatch.setitem(
        sys.modules,
        "autotrade.signals.agentic_signal_generator",
        SimpleNamespace(AgenticSignalGenerator=_SignalGen),
    )

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent.overnight_secondary_cfg = SimpleNamespace(
        top_pick_research_enabled=False,
        historical_revisit_enabled=False,
        signal_mining_enabled=False,
    )

    audit = agent._build_overnight_artifact_audit(
        state,
        trigger_reason="queue_exhausted",
    )

    assert audit["artifact_audit_status"] == "ok"
    assert "morning_game_plan" in audit["generated_artifacts"]
    assert "overnight_research_bundle" in audit["generated_artifacts"]
    assert "signals_ledger" in audit["generated_artifacts"]
    assert (plans_dir / "overnight_research_bundle_latest.json").exists()
    assert (logs_dir / "signals_2026-03-25.json").exists()


def test_build_overnight_artifact_audit_backfills_missing_reflect_artifacts(
    monkeypatch, tmp_path
):
    plans_dir = Path(tmp_path) / "plans"
    logs_dir = Path(tmp_path) / "logs"
    research_dir = Path(tmp_path) / "research"
    reports_dir = Path(tmp_path) / "reports"
    plans_dir.mkdir()
    logs_dir.mkdir()
    research_dir.mkdir()
    reports_dir.mkdir()

    target_date = "2026-03-25"
    plan_path = plans_dir / "morning_game_plan_20260325.json"
    plan_path.write_text(
        json.dumps(
            {
                "date": target_date,
                "signals": [{"symbol": "AAA", "score": 88.0}],
                "full_watchlist": [{"symbol": "AAA", "score": 88.0}],
                "resolved_regime": {"regime": "NEUTRAL"},
            }
        ),
        encoding="utf-8",
    )

    state = {
        "research_complete": True,
        "watchlist": [{"symbol": "AAA", "score": 88.0}],
        "resolved_regime": {"regime": "NEUTRAL"},
        "workflow_completion": {
            "target_trade_date": target_date,
            "game_plan_generated": True,
            "game_plan_path": str(plan_path),
        },
        "secondary_research": {"last_result_by_job": {}},
    }
    (research_dir / "overnight_state.json").write_text(
        json.dumps(state),
        encoding="utf-8",
    )

    class _SignalGen:
        def save_signals(
            self,
            signals,
            target_date=None,
            source=None,
            allow_overwrite=False,
            min_count=1,
            enforce_filters=False,
            metadata=None,
        ):
            (logs_dir / f"signals_{target_date}.json").write_text(
                json.dumps(
                    {"signals": signals, "source": source, "metadata": metadata}
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr(autonomous_agent_mod, "PROJECT_DIR", Path(tmp_path))
    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", plans_dir)
    monkeypatch.setattr(autonomous_agent_mod, "LOG_DIR", logs_dir)
    monkeypatch.setitem(
        sys.modules,
        "autotrade.signals.agentic_signal_generator",
        SimpleNamespace(AgenticSignalGenerator=_SignalGen),
    )

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent.overnight_secondary_cfg = SimpleNamespace(
        top_pick_research_enabled=False,
        historical_revisit_enabled=False,
        signal_mining_enabled=False,
    )
    agent._reflect_done_today = False

    reflect_calls = []

    def _fake_reflect(force=False):
        reflect_calls.append(force)
        (reports_dir / f"daily_lessons_{target_date}.md").write_text(
            "# Lessons\n",
            encoding="utf-8",
        )
        (reports_dir / f"post_market_lesson_{target_date}.json").write_text(
            json.dumps({"date": target_date}),
            encoding="utf-8",
        )
        return {"success": True, "review_session_date": target_date}

    agent._run_reflect_phase = _fake_reflect

    audit = agent._build_overnight_artifact_audit(
        state,
        trigger_reason="queue_exhausted",
    )

    assert reflect_calls == [True]
    assert audit["artifact_audit_status"] == "ok"
    assert "daily_lessons" in audit["generated_artifacts"]
    assert "post_market_lesson" in audit["generated_artifacts"]


def test_build_overnight_artifact_audit_marks_secondary_pending_when_handoff_incomplete(
    monkeypatch, tmp_path
):
    plans_dir = Path(tmp_path) / "plans"
    logs_dir = Path(tmp_path) / "logs"
    research_dir = Path(tmp_path) / "research"
    plans_dir.mkdir()
    logs_dir.mkdir()
    research_dir.mkdir()

    target_date = "2026-04-14"
    plan_path = plans_dir / "morning_game_plan_20260414.json"
    plan_path.write_text(
        json.dumps(
            {
                "date": target_date,
                "signals": [{"symbol": "AAA", "score": 88.0}],
                "full_watchlist": [{"symbol": "AAA", "score": 88.0}],
                "resolved_regime": {"regime": "NEUTRAL"},
            }
        ),
        encoding="utf-8",
    )
    (logs_dir / f"signals_{target_date}.json").write_text(
        json.dumps({"signals": [{"symbol": "AAA", "score": 88.0}]}),
        encoding="utf-8",
    )
    (plans_dir / "overnight_research_bundle_latest.json").write_text(
        json.dumps({"trade_date": target_date, "watchlist": [{"symbol": "AAA"}]}),
        encoding="utf-8",
    )

    state = {
        "research_complete": False,
        "watchlist": [{"symbol": "AAA", "score": 88.0}],
        "workflow_completion": {
            "target_trade_date": target_date,
            "game_plan_generated": True,
            "game_plan_path": str(plan_path),
            "breadth_passed": False,
            "news_coverage_passed": False,
            "completion_reason": "full_cycle_failed_breadth_gate",
        },
        "secondary_research": {"last_result_by_job": {}},
    }
    (research_dir / "overnight_state.json").write_text(
        json.dumps(state),
        encoding="utf-8",
    )

    monkeypatch.setattr(autonomous_agent_mod, "PROJECT_DIR", Path(tmp_path))
    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", plans_dir)
    monkeypatch.setattr(autonomous_agent_mod, "LOG_DIR", logs_dir)

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent.overnight_secondary_cfg = SimpleNamespace(
        run_when_research_complete=True,
        top_pick_research_enabled=True,
        historical_revisit_enabled=True,
        signal_mining_enabled=True,
    )

    audit = agent._build_overnight_artifact_audit(
        state,
        trigger_reason="queue_exhausted",
    )

    assert audit["artifact_audit_status"] == "pending"
    assert audit["missing_artifacts"] == []
    assert audit["handoff_complete"] is False
    assert (
        audit["skipped_artifacts"]["top_pick_deep_research"]
        == "waiting_for_main_handoff:full_cycle_failed_breadth_gate"
    )
    assert any(
        "[OVERNIGHT][ARTIFACT AUDIT][PENDING]" in msg
        for msg in agent.logger.messages["warning"]
    )
    assert agent.logger.messages["error"] == []


def test_build_overnight_artifact_audit_accepts_degraded_historical_artifact(
    monkeypatch, tmp_path
):
    plans_dir = Path(tmp_path) / "plans"
    logs_dir = Path(tmp_path) / "logs"
    research_dir = Path(tmp_path) / "research"
    reports_dir = Path(tmp_path) / "reports"
    plans_dir.mkdir()
    logs_dir.mkdir()
    research_dir.mkdir()
    reports_dir.mkdir()

    target_date = "2026-04-14"
    plan_path = plans_dir / "morning_game_plan_20260414.json"
    plan_path.write_text(
        json.dumps(
            {
                "date": target_date,
                "signals": [{"symbol": "AAA", "score": 88.0}],
                "full_watchlist": [{"symbol": "AAA", "score": 88.0}],
                "resolved_regime": {"regime": "NEUTRAL"},
            }
        ),
        encoding="utf-8",
    )
    (logs_dir / f"signals_{target_date}.json").write_text(
        json.dumps({"signals": [{"symbol": "AAA", "score": 88.0}]}),
        encoding="utf-8",
    )
    (plans_dir / "overnight_research_bundle_latest.json").write_text(
        json.dumps({"trade_date": target_date, "watchlist": [{"symbol": "AAA"}]}),
        encoding="utf-8",
    )

    top_pick_path = reports_dir / "top_pick_deep_research_20260414_000001.json"
    top_pick_path.write_text(
        json.dumps({"rows": [{"symbol": "AAA"}]}), encoding="utf-8"
    )
    signal_mining_path = reports_dir / "signal_mining_20260414_000001.json"
    signal_mining_path.write_text(
        json.dumps({"rows": [{"symbol": "AAA"}]}), encoding="utf-8"
    )
    historical_path = reports_dir / "historical_bullish_revisit_20260414_000001.json"
    historical_path.write_text(
        json.dumps(
            {
                "job": "historical_revisit",
                "degraded_mode": True,
                "degraded_reason": "no_candidate_symbols",
                "rows": [],
            }
        ),
        encoding="utf-8",
    )
    (reports_dir / f"daily_lessons_{target_date}.md").write_text(
        "# Lessons\n",
        encoding="utf-8",
    )
    (reports_dir / f"post_market_lesson_{target_date}.json").write_text(
        json.dumps({"date": target_date}),
        encoding="utf-8",
    )

    state = {
        "research_complete": True,
        "watchlist": [{"symbol": "AAA", "score": 88.0}],
        "workflow_completion": {
            "target_trade_date": target_date,
            "game_plan_generated": True,
            "game_plan_path": str(plan_path),
            "breadth_passed": True,
            "news_coverage_passed": False,
            "degraded_news_coverage": True,
            "completion_reason": "full_cycle_complete_with_failed_news_coverage",
        },
        "secondary_research": {
            "last_result_by_job": {
                "top_pick_research": {
                    "result": {
                        "success": True,
                        "artifact_path": str(top_pick_path),
                    }
                },
                "historical_revisit": {
                    "result": {
                        "success": True,
                        "artifact_path": str(historical_path),
                        "degraded_mode": True,
                        "reason": "no_candidate_symbols",
                    }
                },
                "signal_mining": {
                    "result": {
                        "success": True,
                        "artifact_path": str(signal_mining_path),
                    }
                },
            }
        },
    }
    (research_dir / "overnight_state.json").write_text(
        json.dumps(state),
        encoding="utf-8",
    )

    monkeypatch.setattr(autonomous_agent_mod, "PROJECT_DIR", Path(tmp_path))
    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", plans_dir)
    monkeypatch.setattr(autonomous_agent_mod, "LOG_DIR", logs_dir)

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent.overnight_secondary_cfg = SimpleNamespace(
        run_when_research_complete=True,
        top_pick_research_enabled=True,
        historical_revisit_enabled=True,
        signal_mining_enabled=True,
    )

    audit = agent._build_overnight_artifact_audit(
        state,
        trigger_reason="queue_exhausted",
    )

    assert audit["artifact_audit_status"] == "ok"
    assert audit["missing_artifacts"] == []
    assert audit["critical_missing_artifacts"] == []
    assert "daily_lessons" in audit["generated_artifacts"]
    assert "post_market_lesson" in audit["generated_artifacts"]
    assert "top_pick_deep_research" in audit["generated_artifacts"]
    assert "historical_bullish_revisit" in audit["generated_artifacts"]
    assert "signal_mining" in audit["generated_artifacts"]
    assert agent.logger.messages["error"] == []


def test_assert_overnight_artifact_contract_raises_for_missing_required_artifacts():
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()

    with pytest.raises(AssertionError, match="missing required overnight artifacts"):
        agent._assert_overnight_artifact_contract(
            {
                "handoff_complete": True,
                "missing_artifacts": [
                    "daily_lessons:required_report_missing",
                    "post_market_lesson:required_report_missing",
                    "historical_bullish_revisit:not_run",
                ],
            }
        )


def test_reconcile_overnight_state_from_existing_handoff_artifacts_marks_complete(
    monkeypatch, tmp_path
):
    plans_dir = Path(tmp_path) / "plans"
    logs_dir = Path(tmp_path) / "logs"
    research_dir = Path(tmp_path) / "research"
    plans_dir.mkdir()
    logs_dir.mkdir()
    research_dir.mkdir()

    target_date = "2026-04-14"
    plan_path = plans_dir / "morning_game_plan_20260414.json"
    watchlist_rows = [
        {
            "symbol": f"SYM{i:03d}",
            "confidence": 70.0 + (i % 20),
            "final_score": 60.0 + (i % 15),
        }
        for i in range(250)
    ]
    plan_path.write_text(
        json.dumps(
            {
                "date": target_date,
                "full_watchlist": watchlist_rows,
                "signals": watchlist_rows[:50],
                "buy_signals": watchlist_rows[:50],
                "resolved_regime": {"regime": "NEUTRAL", "available": True},
                "youtube_intelligence": {
                    "available": True,
                    "regime": "NEUTRAL",
                    "report_date": "2026-04-13",
                },
            }
        ),
        encoding="utf-8",
    )
    (logs_dir / f"signals_{target_date}.json").write_text(
        json.dumps({"date": target_date, "signals": watchlist_rows[:50]}),
        encoding="utf-8",
    )
    (research_dir / "overnight_state.json").write_text(
        json.dumps(
            {
                "date": target_date,
                "research_complete": False,
                "watchlist": [],
                "all_candidates": [],
                "strategy_pool_snapshot": {},
                "workflow_completion": {
                    "watchlist_selected": False,
                    "game_plan_generated": False,
                    "youtube_ready": False,
                    "reset_reason": "stale_date_before_market_open",
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(autonomous_agent_mod, "PROJECT_DIR", Path(tmp_path))
    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", plans_dir)
    monkeypatch.setattr(autonomous_agent_mod, "LOG_DIR", logs_dir)

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent._evaluate_overnight_breadth = lambda state: {
        "passed": True,
        "scanned_count": 4825,
        "required_count": 2000,
    }
    agent._persist_research_artifact_bundle_from_latest_plan = lambda now_et=None: (
        plans_dir / "overnight_research_bundle_latest.json"
    )

    initial_state = json.loads(
        (research_dir / "overnight_state.json").read_text(encoding="utf-8")
    )
    reconciled = agent._reconcile_overnight_state_from_existing_handoff_artifacts(
        initial_state,
        now_et=datetime(2026, 4, 14, 5, 45, 0),
        persist=True,
    )

    completion = reconciled["workflow_completion"]
    assert reconciled["research_complete"] is True
    assert reconciled["target_trade_date"] == target_date
    assert len(reconciled["watchlist"]) == 250
    assert completion["watchlist_selected"] is True
    assert completion["watchlist_target_met"] is True
    assert completion["game_plan_generated"] is True
    assert completion["breadth_passed"] is True
    assert completion["degraded_news_coverage"] is True
    assert completion["youtube_ready"] is True
    assert completion["signals_ledger_ready"] is True
    assert (
        completion["completion_reason"] == "reconciled_from_existing_handoff_artifacts"
    )

    saved = json.loads(
        (research_dir / "overnight_state.json").read_text(encoding="utf-8")
    )
    assert saved["research_complete"] is True
    assert saved["workflow_completion"]["game_plan_generated"] is True
    assert saved["workflow_completion"]["target_trade_date"] == target_date


def test_research_freshness_reconciles_existing_handoff_before_status(
    monkeypatch, tmp_path
):
    plans_dir = Path(tmp_path) / "plans"
    logs_dir = Path(tmp_path) / "logs"
    research_dir = Path(tmp_path) / "research"
    plans_dir.mkdir()
    logs_dir.mkdir()
    research_dir.mkdir()

    target_date = "2026-04-14"
    watchlist_rows = [
        {
            "symbol": f"SYM{i:03d}",
            "confidence": 70.0 + (i % 20),
            "final_score": 60.0 + (i % 15),
        }
        for i in range(250)
    ]
    (plans_dir / "morning_game_plan_20260414.json").write_text(
        json.dumps(
            {
                "date": target_date,
                "full_watchlist": watchlist_rows,
                "signals": watchlist_rows[:50],
                "buy_signals": watchlist_rows[:50],
            }
        ),
        encoding="utf-8",
    )
    (logs_dir / f"signals_{target_date}.json").write_text(
        json.dumps({"date": target_date, "signals": watchlist_rows[:50]}),
        encoding="utf-8",
    )
    (research_dir / "overnight_state.json").write_text(
        json.dumps(
            {
                "updated_at": "2026-04-14T05:30:00-04:00",
                "date": target_date,
                "research_complete": False,
                "watchlist": [],
                "all_candidates": [],
                "workflow_completion": {
                    "watchlist_selected": False,
                    "game_plan_generated": False,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(autonomous_agent_mod, "PROJECT_DIR", Path(tmp_path))
    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", plans_dir)
    monkeypatch.setattr(autonomous_agent_mod, "LOG_DIR", logs_dir)

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent.research_freshness_cfg = SimpleNamespace(
        weekday_max_age_hours=18.0,
        monday_max_age_hours=60.0,
    )
    agent._refresh_data_gateway_freshness_snapshot = lambda: {"available": False}
    agent._evaluate_overnight_breadth = lambda state: {
        "passed": True,
        "scanned_count": 4825,
        "required_count": 2000,
    }
    agent._persist_research_artifact_bundle_from_latest_plan = lambda now_et=None: (
        plans_dir / "overnight_research_bundle_latest.json"
    )

    now_et = autonomous_agent_mod.pytz.timezone("US/Eastern").localize(
        datetime(2026, 4, 14, 6, 50, 0)
    )
    result = agent._check_research_freshness(
        persist_metadata=False,
        now_et=now_et,
    )

    assert result["workflow_complete"] is True
    assert result["workflow_reason"] == "workflow_markers"
    assert result["is_fresh"] is True

    saved = json.loads(
        (research_dir / "overnight_state.json").read_text(encoding="utf-8")
    )
    assert saved["research_complete"] is True
    assert (
        saved["workflow_completion"]["completion_reason"]
        == "reconciled_from_existing_handoff_artifacts"
    )


def test_pm_entry_deadline_alert_emits_when_zero_submissions_after_ten():
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()

    exec_state = {
        "pm_candidates_loaded": 6,
        "pm_candidates_accepted_by_day_manager": 0,
        "pm_candidates_submitted": 0,
        "pm_gate_reason_counts": {
            "bad_day_posture_block:cautious_selective_longs": 4,
            "entry_authority_missing_score": 2,
        },
    }

    alert = agent._maybe_emit_pm_entry_deadline_alert(
        exec_state,
        now_local=datetime(2026, 3, 25, 10, 1, 0),
    )

    assert alert is not None
    assert alert["pm_candidates_loaded"] == 6
    assert exec_state["pm_entry_deadline_alert_emitted"] is True
    assert (
        alert["top_gate_reasons"][0]["reason"]
        == "bad_day_posture_block:cautious_selective_longs"
    )


def test_validate_pm_workflow_marks_dead_pm_authority(tmp_path, monkeypatch):
    plans_dir = Path(tmp_path) / "plans"
    logs_dir = Path(tmp_path) / "logs"
    plans_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)

    (plans_dir / "pm_plan_2026-04-17.json").write_text(
        json.dumps(
            {
                "plan_date": "unknown (fallback)",
                "signals": [],
                "entry_orders": [],
                "summary": {},
            }
        ),
        encoding="utf-8",
    )
    (plans_dir / "morning_game_plan_20260417.json").write_text(
        json.dumps({"signals": [{"symbol": "AAA", "score": 81.0}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", plans_dir)
    monkeypatch.setattr(autonomous_agent_mod, "LOG_DIR", logs_dir)
    monkeypatch.setattr(
        autonomous_agent_mod,
        "_pm_plan_date",
        lambda now=None: datetime(2026, 4, 17).date(),
    )

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent.watchlist_optimizer = SimpleNamespace(
        analyze_current_picks=lambda path: {"score": 80.0},
        update_watchlist=lambda *args, **kwargs: None,
    )
    agent.web_researcher = SimpleNamespace(openai_client=None)
    agent.data_fetcher = SimpleNamespace(get_missing_tickers=lambda tickers: [])

    result = agent.validate_pm_workflow()

    assert result["pm_plan_authority_dead"] is False
    assert result["pm_plan_authority_reason"] in {"", "eligible"}
    assert result["actions_taken"][0].startswith("Promoted 1 morning signals")


def test_validate_pm_workflow_promotes_stale_watch_only_pm_plan_from_morning_plan(
    tmp_path, monkeypatch
):
    plans_dir = Path(tmp_path) / "plans"
    logs_dir = Path(tmp_path) / "logs"
    plans_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)

    (plans_dir / "pm_plan_2026-04-17.json").write_text(
        json.dumps(
            {
                "plan_date": "2026-04-17",
                "generated_at": "2026-04-17T00:15:53",
                "signals": [
                    {"symbol": "OLD", "score": 61.0, "recommendation": "WATCH"}
                ],
            }
        ),
        encoding="utf-8",
    )
    (plans_dir / "morning_game_plan_20260417.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-04-17T08:29:14",
                "signals": [
                    {"symbol": "AAA", "score": 81.0, "recommendation": "STRONG BUY"},
                    {"symbol": "BBB", "score": 74.0, "recommendation": "BUY"},
                ],
            }
        ),
        encoding="utf-8",
    )

    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 17, 9, 0, 0, tzinfo=tz)

    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", plans_dir)
    monkeypatch.setattr(autonomous_agent_mod, "LOG_DIR", logs_dir)
    monkeypatch.setattr(autonomous_agent_mod, "datetime", _FixedDateTime)
    monkeypatch.setattr(
        autonomous_agent_mod,
        "_pm_plan_date",
        lambda now=None: _FixedDateTime(2026, 4, 17).date(),
    )

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent.watchlist_optimizer = SimpleNamespace(
        analyze_current_picks=lambda path: {"score": 80.0},
        update_watchlist=lambda *args, **kwargs: None,
    )
    agent.web_researcher = SimpleNamespace(openai_client=None)
    agent.data_fetcher = SimpleNamespace(get_missing_tickers=lambda tickers: [])

    result = agent.validate_pm_workflow()

    promoted_plan = json.loads(
        (plans_dir / "pm_plan_2026-04-17.json").read_text(encoding="utf-8")
    )

    assert [row["symbol"] for row in promoted_plan["signals"]] == ["AAA", "BBB"]
    assert promoted_plan["pm_plan_source"] == "morning_game_plan_20260417.json"
    assert any(
        msg.startswith("Promoted 2 morning signals") for msg in result["actions_taken"]
    )


def test_validate_pm_workflow_does_not_promote_stale_pm_plan_after_execution_started(
    tmp_path, monkeypatch
):
    plans_dir = Path(tmp_path) / "plans"
    logs_dir = Path(tmp_path) / "logs"
    plans_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)

    (plans_dir / "pm_plan_2026-04-17.json").write_text(
        json.dumps(
            {
                "plan_date": "2026-04-17",
                "generated_at": "2026-04-17T00:15:53",
                "signals": [
                    {"symbol": "OLD", "score": 61.0, "recommendation": "WATCH"}
                ],
            }
        ),
        encoding="utf-8",
    )
    (plans_dir / "morning_game_plan_20260417.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-04-17T08:29:14",
                "signals": [
                    {"symbol": "AAA", "score": 81.0, "recommendation": "STRONG BUY"}
                ],
            }
        ),
        encoding="utf-8",
    )
    (plans_dir / ".execution_state_20260417.json").write_text(
        json.dumps({"wave1_done": True}),
        encoding="utf-8",
    )

    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 17, 9, 0, 0, tzinfo=tz)

    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", plans_dir)
    monkeypatch.setattr(autonomous_agent_mod, "LOG_DIR", logs_dir)
    monkeypatch.setattr(autonomous_agent_mod, "datetime", _FixedDateTime)
    monkeypatch.setattr(
        autonomous_agent_mod,
        "_pm_plan_date",
        lambda now=None: _FixedDateTime(2026, 4, 17).date(),
    )

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent.watchlist_optimizer = SimpleNamespace(
        analyze_current_picks=lambda path: {"score": 80.0},
        update_watchlist=lambda *args, **kwargs: None,
    )
    agent.web_researcher = SimpleNamespace(openai_client=None)
    agent.data_fetcher = SimpleNamespace(get_missing_tickers=lambda tickers: [])

    result = agent.validate_pm_workflow()

    preserved_plan = json.loads(
        (plans_dir / "pm_plan_2026-04-17.json").read_text(encoding="utf-8")
    )

    assert [row["symbol"] for row in preserved_plan["signals"]] == ["OLD"]
    assert not any("Promoted" in msg for msg in result["actions_taken"])
    assert any("execution already started" in issue for issue in result["issues"])


def test_load_eod_feedback_rejects_stale_payload(monkeypatch, tmp_path):
    data_dir = Path(tmp_path) / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "eod_feedback_latest.json").write_text(
        json.dumps({"date": "2026-03-18", "win_rate": 0.55}),
        encoding="utf-8",
    )
    monkeypatch.setattr(autonomous_agent_mod, "PROJECT_DIR", Path(tmp_path))
    monkeypatch.setattr(
        autonomous_agent_mod,
        "get_market_now",
        lambda: datetime(2026, 3, 23, 10, 0, 0),
    )

    agent = OvernightResearchEngine.__new__(OvernightResearchEngine)
    agent.logger = _LoggerStub()

    assert agent._load_eod_feedback() == {}


def test_load_eod_feedback_accepts_latest_completed_session(monkeypatch, tmp_path):
    data_dir = Path(tmp_path) / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "eod_feedback_latest.json").write_text(
        json.dumps({"date": "2026-03-20", "win_rate": 0.55}),
        encoding="utf-8",
    )
    monkeypatch.setattr(autonomous_agent_mod, "PROJECT_DIR", Path(tmp_path))
    monkeypatch.setattr(
        autonomous_agent_mod,
        "get_market_now",
        lambda: datetime(2026, 3, 23, 10, 0, 0),
    )

    agent = OvernightResearchEngine.__new__(OvernightResearchEngine)
    agent.logger = _LoggerStub()

    feedback = agent._load_eod_feedback()

    assert feedback["date"] == "2026-03-20"


def test_premarket_keeps_current_day_zero_signal_plan_instead_of_stale_fallback(
    monkeypatch, tmp_path
):
    today = datetime(2026, 3, 24, 8, 0, 0)
    plans_dir = Path(tmp_path)
    (plans_dir / "pm_plan_2026-03-24.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-03-24T00:46:25",
                "signals": [],
                "entry_candidates": [],
                "resolved_regime": {"allow_new_longs": True},
                "positions": [{"symbol": "CAVA"}],
            }
        ),
        encoding="utf-8",
    )
    (plans_dir / "pm_plan_2026-03-23.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-03-23T00:46:25",
                "signals": [{"symbol": "STALE"}],
                "entry_candidates": [{"symbol": "STALE"}],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", plans_dir)
    monkeypatch.setitem(
        sys.modules,
        "autotrade.utils.youtube_readiness",
        SimpleNamespace(get_intelligence_context=lambda: {"available": False}),
    )
    monkeypatch.setitem(
        sys.modules,
        "autotrade.utils.research_retrigger",
        SimpleNamespace(attempt_retrigger_if_stale=lambda *args, **kwargs: False),
    )

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent.scheduler = SimpleNamespace(
        get_current_time=lambda: today,
        get_market_phase=lambda: SimpleNamespace(value="premarket"),
    )
    agent.research_freshness_cfg = SimpleNamespace(
        premarket_block_if_stale=True,
    )
    agent._youtube_context = {}
    agent._premarket_state = {}
    agent._check_research_freshness = lambda **kwargs: {
        "age_hours": 1.0,
        "age_stale": False,
        "workflow_complete": True,
        "workflow_reason": "complete",
        "warning": False,
        "max_age_hours": 12,
    }
    agent._get_strategy_pool_snapshot = lambda: {}
    agent._load_overnight_state = lambda: {}
    agent._strategy_pool_snapshot_changed = lambda *_args, **_kwargs: False

    result = agent._run_premarket_cycle(1)

    assert result == 60
    assert any(
        "Current-day plan has 0 entry signals" in msg
        for msg in agent.logger.messages["warning"]
    )
    assert any("No signals in plan" in msg for msg in agent.logger.messages["warning"])
    assert not any(
        "Using stale plan from 2026-03-23" in msg
        for msg in agent.logger.messages["warning"]
    )


def test_get_account_info_falls_back_to_cached_snapshot(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "alpaca_snapshot_today.json").write_text(
        json.dumps(
            {
                "account": {
                    "cash": "69403.11",
                    "buying_power": "326472.23",
                    "equity": "90531.35",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        autonomous_agent_mod,
        "__file__",
        str(tmp_path / "autotrade" / "core" / "autonomous_agent.py"),
    )

    generator = TomorrowsPlanGenerator.__new__(TomorrowsPlanGenerator)
    generator.logger = _LoggerStub()
    generator.alpaca_client = None

    account = TomorrowsPlanGenerator.get_account_info(generator)

    assert account["cash"] == 69403.11
    assert account["buying_power"] == 326472.23
    assert account["equity"] == 90531.35
    assert account["account_source"] == "alpaca_snapshot_today"


def test_premarket_uses_current_day_entry_orders_from_morning_plan(
    monkeypatch, tmp_path
):
    today = datetime(2026, 3, 24, 8, 0, 0)
    plans_dir = Path(tmp_path)
    (plans_dir / "morning_game_plan_20260324.json").write_text(
        json.dumps(
            {
                "entry_orders": [
                    {
                        "symbol": "BMNR",
                        "entry_price": 23.39,
                        "score": 81.25,
                        "stop_loss": 21.98,
                        "take_profit": 25.84,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (plans_dir / "pm_plan_2026-03-23.json").write_text(
        json.dumps(
            {
                "signals": [{"symbol": "STALE", "entry_price": 10.0, "score": 50.0}],
                "entry_candidates": [
                    {"symbol": "STALE", "entry_price": 10.0, "score": 50.0}
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", plans_dir)
    monkeypatch.setitem(
        sys.modules,
        "autotrade.utils.youtube_readiness",
        SimpleNamespace(get_intelligence_context=lambda: {"available": False}),
    )
    monkeypatch.setitem(
        sys.modules,
        "autotrade.utils.research_retrigger",
        SimpleNamespace(attempt_retrigger_if_stale=lambda *args, **kwargs: False),
    )

    captured = {}

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent.scheduler = SimpleNamespace(
        get_current_time=lambda: today,
        get_market_phase=lambda: SimpleNamespace(value="premarket"),
    )
    agent.research_freshness_cfg = SimpleNamespace(premarket_block_if_stale=True)
    agent._youtube_context = {}
    agent._premarket_state = {
        "last_plan_hash": "current_plan_hash",
        "last_gap_snapshot": {},
        "last_full_scan_time": today - timedelta(minutes=1),
    }
    agent._check_research_freshness = lambda **kwargs: {
        "age_hours": 1.0,
        "age_stale": False,
        "workflow_complete": True,
        "workflow_reason": "complete",
        "warning": False,
        "max_age_hours": 12,
    }
    agent._get_strategy_pool_snapshot = lambda: {}
    agent._load_overnight_state = lambda: {}
    agent._strategy_pool_snapshot_changed = lambda *_args, **_kwargs: False
    agent._build_premarket_actionable_pool = lambda plan, primary_signals: (
        captured.setdefault("primary_signals", list(primary_signals)),
        {
            "primary_count": len(primary_signals),
            "overflow_count": 0,
            "actionable_total": len(primary_signals),
            "from_actionable_top50": 0,
            "from_overflow_signals": 0,
            "from_full_watchlist": 0,
        },
    )
    agent._hash_payload = lambda payload: "current_plan_hash"
    agent._capture_premarket_gap_snapshot = lambda *_args, **_kwargs: {}

    result = agent._run_premarket_cycle(1)

    assert result == 60
    assert captured["primary_signals"][0]["symbol"] == "BMNR"
    assert not any(
        "Using stale plan from 2026-03-23" in msg
        for msg in agent.logger.messages["warning"]
    )
