import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from autotrade.core import autonomous_agent as autonomous_agent_mod
from autotrade.core.autonomous_agent import AutonomousAgent
from autotrade.utils.research_freshness import check_research_freshness


ET = ZoneInfo("America/New_York")


class _LoggerStub:
    def __init__(self):
        self.messages = {"info": [], "warning": [], "error": [], "debug": []}

    def info(self, message, *args):
        self.messages["info"].append(message % args if args else message)

    def warning(self, message, *args):
        self.messages["warning"].append(message % args if args else message)

    def error(self, message, *args):
        self.messages["error"].append(message % args if args else message)

    def debug(self, message, *args):
        self.messages["debug"].append(message % args if args else message)


def _write_common_handoff_artifacts(base: Path, target_date: str) -> dict:
    reports_dir = base / "reports"
    plans_dir = base / "plans"
    logs_dir = base / "logs"
    research_dir = base / "research"
    reports_dir.mkdir()
    plans_dir.mkdir()
    logs_dir.mkdir()
    research_dir.mkdir()

    compact = target_date.replace("-", "")
    plan_path = plans_dir / f"morning_game_plan_{compact}.json"
    plan_path.write_text(
        json.dumps(
            {
                "date": target_date,
                "full_watchlist": [{"symbol": "AAA", "confidence": 82.0}],
                "signals": [{"symbol": "AAA", "confidence": 82.0}],
                "buy_signals": [{"symbol": "AAA", "confidence": 82.0}],
            }
        ),
        encoding="utf-8",
    )
    (plans_dir / "overnight_research_bundle_latest.json").write_text(
        json.dumps({"trade_date": target_date}),
        encoding="utf-8",
    )
    (logs_dir / f"signals_{target_date}.json").write_text(
        json.dumps({"date": target_date, "signals": [{"symbol": "AAA"}]}),
        encoding="utf-8",
    )
    (reports_dir / f"daily_lessons_{target_date}.md").write_text(
        "# lessons",
        encoding="utf-8",
    )
    (reports_dir / f"post_market_lesson_{target_date}.json").write_text(
        json.dumps({"date": target_date}),
        encoding="utf-8",
    )
    return {
        "reports_dir": reports_dir,
        "plans_dir": plans_dir,
        "logs_dir": logs_dir,
        "research_dir": research_dir,
        "plan_path": plan_path,
    }


def test_check_research_freshness_blocks_marker_only_completion_when_artifact_audit_degraded(
    monkeypatch, tmp_path: Path
):
    target_date = "2026-04-21"
    paths = _write_common_handoff_artifacts(tmp_path, target_date)

    monkeypatch.setattr(autonomous_agent_mod, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", paths["plans_dir"])
    monkeypatch.setattr(autonomous_agent_mod, "LOG_DIR", paths["logs_dir"])

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent.overnight_secondary_cfg = SimpleNamespace(
        top_pick_research_enabled=True,
        historical_revisit_enabled=False,
        signal_mining_enabled=False,
        run_when_research_complete=True,
    )
    agent._persist_research_artifact_bundle_from_latest_plan = (
        lambda now_et=None: paths["plans_dir"] / "overnight_research_bundle_latest.json"
    )

    state = {
        "updated_at": "2026-04-20T22:30:00-04:00",
        "research_complete": True,
        "watchlist": [{"symbol": "AAA", "confidence": 82.0}],
        "target_trade_date": target_date,
        "workflow_completion": {
            "watchlist_selected": True,
            "game_plan_generated": True,
            "target_trade_date": target_date,
        },
        "secondary_research": {"last_result_by_job": {}},
    }
    audit = agent._build_overnight_artifact_audit(state, trigger_reason="test")
    assert audit["artifact_audit_status"] == "degraded"
    assert "top_pick_deep_research:not_run" in audit["missing_artifacts"]

    state_path = paths["research_dir"] / "overnight_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    result = check_research_freshness(
        state_path=state_path,
        now_et=datetime(2026, 4, 21, 7, 0, tzinfo=ET),
        persist_metadata=False,
    )

    assert result["workflow_complete"] is False
    assert result["workflow_reason"] == "artifact_audit_degraded"
    assert result["is_fresh"] is False


def test_check_research_freshness_accepts_completion_when_artifact_audit_ok(
    monkeypatch, tmp_path: Path
):
    target_date = "2026-04-21"
    paths = _write_common_handoff_artifacts(tmp_path, target_date)
    top_pick_path = (
        paths["reports_dir"] / "top_pick_deep_research_20260421_000001.json"
    )
    top_pick_path.write_text(
        json.dumps({"symbols": [{"symbol": "AAA"}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(autonomous_agent_mod, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", paths["plans_dir"])
    monkeypatch.setattr(autonomous_agent_mod, "LOG_DIR", paths["logs_dir"])

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent.overnight_secondary_cfg = SimpleNamespace(
        top_pick_research_enabled=True,
        historical_revisit_enabled=False,
        signal_mining_enabled=False,
        run_when_research_complete=True,
    )
    agent._persist_research_artifact_bundle_from_latest_plan = (
        lambda now_et=None: paths["plans_dir"] / "overnight_research_bundle_latest.json"
    )

    state = {
        "updated_at": "2026-04-20T22:30:00-04:00",
        "research_complete": True,
        "watchlist": [{"symbol": "AAA", "confidence": 82.0}],
        "target_trade_date": target_date,
        "workflow_completion": {
            "watchlist_selected": True,
            "game_plan_generated": True,
            "target_trade_date": target_date,
        },
        "secondary_research": {
            "last_result_by_job": {
                "top_pick_research": {
                    "result": {
                        "success": True,
                        "artifact_path": str(top_pick_path),
                    }
                }
            }
        },
    }
    audit = agent._build_overnight_artifact_audit(state, trigger_reason="test")
    assert audit["artifact_audit_status"] == "ok"

    state_path = paths["research_dir"] / "overnight_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    result = check_research_freshness(
        state_path=state_path,
        now_et=datetime(2026, 4, 21, 7, 0, tzinfo=ET),
        persist_metadata=False,
    )

    assert result["workflow_complete"] is True
    assert result["is_fresh"] is True
