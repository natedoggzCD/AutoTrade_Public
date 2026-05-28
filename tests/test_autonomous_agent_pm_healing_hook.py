import sys
import json
from types import SimpleNamespace

import autotrade.core.autonomous_agent as autonomous_agent_mod
from autotrade.core.autonomous_agent import AgentState, AutonomousAgent


class _LogCapture:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.errors = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(str(msg) % args if args else str(msg))

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(str(msg) % args if args else str(msg))

    def error(self, msg, *args, **kwargs):
        self.errors.append(str(msg) % args if args else str(msg))


def test_run_pm_workflow_calls_healing_hook_on_exception(monkeypatch):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LogCapture()
    agent.state = AgentState.IDLE
    agent.error_diagnoser = SimpleNamespace(
        diagnose_error=lambda exc, ctx: {"diagnosis": {"issue": f"{ctx}: {exc}"}}
    )
    agent._ensure_ollama_for_phase = lambda phase: None
    agent._record_ollama_result = lambda *args, **kwargs: None
    agent._check_ollama_health = lambda require_models=False: {"ok": True}

    healing_calls = []
    agent._handle_error_with_healing = lambda exc, ctx: healing_calls.append(
        (type(exc).__name__, ctx)
    )

    class _PMWorkflow:
        def __init__(self, dry_run):
            self.dry_run = dry_run

        def run(self, eod_results=None):
            raise SyntaxError("expected an indented block")

    class _EODReview:
        def __init__(self, dry_run):
            self.dry_run = dry_run

        def run(self):
            return {"ok": True}

    monkeypatch.setitem(
        sys.modules,
        "autotrade.execution.post_market_workflow",
        SimpleNamespace(PostMarketWorkflow=_PMWorkflow),
    )
    monkeypatch.setitem(
        sys.modules,
        "autotrade.core.eod_review",
        SimpleNamespace(EODReview=_EODReview),
    )

    result = agent.run_pm_workflow(dry_run=True, include_review=False)

    assert "error" in result
    assert healing_calls == [("SyntaxError", "PM workflow execution")]
    assert agent.state == AgentState.IDLE


def test_attempt_auto_fix_logs_no_anchor_without_burning_attempts(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(autonomous_agent_mod, "LOG_DIR", tmp_path)

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LogCapture()
    agent.state = AgentState.IDLE
    agent.dry_run = True
    agent.self_heal_live_enabled = False
    agent.task_router = SimpleNamespace(
        route=lambda task: SimpleNamespace(success=False, message="diagnostic failed")
    )
    agent.auto_fix_pipeline = None
    agent.fix_attempts = {}
    agent.self_heal_root_tracker = {}

    agent._append_jsonl = AutonomousAgent._append_jsonl.__get__(agent, AutonomousAgent)
    agent._log_self_heal_event = AutonomousAgent._log_self_heal_event.__get__(
        agent, AutonomousAgent
    )
    agent._log_code_fix_attempt = AutonomousAgent._log_code_fix_attempt.__get__(
        agent, AutonomousAgent
    )
    agent._try_common_fixes = lambda *args, **kwargs: False
    agent._request_day_manager_process_restart = lambda *args, **kwargs: None

    exc = RuntimeError("synthetic runtime failure")

    assert agent._attempt_auto_fix(exc, "Day manager cycle") is False

    app_rows = [
        json.loads(line)
        for line in (tmp_path / "app.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    code_fix_rows = [
        json.loads(line)
        for line in (tmp_path / "code_fixes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert any(row["event"] == "self_heal_blocked_error" for row in app_rows)
    assert not any(row["event"] == "self_heal_runtime_degraded" for row in app_rows)
    assert agent.fix_attempts == {}
    assert agent.self_heal_root_tracker
    assert code_fix_rows[-1]["success"] is False
    assert code_fix_rows[-1]["file"] == "no_anchor"
    assert code_fix_rows[-1]["summary"] == "auto_fix_blocked:no_anchor"


def test_start_momentum_scanner_daemon_records_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(autonomous_agent_mod, "LOG_DIR", tmp_path)

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LogCapture()
    agent.config = SimpleNamespace(momentum_scanner=SimpleNamespace(enabled=True))
    agent._momentum_scanner_thread = None
    agent._momentum_scanner_runtime = {
        "enabled": True,
        "thread_name": "",
        "started_at": None,
        "last_heartbeat": None,
        "last_error": None,
        "alive": False,
    }
    agent._append_jsonl = AutonomousAgent._append_jsonl.__get__(agent, AutonomousAgent)

    class _FakeThread:
        def __init__(self, target, daemon, name):
            self._target = target
            self.daemon = daemon
            self.name = name
            self._alive = False

        def start(self):
            self._alive = True
            self._target()
            self._alive = False

        def is_alive(self):
            return self._alive

    class _BrokenScanner:
        def __init__(self, config):
            self.config = config

        def run_forever(self):
            raise RuntimeError("scanner boom")

    monkeypatch.setattr(autonomous_agent_mod.threading, "Thread", _FakeThread)
    import autotrade.utils.momentum_scanner as momentum_scanner_mod

    monkeypatch.setattr(momentum_scanner_mod, "MomentumScanner", _BrokenScanner)

    AutonomousAgent._start_momentum_scanner_daemon(agent)

    rows = [
        json.loads(line)
        for line in (tmp_path / "app.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row["event"] == "momentum_scanner_thread_started" for row in rows)
    assert any(row["event"] == "momentum_scanner_thread_failed" for row in rows)
    assert "RuntimeError: scanner boom" == agent._momentum_scanner_runtime["last_error"]
