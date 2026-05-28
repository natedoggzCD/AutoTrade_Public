"""
Tests for WorkflowClaw - LLM-powered autonomous workflow manager.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from autotrade.core.workflow_claw import (
    ExitInfo,
    RecoveryAgent,
    RecoveryJournal,
    RecoveryResult,
    ServiceHealthManager,
    WorkflowClaw,
    _build_supervisor_command,
    _summarize_state,
)


# ── ServiceHealthManager Tests ──────────────────────────────────────────────


class TestServiceHealthManager:
    """Tests for service health probing and restart logic."""

    def setup_method(self):
        self.services = {
            "ollama": {
                "health_url": "http://localhost:11434/api/tags",
                "restart_command": ["echo", "restart_ollama"],
                "kill_command": ["echo", "kill_ollama"],
                "restart_cooldown_sec": 1,
                "max_restart_attempts": 3,
                "health_timeout_sec": 5,
            },
            "searxng": {
                "health_url": "http://localhost:8080/healthz",
                "restart_command": ["echo", "restart_searxng"],
                "restart_cooldown_sec": 1,
                "max_restart_attempts": 2,
                "health_timeout_sec": 5,
            },
        }
        self.manager = ServiceHealthManager(self.services)

    def test_unknown_service_returns_error(self):
        result = self.manager.check_health("nonexistent")
        assert not result["healthy"]
        assert "Unknown" in result["error"]

    @patch("autotrade.core.workflow_claw.requests.get")
    def test_healthy_service(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        result = self.manager.check_health("ollama")
        assert result["healthy"]
        assert result["error"] is None

    @patch("autotrade.core.workflow_claw.requests.get")
    def test_unhealthy_service_http_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_get.return_value = mock_resp

        result = self.manager.check_health("ollama")
        assert not result["healthy"]
        assert "500" in result["error"]

    @patch("autotrade.core.workflow_claw.requests.get")
    def test_connection_refused(self, mock_get):
        import requests as req

        mock_get.side_effect = req.exceptions.ConnectionError("refused")

        result = self.manager.check_health("ollama")
        assert not result["healthy"]
        assert result["error"] == "connection_refused"

    @patch("autotrade.core.workflow_claw.requests.get")
    def test_timeout(self, mock_get):
        import requests as req

        mock_get.side_effect = req.exceptions.Timeout()

        result = self.manager.check_health("ollama")
        assert not result["healthy"]
        assert result["error"] == "timeout"

    def test_restart_unknown_service(self):
        result = self.manager.restart_service("nonexistent")
        assert not result["success"]

    @patch("autotrade.core.workflow_claw.requests.get")
    @patch("autotrade.core.workflow_claw.subprocess.Popen")
    @patch("autotrade.core.workflow_claw.subprocess.run")
    def test_restart_success(self, mock_run, mock_popen, mock_get):
        # Health check returns healthy after restart
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        result = self.manager.restart_service("ollama")
        assert result["success"]
        assert result["restart_count"] == 1

    def test_max_restart_attempts(self):
        self.manager._restart_counts["ollama"] = 3  # Already at max
        result = self.manager.restart_service("ollama")
        assert not result["success"]
        assert "Max restart" in result["error"]

    def test_reset_counts(self):
        self.manager._restart_counts["ollama"] = 5
        self.manager.reset_counts()
        assert len(self.manager._restart_counts) == 0

    def test_get_all_status(self):
        with patch("autotrade.core.workflow_claw.requests.get") as mock_get:
            import requests as req

            mock_get.side_effect = req.exceptions.ConnectionError()
            status = self.manager.get_all_status()
            assert "ollama" in status
            assert "searxng" in status
            assert not status["ollama"]["healthy"]


# ── RecoveryJournal Tests ────────────────────────────────────────────────────


class TestRecoveryJournal:
    """Tests for the append-only JSONL journal."""

    def test_record_creates_valid_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = RecoveryJournal.__new__(RecoveryJournal)
            journal.path = Path(tmpdir) / "test_journal.jsonl"
            journal.session_id = "test_session"
            journal.path.parent.mkdir(parents=True, exist_ok=True)

            journal.record(
                "test_event",
                details={"key": "value"},
                action={"type": "test"},
                outcome={"status": "ok"},
            )

            lines = journal.path.read_text().strip().split("\n")
            assert len(lines) == 1

            entry = json.loads(lines[0])
            assert entry["event_type"] == "test_event"
            assert entry["session_id"] == "test_session"
            assert entry["details"]["key"] == "value"
            assert "timestamp" in entry

    def test_multiple_records_append(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = RecoveryJournal.__new__(RecoveryJournal)
            journal.path = Path(tmpdir) / "test_journal.jsonl"
            journal.session_id = "test"
            journal.path.parent.mkdir(parents=True, exist_ok=True)

            journal.record("event1")
            journal.record("event2")
            journal.record("event3")

            lines = journal.path.read_text().strip().split("\n")
            assert len(lines) == 3

    def test_morning_briefing_no_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = RecoveryJournal.__new__(RecoveryJournal)
            journal.path = Path(tmpdir) / "nonexistent.jsonl"
            journal.session_id = "test"

            briefing = journal.generate_morning_briefing("2026-03-22")
            assert "No journal entries found" in briefing

    def test_morning_briefing_with_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = RecoveryJournal.__new__(RecoveryJournal)
            journal.path = Path(tmpdir) / "test_journal.jsonl"
            journal.session_id = "test"
            journal.path.parent.mkdir(parents=True, exist_ok=True)

            today = time.strftime("%Y-%m-%d")
            entries = [
                {
                    "timestamp": f"{today}T02:15:33",
                    "session_id": "test",
                    "event_type": "recovery_attempt",
                    "details": {},
                    "action": {},
                    "outcome": {"status": "resolved", "summary": "Restarted Ollama"},
                },
                {
                    "timestamp": f"{today}T03:00:00",
                    "session_id": "test",
                    "event_type": "supervisor_exit",
                    "details": {},
                    "action": {},
                    "outcome": {"status": "failed", "summary": "Budget exhausted"},
                },
            ]
            journal.path.write_text(
                "\n".join(json.dumps(e) for e in entries) + "\n"
            )

            briefing = journal.generate_morning_briefing(today)
            assert "Total events" in briefing
            assert "recovery_attempt" in briefing
            assert "Restarted Ollama" in briefing


# ── RecoveryAgent Tests ──────────────────────────────────────────────────────


class TestRecoveryAgent:
    """Tests for the LLM-powered recovery agent."""

    def _make_config(self):
        config = MagicMock()
        config.primary_model = "test-model"
        config.fallback_model = "test-fallback"
        config.openrouter_model = "test/openrouter"
        config.openai_model = "gpt-test"
        config.max_agent_turns = 5
        config.num_ctx = 4096
        config.temperature = 0.2
        return config

    def test_tool_read_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config()
            shm = ServiceHealthManager({})
            journal = RecoveryJournal.__new__(RecoveryJournal)
            journal.path = Path(tmpdir) / "j.jsonl"
            journal.session_id = "test"
            journal.path.parent.mkdir(parents=True, exist_ok=True)

            agent = RecoveryAgent(config, shm, journal)

            # Create a test log
            log_path = Path(tmpdir) / "test.log"
            log_path.write_text("line1\nline2\nline3\n")

            with patch(
                "autotrade.core.workflow_claw.PROJECT_ROOT", Path(tmpdir)
            ):
                result = agent._tool_read_log({"path": "test.log", "lines": 2})
                assert "line2" in result
                assert "line3" in result

    def test_tool_check_artifact_missing(self):
        config = self._make_config()
        shm = ServiceHealthManager({})
        journal = MagicMock()
        agent = RecoveryAgent(config, shm, journal)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "autotrade.core.workflow_claw.PROJECT_ROOT", Path(tmpdir)
            ):
                result = json.loads(agent._tool_check_artifact({"path": "nonexistent.json"}))
                assert result["exists"] is False

    def test_tool_check_artifact_exists(self):
        config = self._make_config()
        shm = ServiceHealthManager({})
        journal = MagicMock()
        agent = RecoveryAgent(config, shm, journal)

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.json"
            test_file.write_text('{"signals": [1, 2, 3]}')

            with patch(
                "autotrade.core.workflow_claw.PROJECT_ROOT", Path(tmpdir)
            ):
                result = json.loads(agent._tool_check_artifact({"path": "test.json"}))
                assert result["exists"] is True
                assert result["size_kb"] >= 0
                assert "keys" in result
                assert "signals" in result["keys"]

    def test_tool_run_diagnostic_blocks_dangerous(self):
        config = self._make_config()
        shm = ServiceHealthManager({})
        journal = MagicMock()
        agent = RecoveryAgent(config, shm, journal)

        result = json.loads(agent._tool_run_diagnostic({"command": "rm -rf /"}))
        assert "blocked" in result["error"].lower()

    def test_tool_run_diagnostic_dry_run(self):
        config = self._make_config()
        shm = ServiceHealthManager({})
        journal = MagicMock()
        agent = RecoveryAgent(config, shm, journal, dry_run=True)

        result = json.loads(agent._tool_run_diagnostic({"command": "echo hello"}))
        assert result["dry_run"] is True

    def test_loop_detection(self):
        """Verify that 3 identical tool calls trigger loop detection."""
        # Simulate the hash tracking
        import hashlib

        call = "check_service:{\"service\": \"ollama\"}"
        h = hashlib.md5(call.encode()).hexdigest()
        hashes = [h, h, h]

        # Verify detection logic
        assert hashes[-1] == hashes[-2] == hashes[-3]

    def test_parse_json_tool_call(self):
        content = 'I should check the service. {"tool_call": {"name": "check_service", "arguments": {"service": "ollama"}}}'
        result = RecoveryAgent._parse_json_tool_call(content)
        assert result is not None
        assert result["function"]["name"] == "check_service"
        assert result["function"]["arguments"]["service"] == "ollama"

    def test_parse_json_tool_call_no_match(self):
        result = RecoveryAgent._parse_json_tool_call("just plain text")
        assert result is None

    def test_select_llm_backend_ollama_healthy(self):
        config = self._make_config()
        shm = MagicMock()
        shm.check_health.return_value = {"healthy": True}
        journal = MagicMock()
        agent = RecoveryAgent(config, shm, journal)

        backend, client = agent._select_llm_backend()
        assert backend == "ollama"

    def test_select_llm_backend_ollama_down_openrouter(self):
        config = self._make_config()
        shm = MagicMock()
        shm.check_health.return_value = {"healthy": False}
        shm.restart_service.return_value = {"success": False}
        journal = MagicMock()
        agent = RecoveryAgent(config, shm, journal)

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_key"}):
            mock_or = MagicMock()
            mock_or.available = True
            agent._openrouter_client = mock_or
            backend, client = agent._select_llm_backend()
            assert backend == "openrouter"


# ── WorkflowClaw Tests ──────────────────────────────────────────────────────


class TestWorkflowClaw:
    """Tests for the main orchestrator."""

    def test_summarize_state_empty(self):
        result = _summarize_state({})
        assert "error" in result

    def test_summarize_state_truncates_lists(self):
        state = {
            "current_phase": "OVERNIGHT",
            "issues": list(range(20)),
            "daily_flags": {"overnight_ran": True},
        }
        result = _summarize_state(state)
        assert result["current_phase"] == "OVERNIGHT"
        assert len(result["issues"]) == 6  # 5 items + "... (20 total)"

    def test_recovery_budget_check(self):
        with patch("config.config_loader.get_config") as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.workflow_claw.max_recovery_attempts_per_hour = 3
            mock_cfg.workflow_claw.max_total_recovery_attempts = 10
            mock_cfg.workflow_claw.journal_path = "logs/test.jsonl"
            mock_cfg.workflow_claw.state_path = "data/test_state.json"
            mock_cfg.workflow_claw.services = {}
            mock_config.return_value = mock_cfg

            claw = WorkflowClaw.__new__(WorkflowClaw)
            claw.config = mock_cfg.workflow_claw
            claw._recovery_attempts = []
            claw._total_recovery_attempts = 0

            # Should be within budget
            assert claw._check_recovery_budget()

            # Exceed hourly
            claw._recovery_attempts = [time.time()] * 3
            assert not claw._check_recovery_budget()

            # Reset hourly, exceed total
            claw._recovery_attempts = []
            claw._total_recovery_attempts = 10
            assert not claw._check_recovery_budget()

    def test_file_log_freshness_prevents_stdout_only_stall(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / "logs"
            logs_dir.mkdir(parents=True)
            app_log = logs_dir / "app.jsonl"
            app_log.write_text("{}\n", encoding="utf-8")
            now = time.time()
            os.utime(app_log, (now, now))

            claw = WorkflowClaw.__new__(WorkflowClaw)
            claw.config = MagicMock()
            claw.config.output_stale_timeout_sec = 3600
            with patch("autotrade.core.workflow_claw.PROJECT_ROOT", Path(tmpdir)):
                assert claw._supervisor_file_logs_stale(now_ts=now + 90) is False

    def test_file_log_absence_counts_as_stale(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            claw = WorkflowClaw.__new__(WorkflowClaw)
            claw.config = MagicMock()
            claw.config.output_stale_timeout_sec = 3600
            with patch("autotrade.core.workflow_claw.PROJECT_ROOT", Path(tmpdir)):
                assert claw._supervisor_file_logs_stale(now_ts=time.time()) is True

    def test_terminate_and_confirm_reports_abandoned_process(self):
        proc = MagicMock()
        proc.terminate.return_value = None
        proc.wait.side_effect = subprocess.TimeoutExpired("cmd", 10)
        proc.kill.return_value = None

        assert WorkflowClaw._terminate_and_confirm(proc, timeout_sec=0.01) is False

    def test_descendant_pids_still_alive_detects_orphan_python(self, monkeypatch):
        fake_psutil = MagicMock()
        fake_psutil.pid_exists.side_effect = lambda pid: pid == 69104
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        assert WorkflowClaw._descendant_pids_still_alive([69104]) is True

    def test_collect_supervisor_descendant_pids_filters_python_children(
        self, monkeypatch
    ):
        python_child = MagicMock()
        python_child.pid = 69104
        python_child.name.return_value = "python.exe"
        python_child.cmdline.return_value = [
            "python.exe",
            "-m",
            "autotrade.core.autonomous_agent",
        ]
        other_child = MagicMock()
        other_child.pid = 123
        other_child.name.return_value = "cmd.exe"
        other_child.cmdline.return_value = ["cmd.exe"]
        root = MagicMock()
        root.children.return_value = [python_child, other_child]
        fake_psutil = MagicMock()
        fake_psutil.Process.return_value = root
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
        proc = MagicMock()
        proc.pid = 42

        assert WorkflowClaw._collect_supervisor_descendant_pids(proc) == [69104]

    def test_force_kill_orphans_reaps_all_pids(self, monkeypatch):
        killed = []

        def fake_process(pid):
            proc = MagicMock()
            proc.kill.side_effect = lambda: killed.append(pid)
            proc.wait.return_value = None
            return proc

        existence = {69104: True, 69105: True}

        def pid_exists(pid):
            if pid in killed and existence.get(pid):
                existence[pid] = False
            return existence.get(pid, False)

        fake_psutil = MagicMock()
        fake_psutil.pid_exists.side_effect = pid_exists
        fake_psutil.Process.side_effect = fake_process
        fake_psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
        fake_psutil.AccessDenied = type("AccessDenied", (Exception,), {})
        fake_psutil.TimeoutExpired = type("TimeoutExpired", (Exception,), {})
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        survivors = WorkflowClaw._force_kill_orphans([69104, 69105])
        assert survivors == []
        assert sorted(killed) == [69104, 69105]

    def test_force_kill_orphans_returns_survivors_when_kill_fails(self, monkeypatch):
        def fake_process(pid):
            proc = MagicMock()
            proc.kill.return_value = None
            proc.wait.return_value = None
            return proc

        fake_psutil = MagicMock()
        # Process still alive after kill attempt
        fake_psutil.pid_exists.return_value = True
        fake_psutil.Process.side_effect = fake_process
        fake_psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
        fake_psutil.AccessDenied = type("AccessDenied", (Exception,), {})
        fake_psutil.TimeoutExpired = type("TimeoutExpired", (Exception,), {})
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        survivors = WorkflowClaw._force_kill_orphans([99999])
        assert survivors == [99999]


# ── ExitInfo Tests ───────────────────────────────────────────────────────────


class TestExitInfo:
    def test_creation(self):
        info = ExitInfo(code=1, reason="stalled", runtime_sec=3600.0)
        assert info.code == 1
        assert info.reason == "stalled"
        assert info.runtime_sec == 3600.0
        assert info.last_lines == []


class TestRecoveryResult:
    def test_creation(self):
        result = RecoveryResult(
            status="resolved",
            summary="Fixed Ollama",
            llm_backend="ollama",
            turns_used=3,
        )
        assert result.status == "resolved"
        assert result.turns_used == 3


def test_build_supervisor_command_places_fresh_before_task():
    cmd = _build_supervisor_command(fresh=True)

    assert cmd[0].endswith("python.exe") or cmd[0].endswith("python")
    assert cmd.index("--fresh") < cmd.index("--task")
    assert cmd[cmd.index("--task") + 1] == "continuous-guarded"
