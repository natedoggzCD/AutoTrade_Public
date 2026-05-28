import json
from unittest.mock import MagicMock
from datetime import datetime

import autotrade.core.autonomous_agent as autonomous_agent_mod
from autotrade.core import day_manager as day_manager_mod
from autotrade.core.autonomous_agent import AgentState
from autotrade.core.autonomous_agent import AutonomousAgent


def test_extract_runtime_error_signature_ignores_youtube_had_errors_false():
    agent = AutonomousAgent.__new__(AutonomousAgent)

    line = (
        '{"timestamp": "2026-03-27T04:02:11.619698Z", "level": "INFO", '
        '"logger": "autotrade.utils.youtube_readiness", '
        '"message": "[YOUTUBE][SCAN] \\"had_errors\\": false,"}'
    )

    assert agent._extract_runtime_error_signature(line) == ""


def test_track_runtime_operational_errors_skips_benign_youtube_scan_flags():
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.dry_run = True
    agent.self_heal_live_enabled = False
    agent.runtime_operational_error_threshold = 3
    agent.runtime_operational_error_ttl_seconds = 900
    agent.runtime_operational_error_tracker = {}
    agent.logger = MagicMock()
    agent._attempt_auto_fix = lambda error, context=None: True
    agent._request_cycle_restart = MagicMock()

    line = (
        '{"timestamp": "2026-03-27T04:02:11.619698Z", "level": "INFO", '
        '"logger": "autotrade.utils.youtube_readiness", '
        '"message": "[YOUTUBE][SCAN] \\"had_errors\\": false,"}'
    )

    triggered = agent._track_runtime_operational_errors([line, line, line, line])

    assert triggered == 0
    assert agent.runtime_operational_error_tracker == {}
    agent._request_cycle_restart.assert_not_called()


def test_attempt_auto_fix_requests_full_process_restart_for_day_manager_patch(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(autonomous_agent_mod, "LOG_DIR", tmp_path)

    class _LogCapture:
        def __init__(self):
            self.infos = []
            self.warnings = []

        def info(self, msg, *args, **kwargs):
            self.infos.append(str(msg) % args if args else str(msg))

        def warning(self, msg, *args, **kwargs):
            self.warnings.append(str(msg) % args if args else str(msg))

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LogCapture()
    agent.state = AgentState.IDLE
    agent.dry_run = True
    agent.self_heal_live_enabled = False
    agent.task_router = MagicMock(
        route=MagicMock(return_value=MagicMock(success=False, message="diagnostic failed"))
    )
    agent.auto_fix_pipeline = MagicMock(
        run=MagicMock(
            return_value={
                "success": True,
                "summary": "persisted_by_codeagent",
                "modified_files": ["autotrade/core/day_manager.py"],
            }
        )
    )
    agent.fix_attempts = {}
    agent.self_heal_root_tracker = {}
    agent.process_restart_requested = False
    agent.process_restart_reason = ""
    dm_instance = MagicMock()
    dm_instance._save_state = MagicMock()
    agent._day_manager_instance = dm_instance
    agent._day_manager_date = datetime.now().strftime("%Y-%m-%d")
    agent._day_manager_dry_run = True
    agent._day_manager_source_signature = "old"
    agent._append_jsonl = AutonomousAgent._append_jsonl.__get__(agent, AutonomousAgent)
    agent._log_self_heal_event = AutonomousAgent._log_self_heal_event.__get__(
        agent, AutonomousAgent
    )
    agent._log_code_fix_attempt = AutonomousAgent._log_code_fix_attempt.__get__(
        agent, AutonomousAgent
    )
    agent._try_common_fixes = lambda *args, **kwargs: False

    try:
        raise TypeError("slice indices must be integers or None or have an __index__ method")
    except TypeError as exc:
        fixed = agent._attempt_auto_fix(exc, "Day manager cycle")

    assert fixed is True
    assert agent.process_restart_requested is True
    assert (
        agent.process_restart_reason == "autofix_patch:autotrade/core/day_manager.py"
    )
    dm_instance._save_state.assert_called_once()


def test_attempt_auto_fix_blocks_compile_failed_target_without_pipeline(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(autonomous_agent_mod, "LOG_DIR", tmp_path)

    class _LogCapture:
        def __init__(self):
            self.infos = []
            self.warnings = []

        def info(self, msg, *args, **kwargs):
            self.infos.append(str(msg) % args if args else str(msg))

        def warning(self, msg, *args, **kwargs):
            self.warnings.append(str(msg) % args if args else str(msg))

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LogCapture()
    agent.state = AgentState.IDLE
    agent.dry_run = True
    agent.self_heal_live_enabled = False
    agent.task_router = MagicMock(
        route=MagicMock(return_value=MagicMock(success=False, message="diagnostic failed"))
    )
    agent.auto_fix_pipeline = MagicMock(run=MagicMock())
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

    bad_file = tmp_path / "broken_target.py"
    bad_file.write_text("def broken(\n    pass\n", encoding="utf-8")

    code = compile('raise AttributeError("boom")', str(bad_file), "exec")
    try:
        exec(code, {})
    except AttributeError as exc:
        fixed = agent._attempt_auto_fix(exc, "Day manager cycle")

    assert fixed is False
    agent.auto_fix_pipeline.run.assert_not_called()
    agent.task_router.route.assert_not_called()
    assert agent.fix_attempts == {}

    rows = [
        json.loads(line)
        for line in (tmp_path / "code_fixes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[-1]["summary"].startswith("auto_fix_blocked:compile_failed")
    assert rows[-1]["file"] == str(bad_file.resolve())
    assert rows[-1]["line"] == 1
    assert rows[-1]["failure_category"] == "compile_gate_failed"


def test_expand_attribute_error_anchor_targets_class_definition(tmp_path):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    source = tmp_path / "attr_anchor_case.py"
    source.write_text(
        "\n".join(
            [
                "class DemoClass:",
                "    def existing(self):",
                "        return 1",
                "",
                "def run():",
                "    return DemoClass().missing_method()",
                "",
                "run()",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    compiled = compile(source.read_text(encoding="utf-8"), str(source), "exec")
    try:
        exec(compiled, {})
    except AttributeError as exc:
        file_hint, line_hint, meta = agent._expand_attribute_error_anchor(
            exc, str(source), 6
        )

    assert file_hint == str(source.resolve())
    assert line_hint == 1
    assert meta["attribute_error_class"] == "DemoClass"
    assert meta["attribute_error_member"] == "missing_method"


def test_attempt_auto_fix_signature_retry_cap_logs_exhaustion(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(autonomous_agent_mod, "LOG_DIR", tmp_path)
    monkeypatch.setenv("SELF_HEAL_SIGNATURE_RETRY_CAP", "2")
    monkeypatch.setenv("SELF_HEAL_SIGNATURE_COOLDOWN_SECONDS", "3600")

    class _LogCapture:
        def __init__(self):
            self.infos = []
            self.warnings = []

        def info(self, msg, *args, **kwargs):
            self.infos.append(str(msg) % args if args else str(msg))

        def warning(self, msg, *args, **kwargs):
            self.warnings.append(str(msg) % args if args else str(msg))

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LogCapture()
    agent.state = AgentState.IDLE
    agent.dry_run = True
    agent.self_heal_live_enabled = False
    agent.task_router = MagicMock(
        route=MagicMock(return_value=MagicMock(success=False, message="no tool calls produced"))
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

    target_file = tmp_path / "valid_target.py"
    target_file.write_text("def ok():\n    return 1\n", encoding="utf-8")

    code = compile('raise RuntimeError("repeating runtime failure")', str(target_file), "exec")
    for _ in range(3):
        try:
            exec(code, {})
        except RuntimeError as exc:
            agent._attempt_auto_fix(exc, "runtime_operational_error")

    rows = [
        json.loads(line)
        for line in (tmp_path / "code_fixes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[-1]["summary"] == "retries_exhausted_for_signature"
    assert rows[-1]["failure_category"] == "retries_exhausted_for_signature"
    assert rows[-1]["signature_retry_exhausted"] is True


def test_get_day_manager_reloads_module_when_source_changes_with_empty_cache(
    monkeypatch, tmp_path
):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = MagicMock()
    agent._day_manager_instance = None
    agent._day_manager_date = None
    agent._day_manager_dry_run = None
    agent._day_manager_source_signature = "stale-signature"
    agent.decision_claw = None

    source_file = tmp_path / "day_manager_runtime.py"
    source_file.write_text("# runtime shim\n", encoding="utf-8")

    class _FakeDayManager:
        def __init__(self, dry_run):
            self.dry_run = dry_run

    monkeypatch.setattr(day_manager_mod, "__file__", str(source_file))
    monkeypatch.setattr(day_manager_mod, "DayManager", _FakeDayManager)

    reload_calls = []

    def _fake_reload(module):
        reload_calls.append(module)
        return module

    monkeypatch.setattr("importlib.reload", _fake_reload)

    dm = agent._get_day_manager(dry_run=True)

    assert isinstance(dm, _FakeDayManager)
    assert reload_calls
