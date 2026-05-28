import json
from datetime import datetime
from types import SimpleNamespace

import autotrade.core.master_supervisor as master_supervisor_mod
from autotrade.core.master_supervisor import DetectedIssue, IssueType, MasterSupervisor
from autotrade.utils.market_time import get_pm_plan_date


def test_validate_task_results_uses_adjusted_plan_actionable_pool_when_signals_missing(
    tmp_path, monkeypatch
):
    project_root = tmp_path / "repo"
    plans_dir = project_root / "plans"
    logs_dir = project_root / "logs"
    module_dir = project_root / "autotrade" / "core"
    plans_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    module_dir.mkdir(parents=True)
    fake_module_path = module_dir / "master_supervisor.py"
    fake_module_path.write_text("# test shim\n", encoding="utf-8")

    today = get_pm_plan_date(datetime.now()).strftime("%Y%m%d")
    adjusted_plan = {
        "signals": [],
        "actionable_top50": [{"ticker": "AAA", "confidence": 81.0}],
        "full_watchlist": [{"ticker": "BBB", "confidence": 70.0}],
    }
    (plans_dir / f"adjusted_plan_{today}_0824.json").write_text(
        json.dumps(adjusted_plan), encoding="utf-8"
    )

    monkeypatch.setattr(
        master_supervisor_mod, "__file__", str(fake_module_path), raising=False
    )

    supervisor = MasterSupervisor.__new__(MasterSupervisor)
    validation = supervisor._validate_task_results("continuous")

    assert validation["success"] is False
    assert validation["problem"] == "signals_exist_but_no_execution"
    assert any("actionable_top50" in check for check in validation["checks"])


def test_validate_task_results_detects_compact_trade_decision_log_as_execution(
    tmp_path, monkeypatch
):
    project_root = tmp_path / "repo"
    logs_dir = project_root / "logs"
    module_dir = project_root / "autotrade" / "core"
    logs_dir.mkdir(parents=True)
    module_dir.mkdir(parents=True)
    fake_module_path = module_dir / "master_supervisor.py"
    fake_module_path.write_text("# test shim\n", encoding="utf-8")

    today_dashed = datetime.now().strftime("%Y-%m-%d")
    today_compact = datetime.now().strftime("%Y%m%d")
    (logs_dir / f"signals_{today_dashed}.json").write_text(
        json.dumps([{"ticker": "SQQQ", "confidence": 88.0}]), encoding="utf-8"
    )
    (logs_dir / f"trade_decisions_{today_compact}.json").write_text(
        json.dumps([{"symbol": "SQQQ", "actually_executed": True}]),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        master_supervisor_mod, "__file__", str(fake_module_path), raising=False
    )

    supervisor = MasterSupervisor.__new__(MasterSupervisor)
    validation = supervisor._validate_task_results("continuous")

    assert validation["success"] is True
    assert "problem" not in validation or validation["problem"] in (None, "")
    assert any("Execution artifacts exist" in check for check in validation["checks"])


def test_collect_tracked_changed_python_files_filters_non_python(monkeypatch):
    class _Result:
        def __init__(self, stdout: str):
            self.returncode = 0
            self.stdout = stdout

    outputs = iter(
        [
            _Result("autotrade/core/day_manager.py\nREADME.md\n"),
            _Result("tests/test_premarket_manager.py\nnotes.txt\n"),
        ]
    )

    monkeypatch.setattr(
        master_supervisor_mod.subprocess,
        "run",
        lambda *args, **kwargs: next(outputs),
    )
    monkeypatch.setattr(master_supervisor_mod.Path, "exists", lambda self: True)

    supervisor = MasterSupervisor.__new__(MasterSupervisor)
    files = supervisor._collect_tracked_changed_python_files()

    assert files
    assert all(path.endswith(".py") for path in files)
    assert any("day_manager.py" in path for path in files)
    assert any("test_premarket_manager.py" in path for path in files)


def test_collect_recent_commit_python_files_filters_non_python(monkeypatch):
    class _Result:
        def __init__(self, stdout: str):
            self.returncode = 0
            self.stdout = stdout

    outputs = iter(
        [
            _Result("abc123\ndef456\n"),
            _Result("autotrade/core/day_manager.py\nREADME.md\n"),
            _Result("tests/test_premarket_manager.py\nnotes.txt\n"),
        ]
    )

    monkeypatch.setattr(
        master_supervisor_mod.subprocess,
        "run",
        lambda *args, **kwargs: next(outputs),
    )

    supervisor = MasterSupervisor.__new__(MasterSupervisor)
    supervisor.runtime_recovery_cfg = SimpleNamespace(
        smoke_pytest_recent_commit_window=2
    )
    files = supervisor._collect_recent_commit_python_files()

    assert files
    assert all(path.endswith(".py") for path in files)
    assert any("day_manager.py" in path for path in files)
    assert any("test_premarket_manager.py" in path for path in files)


def test_run_runtime_recovery_smoke_treats_pytest_failure_as_advisory(monkeypatch):
    class _Result:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    outputs = iter(
        [
            _Result(0),
            _Result(
                1,
                stdout=(
                    "FAILED tests/test_premarket_manager.py::test_case - AssertionError\n"
                ),
            ),
        ]
    )

    monkeypatch.setattr(
        master_supervisor_mod.subprocess,
        "run",
        lambda *args, **kwargs: next(outputs),
    )

    supervisor = MasterSupervisor.__new__(MasterSupervisor)
    supervisor.runtime_recovery_cfg = SimpleNamespace(
        preflight_enabled=True,
        smoke_py_compile_changed_files=True,
        smoke_pytest_blocking=False,
        smoke_pytest_recent_commit_window=12,
        smoke_pytest_modules=["tests/test_premarket_manager.py"],
    )
    supervisor._collect_tracked_changed_python_files = lambda: []
    supervisor._collect_recent_commit_python_files = lambda: []

    result = supervisor._run_runtime_recovery_smoke("python")

    assert result["success"] is True
    assert "smoke_pytest_failed:nonblocking_smoke_failure" in result["advisories"]
    assert not result["failures"]


def test_run_runtime_recovery_smoke_uses_isolated_pytest_cache(monkeypatch):
    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    calls = []

    def _fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return _Result()

    monkeypatch.setattr(master_supervisor_mod.subprocess, "run", _fake_run)

    supervisor = MasterSupervisor.__new__(MasterSupervisor)
    supervisor.runtime_recovery_cfg = SimpleNamespace(
        preflight_enabled=True,
        smoke_py_compile_changed_files=True,
        smoke_pytest_blocking=False,
        smoke_pytest_recent_commit_window=12,
        smoke_pytest_modules=["tests/test_premarket_manager.py"],
    )
    supervisor._collect_tracked_changed_python_files = lambda: []

    result = supervisor._run_runtime_recovery_smoke("python")

    assert result["success"] is True
    assert "smoke_pytest_ok" in result["checks"]
    pytest_cmd = calls[1]
    assert "--cache-clear" not in pytest_cmd
    assert "-o" in pytest_cmd
    cache_override = pytest_cmd[pytest_cmd.index("-o") + 1]
    assert cache_override.startswith("cache_dir=")
    assert ".pytest_cache" not in cache_override


def test_run_runtime_recovery_smoke_retries_pytest_internalerror(monkeypatch):
    class _Result:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    outputs = iter(
        [
            _Result(0),
            _Result(
                3,
                stderr=(
                    "INTERNALERROR> PermissionError: [WinError 5] Access is denied: "
                    "'.pytest_cache'"
                ),
            ),
            _Result(0),
        ]
    )

    monkeypatch.setattr(
        master_supervisor_mod.subprocess,
        "run",
        lambda *args, **kwargs: next(outputs),
    )

    supervisor = MasterSupervisor.__new__(MasterSupervisor)
    supervisor.runtime_recovery_cfg = SimpleNamespace(
        preflight_enabled=True,
        smoke_py_compile_changed_files=True,
        smoke_pytest_blocking=True,
        smoke_pytest_recent_commit_window=12,
        smoke_pytest_timeout_seconds=45,
        smoke_pytest_modules=["tests/test_premarket_manager.py"],
    )
    supervisor._collect_tracked_changed_python_files = lambda: []

    result = supervisor._run_runtime_recovery_smoke("python")

    assert result["success"] is True
    assert "smoke_pytest_internalerror_retry" in result["checks"]
    assert "smoke_pytest_ok_after_retry" in result["checks"]
    assert not result["failures"]


def test_run_runtime_recovery_smoke_treats_pytest_timeout_as_advisory(monkeypatch):
    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    calls = {"count": 0}

    def _fake_run(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return _Result()
        raise master_supervisor_mod.subprocess.TimeoutExpired(
            cmd=kwargs.get("args", args[0] if args else []),
            timeout=45,
            output="partial pytest output",
        )

    monkeypatch.setattr(master_supervisor_mod.subprocess, "run", _fake_run)

    supervisor = MasterSupervisor.__new__(MasterSupervisor)
    supervisor.runtime_recovery_cfg = SimpleNamespace(
        preflight_enabled=True,
        smoke_py_compile_changed_files=True,
        smoke_pytest_blocking=False,
        smoke_pytest_recent_commit_window=12,
        smoke_pytest_timeout_seconds=45,
        smoke_pytest_modules=["tests/test_premarket_manager.py"],
    )
    supervisor._collect_tracked_changed_python_files = lambda: []
    supervisor._collect_recent_commit_python_files = lambda: []

    result = supervisor._run_runtime_recovery_smoke("python")

    assert result["success"] is True
    assert "smoke_pytest_timeout:45s" in result["advisories"]
    assert not result["failures"]


def test_run_runtime_recovery_smoke_marks_recent_overlap_for_pytest_failure(
    monkeypatch,
):
    class _Result:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    outputs = iter(
        [
            _Result(0),
            _Result(
                1,
                stdout=(
                    "FAILED tests/test_premarket_manager.py::test_case - AssertionError\n"
                ),
            ),
        ]
    )

    monkeypatch.setattr(
        master_supervisor_mod.subprocess,
        "run",
        lambda *args, **kwargs: next(outputs),
    )

    project_root = master_supervisor_mod.PROJECT_ROOT.resolve()
    overlap_path = str((project_root / "tests" / "test_premarket_manager.py").resolve())

    supervisor = MasterSupervisor.__new__(MasterSupervisor)
    supervisor.runtime_recovery_cfg = SimpleNamespace(
        preflight_enabled=True,
        smoke_py_compile_changed_files=True,
        smoke_pytest_blocking=False,
        smoke_pytest_recent_commit_window=12,
        smoke_pytest_modules=["tests/test_premarket_manager.py"],
    )
    supervisor._collect_tracked_changed_python_files = lambda: []
    supervisor._collect_recent_commit_python_files = lambda: [overlap_path]

    result = supervisor._run_runtime_recovery_smoke("python")

    assert result["success"] is True
    assert "smoke_pytest_failed:recent_change_overlap" in result["advisories"]
    assert any(
        advisory.startswith("smoke_pytest_recent_change_overlap:")
        for advisory in result["advisories"]
    )


def test_run_guarded_continuous_restarts_and_recovers(monkeypatch):
    statuses = []
    supervisor = MasterSupervisor.__new__(MasterSupervisor)
    supervisor.runtime_recovery_cfg = SimpleNamespace(
        enabled=True,
        preflight_enabled=True,
        post_restart_smoke_enabled=True,
        allow_live_local_repair=True,
        allow_live_openai_repair=True,
        allow_live_codex_repair=True,
        restart_backoff_seconds=0,
        max_restarts_per_hour=3,
        max_restarts_per_session=3,
        fatal_signature_threshold=3,
    )
    supervisor.issues_detected = []
    supervisor.issues_fixed = []
    supervisor._guard_restart_times = []
    supervisor._write_recovery_status = lambda payload: statuses.append(
        payload["stage"]
    )
    supervisor._run_runtime_recovery_smoke = lambda python_exe: {
        "success": True,
        "checks": ["ok"],
        "failures": [],
    }
    runs = iter([(1, ["boom"]), (0, [])])
    supervisor._run_monitored_command = lambda *args, **kwargs: next(runs)
    supervisor._build_task_command = lambda python_exe, task_name, args=None: [
        python_exe,
        "-m",
        "autotrade.core.autonomous_agent",
        "--continuous",
    ]
    monkeypatch.setattr(master_supervisor_mod.time, "sleep", lambda seconds: None)

    exit_code, unfixed = supervisor._run_guarded_continuous("python", ["--execute"])

    assert exit_code == 0
    assert unfixed == []
    assert "preflight" in statuses
    assert "failure" in statuses
    assert "complete" in statuses


def test_resolve_issue_file_path_uses_structured_logger_name(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    module_dir = project_root / "autotrade" / "utils"
    module_dir.mkdir(parents=True)
    fake_module_path = project_root / "autotrade" / "core" / "master_supervisor.py"
    fake_module_path.parent.mkdir(parents=True, exist_ok=True)
    fake_module_path.write_text("# test shim\n", encoding="utf-8")
    target_file = module_dir / "youtube_readiness.py"
    target_file.write_text("# test target\n", encoding="utf-8")

    monkeypatch.setattr(
        master_supervisor_mod, "__file__", str(fake_module_path), raising=False
    )

    supervisor = MasterSupervisor.__new__(MasterSupervisor)
    issue = DetectedIssue(
        issue_type=IssueType.CODE_ERROR,
        message=(
            '{"timestamp":"2026-03-24T06:45:00","level":"ERROR",'
            '"logger":"autotrade.utils.youtube_readiness",'
            '"message":"[YOUTUBE] Pending metadata fetch skipped: _BrokenExc: boom"}'
        ),
    )

    resolved = supervisor._resolve_issue_file_path(issue)

    assert resolved == target_file.resolve()


def test_run_code_fix_attempt_accepts_already_applied_codeagent_fix(monkeypatch):
    supervisor = MasterSupervisor.__new__(MasterSupervisor)
    supervisor.task_router = SimpleNamespace(
        route=lambda task: SimpleNamespace(
            success=True,
            message=json.dumps({"summary": "fixed internally", "changes": []}),
            data={
                "model": "openrouter:anthropic/claude-3.5-sonnet",
                "changes_applied": 1,
                "fix_details": [
                    {
                        "file": "autotrade/utils/youtube_readiness.py",
                        "search": "old",
                        "replace": "new",
                        "reason": "normalize encoding handling",
                    }
                ],
            },
            model_used="openrouter:anthropic/claude-3.5-sonnet",
        )
    )

    issue = DetectedIssue(
        issue_type=IssueType.CODE_ERROR,
        message="encoding failure",
        file_path="autotrade/utils/youtube_readiness.py",
        error_type="RecursionError",
    )

    applied_count, fix_data = supervisor._run_code_fix_attempt(
        issue=issue,
        traceback_str="RecursionError: encoding with 'cp1252' codec failed",
        force_escalation=False,
        attempt=1,
    )

    assert applied_count == 1
    assert fix_data is not None
    assert fix_data["summary"] == "fixed internally"
    assert len(fix_data["changes"]) == 1
