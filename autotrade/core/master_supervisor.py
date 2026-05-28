"""
Master Supervisor - Real-Time Process Monitor & Self-Healing Orchestrator

This is the MISSING PIECE that watches child processes and diagnoses issues in real-time.

Architecture:
    [MASTER SUPERVISOR] (this file - runs in main console)
           │
           ├── Spawns child processes in separate consoles
           ├── Monitors stdout/stderr in real-time via pipes
           ├── Detects errors/warnings immediately
           ├── Routes issues to specialized agents via TaskRouter
           ├── Auto-heals: installs packages, fixes imports, restarts
           └── Logs everything for post-mortem analysis

Usage:
    python -m autotrade.core.master_supervisor --task charts
    python -m autotrade.core.master_supervisor --task overnight
    python -m autotrade.core.master_supervisor --task continuous --execute

    Or via batch file menu (auto-invoked)
"""

import os
import sys
import json
import time
import queue
import logging
import threading
import subprocess
import re
from pathlib import Path
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from config.config_loader import get_config

# Configure safe logging (Windows console compatible)
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

try:
    from autotrade.utils.safe_logging import get_safe_logger

    logger = get_safe_logger(
        "MasterSupervisor",
        LOG_DIR / f"supervisor_{datetime.now().strftime('%Y%m%d')}.log",
    )
except ImportError:
    log_file = LOG_DIR / f"supervisor_{datetime.now().strftime('%Y%m%d')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("MasterSupervisor")


class IssueType(Enum):
    """Types of issues the supervisor can detect."""

    MISSING_MODULE = "missing_module"
    MISSING_PACKAGE = "missing_package"
    IMPORT_ERROR = "import_error"
    CONNECTION_ERROR = "connection_error"
    API_ERROR = "api_error"
    CONFIG_ERROR = "config_error"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    CODE_ERROR = "code_error"  # SyntaxError, AttributeError, etc - CodeAgent can fix
    LOGGING_ERROR = "logging_error"  # Unicode/encoding issues in logging
    WARNING = "warning"
    UNKNOWN = "unknown"


@dataclass
class DetectedIssue:
    """An issue detected in child process output."""

    issue_type: IssueType
    message: str
    module_name: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    auto_fixable: bool = False
    fix_command: Optional[str] = None
    fixed: bool = False
    traceback_lines: List[str] = field(
        default_factory=list
    )  # Full traceback for CodeAgent
    error_type: Optional[str] = None  # e.g., 'SyntaxError', 'AttributeError'


class OutputParser:
    """Parses child process output to detect issues in real-time."""

    # Regex patterns for common issues
    PATTERNS = {
        IssueType.MISSING_MODULE: [
            r"No module named '([^']+)'",
            r"ModuleNotFoundError: No module named '([^']+)'",
        ],
        IssueType.MISSING_PACKAGE: [
            r"([a-zA-Z0-9_-]+) not installed",
            r"(?:package|module|dependency)\s+([a-zA-Z0-9_-]+)\s+not available",
            r"([a-zA-Z0-9_-]+)\s+(?:package|module)\s+not available",
            r"skipped \(missing ([a-zA-Z0-9_-]+)\)",  # e.g. "Chart generation skipped (missing mplfinance)"
            r"missing ([a-zA-Z0-9_]+)\)",  # e.g. "(missing mplfinance)"
        ],
        IssueType.IMPORT_ERROR: [
            r"ImportError: cannot import name '([^']+)' from '([^']+)'",
            r"ImportError: (.+)",
        ],
        IssueType.CONNECTION_ERROR: [
            r"ConnectionRefusedError",
            r"Connection refused",
            r"Could not connect to ([^:]+):(\d+)",
            r"Ollama not reachable",
            r"SearXNG not running",
        ],
        IssueType.API_ERROR: [
            r"API key not (set|found)",
            r"401 Unauthorized",
            r"403 Forbidden",
            r"HTTPError: (\d+)",
        ],
        IssueType.CONFIG_ERROR: [
            r"Could not load config: (.+)",
            r"Missing .+ in .env",
            r"Config file not found",
            r"Failed to load tools index: (.+)",
            r"Failed to load .+ index: (.+)",
        ],
        IssueType.WARNING: [
            r"WARNING \| (?!.*[Bb]earish|.*[Bb]ullish|.*sentiment|.*Cooldown active \(|.*Entering cooldown for)(.+)",  # Exclude sentiment + known operational cooldown noise
            r"\[WARN\] (.+)",  # Standard [WARN] tag only, not [WARNING] which is used for sentiment
        ],
        IssueType.LOGGING_ERROR: [
            r"--- Logging error ---",
            r"UnicodeEncodeError: .* can't encode",
            r"'charmap' codec can't encode",
            r"character maps to <undefined>",
            r"UnicodeDecodeError:",
        ],
        IssueType.CODE_ERROR: [
            r"(\w+Error|\w+Exception|SyntaxError): (.+)",
            r"Signal generation failed: (.+)",
            r"Failed to analyze .+ for momentum: (.+)",
            r"\|\s*ERROR\s*\|.*\|\s*(.+)",
        ],
    }

    # Known fixable config issues
    CONFIG_FIXES = {
        "tools index": {
            "file": "tools/tools_index.json",
            "content": "{}",
            "description": "Empty tools index file",
        },
        "charts index": {
            "file": "charts/charts_index.json",
            "content": "{}",
            "description": "Empty charts index file",
        },
    }

    def parse_line(
        self, line: str, recent_lines: Optional[List[str]] = None
    ) -> Optional[DetectedIssue]:
        """Parse a single line of output for issues."""
        line = line.strip()
        if not line:
            return None
        # Cooldown chatter can spike into thousands of lines and is operational noise, not a fixable issue.
        if "Cooldown active (" in line:
            return None
        lowered = line.lower()
        if (
            "[self-heal] repeated operational error signature detected" in lowered
            or "[self-heal] runtime monitor applied" in lowered
            or "cannot fix - file not found: none" in lowered
            or '"had_errors": false' in lowered
        ):
            return None

        for issue_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    issue = self._create_issue(issue_type, line, match, recent_lines)
                    return issue

        return None

    def _create_issue(
        self,
        issue_type: IssueType,
        line: str,
        match: re.Match,
        recent_lines: Optional[List[str]] = None,
    ) -> DetectedIssue:
        """Create a DetectedIssue from a regex match."""
        issue = DetectedIssue(
            issue_type=issue_type,
            message=line,
        )

        # Extract module name for missing module errors
        if issue_type == IssueType.MISSING_MODULE:
            module_name = match.group(1)
            # Get root module (e.g., 'alpaca' from 'alpaca.trading.client')
            root_module = module_name.split(".")[0]
            issue.module_name = root_module
            issue.auto_fixable = True

        elif issue_type == IssueType.MISSING_PACKAGE:
            package_name = match.group(1).lower()
            issue.module_name = package_name
            issue.auto_fixable = True

        elif issue_type == IssueType.CONFIG_ERROR:
            # Check if we have a known fix for this config issue
            for key, fix_info in self.CONFIG_FIXES.items():
                if key in line.lower():
                    issue.file_path = fix_info["file"]
                    issue.fix_command = (
                        f"__CREATE_FILE__:{fix_info['file']}:{fix_info['content']}"
                    )
                    issue.auto_fixable = True
                    break

        elif issue_type == IssueType.LOGGING_ERROR:
            # Logging/encoding errors - track file if we can extract it
            file_match = re.search(r'File "([^"]+)", line (\d+)', line)
            if file_match:
                issue.file_path = file_match.group(1)
                issue.line_number = int(file_match.group(2))
            elif recent_lines:
                # Search backwards in recent lines for a file hint
                for prev in reversed(recent_lines):
                    file_match = re.search(r'File "([^"]+)", line (\d+)', prev)
                    if file_match:
                        issue.file_path = file_match.group(1)
                        issue.line_number = int(file_match.group(2))
                        break

            # These are not auto-fixable but should be flagged prominently
            issue.auto_fixable = False
            # Note: The actual fix is to use safe_logging module
            issue.fix_command = (
                "__INFO__:Replace emoji with ASCII or use autotrade.utils.safe_logging"
            )

        elif issue_type == IssueType.CODE_ERROR:
            # Code error from log line
            if match.lastindex >= 2:
                issue.error_type = match.group(1)
                # If group 1 is not an Error/Exception class, it might be just a message
                if not (
                    issue.error_type.endswith("Error")
                    or issue.error_type.endswith("Exception")
                ):
                    issue.error_type = "RuntimeError"
            else:
                issue.error_type = "RuntimeError"

            # Try to find file path in same line
            file_match = re.search(r'File "([^"]+)", line (\d+)', line)
            if file_match:
                issue.file_path = file_match.group(1)
                issue.line_number = int(file_match.group(2))
            elif recent_lines:
                # Search backwards in recent lines for a file hint
                for prev in reversed(recent_lines):
                    file_match = re.search(r'File "([^"]+)", line (\d+)', prev)
                    if file_match:
                        issue.file_path = file_match.group(1)
                        issue.line_number = int(file_match.group(2))
                        break

            # Treat all CODE_ERROR detections as auto-fix candidates.
            issue.auto_fixable = True

        return issue


class TracebackCollector:
    """
    Collects multi-line Python tracebacks and creates CODE_ERROR issues.

    Tracebacks look like:
        Traceback (most recent call last):
          File "...", line N, in <func>
            <code>
        ErrorType: message

    We need to collect ALL lines to give CodeAgent full context.
    """

    # Error types that CodeAgent can fix
    FIXABLE_ERROR_TYPES = {
        "SyntaxError",
        "AttributeError",
        "TypeError",
        "NameError",
        "ImportError",
        "IndentationError",
        "KeyError",
        "IndexError",
    }

    def __init__(self):
        self.collecting = False
        self.lines: List[str] = []
        self.file_path: Optional[str] = None
        self.line_number: Optional[int] = None
        self.error_type: Optional[str] = None
        self.error_message: Optional[str] = None

    def feed_line(self, line: str) -> Optional[DetectedIssue]:
        """
        Feed a line of output. Returns a DetectedIssue when traceback is complete.
        """
        stripped = line.strip()

        # Start collecting on "Traceback (most recent call last):" or "  File "..."" (for SyntaxError)
        if "Traceback (most recent call last):" in line or (
            line.startswith('  File "') and not self.collecting
        ):
            self.collecting = True
            self.lines = [line]
            self.file_path = None
            self.line_number = None
            self.error_type = None
            self.error_message = None

            # If it's the "  File" line, extract info immediately
            if line.startswith('  File "'):
                file_match = re.search(r'File "([^"]+)", line (\d+)', line)
                if file_match:
                    self.file_path = file_match.group(1)
                    self.line_number = int(file_match.group(2))
            return None

        if not self.collecting:
            return None

        # Accumulate lines
        self.lines.append(line)

        # Extract file/line info from "File "...", line N" lines
        file_match = re.search(r'File "([^"]+)", line (\d+)', line)
        if file_match:
            # Keep the LAST file/line (closest to the error)
            self.file_path = file_match.group(1)
            self.line_number = int(file_match.group(2))

        # Check if this line is the final error line (ErrorType: message)
        error_match = re.match(r"^(\w+Error|\w+Exception|SyntaxError): (.+)$", stripped)
        if error_match:
            self.error_type = error_match.group(1)
            self.error_message = error_match.group(2)
            return self._create_issue()

        # Also catch single-word error types without message
        if (
            stripped in self.FIXABLE_ERROR_TYPES
            or stripped.endswith("Error")
            or stripped.endswith("Exception")
        ):
            self.error_type = stripped
            self.error_message = ""
            return self._create_issue()

        # Safety: if we've collected too many lines, something is wrong
        if len(self.lines) > 100:
            self.collecting = False
            self.lines = []

        return None

    def _create_issue(self) -> DetectedIssue:
        """Create a CODE_ERROR issue from collected traceback."""
        issue = DetectedIssue(
            issue_type=IssueType.CODE_ERROR,
            message=f"{self.error_type}: {self.error_message}",
            file_path=self.file_path,
            line_number=self.line_number,
            traceback_lines=self.lines.copy(),
            error_type=self.error_type,
            auto_fixable=self.error_type in self.FIXABLE_ERROR_TYPES,
        )

        # Reset state
        self.collecting = False
        self.lines = []

        return issue


class ChildProcessMonitor:
    """Monitors a child process in real-time."""

    def __init__(self, command: List[str], supervisor: "MasterSupervisor"):
        self.command = command
        self.supervisor = supervisor
        self.process: Optional[subprocess.Popen] = None
        self.output_queue: queue.Queue = queue.Queue()
        self.issues: List[DetectedIssue] = []
        self.parser = OutputParser()
        self.traceback_collector = TracebackCollector()  # NEW: Collect tracebacks
        from collections import deque

        self.recent_lines = deque(maxlen=20)  # Buffer for path extraction hints
        self.running = False
        self.exit_code: Optional[int] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def start(self) -> bool:
        """Start the child process with output monitoring."""
        try:
            logger.info(f"[SPAWN] Starting child process: {' '.join(self.command)}")
            self.start_time = datetime.now()

            # Ensure all command elements are strings
            cmd = [str(c) for c in self.command]

            # Start process with pipes for stdout/stderr
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr into stdout
                stdin=subprocess.DEVNULL,  # Prevent hanging on input
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=0,  # Unbuffered for real-time responsiveness
                shell=False,  # Use direct list execution
                cwd=Path(__file__).parent.parent.parent,  # AutoTrade root
            )

            self.running = True

            # Start output reader thread
            reader_thread = threading.Thread(target=self._read_output, daemon=True)
            reader_thread.start()

            return True

        except Exception as e:
            logger.error(f"[SPAWN] Failed to start process: {e}")
            return False

    def _read_output(self):
        """Read output from child process in real-time."""
        logger.info("[MONITOR] Output reader thread started")
        try:
            for line in iter(self.process.stdout.readline, ""):
                if not line:
                    logger.debug("[MONITOR] End of output stream")
                    break

                line = line.rstrip()
                self.output_queue.put(line)

                # FIRST: Check for traceback (multi-line error)
                traceback_issue = self.traceback_collector.feed_line(line)
                if traceback_issue:
                    logger.debug(
                        f"[MONITOR] Traceback detected: {traceback_issue.error_type}"
                    )
                    self.issues.append(traceback_issue)
                    self.supervisor.on_issue_detected(traceback_issue)
                    self.recent_lines.append(line)
                    continue  # Don't double-parse the error line

                # SECOND: Parse for single-line issues (imports, warnings, etc)
                issue = self.parser.parse_line(line, list(self.recent_lines))
                if issue:
                    logger.debug(f"[MONITOR] Issue detected: {issue.issue_type}")
                    self.issues.append(issue)
                    # Notify supervisor immediately
                    self.supervisor.on_issue_detected(issue)

                self.recent_lines.append(line)

        except Exception as e:
            logger.error(f"[MONITOR] Output reader error: {e}")
        finally:
            logger.info("[MONITOR] Output reader thread finishing")
            self.running = False
            self.exit_code = self.process.wait()
            self.end_time = datetime.now()
            logger.info(f"[MONITOR] Process finished with exit code {self.exit_code}")

    def get_output(self, timeout: float = 0.1) -> Optional[str]:
        """Get next line of output (non-blocking)."""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def wait(self, timeout: Optional[float] = None) -> int:
        """Wait for process to complete."""
        if self.process:
            return self.process.wait(timeout=timeout)
        return -1

    def terminate(self):
        """Terminate the child process."""
        if self.process and self.running:
            logger.info("[SPAWN] Terminating child process...")
            self.process.terminate()
            self.running = False


class MasterSupervisor:
    """
    The Master Supervisor that orchestrates child processes and self-heals.

    This runs in the main console and:
    1. Spawns child tasks in separate processes
    2. Monitors their output in real-time
    3. Detects errors/warnings immediately
    4. Auto-fixes issues (installs packages, etc.)
    5. Routes complex issues to specialized agents
    6. Provides real-time feedback
    """

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.config = get_config()
        self.runtime_recovery_cfg = getattr(self.config, "runtime_recovery", None)
        self.current_monitor: Optional[ChildProcessMonitor] = None
        self.issues_detected: List[DetectedIssue] = []
        self.issues_fixed: List[DetectedIssue] = []
        self._guard_restart_times: List[datetime] = []
        os.environ.setdefault("AUTOTRADE_CODE_REPAIR_API_ONLY", "1")
        os.environ.setdefault("AUTOTRADE_ENABLE_QWEN3_REPAIR", "0")
        self.task_router = None
        self._init_task_router()

        logger.info("=" * 60)
        logger.info(" MASTER SUPERVISOR INITIALIZED")
        logger.info(f" Mode: {'DRY RUN' if dry_run else 'LIVE'}")
        logger.info(f" TaskRouter: {'Available' if self.task_router else 'Not loaded'}")
        logger.info("=" * 60)

    def _build_task_command(
        self, python_exe: str, task_name: str, args: Optional[List[str]] = None
    ) -> List[str]:
        args = list(args or [])
        special_tasks = {
            "youtube-weekend": ["tools/youtube_weekend_scanner.py"],
            "youtube_weekend": ["tools/youtube_weekend_scanner.py"],
            "weekend-youtube": ["tools/youtube_weekend_scanner.py"],
            "weekend": ["tools/youtube_weekend_scanner.py", "--all", "--verbose"],
        }
        resolved_task = "continuous" if task_name == "continuous-guarded" else task_name
        if resolved_task in special_tasks:
            # Handle sequencing for complex 'weekend' task
            if resolved_task == "weekend":
                logger.info("[WEEKEND] Starting FULL Weekend Mode sequence...")
                # 1. YouTube intelligence
                yt_cmd = [python_exe] + special_tasks["weekend"]
                subprocess.run(yt_cmd, cwd=PROJECT_ROOT)

                # 2. Overnight research (heavy lift)
                logger.info("[WEEKEND] Moving to FULL Overnight Research...")
                overnight_cmd = [
                    python_exe,
                    "-m",
                    "autotrade.core.autonomous_agent",
                    "--overnight",
                    "--fresh",
                ]
                subprocess.run(overnight_cmd, cwd=PROJECT_ROOT)

                # 3. Monday plan generation
                logger.info("[WEEKEND] Generating Monday Morning Plan...")
                return [
                    python_exe,
                    "tools/generate_morning_plan.py",
                ]  # Last one returns to the runner

            return [python_exe] + special_tasks[resolved_task] + args
        return [
            python_exe,
            "-m",
            "autotrade.core.autonomous_agent",
            f"--{resolved_task}",
        ] + args

    def _find_execution_artifacts(
        self,
        *,
        logs_dir: Path,
        today_compact: str,
        today_dashed: str,
    ) -> List[Path]:
        """Return execution artifacts written by the live runtime for today's session."""
        candidates = [
            logs_dir / f"trade_decisions_{today_compact}.json",
            logs_dir / f"trade_decisions_{today_dashed}.json",
            logs_dir / "trade_journal.json",
            logs_dir / "trade_journal.csv",
        ]
        return [path for path in candidates if path.exists()]

    def _collect_tracked_changed_python_files(self) -> List[str]:
        project_root = Path(__file__).parent.parent.parent.resolve()
        commands = [
            ["git", "diff", "--name-only", "--diff-filter=ACMR"],
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        ]
        changed: List[str] = []
        for cmd in commands:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=30,
                    cwd=project_root,
                )
            except Exception:
                continue
            if result.returncode != 0:
                continue
            for line in (result.stdout or "").splitlines():
                rel = str(line or "").strip().replace("\\", "/")
                if not rel.endswith(".py"):
                    continue
                full = project_root / rel
                if full.exists():
                    changed.append(str(full))
        deduped: List[str] = []
        seen = set()
        for path in changed:
            if path in seen:
                continue
            seen.add(path)
            deduped.append(path)
        return deduped

    def _collect_recent_commit_python_files(self) -> List[str]:
        cfg = self.runtime_recovery_cfg
        project_root = Path(__file__).parent.parent.parent.resolve()
        commit_window = int(getattr(cfg, "smoke_pytest_recent_commit_window", 12) or 12)
        if commit_window <= 0:
            return []

        try:
            revs = subprocess.run(
                ["git", "log", f"-n{commit_window}", "--format=%H"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
                cwd=project_root,
            )
        except Exception:
            return []

        if revs.returncode != 0:
            return []

        commit_ids = [
            line.strip() for line in (revs.stdout or "").splitlines() if line.strip()
        ]
        if not commit_ids:
            return []

        changed: List[str] = []
        for commit_id in commit_ids:
            try:
                result = subprocess.run(
                    ["git", "show", "--name-only", "--format=", commit_id],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=30,
                    cwd=project_root,
                )
            except Exception:
                continue
            if result.returncode != 0:
                continue
            for line in (result.stdout or "").splitlines():
                rel = str(line or "").strip().replace("\\", "/")
                if not rel.endswith(".py"):
                    continue
                full = project_root / rel
                changed.append(str(full.resolve()))

        deduped: List[str] = []
        seen = set()
        for path in changed:
            if path in seen:
                continue
            seen.add(path)
            deduped.append(path)
        return deduped

    def _extract_pytest_failure_paths(self, *outputs: str) -> List[str]:
        project_root = Path(__file__).parent.parent.parent.resolve()
        candidates: List[str] = []
        patterns = [
            r"FAILED\s+((?:tests|autotrade|config|tools)[/\\][^\s:]+\.py)",
            r'File "([^"]+\.py)"',
        ]
        for output in outputs:
            if not output:
                continue
            for pattern in patterns:
                for match in re.findall(pattern, output):
                    rel = str(match or "").strip().replace("\\", "/")
                    if not rel:
                        continue
                    path_obj = Path(rel)
                    if not path_obj.is_absolute():
                        path_obj = project_root / rel
                    candidates.append(str(path_obj.resolve()))

        deduped: List[str] = []
        seen = set()
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            deduped.append(path)
        return deduped

    def _summarize_smoke_pytest_failure(
        self,
        *,
        stdout: str,
        stderr: str,
        returncode: int,
    ) -> Dict[str, Any]:
        impacted_paths = self._extract_pytest_failure_paths(stdout, stderr)
        tracked_paths = self._collect_tracked_changed_python_files()
        recent_commit_paths = self._collect_recent_commit_python_files()
        recent_change_paths = set(tracked_paths) | set(recent_commit_paths)
        overlapping_paths = sorted(
            path for path in impacted_paths if path in recent_change_paths
        )
        return {
            "returncode": returncode,
            "impacted_paths": impacted_paths,
            "tracked_paths": tracked_paths,
            "recent_commit_paths": recent_commit_paths,
            "overlapping_paths": overlapping_paths,
            "recent_overlap_detected": bool(overlapping_paths),
        }

    @staticmethod
    def _is_pytest_infrastructure_failure(*outputs: str) -> bool:
        text = "\n".join(output or "" for output in outputs)
        return "INTERNALERROR>" in text or "PermissionError: [WinError 5]" in text

    def _write_recovery_status(self, payload: Dict[str, Any]) -> None:
        cfg = self.runtime_recovery_cfg
        if cfg is None:
            return
        project_root = Path(__file__).parent.parent.parent.resolve()
        for attr_name, append_mode in (
            ("status_json_path", False),
            ("status_jsonl_path", True),
        ):
            rel = getattr(cfg, attr_name, "")
            if not rel:
                continue
            target = Path(rel)
            if not target.is_absolute():
                target = project_root / target
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                if append_mode:
                    with open(target, "a", encoding="utf-8") as f:
                        f.write(json.dumps(payload, default=str) + "\n")
                else:
                    with open(target, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2, default=str)
            except Exception as e:
                logger.warning(
                    f"[GUARDED] Failed writing recovery status to {target}: {e}"
                )

    def _run_runtime_recovery_smoke(self, python_exe: str) -> Dict[str, Any]:
        cfg = self.runtime_recovery_cfg
        result: Dict[str, Any] = {
            "success": True,
            "checks": [],
            "failures": [],
            "advisories": [],
        }
        if cfg is None or not bool(getattr(cfg, "preflight_enabled", True)):
            result["checks"].append("preflight_disabled")
            return result

        workflow_cmd = [
            python_exe,
            "-m",
            "autotrade.core.autonomous_agent",
            "--workflow-validate",
        ]
        try:
            workflow = subprocess.run(
                workflow_cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=900,
                cwd=PROJECT_ROOT,
            )
            if workflow.returncode == 0:
                result["checks"].append("workflow_validate_ok")
            else:
                result["success"] = False
                result["failures"].append("workflow_validate_failed")
        except Exception as e:
            result["success"] = False
            result["failures"].append(f"workflow_validate_error:{e}")

        if bool(getattr(cfg, "smoke_py_compile_changed_files", True)):
            changed_files = self._collect_tracked_changed_python_files()
            if changed_files:
                try:
                    compile_cmd = [python_exe, "-m", "py_compile", *changed_files]
                    compiled = subprocess.run(
                        compile_cmd,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        timeout=900,
                        cwd=PROJECT_ROOT,
                    )
                    if compiled.returncode == 0:
                        result["checks"].append(
                            f"py_compile_changed_files_ok:{len(changed_files)}"
                        )
                    else:
                        result["success"] = False
                        result["failures"].append("py_compile_changed_files_failed")
                except Exception as e:
                    result["success"] = False
                    result["failures"].append(f"py_compile_changed_files_error:{e}")
            else:
                result["checks"].append("py_compile_changed_files_skipped")

        pytest_modules = list(getattr(cfg, "smoke_pytest_modules", []) or [])
        if pytest_modules:
            try:
                smoke_cache_dir = LOG_DIR / "pytest_smoke_cache"
                smoke_cache_dir.mkdir(parents=True, exist_ok=True)
                pytest_cmd = [
                    python_exe,
                    "-m",
                    "pytest",
                    *pytest_modules,
                    "-q",
                    "-o",
                    f"cache_dir={smoke_cache_dir}",
                ]

                # Set environment to force UTF-8 output and prevent cp1252 recursion
                pytest_env = os.environ.copy()
                pytest_env["PYTHONIOENCODING"] = "utf-8"
                pytest_timeout = int(
                    getattr(cfg, "smoke_pytest_timeout_seconds", 180) or 180
                )

                pytest_run = subprocess.run(
                    pytest_cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=pytest_timeout,
                    cwd=PROJECT_ROOT,
                    env=pytest_env,
                )
                if pytest_run.returncode == 0:
                    result["checks"].append("smoke_pytest_ok")
                else:
                    if self._is_pytest_infrastructure_failure(
                        pytest_run.stdout or "", pytest_run.stderr or ""
                    ):
                        result["checks"].append("smoke_pytest_internalerror_retry")
                        logger.warning(
                            "[GUARDED] Pytest infrastructure failure detected; retrying once"
                        )
                        pytest_run = subprocess.run(
                            pytest_cmd,
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            timeout=pytest_timeout,
                            cwd=PROJECT_ROOT,
                            env=pytest_env,
                        )
                        if pytest_run.returncode == 0:
                            result["checks"].append("smoke_pytest_ok_after_retry")
                            return result
                        if self._is_pytest_infrastructure_failure(
                            pytest_run.stdout or "", pytest_run.stderr or ""
                        ):
                            result["advisories"].append(
                                "smoke_pytest_infrastructure_failure"
                            )
                            logger.warning(
                                "[GUARDED] Pytest infrastructure failure persisted after retry; launch will continue"
                            )
                            if pytest_run.stdout:
                                logger.warning(
                                    f"[GUARDED] Pytest stdout: {pytest_run.stdout[:2000]}"
                                )
                            if pytest_run.stderr:
                                logger.warning(
                                    f"[GUARDED] Pytest stderr: {pytest_run.stderr[:2000]}"
                                )
                            return result
                    failure_summary = self._summarize_smoke_pytest_failure(
                        stdout=pytest_run.stdout or "",
                        stderr=pytest_run.stderr or "",
                        returncode=pytest_run.returncode,
                    )
                    recent_overlap = bool(
                        failure_summary.get("recent_overlap_detected", False)
                    )
                    smoke_blocking = bool(getattr(cfg, "smoke_pytest_blocking", False))
                    result["checks"].append(
                        f"smoke_pytest_failed_exit_code:{pytest_run.returncode}"
                    )
                    if recent_overlap:
                        overlap_rel = []
                        project_root = PROJECT_ROOT.resolve()
                        for path in failure_summary.get("overlapping_paths", []):
                            try:
                                overlap_rel.append(
                                    str(Path(path).resolve().relative_to(project_root))
                                )
                            except Exception:
                                overlap_rel.append(path)
                        result["advisories"].append(
                            "smoke_pytest_recent_change_overlap:"
                            + ",".join(overlap_rel[:5])
                        )
                    advisory_reason = (
                        "recent_change_overlap"
                        if recent_overlap
                        else "nonblocking_smoke_failure"
                    )
                    if smoke_blocking and not recent_overlap:
                        result["success"] = False
                        result["failures"].append("smoke_pytest_failed")
                        logger.error(
                            f"[GUARDED] Preflight smoke tests failed (exit code {pytest_run.returncode})"
                        )
                    else:
                        result["advisories"].append(
                            f"smoke_pytest_failed:{advisory_reason}"
                        )
                        logger.warning(
                            "[GUARDED] Preflight smoke tests failed but launch will continue "
                            f"(exit code {pytest_run.returncode}, reason={advisory_reason})"
                        )
                    if pytest_run.stdout:
                        logger.warning(
                            f"[GUARDED] Pytest stdout: {pytest_run.stdout[:2000]}"
                        )
                    if pytest_run.stderr:
                        logger.warning(
                            f"[GUARDED] Pytest stderr: {pytest_run.stderr[:2000]}"
                        )
            except subprocess.TimeoutExpired as e:
                smoke_blocking = bool(getattr(cfg, "smoke_pytest_blocking", False))
                timeout_seconds = int(
                    getattr(cfg, "smoke_pytest_timeout_seconds", 180) or 180
                )
                result["checks"].append(f"smoke_pytest_timeout:{timeout_seconds}s")
                message = f"smoke_pytest_timeout:{timeout_seconds}s"
                if smoke_blocking:
                    result["success"] = False
                    result["failures"].append(message)
                    logger.error(
                        "[GUARDED] Preflight smoke tests timed out after %ss",
                        timeout_seconds,
                    )
                else:
                    result["advisories"].append(message)
                    logger.warning(
                        "[GUARDED] Preflight smoke tests timed out after %ss; launch will continue",
                        timeout_seconds,
                    )
                if e.stdout:
                    logger.warning(f"[GUARDED] Pytest stdout: {str(e.stdout)[:2000]}")
                if e.stderr:
                    logger.warning(f"[GUARDED] Pytest stderr: {str(e.stderr)[:2000]}")
            except Exception as e:
                result["success"] = False
                result["failures"].append(f"smoke_pytest_error:{e}")

        return result

    def _guard_restart_allowed(self) -> bool:
        cfg = self.runtime_recovery_cfg
        if cfg is None:
            return False
        now = datetime.now()
        self._guard_restart_times = [
            ts for ts in self._guard_restart_times if (now - ts).total_seconds() <= 3600
        ]
        max_hour = int(getattr(cfg, "max_restarts_per_hour", 0) or 0)
        max_session = int(getattr(cfg, "max_restarts_per_session", 0) or 0)
        if max_hour > 0 and len(self._guard_restart_times) >= max_hour:
            return False
        if max_session > 0 and len(self._guard_restart_times) >= max_session:
            return False
        return True

    def _record_guard_restart(self) -> None:
        self._guard_restart_times.append(datetime.now())

    def _attempt_critical_failure_repairs(
        self, exit_code: int, issues: List[DetectedIssue]
    ) -> None:
        if exit_code in (0, 64):
            return
        if self.dry_run:
            logger.info(
                "[CRITICAL-REPAIR] Runtime failed, but dry-run mode is active; "
                "repair skipped"
            )
            return

        candidates: List[DetectedIssue] = []
        seen = set()
        for issue in issues:
            if issue.fixed or not issue.auto_fixable:
                continue
            if issue.issue_type in (
                IssueType.MISSING_MODULE,
                IssueType.MISSING_PACKAGE,
            ):
                key = (
                    issue.issue_type.value,
                    issue.module_name or "",
                    issue.message,
                )
            elif issue.issue_type == IssueType.CODE_ERROR:
                key = (
                    issue.issue_type.value,
                    issue.error_type or "",
                    issue.file_path or "",
                    issue.line_number or 0,
                    issue.message,
                )
            else:
                continue
            if key in seen:
                continue
            seen.add(key)
            candidates.append(issue)

        if not candidates:
            logger.info(
                "[CRITICAL-REPAIR] Runtime failed with exit code "
                f"{exit_code}; no auto-fixable critical issue was detected"
            )
            return

        logger.error(
            "[CRITICAL-REPAIR] Runtime failed with exit code "
            f"{exit_code}; attempting {len(candidates)} critical repair(s)"
        )
        for issue in candidates:
            if issue.issue_type in (
                IssueType.MISSING_MODULE,
                IssueType.MISSING_PACKAGE,
            ):
                self._attempt_auto_fix(issue)
            elif issue.issue_type == IssueType.CODE_ERROR:
                self._fix_code_error(issue)

    def _run_monitored_command(
        self,
        task_name: str,
        command: List[str],
        validation_task_name: Optional[str] = None,
    ) -> Tuple[int, List[DetectedIssue]]:
        self.issues_detected = []
        self.issues_fixed = []
        self.current_monitor = ChildProcessMonitor(command, self)
        if not self.current_monitor.start():
            logger.error("[TASK] Failed to start child process")
            return 1, []

        try:
            while self.current_monitor.running:
                line = (
                    self.current_monitor.get_output(timeout=0.5)
                    if self.current_monitor.running
                    else None
                )
                if line:
                    try:
                        print(line)
                    except UnicodeEncodeError:
                        print(line.encode("ascii", errors="replace").decode("ascii"))
        except KeyboardInterrupt:
            logger.info("[TASK] Interrupted by user")
            self.current_monitor.terminate()

        exit_code = self.current_monitor.wait()
        if self.current_monitor.end_time and self.current_monitor.start_time:
            duration = (
                self.current_monitor.end_time - self.current_monitor.start_time
            ).total_seconds()
        else:
            duration = 0.0

        unfixed = [i for i in self.issues_detected if not i.fixed]
        if exit_code not in (0, 64) and unfixed:
            self._attempt_critical_failure_repairs(exit_code, unfixed)
            unfixed = [i for i in self.issues_detected if not i.fixed]

        logger.info("=" * 60)
        logger.info(f"[TASK] Completed: {task_name}")
        logger.info(f"[TASK] Exit code: {exit_code}")
        logger.info(f"[TASK] Duration: {duration:.1f}s")
        logger.info(f"[TASK] Issues detected: {len(self.issues_detected)}")
        logger.info(f"[TASK] Issues fixed: {len(self.issues_fixed)}")
        logger.info(f"[TASK] Issues remaining: {len(unfixed)}")
        logger.info("=" * 60)

        validation_result = self._validate_task_results(
            validation_task_name or task_name
        )
        if not validation_result["success"]:
            logger.warning(
                f"[VALIDATION] Task '{validation_task_name or task_name}' completed but results are INVALID!"
            )
            logger.warning(
                f"[VALIDATION] Problem: {validation_result.get('problem', 'unknown')}"
            )
            if self.task_router:
                self._diagnose_and_fix(
                    validation_task_name or task_name, validation_result
                )

        try:
            self._maybe_run_weekly_signal_audit()
        except Exception as e:
            logger.warning(f"[WEEKLY AUDIT] Skipped due to error: {e}")

        return exit_code, unfixed

    def _run_guarded_continuous(
        self, python_exe: str, args: Optional[List[str]] = None
    ) -> Tuple[int, List[DetectedIssue]]:
        cfg = self.runtime_recovery_cfg
        if cfg is None or not bool(getattr(cfg, "enabled", True)):
            command = self._build_task_command(python_exe, "continuous", args)
            return self._run_monitored_command(
                "continuous-guarded", command, "continuous"
            )

        previous_env = {
            "AUTOTRADE_CODE_REPAIR_API_ONLY": os.environ.get(
                "AUTOTRADE_CODE_REPAIR_API_ONLY"
            ),
            "AUTOTRADE_ENABLE_QWEN3_REPAIR": os.environ.get(
                "AUTOTRADE_ENABLE_QWEN3_REPAIR"
            ),
            "AUTOTRADE_ENABLE_OPENAI_REPAIR": os.environ.get(
                "AUTOTRADE_ENABLE_OPENAI_REPAIR"
            ),
            "AUTOTRADE_ENABLE_CODEX_REPAIR": os.environ.get(
                "AUTOTRADE_ENABLE_CODEX_REPAIR"
            ),
            "SELF_HEAL_ENABLE_LIVE": os.environ.get("SELF_HEAL_ENABLE_LIVE"),
            "SELF_HEAL_RUNTIME_ERROR_THRESHOLD": os.environ.get(
                "SELF_HEAL_RUNTIME_ERROR_THRESHOLD"
            ),
        }
        os.environ["AUTOTRADE_CODE_REPAIR_API_ONLY"] = "1"
        os.environ["AUTOTRADE_ENABLE_QWEN3_REPAIR"] = "0"
        os.environ["AUTOTRADE_ENABLE_OPENAI_REPAIR"] = (
            "1" if bool(getattr(cfg, "allow_live_openai_repair", True)) else "0"
        )
        os.environ["AUTOTRADE_ENABLE_CODEX_REPAIR"] = (
            "1" if bool(getattr(cfg, "allow_live_codex_repair", True)) else "0"
        )
        os.environ["SELF_HEAL_ENABLE_LIVE"] = (
            "1" if bool(getattr(cfg, "allow_live_local_repair", True)) else "0"
        )
        os.environ["SELF_HEAL_RUNTIME_ERROR_THRESHOLD"] = str(
            int(getattr(cfg, "fatal_signature_threshold", 3) or 3)
        )

        try:
            preflight = self._run_runtime_recovery_smoke(python_exe)
            self._write_recovery_status(
                {
                    "timestamp": datetime.now().isoformat(),
                    "task": "continuous-guarded",
                    "stage": "preflight",
                    "success": preflight["success"],
                    "checks": preflight["checks"],
                    "failures": preflight["failures"],
                }
            )
            if not preflight["success"]:
                logger.error(
                    "[GUARDED] Preflight checks failed; refusing to launch guarded continuous mode"
                )
                return 1, []

            attempt = 0
            while True:
                attempt += 1
                self.issues_detected = []
                self.issues_fixed = []
                command = self._build_task_command(
                    python_exe, "continuous", list(args or [])
                )
                self._write_recovery_status(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "task": "continuous-guarded",
                        "stage": "launch",
                        "attempt": attempt,
                        "command": command,
                    }
                )
                exit_code, unfixed = self._run_monitored_command(
                    "continuous-guarded", command, "continuous"
                )
                if exit_code in (0, 64) and not unfixed:
                    self._write_recovery_status(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "task": "continuous-guarded",
                            "stage": "complete",
                            "attempt": attempt,
                            "exit_code": exit_code,
                            "success": True,
                        }
                    )
                    return exit_code, unfixed

                self._write_recovery_status(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "task": "continuous-guarded",
                        "stage": "failure",
                        "attempt": attempt,
                        "exit_code": exit_code,
                        "unfixed_issue_count": len(unfixed),
                    }
                )
                if not self._guard_restart_allowed():
                    logger.error(
                        "[GUARDED] Restart budget exhausted for guarded continuous mode"
                    )
                    return exit_code or 1, unfixed

                if bool(getattr(cfg, "post_restart_smoke_enabled", True)):
                    smoke = self._run_runtime_recovery_smoke(python_exe)
                    self._write_recovery_status(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "task": "continuous-guarded",
                            "stage": "post_failure_smoke",
                            "attempt": attempt,
                            "success": smoke["success"],
                            "checks": smoke["checks"],
                            "failures": smoke["failures"],
                        }
                    )

                self._record_guard_restart()
                backoff = max(1, int(getattr(cfg, "restart_backoff_seconds", 20) or 20))
                logger.warning(
                    f"[GUARDED] Restarting guarded continuous mode after {backoff}s backoff (attempt {attempt})"
                )
                time.sleep(backoff)
        finally:
            for key, old_val in previous_env.items():
                if old_val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_val

    def _init_task_router(self):
        """Initialize TaskRouter for routing issues to specialized agents."""
        try:
            from autotrade.core.agentic_orchestrator import TaskRouter

            self.task_router = TaskRouter(dry_run=self.dry_run)
            logger.info("[INIT] TaskRouter loaded with specialized agents")
        except ImportError as e:
            logger.warning(f"[INIT] TaskRouter not available: {e}")

    def on_issue_detected(self, issue: DetectedIssue):
        """Called when an issue is detected in child output."""
        self.issues_detected.append(issue)

        # Treat missing-module tracebacks as dependency issues.
        if issue.issue_type == IssueType.CODE_ERROR and issue.error_type in (
            "ModuleNotFoundError",
            "ImportError",
        ):
            if "No module named" in issue.message:
                module_match = re.search(r"No module named '([^']+)'", issue.message)
                issue.issue_type = IssueType.MISSING_MODULE
                issue.module_name = (
                    module_match.group(1).split(".")[0] if module_match else None
                )
                issue.auto_fixable = True
                logger.error(f"[DETECT] missing_module: {issue.message}")
                logger.info(
                    "[DETECT] Repair deferred unless the child process exits "
                    "with a critical failure"
                )
                return
        if (
            issue.issue_type == IssueType.CODE_ERROR
            and "pip install" in issue.message.lower()
        ):
            issue.issue_type = IssueType.MISSING_PACKAGE
            issue.auto_fixable = True
            logger.error(f"[DETECT] missing_package: {issue.message}")
            logger.info(
                "[DETECT] Repair deferred unless the child process exits "
                "with a critical failure"
            )
            return

        # Color-coded output for visibility
        if issue.issue_type == IssueType.WARNING:
            logger.warning(f"[DETECT] {issue.message}")
        elif issue.issue_type == IssueType.LOGGING_ERROR:
            # Logging errors get special treatment - they spam but need attention
            logger.error("[LOGGING ERROR] Encoding issue detected!")
            if issue.file_path:
                logger.error(
                    f"[LOGGING ERROR] File: {issue.file_path}:{issue.line_number or '?'}"
                )
            logger.error(
                "[LOGGING ERROR] Fix: Use autotrade.utils.safe_logging or replace emoji with ASCII"
            )
        elif issue.issue_type == IssueType.CODE_ERROR:
            logger.error(f"[CODE ERROR] {issue.error_type}: {issue.message}")
            if issue.file_path:
                logger.error(
                    f"[CODE ERROR] File: {issue.file_path}:{issue.line_number or '?'}"
                )
            if issue.auto_fixable:
                logger.info(
                    "[CODE ERROR] Repair deferred unless the child process exits "
                    "with a critical failure"
                )
            return
        else:
            logger.error(f"[DETECT] {issue.issue_type.value}: {issue.message}")

        if issue.auto_fixable and issue.fix_command:
            logger.info(
                "[DETECT] Auto-fix command noted; repair deferred unless the "
                "child process fails"
            )

    @staticmethod
    def _camel_to_snake(value: str) -> str:
        value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value or "")
        return re.sub(r"_+", "_", value).strip("_").lower()

    @staticmethod
    def _extract_structured_log_metadata(
        message: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Pull logger/file hints from JSON-formatted log payloads."""
        if not message:
            return None, None

        payload = None
        stripped = message.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            candidate = stripped
        else:
            start = stripped.find("{")
            end = stripped.rfind("}")
            candidate = stripped[start : end + 1] if start != -1 and end > start else ""

        if candidate:
            try:
                loaded = json.loads(candidate)
                if isinstance(loaded, dict):
                    payload = loaded
            except Exception:
                payload = None

        if not payload:
            return None, None

        logger_name = (
            payload.get("logger")
            or payload.get("logger_name")
            or payload.get("name")
            or payload.get("module")
        )
        file_path = (
            payload.get("file_path")
            or payload.get("path")
            or payload.get("filename")
            or payload.get("source_file")
        )
        return (
            str(logger_name).strip() if logger_name else None,
            str(file_path).strip() if file_path else None,
        )

    def _resolve_issue_file_path(self, issue: DetectedIssue) -> Optional[Path]:
        project_root = Path(__file__).parent.parent.parent.resolve()

        def _to_project_path(candidate: Optional[str]) -> Optional[Path]:
            if not candidate:
                return None
            p = Path(str(candidate).strip())
            if not p.is_absolute():
                p = project_root / p
            try:
                p = p.resolve()
            except Exception:
                return None
            if not p.exists():
                return None
            try:
                p.relative_to(project_root)
            except ValueError:
                return None
            return p

        resolved = _to_project_path(issue.file_path)
        if resolved:
            return resolved

        msg = str(issue.message or "")
        logger_name = None
        structured_logger_name, structured_file_path = (
            self._extract_structured_log_metadata(msg)
        )
        if structured_file_path:
            resolved = _to_project_path(structured_file_path)
            if resolved:
                return resolved
        if structured_logger_name:
            logger_name = structured_logger_name
        m = re.search(r"\|\s*ERROR\s*\|\s*([^|]+?)\s*\|", msg)
        if m:
            logger_name = m.group(1).strip()
        if not logger_name:
            m = re.search(r"\[LOG:[A-Z]+\]\s*([^:]+):", msg)
            if m:
                logger_name = m.group(1).strip()

        # High-signal hard mappings for recurring issues.
        lower_msg = msg.lower()
        if "failed to analyze" in lower_msg and "momentum" in lower_msg:
            candidate = project_root / "autotrade" / "signals" / "momentum_engine.py"
            if candidate.exists():
                return candidate
        if "opencode.json" in lower_msg and "alpaca" in lower_msg:
            candidate = project_root / "autotrade" / "utils" / "mcp_client.py"
            if candidate.exists():
                return candidate

        if logger_name:
            if logger_name.startswith("autotrade."):
                mod_path = project_root / (logger_name.replace(".", "/") + ".py")
                if mod_path.exists():
                    return mod_path.resolve()
            if logger_name.startswith("AutoTrade."):
                suffix = logger_name.split(".", 1)[1]
                if suffix.startswith("AutonomousAgent"):
                    candidate = (
                        project_root / "autotrade" / "core" / "autonomous_agent.py"
                    )
                    if candidate.exists():
                        return candidate
                leaf = self._camel_to_snake(suffix.split(".")[-1])
                for root in (
                    "autotrade/core",
                    "autotrade/utils",
                    "autotrade/signals",
                    "langgraph_workflow",
                ):
                    candidate = project_root / root / f"{leaf}.py"
                    if candidate.exists():
                        return candidate.resolve()
            if "." not in logger_name:
                for root in (
                    "autotrade/core",
                    "autotrade/utils",
                    "autotrade/signals",
                    "langgraph_workflow",
                ):
                    candidate = project_root / root / f"{logger_name}.py"
                    if candidate.exists():
                        return candidate.resolve()

        return None

    def _attempt_auto_fix(self, issue: DetectedIssue):
        """Attempt to automatically fix a detected issue."""
        if self.dry_run:
            logger.info(f"[AUTO-FIX] DRY RUN - Would execute: {issue.fix_command}")
            return

        # Special handling for missing dependencies - use resolver instead of hard-coded mapping
        if issue.issue_type in (IssueType.MISSING_MODULE, IssueType.MISSING_PACKAGE):
            self._attempt_dependency_fix(issue)
            return

        # Get project root
        project_root = Path(__file__).parent.parent.parent

        try:
            # Handle special file creation commands
            if issue.fix_command.startswith("__CREATE_FILE__:"):
                parts = issue.fix_command.split(":", 2)
                if len(parts) == 3:
                    _, file_path, content = parts
                    full_path = project_root / file_path

                    logger.info(f"[AUTO-FIX] Creating file: {file_path}")

                    # Ensure directory exists
                    full_path.parent.mkdir(parents=True, exist_ok=True)

                    # Write content
                    with open(full_path, "w", encoding="utf-8") as f:
                        f.write(content)

                    logger.info(f"[AUTO-FIX] SUCCESS: Created {file_path}")
                    issue.fixed = True
                    self.issues_fixed.append(issue)
                    return

            # Regular shell command (pip install, etc.)
            # For pip commands, use conda run to ensure correct environment
            command = issue.fix_command
            if command.startswith("pip install"):
                import sys

                package = command.replace("pip install ", "")
                python_exe = sys.executable
                command = f'"{python_exe}" -m pip install {package}'
                logger.info(f"[AUTO-FIX] Using current interpreter pip: {command}")

            logger.info(f"[AUTO-FIX] Executing: {command}")
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=120,  # 2 minute timeout for pip install
                cwd=project_root,
            )

            if result.returncode == 0:
                logger.info(f"[AUTO-FIX] SUCCESS: {issue.fix_command}")
                issue.fixed = True
                self.issues_fixed.append(issue)
            else:
                logger.error(f"[AUTO-FIX] FAILED: {result.stderr}")
                # Route to CodeAgent for deeper analysis
                self._escalate_to_agent(issue)

        except subprocess.TimeoutExpired:
            logger.error(f"[AUTO-FIX] TIMEOUT: {issue.fix_command}")
        except Exception as e:
            logger.error(f"[AUTO-FIX] ERROR: {e}")

    def _attempt_dependency_fix(self, issue: DetectedIssue) -> None:
        """Resolve missing dependencies using agent + heuristics (no hard-coded one-offs)."""
        from autotrade.utils.dependency_resolver import resolve_and_install

        try:
            result = resolve_and_install(
                module_name=issue.module_name,
                error_text=issue.message,
                task_router=self.task_router,
                logger=logger,
                allow_agent=True,
            )
        except Exception as e:
            logger.error(f"[AUTO-FIX] Dependency resolver failed: {e}")
            self._escalate_to_agent(issue)
            return

        if not result.handled:
            logger.error("[AUTO-FIX] Dependency resolver did not recognize error")
            self._escalate_to_agent(issue)
            return

        if result.success:
            logger.info(f"[AUTO-FIX] SUCCESS: Installed {result.package}")
            issue.fixed = True
            self.issues_fixed.append(issue)
            return

        logger.error(
            f"[AUTO-FIX] FAILED: Dependency install attempts failed: {result.attempted}"
        )
        self._escalate_to_agent(issue)

    def _fix_code_error(self, issue: DetectedIssue):
        """
        Fix a code error using the CodeAgent repair cascade.

        Attempt 1: CodeAgent uses the default cloud-first repair cascade.
        Attempt 2: If no changes could be applied (search text not found in file,
                   malformed JSON, etc.), escalate to a stronger OpenRouter-only
                   repair cascade.

        qwen3-coder-next is opt-in only and is appended on escalated retries
        only when explicitly enabled.
        """
        if not self.task_router:
            logger.warning("[CODE-FIX] No TaskRouter available - cannot auto-fix")
            return

        resolved_file = self._resolve_issue_file_path(issue)
        if resolved_file is None:
            logger.warning(f"[CODE-FIX] Cannot fix - file not found: {issue.file_path}")
            return
        issue.file_path = str(resolved_file)

        # Memory guard: skip auto-fix if system RAM is critically low
        try:
            import psutil

            mem = psutil.virtual_memory()
            if mem.percent > 85:
                logger.warning(
                    f"[CODE-FIX] SKIPPING auto-fix - RAM at {mem.percent}% ({mem.available / (1024**3):.1f} GB free)"
                )
                logger.warning(
                    "[CODE-FIX] Fix the error manually to avoid loading large models"
                )
                return
        except ImportError:
            pass  # psutil not available, skip check

        logger.info("=" * 60)
        logger.info("[CODE-FIX] INITIATING AUTOMATIC CODE REPAIR")
        logger.info(f"[CODE-FIX] Error: {issue.error_type}: {issue.message}")
        logger.info(f"[CODE-FIX] File: {issue.file_path}:{issue.line_number}")
        logger.info("=" * 60)

        # Build full traceback string for context
        traceback_str = (
            "\n".join(issue.traceback_lines) if issue.traceback_lines else issue.message
        )

        # Try up to 2 attempts: first normal, then an escalated repair cascade.
        for attempt, force_escalation in enumerate([False, True], start=1):
            if force_escalation:
                logger.info(
                    "[CODE-FIX] Attempt 1 produced no applicable changes — escalating to stronger cloud repair models"
                )

            applied_count, fix_data = self._run_code_fix_attempt(
                issue=issue,
                traceback_str=traceback_str,
                force_escalation=force_escalation,
                attempt=attempt,
            )

            if applied_count > 0:
                break  # Changes were applied — proceed to validation below
            if attempt == 1:
                logger.warning(
                    "[CODE-FIX] No changes applied on first attempt, will escalate"
                )
            else:
                logger.error(
                    "[CODE-FIX] No changes could be applied even after escalation"
                )
                return

        if not fix_data:
            logger.error("[CODE-FIX] fix_data missing after apply loop — aborting")
            return

        # Comprehensive Validation (Phase 7)
        success, message = self._validate_code_fix(issue.file_path)

        if success:
            logger.info(f"[CODE-FIX] SUCCESS - {message}")
            issue.fixed = True
            self.issues_fixed.append(issue)

            # Log the fix for changelog
            self._log_code_fix(issue, fix_data)
        else:
            logger.error(f"[CODE-FIX] VALIDATION FAILED: {message}")
            logger.info("[CODE-FIX] Restoring from backup...")

            backup_path = Path(issue.file_path).with_suffix(".py.bak")
            if backup_path.exists():
                try:
                    with open(backup_path, "r", encoding="utf-8") as f:
                        original_content = f.read()
                    with open(issue.file_path, "w", encoding="utf-8") as out:
                        out.write(original_content)
                    logger.info("[CODE-FIX] Restored original file")
                except Exception as e:
                    logger.error(f"[CODE-FIX] Restore failed: {e}")

    def _validate_code_fix(self, file_path: str) -> Tuple[bool, str]:
        """
        Comprehensive validation of a code fix:
        1. py_compile (Syntax check)
        2. ruff check (Lint check)
        3. pytest (Opt-in targeted check for live repair)
        """
        logger.info(f"[CODE-FIX] Validating fix for {file_path}...")

        # 1. Syntax check (Fastest, catch trivial mistakes)
        try:
            import py_compile

            # doraise=True makes it raise PyCompileError on failure
            py_compile.compile(file_path, doraise=True)
            logger.info("   [1/3] py_compile: PASS")
        except py_compile.PyCompileError as e:
            # Extract just the error message to keep log clean
            clean_err = str(e).split(":", 1)[-1].strip() if ":" in str(e) else str(e)
            return False, f"SyntaxError: {clean_err}"
        except Exception as e:
            return False, f"py_compile error: {e}"

        # 2. Ruff check (Catch logical lint errors like undefined names or bad indents)
        # We select E (Error) and F (Fatal/Pyflakes) rules.
        try:
            # We use subprocess with encoding='utf-8' per mandate
            # --no-cache ensures we don't get stale results
            res = subprocess.run(
                ["ruff", "check", file_path, "--select", "E,F", "--no-cache"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=20,
            )
            if res.returncode != 0:
                # Capture the first few lines of ruff output for the error message
                err_lines = [line for line in res.stdout.split("\n") if line.strip()]
                summary = err_lines[0] if err_lines else "Unknown lint error"
                logger.warning(f"   [2/3] ruff: FAIL - {summary}")
                return False, f"LintError: {summary}"
            logger.info("   [2/3] ruff: PASS")
        except subprocess.TimeoutExpired:
            logger.warning("   [2/3] ruff: TIMEOUT - skipping")
        except Exception as e:
            logger.warning(f"   [2/3] ruff: ERROR - {e}")

        run_live_pytest = os.environ.get(
            "AUTOTRADE_LIVE_REPAIR_PYTEST", "0"
        ).lower() in {"1", "true", "yes", "on"}
        if not run_live_pytest:
            logger.info("   [3/3] pytest: SKIP (disabled for live repair)")
            return True, "py_compile and ruff passed"

        # 3. Pytest check (Functional check)
        # We try to find a test file related to this module to avoid running full suite
        try:
            # Heuristic 1: tests/test_<module_name>.py
            # Heuristic 2: tests/<module_path>/test_<module_name>.py
            file_p = Path(file_path)
            module_name = file_p.stem

            # Common test locations
            test_candidates = [
                PROJECT_ROOT / "tests" / f"test_{module_name}.py",
                PROJECT_ROOT / "tests" / file_p.parent.name / f"test_{module_name}.py",
            ]

            test_file = next((t for t in test_candidates if t.exists()), None)

            if test_file:
                logger.info(f"   [3/3] Running targeted test: {test_file.name}")
                # Set PYTHONIOENCODING=utf-8 to avoid RecursionError on Windows terminals
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"

                res = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        str(test_file),
                        "-v",
                        "--no-header",
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    env=env,
                    timeout=60,
                )
                if res.returncode != 0:
                    # Capture the failing test line if possible
                    fail_summary = "Tests failed"
                    for line in res.stdout.split("\n"):
                        if "FAILED" in line and ":" in line:
                            fail_summary = line.strip()
                            break
                    logger.warning(f"   [3/3] pytest: FAIL - {fail_summary}")
                    return False, f"TestError: {fail_summary}"
                logger.info("   [3/3] pytest: PASS")
            else:
                logger.info(
                    f"   [3/3] pytest: SKIP (no targeted test found for {module_name})"
                )
        except subprocess.TimeoutExpired:
            logger.warning("   [3/3] pytest: TIMEOUT - skipping functional validation")
        except Exception as e:
            logger.warning(f"   [3/3] pytest: ERROR - {e}")

        return True, "All checks passed"

    def _run_code_fix_attempt(
        self,
        issue: DetectedIssue,
        traceback_str: str,
        force_escalation: bool,
        attempt: int,
    ):
        """
        Run a single code-fix attempt via the TaskRouter / CodeAgent.

        Returns:
            (applied_count: int, fix_data: dict | None)
        """
        try:
            from autotrade.core.agentic_orchestrator import Task, TaskType
            import json as json_module

            task = Task(
                type=TaskType.CODE_FIX,
                description=f"Auto-fix {issue.error_type} in {Path(issue.file_path).name} (attempt {attempt})",
                data={
                    "file": issue.file_path,
                    "error": traceback_str,
                    "line": issue.line_number,
                    "error_type": issue.error_type or "",
                    "mode": "auto_fix",
                    "force_escalation": force_escalation,
                },
                priority=1,
            )

            label = "ESCALATED " if force_escalation else ""
            logger.info(f"[CODE-FIX] {label}Routing attempt {attempt} to CodeAgent...")
            result = self.task_router.route(task)

            if not result.success:
                logger.error(
                    f"[CODE-FIX] CodeAgent failed (attempt {attempt}): {result.message}"
                )
                return 0, None

            response = result.message
            if not response:
                logger.error(
                    f"[CODE-FIX] CodeAgent returned empty response (attempt {attempt})"
                )
                return 0, None

            # Strip markdown fences if present
            json_str = response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in json_str:
                json_str = json_str.split("```", 1)[1].split("```", 1)[0]

            try:
                fix_data = json_module.loads(json_str.strip())
            except json_module.JSONDecodeError as e:
                logger.error(
                    f"[CODE-FIX] Could not parse CodeAgent response as JSON (attempt {attempt}): {e}"
                )
                logger.error(f"[CODE-FIX] Response was: {response[:500]}...")
                return 0, None

            changes = fix_data.get("changes", [])
            if not changes:
                already_applied = int(result.data.get("changes_applied", 0) or 0)
                if already_applied > 0:
                    logger.info(
                        f"[CODE-FIX] CodeAgent already applied {already_applied} change(s) "
                        f"internally via {result.data.get('model', result.model_used or 'unknown')}"
                    )
                    if result.data.get("fix_details"):
                        fix_data["changes"] = result.data["fix_details"]
                    return already_applied, fix_data

                logger.warning(
                    f"[CODE-FIX] CodeAgent returned no changes (attempt {attempt})"
                )
                return 0, None

            model_used = result.data.get("model", result.model_used or "unknown")
            logger.info(f"[CODE-FIX] {len(changes)} change(s) proposed by {model_used}")
            logger.info(f"[CODE-FIX] Summary: {fix_data.get('summary', 'No summary')}")

            # Apply changes
            applied_count = 0
            for i, change in enumerate(changes):
                file_path = change.get("file", issue.file_path)
                search = change.get("search", "")
                replace = change.get("replace", "")
                reason = change.get("reason", "No reason given")

                if not search:
                    logger.warning(
                        f"[CODE-FIX] Change {i + 1} has no search text, skipping"
                    )
                    continue

                logger.info(f"[CODE-FIX] Applying change {i + 1}: {reason}")

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    logger.error(f"[CODE-FIX] Could not read file: {e}")
                    continue

                if search not in content:
                    logger.warning(
                        f"[CODE-FIX] Search text not found in file (change {i + 1}), skipping"
                    )
                    logger.debug(f"[CODE-FIX] Searched for: {search[:120]}...")
                    continue

                occurrences = content.count(search)
                if occurrences > 1:
                    logger.warning(
                        f"[CODE-FIX] Search text appears {occurrences} times — need unique match, skipping"
                    )
                    continue

                # Backup
                backup_path = Path(file_path).with_suffix(".py.bak")
                try:
                    with open(backup_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    logger.info(f"[CODE-FIX] Backup saved to {backup_path.name}")
                except Exception as e:
                    logger.warning(f"[CODE-FIX] Could not create backup: {e}")

                new_content = content.replace(search, replace, 1)

                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    logger.info(
                        f"[CODE-FIX] Change {i + 1} applied to {Path(file_path).name}"
                    )
                    applied_count += 1
                except Exception as e:
                    logger.error(f"[CODE-FIX] Could not write file: {e}")
                    if backup_path.exists():
                        with open(backup_path, "r", encoding="utf-8") as f:
                            with open(file_path, "w", encoding="utf-8") as out:
                                out.write(f.read())
                        logger.info(
                            "[CODE-FIX] Restored from backup after write failure"
                        )
                    continue

            return applied_count, fix_data

        except Exception as e:
            logger.error(f"[CODE-FIX] Unexpected error in attempt {attempt}: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return 0, None

    def _log_code_fix(self, issue: DetectedIssue, fix_data: Dict[str, Any]):
        """Log a successful code fix for changelog tracking."""
        project_root = Path(__file__).parent.parent.parent
        fixes_log = project_root / "logs" / "code_fixes.jsonl"

        try:
            import json as json_module

            entry = {
                "timestamp": datetime.now().isoformat(),
                "error_type": issue.error_type,
                "file": issue.file_path,
                "line": issue.line_number,
                "summary": fix_data.get("summary", ""),
                "changes": len(fix_data.get("changes", [])),
            }

            with open(fixes_log, "a", encoding="utf-8") as f:
                f.write(json_module.dumps(entry) + "\n")

            logger.info(f"[CODE-FIX] Fix logged to {fixes_log}")

        except Exception as e:
            logger.warning(f"[CODE-FIX] Could not log fix: {e}")

    def _escalate_to_agent(self, issue: DetectedIssue):
        """Escalate a complex issue to a specialized agent for analysis (not auto-fix)."""
        if not self.task_router:
            logger.warning("[ESCALATE] No TaskRouter available")
            return

        try:
            from autotrade.core.agentic_orchestrator import Task, TaskType

            # Create a task for the CodeAgent
            task = Task(
                type=TaskType.CODE_FIX,
                description=f"Fix issue: {issue.message}",
                data={
                    "issue_type": issue.issue_type.value,
                    "message": issue.message,
                    "module": issue.module_name,
                    "file": issue.file_path,
                },
            )

            logger.info("[ESCALATE] Routing to specialized agent...")
            result = self.task_router.route(task)

            if result.success:
                # Agent provided analysis/suggestion but NOT auto-applied
                # Store suggestion for user prompt
                issue.fix_command = f"__AGENT_SUGGESTION__:{result.message}"
                issue.auto_fixable = False  # Needs user confirmation
                logger.info(f"[ESCALATE] Agent analysis: {result.message[:200]}...")
            else:
                logger.warning(f"[ESCALATE] Agent could not fix: {result.message}")

        except Exception as e:
            logger.error(f"[ESCALATE] Failed: {e}")

    def run_task(
        self, task_name: str, args: List[str] = None
    ) -> Tuple[bool, List[DetectedIssue]]:
        """
        Run a task with full supervision and real-time monitoring.

        Args:
            task_name: The task to run (e.g., 'charts', 'overnight', 'continuous')
            args: Additional arguments to pass

        Returns:
            Tuple of (success, list of unfixed issues)
        """
        args = args or []
        # Per-task accounting: avoid carrying issue counts across separate task runs.
        self.issues_detected = []
        self.issues_fixed = []

        # Build command - extremely robust python executable resolution (Phase 6)
        raw_sys_exe = str(sys.executable)
        logger.info(f"[DEBUG] Raw sys.executable: {raw_sys_exe}")

        raw_exe = raw_sys_exe.strip().strip('"').strip("'")
        python_exe = None

        # Aggressively remove any non-path prefixes (like @)
        clean_exe = re.sub(r"^[^a-zA-Z0-9\/\\]+", "", raw_exe)

        # Priority 1: Cleaned sys.executable
        clean_exe_abs = os.path.abspath(clean_exe)
        if os.path.exists(clean_exe_abs):
            python_exe = clean_exe_abs

        # Priority 2: sys.prefix (Conda/Venv root)
        if not python_exe:
            prefix_exe = os.path.join(sys.prefix, "python.exe")
            if os.path.exists(prefix_exe):
                python_exe = prefix_exe

        # Priority 3: shutil.which
        if not python_exe:
            import shutil

            which_exe = shutil.which("python")
            if which_exe:
                python_exe = which_exe

        # Final fallback: hardcoded stocks-gpu env
        if not python_exe:
            home = os.path.expanduser("~")
            fallback = os.path.join(
                home, "miniconda3", "envs", "stocks-gpu", "python.exe"
            )
            if os.path.exists(fallback):
                python_exe = fallback

        # If all fails, use the regex-cleaned one
        if not python_exe:
            python_exe = clean_exe

        # ULTIMATE SAFETY: Ensure python_exe is a clean string (Phase 6)
        python_exe = str(python_exe).split("@")[-1].strip().strip('"').strip("'")

        logger.info(f"[TASK] FINAL Normalized Python Executable: {python_exe}")

        # HOOK: Run financial DB update before overnight process
        if task_name == "overnight":
            logger.info("[HOOK] Running nightly financial DB update...")

            # Determine mode: Held-only on weeknights, Top-50 on weekends (Fri-Sun)
            is_weekend = datetime.now().weekday() >= 4
            update_args = ["--top", "50"] if is_weekend else ["--held-only"]

            if is_weekend:
                logger.info("   [WEEKEND MODE] Updating top 50 tickers")
            else:
                logger.info("   [WEEKNIGHT MODE] Updating held positions only")

            update_cmd = [python_exe, "tools/update_financial_db.py"] + update_args
            # Run with timeout to prevent hang
            try:
                update_res = subprocess.run(
                    update_cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    cwd=PROJECT_ROOT,
                    timeout=600,
                )
                if update_res.returncode == 0:
                    logger.info("[HOOK] Financial DB update completed successfully.")
                else:
                    logger.warning(
                        f"[HOOK] Financial DB update failed: {update_res.stderr}"
                    )
            except Exception as e:
                logger.error(f"[HOOK] Financial DB update exception: {e}")

        if task_name == "continuous-guarded":
            return self._run_guarded_continuous(python_exe, args)

        command = self._build_task_command(python_exe, task_name, args)

        logger.info("=" * 60)
        logger.info(f"[TASK] Starting supervised task: {task_name}")
        logger.info(f"[TASK] Command: {' '.join(command)}")
        logger.info("=" * 60)

        return self._run_monitored_command(task_name, command)

    def _maybe_run_weekly_signal_audit(self) -> None:
        cfg = None
        try:
            cfg = get_config()
        except Exception as e:
            logger.warning(f"[WEEKLY AUDIT] Cannot load config: {e}")
            return

        if not getattr(cfg.backtest, "enable_weekly_signal_audit", True):
            logger.info("[WEEKLY AUDIT] Disabled via config")
            return

        now = datetime.now()
        if not self._is_weekly_window(now):
            return

        week_end = self._compute_week_end(now.date())
        json_path = LOG_DIR / f"weekly_signal_audit_{week_end:%Y%m%d}.json"
        if json_path.exists():
            return

        week_start = week_end - timedelta(days=4)
        payload = self._run_weekly_signal_audit_cli(week_start, week_end)
        if not payload:
            return

        self._write_weekly_audit_outputs(cfg, week_start, week_end, payload)

    @staticmethod
    def _is_weekly_window(now: datetime) -> bool:
        weekday = now.weekday()
        if weekday == 4:  # Friday
            return now.hour >= 21  # wait until after close
        return weekday in (5, 6, 0)  # weekend or early Monday catch-up

    @staticmethod
    def _compute_week_end(today: date) -> date:
        days_since_friday = (today.weekday() - 4) % 7
        return (
            today - timedelta(days=days_since_friday or 7)
            if today.weekday() < 4
            else today - timedelta(days=days_since_friday)
        )

    def _run_weekly_signal_audit_cli(
        self, start: date, end: date
    ) -> Optional[Dict[str, Any]]:
        cmd = [
            sys.executable,
            "tools/weekly_signal_audit.py",
            "--json",
            "--start",
            start.isoformat(),
            "--end",
            end.isoformat(),
        ]
        logger.info(f"[WEEKLY AUDIT] Running weekly_signal_audit for {start} to {end}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=90,
            cwd=PROJECT_ROOT,
        )
        if result.returncode != 0:
            logger.warning(
                f"[WEEKLY AUDIT] CLI failed (code {result.returncode}): {result.stderr.strip()}"
            )
            return None
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            logger.warning("[WEEKLY AUDIT] Could not parse JSON output from audit tool")
            return None

    def _write_weekly_audit_outputs(
        self, cfg, start: date, end: date, payload: Dict[str, Any]
    ) -> None:
        slug = end.strftime("%Y%m%d")
        json_path = LOG_DIR / f"weekly_signal_audit_{slug}.json"
        txt_path = LOG_DIR / f"weekly_signal_audit_{slug}.txt"

        try:
            json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"[WEEKLY AUDIT] Failed to write JSON: {e}")

        try:
            summary_text = self._render_weekly_audit_text(payload, start, end)
            txt_path.write_text(summary_text, encoding="utf-8")
        except Exception as e:
            logger.warning(f"[WEEKLY AUDIT] Failed to write text summary: {e}")

        log_filename = getattr(cfg.logging, "json_filename", "app.jsonl")
        app_log = LOG_DIR / log_filename
        validation = payload.get("validation", {})
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "weekly_signal_audit",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "win_rate": payload.get("summary", {}).get("win_rate"),
            "avg_ret_5d": payload.get("summary", {}).get("avg_ret_5d"),
            "validated_win_rate": validation.get("validated", {}).get("win_rate"),
            "rejected_win_rate": validation.get("rejected", {}).get("win_rate"),
            "missed_count": len(payload.get("missed_opportunities", [])),
        }
        try:
            with open(app_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"[WEEKLY AUDIT] Failed to append app log: {e}")

    @staticmethod
    def _render_weekly_audit_text(
        payload: Dict[str, Any], start: date, end: date
    ) -> str:
        summary = payload.get("summary", {})
        day_rows = payload.get("by_day", [])
        validation = payload.get("validation", {})
        missed = payload.get("missed_opportunities", [])

        best_night = (
            max(day_rows, key=lambda r: r.get("win_rate", 0)) if day_rows else None
        )
        worst_night = (
            min(day_rows, key=lambda r: r.get("win_rate", 0)) if day_rows else None
        )

        def fmt_pct(val: Optional[float]) -> str:
            return f"{val:+.2f}%" if val is not None else "n/a"

        lines = [
            f"WEEKLY SIGNAL AUDIT {start.isoformat()} -> {end.isoformat()}",
            f"Signals: {summary.get('total_signals', 0)} (priced: {summary.get('signals_with_data', 0)})",
            f"Win rate (>=2% 5D): {summary.get('win_rate', 0):.1f}%",
            f"Avg 5D return: {fmt_pct(summary.get('avg_ret_5d'))}",
            f"Max gain/loss avg (5D): {fmt_pct(summary.get('avg_max_gain'))} / {fmt_pct(summary.get('avg_max_loss'))}",
        ]

        if best_night:
            lines.append(
                f"Best night: {best_night.get('date')} ({best_night.get('win_rate', 0):.1f}% win)"
            )
        if worst_night:
            lines.append(
                f"Worst night: {worst_night.get('date')} ({worst_night.get('win_rate', 0):.1f}% win)"
            )

        validated = validation.get("validated", {})
        rejected = validation.get("rejected", {})
        if validated or rejected:
            lines.append(
                "Validated vs rejected win rate: "
                f"{validated.get('win_rate', 0.0):.1f}% / {rejected.get('win_rate', 0.0):.1f}%"
            )

        lines.append(f"Missed top movers: {len(missed)}")
        return "\n".join(lines)

    def _extract_plan_candidates(
        self, plan_data: Dict[str, Any]
    ) -> Tuple[List[Any], str]:
        """Return the best available candidate list from a plan payload."""
        if not isinstance(plan_data, dict):
            return [], "none"

        for key in (
            "signals",
            "buy_signals",
            "entry_candidates",
            "entry_orders",
            "actionable_top50",
            "full_watchlist",
        ):
            rows = plan_data.get(key)
            if isinstance(rows, list) and rows:
                return rows, key
        return [], "none"

    def _validate_task_results(self, task_name: str) -> Dict[str, Any]:
        """
        Validate that a task actually produced the expected results.

        This is the CRITICAL piece that was missing - we need to verify
        the task actually worked, not just that it exited cleanly.
        """
        from pathlib import Path
        from datetime import datetime
        import json
        from autotrade.utils.market_time import get_pm_plan_date

        project_root = Path(__file__).parent.parent.parent
        plans_dir = project_root / "plans"
        logs_dir = project_root / "logs"
        plan_date = get_pm_plan_date(datetime.now())
        today = plan_date.strftime("%Y%m%d")

        validation = {
            "success": True,
            "task": task_name,
            "checks": [],
            "problem": None,
        }

        if task_name in ["overnight", "research"]:
            # Overnight research should produce signals
            morning_plan = plans_dir / f"morning_game_plan_{today}.json"

            if morning_plan.exists():
                try:
                    with open(morning_plan) as f:
                        plan = json.load(f)
                    signals, signal_source = self._extract_plan_candidates(plan)
                    if len(signals) == 0:
                        validation["success"] = False
                        validation["problem"] = "overnight_research_no_signals"
                        validation["checks"].append(
                            "Morning plan exists but has 0 signals"
                        )
                    else:
                        validation["checks"].append(
                            f"Morning plan has {len(signals)} candidates via {signal_source} - OK"
                        )
                except Exception as e:
                    validation["checks"].append(f"Could not read morning plan: {e}")
            else:
                validation["success"] = False
                validation["problem"] = "overnight_no_plan_generated"
                validation["checks"].append("No morning_game_plan for today")

        elif task_name in ["continuous", "execute"]:
            # Trading tasks should execute trades
            today_date = datetime.now().strftime("%Y-%m-%d")
            today_compact = datetime.now().strftime("%Y%m%d")
            yesterday_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

            execution_marker = plans_dir / f".execution_marker_{today}"
            execution_artifacts = self._find_execution_artifacts(
                logs_dir=logs_dir,
                today_compact=today_compact,
                today_dashed=today_date,
            )

            # MODERN DISCOVERY: Check for adjusted_plan or morning_game_plan first
            # These are now the canonical artifacts for DayManager/Execution
            adjusted_plans = sorted(plans_dir.glob(f"adjusted_plan_{today}_*.json"))
            morning_plan = plans_dir / f"morning_game_plan_{today}.json"

            active_plan_file = None
            if adjusted_plans:
                active_plan_file = adjusted_plans[-1]  # Latest adjusted plan
                validation["checks"].append(
                    f"Found adjusted plan: {active_plan_file.name}"
                )
            elif morning_plan.exists():
                active_plan_file = morning_plan
                validation["checks"].append(f"Found morning plan: {morning_plan.name}")

            signals = []
            signal_source = "none"

            if active_plan_file:
                try:
                    with open(active_plan_file) as f:
                        plan_data = json.load(f)
                    signals, signal_source = self._extract_plan_candidates(plan_data)
                except Exception as e:
                    validation["checks"].append(
                        f"Could not read plan file {active_plan_file.name}: {e}"
                    )

            # LEGACY FALLBACK: Check logs/signals_*.json
            if not signals:
                signals_file = logs_dir / f"signals_{today_date}.json"
                signals_file_yesterday = logs_dir / f"signals_{yesterday_date}.json"

                target_signal_file = None
                if signals_file.exists():
                    target_signal_file = signals_file
                elif signals_file_yesterday.exists():
                    target_signal_file = signals_file_yesterday
                    validation["checks"].append(
                        f"Using yesterday's signals file: {signals_file_yesterday.name}"
                    )

                if target_signal_file:
                    try:
                        with open(target_signal_file) as f:
                            loaded = json.load(f)
                        if isinstance(loaded, dict):
                            signals = loaded.get("signals", [])
                        elif isinstance(loaded, list):
                            signals = loaded
                        signal_source = target_signal_file.name
                    except Exception as e:
                        validation["checks"].append(
                            f"Could not read legacy signals file {target_signal_file.name}: {e}"
                        )

            if signals:
                validation["checks"].append(
                    f"Signals file has {len(signals)} candidates via {signal_source} - OK"
                )

                # Check if execution actually happened
                if not execution_marker.exists():
                    if execution_artifacts:
                        artifact_names = ", ".join(
                            path.name for path in execution_artifacts
                        )
                        validation["checks"].append(
                            f"Execution artifacts exist - execution ran ({artifact_names})"
                        )
                    else:
                        validation["success"] = False
                        validation["problem"] = "signals_exist_but_no_execution"
                        validation["checks"].append(
                            "Signals exist but no execution attempted!"
                        )
            else:
                validation["success"] = False
                validation["problem"] = "no_signals_file"
                validation["checks"].append(
                    f"No active signals or plan files found for {today}"
                )

        elif task_name == "charts":
            # Charts task should generate chart images
            charts_dir = project_root / "charts"
            charts_index = charts_dir / "charts_index.json"

            if charts_index.exists():
                try:
                    with open(charts_index) as f:
                        index = json.load(f)
                    if len(index) > 0:
                        validation["checks"].append(
                            f"Charts index has {len(index)} entries - OK"
                        )
                    else:
                        validation["success"] = False
                        validation["problem"] = "charts_index_empty"
                except Exception:
                    pass

        # Log validation results
        for check in validation["checks"]:
            logger.info(f"[VALIDATE] {check}")

        return validation

    def _diagnose_and_fix(self, task_name: str, validation_result: Dict):
        """
        Use DiagnosticAgent from TaskRouter to analyze and fix workflow problems.

        THIS is where we actually use the agent infrastructure we built!
        """
        if not self.task_router:
            logger.warning("[DIAGNOSE] TaskRouter not available - cannot auto-diagnose")
            return

        problem = validation_result.get("problem", "unknown")
        checks = validation_result.get("checks", [])

        logger.info(f"[DIAGNOSE] Routing '{problem}' to DiagnosticAgent...")

        try:
            # Import Task and TaskType to create proper task object
            from autotrade.core.agentic_orchestrator import Task, TaskType

            # Create proper Task object for the diagnostic agent
            task = Task(
                type=TaskType.HEALTH_CHECK,
                description=f"Task '{task_name}' completed but validation failed: {problem}",
                data={
                    "problem": problem,
                    "task_name": task_name,
                    "checks": checks,
                    "validation_result": validation_result,
                },
                priority=1,  # High priority - workflow failed!
            )

            result = self.task_router.route(task)

            if result.success:
                logger.info(f"[DIAGNOSE] Agent analysis: {result.message}")

                # If agent suggests a fix, apply it
                if result.data.get("fix_applied"):
                    logger.info("[DIAGNOSE] Fix was applied automatically")

                # If agent identified specific fix, try to apply it
                if result.data.get("suggested_fix"):
                    self._apply_suggested_fix(result.data["suggested_fix"], problem)
            else:
                logger.warning(f"[DIAGNOSE] Agent could not diagnose: {result.message}")

        except Exception as e:
            logger.error(f"[DIAGNOSE] Diagnosis failed: {e}")

    def _apply_suggested_fix(self, fix: Dict, problem: str):
        """Apply a fix suggested by the diagnostic agent."""
        fix_type = fix.get("type")

        if fix_type == "reload_signals":
            # Re-run signal loading
            logger.info("[FIX] Reloading signals from morning_game_plan...")
            try:
                from autotrade.core.day_manager import DayManager

                dm = DayManager(dry_run=True)  # Just load signals, don't trade
                count = len(dm.signals)
                logger.info(f"[FIX] Loaded {count} signals")
            except Exception as e:
                logger.error(f"[FIX] Failed to reload signals: {e}")

        elif fix_type == "convert_signals":
            # Convert signals to entry_orders format
            logger.info("[FIX] Converting signals to entry_orders...")
            try:
                from autotrade.core.autonomous_agent import TomorrowsPlanGenerator

                tpg = TomorrowsPlanGenerator()
                plan = tpg._load_latest_plan()
                if plan:
                    logger.info(
                        f"[FIX] Plan loaded with {len(plan.get('entry_orders', []))} orders"
                    )
            except Exception as e:
                logger.error(f"[FIX] Conversion failed: {e}")

        elif fix_type == "trigger_execution":
            # Force trigger execution
            logger.info("[FIX] Would trigger execution (requires market hours)...")

        else:
            logger.warning(f"[FIX] Unknown fix type: {fix_type}")

    def validate_environment(self) -> Dict[str, Any]:
        """
        Validate the environment before running tasks.
        This catches issues like wrong conda env BEFORE spawning children.
        """
        logger.info("[VALIDATE] Checking environment...")

        results = {
            "python": sys.executable,
            "version": sys.version,
            "packages": {},
            "services": {},
            "issues": [],
        }

        # CRITICAL: Check if we're in the right conda environment FIRST
        expected_env = "stocks-gpu"
        current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
        python_path = sys.executable.lower()

        env_valid = expected_env in python_path or current_env == expected_env

        if not env_valid:
            logger.error("=" * 60)
            logger.error("WRONG CONDA ENVIRONMENT!")
            logger.error(f"  Expected: {expected_env}")
            logger.error(f"  Current:  {current_env or 'unknown'}")
            logger.error(f"  Python:   {sys.executable}")
            logger.error("")
            logger.error("FIX: Run these commands:")
            logger.error("  conda activate stocks-gpu")
            logger.error("  Then re-run your command")
            logger.error("=" * 60)
            # Exit immediately - do NOT try to auto-fix wrong env
            sys.exit(1)

        # Check critical packages (just detection, no auto-fix for core packages)
        critical_packages = [
            ("pytz", "pytz"),
            ("yfinance", "yfinance"),
            ("yaml", "pyyaml"),
            ("requests", "requests"),
            ("alpaca", "alpaca-py"),
            ("pandas", "pandas"),
            ("numpy", "numpy"),
        ]

        missing_count = 0
        for module_name, package_name in critical_packages:
            try:
                __import__(module_name)
                results["packages"][module_name] = "OK"
            except ImportError:
                results["packages"][module_name] = "MISSING"
                missing_count += 1

        # If multiple core packages missing, it's wrong env - don't auto-fix
        if missing_count >= 3:
            logger.error("=" * 60)
            logger.error(f"ENVIRONMENT PROBLEM: {missing_count} core packages missing!")
            logger.error("This usually means you're in the wrong conda environment.")
            logger.error("")
            logger.error("FIX: Run these commands:")
            logger.error("  conda activate stocks-gpu")
            logger.error("  Then re-run your command")
            logger.error("=" * 60)
            sys.exit(1)

        # Check services (no auto-fix needed)
        try:
            import requests

            # Ollama
            try:
                resp = requests.get("http://localhost:11434/api/tags", timeout=2)
                results["services"]["ollama"] = (
                    "OK" if resp.status_code == 200 else "ERROR"
                )
            except Exception:
                results["services"]["ollama"] = "NOT RUNNING"

            # SearXNG
            try:
                resp = requests.get("http://localhost:8080/healthz", timeout=2)
                results["services"]["searxng"] = (
                    "OK" if resp.status_code == 200 else "ERROR"
                )
            except Exception:
                results["services"]["searxng"] = "NOT RUNNING"

        except ImportError:
            pass

        # Log results
        logger.info("[VALIDATE] Package status:")
        for pkg, status in results["packages"].items():
            icon = "[OK]" if status == "OK" else "[MISSING]"
            logger.info(f"  {icon} {pkg}: {status}")

        logger.info("[VALIDATE] Service status:")
        for svc, status in results["services"].items():
            icon = "[OK]" if status == "OK" else "[DOWN]"
            logger.info(f"  {icon} {svc}: {status}")

        return results

    def run_supervised_menu_task(self, task_name: str, args: List[str] = None):
        """
        Entry point for batch file integration.
        Validates environment, runs task with supervision, reports results.
        """
        # First validate environment
        validation = self.validate_environment()

        if validation["issues"] and not self.dry_run:
            # Wait for fixes to complete
            time.sleep(2)
            # Re-validate
            validation = self.validate_environment()

        # Run the task
        exit_code, unfixed = self.run_task(task_name, args)

        # Report and offer to fix
        if exit_code == 0 and not unfixed:
            logger.info("[RESULT] Task completed successfully!")
        elif unfixed:
            self._offer_to_fix(unfixed, task_name, args)
        elif exit_code == 64:
            logger.info("[RESULT] Task reached automated deadline")
        else:
            logger.error(f"[RESULT] Task failed with exit code {exit_code}")

        return exit_code

    def _offer_to_fix(
        self, unfixed: List[DetectedIssue], task_name: str, args: List[str]
    ):
        """
        Offer to fix detected issues with 10-second auto-timeout.
        No response = Yes (fix automatically).
        """
        fixable = [i for i in unfixed if i.auto_fixable]
        non_fixable = [i for i in unfixed if not i.auto_fixable]

        print("\n" + "=" * 60)
        print(" ISSUES DETECTED - SELF-HEALING AVAILABLE")
        print("=" * 60)

        if fixable:
            print(
                f"\n[AUTO-FIXABLE] {len(fixable)} issue(s) can be fixed automatically:"
            )
            for i, issue in enumerate(fixable, 1):
                print(f"  {i}. {issue.issue_type.value}: {issue.message[:60]}")
                if issue.fix_command:
                    print(f"     Fix: {issue.fix_command}")

        if non_fixable:
            print(
                f"\n[MANUAL] {len(non_fixable)} issue(s) require manual intervention:"
            )
            for i, issue in enumerate(non_fixable, 1):
                print(f"  {i}. {issue.issue_type.value}: {issue.message[:60]}")

        if not fixable:
            print("\nNo auto-fixable issues found.")
            return

        # Prompt with 10-second timeout (default to Y)
        print("\n" + "-" * 60)
        print("Attempt automatic fixes? (Y/n) [Auto-YES in 10 seconds...]")
        print("-" * 60)

        response = self._input_with_timeout("Fix now? ", timeout=10, default="Y")

        if response.upper() in ("Y", "YES", ""):
            logger.info("[SELF-HEAL] Applying fixes...")
            self._apply_fixes(fixable)

            # Offer to retry the task
            print("\nRetry task after fixes? (Y/n) [Auto-YES in 5 seconds...]")
            retry = self._input_with_timeout("Retry? ", timeout=5, default="Y")

            if retry.upper() in ("Y", "YES", ""):
                logger.info(f"[RETRY] Re-running {task_name}...")
                self.issues_detected = []
                self.issues_fixed = []
                success, new_unfixed = self.run_task(task_name, args)

                if success and not new_unfixed:
                    logger.info("[SELF-HEAL] SUCCESS! Task completed after fixes.")
                elif new_unfixed:
                    logger.warning(
                        f"[SELF-HEAL] Still have {len(new_unfixed)} issues after fixes"
                    )
        else:
            logger.info("[SELF-HEAL] Skipped by user")

    def _input_with_timeout(
        self, prompt: str, timeout: int = 10, default: str = "Y"
    ) -> str:
        """
        Get user input with a timeout. Returns default if no input received.
        Works on Windows using msvcrt.
        """
        import sys

        print(prompt, end="", flush=True)

        # Countdown display
        start_time = time.time()
        input_chars = []

        if sys.platform == "win32":
            import msvcrt

            while (time.time() - start_time) < timeout:
                # Check for keypress
                if msvcrt.kbhit():
                    char = msvcrt.getwch()
                    if char == "\r":  # Enter pressed
                        print()  # New line
                        return "".join(input_chars) if input_chars else default
                    elif char == "\x03":  # Ctrl+C
                        raise KeyboardInterrupt
                    else:
                        input_chars.append(char)
                        print(char, end="", flush=True)

                time.sleep(0.1)

            print(f"\n[TIMEOUT] No response - defaulting to '{default}'")
            return default
        else:
            # Fallback for non-Windows
            import select

            rlist, _, _ = select.select([sys.stdin], [], [], timeout)
            if rlist:
                return sys.stdin.readline().strip() or default
            else:
                print(f"\n[TIMEOUT] No response - defaulting to '{default}'")
                return default

    def _apply_fixes(self, issues: List[DetectedIssue]):
        """Apply fixes for a list of auto-fixable issues."""
        # Get project root
        project_root = Path(__file__).parent.parent.parent

        for issue in issues:
            if issue.fix_command:
                try:
                    # Handle special file creation commands
                    if issue.fix_command.startswith("__CREATE_FILE__:"):
                        parts = issue.fix_command.split(":", 2)
                        if len(parts) == 3:
                            _, file_path, content = parts
                            full_path = project_root / file_path

                            logger.info(f"[FIX] Creating file: {file_path}")

                            # Ensure directory exists
                            full_path.parent.mkdir(parents=True, exist_ok=True)

                            # Write content
                            with open(full_path, "w") as f:
                                f.write(content)

                            logger.info(f"[FIX] SUCCESS: Created {file_path}")
                            issue.fixed = True
                            self.issues_fixed.append(issue)
                            continue

                    # Regular shell command
                    logger.info(f"[FIX] Executing: {issue.fix_command}")
                    result = subprocess.run(
                        issue.fix_command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        timeout=120,
                        cwd=project_root,
                    )

                    if result.returncode == 0:
                        logger.info(f"[FIX] SUCCESS: {issue.fix_command}")
                        issue.fixed = True
                        self.issues_fixed.append(issue)
                    else:
                        logger.error(f"[FIX] FAILED: {result.stderr[:100]}")
                        # Try escalating to agent
                        self._escalate_to_agent(issue)

                except subprocess.TimeoutExpired:
                    logger.error(f"[FIX] TIMEOUT: {issue.fix_command}")
                except Exception as e:
                    logger.error(f"[FIX] ERROR: {e}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Master Supervisor - Real-time Process Monitor"
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Task to run (charts, overnight, continuous, etc.)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Don't actually apply fixes (report only)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Enable self-healing (auto-apply code fixes)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh overnight research (reset state)",
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate environment"
    )
    parser.add_argument("args", nargs="*", help="Additional arguments to pass to task")

    args = parser.parse_args()

    dry_run = not args.execute
    supervisor = MasterSupervisor(dry_run=dry_run)

    if args.validate_only:
        supervisor.validate_environment()
        return

    # Pass fresh flag if present
    task_args = args.args if args.args else []
    if args.fresh:
        task_args.append("--fresh")

    exit_code = supervisor.run_supervised_menu_task(args.task, task_args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
