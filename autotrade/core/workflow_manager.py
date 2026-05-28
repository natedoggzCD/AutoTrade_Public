"""
Workflow Manager - State machine with persistence, watchdog, and retries.

This module provides a durable orchestration layer that can run AutoTrade
workflows in a predictable, observable way without asyncio.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from autotrade.utils.market_time import get_pm_plan_date

try:
    from autotrade.monitoring.collector import get_metrics_collector

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


def _cpu_probe(n: int) -> int:
    """Deterministic CPU-bound probe for validation runs."""
    total = 0
    for i in range(n):
        total += (i * i) % 97
    return total


def _io_probe(label: str) -> Dict[str, Any]:
    """Deterministic IO-bound probe for validation runs."""
    return {"label": label, "status": "ok"}


class WorkflowPhase(Enum):
    INIT = "init"
    OVERNIGHT = "overnight"
    EARLY_PREMARKET = "early_premarket"
    PREMARKET = "premarket"
    MARKET_OPEN = "market_open"
    MARKET_HOURS = "market_hours"
    POWER_HOUR = "power_hour"
    POST_MARKET = "post_market"
    PM_WORKFLOW = "pm_workflow"
    IDLE = "idle"
    ERROR = "error"

    @classmethod
    def from_value(cls, value: Any) -> "WorkflowPhase":
        if isinstance(value, WorkflowPhase):
            return value
        if hasattr(value, "value"):
            value = value.value
        try:
            return cls(str(value))
        except Exception:
            return cls.IDLE


ALLOWED_TRANSITIONS: Dict[WorkflowPhase, List[WorkflowPhase]] = {
    WorkflowPhase.INIT: [
        WorkflowPhase.OVERNIGHT,
        WorkflowPhase.EARLY_PREMARKET,
        WorkflowPhase.PREMARKET,
        WorkflowPhase.MARKET_OPEN,
        WorkflowPhase.MARKET_HOURS,
        WorkflowPhase.POWER_HOUR,
        WorkflowPhase.POST_MARKET,
        WorkflowPhase.PM_WORKFLOW,
        WorkflowPhase.IDLE,
    ],
    WorkflowPhase.OVERNIGHT: [
        WorkflowPhase.EARLY_PREMARKET,
        WorkflowPhase.PREMARKET,
        WorkflowPhase.MARKET_OPEN,
        WorkflowPhase.MARKET_HOURS,
        WorkflowPhase.POWER_HOUR,
        WorkflowPhase.POST_MARKET,
        WorkflowPhase.PM_WORKFLOW,
        WorkflowPhase.OVERNIGHT,
        WorkflowPhase.IDLE,
    ],
    WorkflowPhase.EARLY_PREMARKET: [
        WorkflowPhase.PREMARKET,
        WorkflowPhase.MARKET_OPEN,
        WorkflowPhase.MARKET_HOURS,
        WorkflowPhase.POWER_HOUR,
        WorkflowPhase.POST_MARKET,
        WorkflowPhase.PM_WORKFLOW,
        WorkflowPhase.OVERNIGHT,
        WorkflowPhase.IDLE,
    ],
    WorkflowPhase.PREMARKET: [
        WorkflowPhase.MARKET_OPEN,
        WorkflowPhase.MARKET_HOURS,
        WorkflowPhase.POWER_HOUR,
        WorkflowPhase.POST_MARKET,
        WorkflowPhase.PM_WORKFLOW,
        WorkflowPhase.OVERNIGHT,
        WorkflowPhase.IDLE,
    ],
    WorkflowPhase.MARKET_OPEN: [
        WorkflowPhase.MARKET_HOURS,
        WorkflowPhase.POWER_HOUR,
        WorkflowPhase.POST_MARKET,
        WorkflowPhase.PM_WORKFLOW,
        WorkflowPhase.OVERNIGHT,
        WorkflowPhase.IDLE,
    ],
    WorkflowPhase.MARKET_HOURS: [
        WorkflowPhase.POWER_HOUR,
        WorkflowPhase.POST_MARKET,
        WorkflowPhase.PM_WORKFLOW,
        WorkflowPhase.OVERNIGHT,
        WorkflowPhase.IDLE,
    ],
    WorkflowPhase.POWER_HOUR: [
        WorkflowPhase.POST_MARKET,
        WorkflowPhase.PM_WORKFLOW,
        WorkflowPhase.OVERNIGHT,
        WorkflowPhase.IDLE,
    ],
    WorkflowPhase.POST_MARKET: [
        WorkflowPhase.PM_WORKFLOW,
        WorkflowPhase.OVERNIGHT,
        WorkflowPhase.IDLE,
    ],
    WorkflowPhase.PM_WORKFLOW: [
        WorkflowPhase.OVERNIGHT,
        WorkflowPhase.IDLE,
    ],
    WorkflowPhase.IDLE: [
        WorkflowPhase.OVERNIGHT,
        WorkflowPhase.EARLY_PREMARKET,
        WorkflowPhase.PREMARKET,
        WorkflowPhase.MARKET_OPEN,
        WorkflowPhase.MARKET_HOURS,
        WorkflowPhase.POWER_HOUR,
        WorkflowPhase.POST_MARKET,
        WorkflowPhase.PM_WORKFLOW,
        WorkflowPhase.IDLE,
    ],
    WorkflowPhase.ERROR: [
        WorkflowPhase.IDLE,
        WorkflowPhase.OVERNIGHT,
    ],
}


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay_sec: int = 10
    max_delay_sec: int = 300
    jitter_sec: int = 0

    def backoff(self, attempt: int) -> float:
        delay = min(self.max_delay_sec, self.base_delay_sec * (2**attempt))
        if self.jitter_sec:
            delay += min(self.jitter_sec, self.max_delay_sec - delay)
        return float(delay)


@dataclass
class WorkflowConfig:
    project_dir: Path
    dry_run: bool = True
    validation_mode: bool = False
    output_root: Optional[Path] = None
    state_path: Optional[Path] = None
    metrics_path: Optional[Path] = None
    report_path: Optional[Path] = None
    io_workers: int = 4
    cpu_workers: int = 2
    enable_process_pool: bool = True
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    error_budget: Dict[str, int] = field(
        default_factory=lambda: {"max_failures_per_phase": 3, "cooldown_sec": 300}
    )
    phase_timeouts_sec: Dict[str, int] = field(default_factory=dict)
    checkpoint_max_age_sec: int = 900
    pm_workflow_local_start_hour: int = 18
    pm_workflow_local_start_minute: int = 30

    def __post_init__(self) -> None:
        if not isinstance(self.project_dir, Path):
            self.project_dir = Path(self.project_dir)
        if self.output_root is None:
            if self.validation_mode:
                self.output_root = self.project_dir / "data" / "dry_run"
            else:
                self.output_root = self.project_dir
        if self.state_path is None:
            if self.validation_mode:
                self.state_path = self.output_root / "workflow_state.json"
            else:
                self.state_path = self.project_dir / "data" / "workflow_state.json"
        if self.metrics_path is None:
            date = datetime.now().strftime("%Y%m%d")
            self.metrics_path = (
                self.project_dir / "logs" / f"workflow_metrics_{date}.jsonl"
            )
        if self.report_path is None:
            date = datetime.now().strftime("%Y%m%d")
            self.report_path = (
                self.project_dir / "logs" / f"workflow_report_{date}.json"
            )
        if not self.phase_timeouts_sec:
            self.phase_timeouts_sec = {
                WorkflowPhase.OVERNIGHT.value: 6 * 60 * 60,
                WorkflowPhase.EARLY_PREMARKET.value: 60 * 60,
                WorkflowPhase.PREMARKET.value: 60 * 60,
                WorkflowPhase.MARKET_OPEN.value: 60 * 60,
                WorkflowPhase.MARKET_HOURS.value: 4 * 60 * 60,
                WorkflowPhase.POWER_HOUR.value: 2 * 60 * 60,
                WorkflowPhase.POST_MARKET.value: 3 * 60 * 60,
                WorkflowPhase.PM_WORKFLOW.value: 2 * 60 * 60,
            }
        if not (0 <= int(self.pm_workflow_local_start_hour) <= 23):
            self.pm_workflow_local_start_hour = 18
        if not (0 <= int(self.pm_workflow_local_start_minute) <= 59):
            self.pm_workflow_local_start_minute = 30


@dataclass
class PhaseResult:
    phase: WorkflowPhase
    success: bool
    started_at: str
    ended_at: str
    duration_sec: float
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    retries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "success": self.success,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_sec": self.duration_sec,
            "outputs": self.outputs,
            "error": self.error,
            "retries": self.retries,
        }


@dataclass
class WorkflowState:
    version: int = 1
    current_phase: str = WorkflowPhase.INIT.value
    last_phase: Optional[str] = None
    phase_entered_at: Optional[str] = None
    last_checkpoint_at: Optional[str] = None
    cycle_count: int = 0
    last_date: Optional[str] = None
    daily_flags: Dict[str, bool] = field(default_factory=dict)
    phase_attempts: Dict[str, int] = field(default_factory=dict)
    phase_failures: Dict[str, int] = field(default_factory=dict)
    last_outputs: Dict[str, Any] = field(default_factory=dict)
    last_successful_outputs: Dict[str, Any] = field(default_factory=dict)
    phase_history: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    transition_warnings: List[Dict[str, Any]] = field(default_factory=list)
    cooldown_until: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "current_phase": self.current_phase,
            "last_phase": self.last_phase,
            "phase_entered_at": self.phase_entered_at,
            "last_checkpoint_at": self.last_checkpoint_at,
            "cycle_count": self.cycle_count,
            "last_date": self.last_date,
            "daily_flags": self.daily_flags,
            "phase_attempts": self.phase_attempts,
            "phase_failures": self.phase_failures,
            "last_outputs": self.last_outputs,
            "last_successful_outputs": self.last_successful_outputs,
            "phase_history": self.phase_history,
            "issues": self.issues,
            "transition_warnings": self.transition_warnings,
            "cooldown_until": self.cooldown_until,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowState":
        state = cls()
        for key, val in data.items():
            if hasattr(state, key):
                setattr(state, key, val)
        return state


class WorkflowStateStore:
    def __init__(self, path: Path, logger: Optional[logging.Logger] = None) -> None:
        self.path = Path(path)
        self.logger = logger or logging.getLogger("WorkflowStateStore")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> WorkflowState:
        if not self.path.exists():
            return WorkflowState()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            state = WorkflowState.from_dict(data)
            if self._prune_stale_issues(state):
                self.save(state)
            return state
        except Exception as exc:
            self.logger.warning(f"Failed to load workflow state, starting fresh: {exc}")
            return WorkflowState()

    def save(self, state: WorkflowState) -> None:
        try:
            tmp_path = self.path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
            tmp_path.replace(self.path)
        except Exception as exc:
            self.logger.error(f"Failed to save workflow state: {exc}")

    def _prune_stale_issues(
        self, state: WorkflowState, max_age: timedelta = timedelta(days=3)
    ) -> bool:
        issues = list(state.issues or [])
        if not issues:
            return False

        now = datetime.now()
        kept: List[Dict[str, Any]] = []
        changed = False
        for issue in issues:
            ts_raw = issue.get("timestamp")
            if not ts_raw:
                kept.append(issue)
                continue
            try:
                ts = datetime.fromisoformat(str(ts_raw))
            except Exception:
                changed = True
                continue
            if now - ts <= max_age:
                kept.append(issue)
            else:
                changed = True

        if changed:
            state.issues = kept
        return changed


class WorkflowMetricsWriter:
    def __init__(self, path: Path, logger: Optional[logging.Logger] = None) -> None:
        self.path = Path(path)
        self.logger = logger or logging.getLogger("WorkflowMetrics")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def set_path(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, payload: Dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            **payload,
        }
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except Exception as exc:
            self.logger.error(f"Failed to write workflow metrics: {exc}")


class WorkflowExecutors:
    def __init__(
        self, config: WorkflowConfig, logger: Optional[logging.Logger] = None
    ) -> None:
        self.config = config
        self.logger = logger or logging.getLogger("WorkflowExecutors")
        self._io_pool = ThreadPoolExecutor(max_workers=config.io_workers)
        self._cpu_pool: Optional[ProcessPoolExecutor] = None

    def submit_io(self, fn: Callable[..., Any], *args: Any, **kwargs: Any):
        return self._io_pool.submit(fn, *args, **kwargs)

    def submit_cpu(self, fn: Callable[..., Any], *args: Any, **kwargs: Any):
        if not self.config.enable_process_pool:
            return self._io_pool.submit(fn, *args, **kwargs)
        if self._cpu_pool is None:
            self._cpu_pool = ProcessPoolExecutor(max_workers=self.config.cpu_workers)
        return self._cpu_pool.submit(fn, *args, **kwargs)

    def shutdown(self) -> None:
        self._io_pool.shutdown(wait=False, cancel_futures=False)
        if self._cpu_pool:
            self._cpu_pool.shutdown(wait=False, cancel_futures=False)


class WorkflowWatchdog:
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("WorkflowWatchdog")

    def evaluate(
        self,
        phase: WorkflowPhase,
        result: Optional[PhaseResult],
        state: WorkflowState,
        config: WorkflowConfig,
        expected_outputs: Dict[str, bool],
    ) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        now = datetime.now()

        if state.phase_entered_at:
            try:
                entered = datetime.fromisoformat(state.phase_entered_at)
                elapsed = (now - entered).total_seconds()
                timeout = config.phase_timeouts_sec.get(phase.value)
                if timeout and elapsed > timeout:
                    issues.append(
                        {
                            "type": "phase_stalled",
                            "phase": phase.value,
                            "elapsed_sec": elapsed,
                            "timeout_sec": timeout,
                        }
                    )
            except Exception:
                pass

        if state.last_checkpoint_at:
            try:
                last_ckpt = datetime.fromisoformat(state.last_checkpoint_at)
                since = (now - last_ckpt).total_seconds()
                if since > config.checkpoint_max_age_sec:
                    issues.append(
                        {
                            "type": "checkpoint_stalled",
                            "phase": phase.value,
                            "elapsed_sec": since,
                            "timeout_sec": config.checkpoint_max_age_sec,
                        }
                    )
            except Exception:
                pass

        if expected_outputs:
            missing = [name for name, exists in expected_outputs.items() if not exists]
            if missing:
                issues.append(
                    {
                        "type": "missing_outputs",
                        "phase": phase.value,
                        "missing": missing,
                    }
                )

        failures = state.phase_failures.get(phase.value, 0)
        budget = config.error_budget.get("max_failures_per_phase", 3)
        if failures >= budget:
            issues.append(
                {
                    "type": "error_budget_exceeded",
                    "phase": phase.value,
                    "failures": failures,
                    "budget": budget,
                }
            )

        if result and not result.success:
            issues.append(
                {
                    "type": "phase_failed",
                    "phase": phase.value,
                    "error": result.error or "unknown",
                }
            )

        return issues


class WorkflowManager:
    def __init__(
        self,
        agent: Any,
        scheduler: Any,
        config: WorkflowConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.agent = agent
        self.scheduler = scheduler
        self.config = config
        self.logger = logger or logging.getLogger("WorkflowManager")
        self.state_store = WorkflowStateStore(config.state_path, logger=self.logger)
        self.metrics = WorkflowMetricsWriter(config.metrics_path, logger=self.logger)
        self.executors = WorkflowExecutors(config, logger=self.logger)
        self.watchdog = WorkflowWatchdog(logger=self.logger)
        self.state = self.state_store.load()
        self.history_limit = 50
        self.phase_handlers = self._build_phase_handlers()
        self._market_context_shadow_buckets: set[str] = set()

    def start(self) -> None:
        self.logger.info("WorkflowManager starting")
        self._router_health_check()
        self.resume()

    def resume(self) -> None:
        self.logger.info("WorkflowManager resuming loop")
        self._ensure_output_dirs()
        self._normalize_state_for_resume()
        try:
            self._run_loop()
        finally:
            self.executors.shutdown()

    def _normalize_state_for_resume(self) -> None:
        """Refresh stale persisted heartbeat timestamps before watchdog checks."""
        now = datetime.now()
        now_iso = now.isoformat()
        updated_fields: List[str] = []

        checkpoint_age = self._age_seconds(self.state.last_checkpoint_at, now)
        checkpoint_stale = (
            self.state.last_checkpoint_at is None
            or checkpoint_age is None
            or checkpoint_age > self.config.checkpoint_max_age_sec
            or checkpoint_age < -60
        )
        if checkpoint_stale:
            self.state.last_checkpoint_at = now_iso
            updated_fields.append("last_checkpoint_at")

        current_phase = WorkflowPhase.from_value(self.state.current_phase)
        timeout = self.config.phase_timeouts_sec.get(current_phase.value)
        phase_age = self._age_seconds(self.state.phase_entered_at, now)
        phase_stale = (
            self.state.phase_entered_at is None
            or phase_age is None
            or phase_age < -60
            or (timeout is not None and phase_age > timeout)
        )
        if phase_stale:
            self.state.phase_entered_at = now_iso
            updated_fields.append("phase_entered_at")

        if updated_fields:
            self.state_store.save(self.state)
            self.logger.info(
                "Resume normalized state fields: " + ", ".join(updated_fields)
            )

    @staticmethod
    def _age_seconds(value: Optional[str], now: datetime) -> Optional[float]:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value))
        except Exception:
            return None
        return (now - parsed).total_seconds()

    def checkpoint(
        self, phase: WorkflowPhase, result: Optional[PhaseResult] = None
    ) -> None:
        self.state.last_checkpoint_at = datetime.now().isoformat()
        self.state.current_phase = phase.value
        if result:
            self.state.last_outputs[phase.value] = self._summarize_outputs(
                result.outputs
            )
        self.state_store.save(self.state)
        self.metrics.log(
            "checkpoint",
            {
                "phase": phase.value,
                "cycle": self.state.cycle_count,
            },
        )

    def recover(self, reason: str) -> None:
        self.logger.warning(f"Workflow recovery triggered: {reason}")
        self.metrics.log("recover", {"reason": reason})
        self.state.current_phase = WorkflowPhase.IDLE.value
        self.state_store.save(self.state)

    def run_dry_validation(self) -> Dict[str, Any]:
        self.logger.info("Running workflow dry validation")
        if not self.config.validation_mode:
            self.config.validation_mode = True
            self.config.output_root = self.config.project_dir / "data" / "dry_run"
            self.config.state_path = self.config.output_root / "workflow_state.json"
            self.state_store = WorkflowStateStore(
                self.config.state_path, logger=self.logger
            )
            self.state = self.state_store.load()
        self._ensure_output_dirs()

        phases = [
            WorkflowPhase.OVERNIGHT,
            WorkflowPhase.EARLY_PREMARKET,
            WorkflowPhase.PREMARKET,
            WorkflowPhase.MARKET_OPEN,
            WorkflowPhase.MARKET_HOURS,
            WorkflowPhase.POWER_HOUR,
            WorkflowPhase.POST_MARKET,
            WorkflowPhase.PM_WORKFLOW,
        ]
        report = {
            "mode": "dry_validation",
            "started_at": datetime.now().isoformat(),
            "phases": [],
        }

        for phase in phases:
            result = self._simulate_phase(phase)
            report["phases"].append(result.to_dict())
            expected = self._expected_outputs(phase, result)
            issues = self.watchdog.evaluate(
                phase, result, self.state, self.config, expected
            )
            if issues:
                report.setdefault("issues", []).extend(issues)

        report["completed_at"] = datetime.now().isoformat()
        self._write_report(report, suffix="dry_validation")
        return report

    def _build_phase_handlers(
        self,
    ) -> Dict[WorkflowPhase, Callable[[], Dict[str, Any]]]:
        return {
            WorkflowPhase.OVERNIGHT: self._run_overnight,
            WorkflowPhase.EARLY_PREMARKET: self._run_early_premarket,
            WorkflowPhase.PREMARKET: self._run_premarket,
            WorkflowPhase.MARKET_OPEN: self._run_day_manager,
            WorkflowPhase.MARKET_HOURS: self._run_day_manager,
            WorkflowPhase.POWER_HOUR: self._run_day_manager,
            WorkflowPhase.POST_MARKET: self._run_post_market_idle,
            WorkflowPhase.PM_WORKFLOW: self._run_pm_workflow,
        }

    def _run_loop(self) -> None:
        while True:
            self.state.cycle_count += 1
            now = datetime.now()
            today = now.strftime("%Y-%m-%d")
            if self.state.last_date != today:
                self._reset_daily_flags(today)

            # Self-healing: Detect and clear stalled/failed phases (P5)
            self._self_heal_stalled_phases()

            phase = self._determine_phase()
            self._transition_phase(phase)
            self._maybe_launch_market_context_shadow(phase)

            if self._in_cooldown():
                cooldown_seconds = self._cooldown_remaining()
                if cooldown_seconds <= 0:
                    self.state.cooldown_until = None
                    # Reset checkpoint timestamp so watchdog doesn't see stale gap
                    self.state.last_checkpoint_at = datetime.now().isoformat()
                    self.state_store.save(self.state)
                    self.logger.info("Cooldown expired; resuming workflow")
                else:
                    self.logger.warning(
                        f"Cooldown active ({cooldown_seconds}s remaining)"
                    )
                    time.sleep(min(60, max(1, cooldown_seconds)))
                    continue

            result = self._execute_phase(phase)
            self._record_phase_result(phase, result)

            # Skip watchdog evaluation for phases that were skipped (already ran today)
            was_skipped = (
                result.outputs.get("skipped", False) if result.outputs else False
            )
            if not was_skipped:
                expected_outputs = self._expected_outputs(phase, result)
                issues = self.watchdog.evaluate(
                    phase, result, self.state, self.config, expected_outputs
                )
                if issues:
                    self._handle_issues(phase, issues)

            self.checkpoint(phase, result)
            self._update_report()

            # Use a short interval when phase was skipped (just waiting for next phase)
            if was_skipped:
                interval = 60  # 1 minute poll when idle/skipped
                self.logger.info(
                    f"[WAIT] Phase already ran today, checking again in {interval}s"
                )
            else:
                interval = self._phase_interval(phase)
                self.logger.info(f"[WAIT] Next cycle in {interval}s")
            time.sleep(interval)

    def _ensure_output_dirs(self) -> None:
        base = self.config.output_root
        (base / "plans").mkdir(parents=True, exist_ok=True)
        (base / "logs").mkdir(parents=True, exist_ok=True)

    def _determine_phase(self) -> WorkflowPhase:
        status = self.scheduler.get_status()
        return WorkflowPhase.from_value(status.get("market_phase"))

    def _transition_phase(self, phase: WorkflowPhase) -> None:
        current = WorkflowPhase.from_value(self.state.current_phase)
        if phase == current:
            return
        allowed = ALLOWED_TRANSITIONS.get(current, [])
        if phase not in allowed:
            warning = {
                "from": current.value,
                "to": phase.value,
                "timestamp": datetime.now().isoformat(),
            }
            self.state.transition_warnings.append(warning)
            if len(self.state.transition_warnings) > 50:
                self.state.transition_warnings = self.state.transition_warnings[-50:]
            self.logger.warning(
                f"Unexpected phase transition {current.value} -> {phase.value}"
            )
        self.state.last_phase = current.value
        self.state.current_phase = phase.value
        self.state.phase_entered_at = datetime.now().isoformat()
        self.metrics.log(
            "phase_transition",
            {
                "from": current.value,
                "to": phase.value,
            },
        )

        if MONITORING_AVAILABLE:
            try:
                collector = get_metrics_collector()
                collector.emit_workflow_phase(
                    phase=phase.value,
                    status="transition",
                    attempt=self.state.phase_attempts.get(phase.value, 1),
                )
            except Exception:
                pass

    def _execute_phase(self, phase: WorkflowPhase) -> PhaseResult:
        if self.config.validation_mode:
            return self._simulate_phase(phase)

        handler = self.phase_handlers.get(phase)
        if handler is None:
            return PhaseResult(
                phase=phase,
                success=False,
                started_at=datetime.now().isoformat(),
                ended_at=datetime.now().isoformat(),
                duration_sec=0.0,
                error="no_handler",
                retries=0,
            )

        if not self._should_run_phase(phase):
            # A skipped phase is not a failure; clear stale failure budget.
            self.state.phase_failures[phase.value] = 0
            return PhaseResult(
                phase=phase,
                success=True,
                started_at=datetime.now().isoformat(),
                ended_at=datetime.now().isoformat(),
                duration_sec=0.0,
                outputs={"skipped": True},
                retries=0,
            )

        attempts = 0
        start_time = datetime.now()
        last_error = None
        while attempts < self.config.retry_policy.max_attempts:
            attempts += 1
            self.state.phase_attempts[phase.value] = (
                self.state.phase_attempts.get(phase.value, 0) + 1
            )
            self.metrics.log("phase_start", {"phase": phase.value, "attempt": attempts})

            if MONITORING_AVAILABLE:
                try:
                    collector = get_metrics_collector()
                    collector.emit_workflow_phase(
                        phase=phase.value,
                        status="started",
                        attempt=attempts,
                    )
                except Exception:
                    pass

            try:
                outputs = handler()
                # Successful execution should reset failure budget for this phase.
                self.state.phase_failures[phase.value] = 0
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                if MONITORING_AVAILABLE:
                    try:
                        collector = get_metrics_collector()
                        collector.emit_workflow_phase(
                            phase=phase.value,
                            status="success",
                            duration_sec=duration,
                            attempt=attempts,
                            outputs=outputs or {},
                        )
                    except Exception:
                        pass

                return PhaseResult(
                    phase=phase,
                    success=True,
                    started_at=start_time.isoformat(),
                    ended_at=end_time.isoformat(),
                    duration_sec=duration,
                    outputs=outputs or {},
                    retries=attempts - 1,
                )
            except Exception as exc:
                last_error = str(exc)
                self.state.phase_failures[phase.value] = (
                    self.state.phase_failures.get(phase.value, 0) + 1
                )
                self.metrics.log(
                    "phase_error",
                    {
                        "phase": phase.value,
                        "attempt": attempts,
                        "error": last_error,
                    },
                )

                if MONITORING_AVAILABLE:
                    try:
                        collector = get_metrics_collector()
                        collector.emit_workflow_phase(
                            phase=phase.value,
                            status="error",
                            attempt=attempts,
                        )
                    except Exception:
                        pass

                if attempts < self.config.retry_policy.max_attempts:
                    delay = self.config.retry_policy.backoff(attempts - 1)
                    self.metrics.log(
                        "phase_retry",
                        {
                            "phase": phase.value,
                            "attempt": attempts,
                            "delay_sec": delay,
                        },
                    )
                    time.sleep(delay)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        if MONITORING_AVAILABLE:
            try:
                collector = get_metrics_collector()
                collector.emit_workflow_phase(
                    phase=phase.value,
                    status="failed",
                    duration_sec=duration,
                    attempt=attempts,
                )
            except Exception:
                pass

        return PhaseResult(
            phase=phase,
            success=False,
            started_at=start_time.isoformat(),
            ended_at=end_time.isoformat(),
            duration_sec=duration,
            error=last_error or "unknown",
            retries=attempts - 1,
        )

    def _record_phase_result(self, phase: WorkflowPhase, result: PhaseResult) -> None:
        entry = {
            "phase": phase.value,
            "started_at": result.started_at,
            "ended_at": result.ended_at,
            "duration_sec": result.duration_sec,
            "success": result.success,
            "retries": result.retries,
            "outputs": self._summarize_outputs(result.outputs),
            "error": result.error,
        }
        self.state.phase_history.append(entry)
        if len(self.state.phase_history) > self.history_limit:
            self.state.phase_history = self.state.phase_history[-self.history_limit :]
        if result.success:
            self.state.last_successful_outputs[phase.value] = entry["outputs"]
        self.metrics.log("phase_end", entry)

    def _handle_issues(
        self, phase: WorkflowPhase, issues: List[Dict[str, Any]]
    ) -> None:
        for issue in issues:
            issue["timestamp"] = datetime.now().isoformat()
            self.state.issues.append(issue)
        if len(self.state.issues) > 100:
            self.state.issues = self.state.issues[-100:]
        self.metrics.log("watchdog_issue", {"phase": phase.value, "issues": issues})
        self.logger.warning(f"Watchdog issues detected in {phase.value}: {issues}")

        budget = self.config.error_budget.get("max_failures_per_phase", 3)
        failures = self.state.phase_failures.get(phase.value, 0)
        if failures >= budget:
            cooldown = self.config.error_budget.get("cooldown_sec", 300)
            self.state.cooldown_until = (
                datetime.now() + timedelta(seconds=cooldown)
            ).isoformat()
            self.logger.warning(
                f"Entering cooldown for {cooldown}s due to error budget"
            )

        self._notify_router(phase, issues)

    def _notify_router(
        self, phase: WorkflowPhase, issues: List[Dict[str, Any]]
    ) -> None:
        if not getattr(self.agent, "task_router", None):
            return
        try:
            payload = {
                "phase": phase.value,
                "issues": issues,
                "cycle": self.state.cycle_count,
            }
            response = self.agent.route_task("WORKFLOW_ANALYSIS", payload)
            if response:
                self.metrics.log(
                    "router_response",
                    {
                        "phase": phase.value,
                        "success": response.get("success"),
                        "agent": response.get("agent"),
                    },
                )
        except Exception as exc:
            self.logger.warning(f"TaskRouter notification failed: {exc}")

    def _router_health_check(self) -> None:
        if not getattr(self.agent, "task_router", None):
            return
        try:
            response = self.agent.route_task(
                "HEALTH_CHECK", {"event": "workflow_start"}
            )
            if response:
                self.metrics.log(
                    "router_health_check",
                    {
                        "success": response.get("success"),
                        "agent": response.get("agent"),
                    },
                )
        except Exception as exc:
            self.logger.warning(f"TaskRouter health check failed: {exc}")

    def _phase_interval(self, phase: WorkflowPhase) -> int:
        actions = self.scheduler.get_phase_actions(phase)
        return int(actions.get("interval_seconds", 60))

    def _maybe_launch_market_context_shadow(self, phase: WorkflowPhase) -> None:
        if phase not in {
            WorkflowPhase.PREMARKET,
            WorkflowPhase.MARKET_OPEN,
            WorkflowPhase.MARKET_HOURS,
            WorkflowPhase.POWER_HOUR,
        }:
            return
        try:
            now = self.scheduler.get_current_time()
        except Exception:
            now = datetime.now()
        bucket = self._market_context_shadow_bucket(now)
        if not bucket or bucket in self._market_context_shadow_buckets:
            return

        self._market_context_shadow_buckets.add(bucket)
        try:
            logs_dir = self.config.project_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = (
                logs_dir
                / f"market_context_shadow_scheduler_{now.strftime('%Y-%m-%d')}.log"
            )
            handle = log_path.open("a", encoding="utf-8")
            subprocess.Popen(
                [sys.executable, "tools/market_context_shadow.py"],
                cwd=str(self.config.project_dir),
                stdout=handle,
                stderr=subprocess.STDOUT,
                close_fds=False,
            )
            self.logger.info(
                "[SHADOW] Launched market-context shadow job for %s", bucket
            )
        except Exception as exc:
            self.logger.warning("[SHADOW] Market-context shadow launch failed: %s", exc)

    @staticmethod
    def _market_context_shadow_bucket(now: datetime) -> Optional[str]:
        if now.weekday() >= 5:
            return None
        slots = [(7, 30), (9, 30), (11, 30), (13, 30), (15, 30)]
        current_minutes = now.hour * 60 + now.minute
        for hour, minute in slots:
            slot_minutes = hour * 60 + minute
            if slot_minutes <= current_minutes < slot_minutes + 120:
                return f"{now.strftime('%Y-%m-%d')}T{hour:02d}:{minute:02d}"
        return None

    def _reset_daily_flags(self, today: str) -> None:
        self.logger.info(f"[NEW DAY] {today}")
        previous_date = self.state.last_date
        previous_flags = dict(self.state.daily_flags or {})
        carry_pm_workflow = self._should_carry_pm_workflow_flag(
            previous_date=previous_date,
            today=today,
            previous_flags=previous_flags,
        )

        self.state.last_date = today
        self.state.daily_flags = {
            "overnight_ran": False,
            "premarket_ran": False,
            "pm_workflow_ran": carry_pm_workflow,
            "early_premarket_ran": False,
        }
        if carry_pm_workflow:
            self.logger.info(
                "[NEW DAY] Carrying PM workflow completion from last cycle"
            )
        date = datetime.now().strftime("%Y%m%d")
        self.metrics.set_path(
            self.config.project_dir / "logs" / f"workflow_metrics_{date}.jsonl"
        )
        self.config.report_path = (
            self.config.project_dir / "logs" / f"workflow_report_{date}.json"
        )

    def _should_carry_pm_workflow_flag(
        self,
        previous_date: Optional[str],
        today: str,
        previous_flags: Dict[str, bool],
    ) -> bool:
        if not previous_flags.get("pm_workflow_ran", False):
            return False
        if not previous_date:
            return False

        try:
            prev_dt = datetime.strptime(previous_date, "%Y-%m-%d").date()
            today_dt = datetime.strptime(today, "%Y-%m-%d").date()
        except ValueError:
            return False

        # Only carry across a normal one-day rollover.
        if (today_dt - prev_dt).days != 1:
            return False

        # PM workflow should have happened recently (last evening), not from older catch-up runs.
        return self._has_recent_pm_workflow_completion(max_age_hours=16)

    def _has_recent_pm_workflow_completion(self, max_age_hours: int = 16) -> bool:
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        history = self.state.phase_history or []

        for entry in reversed(history):
            if entry.get("phase") != WorkflowPhase.PM_WORKFLOW.value:
                continue
            if not entry.get("success", False):
                continue

            ended_at_raw = entry.get("ended_at") or entry.get("started_at")
            if not ended_at_raw:
                continue

            try:
                ended_at = datetime.fromisoformat(str(ended_at_raw))
            except ValueError:
                continue

            return ended_at >= cutoff

        return False

    def _should_run_phase(self, phase: WorkflowPhase) -> bool:
        if phase == WorkflowPhase.OVERNIGHT:
            # Always run if PM workflow hasn't completed yet (catch-up pipeline)
            if not self.state.daily_flags.get("pm_workflow_ran", False):
                return True
            return not self.state.daily_flags.get("overnight_ran", False)
        if phase == WorkflowPhase.PREMARKET:
            return not self.state.daily_flags.get("premarket_ran", False)
        if phase == WorkflowPhase.PM_WORKFLOW:
            return not self.state.daily_flags.get("pm_workflow_ran", False)
        if phase == WorkflowPhase.EARLY_PREMARKET:
            return not self.state.daily_flags.get("early_premarket_ran", False)
        return True

    def _simulate_phase(self, phase: WorkflowPhase) -> PhaseResult:
        start = datetime.now()
        io_future = self.executors.submit_io(_io_probe, phase.value)
        cpu_future = self.executors.submit_cpu(_cpu_probe, 25000)
        io_result = io_future.result()
        cpu_result = cpu_future.result()

        outputs = {
            "simulated": True,
            "io": io_result,
            "cpu": {"probe": cpu_result},
        }
        if phase == WorkflowPhase.OVERNIGHT:
            path = self._plan_paths()["morning_plan"]
            self._write_json(
                path, {"buy_signals": [{"symbol": "TEST", "confidence": 0.5}]}
            )
            outputs["generated"] = path.name
        elif phase == WorkflowPhase.PM_WORKFLOW:
            path = self._plan_paths()["pm_plan"]
            self._write_json(path, {"signals": [{"symbol": "TEST", "score": 0.5}]})
            outputs["generated"] = path.name

        end = datetime.now()
        duration = (end - start).total_seconds()
        return PhaseResult(
            phase=phase,
            success=True,
            started_at=start.isoformat(),
            ended_at=end.isoformat(),
            duration_sec=duration,
            outputs=outputs,
            retries=0,
        )

    def _expected_outputs(
        self, phase: WorkflowPhase, result: Optional[PhaseResult] = None
    ) -> Dict[str, bool]:
        paths = self._plan_paths()
        expected: Dict[str, bool] = {}
        outputs = (result.outputs or {}) if result else {}

        # If a phase explicitly skipped (schedule gate), do not enforce output checks.
        if outputs.get("skipped"):
            return expected

        if phase == WorkflowPhase.OVERNIGHT:
            has_morning = paths["morning_plan"].exists()
            # Allow fallback to the most recent plan artifact; before PM/overnight
            # completion we may still rely on yesterday's plan.
            has_recent_plan = self._has_recent_plan(max_age_hours=48)
            expected["overnight_plan"] = has_morning or has_recent_plan
        elif phase == WorkflowPhase.PM_WORKFLOW:
            # PM workflow currently runs daily review and does not guarantee pm_plan output.
            expected = {}
        return expected

    def _has_recent_plan(self, max_age_hours: int = 48) -> bool:
        """Return True when a recent canonical session plan artifact exists."""
        try:
            plans_dir = self.config.output_root / "plans"
            if not plans_dir.exists():
                return False
            now_ts = datetime.now().timestamp()
            for pattern in ("morning_game_plan_*.json",):
                for path in plans_dir.glob(pattern):
                    age_hours = (now_ts - path.stat().st_mtime) / 3600.0
                    if age_hours <= max_age_hours:
                        return True
        except Exception:
            return False
        return False

    def _plan_has_usable_signals(self, plan_path: Path) -> bool:
        """Return True when a plan file exists and contains at least one signal."""
        if not plan_path.exists():
            return False
        try:
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        for key in ("signals", "entry_candidates", "buy_signals"):
            signals = payload.get(key)
            if isinstance(signals, list) and signals:
                return True
        return False

    def _plan_paths(self) -> Dict[str, Path]:
        plan_date = get_pm_plan_date(datetime.now())
        date_dash = plan_date.strftime("%Y-%m-%d")
        date_compact = plan_date.strftime("%Y%m%d")
        base = self.config.output_root
        return {
            "pm_plan": base / "plans" / f"pm_plan_{date_dash}.json",
            "morning_plan": base / "plans" / f"morning_game_plan_{date_compact}.json",
        }

    def _run_overnight(self) -> Dict[str, Any]:
        if hasattr(self.agent, "_ensure_ollama_for_phase"):
            self.agent._ensure_ollama_for_phase("OVERNIGHT")

        pm_plan_path = self._plan_paths()["pm_plan"]

        # Catch up on PM workflow if it hasn't run yet.
        # Overnight research depends on a usable PM plan. If the PM_WORKFLOW
        # phase window was missed or produced an empty plan, run it now.
        pm_workflow_ran = self.state.daily_flags.get("pm_workflow_ran", False)
        pm_plan_exists = pm_plan_path.exists()
        pm_plan_usable = self._plan_has_usable_signals(pm_plan_path)
        if (
            (not pm_workflow_ran)
            or (not pm_plan_exists)
            or (pm_plan_exists and not pm_plan_usable)
        ):
            reason_bits = []
            if not pm_workflow_ran:
                reason_bits.append("pm_workflow_not_marked_complete")
            if not pm_plan_exists:
                reason_bits.append(f"missing_{pm_plan_path.name}")
            elif not pm_plan_usable:
                reason_bits.append(f"empty_{pm_plan_path.name}")
            catchup_reason = ",".join(reason_bits) or "pm_catchup_required"
            self.logger.info(
                "[CATCHUP] PM workflow catch-up required before overnight research (%s)",
                catchup_reason,
            )
            try:
                pm_result = self._run_pm_workflow()
                self.logger.info(
                    f"[CATCHUP] PM workflow completed: {list((pm_result or {}).keys())}"
                )
            except Exception as exc:
                self.logger.error(
                    "[CATCHUP] PM workflow failed during overnight handoff: %s",
                    exc,
                )
                if not pm_plan_path.exists() or not self._plan_has_usable_signals(
                    pm_plan_path
                ):
                    raise RuntimeError(
                        f"PM workflow catch-up failed and {pm_plan_path.name} is still unusable"
                    ) from exc

            if not pm_plan_path.exists() or not self._plan_has_usable_signals(
                pm_plan_path
            ):
                raise RuntimeError(
                    f"PM workflow catch-up completed without producing usable {pm_plan_path.name}"
                )

        report = self.agent.overnight_engine.run_full_overnight_cycle(pm_plan_path)
        pm_report = report.get("pm_report") if isinstance(report, dict) else None
        if pm_report:
            try:
                self.agent._persist_pm_report_to_plan(pm_report, pm_plan_path)
                self.logger.info(
                    f"[PM REPORT] Stored standardized pm_report into {pm_plan_path.name}"
                )
            except Exception as exc:
                self.logger.debug(f"[PM REPORT] Could not persist pm_report: {exc}")
        self.state.daily_flags["overnight_ran"] = True
        return report or {}

    def _run_early_premarket(self) -> Dict[str, Any]:
        if hasattr(self.agent, "_ensure_ollama_for_phase"):
            self.agent._ensure_ollama_for_phase("EARLY_PREMARKET")
        if not self.state.daily_flags.get("overnight_ran", False):
            self._run_overnight()
        paths = self._plan_paths()
        preferred_plan_path = (
            paths["pm_plan"]
            if self._plan_has_usable_signals(paths["pm_plan"])
            else paths["morning_plan"]
            if paths["morning_plan"].exists()
            else paths["pm_plan"]
        )
        trade_plan = self.agent.plan_generator.generate_plan(preferred_plan_path)
        self.state.daily_flags["early_premarket_ran"] = True
        return {"plan_summary": trade_plan.get("summary", "n/a")}

    def _run_premarket(self) -> Dict[str, Any]:
        result = self._normalize_phase_result(
            "premarket_scan",
            self.agent.run_premarket_scan(dry_run=self.config.dry_run),
        )
        self.state.daily_flags["premarket_ran"] = True
        return result

    def _run_day_manager(self) -> Dict[str, Any]:
        return self._normalize_phase_result(
            "day_manager_cycle",
            self.agent.run_day_manager_cycle(dry_run=self.config.dry_run),
        )

    def _run_post_market_idle(self) -> Dict[str, Any]:
        if not self.state.daily_flags.get("pm_workflow_ran", False):
            if not self._is_pm_workflow_local_window_open():
                start_local = (
                    f"{self.config.pm_workflow_local_start_hour:02d}:"
                    f"{self.config.pm_workflow_local_start_minute:02d}"
                )
                self.logger.info(
                    f"[POST_MARKET] PM workflow pending; waiting until {start_local} local"
                )
                return {
                    "idle": True,
                    "pm_workflow_pending": True,
                    "pm_workflow_local_start": start_local,
                }
            self.logger.info("[POST_MARKET] PM workflow pending for today; running now")
            pm_result = self._run_pm_workflow()
            return {
                "pm_workflow_triggered": True,
                "pm_workflow_error": bool((pm_result or {}).get("error")),
            }
        return {"idle": True}

    def _is_pm_workflow_local_window_open(self, now: Optional[datetime] = None) -> bool:
        current = now or datetime.now()
        start = (
            int(self.config.pm_workflow_local_start_hour),
            int(self.config.pm_workflow_local_start_minute),
        )
        return (current.hour, current.minute) >= start

    def _run_pm_workflow(self) -> Dict[str, Any]:
        # Ensure YouTube intelligence is fresh before PM workflow
        try:
            from autotrade.utils.youtube_readiness import (
                ensure_youtube_ready,
                format_readiness_log,
            )

            status = ensure_youtube_ready()
            post = status.get("post_check", {})
            log_status = dict(post or {})
            log_status["scan_ran"] = bool(status.get("scan_ran", False))
            self.logger.info(format_readiness_log(log_status))
        except Exception as e:
            self.logger.warning(f"YouTube readiness check failed: {e}")

        try:
            import inspect

            signature = inspect.signature(self.agent.run_pm_workflow)
            supports_include_signals = "include_signals" in signature.parameters
        except (TypeError, ValueError):
            supports_include_signals = False
        if supports_include_signals:
            raw_result = self.agent.run_pm_workflow(
                dry_run=self.config.dry_run,
                include_signals=False,
            )
        else:
            raw_result = self.agent.run_pm_workflow(dry_run=self.config.dry_run)

        result = self._normalize_phase_result("pm_workflow", raw_result)
        self.state.daily_flags["pm_workflow_ran"] = True
        return result

    @staticmethod
    def _normalize_phase_result(phase_name: str, result: Any) -> Dict[str, Any]:
        """Normalize phase handler outputs and fail fast on explicit error payloads."""
        if result is None:
            return {}
        if isinstance(result, dict):
            error_msg = result.get("error")
            if error_msg:
                raise RuntimeError(f"{phase_name} returned error: {error_msg}")
            return result
        return {"result": result}

    def _summarize_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for key, value in (outputs or {}).items():
            if isinstance(value, list):
                summary[key] = {"count": len(value)}
            elif isinstance(value, dict):
                summary[key] = {"keys": list(value.keys())[:10]}
            else:
                summary[key] = value
        return summary

    def _write_report(
        self, report: Dict[str, Any], suffix: Optional[str] = None
    ) -> None:
        path = self.config.report_path
        if suffix:
            path = path.with_name(path.stem + f"_{suffix}" + path.suffix)
        try:
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except Exception as exc:
            self.logger.error(f"Failed to write workflow report: {exc}")

    def _update_report(self) -> None:
        report = {
            "generated_at": datetime.now().isoformat(),
            "current_phase": self.state.current_phase,
            "cycle_count": self.state.cycle_count,
            "phase_history": self.state.phase_history,
            "last_successful_outputs": self.state.last_successful_outputs,
            "issues": self.state.issues[-10:],
        }
        self._write_report(report)

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _in_cooldown(self) -> bool:
        if not self.state.cooldown_until:
            return False
        try:
            until = datetime.fromisoformat(self.state.cooldown_until)
            return datetime.now() < until
        except Exception:
            return False

    def _cooldown_remaining(self) -> int:
        if not self.state.cooldown_until:
            return 0
        try:
            until = datetime.fromisoformat(self.state.cooldown_until)
            return max(0, int(math.ceil((until - datetime.now()).total_seconds())))
        except Exception:
            return 0

    def _self_heal_stalled_phases(self) -> None:
        """
        Detect and clear stalled or failed phases (P5).
        If a phase has been in a failed state or same-phase window for >20 mins
        without success, reset the failure budget and flags to permit a fresh start.
        """
        now = datetime.now()
        current_phase = WorkflowPhase.from_value(self.state.current_phase)

        # We only heal phases that are actually expected to be running or failed
        if current_phase in [
            WorkflowPhase.IDLE,
            WorkflowPhase.INIT,
            WorkflowPhase.POST_MARKET,
        ]:
            return

        failures = self.state.phase_failures.get(current_phase.value, 0)
        phase_entered_at = self.state.phase_entered_at

        if not phase_entered_at:
            return

        try:
            entered = datetime.fromisoformat(phase_entered_at)
            elapsed_mins = (now - entered).total_seconds() / 60.0

            # P5 Rule: 20 minute stall threshold for self-healing
            STALL_THRESHOLD_MINS = 20.0

            if elapsed_mins > STALL_THRESHOLD_MINS and failures > 0:
                self.logger.warning(
                    f"[SELF-HEAL] Phase {current_phase.value} stalled/failed for {elapsed_mins:.1f} mins. "
                    f"Resetting failure budget ({failures} -> 0) to allow retry."
                )

                # Reset failure counts to allow the retry loop in _execute_phase to trigger again
                self.state.phase_failures[current_phase.value] = 0

                # If we are in a cooldown due to this phase, clear it
                if self.state.cooldown_until:
                    self.logger.info(
                        "[SELF-HEAL] Clearing active cooldown to permit immediate retry."
                    )
                    self.state.cooldown_until = None

                # Update phase_entered_at so we don't immediately heal again next cycle
                self.state.phase_entered_at = now.isoformat()
                self.state_store.save(self.state)

                self.metrics.log(
                    "self_heal",
                    {
                        "phase": current_phase.value,
                        "elapsed_mins": elapsed_mins,
                        "previous_failures": failures,
                    },
                )

        except Exception as e:
            self.logger.error(f"[SELF-HEAL] Error during phase healing check: {e}")
