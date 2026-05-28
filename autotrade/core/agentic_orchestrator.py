"""
Agentic Orchestrator - Always-On System Health Monitor with 14 Modular Agents

This orchestrator runs continuously alongside the server to:
1. Monitor system health and logs
2. Detect failures and attempt auto-recovery via specialized agents
3. Ensure agentic workflows are active
4. Maintain daily context snapshots
5. Route tasks to appropriate Ollama models based on complexity
6. Validate PM workflow picks and optimize watchlists
7. Research stocks and analyze backtests
8. Debug code using local models

AI MODEL POLICY:
- Default to Codex CLI for agentic tasks
- Local Ollama models are optional (toggle via config)
- OpenAI is LAST RESORT only (for large refactoring)
- qwen3 models require "think": False in API calls

Architecture (14 Agents):
    - DiagnosticAgent: Quick health checks (qwen2.5:3b)
    - AnalysisAgent: Log/data analysis (deepseek-r1:8b)
    - CodeAgent: Code analysis/fixes (qwen3-coder:30b)
    - RecoveryAgent: Auto-recovery actions (deepseek-r1:8b)
    - VisionAgent: Screenshot/UI analysis (qwen3-vl:8b)
    - ChartAgent: Trading chart analysis (llama3.2-vision primary, qwen2.5vl fallback)
    - MathAgent: Math/algorithm verification (phi4:14b)
    - PlannerAgent: Task planning (gpt-oss:20b)
    - MassiveContextAgent: 1M token context (nemotron-3-nano:30b)
    - SearchAgent: Web/news search via SearXNG (qwen3:8b)
    - DevSearchAgent: Code/repo/docs search (qwen3-coder:30b)
    - PMValidatorAgent: PM workflow validation (deepseek-r1:8b)
    - ImprovementAgent: Strategy improvements (qwen3:30b)
    - AutonomousResearchAgent: Research/optimization (qwen3:30b)

Usage:
    python -m autotrade.core.agentic_orchestrator [--dry-run] [--interval SECONDS]
    python -m autotrade.core.agentic_orchestrator --once  # Single health check
    python -m autotrade.core.agentic_orchestrator --list-models  # Show available models

    start_server.bat agent       # Server + Orchestrator
    start_server.bat agent-only  # Orchestrator only
"""

import os
import sys
import json
import base64
import time
import re
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import requests

from autotrade.utils.market_time import get_market_now, get_pm_plan_date
from autotrade.utils.openai_client import get_openai_client
from autotrade.utils.openrouter_client import get_openrouter_client
from autotrade.utils.safe_logging import get_safe_logger
from config.config_loader import get_llm_config

# Setup logging
PROJECT_DIR = Path(
    os.environ.get("AUTOTRADE_ROOT", Path(__file__).resolve().parents[2])
)
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = get_safe_logger(
    "AgenticOrchestrator",
    LOG_DIR / f"orchestrator_{datetime.now().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    console_level=logging.INFO,
)


# =============================================================================
# SHARED LINT VALIDATION â€” used by all fallback repair paths
# =============================================================================


def _lint_python_content(content: str, file_path: str) -> Optional[str]:
    """
    Validate Python content with AST parse + ruff lint.

    Mirrors ToolExecutor._lint_python_file() but works with in-memory content.
    Returns error string if validation fails, None if OK.

    Checks:
      1. AST parse â€” catches syntax errors
      2. Ruff lint â€” catches undefined names (F821/F82), unused imports (F401),
         syntax (E9), assert/print (F63), statement errors (F7)
    """
    import ast as _ast
    import tempfile

    # Stage 1: AST parse
    try:
        _ast.parse(content, filename=file_path)
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg}"

    # Stage 2: Ruff lint via temp file
    if not file_path.endswith(".py"):
        return None

    tmp_fd = None
    tmp_path = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="lint_")
        os.close(tmp_fd)
        tmp_fd = None
        Path(tmp_path).write_text(content, encoding="utf-8")

        result = subprocess.run(
            [
                "ruff",
                "check",
                tmp_path,
                "--select",
                "E9,F63,F7,F82,F401",
                "--output-format",
                "concise",
                "--no-cache",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
        )
        if result.returncode != 0 and result.stdout.strip():
            # Replace temp path with real filename for readable errors
            errors = result.stdout.strip().replace(tmp_path, Path(file_path).name)
            return f"Ruff lint errors:\n{errors}"
    except FileNotFoundError:
        pass  # ruff not installed
    except subprocess.TimeoutExpired:
        pass  # ruff hung
    except Exception:
        pass  # don't block repair on lint infra failure
    finally:
        if tmp_fd is not None:
            try:
                os.close(tmp_fd)
            except OSError:
                pass
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return None


# =============================================================================
# MODEL CONFIGURATION - Based on `ollama list` output
# =============================================================================


class ModelTier(Enum):
    """Model tiers for different task complexities."""

    FAST = "fast"  # Quick checks, simple parsing (~1s)
    MEDIUM = "medium"  # Analysis, reasoning (~5s)
    CODE = "code"  # Code generation/analysis
    LARGE = "large"  # Complex reasoning, multi-step
    VISION = "vision"  # General vision tasks (screenshots, UI)
    CHART = "chart"  # Chart/graph analysis specialist
    MATH = "math"  # Math/logic specialist
    MASSIVE = "massive"  # Massive context (logs, repos)
    PLANNER = "planner"  # Agent coordination, tool use
    SEARCH = "search"  # Web search result analysis


# Available models from `ollama list` - ranked by capability per tier
AVAILABLE_MODELS = {
    ModelTier.FAST: [
        "qwen2.5:3b",  # 1.9 GB - fastest for simple tasks
        "qwen3:8b",  # 5.2 GB - good balance
    ],
    ModelTier.MEDIUM: [
        "gemma4:e4b",  # 9.6 GB - fast generalist with stronger structured output
        "qwen3:8b",  # 5.2 GB - lightweight fallback
        "phi4-reasoning:14b",  # 11 GB - high quality reasoning
        "phi4:14b-q4_K_M",  # 9.1 GB - if needed
    ],
    ModelTier.CODE: [
        "qwen2.5-coder:14b",  # 9 GB - good coder, RAM-safe default
        "qwen2.5-coder:7b",  # 4.7 GB - fast lightweight coder
        "qwen3-coder-next:latest",  # 51 GB - ONLY use when explicitly needed
    ],
    ModelTier.LARGE: [
        "phi4-reasoning:14b",  # 11 GB - fastest passing prompt-optimization benchmark
        "glm-4.7-flash:latest",  # 19 GB - fast non-reasoning fallback with solid logic
        "gemma4:26b",  # 17 GB - multimodal, function calling
    ],
    ModelTier.VISION: [
        "qwen3-vl:8b",  # 6.1 GB - best general vision
        "qwen2.5vl:7b",  # 6.0 GB - OCR, UI analysis
    ],
    ModelTier.CHART: [
        "llama3.2-vision:latest",  # fastest stable vision model for chart tasks
        "qwen2.5vl:7b",  # fallback: accurate but slower
        "qwen3-vl:8b",  # tertiary fallback
    ],
    ModelTier.MATH: [
        "phi4:14b-q4_K_M",  # 9.1 GB - competition-level math
        "phi4-reasoning:14b",  # 11 GB - high quality reasoning fallback
    ],
    ModelTier.MASSIVE: [
        "gpt-oss:20b",  # 13 GB - 128K context fallback
        "phi4-reasoning:14b",  # 11 GB - 128K context fallback
    ],
    ModelTier.PLANNER: [
        "gpt-oss:20b",  # 13 GB - tool-use, Harmony format
        "gemma4:26b",  # 17 GB - function calling
    ],
    ModelTier.SEARCH: [
        "gemma4:e4b",  # 9.6 GB - stronger summarization/analysis for search results
        "qwen3:8b",  # 5.2 GB - faster lightweight fallback
        "qwen2.5:3b",  # 1.9 GB - ultra-fast for simple queries
    ],
}

# Default model per tier
DEFAULT_MODELS = {
    # CRITICAL: OpenRouter (Claude 3.5 Sonnet) - Paid (200/day cap)
    "code_heavy": "openrouter:anthropic/claude-3.5-sonnet",
    # HIGH-FIDELITY FREE: OpenRouter Free Tier (1000/day cap)
    "code_standard": "openrouter:qwen/qwen-2.5-coder-32b-instruct:free",
    "planner": "openrouter:meta-llama/llama-3.3-70b-instruct:free",
    "reasoning": "openrouter:google/gemini-2.0-flash-thinking-exp:free",
    "massive": "openrouter:google/gemini-2.0-pro-exp-02-05:free",
    # LOCAL SPECIALIZED (Muscle)
    "vision": "qwen3-vl:8b",
    "chart": "llama3.2-vision:latest",
    "search": "gemma4:e4b",
}


def get_available_ollama_models() -> List[str]:
    """Get list of currently available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            models = []
            for line in lines:
                parts = line.split()
                if parts:
                    models.append(parts[0])
            return models
    except Exception as e:
        logger.warning(f"Could not get Ollama models: {e}")
    return []


def select_model_for_tier(
    tier: ModelTier, available: List[str] = None
) -> Optional[str]:
    """Select the best available model for a given tier."""
    if available is None:
        available = get_available_ollama_models()

    for model in AVAILABLE_MODELS.get(tier, []):
        # Check for exact match first
        if model in available:
            return model
        # Then check for base name match (model without tag)
        base_model = model.split(":")[0]
        for avail in available:
            avail_base = avail.split(":")[0]
            if avail_base == base_model:
                return avail

    # Fallback to default
    return DEFAULT_MODELS.get(tier)


# =============================================================================
# TASK TYPES AND ROUTING
# =============================================================================


class TaskType(Enum):
    """Types of tasks the orchestrator can handle."""

    # Diagnostic tasks
    HEALTH_CHECK = "health_check"
    ADVISOR_CHECK = "advisor_check"

    # Analysis tasks
    LOG_ANALYSIS = "log_analysis"
    CONTEXT_UPDATE = "context_update"

    # Code tasks
    CODE_FIX = "code_fix"
    CODE_REVIEW = "code_review"

    # Recovery tasks
    RECOVERY = "recovery"
    CONFIG_UPDATE = "config_update"

    # Vision tasks (NEW)
    SCREENSHOT_ANALYSIS = "screenshot_analysis"
    CHART_ANALYSIS = "chart_analysis"
    UI_DEBUG = "ui_debug"

    # Reasoning tasks (NEW)
    MATH_PROBLEM = "math_problem"
    LOGIC_VERIFY = "logic_verify"

    # Planning tasks (NEW)
    TASK_PLANNING = "task_planning"
    TOOL_DECISION = "tool_decision"

    # Massive context tasks (NEW)
    REPO_ANALYSIS = "repo_analysis"
    MASSIVE_LOG_ANALYSIS = "massive_log_analysis"

    # Search tasks (NEW)
    WEB_SEARCH = "web_search"
    NEWS_SEARCH = "news_search"
    FINANCE_SEARCH = "finance_search"

    # Developer search tasks (NEW)
    CODE_SEARCH = "code_search"  # Search for code snippets
    REPO_SEARCH = "repo_search"  # Search for repositories
    DOCS_SEARCH = "docs_search"  # Search documentation
    STACKOVERFLOW_SEARCH = "stackoverflow_search"  # Search Stack Overflow
    PACKAGE_SEARCH = "package_search"  # Search PyPI/npm packages

    # PM Workflow validation tasks (NEW)
    PM_WORKFLOW_CHECK = "pm_workflow_check"  # Check if PM workflow ran
    PICKS_VALIDATION = "picks_validation"  # Validate tomorrow's picks
    PLAN_REVIEW = "plan_review"  # Review trading plan

    # Improvement/optimization tasks (NEW)
    BACKTEST_ANALYSIS = "backtest_analysis"  # Analyze backtest results
    STRATEGY_IMPROVEMENT = "strategy_improvement"  # Suggest strategy improvements
    PERFORMANCE_REVIEW = "performance_review"  # Review overall performance
    WORKFLOW_ANALYSIS = "workflow_analysis"  # Analyze workflow efficiency

    # Autonomous research tasks (NEW)
    RESEARCH_STOCK = "research_stock"  # Research a specific stock
    FETCH_DATA = "fetch_data"  # Fetch missing data from yfinance
    OPTIMIZE_WATCHLIST = "optimize_watchlist"  # AI-optimize the watchlist
    CONTINUOUS_IMPROVEMENT = (
        "continuous_improvement"  # Full autonomous improvement cycle
    )


@dataclass
class Task:
    """Represents a task to be executed by an agent."""

    type: Optional[TaskType] = None
    description: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5  # 1=highest, 10=lowest
    created_at: datetime = field(default_factory=datetime.now)
    # Backwards-compatible alias (older code used task_type)
    task_type: Optional[TaskType] = field(default=None, repr=False)

    def __post_init__(self):
        # Handle legacy task_type parameter
        if self.type is None and self.task_type is not None:
            self.type = self.task_type
        # Convert string to TaskType enum
        if isinstance(self.type, str):
            try:
                self.type = TaskType(self.type)
            except ValueError:
                logger.warning(
                    f"Unknown task type string: {self.type}, trying uppercase"
                )
                try:
                    self.type = TaskType(self.type.upper())
                except (ValueError, AttributeError):
                    logger.error(f"Cannot convert task type: {self.type}")
                    raise ValueError(f"Invalid task type: {self.type}")
        if self.type is None:
            raise ValueError("Task.type is required")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "description": self.description,
            "data": self.data,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class TaskResult:
    """Result from executing a task."""

    success: bool
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    agent_used: str = ""
    model_used: str = ""
    duration_ms: int = 0

    # Backwards-compatible alias (older code used .output or .error)
    @property
    def output(self) -> str:
        return self.message

    @property
    def error(self) -> str:
        return self.message


# =============================================================================
# MODULAR AGENTS
# =============================================================================


class BaseAgent:
    """Base class for all agents."""

    def __init__(self, name: str, model_tier: ModelTier):
        self.name = name
        self.model_tier = model_tier
        self.model = None
        self._available_models = None
        self.llm_provider = "local"
        self.codex_command = "codex"
        self.codex_timeout = 120
        self.codex_use_stdin = False
        self.codex_extra_args = []

        try:
            from config.config_loader import get_llm_config

            cfg = get_llm_config(reload=True)  # Always read fresh from disk
            self.llm_provider = getattr(cfg, "provider", "local").lower()
            # Safety: never route to codex (credits exhausted)
            if self.llm_provider == "codex":
                logger.warning(f"{name}: provider was 'codex', forcing to 'local'")
                self.llm_provider = "local"
            self.codex_command = getattr(cfg, "codex_command", "codex")
            self.codex_timeout = getattr(cfg, "codex_timeout", 120)
            self.codex_use_stdin = getattr(cfg, "codex_use_stdin", False)
            self.codex_extra_args = getattr(cfg, "codex_extra_args", [])
        except Exception:
            pass

    def _get_model(self) -> str:
        """Get the best available model for this agent."""
        if self.llm_provider == "codex":
            return "codex"
        if self.model is None:
            if self._available_models is None:
                self._available_models = get_available_ollama_models()
            self.model = select_model_for_tier(self.model_tier, self._available_models)
        return self.model

    @staticmethod
    def _requires_think_false(model: str) -> bool:
        return "qwen3" in str(model or "").lower()

    def _call_ollama(self, prompt: str, system: str = None) -> Optional[str]:
        """Call Ollama API with the agent's model."""
        if self.llm_provider == "codex":
            return self._call_codex(prompt, system)

        model = self._get_model()
        if not model:
            logger.error(f"No model available for {self.name}")
            return None

        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
            }
            if system:
                payload["system"] = system
            if self._requires_think_false(model):
                payload["think"] = False

            response = requests.post(
                "http://localhost:11434/api/generate", json=payload, timeout=120
            )
            if response.status_code == 200:
                return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Ollama call failed for {self.name}: {e}")
        return None

    def _call_ollama_vision(
        self, prompt: str, image_path: str, system: str = None
    ) -> Optional[str]:
        """Call Ollama API with an image."""
        model = self._get_model()
        if not model:
            logger.error(f"No model available for {self.name}")
            return None

        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            payload = {
                "model": model,
                "prompt": prompt,
                "images": [image_data],
                "stream": False,
            }
            if system:
                payload["system"] = system
            if self._requires_think_false(model):
                payload["think"] = False

            response = requests.post(
                "http://localhost:11434/api/generate", json=payload, timeout=180
            )
            if response.status_code == 200:
                return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Ollama vision call failed for {self.name}: {e}")
        return None

    def _call_codex(self, prompt: str, system: str = None) -> Optional[str]:
        """Call Codex CLI with the agent prompt."""
        from autotrade.utils.codex_cli import run_codex

        combined_prompt = prompt
        if system:
            combined_prompt = f"SYSTEM:\\n{system}\\n\\nUSER:\\n{prompt}"

        success, stdout, stderr = run_codex(
            combined_prompt,
            command=self.codex_command,
            timeout=self.codex_timeout,
            use_stdin=self.codex_use_stdin,
            extra_args=self.codex_extra_args,
        )
        if not success:
            logger.error(f"Codex call failed for {self.name}: {stderr}")
            return None
        return stdout

    def execute(self, task: Task) -> TaskResult:
        """Execute a task. Override in subclasses."""
        raise NotImplementedError


class DiagnosticAgent(BaseAgent):
    """Fast agent for health checks and diagnostics."""

    def __init__(self):
        super().__init__("DiagnosticAgent", ModelTier.FAST)

    def execute(self, task: Task) -> TaskResult:
        start = time.time()

        if task.type == TaskType.HEALTH_CHECK:
            # Check if this is a workflow validation failure
            if task.data.get("problem"):
                return self._diagnose_workflow_failure(task, start)
            return self._check_system_health(task, start)
        elif task.type == TaskType.ADVISOR_CHECK:
            return self._check_advisors(task, start)

        return TaskResult(
            success=False,
            message=f"Unknown task type for DiagnosticAgent: {task.type}",
            agent_used=self.name,
        )

    def _diagnose_workflow_failure(self, task: Task, start: float) -> TaskResult:
        """
        Diagnose WHY a workflow task failed validation.

        This is the key piece - when MasterSupervisor detects a task
        ran but didn't produce expected results, this agent figures out why.
        """
        problem = task.data.get("problem", "unknown")
        task_name = task.data.get("task_name", "unknown")
        checks = task.data.get("checks", [])

        logger.info(f"[DiagnosticAgent] Analyzing workflow failure: {problem}")

        diagnosis = {
            "problem": problem,
            "root_cause": None,
            "suggested_fix": None,
            "analysis": [],
        }

        # Problem-specific analysis
        if problem == "overnight_research_no_signals":
            diagnosis["root_cause"] = (
                "Overnight research ran but produced no trading signals"
            )
            diagnosis["analysis"].append(
                "Check market_screener.py or web research modules"
            )
            diagnosis["analysis"].append("May need to lower score thresholds")
            diagnosis["suggested_fix"] = {
                "type": "review_research",
                "action": "check_screener_thresholds",
            }

        elif problem == "overnight_no_plan_generated":
            diagnosis["root_cause"] = "Overnight research did not create any plan files"
            diagnosis["analysis"].append(
                "Option 10 (overnight) may have crashed silently"
            )
            diagnosis["analysis"].append(
                "Check logs/app.jsonl for errors during overnight run"
            )
            diagnosis["suggested_fix"] = {
                "type": "check_logs",
                "action": "review_overnight_errors",
            }

        elif problem == "signals_exist_but_no_execution":
            # This is the EXACT problem user had!
            diagnosis["root_cause"] = (
                "DayManager has signals but execution never triggered"
            )
            diagnosis["analysis"].append("Signals file exists with entries")
            diagnosis["analysis"].append(
                "But trade execution (Option 5 or continuous) never ran"
            )
            diagnosis["analysis"].append("Check if execute_plan() was called")
            diagnosis["suggested_fix"] = {
                "type": "trigger_execution",
                "action": "force_execute_plan",
            }

        elif problem == "empty_signals_file":
            diagnosis["root_cause"] = "Signals file exists but is empty"
            diagnosis["analysis"].append("DayManager created file but loaded 0 signals")
            diagnosis["analysis"].append(
                "Check if _load_signals() is reading the right file"
            )
            diagnosis["analysis"].append(
                "Verify morning_game_plan has buy_signals array"
            )
            diagnosis["suggested_fix"] = {
                "type": "reload_signals",
                "action": "dm.load_signals_from_plan()",
            }

        elif problem == "no_signals_file":
            diagnosis["root_cause"] = "No signals file found for DayManager"
            diagnosis["analysis"].append(
                "DayManager.get_signals_path() returned non-existent path"
            )
            diagnosis["analysis"].append("Need to run load_signals_from_plan() first")
            diagnosis["suggested_fix"] = {
                "type": "reload_signals",
                "action": "dm.load_signals_from_plan()",
            }

        else:
            diagnosis["root_cause"] = f"Unknown workflow problem: {problem}"
            diagnosis["analysis"].append("Manual investigation needed")

        # Save diagnosis for review
        try:
            from datetime import datetime

            diagnosis_path = (
                Path("logs")
                / f"supervisor_diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            import json

            with open(diagnosis_path, "w") as f:
                json.dump(diagnosis, f, indent=2)
            logger.info(f"[DiagnosticAgent] Diagnosis saved: {diagnosis_path.name}")
        except Exception:
            pass

        return TaskResult(
            success=diagnosis["root_cause"] is not None,
            message=diagnosis["root_cause"] or "Could not determine root cause",
            data={
                "diagnosis": diagnosis,
                "suggested_fix": diagnosis.get("suggested_fix"),
                "analysis": diagnosis["analysis"],
            },
            agent_used=self.name,
            model_used="rule-based",
            duration_ms=int((time.time() - start) * 1000),
        )

    def _check_system_health(self, task: Task, start: float) -> TaskResult:
        """Quick system health check.

        Only checks essential services (ollama, logs). The web server is
        optional and not required for scheduled/autonomous operation.
        """
        results = {
            "server": self._check_server(),
            "ollama": self._check_ollama(),
            "logs": self._check_logs_exist(),
        }

        # Server is optional for scheduled mode - only require ollama + logs
        essential_healthy = all(
            v.get("healthy", False) for k, v in results.items() if k != "server"
        )

        return TaskResult(
            success=essential_healthy,
            message="System healthy" if essential_healthy else "Issues detected",
            data=results,
            agent_used=self.name,
            model_used="rule-based",
            duration_ms=int((time.time() - start) * 1000),
        )

    def _check_server(self) -> Dict[str, Any]:
        try:
            resp = requests.get("http://127.0.0.1:8000/api/health", timeout=5)
            return {"healthy": resp.status_code == 200, "status": resp.status_code}
        except:
            return {"healthy": False, "error": "Connection refused"}

    def _check_ollama(self) -> Dict[str, Any]:
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            return {"healthy": resp.status_code == 200}
        except:
            return {"healthy": False, "error": "Ollama not running"}

    def _check_logs_exist(self) -> Dict[str, Any]:
        log_dir = Path("logs")
        return {
            "healthy": log_dir.exists(),
            "app_jsonl": (log_dir / "app.jsonl").exists()
            if log_dir.exists()
            else False,
        }

    def _check_advisors(self, task: Task, start: float) -> TaskResult:
        """Check if advisors are working."""
        try:
            from autotrade.advisors.position_advisor import PositionAdvisor

            advisor = PositionAdvisor()
            llm_ok = advisor._check_llm_available()
            return TaskResult(
                success=llm_ok,
                message="Advisor LLM available"
                if llm_ok
                else "Advisor in fallback mode",
                data={"llm_available": llm_ok, "model": advisor.model},
                agent_used=self.name,
                model_used="rule-based",
                duration_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:
            return TaskResult(
                success=False,
                message=f"Advisor check failed: {e}",
                agent_used=self.name,
                duration_ms=int((time.time() - start) * 1000),
            )


class AnalysisAgent(BaseAgent):
    """Agent for log analysis and pattern detection."""

    def __init__(self):
        super().__init__("AnalysisAgent", ModelTier.MEDIUM)

    def execute(self, task: Task) -> TaskResult:
        start = time.time()

        if task.type == TaskType.LOG_ANALYSIS:
            return self._analyze_logs(task, start)
        elif task.type == TaskType.CONTEXT_UPDATE:
            return self._update_context(task, start)

        return TaskResult(
            success=False,
            message=f"Unknown task type for AnalysisAgent: {task.type}",
            agent_used=self.name,
        )

    def _analyze_logs(self, task: Task, start: float) -> TaskResult:
        """Analyze recent logs for issues."""
        errors = []
        warnings = []

        # Scan app.jsonl
        app_log = Path("logs/app.jsonl")
        if app_log.exists():
            try:
                with open(app_log) as f:
                    lines = f.readlines()[-50:]
                for line in lines:
                    try:
                        entry = json.loads(line)
                        level = entry.get("level", "").upper()
                        if level == "ERROR":
                            errors.append(entry.get("message", "")[:100])
                        elif level == "WARNING":
                            warnings.append(entry.get("message", "")[:100])
                    except:
                        pass
            except Exception as e:
                logger.warning(f"Could not read app.jsonl: {e}")

        # Use LLM to summarize if there are issues
        summary = None
        if errors and self._get_model():
            prompt = f"Summarize these trading system errors briefly:\n" + "\n".join(
                errors[:5]
            )
            summary = self._call_ollama(
                prompt, "You are a trading system diagnostician. Be brief."
            )

        return TaskResult(
            success=len(errors) == 0,
            message=summary or f"Found {len(errors)} errors, {len(warnings)} warnings",
            data={"errors": errors[:10], "warnings": warnings[:10]},
            agent_used=self.name,
            model_used=self._get_model() or "rule-based",
            duration_ms=int((time.time() - start) * 1000),
        )

    def _update_context(self, task: Task, start: float) -> TaskResult:
        """Generate daily context snapshot."""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "modules": self._scan_key_modules(),
            "recent_changes": self._get_recent_changes(),
        }

        snapshot_file = (
            Path("logs") / f"context_snapshot_{datetime.now().strftime('%Y%m%d')}.json"
        )
        with open(snapshot_file, "w") as f:
            json.dump(snapshot, f, indent=2)

        return TaskResult(
            success=True,
            message=f"Context snapshot saved to {snapshot_file}",
            data=snapshot,
            agent_used=self.name,
            model_used="rule-based",
            duration_ms=int((time.time() - start) * 1000),
        )

    def _scan_key_modules(self) -> List[str]:
        """List key Python modules."""
        key_files = [
            "day_manager.py",
            "pm_workflow.py",
            "position_advisor.py",
            "agentic_advisor.py",
            "agentic_orchestrator.py",
        ]
        return [f for f in key_files if Path(f).exists()]

    def _get_recent_changes(self) -> List[str]:
        """Get recently modified files."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD~5"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip().split("\n")[:10]
        except:
            pass
        return []


# =============================================================================
# CODE REPAIR CONTEXT ROUTER
# =============================================================================


class RepairComplexity(Enum):
    """Complexity tier for a code repair task â€” drives model selection."""

    SIMPLE = "simple"  # Syntax / obvious one-liner  â†’ try qwen2.5-coder:7b first
    MEDIUM = "medium"  # Logic / attribute errors     â†’ try qwen2.5-coder:14b first
    COMPLEX = (
        "complex"  # Multi-file / architectural repairs prefer stronger cloud models
    )


@dataclass
class RepairContext:
    """All context that the repair model needs to produce a correct fix."""

    complexity: RepairComplexity
    primary_file: str
    primary_content: str  # full file text
    traceback: str
    error_type: str
    line_hint: Optional[int]
    related_files: Dict[str, str]  # path â†’ excerpted content
    conventions_snippet: str  # key sections of PROJECT_NOTES.md
    prompt_context: str  # assembled prompt block for the LLM
    force_escalation: bool = (
        False  # skip the default cascade and use the escalation cascade
    )


class CodeRepairContextRouter:
    """
    Context assembler + model-cascade selector for the self-healing code repair pipeline.

    Responsibilities:
      1. Classify how complex the error is (SIMPLE / MEDIUM / COMPLEX).
      2. Pull ALL relevant context: full file, related-file excerpts (from traceback),
         project conventions from PROJECT_NOTES.md.
      3. Return an ordered *model cascade* that matches the complexity.
         The caller tries models left-to-right, escalating on failure.

    Model cascade:
      SIMPLE/MEDIUM/COMPLEX â†’ OpenAI first, then primary OpenRouter coder
      force_escalation      â†’ stronger OpenRouter-only escalation path
      optional qwen3        â†’ append only when explicitly enabled
    """

    _PROJECT_ROOT = Path(__file__).parent.parent.parent
    _PROJECT_NOTES_MD = _PROJECT_ROOT / "PROJECT_NOTES.md"

    # Direct error-type â†’ complexity mapping
    _COMPLEXITY_MAP: Dict[str, RepairComplexity] = {
        "SyntaxError": RepairComplexity.SIMPLE,
        "IndentationError": RepairComplexity.SIMPLE,
        "TabError": RepairComplexity.SIMPLE,
        "NameError": RepairComplexity.SIMPLE,
        "UnboundLocalError": RepairComplexity.SIMPLE,
        "KeyError": RepairComplexity.SIMPLE,
        "IndexError": RepairComplexity.SIMPLE,
        "AttributeError": RepairComplexity.MEDIUM,
        "TypeError": RepairComplexity.MEDIUM,
        "ValueError": RepairComplexity.MEDIUM,
        "ImportError": RepairComplexity.MEDIUM,
        "ModuleNotFoundError": RepairComplexity.MEDIUM,
        "RuntimeError": RepairComplexity.COMPLEX,
        "NotImplementedError": RepairComplexity.COMPLEX,
        "RecursionError": RepairComplexity.COMPLEX,
    }

    # Model cascade: local-first when qwen3 repair is enabled, then
    # cost-controlled OpenAI coding models, then Claude as high-precision fallback.
    _MODEL_CASCADE: Dict[RepairComplexity, List[str]] = {
        RepairComplexity.SIMPLE: [
            "openai:gpt-5.4-mini",
            "openai:gpt-5.3-codex",
            "openrouter:anthropic/claude-sonnet-4-6",
        ],
        RepairComplexity.MEDIUM: [
            "openai:gpt-5.4-mini",
            "openai:gpt-5.3-codex",
            "openrouter:anthropic/claude-sonnet-4-6",
        ],
        RepairComplexity.COMPLEX: [
            "openai:gpt-5.4-mini",
            "openai:gpt-5.3-codex",
            "openrouter:anthropic/claude-sonnet-4-6",
        ],
    }

    _ESCALATION_CASCADE: List[str] = [
        "openai:gpt-5.4-mini",
        "openai:gpt-5.3-codex",
        "openrouter:anthropic/claude-sonnet-4-6",
    ]

    # Minimum free RAM (GB) required before loading each model.
    # qwen3-coder-next is excluded â€” we unload everything first and let
    # Ollama handle memory (the OS will page if needed).
    _RAM_REQUIRED: Dict[str, float] = {
        "qwen3:8b": 5.0,
        "qwen2.5-coder:7b": 6.0,
        "qwen2.5-coder:14b": 11.0,
        "devstral:24b-small-2505-q4_K_M": 16.0,
        # qwen3-coder-next deliberately omitted â€” no RAM guard
    }

    # Models that need ALL other models unloaded before loading
    _HEAVY_MODELS = {"qwen3-coder-next:latest"}

    # PROJECT_NOTES.md section headers worth including as conventions
    _CONVENTION_HEADERS = [
        "## Development Conventions",
        "## Critical Data Gotchas",
        "## Module Map",
    ]

    # ------------------------------------------------------------------ helpers

    def _classify(self, error_type: str, traceback_str: str) -> RepairComplexity:
        """Heuristic complexity classification."""
        if error_type in self._COMPLEXITY_MAP:
            return self._COMPLEXITY_MAP[error_type]
        # Multiple project files in traceback â†’ escalate
        tb_files = re.findall(r'File ["\']([^"\']+\.py)["\']', traceback_str)
        project_files = [
            f
            for f in tb_files
            if "autotrade" in f.lower()
            or str(self._PROJECT_ROOT).replace("\\", "/").lower()
            in f.lower().replace("\\", "/")
        ]
        if len(project_files) >= 3:
            return RepairComplexity.COMPLEX
        if len(project_files) >= 2:
            return RepairComplexity.MEDIUM
        return RepairComplexity.MEDIUM

    def _conventions_snippet(self) -> str:
        """Extract key development conventions from PROJECT_NOTES.md (â‰¤3000 chars)."""
        try:
            text = self._PROJECT_NOTES_MD.read_text(encoding="utf-8")
            sections: List[str] = []
            for header in self._CONVENTION_HEADERS:
                idx = text.find(header)
                if idx < 0:
                    continue
                end = text.find("\n## ", idx + 1)
                sections.append(text[idx : end if end > idx else idx + 1500])
            return "\n\n".join(sections)[:3000] if sections else ""
        except Exception:
            return ""

    def _related_files(self, traceback_str: str, primary_path: str) -> Dict[str, str]:
        """
        Return excerpts of project files referenced in the traceback,
        excluding the primary file (already included in full).
        Limits each related file to 2 000 chars to stay within context budget.
        """
        related: Dict[str, str] = {}
        primary = str(Path(primary_path))
        tb_files = re.findall(r'File ["\']([^"\']+\.py)["\']', traceback_str)
        seen: set = set()
        for fp in tb_files:
            p = Path(fp)
            key = str(p)
            if key in seen or key == primary:
                continue
            seen.add(key)
            if not p.exists():
                continue
            # Only include project files
            fp_norm = str(p).lower().replace("\\", "/")
            root_norm = str(self._PROJECT_ROOT).lower().replace("\\", "/")
            if "autotrade" not in fp_norm and root_norm not in fp_norm:
                continue
            try:
                related[key] = p.read_text(encoding="utf-8")[:2000]
            except Exception:
                pass
        return related

    # ------------------------------------------------------------------ public API

    def build(
        self,
        file_path: str,
        traceback_str: str,
        error_type: str,
        line_hint: Optional[int] = None,
        force_escalation: bool = False,
    ) -> RepairContext:
        """
        Assemble a :class:`RepairContext` with full file content, related-file
        excerpts, project conventions, and a ready-to-use ``prompt_context`` string.
        """
        p = Path(file_path)
        content = p.read_text(encoding="utf-8")

        complexity = (
            RepairComplexity.COMPLEX
            if force_escalation
            else self._classify(error_type, traceback_str)
        )

        related = self._related_files(traceback_str, file_path)
        conventions = self._conventions_snippet()

        # Build a generous window around the error line (Â±150 lines).
        # With 64K context, we can afford a 300-line "local map" by default.
        lines = content.splitlines()
        total_lines = len(lines)
        focus_section = ""
        if line_hint:
            start = max(0, line_hint - 150)
            end = min(total_lines, line_hint + 150)
            numbered = [
                f"{'>>>' if i + 1 == line_hint else '   '} {i + 1:4d} | {lines[i]}"
                for i in range(start, end)
            ]
            focus_section = (
                "=" * 60
                + f"\nFOCUS AREA (lines {start + 1}-{end} of {file_path}, {total_lines} lines total) â€” error is on the >>> line"
                + "\n"
                + "=" * 60
                + "\n"
                + "\n".join(numbered)
                + "\n"
            )

        # Assemble the prompt context block â€” NO full file dump
        sections: List[str] = [
            "=" * 60,
            f"PRIMARY FILE: {file_path} ({total_lines} lines)",
            "=" * 60,
        ]
        if focus_section:
            sections.append(focus_section)
        else:
            # No line hint â€” include first + last 30 lines as orientation
            head = "\n".join(
                f"  {i + 1:4d} | {lines[i]}" for i in range(min(30, total_lines))
            )
            tail_start = max(0, total_lines - 30)
            tail = "\n".join(
                f"  {i + 1:4d} | {lines[i]}" for i in range(tail_start, total_lines)
            )
            sections += [head, "  ...", tail]
        sections.append("")
        sections.append("NOTE: Use `read_file` tool to see more of the file if needed.")
        if related:
            sections.append("RELATED FILES (referenced in traceback):")
            for rpath, rcontent in related.items():
                sections += [f"\n--- {rpath} (first 2000 chars) ---", rcontent, ""]
        if conventions:
            sections += ["\nPROJECT CONVENTIONS (from PROJECT_NOTES.md):", conventions, ""]

        prompt_context = "\n".join(sections)

        return RepairContext(
            complexity=complexity,
            primary_file=file_path,
            primary_content=content,
            traceback=traceback_str,
            error_type=error_type or "Unknown",
            line_hint=line_hint,
            related_files=related,
            conventions_snippet=conventions,
            prompt_context=prompt_context,
            force_escalation=force_escalation,
        )

    def get_model_cascade(
        self,
        complexity: RepairComplexity,
        force_escalation: bool = False,
    ) -> List[str]:
        """Return the ordered list of models to try for this repair."""
        if force_escalation:
            return list(self._ESCALATION_CASCADE)

        # Return the configured cascade (now prioritized for OpenAI)
        return list(
            self._MODEL_CASCADE.get(
                complexity, self._MODEL_CASCADE[RepairComplexity.MEDIUM]
            )
        )

    def ram_ok_for_model(self, model: str) -> bool:
        """Return True if the system has enough free RAM to safely load *model*."""
        required = self._RAM_REQUIRED.get(model, 4.0)
        try:
            import psutil

            available_gb = psutil.virtual_memory().available / (1024**3)
            if available_gb < required:
                logger.warning(
                    f"[ContextRouter] Skipping {model}: needs {required:.0f} GB free,"
                    f" only {available_gb:.1f} GB available"
                )
                return False
        except ImportError:
            pass  # can't check â†’ optimistic
        return True


class CodeAgent(BaseAgent):
    """Agentic code repair agent using tool-calling loop.

    Instead of single-shot /api/generate + manual JSON parsing, this agent
    uses OllamaClient.chat_with_tools() to give the model access to:
      - read_file: examine the broken file and related files
      - search_files: ripgrep for symbol usages
      - replace_in_file: apply patches with automatic AST+ruff validation & rollback
      - execute_command: run ruff/py_compile (restricted allowlist)
      - attempt_completion: signal that the fix is done

    The ToolExecutor handles lint-on-edit internally: if a replace_in_file
    introduces syntax errors or undefined names, it rolls back automatically
    and returns the error to the model, which can then retry with a corrected
    patch.  This is the self-correcting loop that prevents death spirals.
    """

    # Tools allowed in repair mode (subset of NATIVE_TOOL_SCHEMAS)
    _REPAIR_TOOL_NAMES = {
        "read_file",
        "search_files",
        "replace_in_file",
        "execute_command",
        "attempt_completion",
    }

    # Keep in sync with CodeRepairContextRouter heavy model set.
    _HEAVY_MODELS = {"qwen3-coder-next:latest"}

    # Commands allowed via execute_command in repair mode
    _ALLOWED_CMD_PREFIXES = [
        "ruff check",
        "ruff format",
        'python -c "import py_compile',
        "python -c 'import py_compile",
        "pytest tests/",
        "python -m pytest tests/",
    ]

    _MAX_REPAIR_TURNS = 200

    # OpenAI daily call limit â€” prevents burning API quota on unfixable errors
    _openai_calls_today: int = 0
    _openai_date: Optional[str] = None
    _openai_daily_limit: int = 10

    def __init__(self):
        super().__init__("CodeAgent", ModelTier.CODE)
        self._context_router = CodeRepairContextRouter()
        self._ollama_client = None
        self._tool_executor = None
        self._repair_openai_enabled = False
        self._repair_codex_enabled = False
        self._repair_qwen3_enabled = False
        try:
            cfg = get_llm_config()
            self._repair_openai_enabled = bool(
                getattr(cfg, "repair_openai_enabled", False)
            )
            self._repair_codex_enabled = bool(
                getattr(cfg, "repair_codex_enabled", False)
            )
            self._repair_qwen3_enabled = bool(
                getattr(cfg, "repair_qwen3_enabled", False)
            )
        except Exception:
            pass

    def _repair_flag_enabled(self, env_name: str, config_value: bool) -> bool:
        env_val = os.environ.get(env_name)
        if env_val is None:
            return bool(config_value)
        return str(env_val).strip().lower() not in {"0", "false", "no", "off"}

    def _repair_qwen3_allowed(self) -> bool:
        return self._repair_flag_enabled(
            "AUTOTRADE_ENABLE_QWEN3_REPAIR", self._repair_qwen3_enabled
        )

    def _repair_api_only_allowed(self) -> bool:
        return self._repair_flag_enabled("AUTOTRADE_CODE_REPAIR_API_ONLY", False)

    def _build_repair_cascade(
        self,
        ctx: Optional["RepairContext"],
        force_escalation: bool,
    ) -> List[str]:
        cascade = (
            self._context_router.get_model_cascade(ctx.complexity, force_escalation)
            if ctx
            else list(CodeRepairContextRouter._ESCALATION_CASCADE)
            if force_escalation
            else list(CodeRepairContextRouter._MODEL_CASCADE[RepairComplexity.MEDIUM])
        )

        if self._repair_api_only_allowed():
            return [
                model_name
                for model_name in cascade
                if str(model_name).startswith(("openai:", "openrouter:"))
            ]

        if self._repair_qwen3_allowed() and "qwen3-coder-next:latest" not in cascade:
            cascade.insert(0, "qwen3-coder-next:latest")

        return cascade

    def _get_ollama_client(self):
        """Lazy-init OllamaClient from local_coding_agent."""
        if self._ollama_client is None:
            try:
                from autotrade.core.local_coding_agent import OllamaClient

                self._ollama_client = OllamaClient()
            except ImportError:
                logger.warning(
                    "[CodeAgent] Could not import OllamaClient, falling back to basic mode"
                )
        return self._ollama_client

    def _get_tool_executor(self):
        """Lazy-init ToolExecutor from local_coding_agent."""
        if self._tool_executor is None:
            try:
                from autotrade.core.local_coding_agent import ToolExecutor

                self._tool_executor = ToolExecutor(PROJECT_DIR)
            except ImportError:
                logger.warning("[CodeAgent] Could not import ToolExecutor")
        return self._tool_executor

    def _get_repair_tool_schemas(self) -> List[Dict]:
        """Return the subset of NATIVE_TOOL_SCHEMAS for repair mode."""
        try:
            from autotrade.core.local_coding_agent import NATIVE_TOOL_SCHEMAS

            return [
                t
                for t in NATIVE_TOOL_SCHEMAS
                if t["function"]["name"] in self._REPAIR_TOOL_NAMES
            ]
        except ImportError:
            return []

    def _is_command_allowed(self, cmd: str) -> bool:
        """Check if a command is in the repair-mode allowlist."""
        cmd_stripped = cmd.strip()
        return any(
            cmd_stripped.startswith(prefix) for prefix in self._ALLOWED_CMD_PREFIXES
        )

    def execute(self, task: Task) -> TaskResult:
        start = time.time()

        if task.type == TaskType.CODE_FIX:
            return self._analyze_code_issue(task, start)

        return TaskResult(
            success=False,
            message=f"Unknown task type for CodeAgent: {task.type}",
            agent_used=self.name,
        )

    def _openrouter_repair(
        self,
        file_path: str,
        error: str,
        error_type: str,
        model: str,
        ctx: Optional["RepairContext"],
        start: float,
    ) -> Optional[TaskResult]:
        """
        High-fidelity repair via OpenRouter API.
        Funnels requests to Claude 3.5 Sonnet or similar high-tier models.
        """
        client = get_openrouter_client()
        if not client.available:
            logger.warning("[CodeAgent] OpenRouter API key missing â€” skipping")
            return None

        abs_file = Path(file_path).resolve()
        try:
            content = abs_file.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"[CodeAgent] Cannot read {file_path}: {e}")
            return None

        # Build a focused excerpt (Â±80 lines around error)
        line_num = None
        line_match = re.search(r"line\s+(\d+)", error, re.IGNORECASE)
        if line_match:
            line_num = int(line_match.group(1))
        elif ctx and ctx.line_hint:
            line_num = ctx.line_hint

        all_lines = content.splitlines()
        exact_block = ""
        if line_num:
            s_line = max(0, line_num - 80)
            e_line = min(len(all_lines), line_num + 80)
            numbered = [f"{i + 1:4d} | {all_lines[i]}" for i in range(s_line, e_line)]
            excerpt = (
                f"# Lines {s_line + 1}-{e_line} of {file_path} ({len(all_lines)} total)\n"
                + "\n".join(numbered)
            )
            block_start = max(0, line_num - 8)
            block_end = min(len(all_lines), line_num + 8)
            exact_block = "\n".join(all_lines[block_start:block_end])
        else:
            numbered = [
                f"{i + 1:4d} | {all_lines[i]}" for i in range(min(150, len(all_lines)))
            ]
            excerpt = (
                f"# First 150 lines of {file_path} ({len(all_lines)} total)\n"
                + "\n".join(numbered)
            )
            exact_block = "\n".join(all_lines[:20])

        prompt = f"""Fix this Python error in the AutoTrade system. 
Return ONLY a valid JSON object. No prose, no markdown fences.

ERROR TYPE: {error_type or "Unknown"}
ERROR: {error}

{excerpt}

EXACT_REPLACE_BLOCK (copy/paste for your search string):
<<<
{exact_block}
>>>

Return this JSON schema:
{{
  "summary": "description of the fix",
  "changes": [
    {{
      "file": "{file_path}",
      "search": "<exact multi-line text from file to replace>",
      "replace": "<new fixed text>",
      "reason": "why this fix",
      "line_start": <optional 1-based start line>,
      "line_end": <optional 1-based end line>
    }}
  ]
}}

Rules:
1. 'search' must be an EXACT substring that appears exactly ONCE in the file.
2. If you cannot guarantee an exact search, use line_start/line_end instead.
3. Produce a minimal, idiomatic Python fix.
"""

        system_prompt = "You are a senior Python developer. Respond ONLY with raw JSON."

        logger.info(f"[CodeAgent] Calling OpenRouter: {model}")
        response = client.chat(
            prompt, system=system_prompt, model=model, temperature=0.1
        )

        if not response.success:
            logger.warning(f"[CodeAgent] OpenRouter {model} failed: {response.error}")
            return None

        reply = response.content.strip()
        # Clean markdown if model ignored instructions
        if "```json" in reply:
            reply = reply.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in reply:
            reply = reply.split("```", 1)[1].split("```", 1)[0].strip()

        try:
            fix_data = json.loads(reply)
            changes = fix_data.get("changes", [])
            if not changes:
                return None

            file_buffers: Dict[str, str] = {}
            applied = 0
            for change in changes:
                target_path = change.get("file") or file_path
                search = change.get("search", "")
                replace = change.get("replace", "")
                line_start = change.get("line_start")
                line_end = change.get("line_end")
                if not search:
                    if not (line_start and line_end):
                        continue
                if target_path not in file_buffers:
                    try:
                        file_buffers[target_path] = Path(target_path).read_text(
                            encoding="utf-8"
                        )
                    except Exception as e:
                        logger.warning(
                            f"[CodeAgent] OpenRouter cannot read {target_path}: {e}"
                        )
                        continue
                buf = file_buffers[target_path]
                if search:
                    if search not in buf:
                        logger.warning(
                            f"[CodeAgent] OpenRouter search text not found in {target_path}, skipping"
                        )
                        continue
                    if buf.count(search) > 1:
                        logger.warning(
                            f"[CodeAgent] OpenRouter search text not unique in {target_path}, skipping"
                        )
                        continue
                    file_buffers[target_path] = buf.replace(search, replace, 1)
                    applied += 1
                else:
                    try:
                        start_idx = int(line_start) - 1
                        end_idx = int(line_end)
                    except (TypeError, ValueError):
                        logger.warning(
                            f"[CodeAgent] OpenRouter invalid line range in {target_path}, skipping"
                        )
                        continue
                    if start_idx < 0 or end_idx <= start_idx:
                        logger.warning(
                            f"[CodeAgent] OpenRouter invalid line bounds in {target_path}, skipping"
                        )
                        continue
                    lines = buf.splitlines(keepends=True)
                    if end_idx > len(lines):
                        logger.warning(
                            f"[CodeAgent] OpenRouter line range out of bounds in {target_path}, skipping"
                        )
                        continue
                    replacement = (
                        replace
                        if replace.endswith("\n") or replace == ""
                        else replace + "\n"
                    )
                    new_lines = lines[:start_idx] + [replacement] + lines[end_idx:]
                    file_buffers[target_path] = "".join(new_lines)
                    applied += 1

            if applied > 0:
                modified_files = sorted(
                    str(Path(path).resolve()) for path in file_buffers
                )
                for target_path, new_content in file_buffers.items():
                    target_file = Path(target_path)
                    if target_file.suffix == ".py":
                        lint_err = _lint_python_content(new_content, target_path)
                        if lint_err:
                            logger.warning(
                                f"[CodeAgent] OpenRouter fix for {target_path} invalid: {lint_err}"
                            )
                            return None
                    backup = target_file.with_suffix(target_file.suffix + ".bak")
                    backup.write_text(
                        target_file.read_text(encoding="utf-8"), encoding="utf-8"
                    )
                    target_file.write_text(new_content, encoding="utf-8")

                summary = fix_data.get("summary", "Fixed via OpenRouter")
                logger.info(f"[CodeAgent] SUCCESS via OpenRouter {model}: {summary}")
                # Return JSON with summary and empty changes list (already applied)
                return TaskResult(
                    success=True,
                    message=json.dumps({"summary": summary, "changes": []}),
                    data={
                        "file": file_path,
                        "model": f"openrouter:{model}",
                        "changes_applied": applied,
                        "fix_details": changes,
                        "modified_files": modified_files,
                        "persisted_to_disk": True,
                    },
                    agent_used=self.name,
                    model_used=f"openrouter:{model}",
                    duration_ms=int((time.time() - start) * 1000),
                )

        except json.JSONDecodeError:
            logger.warning(f"[CodeAgent] OpenRouter {model} returned invalid JSON")

        return None

    def _openrouter_agentic_repair(
        self,
        start: float,
        file_path: str,
        error: str,
        error_type: str,
        ctx: Optional["RepairContext"],
        model: str,
        system_msg: str,
        user_msg: str,
        repair_root: Path,
        tools: List[Dict],
    ) -> Optional[TaskResult]:
        """Minimal tool-calling repair loop using OpenRouter + ToolExecutor."""
        from autotrade.core.local_coding_agent import AgentMode, ToolExecutor

        client = get_openrouter_client()
        if not client.available:
            logger.warning("[CodeAgent] OpenRouter API key missing â€” skipping")
            return None

        executor = ToolExecutor(repair_root)

        abs_file = Path(file_path).resolve()
        executor._files_read.add(str(abs_file))

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        # Pre-seed a large read to avoid tiny-window loops.
        try:
            pre_read = executor.execute(
                "read_file",
                {"path": str(abs_file), "start_line": 1, "end_line": 320},
                AgentMode.ACT,
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "prefetch_read",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps(
                                    {
                                        "path": str(abs_file),
                                        "start_line": 1,
                                        "end_line": 320,
                                    }
                                ),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": "prefetch_read",
                    "content": json.dumps(pre_read, default=str)[:4000],
                }
            )
        except Exception as e:
            logger.warning(f"[CodeAgent] Pre-read failed: {e}")

        fix_applied = False
        completion_result = None
        last_error = "No tool calls produced"
        no_tool_call_strikes = 0
        max_turns = 6

        for turn in range(max_turns):
            logger.info(f"[CodeAgent] Turn {turn + 1}/{max_turns}")

            response = client.chat(
                "",
                messages=messages,
                model=model,
                temperature=0.2,
                max_tokens=2048,
                tools=tools,
                tool_choice="auto",
            )

            tool_calls = response.tool_calls
            content = (response.content or "").strip()

            if not tool_calls:
                if content:
                    messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": "Use the tools to fix the error. Start with read_file over ~300 lines around the error, then replace_in_file, then attempt_completion.",
                        }
                    )
                    no_tool_call_strikes += 1
                    if no_tool_call_strikes >= 2:
                        last_error = f"{model} did not issue tool calls after 2 prompts"
                        logger.warning(f"[CodeAgent] {last_error}")
                        break
                    continue
                last_error = f"{model} returned no tool calls on turn {turn + 1}"
                logger.warning(f"[CodeAgent] {last_error}")
                break

            no_tool_call_strikes = 0
            messages.append(
                {
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": tool_calls,
                }
            )

            for idx, tc in enumerate(tool_calls):
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                args = fn.get("arguments", {})

                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                if "start" in args and "start_line" not in args:
                    args["start_line"] = args.pop("start")
                if "end" in args and "end_line" not in args:
                    args["end_line"] = args.pop("end")

                if "path" in args:
                    p = args["path"].replace("\\", "/")
                    try:
                        p_abs = Path(p).resolve()
                        p = str(p_abs.relative_to(repair_root)).replace("\\", "/")
                    except (ValueError, OSError):
                        pass
                    args["path"] = p

                if tool_name == "execute_command":
                    cmd = args.get("command", "")
                    if not self._is_command_allowed(cmd):
                        tool_result = {
                            "success": False,
                            "error": "Command not allowed in repair mode. Only ruff, py_compile, pytest are permitted.",
                        }
                    else:
                        tool_result = executor.execute(tool_name, args, AgentMode.ACT)
                elif tool_name == "attempt_completion":
                    completion_result = args.get("result", "Fix completed")
                    fix_applied = executor._edits_made > 0
                    tool_result = {
                        "success": True,
                        "completed": True,
                        "result": completion_result,
                    }
                elif tool_name in self._REPAIR_TOOL_NAMES:
                    if (
                        tool_name == "read_file"
                        and "start_line" in args
                        and "end_line" in args
                    ):
                        try:
                            start_line = int(args["start_line"])
                            end_line = int(args["end_line"])
                            if (end_line - start_line) < 200:
                                center = (start_line + end_line) // 2
                                args["start_line"] = max(1, center - 150)
                                args["end_line"] = center + 150
                                logger.info(
                                    f"[CodeAgent] Range too small ({end_line - start_line} lines). Expanding to 300-line context."
                                )
                        except (ValueError, TypeError):
                            pass
                    tool_result = executor.execute(tool_name, args, AgentMode.ACT)
                else:
                    tool_result = {
                        "success": False,
                        "error": f"Tool '{tool_name}' not available in repair mode",
                    }

                if tool_name == "replace_in_file" and tool_result.get("success"):
                    fix_applied = True

                tc_id = tc.get("id") or f"call_{turn}_{idx}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": json.dumps(tool_result, default=str)[:4000],
                    }
                )

            if completion_result is not None:
                break

        if fix_applied:
            summary = completion_result or "Fix applied via OpenRouter tools"
            logger.info(f"[CodeAgent] SUCCESS via OpenRouter tools {model}: {summary}")
            return TaskResult(
                success=True,
                message=summary,
                data={
                    "file": file_path,
                    "model": f"openrouter:{model}",
                    "complexity": ctx.complexity.value if ctx else "unknown",
                    "turns_used": turn + 1,
                    "agentic": True,
                },
                agent_used=self.name,
                model_used=f"openrouter:{model}",
                duration_ms=int((time.time() - start) * 1000),
            )

        logger.warning(f"[CodeAgent] {last_error}")
        return None

    def _analyze_code_issue(self, task: Task, start: float) -> TaskResult:
        """
        Analyze a code issue and fix it using an agentic tool-calling loop.

        The model gets tools (read_file, search_files, replace_in_file,
        execute_command, attempt_completion) and iterates up to 6 turns.
        replace_in_file automatically validates with AST+ruff and rolls back
        on failure, giving the model the error so it can retry.

        Falls back to legacy single-shot JSON mode if OllamaClient/ToolExecutor
        are not available.
        """
        file_path = task.data.get("file")
        error = task.data.get("error", "")
        mode = task.data.get("mode", "")
        line_hint = task.data.get("line")
        force_escalation = bool(task.data.get("force_escalation", False))
        error_type = task.data.get("error_type", "")

        # ------------------------------------------------------------------
        # Resolve file path from traceback if not provided explicitly
        # ------------------------------------------------------------------
        if not file_path:
            tb_text = task.data.get("traceback", "") or error
            matches = re.findall(r'File ["\']([^"\']+\.py)["\'], line (\d+)', tb_text)
            if matches:
                file_path, tb_line = matches[-1]
                if not line_hint:
                    try:
                        line_hint = int(tb_line)
                    except Exception:
                        line_hint = None

        if not file_path or not Path(file_path).exists():
            return TaskResult(
                success=False,
                message=f"File not found: {file_path}",
                agent_used=self.name,
            )

        # ------------------------------------------------------------------
        # Build rich context via the context router
        # ------------------------------------------------------------------
        try:
            ctx = self._context_router.build(
                file_path=file_path,
                traceback_str=error,
                error_type=error_type or self._infer_error_type(error),
                line_hint=int(line_hint) if line_hint else None,
                force_escalation=force_escalation,
            )
        except Exception as e:
            logger.warning(
                f"[CodeAgent] Context router failed ({e}), using basic context"
            )
            ctx = None

        # ------------------------------------------------------------------
        # Try agentic tool-calling loop (preferred path)
        # ------------------------------------------------------------------
        if mode == "auto_fix":
            api_only = self._repair_api_only_allowed()
            client = None if api_only else self._get_ollama_client()
            executor = self._get_tool_executor()
            tools = self._get_repair_tool_schemas()

            if (client or api_only) and executor and tools:
                return self._agentic_repair(
                    task,
                    start,
                    file_path,
                    error,
                    error_type,
                    ctx,
                    force_escalation,
                    client,
                    executor,
                    tools,
                )
            else:
                logger.warning(
                    "[CodeAgent] Agentic tools unavailable, falling back to legacy JSON mode"
                )

        # ------------------------------------------------------------------
        # Fallback: legacy single-shot JSON mode
        # ------------------------------------------------------------------
        return self._legacy_json_repair(
            task,
            start,
            file_path,
            error,
            error_type,
            mode,
            ctx,
            force_escalation,
        )

    # ------------------------------------------------------------------
    # Agentic tool-calling repair loop
    # ------------------------------------------------------------------

    def _agentic_repair(
        self,
        task: Task,
        start: float,
        file_path: str,
        error: str,
        error_type: str,
        ctx: Optional["RepairContext"],
        force_escalation: bool,
        client,
        executor,
        tools: List[Dict],
    ) -> TaskResult:
        """Multi-step agentic repair loop using OllamaClient + ToolExecutor."""
        from autotrade.core.local_coding_agent import (
            should_use_native_tools,
            AgentMode,
            ToolExecutor,
        )

        cascade = self._build_repair_cascade(ctx, force_escalation)

        # Determine the project root for the ToolExecutor.
        # If the file is inside PROJECT_DIR, use PROJECT_DIR.
        # Otherwise (e.g. temp directory in tests), use the file's parent.
        abs_file = Path(file_path).resolve()
        if str(abs_file).startswith(str(PROJECT_DIR)):
            repair_root = PROJECT_DIR
        else:
            repair_root = abs_file.parent

        # Compute relative path from repair_root for the model prompt
        try:
            rel_file = str(abs_file.relative_to(repair_root)).replace("\\", "/")
        except ValueError:
            rel_file = abs_file.name

        # Build the initial user message with context
        focus_section = ""
        if ctx and ctx.prompt_context:
            focus_section = ctx.prompt_context

        system_msg = (
            "You are a code repair agent for the AutoTrade trading system. "
            "Fix the error using the provided tools. Steps:\n"
            "1. Read the broken file to see current content. ALWAYS read at least 200-300 lines "
            "at once to maintain structural context. Avoid tiny 10-line reads.\n"
            "2. Search for symbol usages if needed (search_files)\n"
            "3. Apply the fix with replace_in_file (it auto-validates with AST+ruff)\n"
            "4. If replace_in_file fails, read the error and try a corrected patch\n"
            "5. Optionally run ruff check or py_compile via execute_command\n"
            "6. Call attempt_completion when the fix is verified\n\n"
            "Rules:\n"
            "- Make minimal, safe changes\n"
            "- Do NOT use write_to_file (only replace_in_file)\n"
            "- execute_command is restricted to: ruff, py_compile, pytest\n"
            "- If you cannot fix the error, call attempt_completion with an explanation\n"
            f"- Use this relative path for all file operations: {rel_file}"
        )

        user_msg = (
            f"Fix this error:\n\n"
            f"ERROR TYPE: {error_type or 'Unknown'}\n"
            f"ERROR: {error}\n"
            f"FILE: {rel_file}\n\n"
        )
        if focus_section:
            user_msg += f"{focus_section}\n"

        last_error = "No models produced a fix"
        _models_tried = []  # track models we've loaded for cleanup

        for model_name in cascade:
            is_heavy = model_name in self._HEAVY_MODELS
            is_openai = model_name.startswith("openai:")

            # OpenAI models: skip RAM guard (they run on remote APIs)
            if is_openai:
                logger.info(f"[CodeAgent] Using OpenAI fallback: {model_name}")
                openai_result = self._openai_repair(
                    file_path=file_path,
                    error=error,
                    error_type=error_type,
                    ctx=ctx,
                    start=start,
                    models=[model_name.replace("openai:", "")],
                )
                if openai_result and openai_result.success:
                    return openai_result
                continue

            # OpenRouter models: skip RAM guard and route to OpenRouter API
            if model_name.startswith("openrouter:"):
                real_model = model_name.replace("openrouter:", "")
                logger.info(f"[CodeAgent] Using OpenRouter: {real_model}")
                or_client = get_openrouter_client()
                if (
                    or_client.available
                    and or_client.supports_tools(real_model)
                    and real_model
                    in {
                        "anthropic/claude-3.5-sonnet",
                        "anthropic/claude-sonnet-4-6",
                    }
                ):
                    or_result = self._openrouter_agentic_repair(
                        start=start,
                        file_path=file_path,
                        error=error,
                        error_type=error_type,
                        ctx=ctx,
                        model=real_model,
                        system_msg=system_msg,
                        user_msg=user_msg,
                        repair_root=repair_root,
                        tools=tools,
                    )
                    if not or_result or not or_result.success:
                        or_result = self._openrouter_repair(
                            file_path=file_path,
                            error=error,
                            error_type=error_type,
                            model=real_model,
                            ctx=ctx,
                            start=start,
                        )
                else:
                    or_result = self._openrouter_repair(
                        file_path=file_path,
                        error=error,
                        error_type=error_type,
                        model=real_model,
                        ctx=ctx,
                        start=start,
                    )
                if or_result and or_result.success:
                    return or_result
                continue

            # Skip RAM check for the primary model (qwen3-coder-next stays loaded)
            if not is_heavy and not self._context_router.ram_ok_for_model(model_name):
                logger.warning(
                    f"[CodeAgent] Skipping {model_name} â€” insufficient free RAM"
                )
                continue

            _models_tried.append(model_name)

            logger.info(
                f"[CodeAgent] Agentic repair with {model_name} "
                f"(complexity={ctx.complexity.value if ctx else 'unknown'})"
            )

            use_native = should_use_native_tools(model_name)

            # Create a fresh ToolExecutor rooted at repair_root for each attempt
            executor = ToolExecutor(repair_root)

            # Pre-mark the broken file as readable (we already have context)
            executor._files_read.add(str(abs_file))

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]

            fix_applied = False
            completion_result = None
            ollama_error_retries = 0
            max_ollama_retries = 2 if is_heavy else 0  # retry heavy models on 500

            # Loop detection state
            last_tool_call_hash = None
            consecutive_identical_calls = 0
            read_file_done = False

            for turn in range(self._MAX_REPAIR_TURNS):
                logger.info(f"[CodeAgent] Turn {turn + 1}/{self._MAX_REPAIR_TURNS}")

                if use_native:
                    tool_calls, content, meta = client.chat_with_tools(
                        model=model_name,
                        messages=messages,
                        tools=tools,
                        num_ctx=65536 if is_heavy else 32768,
                        temperature=0.2,
                        num_predict=4096,
                        timeout=300 if is_heavy else 180,
                    )
                else:
                    # XML fallback: use regular chat and parse XML tool calls
                    result = self._xml_tool_call(client, model_name, messages, tools)
                    tool_calls = result.get("tool_calls")
                    content = result.get("content", "")
                    meta = result.get("meta", {})

                # Handle error responses from Ollama â€” retry for heavy models
                if content and content.startswith("[OLLAMA"):
                    if ollama_error_retries < max_ollama_retries:
                        ollama_error_retries += 1
                        wait_secs = (
                            (30 * ollama_error_retries)
                            if is_heavy
                            else (10 * ollama_error_retries)
                        )
                        logger.warning(
                            f"[CodeAgent] Ollama error for {model_name}, "
                            f"retry {ollama_error_retries}/{max_ollama_retries} "
                            f"after {wait_secs}s wait: {content[:120]}"
                        )
                        time.sleep(wait_secs)
                        continue  # retry same turn
                    last_error = f"{model_name}: {content}"
                    logger.warning(f"[CodeAgent] {last_error}")
                    break

                if not tool_calls:
                    # No tool calls â€” model might be done or stuck
                    if content:
                        # Append assistant content and nudge
                        messages.append({"role": "assistant", "content": content})
                        messages.append(
                            {
                                "role": "user",
                                "content": "Use the tools to fix the error. Call read_file, then replace_in_file, then attempt_completion.",
                            }
                        )
                        continue
                    # Empty response â€” try next model
                    last_error = (
                        f"{model_name} returned empty response on turn {turn + 1}"
                    )
                    logger.warning(f"[CodeAgent] {last_error}")
                    break

                # Process each tool call
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    tool_name = fn.get("name", "")
                    args = fn.get("arguments", {})

                    # Ensure args is a dict (some models return strings)
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}

                    # Loop Detection: Hash the tool call
                    current_call_hash = hash(
                        f"{tool_name}:{json.dumps(args, sort_keys=True)}"
                    )
                    if current_call_hash == last_tool_call_hash:
                        consecutive_identical_calls += 1
                    else:
                        consecutive_identical_calls = 0
                    last_tool_call_hash = current_call_hash

                    if consecutive_identical_calls >= 3:
                        tool_result = {
                            "success": False,
                            "error": f"LOOP DETECTED: You have called {tool_name} with these identical arguments {consecutive_identical_calls + 1} times in a row. "
                            "This is not working. You MUST call `read_file` to verify the current state of the file and try a DIFFERENT approach (e.g., a different patch or tool).",
                        }
                        logger.warning(
                            f"[CodeAgent] Loop detected for {tool_name}, forcing change"
                        )
                    else:
                        # Normalize path args: if model sends absolute path or
                        # the original file_path, rewrite to relative
                        if "path" in args:
                            p = args["path"].replace("\\", "/")
                            # If it's the absolute path, make relative
                            try:
                                p_abs = Path(p).resolve()
                                p = str(p_abs.relative_to(repair_root)).replace(
                                    "\\", "/"
                                )
                            except (ValueError, OSError):
                                pass
                            args["path"] = p

                        # Build descriptive log for the tool call
                        arg_summary = []
                        for k, v in args.items():
                            if k == "path":
                                arg_summary.append(f"path='{v}'")
                            elif k in ("start_line", "end_line"):
                                arg_summary.append(f"{k}={v}")
                            elif k in ("old_text", "new_text", "content"):
                                # Snippet of the code being changed
                                snippet = str(v).strip().replace("\n", "\\n")
                                if len(snippet) > 60:
                                    snippet = snippet[:57] + "..."
                                arg_summary.append(f"{k}='{snippet}'")
                            elif k == "command":
                                arg_summary.append(f"cmd='{v}'")
                            else:
                                arg_summary.append(f"{k}={v}")

                        logger.info(
                            f"[CodeAgent]   Tool: {tool_name}({', '.join(arg_summary)})"
                        )

                        # Safety: restrict execute_command in repair mode
                        if tool_name == "execute_command":
                            cmd = args.get("command", "")
                            if not self._is_command_allowed(cmd):
                                tool_result = {
                                    "success": False,
                                    "error": f"Command not allowed in repair mode. Only ruff, py_compile, pytest are permitted.",
                                }
                            else:
                                tool_result = executor.execute(
                                    tool_name, args, AgentMode.ACT
                                )
                        elif tool_name == "attempt_completion":
                            completion_result = args.get("result", "Fix completed")
                            fix_applied = executor._edits_made > 0
                            tool_result = {
                                "success": True,
                                "completed": True,
                                "result": completion_result,
                            }
                        elif tool_name in self._REPAIR_TOOL_NAMES:
                            # ENFORCEMENT: If agent tries to read a tiny range, expand it to 300 lines
                            if tool_name == "read_file":
                                if read_file_done:
                                    tool_result = {
                                        "success": False,
                                        "error": "READ ALREADY PROVIDED. Do not call read_file again. Proceed to replace_in_file.",
                                    }
                                else:
                                    if "start_line" in args and "end_line" in args:
                                        try:
                                            start = int(args["start_line"])
                                            end = int(args["end_line"])
                                            if (end - start) < 200:
                                                center = (start + end) // 2
                                                args["start_line"] = max(
                                                    1, center - 150
                                                )
                                                args["end_line"] = center + 150
                                                logger.info(
                                                    f"[CodeAgent] Range too small ({end - start} lines). Expanding to 300-line context."
                                                )
                                        except (ValueError, TypeError):
                                            pass  # Fallback to original args if cast fails
                                    tool_result = executor.execute(
                                        tool_name, args, AgentMode.ACT
                                    )
                                    if tool_result.get("success"):
                                        read_file_done = True
                            else:
                                tool_result = executor.execute(
                                    tool_name, args, AgentMode.ACT
                                )
                        else:
                            tool_result = {
                                "success": False,
                                "error": f"Tool '{tool_name}' not available in repair mode",
                            }

                    # Track if replace_in_file succeeded
                    if tool_name == "replace_in_file" and tool_result.get("success"):
                        fix_applied = True

                    # Append assistant + tool result to message history
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content or "",
                            "tool_calls": tool_calls,
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "content": json.dumps(tool_result, default=str)[:4000],
                        }
                    )

                # Check if completion was signaled
                if completion_result is not None:
                    break

            # Evaluate result for this model
            if fix_applied:
                summary = completion_result or "Fix applied via agentic loop"
                logger.info(f"[CodeAgent] SUCCESS with {model_name}: {summary}")
                # Return JSON with summary and empty changes list (already applied)
                return TaskResult(
                    success=True,
                    message=json.dumps({"summary": summary, "changes": []}),
                    data={
                        "file": file_path,
                        "model": model_name,
                        "complexity": ctx.complexity.value if ctx else "unknown",
                        "turns_used": turn + 1,
                        "agentic": True,
                        "changes_applied": 1,
                        "persisted_to_disk": True,
                    },
                    agent_used=self.name,
                    model_used=model_name,
                    duration_ms=int((time.time() - start) * 1000),
                )
            else:
                last_error = f"{model_name} did not apply any changes"
                logger.warning(f"[CodeAgent] {last_error} â€” trying next model")

        # All cascade models exhausted â€” optionally try enabled final fallbacks.
        cascade_tried = list(cascade)
        openai_already_tried = any(
            str(model_name).startswith("openai:") for model_name in cascade_tried
        )
        if (
            not openai_already_tried
            and self._repair_flag_enabled(
                "AUTOTRADE_ENABLE_OPENAI_REPAIR", self._repair_openai_enabled
            )
        ):
            logger.warning(
                f"[CodeAgent] All local models failed â€” attempting OpenAI fallback"
            )
            openai_result = self._openai_repair(
                file_path=file_path,
                error=error,
                error_type=error_type,
                ctx=ctx,
                start=start,
            )
            cascade_tried.append("openai")
            if openai_result and openai_result.success:
                return openai_result

        if self._repair_flag_enabled(
            "AUTOTRADE_ENABLE_CODEX_REPAIR", self._repair_codex_enabled
        ):
            logger.warning(
                f"[CodeAgent] Local/OpenAI repair failed â€” attempting Codex fallback"
            )
            codex_result = self._codex_repair(
                file_path=file_path,
                error=error,
                error_type=error_type,
                ctx=ctx,
                start=start,
            )
            cascade_tried.append("codex")
            if codex_result and codex_result.success:
                return codex_result

        logger.error(f"[CodeAgent] All repair paths failed. Last: {last_error}")
        return TaskResult(
            success=False,
            message=f"Agentic repair failed after all enabled fallbacks. Last error: {last_error}",
            data={"file": file_path, "cascade_tried": cascade_tried},
            agent_used=self.name,
            model_used="none",
            duration_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # OpenAI API fallback (last resort when all local models fail)
    # ------------------------------------------------------------------

    _OPENAI_CASCADE = ["gpt-5.4-mini", "gpt-5.3-codex"]
    _CODEX_JSON_MAX_CHANGES = 8

    def _openai_repair(
        self,
        file_path: str,
        error: str,
        error_type: str,
        ctx: Optional["RepairContext"],
        start: float,
        models: Optional[List[str]] = None,
    ) -> Optional[TaskResult]:
        """
        Last-resort repair via OpenAI API (JSON mode, no tool calling).

        Tries cost-controlled coding models by default.
        Uses the legacy JSON search/replace format since OpenAI models
        are fast enough to not need multi-turn tool loops.
        """
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("[CodeAgent] No OPENAI_API_KEY â€” skipping OpenAI fallback")
            return None

        # Daily limit check â€” prevent burning API quota on unfixable errors
        today = datetime.now().strftime("%Y-%m-%d")
        if CodeAgent._openai_date != today:
            CodeAgent._openai_calls_today = 0
            CodeAgent._openai_date = today
        if CodeAgent._openai_calls_today >= CodeAgent._openai_daily_limit:
            logger.warning(
                f"[CodeAgent] OpenAI daily limit reached ({CodeAgent._openai_daily_limit}), skipping"
            )
            return None
        CodeAgent._openai_calls_today += 1

        abs_file = Path(file_path).resolve()
        try:
            content = abs_file.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"[CodeAgent] Cannot read {file_path}: {e}")
            return None

        # Build a focused excerpt (Â±60 lines around error) instead of dumping
        # the entire file â€” the old approach produced 186K tokens and got rejected.
        excerpt = ""
        line_num = None
        # Try to extract line number from error string
        line_match = re.search(r"line\s+(\d+)", error, re.IGNORECASE)
        if line_match:
            line_num = int(line_match.group(1))
        elif ctx and ctx.line_hint:
            line_num = ctx.line_hint

        all_lines = content.splitlines()
        exact_block = ""
        if line_num:
            start = max(0, line_num - 60)
            end = min(len(all_lines), line_num + 60)
            numbered = [f"{i + 1:4d} | {all_lines[i]}" for i in range(start, end)]
            excerpt = (
                f"# Lines {start + 1}-{end} of {file_path} ({len(all_lines)} total)\n"
                + "\n".join(numbered)
            )
            block_start = max(0, line_num - 8)
            block_end = min(len(all_lines), line_num + 8)
            exact_block = "\n".join(all_lines[block_start:block_end])
        elif ctx and ctx.prompt_context:
            # Fallback to the (now slim) prompt_context
            excerpt = ctx.prompt_context[:6000]
            exact_block = excerpt.splitlines()[:20]
            exact_block = "\n".join(exact_block)
        else:
            # Last resort: first 120 lines
            numbered = [
                f"{i + 1:4d} | {all_lines[i]}" for i in range(min(120, len(all_lines)))
            ]
            excerpt = (
                f"# First 120 lines of {file_path} ({len(all_lines)} total)\n"
                + "\n".join(numbered)
            )
            exact_block = "\n".join(all_lines[:20])

        prompt = f"""Fix this Python error. Return ONLY a JSON object, no markdown fences.

ERROR TYPE: {error_type or "Unknown"}
ERROR: {error}

{excerpt}

EXACT_REPLACE_BLOCK (copy/paste for your search string):
<<<
{exact_block}
>>>

Return this exact JSON schema:
{{"summary": "one sentence", "changes": [{{"file": "{file_path}", "search": "<exact text from file>", "replace": "<fixed text>", "reason": "why", "line_start": <optional 1-based start line>, "line_end": <optional 1-based end line>}}]}}

Rules:
- search must be an EXACT substring that appears ONCE in the file
- If exact search is risky, use line_start/line_end
- Fix ALL errors, not just the first one â€” the file will be validated with ruff lint
- Minimal change only"""

        lint_feedback = ""  # Accumulates lint errors from prior attempts

        for model in models or self._OPENAI_CASCADE:
            logger.info(f"[CodeAgent] OpenAI fallback: trying {model}")
            try:
                current_prompt = prompt
                if lint_feedback:
                    current_prompt += (
                        f"\n\nPREVIOUS FIX ATTEMPT WAS REJECTED by ruff lint. "
                        f"You MUST fix ALL of these errors in addition to the original error:\n{lint_feedback}"
                    )

                is_o_series = model.startswith("o3-") or model.startswith("o1-")
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a Python bug-fix expert. Output ONLY raw JSON.",
                        },
                        {"role": "user", "content": current_prompt},
                    ],
                }
                if not is_o_series:
                    payload["temperature"] = 0.1
                if is_o_series:
                    payload["max_completion_tokens"] = 2000
                else:
                    payload["max_tokens"] = 2000
                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=30,
                )

                if resp.status_code != 200:
                    logger.warning(
                        f"[CodeAgent] OpenAI {model} returned {resp.status_code}: {resp.text[:200]}"
                    )
                    continue

                data = resp.json()
                reply = data["choices"][0]["message"]["content"].strip()

                # Strip markdown fences
                if "```json" in reply:
                    reply = reply.split("```json", 1)[1].split("```", 1)[0].strip()
                elif "```" in reply:
                    reply = reply.split("```", 1)[1].split("```", 1)[0].strip()

                fix_data = json.loads(reply)
                changes = fix_data.get("changes", [])
                if not changes:
                    logger.warning(f"[CodeAgent] OpenAI {model} returned empty changes")
                    continue

                # Apply all changes to in-memory buffer first, then validate atomically
                original_content = content  # Preserve for rollback
                applied = 0
                for change in changes:
                    search = change.get("search", "")
                    replace = change.get("replace", "")
                    line_start = change.get("line_start")
                    line_end = change.get("line_end")
                    if not search:
                        if not (line_start and line_end):
                            logger.warning(
                                f"[CodeAgent] OpenAI missing search and line range, skipping"
                            )
                            continue
                    if search:
                        if search not in content:
                            # Fuzzy fallback: normalize trailing whitespace per line
                            norm_search = "\n".join(
                                line.rstrip() for line in search.splitlines()
                            )
                            norm_content = "\n".join(
                                line.rstrip() for line in content.splitlines()
                            )
                            if norm_search in norm_content:
                                search = norm_search
                                content = norm_content
                                logger.info(
                                    "[CodeAgent] OpenAI search matched after whitespace normalization"
                                )
                            else:
                                logger.warning(
                                    f"[CodeAgent] OpenAI search text not found (even after normalization), skipping"
                                )
                                continue
                        if content.count(search) > 1:
                            logger.warning(
                                f"[CodeAgent] OpenAI search text not unique, skipping"
                            )
                            continue

                    if search:
                        content = content.replace(search, replace, 1)
                        applied += 1
                        logger.info(
                            f"[CodeAgent] OpenAI {model} change applied: {change.get('reason', '?')}"
                        )
                    else:
                        try:
                            start_idx = int(line_start) - 1
                            end_idx = int(line_end)
                        except (TypeError, ValueError):
                            logger.warning(
                                f"[CodeAgent] OpenAI invalid line range, skipping"
                            )
                            continue
                        lines = content.splitlines(keepends=True)
                        if (
                            start_idx < 0
                            or end_idx <= start_idx
                            or end_idx > len(lines)
                        ):
                            logger.warning(
                                f"[CodeAgent] OpenAI line range out of bounds, skipping"
                            )
                            continue
                        replacement = (
                            replace
                            if replace.endswith("\n") or replace == ""
                            else replace + "\n"
                        )
                        lines = lines[:start_idx] + [replacement] + lines[end_idx:]
                        content = "".join(lines)
                        applied += 1
                        logger.info(
                            f"[CodeAgent] OpenAI {model} change applied via line range: {change.get('reason', '?')}"
                        )

                if applied > 0:
                    # Validate complete buffer with AST + ruff lint before writing
                    lint_err = _lint_python_content(content, str(abs_file))
                    if lint_err:
                        logger.error(
                            f"[CodeAgent] OpenAI fix rejected by lint: {lint_err}"
                        )
                        lint_feedback = lint_err  # Feed errors to next model
                        content = original_content  # Rollback in-memory buffer
                        continue

                    # Lint passed â€” write to disk with backup
                    backup = abs_file.with_suffix(".py.bak")
                    backup.write_text(original_content, encoding="utf-8")
                    abs_file.write_text(content, encoding="utf-8")

                    cost_est = "$0.01" if model == "gpt-4o" else "$0.05"
                    summary = fix_data.get("summary", "Fixed via OpenAI")
                    modified_files = sorted(
                        {
                            str(Path(change.get("file") or file_path).resolve())
                            for change in changes
                        }
                    )
                    logger.info(
                        f"[CodeAgent] SUCCESS via OpenAI {model} (~{cost_est}): {summary}"
                    )
                    return TaskResult(
                        success=True,
                        message=json.dumps(fix_data),
                        data={
                            "file": file_path,
                            "model": f"openai:{model}",
                            "complexity": ctx.complexity.value if ctx else "unknown",
                            "agentic": True,
                            "openai_fallback": True,
                            "changes_applied": applied,
                            "fix_details": changes,
                            "modified_files": modified_files,
                            "persisted_to_disk": True,
                        },
                        agent_used=self.name,
                        model_used=f"openai:{model}",
                        duration_ms=int((time.time() - start) * 1000),
                    )

            except json.JSONDecodeError as e:
                logger.warning(f"[CodeAgent] OpenAI {model} returned invalid JSON: {e}")
            except Exception as e:
                logger.warning(f"[CodeAgent] OpenAI {model} error: {e}")

        return None

    def _codex_repair(
        self,
        file_path: str,
        error: str,
        error_type: str,
        ctx: Optional["RepairContext"],
        start: float,
    ) -> Optional[TaskResult]:
        """Final repair fallback via Codex CLI using the same JSON patch contract."""
        from autotrade.utils.codex_cli import codex_available, run_codex

        if not codex_available(self.codex_command):
            logger.warning(
                "[CodeAgent] Codex CLI unavailable â€” skipping Codex fallback"
            )
            return None

        abs_file = Path(file_path).resolve()
        try:
            content = abs_file.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"[CodeAgent] Cannot read {file_path}: {e}")
            return None

        excerpt = ""
        line_num = None
        line_match = re.search(r"line\s+(\d+)", error, re.IGNORECASE)
        if line_match:
            line_num = int(line_match.group(1))
        elif ctx and ctx.line_hint:
            line_num = ctx.line_hint

        all_lines = content.splitlines()
        if line_num:
            block_start = max(0, line_num - 60)
            block_end = min(len(all_lines), line_num + 60)
            numbered = [
                f"{i + 1:4d} | {all_lines[i]}" for i in range(block_start, block_end)
            ]
            excerpt = "\n".join(numbered)
        elif ctx and ctx.prompt_context:
            excerpt = ctx.prompt_context[:8000]
        else:
            excerpt = "\n".join(all_lines[:120])

        prompt = f"""Fix this Python error in AutoTrade and return ONLY raw JSON.

ERROR TYPE: {error_type or "Unknown"}
ERROR: {error}
FILE: {file_path}

FILE EXCERPT:
{excerpt}

Return this exact JSON schema:
{{"summary": "one sentence", "changes": [{{"file": "{file_path}", "search": "<exact text>", "replace": "<fixed text>", "reason": "why"}}]}}

Rules:
- Minimal safe changes only
- search must appear exactly once
- No markdown fences
- No prose outside the JSON object
- At most {self._CODEX_JSON_MAX_CHANGES} changes"""

        success, stdout, stderr = run_codex(
            prompt=prompt,
            command=self.codex_command,
            timeout=self.codex_timeout,
            use_stdin=self.codex_use_stdin,
            extra_args=self.codex_extra_args,
        )
        if not success or not stdout:
            logger.warning(
                f"[CodeAgent] Codex fallback failed: {(stderr or stdout or '').strip()[:200]}"
            )
            return None

        reply = stdout.strip()
        if "```json" in reply:
            reply = reply.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in reply:
            reply = reply.split("```", 1)[1].split("```", 1)[0].strip()

        try:
            fix_data = json.loads(reply)
        except json.JSONDecodeError as e:
            logger.warning(f"[CodeAgent] Codex returned invalid JSON: {e}")
            return None

        changes = fix_data.get("changes", [])
        if not changes:
            logger.warning("[CodeAgent] Codex returned no changes")
            return None

        applied = 0
        for change in changes[: self._CODEX_JSON_MAX_CHANGES]:
            search = change.get("search", "")
            replace = change.get("replace", "")
            if not search or search not in content or content.count(search) != 1:
                logger.warning("[CodeAgent] Codex search text missing or non-unique")
                continue
            content = content.replace(search, replace, 1)
            applied += 1

        if applied == 0:
            return None

        backup = abs_file.with_suffix(".py.bak")
        try:
            backup.write_text(abs_file.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass
        abs_file.write_text(content, encoding="utf-8")

        lint_err = _lint_python_content(content, str(abs_file))
        if lint_err:
            logger.error(f"[CodeAgent] Codex fix rejected by lint: {lint_err}")
            if backup.exists():
                abs_file.write_text(
                    backup.read_text(encoding="utf-8"), encoding="utf-8"
                )
            return None

        summary = fix_data.get("summary", "Fixed via Codex")
        logger.info(f"[CodeAgent] SUCCESS via Codex: {summary}")
        # Return JSON with summary and empty changes list (already applied)
        return TaskResult(
            success=True,
            message=json.dumps({"summary": summary, "changes": []}),
            data={
                "file": file_path,
                "model": "codex",
                "complexity": ctx.complexity.value if ctx else "unknown",
                "agentic": True,
                "codex_fallback": True,
            },
            agent_used=self.name,
            model_used="codex",
            duration_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # XML tool-call fallback for models in FORCE_XML_MODELS
    # ------------------------------------------------------------------

    def _xml_tool_call(
        self, client, model_name: str, messages: List[Dict], tools: List[Dict]
    ) -> Dict:
        """Send chat via OllamaClient.chat() and parse XML tool calls from content."""
        from autotrade.core.local_coding_agent import TOOL_DEFINITIONS

        # Prepend XML tool definitions to system message
        xml_messages = list(messages)
        if xml_messages and xml_messages[0]["role"] == "system":
            xml_messages[0] = {
                "role": "system",
                "content": xml_messages[0]["content"] + "\n\n" + TOOL_DEFINITIONS,
            }

        content, meta = client.chat(
            model=model_name,
            messages=xml_messages,
            num_ctx=32768,
            temperature=0.2,
            num_predict=4096,
            timeout=180,
        )

        # Parse XML tool calls from content
        tool_calls = self._parse_xml_tool_calls(content)

        return {
            "tool_calls": tool_calls if tool_calls else None,
            "content": content,
            "meta": meta,
        }

    @staticmethod
    def _parse_xml_tool_calls(content: str) -> Optional[List[Dict]]:
        """Parse XML-formatted tool calls from model content."""
        if "<tool_call>" not in content:
            return None

        calls = []
        for match in re.finditer(
            r"<tool_call>\s*<tool_name>(\w+)</tool_name>(.*?)</tool_call>",
            content,
            re.DOTALL,
        ):
            tool_name = match.group(1)
            params_xml = match.group(2)

            # Extract params from XML tags
            args = {}
            for param_match in re.finditer(r"<(\w+)>(.*?)</\1>", params_xml, re.DOTALL):
                args[param_match.group(1)] = param_match.group(2)

            calls.append(
                {
                    "function": {
                        "name": tool_name,
                        "arguments": args,
                    }
                }
            )

        return calls if calls else None

    # ------------------------------------------------------------------
    # Legacy single-shot JSON repair (fallback)
    # ------------------------------------------------------------------

    def _legacy_json_repair(
        self,
        task: Task,
        start: float,
        file_path: str,
        error: str,
        error_type: str,
        mode: str,
        ctx: Optional["RepairContext"],
        force_escalation: bool,
    ) -> TaskResult:
        """Fallback: single-shot /api/generate with JSON response parsing."""
        content = (
            ctx.primary_content if ctx else Path(file_path).read_text(encoding="utf-8")
        )
        prompt_context = ctx.prompt_context if ctx else content

        if mode == "auto_fix":
            prompt = f"""You are a Python coding agent for the AutoTrade trading system.
Produce a minimal, safe bug fix as JSON ONLY â€” no markdown, no prose outside the JSON object.

ERROR TYPE : {error_type or "Unknown"}
ERROR      : {error}

{prompt_context}

Return JSON in this EXACT schema (nothing else):
{{
  "summary": "one sentence describing the fix",
  "changes": [
    {{
      "file": "{file_path}",
      "search": "<exact multi-line text to find â€” must appear exactly once in the file>",
      "replace": "<replacement text>",
      "reason": "why this change"
    }}
  ]
}}

Critical rules:
- search must be an EXACT substring of the file (copy-paste from the file above).
- search must appear exactly ONCE in the file.
- Keep changes as small as possible.
- Do NOT include any text before or after the JSON object.
"""
        else:
            prompt = f"""Analyze this Python code issue and suggest a fix.

ERROR: {error}

{prompt_context}

Provide:
1. Root cause (1 sentence)
2. Suggested fix (code snippet)
"""

        cascade = self._build_repair_cascade(ctx, force_escalation)

        system_prompt = (
            "You are an expert Python developer specialising in algorithmic trading systems. "
            "When asked for a JSON fix, output ONLY the raw JSON object with no extra text."
        )

        last_error = "No models produced a valid response"
        for model_name in cascade:
            if not self._context_router.ram_ok_for_model(model_name):
                logger.warning(
                    f"[CodeAgent] Skipping {model_name} â€” insufficient free RAM"
                )
                continue

            logger.info(
                f"[CodeAgent] Legacy mode with {model_name} "
                f"(complexity={ctx.complexity.value if ctx else 'unknown'})"
            )

            saved_model = self.model
            self.model = model_name
            try:
                response = self._call_ollama(prompt, system_prompt)
            finally:
                self.model = saved_model

            if not response:
                last_error = f"{model_name} returned empty response"
                logger.warning(f"[CodeAgent] {last_error} â€” trying next model")
                continue

            if mode == "auto_fix":
                parsed = self._parse_fix_json(response, model_name)
                if parsed is None:
                    last_error = f"{model_name} returned unparseable JSON"
                    logger.warning(f"[CodeAgent] {last_error} â€” trying next model")
                    continue
                if not parsed.get("changes"):
                    last_error = f"{model_name} returned empty changes list"
                    logger.warning(f"[CodeAgent] {last_error} â€” trying next model")
                    continue
                logger.info(
                    f"[CodeAgent] SUCCESS with {model_name}: "
                    f"{parsed.get('summary', 'no summary')}"
                )
                return TaskResult(
                    success=True,
                    message=response,
                    data={
                        "file": file_path,
                        "analysis": response,
                        "model": model_name,
                        "complexity": ctx.complexity.value if ctx else "unknown",
                        "agentic": False,
                    },
                    agent_used=self.name,
                    model_used=model_name,
                    duration_ms=int((time.time() - start) * 1000),
                )

            logger.info(f"[CodeAgent] Analysis complete with {model_name}")
            return TaskResult(
                success=True,
                message=response,
                data={"file": file_path, "analysis": response, "model": model_name},
                agent_used=self.name,
                model_used=model_name,
                duration_ms=int((time.time() - start) * 1000),
            )

        logger.error(
            f"[CodeAgent] All models in cascade failed. Last error: {last_error}"
        )
        return TaskResult(
            success=False,
            message=f"Code repair failed after trying all models. Last error: {last_error}",
            data={"file": file_path, "cascade_tried": cascade},
            agent_used=self.name,
            model_used="none",
            duration_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unload_model(model_name: str):
        """Send keep_alive=0 to Ollama to unload a model from RAM/VRAM."""
        try:
            requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model_name, "keep_alive": 0},
                timeout=10,
            )
            logger.info(f"[CodeAgent] Unloaded {model_name} to free RAM")
        except Exception:
            pass  # best-effort

    @staticmethod
    def _unload_all_models():
        """Query Ollama for loaded models and unload every one of them.

        This is used before loading a heavy model (qwen3-coder-next) to
        ensure maximum available RAM.
        """
        try:
            resp = requests.get("http://localhost:11434/api/ps", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("models", [])
                for m in models:
                    name = m.get("name", "")
                    if name:
                        try:
                            requests.post(
                                "http://localhost:11434/api/generate",
                                json={"model": name, "keep_alive": 0},
                                timeout=10,
                            )
                            logger.info(
                                f"[CodeAgent] Unloaded {name} (pre-heavy-model cleanup)"
                            )
                        except Exception:
                            pass
                if models:
                    # Give Ollama a moment to actually free the memory
                    time.sleep(3)
                    logger.info(
                        f"[CodeAgent] Unloaded {len(models)} model(s), waiting for RAM to free"
                    )
                else:
                    logger.info("[CodeAgent] No models currently loaded in Ollama")
        except Exception as exc:
            logger.warning(
                f"[CodeAgent] Could not query Ollama for loaded models: {exc}"
            )

    @staticmethod
    def _infer_error_type(error_text: str) -> str:
        """Extract the error type from a traceback string."""
        for line in reversed(error_text.splitlines()):
            line = line.strip()
            match = re.match(
                r"^([A-Za-z][A-Za-z0-9_]*Error|[A-Za-z][A-Za-z0-9_]*Exception):", line
            )
            if match:
                return match.group(1)
        return "Unknown"

    @staticmethod
    def _parse_fix_json(response: str, model_name: str):
        """
        Try to parse the JSON fix from a model response.
        Handles responses that may have a markdown code fence wrapper.
        Returns the parsed dict or None on failure.
        """
        import json as _json

        text = response.strip()
        # Strip markdown fences
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]
        text = text.strip()
        try:
            return _json.loads(text)
        except _json.JSONDecodeError as exc:
            logger.debug(f"[CodeAgent] JSON parse error from {model_name}: {exc}")
            return None


class RecoveryAgent(BaseAgent):
    """Agent for system recovery actions."""

    def __init__(self, dry_run: bool = True):
        super().__init__("RecoveryAgent", ModelTier.MEDIUM)
        self.dry_run = dry_run

    def execute(self, task: Task) -> TaskResult:
        start = time.time()

        if task.type == TaskType.RECOVERY:
            return self._attempt_recovery(task, start)
        elif task.type == TaskType.CONFIG_UPDATE:
            return self._update_config(task, start)

        return TaskResult(
            success=False,
            message=f"Unknown task type for RecoveryAgent: {task.type}",
            agent_used=self.name,
        )

    def _attempt_recovery(self, task: Task, start: float) -> TaskResult:
        """Attempt to recover from an issue."""
        issue = task.data.get("issue", "")

        if self.dry_run:
            return TaskResult(
                success=True,
                message=f"[DRY RUN] Would attempt recovery for: {issue}",
                data={"dry_run": True, "issue": issue},
                agent_used=self.name,
                model_used="dry-run",
                duration_ms=int((time.time() - start) * 1000),
            )

        # Real recovery logic here
        actions_taken = []

        if "advisor" in issue.lower():
            # Try to reinitialize advisor
            try:
                from autotrade.advisors.position_advisor import PositionAdvisor

                advisor = PositionAdvisor()
                if advisor._check_llm_available():
                    actions_taken.append("Reinitialized PositionAdvisor")
            except Exception as e:
                actions_taken.append(f"Advisor reinit failed: {e}")

        return TaskResult(
            success=len(actions_taken) > 0,
            message=f"Recovery actions: {', '.join(actions_taken) or 'none'}",
            data={"actions": actions_taken},
            agent_used=self.name,
            model_used="rule-based",
            duration_ms=int((time.time() - start) * 1000),
        )

    def _update_config(self, task: Task, start: float) -> TaskResult:
        """Update configuration."""
        return TaskResult(
            success=False,
            message="Config updates not implemented yet",
            agent_used=self.name,
            duration_ms=int((time.time() - start) * 1000),
        )


class VisionAgent(BaseAgent):
    """Agent for screenshot and UI analysis using VL models."""

    def __init__(self):
        super().__init__("VisionAgent", ModelTier.VISION)

    def execute(self, task: Task) -> TaskResult:
        start = time.time()

        if task.type == TaskType.SCREENSHOT_ANALYSIS:
            return self._analyze_screenshot(task, start)
        elif task.type == TaskType.UI_DEBUG:
            return self._debug_ui(task, start)

        return TaskResult(
            success=False,
            message=f"Unknown task type for VisionAgent: {task.type}",
            agent_used=self.name,
        )

    def _analyze_screenshot(self, task: Task, start: float) -> TaskResult:
        """Analyze a screenshot image."""
        image_path = task.data.get("image_path")
        question = task.data.get("question", "Describe what you see in this image.")

        if not image_path:
            return TaskResult(
                success=False, message="No image_path provided", agent_used=self.name
            )

        # For now, return placeholder - actual vision requires multimodal API
        # TODO: Implement actual vision model call with image
        return TaskResult(
            success=False,
            message="Vision analysis not yet implemented - requires multimodal API integration",
            data={"image_path": image_path, "question": question},
            agent_used=self.name,
            model_used=self._get_model() or "none",
            duration_ms=int((time.time() - start) * 1000),
        )

    def _debug_ui(self, task: Task, start: float) -> TaskResult:
        """Debug UI issues from screenshot + code."""
        image_description = task.data.get("image_description", "")
        html_code = task.data.get("html_code", "")
        css_code = task.data.get("css_code", "")
        issue = task.data.get("issue", "")

        prompt = f"""Analyze this UI debugging issue:

Issue reported: {issue}

Visual description: {image_description}

HTML (excerpt):
```html
{html_code[:2000]}
```

CSS (excerpt):
```css
{css_code[:1000]}
```

Identify the likely cause and suggest a fix.
"""

        response = self._call_ollama(
            prompt,
            "You are a frontend developer debugging UI issues. Be specific about CSS/HTML fixes.",
        )

        return TaskResult(
            success=response is not None,
            message=response or "Could not analyze UI issue",
            data={"analysis": response},
            agent_used=self.name,
            model_used=self._get_model() or "none",
            duration_ms=int((time.time() - start) * 1000),
        )


class ChartAgent(BaseAgent):
    """Agent for trading chart analysis using vision-capable models."""

    def __init__(self):
        super().__init__("ChartAgent", ModelTier.CHART)

    def execute(self, task: Task) -> TaskResult:
        start = time.time()

        if task.type == TaskType.CHART_ANALYSIS:
            return self._analyze_chart(task, start)

        return TaskResult(
            success=False,
            message=f"Unknown task type for ChartAgent: {task.type}",
            agent_used=self.name,
        )

    def _analyze_chart(self, task: Task, start: float) -> TaskResult:
        """Analyze a trading chart image."""
        image_path = task.data.get("image_path")
        if not image_path or not Path(image_path).exists():
            return TaskResult(
                success=False,
                message=f"Chart image not found: {image_path}",
                agent_used=self.name,
            )

        symbol = task.data.get("symbol", "?")
        question = task.data.get(
            "question",
            f"Analyze this trading chart for {symbol}. What is the current trend, support/resistance levels, and recommended entry strategy?",
        )

        system = "You are a professional technical analyst specializing in trading chart patterns. Identify trends, support/resistance, and volume profiles."

        response = self._call_ollama_vision(question, image_path, system)

        return TaskResult(
            success=response is not None,
            message=response or "Chart analysis failed",
            data={"symbol": symbol, "image_path": image_path, "analysis": response},
            agent_used=self.name,
            model_used=self._get_model() or "none",
            duration_ms=int((time.time() - start) * 1000),
        )


class MathAgent(BaseAgent):
    """Agent for math and logic problems using phi4."""

    def __init__(self):
        super().__init__("MathAgent", ModelTier.MATH)

    def execute(self, task: Task) -> TaskResult:
        start = time.time()

        if task.type == TaskType.MATH_PROBLEM:
            return self._solve_math(task, start)
        elif task.type == TaskType.LOGIC_VERIFY:
            return self._verify_logic(task, start)

        return TaskResult(
            success=False,
            message=f"Unknown task type for MathAgent: {task.type}",
            agent_used=self.name,
        )

    def _solve_math(self, task: Task, start: float) -> TaskResult:
        """Solve a math problem step by step."""
        problem = task.data.get("problem", "")

        prompt = f"""Solve this problem step by step:

{problem}

Show your reasoning at each step, then provide the final answer.
"""

        response = self._call_ollama(
            prompt,
            "You are a mathematics expert. Solve problems step by step with clear reasoning.",
        )

        return TaskResult(
            success=response is not None,
            message=response or "Could not solve problem",
            data={"problem": problem, "solution": response},
            agent_used=self.name,
            model_used=self._get_model() or "none",
            duration_ms=int((time.time() - start) * 1000),
        )

    def _verify_logic(self, task: Task, start: float) -> TaskResult:
        """Verify logical reasoning or a proposed solution."""
        claim = task.data.get("claim", "")
        reasoning = task.data.get("reasoning", "")

        prompt = f"""Verify this logical claim and reasoning:

Claim: {claim}

Reasoning provided:
{reasoning}

Is the reasoning valid? Are there any logical errors or missing steps?
"""

        response = self._call_ollama(
            prompt,
            "You are a logic expert. Carefully verify reasoning and identify any flaws.",
        )

        return TaskResult(
            success=response is not None,
            message=response or "Could not verify logic",
            data={"claim": claim, "verification": response},
            agent_used=self.name,
            model_used=self._get_model() or "none",
            duration_ms=int((time.time() - start) * 1000),
        )


class PlannerAgent(BaseAgent):
    """Agent for high-level planning and tool orchestration."""

    def __init__(self):
        super().__init__("PlannerAgent", ModelTier.PLANNER)

    def execute(self, task: Task) -> TaskResult:
        start = time.time()

        if task.type == TaskType.TASK_PLANNING:
            return self._plan_task(task, start)
        elif task.type == TaskType.TOOL_DECISION:
            return self._decide_tool(task, start)

        return TaskResult(
            success=False,
            message=f"Unknown task type for PlannerAgent: {task.type}",
            agent_used=self.name,
        )

    def _plan_task(self, task: Task, start: float) -> TaskResult:
        """Create a plan for a complex task."""
        goal = task.data.get("goal", "")
        context = task.data.get("context", "")
        available_tools = task.data.get("available_tools", [])

        prompt = f"""Create a step-by-step plan to accomplish this goal:

Goal: {goal}

Context: {context}

Available tools/agents:
{chr(10).join(f"- {t}" for t in available_tools)}

Output a numbered plan with specific actions. For each step, indicate which tool to use.
"""

        response = self._call_ollama(
            prompt,
            "You are an expert planner and task orchestrator. Create clear, actionable plans.",
        )

        return TaskResult(
            success=response is not None,
            message=response or "Could not create plan",
            data={"goal": goal, "plan": response},
            agent_used=self.name,
            model_used=self._get_model() or "none",
            duration_ms=int((time.time() - start) * 1000),
        )

    def _decide_tool(self, task: Task, start: float) -> TaskResult:
        """Decide which tool/agent to use for a task."""
        task_description = task.data.get("task_description", "")
        available_tools = task.data.get("available_tools", {})

        tools_desc = "\n".join(
            f"- {name}: {desc}" for name, desc in available_tools.items()
        )

        prompt = f"""Given this task, which tool should be used?

Task: {task_description}

Available tools:
{tools_desc}

Respond with just the tool name and a brief reason.
"""

        response = self._call_ollama(
            prompt,
            "You are a tool selection expert. Choose the best tool for each task.",
        )

        return TaskResult(
            success=response is not None,
            message=response or "Could not decide tool",
            data={"task": task_description, "decision": response},
            agent_used=self.name,
            model_used=self._get_model() or "none",
            duration_ms=int((time.time() - start) * 1000),
        )


class MassiveContextAgent(BaseAgent):
    """Agent for tasks requiring massive context (logs, repos) using Nemotron."""

    def __init__(self):
        super().__init__("MassiveContextAgent", ModelTier.MASSIVE)

    def execute(self, task: Task) -> TaskResult:
        start = time.time()

        if task.type == TaskType.REPO_ANALYSIS:
            return self._analyze_repo(task, start)
        elif task.type == TaskType.MASSIVE_LOG_ANALYSIS:
            return self._analyze_massive_logs(task, start)

        return TaskResult(
            success=False,
            message=f"Unknown task type for MassiveContextAgent: {task.type}",
            agent_used=self.name,
        )

    def _analyze_repo(self, task: Task, start: float) -> TaskResult:
        """Analyze an entire repository or large codebase."""
        repo_path = task.data.get("repo_path", ".")
        question = task.data.get("question", "Summarize this codebase.")
        max_files = task.data.get("max_files", 50)

        # Gather code from repo
        code_content = []
        for ext in ["*.py", "*.js", "*.ts", "*.html", "*.css"]:
            for file_path in Path(repo_path).rglob(ext):
                if "__pycache__" in str(file_path) or "node_modules" in str(file_path):
                    continue
                try:
                    with open(file_path) as f:
                        content = f.read()
                    code_content.append(f"=== {file_path} ===\n{content[:5000]}")
                    if len(code_content) >= max_files:
                        break
                except:
                    pass
            if len(code_content) >= max_files:
                break

        full_context = "\n\n".join(code_content)

        prompt = f"""Analyze this codebase and answer the question:

Question: {question}

Codebase ({len(code_content)} files):
{full_context[:100000]}  # Limit to ~100K chars for safety

Provide a comprehensive answer.
"""

        response = self._call_ollama(
            prompt,
            "You are a senior software architect analyzing a codebase. Be thorough and specific.",
        )

        return TaskResult(
            success=response is not None,
            message=response or "Could not analyze repo",
            data={"files_analyzed": len(code_content), "analysis": response},
            agent_used=self.name,
            model_used=self._get_model() or "none",
            duration_ms=int((time.time() - start) * 1000),
        )

    def _analyze_massive_logs(self, task: Task, start: float) -> TaskResult:
        """Analyze very large log files."""
        log_path = task.data.get("log_path")
        question = task.data.get("question", "What errors or issues do you see?")

        if not log_path or not Path(log_path).exists():
            return TaskResult(
                success=False,
                message=f"Log file not found: {log_path}",
                agent_used=self.name,
            )

        with open(log_path) as f:
            log_content = f.read()

        prompt = f"""Analyze these logs and answer the question:

Question: {question}

Logs ({len(log_content)} characters):
{log_content[:200000]}  # Nemotron can handle ~1M tokens

Provide a detailed analysis of any issues, patterns, or anomalies.
"""

        response = self._call_ollama(
            prompt,
            "You are a log analysis expert. Find patterns, errors, and anomalies in logs.",
        )

        return TaskResult(
            success=response is not None,
            message=response or "Could not analyze logs",
            data={
                "log_path": log_path,
                "log_size": len(log_content),
                "analysis": response,
            },
            agent_used=self.name,
            model_used=self._get_model() or "none",
            duration_ms=int((time.time() - start) * 1000),
        )


class SearchAgent(BaseAgent):
    """Agent for web search using local SearXNG instance."""

    def __init__(self, searxng_host: Optional[str] = None):
        super().__init__("SearchAgent", ModelTier.SEARCH)
        self.searxng_host = searxng_host
        self._client = None

    def _get_client(self):
        """Lazy-load the SearXNG client."""
        if self._client is None:
            try:
                from autotrade.analysis.searxng_client import SearXNGClient

                self._client = SearXNGClient(host=self.searxng_host)
            except ImportError:
                logger.error("searxng_client module not found")
                return None
        return self._client

    def execute(self, task: Task) -> TaskResult:
        start = time.time()

        if task.type == TaskType.WEB_SEARCH:
            return self._web_search(task, start)
        elif task.type == TaskType.NEWS_SEARCH:
            return self._news_search(task, start)
        elif task.type == TaskType.FINANCE_SEARCH:
            return self._finance_search(task, start)

        return TaskResult(
            success=False,
            message=f"Unknown task type for SearchAgent: {task.type}",
            agent_used=self.name,
        )

    def _web_search(self, task: Task, start: float) -> TaskResult:
        """Perform general web search and synthesize results."""
        query = task.data.get("query", "")
        max_results = task.data.get("max_results", 5)
        synthesize = task.data.get("synthesize", True)

        if not query:
            return TaskResult(
                success=False, message="No query provided", agent_used=self.name
            )

        client = self._get_client()
        if not client:
            return TaskResult(
                success=False,
                message="SearXNG client not available",
                agent_used=self.name,
            )

        # Perform search
        response = client.search(query)

        if not response.results:
            return TaskResult(
                success=False,
                message=f"No results found for: {query}",
                agent_used=self.name,
                duration_ms=int((time.time() - start) * 1000),
            )

        # Get raw results text
        results_text = response.to_text(max_results=max_results)

        if not synthesize:
            return TaskResult(
                success=True,
                message=results_text,
                data={"query": query, "num_results": len(response.results)},
                agent_used=self.name,
                duration_ms=int((time.time() - start) * 1000),
            )

        # Use LLM to synthesize results
        prompt = f"""Based on these search results, provide a concise answer:

{results_text}

Synthesize the key information into a clear, helpful response.
"""

        synthesis = self._call_ollama(
            prompt,
            "You are a research assistant. Synthesize search results into clear, accurate answers.",
        )

        return TaskResult(
            success=True,
            message=synthesis or results_text,
            data={
                "query": query,
                "num_results": len(response.results),
                "raw_results": results_text,
                "synthesis": synthesis,
            },
            agent_used=self.name,
            model_used=self._get_model() or "none",
            duration_ms=int((time.time() - start) * 1000),
        )

    def _news_search(self, task: Task, start: float) -> TaskResult:
        """Search for news articles."""
        query = task.data.get("query", "")
        max_results = task.data.get("max_results", 5)

        client = self._get_client()
        if not client:
            return TaskResult(
                success=False,
                message="SearXNG client not available",
                agent_used=self.name,
            )

        response = client.search_news(query)
        results_text = response.to_text(max_results=max_results)

        return TaskResult(
            success=len(response.results) > 0,
            message=results_text if response.results else f"No news found for: {query}",
            data={"query": query, "num_results": len(response.results)},
            agent_used=self.name,
            duration_ms=int((time.time() - start) * 1000),
        )

    def _finance_search(self, task: Task, start: float) -> TaskResult:
        """Search for financial/stock information."""
        query = task.data.get("query", "")
        symbol = task.data.get("symbol", "")
        max_results = task.data.get("max_results", 5)

        # Enhance query with finance context
        search_query = f"{symbol} {query} stock market" if symbol else query

        client = self._get_client()
        if not client:
            return TaskResult(
                success=False,
                message="SearXNG client not available",
                agent_used=self.name,
            )

        response = client.search_finance(search_query)
        results_text = response.to_text(max_results=max_results)

        # Synthesize financial info
        prompt = f"""Analyze these financial search results for {symbol or "the query"}:

{results_text}

Provide a concise summary of the key financial information, news, and market sentiment.
"""

        synthesis = self._call_ollama(
            prompt,
            "You are a financial analyst. Extract and summarize key market information.",
        )

        return TaskResult(
            success=len(response.results) > 0,
            message=synthesis or results_text,
            data={
                "query": search_query,
                "symbol": symbol,
                "num_results": len(response.results),
                "synthesis": synthesis,
            },
            agent_used=self.name,
            model_used=self._get_model() or "none",
            duration_ms=int((time.time() - start) * 1000),
        )


class DevSearchAgent(BaseAgent):
    """Agent for developer-focused searches: code, repos, docs, Stack Overflow."""

    def __init__(self, searxng_host: Optional[str] = None):
        super().__init__("DevSearchAgent", ModelTier.CODE)  # Use CODE tier for analysis
        self.searxng_host = searxng_host
        self._client = None

    def _get_client(self):
        """Lazy-load the SearXNG client."""
        if self._client is None:
            try:
                from autotrade.analysis.searxng_client import SearXNGClient

                self._client = SearXNGClient(host=self.searxng_host)
            except ImportError:
                logger.error("searxng_client module not found")
                return None
        return self._client

    def execute(self, task: Task) -> TaskResult:
        start = time.time()

        handlers = {
            TaskType.CODE_SEARCH: self._code_search,
            TaskType.REPO_SEARCH: self._repo_search,
            TaskType.DOCS_SEARCH: self._docs_search,
            TaskType.STACKOVERFLOW_SEARCH: self._stackoverflow_search,
            TaskType.PACKAGE_SEARCH: self._package_search,
        }

        handler = handlers.get(task.type)
        if handler:
            return handler(task, start)

        return TaskResult(
            success=False,
            message=f"Unknown task type for DevSearchAgent: {task.type}",
            agent_used=self.name,
        )

    def _code_search(self, task: Task, start: float) -> TaskResult:
        """Search for code snippets and examples."""
        query = task.data.get("query", "")
        language = task.data.get("language")
        max_results = task.data.get("max_results", 5)
        synthesize = task.data.get("synthesize", True)

        client = self._get_client()
        if not client:
            return TaskResult(
                success=False,
                message="SearXNG client not available",
                agent_used=self.name,
            )

        response = client.search_code(query, language=language)
        results_text = response.to_text(max_results=max_results)

        if not response.results:
            return TaskResult(
                success=False,
                message=f"No code results found for: {query}",
                agent_used=self.name,
                duration_ms=int((time.time() - start) * 1000),
            )

        if not synthesize:
            return TaskResult(
                success=True,
                message=results_text,
                data={
                    "query": query,
                    "language": language,
                    "num_results": len(response.results),
                },
                agent_used=self.name,
                duration_ms=int((time.time() - start) * 1000),
            )

        # Use code model to analyze and synthesize results
        lang_hint = f"in {language}" if language else ""
        prompt = f"""Analyze these code search results {lang_hint}:

{results_text}

Extract and explain:
1. The most relevant code patterns or solutions
2. Best practices shown
3. Any caveats or gotchas mentioned
4. Recommended approach for implementation
"""

        synthesis = self._call_ollama(
            prompt,
            "You are a senior developer analyzing code search results. Be specific about code patterns and provide actionable guidance.",
        )

        return TaskResult(
            success=True,
            message=synthesis or results_text,
            data={
                "query": query,
                "language": language,
                "num_results": len(response.results),
                "raw_results": results_text,
                "synthesis": synthesis,
            },
            agent_used=self.name,
            model_used=self._get_model() or "none",
            duration_ms=int((time.time() - start) * 1000),
        )

    def _repo_search(self, task: Task, start: float) -> TaskResult:
        """Search for repositories on GitHub/GitLab."""
        query = task.data.get("query", "")
        language = task.data.get("language")
        max_results = task.data.get("max_results", 5)

        client = self._get_client()
        if not client:
            return TaskResult(
                success=False,
                message="SearXNG client not available",
                agent_used=self.name,
            )

        response = client.search_repos(query, language=language)
        results_text = response.to_text(max_results=max_results)

        if not response.results:
            return TaskResult(
                success=False,
                message=f"No repositories found for: {query}",
                agent_used=self.name,
                duration_ms=int((time.time() - start) * 1000),
            )

        # Analyze repos to recommend best options
        prompt = f"""Analyze these repository search results:

{results_text}

For each repository:
1. What problem does it solve?
2. Is it actively maintained (if you can tell)?
3. How might it be useful for an agentic trading workflow?

Recommend the best options and explain why.
"""

        synthesis = self._call_ollama(
            prompt,
            "You are a developer evaluating open-source repositories. Focus on relevance, quality, and maintainability.",
        )

        return TaskResult(
            success=True,
            message=synthesis or results_text,
            data={
                "query": query,
                "language": language,
                "num_results": len(response.results),
                "synthesis": synthesis,
            },
            agent_used=self.name,
            model_used=self._get_model() or "none",
            duration_ms=int((time.time() - start) * 1000),
        )

    def _docs_search(self, task: Task, start: float) -> TaskResult:
        """Search for documentation and tutorials."""
        query = task.data.get("query", "")
        framework = task.data.get("framework")
        max_results = task.data.get("max_results", 5)

        client = self._get_client()
        if not client:
            return TaskResult(
                success=False,
                message="SearXNG client not available",
                agent_used=self.name,
            )

        response = client.search_docs(query, framework=framework)
        results_text = response.to_text(max_results=max_results)

        if not response.results:
            return TaskResult(
                success=False,
                message=f"No documentation found for: {query}",
                agent_used=self.name,
                duration_ms=int((time.time() - start) * 1000),
            )

        # Synthesize documentation into actionable guide
        framework_hint = f"for {framework}" if framework else ""
        prompt = f"""Synthesize these documentation search results {framework_hint}:

{results_text}

Create a concise guide that:
1. Explains the key concepts
2. Provides step-by-step instructions
3. Notes common pitfalls to avoid
4. Links to the most authoritative sources
"""

        synthesis = self._call_ollama(
            prompt,
            "You are a technical writer creating clear, actionable documentation guides.",
        )

        return TaskResult(
            success=True,
            message=synthesis or results_text,
            data={
                "query": query,
                "framework": framework,
                "num_results": len(response.results),
                "synthesis": synthesis,
            },
            agent_used=self.name,
            model_used=self._get_model() or "none",
            duration_ms=int((time.time() - start) * 1000),
        )

    def _stackoverflow_search(self, task: Task, start: float) -> TaskResult:
        """Search Stack Overflow for solutions."""
        query = task.data.get("query", "")
        tags = task.data.get("tags", [])
        max_results = task.data.get("max_results", 5)

        client = self._get_client()
        if not client:
            return TaskResult(
                success=False,
                message="SearXNG client not available",
                agent_used=self.name,
            )

        response = client.search_stackoverflow(query, tags=tags)
        results_text = response.to_text(max_results=max_results)

        if not response.results:
            return TaskResult(
                success=False,
                message=f"No Stack Overflow results for: {query}",
                agent_used=self.name,
                duration_ms=int((time.time() - start) * 1000),
            )

        # Extract key solutions
        tags_str = ", ".join(tags) if tags else "general"
        prompt = f"""Analyze these Stack Overflow results for [{tags_str}]:

{results_text}

Extract:
1. The core problem being solved
2. The accepted/best solution approach
3. Code snippets if available
4. Common mistakes to avoid
5. Alternative approaches mentioned
"""

        synthesis = self._call_ollama(
            prompt,
            "You are a senior developer extracting solutions from Stack Overflow. Focus on working code and best practices.",
        )

        return TaskResult(
            success=True,
            message=synthesis or results_text,
            data={
                "query": query,
                "tags": tags,
                "num_results": len(response.results),
                "synthesis": synthesis,
            },
            agent_used=self.name,
            model_used=self._get_model() or "none",
            duration_ms=int((time.time() - start) * 1000),
        )

    def _package_search(self, task: Task, start: float) -> TaskResult:
        """Search for Python/npm packages."""
        query = task.data.get("query", "")
        package_type = task.data.get("package_type", "pypi")  # pypi or npm
        max_results = task.data.get("max_results", 5)

        client = self._get_client()
        if not client:
            return TaskResult(
                success=False,
                message="SearXNG client not available",
                agent_used=self.name,
            )

        if package_type == "npm":
            response = client.search_npm(query)
        else:
            response = client.search_pypi(query)

        results_text = response.to_text(max_results=max_results)

        if not response.results:
            return TaskResult(
                success=False,
                message=f"No packages found for: {query}",
                agent_used=self.name,
                duration_ms=int((time.time() - start) * 1000),
            )

        # Evaluate packages
        prompt = f"""Evaluate these {package_type.upper()} package search results:

{results_text}

For each package:
1. What does it do?
2. Is it actively maintained?
3. Pros and cons
4. Installation command

Recommend the best option for a production trading system.
"""

        synthesis = self._call_ollama(
            prompt,
            "You are evaluating packages for a production system. Focus on reliability, maintenance, and documentation quality.",
        )

        return TaskResult(
            success=True,
            message=synthesis or results_text,
            data={
                "query": query,
                "package_type": package_type,
                "num_results": len(response.results),
                "synthesis": synthesis,
            },
            agent_used=self.name,
            model_used=self._get_model() or "none",
            duration_ms=int((time.time() - start) * 1000),
        )


class PMValidatorAgent(BaseAgent):
    """Agent for validating PM workflow, picks, and trading plans.

    Can auto-execute PM workflow if it hasn't run.
    """

    def __init__(self, auto_execute: bool = True):
        super().__init__("PMValidatorAgent", ModelTier.MEDIUM)
        self._openai_client = None
        self.auto_execute = auto_execute  # Auto-run PM if not executed

    def _get_openai(self):
        """Lazy-load OpenAI client for complex validation."""
        if self._openai_client is None:
            try:
                from autotrade.utils.openai_client import get_openai_client

                self._openai_client = get_openai_client()
            except ImportError:
                logger.warning("OpenAI client not available")
        return self._openai_client

    def execute(self, task: Task) -> TaskResult:
        start = time.time()

        handlers = {
            TaskType.PM_WORKFLOW_CHECK: self._check_pm_workflow,
            TaskType.PICKS_VALIDATION: self._validate_picks,
            TaskType.PLAN_REVIEW: self._review_plan,
        }

        handler = handlers.get(task.type)
        if handler:
            return handler(task, start)

        return TaskResult(
            success=False,
            message=f"Unknown task type for PMValidatorAgent: {task.type}",
            agent_used=self.name,
        )

    def _run_pm_workflow(self) -> Dict[str, Any]:
        """Execute the PM workflow and return results."""
        logger.info("Auto-executing PM workflow...")

        try:
            from autotrade.execution.post_market_workflow import PostMarketWorkflow

            # Run with execute=True to save the plan
            pm = PostMarketWorkflow(dry_run=False)
            result = pm.run()

            # Save the plan
            plan_date = get_pm_plan_date()
            plan_path = Path("plans") / f"pm_plan_{plan_date.strftime('%Y-%m-%d')}.json"
            with open(plan_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

            logger.info(f"PM workflow executed, plan saved to {plan_path}")

            return {
                "success": True,
                "plan_path": str(plan_path),
                "positions": len(result.get("positions", [])),
                "result": result,
            }

        except Exception as e:
            logger.error(f"PM workflow execution failed: {e}")
            return {"success": False, "error": str(e)}

    def _analyze_pm_logs(
        self, candidate_dates: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """Analyze PM workflow logs for actionable issues."""
        from datetime import date, timedelta

        logs_dir = Path("logs")
        searched_paths: List[Path] = []
        seen_paths = set()

        def _add_candidate_log(dt_obj: Any) -> None:
            if dt_obj is None or not hasattr(dt_obj, "strftime"):
                return
            p = logs_dir / f"pm_workflow_{dt_obj.strftime('%Y-%m-%d')}.log"
            key = str(p.resolve()) if p.exists() else str(p)
            if key in seen_paths:
                return
            seen_paths.add(key)
            searched_paths.append(p)

        for dt_obj in candidate_dates or []:
            _add_candidate_log(dt_obj)

        market_now = get_market_now()
        _add_candidate_log(market_now.date())
        _add_candidate_log((market_now - timedelta(days=1)).date())
        _add_candidate_log(date.today())
        _add_candidate_log(date.today() - timedelta(days=1))

        log_path = next((p for p in searched_paths if p.exists()), None)
        if log_path is None:
            return {
                "exists": False,
                "issues": [],
                "note": "No PM log found for expected dates",
                "searched_logs": [str(p) for p in searched_paths],
            }

        try:
            content = log_path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()

            error_level_re = re.compile(r"\|\s*(ERROR|CRITICAL)\s*\|", re.IGNORECASE)
            warning_level_re = re.compile(r"\|\s*WARNING\s*\|", re.IGNORECASE)
            failure_token_re = re.compile(
                r"\b(traceback|exception|failed|failure)\b", re.IGNORECASE
            )

            errors: List[str] = []
            warnings: List[str] = []

            for line in lines:
                lower_line = line.lower()
                if error_level_re.search(line):
                    errors.append(line)
                    continue
                if warning_level_re.search(line):
                    warnings.append(line)
                    # Treat explicit failures in warnings as issues (but ignore FAILSAFE telemetry).
                    if (
                        failure_token_re.search(lower_line)
                        and "[failsafe]" not in lower_line
                    ):
                        errors.append(line)
                    continue
                # Some logs may not include a normalized level block.
                if (
                    failure_token_re.search(lower_line)
                    and "[failsafe]" not in lower_line
                ):
                    errors.append(line)

            success_indicators = [
                "pm workflow - post-market analysis",
                "plan saved",
                "[pm workflow][complete]",
            ]
            ran_successfully = any(ind in content.lower() for ind in success_indicators)

            return {
                "exists": True,
                "log_path": str(log_path),
                "searched_logs": [str(p) for p in searched_paths],
                "total_lines": len(lines),
                "errors": errors[-5:],  # Last 5 actionable errors
                "warnings": warnings[-5:],
                "ran_successfully": ran_successfully,
                "issues": errors[-5:] if errors else [],
            }

        except Exception as e:
            return {
                "exists": True,
                "log_path": str(log_path),
                "error": str(e),
                "issues": [str(e)],
            }

    def _check_pm_workflow(self, task: Task, start: float) -> TaskResult:
        """Check if PM workflow ran and produced valid output.

        If not run and auto_execute is True, will execute PM workflow.
        """
        from datetime import timedelta

        auto_execute = task.data.get("auto_execute", self.auto_execute)

        plans_dir = Path("plans")
        now = get_market_now()
        today = now.date()
        yesterday = today - timedelta(days=1)
        target_date = get_pm_plan_date(now)

        # Check for today's plan (generated yesterday evening)
        plan_dates = []
        for dt in (target_date, today, yesterday):
            if dt not in plan_dates:
                plan_dates.append(dt)
        plan_files = {
            f"{dt.isoformat()}": plans_dir / f"pm_plan_{dt.strftime('%Y-%m-%d')}.json"
            for dt in plan_dates
        }

        results = {
            "plans_found": {},
            "latest_plan": None,
            "pm_ran_recently": False,
            "issues": [],
            "log_analysis": None,
            "auto_executed": False,
        }

        for label, path in plan_files.items():
            if path.exists():
                results["plans_found"][label] = str(path)
                try:
                    with open(path) as f:
                        plan = json.load(f)
                    results["latest_plan"] = plan
                    results["pm_ran_recently"] = True

                    # Validate plan structure
                    required_keys = ["generated_at", "positions", "account"]
                    missing = [k for k in required_keys if k not in plan]
                    if missing:
                        results["issues"].append(f"Plan missing keys: {missing}")

                    # Check positions count
                    positions = plan.get("positions", [])
                    if not positions:
                        results["issues"].append("No positions in plan")
                    else:
                        results["position_count"] = len(positions)

                        # Check for exits planned
                        exits = [
                            p
                            for p in positions
                            if p.get("recommended_action")
                            in ["exit", "exit_immediately"]
                        ]
                        results["exits_planned"] = len(exits)

                    break  # Found a valid plan

                except Exception as e:
                    results["issues"].append(f"Error reading {path}: {e}")

        # If PM didn't run, try to execute it
        if not results["pm_ran_recently"]:
            results["issues"].append("No PM plan found for today or yesterday")

            if auto_execute:
                logger.warning("PM workflow not found - auto-executing...")
                exec_result = self._run_pm_workflow()
                results["auto_executed"] = True
                results["execution_result"] = exec_result

                if exec_result["success"]:
                    results["pm_ran_recently"] = True
                    results["issues"] = []  # Clear issues since we fixed it
                    results["latest_plan"] = exec_result.get("result")

                    # Re-analyze logs after execution
                    results["log_analysis"] = self._analyze_pm_logs(plan_dates)
                else:
                    results["issues"].append(
                        f"Auto-execution failed: {exec_result.get('error')}"
                    )

        # Always analyze logs
        if results["log_analysis"] is None:
            results["log_analysis"] = self._analyze_pm_logs(plan_dates)

        # Check logs for issues
        log_issues = results["log_analysis"].get("issues", [])
        if log_issues:
            results["issues"].extend([f"Log error: {e[:50]}" for e in log_issues[:3]])

        success = results["pm_ran_recently"] and len(results["issues"]) == 0

        if results["auto_executed"] and results["pm_ran_recently"]:
            message = f"PM workflow auto-executed successfully. {results.get('position_count', 0)} positions analyzed."
        elif success:
            message = f"PM workflow validated. {results.get('position_count', 0)} positions, {results.get('exits_planned', 0)} exits planned."
        else:
            message = f"PM workflow issues: {', '.join(results['issues'][:3])}"

        return TaskResult(
            success=success,
            message=message,
            data=results,
            agent_used=self.name,
            model_used="rule-based",
            duration_ms=int((time.time() - start) * 1000),
        )

    def _validate_picks(self, task: Task, start: float) -> TaskResult:
        """Validate tomorrow's picks using local model + optional OpenAI."""
        use_openai = task.data.get("use_openai", False)

        # Check if picks were passed directly in task data (premarket validation)
        picks = task.data.get("picks", [])
        pick_count = task.data.get("pick_count", len(picks))
        cleanup_count = task.data.get("cleanup_count", 0)
        plan_date = task.data.get("plan_date", "unknown")
        market_phase = task.data.get("market_phase", "unknown")

        # If picks were passed directly, validate those
        if picks:
            validation_results = {
                "plan_date": plan_date,
                "market_phase": market_phase,
                "total_picks": pick_count,
                "cleanup_orders": cleanup_count,
                "picks_validated": len(picks),
                "verdict": "approve",
                "affected_symbols": [],
                "reasons": [],
                "warnings": [],
                "risk_flags": [],
            }

            # Basic validation of picks
            for pick in picks[:10]:
                symbol = pick.get("symbol", "???") if isinstance(pick, dict) else pick
                confidence = pick.get("confidence") if isinstance(pick, dict) else 0
                entry_price = pick.get("entry_price") if isinstance(pick, dict) else 0
                stop_loss = pick.get("stop_loss") if isinstance(pick, dict) else 0

                # Ensure numeric values (could be None)
                confidence = confidence if confidence is not None else 0
                entry_price = entry_price if entry_price is not None else 0
                stop_loss = stop_loss if stop_loss is not None else 0
                try:
                    confidence = float(confidence)
                except (TypeError, ValueError):
                    confidence = 0.0
                try:
                    entry_price = float(entry_price)
                except (TypeError, ValueError):
                    entry_price = 0.0
                try:
                    stop_loss = float(stop_loss)
                except (TypeError, ValueError):
                    stop_loss = 0.0

                # Flag low confidence picks
                if isinstance(pick, dict) and confidence < 50:
                    reason = f"{symbol}: Low confidence ({confidence:.0f})"
                    validation_results["warnings"].append(reason)
                    validation_results["reasons"].append(reason)

                # Flag missing stop loss
                invalid_stop = (
                    isinstance(pick, dict)
                    and entry_price > 0
                    and (stop_loss <= 0 or stop_loss >= entry_price)
                )
                if invalid_stop:
                    reason = (
                        f"{symbol}: Invalid stop loss"
                        if stop_loss > 0
                        else f"{symbol}: Missing stop loss"
                    )
                    validation_results["risk_flags"].append(reason)
                    validation_results["reasons"].append(reason)
                    if symbol:
                        validation_results["affected_symbols"].append(
                            str(symbol).upper()
                        )

            affected = list(dict.fromkeys(validation_results["affected_symbols"]))
            validation_results["affected_symbols"] = affected
            if affected:
                validation_results["verdict"] = (
                    "reject_all" if len(affected) >= len(picks) else "reject_partial"
                )
            elif validation_results["warnings"]:
                validation_results["verdict"] = "approve_with_warnings"

            # Create picks summary for LLM
            picks_summary = []
            for p in picks[:10]:
                if isinstance(p, dict):
                    entry_p = p.get("entry_price") or 0
                    stop_p = p.get("stop_loss") or 0
                    conf = p.get("confidence") or 0
                    picks_summary.append(
                        f"  {p.get('symbol')}: entry=${entry_p:.2f}, "
                        f"stop=${stop_p:.2f}, conf={conf:.0f}"
                    )
                else:
                    picks_summary.append(f"  {p}")

            prompt = f"""Validate this trading plan for {plan_date}:

Phase: {market_phase}
Total new entries: {pick_count}
Cleanup orders: {cleanup_count}

Top picks to validate:
{chr(10).join(picks_summary)}

Risk flags: {validation_results["risk_flags"][:3]}
Warnings: {validation_results["warnings"][:3]}

Return one brief sentence. The structured verdict is already set to
{validation_results["verdict"]}; do not contradict it.
"""

            summary = self._call_ollama(
                prompt, "You are a trading risk manager. Briefly validate the plan."
            )

            validation_results["summary"] = summary
            has_issues = validation_results["verdict"] in {
                "reject_partial",
                "reject_all",
            }

            return TaskResult(
                success=not has_issues,
                message=summary
                or f"Validated {pick_count} picks, {len(validation_results['risk_flags'])} risk flags",
                data=validation_results,
                agent_used=self.name,
                model_used=self._get_model() or "rule-based",
                duration_ms=int((time.time() - start) * 1000),
            )

        # Fall back to loading from file if no picks passed
        plan_path = task.data.get("plan_path")

        # Find latest plan
        if not plan_path:
            plans_dir = Path("plans")
            target_date = get_pm_plan_date()
            target_path = plans_dir / f"pm_plan_{target_date.strftime('%Y-%m-%d')}.json"
            if target_path.exists():
                plan_path = target_path
            else:
                plan_files = sorted(plans_dir.glob("pm_plan_*.json"), reverse=True)
                if plan_files:
                    plan_path = plan_files[0]

        if not plan_path or not Path(plan_path).exists():
            return TaskResult(
                success=False,
                message="No trading plan found to validate",
                agent_used=self.name,
                duration_ms=int((time.time() - start) * 1000),
            )

        with open(plan_path) as f:
            plan = json.load(f)

        positions = plan.get("positions", [])
        exits = [
            p
            for p in positions
            if p.get("recommended_action") in ["exit", "exit_immediately"]
        ]
        holds = [p for p in positions if p.get("recommended_action") == "hold"]
        adds = [p for p in positions if p.get("recommended_action") == "add"]

        validation_results = {
            "plan_date": plan.get("generated_at", "unknown"),
            "total_positions": len(positions),
            "exits": len(exits),
            "holds": len(holds),
            "adds": len(adds),
            "exit_symbols": [p["symbol"] for p in exits],
            "warnings": [],
            "risk_flags": [],
        }

        # Basic validation rules
        for pos in positions:
            symbol = pos.get("symbol", "???")
            conviction = pos.get("conviction_score", 50)
            pnl = pos.get("pnl_pct", 0)
            action = pos.get("recommended_action", "hold")

            # Flag low conviction holds
            if action == "hold" and conviction < 40:
                validation_results["warnings"].append(
                    f"{symbol}: Holding despite low conviction ({conviction:.1f})"
                )

            # Flag big losses not being exited
            if pnl < -5 and action != "exit":
                validation_results["risk_flags"].append(
                    f"{symbol}: {pnl:.1f}% loss but not exiting"
                )

            # Flag exits of profitable positions
            if pnl > 5 and action == "exit":
                validation_results["warnings"].append(
                    f"{symbol}: Exiting +{pnl:.1f}% winner - verify reason"
                )

        # Use OpenAI for deeper validation if requested
        openai_analysis = None
        if use_openai:
            openai = self._get_openai()
            if openai and openai.available:
                response = openai.validate_picks(
                    picks=positions,
                    market_context=f"Account equity: ${plan.get('account', {}).get('equity', 0):,.2f}",
                )
                if response.success:
                    openai_analysis = response.content
                    validation_results["openai_analysis"] = openai_analysis
                    validation_results["openai_cost"] = response.cost_usd

        # Summarize with local model
        prompt = f"""Validate this trading plan:

Total positions: {len(positions)}
Planned exits: {len(exits)} - {[p["symbol"] for p in exits]}
Risk flags: {validation_results["risk_flags"]}
Warnings: {validation_results["warnings"]}

Provide a brief validation summary (2-3 sentences).
"""

        summary = self._call_ollama(
            prompt, "You are a trading risk manager. Validate the plan briefly."
        )

        validation_results["summary"] = summary

        has_issues = len(validation_results["risk_flags"]) > 0

        return TaskResult(
            success=not has_issues,
            message=summary
            or f"Validated {len(positions)} positions, {len(validation_results['risk_flags'])} risk flags",
            data=validation_results,
            agent_used=self.name,
            model_used=self._get_model() or "rule-based",
            duration_ms=int((time.time() - start) * 1000),
        )

    def _review_plan(self, task: Task, start: float) -> TaskResult:
        """Full review of trading plan with LLM."""
        plan_path = task.data.get("plan_path")

        # Find latest plan
        if not plan_path:
            plans_dir = Path("plans")
            target_date = get_pm_plan_date()
            target_path = plans_dir / f"pm_plan_{target_date.strftime('%Y-%m-%d')}.json"
            if target_path.exists():
                plan_path = target_path
            else:
                plan_files = sorted(plans_dir.glob("pm_plan_*.json"), reverse=True)
                if plan_files:
                    plan_path = plan_files[0]

        if not plan_path or not Path(plan_path).exists():
            return TaskResult(
                success=False,
                message="No trading plan found to review",
                agent_used=self.name,
            )

        with open(plan_path) as f:
            plan = json.load(f)

        # Create summary for LLM
        positions_summary = []
        for p in plan.get("positions", [])[:15]:
            positions_summary.append(
                f"  {p.get('symbol')}: {p.get('pnl_pct', 0):.1f}% PnL, "
                f"conviction={p.get('conviction_score', 0):.0f}, "
                f"action={p.get('recommended_action')}"
            )

        prompt = f"""Review this trading plan:

Account: ${plan.get("account", {}).get("equity", 0):,.2f}
Generated: {plan.get("generated_at", "unknown")}

Positions:
{chr(10).join(positions_summary)}

Provide:
1. Overall assessment (1-2 sentences)
2. Top risk (1 sentence)
3. Top opportunity (1 sentence)
4. Suggested adjustment (1 sentence)
"""

        review = self._call_ollama(
            prompt, "You are a trading portfolio manager reviewing the next day's plan."
        )

        return TaskResult(
            success=True,
            message=review or "Plan review complete",
            data={"plan_path": str(plan_path), "review": review},
            agent_used=self.name,
            model_used=self._get_model() or "none",
            duration_ms=int((time.time() - start) * 1000),
        )


class ImprovementAgent(BaseAgent):
    """Agent for continuous analysis and improvement of trading strategy."""

    def __init__(self):
        super().__init__("ImprovementAgent", ModelTier.LARGE)
        self._openai_client = None

    def _get_openai(self):
        """Lazy-load OpenAI client for complex analysis."""
        if self._openai_client is None:
            try:
                from autotrade.utils.openai_client import get_openai_client

                self._openai_client = get_openai_client()
            except ImportError:
                logger.warning("OpenAI client not available")
        return self._openai_client

    def execute(self, task: Task) -> TaskResult:
        start = time.time()

        handlers = {
            TaskType.BACKTEST_ANALYSIS: self._analyze_backtest,
            TaskType.STRATEGY_IMPROVEMENT: self._suggest_improvements,
            TaskType.PERFORMANCE_REVIEW: self._review_performance,
            TaskType.WORKFLOW_ANALYSIS: self._analyze_workflow,
        }

        handler = handlers.get(task.type)
        if handler:
            return handler(task, start)

        return TaskResult(
            success=False,
            message=f"Unknown task type for ImprovementAgent: {task.type}",
            agent_used=self.name,
        )

    def _analyze_backtest(self, task: Task, start: float) -> TaskResult:
        """Analyze backtest results and extract insights."""
        results_path = task.data.get("results_path")
        use_openai = task.data.get("use_openai", False)

        # Find latest backtest results
        if not results_path:
            logs_dir = Path("logs")
            result_files = sorted(
                logs_dir.glob("backtest_results_*.json"), reverse=True
            )
            agentic_files = sorted(
                logs_dir.glob("agentic_backtest_*.json"), reverse=True
            )
            all_files = result_files + agentic_files
            all_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            if all_files:
                results_path = all_files[0]

        if not results_path or not Path(results_path).exists():
            return TaskResult(
                success=False, message="No backtest results found", agent_used=self.name
            )

        with open(results_path) as f:
            results = json.load(f)

        # Extract key metrics
        metrics = {
            "file": str(results_path),
            "total_trades": results.get("total_trades", 0),
            "win_rate": results.get("win_rate", 0),
            "profit_factor": results.get("profit_factor", 0),
            "total_pnl": results.get("total_pnl", 0),
            "avg_trade": results.get("avg_trade", 0),
            "max_drawdown": results.get("max_drawdown", 0),
        }

        # Build analysis prompt
        prompt = f"""Analyze these backtest results:

Total Trades: {metrics["total_trades"]}
Win Rate: {metrics["win_rate"]:.1%}
Profit Factor: {metrics["profit_factor"]:.2f}
Total P&L: ${metrics["total_pnl"]:,.2f}
Avg Trade: ${metrics["avg_trade"]:,.2f}
Max Drawdown: {metrics["max_drawdown"]:.1%}

Provide:
1. Overall assessment (good/needs work/poor)
2. Biggest strength
3. Biggest weakness
4. One specific improvement suggestion
"""

        # Use OpenAI for deeper analysis if requested
        if use_openai:
            openai = self._get_openai()
            if openai and openai.available:
                # Load strategy code for context
                strategy_code = ""
                try:
                    with open("unified_strategy.py") as f:
                        strategy_code = f.read()[:3000]
                except:
                    pass

                response = openai.improve_strategy(
                    strategy_code=strategy_code,
                    backtest_results=results,
                    lessons_learned=[],
                )

                if response.success:
                    return TaskResult(
                        success=True,
                        message=response.content,
                        data={
                            "metrics": metrics,
                            "openai_analysis": response.content,
                            "openai_cost": response.cost_usd,
                        },
                        agent_used=self.name,
                        model_used=f"OpenAI {response.model}",
                        duration_ms=int((time.time() - start) * 1000),
                    )

        # Use local model
        analysis = self._call_ollama(
            prompt,
            "You are a quantitative trading strategist analyzing backtest results.",
        )

        return TaskResult(
            success=True,
            message=analysis
            or f"Backtest: {metrics['win_rate']:.0%} win rate, PF {metrics['profit_factor']:.2f}",
            data={"metrics": metrics, "analysis": analysis},
            agent_used=self.name,
            model_used=self._get_model() or "none",
            duration_ms=int((time.time() - start) * 1000),
        )

    def _suggest_improvements(self, task: Task, start: float) -> TaskResult:
        """Suggest improvements to trading strategy using OpenAI."""
        use_openai = task.data.get("use_openai", True)

        # Gather context
        strategy_code = ""
        try:
            with open("unified_strategy.py") as f:
                strategy_code = f.read()
        except:
            pass

        # Get recent backtest results
        backtest_results = {}
        logs_dir = Path("logs")
        result_files = sorted(logs_dir.glob("backtest_results_*.json"), reverse=True)
        if result_files:
            with open(result_files[0]) as f:
                backtest_results = json.load(f)

        # Get lessons learned
        lessons = []
        try:
            from autotrade.signals.trade_learner import TradeLearner

            learner = TradeLearner()
            lessons = list(learner.lessons.keys())[:20]
        except:
            pass

        if use_openai:
            openai = self._get_openai()
            if openai and openai.available:
                response = openai.improve_strategy(
                    strategy_code=strategy_code[:5000],
                    backtest_results=backtest_results,
                    lessons_learned=lessons,
                )

                if response.success:
                    return TaskResult(
                        success=True,
                        message=response.content,
                        data={
                            "openai_analysis": response.content,
                            "openai_cost": response.cost_usd,
                            "lessons_count": len(lessons),
                            "has_backtest": bool(backtest_results),
                        },
                        agent_used=self.name,
                        model_used=f"OpenAI {response.model}",
                        duration_ms=int((time.time() - start) * 1000),
                    )

        # Fallback to local model
        prompt = f"""Based on this strategy code excerpt, suggest improvements:

```python
{strategy_code[:2000]}
```

Recent lessons learned: {lessons[:10]}

Provide 3-5 specific improvement suggestions.
"""

        suggestions = self._call_ollama(
            prompt, "You are a quantitative trading strategist suggesting improvements."
        )

        return TaskResult(
            success=True,
            message=suggestions or "Unable to generate suggestions",
            data={"suggestions": suggestions},
            agent_used=self.name,
            model_used=self._get_model() or "none",
            duration_ms=int((time.time() - start) * 1000),
        )

    def _review_performance(self, task: Task, start: float) -> TaskResult:
        """Review overall trading performance."""
        days = task.data.get("days", 7)

        # Gather performance data from plans
        plans_dir = Path("plans")
        plan_files = sorted(plans_dir.glob("pm_plan_*.json"), reverse=True)[:days]

        performance_data = []
        for pf in plan_files:
            try:
                with open(pf) as f:
                    plan = json.load(f)
                performance_data.append(
                    {
                        "date": plan.get("generated_at", "")[:10],
                        "equity": plan.get("account", {}).get("equity", 0),
                        "positions": len(plan.get("positions", [])),
                    }
                )
            except:
                pass

        if not performance_data:
            return TaskResult(
                success=False, message="No performance data found", agent_used=self.name
            )

        # Calculate trends
        if len(performance_data) >= 2:
            equity_change = (
                performance_data[0]["equity"] - performance_data[-1]["equity"]
            )
            equity_pct = (
                (equity_change / performance_data[-1]["equity"]) * 100
                if performance_data[-1]["equity"] > 0
                else 0
            )
        else:
            equity_change = 0
            equity_pct = 0

        summary = {
            "period_days": len(performance_data),
            "latest_equity": performance_data[0]["equity"] if performance_data else 0,
            "equity_change": equity_change,
            "equity_change_pct": equity_pct,
            "avg_positions": sum(p["positions"] for p in performance_data)
            / len(performance_data)
            if performance_data
            else 0,
        }

        fallback_message = (
            f"Performance: {summary['equity_change_pct']:.1f}% over "
            f"{summary['period_days']} days"
        )
        if not bool(task.data.get("use_llm", True)):
            return TaskResult(
                success=True,
                message=fallback_message,
                data=summary,
                agent_used=self.name,
                model_used="disabled",
                duration_ms=int((time.time() - start) * 1000),
            )

        prompt = f"""Review this trading performance:

Period: {summary["period_days"]} days
Current Equity: ${summary["latest_equity"]:,.2f}
Equity Change: ${summary["equity_change"]:,.2f} ({summary["equity_change_pct"]:.1f}%)
Avg Positions: {summary["avg_positions"]:.1f}

Provide:
1. Performance assessment (1 sentence)
2. Key observation (1 sentence)
3. Recommendation (1 sentence)
"""

        review = self._call_ollama(
            prompt, "You are a portfolio manager reviewing trading performance."
        )

        return TaskResult(
            success=True,
            message=review or fallback_message,
            data=summary,
            agent_used=self.name,
            model_used=self._get_model() or "none",
            duration_ms=int((time.time() - start) * 1000),
        )

    def _analyze_workflow(self, task: Task, start: float) -> TaskResult:
        """Analyze workflow efficiency and suggest optimizations."""

        # Check which components have run recently
        workflow_status = {
            "pm_workflow": Path("logs").glob("pm_workflow_*.log"),
            "day_manager": Path("logs").glob("day_manager_*.log"),
            "premarket": Path("logs").glob("premarket_*.log"),
            "orchestrator": Path("logs").glob("orchestrator_*.log"),
        }

        component_status = {}
        for component, files in workflow_status.items():
            file_list = list(files)
            component_status[component] = {
                "log_count": len(file_list),
                "latest": str(file_list[0]) if file_list else None,
            }

        # Check module health
        modules_to_check = [
            "pm_workflow.py",
            "day_manager.py",
            "position_advisor.py",
            "unified_strategy.py",
            "agentic_orchestrator.py",
        ]

        module_status = {}
        for mod in modules_to_check:
            if Path(mod).exists():
                stat = Path(mod).stat()
                module_status[mod] = {
                    "exists": True,
                    "size_kb": stat.st_size / 1024,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            else:
                module_status[mod] = {"exists": False}

        analysis = {
            "components": component_status,
            "modules": module_status,
            "recommendations": [],
        }

        # Generate recommendations
        if not component_status.get("pm_workflow", {}).get("log_count"):
            analysis["recommendations"].append(
                "PM workflow not running - schedule it for 6:30 PM"
            )

        if not component_status.get("orchestrator", {}).get("log_count"):
            analysis["recommendations"].append(
                "Orchestrator not running - start start_orchestrator.bat"
            )

        prompt = f"""Analyze this trading workflow status:

Components: {json.dumps(component_status, indent=2)}

Provide:
1. Overall health assessment
2. Any bottlenecks or issues
3. Optimization suggestions
"""

        workflow_review = self._call_ollama(
            prompt, "You are a systems analyst reviewing a trading workflow."
        )

        analysis["review"] = workflow_review

        return TaskResult(
            success=True,
            message=workflow_review
            or f"Workflow analysis: {len(analysis['recommendations'])} recommendations",
            data=analysis,
            agent_used=self.name,
            model_used=self._get_model() or "none",
            duration_ms=int((time.time() - start) * 1000),
        )


class AutonomousResearchAgent(BaseAgent):
    """
    Truly autonomous agent for research, data fetching, and continuous improvement.

    Capabilities:
    1. Research stocks via web search (SearXNG + OpenAI)
    2. Fetch missing data from yfinance
    3. Optimize watchlist using AI
    4. Run full autonomous improvement cycles
    5. Update strategies based on learning
    """

    def __init__(self):
        super().__init__("AutonomousResearchAgent", ModelTier.LARGE)
        self._autonomous_agent = None

    def _get_agent(self):
        """Lazy-load the autonomous agent."""
        if self._autonomous_agent is None:
            try:
                from autotrade.core.autonomous_agent import AutonomousAgent

                self._autonomous_agent = AutonomousAgent()
                logger.info("AutonomousAgent loaded successfully")
            except ImportError as e:
                logger.error(f"Failed to load AutonomousAgent: {e}")
        return self._autonomous_agent

    def execute(self, task: Task) -> TaskResult:
        start = time.time()

        handlers = {
            TaskType.RESEARCH_STOCK: self._research_stock,
            TaskType.FETCH_DATA: self._fetch_data,
            TaskType.OPTIMIZE_WATCHLIST: self._optimize_watchlist,
            TaskType.CONTINUOUS_IMPROVEMENT: self._continuous_improvement,
        }

        handler = handlers.get(task.type)
        if handler:
            return handler(task, start)

        return TaskResult(
            success=False,
            message=f"Unknown task type for AutonomousResearchAgent: {task.type}",
            agent_used=self.name,
        )

    def _research_stock(self, task: Task, start: float) -> TaskResult:
        """Research a specific stock using web search and AI analysis."""
        symbol = task.data.get("symbol")
        if not symbol:
            return TaskResult(
                success=False,
                message="No symbol provided for research",
                agent_used=self.name,
            )

        agent = self._get_agent()
        if not agent:
            return TaskResult(
                success=False,
                message="AutonomousAgent not available",
                agent_used=self.name,
            )

        try:
            research = agent.web_researcher.research_stock(symbol)

            return TaskResult(
                success=True,
                message=f"Researched {symbol}: {research.get('sentiment', 'neutral')} sentiment, {len(research.get('news', []))} news items",
                data=research,
                agent_used=self.name,
                model_used="SearXNG+OpenAI",
                duration_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:
            return TaskResult(
                success=False,
                message=f"Research failed: {str(e)}",
                agent_used=self.name,
                duration_ms=int((time.time() - start) * 1000),
            )

    def _fetch_data(self, task: Task, start: float) -> TaskResult:
        """Fetch missing stock data from yfinance."""
        symbols = task.data.get("symbols", [])
        symbol = task.data.get("symbol")
        if symbol:
            symbols = [symbol]

        if not symbols:
            return TaskResult(
                success=False,
                message="No symbols provided for data fetch",
                agent_used=self.name,
            )

        agent = self._get_agent()
        if not agent:
            return TaskResult(
                success=False,
                message="AutonomousAgent not available",
                agent_used=self.name,
            )

        try:
            # Check which need fetching
            missing = agent.data_fetcher.get_missing_tickers(symbols)

            if not missing:
                return TaskResult(
                    success=True,
                    message=f"All {len(symbols)} symbols already have current data",
                    data={"symbols": symbols, "fetched": []},
                    agent_used=self.name,
                    duration_ms=int((time.time() - start) * 1000),
                )

            # Fetch missing data
            fetched = agent.data_fetcher.bulk_fetch(missing)

            return TaskResult(
                success=True,
                message=f"Fetched data for {len(fetched)}/{len(missing)} missing tickers",
                data={
                    "requested": symbols,
                    "missing": missing,
                    "fetched": [d["ticker"] for d in fetched],
                },
                agent_used=self.name,
                model_used="yfinance",
                duration_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:
            return TaskResult(
                success=False,
                message=f"Data fetch failed: {str(e)}",
                agent_used=self.name,
                duration_ms=int((time.time() - start) * 1000),
            )

    def _optimize_watchlist(self, task: Task, start: float) -> TaskResult:
        """Use AI to optimize the current watchlist."""
        agent = self._get_agent()
        if not agent:
            return TaskResult(
                success=False,
                message="AutonomousAgent not available",
                agent_used=self.name,
            )

        try:
            # Validate and optimize PM workflow
            result = agent.validate_pm_workflow()

            # If OpenAI optimization was performed
            optimization = result.get("optimization", {})

            return TaskResult(
                success=True,
                message=f"Watchlist optimization complete. Issues: {len(result.get('issues', []))}, Actions: {len(result.get('actions_taken', []))}",
                data=result,
                agent_used=self.name,
                model_used="OpenAI+Autonomous",
                duration_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:
            return TaskResult(
                success=False,
                message=f"Optimization failed: {str(e)}",
                agent_used=self.name,
                duration_ms=int((time.time() - start) * 1000),
            )

    def _continuous_improvement(self, task: Task, start: float) -> TaskResult:
        """Run a full autonomous improvement cycle."""
        agent = self._get_agent()
        if not agent:
            return TaskResult(
                success=False,
                message="AutonomousAgent not available",
                agent_used=self.name,
            )

        try:
            logger.info("Starting continuous improvement cycle...")

            # 1. Validate PM workflow
            pm_result = agent.validate_pm_workflow()

            # 2. Analyze backtests
            backtest_result = agent.run_backtest_analysis()

            # 3. Get strategy improvements (uses OpenAI)
            improvements = agent.suggest_strategy_improvements()

            result = {
                "pm_validation": pm_result,
                "backtest_analysis": backtest_result,
                "improvements": improvements,
                "completed_at": datetime.now().isoformat(),
            }

            # Save cycle results
            cycle_path = (
                Path("logs")
                / f"improvement_cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(cycle_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

            # Count total suggestions
            total_suggestions = len(improvements.get("priority_improvements", []))

            return TaskResult(
                success=True,
                message=f"Improvement cycle complete. {total_suggestions} priority improvements suggested. Saved to {cycle_path.name}",
                data=result,
                agent_used=self.name,
                model_used="OpenAI+Full",
                duration_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:
            logger.error(f"Improvement cycle failed: {e}")
            return TaskResult(
                success=False,
                message=f"Improvement cycle failed: {str(e)}",
                agent_used=self.name,
                duration_ms=int((time.time() - start) * 1000),
            )


# =============================================================================
# RECURRING ERROR REGISTRY
# =============================================================================


class RecurringErrorRegistry:
    """Tracks recurring errors to prevent infinite fix loops."""

    MAX_DAILY_ATTEMPTS = 6

    def __init__(self, max_failures: int = 3, cooldown_seconds: int = 300):
        self.max_failures = max_failures
        self.cooldown_seconds = cooldown_seconds
        self.error_counts: Dict[str, int] = {}
        self.error_timestamps: Dict[str, datetime] = {}
        self.error_messages: Dict[str, List[str]] = {}
        # Daily limits â€” prevent infinite retry loops across cooldown resets
        self.daily_counts: Dict[str, int] = {}
        self.daily_date: Optional[str] = None

    def _reset_daily_if_needed(self):
        """Reset daily counters at midnight."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.daily_date != today:
            self.daily_counts.clear()
            self.daily_date = today

    def record_failure(self, error_key: str, message: str):
        """Record a failure for the given error key."""
        self._reset_daily_if_needed()
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        self.error_timestamps[error_key] = datetime.now()
        self.daily_counts[error_key] = self.daily_counts.get(error_key, 0) + 1

        if error_key not in self.error_messages:
            self.error_messages[error_key] = []
        self.error_messages[error_key].append(message)

        # Keep only last 5 messages
        if len(self.error_messages[error_key]) > 5:
            self.error_messages[error_key] = self.error_messages[error_key][-5:]

        if self.error_counts[error_key] >= self.max_failures:
            logger.warning(
                f"[ErrorRegistry] {error_key} has failed {self.error_counts[error_key]} times. "
                f"Recent errors: {self.error_messages[error_key][-2:]}"
            )

    def record_success(self, error_key: str):
        """Record a success and reset the error count."""
        if error_key in self.error_counts:
            logger.info(
                f"[ErrorRegistry] {error_key} succeeded, resetting failure count"
            )
            self.error_counts.pop(error_key, None)
            self.error_timestamps.pop(error_key, None)
            self.error_messages.pop(error_key, None)

    def should_skip(self, error_key: str) -> bool:
        """Check if we should skip this operation due to recurring failures."""
        self._reset_daily_if_needed()

        # Hard daily cap â€” no more retries after MAX_DAILY_ATTEMPTS per error
        if self.daily_counts.get(error_key, 0) >= self.MAX_DAILY_ATTEMPTS:
            logger.warning(
                f"[ErrorRegistry] Daily limit reached for {error_key} "
                f"({self.daily_counts[error_key]}/{self.MAX_DAILY_ATTEMPTS}), skipping for today"
            )
            return True

        if error_key not in self.error_counts:
            return False

        # Check if cooldown period has passed
        if error_key in self.error_timestamps:
            elapsed = (
                datetime.now() - self.error_timestamps[error_key]
            ).total_seconds()
            if elapsed > self.cooldown_seconds:
                logger.info(
                    f"[ErrorRegistry] Cooldown expired for {error_key}, allowing retry"
                )
                self.error_counts.pop(error_key, None)
                self.error_timestamps.pop(error_key, None)
                return False

        return self.error_counts.get(error_key, 0) >= self.max_failures

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all recurring errors."""
        return {
            error_key: {
                "count": count,
                "last_seen": self.error_timestamps.get(
                    error_key, datetime.now()
                ).isoformat(),
                "recent_messages": self.error_messages.get(error_key, [])[-2:],
            }
            for error_key, count in self.error_counts.items()
            if count >= self.max_failures
        }


# =============================================================================
# TASK ROUTER
# =============================================================================


class TaskRouter:
    """Routes tasks to appropriate agents based on task type."""

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.error_registry = RecurringErrorRegistry()

        # Create agent instances
        diagnostic_agent = DiagnosticAgent()
        analysis_agent = AnalysisAgent()
        code_agent = CodeAgent()
        recovery_agent = RecoveryAgent(dry_run=dry_run)
        vision_agent = VisionAgent()
        chart_agent = ChartAgent()
        math_agent = MathAgent()
        planner_agent = PlannerAgent()
        massive_agent = MassiveContextAgent()
        search_agent = SearchAgent()
        dev_search_agent = DevSearchAgent()
        pm_validator_agent = PMValidatorAgent()
        improvement_agent = ImprovementAgent()
        autonomous_research_agent = AutonomousResearchAgent()

        self.agents = {
            # Diagnostic tasks
            TaskType.HEALTH_CHECK: diagnostic_agent,
            TaskType.ADVISOR_CHECK: diagnostic_agent,
            # Analysis tasks
            TaskType.LOG_ANALYSIS: analysis_agent,
            TaskType.CONTEXT_UPDATE: analysis_agent,
            # Code tasks
            TaskType.CODE_FIX: code_agent,
            # Recovery tasks
            TaskType.RECOVERY: recovery_agent,
            TaskType.CONFIG_UPDATE: recovery_agent,
            # Vision tasks (NEW)
            TaskType.SCREENSHOT_ANALYSIS: vision_agent,
            TaskType.UI_DEBUG: vision_agent,
            # Chart tasks (NEW)
            TaskType.CHART_ANALYSIS: chart_agent,
            # Math tasks (NEW)
            TaskType.MATH_PROBLEM: math_agent,
            TaskType.LOGIC_VERIFY: math_agent,
            # Planning tasks (NEW)
            TaskType.TASK_PLANNING: planner_agent,
            TaskType.TOOL_DECISION: planner_agent,
            # Massive context tasks (NEW)
            TaskType.REPO_ANALYSIS: massive_agent,
            TaskType.MASSIVE_LOG_ANALYSIS: massive_agent,
            # Search tasks (NEW)
            TaskType.WEB_SEARCH: search_agent,
            TaskType.NEWS_SEARCH: search_agent,
            TaskType.FINANCE_SEARCH: search_agent,
            # Developer search tasks (NEW)
            TaskType.CODE_SEARCH: dev_search_agent,
            TaskType.REPO_SEARCH: dev_search_agent,
            TaskType.DOCS_SEARCH: dev_search_agent,
            TaskType.STACKOVERFLOW_SEARCH: dev_search_agent,
            TaskType.PACKAGE_SEARCH: dev_search_agent,
            # PM Validation tasks (NEW)
            TaskType.PM_WORKFLOW_CHECK: pm_validator_agent,
            TaskType.PICKS_VALIDATION: pm_validator_agent,
            TaskType.PLAN_REVIEW: pm_validator_agent,
            # Improvement tasks (NEW)
            TaskType.BACKTEST_ANALYSIS: improvement_agent,
            TaskType.STRATEGY_IMPROVEMENT: improvement_agent,
            TaskType.PERFORMANCE_REVIEW: improvement_agent,
            TaskType.WORKFLOW_ANALYSIS: improvement_agent,
            # Autonomous research tasks (NEW)
            TaskType.RESEARCH_STOCK: autonomous_research_agent,
            TaskType.FETCH_DATA: autonomous_research_agent,
            TaskType.OPTIMIZE_WATCHLIST: autonomous_research_agent,
            TaskType.CONTINUOUS_IMPROVEMENT: autonomous_research_agent,
        }

        unique_agents = set(a.name for a in self.agents.values())
        logger.info(
            f"TaskRouter initialized with {len(unique_agents)} agents: "
            + ", ".join(sorted(unique_agents))
        )

    @staticmethod
    def _coerce_task(task: Any) -> Optional[Task]:
        """
        Normalize legacy task payloads into canonical Task objects.
        Accepts Task instances, dict payloads, and objects with `type`/`task_type`.
        """
        if isinstance(task, Task):
            return task

        if isinstance(task, dict):
            try:
                data = task.get("data", {})
                if not isinstance(data, dict):
                    data = {}
                return Task(
                    type=task.get("type"),
                    task_type=task.get("task_type"),
                    description=str(task.get("description", "")),
                    data=data,
                    priority=int(task.get("priority", 5)),
                )
            except Exception:
                return None

        if hasattr(task, "type") or hasattr(task, "task_type"):
            try:
                raw_data = getattr(task, "data", {})
                if not isinstance(raw_data, dict):
                    raw_data = {}
                return Task(
                    type=getattr(task, "type", None),
                    task_type=getattr(task, "task_type", None),
                    description=str(getattr(task, "description", "")),
                    data=raw_data,
                    priority=int(getattr(task, "priority", 5)),
                )
            except Exception:
                return None

        return None

    def route(self, task: Task) -> TaskResult:
        """Route a task to the appropriate agent with error handling."""
        task_obj = self._coerce_task(task)
        if task_obj is None:
            error_msg = (
                f"Task payload is invalid or missing schema-required fields: {task}"
            )
            logger.error(f"[TaskRouter] {error_msg}")
            return TaskResult(
                success=False,
                message=error_msg,
                data={"error_type": "invalid_task_payload"},
            )
        task = task_obj

        # Validate task has required type
        if not hasattr(task, "type") or task.type is None:
            error_msg = f"Task missing 'type' attribute: {task}"
            logger.error(f"[TaskRouter] {error_msg}")
            return TaskResult(
                success=False, message=error_msg, data={"error_type": "missing_type"}
            )

        agent = self.agents.get(task.type)
        if not agent:
            error_msg = f"No agent registered for task type: {task.type}"
            logger.warning(f"[TaskRouter] {error_msg}")
            return TaskResult(
                success=False,
                message=error_msg,
                data={"error_type": "unknown_task_type", "task_type": str(task.type)},
            )

        # Check for recurring errors before executing
        error_key = f"task_routing:{task.type.value}"
        if self.error_registry.should_skip(error_key):
            logger.warning(
                f"[TaskRouter] Skipping {task.type.value} due to recurring errors"
            )
            return TaskResult(
                success=False,
                message=f"Task type {task.type.value} temporarily disabled due to recurring failures",
                data={"error_type": "rate_limited"},
            )

        logger.info(f"[TaskRouter] Routing {task.type.value} to {agent.name}")

        try:
            result = agent.execute(task)
            # Track successful execution
            if result.success:
                self.error_registry.record_success(error_key)
            else:
                self.error_registry.record_failure(error_key, result.message)
            return result
        except Exception as e:
            error_msg = f"Task execution failed: {str(e)}"
            logger.error(f"[TaskRouter] {error_msg}", exc_info=True)
            self.error_registry.record_failure(error_key, error_msg)
            return TaskResult(
                success=False,
                message=error_msg,
                data={"error_type": "execution_error", "exception": str(e)},
            )

    def run_diagnostic_check(
        self, checks: Optional[List[str]] = None, task_name: str = "health_check"
    ) -> TaskResult:
        """Public helper to invoke DiagnosticAgent health checks."""
        task = Task(
            type=TaskType.HEALTH_CHECK,
            description=f"Diagnostic check: {task_name}",
            data={
                "task_name": task_name,
                "checks": checks or [],
            },
            priority=1,
        )
        return self.route(task)

    def run_pm_validator(
        self, plan_path: Optional[str] = None, auto_execute: bool = False
    ) -> TaskResult:
        """Public helper to invoke PMValidatorAgent."""
        data: Dict[str, Any] = {"auto_execute": auto_execute}
        if plan_path:
            data["plan_path"] = plan_path
        task = Task(
            type=TaskType.PM_WORKFLOW_CHECK,
            description="Validate PM workflow plan",
            data=data,
            priority=1,
        )
        return self.route(task)

    def run_chart_analysis(
        self, image_path: str, symbol: str = "", timeframe: str = "3M"
    ) -> TaskResult:
        """Public helper to invoke ChartAgent analysis."""
        task = Task(
            type=TaskType.CHART_ANALYSIS,
            description=f"Chart analysis for {symbol or 'symbol'}",
            data={
                "image_path": image_path,
                "symbol": symbol,
                "timeframe": timeframe,
            },
            priority=1,
        )
        return self.route(task)

    def run_improvement_review(
        self, lessons_context: Optional[Dict[str, Any]] = None
    ) -> TaskResult:
        """Public helper to invoke ImprovementAgent strategy suggestions."""
        task = Task(
            type=TaskType.STRATEGY_IMPROVEMENT,
            description="Lessons-informed strategy improvement review",
            data={
                "use_openai": False,
                "lessons_context": lessons_context or {},
            },
            priority=1,
        )
        return self.route(task)

    def get_agent_models(self) -> Dict[str, str]:
        """Get the model each agent is using."""
        models = {}
        for agent in set(self.agents.values()):
            models[agent.name] = agent._get_model() or "none"
        return models

    def test_legacy_compatibility(self) -> Dict[str, Any]:
        """Test that TaskRouter handles legacy code patterns gracefully.

        This verifies that:
        1. Tasks can be created with task_type instead of type (backwards compat)
        2. TaskResult has output/error properties
        3. String task types are converted to enums
        4. Invalid task types are handled gracefully

        Returns:
            Dict with test results
        """
        results = {"passed": [], "failed": []}

        # Test 1: Legacy task_type parameter
        try:
            task = Task(task_type=TaskType.HEALTH_CHECK, description="Legacy test")
            assert task.type == TaskType.HEALTH_CHECK
            results["passed"].append("task_type_alias")
            logger.info("[COMPAT TEST] âœ“ task_type alias works")
        except Exception as e:
            results["failed"].append(("task_type_alias", str(e)))
            logger.error(f"[COMPAT TEST] âœ— task_type alias failed: {e}")

        # Test 2: String to enum conversion
        try:
            task = Task(type="health_check", description="String type test")
            assert task.type == TaskType.HEALTH_CHECK
            results["passed"].append("string_type_conversion")
            logger.info("[COMPAT TEST] âœ“ String type conversion works")
        except Exception as e:
            results["failed"].append(("string_type_conversion", str(e)))
            logger.error(f"[COMPAT TEST] âœ— String type conversion failed: {e}")

        # Test 3: TaskResult backwards compat properties
        try:
            result = TaskResult(success=True, message="Test message")
            assert result.output == "Test message"
            assert result.error == "Test message"
            results["passed"].append("task_result_properties")
            logger.info("[COMPAT TEST] âœ“ TaskResult properties work")
        except Exception as e:
            results["failed"].append(("task_result_properties", str(e)))
            logger.error(f"[COMPAT TEST] âœ— TaskResult properties failed: {e}")

        # Test 4: Error registry prevents loops
        try:
            error_key = "test:compatibility_check"
            for i in range(5):
                self.error_registry.record_failure(error_key, f"Test failure {i}")

            should_skip = self.error_registry.should_skip(error_key)
            assert should_skip, "Error registry should rate-limit after max failures"
            results["passed"].append("error_registry_rate_limiting")
            logger.info("[COMPAT TEST] âœ“ Error registry rate limiting works")

            # Clean up test data
            self.error_registry.record_success(error_key)
        except Exception as e:
            results["failed"].append(("error_registry_rate_limiting", str(e)))
            logger.error(f"[COMPAT TEST] âœ— Error registry test failed: {e}")

        # Test 5: Invalid task type handling
        try:
            task = Task(type=TaskType.HEALTH_CHECK, description="Test")
            task.type = None  # Simulate missing type
            result = self.route(task)
            assert not result.success
            assert (
                "missing" in result.message.lower() or "type" in result.message.lower()
            )
            results["passed"].append("missing_type_handling")
            logger.info("[COMPAT TEST] âœ“ Missing type handling works")
        except Exception as e:
            results["failed"].append(("missing_type_handling", str(e)))
            logger.error(f"[COMPAT TEST] âœ— Missing type handling failed: {e}")

        logger.info(
            f"[COMPAT TEST] Results: {len(results['passed'])} passed, {len(results['failed'])} failed"
        )
        return results


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================


class AgenticOrchestrator:
    """
    Always-on orchestrator that monitors system health and ensures
    agentic workflows are running properly using modular agents.
    """

    def __init__(self, dry_run: bool = True, check_interval: int = 30):
        self.dry_run = dry_run
        self.check_interval = check_interval
        self.server_url = "http://127.0.0.1:8000"
        self.running = False
        self.last_health_check = None
        self.last_context_update = None
        self.failure_count = 0
        self.max_failures_before_recovery = 3

        # Initialize task router with all agents
        self.router = TaskRouter(dry_run=dry_run)

        # Log TaskRouter capabilities
        logger.info(
            f"TaskRouter initialized with error registry (max_failures=3, cooldown=300s)"
        )
        logger.debug(f"Available task types: {len(self.router.agents)} registered")

        # Demonstrate legacy compatibility
        logger.info("[COMPAT] TaskRouter supports legacy code patterns:")
        logger.info("  - Task(task_type=...) alias for Task(type=...)")
        logger.info("  - String task types auto-convert to enums")
        logger.info("  - TaskResult.output and .error properties")
        logger.info("  - RecurringErrorRegistry prevents infinite loops")

        # State tracking
        self.state = {
            "server_healthy": False,
            "advisor_status": "unknown",
            "signals_status": "unknown",
            "last_error": None,
            "recovery_attempts": 0,
            "last_check": None,
            "models_in_use": {},
            "error_registry_summary": {},
        }

        # Log available models
        self._log_model_info()

        logger.info(
            f"Orchestrator initialized (dry_run={dry_run}, interval={check_interval}s)"
        )

    def _log_model_info(self):
        """Log which models are being used by each agent."""
        models = self.router.get_agent_models()
        self.state["models_in_use"] = models
        logger.info("=" * 50)
        logger.info("AGENT MODEL CONFIGURATION:")
        for agent, model in models.items():
            logger.info(f"  {agent}: {model}")
        logger.info("=" * 50)

    def run_health_check(self) -> Dict[str, Any]:
        """Run a complete health check using modular agents."""
        logger.info("=" * 50)
        logger.info("Running health check...")

        # Create tasks for each check
        tasks = [
            Task(TaskType.HEALTH_CHECK, "System health check"),
            Task(TaskType.ADVISOR_CHECK, "Advisor status check"),
            Task(TaskType.LOG_ANALYSIS, "Analyze recent logs"),
            Task(TaskType.PM_WORKFLOW_CHECK, "Check if PM workflow ran"),
            Task(TaskType.PICKS_VALIDATION, "Validate tomorrow's picks"),
        ]

        results = {}
        for task in tasks:
            result = self.router.route(task)
            results[task.type.value] = {
                "success": result.success,
                "message": result.message,
                "agent": result.agent_used,
                "model": result.model_used,
                "duration_ms": result.duration_ms,
                "data": result.data,
            }

            # Update state based on results
            if task.type == TaskType.HEALTH_CHECK:
                self.state["server_healthy"] = result.data.get("server", {}).get(
                    "healthy", False
                )
            elif task.type == TaskType.ADVISOR_CHECK:
                self.state["advisor_status"] = (
                    "agentic" if result.success else "fallback"
                )

        self.state["last_check"] = datetime.now().isoformat()
        self.last_health_check = datetime.now()

        # Get error registry summary
        self.state["error_registry_summary"] = self.router.error_registry.get_summary()
        if self.state["error_registry_summary"]:
            logger.warning(
                f"[ErrorRegistry] {len(self.state['error_registry_summary'])} recurring errors tracked"
            )

        # Log summary
        logger.info("-" * 30)
        logger.info("HEALTH CHECK SUMMARY:")
        for check_type, result in results.items():
            status = "[OK]" if result["success"] else "[FAIL]"
            logger.info(f"  {status} {check_type}: {result['message'][:60]}")
        logger.info("-" * 30)

        # Check for critical failures
        health_result = results.get("health_check", {})
        if not health_result.get("success"):
            self.failure_count += 1
            self.state["last_error"] = health_result.get("message")
            logger.error(f"Health check failed: {self.state['last_error']}")
        else:
            self.failure_count = 0

        # Attempt recovery if needed
        if self.failure_count >= self.max_failures_before_recovery:
            logger.warning(
                f"Too many failures ({self.failure_count}), triggering recovery..."
            )
            recovery_task = Task(
                TaskType.RECOVERY,
                "Auto-recovery after multiple failures",
                data={"issue": self.state["last_error"]},
            )
            recovery_result = self.router.route(recovery_task)
            results["recovery"] = {
                "success": recovery_result.success,
                "message": recovery_result.message,
                "agent": recovery_result.agent_used,
            }

        return results

    def generate_context_snapshot(self) -> Dict[str, Any]:
        """Generate daily context snapshot using AnalysisAgent."""
        task = Task(TaskType.CONTEXT_UPDATE, "Daily context snapshot")
        result = self.router.route(task)

        if result.success:
            self.last_context_update = datetime.now()
            logger.info(result.message)
        else:
            logger.error(f"Context snapshot failed: {result.message}")

        return result.data

    def check_signals_pipeline(self) -> Dict[str, Any]:
        """Check if signals are being generated."""
        signals_dir = Path("logs")  # signals are in logs/ per PROJECT_CONTEXT

        # Find most recent signals file
        signal_files = list(signals_dir.glob("signals_*.json"))
        if not signal_files:
            self.state["signals_status"] = "no_files"
            return {"active": False, "error": "No signal files found"}

        latest = max(signal_files, key=lambda f: f.stat().st_mtime)
        age_seconds = time.time() - latest.stat().st_mtime
        age_hours = age_seconds / 3600

        if age_hours > 24:
            self.state["signals_status"] = "stale"
            return {
                "active": False,
                "latest": str(latest),
                "age_hours": round(age_hours, 1),
            }

        self.state["signals_status"] = "active"
        return {"active": True, "latest": str(latest), "age_hours": round(age_hours, 1)}

    def run(self):
        """Main orchestrator loop."""
        self.running = True
        logger.info("=" * 60)
        logger.info("AGENTIC ORCHESTRATOR STARTING")
        logger.info(f"   Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        logger.info(f"   Check interval: {self.check_interval} seconds")
        logger.info("=" * 60)

        # Initial context snapshot
        self.generate_context_snapshot()

        try:
            while self.running:
                try:
                    # Run health check
                    self.run_health_check()

                    # Check signals pipeline
                    signals = self.check_signals_pipeline()
                    logger.info(f"Signals: {signals}")

                    # Generate daily snapshot if needed
                    if self.last_context_update:
                        hours_since_snapshot = (
                            datetime.now() - self.last_context_update
                        ).total_seconds() / 3600
                        if hours_since_snapshot >= 24:
                            self.generate_context_snapshot()

                    # Wait for next check
                    logger.info(f"Next check in {self.check_interval} seconds...")
                    logger.info("")
                    time.sleep(self.check_interval)

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error(f"Error in orchestrator loop: {e}")
                    import traceback

                    traceback.print_exc()
                    time.sleep(self.check_interval)

        except KeyboardInterrupt:
            logger.info("\nOrchestrator stopped by user")
        finally:
            self.running = False
            logger.info("Orchestrator shutdown complete")

    def stop(self):
        """Stop the orchestrator."""
        self.running = False


def main():
    parser = argparse.ArgumentParser(
        description="Agentic Orchestrator for Trading System"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run in dry-run mode (no real actions)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live mode (enables recovery actions)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Health check interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--once", action="store_true", help="Run a single health check and exit"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available Ollama models and exit",
    )
    parser.add_argument(
        "--test-compat",
        action="store_true",
        help="Run TaskRouter compatibility tests and exit",
    )

    args = parser.parse_args()

    # Compatibility test mode
    if args.test_compat:
        print("\n" + "=" * 60)
        print("TASKROUTER COMPATIBILITY TEST")
        print("=" * 60)
        router = TaskRouter(dry_run=True)
        results = router.test_legacy_compatibility()
        print(f"\nPassed: {len(results['passed'])}")
        for test in results["passed"]:
            print(f"  [OK] {test}")
        if results["failed"]:
            print(f"\nFailed: {len(results['failed'])}")
            for test, error in results["failed"]:
                print(f"  [X] {test}: {error}")
        else:
            print("\nAll tests passed! [OK]")
        print("=" * 60)
        return

    # List models mode
    if args.list_models:
        print("\n" + "=" * 60)
        print("AVAILABLE OLLAMA MODELS")
        print("=" * 60)
        models = get_available_ollama_models()
        for model in models:
            print(f"  - {model}")
        print("\n" + "=" * 60)
        print("MODEL TIER ASSIGNMENTS")
        print("=" * 60)
        for tier in ModelTier:
            selected = select_model_for_tier(tier, models)
            print(f"\n{tier.value.upper()}:")
            print(f"  Selected: {selected}")
            print(f"  Candidates: {', '.join(AVAILABLE_MODELS.get(tier, []))}")
        return

    dry_run = not args.live
    orchestrator = AgenticOrchestrator(dry_run=dry_run, check_interval=args.interval)

    if args.once:
        results = orchestrator.run_health_check()
        print("\n" + json.dumps(results, indent=2, default=str))
    else:
        orchestrator.run()


if __name__ == "__main__":
    main()

