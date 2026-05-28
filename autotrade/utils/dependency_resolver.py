"""
Dependency resolver for missing Python modules.

Goal: resolve missing-module errors using the agent (TaskRouter) only.
Strategy:
1) Ask TaskRouter PACKAGE_SEARCH for a recommendation.
2) Parse a single `pip install ...` from the agent output.
"""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


_MISSING_MODULE_PATTERNS = [
    r"No module named '([^']+)'",
    r"ModuleNotFoundError: No module named '([^']+)'",
]
_PIP_HINT_RE = re.compile(r"pip install\s+([A-Za-z0-9_.-]+)", re.IGNORECASE)


@dataclass
class ResolveResult:
    handled: bool
    success: bool
    package: Optional[str] = None
    attempted: List[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "handled": self.handled,
            "success": self.success,
            "package": self.package,
            "attempted": self.attempted or [],
            "error": self.error,
        }


def extract_missing_module(error_text: str) -> Optional[str]:
    if not error_text:
        return None
    for pattern in _MISSING_MODULE_PATTERNS:
        m = re.search(pattern, error_text)
        if m:
            return m.group(1)
    return None


def _parse_first_pip_install(text: str) -> Optional[str]:
    if not text:
        return None
    m = _PIP_HINT_RE.search(text)
    if m:
        return m.group(1)
    return None


def _try_get_task_router(logger=None):
    try:
        from autotrade.core.agentic_orchestrator import TaskRouter
        return TaskRouter(dry_run=False)
    except Exception as e:
        if logger:
            logger.debug(f"[DEPS] TaskRouter unavailable: {e}")
        return None


def _agent_suggest_package(query: str, task_router, logger=None) -> Optional[str]:
    if not task_router:
        return None
    try:
        from autotrade.core.agentic_orchestrator import Task, TaskType
    except Exception as e:
        if logger:
            logger.debug(f"[DEPS] Task import failed: {e}")
        return None

    task = Task(
        type=TaskType.PACKAGE_SEARCH,
        description=f"Resolve missing dependency: {query}",
        data={"query": query, "package_type": "pypi", "max_results": 5},
        priority=1,
    )
    result = task_router.route(task)
    if not result.success:
        if logger:
            logger.debug(f"[DEPS] Package search failed: {result.message}")
        return None
    return _parse_first_pip_install(result.message or "")


def _build_candidates(
    module_name: Optional[str],
    error_text: Optional[str],
    task_router=None,
    logger=None,
) -> List[str]:
    candidates: List[str] = []

    root = None
    if module_name:
        root = module_name.split(".")[0]

    if root:
        query = f"python package for import {root}"
        agent_pkg = _agent_suggest_package(query, task_router, logger=logger)
        if agent_pkg:
            candidates.append(agent_pkg)
    elif error_text:
        query = f"python package to fix error: {error_text}"
        agent_pkg = _agent_suggest_package(query, task_router, logger=logger)
        if agent_pkg:
            candidates.append(agent_pkg)

    # De-dupe while preserving order
    seen = set()
    unique = []
    for c in candidates:
        if c and c not in seen:
            unique.append(c)
            seen.add(c)
    return unique


def _install_package(package: str, logger=None) -> bool:
    try:
        if logger:
            logger.info(f"[DEPS] Installing: {package}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
        )
        if result.returncode == 0:
            if logger:
                logger.info(f"[DEPS] Installed: {package}")
            return True
        if logger:
            logger.warning(f"[DEPS] Install failed for {package}: {result.stderr[:200]}")
        return False
    except Exception as e:
        if logger:
            logger.warning(f"[DEPS] Install error for {package}: {e}")
        return False


def resolve_and_install(
    module_name: Optional[str] = None,
    error_text: Optional[str] = None,
    task_router=None,
    logger=None,
    allow_agent: bool = True,
) -> ResolveResult:
    """
    Resolve and install a missing dependency.
    Returns ResolveResult with handled=True only if the error looked like a missing dependency.
    """
    if not module_name:
        module_name = extract_missing_module(error_text or "")

    if not module_name:
        return ResolveResult(handled=False, success=False, error="No dependency signal", attempted=[])

    if allow_agent and task_router is None:
        task_router = _try_get_task_router(logger=logger)
    if not task_router:
        return ResolveResult(
            handled=True,
            success=False,
            error="TaskRouter unavailable for agent-driven dependency resolution",
            attempted=[],
        )

    candidates = _build_candidates(
        module_name=module_name,
        error_text=error_text,
        task_router=task_router,
        logger=logger,
    )

    attempted: List[str] = []
    for pkg in candidates:
        attempted.append(pkg)
        if _install_package(pkg, logger=logger):
            return ResolveResult(handled=True, success=True, package=pkg, attempted=attempted)

    return ResolveResult(
        handled=True,
        success=False,
        package=None,
        attempted=attempted,
        error="All install attempts failed",
    )
