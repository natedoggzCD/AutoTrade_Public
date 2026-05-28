"""
Global exception hook to route errors through agentic auto-fix.

This installs a sys.excepthook that:
1) Resolves missing-module errors via dependency_resolver (agentic).
2) Routes code errors through AutoFixPipeline + CodeAgent.
"""
from __future__ import annotations

import os
import sys
import traceback
from typing import Optional


def _should_skip(exc_type: type) -> bool:
    if exc_type in (KeyboardInterrupt, SystemExit):
        return True
    return False


def _default_excepthook(exc_type, exc, tb) -> None:
    sys.__excepthook__(exc_type, exc, tb)


def _route_exception(
    exc: Exception,
    err_text: Optional[str],
    file_hint: Optional[str],
    line_hint: Optional[int],
    context: str,
) -> dict:
    # Dependency errors first
    try:
        from autotrade.utils.dependency_resolver import resolve_and_install
        dep_result = resolve_and_install(error_text=str(exc), allow_agent=True)
    except Exception:
        dep_result = None

    if dep_result and dep_result.handled and dep_result.success:
        return {"success": True, "type": "dependency", "detail": dep_result.to_dict()}

    try:
        from autotrade.core.auto_fix_pipeline import AutoFixPipeline
        from autotrade.core.agentic_orchestrator import TaskRouter
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[2]
        router = TaskRouter(dry_run=False)
        pipeline = AutoFixPipeline(project_root)
        fix_result = pipeline.run(
            router,
            exc,
            err_text or str(exc),
            context=context,
            file_hint=file_hint,
            line_hint=line_hint,
        )
        return fix_result
    except Exception as e:
        return {"success": False, "reason": f"routing failed: {e}"}


def _handle_exception(exc_type, exc, tb) -> None:
    if _should_skip(exc_type):
        _default_excepthook(exc_type, exc, tb)
        return

    # Prevent recursion
    if os.environ.get("AUTOTRADE_AGENTIC_EXC_GUARD") == "1":
        _default_excepthook(exc_type, exc, tb)
        return

    os.environ["AUTOTRADE_AGENTIC_EXC_GUARD"] = "1"
    try:
        err_text = "".join(traceback.format_exception(exc_type, exc, tb))
        file_hint = None
        line_hint = None
        try:
            frames = traceback.extract_tb(tb)
            if frames:
                last = frames[-1]
                file_hint = last.filename
                line_hint = last.lineno
        except Exception:
            file_hint = None
            line_hint = None

        fix_result = _route_exception(
            exc=exc,
            err_text=err_text,
            file_hint=file_hint,
            line_hint=line_hint,
            context="global_excepthook",
        )

        # Print original error regardless.
        _default_excepthook(exc_type, exc, tb)

        if not fix_result.get("success"):
            return

        # Retry once after successful fix
        if os.environ.get("AUTOTRADE_AGENTIC_EXC_RETRY") == "1":
            return
        os.environ["AUTOTRADE_AGENTIC_EXC_RETRY"] = "1"
        try:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception:
            return
        return
    finally:
        os.environ.pop("AUTOTRADE_AGENTIC_EXC_GUARD", None)


def route_exception(
    exc: Exception,
    context: str = "manual_route",
    file_hint: Optional[str] = None,
    line_hint: Optional[int] = None,
) -> dict:
    err_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return _route_exception(
        exc=exc,
        err_text=err_text,
        file_hint=file_hint,
        line_hint=line_hint,
        context=context,
    )


def install() -> None:
    if os.environ.get("AUTOTRADE_AGENTIC_EXCEPTIONS", "1") in ("0", "false", "False"):
        return
    sys.excepthook = _handle_exception
    try:
        from autotrade.utils.agentic_log_router import install_log_router
        install_log_router()
    except Exception:
        pass
