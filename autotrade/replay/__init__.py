"""Runtime replay utilities for non-live session reconstruction."""

from __future__ import annotations

from typing import Any

__all__ = [
    "RuntimeSessionReplay",
    "replay_runtime_session",
    "wait_for_runtime_replay_benchmark_completion",
    "run_runtime_replay_benchmark",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .runtime_session_replay import (
            RuntimeSessionReplay,
            replay_runtime_session,
            wait_for_runtime_replay_benchmark_completion,
            run_runtime_replay_benchmark,
        )

        exports = {
            "RuntimeSessionReplay": RuntimeSessionReplay,
            "replay_runtime_session": replay_runtime_session,
            "wait_for_runtime_replay_benchmark_completion": wait_for_runtime_replay_benchmark_completion,
            "run_runtime_replay_benchmark": run_runtime_replay_benchmark,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
