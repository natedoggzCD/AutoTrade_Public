"""LLM disabled stub for the public backtesting-focused distribution.

The public distribution can run without any LLM. This stub keeps
import paths valid for code that transitively references OpenAIClient,
but any call returns a failed/no-op response.
"""
from types import SimpleNamespace


def _disabled_response(*_a, **_kw):
    return SimpleNamespace(
        success=False,
        content="",
        error="LLM disabled in public backtesting distribution",
        tool_calls=None,
    )


class OpenAIClient:
    available = False

    def __init__(self, *_a, **_kw):
        pass

    def chat(self, *_a, **_kw):
        return _disabled_response()

    def complete(self, *_a, **_kw):
        return _disabled_response()

