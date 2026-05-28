"""LLM disabled stub for the public backtesting-focused distribution."""
from types import SimpleNamespace


def _disabled_response(*_a, **_kw):
    return SimpleNamespace(
        success=False,
        content="",
        error="LLM disabled in public backtesting distribution",
        tool_calls=None,
    )


class OpenRouterClient:
    available = False

    def __init__(self, *_a, **_kw):
        pass

    def chat(self, *_a, **_kw):
        return _disabled_response()

