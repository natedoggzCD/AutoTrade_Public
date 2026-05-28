from __future__ import annotations

from autotrade.core.autonomous_agent import AutonomousAgent


class _Logger:
    def __init__(self):
        self.warnings = []

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(str(msg) % args if args else str(msg))


def test_initialize_fast_loop_runtime_builds_runtime_without_enabling_stream(monkeypatch):
    monkeypatch.setenv("AUTOTRADE_FAST_LOOP_ENABLED", "0")

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _Logger()
    agent.api_key = None
    agent.secret_key = None
    agent._fast_loop_status = {
        "initialized": False,
        "enabled": False,
        "stream_ready": False,
        "error": None,
    }

    agent._initialize_fast_loop_runtime()

    assert agent.fast_loop_runtime is not None
    assert agent.fast_loop_stream is None
    assert agent._fast_loop_status["initialized"] is True
    assert agent._fast_loop_status["stream_ready"] is False


def test_initialize_fast_loop_runtime_builds_stream_when_enabled(monkeypatch):
    monkeypatch.setenv("AUTOTRADE_FAST_LOOP_ENABLED", "1")
    created = {}

    class _Bridge:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr("autotrade.core.autonomous_agent.AlpacaStreamBridge", _Bridge)

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _Logger()
    agent.api_key = "key"
    agent.secret_key = "secret"
    agent._fast_loop_status = {
        "initialized": False,
        "enabled": False,
        "stream_ready": False,
        "error": None,
    }

    agent._initialize_fast_loop_runtime()

    assert agent.fast_loop_runtime is not None
    assert isinstance(agent.fast_loop_stream, _Bridge)
    assert created["api_key"] == "key"
    assert created["secret_key"] == "secret"
    assert agent._fast_loop_status["stream_ready"] is True
