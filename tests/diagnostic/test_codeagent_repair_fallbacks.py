import time

import autotrade.core.agentic_orchestrator as orchestrator_mod
from autotrade.core.agentic_orchestrator import CodeAgent


def test_codex_repair_applies_json_patch(monkeypatch, tmp_path):
    target = tmp_path / "broken.py"
    target.write_text("value = 1\npritn(value)\n", encoding="utf-8")

    monkeypatch.setattr(
        orchestrator_mod,
        "get_llm_config",
        lambda: type(
            "Cfg",
            (),
            {
                "provider": "local",
                "codex_command": "codex",
                "codex_timeout": 120,
                "codex_use_stdin": False,
                "codex_extra_args": [],
                "repair_openai_enabled": False,
                "repair_codex_enabled": True,
            },
        )(),
    )

    monkeypatch.setenv("AUTOTRADE_ENABLE_CODEX_REPAIR", "1")

    monkeypatch.setattr(
        orchestrator_mod,
        "get_available_ollama_models",
        lambda: ["qwen2.5-coder:7b"],
    )

    def _fake_available(command: str = "codex") -> bool:
        return True

    def _fake_run_codex(**kwargs):
        return (
            True,
            '{"summary":"Fixed print typo","changes":[{"file":"%s","search":"pritn(value)","replace":"print(value)","reason":"fix typo"}]}'
            % str(target).replace("\\", "\\\\"),
            "",
        )

    monkeypatch.setattr("autotrade.utils.codex_cli.codex_available", _fake_available)
    monkeypatch.setattr("autotrade.utils.codex_cli.run_codex", _fake_run_codex)

    agent = CodeAgent()
    result = agent._codex_repair(
        file_path=str(target),
        error='NameError: name "pritn" is not defined',
        error_type="NameError",
        ctx=None,
        start=time.time(),
    )

    assert result is not None
    assert result.success is True
    assert "print(value)" in target.read_text(encoding="utf-8")
