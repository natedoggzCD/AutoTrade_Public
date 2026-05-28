from autotrade.core.agentic_orchestrator import (
    CodeAgent,
    CodeRepairContextRouter,
    RepairComplexity,
)


def test_force_escalation_prefers_openrouter_and_excludes_qwen_by_default():
    router = CodeRepairContextRouter()

    cascade = router.get_model_cascade(
        RepairComplexity.COMPLEX,
        force_escalation=True,
    )

    assert cascade == ["openrouter:google/gemini-2.0-pro-exp-02-05"]
    assert "qwen3-coder-next:latest" not in cascade


def test_force_escalation_appends_qwen_only_when_opted_in(monkeypatch):
    monkeypatch.delenv("AUTOTRADE_ENABLE_QWEN3_REPAIR", raising=False)

    agent = CodeAgent.__new__(CodeAgent)
    agent._context_router = CodeRepairContextRouter()
    agent._repair_qwen3_enabled = True
    agent._repair_flag_enabled = CodeAgent._repair_flag_enabled.__get__(agent, CodeAgent)
    agent._repair_qwen3_allowed = CodeAgent._repair_qwen3_allowed.__get__(agent, CodeAgent)
    agent._build_repair_cascade = CodeAgent._build_repair_cascade.__get__(agent, CodeAgent)

    cascade = agent._build_repair_cascade(ctx=None, force_escalation=True)

    assert cascade[-1] == "qwen3-coder-next:latest"
    assert cascade[0] == "openrouter:google/gemini-2.0-pro-exp-02-05"


def test_default_cascade_still_starts_with_openai_when_openai_flag_is_false():
    agent = CodeAgent.__new__(CodeAgent)
    agent._context_router = CodeRepairContextRouter()
    agent._repair_openai_enabled = False
    agent._build_repair_cascade = CodeAgent._build_repair_cascade.__get__(agent, CodeAgent)

    cascade = agent._build_repair_cascade(ctx=None, force_escalation=False)

    assert cascade[0].startswith("openai:")
