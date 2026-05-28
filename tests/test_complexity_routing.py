"""Pytest coverage for task complexity routing and model selection."""

from __future__ import annotations

from autotrade.core.local_coding_agent import (
    AgentMode,
    AgentModelConfig,
    LocalCodingAgent,
    MODEL_CAPABILITIES,
    TaskComplexity,
    should_use_native_tools,
)


def _make_agent() -> LocalCodingAgent:
    agent = LocalCodingAgent.__new__(LocalCodingAgent)
    agent.models = AgentModelConfig()
    agent.mode = AgentMode.PLAN
    agent.task_complexity = TaskComplexity.STANDARD
    return agent


def test_task_complexity_enum_values():
    assert TaskComplexity.LIGHT.value == "light"
    assert TaskComplexity.STANDARD.value == "standard"
    assert TaskComplexity.HEAVY.value == "heavy"


def test_classify_light_keywords():
    agent = _make_agent()
    prompt = "verify that the dedup function is already implemented"
    assert agent._classify_task_complexity(prompt) == TaskComplexity.LIGHT


def test_classify_standard_default():
    agent = _make_agent()
    prompt = "implement the data processing pipeline"
    assert agent._classify_task_complexity(prompt) == TaskComplexity.STANDARD


def test_classify_heavy_keywords():
    agent = _make_agent()
    prompt = "refactor across all modules to use the new API pattern"
    assert agent._classify_task_complexity(prompt) == TaskComplexity.HEAVY


def test_long_prompt_with_light_keyword_is_standard():
    agent = _make_agent()
    prompt = "verify that " + "x " * 3000
    assert agent._classify_task_complexity(prompt) == TaskComplexity.STANDARD


def test_model_routing_plan_mode():
    agent = _make_agent()
    agent.mode = AgentMode.PLAN

    agent.task_complexity = TaskComplexity.LIGHT
    model, ctx = agent._get_model_for_mode()
    assert model == agent.models.light_planner
    assert ctx == agent.models.light_planner_ctx

    agent.task_complexity = TaskComplexity.STANDARD
    model, ctx = agent._get_model_for_mode()
    assert model == agent.models.planner
    assert ctx == agent.models.planner_ctx


def test_model_routing_act_mode():
    agent = _make_agent()
    agent.mode = AgentMode.ACT

    agent.task_complexity = TaskComplexity.LIGHT
    model, ctx = agent._get_model_for_mode()
    assert model == agent.models.light_coder
    assert ctx == agent.models.light_coder_ctx

    agent.task_complexity = TaskComplexity.STANDARD
    model, ctx = agent._get_model_for_mode()
    assert model == agent.models.coder
    assert ctx == agent.models.coder_ctx

    agent.task_complexity = TaskComplexity.HEAVY
    model, ctx = agent._get_model_for_mode()
    assert model == agent.models.heavy_coder
    assert ctx == agent.models.heavy_coder_ctx


def test_model_capabilities_registry():
    assert "qwen3:8b" in MODEL_CAPABILITIES
    assert should_use_native_tools("qwen3:8b")
    assert "granite-code:8b" in MODEL_CAPABILITIES
    assert not should_use_native_tools("granite-code:8b")
