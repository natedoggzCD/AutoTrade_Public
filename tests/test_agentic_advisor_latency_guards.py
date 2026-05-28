from __future__ import annotations

from autotrade.core.agentic_advisor import AgenticAdvisor, _AdviceCache


def test_agentic_advisor_initializes_langgraph_advisory_only_by_default(monkeypatch):
    captured = {}

    class _FakeTradingGraph:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("langgraph_workflow.graph.TradingGraph", _FakeTradingGraph)

    advisor = AgenticAdvisor(
        trading_client=object(),
        dry_run=False,
    )
    advisor._ensure_initialized()

    assert captured["trading_client"] is advisor.trading_client
    assert captured["dry_run"] is True
    assert captured["allow_live_execute"] is False


def test_clear_signal_bypasses_langgraph(monkeypatch):
    advisor = AgenticAdvisor(dry_run=True)
    monkeypatch.setattr(
        advisor,
        "_ensure_initialized",
        lambda: (_ for _ in ()).throw(AssertionError("graph should be bypassed")),
    )

    advice = advisor.get_advice(
        {
            "symbol": "ABCD",
            "final_score": 91,
            "entry_source": "overnight_plan",
            "validation_gated": False,
            "l2_spread_pct": 0.05,
            "critical_news": False,
            "entry_authority": {"eligible": True},
            "strict_plan_authority": True,
        }
    )

    assert advice["action"] == "buy"
    assert advice["advisor_type"] == "fast_path_bypass"
    assert advice["flags"] == ["llm_bypassed"]


def test_cooldown_budget_fallback_forces_hold_except_hard_stop():
    assert AgenticAdvisor._should_force_hold_on_cooldown_fallback(
        action="trim",
        reasoning="fallback:cooldown_or_budget: model unavailable",
        context={"pnl_pct": 2.5},
    )
    assert not AgenticAdvisor._should_force_hold_on_cooldown_fallback(
        action="exit",
        reasoning="fallback:cooldown_or_budget: hard stop",
        context={"pnl_pct": -8.0},
    )


def test_advice_cache_quantizes_and_returns_copy():
    cache = _AdviceCache(ttl_seconds=300)
    ctx = {"symbol": "ABCD", "pnl_pct": 1.01, "score": 86.9, "qty": 100}
    cache.put(ctx, {"action": "hold", "flags": []})

    cached = cache.get({"symbol": "abcd", "pnl_pct": 1.0, "score": 87.1, "qty": 100})
    assert cached == {"action": "hold", "flags": []}
    cached["flags"].append("mutated")

    assert cache.get(ctx) == {"action": "hold", "flags": []}
