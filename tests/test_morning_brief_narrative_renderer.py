from datetime import UTC, datetime

import requests

from autotrade.reporting.morning_brief.fact_assembler import (
    FactAssemblyResult,
    SourceFreshness,
    TapeContext,
    TickerFacts,
)
from autotrade.reporting.morning_brief.html_renderer import render_html
from autotrade.reporting.morning_brief.narrative_llm import build_narrative
from autotrade.reporting.morning_brief.tier_engine import assign_tiers


def _facts():
    return FactAssemblyResult(
        facts=[
            TickerFacts(
                symbol="AAA",
                price=10,
                gap_pct=2.5,
                atr_14=0.7,
                rel_strength_5d=4.2,
                source_flags=frozenset({"overnight_pick", "claw_yesterday", "pm_plan"}),
                catalyst_flags=frozenset({"earnings_within_5d"}),
                thesis_state="intact",
                key_level=12,
                score=90,
            )
        ],
        freshness=[
            SourceFreshness("overnight_signals", "logs/signals.json", True, "fresh"),
            SourceFreshness("position_thesis", None, False, "missing source"),
        ],
        tape=TapeContext(
            regime_label="neutral",
            breadth_pct=58,
            source_label="plan posture, not live tape",
            is_live=False,
            claw_theme="DecisionClaw confirms AAA",
            fade_theme="avoid weak fades",
        ),
    )


def test_narrative_falls_back_when_ollama_unavailable(monkeypatch):
    def _raise(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr("requests.post", _raise)

    facts = _facts()
    tiers = assign_tiers(facts.facts, shorts_disabled=True)
    narrative = build_narrative(facts, tiers, timeout=0.01)

    assert narrative.banner is not None
    assert len(narrative.tape_lines) == 3
    assert narrative.why_by_symbol["AAA"]


def test_render_html_contains_empty_s_explanation_and_mobile_controls():
    facts = _facts()
    weak_facts = FactAssemblyResult(
        facts=[
            TickerFacts(
                symbol="BBB",
                price=10,
                gap_pct=None,
                atr_14=0.7,
                rel_strength_5d=4.2,
                source_flags=frozenset({"overnight_pick", "pm_plan"}),
                catalyst_flags=frozenset(),
                thesis_state="none",
                score=70,
            )
        ],
        freshness=facts.freshness,
        tape=facts.tape,
    )
    tiers = assign_tiers(weak_facts.facts, shorts_disabled=True)
    narrative = build_narrative(
        weak_facts, tiers, timeout=0.01, ollama_url="http://127.0.0.1:9"
    )

    html = render_html(
        weak_facts,
        tiers,
        narrative,
        generated_at=datetime(2026, 5, 21, 11, 0, tzinfo=UTC),
    )

    assert "No 3+ source agreement today" in html
    assert 'button class="symbol"' in html
    assert "min-height:44px" in html
    assert "Tier D" not in html
    assert "Watch Bench" not in html
    assert "candidates scanned" in html
    assert "Source warning" in html
    assert "overflow:hidden" in html


def test_render_html_shows_input_gate_rejection_banner():
    facts = FactAssemblyResult(
        facts=[],
        freshness=[
            SourceFreshness(
                "morning_brief_input_gate",
                "logs/morning_brief_rejects_2026-05-20.json",
                True,
                "filtered 2 malformed signals",
                rejected_count=2,
                reject_path="logs/morning_brief_rejects_2026-05-20.json",
            )
        ],
    )
    tiers = assign_tiers(facts.facts, shorts_disabled=True)
    narrative = build_narrative(
        facts, tiers, timeout=0.01, ollama_url="http://127.0.0.1:9"
    )

    html = render_html(facts, tiers, narrative)

    assert "Filtered 2 malformed signals" in html
    assert "logs/morning_brief_rejects_2026-05-20.json" in html


def test_narrative_rejects_object_tape_lines(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": '{"tape_lines":[{"regime":"SELLOFF"},"two","three"],"why_by_symbol":{}}'
            }

    monkeypatch.setattr("requests.post", lambda *args, **kwargs: Response())

    facts = _facts()
    tiers = assign_tiers(facts.facts, shorts_disabled=True)
    narrative = build_narrative(facts, tiers)

    assert (
        narrative.tape_lines[0]
        == "Live tape unavailable; last plan posture: neutral with 58% breadth."
    )


def test_narrative_uses_deterministic_tape_when_context_is_not_live(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": '{"tape_lines":["Today is raging risk-on","ignore source","ignore"],"why_by_symbol":{}}'
            }

    monkeypatch.setattr("requests.post", lambda *args, **kwargs: Response())

    facts = _facts()
    tiers = assign_tiers(facts.facts, shorts_disabled=True)
    narrative = build_narrative(facts, tiers)

    assert narrative.tape_lines[0].startswith(
        "Live tape unavailable; last plan posture:"
    )
    assert "Today is raging" not in narrative.tape_lines[0]
