from autotrade.reporting.morning_brief.fact_assembler import TickerFacts
from autotrade.reporting.morning_brief.tier_engine import assign_tiers


def _fact(symbol, sources, score=50, price=10, catalysts=(), thesis="none", losses=0):
    return TickerFacts(
        symbol=symbol,
        price=price,
        gap_pct=None,
        atr_14=0.5,
        rel_strength_5d=1.0,
        source_flags=frozenset(sources),
        catalyst_flags=frozenset(catalysts),
        thesis_state=thesis,
        score=score,
        recent_loss_count=losses,
    )


def test_tier_rules_caps_highest_tier_and_badges():
    facts = [
        _fact("AAA", {"overnight_pick", "claw_yesterday", "pm_plan"}, score=90),
        _fact("BBB", {"overnight_pick", "strength_reentry"}, score=80),
        _fact("CCC", {"overnight_pick", "pm_plan"}, score=70),
        _fact("DDD", {"overnight_pick"}, score=100),
    ]

    result = assign_tiers(facts, shorts_disabled=True)

    assert [row.facts.symbol for row in result.tiers["S"]] == ["AAA"]
    assert "B" in result.tiers["S"][0].badges
    assert [row.facts.symbol for row in result.tiers["A"]] == ["BBB"]
    assert [row.facts.symbol for row in result.tiers["B"]] == ["CCC"]
    assert "DDD" not in {
        row.facts.symbol for tier, rows in result.tiers.items() for row in rows
    }
    assert "D" not in result.tiers


def test_empty_s_has_explanation_without_promotion():
    result = assign_tiers(
        [_fact("AAA", {"overnight_pick", "pm_plan"}, score=90)],
        shorts_disabled=True,
    )

    assert result.tiers["S"] == []
    assert result.tiers["B"][0].facts.symbol == "AAA"
    assert "No 3+ source agreement" in result.explanations["S"]


def test_one_source_candidates_do_not_pad_primary_tiers():
    result = assign_tiers(
        [_fact("AAA", {"overnight_pick"}, score=90)],
        shorts_disabled=True,
    )

    assert result.tiers["S"] == []
    assert result.tiers["A"] == []
    assert result.tiers["B"] == []
    assert "D" not in result.tiers


def test_avoid_list_uses_broken_thesis_or_recent_losses():
    result = assign_tiers(
        [
            _fact("BROK", {"overnight_pick", "pm_plan"}, thesis="broken"),
            _fact("LOSS", {"overnight_pick", "pm_plan"}, losses=2),
        ],
        shorts_disabled=True,
    )

    assert result.avoid == ["BROK", "LOSS"]
