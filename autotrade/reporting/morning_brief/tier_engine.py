from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from autotrade.reporting.morning_brief.fact_assembler import TickerFacts


TIER_ORDER = ("S", "A", "B", "C")


@dataclass(frozen=True)
class TieredTicker:
    facts: TickerFacts
    tier: str
    badges: tuple[str, ...] = ()


@dataclass(frozen=True)
class TieredBrief:
    tiers: dict[str, list[TieredTicker]]
    avoid: list[str]
    explanations: dict[str, str] = field(default_factory=dict)


def assign_tiers(
    facts: list[TickerFacts],
    shorts_disabled: bool | None = None,
    disable_shorts_path: str | Path = "flags/disable_shorts.flag",
) -> TieredBrief:
    if shorts_disabled is None:
        shorts_disabled = Path(disable_shorts_path).exists()

    tiers: dict[str, list[TieredTicker]] = {tier: [] for tier in TIER_ORDER}
    assigned: set[str] = set()

    sorted_facts = sorted(
        facts, key=lambda row: (-row.score, -len(row.source_flags), row.symbol)
    )
    for tier, cap in (("S", 3), ("A", 5), ("B", 7), ("C", 5)):
        if tier == "C" and shorts_disabled:
            continue
        for row in sorted_facts:
            if row.symbol in assigned or len(tiers[tier]) >= cap:
                continue
            if _qualifies(row, tier, shorts_disabled):
                badges = tuple(
                    other
                    for other in TIER_ORDER
                    if other != tier and _qualifies(row, other, shorts_disabled)
                )
                tiers[tier].append(TieredTicker(row, tier, badges))
                assigned.add(row.symbol)

    avoid = [
        row.symbol
        for row in sorted_facts
        if row.thesis_state == "broken" or row.recent_loss_count >= 2
    ][:10]

    explanations: dict[str, str] = {}
    if not tiers["S"]:
        explanations["S"] = "No 3+ source agreement today - selectivity by design."
    if shorts_disabled:
        explanations["C"] = "Shorts are disabled by flags/disable_shorts.flag."
    elif not tiers["C"]:
        explanations["C"] = "No dedicated short-side candidates found."
    for tier in ("A", "B"):
        if not tiers[tier]:
            explanations[tier] = "No qualifying multi-source candidates."
    if not avoid:
        explanations["Avoid"] = "No broken thesis or repeated-loss names flagged."

    return TieredBrief(tiers=tiers, avoid=avoid, explanations=explanations)


def _qualifies(row: TickerFacts, tier: str, shorts_disabled: bool) -> bool:
    source_count = len(row.source_flags)
    if tier == "S":
        return (
            source_count >= 3
            and "recent_offering" not in row.catalyst_flags
            and "dilution_risk" not in row.catalyst_flags
            and 2.0 <= row.price <= 200.0
        )
    if tier == "A":
        return source_count >= 2 and "strength_reentry" in row.source_flags
    if tier == "B":
        return source_count >= 2 and "overnight_pick" in row.source_flags
    if tier == "C":
        return (
            not shorts_disabled
            and source_count >= 2
            and (
                row.thesis_state == "broken"
                or row.recent_loss_count >= 2
                or "dilution_risk" in row.catalyst_flags
            )
        )
    return False
