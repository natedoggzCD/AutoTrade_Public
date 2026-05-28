"""Mobile morning opportunity brief rendered behind the legacy top10 path."""

from autotrade.reporting.morning_brief.fact_assembler import (
    FactAssemblyResult,
    SourceFreshness,
    TickerFacts,
    assemble_facts,
)
from autotrade.reporting.morning_brief.html_renderer import render_html, write_html_report
from autotrade.reporting.morning_brief.narrative_llm import NarrativeResult, build_narrative
from autotrade.reporting.morning_brief.tier_engine import (
    TieredBrief,
    TieredTicker,
    assign_tiers,
)

__all__ = [
    "FactAssemblyResult",
    "NarrativeResult",
    "SourceFreshness",
    "TickerFacts",
    "TieredBrief",
    "TieredTicker",
    "assemble_facts",
    "assign_tiers",
    "build_narrative",
    "render_html",
    "write_html_report",
]
