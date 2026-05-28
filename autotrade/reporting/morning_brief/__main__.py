from __future__ import annotations

import argparse
from pathlib import Path

from autotrade.reporting.morning_brief.fact_assembler import assemble_facts
from autotrade.reporting.morning_brief.html_renderer import DEFAULT_OUTPUT, write_html_report
from autotrade.reporting.morning_brief.narrative_llm import build_narrative
from autotrade.reporting.morning_brief.tier_engine import assign_tiers


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the mobile morning opportunity brief.")
    parser.add_argument("--root", default=".", help="Repository root containing logs/plans/data.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="HTML output path.")
    parser.add_argument("--no-archive", action="store_true", help="Do not archive an existing output file.")
    parser.add_argument("--skip-llm", action="store_true", help="Render deterministic labels without Ollama.")
    args = parser.parse_args()

    facts = assemble_facts(args.root)
    tiers = assign_tiers(facts.facts, disable_shorts_path=Path(args.root) / "flags" / "disable_shorts.flag")
    if args.skip_llm:
        from autotrade.reporting.morning_brief.narrative_llm import _fallback_narrative, _narrative_payload

        narrative = _fallback_narrative(_narrative_payload(facts, tiers), "LLM skipped by CLI flag.")
    else:
        narrative = build_narrative(facts, tiers)
    output = write_html_report(
        facts,
        tiers,
        narrative,
        output_path=args.output,
        archive_existing=not args.no_archive,
    )
    print(f"Morning brief written: {output}")
    print(f"Candidates: {len(facts.facts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
