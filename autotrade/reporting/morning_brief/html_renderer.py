from __future__ import annotations

import shutil
from datetime import datetime
from html import escape
from pathlib import Path

from autotrade.reporting.morning_brief.fact_assembler import (
    FactAssemblyResult,
    TickerFacts,
)
from autotrade.reporting.morning_brief.narrative_llm import NarrativeResult
from autotrade.reporting.morning_brief.tier_engine import TieredBrief, TieredTicker


DEFAULT_OUTPUT = Path("reports/Top10_Robinhood/top10_latest.html")


def write_html_report(
    facts: FactAssemblyResult,
    tiers: TieredBrief,
    narrative: NarrativeResult,
    output_path: str | Path = DEFAULT_OUTPUT,
    archive_existing: bool = True,
    generated_at: datetime | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if archive_existing and output.exists():
        archive_dir = output.parent / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = (generated_at or datetime.now()).strftime("%Y%m%d")
        archive_path = archive_dir / f"top10_{stamp}.html"
        if output.resolve() != archive_path.resolve():
            shutil.copy2(output, archive_path)
    output.write_text(
        render_html(facts, tiers, narrative, generated_at or datetime.now()),
        encoding="utf-8",
    )
    return output


def render_html(
    facts: FactAssemblyResult,
    tiers: TieredBrief,
    narrative: NarrativeResult,
    generated_at: datetime | None = None,
) -> str:
    timestamp = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M")
    banners = _banners(facts, narrative)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Morning Opportunity Brief</title>
<style>
:root {{ color-scheme: dark; --bg:#101114; --panel:#17191f; --line:#2b3038; --text:#f2f5f8; --muted:#aab2bd; --green:#45c26b; --amber:#e7b84a; --red:#ef6461; --blue:#6fb7ff; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font-family:system-ui,-apple-system,Segoe UI,Arial,sans-serif; letter-spacing:0; }}
main {{ max-width:860px; margin:0 auto; padding:14px; }}
.top {{ border-bottom:1px solid var(--line); padding-bottom:12px; margin-bottom:12px; }}
h1 {{ font-size:1.35rem; margin:0 0 8px; }}
.meta,.fresh {{ color:var(--muted); font-size:.9rem; line-height:1.35; }}
.summary {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin:10px 0; }}
.tile {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px; min-height:54px; }}
.tile strong {{ display:block; font-size:1.25rem; color:var(--text); }}
.tile span {{ color:var(--muted); font-size:.78rem; }}
.tape {{ display:grid; gap:6px; margin:10px 0; }}
.tape div,.banner,.empty {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px; }}
.banner {{ border-color:var(--amber); color:#ffe0a3; margin:8px 0; }}
details {{ border:1px solid var(--line); border-radius:8px; margin:10px 0; background:#14161b; overflow:hidden; }}
summary {{ min-height:48px; padding:13px 12px; cursor:pointer; font-weight:700; }}
.rows {{ display:grid; gap:8px; padding:0 10px 10px; }}
.row {{ display:grid; grid-template-columns:minmax(80px,.7fr) minmax(0,1.8fr); gap:9px; align-items:start; min-height:76px; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px; }}
button.symbol {{ min-height:44px; width:100%; border:1px solid var(--line); border-radius:7px; background:#202632; color:var(--green); font-size:1.05rem; font-weight:800; }}
.why {{ min-width:0; line-height:1.3; }}
.line {{ margin-top:4px; }}
.stats,.badges {{ color:var(--muted); font-size:.86rem; margin-top:4px; display:flex; flex-wrap:wrap; gap:5px; }}
.badge {{ color:var(--text); border:1px solid var(--line); border-radius:999px; padding:2px 7px; }}
.cat {{ color:#ffd98a; }}
.avoid {{ color:var(--red); }}
@media (max-width:430px) {{ main {{ padding:10px; }} .summary {{ grid-template-columns:repeat(2,1fr); }} .row {{ grid-template-columns:1fr; }} button.symbol {{ text-align:left; padding-left:12px; }} }}
</style>
</head>
<body>
<main>
<section class="top">
<h1>Morning Opportunity Brief</h1>
<div class="meta">Generated {escape(timestamp)} local time. New-opportunity scan behind legacy top10_latest.html.</div>
{"".join(banners)}
{_summary_tiles(facts, tiers)}
<div class="tape">{"".join(f"<div>{escape(line)}</div>" for line in narrative.tape_lines)}</div>
<div class="fresh">{_freshness_footer(facts)}</div>
</section>
{_tier_section("S", "Conviction", tiers.tiers["S"], tiers, narrative, open_section=True)}
{_tier_section("A", "Strength Re-entry", tiers.tiers["A"], tiers, narrative)}
{_tier_section("B", "Fresh Setups", tiers.tiers["B"], tiers, narrative)}
{_tier_section("C", "Shorts / Failed Breakouts", tiers.tiers["C"], tiers, narrative)}
{_avoid_section(tiers)}
</main>
<script>
document.querySelectorAll('button.symbol').forEach((button) => {{
  button.addEventListener('click', async () => {{
    try {{ await navigator.clipboard.writeText(button.dataset.symbol); button.textContent = button.dataset.symbol + ' copied'; }}
    catch (_) {{ button.textContent = button.dataset.symbol; }}
    setTimeout(() => button.textContent = button.dataset.symbol, 900);
  }});
}});
</script>
</body>
</html>
"""


def _banners(facts: FactAssemblyResult, narrative: NarrativeResult) -> list[str]:
    banners = []
    for source in facts.freshness:
        if source.name == "morning_brief_input_gate" and source.rejected_count > 0:
            banners.append(
                '<div class="banner">'
                f"Filtered {source.rejected_count} malformed signals "
                "(zero stop, target &lt;= entry, R:R &lt; 1.2, target &gt; 40% above entry). "
                f"See {escape(source.reject_path or source.path or 'logs/morning_brief_rejects_DATE.json')} for detail."
                "</div>"
            )
            continue
        if not source.fresh:
            label = f"{source.name}: {source.message}"
            banners.append(
                f'<div class="banner">Source warning - {escape(label)}</div>'
            )
    if narrative.banner:
        banners.append(f'<div class="banner">{escape(narrative.banner)}</div>')
    return banners


def _tier_section(
    tier: str,
    title: str,
    rows: list[TieredTicker],
    tiers: TieredBrief,
    narrative: NarrativeResult,
    open_section: bool = False,
) -> str:
    open_attr = " open" if open_section else ""
    if not rows:
        body = f'<div class="empty">{escape(tiers.explanations.get(tier, "No names."))}</div>'
    else:
        body = "".join(_ticker_row(row, narrative) for row in rows)
    return (
        f"<details{open_attr}><summary>Tier {escape(tier)} - {escape(title)} "
        f'({len(rows)})</summary><div class="rows">{body}</div></details>'
    )


def _ticker_row(row: TieredTicker, narrative: NarrativeResult) -> str:
    fact = row.facts
    why = narrative.why_by_symbol.get(fact.symbol) or _fallback_why(fact)
    badges = list(row.badges) + sorted(fact.source_flags)
    catalyst = " ".join(_catalyst_label(flag) for flag in sorted(fact.catalyst_flags))
    key_level = f" | Key ${fact.key_level:.2f}" if fact.key_level else ""
    gap = f" | Gap {fact.gap_pct:+.1f}%" if fact.gap_pct is not None else ""
    return f"""<article class="row">
<button class="symbol" data-symbol="{escape(fact.symbol)}">{escape(fact.symbol)}</button>
<div class="why">
<div>{escape(why)}</div>
<div class="line">{escape(_trade_line(fact))}</div>
<div class="stats">${fact.price:.2f}{gap} | ATR {fact.atr_14:.2f} | RS5D {fact.rel_strength_5d:+.1f}%{escape(key_level)}</div>
<div class="stats">{escape(_quality_line(fact))}</div>
<div class="badges">{"".join(f'<span class="badge">{escape(str(badge))}</span>' for badge in badges)} <span class="cat">{escape(catalyst)}</span></div>
</div>
</article>"""


def _summary_tiles(facts: FactAssemblyResult, tiers: TieredBrief) -> str:
    tiered_count = sum(len(rows) for rows in tiers.tiers.values())
    stale_count = sum(1 for source in facts.freshness if not source.fresh)
    return (
        '<div class="summary">'
        f'<div class="tile"><strong>{len(facts.facts)}</strong><span>candidates scanned</span></div>'
        f'<div class="tile"><strong>{tiered_count}</strong><span>names shown</span></div>'
        f'<div class="tile"><strong>{stale_count}</strong><span>source warnings</span></div>'
        "</div>"
    )


def _trade_line(fact: TickerFacts) -> str:
    pieces = []
    if fact.setup_type:
        pieces.append(fact.setup_type.replace("_", " "))
    if fact.stop_loss:
        pieces.append(f"stop ${fact.stop_loss:.2f}")
    if fact.target:
        pieces.append(f"target ${fact.target:.2f}")
    if fact.risk_reward:
        pieces.append(f"R:R {fact.risk_reward:.2f}")
    return " | ".join(pieces) if pieces else "No trade levels in source row"


def _quality_line(fact: TickerFacts) -> str:
    pieces = [f"score {fact.score:.1f}"]
    if fact.confidence is not None:
        pieces.append(f"conf {fact.confidence:.0f}")
    if fact.volume_ratio is not None:
        pieces.append(f"vol {fact.volume_ratio:.1f}x")
    if fact.rsi_14 is not None:
        pieces.append(f"RSI {fact.rsi_14:.0f}")
    if fact.overnight_expected_open_to_high_pct is not None:
        pieces.append(f"O-H {fact.overnight_expected_open_to_high_pct:+.1f}%")
    if fact.overnight_expected_open_to_close_pct is not None:
        pieces.append(f"O-C {fact.overnight_expected_open_to_close_pct:+.1f}%")
    return " | ".join(pieces)


def _avoid_section(tiers: TieredBrief) -> str:
    if tiers.avoid:
        body = "".join(
            f'<div class="empty avoid">{escape(symbol)}</div>' for symbol in tiers.avoid
        )
    else:
        body = f'<div class="empty">{escape(tiers.explanations.get("Avoid", "No avoid names."))}</div>'
    return f'<details><summary>Avoid List ({len(tiers.avoid)})</summary><div class="rows">{body}</div></details>'


def _fallback_why(fact: TickerFacts) -> str:
    return f"{len(fact.source_flags)} sources; score {fact.score:.1f}; thesis {fact.thesis_state}"


def _catalyst_label(flag: str) -> str:
    labels = {
        "earnings_within_5d": "Earnings!",
        "recent_offering": "Offering!",
        "dilution_risk": "Dilution!",
        "news_today": "News",
    }
    return labels.get(flag, flag)


def _freshness_footer(facts: FactAssemblyResult) -> str:
    parts = []
    for source in facts.freshness:
        state = "fresh" if source.fresh else "stale/missing"
        parts.append(f"{source.name}: {state}")
    return "Sources - " + "; ".join(escape(part) for part in parts)
