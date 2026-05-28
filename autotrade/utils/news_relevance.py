"""Ticker-scoped news relevance helpers.

These helpers intentionally use deterministic text checks. They are used as a
precondition before sentiment can influence live trading decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from autotrade.utils.security_metadata import get_company_aliases


@dataclass(frozen=True)
class NewsRelevance:
    label: str
    score: float
    title_ticker_mentions: int
    body_ticker_mentions: int
    company_mentions: int
    snippet: str
    reason: str


MATERIAL_KEYWORDS = (
    "acquisition",
    "approval",
    "beat",
    "beats",
    "buyback",
    "contract",
    "deal",
    "disclosed",
    "downgrade",
    "earnings",
    "fda",
    "guidance",
    "jumped",
    "lawsuit",
    "merger",
    "miss",
    "offering",
    "partnership",
    "purchase",
    "raises",
    "rejection",
    "shares fell",
    "shares jumped",
    "shelf",
    "upgrade",
    "warrant",
)

TANGENTIAL_KEYWORDS = (
    "comparison",
    "ipo discussion",
    "sector",
    "spacex",
    "versus",
    "vs",
    "which stock",
)


def count_phrase_mentions(text: str, phrase: str) -> int:
    phrase = str(phrase or "").strip()
    if not text or not phrase:
        return 0
    pattern = rf"(?<![A-Za-z0-9]){re.escape(phrase)}(?![A-Za-z0-9])"
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def count_ticker_mentions(text: str, ticker: str) -> int:
    return count_phrase_mentions(text, str(ticker or "").upper().strip())


def count_keyword_hits(text: str, keywords: tuple[str, ...]) -> int:
    return sum(1 for keyword in keywords if count_phrase_mentions(text, keyword))


def company_aliases(ticker: str) -> tuple[str, ...]:
    return get_company_aliases(ticker)


def split_article_units(body: str) -> list[str]:
    units: list[str] = []
    for block in re.split(r"\n\s*\n+", body or ""):
        block = block.strip()
        if not block:
            continue
        lines = [
            line.strip(" -\t") for line in block.splitlines() if line.strip(" -\t")
        ]
        if len(lines) > 1:
            units.extend(lines)
        else:
            units.append(block)
    return units


def extract_target_snippet(
    ticker: str,
    title: str,
    body: str,
    aliases: Optional[tuple[str, ...]] = None,
    max_chars: int = 1800,
) -> str:
    aliases = aliases if aliases is not None else company_aliases(ticker)
    targets = (str(ticker or "").upper().strip(), *aliases)
    matched_units: list[str] = []
    for unit in split_article_units(body):
        if any(count_phrase_mentions(unit, target) for target in targets if target):
            matched_units.append(unit)
    if not matched_units:
        return ""
    snippet = "\n".join(matched_units)
    if title and any(
        count_phrase_mentions(title, target) for target in targets if target
    ):
        snippet = f"{title}\n{snippet}"
    return snippet[:max_chars].strip()


def assess_news_relevance(ticker: str, title: str, body: str = "") -> NewsRelevance:
    ticker = str(ticker or "").upper().strip()
    title = str(title or "")
    body = str(body or "")
    aliases = company_aliases(ticker)
    title_mentions = count_ticker_mentions(title, ticker)
    body_mentions = count_ticker_mentions(body, ticker)
    combined_text = f"{title}\n{body}"
    title_company_mentions = sum(
        count_phrase_mentions(title, alias) for alias in aliases
    )
    company_mentions = sum(
        count_phrase_mentions(combined_text, alias) for alias in aliases
    )
    snippet = extract_target_snippet(ticker, title, body, aliases=aliases)
    scoring_text = f"{title}\n{snippet or body}"
    material_hits = count_keyword_hits(scoring_text, MATERIAL_KEYWORDS)
    tangential_hits = count_keyword_hits(scoring_text, TANGENTIAL_KEYWORDS)
    total_mentions = title_mentions + body_mentions + company_mentions

    if total_mentions == 0:
        return NewsRelevance(
            label="irrelevant",
            score=0.0,
            title_ticker_mentions=title_mentions,
            body_ticker_mentions=body_mentions,
            company_mentions=company_mentions,
            snippet="",
            reason="no_ticker_or_company_mentions",
        )

    score = 0.15 * title_mentions + 0.20 * body_mentions + 0.15 * company_mentions
    score += min(0.35, 0.12 * material_hits)
    score -= min(0.25, 0.10 * tangential_hits)
    score = max(0.0, min(1.0, score))

    if (
        tangential_hits
        and not (title_mentions or title_company_mentions)
        and body_mentions < 3
    ):
        label = "tangential"
        reason = "company_mention_in_tangential_context"
    elif body_mentions >= 3 or company_mentions >= 2:
        label = "relevant"
        reason = "repeated_target_mentions"
    elif body_mentions == 0 and tangential_hits:
        label = "tangential"
        reason = "company_mention_in_tangential_context"
    elif (
        (title_mentions or body_mentions or company_mentions)
        and material_hits
        and not tangential_hits
    ):
        label = "relevant"
        reason = "target_snippet_has_material_keyword"
    else:
        label = "tangential"
        reason = "mentions_without_material_target_context"

    return NewsRelevance(
        label=label,
        score=round(score, 3),
        title_ticker_mentions=title_mentions,
        body_ticker_mentions=body_mentions,
        company_mentions=company_mentions,
        snippet=snippet,
        reason=reason,
    )


def is_relevant_news(ticker: str, title: str, body: str = "") -> bool:
    return assess_news_relevance(ticker, title, body).label == "relevant"
