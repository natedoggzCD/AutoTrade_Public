from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from autotrade.utils.financial_db import FinancialDB


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class CatalystSnapshot:
    symbol: str
    has_catalyst: bool
    catalyst_score: float
    catalyst_tags: List[str]
    catalyst_note: str
    earnings_context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "has_catalyst": self.has_catalyst,
            "catalyst_score": self.catalyst_score,
            "catalyst_tags": list(self.catalyst_tags),
            "catalyst_note": self.catalyst_note,
            "earnings_context": dict(self.earnings_context),
        }


class CatalystTagBridge:
    """Attach catalyst metadata from headlines and earnings/event feeds."""

    _KEYWORD_TAGS = {
        "earnings": ("earnings", 30.0),
        "guidance": ("guidance", 18.0),
        "fda": ("fda", 25.0),
        "approval": ("approval", 18.0),
        "contract": ("contract", 12.0),
        "acquisition": ("mna", 18.0),
        "merger": ("mna", 18.0),
        "analyst": ("analyst", 10.0),
        "upgrade": ("analyst", 10.0),
        "downgrade": ("analyst", 10.0),
        "offering": ("offering", 12.0),
        "dilution": ("offering", 12.0),
    }

    def __init__(
        self,
        financial_db: Optional[FinancialDB] = None,
        financial_db_path: Optional[Path] = None,
    ) -> None:
        self.financial_db = financial_db or FinancialDB(db_path=financial_db_path)

    def extract(
        self,
        symbol: str,
        headlines: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        symbol_key = str(symbol or "").strip().upper()
        tags: List[str] = []
        note = ""
        score = 0.0

        for item in list(headlines or []):
            title = str(item.get("title") or item.get("headline") or "").strip()
            lowered = title.lower()
            for needle, (tag, weight) in self._KEYWORD_TAGS.items():
                if needle in lowered:
                    if tag not in tags:
                        tags.append(tag)
                    score += weight
                    if not note:
                        note = title[:120]

        earnings_context: Dict[str, Any] = {
            "upcoming": [],
            "recent_surprise": [],
        }
        if symbol_key:
            upcoming = self.financial_db.get_upcoming_earnings(days=7, tickers=[symbol_key])
            recent = [
                row
                for row in self.financial_db.get_recent_surprises(days=7, min_surprise=5.0)
                if str(row.get("ticker", "")).upper() == symbol_key
            ]
            if upcoming:
                earnings_context["upcoming"] = upcoming
                if "earnings" not in tags:
                    tags.append("earnings")
                score += 20.0
                if not note:
                    first = upcoming[0]
                    note = f"Earnings {first.get('earnings_date')} {first.get('time_of_day', 'TBD')}".strip()
            if recent:
                earnings_context["recent_surprise"] = recent
                if "earnings_surprise" not in tags:
                    tags.append("earnings_surprise")
                score += 25.0
                if not note:
                    first = recent[0]
                    note = (
                        f"Recent earnings surprise {float(first.get('surprise_pct') or 0.0):+.1f}%"
                    )

        score = _clamp(score, 0.0, 100.0)
        return CatalystSnapshot(
            symbol=symbol_key,
            has_catalyst=bool(tags),
            catalyst_score=round(score, 2),
            catalyst_tags=tags,
            catalyst_note=note,
            earnings_context=earnings_context,
        ).to_dict()
