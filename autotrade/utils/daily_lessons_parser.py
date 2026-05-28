"""
Utility to parse daily lessons markdown reports into structured preferences.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_DIR / "reports"
LESSON_BIAS_EXCLUDED_SYMBOLS = {"SPY", "QQQ"}


@dataclass
class ParsedLessons:
    date: str
    executive_summary: str
    success_patterns: List[str]
    failure_patterns: List[str]
    missed_opportunities: List[Dict[str, Any]]
    actionable_lessons: List[str]
    prefer_symbols: List[str]
    avoid_symbols: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "executive_summary": self.executive_summary,
            "success_patterns": self.success_patterns,
            "failure_patterns": self.failure_patterns,
            "missed_opportunities": self.missed_opportunities,
            "actionable_lessons": self.actionable_lessons,
            "prefer_symbols": self.prefer_symbols,
            "avoid_symbols": self.avoid_symbols,
        }


class DailyLessonsParser:
    """
    Parse markdown daily lessons reports (produced by DailyLessonsAnalyzer) into
    a lightweight, machine-usable structure for prioritization.
    """

    def __init__(self, reports_dir: Path = REPORTS_DIR) -> None:
        self.reports_dir = Path(reports_dir)

    def parse_latest(self) -> Dict[str, Any]:
        """Parse the most recent daily_lessons_*.md file."""
        paths = sorted(self.reports_dir.glob("daily_lessons_*.md"))
        if not paths:
            logger.debug("No daily lessons files found in %s", self.reports_dir)
            return self._empty_payload()
        return self.parse_file(paths[-1])

    def parse_recent(self, n: int = 5) -> Dict[str, Any]:
        """
        Parse the most recent N daily lessons files and return an aggregated view.
        """
        paths = sorted(self.reports_dir.glob("daily_lessons_*.md"))[-int(max(1, n)) :]
        if not paths:
            return {
                "days": [],
                "prefer_symbols": [],
                "avoid_symbols": [],
                "mixed_symbols": [],
            }

        daily: List[Dict[str, Any]] = []
        prefer_all: List[str] = []
        avoid_all: List[str] = []
        for p in paths:
            parsed = self.parse_file(p)
            parsed["file"] = p.name
            daily.append(parsed)
            prefer_all.extend(parsed.get("prefer_symbols", []))
            avoid_all.extend(parsed.get("avoid_symbols", []))

        prefer_set = {s.upper() for s in prefer_all}
        avoid_set = {s.upper() for s in avoid_all}
        mixed = sorted(prefer_set & avoid_set)
        prefer_only = sorted(prefer_set - avoid_set)
        avoid_only = sorted(avoid_set - prefer_set)

        return {
            "days": daily,
            "prefer_symbols": prefer_only,
            "avoid_symbols": avoid_only,
            "mixed_symbols": mixed,
        }

    def parse_file(self, path: Path) -> Dict[str, Any]:
        text = Path(path).read_text(encoding="utf-8")
        return self.parse_text(text, source_path=path)

    def parse_text(self, text: str, source_path: Optional[Path] = None) -> Dict[str, Any]:
        date = self._extract_date(text, source_path)
        sections = self._split_sections(text)

        def _get_section(prefix: str) -> str:
            for key, val in sections.items():
                if key.startswith(prefix):
                    return val
            return ""

        exec_summary = _get_section("executive summary").strip()
        trades_section = _get_section("trades executed")
        missed_section = _get_section("missed opportunities")
        lessons_section = _get_section("actionable lessons")

        prefer_symbols: List[str] = []
        avoid_symbols: List[str] = []
        success_patterns: List[str] = []
        failure_patterns: List[str] = []

        # Trades: classify winners/losers
        for line in self._iter_lines(trades_section):
            symbol = self._extract_symbol(line)
            if not symbol:
                continue
            lower = line.lower()
            if any(token in lower for token in ["sold", "stop", "exit", "-"]):
                avoid_symbols.append(symbol)
                failure_patterns.append(line.strip())
            if any(token in lower for token in ["profit", "gain", "+"]):
                success_patterns.append(line.strip())

        missed_opportunities: List[Dict[str, Any]] = []
        for line in self._iter_lines(missed_section):
            symbol = self._extract_symbol(line)
            if not symbol:
                continue
            prefer_symbols.append(symbol)
            move = self._extract_move_percent(line)
            missed_opportunities.append({"symbol": symbol, "note": line.strip(), "move_pct": move})

        actionable_lessons = [
            line.strip(" .")
            for line in self._iter_lines(lessons_section, include_numbers=True)
            if line.strip()
        ]

        payload = ParsedLessons(
            date=date,
            executive_summary=exec_summary,
            success_patterns=_unique(success_patterns, force_upper=False),
            failure_patterns=_unique(failure_patterns, force_upper=False),
            missed_opportunities=missed_opportunities,
            actionable_lessons=actionable_lessons,
            prefer_symbols=_unique(prefer_symbols, force_upper=True),
            avoid_symbols=_unique(avoid_symbols, force_upper=True),
        )
        return payload.to_dict()

    # ---- helpers -----------------------------------------------------
    def _split_sections(self, text: str) -> Dict[str, str]:
        """
        Split markdown into sections keyed by lower-case heading text without numbering.
        """
        sections: Dict[str, str] = {}
        current = "executive summary"
        sections[current] = []
        for line in text.splitlines():
            heading = re.match(r"^##+\s+\d?\.?\s*(.+)$", line.strip(), flags=re.IGNORECASE)
            if heading:
                current = heading.group(1).strip().lower()
                sections.setdefault(current, [])
                continue
            sections.setdefault(current, [])
            sections[current].append(line)
        return {k: "\n".join(v).strip() for k, v in sections.items()}

    def _extract_date(self, text: str, source_path: Optional[Path]) -> str:
        match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
        if match:
            return match.group(1)
        if source_path:
            stem = Path(source_path).stem
            m2 = re.search(r"(\d{4}-\d{2}-\d{2})", stem)
            if m2:
                return m2.group(1)
        return ""

    def _iter_lines(self, section: str, include_numbers: bool = False) -> Sequence[str]:
        lines = []
        for raw in section.splitlines():
            if not raw.strip():
                continue
            if raw.lstrip().startswith("-") or raw.lstrip().startswith("*"):
                lines.append(raw.strip("-* ").strip())
            elif include_numbers and re.match(r"^\d+\.\s+", raw.lstrip()):
                cleaned = re.sub(r"^\d+\.\s+", "", raw.lstrip())
                lines.append(cleaned)
        return lines

    def _extract_symbol(self, line: str) -> Optional[str]:
        m = re.search(r"\*\*([A-Z]{1,6})\*\*", line)
        if m:
            return m.group(1)
        m = re.search(r"\b([A-Z]{1,6})\b", line)
        if m:
            return m.group(1)
        return None

    def _extract_move_percent(self, line: str) -> Optional[float]:
        m = re.search(r"([-+]?\d+\.\d+)%", line)
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    def _empty_payload(self) -> Dict[str, Any]:
        return ParsedLessons(
            date="",
            executive_summary="",
            success_patterns=[],
            failure_patterns=[],
            missed_opportunities=[],
            actionable_lessons=[],
            prefer_symbols=[],
            avoid_symbols=[],
        ).to_dict()


def apply_lesson_bias(
    candidates: List[Any],
    lessons: Dict[str, Any],
    prefer_boost: float = 3.0,
    avoid_penalty: float = 3.0,
) -> List[Any]:
    """
    Adjust candidate scores using parsed lessons.

    - Adds `prefer_boost` to symbols in lessons['prefer_symbols']
    - Subtracts `avoid_penalty` from symbols in lessons['avoid_symbols']
    - Annotates `metadata['daily_lessons_tag']`
    """

    prefer = {
        s.upper()
        for s in lessons.get("prefer_symbols", [])
        if s and s.upper() not in LESSON_BIAS_EXCLUDED_SYMBOLS
    }
    avoid = {
        s.upper()
        for s in lessons.get("avoid_symbols", [])
        if s and s.upper() not in LESSON_BIAS_EXCLUDED_SYMBOLS
    }

    for cand in candidates:
        sym = getattr(cand, "symbol", "") or ""
        sym_upper = sym.upper()
        tag: Optional[str] = None

        if sym_upper in prefer:
            cand.final_score = max(0.0, min(100.0, float(cand.final_score) + prefer_boost))
            tag = "prefer"
        if sym_upper in avoid:
            cand.final_score = max(0.0, min(100.0, float(cand.final_score) - avoid_penalty))
            tag = "avoid"

        if tag:
            cand.metadata = getattr(cand, "metadata", {}) or {}
            cand.metadata["daily_lessons_tag"] = tag

    candidates.sort(key=lambda c: float(getattr(c, "final_score", 0.0)), reverse=True)
    return candidates


def _unique(items: List[str], force_upper: bool = False) -> List[str]:
    seen = set()
    result = []
    for item in items:
        val = item.strip()
        if not val:
            continue
        key = val.upper() if force_upper else val
        if key in seen:
            continue
        seen.add(key)
        result.append(val.upper() if force_upper else val)
    return result
