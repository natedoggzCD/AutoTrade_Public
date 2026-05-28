import pytest

from autotrade.signals.agentic_signal_generator import EntryCandidate
from autotrade.utils.daily_lessons_parser import (
    DailyLessonsParser,
    apply_lesson_bias,
)


SAMPLE_MARKDOWN = """# Daily Trading Lessons: 2026-02-25

## 1. Executive Summary
Today was a mixed day for the trading system with several exits and a few small wins.

## 2. Trades Executed & Risk Management
- **ASC**: Sold at the close after stop triggered (-4.2%).
- **CCK**: Held flat, small profit of +0.2%.

## 3. Missed Opportunities
- **CLDX**: +15.13% breakout not taken.
- **HSIC**: +1.65% gap-and-go ignored.

## 4. Actionable Lessons & System Improvements
1. Review conviction score thresholds for momentum breakouts.
2. Tighten trailing stops on lagging positions to cut losers earlier.
"""


def test_parser_extracts_symbols_and_lessons():
    parser = DailyLessonsParser()
    result = parser.parse_text(SAMPLE_MARKDOWN)

    assert result["date"] == "2026-02-25"
    assert any("mixed day" in result["executive_summary"].lower() for _ in [0])

    missed_symbols = {m["symbol"] for m in result["missed_opportunities"]}
    assert {"CLDX", "HSIC"} <= missed_symbols

    assert "ASC" in result["avoid_symbols"]
    assert {"CLDX", "HSIC"} <= set(result["prefer_symbols"])
    assert len(result["actionable_lessons"]) == 2


def test_apply_lesson_bias_boosts_prefer_and_penalizes_avoid():
    candidates = [
        EntryCandidate(symbol="CLDX", price=10.0, final_score=70.0),
        EntryCandidate(symbol="ASC", price=20.0, final_score=80.0),
    ]

    lessons = {"prefer_symbols": ["CLDX"], "avoid_symbols": ["ASC"]}
    boosted = apply_lesson_bias(
        candidates, lessons, prefer_boost=5.0, avoid_penalty=7.0
    )

    scores = {c.symbol: c.final_score for c in boosted}
    assert scores["CLDX"] == pytest.approx(75.0)
    assert scores["ASC"] == pytest.approx(73.0)

    metadata = {c.symbol: c.metadata.get("daily_lessons_tag") for c in boosted}
    assert metadata["CLDX"] == "prefer"
    assert metadata["ASC"] == "avoid"


def test_apply_lesson_bias_ignores_market_context_symbols():
    candidates = [
        EntryCandidate(symbol="SPY", price=400.0, final_score=70.0),
        EntryCandidate(symbol="QQQ", price=350.0, final_score=70.0),
        EntryCandidate(symbol="CLDX", price=10.0, final_score=70.0),
    ]

    boosted = apply_lesson_bias(
        candidates,
        {"prefer_symbols": ["SPY", "CLDX"], "avoid_symbols": ["QQQ"]},
        prefer_boost=5.0,
        avoid_penalty=7.0,
    )

    scores = {c.symbol: c.final_score for c in boosted}
    assert scores["SPY"] == 70.0
    assert scores["QQQ"] == 70.0
    assert scores["CLDX"] == 75.0


def test_parse_recent_merges_conflicts(tmp_path):
    parser = DailyLessonsParser(reports_dir=tmp_path)

    files = {
        "daily_lessons_2026-03-01.md": """
## 2. Trades Executed & Risk Management
- **AAA** stopped out -4.0%
## 3. Missed Opportunities
- **BBB** +6.5% breakout not taken
""",
        "daily_lessons_2026-03-02.md": """
## 2. Trades Executed & Risk Management
- **BBB** stopped out -3.1%
## 3. Missed Opportunities
- **CCC** +4.2% continuation
""",
        "daily_lessons_2026-03-03.md": """
## 2. Trades Executed & Risk Management
- **DDD** trimmed +1.0%
## 3. Missed Opportunities
- **AAA** +5.0% momentum
""",
    }

    for name, content in files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    agg = parser.parse_recent(n=5)

    assert set(agg["mixed_symbols"]) == {"AAA", "BBB"}
    assert set(agg["prefer_symbols"]) == {"CCC"}
    assert set(agg["avoid_symbols"]) == set()
