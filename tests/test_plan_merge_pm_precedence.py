"""H4 plan-merge with PM precedence (day_manager._merge_plan_signals_pm_precedence).

day-manager 2026-05-19 (H4): pins the merge behaviour that replaces the prior
"morning_is_richer" row-count heuristic. PM is the conviction set and wins on
per-symbol conflict; morning fills the tail.
"""

from __future__ import annotations

from autotrade.core.day_manager import DayManager


def _merge(pm, morning):
    # Bind the bound method without constructing a full DayManager: the helper
    # is pure (only uses module-level ``logger``).
    return DayManager._merge_plan_signals_pm_precedence(  # type: ignore[arg-type]
        None, pm, morning
    )


def test_pm_precedence_keeps_pm_rows_verbatim_on_conflict():
    pm = [
        {"symbol": "AAPL", "score": 85.0, "entry_price": 100.0},
        {"symbol": "TSLA", "score": 72.0, "entry_price": 250.0},
    ]
    morning = [
        {"symbol": "AAPL", "score": 40.0, "entry_price": 999.0},
        {"symbol": "NVDA", "score": 60.0, "entry_price": 500.0},
    ]
    merged = _merge(pm, morning)
    assert [r["symbol"] for r in merged] == ["AAPL", "TSLA", "NVDA"]
    aapl = next(r for r in merged if r["symbol"] == "AAPL")
    assert aapl["score"] == 85.0
    assert aapl["entry_price"] == 100.0
    assert aapl["plan_source"] == "pm"
    nvda = next(r for r in merged if r["symbol"] == "NVDA")
    assert nvda["plan_source"] == "morning_tail"


def test_disjoint_plans_concat_with_provenance():
    pm = [{"symbol": "A"}, {"symbol": "B"}]
    morning = [{"symbol": "C"}, {"symbol": "D"}]
    merged = _merge(pm, morning)
    assert len(merged) == 4
    assert [r["plan_source"] for r in merged] == [
        "pm",
        "pm",
        "morning_tail",
        "morning_tail",
    ]


def test_empty_morning_returns_pm_only():
    pm = [{"symbol": "A"}, {"symbol": "B"}]
    merged = _merge(pm, [])
    assert [r["symbol"] for r in merged] == ["A", "B"]
    assert all(r["plan_source"] == "pm" for r in merged)


def test_empty_pm_returns_morning_only():
    morning = [{"symbol": "X"}, {"symbol": "Y"}]
    merged = _merge([], morning)
    assert [r["symbol"] for r in merged] == ["X", "Y"]
    assert all(r["plan_source"] == "morning_tail" for r in merged)


def test_none_inputs_safe():
    assert _merge(None, None) == []
    assert _merge(None, [{"symbol": "X"}]) == [
        {"symbol": "X", "plan_source": "morning_tail"}
    ]


def test_ticker_field_recognized_when_symbol_missing():
    pm = [{"ticker": "PM1"}]
    morning = [{"ticker": "PM1"}, {"ticker": "M1"}]
    merged = _merge(pm, morning)
    assert [r.get("ticker") for r in merged] == ["PM1", "M1"]
    assert merged[0]["plan_source"] == "pm"
    assert merged[1]["plan_source"] == "morning_tail"


def test_case_insensitive_symbol_match():
    pm = [{"symbol": "aapl"}]
    morning = [{"symbol": "AAPL"}, {"symbol": "msft"}]
    merged = _merge(pm, morning)
    assert len(merged) == 2
    assert merged[0]["symbol"] == "aapl"
    assert merged[1]["symbol"] == "msft"


def test_rows_without_symbol_dropped():
    pm = [{"symbol": "A"}, {"score": 50.0}]
    morning = [{"ticker": ""}, {"symbol": "B"}]
    merged = _merge(pm, morning)
    assert [r["symbol"] if "symbol" in r else r.get("ticker") for r in merged] == [
        "A",
        "B",
    ]


def test_today_repro_scale():
    """Today (2026-05-19): PM=30 actionable, morning=393, overlap=29.

    Pre-fix: morning_is_richer demoted PM. Post-fix: 30 PM + 364 morning_tail.
    """
    pm = [{"symbol": f"PM{i:03d}"} for i in range(30)]
    overlap = [{"symbol": f"PM{i:03d}"} for i in range(29)]  # 29 shared
    morning_only = [{"symbol": f"MO{i:03d}"} for i in range(364)]
    morning = overlap + morning_only
    merged = _merge(pm, morning)
    assert len(merged) == 30 + 364
    pm_count = sum(1 for r in merged if r["plan_source"] == "pm")
    tail_count = sum(1 for r in merged if r["plan_source"] == "morning_tail")
    assert pm_count == 30
    assert tail_count == 364
