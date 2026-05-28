"""Tests for tools/missed_opportunities_report.py (H9)."""

import csv
import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.missed_opportunities_report import (
    MIN_SCORE,
    SLIPPAGE_BPS,
    _apply_phantom_slippage,
    _conviction_tier,
    build_report,
)


def _write_funnel(logs_dir: Path, target: date, cycles):
    path = logs_dir / f"entry_funnel_{target.isoformat()}.json"
    path.write_text(json.dumps(cycles), encoding="utf-8")


def _write_pm_plan(plans_dir: Path, target: date, signals):
    path = plans_dir / f"pm_plan_{target.isoformat()}.json"
    path.write_text(json.dumps({"signals": signals}), encoding="utf-8")


def _write_morning_plan(plans_dir: Path, target: date, signals):
    path = plans_dir / f"morning_game_plan_{target.strftime('%Y%m%d')}.json"
    path.write_text(json.dumps({"signals": signals}), encoding="utf-8")


@pytest.fixture()
def env(tmp_path):
    logs = tmp_path / "logs"
    plans = tmp_path / "plans"
    reports = tmp_path / "reports"
    for d in (logs, plans, reports):
        d.mkdir()
    return logs, plans, reports


def test_conviction_tier_buckets_match_directive():
    assert _conviction_tier(95) == "90+"
    assert _conviction_tier(85) == "80-89"
    assert _conviction_tier(75) == "70-79"
    assert _conviction_tier(65) == "60-69"
    assert _conviction_tier(55) == "<60"
    assert _conviction_tier(None) == "unknown"


def test_apply_phantom_slippage_marks_down_by_10bps():
    # qty=100, entry=$50 → notional $5000 → 10bps = $5 slippage
    result = _apply_phantom_slippage(100.0, qty=100.0, entry=50.0)
    assert result == 95.0
    # No notional info → flat % off pnl
    flat = _apply_phantom_slippage(100.0, qty=None, entry=None)
    assert flat == pytest.approx(100.0 * (1 - SLIPPAGE_BPS / 10000.0))
    assert _apply_phantom_slippage(None, 100.0, 50.0) is None


def test_score_below_min_is_excluded(env):
    logs, plans, reports = env
    target = date(2026, 5, 19)
    _write_funnel(
        logs,
        target,
        [
            {
                "timestamp": "2026-05-19T10:00:00",
                "phase": "core_trading",
                "rows": [
                    {
                        "symbol": "LOWSCORE",
                        "score": 50.0,
                        "decision": "skipped",
                        "stage": "preflight",
                        "reason": "candidate_not_submitted",
                    },
                    {
                        "symbol": "HIGHSCORE",
                        "score": 80.0,
                        "decision": "skipped",
                        "stage": "preflight",
                        "reason": "candidate_not_submitted",
                    },
                ],
            }
        ],
    )
    summary = build_report(
        target_date=target, logs_dir=logs, plans_dir=plans, reports_dir=reports
    )
    assert summary.total_missed == 1


def test_filled_rows_are_excluded(env):
    logs, plans, reports = env
    target = date(2026, 5, 19)
    _write_funnel(
        logs,
        target,
        [
            {
                "timestamp": "2026-05-19T10:00:00",
                "rows": [
                    {
                        "symbol": "FILLED",
                        "score": 90.0,
                        "decision": "filled",
                        "stage": "execution",
                        "reason": "ok",
                    },
                    {
                        "symbol": "SKIPPED",
                        "score": 90.0,
                        "decision": "skipped",
                        "stage": "posture",
                        "reason": "bearish_session",
                    },
                ],
            }
        ],
    )
    summary = build_report(target_date=target, logs_dir=logs, plans_dir=plans, reports_dir=reports)
    assert summary.total_missed == 1


def test_never_triggered_is_tagged_and_counted(env):
    logs, plans, reports = env
    target = date(2026, 5, 19)
    _write_funnel(
        logs,
        target,
        [
            {
                "timestamp": "2026-05-19T10:00:00",
                "rows": [
                    {
                        "symbol": "NEVER",
                        "score": 75.0,
                        "decision": "skipped",
                        "stage": "never_triggered",
                        "reason": "price_never_crossed_entry",
                        "phantom_outcome": {"status": "never_triggered"},
                    },
                ],
            }
        ],
    )
    summary = build_report(target_date=target, logs_dir=logs, plans_dir=plans, reports_dir=reports)
    assert summary.total_missed == 1
    assert summary.never_triggered_count == 1


def test_top_three_sorted_by_phantom_pnl_descending(env):
    logs, plans, reports = env
    target = date(2026, 5, 19)
    _write_funnel(
        logs,
        target,
        [
            {
                "timestamp": "2026-05-19T10:00:00",
                "rows": [
                    {
                        "symbol": f"SYM{i}",
                        "score": 70.0 + i,
                        "decision": "skipped",
                        "stage": "posture",
                        "reason": "gate",
                        "planned_size_shares": 100,
                        "planned_entry_price": 10.0,
                        "phantom_outcome": {"status": "filled", "pnl_dollars": float(pnl)},
                    }
                    for i, pnl in enumerate([50.0, 200.0, 10.0, 500.0, 1000.0])
                ],
            }
        ],
    )
    summary = build_report(target_date=target, logs_dir=logs, plans_dir=plans, reports_dir=reports)
    assert len(summary.top_three) == 3
    symbols = [r.symbol for r in summary.top_three]
    # SYM4 (1000), SYM3 (500), SYM1 (200), each marked down by 10bps * notional (1000) = $1
    assert symbols == ["SYM4", "SYM3", "SYM1"]
    assert summary.top_three[0].phantom_pnl_dollars == 999.0


def test_outputs_markdown_and_csv(env):
    logs, plans, reports = env
    target = date(2026, 5, 19)
    _write_funnel(
        logs,
        target,
        [
            {
                "timestamp": "2026-05-19T10:00:00",
                "rows": [
                    {
                        "symbol": "RBRK",
                        "score": 90.0,
                        "decision": "skipped",
                        "stage": "posture",
                        "reason": "bearish_session_gate",
                        "planned_entry_price": 50.0,
                        "planned_stop_price": 47.5,
                        "planned_target_price": 55.0,
                        "phantom_outcome": {"status": "filled", "pnl_dollars": 150.0, "pnl_pct": 0.03},
                    }
                ],
            }
        ],
    )
    summary = build_report(target_date=target, logs_dir=logs, plans_dir=plans, reports_dir=reports)
    assert summary.markdown_path.exists()
    assert summary.csv_path.exists()

    md = summary.markdown_path.read_text(encoding="utf-8")
    assert "RBRK" in md
    assert "posture" in md
    assert "Top 3" in md  # TL;DR present

    with summary.csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["symbol"] == "RBRK"
    assert rows[0]["stage"] == "posture"


def test_pm_plan_score_used_when_funnel_row_missing_score(env):
    logs, plans, reports = env
    target = date(2026, 5, 19)
    _write_pm_plan(plans, target, [{"symbol": "FROMPLAN", "confidence": 85.0}])
    _write_funnel(
        logs,
        target,
        [
            {
                "timestamp": "2026-05-19T10:00:00",
                "rows": [
                    {
                        "symbol": "FROMPLAN",
                        # no score in funnel row
                        "decision": "skipped",
                        "stage": "sizing",
                        "reason": "qty_zero",
                    },
                ],
            }
        ],
    )
    summary = build_report(target_date=target, logs_dir=logs, plans_dir=plans, reports_dir=reports)
    assert summary.total_missed == 1


def test_tldr_lines_handle_empty_report(env):
    logs, plans, reports = env
    target = date(2026, 5, 19)
    _write_funnel(logs, target, [])
    summary = build_report(target_date=target, logs_dir=logs, plans_dir=plans, reports_dir=reports)
    lines = summary.tldr_lines()
    assert lines and "0 candidates" in lines[0]


def test_grouping_by_stage_separates_rejection_kinds(env):
    logs, plans, reports = env
    target = date(2026, 5, 19)
    _write_funnel(
        logs,
        target,
        [
            {
                "timestamp": "2026-05-19T10:00:00",
                "rows": [
                    {"symbol": "A", "score": 80, "decision": "skipped", "stage": "posture", "reason": "x"},
                    {"symbol": "B", "score": 80, "decision": "skipped", "stage": "sizing", "reason": "y"},
                    {"symbol": "C", "score": 80, "decision": "skipped", "stage": "posture", "reason": "x"},
                ],
            }
        ],
    )
    summary = build_report(target_date=target, logs_dir=logs, plans_dir=plans, reports_dir=reports)
    assert set(summary.by_stage_phantom_pnl.keys()) == {"posture", "sizing"}


def test_constants_match_directive():
    assert MIN_SCORE == 60.0
    assert SLIPPAGE_BPS == 10.0
