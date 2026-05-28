import json
from datetime import datetime
from pathlib import Path

import autotrade.core.day_manager as day_manager_mod
from autotrade.core.day_manager import DayManager
from tools.entry_funnel_review import build_review


def _dm_stub() -> DayManager:
    dm = DayManager.__new__(DayManager)
    dm.signal_status = {}
    dm._active_entry_audit = None
    dm._last_entry_audit = {}
    return dm


def test_entry_audit_persists_entry_funnel_rows_with_phantom_placeholder(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(day_manager_mod, "LOG_DIR", tmp_path)
    dm = _dm_stub()
    dm._begin_entry_audit(
        candidates=[
            {
                "ticker": "PASS",
                "realtime_score": 81.0,
                "entry_source": "overnight_plan",
                "entry_price": 10.0,
                "stop_loss": 9.5,
                "target_price": 11.0,
                "quantity": 100,
            },
            {
                "ticker": "FAIL",
                "realtime_score": 55.0,
                "entry_source": "overnight_plan",
                "entry_price": 20.0,
                "stop_loss": 19.0,
                "target_price": 22.0,
                "quantity": 50,
            },
        ],
        phase=day_manager_mod.TradingPhase.CORE_TRADING,
        open_slots=2,
        max_new_entries=2,
    )
    dm._record_entry_audit_selected("PASS", submitted=True)
    dm._record_entry_audit_skip(
        "FAIL", "bad_day_posture_block:cautious_selective_longs"
    )

    dm._finalize_entry_audit(
        {
            "candidate_count": 2,
            "open_slots": 2,
            "max_new_entries": 2,
            "entries_submitted": 1,
            "selected_entry_symbols": ["PASS"],
            "phase": "core_trading",
        }
    )

    path = tmp_path / f"entry_funnel_{datetime.now().strftime('%Y-%m-%d')}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload[-1]["rows"]
    assert {row["symbol"] for row in rows} == {"PASS", "FAIL"}
    passed = next(row for row in rows if row["symbol"] == "PASS")
    failed = next(row for row in rows if row["symbol"] == "FAIL")
    assert passed["decision"] == "filled"
    assert passed["planned_entry_price"] == 10.0
    assert failed["decision"] == "skipped"
    assert failed["stage"] == "posture"
    assert failed["phantom_outcome"]["status"] == "pending_review"


def test_entry_funnel_review_aggregates_stage_phantom_pnl(tmp_path: Path):
    logs_dir = tmp_path / "logs"
    reports_dir = tmp_path / "reports"
    logs_dir.mkdir()
    (logs_dir / "entry_funnel_2026-05-19.json").write_text(
        json.dumps(
            [
                {
                    "timestamp": "2026-05-19T10:00:00",
                    "rows": [
                        {
                            "symbol": "A",
                            "decision": "skipped",
                            "stage": "vwap",
                            "phantom_outcome": {
                                "status": "closed",
                                "pnl_dollars": 50.0,
                            },
                        },
                        {
                            "symbol": "B",
                            "decision": "skipped",
                            "stage": "vwap",
                            "phantom_outcome": {"status": "pending_review"},
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    outputs = build_review(
        logs_dir=logs_dir,
        reports_dir=reports_dir,
        since=datetime(2026, 5, 19).date(),
        report_date=datetime(2026, 5, 19).date(),
    )

    markdown = outputs["markdown"].read_text(encoding="utf-8")
    csv_text = outputs["csv"].read_text(encoding="utf-8")
    assert "vwap" in markdown
    assert "50.00" in markdown
    assert "phantom_pending" in csv_text
