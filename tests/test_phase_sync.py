import json
from pathlib import Path

from autotrade.utils.phase_sync import (
    normalize_watchlist,
    record_decisions,
    update_phase_snapshot,
)


def test_normalize_watchlist_handles_mixed_inputs():
    items = [
        {"symbol": "aapl", "score": "5", "has_catalyst": True, "sector": "Tech"},
        "TSLA",
    ]

    normalized = normalize_watchlist(items, source="pm_report", phase="overnight")

    assert normalized[0]["symbol"] == "AAPL"
    assert normalized[0]["score"] == 5.0
    assert normalized[0]["has_catalyst"] is True
    assert normalized[1]["ticker"] == "TSLA"
    assert normalized[1]["score"] == 0.0
    assert all(item["phase"] == "overnight" for item in normalized)
    assert all(item["source"] == "pm_report" for item in normalized)


def test_update_phase_snapshot_merges_sections(tmp_path: Path):
    path = update_phase_snapshot(
        "2026-03-09",
        plans_dir=tmp_path,
        overnight={"full_watchlist": [{"symbol": "AAPL"}]},
    )
    data = json.loads(path.read_text())
    assert data["overnight"]["full_watchlist"][0]["symbol"] == "AAPL"

    path = update_phase_snapshot(
        "2026-03-09",
        plans_dir=tmp_path,
        premarket={"ranked_watchlist": [{"symbol": "TSLA"}]},
    )
    data = json.loads(path.read_text())
    assert "overnight" in data
    assert data["premarket"]["ranked_watchlist"][0]["symbol"] == "TSLA"


def test_record_decisions_appends_jsonl(tmp_path: Path):
    log_path = record_decisions(
        [{"symbol": "AAPL", "action": "selected", "score": 1.2}],
        phase="overnight",
        day="2026-03-09",
        log_dir=tmp_path,
    )

    assert log_path is not None
    assert log_path.exists()

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["symbol"] == "AAPL"
    assert entry["phase"] == "overnight"
