import json
from pathlib import Path

from autotrade.utils.pnl_audit import compare_eod_sources


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compare_eod_sources_calculates_delta(tmp_path):
    data_dir = tmp_path
    review = {
        "date": "2026-03-03",
        "total_trades": 2,
        "avg_pnl": 5.0,
        "trades": [
            {"symbol": "AAA", "unrealized_pl": 8.5},
            {"symbol": "BBB", "unrealized_pl": -1.5},
        ],
    }
    feedback = {
        "date": "2026-03-03",
        "total_pnl": 12.0,
        "win_rate": 0.5,
    }

    _write_json(data_dir / "eod_review_2026-03-03.json", review)
    _write_json(data_dir / "eod_feedback_latest.json", feedback)

    result = compare_eod_sources("2026-03-03", data_dir=data_dir)

    assert result["status"] == "ok"
    assert result["eod_review_total_pnl"] == 7.0
    assert result["eod_feedback_total_pnl"] == 12.0
    assert result["delta"] == 5.0


def test_compare_eod_sources_handles_missing_files(tmp_path):
    result = compare_eod_sources("2026-03-03", data_dir=tmp_path)

    assert result["status"] == "missing"
    assert "eod_review" in result["missing"]
    assert "eod_feedback" in result["missing"]
