import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from autotrade.utils import incident_analysis


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_filter_jsonl_window_respects_local_time(tmp_path):
    log_path = tmp_path / "app.jsonl"
    rows = [
        {
            "timestamp": "2026-03-02T23:59:00Z",
            "level": "INFO",
            "logger": "test",
            "message": "before window",
        },
        {
            "timestamp": "2026-03-03T00:30:00Z",
            "level": "INFO",
            "logger": "test",
            "message": "in window",
        },
        {
            "timestamp": "2026-03-03T02:00:00Z",
            "level": "INFO",
            "logger": "test",
            "message": "after window",
        },
    ]
    _write_jsonl(log_path, rows)

    tz = ZoneInfo("America/Chicago")
    start_local = datetime(2026, 3, 2, 18, 0, tzinfo=tz)
    end_local = datetime(2026, 3, 2, 19, 0, tzinfo=tz)

    entries = incident_analysis.filter_jsonl_window(
        log_path, start_local=start_local, end_local=end_local, tz_name="America/Chicago"
    )

    assert len(entries) == 1
    assert entries[0]["record"]["message"] == "in window"


def test_scan_logs_for_keywords_filters_messages(tmp_path):
    log_path = tmp_path / "app.jsonl"
    rows = [
        {
            "timestamp": "2026-03-03T01:00:00Z",
            "level": "INFO",
            "logger": "scheduler",
            "message": "overnight research started",
        },
        {
            "timestamp": "2026-03-03T01:05:00Z",
            "level": "ERROR",
            "logger": "network",
            "message": "DNS resolution failed",
        },
        {
            "timestamp": "2026-03-03T01:10:00Z",
            "level": "INFO",
            "logger": "misc",
            "message": "heartbeat ok",
        },
    ]
    _write_jsonl(log_path, rows)

    tz = ZoneInfo("America/Chicago")
    start_local = datetime(2026, 3, 2, 19, 0, tzinfo=tz)
    end_local = datetime(2026, 3, 2, 20, 0, tzinfo=tz)

    entries = incident_analysis.scan_logs_for_keywords(
        [log_path],
        start_local=start_local,
        end_local=end_local,
        keywords=["dns", "overnight"],
        tz_name="America/Chicago",
    )

    messages = [entry["message"] for entry in entries]
    assert "overnight research started" in messages
    assert "DNS resolution failed" in messages
    assert "heartbeat ok" not in messages
