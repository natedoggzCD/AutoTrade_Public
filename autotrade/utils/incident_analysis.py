"""Incident analysis helpers for JSONL logs and overnight windows."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional
from zoneinfo import ZoneInfo

from autotrade.utils.safe_logging import get_safe_logger


LOGGER = get_safe_logger("incident_analysis")


def _parse_timestamp(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    value = raw.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
    return parsed


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError:
                continue


def filter_jsonl_window(
    log_path: Path | str,
    start_local: datetime,
    end_local: datetime,
    tz_name: str = "America/Chicago",
) -> List[Dict[str, Any]]:
    """
    Filter JSONL entries whose timestamps fall within a local-time window.
    Returns entries with `record` and localized `timestamp`.
    """
    path = Path(log_path)
    tz = ZoneInfo(tz_name)
    results: List[Dict[str, Any]] = []

    for record in _iter_jsonl(path):
        ts_raw = record.get("timestamp") or record.get("asctime")
        parsed = _parse_timestamp(str(ts_raw)) if ts_raw else None
        if parsed is None:
            continue
        local_ts = parsed.astimezone(tz)
        if start_local <= local_ts <= end_local:
            results.append({"record": record, "timestamp": local_ts})

    return results


def scan_logs_for_keywords(
    log_paths: Iterable[Path | str],
    start_local: datetime,
    end_local: datetime,
    keywords: Iterable[str],
    tz_name: str = "America/Chicago",
) -> List[Dict[str, Any]]:
    """
    Scan one or more JSONL logs for keyword matches in a local-time window.
    Returns flat entries with extracted metadata.
    """
    lowered = [k.lower() for k in keywords if k]
    matches: List[Dict[str, Any]] = []

    for log_path in log_paths:
        entries = filter_jsonl_window(
            log_path, start_local=start_local, end_local=end_local, tz_name=tz_name
        )
        for entry in entries:
            record = entry["record"]
            message = str(record.get("message", ""))
            if not lowered or any(key in message.lower() for key in lowered):
                matches.append(
                    {
                        "path": str(log_path),
                        "timestamp": entry["timestamp"],
                        "level": record.get("level"),
                        "logger": record.get("logger"),
                        "message": message,
                        "record": record,
                    }
                )

    return matches
