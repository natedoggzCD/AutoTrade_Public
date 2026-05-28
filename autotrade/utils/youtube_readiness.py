"""
YouTube Readiness Checker — Intelligence Pipeline Status & Auto-Scan
====================================================================

Central module for the trading agent to verify YouTube intelligence is current
and trigger scanning when it's not. Used by:
  - pm_workflow.py      (PM phase: scan channels, generate report)
  - autonomous_agent.py (overnight: load regime for screening adjustments)
  - day_manager.py      (intraday: regime-aware conviction, sizing)
  - workflow_manager.py  (phase transitions: ensure freshness)

Usage:
    from autotrade.utils.youtube_readiness import (
        check_readiness, ensure_youtube_ready, get_intelligence_context
    )

    # Quick status check
    status = check_readiness()
    if not status["report_ready"]:
        ensure_youtube_ready()  # auto-scans and generates report

    # Get intelligence context for trading decisions
    ctx = get_intelligence_context()
    regime = ctx["regime"]
    sizing = ctx["sizing_multiplier"]
"""

import concurrent.futures
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from autotrade.utils.safe_logging import safe_exception_text, setup_safe_logging_for_module


class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except Exception:
            return str(obj)

logger = logging.getLogger(__name__)
setup_safe_logging_for_module(logger)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RAG_DIR = PROJECT_ROOT / "data" / "youtube" / "rag"
TRACK_FILE = PROJECT_ROOT / "data" / "youtube" / "processed_videos.json"


# ── Status Checking ──────────────────────────────────────────────


def _next_trading_date() -> str:
    """Get the next trading date (today if weekday and after 4PM, else next weekday)."""
    now = datetime.now()
    # If it's a weekday evening (PM workflow time), next trading day is tomorrow
    # If it's weekend, next trading day is Monday
    if now.weekday() >= 5:  # Sat/Sun
        days_ahead = 7 - now.weekday()  # Mon
        return (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    if now.hour >= 16:  # After market close
        next_day = now + timedelta(days=1)
        # Skip weekend
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        return next_day.strftime("%Y-%m-%d")
    return now.strftime("%Y-%m-%d")


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _session_dates_for_readiness(target_date: str, today_date: str) -> List[str]:
    """
    Build the list of dates that should count toward the current session.

    Weekend rule: when the next trading day is Monday, include Friday/Saturday/Sunday
    so weekend content is not dropped from readiness decisions.
    """
    dates: List[str] = []

    def _add(d: str) -> None:
        if d and d not in dates:
            dates.append(d)

    _add(target_date)
    _add(today_date)
    _add((datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"))

    try:
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        if target_dt.weekday() == 0:  # Monday
            for offset in (3, 2, 1):  # Fri/Sat/Sun
                _add((target_dt - timedelta(days=offset)).strftime("%Y-%m-%d"))
    except Exception:
        pass

    return dates


def get_processed_videos() -> Dict:
    """Load the processed videos tracking file."""
    if TRACK_FILE.exists():
        try:
            with open(TRACK_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"processed": {}}


def get_channel_config() -> Dict:
    """Load youtube_channels.yaml."""
    config_path = PROJECT_ROOT / "config" / "youtube_channels.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def _detect_pending_videos(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lightweight metadata check to detect new videos not yet processed.

    Returns:
        {
            "checked": bool,
            "pending_count": int,
            "pending_channels": List[str],
            "error": Optional[str],
        }
    """
    try:
        from tools.youtube_daily_scanner import (
            get_recent_videos,
            filter_new_videos,
        )
    except Exception as e:
        return {
            "checked": False,
            "pending_count": 0,
            "pending_channels": [],
            "error": f"scanner_import_failed: {safe_exception_text(e)}",
        }

    channels = config.get("channels", {}) or {}
    if not channels:
        return {"checked": True, "pending_count": 0, "pending_channels": []}

    scanner_config = config.get("scanner", {}) or {}
    rag_dir = Path(scanner_config.get("rag_dir", "data/youtube/rag"))
    perf_cfg = scanner_config.get("performance", {}) or {}
    max_channel_workers = max(1, int(perf_cfg.get("max_channel_workers", 4)))

    processed = get_processed_videos()

    def _fetch(
        item: Tuple[str, Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
        channel_key, channel_cfg = item
        channel_url = channel_cfg["channel_url"]
        max_age = channel_cfg.get("schedule", {}).get("max_age_hours", 36)
        videos = get_recent_videos(channel_url, max_age_hours=max_age)
        return channel_key, channel_cfg, videos

    pending_total = 0
    pending_channels: List[str] = []
    channel_items = list(channels.items())
    fetched: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]] = []

    if len(channel_items) == 1 or max_channel_workers == 1:
        fetched = [_fetch(item) for item in channel_items]
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_channel_workers
        ) as pool:
            futures = [pool.submit(_fetch, item) for item in channel_items]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    fetched.append(fut.result())
                except Exception as e:
                    logger.debug(
                        "[YOUTUBE] Pending metadata fetch skipped: %s",
                        safe_exception_text(e),
                    )

    fetched.sort(key=lambda row: row[1].get("priority", 99))
    for channel_key, channel_cfg, videos in fetched:
        new = filter_new_videos(videos, processed, channel_key, rag_dir=rag_dir)
        if new:
            pending_total += len(new)
            pending_channels.append(channel_key)

    return {
        "checked": True,
        "pending_count": int(pending_total),
        "pending_channels": sorted(set(pending_channels)),
    }


def _get_summary_override_args() -> List[str]:
    """
    Resolve scanner CLI overrides for fast report synthesis.

    Defaults come from `config/youtube_channels.yaml` so readiness flows use the
    currently configured synthesis provider/model.
    """
    config = get_channel_config()
    ext_cfg = (config.get("scanner", {}) or {}).get("extraction", {}) or {}
    provider = (
        str(ext_cfg.get("synthesis_provider", "openrouter")).strip() or "openrouter"
    )
    model = (
        str(ext_cfg.get("report_model", "gpt-4.1")).strip()
        or "gpt-4.1"
    )
    return ["--synthesis-provider", provider, "--report-model", model]


def _resolve_output_dir(config: Dict[str, Any]) -> Path:
    """Resolve scanner output directory from config."""
    scanner_cfg = config.get("scanner", {}) if isinstance(config, dict) else {}
    raw = str(scanner_cfg.get("output_dir", "data/youtube")).strip() or "data/youtube"
    out = Path(raw)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    return out


def _count_transcript_artifacts(output_dir: Path, dates: List[str]) -> int:
    """
    Count transcript `.txt` artifacts for the current session date set.

    Uses file mtime date as a best-effort fallback when extraction artifacts are missing.
    """
    if not output_dir.exists():
        return 0
    date_set = set(dates)
    count = 0
    for txt_file in output_dir.glob("*.txt"):
        try:
            mdate = datetime.fromtimestamp(txt_file.stat().st_mtime).strftime(
                "%Y-%m-%d"
            )
        except Exception:
            continue
        if mdate in date_set:
            count += 1
    return count


def _parse_channel_key_from_stem(stem: str, known_channels: List[str]) -> str:
    """Parse channel key from extraction filename stem, tolerant to underscores in video IDs."""
    for ck in sorted([str(c) for c in known_channels if c], key=len, reverse=True):
        prefix = f"{ck}_"
        if stem.startswith(prefix):
            return ck
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def _load_json_payload(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _is_usable_extraction_payload(payload: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(payload, dict):
        return False
    meta = payload.get("_meta", {}) if isinstance(payload.get("_meta"), dict) else {}
    if bool(meta.get("extraction_failed")):
        return False
    raw_extraction = str(payload.get("raw_extraction", "") or "").strip()
    if raw_extraction:
        return True
    ignored_keys = {"_meta", "_source", "transcript_context", "raw_extraction"}
    return any(key not in ignored_keys for key in payload.keys())


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    return _safe_int(raw, default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}


def _safe_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except Exception:
        return 0.0


def _resolve_watchdog_settings(
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config or get_channel_config()
    readiness_cfg = (cfg.get("scanner", {}) or {}).get("readiness", {}) or {}

    scan_stall = _safe_int(readiness_cfg.get("scan_stall_timeout_seconds", 7200), 7200)
    scan_hard = _safe_int(readiness_cfg.get("scan_hard_timeout_seconds", 0), 0)
    report_stall = _safe_int(
        readiness_cfg.get("report_stall_timeout_seconds", 3600), 3600
    )
    report_hard = _safe_int(readiness_cfg.get("report_hard_timeout_seconds", 0), 0)
    poll_seconds = _safe_int(readiness_cfg.get("progress_poll_seconds", 5), 5)
    verbose_logs = bool(readiness_cfg.get("verbose_subprocess_logs", False))

    scan_stall = max(300, _env_int("YOUTUBE_SCAN_STALL_TIMEOUT_SECONDS", scan_stall))
    scan_hard = max(0, _env_int("YOUTUBE_SCAN_HARD_TIMEOUT_SECONDS", scan_hard))
    report_stall = max(
        300, _env_int("YOUTUBE_REPORT_STALL_TIMEOUT_SECONDS", report_stall)
    )
    report_hard = max(0, _env_int("YOUTUBE_REPORT_HARD_TIMEOUT_SECONDS", report_hard))
    poll_seconds = max(1, _env_int("YOUTUBE_PROGRESS_POLL_SECONDS", poll_seconds))
    verbose_logs = _env_bool("YOUTUBE_VERBOSE_WATCHDOG_LOGS", verbose_logs)

    return {
        "scan_stall_timeout_seconds": scan_stall,
        "scan_hard_timeout_seconds": scan_hard,
        "report_stall_timeout_seconds": report_stall,
        "report_hard_timeout_seconds": report_hard,
        "progress_poll_seconds": poll_seconds,
        "verbose_subprocess_logs": bool(verbose_logs),
    }


def _load_session_scan_status(target_date: str, today_date: str) -> Dict[str, Any]:
    """
    Load the most relevant session scan status artifact (if present).
    """
    candidates: List[str] = []
    for d in (
        target_date,
        today_date,
        (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
    ):
        if d not in candidates:
            candidates.append(d)

    for d in candidates:
        status_file = RAG_DIR / f"session_scan_status_{d}.json"
        if not status_file.exists():
            continue
        try:
            payload = json.loads(status_file.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload["_artifact_file"] = str(status_file)
                return payload
        except Exception:
            continue
    return {}


def _scan_progress_snapshot(
    output_dir: Path, today: str, target: str
) -> Tuple[float, int, float, int, float, float]:
    today_dir = RAG_DIR / "by_date" / today
    target_dir = RAG_DIR / "by_date" / target

    today_count = len(list(today_dir.glob("*.json"))) if today_dir.exists() else 0
    target_count = len(list(target_dir.glob("*.json"))) if target_dir.exists() else 0

    return (
        _safe_mtime(TRACK_FILE),
        today_count,
        _safe_mtime(today_dir),
        target_count,
        _safe_mtime(target_dir),
        _safe_mtime(output_dir),
    )


def _report_progress_snapshot(
    today: str, target: str
) -> Tuple[float, float, float, float, float]:
    report_dir = RAG_DIR / "daily_reports"
    today_report = report_dir / f"{today}_consolidated.json"
    target_report = report_dir / f"{target}_consolidated.json"
    today_weekend = report_dir / f"{today}_weekend_consolidated.json"
    target_weekend = report_dir / f"{target}_weekend_consolidated.json"
    return (
        _safe_mtime(report_dir),
        _safe_mtime(today_report),
        _safe_mtime(target_report),
        _safe_mtime(today_weekend),
        _safe_mtime(target_weekend),
    )


def _terminate_process(proc: subprocess.Popen, label: str) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=15)
    except Exception:
        logger.warning(f"{label} did not terminate gracefully; killing process")
        try:
            proc.kill()
            proc.wait(timeout=10)
        except Exception:
            pass


def _run_subprocess_with_watchdog(
    cmd: List[str],
    *,
    cwd: Path,
    stall_timeout: Optional[int],
    hard_timeout: Optional[int],
    progress_poll_seconds: int,
    progress_snapshot: Optional[Callable[[], Tuple[Any, ...]]] = None,
    label: str = "[YOUTUBE]",
    verbose_logs: bool = False,
) -> Dict[str, Any]:
    started_at = time.time()
    stall_timeout_s = max(0, int(stall_timeout or 0))
    hard_timeout_s = max(0, int(hard_timeout or 0))
    poll_s = max(1, int(progress_poll_seconds or 5))

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )

    line_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
    stdout_tail: deque[str] = deque(maxlen=2000)
    stderr_tail: deque[str] = deque(maxlen=2000)
    stdout_lines = 0
    stderr_lines = 0
    terminated_reason = None
    last_progress_at = started_at
    last_log_at = started_at
    last_snapshot = None
    if progress_snapshot is not None:
        try:
            last_snapshot = progress_snapshot()
        except Exception:
            last_snapshot = None

    def _reader(name: str, stream: Optional[Any]) -> None:
        if stream is None:
            return
        try:
            for raw in iter(stream.readline, ""):
                if raw == "":
                    break
                line_queue.put((name, raw.rstrip("\r\n")))
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _is_progress_line(line: str) -> bool:
        low = (line or "").lower()
        markers = (
            "scanning:",
            "found ",
            "new videos",
            "transcribing:",
            "extracting intelligence",
            "stored extraction",
            "report saved",
            "scan complete",
            "report-only mode",
            "generating daily report",
            "error",
            "warning",
        )
        return any(m in low for m in markers)

    def _normalize_progress_line(line: str) -> Optional[str]:
        if not line:
            return None
        if verbose_logs:
            return line
        low = line.lower()
        if "raw_report" in low:
            return None
        if len(line) > 280:
            if "error" in low or "warning" in low:
                return f"{line[:280]}... [truncated]"
            return None
        return line

    threads = [
        threading.Thread(target=_reader, args=("stdout", proc.stdout), daemon=True),
        threading.Thread(target=_reader, args=("stderr", proc.stderr), daemon=True),
    ]
    for t in threads:
        t.start()

    def _drain_lines() -> None:
        nonlocal stdout_lines, stderr_lines, last_progress_at
        while True:
            try:
                stream_name, line = line_queue.get_nowait()
            except queue.Empty:
                break
            if stream_name == "stdout":
                stdout_lines += 1
                if line:
                    stdout_tail.append(line)
            else:
                stderr_lines += 1
                if line:
                    stderr_tail.append(line)
            if line and _is_progress_line(line):
                compact = _normalize_progress_line(line)
                if compact:
                    logger.info(f"{label} {compact}")
            last_progress_at = time.time()

    while True:
        _drain_lines()

        now = time.time()
        if progress_snapshot is not None:
            try:
                snap = progress_snapshot()
            except Exception:
                snap = None
            if snap is not None and snap != last_snapshot:
                last_snapshot = snap
                last_progress_at = now
                logger.info(f"{label} Progress heartbeat: artifacts updated")

        elapsed = now - started_at
        idle = now - last_progress_at
        if now - last_log_at >= 60:
            logger.info(
                f"{label} Running {elapsed:.1f}s (idle {idle:.1f}s, "
                f"stall_timeout={stall_timeout_s}s, hard_timeout={hard_timeout_s}s)"
            )
            last_log_at = now

        if hard_timeout_s > 0 and elapsed > hard_timeout_s:
            terminated_reason = "hard_timeout"
            logger.warning(
                f"{label} Hard timeout reached after {elapsed:.1f}s "
                f"(limit={hard_timeout_s}s) - terminating"
            )
            _terminate_process(proc, label)
            break

        if stall_timeout_s > 0 and idle > stall_timeout_s:
            terminated_reason = "stall_timeout"
            logger.warning(
                f"{label} No progress detected for {idle:.1f}s "
                f"(stall timeout={stall_timeout_s}s) - terminating"
            )
            _terminate_process(proc, label)
            break

        if proc.poll() is not None:
            break
        time.sleep(poll_s)

    for t in threads:
        t.join(timeout=1.0)
    _drain_lines()

    try:
        return_code = proc.wait(timeout=1.0)
    except Exception:
        return_code = proc.poll()

    return {
        "returncode": int(return_code if return_code is not None else -1),
        "elapsed_seconds": max(time.time() - started_at, 0.0),
        "stdout_lines": stdout_lines,
        "stderr_lines": stderr_lines,
        "stdout_tail": list(stdout_tail),
        "stderr_tail": list(stderr_tail),
        "terminated_reason": terminated_reason,
    }


def check_readiness(target_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Check if YouTube intelligence is ready for the next trading session.

    Returns a status dict:
        {
            "report_ready": bool,       # consolidated report exists
            "report_date": str,         # date of most recent report
            "report_stale": bool,       # report is >24h old
            "channels_scraped": [...],  # channels with extractions today
            "channels_missing": [...],  # channels with no recent extraction
            "channels_checked": [...],  # channels metadata-scanned this session
            "channels_unchecked": [...],  # channels not metadata-scanned this session
            "videos_processed_today": int,
            "videos_available": int,    # unprocessed videos detected
            "needs_scan": bool,         # should we trigger a scan?
            "needs_report": bool,       # should we regenerate report?
        }
    """
    today = _today_str()
    target = target_date or _next_trading_date()
    report_dir = RAG_DIR / "daily_reports"
    session_dates = _session_dates_for_readiness(target_date=target, today_date=today)

    config = get_channel_config()
    output_dir = _resolve_output_dir(config)
    known_channels = list((config.get("channels", {}) or {}).keys())
    all_channels = set(known_channels)
    readiness_cfg = (config.get("scanner", {}) or {}).get("readiness", {}) or {}
    always_check_new = _env_bool(
        "YOUTUBE_ALWAYS_CHECK_NEW",
        bool(readiness_cfg.get("always_check_new_videos", True)),
    )

    # Check for a CURRENT consolidated report first. Older fallback reports are
    # tracked separately and must not mark the current session as ready.
    report_ready = False
    report_current = False
    report_date = None
    report_stale = True
    fallback_report_available = False
    fallback_report_date = None

    report_dates: List[str] = []
    for d in [target] + [
        (datetime.now() - timedelta(days=offset)).strftime("%Y-%m-%d")
        for offset in range(4)
    ]:
        if d not in report_dates:
            report_dates.append(d)

    for check_date in report_dates:
        report_file = report_dir / f"{check_date}_consolidated.json"
        weekend_file = report_dir / f"{check_date}_weekend_consolidated.json"

        for f in [weekend_file, report_file]:
            if f.exists():
                report_date = check_date
                age_hours = (
                    datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)
                ).total_seconds() / 3600
                report_stale = age_hours > 24
                if check_date == target:
                    report_ready = True
                    report_current = True
                else:
                    fallback_report_available = True
                    fallback_report_date = check_date
                break
        if report_current:
            break

    # Check which channels have extractions.
    # We only count artifacts from TODAY'S actual date for the 'processed_today' metric
    # to avoid falsely skipping scans based on yesterday's material.
    # CRITICAL: We also check file system creation/modification time to ensure 
    # a file in the session folder isn't actually from last night.
    session_date_set = set(session_dates)
    channels_scraped_session = set()
    channels_scraped_today = set()
    channels_failed_extraction = set()
    usable_videos_processed_today = 0
    videos_in_session = 0
    
    # 1. Count actual TODAY'S artifacts (Calendar Day)
    # Even if they are stored in tomorrow's session folder, we care if they were downloaded TODAY.
    for check_date in session_dates:
        check_dir = RAG_DIR / "by_date" / check_date
        if not check_dir.exists():
            continue
            
        for f in check_dir.glob("*.json"):
            channel_key = _parse_channel_key_from_stem(f.stem, known_channels=known_channels)
            payload = _load_json_payload(f)
            usable_payload = _is_usable_extraction_payload(payload)
            
            # Check file timestamp
            try:
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                is_fresh = mtime.date() == datetime.now().date()
            except Exception:
                is_fresh = (check_date == today) # Fallback to folder name
            
            if check_date in session_date_set and usable_payload:
                channels_scraped_session.add(channel_key)
                videos_in_session += 1
                
            if is_fresh and usable_payload:
                channels_scraped_today.add(channel_key)
                usable_videos_processed_today += 1
            elif is_fresh and not usable_payload:
                channels_failed_extraction.add(channel_key)

    # 2. Collect other recent channels (for UI/context only)
    channels_scraped_recent = set()
    for check_date in session_dates:
        check_dir = RAG_DIR / "by_date" / check_date
        if check_dir.exists():
            for f in check_dir.glob("*.json"):
                payload = _load_json_payload(f)
                if not _is_usable_extraction_payload(payload):
                    continue
                channel_key = _parse_channel_key_from_stem(f.stem, known_channels=known_channels)
                channels_scraped_recent.add(channel_key)

    scan_status = _load_session_scan_status(target_date=target, today_date=today)
    channels_checked_session = {
        str(c)
        for c in (
            scan_status.get("channels_checked", [])
            if isinstance(scan_status, dict)
            else []
        )
        if str(c) in all_channels
    }
    # Fallback for older scanner runs without session status artifact.
    if not channels_checked_session:
        channels_checked_session = set(channels_scraped_session) | set(channels_scraped_today)

    channels_unchecked = all_channels - channels_checked_session
    channels_missing_content = all_channels - channels_scraped_today
    channels_missing = set(channels_missing_content)

    # Check processed videos tracking for unprocessed count
    readiness_cfg = (config.get("scanner", {}) or {}).get("readiness", {}) or {}
    require_all_channels = bool(readiness_cfg.get("require_all_channels", False))
    min_videos_to_skip_scan = max(
        1, int(readiness_cfg.get("min_videos_to_skip_scan", 4))
    )
    periodic_recheck_enabled = bool(
        readiness_cfg.get("periodic_recheck_enabled", False)
    )
    scan_for_missing_content_cfg = readiness_cfg.get("scan_for_missing_content", None)
    if scan_for_missing_content_cfg is None:
        scan_for_missing_content = True
    else:
        scan_for_missing_content = bool(scan_for_missing_content_cfg)
    if require_all_channels:
        scan_for_missing_content = True

    # Transcript artifacts for TODAY ONLY.
    transcript_artifacts = _count_transcript_artifacts(
        output_dir=output_dir, dates=[today, target]
    )
    videos_processed_today = usable_videos_processed_today

    # Check scan cooldown — don't re-scan if we scanned within the last N minutes
    scan_cooldown_minutes = max(5, int(readiness_cfg.get("scan_cooldown_minutes", 30)))
    scan_marker = RAG_DIR / ".last_scan_at"
    minutes_since_last_scan = None
    scan_on_cooldown = False
    if scan_marker.exists():
        try:
            age_sec = (
                datetime.now() - datetime.fromtimestamp(scan_marker.stat().st_mtime)
            ).total_seconds()
            minutes_since_last_scan = age_sec / 60
            scan_on_cooldown = minutes_since_last_scan < scan_cooldown_minutes
        except Exception:
            pass

    # Determine if we need to scan.
    # New policy: a session is only considered to have material if the threshold
    # is met AND no channels are missing fresh content for the calendar day.
    session_has_material = (
        usable_videos_processed_today >= min_videos_to_skip_scan
    ) and (not channels_missing_content)

    # Scan reasons (any one is sufficient)
    # Periodic recheck is important because "channels scraped today" does not
    # guarantee there are no *new* uploads since the last scan.
    scan_for_periodic_recheck = periodic_recheck_enabled and (not scan_on_cooldown)
    scan_for_unchecked_channels = bool(channels_unchecked)
    scan_for_missing_channels = scan_for_missing_content and bool(
        channels_missing_content
    )
    scan_for_stale_report = bool(report_stale) and not session_has_material
    scan_for_no_report = not report_ready

    needs_scan_reason_str = ""
    if not session_has_material:
        needs_scan_reason_str = (
            "missing_fresh_content "
            f"(usable={usable_videos_processed_today} < min={min_videos_to_skip_scan} or channels_missing)"
        )
    elif scan_for_missing_channels:
        needs_scan_reason_str = f"missing_channels={list(channels_missing_content)}"
    elif scan_for_unchecked_channels:
        needs_scan_reason_str = f"unchecked_channels={list(channels_unchecked)}"
    elif scan_for_stale_report:
        needs_scan_reason_str = "stale_report"
    elif scan_for_no_report:
        needs_scan_reason_str = "no_report"
    elif scan_for_periodic_recheck:
        needs_scan_reason_str = "periodic_recheck"

    # Check for pending videos using tools/youtube_daily_scanner.py logic
    pending_check = {"checked": False, "pending_count": 0, "pending_channels": []}
    if always_check_new:
        pending_check = _detect_pending_videos(config)
        if pending_check.get("pending_count", 0) > 0:
            needs_scan_reason_str = f"pending_videos={pending_check['pending_count']}"

    # Apply cooldown: suppress the scan ONLY when cooldown is active AND
    # (a) report exists and is not stale, AND
    # (b) we already have enough session material (>= min_videos_to_skip_scan).
    # Missing channels with insufficient material should ALWAYS trigger a scan
    # so all 4 channels get covered, not just the first one with a video.
    if needs_scan_reason_str and scan_on_cooldown:
        # Never suppress when report is missing/stale
        if scan_for_stale_report or scan_for_no_report:
            needs_scan = True
        # Never suppress unchecked channels until we have enough material.
        elif scan_for_unchecked_channels and not session_has_material:
            needs_scan = True
        # Optional strict mode: missing channel content can force scan too.
        elif scan_for_missing_channels and not session_has_material:
            needs_scan = True
        else:
            # Only periodic recheck or post-material missing channels — respect cooldown
            needs_scan = False
    else:
        needs_scan = bool(needs_scan_reason_str)

    # If session material exists and report is stale, regenerate report without forcing scan.
    needs_report = (not report_current) or (bool(report_stale) and session_has_material)

    channels_total = len(all_channels)
    channels_processed = len(channels_scraped_today)
    coverage_pct = (
        round(float(channels_processed) / float(channels_total), 4)
        if channels_total > 0
        else 0.0
    )
    if coverage_pct >= 0.75:
        coverage_grade = "complete"
    elif coverage_pct >= 0.5:
        coverage_grade = "partial"
    else:
        coverage_grade = "minimal"

    status = {
        "report_ready": report_ready,
        "report_current": report_current,
        "report_date": report_date,
        "report_stale": report_stale,
        "fallback_report_available": fallback_report_available,
        "fallback_report_date": fallback_report_date,
        "channels_scraped": sorted(channels_scraped_today),
        "channels_scraped_session": sorted(channels_scraped_session),
        "channels_checked": sorted(channels_checked_session),
        "channels_unchecked": sorted(channels_unchecked),
        "channels_missing_content": sorted(channels_missing_content),
        "channels_missing": sorted(channels_missing),
        "channels_total": channels_total,
        "channels_processed": channels_processed,
        "coverage_pct": coverage_pct,
        "coverage_grade": coverage_grade,
        "videos_processed_today": videos_processed_today,
        "usable_videos_processed_today": usable_videos_processed_today,
        "videos_in_session": videos_in_session,
        "transcript_artifacts": transcript_artifacts,
        "channels_failed_extraction": sorted(channels_failed_extraction),
        "videos_available": int(pending_check.get("pending_count", 0)),
        "needs_scan": needs_scan,
        "needs_scan_reason": needs_scan_reason_str,
        "needs_report": needs_report,
        "target_date": target,
        "scan_policy": {
            "require_all_channels": require_all_channels,
            "min_videos_to_skip_scan": min_videos_to_skip_scan,
            "session_has_material": session_has_material,
            "scan_on_cooldown": scan_on_cooldown,
            "minutes_since_last_scan": round(minutes_since_last_scan, 1)
            if minutes_since_last_scan is not None
            else None,
            "scan_cooldown_minutes": scan_cooldown_minutes,
            "scan_for_periodic_recheck": scan_for_periodic_recheck,
            "scan_for_unchecked_channels": scan_for_unchecked_channels,
            "scan_for_missing_content": scan_for_missing_channels,
            "periodic_recheck_enabled": periodic_recheck_enabled,
            "channels_scraped_recent": sorted(channels_scraped_recent),
            "scan_status_artifact": scan_status.get("_artifact_file")
            if isinstance(scan_status, dict)
            else None,
            "always_check_new_videos": always_check_new,
            "pending_videos": int(pending_check.get("pending_count", 0)),
            "pending_channels": pending_check.get("pending_channels", []),
            "pending_check_error": pending_check.get("error"),
        },
    }

    return status


def ensure_youtube_ready(
    force_scan: bool = False,
    force_report: bool = False,
    timeout: Optional[int] = 1800,
) -> Dict[str, Any]:
    """
    Ensure YouTube intelligence is ready for the next trading session.

    This is the main function called by PM workflow, overnight agent, etc.
    It checks readiness, scans if needed, and generates the report.

    Returns:
        Status dict with scan results and report availability.
    """
    status = check_readiness()
    config = get_channel_config()
    watchdog = _resolve_watchdog_settings(config)

    # Backward compatibility: historical callers pass timeout=1800 expecting
    # "do not hang forever". We now interpret this as a stall timeout floor,
    # not a hard wall-clock kill.
    legacy_timeout = max(0, int(timeout or 0))
    scan_stall_timeout = watchdog["scan_stall_timeout_seconds"]
    report_stall_timeout = watchdog["report_stall_timeout_seconds"]
    if legacy_timeout > 0:
        scan_stall_timeout = max(scan_stall_timeout, legacy_timeout)
        report_stall_timeout = max(report_stall_timeout, legacy_timeout)

    logger.info(
        f"[YOUTUBE] Readiness check: report={'OK' if status['report_ready'] else 'MISSING'}, "
        f"stale={status['report_stale']}, channels={len(status['channels_scraped'])}/{len(status['channels_scraped']) + len(status['channels_missing'])}"
    )

    if status["channels_missing"]:
        logger.info(
            f"[YOUTUBE] Missing channels: {', '.join(status['channels_missing'])}"
        )

    result = {
        "pre_check": status,
        "scan_ran": False,
        "scan_result": None,
        "report_generated": False,
        "intelligence_loaded": False,
        "watchdog": {
            "scan_stall_timeout_seconds": scan_stall_timeout,
            "scan_hard_timeout_seconds": watchdog["scan_hard_timeout_seconds"],
            "report_stall_timeout_seconds": report_stall_timeout,
            "report_hard_timeout_seconds": watchdog["report_hard_timeout_seconds"],
            "progress_poll_seconds": watchdog["progress_poll_seconds"],
            "verbose_subprocess_logs": bool(
                watchdog.get("verbose_subprocess_logs", False)
            ),
        },
    }

    # Run scan if needed
    if force_scan or status["needs_scan"]:
        logger.info(
            "[YOUTUBE] Running channel scan for latest videos "
            f"(stall_timeout={scan_stall_timeout}s, hard_timeout={watchdog['scan_hard_timeout_seconds']}s)..."
        )
        try:
            scan_result = _run_youtube_scan(
                stall_timeout=scan_stall_timeout,
                hard_timeout=watchdog["scan_hard_timeout_seconds"],
                progress_poll_seconds=watchdog["progress_poll_seconds"],
                target_date=status.get("target_date"),
                verbose_logs=bool(watchdog.get("verbose_subprocess_logs", False)),
            )
            result["scan_ran"] = True
            result["scan_result"] = scan_result
            # Update scan cooldown marker so we don't re-scan too quickly
            try:
                RAG_DIR.mkdir(parents=True, exist_ok=True)
                scan_marker = RAG_DIR / ".last_scan_at"
                scan_marker.write_text(datetime.now().isoformat(), encoding="utf-8")
            except Exception:
                pass
        except Exception as e:
            logger.warning("[YOUTUBE] Scan failed: %s", safe_exception_text(e))
            result["scan_error"] = safe_exception_text(e)

    status_after_scan = status
    if result["scan_ran"]:
        try:
            status_after_scan = check_readiness(target_date=status.get("target_date"))
            result["post_scan_check"] = status_after_scan
        except Exception:
            # Keep conservative pre-scan status when post-scan check fails.
            status_after_scan = status

    scan_result = result.get("scan_result")
    scan_report_generated = bool(
        isinstance(scan_result, dict) and scan_result.get("report_generated")
    )
    scan_degraded = bool(
        isinstance(scan_result, dict) and scan_result.get("scan_degraded")
    )
    scan_new_videos = 0
    if isinstance(scan_result, dict):
        try:
            scan_new_videos = max(0, int(scan_result.get("new_videos_processed", 0)))
        except Exception:
            scan_new_videos = 0
    if scan_degraded:
        result["scan_error"] = (
            "youtube_scan_degraded:"
            + ",".join(scan_result.get("metadata_failed_channels") or [])
        )
        logger.warning(
            "[YOUTUBE] Scan degraded; metadata failed for channels: %s",
            ", ".join(scan_result.get("metadata_failed_channels") or ["unknown"]),
        )

    report_needed_from_scan = scan_new_videos > 0 and not scan_report_generated
    should_generate_report = bool(
        force_report
        or (
            status_after_scan.get("needs_report", status.get("needs_report", False))
            and not scan_degraded
        )
        or report_needed_from_scan
    )

    # Generate report if needed
    if should_generate_report:
        logger.info(
            "[YOUTUBE] Generating consolidated report "
            f"(stall_timeout={report_stall_timeout}s, hard_timeout={watchdog['report_hard_timeout_seconds']}s)..."
        )
        try:
            report_ok = _generate_report(
                stall_timeout=report_stall_timeout,
                hard_timeout=watchdog["report_hard_timeout_seconds"],
                progress_poll_seconds=watchdog["progress_poll_seconds"],
                target_date=status.get("target_date"),
                verbose_logs=bool(watchdog.get("verbose_subprocess_logs", False)),
            )
            result["report_generated"] = report_ok
        except Exception as e:
            logger.warning(
                "[YOUTUBE] Report generation failed: %s", safe_exception_text(e)
            )
    else:
        logger.info(
            "[YOUTUBE] Skipping report generation "
            f"(force_report={force_report}, needs_report={status_after_scan.get('needs_report', False)}, "
            f"scan_new_videos={scan_new_videos}, scan_report_generated={scan_report_generated})"
        )

    # Verify final state
    final_status = check_readiness()
    result["post_check"] = final_status
    result["intelligence_loaded"] = final_status["report_ready"]

    if final_status["report_ready"]:
        logger.info(
            f"[YOUTUBE] Intelligence ready (report date: {final_status['report_date']})"
        )
    else:
        logger.warning("[YOUTUBE] Intelligence NOT ready after scan attempt")

    return result


def _run_youtube_scan(
    stall_timeout: int = 7200,
    hard_timeout: int = 0,
    progress_poll_seconds: int = 5,
    target_date: Optional[str] = None,
    verbose_logs: bool = False,
) -> Optional[Dict]:
    """Run the YouTube daily scanner as a subprocess."""
    scanner_path = PROJECT_ROOT / "tools" / "youtube_daily_scanner.py"
    if not scanner_path.exists():
        logger.error(f"Scanner not found: {scanner_path}")
        return None

    config = get_channel_config()
    output_dir = _resolve_output_dir(config)
    today = _today_str()
    target = target_date or _next_trading_date()

    try:
        cmd = [
            sys.executable,
            str(scanner_path),
            "--verbose",
            *_get_summary_override_args(),
        ]
        logger.info(
            f"[YOUTUBE] Launching scanner subprocess "
            f"(stall_timeout={stall_timeout}s, hard_timeout={hard_timeout}s): {' '.join(cmd)}"
        )
        run_meta = _run_subprocess_with_watchdog(
            cmd,
            cwd=PROJECT_ROOT,
            stall_timeout=stall_timeout,
            hard_timeout=hard_timeout,
            progress_poll_seconds=progress_poll_seconds,
            progress_snapshot=lambda: _scan_progress_snapshot(
                output_dir, today=today, target=target
            ),
            label="[YOUTUBE][SCAN]",
            verbose_logs=verbose_logs,
        )
        elapsed = float(run_meta.get("elapsed_seconds", 0.0))
        stdout_lines = int(run_meta.get("stdout_lines", 0))
        stderr_lines = int(run_meta.get("stderr_lines", 0))
        if int(run_meta.get("returncode", -1)) == 0:
            logger.info(
                f"[YOUTUBE] Scan completed successfully in {elapsed:.1f}s "
                f"(stdout_lines={stdout_lines}, stderr_lines={stderr_lines})"
            )
            # Try to parse JSON summary from stdout
            try:
                # Scanner prints JSON summary at the end (pretty-printed multiline JSON).
                lines = [
                    ln for ln in run_meta.get("stdout_tail", []) if isinstance(ln, str)
                ]
                for idx in range(len(lines) - 1, -1, -1):
                    probe = lines[idx].strip()
                    if probe.startswith("{"):
                        return json.loads("\n".join(lines[idx:]))
            except (json.JSONDecodeError, IndexError, TypeError):
                pass
            return {
                "status": "ok",
                "stdout_lines": stdout_lines,
                "stderr_lines": stderr_lines,
                "elapsed_seconds": elapsed,
            }
        else:
            terminated_reason = run_meta.get("terminated_reason")
            logger.warning(
                f"[YOUTUBE] Scan exited with code {run_meta.get('returncode')} after {elapsed:.1f}s "
                f"(stdout_lines={stdout_lines}, stderr_lines={stderr_lines}, reason={terminated_reason or 'nonzero_exit'})"
            )
            stderr_tail = run_meta.get("stderr_tail", [])[-12:]
            if stderr_tail:
                logger.warning("[YOUTUBE] stderr tail:\n" + "\n".join(stderr_tail))
            return None
    except Exception as e:
        logger.error("[YOUTUBE] Scan error: %s", safe_exception_text(e))
        return None


def _generate_report(
    stall_timeout: int = 3600,
    hard_timeout: int = 0,
    progress_poll_seconds: int = 5,
    target_date: Optional[str] = None,
    verbose_logs: bool = False,
) -> bool:
    """Generate/regenerate the daily consolidated report."""
    scanner_path = PROJECT_ROOT / "tools" / "youtube_daily_scanner.py"
    if not scanner_path.exists():
        return False

    today = _today_str()
    target = target_date or _next_trading_date()

    try:
        cmd = [
            sys.executable,
            str(scanner_path),
            "--report-only",
            *_get_summary_override_args(),
        ]
        logger.info(
            f"[YOUTUBE] Launching report subprocess "
            f"(stall_timeout={stall_timeout}s, hard_timeout={hard_timeout}s): {' '.join(cmd)}"
        )
        run_meta = _run_subprocess_with_watchdog(
            cmd,
            cwd=PROJECT_ROOT,
            stall_timeout=stall_timeout,
            hard_timeout=hard_timeout,
            progress_poll_seconds=progress_poll_seconds,
            progress_snapshot=lambda: _report_progress_snapshot(
                today=today, target=target
            ),
            label="[YOUTUBE][REPORT]",
            verbose_logs=verbose_logs,
        )
        elapsed = float(run_meta.get("elapsed_seconds", 0.0))
        stdout_lines = int(run_meta.get("stdout_lines", 0))
        stderr_lines = int(run_meta.get("stderr_lines", 0))
        if int(run_meta.get("returncode", -1)) == 0:
            logger.info(
                f"[YOUTUBE] Report generation completed in {elapsed:.1f}s "
                f"(stdout_lines={stdout_lines}, stderr_lines={stderr_lines})"
            )
            return True
        terminated_reason = run_meta.get("terminated_reason")
        logger.warning(
            f"[YOUTUBE] Report generation exited with code {run_meta.get('returncode')} after {elapsed:.1f}s "
            f"(stdout_lines={stdout_lines}, stderr_lines={stderr_lines}, reason={terminated_reason or 'nonzero_exit'})"
        )
        stderr_tail = run_meta.get("stderr_tail", [])[-12:]
        if stderr_tail:
            logger.warning("[YOUTUBE] report stderr tail:\n" + "\n".join(stderr_tail))
        return False
    except Exception as e:
        logger.warning(
            "[YOUTUBE] Report generation error: %s", safe_exception_text(e)
        )
        return False


# ── Intelligence Context for Trading ──────────────────────────────


def _coverage_context_from_status(status: Dict[str, Any]) -> Dict[str, Any]:
    channels_total = int(status.get("channels_total", 0) or 0)
    channels_processed = int(status.get("channels_processed", 0) or 0)
    coverage_pct = float(status.get("coverage_pct", 0.0) or 0.0)
    coverage_grade = str(status.get("coverage_grade", "") or "").lower()
    if coverage_grade not in {"complete", "partial", "minimal"}:
        if channels_total <= 0:
            coverage_grade = "minimal"
        else:
            ratio = channels_processed / channels_total
            coverage_grade = (
                "complete" if ratio >= 0.75 else "partial" if ratio >= 0.5 else "minimal"
            )
            coverage_pct = round(ratio, 4)
    return {
        "channels_total": channels_total,
        "channels_processed": channels_processed,
        "channels_missing": list(status.get("channels_missing", []) or []),
        "coverage_pct": coverage_pct,
        "coverage_grade": coverage_grade,
    }


def get_intelligence_context(fallback_days: int = 3) -> Dict[str, Any]:
    """
    Load the latest YouTube intelligence and return a flat context dict
    ready for consumption by trading workflows.

    This is the ONE function that all trading modules should call.
    Returns safe defaults if no intelligence is available.

    Context dict:
        regime: str              - Market regime classification
        regime_confidence: int   - 0-100
        sizing_multiplier: float - Position sizing multiplier (0.0-1.5)
        avoid_sectors: list      - Sectors to avoid/underweight
        favor_sectors: list      - Sectors to overweight
        smallcap_ok: bool        - Is small-cap health acceptable?
        trigger_levels: dict     - Key levels for SPY/QQQ/IWM
        directives: list         - Agent action directives
        report_date: str         - Date of the report
        report_source: str       - "daily" or "weekend_consolidated"
    available: bool          - Whether any intelligence was loaded
    """
    try:
        from autotrade.utils.market_intelligence import load_market_intelligence

        status = check_readiness()
        coverage_context = _coverage_context_from_status(status)
        if not bool(status.get("report_ready")):
            context = _default_context()
            context.update(coverage_context)
            return context
        report_target = str(
            status.get("target_date")
            or status.get("report_date")
            or _next_trading_date()
        )
        report = load_market_intelligence(
            date_str=report_target,
            fallback_days=0,
        )
    except ImportError:
        report = None

    if not report:
        return _default_context()

    # Extract regime
    regime = report.get("market_regime", "NEUTRAL")
    confidence = report.get("regime_confidence", 50)

    # Extract trading signals
    signals = report.get("trading_signals", {})
    if not isinstance(signals, dict):
        signals = {}
    sizing = float(signals.get("sizing_multiplier", 1.0))

    # Extract sector bias
    avoid_sectors = []
    favor_sectors = []
    for sb in signals.get("sector_bias", []):
        bias = sb.get("bias", "").lower()
        sector = sb.get("sector", "")
        if bias in ("avoid", "underweight"):
            avoid_sectors.append(sector)
        elif bias == "overweight":
            favor_sectors.append(sector)

    sector_demotion_available = coverage_context.get("coverage_grade") != "minimal"
    youtube_authority_weight = 1.0
    if not sector_demotion_available:
        youtube_authority_weight = 0.5
        if avoid_sectors:
            logger.warning(
                "[YOUTUBE] minimal coverage (%s/%s channels) - sector demotion suppressed",
                coverage_context.get("channels_processed", 0),
                coverage_context.get("channels_total", 0),
            )
        avoid_sectors = []
        sizing = 1.0 + ((float(sizing) - 1.0) * youtube_authority_weight)

    # Small-cap health
    sc = report.get("smallcap_health", {})
    sc_status = sc.get("status", "unknown").lower()
    smallcap_ok = sc_status in ("healthy", "cautious", "unknown")

    # Trigger levels
    trigger_levels = signals.get("trigger_levels", {})

    # Directives
    directives = report.get("overnight_directives", report.get("agent_actions", []))
    raw_text = "\n".join(
        str(part or "")
        for part in (
            report.get("raw_report", ""),
            report.get("executive_summary", ""),
            report.get("regime_summary", ""),
            "\n".join(str(item) for item in directives),
        )
    ).upper()
    if any(
        marker in raw_text
        for marker in ("NO NEW LONGS", "MAXIMUM DEFENSIVE", "RISK-OFF CONFIRMED")
    ):
        regime = "RISK-OFF"
        confidence = max(int(confidence or 0), 85)
        sizing = min(float(sizing), 0.25)
        smallcap_ok = False
        if not directives:
            directives = [
                "NO NEW LONGS until breadth and index support stabilize.",
                "Remain defensive and prioritize capital preservation.",
            ]

    # Report metadata
    meta = report.get("_meta", {})
    report_date = meta.get("date", report.get("date", "unknown"))
    report_source = meta.get("_source", "daily")

    context = {
        "regime": regime,
        "regime_confidence": confidence,
        "sizing_multiplier": sizing,
        "avoid_sectors": avoid_sectors,
        "favor_sectors": favor_sectors,
        "smallcap_ok": smallcap_ok,
        "trigger_levels": trigger_levels,
        "directives": directives,
        "report_date": report_date,
        "report_source": report_source,
        "available": True,
        "sector_demotion_available": sector_demotion_available,
        "youtube_authority_weight": youtube_authority_weight,
        "executive_summary": report.get("executive_summary", ""),
        "regime_summary": report.get("regime_summary", ""),
    }
    context.update(coverage_context)
    return context


def _default_context() -> Dict[str, Any]:
    """Return safe default context when no intelligence is available."""
    return {
        "regime": "NEUTRAL",
        "regime_confidence": 0,
        "sizing_multiplier": 1.0,
        "avoid_sectors": [],
        "favor_sectors": [],
        "smallcap_ok": True,
        "trigger_levels": {},
        "directives": [],
        "report_date": "none",
        "report_source": "none",
        "available": False,
        "sector_demotion_available": False,
        "youtube_authority_weight": 0.0,
        "channels_total": 0,
        "channels_processed": 0,
        "channels_missing": [],
        "coverage_pct": 0.0,
        "coverage_grade": "minimal",
        "executive_summary": "",
        "regime_summary": "",
    }


def format_readiness_log(status: Dict) -> str:
    """Format readiness status for logging."""
    report_state = "READY" if status.get("report_ready") else "MISSING"
    report_scope = "current" if status.get("report_current") else "current_missing"
    fallback_available = bool(status.get("fallback_report_available"))
    fallback_date = status.get("fallback_report_date", "none")

    def _csv_list(key: str) -> str:
        values = status.get(key) or []
        if not isinstance(values, (list, tuple, set)):
            values = [values]
        return ", ".join(str(value) for value in values if value) or "none"

    lines = [
        "[YOUTUBE INTELLIGENCE STATUS]",
        f"   Report: {report_state} ({report_scope}, date: {status.get('report_date', 'none')}, "
        f"stale: {status.get('report_stale', False)})",
        f"   Fallback report available: {fallback_available} (date: {fallback_date})",
        f"   Channels scraped: {_csv_list('channels_scraped')}",
        f"   Channels checked: {_csv_list('channels_checked')}",
        f"   Channels unchecked: {_csv_list('channels_unchecked')}",
        f"   Channels missing content: {_csv_list('channels_missing_content')}",
        f"   Channels failed extraction: {_csv_list('channels_failed_extraction')}",
        f"   Channels missing: {_csv_list('channels_missing')}",
        f"   Session artifacts available: {status.get('videos_processed_today', 0)}",
        f"   Usable session artifacts: {status.get('usable_videos_processed_today', status.get('videos_processed_today', 0))}",
        f"   Transcript artifacts: {status.get('transcript_artifacts', 0)}",
        f"   Needs scan: {status.get('needs_scan', True)}",
        f"   Needs report: {status.get('needs_report', False)}",
    ]
    if "scan_ran" in status:
        lines.append(f"   Scan ran this cycle: {bool(status.get('scan_ran'))}")
    return "\n".join(lines)
