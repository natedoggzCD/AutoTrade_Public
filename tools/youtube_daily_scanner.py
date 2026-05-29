#!/usr/bin/env python3
"""
YouTube Daily Scanner for AutoTrade
====================================
Automated daily scanning of trading YouTube channels.

Workflow:
1. Check each channel for new videos (via yt-dlp)
2. Download + GPU-transcribe new videos (faster-whisper on mlgpu)
3. Run transcripts through producer-specific extraction prompts (Ollama)
4. Store structured extractions for RAG retrieval
5. Generate consolidated daily market intelligence report

Usage:
    python tools/youtube_daily_scanner.py                    # Scan all channels
    python tools/youtube_daily_scanner.py --channel trade_brigade  # Scan one channel
    python tools/youtube_daily_scanner.py --report-only      # Re-generate report from existing extractions
    python tools/youtube_daily_scanner.py --list-pending      # Show unprocessed videos
    python tools/youtube_daily_scanner.py --force-rescan      # Ignore dedup, rescan everything

Designed to run as part of the overnight/premarket workflow in autonomous_agent.py.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from google import genai
except ImportError:
    genai = None

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autotrade.utils.youtube_rag import get_rag_manager  # noqa: E402

logger = logging.getLogger("youtube_scanner")

# ── Config Loading ──────────────────────────────────────────────────

def load_channel_config(config_path: Optional[Path] = None) -> dict:
    """Load youtube_channels.yaml configuration."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "youtube_channels.yaml"
    
    try:
        import yaml
    except ImportError:
        raise RuntimeError("PyYAML required. Run: pip install pyyaml")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_processed_videos(track_file: Path) -> dict:
    """Load the deduplication tracking file."""
    if track_file.exists():
        with open(track_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"processed": {}}


def save_processed_videos(track_file: Path, data: dict) -> None:
    """Save deduplication tracking data."""
    track_file.parent.mkdir(parents=True, exist_ok=True)
    with open(track_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Channel Scanning ───────────────────────────────────────────────

def _is_dns_error(exc: Exception) -> bool:
    """Check if exception is related to DNS resolution failure."""
    msg = str(exc).lower()
    return "getaddrinfo failed" in msg or "non-recoverable failure" in msg or "socket.gaierror" in msg

# yt-dlp requires a JavaScript runtime for YouTube's signature/n challenges.
# Node.js is available on this system; Deno (yt-dlp's default) is not.
_YDL_JS_RUNTIME = {"js_runtimes": {"node": {}}}

_cookie_opts_cache: Optional[dict] = None
_cookie_opts_validated: bool = False
_recent_video_status: Dict[str, Dict[str, Any]] = {}


def _configure_utf8_stdio() -> None:
    """Prefer UTF-8 stdio on Windows so yt-dlp/log output cannot break on cp1252."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _get_cookie_opts() -> dict:
    """Return explicit cookie-file opts when the user exported one."""
    global _cookie_opts_cache, _cookie_opts_validated
    if _cookie_opts_validated:
        return _cookie_opts_cache or {}

    cookie_file = Path(__file__).parent.parent / "config" / "youtube_cookies.txt"
    if cookie_file.exists() and cookie_file.stat().st_size > 100:
        _cookie_opts_cache = {"cookiefile": str(cookie_file)}
        _cookie_opts_validated = True
        return _cookie_opts_cache

    _cookie_opts_cache = {}
    _cookie_opts_validated = True
    return {}


def _looks_like_auth_error(exc: Exception) -> bool:
    """Return True when yt-dlp likely needs authenticated cookies for this item."""
    msg = _safe_exception_text(exc, max_len=2000).lower()
    markers = (
        "sign in to confirm",
        "use --cookies",
        "members-only",
        "private video",
        "age-restricted",
        "age restricted",
        "login required",
        "authentication required",
        "this video is unavailable",
        "http error 429",
        "too many requests",
    )
    return any(marker in msg for marker in markers)


def _set_recent_video_status(channel_url: str, **status: Any) -> None:
    _recent_video_status[channel_url] = dict(status)


def get_recent_videos_status(channel_url: str) -> Dict[str, Any]:
    """Return status from the most recent get_recent_videos call for this channel."""
    return dict(_recent_video_status.get(channel_url, {}))


def get_recent_videos(
    channel_url: str,
    max_age_hours: int = 36,
    max_results: int = 5,
    retries: int = 3
) -> List[dict]:
    """
    Get recent videos from a YouTube channel using yt-dlp.

    Returns list of dicts with: id, title, url, upload_date, duration

    Live broadcasts (the /streams tab) are intentionally NOT scanned: by
    the time a premarket live stream is available as a VOD, the trading
    window it covers has already passed.
    """
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        raise RuntimeError("yt-dlp required. Run: pip install yt-dlp")

    # Use /videos tab to get uploads
    videos_url = channel_url.rstrip("/") + "/videos"

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "encoding": "utf-8",
        "extract_flat": True,        # Don't download, just get metadata
        "playlistend": max_results,   # Only check recent N videos
        "ignoreerrors": True,
        **_YDL_JS_RUNTIME,
    }
    cookie_opts = _get_cookie_opts()

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

    for attempt in range(1, retries + 1):
        status = {
            "attempt": attempt,
            "entries_seen": 0,
            "valid_entries_seen": 0,
            "detail_failures": 0,
            "missing_upload_dates": 0,
            "used_cookie_fallback": False,
            "had_errors": False,
            "metadata_degraded": False,
        }
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(videos_url, download=False)

                if not info:
                    status["had_errors"] = True
                    _set_recent_video_status(channel_url, **status)
                    logger.warning(f"No info returned for {channel_url} (attempt {attempt})")
                    if attempt < retries:
                        time.sleep(2 ** attempt)
                        continue
                    return []

                entries = info.get("entries") or []
                status["entries_seen"] = len(entries)
                videos = []
                for entry in entries:
                    if not entry:
                        continue

                    status["valid_entries_seen"] += 1
                    video_id = entry.get("id", "")
                    title = entry.get("title", "Unknown")
                    upload_date_str = entry.get("upload_date", "") or ""
                    duration = entry.get("duration") or 0

                    # If upload_date missing, attempt a detailed fetch for this video
                    if not upload_date_str and video_id:
                        detail_url = f"https://www.youtube.com/watch?v={video_id}"
                        try:
                            detail_opts = {
                                **ydl_opts,
                                "extract_flat": False,
                                "skip_download": True,
                            }
                            with YoutubeDL(detail_opts) as ydl_detail:
                                detail = ydl_detail.extract_info(detail_url, download=False)
                                if detail:
                                    upload_date_str = detail.get("upload_date") or upload_date_str
                                    duration = detail.get("duration") or duration
                        except Exception as e:
                            status["detail_failures"] += 1
                            if cookie_opts and _looks_like_auth_error(e):
                                try:
                                    detail_opts = {
                                        **ydl_opts,
                                        **cookie_opts,
                                        "extract_flat": False,
                                        "skip_download": True,
                                    }
                                    with YoutubeDL(detail_opts) as ydl_detail:
                                        detail = ydl_detail.extract_info(detail_url, download=False)
                                        if detail:
                                            upload_date_str = detail.get("upload_date") or upload_date_str
                                            duration = detail.get("duration") or duration
                                            status["used_cookie_fallback"] = True
                                except Exception as cookie_err:
                                    logger.debug(
                                        "Cookie fallback detail fetch failed for %s: %s",
                                        video_id,
                                        _safe_exception_text(cookie_err),
                                    )
                            elif _looks_like_auth_error(e):
                                logger.warning(
                                    "YouTube auth required for %s and no cookie file is configured. "
                                    "Export cookies to config/youtube_cookies.txt if this video should be accessible.",
                                    video_id,
                                )
                            else:
                                logger.debug(
                                    "Detail fetch failed for %s: %s",
                                    video_id,
                                    _safe_exception_text(e),
                                )

                    try:
                        duration = int(float(duration))
                    except (TypeError, ValueError):
                        duration = 0

                    # Parse upload date
                    upload_date = None
                    if upload_date_str:
                        try:
                            cleaned = upload_date_str.strip().replace("-", "")
                            upload_date = datetime.strptime(cleaned, "%Y%m%d").replace(
                                tzinfo=timezone.utc
                            )
                        except ValueError:
                            pass

                    # Filter by age
                    if upload_date and upload_date < cutoff:
                        continue
                    if not upload_date:
                        status["missing_upload_dates"] += 1
                        logger.debug(f"Skipping video with missing upload_date: {video_id} ({title})")
                        continue

                    url = f"https://www.youtube.com/watch?v={video_id}"
                    videos.append({
                        "id": video_id,
                        "title": title,
                        "url": url,
                        "upload_date": upload_date_str,
                        "upload_datetime": upload_date.isoformat() if upload_date else None,
                        "duration": duration,
                    })
                status["metadata_degraded"] = bool(
                    status["valid_entries_seen"] > 0
                    and not videos
                    and status["missing_upload_dates"] >= status["valid_entries_seen"]
                )
                status["had_errors"] = bool(status["metadata_degraded"] or status["detail_failures"])
                _set_recent_video_status(channel_url, **status)
                return videos
        except Exception as e:
            status["had_errors"] = True
            _set_recent_video_status(channel_url, **status)
            if _is_dns_error(e):
                logger.warning(
                    "DNS error scanning %s (attempt %s): %s",
                    channel_url,
                    attempt,
                    _safe_exception_text(e),
                )
            else:
                logger.error(
                    "Error scanning %s (attempt %s): %s",
                    channel_url,
                    attempt,
                    _safe_exception_text(e),
                )

            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                return []

    _set_recent_video_status(channel_url, had_errors=True, metadata_degraded=False)
    return []

def filter_new_videos(
    videos: List[dict],
    processed: dict,
    channel_key: str,
    rag_dir: Optional[Path] = None,
) -> List[dict]:
    """Filter out already-processed videos."""
    processed_ids = set()

    def _is_terminal_state(info: dict) -> bool:
        """Return True when a video can be safely skipped on future scans."""
        if not isinstance(info, dict):
            return False
        if bool(info.get("recovered_from_rag")):
            return True
        status = str(info.get("status", "")).strip().lower()
        if status == "complete":
            return True
        if status == "complete_with_fallback":
            return not bool(info.get("error"))
        return False

    for vid_id, vid_info in processed.get("processed", {}).items():
        if _is_terminal_state(vid_info):
            processed_ids.add(vid_id)
        else:
            logger.debug(
                "Retrying non-terminal processed video state: %s (status=%s)",
                vid_id,
                (vid_info or {}).get("status"),
            )
    
    new = []
    for v in videos:
        vid = str(v.get("id") or "")
        if vid not in processed_ids and rag_dir is not None:
            channel_dir = rag_dir / "by_channel" / channel_key
            if channel_dir.exists():
                # Recover dedup state from persisted extraction artifacts.
                matched_files = list(channel_dir.glob(f"*_{vid}.json"))
                usable_match = False
                for artifact_path in matched_files:
                    try:
                        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    if _is_usable_extraction_payload(payload):
                        usable_match = True
                        break
                if usable_match:
                    processed.setdefault("processed", {})[vid] = {
                        "channel": channel_key,
                        "title": v.get("title", ""),
                        "date": v.get("upload_date", ""),
                        "status": "complete",
                        "timestamp": datetime.now().isoformat(),
                        "recovered_from_rag": True,
                    }
                    processed_ids.add(vid)
                    logger.debug(f"Recovered processed state from RAG: {v.get('title', vid)}")
        if vid not in processed_ids:
            new.append(v)
        else:
            logger.debug(f"Skipping already processed: {v['title']}")
    
    return new


# ── Transcription ──────────────────────────────────────────────────

def slugify(text: str, max_len: int = 100) -> str:
    """Convert text to filesystem-safe slug."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s_-]+", "_", text)[:max_len]


def _to_wsl_path(path: Path) -> str:
    """Convert a Windows path to a WSL /mnt/<drive>/ path."""
    p = Path(path).resolve()
    drive = p.drive.replace(":", "").lower()
    rest = p.as_posix()
    if ":" in rest:
        rest = rest.split(":", 1)[1]
    if not rest.startswith("/"):
        rest = "/" + rest
    return f"/mnt/{drive}{rest}"


def _sh_escape(value: str) -> str:
    """Escape a string for single-quoted bash."""
    return value.replace("'", "'\"'\"'")


def _parse_bool(value: Any, default: bool = True) -> bool:
    """Parse bool-like config values."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _safe_console_text(text: Any, depth: int = 0) -> str:
    """Best-effort console-safe text for Windows terminals, preventing RecursionError."""
    if text is None:
        return ""
    if depth > 3:
        return "[RECURSION_LIMIT_REACHED]"
    
    try:
        s = str(text)
        # On Windows, stdout encoding might be cp1252/utf-8. 
        # We target utf-8 but handle failures gracefully.
        encoding = (getattr(sys.stdout, "encoding", None) or "utf-8").lower()
        if encoding == "cp1252" or encoding == "charmap":
            # If we're stuck in a legacy terminal, try to normalize to ascii/safe chars
            return s.encode("ascii", errors="replace").decode("ascii")
        return s
    except Exception:
        try:
            return repr(text)
        except Exception:
            return "[UNRENDERABLE_OBJECT]"


def _safe_exception_text(exc: Exception, max_len: int = 500) -> str:
    """Best-effort exception rendering for scanner logs, protecting against RecursionError."""
    try:
        exc_type = type(exc).__name__ or "Exception"
        try:
            message = str(exc).strip()
        except Exception:
            try:
                message = repr(exc).strip()
            except Exception:
                message = "[unrenderable]"
        
        # Avoid recursion by using primitive encoding if needed
        safe_message = message.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        
        if max_len > 0 and len(safe_message) > max_len:
            safe_message = safe_message[: max_len - 15].rstrip() + "... [truncated]"
            
        if not safe_message or safe_message == exc_type:
            return exc_type
        if safe_message.startswith(f"{exc_type}:"):
            return safe_message
        return f"{exc_type}: {safe_message}"
    except Exception:
        return "UnrenderableException"


def _get_env_var(name: str) -> Optional[str]:
    """
    Resolve an environment variable, falling back to PROJECT_ROOT/.env when present.
    """
    value = os.environ.get(name)
    if value:
        return value

    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return None

    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, raw_val = stripped.split("=", 1)
            if key.strip() != name:
                continue
            val = raw_val.strip().strip('"').strip("'")
            return val or None
    except Exception:
        return None
    return None


def _run_transcriber_entries(
    entries: List[dict],
    output_dir: Path,
    config: dict,
    timeout_seconds: Optional[int] = None,
) -> Optional[List[dict]]:
    """
    Run youtube_transcriber.py once for a batch of entries.

    Returns parsed result list or None on hard failure.
    """
    if not entries:
        return []

    tx_config = config.get("scanner", {}).get("transcription", {})
    python_path = tx_config.get("python_path", "python")
    env_vars = tx_config.get("env_vars", {})
    model = tx_config.get("model", "large-v3")
    device = tx_config.get("device", "cuda")
    dtype = tx_config.get("dtype", "float16")
    ffmpeg_location = tx_config.get("ffmpeg_location")
    wsl_cfg = tx_config.get("wsl", {})

    beam_size = int(tx_config.get("beam_size", 1))
    best_of = int(tx_config.get("best_of", beam_size))
    condition_prev = _parse_bool(tx_config.get("condition_on_previous_text", True), default=True)
    vad_filter = _parse_bool(tx_config.get("vad_filter", True), default=True)
    artifact_mode = str(tx_config.get("artifact_mode", "scanner"))

    env = os.environ.copy()
    for k, v in env_vars.items():
        env[k] = str(v)

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / ".scanner_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    batch_fd, batch_path = tempfile.mkstemp(prefix="yt_batch_", suffix=".json", dir=str(tmp_dir))
    result_fd, result_path = tempfile.mkstemp(prefix="yt_result_", suffix=".json", dir=str(tmp_dir))
    os.close(batch_fd)
    os.close(result_fd)
    batch_file = Path(batch_path)
    result_file = Path(result_path)
    batch_file.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

    transcriber = str(PROJECT_ROOT / "tools" / "youtube_transcriber.py")
    cmd_timeout = timeout_seconds or (max(1, len(entries)) * 600 + 120)

    try:
        if wsl_cfg.get("enabled"):
            distro = wsl_cfg.get("distro")
            if not distro:
                raise RuntimeError(
                    "WSL transcription enabled but no distro configured (scanner.transcription.wsl.distro)"
                )
            wsl_python = wsl_cfg.get("python", "python3")
            wsl_project = _to_wsl_path(PROJECT_ROOT)
            wsl_output = _to_wsl_path(output_dir)
            wsl_transcriber = f"{wsl_project}/tools/youtube_transcriber.py"
            wsl_batch = _to_wsl_path(batch_file)
            wsl_result = _to_wsl_path(result_file)

            exports = []
            for k, v in env_vars.items():
                exports.append(f"export {k}='{_sh_escape(str(v))}';")
            if ffmpeg_location:
                exports.append(f"export PATH='{_sh_escape(str(ffmpeg_location))}':\"$PATH\";")
            export_cmd = " ".join(exports)

            wsl_cmd = (
                f"cd '{_sh_escape(wsl_project)}' && "
                f"{export_cmd} {wsl_python} '{_sh_escape(wsl_transcriber)}' "
                f"--batch-json '{_sh_escape(wsl_batch)}' "
                f"--result-json '{_sh_escape(wsl_result)}' "
                f"--model '{_sh_escape(model)}' "
                f"--device '{_sh_escape(device)}' "
                f"--dtype '{_sh_escape(dtype)}' "
                f"--output-dir '{_sh_escape(wsl_output)}' "
                f"--beam-size {beam_size} "
                f"--best-of {best_of} "
                f"--condition-on-previous-text {'true' if condition_prev else 'false'} "
                f"--vad-filter {'true' if vad_filter else 'false'} "
                f"--artifact-mode '{_sh_escape(artifact_mode)}'"
            )
            if ffmpeg_location:
                wsl_cmd += f" --ffmpeg-location '{_sh_escape(str(ffmpeg_location))}'"

            cmd = ["wsl", "-d", distro, "--", "bash", "-lc", wsl_cmd]
        else:
            cmd = [
                python_path,
                transcriber,
                "--batch-json",
                str(batch_file),
                "--result-json",
                str(result_file),
                "--model",
                model,
                "--device",
                device,
                "--dtype",
                dtype,
                "--output-dir",
                str(output_dir),
                "--beam-size",
                str(beam_size),
                "--best-of",
                str(best_of),
                "--condition-on-previous-text",
                "true" if condition_prev else "false",
                "--vad-filter",
                "true" if vad_filter else "false",
                "--artifact-mode",
                artifact_mode,
            ]
            if ffmpeg_location:
                cmd.extend(["--ffmpeg-location", str(ffmpeg_location)])

        logger.debug(f"Transcriber command: {' '.join(cmd)}")
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=cmd_timeout,
            env=env,
            cwd=str(PROJECT_ROOT),
        )

        result_payload: Optional[List[dict]] = None
        if result_file.exists():
            try:
                parsed = json.loads(result_file.read_text(encoding="utf-8"))
                if isinstance(parsed, list):
                    result_payload = parsed
            except Exception as parse_err:
                logger.error(
                    "Failed parsing transcriber result json: %s",
                    _safe_exception_text(parse_err),
                )

        if proc.returncode != 0:
            stdout = (proc.stdout or "").strip()
            stderr = (proc.stderr or "").strip()
            logger.error(f"Transcriber batch failed (exit {proc.returncode}) for {len(entries)} video(s).")
            if stdout:
                logger.error(f"Transcriber stdout: {stdout[:2000]}")
            if stderr:
                logger.error(f"Transcriber stderr: {stderr[:2000]}")
            # Return partial payload if available (transcriber can fail late with partial successes).
            if result_payload is not None:
                return result_payload
            return None

        return result_payload or []
    except subprocess.TimeoutExpired:
        logger.error(f"Transcriber batch timed out for {len(entries)} video(s)")
        return None
    except Exception as e:
        logger.error("Transcriber batch error: %s", _safe_exception_text(e))
        return None
    finally:
        for p in (batch_file, result_file):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass


def transcribe_video(
    url: str,
    output_dir: Path,
    config: dict,
    video_id: Optional[str] = None,
    video_title: Optional[str] = None,
) -> Optional[dict]:
    """
    Transcribe a video using the mlgpu environment.

    Uses subprocess to call the transcriber and parse structured result output.
    Returns transcription result dict or None on failure.
    """
    logger.info(f"Transcribing: {url}")
    entry_video_id = video_id or ""
    entry_prefix = slugify(f"{entry_video_id}_{video_title}" if entry_video_id and video_title else (entry_video_id or video_title or "transcript"))
    results = _run_transcriber_entries(
        entries=[
            {
                "url": url,
                "video_id": entry_video_id or None,
                "output_prefix": entry_prefix,
            }
        ],
        output_dir=output_dir,
        config=config,
        timeout_seconds=720,
    )
    if not results:
        return None

    item = results[0]
    if not item.get("success"):
        logger.error(f"Transcription failed for {url}: {item.get('error', 'unknown error')}")
        return None

    transcript_text = item.get("transcript", "")
    if not transcript_text:
        txt_path = (item.get("files") or {}).get("txt")
        if txt_path and Path(txt_path).exists():
            transcript_text = Path(txt_path).read_text(encoding="utf-8")

    return {
        "success": True,
        "elapsed_seconds": float(item.get("elapsed_seconds") or 0.0),
        "transcript": transcript_text,
        "files": item.get("files", {}),
    }


def transcribe_videos_batch(
    videos: List[dict],
    output_dir: Path,
    config: dict,
) -> Dict[str, dict]:
    """
    Transcribe many videos in one transcriber process (single Whisper model load).

    Returns map: video_id -> transcribe result payload.
    """
    if not videos:
        return {}

    def _load_cached_transcript(video: dict) -> Optional[dict]:
        """Best-effort reuse of existing transcript artifacts to avoid re-transcription."""
        video_id = str(video.get("id") or "").strip()
        title = str(video.get("title") or "")
        prefixes: List[str] = []
        primary = slugify(f"{video_id}_{title}" if video_id else title)
        if primary:
            prefixes.append(primary)
        if video_id:
            vid_slug = slugify(video_id)
            if vid_slug and vid_slug not in prefixes:
                prefixes.append(vid_slug)

        candidates: List[Path] = []
        seen: set[str] = set()
        for pref in prefixes:
            for txt_file in sorted(output_dir.glob(f"{pref}*.txt"), key=lambda p: p.stat().st_mtime, reverse=True):
                key = str(txt_file.resolve())
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(txt_file)

        for txt_path in candidates:
            try:
                transcript_text = txt_path.read_text(encoding="utf-8")
            except Exception:
                continue
            if len((transcript_text or "").strip()) < 100:
                continue
            json_path = txt_path.with_suffix(".json")
            files = {"txt": str(txt_path), "json": str(json_path) if json_path.exists() else None}
            logger.info(f"Reusing cached transcript for {video_id or title}: {txt_path.name}")
            return {
                "success": True,
                "elapsed_seconds": 0.0,
                "transcript": transcript_text,
                "files": files,
                "error": None,
                "reused_cached_transcript": True,
            }
        return None

    entries: List[dict] = []
    mapped: Dict[str, dict] = {}
    for v in videos:
        video_id = str(v.get("id") or "").strip()
        cached = _load_cached_transcript(v)
        if cached is not None:
            mapped[video_id] = cached
            continue
        title = str(v.get("title") or "")
        prefix = slugify(f"{video_id}_{title}" if video_id else title)
        entries.append(
            {
                "url": v.get("url"),
                "video_id": video_id or None,
                "output_prefix": prefix,
            }
        )

    if not entries:
        return mapped

    batch_results = _run_transcriber_entries(entries=entries, output_dir=output_dir, config=config)
    if batch_results is None:
        # Hard failure: fall back to single-video calls.
        fallback: Dict[str, dict] = dict(mapped)
        entry_map: Dict[str, dict] = {str(v.get("id") or "").strip(): v for v in videos}
        for vid, v in entry_map.items():
            if vid in fallback:
                continue
            vid = str(v.get("id") or "")
            fallback[vid] = transcribe_video(
                url=v.get("url", ""),
                output_dir=output_dir,
                config=config,
                video_id=vid,
                video_title=v.get("title"),
            ) or {"success": False, "error": "fallback_failed"}
        return fallback

    for item in batch_results:
        vid = str(item.get("video_id") or "").strip()
        transcript_text = item.get("transcript", "")
        if not transcript_text:
            txt_path = (item.get("files") or {}).get("txt")
            if txt_path and Path(txt_path).exists():
                transcript_text = Path(txt_path).read_text(encoding="utf-8")
        mapped[vid] = {
            "success": bool(item.get("success")),
            "elapsed_seconds": float(item.get("elapsed_seconds") or 0.0),
            "transcript": transcript_text,
            "files": item.get("files", {}),
            "error": item.get("error"),
            "reused_cached_transcript": False,
        }
    return mapped


# ── Model Selection ────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    return len(text) // 4


def select_extraction_model(
    transcript_length: int,
    video_duration_sec: int,
    channel_priority: int,
    config: dict,
) -> Tuple[str, int]:
    """
    Select the best extraction model based on video/transcript characteristics.

    Returns (model_name, num_ctx).

    Logic:
    - Short (<5000 chars) or short video (<20min): phi4:14b, ctx=16384
    - Medium (5000-20000 chars) + high priority channel: glm-4.7-flash, ctx=32768
    - Long (>20000 chars) or long video (>=40min): glm-4.7-flash, ctx=65536
    """
    weekend_cfg = config.get("scanner", {}).get("weekend", {})
    short_model = weekend_cfg.get("extraction_model_short", "phi4:14b-q4_K_M")
    long_model = weekend_cfg.get("extraction_model_long", "glm-4.7-flash")
    escalation_threshold = weekend_cfg.get("extraction_model_escalation_threshold", 20000)

    video_duration_min = video_duration_sec / 60 if video_duration_sec else 0

    # Short video / short transcript
    if transcript_length < 5000 or (video_duration_min > 0 and video_duration_min < 20):
        return short_model, 16384

    # Long video or long transcript
    if transcript_length > escalation_threshold or video_duration_min >= 40:
        return long_model, 65536

    # Medium transcript + high priority channel
    if channel_priority <= 2:
        return long_model, 32768

    # Medium transcript + low priority channel
    return short_model, 16384


# ── LLM Extraction ─────────────────────────────────────────────────

def load_extraction_template(template_path: str, long_video: bool = False) -> Optional[str]:
    """Load and return the extraction prompt from a producer template.

    If long_video=True, appends the '## Long Video Addendum' section from the
    template (if present) to the base prompt for more thorough extraction.
    """
    full_path = PROJECT_ROOT / template_path
    if not full_path.exists():
        logger.error(f"Template not found: {full_path}")
        return None

    content = full_path.read_text(encoding="utf-8")

    # Extract the prompt between ```  blocks in the "Extraction Prompt" section
    # Look for the extraction prompt code block
    match = re.search(
        r"## Extraction Prompt\s*\n+```[^\n]*\n(.*?)```",
        content,
        re.DOTALL
    )

    prompt = None
    if match:
        prompt = match.group(1).strip()
    else:
        # Fallback: try to find any prompt-like section
        match = re.search(
            r"(?:EXTRACT THE FOLLOWING|You are analyzing)(.*?)(?:OUTPUT FORMAT|```)",
            content,
            re.DOTALL
        )
        if match:
            prompt = match.group(0).strip()
        else:
            logger.warning(f"Could not extract prompt from {template_path}, using full content")
            prompt = content

    # Append long video addendum if requested
    if long_video and prompt:
        addendum_match = re.search(
            r"## Long Video Addendum\s*\n(.*?)(?=\n## |\Z)",
            content,
            re.DOTALL,
        )
        if addendum_match:
            addendum = addendum_match.group(1).strip()
            prompt += f"\n\nADDITIONAL INSTRUCTIONS (LONG VIDEO - be extra thorough):\n{addendum}"
            logger.info("Applied long video addendum from template")
        else:
            prompt += (
                "\n\nADDITIONAL INSTRUCTIONS (LONG VIDEO):\n"
                "This is a longer-form video with more detail than usual. Be extra thorough:\n"
                "- Extract ALL specific levels, tickers, and data points mentioned\n"
                "- Capture forward-looking analysis and multi-day/week outlook\n"
                "- Include any deep-dive segments that provide unique insight\n"
                "- Do not truncate or summarize — extract comprehensively"
            )
            logger.info("Applied default long video addendum (template has no ## Long Video Addendum)")

    return prompt


def extract_with_llm(
    transcript: str,
    template_prompt: str,
    channel_name: str,
    video_title: str,
    video_date: str,
    config: dict,
    model_override: Optional[str] = None,
    num_ctx_override: Optional[int] = None,
) -> Optional[dict]:
    """
    Run transcript through the configured extraction provider with safe fallbacks.

    Returns structured extraction dict or None on failure.

    Args:
        model_override: Use this model instead of config default.
        num_ctx_override: Use this context window instead of config default.
    """
    ext_config = config.get("scanner", {}).get("extraction", {})
    model = model_override or ext_config.get("model", "qwen3:8b")
    extraction_provider = ext_config.get("extraction_provider", "ollama").strip().lower()
    requested_model = str(model)
    requested_provider = str(extraction_provider)
    ollama_url = ext_config.get("ollama_url", "http://localhost:11434")
    timeout = ext_config.get("timeout", 300)

    num_ctx = num_ctx_override or ext_config.get("num_ctx", 32768)
    fallback_model = ext_config.get("fallback_model", "qwen3:30b")
    local_fallback_model = ext_config.get("local_fallback_model", fallback_model)
    openai_fallback_enabled = _parse_bool(
        ext_config.get("openai_fallback_enabled", False),
        default=False,
    )
    openai_fallback_model = str(
        ext_config.get(
            "openai_fallback_extraction_model",
            ext_config.get("openai_fallback_model", "gpt-4.1"),
        )
    ).strip() or "gpt-4.1"

    # OpenRouter config (shared with synthesis)
    openrouter_url = ext_config.get("openrouter_url", "https://openrouter.ai/api/v1/chat/completions")
    openrouter_key_env = ext_config.get("openrouter_api_key_env", "OPENROUTER_API_KEY")
    openrouter_referer = ext_config.get("openrouter_referer")
    openrouter_title = ext_config.get("openrouter_title", "AutoTrade YouTube Summary")
    openrouter_max_tokens = int(ext_config.get("openrouter_max_tokens", 4096))
    openai_fallback_max_tokens = int(
        ext_config.get("openai_fallback_max_tokens", openrouter_max_tokens)
    )

    # Allow full transcripts for large-context cloud providers and known large local models.
    large_ctx_models = ("27b", "30b", "glm-4.7", "glm-4", "nemotron")
    if extraction_provider in {"openrouter", "openai", "gemini"}:
        max_transcript_chars = 50000
    else:
        max_transcript_chars = 50000 if any(m in model for m in large_ctx_models) else 15000
    prompt = template_prompt.replace("{transcript}", transcript[:max_transcript_chars])

    # Add metadata header
    header = f"""VIDEO METADATA:
- Channel: {channel_name}
- Title: {video_title}
- Date: {video_date}
- Extraction Time: {datetime.now().isoformat()}

"""
    full_prompt = header + prompt

    try:
        import requests
    except ImportError:
        raise RuntimeError("requests required. Run: pip install requests")

    logger.info(f"Extracting intelligence from {channel_name}: {video_title} (provider: {extraction_provider}, model: {model})")
    start = time.time()

    def _call_model_ollama(mdl: str, ctx: int) -> requests.Response:
        """Call Ollama with a specific model."""
        return requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": mdl,
                "messages": [{"role": "user", "content": full_prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_ctx": ctx,
                }
            },
            timeout=timeout,
        )

    def _call_model_openrouter(mdl: str) -> requests.Response:
        """Call OpenRouter with a specific model."""
        api_key = _get_env_var(str(openrouter_key_env)) or _get_env_var("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"Missing OpenRouter key. Set env var {openrouter_key_env}=<your_key> in .env"
            )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if openrouter_referer:
            headers["HTTP-Referer"] = str(openrouter_referer)
        if openrouter_title:
            headers["X-Title"] = str(openrouter_title)
        return requests.post(
            openrouter_url,
            headers=headers,
            json={
                "model": mdl,
                "messages": [{"role": "user", "content": full_prompt}],
                "temperature": 0.1,
                "max_tokens": openrouter_max_tokens,
            },
            timeout=timeout,
        )

    def _call_model_openai(mdl: str) -> tuple:
        """Call OpenAI with a specific model."""
        api_key = _get_env_var("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY for YouTube extraction fallback")

        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=mdl,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.1,
            max_tokens=openai_fallback_max_tokens,
        )
        payload = (
            response.model_dump()
            if hasattr(response, "model_dump")
            else response.to_dict()
        )
        content = ((payload.get("choices") or [{}])[0].get("message") or {}).get(
            "content", ""
        )
        return payload, content

    def _call_model_gemini(mdl: str) -> tuple:
        """Call Gemini with a specific model."""
        if genai is None:
            raise RuntimeError(
                "google-genai SDK not available. Run: pip install google-genai"
            )

        api_key = _get_env_var("GOOGLE_AI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GOOGLE_AI_API_KEY for Gemini extraction")

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=mdl,
            contents=full_prompt,
            config={
                "temperature": 0.1,
                "max_output_tokens": openrouter_max_tokens,
            },
        )
        payload = {
            "model": mdl,
            "usage": {
                "prompt_tokens": getattr(
                    response.usage_metadata, "prompt_token_count", 0
                ),
                "completion_tokens": getattr(
                    response.usage_metadata, "candidates_token_count", 0
                ),
                "total_tokens": getattr(
                    response.usage_metadata, "total_token_count", 0
                ),
            },
        }
        content = response.text or ""
        return payload, content

    def _call_model(mdl: str, ctx: int, provider_name: Optional[str] = None) -> tuple:
        """Call LLM via configured provider. Returns (response_data, content)."""
        active_provider = str(provider_name or extraction_provider).strip().lower()
        if active_provider == "openrouter":
            resp = _call_model_openrouter(mdl)
            resp.raise_for_status()
            payload = resp.json()
            content = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content", "")
            return payload, content
        if active_provider == "openai":
            return _call_model_openai(mdl)
        if active_provider == "gemini":
            return _call_model_gemini(mdl)
        if active_provider == "ollama":
            resp = _call_model_ollama(mdl, ctx)
            resp.raise_for_status()
            payload = resp.json()
            content = payload.get("message", {}).get("content", "")
            return payload, content
        raise RuntimeError(f"Unsupported extraction provider: {active_provider}")

    try:
        candidates: List[Tuple[str, str]] = []

        def _add_candidate(provider_name: str, model_name: Optional[str]) -> None:
            candidate_provider = str(provider_name or "").strip().lower()
            candidate_model = str(model_name or "").strip()
            if not candidate_provider or not candidate_model:
                return
            candidate = (candidate_provider, candidate_model)
            if candidate not in candidates:
                candidates.append(candidate)

        _add_candidate(extraction_provider, model)
        _add_candidate("ollama", local_fallback_model)
        _add_candidate("ollama", fallback_model)
        if openai_fallback_enabled:
            _add_candidate("openai", openai_fallback_model)

        data = {}
        content = ""
        used_provider = extraction_provider
        last_error: Optional[Exception] = None

        for idx, (candidate_provider, candidate_model) in enumerate(candidates):
            try:
                data, content = _call_model(
                    candidate_model,
                    num_ctx,
                    provider_name=candidate_provider,
                )
                if not str(content or "").strip():
                    raise RuntimeError(
                        f"{candidate_provider}:{candidate_model} returned empty content"
                    )
                model = candidate_model
                used_provider = candidate_provider
                break
            except Exception as candidate_err:
                last_error = candidate_err
                if idx < len(candidates) - 1:
                    next_provider, next_model = candidates[idx + 1]
                    logger.warning(
                        f"Model {candidate_model} failed via {candidate_provider}: "
                        f"{_safe_exception_text(candidate_err)}, "
                        f"falling back to {next_provider}:{next_model}"
                    )
                    continue
                raise last_error
        
        elapsed = time.time() - start
        logger.info(f"Extraction complete in {elapsed:.1f}s")
        
        # Try to parse JSON from response
        extraction = _extract_json(content)
        
        if extraction:
            extraction["_meta"] = {
                "channel": channel_name,
                "video_title": video_title,
                "video_date": video_date,
                "extracted_at": datetime.now().isoformat(),
                "model": model,
                "provider": used_provider,
                "requested_model": requested_model,
                "requested_provider": requested_provider,
                "elapsed_seconds": elapsed,
            }
            return extraction
        else:
            # Return raw text if JSON parsing fails
            return {
                "raw_extraction": content,
                "_meta": {
                    "channel": channel_name,
                    "video_title": video_title,
                    "video_date": video_date,
                    "extracted_at": datetime.now().isoformat(),
                    "model": model,
                    "provider": used_provider,
                    "requested_model": requested_model,
                    "requested_provider": requested_provider,
                    "elapsed_seconds": elapsed,
                    "json_parse_failed": True,
                }
            }
            
    except Exception as e:
        logger.error("LLM extraction failed: %s", _safe_exception_text(e))
        # Fallback payload: persist transcript context so downstream synthesis
        # can still use this video and we avoid endless reprocessing loops.
        return {
            "raw_extraction": "",
            "transcript_context": transcript[:12000],
            "_meta": {
                "channel": channel_name,
                "video_title": video_title,
                "video_date": video_date,
                "extracted_at": datetime.now().isoformat(),
                "model": requested_model,
                "provider": requested_provider,
                "requested_model": requested_model,
                "requested_provider": requested_provider,
                "elapsed_seconds": time.time() - start,
                "extraction_failed": True,
                "error": _safe_exception_text(e),
            },
        }


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON from LLM response."""
    # Strip think tags if present (qwen3 thinking)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    # Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    
    # From code block
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass
    
    # Find JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass
    
    return None


class ReportSchemaValidationError(ValueError):
    """Raised when synthesis returns text that cannot satisfy report schema."""

    def __init__(self, message: str, *, parseable_json: bool = False):
        super().__init__(message)
        self.parseable_json = bool(parseable_json)


def _normalize_report_schema_defaults(data: Optional[dict]) -> Optional[dict]:
    """Fill non-critical report keys that downstream consumers expect."""
    if not isinstance(data, dict):
        return data

    trading_signals = data.get("trading_signals")
    if not isinstance(trading_signals, dict):
        trading_signals = {}
        data["trading_signals"] = trading_signals

    sector_bias = trading_signals.get("sector_bias")
    if not isinstance(sector_bias, list):
        trading_signals["sector_bias"] = []

    return data


def _missing_required_report_keys(
    data: Optional[dict],
    required_keys: Optional[List[str]] = None,
) -> List[str]:
    if not isinstance(data, dict):
        return list(required_keys or [])
    missing: List[str] = []
    for key in required_keys or []:
        if "." in key:
            parts = key.split(".")
            curr: Any = data
            for part in parts:
                if not isinstance(curr, dict) or part not in curr:
                    missing.append(key)
                    break
                curr = curr[part]
        elif key not in data:
            missing.append(key)
    return missing


def _build_conservative_report_from_extractions(
    *,
    date_str: str,
    extractions: dict,
    reason: str,
) -> dict:
    """Build a valid low-confidence report when provider synthesis is unusable."""
    channel_names = sorted(str(key) for key in (extractions or {}).keys())
    reason_text = str(reason or "synthesis_schema_fallback")
    return {
        "date": date_str,
        "executive_summary": (
            "Provider synthesis did not return the full report schema. "
            "Using available channel extractions with conservative risk settings."
        ),
        "market_regime": "NEUTRAL",
        "regime_confidence": 25,
        "regime_summary": (
            "Automated synthesis fallback; defer to raw channel extraction artifacts."
        ),
        "smallcap_health": {
            "status": "cautious",
            "iwm_assessment": "unavailable",
            "breadth_assessment": "unavailable",
            "rotation_signal": "unavailable",
        },
        "trading_signals": {
            "sizing_multiplier": 0.5,
            "sizing_rationale": reason_text,
            "sector_bias": [],
            "trigger_levels": {
                "spy": {"bull_above": 0, "bear_below": 0},
                "qqq": {"bull_above": 0, "bear_below": 0},
                "iwm": {"bull_above": 0, "bear_below": 0},
            },
            "time_alerts": [],
            "earnings_impact": [],
        },
        "consensus": {
            "themes": ["fallback_report"],
            "risks": [reason_text],
            "conflicts": [],
        },
        "trade_ideas": [],
        "inverse_etf_signal": {
            "signal": "none",
            "instruments": [],
            "reason": "fallback report",
        },
        "overnight_directives": [
            "Use conservative sizing until a full YouTube synthesis report is available.",
            "Review raw YouTube extraction artifacts before increasing risk.",
        ],
        "channel_agreement": {
            "bullish_channels": [],
            "bearish_channels": [],
            "neutral_channels": channel_names,
            "consensus_strength": "limited",
        },
        "_schema_fallback": True,
        "_schema_fallback_reason": reason_text,
    }


def _extract_and_validate_report_json(
    text: str,
    required_keys: Optional[List[str]] = None,
) -> Optional[dict]:
    """
    Extract JSON and verify presence of mandatory keys.
    Supports dot-notation for nested keys (e.g. 'trading_signals.sector_bias').
    """
    data = _extract_json(text)
    if not data:
        return None
    data = _normalize_report_schema_defaults(data)
    if not required_keys:
        return data

    if _missing_required_report_keys(data, required_keys):
        return None
    return data


def _normalize_parseable_report_content(content: str) -> Optional[str]:
    """Return normalized JSON content when provider output is parseable."""
    data = _extract_json(content)
    if not isinstance(data, dict):
        return None
    normalized = _normalize_report_schema_defaults(data)
    if not isinstance(normalized, dict):
        return None
    return json.dumps(normalized, ensure_ascii=False)


def _is_usable_extraction_payload(payload: Any) -> bool:
    """Return True when a stored extraction can be used for report synthesis."""
    if not isinstance(payload, dict):
        return bool(payload)
    meta = payload.get("_meta", {}) if isinstance(payload.get("_meta"), dict) else {}
    if bool(meta.get("extraction_failed")):
        return False
    raw_extraction = str(payload.get("raw_extraction", "") or "").strip()
    if raw_extraction:
        return True
    ignored_keys = {"_meta", "_source", "transcript_context", "raw_extraction"}
    return any(key not in ignored_keys for key in payload.keys())


# ── RAG Storage ────────────────────────────────────────────────────

def store_extraction(
    extraction: dict,
    channel_key: str,
    video_id: str,
    video_date: str,
    rag_dir: Path,
) -> Path:
    """
    Store extraction in RAG-ready format.
    
    Directory structure:
        data/youtube/rag/
            by_date/
                2026-02-05/
                    trade_brigade_abc123.json
                    click_capital_def456.json
            by_channel/
                trade_brigade/
                    2026-02-05_abc123.json
                    2026-02-04_xyz789.json
            daily_reports/
                2026-02-05_consolidated.json
    """
    # Parse date
    date_str = video_date
    if len(date_str) == 8:  # YYYYMMDD format
        date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    elif not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    
    # Store by date
    date_dir = rag_dir / "by_date" / date_str
    date_dir.mkdir(parents=True, exist_ok=True)
    date_file = date_dir / f"{channel_key}_{video_id}.json"
    
    with open(date_file, "w", encoding="utf-8") as f:
        json.dump(extraction, f, indent=2, ensure_ascii=False)
    
    # Store by channel
    channel_dir = rag_dir / "by_channel" / channel_key
    channel_dir.mkdir(parents=True, exist_ok=True)
    channel_file = channel_dir / f"{date_str}_{video_id}.json"
    
    with open(channel_file, "w", encoding="utf-8") as f:
        json.dump(extraction, f, indent=2, ensure_ascii=False)
    
    # 3. Index in vector store (NEW Phase 2)
    try:
        rag = get_rag_manager()
        rag.index_extraction(
            ticker="global", # Extractions are usually market-wide
            date_str=date_str,
            extraction=extraction
        )
    except Exception as e:
        logger.warning(
            "Failed to index extraction in vector store: %s",
            _safe_exception_text(e),
        )

    logger.info(f"Stored extraction: {date_file.name}")
    return date_file


# ── Daily Report Generation ───────────────────────────────────────

def generate_daily_report(
    date_str: str,
    rag_dir: Path,
    config: dict,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    output_filename: Optional[str] = None,
    save_report: bool = True,
) -> Optional[dict]:
    """
    Generate consolidated daily market intelligence report from all channel extractions.
    
    This is the report the agent consumes for decision-making.
    """
    from datetime import timedelta
    # Resolve target dates for the rolling window (today + prior evening)
    try:
        report_dt = datetime.strptime(date_str, "%Y-%m-%d")
        yesterday_str = (report_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    except ValueError:
        logger.error(f"Invalid date format: {date_str}")
        return None

    target_dirs = []
    d_today = rag_dir / "by_date" / date_str
    if d_today.exists():
        target_dirs.append((d_today, False))  # (path, is_yesterday)
    
    d_yesterday = rag_dir / "by_date" / yesterday_str
    if d_yesterday.exists():
        target_dirs.append((d_yesterday, True))

    if not target_dirs:
        logger.warning(f"No extractions found for {date_str} or {yesterday_str}")
        return None
    
    # Collect all usable extractions within the window.
    extractions: Dict[str, dict] = {}
    total_video_count = 0
    skipped_failed = 0
    skipped_stale = 0
    seen_video_ids = set()
    known_channels = list((config.get("channels", {}) or {}).keys())

    for date_dir, is_yesterday in target_dirs:
        for f in sorted(date_dir.glob("*.json")):
            channel_key, video_id = _split_channel_and_video_id(f.stem, known_channels=known_channels)
            
            # Avoid processing the same video twice if it appears in both folders (rare but possible)
            if video_id in seen_video_ids:
                continue

            with open(f, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            
            if not _is_usable_extraction_payload(payload):
                skipped_failed += 1
                continue

            # Filtering for prior day: only include evening updates (post-16:00 ET)
            if is_yesterday:
                meta = payload.get("_meta", {})
                extracted_at = meta.get("extracted_at")
                if extracted_at:
                    try:
                        # Handle potential ISO formats (with or without Z/offsets)
                        ext_dt = datetime.fromisoformat(extracted_at.replace("Z", "+00:00"))
                        # We use 16:00 (4 PM) as the cutoff for 'next day relevance'
                        if ext_dt.hour < 16:
                            skipped_stale += 1
                            continue
                    except ValueError:
                        pass # If timestamp is unparseable, we lean toward inclusion if it's a known channel
            
            seen_video_ids.add(video_id)
            total_video_count += 1

            channel_bucket = extractions.setdefault(
                channel_key,
                {
                    "channel": channel_key,
                    "videos": [],
                },
            )
            normalized_payload = payload if isinstance(payload, dict) else {"raw_extraction": payload}
            normalized_payload.setdefault(
                "_source",
                {
                    "video_id": video_id,
                    "file": f.name,
                    "date": date_dir.name,
                },
            )
            channel_bucket["videos"].append(normalized_payload)
    
    if skipped_stale:
        logger.debug(f"Skipped {skipped_stale} stale extractions from {yesterday_str} (pre-16:00)")

    
    if not extractions:
        if skipped_failed:
            logger.warning(
                f"No usable extractions for {date_str} (skipped_failed={skipped_failed})"
            )
        return None
    
    # Load prior day's report for continuity context
    prior_report = None
    try:
        prev_date = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        prev_file = rag_dir / "daily_reports" / f"{prev_date}_consolidated.json"
        if prev_file.exists():
            with open(prev_file, "r", encoding="utf-8") as fh:
                prior_report = json.load(fh)
            logger.info(f"Loaded prior day report: {prev_date}")
    except Exception:
        pass
    
    # Build synthesis prompt — use master prompt from youtube_daily_report.md if available
    ext_config = config.get("scanner", {}).get("extraction", {})
    synthesis_provider = str(provider_override or ext_config.get("synthesis_provider", "openrouter")).strip().lower()
    requested_provider = synthesis_provider
    synthesis_model = ext_config.get("synthesis_model", ext_config.get("model", "qwen3:30b"))
    report_model = model_override or ext_config.get("report_model", synthesis_model)
    ollama_url = ext_config.get("ollama_url", "http://localhost:11434")
    synthesis_timeout = ext_config.get("synthesis_timeout", 1200)
    synthesis_num_ctx = ext_config.get("synthesis_num_ctx", 65536)
    fallback = ext_config.get("fallback_model", "qwen3:30b")
    report_fallback = ext_config.get("report_fallback_model", fallback)
    local_fallback_model = ext_config.get("local_fallback_model", report_fallback)
    openai_fallback_enabled = _parse_bool(
        ext_config.get("openai_fallback_enabled", False),
        default=False,
    )
    openai_fallback_model = str(
        ext_config.get("openai_fallback_model", "gpt-4.1")
    ).strip() or "gpt-4.1"
    openrouter_url = ext_config.get("openrouter_url", "https://openrouter.ai/api/v1/chat/completions")
    openrouter_key_env = ext_config.get("openrouter_api_key_env", "OPENROUTER_API_KEY")
    openrouter_referer = ext_config.get("openrouter_referer")
    openrouter_title = ext_config.get("openrouter_title", "AutoTrade YouTube Summary")
    openrouter_max_tokens = int(ext_config.get("openrouter_max_tokens", 4096))
    openai_fallback_max_tokens = int(
        ext_config.get("openai_fallback_max_tokens", openrouter_max_tokens)
    )
    
    extraction_payload_chars = sum(len(json.dumps(v, ensure_ascii=False)) for v in extractions.values())
    extraction_payload_tokens = _estimate_tokens("".join(json.dumps(v, ensure_ascii=False) for v in extractions.values()))
    expected_output_tokens = _resolve_expected_output_tokens(
        ext_config=ext_config,
        provider=synthesis_provider,
        extraction_payload_tokens=extraction_payload_tokens,
        channel_count=len(extractions),
        video_count=total_video_count,
    )
    input_budget_tokens = _resolve_synthesis_input_budget_tokens(
        ext_config=ext_config,
        provider=synthesis_provider,
        model=report_model,
        output_reserve_override=expected_output_tokens,
    )
    dynamic_max_chars_per_channel = _resolve_dynamic_max_chars_per_channel(
        ext_config=ext_config,
        input_budget_tokens=input_budget_tokens,
        channel_count=len(extractions),
        video_count=total_video_count,
    )
    max_prior_chars = int(ext_config.get("synthesis_max_prior_chars", 6000))
    chunking_enabled = _parse_bool(ext_config.get("synthesis_chunking_enabled", True), default=True)
    chunk_threshold_ratio = float(ext_config.get("synthesis_chunk_threshold_ratio", 0.80))
    should_chunk = chunking_enabled and extraction_payload_tokens > int(input_budget_tokens * chunk_threshold_ratio)

    logger.info(
        f"Generating daily report with provider={synthesis_provider}, model={report_model} "
        f"(ctx: {synthesis_num_ctx}, timeout: {synthesis_timeout}s, channels: {list(extractions.keys())}, "
        f"videos={total_video_count}, input_budget_tokens~{input_budget_tokens}, "
        f"expected_output_tokens~{expected_output_tokens}, extraction_tokens~{extraction_payload_tokens}, "
        f"chunk={should_chunk})"
    )
    
    try:
        import requests

        def _synth_call_ollama(mdl: str, prompt_text: str) -> requests.Response:
            # Legacy local path retained for future fallback use.
            return requests.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": mdl,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_ctx": synthesis_num_ctx},
                },
                timeout=synthesis_timeout,
            )

        def _synth_call_openrouter(mdl: str, prompt_text: str) -> requests.Response:
            api_key = _get_env_var(str(openrouter_key_env)) or _get_env_var("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError(
                    f"Missing OpenRouter key. Set env var {openrouter_key_env}=<your_key> "
                    "or add it to .env before running youtube summary generation."
                )
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            if openrouter_referer:
                headers["HTTP-Referer"] = str(openrouter_referer)
            if openrouter_title:
                headers["X-Title"] = str(openrouter_title)
            return requests.post(
                openrouter_url,
                headers=headers,
                json={
                    "model": mdl,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "temperature": 0.1,
                    "max_tokens": openrouter_max_tokens,  # Let the model use full budget (reasoning + content)
                    "response_format": {"type": "json_object"},
                },
                timeout=synthesis_timeout,
            )

        def _synth_call_openai(mdl: str, prompt_text: str) -> tuple[dict, str]:
            api_key = _get_env_var("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("Missing OPENAI_API_KEY for YouTube report fallback")

            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=mdl,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.1,
                max_tokens=openai_fallback_max_tokens,
                response_format={"type": "json_object"},
            )
            payload = (
                response.model_dump()
                if hasattr(response, "model_dump")
                else response.to_dict()
            )
            content = ((payload.get("choices") or [{}])[0].get("message") or {}).get(
                "content", ""
            )
            return payload, content

        def _synth_call_gemini(mdl: str, prompt_text: str) -> tuple[dict, str]:
            if genai is None:
                raise RuntimeError("google-genai SDK not available. Run: pip install google-genai")
            
            api_key = _get_env_var("GOOGLE_AI_API_KEY")
            if not api_key:
                raise RuntimeError("Missing GOOGLE_AI_API_KEY for Gemini synthesis")

            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=mdl,
                contents=prompt_text,
                config={
                    "temperature": 0.1,
                    "max_output_tokens": openrouter_max_tokens,
                }
            )
            
            payload = {
                "model": mdl,
                "usage": {
                    "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                    "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                    "total_tokens": getattr(response.usage_metadata, "total_token_count", 0),
                }
            }
            content = response.text or ""
            return payload, content

        def _call_provider(
            mdl: str,
            prompt_text: str,
            provider_name: Optional[str] = None,
        ) -> tuple[dict, str]:
            active_provider = str(provider_name or synthesis_provider).strip().lower()
            if active_provider == "openrouter":
                response = _synth_call_openrouter(mdl, prompt_text)
                response.raise_for_status()
                payload = response.json()
                content = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content", "")
                return payload, content
            if active_provider == "openai":
                return _synth_call_openai(mdl, prompt_text)
            if active_provider == "gemini":
                return _synth_call_gemini(mdl, prompt_text)
            if active_provider == "ollama":
                response = _synth_call_ollama(mdl, prompt_text)
                response.raise_for_status()
                payload = response.json()
                content = payload.get("message", {}).get("content", "")
                return payload, content
            raise RuntimeError(f"Unsupported synthesis provider: {active_provider}")

        candidates: List[Tuple[str, str]] = []

        def _add_candidate(provider_name: str, model_name: Optional[str]) -> None:
            candidate_provider = str(provider_name or "").strip().lower()
            candidate_model = str(model_name or "").strip()
            if not candidate_provider or not candidate_model:
                return
            candidate = (candidate_provider, candidate_model)
            if candidate not in candidates:
                candidates.append(candidate)

        _add_candidate(synthesis_provider, report_model)
        _add_candidate("ollama", local_fallback_model)
        _add_candidate("ollama", report_fallback)
        if openai_fallback_enabled:
            _add_candidate("openai", openai_fallback_model)

        def _call_with_fallback(
            prompt_text: str,
            primary_model: str,
            required_keys: Optional[List[str]] = None,
        ) -> tuple[dict, str, str, str]:
            try:
                _ = primary_model
                first_schema_error: Optional[ReportSchemaValidationError] = None
                for idx, (candidate_provider, candidate_model) in enumerate(candidates):
                    try:
                        payload, content = _call_provider(
                            candidate_model,
                            prompt_text,
                            provider_name=candidate_provider,
                        )
                        if not content.strip():
                            usage = payload.get("usage") or {}
                            reasoning_tok = (
                                (usage.get("completion_tokens_details") or {}).get(
                                    "reasoning_tokens", 0
                                )
                            )
                            raise RuntimeError(
                                f"empty content (reasoning_tokens={reasoning_tok})"
                            )
                        
                        # Validate structure
                        data = _extract_and_validate_report_json(
                            content,
                            required_keys=required_keys,
                        )
                        if not data:
                            parsed_content = _extract_json(content)
                            missing_keys = _missing_required_report_keys(
                                parsed_content,
                                required_keys,
                            )
                            normalized_content = _normalize_parseable_report_content(
                                content
                            )
                            if normalized_content is not None:
                                normalized_data = _extract_and_validate_report_json(
                                    normalized_content,
                                    required_keys=required_keys,
                                )
                                if normalized_data is not None:
                                    logger.warning(
                                        "Model %s (%s) omitted defaultable report keys; "
                                        "continuing with normalized schema defaults.",
                                        candidate_model,
                                        candidate_provider,
                                    )
                                    return (
                                        payload,
                                        normalized_content,
                                        candidate_model,
                                        candidate_provider,
                                    )
                                logger.warning(
                                    f"Model {candidate_model} ({candidate_provider}) failed schema validation. "
                                    f"Required: {missing_keys if missing_keys else required_keys}"
                                )
                                raise ReportSchemaValidationError(
                                    "Invalid JSON schema or missing keys: "
                                    f"{missing_keys or required_keys}",
                                    parseable_json=isinstance(parsed_content, dict),
                                )

                        return payload, content, candidate_model, candidate_provider
                    except Exception as candidate_err:
                        if (
                            first_schema_error is None
                            and isinstance(candidate_err, ReportSchemaValidationError)
                        ):
                            first_schema_error = candidate_err
                        if idx < len(candidates) - 1:
                            next_provider, next_model = candidates[idx + 1]
                            logger.warning(
                                f"Report model {candidate_model} failed via {candidate_provider}: "
                                f"{_safe_exception_text(candidate_err)}, "
                                f"falling back to {next_provider}:{next_model}"
                            )
                            continue
                        if first_schema_error is not None:
                            raise first_schema_error from None
                        raise
            except Exception:
                raise

        effective_extractions = extractions
        chunk_passes = 0
        chunk_models_used: List[str] = []
        chunk_providers_used: List[str] = []
        if should_chunk:
            # Build chunks dynamically based on available input budget.
            chunk_target_tokens = int(
                ext_config.get(
                    "synthesis_chunk_target_tokens",
                    max(2500, int(input_budget_tokens * 0.45)),
                )
            )
            channel_items: List[tuple[str, dict, int]] = []
            for ck, ext in extractions.items():
                videos = ext.get("videos") if isinstance(ext, dict) else None
                if isinstance(videos, list) and videos:
                    for idx, video_ext in enumerate(videos, start=1):
                        chunk_payload = {
                            "channel": ck,
                            "video_index": idx,
                            "video_count": len(videos),
                            "extraction": video_ext,
                        }
                        raw = json.dumps(chunk_payload, ensure_ascii=False)
                        channel_items.append((ck, chunk_payload, _estimate_tokens(raw)))
                else:
                    raw = json.dumps(ext, ensure_ascii=False)
                    channel_items.append((ck, ext, _estimate_tokens(raw)))

            channel_items.sort(key=lambda row: row[2], reverse=True)

            chunks: List[List[tuple[str, dict, int]]] = []
            current: List[tuple[str, dict, int]] = []
            current_tokens = 0
            for item in channel_items:
                if current and (current_tokens + item[2]) > chunk_target_tokens:
                    chunks.append(current)
                    current = []
                    current_tokens = 0
                current.append(item)
                current_tokens += item[2]
            if current:
                chunks.append(current)

            chunk_summaries: Dict[str, dict] = {}
            for idx, chunk in enumerate(chunks, start=1):
                chunk_data = [
                    {"channel": ck, "payload": ext}
                    for ck, ext, _tok in chunk
                ]
                chunk_prompt = (
                    "You are summarizing a subset of daily YouTube channel extractions. "
                    "Return strict JSON only with keys: "
                    "executive_summary, market_regime, regime_confidence, "
                    "smallcap_health, trading_signals, consensus, trade_ideas, risks, overnight_directives.\n\n"
                    f"DATE: {date_str}\n"
                    f"CHUNK: {idx}/{len(chunks)}\n"
                    f"ITEMS: {len(chunk_data)}\n\n"
                    f"DATA:\n{json.dumps(chunk_data, ensure_ascii=False, indent=2)}"
                )
                payload_chunk, content_chunk, chunk_model, chunk_provider = _call_with_fallback(
                    chunk_prompt, 
                    report_model,
                    required_keys=[],
                )
                _ = payload_chunk
                chunk_models_used.append(chunk_model)
                chunk_providers_used.append(chunk_provider)
                parsed_chunk = _extract_json(content_chunk)
                chunk_summaries[f"chunk_{idx}"] = parsed_chunk
                chunk_passes += 1

            if chunk_summaries:
                effective_extractions = chunk_summaries

        synthesis_prompt = _build_synthesis_prompt(
            effective_extractions,
            date_str,
            prior_report,
            max_chars_per_channel=dynamic_max_chars_per_channel,
            max_prior_chars=max_prior_chars,
        )

        # sector_bias is defaultable. Do not fail provider synthesis just because
        # a parseable report omitted it; normalization fills conservative defaults.
        try:
            payload, content, used_model, used_provider = _call_with_fallback(
                synthesis_prompt,
                report_model,
                required_keys=[],
            )
        except ReportSchemaValidationError as schema_err:
            if (
                not getattr(schema_err, "parseable_json", False)
                and synthesis_provider != "gemini"
            ):
                raise
            logger.warning(
                "Report synthesis failed schema validation; "
                "using conservative fallback report: %s",
                _safe_exception_text(schema_err),
            )
            report_fallback = _build_conservative_report_from_extractions(
                date_str=date_str,
                extractions=extractions,
                reason=_safe_exception_text(schema_err),
            )
            payload = {}
            content = json.dumps(report_fallback, ensure_ascii=False)
            used_model = f"{report_model}:schema_fallback"
            used_provider = f"{synthesis_provider}_schema_fallback"
        served_model = ""
        if isinstance(payload, dict) and payload:
            served_model = str(payload.get("model") or "")
        
        report = _normalize_report_schema_defaults(_extract_json(content) or {})

        # Salvage: if synthesis returned prose (or near-empty JSON), reconstruct
        # structured fields from the markdown body via local Ollama so the persisted
        # file is usable by downstream consumers without re-extraction at every load.
        structured_fields = ("market_regime", "regime_confidence", "executive_summary",
                             "overnight_directives", "smallcap_health")
        has_structure = any(k in (report or {}) for k in structured_fields)
        ts = (report or {}).get("trading_signals") or {}
        has_signals = bool(ts.get("sector_bias") or ts.get("sizing_multiplier"))
        if (not has_structure) and (not has_signals) and content and content.strip():
            try:
                from autotrade.utils.market_intelligence import (
                    _convert_markdown_to_json_ollama,
                )
                salvaged = _convert_markdown_to_json_ollama(content)
            except Exception as salvage_err:  # noqa: BLE001
                logger.warning("Markdown salvage failed: %s", _safe_exception_text(salvage_err))
                salvaged = None
            if isinstance(salvaged, dict) and any(k in salvaged for k in structured_fields):
                logger.info("Salvaged %d structured fields from markdown synthesis output",
                            sum(1 for k in structured_fields if k in salvaged))
                for k, v in salvaged.items():
                    if k not in ("_meta", "raw_report"):
                        report[k] = v
                report = _normalize_report_schema_defaults(report)
                report["_salvaged_from_markdown"] = True

        if not report:
            raise ReportSchemaValidationError(
                "Synthesis failed to produce parseable JSON after validation."
            )
        
        report["_meta"] = {
            "date": date_str,
            "channels_included": list(extractions.keys()),
            "generated_at": datetime.now().isoformat(),
            "model": used_model,
            "requested_model": report_model,
            "requested_provider": requested_provider,
            "served_model": served_model,
            "fallback_used": bool(used_model != report_model),
            "provider": used_provider,
            "num_ctx": synthesis_num_ctx,
            "video_count": total_video_count,
            "input_budget_tokens": input_budget_tokens,
            "expected_output_tokens": expected_output_tokens,
            "extraction_payload_chars": extraction_payload_chars,
            "extraction_payload_tokens_est": extraction_payload_tokens,
            "skipped_failed_extractions": skipped_failed,
            "chunking_applied": bool(chunk_passes),
            "chunk_passes": chunk_passes,
            "chunk_models": chunk_models_used,
            "chunk_providers": chunk_providers_used,
            "dynamic_max_chars_per_channel": dynamic_max_chars_per_channel,
        }

        # Keep the raw report body for agent regex fallback
        report["raw_report"] = content

        if used_provider in {"openrouter", "openai"}:
            usage = payload.get("usage") or {}
            report["_meta"]["token_usage"] = usage

        if save_report:
            report_dir = rag_dir / "daily_reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            file_name = output_filename or f"{date_str}_consolidated.json"
            report_file = report_dir / file_name
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Daily report saved: {report_file}")
        return report
        
    except ReportSchemaValidationError as e:
        logger.error("Report schema validation failed: %s", _safe_exception_text(e))
        raise
    except Exception as e:
        logger.error("Report generation failed: %s", _safe_exception_text(e))
        return None


def _build_synthesis_prompt(
    extractions: dict,
    date_str: str,
    prior_report: Optional[dict] = None,
    max_chars_per_channel: Optional[int] = None,
    max_prior_chars: Optional[int] = None,
) -> str:
    """Build the cross-channel synthesis prompt.
    
    Tries to load the comprehensive master prompt from prompts/llm/youtube_daily_report.md.
    Falls back to a built-in prompt if the file is unavailable.
    """
    # ── Prepare channel extraction data ───────────────────────────
    channel_summaries = []
    for channel_key, ext in extractions.items():
        ext_str = json.dumps(ext, indent=2, ensure_ascii=False)
        # Allow larger payloads — nemotron has 1M context
        if max_chars_per_channel and len(ext_str) > max_chars_per_channel:
            ext_str = ext_str[:max_chars_per_channel] + "\n... (truncated)"
        channel_summaries.append(f"### {channel_key}\n```json\n{ext_str}\n```")
    
    extractions_block = chr(10).join(channel_summaries)
    
    # ── Prior-day continuity context ──────────────────────────────
    prior_block = ""
    if prior_report:
        prior_str = json.dumps(prior_report, indent=2, ensure_ascii=False)
        if max_prior_chars and len(prior_str) > max_prior_chars:
            prior_str = prior_str[:max_prior_chars] + "\n... (truncated)"
        prior_block = f"""\n\nPRIOR DAY REPORT (for continuity — note changes and evolving themes):\n```json\n{prior_str}\n```\n"""
    
    # ── Try loading master prompt from file ───────────────────────
    master_prompt_path = PROJECT_ROOT / "prompts" / "llm" / "youtube_daily_report.md"
    if master_prompt_path.exists():
        try:
            master_content = master_prompt_path.read_text(encoding="utf-8")
            # Extract the synthesis prompt section
            # Look for content between "## Master Synthesis Prompt" and the next ## or end
            match = re.search(
                r"## (?:Master )?Synthesis Prompt\s*\n(.*?)(?=\n## |\Z)",
                master_content,
                re.DOTALL,
            )
            if match:
                prompt_template = match.group(1).strip()
                # Strip the surrounding ``` code block markers if present
                prompt_template = re.sub(r"^```[^\n]*\n", "", prompt_template)
                prompt_template = re.sub(r"\n```\s*$", "", prompt_template)
                # Replace placeholders with actual data
                prompt_template = prompt_template.replace("{date}", date_str)
                prompt_template = prompt_template.replace("{time_et}", datetime.now().strftime("%H:%M"))
                prompt_template = prompt_template.replace("{channel_count}", str(len(extractions)))
                prompt_template = prompt_template.replace("{extractions}", extractions_block)
                prompt_template = prompt_template.replace("{channel_extractions}", extractions_block)
                prompt_template = prompt_template.replace("{prior_day_report}", prior_block)
                logger.info("Loaded master synthesis prompt from youtube_daily_report.md")
                return prompt_template
        except Exception as e:
            logger.warning(
                "Failed to load master prompt: %s, using built-in",
                _safe_exception_text(e),
            )
    
    # ── Built-in fallback prompt ──────────────────────────────────
    return f"""You are the AutoTrade Market Intelligence Synthesizer.

DATE: {date_str}

You have extractions from {len(extractions)} YouTube trading channels. Each channel provides 
a different perspective on the market. Your job is to synthesize these into a single, 
actionable intelligence report for AutoTrade's trading agent.

CONTEXT:
- AutoTrade trades ~1600 small/mid-cap stocks ($2-$200 price range)
- We care about small-cap health MORE than mega-cap price action
- SPY/QQQ/IWM levels matter as REGIME indicators, not as trade targets  
- We need: risk regime, position sizing guidance, sector bias, and specific alerts

CHANNEL EXTRACTIONS:

{extractions_block}
{prior_block}
SYNTHESIZE INTO THIS JSON FORMAT:
{{
    "date": "{date_str}",
    "executive_summary": "2-3 sentence synthesis of all channels",
    "market_regime": "STRONG-RISK-ON | RISK-ON | LEAN-BULLISH | NEUTRAL | LEAN-BEARISH | RISK-OFF | CRASH",
    "regime_confidence": 0-100,
    "regime_summary": "1-2 sentence regime description",
    
    "smallcap_health": {{
        "status": "healthy | cautious | defensive | danger",
        "iwm_assessment": "from Trade Brigade if available",
        "breadth_assessment": "from Mike if available",
        "rotation_signal": "whether rotation is supporting or hurting small-caps"
    }},
    
    "trading_signals": {{
        "sizing_multiplier": 0.0-1.5,
        "sizing_rationale": "why this multiplier",
        "sector_bias": [{{
            "sector": "name",
            "bias": "overweight | neutral | underweight | avoid",
            "reason": "why"
        }}],
        "trigger_levels": {{
            "spy": {{"bull_above": 0, "bear_below": 0}},
            "qqq": {{"bull_above": 0, "bear_below": 0}},
            "iwm": {{"bull_above": 0, "bear_below": 0}}
        }},
        "time_alerts": ["economic event or data release to watch"],
        "earnings_impact": ["tickers with upcoming earnings that affect our universe"]
    }},
    
    "consensus": {{
        "themes": ["theme1", "theme2"],
        "risks": ["risk1", "risk2"],
        "conflicts": [{{"topic": "...", "bull_case": "...", "bear_case": "...", "resolution": "..."}}]
    }},
    
    "trade_ideas": [
        {{
            "ticker": "XYZ",
            "mentioned_by": ["channel1"],
            "direction": "long | short",
            "conviction": "high | medium | low",
            "in_our_universe": true,
            "setup": "description"
        }}
    ],
    
    "inverse_etf_signal": {{
        "signal": "none | consider | strong",
        "instruments": ["SH", "PSQ", "SQQQ"],
        "reason": "why or why not"
    }},
    
    "overnight_directives": [
        "Specific instruction 1 for the trading agent",
        "Specific instruction 2"
    ],
    
    "channel_agreement": {{
        "bullish_channels": [],
        "bearish_channels": [],
        "neutral_channels": [],
        "consensus_strength": "strong | moderate | mixed | contradictory"
    }}
}}

RULES:
1. sizing_multiplier is the MOST IMPORTANT output — it directly controls how much capital we risk
2. If channels DISAGREE, weight Trade Brigade for IWM/small-cap, RTA for VIX/regime, Mike for breadth
3. If ANY channel signals RISK-OFF or danger, sizing_multiplier must be <= 0.5
4. Cross-reference trade ideas across channels — consensus picks get HIGH conviction
5. ALWAYS include overnight_directives — these drive the next trading session
6. If only 1-2 channels available, note limited coverage and be more conservative
"""


# ── Main Scanner Pipeline ─────────────────────────────────────────

def _timing_stats(values: List[float]) -> dict:
    """Return compact timing stats for a list of durations in seconds."""
    clean = [float(v) for v in values if v is not None and v >= 0]
    if not clean:
        return {"count": 0, "min": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(clean)
    p95_idx = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1))
    return {
        "count": len(ordered),
        "min": round(ordered[0], 3),
        "median": round(statistics.median(ordered), 3),
        "p95": round(ordered[p95_idx], 3),
        "max": round(ordered[-1], 3),
    }


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _safe_processed_write(
    processed: dict,
    lock: Optional[threading.Lock],
    video_id: str,
    payload: dict,
    track_file: Optional[Path] = None,
) -> None:
    """Write to processed_videos safely when extraction is parallelized."""
    if lock:
        with lock:
            processed.setdefault("processed", {})[video_id] = payload
            if track_file is not None:
                save_processed_videos(track_file, processed)
    else:
        processed.setdefault("processed", {})[video_id] = payload
        if track_file is not None:
            save_processed_videos(track_file, processed)


def _split_channel_and_video_id(stem: str, known_channels: Optional[List[str]] = None) -> Tuple[str, str]:
    """
    Split filename stem into (channel_key, video_id) using the final underscore.
    """
    if known_channels:
        for ck in sorted([str(c) for c in known_channels if c], key=len, reverse=True):
            prefix = f"{ck}_"
            if stem.startswith(prefix):
                return ck, stem[len(prefix):]
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return stem, ""


def _resolve_expected_output_tokens(
    ext_config: dict,
    provider: str,
    extraction_payload_tokens: int,
    channel_count: int,
    video_count: int,
) -> int:
    """
    Resolve target output tokens for one synthesis call.
    """
    provider = (provider or "").strip().lower()
    min_tokens = int(ext_config.get("synthesis_output_tokens_min", 4096))  # JSON report needs at least ~2000 tokens + reasoning overhead
    max_tokens_default = int(ext_config.get("synthesis_output_tokens_max", 8192))

    if provider in {"openrouter", "openai", "gemini"}:
        # Use cloud-provider output cap, not the Ollama expected-output path.
        cloud_output_limit = ext_config.get(
            "openai_fallback_max_tokens",
            ext_config.get("openrouter_max_tokens", max_tokens_default),
        )
        max_tokens_default = int(ext_config.get("synthesis_max_output_tokens",
                                 cloud_output_limit))
    else:
        max_tokens_default = int(ext_config.get("ollama_expected_output_tokens", max_tokens_default))

    # Scale with both payload and number of videos/channels, then clamp.
    payload_scaled = int(extraction_payload_tokens * 0.28)
    structure_scaled = int(channel_count * 220 + video_count * 180)
    target = max(min_tokens, payload_scaled + structure_scaled)
    return max(min_tokens, min(max_tokens_default, target))


def _resolve_dynamic_max_chars_per_channel(
    ext_config: dict,
    input_budget_tokens: int,
    channel_count: int,
    video_count: int,
) -> int:
    """
    Dynamic per-channel prompt char cap based on current input budget and daily volume.
    """
    hard_cap = int(ext_config.get("synthesis_dynamic_max_chars_per_channel", 18000))
    hard_min = int(ext_config.get("synthesis_min_chars_per_channel", 2500))

    # Reserve ~25% for instruction/prompt scaffolding and continuity context.
    usable_prompt_chars = max(4000, int(input_budget_tokens * 4 * 0.75))
    divisor = max(1, channel_count + max(0, video_count - channel_count))
    dynamic = int(usable_prompt_chars / divisor)
    return max(hard_min, min(hard_cap, dynamic))


def _resolve_synthesis_input_budget_tokens(
    ext_config: dict,
    provider: str,
    model: str,
    output_reserve_override: Optional[int] = None,
) -> int:
    """
    Resolve input token budget for one synthesis session.

    Budget is dynamic: model context limit - expected output reserve.
    """
    provider = (provider or "").strip().lower()
    model_limits = ext_config.get("model_input_token_limits", {}) or {}
    output_reserve = int(output_reserve_override) if output_reserve_override else 2048
    model_limit = 32768

    if provider == "openrouter":
        model_limits = ext_config.get("openrouter_model_input_tokens", model_limits) or {}
        model_limit = int(model_limits.get(model, ext_config.get("openrouter_default_input_tokens", 32768)))
        if output_reserve_override is None:
            output_reserve = int(ext_config.get("openrouter_max_tokens", 4096))
    elif provider == "openai":
        model_limits = ext_config.get("openai_model_input_tokens", model_limits) or {}
        model_limit = int(
            model_limits.get(model, ext_config.get("openai_default_input_tokens", 120000))
        )
        if output_reserve_override is None:
            output_reserve = int(ext_config.get("openai_fallback_max_tokens", 8192))
    elif provider == "gemini":
        # Gemini has 1M context. We default to a safe 128k but can go higher.
        model_limit = int(ext_config.get("gemini_default_input_tokens", 128000))
        if output_reserve_override is None:
            output_reserve = int(ext_config.get("openrouter_max_tokens", 8192))
    else:
        # Ollama path.
        model_limit = int(model_limits.get(model, ext_config.get("synthesis_num_ctx", 65536)))
        if output_reserve_override is None:
            output_reserve = int(ext_config.get("ollama_expected_output_tokens", 2048))

    safety_margin = int(ext_config.get("synthesis_input_safety_margin_tokens", 1024))
    input_budget = max(4000, model_limit - output_reserve - safety_margin)
    return input_budget


def _session_report_date(now: Optional[datetime] = None) -> str:
    """
    Resolve the trading-session date for report generation/storage.

    Holiday-aware: delegates to ``autotrade.utils.market_time.get_pm_plan_date``
    so the youtube scanner agrees with PM-plan / EOD pipelines on the
    target session date. At 12:15 AM on a market holiday (MLK Day, Good
    Friday, Memorial Day, etc.) this rolls forward to the next actual
    trading session instead of keying the report under an orphan
    holiday date.
    """
    try:
        from autotrade.utils.market_time import get_pm_plan_date

        ts = now or datetime.now()
        return get_pm_plan_date(ts).strftime("%Y-%m-%d")
    except Exception:
        # Fallback to weekday-only resolution if market_time is unavailable
        # (e.g., partial install). Holiday handling lost but core flow ok.
        ts = now or datetime.now()
        if ts.weekday() >= 5:
            days_ahead = 7 - ts.weekday()
            return (ts + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        if ts.hour >= 16:
            nxt = ts + timedelta(days=1)
            while nxt.weekday() >= 5:
                nxt += timedelta(days=1)
            return nxt.strftime("%Y-%m-%d")
        return ts.strftime("%Y-%m-%d")


def _write_session_scan_status(
    *,
    rag_dir: Path,
    session_date: str,
    channel_status: Dict[str, Dict[str, Any]],
    channels_total: int,
) -> Optional[Path]:
    """
    Persist scan coverage/status so readiness checks can distinguish:
    - channel checked but no fresh uploads
    - channel never checked this session
    """
    if not session_date:
        return None
    try:
        rag_dir.mkdir(parents=True, exist_ok=True)
        status_file = rag_dir / f"session_scan_status_{session_date}.json"
        payload = {
            "session_date": session_date,
            "generated_at": datetime.now().isoformat(),
            "channels_total": int(channels_total),
            "channels_checked": sorted(channel_status.keys()),
            "channel_status": channel_status,
        }
        status_file.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return status_file
    except Exception as e:
        logger.warning(
            "Failed writing session scan status artifact: %s",
            _safe_exception_text(e),
        )
        return None


def _weekend_adjusted_max_age(configured_max_age_hours: int) -> int:
    """
    On weekends (Sat/Sun), extend max_age_hours so Friday videos are captured.

    Friday close is ~4 PM ET.  By Sunday 10 PM that's ~54 hours ago, easily
    beyond the default 36h window.  We extend to at least 96h (4 days) on
    weekends so the full Fri-Sat-Sun window is covered.  On weekdays the
    configured value is returned unchanged.
    """
    now = datetime.now()
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return max(configured_max_age_hours, 96)
    return configured_max_age_hours


def scan_channel(
    channel_key: str,
    channel_config: dict,
    scanner_config: dict,
    processed: dict,
    track_file: Optional[Path] = None,
    force: bool = False,
    videos_override: Optional[List[dict]] = None,
    processed_lock: Optional[threading.Lock] = None,
    timing_buckets: Optional[Dict[str, List[float]]] = None,
    session_date: Optional[str] = None,
    channel_status: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[dict]:
    """
    Scan a single channel for new videos, transcribe, and extract.
    
    Returns list of extraction results.
    """
    channel_name = channel_config["name"]
    channel_url = channel_config["channel_url"]
    template_path = channel_config["template"]
    max_age = _weekend_adjusted_max_age(channel_config.get("schedule", {}).get("max_age_hours", 36))
    output_dir = Path(scanner_config.get("output_dir", "data/youtube"))
    rag_dir = Path(scanner_config.get("rag_dir", "data/youtube/rag"))
    full_config = {"scanner": scanner_config}
    perf_cfg = scanner_config.get("performance", {})
    max_extraction_workers = max(1, int(perf_cfg.get("max_extraction_workers", 1)))
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Scanning: {channel_name} ({channel_key})")
    logger.info(f"{'='*60}")
    
    # 1. Get recent videos (can be pre-fetched in parallel by caller)
    scan_start = time.time()
    if videos_override is None:
        meta_start = time.time()
        videos = get_recent_videos(channel_url, max_age_hours=max_age)
        meta_elapsed = time.time() - meta_start
        if timing_buckets is not None:
            timing_buckets.setdefault("metadata_seconds", []).append(meta_elapsed)
    else:
        videos = videos_override
    meta_status = get_recent_videos_status(channel_url)
    recent_found_count = len(videos)
    channel_had_errors = bool(meta_status.get("had_errors"))
    logger.info(f"Found {len(videos)} recent videos")
    
    if not videos:
        metadata_degraded = bool(meta_status.get("metadata_degraded"))
        if metadata_degraded:
            logger.warning(
                "Metadata fetch degraded for %s: entries_seen=%s, missing_upload_dates=%s, detail_failures=%s",
                channel_key,
                meta_status.get("entries_seen", 0),
                meta_status.get("missing_upload_dates", 0),
                meta_status.get("detail_failures", 0),
            )
        if channel_status is not None:
            channel_status[channel_key] = {
                "checked_at": datetime.now().isoformat(),
                "recent_found_count": int(recent_found_count),
                "raw_entry_count": int(meta_status.get("entries_seen", recent_found_count)),
                "new_candidate_count": 0,
                "new_processed_count": 0,
                "had_errors": bool(channel_had_errors),
                "metadata_degraded": metadata_degraded,
                "detail_failures": int(meta_status.get("detail_failures", 0)),
                "used_cookie_fallback": bool(meta_status.get("used_cookie_fallback", False)),
            }
        if timing_buckets is not None:
            timing_buckets.setdefault("channel_total_seconds", []).append(time.time() - scan_start)
        return []
    
    # 2. Filter already processed
    if not force:
        videos = filter_new_videos(
            videos=videos,
            processed=processed,
            channel_key=channel_key,
            rag_dir=rag_dir,
        )
        logger.info(f"{len(videos)} new videos to process")
    new_candidate_count = len(videos)

    if not videos:
        logger.info("All videos already processed")
        if channel_status is not None:
            channel_status[channel_key] = {
                "checked_at": datetime.now().isoformat(),
                "recent_found_count": int(recent_found_count),
                "raw_entry_count": int(meta_status.get("entries_seen", recent_found_count)),
                "new_candidate_count": int(new_candidate_count),
                "new_processed_count": 0,
                "had_errors": bool(channel_had_errors),
                "metadata_degraded": bool(meta_status.get("metadata_degraded", False)),
                "detail_failures": int(meta_status.get("detail_failures", 0)),
                "used_cookie_fallback": bool(meta_status.get("used_cookie_fallback", False)),
            }
        if timing_buckets is not None:
            timing_buckets.setdefault("channel_total_seconds", []).append(time.time() - scan_start)
        return []
    
    # 3. Load extraction template
    template = load_extraction_template(template_path)
    if not template:
        logger.error(f"Failed to load template: {template_path}")
        if channel_status is not None:
            channel_status[channel_key] = {
                "checked_at": datetime.now().isoformat(),
                "recent_found_count": int(recent_found_count),
                "raw_entry_count": int(meta_status.get("entries_seen", recent_found_count)),
                "new_candidate_count": int(new_candidate_count),
                "new_processed_count": 0,
                "had_errors": True,
                "metadata_degraded": bool(meta_status.get("metadata_degraded", False)),
                "detail_failures": int(meta_status.get("detail_failures", 0)),
                "used_cookie_fallback": bool(meta_status.get("used_cookie_fallback", False)),
            }
        if timing_buckets is not None:
            timing_buckets.setdefault("channel_total_seconds", []).append(time.time() - scan_start)
        return []
    
    results = []

    # 4. Transcribe in one process for this channel (single model load)
    tx_batch_start = time.time()
    tx_map = transcribe_videos_batch(videos=videos, output_dir=output_dir, config=full_config)
    tx_batch_elapsed = time.time() - tx_batch_start
    if timing_buckets is not None:
        timing_buckets.setdefault("transcribe_batch_seconds", []).append(tx_batch_elapsed)

    ready_for_extraction: List[tuple[dict, dict, str, str]] = []
    for video in videos:
        video_id = video["id"]
        video_title = video["title"]
        video_date = video.get("upload_date", datetime.now().strftime("%Y%m%d"))
        tx_result = tx_map.get(video_id) or {"success": False, "error": "missing_batch_result"}

        per_video_tx_elapsed = float(tx_result.get("elapsed_seconds") or 0.0)
        if timing_buckets is not None and per_video_tx_elapsed > 0:
            timing_buckets.setdefault("transcribe_seconds", []).append(per_video_tx_elapsed)

        if not tx_result.get("success"):
            channel_had_errors = True
            logger.error(f"Transcription failed for {video_title}: {tx_result.get('error', 'unknown')}")
            _safe_processed_write(
                processed,
                processed_lock,
                video_id,
                {
                    "channel": channel_key,
                    "title": video_title,
                    "date": video_date,
                    "status": "transcription_failed",
                    "timestamp": datetime.now().isoformat(),
                    "error": tx_result.get("error"),
                },
                track_file=track_file,
            )
            continue

        transcript = str(tx_result.get("transcript") or "")
        if len(transcript.strip()) < 100:
            channel_had_errors = True
            logger.warning(f"Transcript too short ({len(transcript)} chars), skipping extraction")
            _safe_processed_write(
                processed,
                processed_lock,
                video_id,
                {
                    "channel": channel_key,
                    "title": video_title,
                    "date": video_date,
                    "status": "transcript_too_short",
                    "timestamp": datetime.now().isoformat(),
                    "files": tx_result.get("files", {}),
                },
                track_file=track_file,
            )
            continue

        ready_for_extraction.append((video, tx_result, transcript, video_date))

    def _extract_one(item: tuple[dict, dict, str, str]) -> tuple[dict, dict, Optional[dict], float]:
        video, tx_result, transcript, video_date = item
        started = time.time()
        extraction = extract_with_llm(
            transcript=transcript,
            template_prompt=template,
            channel_name=channel_name,
            video_title=video["title"],
            video_date=video_date,
            config=full_config,
        )
        return video, tx_result, extraction, time.time() - started

    extraction_outputs: List[tuple[dict, dict, Optional[dict], float]] = []
    if ready_for_extraction:
        if max_extraction_workers == 1:
            for item in ready_for_extraction:
                extraction_outputs.append(_extract_one(item))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_extraction_workers) as pool:
                futures = [pool.submit(_extract_one, item) for item in ready_for_extraction]
                for fut in concurrent.futures.as_completed(futures):
                    extraction_outputs.append(fut.result())

    for video, tx_result, extraction, extract_elapsed in extraction_outputs:
        video_id = video["id"]
        video_title = video["title"]
        video_date = video.get("upload_date", datetime.now().strftime("%Y%m%d"))

        if timing_buckets is not None:
            timing_buckets.setdefault("extract_seconds", []).append(extract_elapsed)

        if extraction:
            extraction_meta = extraction.get("_meta", {}) if isinstance(extraction, dict) else {}
            extraction_failed = bool(extraction_meta.get("extraction_failed"))
            store_extraction(
                extraction=extraction,
                channel_key=channel_key,
                video_id=video_id,
                video_date=session_date or video_date,
                rag_dir=rag_dir,
            )
            results.append(extraction)
            _safe_processed_write(
                processed,
                processed_lock,
                video_id,
                {
                    "channel": channel_key,
                    "title": video_title,
                    "date": video_date,
                    "status": "extraction_failed" if extraction_failed else "complete",
                    "timestamp": datetime.now().isoformat(),
                    "files": tx_result.get("files", {}),
                    "error": extraction_meta.get("error") if extraction_failed else None,
                },
                track_file=track_file,
            )
        else:
            channel_had_errors = True
            _safe_processed_write(
                processed,
                processed_lock,
                video_id,
                {
                    "channel": channel_key,
                    "title": video_title,
                    "date": video_date,
                    "status": "extraction_failed",
                    "timestamp": datetime.now().isoformat(),
                },
                track_file=track_file,
            )

    if channel_status is not None:
        channel_status[channel_key] = {
            "checked_at": datetime.now().isoformat(),
            "recent_found_count": int(recent_found_count),
            "raw_entry_count": int(meta_status.get("entries_seen", recent_found_count)),
            "new_candidate_count": int(new_candidate_count),
            "new_processed_count": int(len(results)),
            "had_errors": bool(channel_had_errors),
            "metadata_degraded": bool(meta_status.get("metadata_degraded", False)),
            "detail_failures": int(meta_status.get("detail_failures", 0)),
            "used_cookie_fallback": bool(meta_status.get("used_cookie_fallback", False)),
        }

    if timing_buckets is not None:
        timing_buckets.setdefault("channel_total_seconds", []).append(time.time() - scan_start)
    return results


def run_full_scan(
    config: dict,
    channel_filter: Optional[str] = None,
    force: bool = False,
    report_only: bool = False,
    report_provider_override: Optional[str] = None,
    report_model_override: Optional[str] = None,
    report_output_filename: Optional[str] = None,
) -> dict:
    """
    Run the full scanning pipeline across all (or specified) channels.
    
    Returns summary dict.
    """
    channels = config.get("channels", {})
    scanner_config = config.get("scanner", {})
    
    rag_dir = Path(scanner_config.get("rag_dir", "data/youtube/rag"))
    track_file = Path(scanner_config.get("dedup", {}).get(
        "track_file", "data/youtube/processed_videos.json"
    ))
    
    today = datetime.now().strftime("%Y-%m-%d")
    session_date = _session_report_date()
    
    if report_only:
        logger.info("Report-only mode: regenerating daily report from existing extractions")
        report = generate_daily_report(
            session_date,
            rag_dir,
            config,
            provider_override=report_provider_override,
            model_override=report_model_override,
            output_filename=report_output_filename,
        )
        if report is None and session_date != today:
            report = generate_daily_report(
                today,
                rag_dir,
                config,
                provider_override=report_provider_override,
                model_override=report_model_override,
                output_filename=report_output_filename,
            )
        return {
            "mode": "report_only",
            "date": session_date,
            "report": report,
        }
    
    # Load dedup tracking
    processed = load_processed_videos(track_file)
    
    # Filter channels if specified
    if channel_filter:
        if channel_filter not in channels:
            logger.error(f"Unknown channel: {channel_filter}. Available: {list(channels.keys())}")
            return {"error": f"Unknown channel: {channel_filter}"}
        channels = {channel_filter: channels[channel_filter]}
    
    # Sort by priority (lower = higher priority)
    sorted_channels = sorted(channels.items(), key=lambda x: x[1].get("priority", 99))
    perf_cfg = scanner_config.get("performance", {})
    max_channel_workers = max(1, int(perf_cfg.get("max_channel_workers", 4)))
    timings: Dict[str, List[float]] = {}
    processed_lock = threading.Lock()
    channel_scan_status: Dict[str, Dict[str, Any]] = {}

    # Stage A: metadata fetch in parallel
    stage_meta_start = time.time()
    pre_scanned: Dict[str, List[dict]] = {}

    def _fetch_channel_videos(item: tuple[str, dict]) -> tuple[str, List[dict], float]:
        ck, cfg = item
        channel_url = cfg["channel_url"]
        max_age = _weekend_adjusted_max_age(cfg.get("schedule", {}).get("max_age_hours", 36))
        started = time.time()
        vids = get_recent_videos(channel_url, max_age_hours=max_age)
        return ck, vids, time.time() - started

    if len(sorted_channels) == 1 or max_channel_workers == 1:
        for item in sorted_channels:
            ck, vids, elapsed = _fetch_channel_videos(item)
            pre_scanned[ck] = vids
            timings.setdefault("metadata_seconds", []).append(elapsed)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_channel_workers) as pool:
            future_map = {
                pool.submit(_fetch_channel_videos, item): item[0]
                for item in sorted_channels
            }
            for fut in concurrent.futures.as_completed(future_map):
                ck = future_map[fut]
                try:
                    _, vids, elapsed = fut.result()
                    pre_scanned[ck] = vids
                    timings.setdefault("metadata_seconds", []).append(elapsed)
                except Exception as e:
                    logger.error(
                        "Failed metadata scan for %s: %s",
                        ck,
                        _safe_exception_text(e),
                    )
                    pre_scanned[ck] = []
                    timings.setdefault("metadata_seconds", []).append(0.0)
    timings.setdefault("metadata_stage_seconds", []).append(time.time() - stage_meta_start)

    all_results = {}
    total_new = 0
    total_processed = 0
    
    for channel_key, channel_cfg in sorted_channels:
        results = scan_channel(
            channel_key=channel_key,
            channel_config=channel_cfg,
            scanner_config=scanner_config,
            processed=processed,
            track_file=track_file,
            force=force,
            videos_override=pre_scanned.get(channel_key),
            processed_lock=processed_lock,
            timing_buckets=timings,
            session_date=session_date,
            channel_status=channel_scan_status,
        )
        
        all_results[channel_key] = results
        total_new += len(results)
        total_processed += 1
    
    # Save dedup tracking
    save_processed_videos(track_file, processed)
    
    metadata_failed_channels = sorted(
        channel_key
        for channel_key, status in channel_scan_status.items()
        if bool((status or {}).get("metadata_degraded"))
    )
    scan_degraded = bool(metadata_failed_channels)

    # Generate daily report if we got any new data and the scan itself did not degrade.
    report = None
    should_generate_report = bool(
        total_new > 0
        or (
            config.get("scanner", {}).get("report", {}).get("generate_daily_report", True)
            and not scan_degraded
        )
    )
    if should_generate_report:
        report = generate_daily_report(
            session_date,
            rag_dir,
            config,
            provider_override=report_provider_override,
            model_override=report_model_override,
            output_filename=report_output_filename,
        )
        if report is None and session_date != today:
            report = generate_daily_report(
                today,
                rag_dir,
                config,
                provider_override=report_provider_override,
                model_override=report_model_override,
                output_filename=report_output_filename,
            )

    summary = {
        "date": session_date,
        "channels_scanned": total_processed,
        "new_videos_processed": total_new,
        "channels": {k: len(v) for k, v in all_results.items()},
        "channel_scan_status": channel_scan_status,
        "metadata_failed_channels": metadata_failed_channels,
        "scan_degraded": scan_degraded,
        "report_generated": report is not None,
        "performance": {
            "metadata_seconds": _timing_stats(timings.get("metadata_seconds", [])),
            "transcribe_batch_seconds": _timing_stats(timings.get("transcribe_batch_seconds", [])),
            "transcribe_seconds": _timing_stats(timings.get("transcribe_seconds", [])),
            "extract_seconds": _timing_stats(timings.get("extract_seconds", [])),
            "channel_total_seconds": _timing_stats(timings.get("channel_total_seconds", [])),
        },
    }

    scan_status_file = _write_session_scan_status(
        rag_dir=rag_dir,
        session_date=session_date,
        channel_status=channel_scan_status,
        channels_total=len(sorted_channels),
    )
    if scan_status_file is not None:
        summary["session_scan_status_file"] = str(scan_status_file)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"SCAN COMPLETE: {total_new} new videos processed across {total_processed} channels")
    logger.info(f"{'='*60}")
    
    return summary


def list_pending(config: dict) -> None:
    """Show videos that haven't been processed yet."""
    channels = config.get("channels", {})
    scanner_config = config.get("scanner", {})
    rag_dir = Path(scanner_config.get("rag_dir", "data/youtube/rag"))
    perf_cfg = scanner_config.get("performance", {})
    max_channel_workers = max(1, int(perf_cfg.get("max_channel_workers", 4)))
    track_file = Path(scanner_config.get("dedup", {}).get(
        "track_file", "data/youtube/processed_videos.json"
    ))
    
    processed = load_processed_videos(track_file)
    
    print(f"\n{'='*70}")
    print("  PENDING VIDEOS (not yet processed)")
    print(f"{'='*70}")
    
    total_pending = 0
    
    channel_items = list(channels.items())

    def _fetch(item: tuple[str, dict]) -> tuple[str, dict, List[dict]]:
        channel_key, channel_cfg = item
        channel_url = channel_cfg["channel_url"]
        max_age = channel_cfg.get("schedule", {}).get("max_age_hours", 36)
        videos = get_recent_videos(channel_url, max_age_hours=max_age)
        return channel_key, channel_cfg, videos

    fetched: List[tuple[str, dict, List[dict]]] = []
    if len(channel_items) == 1 or max_channel_workers == 1:
        fetched = [_fetch(item) for item in channel_items]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_channel_workers) as pool:
            futures = [pool.submit(_fetch, item) for item in channel_items]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    fetched.append(fut.result())
                except Exception as e:
                    logger.error(
                        "Pending fetch error: %s",
                        _safe_exception_text(e),
                    )

    fetched.sort(key=lambda row: row[1].get("priority", 99))
    for channel_key, channel_cfg, videos in fetched:
        new = filter_new_videos(videos, processed, channel_key, rag_dir=rag_dir)
        
        if new:
            print(f"\n  [{_safe_console_text(channel_cfg['name'])}] ({len(new)} pending)")
            for v in new:
                dur = f" ({v['duration']//60}m)" if v.get('duration') else ""
                print(f"    - {_safe_console_text(v['title'])}{dur}")
                print(f"      {_safe_console_text(v['url'])}")
            total_pending += len(new)
        else:
            print(f"\n  [{_safe_console_text(channel_cfg['name'])}] [OK] Up to date")
    
    print(f"\n{'='*70}")
    print(f"  Total pending: {total_pending}")
    print(f"{'='*70}\n")


# ── CLI Entry Point ────────────────────────────────────────────────

def main():
    _configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="AutoTrade YouTube Daily Scanner - Scan channels for market intelligence"
    )
    parser.add_argument(
        "--channel", "-c",
        help="Scan specific channel only (e.g., trade_brigade, click_capital)"
    )
    parser.add_argument(
        "--report-only", "-r",
        action="store_true",
        help="Regenerate daily report from existing extractions (no scanning)"
    )
    parser.add_argument(
        "--list-pending", "-l",
        action="store_true",
        help="List videos that haven't been processed yet"
    )
    parser.add_argument(
        "--force-rescan", "-f",
        action="store_true",
        help="Ignore dedup tracking, rescan everything"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to youtube_channels.yaml config"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--synthesis-provider",
        default=None,
        help="Override report synthesis provider (openai, openrouter, or ollama)"
    )
    parser.add_argument(
        "--report-model",
        default=None,
        help="Override report synthesis model"
    )
    parser.add_argument(
        "--report-output-file",
        default=None,
        help="Custom output filename for report (stored in data/youtube/rag/daily_reports)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Load config
    config_path = Path(args.config) if args.config else None
    config = load_channel_config(config_path)
    
    if args.list_pending:
        list_pending(config)
        return
    
    summary = run_full_scan(
        config=config,
        channel_filter=args.channel,
        force=args.force_rescan,
        report_only=args.report_only,
        report_provider_override=args.synthesis_provider,
        report_model_override=args.report_model,
        report_output_filename=args.report_output_file,
    )
    
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
