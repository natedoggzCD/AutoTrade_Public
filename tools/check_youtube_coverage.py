"""
YouTube channel coverage checker.

Compares what each configured channel *actually published* (via the channel's
public RSS feed) against what the scanner *captured* (via processed_videos.json
and data/youtube/rag/by_date/<date>/). Surfaces the gap that the scanner
otherwise hides — specifically "silent skips" caused by yt-dlp flat-extract
returning no upload_date and the per-video detail fetch failing.

Exit code:
  0 — all daily publishers caught a video in the last N days and no silent
      misses found
  1 — at least one silent miss OR a daily publisher missing >=2 consecutive
      trading days

Usage:
  python tools/check_youtube_coverage.py
  python tools/check_youtube_coverage.py --days 14
  python tools/check_youtube_coverage.py --json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHANNELS_YAML = PROJECT_ROOT / "config" / "youtube_channels.yaml"
PROCESSED_FILE = PROJECT_ROOT / "data" / "youtube" / "processed_videos.json"
BY_DATE_DIR = PROJECT_ROOT / "data" / "youtube" / "rag" / "by_date"

# Daily publishers — channels whose REGULAR VOD cadence is ~daily on weekdays.
# If these go silent for >=2 weekdays without a caught video, scanner is broken.
# Excludes rta_trading: their non-live VOD cadence is only ~3-4/week (their
# daily content is the PREMARKET LIVE stream, which we intentionally skip).
DAILY_PUBLISHERS = {"mike_market", "stockedup"}

RSS_URL = "https://www.youtube.com/feeds/videos.xml?channel_id={}"
RSS_ENTRY_RE = re.compile(
    r"<entry>.*?<yt:videoId>([^<]+)</yt:videoId>.*?<title>([^<]+)</title>.*?"
    r"<published>([0-9T:\-+Z]+)</published>",
    re.DOTALL,
)


def load_channels() -> dict[str, dict]:
    with open(CHANNELS_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)["channels"]


def fetch_rss(channel_id: str, timeout: int = 15) -> list[dict]:
    url = RSS_URL.format(channel_id)
    req = urllib.request.Request(url, headers={"User-Agent": "AutoTradeCoverageCheck/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    out = []
    for vid, title, published in RSS_ENTRY_RE.findall(body):
        try:
            pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        except ValueError:
            continue
        out.append({"id": vid, "title": title.strip(), "published": pub_dt})
    return out


def load_processed() -> dict[str, dict]:
    if not PROCESSED_FILE.exists():
        return {}
    blob = json.loads(PROCESSED_FILE.read_text(encoding="utf-8"))
    return blob.get("processed", {})


_EXTRACTION_INDEX: dict[str, str] | None = None
_EXTRACTION_INDEX_BUILT: dict[str, str] = {}


def _build_extraction_index(channel_keys: list[str]) -> dict[str, str]:
    """Map video_id -> by_date folder containing its extraction (scanner-run date).

    Files are named <channel_key>_<video_id>.json — video IDs can contain
    underscores, so strip the known channel-key prefix rather than splitting.
    """
    global _EXTRACTION_INDEX
    if _EXTRACTION_INDEX is not None:
        return _EXTRACTION_INDEX
    idx: dict[str, str] = {}
    prefixes = sorted((f"{k}_" for k in channel_keys), key=len, reverse=True)
    if BY_DATE_DIR.exists():
        for day_dir in BY_DATE_DIR.iterdir():
            if not day_dir.is_dir():
                continue
            for p in day_dir.glob("*.json"):
                stem = p.stem
                vid = None
                for pref in prefixes:
                    if stem.startswith(pref):
                        vid = stem[len(pref):]
                        break
                if vid:
                    idx[vid] = day_dir.name
    _EXTRACTION_INDEX = idx
    return idx


def _channel_first_seen(channel_key: str, processed: dict[str, dict]) -> date | None:
    dates = []
    for v in processed.values():
        if v.get("channel") != channel_key:
            continue
        ds = v.get("date", "")
        try:
            dates.append(datetime.strptime(ds, "%Y%m%d").date())
        except (ValueError, TypeError):
            continue
    return min(dates) if dates else None


_LIVE_STREAM_TITLE_RE = re.compile(
    r"\b(?:premarket\s+live|live\s+stream|earnings\s+live|"
    r"live\s+trading|live\s*(?::|\s|$)|going\s+live)\b",
    re.IGNORECASE,
)


def _looks_like_live_stream(title: str) -> bool:
    return bool(_LIVE_STREAM_TITLE_RE.search(title or ""))


def classify_upload(
    vid: str,
    channel_key: str,
    upload_date: date,
    processed: dict[str, dict],
    title: str = "",
) -> str:
    rec = processed.get(vid)
    if rec is None:
        first_seen = _channel_first_seen(channel_key, processed)
        if first_seen and upload_date < first_seen:
            return "before_channel_added"
        if _looks_like_live_stream(title):
            # Live broadcasts are intentionally skipped — VOD lands too late
            # to be useful for the trading window the stream covered.
            return "live_stream_skipped"
        return "silently_missed"
    status = rec.get("status", "")
    if status == "complete":
        if vid in _EXTRACTION_INDEX_BUILT:
            return "caught"
        return "complete_but_no_extraction"
    if status in ("transcription_failed", "extraction_failed"):
        return status
    return f"unknown:{status}"


def _ascii_safe(s: str) -> str:
    return s.encode("ascii", errors="replace").decode("ascii")


def run(days: int, as_json: bool) -> int:
    channels = load_channels()
    processed = load_processed()
    global _EXTRACTION_INDEX_BUILT
    _EXTRACTION_INDEX_BUILT = _build_extraction_index(list(channels.keys()))
    cutoff = datetime.now(tz=None).date() - timedelta(days=days)

    per_channel: dict[str, dict] = {}
    silent_misses_total = 0
    daily_publisher_gaps: dict[str, list[str]] = {}

    for key, cfg in channels.items():
        cid = cfg.get("channel_id")
        if not cid:
            continue
        try:
            uploads = fetch_rss(cid)
        except Exception as exc:  # noqa: BLE001
            per_channel[key] = {"error": f"RSS_FAILED: {exc}"}
            continue

        today = datetime.now().date()
        in_window = [
            u for u in uploads
            if cutoff <= u["published"].date() < today
        ]
        statuses: dict[str, int] = defaultdict(int)
        rows = []
        for u in in_window:
            d = u["published"].date()
            cls = classify_upload(u["id"], key, d, processed, u.get("title", ""))
            statuses[cls] += 1
            rows.append({"date": d.isoformat(), "id": u["id"], "title": u["title"], "status": cls})
            if cls == "silently_missed":
                silent_misses_total += 1
            if cls == "before_channel_added":
                continue
            rows[-1]["status"] = cls  # no-op but keeps grouping consistent

        last_caught = next(
            (r["date"] for r in rows if r["status"] == "caught"), None
        )
        per_channel[key] = {
            "name": cfg.get("name"),
            "priority": cfg.get("priority"),
            "channel_id": cid,
            "rss_uploads_in_window": len(in_window),
            "status_counts": dict(statuses),
            "last_caught": last_caught,
            "uploads": rows,
        }

        if key in DAILY_PUBLISHERS:
            today = datetime.now().date()
            expected_days = [
                today - timedelta(days=i)
                for i in range(1, 8)
                if (today - timedelta(days=i)).weekday() < 5
            ]
            caught_days = {
                datetime.fromisoformat(r["date"]).date()
                for r in rows
                if r["status"] == "caught"
            }
            missing = [d.isoformat() for d in expected_days if d not in caught_days]
            consecutive = 0
            max_consec = 0
            prev = None
            for d in sorted(expected_days):
                if d not in caught_days:
                    if prev and (d - prev).days == 1:
                        consecutive += 1
                    else:
                        consecutive = 1
                    max_consec = max(max_consec, consecutive)
                    prev = d
                else:
                    consecutive = 0
                    prev = d
            if max_consec >= 2:
                daily_publisher_gaps[key] = missing

    summary = {
        "days_window": days,
        "silent_misses_total": silent_misses_total,
        "daily_publisher_gaps": daily_publisher_gaps,
        "per_channel": per_channel,
    }

    if as_json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        try:
            _print_human(summary)
        except UnicodeEncodeError:
            # Fallback: re-print with all titles ASCII-stripped
            for info in summary["per_channel"].values():
                for row in info.get("uploads", []):
                    row["title"] = _ascii_safe(row["title"])
            _print_human(summary)

    if silent_misses_total > 0 or daily_publisher_gaps:
        return 1
    return 0


def _print_human(s: dict) -> None:
    print(f"YouTube Coverage — last {s['days_window']} days")
    print("=" * 78)
    for key, info in s["per_channel"].items():
        if "error" in info:
            print(f"  {key:18s} ERROR: {info['error']}")
            continue
        counts = info["status_counts"]
        line = (
            f"  {key:18s} pri={info['priority']}  "
            f"published={info['rss_uploads_in_window']:>2}  "
            f"caught={counts.get('caught',0):>2}  "
            f"silent_miss={counts.get('silently_missed',0):>2}  "
            f"tr_fail={counts.get('transcription_failed',0)}  "
            f"ex_fail={counts.get('extraction_failed',0)}  "
            f"last_caught={info['last_caught'] or '—'}"
        )
        print(line)
        for row in info["uploads"]:
            if row["status"] not in ("caught",):
                title = _ascii_safe(row["title"])[:48]
                print(f"      {row['date']}  {row['status']:30s}  {row['id']}  {title}")
    print("=" * 78)
    print(f"Silent misses total: {s['silent_misses_total']}")
    if s["daily_publisher_gaps"]:
        print("!! Daily-publisher gaps (>=2 consecutive missing weekdays):")
        for k, days in s["daily_publisher_gaps"].items():
            print(f"   {k}: {', '.join(days)}")
    else:
        print("Daily publishers: OK")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    return run(args.days, args.json)


if __name__ == "__main__":
    sys.exit(main())
