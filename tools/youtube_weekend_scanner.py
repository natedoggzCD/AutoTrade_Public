#!/usr/bin/env python3
"""
YouTube Weekend Scanner for AutoTrade
======================================
Interactive weekend scanning mode for longer deep-dive videos with smart model
selection and OpenAI-powered synthesis.

Weekend videos (Fri-Sun) are typically longer, more analytical, and contain
forward-looking content that is critical for Monday morning trading.

Features:
- Interactive video selection menu
- Smart model selection based on video length/transcript size
- Long video addendum for channel-specific extraction prompts
- OpenAI gpt-4.1-mini synthesis with local fallback
- Cost tracking to logs/openai_costs.jsonl

Usage:
    python tools/youtube_weekend_scanner.py              # Interactive menu
    python tools/youtube_weekend_scanner.py --all        # Process all weekend videos
    python tools/youtube_weekend_scanner.py --report-only # Regenerate from existing
    python tools/youtube_weekend_scanner.py --verbose    # Verbose logging
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Fix Windows CP1252 console crash on emoji/unicode characters
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.youtube_daily_scanner import (
    _extract_json,
    extract_with_llm,
    filter_new_videos,
    get_recent_videos,
    load_channel_config,
    load_extraction_template,
    load_processed_videos,
    save_processed_videos,
    select_extraction_model,
    store_extraction,
    transcribe_video,
)

logger = logging.getLogger("youtube_weekend")


# ── Helpers ────────────────────────────────────────────────────────

def _get_next_monday(ref: Optional[datetime] = None) -> str:
    """Return the next Monday date as YYYY-MM-DD. If today is Monday, return today."""
    now = ref or datetime.now()
    days_ahead = (7 - now.weekday()) % 7  # 0 = Monday
    if days_ahead == 0 and now.weekday() != 0:
        days_ahead = 7
    monday = now + timedelta(days=days_ahead)
    # If today is already Monday, use today
    if now.weekday() == 0:
        monday = now
    return monday.strftime("%Y-%m-%d")


def _is_weekend_video(upload_date_str: str) -> bool:
    """Check if a video was uploaded on Friday, Saturday, or Sunday."""
    if not upload_date_str:
        return False  # Unknown date: exclude from weekend filter
    try:
        cleaned = upload_date_str.strip().replace("-", "")
        dt = datetime.strptime(cleaned, "%Y%m%d")
        return dt.weekday() >= 4  # 4=Fri, 5=Sat, 6=Sun
    except ValueError:
        return False


def _format_duration(seconds: int) -> str:
    """Format duration in human-readable form."""
    if not seconds:
        return "??m"
    minutes = seconds // 60
    if minutes >= 60:
        return f"{minutes // 60}h{minutes % 60:02d}m"
    return f"{minutes}m"

def _find_split_index(text: str, approx_idx: int) -> int:
    """Find a whitespace boundary near approx_idx for a clean split."""
    if approx_idx <= 0:
        return 0
    if approx_idx >= len(text):
        return len(text)
    left = approx_idx
    right = approx_idx
    while left > 0 or right < len(text) - 1:
        if left > 0 and text[left].isspace():
            return left
        if right < len(text) and text[right].isspace():
            return right
        left -= 1
        right += 1
    return approx_idx


def _split_transcript(text: str, overlap: int = 2000) -> List[str]:
    """Split transcript into two parts with optional overlap."""
    n = len(text)
    if n <= 1:
        return [text]
    mid = n // 2
    split_idx = _find_split_index(text, mid)
    if split_idx <= 0 or split_idx >= n:
        return [text]
    start2 = max(0, split_idx - overlap)
    end1 = min(n, split_idx + overlap)
    part1 = text[:end1]
    part2 = text[start2:]
    return [part1, part2]


def _merge_partials(
    partials: List[dict],
    template_prompt: str,
    channel_name: str,
    video_title: str,
    video_date: str,
    config: dict,
    model_override: Optional[str] = None,
    num_ctx_override: Optional[int] = None,
) -> Optional[dict]:
    """Merge multiple partial extractions into a single extraction via LLM."""
    import json as _json

    safe_template = template_prompt.replace("{transcript}", "<TRANSCRIPT>")
    merge_template = (
        "You are merging partial extractions from the SAME video.\n"
        "Return a single JSON object only.\n"
        "Use the SAME schema and keys as described in the TEMPLATE below.\n\n"
        "TEMPLATE:\n"
        f"{safe_template}\n\n"
        "PARTIAL_EXTRACTIONS:\n"
        "{transcript}\n"
    )

    payload = _json.dumps(partials, ensure_ascii=False)
    merged = extract_with_llm(
        transcript=payload,
        template_prompt=merge_template,
        channel_name=channel_name,
        video_title=f"{video_title} (merged)",
        video_date=video_date,
        config=config,
        model_override=model_override,
        num_ctx_override=num_ctx_override,
    )
    if merged and isinstance(merged, dict):
        merged.setdefault("_meta", {})
        merged["_meta"]["merged_partials"] = len(partials)
    return merged


# ---- Agent test hook (intentional, easily removable) ----
def _agent_test_complex(config: dict) -> None:
    """
    Intentional complex error for coding-agent validation.
    Triggered when AUTOTRADE_AGENT_TEST_COMPLEX is set.
    Modes:
      - "2" (default): multi-file error
      - "1": single-file error
      - "0": disabled
    """
    mode = os.environ.get("AUTOTRADE_AGENT_TEST_COMPLEX", "2")
    if mode in ("0", "false", "False"):
        return
    if mode == "2":
        from autotrade.utils.agent_test_multifile import run_multifile_test
        run_multifile_test(config)
        return

    weekend_cfg = config.get("scanner", {}).get("weekend", {})
    probe = {
        "cfg": weekend_cfg,
        "age": weekend_cfg.get("max_age_hours", 72),
        "count": weekend_cfg.get("max_results_per_channel", 5),
    }
    # INTENTIONAL bug: shadow dict with JSON string, then treat as dict.
    # probe = json.dumps(probe)  # COMMENTED OUT to fix type error
    if probe["age"] > 0:
        pass


# ── Scanning ──────────────────────────────────────────────────────

def scan_weekend_videos(config: dict) -> Dict[str, List[dict]]:
    """
    Scan all 4 channels for videos uploaded Friday-Sunday (72h window).

    Returns dict keyed by channel_key with list of video dicts.
    """
    # INTENTIONAL TYPE ERROR for agent test (before any scanning/network)
    # len(5)  # raises TypeError

    channels = config.get("channels", {})
    weekend_cfg = config.get("scanner", {}).get("weekend", {})
    max_age_hours = weekend_cfg.get("max_age_hours", 72)
    max_results = weekend_cfg.get("max_results_per_channel", 5)

    all_videos: Dict[str, List[dict]] = {}

    for channel_key, channel_cfg in channels.items():
        channel_url = channel_cfg["channel_url"]
        logger.info(f"Scanning {channel_cfg['name']} for weekend videos...")

        videos = get_recent_videos(
            channel_url,
            max_age_hours=max_age_hours,
            max_results=max_results,
        )

        missing_dates = sum(1 for v in videos if not v.get("upload_date"))
        if missing_dates:
            logger.warning(f"  {missing_dates} videos missing upload_date (excluded from weekend filter)")

        # Filter to weekend uploads only
        weekend_vids = [v for v in videos if _is_weekend_video(v.get("upload_date", ""))]

        if weekend_vids:
            all_videos[channel_key] = weekend_vids
            logger.info(f"  Found {len(weekend_vids)} weekend videos")
        else:
            logger.info(f"  No weekend videos found")

    return all_videos


# ── Interactive Menu ──────────────────────────────────────────────

def display_video_menu(
    all_videos: Dict[str, List[dict]],
    config: dict,
) -> List[Tuple[str, dict]]:
    """
    Display numbered list of weekend videos and let user pick.

    Returns list of (channel_key, video_dict) tuples to process.
    """
    channels = config.get("channels", {})

    # Build flat list for numbering
    flat: List[Tuple[str, dict]] = []
    for channel_key, videos in all_videos.items():
        for v in videos:
            flat.append((channel_key, v))

    if not flat:
        print("\n  No weekend videos found across any channel.")
        return []

    # Sort by priority then date
    def sort_key(item):
        ck, v = item
        priority = channels.get(ck, {}).get("priority", 99)
        date = v.get("upload_date", "00000000")
        return (priority, date)

    flat.sort(key=sort_key)

    print(f"\n{'='*70}")
    print(f"  WEEKEND VIDEOS ({len(flat)} found)")
    print(f"{'='*70}")

    for i, (ck, v) in enumerate(flat, 1):
        ch_name = channels.get(ck, {}).get("name", ck)
        dur = _format_duration(v.get("duration", 0))
        title = v.get("title", "Unknown")[:50]
        date_raw = v.get("upload_date", "")
        date = "????-??-??"
        if date_raw:
            cleaned = str(date_raw).strip().replace("-", "")
            if len(cleaned) == 8 and cleaned.isdigit():
                date = f"{cleaned[0:4]}-{cleaned[4:6]}-{cleaned[6:8]}"
        pri = channels.get(ck, {}).get("priority", "?")
        print(f"  [{i:2d}] [{ch_name:16s}] {date}  {dur:>6s}  P{pri}  {title}")

    print(f"\n{'='*70}")
    print(f"  Enter numbers (e.g. 1,3,5), 'all', or 'q' to quit:")

    try:
        choice = input("  > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return []

    if choice == "q" or choice == "":
        return []

    if choice == "all":
        return flat

    # Parse comma-separated numbers
    selected = []
    for part in choice.split(","):
        part = part.strip()
        try:
            idx = int(part) - 1
            if 0 <= idx < len(flat):
                selected.append(flat[idx])
            else:
                print(f"  Invalid number: {part}")
        except ValueError:
            print(f"  Invalid input: {part}")

    return selected


# ── Video Processing ──────────────────────────────────────────────

def process_weekend_video(
    channel_key: str,
    video: dict,
    config: dict,
    processed: dict,
) -> Optional[dict]:
    """
    Process a single weekend video: transcribe -> model select -> extract -> store.

    Returns extraction dict or None on failure.
    """
    channels = config.get("channels", {})
    channel_cfg = channels.get(channel_key, {})
    channel_name = channel_cfg.get("name", channel_key)
    template_path = channel_cfg.get("template", "")
    priority = channel_cfg.get("priority", 3)

    video_id = video["id"]
    video_title = video["title"]
    video_date = video.get("upload_date", datetime.now().strftime("%Y%m%d"))
    video_duration = video.get("duration", 0)

    scanner_config = config.get("scanner", {})
    output_dir = Path(scanner_config.get("output_dir", "data/youtube"))
    rag_dir = Path(scanner_config.get("rag_dir", "data/youtube/rag"))

    print(f"\n  Processing: {video_title}")
    print(f"  Channel:    {channel_name}")
    print(f"  Duration:   {_format_duration(video_duration)}")

    # 1. Transcribe
    print(f"  [1/3] Transcribing...")
    tx_result = transcribe_video(
        video["url"],
        output_dir,
        config,
        video_id=video_id,
        video_title=video_title,
    )

    if not tx_result or not tx_result.get("success"):
        print(f"  FAILED: Transcription failed")
        processed.setdefault("processed", {})[video_id] = {
            "channel": channel_key,
            "title": video_title,
            "date": video_date,
            "status": "transcription_failed",
            "timestamp": datetime.now().isoformat(),
            "source": "weekend_scanner",
        }
        return None

    transcript = tx_result["transcript"]
    if len(transcript.strip()) < 100:
        print(f"  FAILED: Transcript too short ({len(transcript)} chars)")
        return None

    transcript_len = len(transcript)
    print(f"  Transcript: {transcript_len:,} chars ({transcript_len // 4:,} est. tokens)")

    # 2. Select model
    model, num_ctx = select_extraction_model(
        transcript_length=transcript_len,
        video_duration_sec=video_duration,
        channel_priority=priority,
        config=config,
    )
    is_long = transcript_len > 20000 or (video_duration and video_duration >= 2400)
    print(f"  [2/3] Extracting with {model} (ctx={num_ctx}, long={is_long})")

    # 3. Load template with long video addendum
    template = load_extraction_template(template_path, long_video=is_long)
    if not template:
        print(f"  FAILED: Could not load template {template_path}")
        return None

    # 4. Extract (split long transcript into two passes if needed)
    weekend_cfg = config.get("scanner", {}).get("weekend", {})
    split_enabled = weekend_cfg.get("split_long_transcripts", True)
    split_chars = weekend_cfg.get("split_long_transcript_chars", 60000)
    split_seconds = weekend_cfg.get("split_long_video_seconds", 3600)
    split_overlap = weekend_cfg.get("split_transcript_overlap_chars", 2000)

    should_split = (
        split_enabled
        and (transcript_len >= split_chars or (video_duration and video_duration >= split_seconds))
    )

    if should_split:
        print("  [2/3] Splitting long transcript into 2 parts...")
        parts = _split_transcript(transcript, overlap=split_overlap)
        partials: List[dict] = []
        for idx, part in enumerate(parts, 1):
            print(f"    - Extracting part {idx}/{len(parts)} ({len(part):,} chars)")
            part_extraction = extract_with_llm(
                transcript=part,
                template_prompt=template,
                channel_name=channel_name,
                video_title=f"{video_title} (Part {idx}/{len(parts)})",
                video_date=video_date,
                config=config,
                model_override=model,
                num_ctx_override=num_ctx,
            )
            if part_extraction:
                partials.append(part_extraction)

        if len(partials) == 1:
            extraction = partials[0]
        elif len(partials) >= 2:
            print("  [2/3] Merging partial extractions...")
            extraction = _merge_partials(
                partials=partials,
                template_prompt=template,
                channel_name=channel_name,
                video_title=video_title,
                video_date=video_date,
                config=config,
                model_override=model,
                num_ctx_override=num_ctx,
            )
        else:
            extraction = None
    else:
        extraction = extract_with_llm(
            transcript=transcript,
            template_prompt=template,
            channel_name=channel_name,
            video_title=video_title,
            video_date=video_date,
            config=config,
            model_override=model,
            num_ctx_override=num_ctx,
        )

    if extraction:
        # 5. Store in RAG
        print(f"  [3/3] Storing extraction in RAG...")
        store_extraction(
            extraction=extraction,
            channel_key=channel_key,
            video_id=video_id,
            video_date=video_date,
            rag_dir=rag_dir,
        )

        # Track as processed
        processed.setdefault("processed", {})[video_id] = {
            "channel": channel_key,
            "title": video_title,
            "date": video_date,
            "status": "complete",
            "timestamp": datetime.now().isoformat(),
            "source": "weekend_scanner",
            "model_used": model,
            "transcript_length": transcript_len,
            "long_video": is_long,
            "split_parts": 2 if should_split else 1,
            "files": tx_result.get("files", {}),
        }

        print(f"  OK: Extraction complete ({model})")
        return extraction
    else:
        processed.setdefault("processed", {})[video_id] = {
            "channel": channel_key,
            "title": video_title,
            "date": video_date,
            "status": "extraction_failed",
            "timestamp": datetime.now().isoformat(),
            "source": "weekend_scanner",
        }
        print(f"  FAILED: LLM extraction returned nothing")
        return None


# ── Weekend Report Generation ─────────────────────────────────────

def _log_openai_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> None:
    """Append cost record to logs/openai_costs.jsonl."""
    log_file = PROJECT_ROOT / "logs" / "openai_costs.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost_usd, 6),
        "source": "weekend_scanner",
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _load_synthesis_prompt(config: dict) -> str:
    """Load the weekend synthesis prompt template."""
    prompt_path = PROJECT_ROOT / "prompts" / "llm" / "youtube_weekend_synthesis.md"

    if prompt_path.exists():
        import re
        content = prompt_path.read_text(encoding="utf-8")
        match = re.search(
            r"## Synthesis Prompt\s*\n+```[^\n]*\n(.*?)```",
            content,
            re.DOTALL,
        )
        if match:
            return match.group(1).strip()

    # Fallback built-in prompt
    return (
        "You are the AutoTrade Weekend Intelligence Synthesizer. Consolidate the "
        "following weekend YouTube channel extractions into a single JSON report "
        "for Monday morning trading. Weight forward-looking content higher than "
        "recaps. Saturday deep dives supersede Friday dailies on conflicts.\n\n"
        "CHANNEL EXTRACTIONS:\n{channel_extractions}\n\n"
        "{prior_day_report}\n\n"
        "Output a comprehensive JSON report with: date, report_type, executive_summary, "
        "market_regime, regime_confidence, weekend_specific (weekly_chart_levels, "
        "next_week_outlook, monday_opening_strategy, key_events_next_week), "
        "smallcap_health, trading_signals (sizing_multiplier, sector_bias, trigger_levels), "
        "consensus, trade_ideas, overnight_directives, channel_agreement."
    )


def generate_weekend_report(
    extractions: Dict[str, dict],
    config: dict,
    monday_date: str,
) -> Optional[dict]:
    """
    Generate consolidated weekend report using OpenAI with local fallback.

    Saves as BOTH {monday}_weekend_consolidated.json AND {monday}_consolidated.json
    so load_market_intelligence() finds it automatically.
    """
    if not extractions:
        logger.warning("No extractions to synthesize")
        return None

    weekend_cfg = config.get("scanner", {}).get("weekend", {})
    synthesis_provider = weekend_cfg.get("synthesis_provider", "openai")
    openai_model = weekend_cfg.get("synthesis_model_openai", "gpt-4.1-mini")
    local_fallback = weekend_cfg.get("synthesis_model_local_fallback", "glm-4.7-flash")
    max_tokens = weekend_cfg.get("synthesis_max_tokens", 8192)
    cost_budget = weekend_cfg.get("cost_budget_per_weekend", 0.50)

    rag_dir = Path(config.get("scanner", {}).get("rag_dir", "data/youtube/rag"))
    report_dir = rag_dir / "daily_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Build channel extractions block
    channel_blocks = []
    for channel_key, ext in extractions.items():
        ext_str = json.dumps(ext, indent=2, ensure_ascii=False)
        if len(ext_str) > 15000:
            ext_str = ext_str[:15000] + "\n... (truncated)"
        channel_blocks.append(f"### {channel_key}\n```json\n{ext_str}\n```")
    extractions_block = "\n\n".join(channel_blocks)

    # Load prior report for continuity
    prior_block = ""
    try:
        prev_date = (datetime.strptime(monday_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        prev_file = report_dir / f"{prev_date}_consolidated.json"
        if prev_file.exists():
            with open(prev_file, "r", encoding="utf-8") as f:
                prior = json.load(f)
            prior_str = json.dumps(prior, indent=2, ensure_ascii=False)
            if len(prior_str) > 6000:
                prior_str = prior_str[:6000] + "\n... (truncated)"
            prior_block = f"PRIOR DAY REPORT (for continuity):\n```json\n{prior_str}\n```"
    except Exception:
        pass

    # Load and fill synthesis prompt
    prompt_template = _load_synthesis_prompt(config)
    prompt = prompt_template.replace("{date}", monday_date)
    prompt = prompt.replace("{channel_count}", str(len(extractions)))
    prompt = prompt.replace("{channel_extractions}", extractions_block)
    prompt = prompt.replace("{prior_day_report}", prior_block)

    report = None

    # Try OpenAI first
    if synthesis_provider == "openai":
        report = _synthesize_openai(prompt, openai_model, max_tokens, cost_budget, monday_date)

    # Try OpenRouter
    if report is None and synthesis_provider == "openrouter":
        ext_config = config.get("scanner", {}).get("extraction", {})
        or_model = ext_config.get("report_model", local_fallback)
        print(f"  Synthesizing with OpenRouter {or_model}...")
        report = _synthesize_openrouter(prompt, or_model, config, monday_date)

    # Fallback to local Ollama
    if report is None:
        print(f"  Using local fallback: {local_fallback}")
        report = _synthesize_local(prompt, local_fallback, config, monday_date)

    if report is None:
        logger.error("Weekend report generation failed (both OpenAI and local)")
        return None

    # Add metadata
    report["_meta"] = {
        "date": monday_date,
        "report_type": "weekend_consolidated",
        "channels_included": list(extractions.keys()),
        "generated_at": datetime.now().isoformat(),
        "_source": "weekend_consolidated",
    }

    # Save as BOTH weekend and regular consolidated
    weekend_file = report_dir / f"{monday_date}_weekend_consolidated.json"
    regular_file = report_dir / f"{monday_date}_consolidated.json"

    for filepath in [weekend_file, regular_file]:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {filepath}")

    print(f"\n  Weekend report saved:")
    print(f"    {weekend_file}")
    print(f"    {regular_file}")

    return report


def _synthesize_openai(
    prompt: str,
    model: str,
    max_tokens: int,
    cost_budget: float,
    monday_date: str,
) -> Optional[dict]:
    """Try OpenAI synthesis. Returns parsed report or None."""
    try:
        import openai
    except ImportError:
        logger.warning("openai package not installed, skipping OpenAI synthesis")
        return None

    # Estimate cost before calling
    input_tokens_est = len(prompt) // 4
    # gpt-4.1-mini pricing (approximate): $0.40/1M input, $1.60/1M output
    est_cost = (input_tokens_est * 0.40 / 1_000_000) + (max_tokens * 1.60 / 1_000_000)

    if est_cost > cost_budget:
        logger.warning(f"Estimated cost ${est_cost:.4f} exceeds budget ${cost_budget:.2f}, skipping OpenAI")
        return None

    print(f"  Synthesizing with OpenAI {model} (est. cost: ${est_cost:.4f})...")

    try:
        client = openai.OpenAI()  # Uses OPENAI_API_KEY from env
        start = time.time()

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1,
        )

        elapsed = time.time() - start
        content = response.choices[0].message.content or ""
        usage = response.usage

        input_tokens = usage.prompt_tokens if usage else input_tokens_est
        output_tokens = usage.completion_tokens if usage else len(content) // 4
        actual_cost = (input_tokens * 0.40 / 1_000_000) + (output_tokens * 1.60 / 1_000_000)

        _log_openai_cost(model, input_tokens, output_tokens, actual_cost)

        print(f"  OpenAI complete in {elapsed:.1f}s (cost: ${actual_cost:.4f})")

        report = _extract_json(content)
        if report:
            report["_synthesis_model"] = model
            report["_synthesis_provider"] = "openai"
            report["_synthesis_cost_usd"] = round(actual_cost, 6)
            return report
        else:
            logger.warning("Failed to parse JSON from OpenAI response")
            return {"raw_report": content, "_synthesis_model": model, "_synthesis_provider": "openai"}

    except Exception as e:
        logger.error(f"OpenAI synthesis failed: {e}")
        return None


def _synthesize_local(
    prompt: str,
    model: str,
    config: dict,
    monday_date: str,
) -> Optional[dict]:
    """Fallback local Ollama synthesis."""
    ext_config = config.get("scanner", {}).get("extraction", {})
    ollama_url = ext_config.get("ollama_url", "http://localhost:11434")
    timeout = ext_config.get("synthesis_timeout", 1200)

    print(f"  Synthesizing with local {model}...")

    try:
        import requests

        start = time.time()
        response = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_ctx": 65536,
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()

        elapsed = time.time() - start
        content = response.json().get("message", {}).get("content", "")

        print(f"  Local synthesis complete in {elapsed:.1f}s")

        report = _extract_json(content)
        if report:
            report["_synthesis_model"] = model
            report["_synthesis_provider"] = "local"
            return report
        else:
            return {"raw_report": content, "_synthesis_model": model, "_synthesis_provider": "local"}

    except Exception as e:
        logger.error(f"Local synthesis failed: {e}")
        return None


def _synthesize_openrouter(
    prompt: str,
    model: str,
    config: dict,
    monday_date: str,
) -> Optional[dict]:
    """Synthesize weekend report via OpenRouter."""
    ext_config = config.get("scanner", {}).get("extraction", {})
    weekend_cfg = config.get("scanner", {}).get("weekend", {})
    openrouter_url = ext_config.get("openrouter_url", "https://openrouter.ai/api/v1/chat/completions")
    openrouter_key_env = ext_config.get("openrouter_api_key_env", "OPENROUTER_API_KEY")
    openrouter_referer = ext_config.get("openrouter_referer")
    openrouter_title = ext_config.get("openrouter_title", "AutoTrade YouTube Summary")
    # Weekend reports are large — use weekend max_tokens (8192) not the extraction default (4096)
    openrouter_max_tokens = int(weekend_cfg.get("synthesis_max_tokens", ext_config.get("openrouter_max_tokens", 8192)))
    timeout = ext_config.get("synthesis_timeout", 1200)
    fallback_model = ext_config.get("report_fallback_model", model)

    try:
        import requests
        from tools.youtube_daily_scanner import _get_env_var

        api_key = _get_env_var(str(openrouter_key_env)) or _get_env_var("OPENROUTER_API_KEY")
        if not api_key:
            logger.warning(f"Missing OpenRouter key ({openrouter_key_env}), skipping")
            return None

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if openrouter_referer:
            headers["HTTP-Referer"] = str(openrouter_referer)
        if openrouter_title:
            headers["X-Title"] = str(openrouter_title)

        def _call(mdl: str) -> Optional[dict]:
            start = time.time()
            resp = requests.post(
                openrouter_url,
                headers=headers,
                json={
                    "model": mdl,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": openrouter_max_tokens,
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            content = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content", "")
            elapsed = time.time() - start
            print(f"  OpenRouter synthesis complete in {elapsed:.1f}s (model: {mdl})")
            report = _extract_json(content)
            if report:
                report["_synthesis_model"] = mdl
                report["_synthesis_provider"] = "openrouter"
                return report
            elif content:
                return {"raw_report": content, "_synthesis_model": mdl, "_synthesis_provider": "openrouter"}
            return None

        # Try primary, then fallback
        try:
            result = _call(model)
            if result:
                return result
        except Exception as e:
            logger.warning(f"OpenRouter primary model {model} failed: {e}")

        if fallback_model != model:
            try:
                return _call(fallback_model)
            except Exception as e:
                logger.warning(f"OpenRouter fallback {fallback_model} failed: {e}")

        return None

    except Exception as e:
        logger.error(f"OpenRouter synthesis failed: {e}")
        return None


# ── Main Pipeline ─────────────────────────────────────────────────

def run_weekend_scan(
    config: dict,
    process_all: bool = False,
    report_only: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Run the full weekend scanning pipeline.

    Args:
        process_all: Skip interactive menu, process everything
        report_only: Regenerate report from existing extractions
        verbose: Extra logging
    """
    scanner_config = config.get("scanner", {})
    rag_dir = Path(scanner_config.get("rag_dir", "data/youtube/rag"))
    track_file = Path(scanner_config.get("dedup", {}).get(
        "track_file", "data/youtube/processed_videos.json"
    ))

    monday_date = _get_next_monday()

    print(f"\n{'='*70}")
    print(f"  WEEKEND YOUTUBE INTELLIGENCE SCANNER")
    print(f"  Target Monday: {monday_date}")
    print(f"{'='*70}")

    if report_only:
        print("\n  Report-only mode: regenerating from existing extractions")
        # Collect extractions from weekend dates (Fri/Sat/Sun before Monday)
        monday_dt = datetime.strptime(monday_date, "%Y-%m-%d")
        all_extractions = {}
        for days_back in range(1, 4):  # Sun, Sat, Fri
            date_str = (monday_dt - timedelta(days=days_back)).strftime("%Y-%m-%d")
            date_dir = rag_dir / "by_date" / date_str
            if date_dir.exists():
                for f in date_dir.glob("*.json"):
                    channel_key = f.stem.rsplit("_", 1)[0]
                    with open(f, "r", encoding="utf-8") as fh:
                        # Use latest per channel
                        all_extractions[channel_key] = json.load(fh)

        if not all_extractions:
            print("  No existing extractions found for this weekend")
            return {"status": "no_data"}

        print(f"  Found extractions from: {list(all_extractions.keys())}")
        report = generate_weekend_report(all_extractions, config, monday_date)
        return {
            "mode": "report_only",
            "monday": monday_date,
            "channels": list(all_extractions.keys()),
            "report_generated": report is not None,
        }

    # Scan for weekend videos
    print("\n  Scanning channels for weekend videos...")
    all_videos = scan_weekend_videos(config)

    if not all_videos:
        print("\n  No weekend videos found. Nothing to process.")
        return {"status": "no_videos", "monday": monday_date}

    total_videos = sum(len(v) for v in all_videos.values())
    print(f"\n  Found {total_videos} weekend videos across {len(all_videos)} channels")

    # Select videos to process
    if process_all:
        selected = []
        for ck, videos in all_videos.items():
            for v in videos:
                selected.append((ck, v))
        print(f"\n  Processing all {len(selected)} videos...")
    else:
        selected = display_video_menu(all_videos, config)

    if not selected:
        print("\n  No videos selected. Exiting.")
        return {"status": "cancelled", "monday": monday_date}

    # Load dedup tracking
    processed = load_processed_videos(track_file)

    # Process selected videos
    extractions: Dict[str, dict] = {}
    success_count = 0
    fail_count = 0

    for channel_key, video in selected:
        # Skip already processed (unless forced)
        if video["id"] in processed.get("processed", {}):
            existing = processed["processed"][video["id"]]
            if existing.get("status") == "complete":
                print(f"\n  Skipping (already processed): {video['title']}")
                # Load existing extraction for report
                video_date = video.get("upload_date", "")
                if len(video_date) == 8:
                    date_str = f"{video_date[:4]}-{video_date[4:6]}-{video_date[6:]}"
                else:
                    date_str = datetime.now().strftime("%Y-%m-%d")
                ext_file = rag_dir / "by_date" / date_str / f"{channel_key}_{video['id']}.json"
                if ext_file.exists():
                    with open(ext_file, "r", encoding="utf-8") as f:
                        extractions[channel_key] = json.load(f)
                continue

        result = process_weekend_video(channel_key, video, config, processed)
        if result:
            extractions[channel_key] = result
            success_count += 1
        else:
            fail_count += 1

    # Save dedup tracking
    save_processed_videos(track_file, processed)

    print(f"\n{'='*70}")
    print(f"  Processing complete: {success_count} OK, {fail_count} failed")
    print(f"{'='*70}")

    # Generate weekend report
    if extractions:
        print(f"\n  Generating weekend consolidated report...")
        report = generate_weekend_report(extractions, config, monday_date)

        if report:
            # Print summary
            regime = report.get("market_regime", "UNKNOWN")
            sizing = report.get("trading_signals", {}).get("sizing_multiplier", "N/A")
            summary = report.get("executive_summary", "")
            print(f"\n{'='*70}")
            print(f"  WEEKEND REPORT SUMMARY")
            print(f"{'='*70}")
            print(f"  Regime:  {regime}")
            print(f"  Sizing:  {sizing}x")
            if summary:
                print(f"  Summary: {summary[:200]}")
            monday_strat = report.get("weekend_specific", {}).get("monday_opening_strategy", "")
            if monday_strat:
                print(f"  Monday:  {monday_strat[:200]}")
            print(f"{'='*70}")
    else:
        report = None
        print("\n  No successful extractions - skipping report generation")

    return {
        "mode": "interactive" if not process_all else "all",
        "monday": monday_date,
        "videos_processed": success_count,
        "videos_failed": fail_count,
        "channels": list(extractions.keys()),
        "report_generated": report is not None,
    }


# ── CLI Entry Point ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AutoTrade Weekend YouTube Scanner - Deep dive analysis with OpenAI synthesis"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Process all weekend videos (skip interactive menu)"
    )
    parser.add_argument(
        "--report-only", "-r",
        action="store_true",
        help="Regenerate weekend report from existing extractions"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to youtube_channels.yaml config"
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

    # Intentional complex error before any scanning/network
    _agent_test_complex(config)

    summary = run_weekend_scan(
        config=config,
        process_all=args.all,
        report_only=args.report_only,
        verbose=args.verbose,
    )

    print(f"\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
