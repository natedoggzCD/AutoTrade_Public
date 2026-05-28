from pathlib import Path
import os
from datetime import datetime
from unittest.mock import MagicMock

import autotrade.utils.market_intelligence as mi
from autotrade.utils import youtube_readiness as yr


def test_detect_pending_videos_logs_safe_exception_text(monkeypatch, tmp_path: Path):
    class _BrokenExc(Exception):
        def __str__(self):
            raise RecursionError("maximum recursion depth exceeded while getting str")

    rag_dir = tmp_path / "rag"
    rag_dir.mkdir(parents=True, exist_ok=True)

    def _boom(*args, **kwargs):  # noqa: ANN001
        raise _BrokenExc("boom")

    monkeypatch.setattr(yr, "logger", MagicMock())
    monkeypatch.setattr(yr, "get_processed_videos", lambda: {})
    monkeypatch.setattr("tools.youtube_daily_scanner.get_recent_videos", _boom)
    monkeypatch.setattr(
        "tools.youtube_daily_scanner.filter_new_videos",
        lambda *args, **kwargs: [],
    )

    config = {
        "scanner": {
            "rag_dir": str(rag_dir),
            "performance": {"max_channel_workers": 2},
        },
        "channels": {
            "alpha": {"channel_url": "https://example.com/a"},
            "beta": {"channel_url": "https://example.com/b"},
        },
    }

    status = yr._detect_pending_videos(config)

    assert status["checked"] is True
    assert status["pending_count"] == 0
    assert yr.logger.debug.call_count >= 1
    rendered = " ".join(
        str(args[1]) for args, _kwargs in yr.logger.debug.call_args_list if len(args) > 1
    )
    assert "_BrokenExc" in rendered


def test_check_readiness_counts_target_session_date(monkeypatch, tmp_path: Path):
    rag_dir = tmp_path / "rag"
    by_date = rag_dir / "by_date"
    daily_reports = rag_dir / "daily_reports"
    output_dir = tmp_path / "youtube_out"
    by_date.mkdir(parents=True, exist_ok=True)
    daily_reports.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_date = "2026-02-13"
    today_date = "2026-02-12"

    # Target-session extraction exists (overnight after close), local "today" folder does not.
    target_dir = by_date / target_date
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "rta_trading_abc123.json").write_text(
        '{"channel": "rta_trading", "notes": "sample extraction"}',
        encoding="utf-8",
    )

    # A recent consolidated report for target session.
    (daily_reports / f"{target_date}_consolidated.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(yr, "RAG_DIR", rag_dir)
    monkeypatch.setattr(yr, "TRACK_FILE", tmp_path / "processed_videos.json")
    monkeypatch.setattr(yr, "_today_str", lambda: today_date)
    monkeypatch.setattr(yr, "_next_trading_date", lambda: target_date)
    monkeypatch.setattr(
        yr,
        "get_channel_config",
        lambda: {
            "scanner": {"output_dir": str(output_dir)},
            "channels": {
                "click_capital": {},
                "mike_market": {},
                "rta_trading": {},
                "trade_brigade": {},
            }
        },
    )

    status = yr.check_readiness()

    assert status["target_date"] == target_date
    assert status["videos_processed_today"] == 1
    assert "rta_trading" in status["channels_scraped"]
    assert "rta_trading" not in status["channels_missing"]


def test_get_processed_videos_reads_tracking_file_without_custom_decoder(
    monkeypatch, tmp_path: Path
):
    track_file = tmp_path / "processed_videos.json"
    track_file.write_text(
        '{"processed": {"rta_trading_demo": {"video_id": "demo123"}}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(yr, "TRACK_FILE", track_file)

    payload = yr.get_processed_videos()

    assert payload["processed"]["rta_trading_demo"]["video_id"] == "demo123"


def test_check_readiness_counts_transcript_artifacts_when_no_rag(monkeypatch, tmp_path: Path):
    rag_dir = tmp_path / "rag"
    (rag_dir / "daily_reports").mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "youtube_out"
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript_file = output_dir / "abc_example_video.txt"
    transcript_file.write_text("hello world", encoding="utf-8")

    target_date = "2026-02-13"
    today_date = "2026-02-12"

    monkeypatch.setattr(yr, "RAG_DIR", rag_dir)
    monkeypatch.setattr(yr, "TRACK_FILE", tmp_path / "processed_videos.json")
    monkeypatch.setattr(yr, "_today_str", lambda: today_date)
    monkeypatch.setattr(yr, "_next_trading_date", lambda: target_date)
    monkeypatch.setattr(
        yr,
        "get_channel_config",
        lambda: {
            "scanner": {
                "output_dir": str(output_dir),
                "readiness": {"always_check_new_videos": False},
            },
            "channels": {"rta_trading": {}},
        },
    )

    # Force mtime into one of the session dates.
    fixed_ts = datetime(2026, 2, 12, 12, 0, 0).timestamp()
    os.utime(transcript_file, (fixed_ts, fixed_ts))

    status = yr.check_readiness()

    assert status["transcript_artifacts"] >= 1
    assert status["videos_processed_today"] == 0
    assert status["needs_scan"] is True


def test_check_readiness_parses_channel_keys_with_underscored_video_ids(monkeypatch, tmp_path: Path):
    rag_dir = tmp_path / "rag"
    by_date = rag_dir / "by_date" / "2026-02-13"
    by_date.mkdir(parents=True, exist_ok=True)
    (by_date / "trade_brigade_9DKuZdjrl_c.json").write_text(
        '{"channel": "trade_brigade", "notes": "sample extraction"}',
        encoding="utf-8",
    )

    output_dir = tmp_path / "youtube_out"
    output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(yr, "RAG_DIR", rag_dir)
    monkeypatch.setattr(yr, "TRACK_FILE", tmp_path / "processed_videos.json")
    monkeypatch.setattr(yr, "_today_str", lambda: "2026-02-12")
    monkeypatch.setattr(yr, "_next_trading_date", lambda: "2026-02-13")
    monkeypatch.setattr(
        yr,
        "get_channel_config",
        lambda: {
            "scanner": {"output_dir": str(output_dir)},
            "channels": {
                "trade_brigade": {},
                "rta_trading": {},
            },
        },
    )

    status = yr.check_readiness()

    assert "trade_brigade" in status["channels_scraped"]
    assert "trade_brigade" not in status["channels_missing"]


def test_check_readiness_scans_when_missing_channels_and_insufficient_material(monkeypatch, tmp_path: Path):
    """With channels missing and < min_videos_to_skip_scan artifacts, needs_scan must be True
    even if cooldown is active.  This prevents the scanner from quitting after 1 video."""
    rag_dir = tmp_path / "rag"
    daily_reports = rag_dir / "daily_reports"
    daily_reports.mkdir(parents=True, exist_ok=True)
    # Stale report exists from two days ago.
    stale_report = daily_reports / "2026-02-10_consolidated.json"
    stale_report.write_text("{}", encoding="utf-8")
    old_ts = 946684800  # 2000-01-01
    os.utime(stale_report, (old_ts, old_ts))

    output_dir = tmp_path / "youtube_out"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Local transcript artifact indicates today's session has some material (but < 4).
    tx = output_dir / "abc_today_video.txt"
    tx.write_text("hello world", encoding="utf-8")
    fixed_ts = datetime(2026, 2, 12, 12, 0, 0).timestamp()
    os.utime(tx, (fixed_ts, fixed_ts))

    monkeypatch.setattr(yr, "RAG_DIR", rag_dir)
    monkeypatch.setattr(yr, "TRACK_FILE", tmp_path / "processed_videos.json")
    monkeypatch.setattr(yr, "_today_str", lambda: "2026-02-12")
    monkeypatch.setattr(yr, "_next_trading_date", lambda: "2026-02-13")
    monkeypatch.setattr(
        yr,
        "get_channel_config",
        lambda: {
            "scanner": {"output_dir": str(output_dir)},
            "channels": {
                "click_capital": {},
                "mike_market": {},
                "rta_trading": {},
                "trade_brigade": {},
            },
        },
    )

    status = yr.check_readiness()

    assert status["transcript_artifacts"] >= 1
    # With 4 channels missing and only 1 artifact, scan must trigger
    assert status["needs_scan"] is True
    assert status["needs_report"] is True


def test_check_readiness_can_require_full_channel_coverage(monkeypatch, tmp_path: Path):
    rag_dir = tmp_path / "rag"
    by_date = rag_dir / "by_date" / "2026-02-13"
    by_date.mkdir(parents=True, exist_ok=True)
    (by_date / "rta_trading_abc123.json").write_text(
        '{"channel": "rta_trading", "notes": "sample extraction"}',
        encoding="utf-8",
    )
    daily_reports = rag_dir / "daily_reports"
    daily_reports.mkdir(parents=True, exist_ok=True)
    (daily_reports / "2026-02-13_consolidated.json").write_text("{}", encoding="utf-8")

    output_dir = tmp_path / "youtube_out"
    output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(yr, "RAG_DIR", rag_dir)
    monkeypatch.setattr(yr, "TRACK_FILE", tmp_path / "processed_videos.json")
    monkeypatch.setattr(yr, "_today_str", lambda: "2026-02-12")
    monkeypatch.setattr(yr, "_next_trading_date", lambda: "2026-02-13")
    monkeypatch.setattr(
        yr,
        "get_channel_config",
        lambda: {
            "scanner": {
                "output_dir": str(output_dir),
                "readiness": {"require_all_channels": True},
            },
            "channels": {
                "click_capital": {},
                "rta_trading": {},
            },
        },
    )

    status = yr.check_readiness()

    assert status["needs_scan"] is True
    assert "click_capital" in status["channels_missing"]


def test_check_readiness_does_not_rescan_when_channels_checked_but_no_new_content(monkeypatch, tmp_path: Path):
    rag_dir = tmp_path / "rag"
    daily_reports = rag_dir / "daily_reports"
    daily_reports.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "youtube_out"
    output_dir.mkdir(parents=True, exist_ok=True)

    target_date = "2026-02-13"
    today_date = "2026-02-12"

    # Ready report for the target session.
    (daily_reports / f"{target_date}_consolidated.json").write_text("{}", encoding="utf-8")

    # Session scan status says all channels were checked this session, even if they had 0 uploads.
    status_file = rag_dir / f"session_scan_status_{target_date}.json"
    status_file.write_text(
        """
{
  "session_date": "2026-02-13",
  "channels_checked": ["click_capital", "mike_market", "rta_trading", "trade_brigade"],
  "channel_status": {}
}
""".strip(),
        encoding="utf-8",
    )

    # Cooldown marker exists (scan happened recently); periodic recheck disabled by default.
    scan_marker = rag_dir / ".last_scan_at"
    scan_marker.write_text("now", encoding="utf-8")

    monkeypatch.setattr(yr, "RAG_DIR", rag_dir)
    monkeypatch.setattr(yr, "TRACK_FILE", tmp_path / "processed_videos.json")
    monkeypatch.setattr(yr, "_today_str", lambda: today_date)
    monkeypatch.setattr(yr, "_next_trading_date", lambda: target_date)
    monkeypatch.setattr(
        yr,
        "get_channel_config",
        lambda: {
            "scanner": {
                "output_dir": str(output_dir),
                "readiness": {"scan_for_missing_content": False},
            },
            "channels": {
                "click_capital": {},
                "mike_market": {},
                "rta_trading": {},
                "trade_brigade": {},
            },
        },
    )

    status = yr.check_readiness()

    assert status["report_ready"] is True
    assert status["channels_unchecked"] == []
    assert set(status["channels_missing_content"]) == {
        "click_capital",
        "mike_market",
        "rta_trading",
        "trade_brigade",
    }
    assert status["needs_scan"] is False
    assert status["needs_report"] is False


def test_format_readiness_log_includes_scan_cycle_flag():
    text = yr.format_readiness_log(
        {
            "report_ready": True,
            "report_current": True,
            "report_date": "2026-02-17",
            "report_stale": False,
            "fallback_report_available": False,
            "fallback_report_date": None,
            "channels_scraped": ["mike_market"],
            "channels_checked": ["mike_market"],
            "channels_unchecked": ["click_capital"],
            "channels_missing_content": ["click_capital"],
            "channels_failed_extraction": ["trade_brigade"],
            "channels_missing": ["click_capital"],
            "videos_processed_today": 1,
            "usable_videos_processed_today": 1,
            "transcript_artifacts": 1,
            "needs_scan": False,
            "needs_report": False,
            "scan_ran": False,
        }
    )

    assert "Fallback report available: False" in text
    assert "Channels failed extraction: trade_brigade" in text
    assert "Session artifacts available: 1" in text
    assert "Usable session artifacts: 1" in text
    assert "Scan ran this cycle: False" in text


def test_check_readiness_failed_extraction_does_not_mark_current_report_ready(
    monkeypatch, tmp_path: Path
):
    rag_dir = tmp_path / "rag"
    target_date = "2026-02-13"
    by_date = rag_dir / "by_date" / target_date
    by_date.mkdir(parents=True, exist_ok=True)
    (by_date / "rta_trading_abc123.json").write_text(
        """
{
  "raw_extraction": "",
  "transcript_context": "noisy transcript",
  "_meta": {
    "extraction_failed": true,
    "error": "401 unauthorized"
  }
}
""".strip(),
        encoding="utf-8",
    )
    daily_reports = rag_dir / "daily_reports"
    daily_reports.mkdir(parents=True, exist_ok=True)
    (daily_reports / "2026-02-12_consolidated.json").write_text("{}", encoding="utf-8")

    output_dir = tmp_path / "youtube_out"
    output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(yr, "RAG_DIR", rag_dir)
    monkeypatch.setattr(yr, "TRACK_FILE", tmp_path / "processed_videos.json")
    monkeypatch.setattr(yr, "_today_str", lambda: "2026-02-12")
    monkeypatch.setattr(yr, "_next_trading_date", lambda: target_date)
    monkeypatch.setattr(
        yr,
        "get_channel_config",
        lambda: {
            "scanner": {
                "output_dir": str(output_dir),
                "readiness": {"always_check_new_videos": False},
            },
            "channels": {"rta_trading": {}},
        },
    )

    status = yr.check_readiness()

    assert status["report_ready"] is False
    assert status["videos_processed_today"] == 0
    assert status["channels_failed_extraction"] == ["rta_trading"]
    assert status["needs_report"] is True


def test_get_intelligence_context_ignores_fallback_only_reports(monkeypatch):
    calls = {"load": 0}

    def _fake_load_market_intelligence(*args, **kwargs):  # noqa: ANN002, ANN003
        calls["load"] += 1
        return {"market_regime": "BULL"}

    monkeypatch.setattr(mi, "load_market_intelligence", _fake_load_market_intelligence)
    monkeypatch.setattr(
        yr,
        "check_readiness",
        lambda: {
            "report_ready": False,
            "report_current": False,
            "fallback_report_available": True,
            "fallback_report_date": "2026-02-12",
            "target_date": "2026-02-13",
        },
    )

    ctx = yr.get_intelligence_context()

    assert ctx["available"] is False
    assert calls["load"] == 0


def test_ensure_youtube_ready_skips_report_when_scan_had_no_new_videos(monkeypatch):
    status = {
        "report_ready": True,
        "report_date": "2026-03-11",
        "report_stale": False,
        "channels_scraped": ["mike_market", "rta_trading"],
        "channels_missing": ["click_capital", "trade_brigade"],
        "needs_scan": True,
        "needs_report": False,
        "target_date": "2026-03-11",
    }
    status_after_scan = dict(status)
    final_status = dict(status)

    calls = {"check": 0, "generate_report": 0}

    def _check_readiness(target_date=None):
        calls["check"] += 1
        if calls["check"] == 1:
            return status
        if calls["check"] == 2:
            return status_after_scan
        return final_status

    monkeypatch.setattr(yr, "check_readiness", _check_readiness)
    monkeypatch.setattr(yr, "get_channel_config", lambda: {"scanner": {}, "channels": {}})
    monkeypatch.setattr(
        yr,
        "_resolve_watchdog_settings",
        lambda _cfg: {
            "scan_stall_timeout_seconds": 7200,
            "scan_hard_timeout_seconds": 0,
            "report_stall_timeout_seconds": 3600,
            "report_hard_timeout_seconds": 0,
            "progress_poll_seconds": 1,
            "verbose_subprocess_logs": False,
        },
    )
    monkeypatch.setattr(
        yr,
        "_run_youtube_scan",
        lambda **_kwargs: {"new_videos_processed": 0, "report_generated": True},
    )

    def _generate_report(**_kwargs):
        calls["generate_report"] += 1
        return True

    monkeypatch.setattr(yr, "_generate_report", _generate_report)

    result = yr.ensure_youtube_ready()

    assert result["scan_ran"] is True
    assert calls["generate_report"] == 0


def test_get_intelligence_context_surfaces_coverage_and_suppresses_minimal_sector_demotion(monkeypatch):
    status = {
        "report_ready": True,
        "target_date": "2026-05-14",
        "channels_total": 6,
        "channels_processed": 2,
        "channels_missing": ["click_capital", "trade_brigade", "stockedup", "finfluential_tv"],
        "coverage_pct": 0.3333,
        "coverage_grade": "minimal",
    }

    monkeypatch.setattr(yr, "check_readiness", lambda: status)
    monkeypatch.setattr(
        mi,
        "load_market_intelligence",
        lambda **_kwargs: {
            "market_regime": "RISK-OFF",
            "regime_confidence": 80,
            "trading_signals": {
                "sizing_multiplier": 0.5,
                "sector_bias": [{"sector": "Technology", "bias": "avoid"}],
            },
            "_meta": {"date": "2026-05-14", "_source": "daily"},
        },
    )

    ctx = yr.get_intelligence_context()

    assert ctx["channels_total"] == 6
    assert ctx["channels_processed"] == 2
    assert ctx["coverage_grade"] == "minimal"
    assert ctx["sector_demotion_available"] is False
    assert ctx["avoid_sectors"] == []
    assert ctx["sizing_multiplier"] == 0.75


def test_ensure_youtube_ready_generates_report_when_scan_found_new_videos_without_report(
    monkeypatch,
):
    status = {
        "report_ready": True,
        "report_date": "2026-03-11",
        "report_stale": False,
        "channels_scraped": ["mike_market", "rta_trading"],
        "channels_missing": ["click_capital", "trade_brigade"],
        "needs_scan": True,
        "needs_report": False,
        "target_date": "2026-03-11",
    }
    status_after_scan = dict(status)
    final_status = dict(status)

    calls = {"check": 0, "generate_report": 0}

    def _check_readiness(target_date=None):
        calls["check"] += 1
        if calls["check"] == 1:
            return status
        if calls["check"] == 2:
            return status_after_scan
        return final_status

    monkeypatch.setattr(yr, "check_readiness", _check_readiness)
    monkeypatch.setattr(yr, "get_channel_config", lambda: {"scanner": {}, "channels": {}})
    monkeypatch.setattr(
        yr,
        "_resolve_watchdog_settings",
        lambda _cfg: {
            "scan_stall_timeout_seconds": 7200,
            "scan_hard_timeout_seconds": 0,
            "report_stall_timeout_seconds": 3600,
            "report_hard_timeout_seconds": 0,
            "progress_poll_seconds": 1,
            "verbose_subprocess_logs": False,
        },
    )
    monkeypatch.setattr(
        yr,
        "_run_youtube_scan",
        lambda **_kwargs: {"new_videos_processed": 2, "report_generated": False},
    )

    def _generate_report(**_kwargs):
        calls["generate_report"] += 1
        return True

    monkeypatch.setattr(yr, "_generate_report", _generate_report)

    result = yr.ensure_youtube_ready()

    assert result["scan_ran"] is True
    assert calls["generate_report"] == 1


def test_ensure_youtube_ready_skips_report_when_scan_degraded(monkeypatch):
    status = {
        "report_ready": False,
        "report_date": None,
        "report_stale": True,
        "channels_scraped": [],
        "channels_missing": ["rta_trading"],
        "needs_scan": True,
        "needs_report": True,
        "target_date": "2026-03-11",
    }
    final_status = dict(status)

    calls = {"check": 0, "generate_report": 0}

    def _check_readiness(target_date=None):
        calls["check"] += 1
        if calls["check"] == 1:
            return status
        return final_status

    monkeypatch.setattr(yr, "check_readiness", _check_readiness)
    monkeypatch.setattr(yr, "get_channel_config", lambda: {"scanner": {}, "channels": {}})
    monkeypatch.setattr(
        yr,
        "_resolve_watchdog_settings",
        lambda _cfg: {
            "scan_stall_timeout_seconds": 7200,
            "scan_hard_timeout_seconds": 0,
            "report_stall_timeout_seconds": 3600,
            "report_hard_timeout_seconds": 0,
            "progress_poll_seconds": 1,
            "verbose_subprocess_logs": False,
        },
    )
    monkeypatch.setattr(
        yr,
        "_run_youtube_scan",
        lambda **_kwargs: {
            "new_videos_processed": 0,
            "report_generated": False,
            "scan_degraded": True,
            "metadata_failed_channels": ["rta_trading"],
        },
    )

    def _generate_report(**_kwargs):
        calls["generate_report"] += 1
        return True

    monkeypatch.setattr(yr, "_generate_report", _generate_report)

    result = yr.ensure_youtube_ready()

    assert result["scan_ran"] is True
    assert result["scan_error"] == "youtube_scan_degraded:rta_trading"
    assert calls["generate_report"] == 0
