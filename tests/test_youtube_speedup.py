import json
import sys
import types
from pathlib import Path

from tools.youtube_daily_scanner import (
    _safe_console_text,
    _split_channel_and_video_id,
    _timing_stats,
    extract_with_llm,
    filter_new_videos,
    get_recent_videos,
    get_recent_videos_status,
    run_full_scan,
    transcribe_videos_batch,
)
from tools.youtube_transcriber import _parse_bool
from tools import youtube_transcriber as ytt


def test_parse_bool_variants() -> None:
    assert _parse_bool("true") is True
    assert _parse_bool("false") is False
    assert _parse_bool("1") is True
    assert _parse_bool("0") is False
    assert _parse_bool(None, default=False) is False


def test_resolve_preferred_transcriber_python_reads_config_override(monkeypatch, tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    preferred_python = tmp_path / "miniconda3" / "envs" / "mlgpu" / "python.exe"
    preferred_python.parent.mkdir(parents=True, exist_ok=True)
    preferred_python.write_text("", encoding="utf-8")
    (config_dir / "youtube_channels.yaml").write_text(
        "scanner:\n  transcription:\n    python_path: \"" + str(preferred_python).replace("\\", "\\\\") + "\"\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(ytt, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("YOUTUBE_TRANSCRIBER_PYTHON", raising=False)

    assert ytt._resolve_preferred_transcriber_python() == str(preferred_python)


def test_timing_stats_summary() -> None:
    stats = _timing_stats([1.0, 2.0, 3.0, 4.0])
    assert stats["count"] == 4
    assert stats["min"] == 1.0
    assert stats["median"] == 2.5
    assert stats["max"] == 4.0
    assert stats["p95"] >= stats["median"]


def test_safe_console_text_handles_unicode() -> None:
    value = _safe_console_text("Hello 🚨")
    assert isinstance(value, str)
    assert "Hello" in value


def test_get_recent_videos_marks_metadata_degraded_when_detail_fetches_all_fail(monkeypatch) -> None:
    class _Boom(Exception):
        pass

    seen_opts = []

    class _FakeYoutubeDL:
        def __init__(self, opts):
            self.opts = opts
            seen_opts.append(opts)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download=False):
            if url.endswith("/videos"):
                return {
                    "entries": [
                        {"id": "abc123", "title": "Video 1", "upload_date": "", "duration": 10},
                        {"id": "def456", "title": "Video 2", "upload_date": "", "duration": 20},
                    ]
                }
            raise _Boom("detail fetch failed")

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=_FakeYoutubeDL))
    monkeypatch.setattr("tools.youtube_daily_scanner._get_cookie_opts", lambda: {})

    videos = get_recent_videos("https://www.youtube.com/@Example", max_age_hours=200, max_results=5, retries=1)
    status = get_recent_videos_status("https://www.youtube.com/@Example")

    assert videos == []
    assert status["entries_seen"] == 2
    assert status["missing_upload_dates"] == 2
    assert status["metadata_degraded"] is True
    assert status["had_errors"] is True
    assert seen_opts
    assert all(opts.get("encoding") == "utf-8" for opts in seen_opts)


def test_transcribe_videos_batch_maps_results(monkeypatch, tmp_path: Path) -> None:
    txt_file = tmp_path / "abc.txt"
    txt_file.write_text("transcript from file", encoding="utf-8")

    def fake_runner(entries, output_dir, config, timeout_seconds=None):  # noqa: ANN001
        return [
            {
                "video_id": "abc",
                "success": True,
                "elapsed_seconds": 12.3,
                "files": {"txt": str(txt_file), "json": str(tmp_path / "abc.json")},
                "transcript": "",
            }
        ]

    monkeypatch.setattr("tools.youtube_daily_scanner._run_transcriber_entries", fake_runner)

    videos = [{"id": "abc", "title": "My Video", "url": "https://example.com"}]
    out = transcribe_videos_batch(videos=videos, output_dir=tmp_path, config={"scanner": {}})

    assert "abc" in out
    assert out["abc"]["success"] is True
    assert out["abc"]["transcript"] == "transcript from file"
    assert out["abc"]["elapsed_seconds"] == 12.3


def test_transcribe_videos_batch_reuses_cached_transcript(monkeypatch, tmp_path: Path) -> None:
    video_id = "abc"
    title = "My Video"
    cached_txt = tmp_path / "abc_my_video.txt"
    cached_txt.write_text("cached transcript " * 20, encoding="utf-8")
    cached_json = tmp_path / "abc_my_video.json"
    cached_json.write_text("{}", encoding="utf-8")

    called = {"runner": 0}

    def fake_runner(entries, output_dir, config, timeout_seconds=None):  # noqa: ANN001
        called["runner"] += 1
        return []

    monkeypatch.setattr("tools.youtube_daily_scanner._run_transcriber_entries", fake_runner)

    videos = [{"id": video_id, "title": title, "url": "https://example.com"}]
    out = transcribe_videos_batch(videos=videos, output_dir=tmp_path, config={"scanner": {}})

    assert called["runner"] == 0
    assert out[video_id]["success"] is True
    assert out[video_id]["reused_cached_transcript"] is True
    assert "cached transcript" in out[video_id]["transcript"]


def test_split_channel_and_video_id_with_underscores() -> None:
    ck, vid = _split_channel_and_video_id(
        "trade_brigade_9DKuZdjrl_c",
        known_channels=["trade_brigade", "rta_trading"],
    )
    assert ck == "trade_brigade"
    assert vid == "9DKuZdjrl_c"


def test_extract_with_llm_returns_fallback_on_exception(monkeypatch) -> None:
    class _Resp:
        def raise_for_status(self) -> None:  # pragma: no cover
            return None

        def json(self):  # pragma: no cover
            return {}

    def _boom(*args, **kwargs):  # noqa: ANN001
        raise TimeoutError("simulated timeout")

    monkeypatch.setattr("requests.post", _boom)

    cfg = {
        "scanner": {
            "extraction": {
                "model": "fake-model",
                "fallback_model": "fake-fallback",
                "ollama_url": "http://localhost:11434",
                "timeout": 1,
            }
        }
    }
    out = extract_with_llm(
        transcript="hello world " * 1000,
        template_prompt="{transcript}",
        channel_name="Test Channel",
        video_title="Test Video",
        video_date="20260213",
        config=cfg,
    )

    assert isinstance(out, dict)
    meta = out.get("_meta", {})
    assert meta.get("extraction_failed") is True
    assert "timeout" in str(meta.get("error", "")).lower()


def test_extract_with_llm_survives_unrenderable_exception_text(monkeypatch) -> None:
    class _BrokenExc(Exception):
        def __str__(self):
            raise RecursionError("maximum recursion depth exceeded while getting str")

    def _boom(*args, **kwargs):  # noqa: ANN001
        raise _BrokenExc("boom")

    monkeypatch.setattr("requests.post", _boom)

    cfg = {
        "scanner": {
            "extraction": {
                "model": "fake-model",
                "fallback_model": "fake-fallback",
                "ollama_url": "http://localhost:11434",
                "timeout": 1,
            }
        }
    }
    out = extract_with_llm(
        transcript="hello world " * 1000,
        template_prompt="{transcript}",
        channel_name="Test Channel",
        video_title="Test Video",
        video_date="20260213",
        config=cfg,
    )

    assert isinstance(out, dict)
    meta = out.get("_meta", {})
    assert meta.get("extraction_failed") is True
    assert "_BrokenExc" in str(meta.get("error", ""))


def test_filter_new_videos_retries_failed_entries(tmp_path: Path) -> None:
    processed = {
        "processed": {
            "abc": {"status": "extraction_failed"},
            "def": {"status": "complete"},
            "ghi": {"status": "complete_with_fallback"},
        }
    }
    videos = [
        {"id": "abc", "title": "retry me"},
        {"id": "def", "title": "skip complete"},
        {"id": "ghi", "title": "skip fallback complete"},
        {"id": "xyz", "title": "new"},
    ]
    out = filter_new_videos(videos=videos, processed=processed, channel_key="rta_trading", rag_dir=tmp_path)
    out_ids = {v["id"] for v in out}
    assert "abc" in out_ids
    assert "xyz" in out_ids
    assert "def" not in out_ids
    assert "ghi" not in out_ids


def test_filter_new_videos_retries_complete_with_fallback_when_error_present(tmp_path: Path) -> None:
    processed = {
        "processed": {
            "abc": {
                "status": "complete_with_fallback",
                "error": "401 unauthorized",
            }
        }
    }
    videos = [{"id": "abc", "title": "retry me"}]

    out = filter_new_videos(videos=videos, processed=processed, channel_key="rta_trading", rag_dir=tmp_path)

    assert [video["id"] for video in out] == ["abc"]


def test_filter_new_videos_does_not_recover_unusable_rag_artifact(tmp_path: Path) -> None:
    rag_dir = tmp_path / "rag"
    channel_dir = rag_dir / "by_channel" / "rta_trading"
    channel_dir.mkdir(parents=True, exist_ok=True)
    (channel_dir / "rta_trading_abc.json").write_text(
        json.dumps(
            {
                "raw_extraction": "",
                "transcript_context": "bad transcript",
                "_meta": {"extraction_failed": True},
            }
        ),
        encoding="utf-8",
    )

    out = filter_new_videos(
        videos=[{"id": "abc", "title": "retry me"}],
        processed={"processed": {}},
        channel_key="rta_trading",
        rag_dir=rag_dir,
    )

    assert [video["id"] for video in out] == ["abc"]


def test_run_full_scan_writes_session_scan_status_artifact(monkeypatch, tmp_path: Path) -> None:
    rag_dir = tmp_path / "rag"
    out_dir = tmp_path / "out"
    rag_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("tools.youtube_daily_scanner.load_processed_videos", lambda _path: {"processed": {}})
    monkeypatch.setattr("tools.youtube_daily_scanner.save_processed_videos", lambda _path, _data: None)
    monkeypatch.setattr("tools.youtube_daily_scanner.get_recent_videos", lambda *_a, **_kw: [])
    monkeypatch.setattr("tools.youtube_daily_scanner.generate_daily_report", lambda *_a, **_kw: None)

    config = {
        "channels": {
            "trade_brigade": {
                "name": "Trade Brigade",
                "channel_url": "https://example.com/trade_brigade",
                "template": "prompts/llm/youtube_trade_brigade.md",
                "priority": 1,
                "schedule": {"max_age_hours": 36},
            },
            "rta_trading": {
                "name": "Arete Trading",
                "channel_url": "https://example.com/rta_trading",
                "template": "prompts/llm/youtube_rta_trading.md",
                "priority": 2,
                "schedule": {"max_age_hours": 36},
            },
        },
        "scanner": {
            "output_dir": str(out_dir),
            "rag_dir": str(rag_dir),
            "report": {"generate_daily_report": False},
            "performance": {"max_channel_workers": 2, "max_extraction_workers": 1},
            "dedup": {"track_file": str(tmp_path / "processed_videos.json")},
        },
    }

    summary = run_full_scan(config=config)

    status_file = Path(summary["session_scan_status_file"])
    assert status_file.exists()
    payload = json.loads(status_file.read_text(encoding="utf-8"))
    assert payload["session_date"] == summary["date"]
    assert set(payload["channels_checked"]) == {"trade_brigade", "rta_trading"}
    assert payload["channel_status"]["trade_brigade"]["recent_found_count"] == 0
    assert payload["channel_status"]["rta_trading"]["new_processed_count"] == 0


def test_run_full_scan_skips_report_when_metadata_degraded(monkeypatch, tmp_path: Path) -> None:
    rag_dir = tmp_path / "rag"
    out_dir = tmp_path / "out"
    rag_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("tools.youtube_daily_scanner.load_processed_videos", lambda _path: {"processed": {}})
    monkeypatch.setattr("tools.youtube_daily_scanner.save_processed_videos", lambda _path, _data: None)
    monkeypatch.setattr("tools.youtube_daily_scanner.get_recent_videos", lambda *_a, **_kw: [])
    monkeypatch.setattr(
        "tools.youtube_daily_scanner.get_recent_videos_status",
        lambda _url: {
            "entries_seen": 2,
            "missing_upload_dates": 2,
            "detail_failures": 2,
            "metadata_degraded": True,
            "had_errors": True,
            "used_cookie_fallback": False,
        },
    )

    called = {"report": 0}

    def _report(*_args, **_kwargs):
        called["report"] += 1
        return {"ok": True}

    monkeypatch.setattr("tools.youtube_daily_scanner.generate_daily_report", _report)

    config = {
        "channels": {
            "trade_brigade": {
                "name": "Trade Brigade",
                "channel_url": "https://example.com/trade_brigade",
                "template": "prompts/llm/youtube_trade_brigade.md",
                "priority": 1,
                "schedule": {"max_age_hours": 36},
            },
        },
        "scanner": {
            "output_dir": str(out_dir),
            "rag_dir": str(rag_dir),
            "report": {"generate_daily_report": True},
            "performance": {"max_channel_workers": 1, "max_extraction_workers": 1},
            "dedup": {"track_file": str(tmp_path / "processed_videos.json")},
        },
    }

    summary = run_full_scan(config=config)

    assert summary["scan_degraded"] is True
    assert summary["metadata_failed_channels"] == ["trade_brigade"]
    assert summary["report_generated"] is False
    assert called["report"] == 0
