#!/usr/bin/env python3
"""
YouTube Transcriber for AutoTrade
=================================
Transcribe YouTube videos to text for market sentiment analysis.

Based on YouTubeScraper project - uses faster_whisper + yt_dlp for 
GPU-accelerated transcription of trading commentary videos.

Usage:
    python tools/youtube_transcriber.py --url "https://youtube.com/watch?v=..."
    python tools/youtube_transcriber.py --urls urls.txt
    python tools/youtube_transcriber.py --audio local_file.mp3

Output:
    - .txt: Plain transcript
    - .srt/.vtt: Subtitles with timestamps
    - .json: Structured segments with timing
    - _chunks.md: Chunked for LLM analysis

Stock Universe Context:
    AutoTrade trades ~1600 small/mid-cap stocks ($2-$200) from the DownDay dataset.
    NOT mega-caps like AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META.
    Transcripts are analyzed for sentiment on small-cap momentum plays.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent

# OpenMP duplicate runtime guard (common on Windows)
if not os.environ.get("KMP_DUPLICATE_LIB_OK"):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# CUDA DLL path setup for Windows
def ensure_cuda_dll_paths(verbose: bool = False) -> None:
    """Add CUDA DLL directories to PATH on Windows."""
    if os.name != "nt":
        return
    cuda_dirs = [
        os.path.join(os.environ.get("CUDA_PATH", ""), "bin"),
        os.path.join(os.environ.get("CUDA_HOME", ""), "bin"),
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
    ]
    for d in cuda_dirs:
        if d and os.path.isdir(d):
            if verbose:
                print(f"[CUDA] Adding to DLL search: {d}")
            try:
                os.add_dll_directory(d)
            except (OSError, AttributeError):
                pass
            if d not in os.environ.get("PATH", ""):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")

def debug_cuda_environment(always: bool = False) -> None:
    """Print CUDA diagnostic information."""
    if not always:
        return
    print("\n=== CUDA Environment Diagnostics ===")
    print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'not set')}")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'not set')}")
    
    # Check for cublas64 DLL
    cublas_found = False
    for p in os.environ.get("PATH", "").split(os.pathsep):
        for dll in ["cublas64_12.dll", "cublas64_11.dll"]:
            check = os.path.join(p, dll)
            if os.path.isfile(check):
                print(f"Found: {check}")
                cublas_found = True
    if not cublas_found:
        print("WARNING: cuBLAS DLL not found in PATH")
    print("=====================================\n")

# Lazy imports
try:
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TimeElapsedColumn
    from rich.table import Table
    from rich import box
    console = Console()
except ImportError:
    console = None
    Progress = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    from yt_dlp import YoutubeDL
except ImportError:
    YoutubeDL = None


def _configure_utf8_stdio() -> None:
    """Prefer UTF-8 stdio on Windows so yt-dlp output cannot break on cp1252."""
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


def _safe_console_text(text: Any) -> str:
    if text is None:
        return ""
    encoding = (getattr(sys.stdout, "encoding", None) or "utf-8")
    try:
        return str(text).encode(encoding, errors="replace").decode(encoding, errors="replace")
    except Exception:
        return str(text).encode("ascii", errors="replace").decode("ascii", errors="replace")


def _safe_exception_text(exc: Exception, max_len: int = 500) -> str:
    exc_type = type(exc).__name__ or "Exception"
    try:
        message = str(exc).strip()
    except Exception:
        try:
            message = repr(exc).strip()
        except Exception:
            message = "<unrenderable>"
    message = _safe_console_text(message)
    if max_len > 0 and len(message) > max_len:
        message = message[: max_len - 15].rstrip() + "... [truncated]"
    if not message or message == exc_type:
        return exc_type
    if message.startswith(f"{exc_type}:"):
        return message
    return f"{exc_type}: {message}"


def slugify(text: str, max_len: int = 100) -> str:
    """Convert text to filesystem-safe slug."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s_-]+", "_", text)[:max_len]


def ts_srt(sec: float) -> str:
    """Format seconds as SRT timestamp."""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")


def write_srt(segs: List[dict], p: Path) -> None:
    """Write SRT subtitle file."""
    with p.open("w", encoding="utf-8") as f:
        for i, s in enumerate(segs, 1):
            f.write(f"{i}\n{ts_srt(s['start'])} --> {ts_srt(s['end'])}\n{s['text'].strip()}\n\n")


def write_vtt(segs: List[dict], p: Path) -> None:
    """Write VTT subtitle file."""
    with p.open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for s in segs:
            a = ts_srt(s['start']).replace(",", ".")
            b = ts_srt(s['end']).replace(",", ".")
            f.write(f"{a} --> {b}\n{s['text'].strip()}\n\n")


def chunk_text(text: str, max_chars: int = 2000) -> List[str]:
    """Split text into chunks for LLM processing."""
    chunks, buf, size = [], [], 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if size + len(line) + 1 > max_chars and buf:
            chunks.append("\n".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += len(line) + 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks


# yt-dlp requires a JavaScript runtime for YouTube's signature/n challenges.
_YDL_JS_RUNTIME = {"js_runtimes": {"node": {}}}


def _get_cookie_opts() -> dict:
    """Return explicit cookie-file opts when the user exported one."""
    cookie_file = Path(__file__).parent.parent / "config" / "youtube_cookies.txt"
    if cookie_file.exists() and cookie_file.stat().st_size > 100:
        return {"cookiefile": str(cookie_file)}
    return {}


def _looks_like_auth_error(exc: Exception) -> bool:
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
        "http error 429",
        "too many requests",
    )
    return any(marker in msg for marker in markers)


def _resolve_preferred_transcriber_python() -> Optional[str]:
    """Resolve the preferred Python for faster-whisper transcription."""
    env_override = os.environ.get("YOUTUBE_TRANSCRIBER_PYTHON")
    if env_override and Path(env_override).exists():
        return env_override

    config_path = PROJECT_ROOT / "config" / "youtube_channels.yaml"
    if config_path.exists():
        try:
            import yaml

            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            python_path = (
                ((config.get("scanner") or {}).get("transcription") or {}).get("python_path")
            )
            if python_path and Path(str(python_path)).exists():
                return str(python_path)
        except Exception:
            pass

    fallback = Path.home() / "miniconda3" / "envs" / "mlgpu" / "python.exe"
    if fallback.exists():
        return str(fallback)
    return None


def _maybe_reexec_in_preferred_env() -> None:
    """Re-launch under the configured transcription env when faster-whisper is absent."""
    if WhisperModel is not None:
        return
    if os.environ.get("_AUTOTRADE_YT_REEXEC") == "1":
        return

    preferred_python = _resolve_preferred_transcriber_python()
    if not preferred_python:
        return
    if Path(preferred_python).resolve() == Path(sys.executable).resolve():
        return

    print(
        f"[INFO] faster-whisper unavailable in {sys.executable}; "
        f"re-launching youtube_transcriber.py with {preferred_python}"
    )
    env = os.environ.copy()
    env["_AUTOTRADE_YT_REEXEC"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    cmd = [preferred_python, str(Path(__file__).resolve()), *sys.argv[1:]]
    raise SystemExit(subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT)).returncode)


def download_audio(url: str, out_dir: Path, ffmpeg_dir: Optional[str] = None) -> Path:
    """Download audio from YouTube URL."""
    if YoutubeDL is None:
        raise RuntimeError("yt-dlp is missing. Run: pip install yt-dlp")

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.resolve() for p in out_dir.glob("*.m4a")}

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(title)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "m4a",
            "preferredquality": "192"
        }],
        **_YDL_JS_RUNTIME,
    }
    if ffmpeg_dir:
        ydl_opts["ffmpeg_location"] = ffmpeg_dir
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as exc:
        cookie_opts = _get_cookie_opts()
        if not (cookie_opts and _looks_like_auth_error(exc)):
            raise
        if console:
            console.print("[yellow]Retrying download with exported YouTube cookies.[/yellow]")
        else:
            print("Retrying download with exported YouTube cookies.")
        retry_opts = {**ydl_opts, **cookie_opts}
        with YoutubeDL(retry_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    
    # Find the downloaded file
    candidates: list[Path] = []
    
    def register(path_like: Optional[str]) -> None:
        if not path_like:
            return
        base = Path(path_like)
        possible = [
            base,
            base.with_suffix(".m4a"),
            out_dir / base.name,
            (out_dir / base.name).with_suffix(".m4a"),
        ]
        for candidate in possible:
            resolved = candidate if candidate.is_absolute() else (out_dir / candidate)
            resolved = resolved.resolve()
            if resolved not in candidates:
                candidates.append(resolved)
    
    register(info.get("_filename"))
    register(info.get("filepath"))
    for entry in info.get("requested_downloads") or []:
        register(entry.get("filepath"))
        register(entry.get("_filename"))
    
    for path in candidates:
        if path.exists():
            return path
    
    # Fallback: find newest .m4a file
    new_files = [p for p in out_dir.glob("*.m4a") if p.resolve() not in existing]
    if new_files:
        return max(new_files, key=lambda p: p.stat().st_mtime)
    
    all_files = list(out_dir.glob("*.m4a"))
    if all_files:
        return max(all_files, key=lambda p: p.stat().st_mtime)
    
    raise FileNotFoundError(f"Unable to locate downloaded audio for {url}")


def _parse_bool(value: Any, default: bool = True) -> bool:
    """Parse bool-like CLI/config values."""
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


def _resolve_device_dtype(device: str, dtype: str) -> Tuple[str, str, str]:
    """Resolve effective device/dtype and preserve requested device for fallback logic."""
    requested_device = (device or "auto").lower()
    actual_device = requested_device
    actual_dtype = dtype

    if requested_device in ("auto", "cuda"):
        ensure_cuda_dll_paths(verbose=False)
    if requested_device == "auto":
        actual_device = "cuda"
    if actual_device == "cpu" and str(actual_dtype).lower() in {"float16", "bfloat16", "int8_float16", "int8_bfloat16"}:
        actual_dtype = "int8" if "int8" in str(dtype).lower() else "float32"

    return requested_device, actual_device, actual_dtype


def _init_whisper_model(model: str, device: str, dtype: str):
    """Initialize faster-whisper with graceful fallback."""
    if WhisperModel is None:
        raise RuntimeError("faster-whisper is not installed. Run: pip install faster-whisper")

    requested_device, actual_device, actual_dtype = _resolve_device_dtype(device, dtype)

    if console:
        console.print(
            f"[bold]Model:[/bold] {model}  [bold]Device:[/bold] {actual_device}  [bold]DType:[/bold] {actual_dtype}"
        )
    else:
        print(f"Model: {model}  Device: {actual_device}  DType: {actual_dtype}")

    try:
        wmodel = WhisperModel(model, device=actual_device, compute_type=actual_dtype)
    except RuntimeError as e:
        msg = str(e).lower()
        if console:
            console.print(f"[red]Whisper model init error: {e}[/red]")
        if (requested_device == "auto") and ("cublas" in msg or "library" in msg):
            if console:
                console.print("[yellow]CUDA/cuBLAS not available; falling back to CPU.[/yellow]")
            else:
                print("CUDA/cuBLAS not available; falling back to CPU.")
            actual_device = "cpu"
            actual_dtype = "int8" if "int8" in str(dtype).lower() else "float32"
            wmodel = WhisperModel(model, device=actual_device, compute_type=actual_dtype)
        else:
            raise

    return wmodel, requested_device, actual_device, actual_dtype


def transcribe_file(
    audio: Path,
    model: str = "large-v3",
    device: str = "auto",
    dtype: str = "float16",
    language: Optional[str] = None,
    out_dir: Path = Path("outputs"),
    max_chars: int = 2000,
    beam_size: int = 5,
    best_of: int = 5,
    condition_on_previous_text: bool = True,
    vad_filter: bool = True,
    output_prefix: Optional[str] = None,
    artifact_mode: str = "full",
    whisper_model=None,
    requested_device_hint: Optional[str] = None,
) -> dict:
    """
    Transcribe audio file to multiple formats.

    Returns dict with paths to output files and full transcript.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base = slugify(output_prefix or audio.stem)
    artifact_mode = (artifact_mode or "full").strip().lower()
    write_full_artifacts = artifact_mode == "full"

    if whisper_model is None:
        whisper_model, requested_device, _, _ = _init_whisper_model(model, device, dtype)
    else:
        requested_device = (requested_device_hint or device or "auto").lower()

    decode_kwargs: Dict[str, Any] = {
        "language": language,
        "vad_filter": bool(vad_filter),
        "beam_size": max(1, int(beam_size)),
        "best_of": max(1, int(best_of)),
        "condition_on_previous_text": bool(condition_on_previous_text),
    }
    if bool(vad_filter):
        decode_kwargs["vad_parameters"] = dict(min_silence_duration_ms=500)

    try:
        segments, info = whisper_model.transcribe(str(audio), **decode_kwargs)
    except RuntimeError as e:
        msg = str(e).lower()
        if (requested_device == "auto") and ("cublas" in msg or "library" in msg):
            if console:
                console.print("[yellow]CUDA error during transcription; retrying on CPU.[/yellow]")
            cpu_model = WhisperModel(model, device="cpu", compute_type="float32")
            segments, info = cpu_model.transcribe(str(audio), **decode_kwargs)
        else:
            raise

    segs: List[dict] = []
    if Progress and console:
        with Progress(
            "{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
        ) as prog:
            task = prog.add_task("Transcribing...", total=None)
            for s in segments:
                segs.append({"start": s.start, "end": s.end, "text": s.text})
                prog.advance(task)
    else:
        print("Transcribing...")
        for s in segments:
            segs.append({"start": s.start, "end": s.end, "text": s.text})

    txt = out_dir / f"{base}.txt"
    srt = out_dir / f"{base}.srt"
    vtt = out_dir / f"{base}.vtt"
    jsn = out_dir / f"{base}.json"
    chm = out_dir / f"{base}_chunks.md"

    full = "\n".join(s["text"].strip() for s in segs)
    txt.write_text(full, encoding="utf-8")
    if write_full_artifacts:
        write_srt(segs, srt)
        write_vtt(segs, vtt)

    jsn.write_text(
        json.dumps(
            {
                "language": info.language,
                "duration": info.duration,
                "segments": segs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    chunks = chunk_text(full, max_chars=max_chars)
    if write_full_artifacts:
        chm.write_text(
            "\n\n".join(f"## Chunk {i+1}\n\n{c}" for i, c in enumerate(chunks)),
            encoding="utf-8",
        )

    if console:
        tab = Table(title="Outputs", box=box.SIMPLE)
        tab.add_column("File")
        tab.add_column("Path")
        files_to_show = [txt, jsn]
        if write_full_artifacts:
            files_to_show.extend([srt, vtt, chm])
        for p in files_to_show:
            tab.add_row(p.name, str(p.resolve()))
        console.print(tab)
    else:
        print(f"\nOutputs saved to {out_dir}:")
        print(f"  - {txt.name}")
        print(f"  - {jsn.name}")
        if write_full_artifacts:
            print(f"  - {srt.name}")
            print(f"  - {vtt.name}")
            print(f"  - {chm.name}")

    return {
        "transcript": full,
        "language": info.language,
        "duration": info.duration,
        "segments": segs,
        "chunks": chunks,
        "artifact_mode": artifact_mode,
        "files": {
            "txt": str(txt),
            "srt": str(srt) if write_full_artifacts else None,
            "vtt": str(vtt) if write_full_artifacts else None,
            "json": str(jsn),
            "chunks_md": str(chm) if write_full_artifacts else None,
        },
    }


def transcribe_url(
    url: str,
    model: str = "large-v3",
    device: str = "auto",
    dtype: str = "float16",
    language: Optional[str] = None,
    out_dir: Path = Path("data/youtube"),
    ffmpeg_dir: Optional[str] = None,
    max_chars: int = 2000,
    beam_size: int = 5,
    best_of: int = 5,
    condition_on_previous_text: bool = True,
    vad_filter: bool = True,
    output_prefix: Optional[str] = None,
    artifact_mode: str = "full",
    video_id: Optional[str] = None,
    whisper_model=None,
    requested_device_hint: Optional[str] = None,
) -> dict:
    """
    Download and transcribe a YouTube video.

    This is the main entry point for AutoTrade integration.
    """
    out_dir = Path(out_dir)
    dl_dir = out_dir / "downloads"

    if console:
        console.print(f"[cyan]Downloading:[/cyan] {url}")
    else:
        print(f"Downloading: {url}")

    audio = download_audio(url, dl_dir, ffmpeg_dir)
    result = transcribe_file(
        audio=audio,
        model=model,
        device=device,
        dtype=dtype,
        language=language,
        out_dir=out_dir,
        max_chars=max_chars,
        beam_size=beam_size,
        best_of=best_of,
        condition_on_previous_text=condition_on_previous_text,
        vad_filter=vad_filter,
        output_prefix=output_prefix,
        artifact_mode=artifact_mode,
        whisper_model=whisper_model,
        requested_device_hint=requested_device_hint,
    )
    result["url"] = url
    result["audio_file"] = str(audio)
    if video_id:
        result["video_id"] = video_id

    return result


def _process_batch(
    entries: List[dict],
    model: str,
    device: str,
    dtype: str,
    language: Optional[str],
    out_dir: Path,
    ffmpeg_location: Optional[str],
    max_chars: int,
    beam_size: int,
    best_of: int,
    condition_on_previous_text: bool,
    vad_filter: bool,
    artifact_mode: str,
) -> List[dict]:
    """Process a list of URLs with one shared Whisper model load."""
    results: List[dict] = []
    whisper_model = None
    requested_device = (device or "auto").lower()

    if entries:
        whisper_model, requested_device, _, _ = _init_whisper_model(model, device, dtype)

    for entry in entries:
        url = str(entry.get("url") or "").strip()
        video_id = str(entry.get("video_id") or "").strip() or None
        output_prefix = entry.get("output_prefix")
        if not url:
            results.append({"url": url, "video_id": video_id, "success": False, "error": "missing_url"})
            continue

        try:
            started = time.time()
            result = transcribe_url(
                url=url,
                model=model,
                device=device,
                dtype=dtype,
                language=language,
                out_dir=out_dir,
                ffmpeg_dir=ffmpeg_location,
                max_chars=max_chars,
                beam_size=beam_size,
                best_of=best_of,
                condition_on_previous_text=condition_on_previous_text,
                vad_filter=vad_filter,
                output_prefix=output_prefix,
                artifact_mode=artifact_mode,
                video_id=video_id,
                whisper_model=whisper_model,
                requested_device_hint=requested_device,
            )
            results.append(
                {
                    "url": url,
                    "video_id": video_id,
                    "success": True,
                    "elapsed_seconds": time.time() - started,
                    "transcript_chars": len(result.get("transcript", "")),
                    "duration": result.get("duration"),
                    "files": result.get("files", {}),
                    "transcript": result.get("transcript", ""),
                }
            )
        except Exception as e:
            results.append({"url": url, "video_id": video_id, "success": False, "error": str(e)})

    return results


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Transcribe YouTube videos for market sentiment analysis")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--url", action="append", help="YouTube URL(s) to transcribe")
    g.add_argument("--urls", help="Text file with URLs (one per line)")
    g.add_argument("--audio", help="Local audio file to transcribe")
    g.add_argument("--batch-json", help="JSON file with list of {'url','video_id','output_prefix'} entries")

    ap.add_argument("--model", default="large-v3", help="Whisper model (default: large-v3)")
    ap.add_argument("--device", default="auto", help="Device: auto, cuda, cpu (default: auto)")
    ap.add_argument("--dtype", default="float16", help="Compute type (default: float16)")
    ap.add_argument("--language", default=None, help="Force language (default: auto-detect)")
    ap.add_argument("--output-dir", default="data/youtube", help="Output directory")
    ap.add_argument("--max-chars", type=int, default=2000, help="Chunk size for optional chunk artifacts")
    ap.add_argument("--ffmpeg-location", default=None, help="Path to ffmpeg directory")
    ap.add_argument("--beam-size", type=int, default=5, help="Whisper beam size")
    ap.add_argument("--best-of", type=int, default=5, help="Whisper best_of")
    ap.add_argument(
        "--condition-on-previous-text",
        default="true",
        help="Use previous text conditioning (true/false)",
    )
    ap.add_argument("--vad-filter", default="true", help="Enable VAD filtering (true/false)")
    ap.add_argument(
        "--artifact-mode",
        choices=["full", "scanner"],
        default="full",
        help="full=txt/json/srt/vtt/chunks, scanner=txt/json only",
    )
    ap.add_argument("--output-prefix", default=None, help="Custom output prefix for generated files")
    ap.add_argument("--video-id", default=None, help="Optional video id to include in metadata")
    ap.add_argument("--result-json", default=None, help="Write structured results to this JSON file")
    ap.add_argument("--diagnose-cuda", action="store_true", help="Print CUDA diagnostics")

    args = ap.parse_args()

    if args.diagnose_cuda:
        debug_cuda_environment(always=True)
        return

    condition_on_previous_text = _parse_bool(args.condition_on_previous_text, default=True)
    vad_filter = _parse_bool(args.vad_filter, default=True)
    out_dir = Path(args.output_dir)

    urls = args.url or []
    if args.urls:
        url_file = Path(args.urls)
        if url_file.exists():
            urls = [line.strip() for line in url_file.read_text().splitlines() if line.strip() and not line.startswith("#")]

    batch_entries: List[dict] = []
    if args.batch_json:
        batch_path = Path(args.batch_json)
        if not batch_path.exists():
            ap.error(f"--batch-json not found: {batch_path}")
        payload = json.loads(batch_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            ap.error("--batch-json must contain a JSON list")
        batch_entries = payload
    elif urls:
        batch_entries = [{"url": u, "video_id": None, "output_prefix": None} for u in urls]

    if not batch_entries and not args.audio:
        user_input = input("Paste YouTube URL (or local audio path): ").strip()
        if not user_input:
            ap.error("Provide --url/--urls/--batch-json/--audio (or run with --diagnose-cuda)")
        if re.match(r"^https?://", user_input) or "youtube.com" in user_input or "youtu.be" in user_input:
            batch_entries = [{"url": user_input, "video_id": args.video_id, "output_prefix": args.output_prefix}]
        else:
            args.audio = user_input

    results: List[dict] = []
    if batch_entries:
        results = _process_batch(
            entries=batch_entries,
            model=args.model,
            device=args.device,
            dtype=args.dtype,
            language=args.language,
            out_dir=out_dir,
            ffmpeg_location=args.ffmpeg_location,
            max_chars=args.max_chars,
            beam_size=args.beam_size,
            best_of=args.best_of,
            condition_on_previous_text=condition_on_previous_text,
            vad_filter=vad_filter,
            artifact_mode=args.artifact_mode,
        )
    elif args.audio:
        p = Path(args.audio)
        if not p.exists():
            print(f"[ERROR] Audio not found: {_safe_console_text(p)}")
            sys.exit(1)
        try:
            single = transcribe_file(
                audio=p,
                model=args.model,
                device=args.device,
                dtype=args.dtype,
                language=args.language,
                out_dir=out_dir,
                max_chars=args.max_chars,
                beam_size=args.beam_size,
                best_of=args.best_of,
                condition_on_previous_text=condition_on_previous_text,
                vad_filter=vad_filter,
                output_prefix=args.output_prefix,
                artifact_mode=args.artifact_mode,
            )
            results = [{"url": None, "video_id": args.video_id, "success": True, "files": single.get("files", {}), "transcript": single.get("transcript", ""), "transcript_chars": len(single.get("transcript", ""))}]
        except Exception as e:
            results = [{"url": None, "video_id": args.video_id, "success": False, "error": _safe_exception_text(e)}]

    if args.result_json:
        result_path = Path(args.result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    failed = [r for r in results if not r.get("success")]
    if failed:
        print(f"[WARN] {len(failed)} transcription task(s) failed.")
        if not args.result_json:
            for item in failed:
                print(
                    f"  - {_safe_console_text(item.get('url') or item.get('video_id'))}: "
                    f"{_safe_console_text(item.get('error', 'unknown error'))}"
                )
        sys.exit(1)


if __name__ == "__main__":
    _configure_utf8_stdio()
    _maybe_reexec_in_preferred_env()
    main()
