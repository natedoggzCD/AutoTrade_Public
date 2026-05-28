#!/usr/bin/env python3
"""
Download or sync AutoTrade backtesting data files with Hugging Face.

Usage:
    python setup_backtest_data.py
    python setup_backtest_data.py --sync

Source dataset:
    https://huggingface.co/datasets/natedoggztn/autotrade-backtest-data
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HF_REPO_ID = "natedoggztn/autotrade-backtest-data"
HF_REPO_TYPE = "dataset"
DEFAULT_DATA_DIR = Path("data/downday")
DEFAULT_TRADINGMETHODS_DIR = Path.home() / "Desktop" / "TradingMethods"
DEFAULT_DOWNDAY_DIR = Path.home() / "Desktop" / "DownDay"

DATA_FILES = [
    {
        "hf_filename": "daily_features.h5",
        "size_hint": "~4.0 GB",
        "required": True,
        "description": "Primary feature store used by backtesting engine",
        "source_candidates": [
            DEFAULT_TRADINGMETHODS_DIR / "daily_features.h5",
            DEFAULT_DOWNDAY_DIR / "daily_features.h5",
        ],
    },
    {
        "hf_filename": "daily_features.parquet",
        "size_hint": "~2.4 GB",
        "required": True,
        "description": "Parquet mirror of feature store (faster pandas I/O)",
        "source_candidates": [
            DEFAULT_DOWNDAY_DIR / "data" / "daily_features.parquet",
            DEFAULT_TRADINGMETHODS_DIR / "daily_features.parquet",
        ],
    },
    {
        "hf_filename": "prices_hourly.parquet",
        "size_hint": "~312 MB",
        "required": True,
        "description": "Compressed hourly OHLCV bars used for intraday signal features",
        "source_candidates": [
            DEFAULT_DOWNDAY_DIR / "data" / "prices_hourly.parquet",
            DEFAULT_TRADINGMETHODS_DIR / "prices_hourly.parquet",
        ],
    },
    {
        "hf_filename": "prices_daily.csv",
        "size_hint": "~950 MB",
        "required": True,
        "description": "Daily OHLCV CSV from TradingMethods",
        "source_candidates": [
            DEFAULT_TRADINGMETHODS_DIR / "prices_daily.csv",
            DEFAULT_DOWNDAY_DIR / "prices_daily.csv",
        ],
    },
    {
        "hf_filename": "prices_hourly.csv",
        "size_hint": "~2.3 GB",
        "required": True,
        "description": "Hourly OHLCV CSV from TradingMethods",
        "source_candidates": [
            DEFAULT_TRADINGMETHODS_DIR / "prices_hourly.csv",
            DEFAULT_DOWNDAY_DIR / "prices_hourly.csv",
        ],
    },
    {
        "hf_filename": "nasdaq_screener.csv",
        "size_hint": "~1 MB",
        "required": True,
        "description": "NASDAQ screener metadata from TradingMethods",
        "source_candidates": [
            DEFAULT_TRADINGMETHODS_DIR / "nasdaq_screener.csv",
            DEFAULT_DOWNDAY_DIR / "nasdaq_screener.csv",
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download AutoTrade backtest data from Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        default=HF_REPO_ID,
        help=f"Hugging Face dataset repo ID (default: {HF_REPO_ID})",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help=(
            "Upload local source files to Hugging Face first, then download/verify "
            "the repo-local cache."
        ),
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload local source files to Hugging Face without downloading.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Local destination directory (default: {DEFAULT_DATA_DIR.as_posix()})",
    )
    parser.add_argument(
        "--tradingmethods-dir",
        default=str(DEFAULT_TRADINGMETHODS_DIR),
        help=(
            "TradingMethods source directory for upload/sync "
            f"(default: {DEFAULT_TRADINGMETHODS_DIR})"
        ),
    )
    parser.add_argument(
        "--downday-dir",
        default=str(DEFAULT_DOWNDAY_DIR),
        help=f"DownDay source directory for upload/sync (default: {DEFAULT_DOWNDAY_DIR})",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        choices=[entry["hf_filename"] for entry in DATA_FILES],
        help="Optional subset of dataset filenames to upload/download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even when they already exist locally.",
    )
    return parser.parse_args()


def _selected_entries(args: argparse.Namespace) -> list[dict]:
    selected = set(args.files or [])
    if not selected:
        return DATA_FILES
    return [entry for entry in DATA_FILES if entry["hf_filename"] in selected]


def _resolve_source_candidates(args: argparse.Namespace, entry: dict) -> list[Path]:
    tradingmethods_dir = Path(args.tradingmethods_dir)
    downday_dir = Path(args.downday_dir)
    filename = entry["hf_filename"]

    if filename == "daily_features.parquet":
        return [downday_dir / "data" / filename, tradingmethods_dir / filename]
    if filename == "prices_hourly.parquet":
        return [downday_dir / "data" / filename, tradingmethods_dir / filename]
    return [tradingmethods_dir / filename, downday_dir / filename]


def _find_source_path(args: argparse.Namespace, entry: dict) -> Path | None:
    for candidate in _resolve_source_candidates(args, entry):
        if candidate.exists():
            return candidate
    return None


def _load_huggingface_tools():
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        print("  Run: pip install huggingface_hub")
        sys.exit(1)
    return HfApi, hf_hub_download


def _upload_files(args: argparse.Namespace, token: str | None) -> None:
    HfApi, _ = _load_huggingface_tools()
    api = HfApi(token=token)
    errors = []

    print(f"Uploading backtesting data to HF repo: {args.repo_id}")
    print(f"TradingMethods source: {Path(args.tradingmethods_dir)}")
    print(f"DownDay source: {Path(args.downday_dir)}")
    print()

    for entry in _selected_entries(args):
        source = _find_source_path(args, entry)
        if source is None:
            msg = f"missing local source for {entry['hf_filename']}"
            print(f"  [{'ERROR' if entry['required'] else 'warn'}] {msg}")
            if entry["required"]:
                errors.append(msg)
            continue

        size_mb = round(source.stat().st_size / 1_048_576, 1)
        print(f"  [upload] {source} -> {entry['hf_filename']} ({size_mb} MB)")
        try:
            api.upload_file(
                path_or_fileobj=str(source),
                path_in_repo=entry["hf_filename"],
                repo_id=args.repo_id,
                repo_type=HF_REPO_TYPE,
                commit_message=f"Sync {entry['hf_filename']}",
            )
            print(f"  [ok]     {entry['hf_filename']}")
        except Exception as exc:
            msg = f"{entry['hf_filename']}: {exc}"
            print(f"  [ERROR]  {msg}")
            errors.append(msg)

    if errors:
        print()
        print("FAILED: upload errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)


def _download_files(args: argparse.Namespace, token: str | None) -> None:
    _, hf_hub_download = _load_huggingface_tools()
    repo_root = Path(__file__).parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = repo_root / data_dir

    print(f"Downloading backtesting data from HF repo: {args.repo_id}")
    print(f"Destination: {data_dir}")
    print()

    errors = []
    for entry in _selected_entries(args):
        local = data_dir / entry["hf_filename"]
        if local.exists() and not args.force:
            size_mb = round(local.stat().st_size / 1_048_576, 1)
            print(f"  [skip] {local.name} already present ({size_mb} MB)")
            continue

        data_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"  [download] {entry['hf_filename']}  {entry['size_hint']}  - "
            f"{entry['description']}"
        )

        try:
            hf_hub_download(
                repo_id=args.repo_id,
                filename=entry["hf_filename"],
                repo_type=HF_REPO_TYPE,
                local_dir=str(data_dir),
                force_download=args.force,
                token=token,
            )
            size_mb = round(local.stat().st_size / 1_048_576, 1) if local.exists() else 0
            print(f"  [ok]   {local.name}  ({size_mb} MB)")
        except Exception as exc:
            msg = f"  [{'ERROR' if entry['required'] else 'warn'}] {entry['hf_filename']}: {exc}"
            print(msg)
            if entry["required"]:
                errors.append(entry["hf_filename"])

    print()
    if errors:
        print(f"FAILED: required files not downloaded: {errors}")
        print("Backtesting will not work without these files.")
        sys.exit(1)

    print(f"Data setup complete. Files are in {data_dir}")


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")

    if args.upload or args.sync:
        _upload_files(args, token)

    if not args.upload or args.sync:
        _download_files(args, token)


if __name__ == "__main__":
    main()
