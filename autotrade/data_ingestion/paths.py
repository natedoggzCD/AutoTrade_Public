"""
Shared path resolution for data ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from config.config_loader import get_config


@dataclass(frozen=True)
class IngestionPaths:
    downday_root: Path
    nasdaq_screener_csv: Path
    daily_features_parquet: Path
    hourly_prices_parquet: Path
    daily_features_h5: Path
    market_data_duckdb: Path
    prices_daily_csv: Path
    prices_hourly_csv: Path


def _resolve_against_root(root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def get_ingestion_paths() -> IngestionPaths:
    """Resolve all ingestion paths from typed config in one place."""
    cfg = get_config().data
    root = Path(cfg.downday_root)

    return IngestionPaths(
        downday_root=root,
        nasdaq_screener_csv=_resolve_against_root(
            root,
            getattr(
                cfg,
                "nasdaq_screener_csv",
                "nasdaq_screener.csv",
            ),
        ),
        daily_features_parquet=_resolve_against_root(root, cfg.daily_features_parquet),
        hourly_prices_parquet=_resolve_against_root(root, cfg.hourly_prices_parquet),
        daily_features_h5=_resolve_against_root(root, cfg.daily_features_h5),
        market_data_duckdb=_resolve_against_root(root, cfg.market_data_duckdb),
        prices_daily_csv=root / "prices_daily.csv",
        prices_hourly_csv=root / "prices_hourly.csv",
    )


def get_primary_parquet_path() -> Path:
    return get_ingestion_paths().daily_features_parquet


def get_bootstrap_h5_candidates() -> List[Path]:
    paths = get_ingestion_paths()
    cfg_h5 = paths.daily_features_h5
    candidates = [
        cfg_h5,
        paths.downday_root / "daily_features.h5",
        Path("daily_features.h5"),
    ]

    deduped: List[Path] = []
    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def describe_ingestion_paths() -> Dict[str, str]:
    paths = get_ingestion_paths()
    return {
        "downday_root": str(paths.downday_root),
        "nasdaq_screener_csv": str(paths.nasdaq_screener_csv),
        "daily_features_parquet": str(paths.daily_features_parquet),
        "hourly_prices_parquet": str(paths.hourly_prices_parquet),
        "daily_features_h5": str(paths.daily_features_h5),
        "market_data_duckdb": str(paths.market_data_duckdb),
        "prices_daily_csv": str(paths.prices_daily_csv),
        "prices_hourly_csv": str(paths.prices_hourly_csv),
    }
