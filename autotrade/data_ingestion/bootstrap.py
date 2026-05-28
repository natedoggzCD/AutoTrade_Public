"""
Centralized core-data readiness checks and optional H5 bootstrap.
"""

from __future__ import annotations

import logging
import threading
from datetime import date, datetime, time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from config.config_loader import get_config

from autotrade.data_ingestion.errors import (
    CoreDataMissingError,
    DataBootstrapError,
    DataFreshnessError,
    DataPathError,
)
from autotrade.data_ingestion.paths import (
    get_bootstrap_h5_candidates,
    get_primary_parquet_path,
)
from autotrade.data_ingestion.schemas import (
    DataFreshnessLevel,
    DataFreshnessStatus,
    IngestionHealthReport,
    classify_freshness,
)
from autotrade.utils.market_time import (
    is_trading_day,
    expected_core_market_data_date,
)

logger = logging.getLogger(__name__)

_BOOTSTRAP_LOCK = threading.Lock()


def _normalize_as_of_datetime(as_of: Optional[date | datetime] = None) -> datetime:
    """Treat date-only inputs as pre-close session checks."""
    if isinstance(as_of, datetime):
        return as_of
    if isinstance(as_of, date):
        return datetime.combine(as_of, time.min)
    return datetime.now()


def get_expected_latest_date(as_of: Optional[date | datetime] = None) -> Optional[date]:
    """Calculate what the latest data date SHOULD be, skipping holidays."""
    as_of_dt = _normalize_as_of_datetime(as_of)
    expected_iso = expected_core_market_data_date(reference_dt=as_of_dt)
    return date.fromisoformat(expected_iso)


def _normalize_bootstrap_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise DataBootstrapError("H5 payload is not a DataFrame")
    out = df.copy()
    if out.empty:
        return out

    if isinstance(out.index, pd.MultiIndex) or out.index.name is not None:
        out = out.reset_index()

    canonical = {
        "date": "Date",
        "datetime": "Date",
        "timestamp": "Date",
        "ticker": "ticker",
        "symbol": "ticker",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    rename_map = {}
    existing = set(out.columns)
    for col in out.columns:
        mapped = canonical.get(str(col).strip().lower())
        if mapped and mapped not in existing:
            rename_map[col] = mapped
    if rename_map:
        out = out.rename(columns=rename_map)

    required = ["Date", "ticker", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise DataBootstrapError(f"H5 data missing required columns: {missing}")

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"])
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out = out[out["ticker"] != ""]
    return out


def _select_h5_key(keys: List[str]) -> str:
    if not keys:
        raise DataBootstrapError("No datasets found in H5 file")
    sorted_keys = sorted(keys)
    preferred = [k for k in sorted_keys if ("daily" in k.lower() or "feature" in k.lower())]
    return preferred[0] if preferred else sorted_keys[0]


def _convert_h5_to_parquet(h5_path: Path, parquet_path: Path) -> int:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = parquet_path.with_suffix(".tmp.parquet")
    if tmp_path.exists():
        tmp_path.unlink()

    try:
        with pd.HDFStore(str(h5_path), mode="r") as store:
            key = _select_h5_key(list(store.keys()))
            try:
                frame = store.select(key)
            except Exception:
                frame = store[key]
    except ImportError as exc:
        raise DataBootstrapError(
            "PyTables dependency missing for H5 bootstrap (`pip install tables`).",
            source_path=h5_path,
        ) from exc

    frame = _normalize_bootstrap_df(frame)
    if frame.empty:
        raise DataBootstrapError("H5 bootstrap produced empty DataFrame", source_path=h5_path)

    frame.sort_values(["Date", "ticker"], inplace=True)
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(parquet_path)
    return len(frame)


def _read_latest_date_and_count(parquet_path: Path) -> Tuple[Optional[date], int]:
    if not parquet_path.exists():
        return None, 0

    try:
        df = pd.read_parquet(parquet_path, columns=["ticker", "Date"])
    except Exception as exc:
        raise DataPathError(f"Failed to read parquet at {parquet_path}: {exc}", path=parquet_path) from exc

    if df.empty:
        return None, 0

    latest = pd.to_datetime(df["Date"]).max()
    ticker_count = int(df["ticker"].nunique()) if "ticker" in df.columns else 0
    if hasattr(latest, "date"):
        return latest.date(), ticker_count
    return latest, ticker_count


def _build_freshness_status(
    parquet_path: Path,
    max_staleness_days: int,
) -> DataFreshnessStatus:
    latest_date, ticker_count = _read_latest_date_and_count(parquet_path)
    expected_date = get_expected_latest_date()
    level, staleness = classify_freshness(latest_date, expected_date, max_staleness_days)
    return DataFreshnessStatus(
        level=level,
        latest_date=latest_date,
        expected_date=expected_date,
        staleness_days=staleness,
        ticker_count=ticker_count,
        is_trading_day=is_trading_day(),
        source_path=str(parquet_path),
        checked_at=datetime.now(),
    )


def ensure_core_market_data_ready(
    max_staleness_days: Optional[int] = None,
    bootstrap_from_h5_enabled: Optional[bool] = None,
    fail_fast: Optional[bool] = None,
    parquet_path: Optional[Path] = None,
) -> IngestionHealthReport:
    """
    Single readiness gateway:
    1) ensure parquet exists (optionally bootstrap from H5)
    2) ensure parquet is readable
    3) enforce freshness policy
    """
    cfg = get_config().data
    max_staleness_days = cfg.max_staleness_days if max_staleness_days is None else max_staleness_days
    bootstrap_from_h5_enabled = (
        cfg.bootstrap_from_h5_enabled
        if bootstrap_from_h5_enabled is None
        else bootstrap_from_h5_enabled
    )
    fail_fast = cfg.fail_fast_on_missing_core_data if fail_fast is None else fail_fast

    parquet = parquet_path or get_primary_parquet_path()
    h5_candidates = get_bootstrap_h5_candidates()

    report = IngestionHealthReport(
        is_healthy=False,
        can_trade=False,
        primary_source_ready=False,
        bootstrap_available=any(path.exists() for path in h5_candidates),
    )

    with _BOOTSTRAP_LOCK:
        if not parquet.exists() and bootstrap_from_h5_enabled:
            for h5_path in h5_candidates:
                if not h5_path.exists():
                    continue
                try:
                    rows = _convert_h5_to_parquet(h5_path, parquet)
                    report.recommendations.append(
                        f"Initialized parquet from H5 source ({h5_path.name}, rows={rows})"
                    )
                    break
                except Exception as exc:
                    report.add_warning(f"H5 bootstrap failed ({h5_path}): {exc}")

        report.primary_source_ready = parquet.exists()
        if not report.primary_source_ready:
            message = f"Primary core data missing: {parquet}"
            report.add_error(message)
            report.sources = {
                "primary_parquet": {"exists": False, "path": str(parquet)},
                "bootstrap_h5_candidates": [str(path) for path in h5_candidates],
            }
            if fail_fast:
                raise CoreDataMissingError(message, data_path=parquet)
            return report

        try:
            freshness = _build_freshness_status(parquet, max_staleness_days=max_staleness_days)
        except Exception as exc:
            message = f"Primary data unreadable: {exc}"
            report.add_error(message)
            if fail_fast:
                raise DataPathError(message, path=parquet) from exc
            return report

        report.freshness = freshness
        report.is_healthy = True
        report.sources = {
            "primary_parquet": {
                "exists": True,
                "path": str(parquet),
                "latest_date": freshness.latest_date.isoformat() if freshness.latest_date else None,
                "ticker_count": freshness.ticker_count,
            },
            "bootstrap_h5_candidates": [str(path) for path in h5_candidates],
        }

        if freshness.level == DataFreshnessLevel.MISSING:
            report.add_error("Primary parquet is empty or missing required date rows")
        elif freshness.level == DataFreshnessLevel.OLD:
            report.add_error(f"Core data is too old ({freshness.staleness_days} trading days)")
        elif freshness.level == DataFreshnessLevel.STALE:
            report.add_warning(f"Core data is stale ({freshness.staleness_days} trading days)")

        report.can_trade = report.is_healthy and freshness.is_usable
        if report.errors:
            report.recommendations.append("Restore or regenerate core market data before trading.")
        else:
            report.recommendations.append("Core market data readiness check passed.")

        if fail_fast and freshness.level in (DataFreshnessLevel.MISSING, DataFreshnessLevel.OLD):
            raise DataFreshnessError(
                f"Core data freshness violation: {freshness.level.value}",
                staleness_days=freshness.staleness_days,
            )

        return report
