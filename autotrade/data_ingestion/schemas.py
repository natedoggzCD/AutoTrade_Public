"""
Data Ingestion Schemas
======================
Typed dataclasses for data ingestion module.

These schemas define the contract between data providers and consumers,
ensuring type safety and consistent data handling across the system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataFreshnessLevel(Enum):
    """Classification of data freshness."""

    FRESH = "fresh"  # Data from today (trading day)
    STALE = "stale"  # Data from yesterday
    OLD = "old"  # Data older than 1 day
    UNKNOWN = "unknown"  # Cannot determine freshness
    MISSING = "missing"  # No data available


class DataSourceType(Enum):
    """Type of data source."""

    LOCAL_PARQUET = "local_parquet"
    LOCAL_H5 = "local_h5"
    LOCAL_CSV = "local_csv"
    LIVE_API = "live_api"
    DUCKDB = "duckdb"
    UNKNOWN = "unknown"


@dataclass
class DataRequest:
    """
    Request for historical or real-time data.

    This is the primary input schema for data ingestion operations.
    """

    ticker: str
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    days: Optional[int] = None

    include_features: bool = True
    include_intraday: bool = False

    force_live: bool = False

    max_age_minutes: int = 15

    def __post_init__(self):
        if self.days is None and self.start_date is None:
            self.days = 250

        if self.start_date is not None and isinstance(self.start_date, str):
            self.start_date = pd.to_datetime(self.start_date).date()

        if self.end_date is not None and isinstance(self.end_date, str):
            self.end_date = pd.to_datetime(self.end_date).date()

    @property
    def ticker_upper(self) -> str:
        return self.ticker.upper()


@dataclass
class DataFrameSnapshot:
    """
    Wrapper for data returned from ingestion operations.

    Provides metadata alongside the actual DataFrame for proper
    consumption by downstream modules.
    """

    df: pd.DataFrame

    ticker: str

    source: DataSourceType

    fetched_at: datetime = field(default_factory=datetime.now)

    start_date: Optional[date] = None
    end_date: Optional[date] = None

    row_count: int = 0
    column_count: int = 0

    freshness: DataFreshnessLevel = DataFreshnessLevel.UNKNOWN

    data_age_minutes: Optional[int] = None

    warnings: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.df is not None and not self.df.empty:
            self.row_count = len(self.df)
            self.column_count = len(self.df.columns)

            if self.start_date is None and "Date" in self.df.columns:
                self.start_date = pd.to_datetime(self.df["Date"]).min().date()
            elif self.start_date is None and isinstance(
                self.df.index, pd.DatetimeIndex
            ):
                self.start_date = self.df.index.min().date()

            if self.end_date is None and "Date" in self.df.columns:
                self.end_date = pd.to_datetime(self.df["Date"]).max().date()
            elif self.end_date is None and isinstance(self.df.index, pd.DatetimeIndex):
                self.end_date = self.df.index.max().date()

    @property
    def is_empty(self) -> bool:
        return self.df is None or self.df.empty

    @property
    def is_fresh(self) -> bool:
        return self.freshness == DataFreshnessLevel.FRESH

    @property
    def date_range(self) -> Tuple[Optional[date], Optional[date]]:
        return (self.start_date, self.end_date)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "source": self.source.value,
            "fetched_at": self.fetched_at.isoformat(),
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "freshness": self.freshness.value,
            "data_age_minutes": self.data_age_minutes,
            "warnings": self.warnings,
            "columns": list(self.df.columns) if self.df is not None else [],
        }


@dataclass
class DataFreshnessStatus:
    """
    Freshness status for a data source.

    Provides detailed information about how current the data is
    and whether it's suitable for various operations.
    """

    level: DataFreshnessLevel

    latest_date: Optional[date] = None
    expected_date: Optional[date] = None

    staleness_days: int = 0

    ticker_count: int = 0

    is_trading_day: bool = False

    source_path: Optional[str] = None

    checked_at: datetime = field(default_factory=datetime.now)

    message: str = ""

    def __post_init__(self):
        if self.message and not self.message:
            self.message = self._generate_message()

    def _generate_message(self) -> str:
        if self.level == DataFreshnessLevel.FRESH:
            return f"Data is fresh ({self.latest_date})"
        elif self.level == DataFreshnessLevel.STALE:
            return f"Data is stale: {self.latest_date} (expected {self.expected_date}, {self.staleness_days} day(s) old)"
        elif self.level == DataFreshnessLevel.OLD:
            return f"Data is old: {self.latest_date} ({self.staleness_days} days old)"
        elif self.level == DataFreshnessLevel.MISSING:
            return "No data available"
        else:
            return "Data freshness unknown"

    @property
    def is_usable(self) -> bool:
        return self.level in (DataFreshnessLevel.FRESH, DataFreshnessLevel.STALE)

    @property
    def requires_warning(self) -> bool:
        return self.level != DataFreshnessLevel.FRESH


@dataclass
class IngestionHealthReport:
    """
    Comprehensive health report for the data ingestion system.

    This is the main output from health checks and should be used
    to determine whether the system can proceed with trading operations.
    """

    is_healthy: bool

    can_trade: bool = False

    primary_source_ready: bool = False

    bootstrap_available: bool = False

    freshness: Optional[DataFreshnessStatus] = None

    sources: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    checked_at: datetime = field(default_factory=datetime.now)

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    recommendations: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def should_proceed(self) -> bool:
        return self.is_healthy and self.primary_source_ready

    @property
    def should_abort(self) -> bool:
        return not self.should_proceed

    def add_error(self, error: str):
        self.errors.append(error)
        self.is_healthy = False

    def add_warning(self, warning: str):
        self.warnings.append(warning)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_healthy": self.is_healthy,
            "can_trade": self.can_trade,
            "primary_source_ready": self.primary_source_ready,
            "bootstrap_available": self.bootstrap_available,
            "checked_at": self.checked_at.isoformat(),
            "errors": self.errors,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "sources": self.sources,
            "freshness": {
                "level": self.freshness.level.value if self.freshness else None,
                "latest_date": self.freshness.latest_date.isoformat()
                if self.freshness and self.freshness.latest_date
                else None,
                "staleness_days": self.freshness.staleness_days
                if self.freshness
                else None,
            }
            if self.freshness
            else None,
        }


@dataclass
class BulkDataRequest:
    """
    Request for bulk data operations.

    Used when requesting data for multiple tickers at once.
    """

    tickers: List[str]

    days: int = 250

    include_features: bool = True

    force_live: bool = False

    batch_size: int = 50

    parallel: bool = True

    def __post_init__(self):
        if isinstance(self.tickers, str):
            self.tickers = [t.strip() for t in self.tickers.split(",")]


@dataclass
class BulkDataResult:
    """
    Result from bulk data operations.
    """

    results: Dict[str, DataFrameSnapshot]

    successful: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)

    total_tickers: int = 0
    success_count: int = 0
    failure_count: int = 0

    elapsed_seconds: float = 0.0

    errors: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.total_tickers = len(self.results) + len(self.errors)
        self.success_count = len(self.successful)
        self.failure_count = len(self.failed)

    @property
    def success_rate(self) -> float:
        if self.total_tickers == 0:
            return 0.0
        return self.success_count / self.total_tickers

    @property
    def is_fully_successful(self) -> bool:
        return self.failure_count == 0

    @property
    def is_partially_successful(self) -> bool:
        return self.success_count > 0 and self.failure_count > 0


def classify_freshness(
    latest_date: Optional[date], expected_date: date, max_staleness_days: int = 1
) -> Tuple[DataFreshnessLevel, int]:
    """
    Classify data freshness based on dates.

    Args:
        latest_date: The most recent date in the data
        expected_date: The expected latest date (usually today or yesterday)
        max_staleness_days: Maximum days allowed before marking as stale

    Returns:
        Tuple of (freshness_level, staleness_days)
    """
    if latest_date is None:
        return DataFreshnessLevel.MISSING, 999

    if latest_date == expected_date:
        return DataFreshnessLevel.FRESH, 0

    staleness = 0
    check_date = latest_date
    while check_date < expected_date:
        check_date += timedelta(days=1)
        if check_date.weekday() < 5:
            staleness += 1

    if staleness == 0:
        return DataFreshnessLevel.FRESH, 0
    elif staleness <= max_staleness_days:
        return DataFreshnessLevel.STALE, staleness
    else:
        return DataFreshnessLevel.OLD, staleness


def calculate_data_age_minutes(file_mtime: float) -> int:
    """Calculate how old a file is in minutes based on modification time."""
    file_time = datetime.fromtimestamp(file_mtime)
    age = datetime.now() - file_time
    return int(age.total_seconds() / 60)
