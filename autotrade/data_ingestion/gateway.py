"""
Data Ingestion Gateway
======================
Centralized gateway for data ingestion operations.

Provides a unified interface that wraps existing utilities while
implementing the DataIngestionGateway protocol.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from config.config_loader import get_config

from autotrade.data_ingestion.adapters import (
    DataQualityAdapter,
    DataSyncAdapter,
    LocalDataProviderAdapter,
)
from autotrade.data_ingestion.bootstrap import (
    ensure_core_market_data_ready,
    get_expected_latest_date,
)
from autotrade.data_ingestion.interfaces import SchemaNormalizer
from autotrade.data_ingestion.paths import (
    get_bootstrap_h5_candidates,
    get_primary_parquet_path,
)
from autotrade.data_ingestion.schemas import (
    DataFrameSnapshot,
    DataFreshnessLevel,
    DataFreshnessStatus,
    DataRequest,
    DataSourceType,
    IngestionHealthReport,
    classify_freshness,
)

logger = logging.getLogger(__name__)


def get_primary_data_path() -> Path:
    """Get the primary data file path."""
    return get_primary_parquet_path()


def get_bootstrap_data_path() -> Path:
    """Get the preferred bootstrap H5 data file path."""
    candidates = get_bootstrap_h5_candidates()
    return candidates[0]


def check_primary_data_exists() -> bool:
    """Check if primary data file exists."""
    return get_primary_parquet_path().exists()


def check_bootstrap_available() -> bool:
    """Check if any bootstrap (H5) source is available."""
    return any(path.exists() for path in get_bootstrap_h5_candidates())


def get_latest_data_date() -> Tuple[Optional[date], int]:
    """Get latest date and ticker count from the primary parquet."""
    parquet_path = get_primary_parquet_path()
    if not parquet_path.exists():
        return None, 0

    try:
        df = pd.read_parquet(parquet_path, columns=["ticker", "Date"])
        if df.empty:
            return None, 0
        latest = pd.to_datetime(df["Date"]).max()
        ticker_count = int(df["ticker"].nunique())
        if hasattr(latest, "date"):
            return latest.date(), ticker_count
        return latest, ticker_count
    except Exception as exc:
        logger.error(f"Failed to get latest date from {parquet_path}: {exc}")
        return None, 0


def compute_freshness(max_staleness_days: int = 1) -> DataFreshnessStatus:
    """Compute freshness status of primary data."""
    latest_date, ticker_count = get_latest_data_date()
    expected_date = get_expected_latest_date()

    if latest_date is None:
        return DataFreshnessStatus(
            level=DataFreshnessLevel.MISSING,
            latest_date=None,
            expected_date=expected_date,
            staleness_days=999,
            ticker_count=0,
            is_trading_day=date.today().weekday() < 5,
            source_path=str(get_primary_parquet_path()),
            checked_at=datetime.now(),
        )

    level, staleness = classify_freshness(
        latest_date,
        expected_date,
        max_staleness_days=max_staleness_days,
    )
    return DataFreshnessStatus(
        level=level,
        latest_date=latest_date,
        expected_date=expected_date,
        staleness_days=staleness,
        ticker_count=ticker_count,
        is_trading_day=date.today().weekday() < 5,
        source_path=str(get_primary_parquet_path()),
        checked_at=datetime.now(),
    )


def check_data_health(max_staleness_days: int = 1) -> IngestionHealthReport:
    """Perform comprehensive health check using the shared bootstrap gateway."""
    return ensure_core_market_data_ready(
        max_staleness_days=max_staleness_days,
        fail_fast=False,
    )


def ensure_data_ready(max_staleness_days: int = 1) -> bool:
    """Ensure all data sources are ready for operation."""
    report = ensure_core_market_data_ready(
        max_staleness_days=max_staleness_days,
        fail_fast=False,
    )
    return report.can_trade


class IngestionGateway:
    """Main gateway implementing ingestion operations and compatibility wrappers."""

    def __init__(
        self,
        max_staleness_days: Optional[int] = None,
        enable_bootstrap: Optional[bool] = None,
    ):
        data_cfg = get_config().data
        self.max_staleness_days = (
            data_cfg.max_staleness_days
            if max_staleness_days is None
            else max_staleness_days
        )
        self.enable_bootstrap = (
            data_cfg.bootstrap_from_h5_enabled
            if enable_bootstrap is None
            else enable_bootstrap
        )

        self._local_provider: Optional[LocalDataProviderAdapter] = None
        self._sync_adapter: Optional[DataSyncAdapter] = None
        self._quality_adapter: Optional[DataQualityAdapter] = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        try:
            self._local_provider = LocalDataProviderAdapter()
        except Exception as exc:
            logger.warning(f"Could not initialize LocalDataProviderAdapter: {exc}")

        try:
            self._sync_adapter = DataSyncAdapter()
        except Exception as exc:
            logger.debug(f"Could not initialize DataSyncAdapter: {exc}")

        try:
            self._quality_adapter = DataQualityAdapter()
        except Exception as exc:
            logger.debug(f"Could not initialize DataQualityAdapter: {exc}")

        self._initialized = True

    def get_data(self, request: DataRequest) -> DataFrameSnapshot:
        """Get data according to request specification."""
        self._ensure_initialized()

        # Enforce shared readiness policy before reads.
        ensure_core_market_data_ready(
            max_staleness_days=self.max_staleness_days,
            bootstrap_from_h5_enabled=self.enable_bootstrap,
            fail_fast=False,
        )

        source = DataSourceType.UNKNOWN
        df: Optional[pd.DataFrame] = None

        if self._local_provider is not None:
            df = self._local_provider.get_ticker_data(
                ticker=request.ticker_upper,
                days=request.days or 250,
                include_features=request.include_features,
            )
            source = DataSourceType.LOCAL_PARQUET
        else:
            df = self._load_from_parquet(request)
            source = DataSourceType.LOCAL_PARQUET

        if df is not None and not df.empty:
            df = SchemaNormalizer.standardize(df)

        freshness = self.get_freshness()
        snapshot = DataFrameSnapshot(
            df=df,
            ticker=request.ticker_upper,
            source=source,
            freshness=freshness.level,
        )
        if df is None or df.empty:
            snapshot.warnings.append("No data returned for ticker")
        return snapshot

    def _load_from_parquet(self, request: DataRequest) -> Optional[pd.DataFrame]:
        parquet_path = get_primary_parquet_path()
        if not parquet_path.exists():
            return None

        try:
            cols = ["ticker", "Date", "Open", "High", "Low", "Close", "Volume"]
            if request.include_features:
                cols.extend(["SMA_10", "SMA_20", "RSI_14", "MACD", "atr_14"])

            df = pd.read_parquet(parquet_path, columns=cols)
            df = df[df["ticker"] == request.ticker_upper]
            if df.empty:
                return None

            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")
            if request.days:
                df = df.tail(request.days)

            df.set_index("Date", inplace=True)
            df.drop(columns=["ticker"], inplace=True, errors="ignore")
            return df
        except Exception as exc:
            logger.error(f"Failed to load {request.ticker_upper} from parquet: {exc}")
            return None

    def get_freshness(self) -> DataFreshnessStatus:
        return compute_freshness(self.max_staleness_days)

    def check_health(self) -> IngestionHealthReport:
        return check_data_health(self.max_staleness_days)

    def ensure_ready(self) -> bool:
        return ensure_data_ready(self.max_staleness_days)

    # Compatibility helpers for legacy call sites.
    def ensure_synced(self) -> bool:
        self._ensure_initialized()
        if self._sync_adapter is None:
            return self.ensure_ready()
        return self._sync_adapter.ensure_synced()

    def validate_signal_data(self, as_of_date=None):
        self._ensure_initialized()
        if self._quality_adapter is None:
            health = self.check_health()
            return health.can_trade, health
        return self._quality_adapter.validate_data_for_signals(as_of_date=as_of_date)


_gateway: Optional[IngestionGateway] = None


def get_ingestion_gateway() -> IngestionGateway:
    """Get or create singleton IngestionGateway instance."""
    global _gateway
    if _gateway is None:
        _gateway = IngestionGateway()
    return _gateway
