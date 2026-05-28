"""
Data Ingestion Module
=====================
Centralized data ingestion layer with typed interfaces, path resolution,
and bootstrap readiness checks.
"""

from autotrade.data_ingestion.adapters import (
    DataQualityAdapter,
    DataSyncAdapter,
    LocalDataProviderAdapter,
)
from autotrade.data_ingestion.bootstrap import (
    ensure_core_market_data_ready,
    get_expected_latest_date,
)
from autotrade.data_ingestion.errors import (
    CoreDataMissingError,
    DataBootstrapError,
    DataFreshnessError,
    DataIngestionError,
    DataPathError,
)
from autotrade.data_ingestion.gateway import (
    IngestionGateway,
    check_data_health,
    ensure_data_ready,
    get_ingestion_gateway,
)
from autotrade.data_ingestion.interfaces import (
    DataBootstrapper,
    DataIngestionGateway,
    DataTransformer,
    DataValidator,
    HistoricalDataSource,
    RealtimePriceSource,
    SchemaNormalizer,
)
from autotrade.data_ingestion.paths import (
    describe_ingestion_paths,
    get_bootstrap_h5_candidates,
    get_ingestion_paths,
    get_primary_parquet_path,
)
from autotrade.data_ingestion.schemas import (
    BulkDataRequest,
    BulkDataResult,
    DataFrameSnapshot,
    DataFreshnessLevel,
    DataFreshnessStatus,
    DataRequest,
    DataSourceType,
    IngestionHealthReport,
    calculate_data_age_minutes,
    classify_freshness,
)
from autotrade.data_ingestion.stream_bridge import (
    AlpacaStreamBridge,
    QuoteStreamEvent,
    TradeStreamEvent,
)
from autotrade.data_ingestion.fast_cache import (
    FastMarketDataCache,
    FastSymbolSnapshot,
)


__all__ = [
    "BulkDataRequest",
    "BulkDataResult",
    "CoreDataMissingError",
    "DataBootstrapError",
    "DataBootstrapper",
    "DataFreshnessError",
    "DataFreshnessLevel",
    "DataFreshnessStatus",
    "DataFrameSnapshot",
    "DataIngestionError",
    "DataIngestionGateway",
    "DataPathError",
    "DataQualityAdapter",
    "DataRequest",
    "DataSourceType",
    "DataSyncAdapter",
    "DataTransformer",
    "DataValidator",
    "HistoricalDataSource",
    "IngestionGateway",
    "IngestionHealthReport",
    "LocalDataProviderAdapter",
    "RealtimePriceSource",
    "SchemaNormalizer",
    "AlpacaStreamBridge",
    "QuoteStreamEvent",
    "TradeStreamEvent",
    "FastMarketDataCache",
    "FastSymbolSnapshot",
    "calculate_data_age_minutes",
    "check_data_health",
    "classify_freshness",
    "describe_ingestion_paths",
    "ensure_core_market_data_ready",
    "ensure_data_ready",
    "get_bootstrap_h5_candidates",
    "get_expected_latest_date",
    "get_ingestion_gateway",
    "get_ingestion_paths",
    "get_primary_parquet_path",
]

__version__ = "0.13.0"
