"""
Data Ingestion Interfaces
==========================
Protocol definitions for data ingestion abstractions.

These protocols define the contracts that any data source adapter
must implement, enabling polymorphic data access throughout the system.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
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

from autotrade.data_ingestion.schemas import (
    DataRequest,
    DataFrameSnapshot,
    DataFreshnessStatus,
    DataFreshnessLevel,
    IngestionHealthReport,
    BulkDataRequest,
    BulkDataResult,
    classify_freshness,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class HistoricalDataSource(Protocol):
    """
    Protocol for historical data providers.

    Any class providing historical OHLCV data should implement this protocol.
    """

    def get_ticker_data(
        self, ticker: str, days: int = 250, include_features: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get historical data for a ticker.

        Args:
            ticker: Stock ticker symbol
            days: Number of trading days to fetch
            include_features: Include technical indicators

        Returns:
            DataFrame with OHLCV data (and features if requested), or None if not found
        """
        ...

    def has_ticker(self, ticker: str) -> bool:
        """Check if ticker exists in this data source."""
        ...

    def get_available_tickers(self) -> List[str]:
        """Get list of all available tickers."""
        ...


@runtime_checkable
class RealtimePriceSource(Protocol):
    """
    Protocol for real-time price providers.

    Any class providing live or near-live price data should implement this.
    """

    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        Get current/latest price for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Current price, or None if unavailable
        """
        ...

    def get_prices(self, tickers: List[str]) -> Dict[str, Optional[float]]:
        """
        Get current prices for multiple tickers.

        Args:
            tickers: List of stock ticker symbols

        Returns:
            Dict mapping ticker -> price (or None if unavailable)
        """
        ...


@runtime_checkable
class DataBootstrapper(Protocol):
    """
    Protocol for data bootstrap operations.

    Any class that can bootstrap from alternative data sources
    (e.g., H5 -> parquet conversion) should implement this.
    """

    def bootstrap_if_needed(self) -> bool:
        """
        Run bootstrap process if needed.

        Returns:
            True if bootstrap succeeded or was not needed
        """
        ...

    def is_bootstrap_available(self) -> bool:
        """Check if bootstrap is available as a fallback."""
        ...

    def get_bootstrap_status(self) -> Dict[str, Any]:
        """
        Get status of bootstrap capability.

        Returns:
            Dict with bootstrap availability and details
        """
        ...


@runtime_checkable
class DataIngestionGateway(Protocol):
    """
    Main gateway protocol for data ingestion.

    This is the primary interface that consumers should use.
    Combines historical, realtime, and bootstrap capabilities.
    """

    def get_data(self, request: DataRequest) -> DataFrameSnapshot:
        """
        Get data according to request specification.

        Args:
            request: Data request with parameters

        Returns:
            DataFrameSnapshot with data and metadata
        """
        ...

    def get_freshness(self) -> DataFreshnessStatus:
        """
        Get current freshness status of primary data source.

        Returns:
            DataFreshnessStatus with freshness details
        """
        ...

    def check_health(self) -> IngestionHealthReport:
        """
        Perform comprehensive health check.

        Returns:
            IngestionHealthReport with system health status
        """
        ...

    def ensure_ready(self) -> bool:
        """
        Ensure all data sources are ready for operation.

        Returns:
            True if ready, False otherwise
        """
        ...


class DataTransformer(Protocol):
    """
    Protocol for data transformation operations.

    Pure transforms that don't perform IO should implement this.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations to DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        ...

    @property
    def transform_name(self) -> str:
        """Name of this transformation."""
        ...


class SchemaNormalizer:
    """
    Utility class for normalizing DataFrame schemas.

    Provides common transformations to ensure consistent column
    names and types across different data sources.
    """

    STANDARD_COLUMNS = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Adj Close": "adj_close",
        "ticker": "symbol",
    }

    FEATURE_COLUMNS = [
        "SMA_10",
        "SMA_20",
        "SMA_50",
        "SMA_200",
        "EMA_10",
        "EMA_20",
        "EMA_50",
        "RSI_14",
        "RSI",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        "atr_14",
        "ATR",
    ]

    @classmethod
    def normalize_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase."""
        if df is None or df.empty:
            return df

        rename_map = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            for std_col, std_lower in cls.STANDARD_COLUMNS.items():
                if col == std_col or col_lower == std_lower:
                    rename_map[col] = std_lower
                    break

        if rename_map:
            df = df.rename(columns=rename_map)

        return df

    @classmethod
    def normalize_date_index(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure date column or index is properly typed."""
        if df is None or df.empty:
            return df

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        elif isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        return df

    @classmethod
    def ensure_numeric(
        cls, df: pd.DataFrame, columns: List[str] = None
    ) -> pd.DataFrame:
        """Ensure specified columns are numeric."""
        if df is None or df.empty:
            return df

        if columns is None:
            columns = ["open", "high", "low", "close", "volume", "adj_close"]

        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    @classmethod
    def standardize(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all standardizations."""
        df = cls.normalize_columns(df)
        df = cls.normalize_date_index(df)
        df = cls.ensure_numeric(df)
        return df


class DataValidator:
    """
    Utility class for validating data quality.

    Provides common validation checks for DataFrames.
    """

    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate OHLCV data integrity."""
        errors = []

        if df is None or df.empty:
            return False, ["DataFrame is empty"]

        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")

        if errors:
            return False, errors

        if (df["high"] < df["low"]).any():
            errors.append("High price is less than low price in some rows")

        if (df["close"] > df["high"]).any():
            errors.append("Close price exceeds high in some rows")

        if (df["close"] < df["low"]).any():
            errors.append("Close price is below low in some rows")

        if (df["open"] > df["high"]).any():
            errors.append("Open price exceeds high in some rows")

        if (df["open"] < df["low"]).any():
            errors.append("Open price is below low in some rows")

        if (df["volume"] < 0).any():
            errors.append("Negative volume found")

        return len(errors) == 0, errors

    @staticmethod
    def check_required_columns(
        df: pd.DataFrame, required: List[str]
    ) -> Tuple[bool, List[str]]:
        """Check if required columns are present."""
        missing = [col for col in required if col not in df.columns]
        return len(missing) == 0, missing

    @staticmethod
    def get_column_stats(df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about DataFrame columns."""
        stats = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": {col: int(df[col].isna().sum()) for col in df.columns},
        }

        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            stats["numeric_summary"] = df[numeric_cols].describe().to_dict()

        return stats
