"""
Typed errors for the data ingestion module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class DataIngestionError(RuntimeError):
    """Base class for all ingestion-layer errors."""


class DataPathError(DataIngestionError):
    """Raised when required data paths are invalid or unreadable."""

    def __init__(self, message: str, path: Optional[Path] = None):
        super().__init__(message)
        self.path = path


class DataBootstrapError(DataIngestionError):
    """Raised when bootstrap from fallback sources fails."""

    def __init__(self, message: str, source_path: Optional[Path] = None):
        super().__init__(message)
        self.source_path = source_path


class CoreDataMissingError(DataIngestionError):
    """Raised when core market data is unavailable after bootstrap attempts."""

    def __init__(self, message: str, data_path: Optional[Path] = None):
        super().__init__(message)
        self.data_path = data_path


class DataFreshnessError(DataIngestionError):
    """Raised when data freshness violates configured policy."""

    def __init__(self, message: str, staleness_days: Optional[int] = None):
        super().__init__(message)
        self.staleness_days = staleness_days

