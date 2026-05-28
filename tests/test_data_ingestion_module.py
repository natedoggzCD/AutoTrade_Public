"""
Tests for Data Ingestion Module
================================
Unit tests for Phase 1 - Contract + Interface Layer.
"""

import pytest
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import numpy as np

from autotrade.data_ingestion.schemas import (
    DataRequest,
    DataFrameSnapshot,
    DataFreshnessStatus,
    DataFreshnessLevel,
    IngestionHealthReport,
    BulkDataRequest,
    BulkDataResult,
    DataSourceType,
    classify_freshness,
    calculate_data_age_minutes,
)
from autotrade.data_ingestion.errors import CoreDataMissingError
from autotrade.data_ingestion.interfaces import SchemaNormalizer
import autotrade.data_ingestion.bootstrap as ingestion_bootstrap
import autotrade.data_ingestion.paths as ingestion_paths


class TestDataRequest:
    """Tests for DataRequest dataclass."""

    def test_basic_request(self):
        """Test creating a basic data request."""
        req = DataRequest(ticker="AAPL")
        assert req.ticker == "AAPL"
        assert req.days == 250
        assert req.include_features is True

    def test_ticker_uppercase(self):
        """Test ticker is normalized to uppercase."""
        req = DataRequest(ticker="aapl")
        assert req.ticker_upper == "AAPL"

    def test_with_date_range(self):
        """Test request with date range."""
        start = date(2025, 1, 1)
        end = date(2025, 12, 31)
        req = DataRequest(ticker="AAPL", start_date=start, end_date=end)
        assert req.start_date == start
        assert req.end_date == end

    def test_with_string_dates(self):
        """Test request with string dates."""
        req = DataRequest(ticker="AAPL", start_date="2025-01-01", end_date="2025-12-31")
        assert req.start_date == date(2025, 1, 1)
        assert req.end_date == date(2025, 12, 31)

    def test_with_days(self):
        """Test request specifying days."""
        req = DataRequest(ticker="AAPL", days=100)
        assert req.days == 100


class TestDataFrameSnapshot:
    """Tests for DataFrameSnapshot dataclass."""

    def test_empty_snapshot(self):
        """Test creating empty snapshot."""
        snap = DataFrameSnapshot(
            df=None,
            ticker="AAPL",
            source=DataSourceType.LOCAL_PARQUET,
        )
        assert snap.is_empty
        assert snap.ticker == "AAPL"

    def test_snapshot_with_data(self):
        """Test snapshot with actual DataFrame."""
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [105.0, 106.0],
                "low": [99.0, 100.0],
                "close": [102.0, 103.0],
                "volume": [1000000, 1100000],
            },
            index=pd.date_range("2025-01-01", periods=2),
        )

        snap = DataFrameSnapshot(
            df=df,
            ticker="AAPL",
            source=DataSourceType.LOCAL_PARQUET,
        )

        assert not snap.is_empty
        assert snap.row_count == 2
        assert snap.column_count == 5
        assert snap.start_date == date(2025, 1, 1)
        assert snap.end_date == date(2025, 1, 2)

    def test_freshness_property(self):
        """Test freshness property."""
        snap = DataFrameSnapshot(
            df=pd.DataFrame(),
            ticker="AAPL",
            source=DataSourceType.LOCAL_PARQUET,
            freshness=DataFreshnessLevel.FRESH,
        )
        assert snap.is_fresh


class TestDataFreshnessStatus:
    """Tests for DataFreshnessStatus dataclass."""

    def test_fresh_status(self):
        """Test fresh data status."""
        today = date.today()
        status = DataFreshnessStatus(
            level=DataFreshnessLevel.FRESH,
            latest_date=today,
            expected_date=today,
            staleness_days=0,
        )
        assert status.is_usable
        assert not status.requires_warning

    def test_stale_status(self):
        """Test stale data status."""
        yesterday = date.today() - timedelta(days=1)
        today = date.today()
        status = DataFreshnessStatus(
            level=DataFreshnessLevel.STALE,
            latest_date=yesterday,
            expected_date=today,
            staleness_days=1,
        )
        assert status.is_usable
        assert status.requires_warning

    def test_old_status(self):
        """Test old data status."""
        old_date = date.today() - timedelta(days=5)
        today = date.today()
        status = DataFreshnessStatus(
            level=DataFreshnessLevel.OLD,
            latest_date=old_date,
            expected_date=today,
            staleness_days=5,
        )
        assert not status.is_usable

    def test_missing_status(self):
        """Test missing data status."""
        status = DataFreshnessStatus(
            level=DataFreshnessLevel.MISSING,
        )
        assert not status.is_usable


class TestIngestionHealthReport:
    """Tests for IngestionHealthReport dataclass."""

    def test_healthy_report(self):
        """Test healthy report."""
        freshness = DataFreshnessStatus(
            level=DataFreshnessLevel.FRESH,
            latest_date=date.today(),
            expected_date=date.today(),
        )
        report = IngestionHealthReport(
            is_healthy=True,
            primary_source_ready=True,
            freshness=freshness,
        )
        assert report.should_proceed
        assert not report.should_abort

    def test_unhealthy_report(self):
        """Test unhealthy report."""
        report = IngestionHealthReport(
            is_healthy=False,
            primary_source_ready=False,
        )
        report.add_error("Primary data not found")

        assert not report.should_proceed
        assert report.should_abort

    def test_report_with_warnings(self):
        """Test report with warnings."""
        report = IngestionHealthReport(
            is_healthy=True,
            primary_source_ready=True,
        )
        report.add_warning("Data is stale")

        assert len(report.warnings) == 1
        assert report.should_proceed

    def test_to_dict(self):
        """Test report serialization."""
        report = IngestionHealthReport(
            is_healthy=True,
            primary_source_ready=True,
            can_trade=True,
        )
        d = report.to_dict()

        assert d["is_healthy"] is True
        assert d["can_trade"] is True


class TestClassifyFreshness:
    """Tests for classify_freshness function."""

    def test_fresh_data(self):
        """Test fresh data classification."""
        today = date.today()
        level, days = classify_freshness(today, today)
        assert level == DataFreshnessLevel.FRESH
        assert days == 0

    def test_stale_data(self):
        """Test stale data classification."""
        expected_date = date(2026, 2, 27)  # Friday
        latest_date = date(2026, 2, 26)    # Thursday
        level, days = classify_freshness(latest_date, expected_date, max_staleness_days=1)
        assert level == DataFreshnessLevel.STALE
        assert days == 1

    def test_old_data(self):
        """Test old data classification."""
        expected_date = date(2026, 2, 27)  # Friday
        old_date = date(2026, 2, 23)       # Monday
        level, days = classify_freshness(old_date, expected_date, max_staleness_days=1)
        assert level == DataFreshnessLevel.OLD

    def test_missing_data(self):
        """Test missing data classification."""
        level, days = classify_freshness(None, date.today())
        assert level == DataFreshnessLevel.MISSING


class TestBulkDataRequest:
    """Tests for BulkDataRequest dataclass."""

    def test_from_list(self):
        """Test creating from list."""
        req = BulkDataRequest(tickers=["AAPL", "MSFT", "GOOGL"])
        assert len(req.tickers) == 3
        assert "AAPL" in req.tickers

    def test_from_string(self):
        """Test creating from comma-separated string."""
        req = BulkDataRequest(tickers="AAPL,MSFT,GOOGL")
        assert len(req.tickers) == 3

    def test_defaults(self):
        """Test default values."""
        req = BulkDataRequest(tickers=["AAPL"])
        assert req.days == 250
        assert req.include_features is True
        assert req.batch_size == 50


class TestBulkDataResult:
    """Tests for BulkDataResult dataclass."""

    def test_empty_result(self):
        """Test empty result."""
        result = BulkDataResult(results={}, errors={})
        assert result.total_tickers == 0
        assert result.success_rate == 0.0

    def test_full_success(self):
        """Test fully successful result."""
        df = pd.DataFrame({"close": [100]})
        results = {
            "AAPL": DataFrameSnapshot(
                df=df, ticker="AAPL", source=DataSourceType.LOCAL_PARQUET
            ),
            "MSFT": DataFrameSnapshot(
                df=df, ticker="MSFT", source=DataSourceType.LOCAL_PARQUET
            ),
        }
        result = BulkDataResult(
            results=results,
            successful=["AAPL", "MSFT"],
        )
        assert result.success_count == 2
        assert result.failure_count == 0
        assert result.is_fully_successful

    def test_partial_success(self):
        """Test partially successful result."""
        results = {
            "AAPL": DataFrameSnapshot(
                df=pd.DataFrame(), ticker="AAPL", source=DataSourceType.LOCAL_PARQUET
            ),
        }
        errors = {"MSFT": "Not found"}

        result = BulkDataResult(
            results=results,
            successful=["AAPL"],
            failed=["MSFT"],
            errors=errors,
        )
        assert result.success_count == 1
        assert result.failure_count == 1
        assert result.is_partially_successful
        assert not result.is_fully_successful
        assert result.success_rate == 0.5


class TestSchemaNormalizer:
    """Tests for SchemaNormalizer utility."""

    def test_normalize_columns(self):
        """Test column normalization."""
        df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [105.0],
                "CLOSE": [102.0],
            }
        )

        from autotrade.data_ingestion.interfaces import SchemaNormalizer

        result = SchemaNormalizer.normalize_columns(df)

        assert "open" in result.columns
        assert "high" in result.columns
        assert "close" in result.columns

    def test_ensure_numeric(self):
        """Test numeric conversion."""
        df = pd.DataFrame(
            {
                "open": ["100.0", "101.0"],
                "close": [102.0, 103.0],
            }
        )

        from autotrade.data_ingestion.interfaces import SchemaNormalizer

        result = SchemaNormalizer.ensure_numeric(df)

        assert result["open"].dtype in (np.float64, np.int64)


class TestIngestionPaths:
    def test_paths_resolve_from_typed_config(self, monkeypatch):
        cfg = SimpleNamespace(
            data=SimpleNamespace(
                downday_root="C:/tmp/down",
                daily_features_parquet="data/daily_features.parquet",
                hourly_prices_parquet="data/prices_hourly.parquet",
                daily_features_h5="daily_features.h5",
                market_data_duckdb="data/market_data.duckdb",
            )
        )
        monkeypatch.setattr(ingestion_paths, "get_config", lambda: cfg)

        resolved = ingestion_paths.get_ingestion_paths()
        assert str(resolved.downday_root).endswith("tmp\\down") or str(resolved.downday_root).endswith("tmp/down")
        assert str(resolved.daily_features_parquet).endswith("data\\daily_features.parquet") or str(resolved.daily_features_parquet).endswith("data/daily_features.parquet")
        assert str(resolved.daily_features_h5).endswith("daily_features.h5")


class TestBootstrapBehavior:
    def _cfg(self, fail_fast: bool = False):
        return SimpleNamespace(
            data=SimpleNamespace(
                max_staleness_days=1,
                bootstrap_from_h5_enabled=True,
                fail_fast_on_missing_core_data=fail_fast,
            )
        )

    def test_bootstrap_runs_when_parquet_missing(self, monkeypatch, tmp_path):
        parquet_path = tmp_path / "data" / "daily_features.parquet"
        h5_path = tmp_path / "daily_features.h5"
        h5_path.write_text("placeholder")

        calls = {"convert": 0}

        def _fake_convert(src: Path, dst: Path) -> int:
            calls["convert"] += 1
            df = pd.DataFrame(
                {
                    "ticker": ["AAPL", "MSFT"],
                    "Date": pd.to_datetime(["2026-02-11", "2026-02-11"]),
                    "Open": [100.0, 200.0],
                    "High": [101.0, 201.0],
                    "Low": [99.0, 199.0],
                    "Close": [100.5, 200.5],
                    "Volume": [1_000_000, 2_000_000],
                }
            )
            dst.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dst, index=False)
            return len(df)

        monkeypatch.setattr(ingestion_bootstrap, "get_config", lambda: self._cfg(False))
        monkeypatch.setattr(ingestion_bootstrap, "get_primary_parquet_path", lambda: parquet_path)
        monkeypatch.setattr(ingestion_bootstrap, "get_bootstrap_h5_candidates", lambda: [h5_path])
        monkeypatch.setattr(ingestion_bootstrap, "_convert_h5_to_parquet", _fake_convert)

        report1 = ingestion_bootstrap.ensure_core_market_data_ready()
        report2 = ingestion_bootstrap.ensure_core_market_data_ready()

        assert parquet_path.exists()
        assert report1.primary_source_ready
        assert report2.primary_source_ready
        assert calls["convert"] == 1  # deterministic: no repeated conversion once parquet exists

    def test_fail_fast_raises_when_core_data_missing(self, monkeypatch, tmp_path):
        missing_parquet = tmp_path / "data" / "daily_features.parquet"

        monkeypatch.setattr(ingestion_bootstrap, "get_config", lambda: self._cfg(True))
        monkeypatch.setattr(ingestion_bootstrap, "get_primary_parquet_path", lambda: missing_parquet)
        monkeypatch.setattr(ingestion_bootstrap, "get_bootstrap_h5_candidates", lambda: [])

        with pytest.raises(CoreDataMissingError):
            ingestion_bootstrap.ensure_core_market_data_ready(
                bootstrap_from_h5_enabled=False,
                fail_fast=True,
            )

    def test_stale_data_is_classified_as_old(self, monkeypatch, tmp_path):
        parquet_path = tmp_path / "data" / "daily_features.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        old_date = date.today() - timedelta(days=10)
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "Date": pd.to_datetime([old_date]),
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.5],
                "Volume": [1_000_000],
            }
        )
        df.to_parquet(parquet_path, index=False)

        monkeypatch.setattr(ingestion_bootstrap, "get_config", lambda: self._cfg(False))
        monkeypatch.setattr(ingestion_bootstrap, "get_primary_parquet_path", lambda: parquet_path)
        monkeypatch.setattr(ingestion_bootstrap, "get_bootstrap_h5_candidates", lambda: [])

        report = ingestion_bootstrap.ensure_core_market_data_ready(
            max_staleness_days=1,
            bootstrap_from_h5_enabled=False,
            fail_fast=False,
            parquet_path=parquet_path,
        )

        assert report.freshness is not None
        assert report.freshness.level == DataFreshnessLevel.OLD
        assert report.can_trade is False


class TestGoldenFixtures:
    def test_schema_normalization_golden_fixture(self):
        fixture_dir = Path("tests/fixtures/data_ingestion")
        raw_df = pd.read_csv(fixture_dir / "raw_daily_features.csv")
        expected = json.loads((fixture_dir / "golden_normalized_columns.json").read_text())

        normalized = SchemaNormalizer.standardize(raw_df)
        assert list(normalized.columns) == expected["normalized_columns"]

    def test_ticker_universe_golden_fixture(self):
        fixture_dir = Path("tests/fixtures/data_ingestion")
        raw_df = pd.read_csv(fixture_dir / "raw_daily_features.csv")
        expected = json.loads((fixture_dir / "golden_ticker_universe.json").read_text())

        normalized = SchemaNormalizer.standardize(raw_df)
        tickers = sorted(
            normalized["symbol"].dropna().astype(str).str.upper().unique().tolist()
        )
        assert tickers == expected["tickers"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
