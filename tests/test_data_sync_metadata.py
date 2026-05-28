from __future__ import annotations

import duckdb
import pandas as pd

from autotrade.utils import data_sync as data_sync_module
from autotrade.utils.data_sync import DataSyncManager


def test_sync_duckdb_enriches_daily_prices_with_security_metadata(
    tmp_path, monkeypatch
):
    parquet_path = tmp_path / "daily_features.parquet"
    duckdb_path = tmp_path / "market_data.duckdb"
    hourly_path = tmp_path / "hourly_prices.parquet"
    metadata_path = tmp_path / "nasdaq_screener.csv"

    pd.DataFrame(
        [
            {
                "ticker": "ABCD",
                "Date": pd.Timestamp("2026-03-08"),
                "Open": 20.0,
                "High": 21.0,
                "Low": 19.5,
                "Close": 20.5,
                "Adj Close": 20.5,
                "Volume": 1_500_000,
                "SMA_10": 19.8,
                "SMA_20": 19.5,
                "EMA_10": 19.9,
                "EMA_20": 19.6,
                "RSI_14": 54.0,
                "MACD": 0.4,
                "atr_14": 1.2,
            }
        ]
    ).to_parquet(parquet_path, index=False)
    pd.DataFrame(
        [{"Symbol": "ABCD", "Sector": "Industrials", "Market Cap": 4500000000}]
    ).to_csv(
        metadata_path,
        index=False,
    )

    monkeypatch.setitem(
        data_sync_module.DATA_SOURCES, "daily_features_parquet", parquet_path
    )
    monkeypatch.setitem(
        data_sync_module.DATA_SOURCES, "market_data_duckdb", duckdb_path
    )
    monkeypatch.setitem(
        data_sync_module.DATA_SOURCES, "hourly_prices_parquet", hourly_path
    )
    monkeypatch.setattr(
        data_sync_module, "get_nasdaq_screener_path", lambda: metadata_path
    )

    sync = DataSyncManager()

    assert sync.sync_duckdb() is True

    con = duckdb.connect(str(duckdb_path), read_only=True)
    row = con.execute(
        "SELECT symbol, sector, market_cap FROM daily_prices WHERE symbol = 'ABCD'"
    ).fetchone()
    con.close()

    assert row == ("ABCD", "Industrials", 4_500_000_000.0)


def test_core_market_data_readiness_fails_closed_when_primary_date_is_stale(
    monkeypatch,
):
    class _StubManager:
        def check_sync_status(self):
            return {
                "primary_date": "2026-03-05",
                "primary_tickers": 1234,
                "all_synced": True,
            }

    monkeypatch.setattr(data_sync_module, "get_manager", lambda: _StubManager())
    monkeypatch.setattr(
        data_sync_module,
        "expected_core_market_data_date",
        lambda reference_dt=None, market_close_local=data_sync_module.MARKET_CLOSE_LOCAL: "2026-03-07",
    )

    readiness = data_sync_module.get_core_market_data_readiness()

    assert readiness["is_fresh"] is False
    assert readiness["pm_ready_for_execution"] is False
    assert readiness["primary_date"] == "2026-03-05"
    assert readiness["expected_date"] == "2026-03-07"
    assert "core_data_stale:2026-03-05->2026-03-07" in readiness["blocking_reasons"]


def test_check_sync_status_reuses_cached_summary_when_files_unchanged(
    tmp_path, monkeypatch
):
    parquet_path = tmp_path / "daily_features.parquet"
    duckdb_path = tmp_path / "market_data.duckdb"
    hourly_path = tmp_path / "hourly_prices.parquet"
    h5_path = tmp_path / "daily_features.h5"
    daily_csv_path = tmp_path / "prices_daily.csv"
    hourly_csv_path = tmp_path / "prices_hourly.csv"
    for path in (
        parquet_path,
        duckdb_path,
        hourly_path,
        h5_path,
        daily_csv_path,
        hourly_csv_path,
    ):
        path.write_text("stub", encoding="utf-8")

    monkeypatch.setitem(
        data_sync_module.DATA_SOURCES, "daily_features_parquet", parquet_path
    )
    monkeypatch.setitem(
        data_sync_module.DATA_SOURCES, "market_data_duckdb", duckdb_path
    )
    monkeypatch.setitem(
        data_sync_module.DATA_SOURCES, "hourly_prices_parquet", hourly_path
    )
    monkeypatch.setitem(data_sync_module.DATA_SOURCES, "daily_features_h5", h5_path)
    monkeypatch.setitem(data_sync_module.DATA_SOURCES, "prices_daily_csv", daily_csv_path)
    monkeypatch.setitem(
        data_sync_module.DATA_SOURCES, "prices_hourly_csv", hourly_csv_path
    )

    sync = DataSyncManager()
    calls = {"source_checks": 0}

    monkeypatch.setattr(
        sync,
        "get_primary_latest_date",
        lambda: ("2026-03-09", 4806),
    )
    original_check_source_status = sync.check_source_status

    def _counting_check_source_status(source_name):
        calls["source_checks"] += 1
        return original_check_source_status(source_name)

    monkeypatch.setattr(sync, "check_source_status", _counting_check_source_status)

    first = sync.check_sync_status()
    second = sync.check_sync_status()

    assert first["primary_date"] == "2026-03-09"
    assert second["primary_date"] == "2026-03-09"
    assert calls["source_checks"] == len(data_sync_module.DATA_SOURCES)
