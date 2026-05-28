from __future__ import annotations

from datetime import datetime

import pandas as pd

from autotrade.signals.universe_scanner import UniverseScanner


def _make_test_parquet(path) -> None:
    rows = []
    base_date = pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=39)
    for idx in range(40):
        close = 20.0 + idx * 0.7
        volume = 5_000_000 if idx == 39 else 1_200_000 + idx * 1_000
        rows.append(
            {
                "ticker": "ABCD",
                "Date": base_date + pd.Timedelta(days=idx),
                "Open": close - 0.2,
                "High": close + 0.4,
                "Low": close - 0.5,
                "Close": close,
                "Volume": volume,
                "RSI_14": 55.0,
                "RSI_14_lag_1": 53.0,
                "SMA_20": max(5.0, close - 0.8),
                "atr_14": 1.5,
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _make_security_csv(path) -> None:
    pd.DataFrame(
        [
            {
                "Symbol": "ABCD",
                "Sector": "Industrials",
                "Industry": "Machinery",
                "Market Cap": 4_500_000_000,
            }
        ]
    ).to_csv(path, index=False)


def test_build_scored_query_skips_market_cap_filter_when_column_missing(tmp_path):
    parquet_path = tmp_path / "daily_features.parquet"
    _make_test_parquet(parquet_path)
    scanner = UniverseScanner(parquet_path=parquet_path)

    query, _ = scanner._build_scored_query(
        lookback_days=5,
        has_sector=False,
        has_market_cap=False,
        parquet_has_sector=False,
        parquet_has_market_cap=False,
        metadata_path=None,
    )

    assert "l.market_cap" not in query


def test_build_scored_query_uses_small_mid_cap_ceiling_and_blocklist(tmp_path):
    parquet_path = tmp_path / "daily_features.parquet"
    _make_test_parquet(parquet_path)
    scanner = UniverseScanner(parquet_path=parquet_path)

    query, _ = scanner._build_scored_query(
        lookback_days=5,
        has_sector=False,
        has_market_cap=True,
        parquet_has_sector=False,
        parquet_has_market_cap=True,
        metadata_path=None,
    )

    assert "10000000000.0" in query
    assert "'SNOW'" in query
    assert "volume_ratio_20d" in query


def test_run_union_scan_handles_parquet_without_market_cap(tmp_path):
    parquet_path = tmp_path / "daily_features.parquet"
    _make_test_parquet(parquet_path)
    scanner = UniverseScanner(parquet_path=parquet_path)

    result = scanner._run_union_scan(max_candidates=10, lookback_days=5)

    assert isinstance(result.data, pd.DataFrame)


def test_run_union_scan_enriches_sector_from_security_metadata(tmp_path, monkeypatch):
    parquet_path = tmp_path / "daily_features.parquet"
    metadata_path = tmp_path / "nasdaq_screener.csv"
    _make_test_parquet(parquet_path)
    _make_security_csv(metadata_path)
    scanner = UniverseScanner(parquet_path=parquet_path)
    monkeypatch.setattr(
        scanner,
        "_resolve_security_metadata_path",
        lambda: metadata_path,
    )

    result = scanner._run_union_scan(max_candidates=10, lookback_days=5)

    assert "sector" in result.data.columns
    assert "market_cap" in result.data.columns
    assert result.data["sector"].dropna().iloc[0] == "Industrials"
    assert result.data["market_cap"].dropna().iloc[0] == 4_500_000_000
