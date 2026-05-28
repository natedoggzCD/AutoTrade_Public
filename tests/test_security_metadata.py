from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from autotrade.utils.security_metadata import (
    get_company_aliases,
    get_nasdaq_screener_path,
    load_security_metadata,
    load_company_aliases,
    normalize_sector_label,
    validate_ticker_sector_sidecar,
)


def test_get_nasdaq_screener_path_prefers_configured_existing_file(
    tmp_path, monkeypatch
):
    configured = tmp_path / "TradingMethods" / "nasdaq_screener.csv"
    configured.parent.mkdir()
    configured.write_text("Symbol,Sector,Industry,Market Cap\n", encoding="utf-8")
    downday_root = tmp_path / "DownDay"
    downday_root.mkdir()
    (downday_root / "nasdaq_screener.csv").write_text(
        "Symbol,Sector,Industry,Market Cap\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        "autotrade.utils.security_metadata.get_ingestion_paths",
        lambda: SimpleNamespace(
            nasdaq_screener_csv=configured,
            downday_root=downday_root,
        ),
    )

    assert get_nasdaq_screener_path() == configured


def test_get_nasdaq_screener_path_falls_back_to_downday_when_config_missing(
    tmp_path, monkeypatch
):
    downday_root = tmp_path / "DownDay"
    downday_root.mkdir()
    fallback = downday_root / "nasdaq_screener.csv"
    monkeypatch.setattr(
        "autotrade.utils.security_metadata.get_ingestion_paths",
        lambda: SimpleNamespace(
            nasdaq_screener_csv=tmp_path / "missing" / "nasdaq_screener.csv",
            downday_root=downday_root,
        ),
    )

    assert get_nasdaq_screener_path() == fallback


def test_load_security_metadata_normalizes_sector_and_market_cap(tmp_path):
    path = tmp_path / "nasdaq_screener.csv"
    pd.DataFrame(
        [
            {
                "Symbol": "abcd",
                "Sector": " Industrials ",
                "Industry": " Machinery ",
                "Market Cap": "4,500,000,000",
            },
            {
                "Symbol": "efgh",
                "Sector": "",
                "Industry": "",
                "Market Cap": "",
            },
        ]
    ).to_csv(path, index=False)

    df = load_security_metadata(path)

    assert list(df.columns) == ["ticker", "sector", "industry", "market_cap"]
    assert df.loc[df["ticker"] == "ABCD", "sector"].iloc[0] == "Industrials"
    assert df.loc[df["ticker"] == "ABCD", "industry"].iloc[0] == "Machinery"
    assert df.loc[df["ticker"] == "ABCD", "market_cap"].iloc[0] == 4_500_000_000
    assert pd.isna(df.loc[df["ticker"] == "EFGH", "sector"]).iloc[0]
    assert pd.isna(df.loc[df["ticker"] == "EFGH", "industry"]).iloc[0]


def test_load_security_metadata_parses_scientific_market_cap(tmp_path):
    path = tmp_path / "nasdaq_screener.csv"
    path.write_text(
        "\n".join(
            [
                "Symbol,Sector,Industry,Market Cap",
                "abcd,Industrials,Machinery,4.5E+09",
                'efgh,Technology,Software,"$5,250,000,000"',
            ]
        ),
        encoding="utf-8",
    )

    df = load_security_metadata(path)

    assert list(df.columns) == ["ticker", "sector", "industry", "market_cap"]
    assert df.loc[df["ticker"] == "ABCD", "market_cap"].iloc[0] == 4_500_000_000
    assert df.loc[df["ticker"] == "EFGH", "market_cap"].iloc[0] == 5_250_000_000


def test_load_company_aliases_derives_clean_names_from_nasdaq_screener(tmp_path):
    path = tmp_path / "nasdaq_screener.csv"
    path.write_text(
        "\n".join(
            [
                "Symbol,Name,Sector,Industry,Market Cap",
                'AMD,"Advanced Micro Devices, Inc. Common Stock",Technology,Semiconductors,1',
                'TSLA,"Tesla, Inc. Common Stock",Consumer Discretionary,Auto,1',
                "AACBR,Artius II Acquisition Inc. Rights,,,0",
                'GDRX,"GoodRx Holdings, Inc. Class A Common Stock",Healthcare,Services,1',
                'RYN,"Rayonier Inc. REIT Common Stock",Real Estate,REIT,1',
                'MRK,"Merck & Company, Inc. Common Stock",Healthcare,Drug Manufacturers,1',
            ]
        ),
        encoding="utf-8",
    )

    aliases = load_company_aliases(path)

    assert aliases["AMD"] == ("Advanced Micro Devices, Inc", "Advanced Micro Devices")
    assert aliases["TSLA"] == ("Tesla, Inc", "Tesla")
    assert aliases["AACBR"] == ("Artius II Acquisition Inc", "Artius II Acquisition")
    assert aliases["GDRX"] == ("GoodRx Holdings, Inc", "GoodRx")
    assert aliases["RYN"] == ("Rayonier Inc. REIT", "Rayonier")
    assert aliases["MRK"] == ("Merck & Company, Inc", "Merck")


def test_get_company_aliases_returns_empty_for_missing_symbol(tmp_path):
    path = tmp_path / "nasdaq_screener.csv"
    path.write_text("Symbol,Name\nABC,ABC Corp. Common Stock\n", encoding="utf-8")

    assert get_company_aliases("XYZ", path) == ()


def test_sector_sidecar_validation_requires_schema_and_canonical_sectors(tmp_path):
    path = tmp_path / "ticker_sectors.parquet"
    pd.DataFrame(
        [
            {
                "ticker": "ABCD",
                "sector": "financial_services",
                "updated_at": "2026-05-05T12:00:00+00:00",
            }
        ]
    ).to_parquet(path, index=False)

    validation = validate_ticker_sector_sidecar(path, max_age_days=3650)

    assert validation.valid is False
    assert validation.missing_columns == ("industry",)
    assert validation.invalid_sectors == ("financial_services",)
    assert normalize_sector_label("Financial Services") == "financials"


def test_sector_sidecar_validation_accepts_canonical_fresh_schema(tmp_path):
    path = tmp_path / "ticker_sectors.parquet"
    pd.DataFrame(
        [
            {
                "ticker": "ABCD",
                "sector": "financials",
                "industry": "capital_markets",
                "updated_at": pd.Timestamp.utcnow().isoformat(),
            }
        ]
    ).to_parquet(path, index=False)

    validation = validate_ticker_sector_sidecar(path, max_age_days=1)

    assert validation.valid is True
    assert validation.row_count == 1
