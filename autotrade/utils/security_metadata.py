"""
Security metadata helpers sourced from the configured NASDAQ screener CSV.
"""

from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Optional

import pandas as pd

from autotrade.data_ingestion.paths import get_ingestion_paths

DEFAULT_NASDAQ_SCREENER_CSV = Path("data/downday/nasdaq_screener.csv")

CANONICAL_SECTOR_LABELS = {
    "materials",
    "communication_services",
    "energy",
    "financials",
    "industrials",
    "technology",
    "consumer_staples",
    "real_estate",
    "utilities",
    "healthcare",
    "consumer_discretionary",
}

SECTOR_ALIASES = {
    "basic_materials": "materials",
    "communication": "communication_services",
    "communications": "communication_services",
    "consumer_cyclical": "consumer_discretionary",
    "financial": "financials",
    "finance": "financials",
    "financial_services": "financials",
    "health_care": "healthcare",
    "real_estate_services": "real_estate",
    "telecommunication": "communication_services",
    "telecommunications": "communication_services",
}


@dataclass(frozen=True)
class SectorSidecarValidation:
    valid: bool
    path: str
    row_count: int = 0
    missing_columns: tuple[str, ...] = ()
    invalid_sectors: tuple[str, ...] = ()
    stale: bool = False
    max_updated_at: Optional[datetime] = None
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def reason(self) -> str:
        parts = []
        if self.missing_columns:
            parts.append(f"missing_columns={','.join(self.missing_columns)}")
        if self.invalid_sectors:
            parts.append(f"invalid_sectors={','.join(self.invalid_sectors[:5])}")
        if self.stale:
            parts.append("stale_updated_at")
        if self.errors:
            parts.extend(self.errors)
        if not parts and not self.valid:
            parts.append("invalid_sector_sidecar")
        return ";".join(parts)


def get_nasdaq_screener_path() -> Path:
    """Resolve the configured Nasdaq screener CSV, falling back to DownDay."""
    paths = get_ingestion_paths()
    configured = Path(
        getattr(paths, "nasdaq_screener_csv", DEFAULT_NASDAQ_SCREENER_CSV)
    )
    if configured.exists():
        return configured
    return paths.downday_root / "nasdaq_screener.csv"


def normalize_sector_label(value: object) -> Optional[str]:
    """Normalize a raw sector label to the canonical sector taxonomy."""
    if value is None or pd.isna(value):
        return None
    cleaned = str(value).strip().lower().replace("&", "and")
    cleaned = cleaned.replace("/", " ").replace("-", " ")
    cleaned = "_".join(cleaned.split())
    if cleaned in {"", "unknown", "n/a", "na", "none", "null", "miscellaneous"}:
        return None
    return SECTOR_ALIASES.get(cleaned, cleaned)


def validate_ticker_sector_sidecar(
    path: Path,
    *,
    max_age_days: int = 14,
) -> SectorSidecarValidation:
    """Validate the ticker sector sidecar before it is used for live regime joins."""
    resolved = Path(path)
    if not resolved.exists():
        return SectorSidecarValidation(
            valid=False,
            path=str(resolved),
            errors=("missing_file",),
        )

    required = {"ticker", "sector", "industry", "updated_at"}
    try:
        df = pd.read_parquet(resolved)
    except Exception as exc:
        return SectorSidecarValidation(
            valid=False,
            path=str(resolved),
            errors=(f"read_failed:{type(exc).__name__}",),
        )

    missing = tuple(sorted(required - set(df.columns)))
    invalid_sectors: tuple[str, ...] = ()
    if "sector" in df.columns:
        sectors = {
            str(value).strip()
            for value in df["sector"].dropna().unique().tolist()
            if str(value).strip()
        }
        invalid_sectors = tuple(sorted(sectors - CANONICAL_SECTOR_LABELS))

    max_updated_at = None
    stale = False
    if "updated_at" in df.columns and not df.empty:
        parsed = pd.to_datetime(df["updated_at"], errors="coerce", utc=True)
        if parsed.notna().any():
            max_updated_at = parsed.max().to_pydatetime()
            age = datetime.now(timezone.utc) - max_updated_at.astimezone(timezone.utc)
            stale = age.days > int(max_age_days)
        else:
            stale = True

    valid = bool(not missing and not invalid_sectors and not stale)
    return SectorSidecarValidation(
        valid=valid,
        path=str(resolved),
        row_count=int(len(df)),
        missing_columns=missing,
        invalid_sectors=invalid_sectors,
        stale=stale,
        max_updated_at=max_updated_at,
    )


@lru_cache(maxsize=4)
def _load_security_metadata_cached(path_str: str, mtime_ns: int) -> pd.DataFrame:
    path = Path(path_str)
    cols = ["Symbol", "Sector", "Industry", "Market Cap"]
    df = pd.read_csv(path, usecols=lambda c: c in cols)
    df = df.rename(
        columns={
            "Symbol": "ticker",
            "Sector": "sector",
            "Industry": "industry",
            "Market Cap": "market_cap",
        }
    )
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["sector"] = df["sector"].astype("string").str.strip()
    df["sector"] = df["sector"].where(df["sector"].notna() & (df["sector"] != ""))
    if "industry" not in df.columns:
        df["industry"] = pd.NA
    df["industry"] = df["industry"].astype("string").str.strip()
    df["industry"] = df["industry"].where(
        df["industry"].notna() & (df["industry"] != "")
    )
    market_cap = (
        df["market_cap"]
        .astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    df["market_cap"] = pd.to_numeric(market_cap, errors="coerce")
    df = df.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"], keep="first")
    return df[["ticker", "sector", "industry", "market_cap"]].reset_index(drop=True)


def load_security_metadata(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load per-ticker metadata from the configured NASDAQ screener CSV.

    Returns an empty DataFrame with the expected columns when the file is absent.
    """
    resolved = Path(path) if path else get_nasdaq_screener_path()
    if not resolved.exists():
        return pd.DataFrame(columns=["ticker", "sector", "industry", "market_cap"])
    return _load_security_metadata_cached(
        str(resolved), resolved.stat().st_mtime_ns
    ).copy()


_SECURITY_NAME_SUFFIX_RE = re.compile(
    r"\b("
    r"american depositary shares|american depositary share|common stock|ordinary shares|"
    r"class [a-z] ordinary shares|class [a-z] common stock|common shares|capital stock|"
    r"depositary shares|preferred stock|warrants?|rights?|units?"
    r")\b.*$",
    re.IGNORECASE,
)
_LEGAL_SUFFIX_RE = re.compile(
    r"\b("
    r"incorporated|inc|corporation|corp|company|co|limited|ltd|plc|"
    r"holdings?|group|reit|s\.a\.|sa|n\.v\.|nv|ag|se"
    r")\b\.?",
    re.IGNORECASE,
)


def _normalize_company_alias(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = _SECURITY_NAME_SUFFIX_RE.sub("", text).strip(" ,.-")
    text = re.sub(r"\s+", " ", text)
    if not text:
        return None
    return text


def _alias_variants(name: str) -> tuple[str, ...]:
    variants: list[str] = []
    cleaned = _normalize_company_alias(name)
    if cleaned:
        variants.append(cleaned)
        legal_stripped = _LEGAL_SUFFIX_RE.sub("", cleaned).strip(" ,.-")
        legal_stripped = re.sub(r"\s*(?:&|and)\s*$", "", legal_stripped).strip(" ,.-")
        legal_stripped = re.sub(r"\s+", " ", legal_stripped)
        if legal_stripped and legal_stripped != cleaned:
            variants.append(legal_stripped)

    out: list[str] = []
    seen: set[str] = set()
    for alias in variants:
        key = alias.casefold()
        if len(alias) < 3 or key in seen:
            continue
        seen.add(key)
        out.append(alias)
    return tuple(out)


@lru_cache(maxsize=4)
def _load_company_aliases_cached(
    path_str: str, mtime_ns: int
) -> dict[str, tuple[str, ...]]:
    path = Path(path_str)
    df = pd.read_csv(path, usecols=lambda c: c in {"Symbol", "Name"})
    if "Symbol" not in df.columns or "Name" not in df.columns:
        return {}

    aliases: dict[str, tuple[str, ...]] = {}
    for _, row in df.iterrows():
        ticker = str(row.get("Symbol", "") or "").upper().strip()
        if not ticker:
            continue
        variants = _alias_variants(row.get("Name"))
        if variants:
            aliases[ticker] = variants
    return aliases


def load_company_aliases(path: Optional[Path] = None) -> dict[str, tuple[str, ...]]:
    """
    Load ticker -> company-name aliases from the configured Nasdaq screener CSV.

    The result is cached by source file mtime so aliases refresh automatically
    after the screener is updated.
    """
    resolved = Path(path) if path else get_nasdaq_screener_path()
    if not resolved.exists():
        return {}
    return dict(
        _load_company_aliases_cached(str(resolved), resolved.stat().st_mtime_ns)
    )


def get_company_aliases(ticker: str, path: Optional[Path] = None) -> tuple[str, ...]:
    symbol = str(ticker or "").upper().strip()
    if not symbol:
        return ()
    return load_company_aliases(path).get(symbol, ())
