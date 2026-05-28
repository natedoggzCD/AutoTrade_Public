from __future__ import annotations

from typing import Any


DEFAULT_MIN_MARKET_CAP = 2_000_000_000.0
DEFAULT_MAX_MARKET_CAP = 10_000_000_000.0
DEFAULT_POST_SPIKE_VOLUME_THRESHOLD = 5.0
DEFAULT_POST_SPIKE_RANGE_ATR_THRESHOLD = 1.5

HARD_BLOCK_MEGA_CAP = {
    "SNOW",
    "PANW",
    "MRVL",
    "VEEV",
    "WDAY",
    "ZS",
    "CRWD",
    "ADBE",
    "ORCL",
    "CRM",
    "NOW",
    "INTU",
    "AMD",
    "AVGO",
    "QCOM",
    "TXN",
    "OXY",
    "XOM",
    "CVX",
    "COP",
    "PSX",
    "MPC",
    "VLO",
    "JPM",
    "BAC",
    "WFC",
    "C",
    "GS",
    "MS",
    "UNH",
    "JNJ",
    "LLY",
    "PFE",
    "MRK",
    "ABBV",
    "WMT",
    "COST",
    "HD",
    "LOW",
    "TGT",
    "DIS",
    "NFLX",
    "CMCSA",
    "T",
    "VZ",
}


def clean_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def atr_fraction(row: dict[str, Any], close: float) -> float:
    atr_pct = safe_float(
        row.get("atr_14_pct", row.get("atr_pct", row.get("atr_percent"))),
        0.0,
    )
    if atr_pct > 1.0:
        return atr_pct / 100.0
    if atr_pct > 0.0:
        return atr_pct
    atr = safe_float(row.get("atr_14"), 0.0)
    return atr / close if close > 0 and atr > 0 else 0.0


def is_post_spike_long_candidate(
    row: dict[str, Any],
    *,
    volume_threshold: float = DEFAULT_POST_SPIKE_VOLUME_THRESHOLD,
    range_atr_threshold: float = DEFAULT_POST_SPIKE_RANGE_ATR_THRESHOLD,
) -> bool:
    volume_ratio = safe_float(row.get("volume_ratio", row.get("volume_ratio_20d")), 0.0)
    if volume_ratio < volume_threshold:
        return False

    high = safe_float(row.get("prior_high", row.get("high", row.get("High"))), 0.0)
    low = safe_float(row.get("prior_low", row.get("low", row.get("Low"))), 0.0)
    close = safe_float(
        row.get(
            "prior_close", row.get("close", row.get("Close", row.get("entry_price")))
        ),
        0.0,
    )
    if high <= low or close <= 0:
        return False

    atr_pct = atr_fraction(row, close)
    if atr_pct <= 0:
        return False

    range_pct = (high - low) / close
    return range_pct >= range_atr_threshold * atr_pct


def signal_universe_rejection_reason(
    row: dict[str, Any],
    *,
    min_market_cap: float = DEFAULT_MIN_MARKET_CAP,
    max_market_cap: float = DEFAULT_MAX_MARKET_CAP,
    post_spike_volume_threshold: float = DEFAULT_POST_SPIKE_VOLUME_THRESHOLD,
    post_spike_range_atr_threshold: float = DEFAULT_POST_SPIKE_RANGE_ATR_THRESHOLD,
) -> str | None:
    symbol = clean_symbol(row.get("symbol") or row.get("ticker"))
    if symbol in HARD_BLOCK_MEGA_CAP:
        return "hard_block_mega_cap"

    market_cap = safe_float(row.get("market_cap"), 0.0)
    if market_cap > 0 and not (min_market_cap <= market_cap <= max_market_cap):
        return "market_cap_out_of_range"

    if is_post_spike_long_candidate(
        row,
        volume_threshold=post_spike_volume_threshold,
        range_atr_threshold=post_spike_range_atr_threshold,
    ):
        return "post_spike_long_exclusion"

    return None
