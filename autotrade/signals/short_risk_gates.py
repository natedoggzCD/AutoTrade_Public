"""Risk gates for stock-side short entries."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from autotrade.utils.safe_logging import get_safe_logger

logger = get_safe_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINANCIAL_DB = PROJECT_ROOT / "data" / "financial.db"


def _float_value(payload: Dict[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        value = payload.get(name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _next_earnings_days(
    conn: sqlite3.Connection, symbol: str, *, as_of_date: date
) -> Optional[int]:
    if not _table_exists(conn, "earnings_calendar"):
        return None
    cols = {
        str(row[1]).lower(): str(row[1])
        for row in conn.execute("PRAGMA table_info(earnings_calendar)").fetchall()
    }
    symbol_col = cols.get("symbol") or cols.get("ticker")
    date_col = (
        cols.get("earnings_date")
        or cols.get("report_date")
        or cols.get("date")
        or cols.get("event_date")
    )
    if not symbol_col or not date_col:
        return None
    row = conn.execute(
        f"""
        SELECT {date_col}
        FROM earnings_calendar
        WHERE UPPER({symbol_col}) = ?
          AND DATE({date_col}) >= DATE(?)
        ORDER BY DATE({date_col}) ASC
        LIMIT 1
        """,
        (symbol.upper(), as_of_date.isoformat()),
    ).fetchone()
    if row is None:
        return None
    try:
        earnings_date = date.fromisoformat(str(row[0])[:10])
    except ValueError:
        return None
    return (earnings_date - as_of_date).days


def _key_stats(conn: sqlite3.Connection, symbol: str) -> Dict[str, Any]:
    if not _table_exists(conn, "key_stats"):
        return {}
    cols = [
        str(row[1]) for row in conn.execute("PRAGMA table_info(key_stats)").fetchall()
    ]
    lower = {c.lower(): c for c in cols}
    symbol_col = lower.get("symbol") or lower.get("ticker")
    if not symbol_col:
        return {}
    selected = ", ".join(cols)
    row = conn.execute(
        f"SELECT {selected} FROM key_stats WHERE UPPER({symbol_col}) = ? LIMIT 1",
        (symbol.upper(),),
    ).fetchone()
    if row is None:
        return {}
    return dict(zip(cols, row))


def passes_short_gates(
    symbol: str,
    signal: Dict[str, Any],
    *,
    financial_db_path: Path = DEFAULT_FINANCIAL_DB,
    as_of_date: Optional[date] = None,
) -> Tuple[bool, str]:
    """Return (allowed, reason) for a short signal."""
    ticker = str(symbol or signal.get("ticker") or signal.get("symbol") or "").upper()
    if not ticker:
        return False, "missing_symbol"
    signal = dict(signal or {})
    price = _float_value(signal, "price", "entry", "entry_price", "close")
    if price < 5.0:
        return False, "price_floor"

    close = _float_value(signal, "close", "price", "entry", "entry_price")
    ema20 = _float_value(signal, "ema_20", "ema20")
    ema50 = _float_value(signal, "ema_50", "ema50")
    if close > 0 and ema20 > 0 and ema50 > 0 and close > ema20 and close > ema50:
        return False, "uptrend_block"

    stats: Dict[str, Any] = {}
    if financial_db_path.exists():
        try:
            with sqlite3.connect(financial_db_path) as conn:
                days = _next_earnings_days(
                    conn,
                    ticker,
                    as_of_date=as_of_date or date.today(),
                )
                if days is not None and days < 5:
                    return False, "earnings_block"
                stats = _key_stats(conn, ticker)
        except Exception as exc:
            logger.warning("[SHORT GATES] DB lookup failed for %s: %s", ticker, exc)

    merged = {**stats, **signal}
    short_interest = _float_value(
        merged,
        "short_interest_pct_float",
        "short_percent_of_float",
        "short_pct_float",
    )
    days_to_cover = _float_value(merged, "days_to_cover", "short_ratio")
    if short_interest > 25.0 and days_to_cover > 4.0:
        return False, "crowded_short_block"

    float_shares = _float_value(
        merged,
        "float_shares",
        "shares_float",
        "float",
        default=5_000_000.0,
    )
    if float_shares < 5_000_000:
        return False, "float_floor"

    return True, "ok"
