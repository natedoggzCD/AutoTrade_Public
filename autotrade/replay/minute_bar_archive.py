from __future__ import annotations

import json
import logging
import time as time_module
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

ET = ZoneInfo("America/New_York")
CT = ZoneInfo("America/Chicago")
UTC = ZoneInfo("UTC")
REGULAR_START_ET = time(9, 30)
REGULAR_END_ET = time(16, 0)

logger = logging.getLogger("minute_bar_archive")

# Module-level flag: once Alpaca returns a subscription/permission error for
# recent SIP minute data, skip it for the remainder of the process and go
# straight to yfinance. Avoids 200+ identical WARNs per archive cycle.
_ALPACA_RECENT_DATA_DENIED: bool = False
_ALPACA_DENIED_LOGGED: bool = False
_ALPACA_DENIED_TOKENS = (
    "subscription does not permit",
    "not permitted",
    "sip data",
)


@dataclass
class SessionArchiveLoad:
    bars_by_symbol: Dict[str, pd.DataFrame]
    manifest_by_symbol: Dict[str, Dict[str, Any]]
    archive_exists: bool


def resolve_archive_path(path_value: str, *, project_dir: Path) -> Path:
    path = Path(str(path_value or "").strip() or "data/replay_minute_bars.duckdb")
    if not path.is_absolute():
        path = project_dir / path
    return path.resolve()


def _parse_et_time(value: str, *, fallback: time) -> time:
    text = str(value or "").strip()
    if not text:
        return fallback
    try:
        hour_text, minute_text = text.split(":", 1)
        return time(hour=int(hour_text), minute=int(minute_text))
    except Exception:
        return fallback


def normalize_bars_frame(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    if df.empty:
        return df
    index = pd.DatetimeIndex(df.index)
    if index.tz is None:
        index = index.tz_localize(UTC)
    df.index = index.tz_convert(CT)
    return df.sort_index()


def _ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS session_minute_bars (
            session_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timestamp_utc TEXT NOT NULL,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            trade_count BIGINT,
            vwap DOUBLE,
            source_window TEXT,
            archived_at_utc TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS session_minute_bar_manifest (
            session_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            capture_status TEXT,
            capture_window_start_et TEXT,
            capture_window_end_et TEXT,
            requested_start_et TEXT,
            requested_end_et TEXT,
            has_premarket_data BOOLEAN,
            used_regular_session_only BOOLEAN,
            first_bar_ts_utc TEXT,
            last_bar_ts_utc TEXT,
            bar_count BIGINT,
            source_tags_json TEXT,
            notes TEXT,
            archived_at_utc TEXT
        )
        """
    )


def _archive_exists(db_path: Path) -> bool:
    return db_path.exists() and db_path.is_file()


def _coerce_symbol_rows(
    symbol_sources: Mapping[str, Iterable[str]],
) -> Dict[str, List[str]]:
    rows: Dict[str, List[str]] = {}
    for raw_symbol, raw_tags in (symbol_sources or {}).items():
        symbol = str(raw_symbol or "").upper().strip()
        if not symbol:
            continue
        cleaned = sorted(
            {
                str(tag).strip()
                for tag in (raw_tags or [])
                if str(tag or "").strip()
            }
        )
        rows[symbol] = cleaned
    return rows


def fetch_symbol_minute_bars(
    client: Any,
    *,
    session_date: str,
    symbol: str,
    start_et: time,
    end_et: time,
) -> pd.DataFrame:
    day_value = date.fromisoformat(session_date)
    start_utc = datetime.combine(day_value, start_et, tzinfo=ET).astimezone(UTC)
    end_utc = datetime.combine(day_value, end_et, tzinfo=ET).astimezone(UTC)
    
    # Try Alpaca first -- but skip once we know the subscription tier
    # doesn't permit recent SIP minute data (basic/IEX-only plans).
    global _ALPACA_RECENT_DATA_DENIED, _ALPACA_DENIED_LOGGED
    if not _ALPACA_RECENT_DATA_DENIED:
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=start_utc,
                end=end_utc,
            )
            bars = client.get_stock_bars(request)
            if hasattr(bars, "df") and not bars.df.empty:
                df = bars.df
                if isinstance(df.index, pd.MultiIndex):
                    if symbol in df.index.get_level_values(0):
                        df = df.loc[symbol]
                    else:
                        df = pd.DataFrame()
                if not df.empty:
                    return normalize_bars_frame(df)
        except Exception as e:
            err_text = str(e).lower()
            if any(tok in err_text for tok in _ALPACA_DENIED_TOKENS):
                _ALPACA_RECENT_DATA_DENIED = True
                if not _ALPACA_DENIED_LOGGED:
                    logger.info(
                        "Alpaca subscription does not permit recent SIP minute data; "
                        "switching to yfinance for the remainder of this session."
                    )
                    _ALPACA_DENIED_LOGGED = True
            else:
                logger.warning(
                    f"Alpaca fetch failed for {symbol} on {session_date}: {e}"
                )

    # Fallback to yfinance if Alpaca fails or returns no data
    try:
        import yfinance as yf
        
        # yfinance download for the specific day
        # We fetch slightly more to be safe, then filter
        download_start = day_value.isoformat()
        download_end = (day_value + pd.Timedelta(days=1)).isoformat()
        
        yf_ticker = yf.Ticker(symbol)
        df = yf_ticker.history(start=download_start, end=download_end, interval="1m")
        
        if df is not None and not df.empty:
            # Normalize columns
            df.columns = [c.lower() for c in df.columns]
            # Map columns to match archive schema if needed
            # (archive expects: open, high, low, close, volume, trade_count, vwap)
            if "trade_count" not in df.columns and "trades" in df.columns:
                df = df.rename(columns={"trades": "trade_count"})
            
            # yfinance index is already DatetimeIndex, usually localized or UTC
            if df.index.tz is None:
                df.index = df.index.tz_localize(ET)
            
            # Filter to requested window (ET)
            mask = (df.index.tz_convert(ET).time >= start_et) & (df.index.tz_convert(ET).time <= end_et)
            df = df.loc[mask]
            
            if not df.empty:
                # Demoted to debug: when Alpaca tier denies recent SIP data,
                # yfinance is the expected primary path -- not a fallback worth
                # logging per symbol every cycle.
                logger.debug(
                    f"yfinance fallback success for {symbol} on {session_date} ({len(df)} bars)"
                )
                # normalize_bars_frame expects UTC or local, converts to CT
                return normalize_bars_frame(df)
    except ImportError:
        logger.debug("yfinance not installed, cannot use fallback")
    except Exception as e:
        logger.warning(f"yfinance fallback failed for {symbol} on {session_date}: {e}")

    return pd.DataFrame()


def archive_session_minute_bars(
    *,
    db_path: Path,
    session_date: str,
    client: Any,
    symbol_sources: Mapping[str, Iterable[str]],
    preferred_start_et: time,
    preferred_end_et: time,
    fallback_regular_session_only: bool,
    replace_existing_symbols_only: bool = False,
) -> Dict[str, Any]:
    symbols = _coerce_symbol_rows(symbol_sources)
    archived_at = datetime.now(tz=UTC).isoformat()
    status_counts = {"archived": 0, "archived_regular_session_only": 0, "missing": 0}
    manifest_rows: List[Dict[str, Any]] = []
    bar_frames: List[pd.DataFrame] = []
    missing_symbols: List[str] = []
    premarket_missing_symbols: List[str] = []

    for symbol, source_tags in sorted(symbols.items()):
        requested_start_text = preferred_start_et.strftime("%H:%M")
        requested_end_text = preferred_end_et.strftime("%H:%M")
        used_start = preferred_start_et
        used_end = preferred_end_et
        notes: List[str] = []
        capture_status = "archived"
        has_premarket_data = False
        used_regular_only = False

        df = fetch_symbol_minute_bars(
            client,
            session_date=session_date,
            symbol=symbol,
            start_et=preferred_start_et,
            end_et=preferred_end_et,
        )
        if df.empty and fallback_regular_session_only:
            df = fetch_symbol_minute_bars(
                client,
                session_date=session_date,
                symbol=symbol,
                start_et=REGULAR_START_ET,
                end_et=REGULAR_END_ET,
            )
            if not df.empty:
                used_start = REGULAR_START_ET
                used_end = REGULAR_END_ET
                used_regular_only = True
                capture_status = "archived_regular_session_only"
                notes.append("premarket_unavailable_regular_session_only")
        if not df.empty:
            has_premarket_data = bool(
                any(ts.tz_convert(ET).time() < REGULAR_START_ET for ts in pd.DatetimeIndex(df.index))
            )
            if not has_premarket_data:
                premarket_missing_symbols.append(symbol)
            export = df.reset_index(drop=True).copy()
            export["timestamp"] = pd.DatetimeIndex(df.index)
            export["session_date"] = session_date
            export["symbol"] = symbol
            export["timestamp_utc"] = (
                pd.to_datetime(export["timestamp"], utc=True)
                .dt.tz_convert(UTC)
                .dt.strftime("%Y-%m-%dT%H:%M:%S%z")
            )
            export["source_window"] = f"{used_start.strftime('%H:%M')}-{used_end.strftime('%H:%M')}"
            export["archived_at_utc"] = archived_at
            keep_columns = [
                "session_date",
                "symbol",
                "timestamp_utc",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "trade_count",
                "vwap",
                "source_window",
                "archived_at_utc",
            ]
            for optional in ("trade_count", "vwap"):
                if optional not in export.columns:
                    export[optional] = None
            bar_frames.append(export[keep_columns])
            status_counts[capture_status] += 1
        else:
            capture_status = "missing"
            missing_symbols.append(symbol)
            notes.append("no_minute_bars_available")
            status_counts["missing"] += 1

        manifest_rows.append(
            {
                "session_date": session_date,
                "symbol": symbol,
                "capture_status": capture_status,
                "capture_window_start_et": used_start.strftime("%H:%M"),
                "capture_window_end_et": used_end.strftime("%H:%M"),
                "requested_start_et": requested_start_text,
                "requested_end_et": requested_end_text,
                "has_premarket_data": bool(has_premarket_data),
                "used_regular_session_only": bool(used_regular_only),
                "first_bar_ts_utc": (
                    pd.DatetimeIndex(df.index)[0].tz_convert(UTC).isoformat() if not df.empty else ""
                ),
                "last_bar_ts_utc": (
                    pd.DatetimeIndex(df.index)[-1].tz_convert(UTC).isoformat() if not df.empty else ""
                ),
                "bar_count": int(len(df.index)) if not df.empty else 0,
                "source_tags_json": json.dumps(source_tags),
                "notes": ";".join(notes),
                "archived_at_utc": archived_at,
            }
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as conn:
        _ensure_schema(conn)
        if replace_existing_symbols_only and symbols:
            delete_symbols = list(sorted(symbols.keys()))
            placeholders = ", ".join(["?"] * len(delete_symbols))
            conn.execute(
                f"DELETE FROM session_minute_bars WHERE session_date = ? AND symbol IN ({placeholders})",
                [session_date, *delete_symbols],
            )
            conn.execute(
                f"DELETE FROM session_minute_bar_manifest WHERE session_date = ? AND symbol IN ({placeholders})",
                [session_date, *delete_symbols],
            )
        else:
            conn.execute("DELETE FROM session_minute_bars WHERE session_date = ?", [session_date])
            conn.execute(
                "DELETE FROM session_minute_bar_manifest WHERE session_date = ?",
                [session_date],
            )
        manifest_df = pd.DataFrame(manifest_rows)
        if not manifest_df.empty:
            conn.register("manifest_df", manifest_df)
            conn.execute("INSERT INTO session_minute_bar_manifest SELECT * FROM manifest_df")
            conn.unregister("manifest_df")
        if bar_frames:
            bars_df = pd.concat(bar_frames, ignore_index=True)
            conn.register("bars_df", bars_df)
            conn.execute("INSERT INTO session_minute_bars SELECT * FROM bars_df")
            conn.unregister("bars_df")

    total_symbols = len(symbols)
    return {
        "session_date": session_date,
        "db_path": str(db_path),
        "total_symbols_requested": total_symbols,
        "archived_symbols": int(status_counts["archived"] + status_counts["archived_regular_session_only"]),
        "missing_symbols_count": int(status_counts["missing"]),
        "premarket_covered_symbols": int(
            sum(1 for row in manifest_rows if bool(row.get("has_premarket_data")))
        ),
        "regular_session_only_symbols": int(
            sum(1 for row in manifest_rows if bool(row.get("used_regular_session_only")))
        ),
        "missing_symbols": missing_symbols[:50],
        "premarket_missing_symbols": sorted(set(premarket_missing_symbols))[:50],
        "status_counts": status_counts,
        "manifest_rows": manifest_rows,
    }


def load_session_archive(
    *,
    db_path: Path,
    session_date: str,
    symbols: Sequence[str],
) -> SessionArchiveLoad:
    requested_symbols = sorted(
        {
            str(symbol or "").upper().strip()
            for symbol in (symbols or [])
            if str(symbol or "").strip()
        }
    )
    if not _archive_exists(db_path):
        return SessionArchiveLoad({}, {}, False)
    if not requested_symbols:
        return SessionArchiveLoad({}, {}, True)

    placeholders = ", ".join(["?"] * len(requested_symbols))
    bars_query = f"""
        SELECT *
        FROM session_minute_bars
        WHERE session_date = ?
          AND symbol IN ({placeholders})
        ORDER BY symbol, timestamp_utc
    """
    manifest_query = f"""
        SELECT *
        FROM session_minute_bar_manifest
        WHERE session_date = ?
          AND symbol IN ({placeholders})
    """
    params = [session_date, *requested_symbols]
    conn = None
    retry_attempts = 5
    while retry_attempts > 0:
        try:
            conn = duckdb.connect(str(db_path))
            break
        except IOError:
            retry_attempts -= 1
            if retry_attempts <= 0:
                raise
            time_module.sleep(2)
    try:
        _ensure_schema(conn)
        bars_df = conn.execute(bars_query, params).fetchdf()
        manifest_df = conn.execute(manifest_query, params).fetchdf()
    finally:
        conn.close()

    manifest_by_symbol: Dict[str, Dict[str, Any]] = {}
    if not manifest_df.empty:
        for row in manifest_df.to_dict(orient="records"):
            symbol = str(row.get("symbol") or "").upper().strip()
            if not symbol:
                continue
            row["has_premarket_data"] = bool(row.get("has_premarket_data"))
            row["used_regular_session_only"] = bool(row.get("used_regular_session_only"))
            manifest_by_symbol[symbol] = row

    bars_by_symbol: Dict[str, pd.DataFrame] = {}
    if not bars_df.empty:
        bars_df["timestamp_utc"] = pd.to_datetime(bars_df["timestamp_utc"], utc=True)
        for symbol, group in bars_df.groupby("symbol"):
            group = group.sort_values("timestamp_utc").copy()
            group.index = pd.DatetimeIndex(group["timestamp_utc"]).tz_convert(CT)
            frame = group.drop(
                columns=[
                    "session_date",
                    "symbol",
                    "timestamp_utc",
                    "source_window",
                    "archived_at_utc",
                ],
                errors="ignore",
            )
            frame = frame.dropna(axis=1, how="all")
            bars_by_symbol[str(symbol).upper()] = frame

    return SessionArchiveLoad(
        bars_by_symbol=bars_by_symbol,
        manifest_by_symbol=manifest_by_symbol,
        archive_exists=True,
    )


def archive_summary_payload(
    *,
    capture_result: Mapping[str, Any],
    source_map: Mapping[str, Iterable[str]],
) -> Dict[str, Any]:
    rows = _coerce_symbol_rows(source_map)
    return {
        "session_date": str(capture_result.get("session_date") or ""),
        "db_path": str(capture_result.get("db_path") or ""),
        "total_symbols_requested": int(capture_result.get("total_symbols_requested", 0) or 0),
        "archived_symbols": int(capture_result.get("archived_symbols", 0) or 0),
        "missing_symbols_count": int(capture_result.get("missing_symbols_count", 0) or 0),
        "premarket_covered_symbols": int(
            capture_result.get("premarket_covered_symbols", 0) or 0
        ),
        "regular_session_only_symbols": int(
            capture_result.get("regular_session_only_symbols", 0) or 0
        ),
        "status_counts": dict(capture_result.get("status_counts") or {}),
        "source_map_size": len(rows),
        "sample_symbols": [
            {"symbol": symbol, "source_tags": tags}
            for symbol, tags in list(sorted(rows.items()))[:15]
        ],
        "missing_symbols": list(capture_result.get("missing_symbols") or []),
        "premarket_missing_symbols": list(
            capture_result.get("premarket_missing_symbols") or []
        ),
    }
