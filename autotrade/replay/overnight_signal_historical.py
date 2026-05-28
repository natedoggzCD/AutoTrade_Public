from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import duckdb
import pandas as pd
import yfinance as yf

from autotrade.signals.overnight_model_fit import build_historical_edge_context


PROJECT_DIR = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_DIR / "logs"
PLANS_DIR = PROJECT_DIR / "plans"
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_BARS_DB = DATA_DIR / "overnight_daily_bars.duckdb"
LOCAL_DAILY_FEATURE_CANDIDATES = (
    DATA_DIR / "daily_features.parquet",
    DATA_DIR / "downday" / "daily_features.parquet",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def trading_dates_from_logs(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> List[date]:
    dates: List[date] = []
    for path in sorted(LOGS_DIR.glob("signals_*.json")):
        try:
            session_date = datetime.fromisoformat(path.stem.split("_", 1)[1]).date()
        except Exception:
            continue
        if start_date and session_date < start_date:
            continue
        if end_date and session_date > end_date:
            continue
        dates.append(session_date)
    return dates


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _local_daily_features_path() -> Optional[Path]:
    for path in LOCAL_DAILY_FEATURE_CANDIDATES:
        if path.exists():
            return path
    return None


def _load_plan_payload(session_date: date) -> Optional[Dict[str, Any]]:
    compact = session_date.strftime("%Y%m%d")
    candidates = sorted(
        PLANS_DIR.glob(f"*{compact}*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        if not any(
            prefix in path.name
            for prefix in ("morning_game_plan", "pm_plan", "adjusted_plan")
        ):
            continue
        try:
            payload = _load_json(path)
        except Exception:
            continue
        if isinstance(payload, dict):
            payload["_source_path"] = str(path)
            return payload
    return None


def load_candidate_pool_for_date(
    session_date: date,
    *,
    prefer_full_watchlist: bool = True,
) -> List[Dict[str, Any]]:
    payload = _load_plan_payload(session_date)
    rows: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        source_keys: Sequence[str] = (
            ("full_watchlist", "actionable_top50", "signals", "buy_signals")
            if prefer_full_watchlist
            else ("signals", "actionable_top50", "full_watchlist", "buy_signals")
        )
        for key in source_keys:
            value = payload.get(key)
            if isinstance(value, list) and value:
                rows = [dict(row) for row in value if isinstance(row, dict)]
                break
    if rows:
        return rows

    log_path = LOGS_DIR / f"signals_{session_date.isoformat()}.json"
    if log_path.exists():
        try:
            payload = _load_json(log_path)
        except Exception:
            return []
        signals = payload.get("signals") if isinstance(payload, dict) else payload
        if isinstance(signals, list):
            return [dict(row) for row in signals if isinstance(row, dict)]
    return []


def load_signal_rows(
    start_date: date,
    end_date: date,
    *,
    prefer_full_watchlist: bool = True,
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    current = start_date
    while current <= end_date:
        rows = load_candidate_pool_for_date(
            current, prefer_full_watchlist=prefer_full_watchlist
        )
        for rank, row in enumerate(rows, start=1):
            ticker = str(row.get("symbol") or row.get("ticker") or "").upper().strip()
            if not ticker or ticker in {"SPY", "QQQ"}:
                continue
            item = dict(row)
            item["sig_date"] = current.isoformat()
            item["ticker"] = ticker
            item["symbol"] = ticker
            item["rank"] = rank
            records.append(item)
        current += timedelta(days=1)
    return pd.DataFrame(records)


def _ensure_db(db_path: Path) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_bars (
            ticker VARCHAR,
            trade_date DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            source VARCHAR,
            fetched_at TIMESTAMP
        )
        """
    )
    return conn


def _existing_pairs(
    conn: duckdb.DuckDBPyConnection,
    tickers: Sequence[str],
    start_date: date,
    end_date: date,
) -> set[tuple[str, str]]:
    if not tickers:
        return set()
    ticker_literals = ", ".join(
        "'" + str(t).replace("'", "''") + "'" for t in sorted(set(tickers))
    )
    query = f"""
    SELECT ticker, CAST(trade_date AS VARCHAR) AS trade_date
    FROM daily_bars
    WHERE ticker IN ({ticker_literals})
      AND trade_date BETWEEN DATE '{start_date.isoformat()}' AND DATE '{end_date.isoformat()}'
    """
    return {(str(row[0]), str(row[1])) for row in conn.execute(query).fetchall()}


def _download_daily_bars(
    tickers: Sequence[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    unique_tickers = sorted({str(t).upper().strip() for t in tickers if str(t).strip()})
    if not unique_tickers:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    batch_size = 40
    fetch_end = end_date + timedelta(days=1)
    for idx in range(0, len(unique_tickers), batch_size):
        batch = unique_tickers[idx : idx + batch_size]
        data = yf.download(
            tickers=" ".join(batch),
            start=start_date.isoformat(),
            end=fetch_end.isoformat(),
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if data is None or data.empty:
            continue
        if isinstance(data.columns, pd.MultiIndex):
            for ticker in batch:
                if ticker not in data.columns.get_level_values(0):
                    continue
                frame = data[ticker].copy()
                if frame.empty:
                    continue
                frame = frame.reset_index()
                frame["ticker"] = ticker
                frames.append(frame)
        else:
            frame = data.reset_index()
            frame["ticker"] = batch[0]
            frames.append(frame)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    rename_map = {
        "Date": "trade_date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    combined = combined.rename(columns=rename_map)
    keep_cols = ["ticker", "trade_date", "open", "high", "low", "close", "volume"]
    combined = combined[[col for col in keep_cols if col in combined.columns]].copy()
    combined["trade_date"] = pd.to_datetime(combined["trade_date"]).dt.date
    combined["source"] = "yfinance"
    combined["fetched_at"] = datetime.now()
    return combined


def _load_daily_bars_from_local_parquet(
    tickers: Sequence[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    parquet_path = _local_daily_features_path()
    unique_tickers = sorted({str(t).upper().strip() for t in tickers if str(t).strip()})
    if parquet_path is None or not unique_tickers:
        return pd.DataFrame()
    ticker_literals = ", ".join(
        "'" + ticker.replace("'", "''") + "'" for ticker in unique_tickers
    )
    conn = duckdb.connect()
    try:
        query = f"""
        SELECT
            UPPER(CAST(ticker AS VARCHAR)) AS ticker,
            CAST(Date AS DATE) AS trade_date,
            CAST(Open AS DOUBLE) AS open,
            CAST(High AS DOUBLE) AS high,
            CAST(Low AS DOUBLE) AS low,
            CAST(Close AS DOUBLE) AS close,
            CAST(Volume AS DOUBLE) AS volume,
            'local_parquet' AS source,
            CURRENT_TIMESTAMP AS fetched_at
        FROM read_parquet('{parquet_path.as_posix()}')
        WHERE UPPER(CAST(ticker AS VARCHAR)) IN ({ticker_literals})
          AND CAST(Date AS DATE) BETWEEN DATE '{start_date.isoformat()}' AND DATE '{end_date.isoformat()}'
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY UPPER(CAST(ticker AS VARCHAR)), CAST(Date AS DATE)
            ORDER BY CAST(Date AS TIMESTAMP) DESC
        ) = 1
        """
        return conn.execute(query).df()
    finally:
        conn.close()


def _upsert_daily_bars(conn: duckdb.DuckDBPyConnection, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    conn.register("daily_bars_stage_df", frame)
    conn.execute(
        """
        DELETE FROM daily_bars
        USING daily_bars_stage_df d
        WHERE daily_bars.ticker = d.ticker
          AND daily_bars.trade_date = d.trade_date
        """
    )
    conn.execute(
        """
        INSERT INTO daily_bars
        SELECT ticker, trade_date, open, high, low, close, volume, source, fetched_at
        FROM daily_bars_stage_df
        """
    )


def ensure_daily_bars_cached(
    tickers: Sequence[str],
    start_date: date,
    end_date: date,
    *,
    db_path: Path = DEFAULT_BARS_DB,
    fetch_missing: bool = False,
) -> None:
    conn = _ensure_db(db_path)
    try:
        existing = _existing_pairs(conn, tickers, start_date, end_date)
        expected_pairs = {
            (ticker, session_date.isoformat())
            for ticker in sorted(
                {str(t).upper().strip() for t in tickers if str(t).strip()}
            )
            for session_date in pd.bdate_range(start_date, end_date).date
        }
        if expected_pairs.issubset(existing):
            return
        missing_tickers = sorted({ticker for ticker, _ in (expected_pairs - existing)})

        local_rows = _load_daily_bars_from_local_parquet(
            missing_tickers, start_date, end_date
        )
        if not local_rows.empty:
            _upsert_daily_bars(conn, local_rows)
            existing = _existing_pairs(conn, tickers, start_date, end_date)
            if expected_pairs.issubset(existing):
                return

        if not fetch_missing:
            return
        missing_tickers = sorted({ticker for ticker, _ in (expected_pairs - existing)})
        downloaded = _download_daily_bars(missing_tickers, start_date, end_date)
        if downloaded.empty:
            return
        _upsert_daily_bars(conn, downloaded)
    finally:
        conn.close()


def compute_outcomes_from_local_bars(
    signals_df: pd.DataFrame,
    *,
    db_path: Path = DEFAULT_BARS_DB,
    fetch_missing: bool = False,
) -> pd.DataFrame:
    if signals_df.empty:
        return signals_df.copy()
    tickers = sorted({str(x).upper() for x in signals_df["ticker"].dropna().tolist()})
    start_date = pd.to_datetime(signals_df["sig_date"]).dt.date.min()
    end_date = pd.to_datetime(signals_df["sig_date"]).dt.date.max()
    ensure_daily_bars_cached(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        db_path=db_path,
        fetch_missing=fetch_missing,
    )

    conn = _ensure_db(db_path)
    try:
        key_df = signals_df.reset_index(drop=True).copy()
        if "_row_id" in key_df.columns:
            key_df = key_df.drop(columns=["_row_id"]).copy()
        key_df.insert(len(key_df.columns), "_row_id", range(len(key_df)))
        key_df["ticker"] = key_df["ticker"].astype(str)
        key_df["sig_date"] = key_df["sig_date"].astype(str)
        conn.register("signal_keys_df", key_df[["_row_id", "ticker", "sig_date"]])
        outcome_df = conn.execute(
            """
            SELECT
                s._row_id,
                b.open,
                b.high,
                b.low,
                b.close,
                b.volume,
                CASE WHEN b.open IS NOT NULL AND b.open != 0 THEN ((b.high - b.open) / b.open) * 100 ELSE NULL END AS open_to_high_pct,
                CASE WHEN b.open IS NOT NULL AND b.open != 0 THEN ((b.close - b.open) / b.open) * 100 ELSE NULL END AS open_to_close_pct,
                CASE WHEN b.open IS NOT NULL AND b.open != 0 THEN ((b.low - b.open) / b.open) * 100 ELSE NULL END AS open_to_low_pct
            FROM signal_keys_df s
            LEFT JOIN daily_bars b
              ON UPPER(CAST(s.ticker AS VARCHAR)) = UPPER(CAST(b.ticker AS VARCHAR))
             AND CAST(s.sig_date AS DATE) = b.trade_date
            ORDER BY s._row_id
            """
        ).df()
    finally:
        conn.close()
    merged = key_df.merge(outcome_df, on="_row_id", how="left")
    return merged.drop(columns=["_row_id"])


def apply_outcome_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    derived_columns = [
        "profit_proxy",
        "fade_span_pct",
        "hit_2pct",
        "positive_close",
        "bad_close",
        "trap_risk",
        "fade_risk",
        "stickiness_score",
    ]
    enriched = df.drop(columns=derived_columns, errors="ignore").copy()
    open_to_high = pd.to_numeric(enriched["open_to_high_pct"], errors="coerce")
    open_to_close = pd.to_numeric(enriched["open_to_close_pct"], errors="coerce")
    fade_span_pct = open_to_high - open_to_close
    positive_retention = open_to_close.clip(lower=0.0)

    derived = pd.DataFrame(
        {
            "profit_proxy": open_to_high * 0.65 + open_to_close * 0.35,
            "fade_span_pct": fade_span_pct,
            "hit_2pct": open_to_high >= 2.0,
            "positive_close": open_to_close > 0.0,
            "bad_close": open_to_close <= -2.0,
            "trap_risk": (
                (open_to_close <= -0.75)
                | (
                    (open_to_high >= 2.0)
                    & (fade_span_pct >= 2.25)
                    & (open_to_close <= 0.15)
                )
            ).fillna(False),
            "fade_risk": (
                (open_to_high >= 2.5) & (fade_span_pct >= 1.75) & (open_to_close < 0.4)
            ).fillna(False),
            "stickiness_score": (positive_retention / open_to_high.clip(lower=0.75))
            .clip(lower=0.0, upper=1.0)
            .fillna(0.0),
        }
    )
    return pd.concat([enriched.reset_index(drop=True), derived], axis=1)


def training_rows_before_date(
    session_date: date,
    *,
    prefer_full_watchlist: bool = True,
    min_start_date: Optional[date] = None,
    db_path: Path = DEFAULT_BARS_DB,
    fetch_missing: bool = False,
) -> pd.DataFrame:
    all_dates = [
        d
        for d in trading_dates_from_logs(
            min_start_date, session_date - timedelta(days=1)
        )
    ]
    if not all_dates:
        return pd.DataFrame()
    signals_df = load_signal_rows(
        start_date=all_dates[0],
        end_date=all_dates[-1],
        prefer_full_watchlist=prefer_full_watchlist,
    )
    outcomes = compute_outcomes_from_local_bars(
        signals_df,
        db_path=db_path,
        fetch_missing=fetch_missing,
    )
    return apply_outcome_metrics(outcomes)


def historical_edge_context_before_date(
    session_date: date,
    *,
    prefer_full_watchlist: bool = True,
    min_start_date: Optional[date] = None,
    db_path: Path = DEFAULT_BARS_DB,
    fetch_missing: bool = False,
) -> Dict[str, Any]:
    rows_df = training_rows_before_date(
        session_date,
        prefer_full_watchlist=prefer_full_watchlist,
        min_start_date=min_start_date,
        db_path=db_path,
        fetch_missing=fetch_missing,
    )
    return build_historical_edge_context(rows_df)
