"""
Financial Database — SQLite CRUD helpers for all financial data.

Database: data/financial.db
Tables: earnings_calendar, financial_events, analyst_ratings, key_stats,
        financial_statements, dividend_history, valuation_metrics, options_summary,
        balance_sheet_annual, balance_sheet_quarterly,
        inverse_etf_universe, inverse_etf_screens

Usage:
    from autotrade.utils.financial_db import FinancialDB
    db = FinancialDB()
    upcoming = db.get_upcoming_earnings(days=5)
    bs_annual = db.get_balance_sheet('AAPL', 'annual')
    bs_quarterly = db.get_balance_sheet('AAPL', 'quarterly')
    divs = db.get_dividends('AAPL')
"""

import sqlite3
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "data" / "financial.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS earnings_calendar (
    ticker TEXT NOT NULL,
    earnings_date DATE NOT NULL,
    time_of_day TEXT DEFAULT 'TBD',
    eps_estimate REAL,
    eps_actual REAL,
    revenue_estimate REAL,
    revenue_actual REAL,
    surprise_pct REAL,
    updated_at TIMESTAMP,
    PRIMARY KEY (ticker, earnings_date)
);

CREATE TABLE IF NOT EXISTS financial_events (
    ticker TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_date DATE NOT NULL,
    details TEXT,
    updated_at TIMESTAMP,
    PRIMARY KEY (ticker, event_type, event_date)
);

CREATE TABLE IF NOT EXISTS analyst_ratings (
    ticker TEXT NOT NULL,
    firm TEXT NOT NULL,
    rating TEXT,
    price_target REAL,
    date DATE NOT NULL,
    updated_at TIMESTAMP,
    PRIMARY KEY (ticker, firm, date)
);

CREATE TABLE IF NOT EXISTS key_stats (
    ticker TEXT NOT NULL PRIMARY KEY,
    market_cap REAL,
    pe_ratio REAL,
    forward_pe REAL,
    peg_ratio REAL,
    beta REAL,
    short_pct REAL,
    inst_ownership_pct REAL,
    updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS financial_statements (
    ticker TEXT NOT NULL,
    statement_type TEXT NOT NULL,  -- 'balance_sheet', 'income_statement', 'cash_flow'
    frequency TEXT NOT NULL,       -- 'annual', 'quarterly'
    metric TEXT NOT NULL,
    period_end DATE NOT NULL,
    value REAL,
    updated_at TIMESTAMP,
    PRIMARY KEY (ticker, statement_type, frequency, metric, period_end)
);

CREATE TABLE IF NOT EXISTS dividend_history (
    ticker TEXT NOT NULL,
    ex_date DATE NOT NULL,
    amount REAL,
    updated_at TIMESTAMP,
    PRIMARY KEY (ticker, ex_date)
);

CREATE TABLE IF NOT EXISTS valuation_metrics (
    ticker TEXT NOT NULL PRIMARY KEY,
    trailing_pe REAL,
    forward_pe REAL,
    peg_ratio REAL,
    price_to_book REAL,
    price_to_sales REAL,
    ev_to_revenue REAL,
    ev_to_ebitda REAL,
    profit_margins REAL,
    operating_margins REAL,
    gross_margins REAL,
    return_on_equity REAL,
    return_on_assets REAL,
    debt_to_equity REAL,
    current_ratio REAL,
    quick_ratio REAL,
    revenue_per_share REAL,
    earnings_growth REAL,
    revenue_growth REAL,
    updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS options_summary (
    ticker TEXT NOT NULL,
    expiration DATE NOT NULL,
    calls_oi INTEGER,
    puts_oi INTEGER,
    calls_volume INTEGER,
    puts_volume INTEGER,
    calls_avg_iv REAL,
    puts_avg_iv REAL,
    put_call_oi_ratio REAL,
    updated_at TIMESTAMP,
    PRIMARY KEY (ticker, expiration)
);

CREATE TABLE IF NOT EXISTS inverse_etf_universe (
    ticker TEXT NOT NULL PRIMARY KEY,
    name TEXT NOT NULL,
    leverage INTEGER NOT NULL DEFAULT 1,
    underlying TEXT NOT NULL,
    category TEXT NOT NULL,
    provider TEXT DEFAULT '',
    expense_ratio REAL DEFAULT 0.0,
    avg_daily_volume INTEGER DEFAULT 0,
    aum_millions REAL DEFAULT 0.0,
    is_active INTEGER DEFAULT 1,
    last_data_check TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS inverse_etf_screens (
    ticker TEXT NOT NULL,
    screen_date DATE NOT NULL,
    screen_time TEXT NOT NULL,
    volume_ratio REAL DEFAULT 0.0,
    momentum_score REAL DEFAULT 0.0,
    vwap_distance_pct REAL DEFAULT 0.0,
    rsi_14 REAL DEFAULT 0.0,
    spread_bps REAL DEFAULT 0.0,
    signal TEXT DEFAULT 'NEUTRAL',
    entry_price REAL DEFAULT 0.0,
    notes TEXT DEFAULT '',
    updated_at TIMESTAMP,
    PRIMARY KEY (ticker, screen_date, screen_time)
);
"""


class FinancialDB:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA_SQL)

    # ── Earnings Calendar ─────────────────────────────────────

    def upsert_earnings(self, rows: List[Dict]):
        if not rows:
            return
        with self._conn() as conn:
            conn.executemany(
                """INSERT INTO earnings_calendar
                   (ticker, earnings_date, time_of_day, eps_estimate, eps_actual,
                    revenue_estimate, revenue_actual, surprise_pct, updated_at)
                   VALUES (:ticker, :earnings_date, :time_of_day, :eps_estimate,
                           :eps_actual, :revenue_estimate, :revenue_actual,
                           :surprise_pct, :updated_at)
                   ON CONFLICT(ticker, earnings_date) DO UPDATE SET
                     time_of_day=excluded.time_of_day,
                     eps_estimate=excluded.eps_estimate,
                     eps_actual=excluded.eps_actual,
                     revenue_estimate=excluded.revenue_estimate,
                     revenue_actual=excluded.revenue_actual,
                     surprise_pct=excluded.surprise_pct,
                     updated_at=excluded.updated_at""",
                rows,
            )

    def get_upcoming_earnings(self, days: int = 5, tickers: Optional[List[str]] = None) -> List[Dict]:
        today = date.today().isoformat()
        end = (date.today() + timedelta(days=days)).isoformat()
        sql = "SELECT * FROM earnings_calendar WHERE earnings_date BETWEEN ? AND ?"
        params: list = [today, end]
        if tickers:
            placeholders = ",".join("?" for _ in tickers)
            sql += f" AND ticker IN ({placeholders})"
            params.extend([t.upper() for t in tickers])
        sql += " ORDER BY earnings_date, ticker"
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def get_recent_surprises(self, days: int = 5, min_surprise: float = 5.0) -> List[Dict]:
        """Find tickers with recent positive earnings surprises."""
        start = (date.today() - timedelta(days=days)).isoformat()
        today = date.today().isoformat()
        sql = """SELECT * FROM earnings_calendar 
                 WHERE earnings_date BETWEEN ? AND ? 
                 AND surprise_pct >= ? 
                 ORDER BY earnings_date DESC"""
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, [start, today, min_surprise]).fetchall()]

    def has_earnings_soon(self, ticker: str, days: int = 2) -> bool:
        rows = self.get_upcoming_earnings(days=days, tickers=[ticker])
        return len(rows) > 0

    def get_last_update(self, ticker: str) -> Optional[datetime]:
        """Get the last update timestamp for a ticker (from key_stats)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT updated_at FROM key_stats WHERE ticker = ?", (ticker,)
            ).fetchone()
            if row and row["updated_at"]:
                try:
                    return datetime.fromisoformat(str(row["updated_at"]))
                except (ValueError, TypeError):
                    pass
        return None

    # ── Financial Events ──────────────────────────────────────

    def upsert_events(self, rows: List[Dict]):
        if not rows:
            return
        with self._conn() as conn:
            conn.executemany(
                """INSERT INTO financial_events
                   (ticker, event_type, event_date, details, updated_at)
                   VALUES (:ticker, :event_type, :event_date, :details, :updated_at)
                   ON CONFLICT(ticker, event_type, event_date) DO UPDATE SET
                     details=excluded.details, updated_at=excluded.updated_at""",
                rows,
            )

    def get_events(self, ticker: str, event_type: Optional[str] = None, limit: int = 20) -> List[Dict]:
        sql = "SELECT * FROM financial_events WHERE ticker = ?"
        params: list = [ticker.upper()]
        if event_type:
            sql += " AND event_type = ?"
            params.append(event_type)
        sql += " ORDER BY event_date DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    # ── Analyst Ratings ───────────────────────────────────────

    def upsert_ratings(self, rows: List[Dict]):
        if not rows:
            return
        with self._conn() as conn:
            conn.executemany(
                """INSERT INTO analyst_ratings
                   (ticker, firm, rating, price_target, date, updated_at)
                   VALUES (:ticker, :firm, :rating, :price_target, :date, :updated_at)
                   ON CONFLICT (ticker, firm, date) DO UPDATE SET
                       rating = EXCLUDED.rating,
                       price_target = EXCLUDED.price_target,
                       updated_at = EXCLUDED.updated_at""",
                rows,
            )

    def get_ratings(self, ticker: str, limit: int = 10) -> List[Dict]:
        with self._conn() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM analyst_ratings WHERE ticker = ? ORDER BY date DESC LIMIT ?",
                    [ticker.upper(), limit],
                ).fetchall()
            ]

    # ── Key Stats ─────────────────────────────────────────────

    def upsert_key_stats(self, row: Dict):
        if not row:
            return
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO key_stats
                   (ticker, market_cap, pe_ratio, forward_pe, peg_ratio,
                    beta, short_pct, inst_ownership_pct, updated_at)
                   VALUES (:ticker, :market_cap, :pe_ratio, :forward_pe, :peg_ratio,
                           :beta, :short_pct, :inst_ownership_pct, :updated_at)
                   ON CONFLICT(ticker) DO UPDATE SET
                     market_cap=excluded.market_cap,
                     pe_ratio=excluded.pe_ratio,
                     forward_pe=excluded.forward_pe,
                     peg_ratio=excluded.peg_ratio,
                     beta=excluded.beta,
                     short_pct=excluded.short_pct,
                     inst_ownership_pct=excluded.inst_ownership_pct,
                     updated_at=excluded.updated_at""",
                row,
            )

    def get_key_stats(self, ticker: str) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM key_stats WHERE ticker = ?", [ticker.upper()]
            ).fetchone()
            return dict(row) if row else None

    def get_high_short_interest_tickers(self, min_short_pct: float = 15.0) -> List[Dict]:
        """Find tickers with high short interest percent of float."""
        sql = "SELECT * FROM key_stats WHERE short_pct >= ? ORDER BY short_pct DESC"
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, [min_short_pct]).fetchall()]

    # ── Financial Statements (balance sheet, income, cash flow) ──

    def upsert_statements(self, rows: List[Dict]):
        """Upsert rows into financial_statements.

        Each row must have: ticker, statement_type, frequency, metric, period_end, value, updated_at
        """
        if not rows:
            return
        with self._conn() as conn:
            conn.executemany(
                """INSERT INTO financial_statements
                   (ticker, statement_type, frequency, metric, period_end, value, updated_at)
                   VALUES (:ticker, :statement_type, :frequency, :metric, :period_end, :value, :updated_at)
                   ON CONFLICT(ticker, statement_type, frequency, metric, period_end) DO UPDATE SET
                     value=excluded.value, updated_at=excluded.updated_at""",
                rows,
            )

    def get_statements(
        self,
        ticker: str,
        statement_type: str,
        frequency: str = "annual",
        metrics: Optional[List[str]] = None,
        limit_periods: int = 8,
    ) -> List[Dict]:
        """Query financial statement rows for a ticker.

        Returns rows sorted by period_end DESC, metric.
        """
        sql = "SELECT * FROM financial_statements WHERE ticker = ? AND statement_type = ? AND frequency = ?"
        params: list = [ticker.upper(), statement_type, frequency]
        if metrics:
            placeholders = ",".join("?" for _ in metrics)
            sql += f" AND metric IN ({placeholders})"
            params.extend(metrics)
        sql += " ORDER BY period_end DESC, metric LIMIT ?"
        params.append(limit_periods * 100)  # generous limit for many metrics
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def get_latest_metric(
        self, ticker: str, statement_type: str, metric: str, frequency: str = "annual"
    ) -> Optional[Dict]:
        """Get the most recent value for a single metric."""
        with self._conn() as conn:
            row = conn.execute(
                """SELECT * FROM financial_statements
                   WHERE ticker = ? AND statement_type = ? AND metric = ? AND frequency = ?
                   ORDER BY period_end DESC LIMIT 1""",
                [ticker.upper(), statement_type, metric, frequency],
            ).fetchone()
            return dict(row) if row else None

    # ── Dividend History ──────────────────────────────────────

    def upsert_dividends(self, rows: List[Dict]):
        if not rows:
            return
        with self._conn() as conn:
            conn.executemany(
                """INSERT INTO dividend_history (ticker, ex_date, amount, updated_at)
                   VALUES (:ticker, :ex_date, :amount, :updated_at)
                   ON CONFLICT(ticker, ex_date) DO UPDATE SET
                     amount=excluded.amount, updated_at=excluded.updated_at""",
                rows,
            )

    def get_dividends(self, ticker: str, limit: int = 20) -> List[Dict]:
        with self._conn() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM dividend_history WHERE ticker = ? ORDER BY ex_date DESC LIMIT ?",
                    [ticker.upper(), limit],
                ).fetchall()
            ]

    def get_dividend_summary(self, ticker: str) -> Optional[Dict]:
        """Return last 4 dividends + estimated annual yield info."""
        divs = self.get_dividends(ticker, limit=4)
        if not divs:
            return None
        total = sum(d["amount"] for d in divs if d["amount"])
        return {
            "ticker": ticker.upper(),
            "last_4_payments": divs,
            "last_4_total": round(total, 4),
            "count": len(divs),
            "latest_date": divs[0]["ex_date"],
        }

    # ── Valuation Metrics ────────────────────────────────────

    def upsert_valuation(self, row: Dict):
        if not row:
            return
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO valuation_metrics
                   (ticker, trailing_pe, forward_pe, peg_ratio, price_to_book,
                    price_to_sales, ev_to_revenue, ev_to_ebitda, profit_margins,
                    operating_margins, gross_margins, return_on_equity, return_on_assets,
                    debt_to_equity, current_ratio, quick_ratio, revenue_per_share,
                    earnings_growth, revenue_growth, updated_at)
                   VALUES (:ticker, :trailing_pe, :forward_pe, :peg_ratio, :price_to_book,
                           :price_to_sales, :ev_to_revenue, :ev_to_ebitda, :profit_margins,
                           :operating_margins, :gross_margins, :return_on_equity, :return_on_assets,
                           :debt_to_equity, :current_ratio, :quick_ratio, :revenue_per_share,
                           :earnings_growth, :revenue_growth, :updated_at)
                   ON CONFLICT(ticker) DO UPDATE SET
                     trailing_pe=excluded.trailing_pe, forward_pe=excluded.forward_pe,
                     peg_ratio=excluded.peg_ratio, price_to_book=excluded.price_to_book,
                     price_to_sales=excluded.price_to_sales, ev_to_revenue=excluded.ev_to_revenue,
                     ev_to_ebitda=excluded.ev_to_ebitda, profit_margins=excluded.profit_margins,
                     operating_margins=excluded.operating_margins, gross_margins=excluded.gross_margins,
                     return_on_equity=excluded.return_on_equity, return_on_assets=excluded.return_on_assets,
                     debt_to_equity=excluded.debt_to_equity, current_ratio=excluded.current_ratio,
                     quick_ratio=excluded.quick_ratio, revenue_per_share=excluded.revenue_per_share,
                     earnings_growth=excluded.earnings_growth, revenue_growth=excluded.revenue_growth,
                     updated_at=excluded.updated_at""",
                row,
            )

    def get_valuation(self, ticker: str) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM valuation_metrics WHERE ticker = ?", [ticker.upper()]
            ).fetchone()
            return dict(row) if row else None

    # ── Options Summary ──────────────────────────────────────

    def upsert_options_summary(self, rows: List[Dict]):
        if not rows:
            return
        with self._conn() as conn:
            conn.executemany(
                """INSERT INTO options_summary
                   (ticker, expiration, calls_oi, puts_oi, calls_volume, puts_volume,
                    calls_avg_iv, puts_avg_iv, put_call_oi_ratio, updated_at)
                   VALUES (:ticker, :expiration, :calls_oi, :puts_oi, :calls_volume, :puts_volume,
                           :calls_avg_iv, :puts_avg_iv, :put_call_oi_ratio, :updated_at)
                   ON CONFLICT(ticker, expiration) DO UPDATE SET
                     calls_oi=excluded.calls_oi, puts_oi=excluded.puts_oi,
                     calls_volume=excluded.calls_volume, puts_volume=excluded.puts_volume,
                     calls_avg_iv=excluded.calls_avg_iv, puts_avg_iv=excluded.puts_avg_iv,
                     put_call_oi_ratio=excluded.put_call_oi_ratio,
                     updated_at=excluded.updated_at""",
                rows,
            )

    def get_options_summary(self, ticker: str, limit: int = 10) -> List[Dict]:
        with self._conn() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM options_summary WHERE ticker = ? ORDER BY expiration LIMIT ?",
                    [ticker.upper(), limit],
                ).fetchall()
            ]

    # ── Inverse ETF Universe ──────────────────────────────────

    def upsert_inverse_etf(self, row: Dict):
        """Upsert a single inverse ETF metadata record."""
        if not row:
            return
        now = datetime.now().isoformat()
        r = {
            "ticker": row["ticker"].upper(),
            "name": row.get("name", ""),
            "leverage": int(row.get("leverage", 1)),
            "underlying": row.get("underlying", ""),
            "category": row.get("category", "index"),
            "provider": row.get("provider", ""),
            "expense_ratio": float(row.get("expense_ratio", 0.0)),
            "avg_daily_volume": int(row.get("avg_daily_volume", 0)),
            "aum_millions": float(row.get("aum_millions", 0.0)),
            "is_active": int(row.get("is_active", 1)),
            "last_data_check": row.get("last_data_check"),
            "updated_at": row.get("updated_at", now),
        }
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO inverse_etf_universe
                   (ticker, name, leverage, underlying, category, provider,
                    expense_ratio, avg_daily_volume, aum_millions, is_active,
                    last_data_check, updated_at)
                   VALUES (:ticker, :name, :leverage, :underlying, :category, :provider,
                           :expense_ratio, :avg_daily_volume, :aum_millions, :is_active,
                           :last_data_check, :updated_at)
                   ON CONFLICT(ticker) DO UPDATE SET
                     name=excluded.name, leverage=excluded.leverage,
                     underlying=excluded.underlying, category=excluded.category,
                     provider=excluded.provider, expense_ratio=excluded.expense_ratio,
                     avg_daily_volume=excluded.avg_daily_volume, aum_millions=excluded.aum_millions,
                     is_active=excluded.is_active, last_data_check=excluded.last_data_check,
                     updated_at=excluded.updated_at""",
                r,
            )

    def upsert_inverse_etfs(self, rows: List[Dict]):
        """Bulk upsert inverse ETF metadata."""
        for row in rows:
            self.upsert_inverse_etf(row)

    def get_inverse_etf(self, ticker: str) -> Optional[Dict]:
        """Look up a single inverse ETF by ticker."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM inverse_etf_universe WHERE ticker = ?",
                [ticker.upper()],
            ).fetchone()
            return dict(row) if row else None

    def get_all_inverse_etfs(
        self,
        active_only: bool = True,
        category: Optional[str] = None,
        min_volume: int = 0,
    ) -> List[Dict]:
        """Query inverse ETFs with optional filters."""
        sql = "SELECT * FROM inverse_etf_universe WHERE 1=1"
        params: list = []
        if active_only:
            sql += " AND is_active = 1"
        if category:
            sql += " AND category = ?"
            params.append(category)
        if min_volume > 0:
            sql += " AND avg_daily_volume >= ?"
            params.append(min_volume)
        sql += " ORDER BY category, leverage, ticker"
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def get_inverse_etfs_for_underlying(self, underlying: str) -> List[Dict]:
        """Get all inverse ETFs that short a given underlying (e.g., 'SPY')."""
        with self._conn() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM inverse_etf_universe WHERE underlying = ? AND is_active = 1 ORDER BY leverage",
                    [underlying.upper()],
                ).fetchall()
            ]

    def get_inverse_etf_tickers(self, active_only: bool = True) -> set:
        """Return set of all inverse ETF tickers for fast membership checks."""
        rows = self.get_all_inverse_etfs(active_only=active_only)
        return {r["ticker"] for r in rows}

    # ── Inverse ETF Screens ───────────────────────────────────

    def upsert_screen_result(self, row: Dict):
        """Save a single screening result."""
        if not row:
            return
        now = datetime.now().isoformat()
        r = {
            "ticker": row["ticker"].upper(),
            "screen_date": row.get("screen_date", date.today().isoformat()),
            "screen_time": row.get("screen_time", ""),
            "volume_ratio": float(row.get("volume_ratio", 0.0)),
            "momentum_score": float(row.get("momentum_score", 0.0)),
            "vwap_distance_pct": float(row.get("vwap_distance_pct", 0.0)),
            "rsi_14": float(row.get("rsi_14", 0.0)),
            "spread_bps": float(row.get("spread_bps", 0.0)),
            "signal": row.get("signal", "NEUTRAL"),
            "entry_price": float(row.get("entry_price", 0.0)),
            "notes": row.get("notes", ""),
            "updated_at": row.get("updated_at", now),
        }
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO inverse_etf_screens
                   (ticker, screen_date, screen_time, volume_ratio, momentum_score,
                    vwap_distance_pct, rsi_14, spread_bps, signal, entry_price, notes, updated_at)
                   VALUES (:ticker, :screen_date, :screen_time, :volume_ratio, :momentum_score,
                           :vwap_distance_pct, :rsi_14, :spread_bps, :signal, :entry_price,
                           :notes, :updated_at)
                   ON CONFLICT(ticker, screen_date, screen_time) DO UPDATE SET
                     volume_ratio=excluded.volume_ratio, momentum_score=excluded.momentum_score,
                     vwap_distance_pct=excluded.vwap_distance_pct, rsi_14=excluded.rsi_14,
                     spread_bps=excluded.spread_bps, signal=excluded.signal,
                     entry_price=excluded.entry_price, notes=excluded.notes,
                     updated_at=excluded.updated_at""",
                r,
            )

    def get_latest_screens(
        self, screen_date: Optional[str] = None, signal: Optional[str] = None
    ) -> List[Dict]:
        """Retrieve screening results, optionally filtered by date and signal."""
        target_date = screen_date or date.today().isoformat()
        sql = "SELECT * FROM inverse_etf_screens WHERE screen_date = ?"
        params: list = [target_date]
        if signal:
            sql += " AND signal = ?"
            params.append(signal.upper())
        sql += " ORDER BY screen_time DESC, momentum_score DESC"
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    # ── Convenience ───────────────────────────────────────────

    def table_counts(self) -> Dict[str, int]:
        with self._conn() as conn:
            counts = {}
            for t in ["earnings_calendar", "financial_events", "analyst_ratings", "key_stats",
                      "financial_statements", "dividend_history", "valuation_metrics", "options_summary",
                      "inverse_etf_universe", "inverse_etf_screens"]:
                counts[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            return counts


# ── Module-level shortcuts ────────────────────────────────────

def get_upcoming_earnings(days: int = 5, tickers: Optional[List[str]] = None) -> List[Dict]:
    return FinancialDB().get_upcoming_earnings(days=days, tickers=tickers)


def has_earnings_soon(ticker: str, days: int = 2) -> bool:
    return FinancialDB().has_earnings_soon(ticker, days=days)
