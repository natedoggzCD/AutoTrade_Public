"""
yfinance-based fetcher for financial statements, dividends, valuation metrics, and options.

Returns normalized rows/dicts ready for FinancialDB upsert methods.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _safe_float(value):
    try:
        if value is None:
            return None
        import math
        v = float(value)
        return None if math.isnan(v) else v
    except Exception:
        return None


def _df_to_rows(
    df, ticker: str, statement_type: str, frequency: str
) -> List[Dict]:
    """Convert a yfinance DataFrame (metrics x periods) to normalized rows."""
    rows: List[Dict] = []
    if df is None or df.empty:
        return rows
    now = datetime.utcnow().isoformat(timespec="seconds")
    for metric in df.index:
        for col in df.columns:
            val = _safe_float(df.loc[metric, col])
            if val is None:
                continue
            try:
                period_end = col.to_pydatetime().date().isoformat() if hasattr(col, "to_pydatetime") else str(col)
            except Exception:
                continue
            rows.append({
                "ticker": ticker.upper(),
                "statement_type": statement_type,
                "frequency": frequency,
                "metric": str(metric),
                "period_end": period_end,
                "value": val,
                "updated_at": now,
            })
    return rows


class YFinanceFinancials:
    def __init__(self):
        import yfinance as yf
        self.yf = yf

    def fetch_balance_sheet(self, ticker: str) -> List[Dict]:
        """Fetch annual + quarterly balance sheet rows."""
        t = self.yf.Ticker(ticker)
        rows: List[Dict] = []
        try:
            rows.extend(_df_to_rows(t.balance_sheet, ticker, "balance_sheet", "annual"))
        except Exception as e:
            logger.warning(f"{ticker} balance_sheet annual: {e}")
        try:
            rows.extend(_df_to_rows(t.quarterly_balance_sheet, ticker, "balance_sheet", "quarterly"))
        except Exception as e:
            logger.warning(f"{ticker} balance_sheet quarterly: {e}")
        return rows

    def fetch_income_statement(self, ticker: str) -> List[Dict]:
        """Fetch annual + quarterly income statement rows."""
        t = self.yf.Ticker(ticker)
        rows: List[Dict] = []
        try:
            rows.extend(_df_to_rows(t.income_stmt, ticker, "income_statement", "annual"))
        except Exception as e:
            logger.warning(f"{ticker} income_statement annual: {e}")
        try:
            rows.extend(_df_to_rows(t.quarterly_income_stmt, ticker, "income_statement", "quarterly"))
        except Exception as e:
            logger.warning(f"{ticker} income_statement quarterly: {e}")
        return rows

    def fetch_cash_flow(self, ticker: str) -> List[Dict]:
        """Fetch annual + quarterly cash flow rows."""
        t = self.yf.Ticker(ticker)
        rows: List[Dict] = []
        try:
            rows.extend(_df_to_rows(t.cashflow, ticker, "cash_flow", "annual"))
        except Exception as e:
            logger.warning(f"{ticker} cash_flow annual: {e}")
        try:
            rows.extend(_df_to_rows(t.quarterly_cashflow, ticker, "cash_flow", "quarterly"))
        except Exception as e:
            logger.warning(f"{ticker} cash_flow quarterly: {e}")
        return rows

    def fetch_all_statements(self, ticker: str) -> List[Dict]:
        """Fetch all 3 financial statements (convenience method)."""
        rows: List[Dict] = []
        rows.extend(self.fetch_balance_sheet(ticker))
        rows.extend(self.fetch_income_statement(ticker))
        rows.extend(self.fetch_cash_flow(ticker))
        return rows

    # ── Dividends ────────────────────────────────────────────

    def fetch_dividends(self, ticker: str, max_rows: int = 50) -> List[Dict]:
        """Fetch dividend history as rows for dividend_history table."""
        t = self.yf.Ticker(ticker)
        rows: List[Dict] = []
        try:
            divs = t.dividends
        except Exception:
            return rows
        if divs is None or len(divs) == 0:
            return rows
        now = datetime.utcnow().isoformat(timespec="seconds")
        for idx, val in divs.tail(max_rows).items():
            try:
                dt = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
                rows.append({
                    "ticker": ticker.upper(),
                    "ex_date": dt.date().isoformat(),
                    "amount": _safe_float(val),
                    "updated_at": now,
                })
            except Exception:
                continue
        return rows

    # ── Valuation Metrics ────────────────────────────────────

    def fetch_valuation_metrics(self, ticker: str) -> Optional[Dict]:
        """Fetch valuation, profitability, and efficiency metrics from yfinance info."""
        t = self.yf.Ticker(ticker)
        try:
            info = t.info or {}
        except Exception:
            return None
        if not info:
            return None
        return {
            "ticker": ticker.upper(),
            "trailing_pe": _safe_float(info.get("trailingPE")),
            "forward_pe": _safe_float(info.get("forwardPE")),
            "peg_ratio": _safe_float(info.get("pegRatio")),
            "price_to_book": _safe_float(info.get("priceToBook")),
            "price_to_sales": _safe_float(info.get("priceToSalesTrailing12Months")),
            "ev_to_revenue": _safe_float(info.get("enterpriseToRevenue")),
            "ev_to_ebitda": _safe_float(info.get("enterpriseToEbitda")),
            "profit_margins": _safe_float(info.get("profitMargins")),
            "operating_margins": _safe_float(info.get("operatingMargins")),
            "gross_margins": _safe_float(info.get("grossMargins")),
            "return_on_equity": _safe_float(info.get("returnOnEquity")),
            "return_on_assets": _safe_float(info.get("returnOnAssets")),
            "debt_to_equity": _safe_float(info.get("debtToEquity")),
            "current_ratio": _safe_float(info.get("currentRatio")),
            "quick_ratio": _safe_float(info.get("quickRatio")),
            "revenue_per_share": _safe_float(info.get("revenuePerShare")),
            "earnings_growth": _safe_float(info.get("earningsGrowth")),
            "revenue_growth": _safe_float(info.get("revenueGrowth")),
            "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
        }

    # ── Options Summary ──────────────────────────────────────

    def fetch_options_summary(self, ticker: str, max_expirations: int = 6) -> List[Dict]:
        """Fetch summarized options chain data (OI, volume, avg IV per expiration)."""
        t = self.yf.Ticker(ticker)
        rows: List[Dict] = []
        try:
            expirations = t.options
        except Exception:
            return rows
        if not expirations:
            return rows
        now = datetime.utcnow().isoformat(timespec="seconds")
        for exp in expirations[:max_expirations]:
            try:
                chain = t.option_chain(exp)
                calls = chain.calls
                puts = chain.puts
                calls_oi = int(calls["openInterest"].sum()) if "openInterest" in calls.columns else 0
                puts_oi = int(puts["openInterest"].sum()) if "openInterest" in puts.columns else 0
                calls_vol = int(calls["volume"].sum()) if "volume" in calls.columns else 0
                puts_vol = int(puts["volume"].sum()) if "volume" in puts.columns else 0
                calls_iv = _safe_float(calls["impliedVolatility"].mean()) if "impliedVolatility" in calls.columns else None
                puts_iv = _safe_float(puts["impliedVolatility"].mean()) if "impliedVolatility" in puts.columns else None
                pc_ratio = round(puts_oi / calls_oi, 3) if calls_oi > 0 else None
                rows.append({
                    "ticker": ticker.upper(),
                    "expiration": exp,
                    "calls_oi": calls_oi,
                    "puts_oi": puts_oi,
                    "calls_volume": calls_vol,
                    "puts_volume": puts_vol,
                    "calls_avg_iv": round(calls_iv, 4) if calls_iv else None,
                    "puts_avg_iv": round(puts_iv, 4) if puts_iv else None,
                    "put_call_oi_ratio": pc_ratio,
                    "updated_at": now,
                })
            except Exception as e:
                logger.warning(f"{ticker} options {exp}: {e}")
                continue
        return rows
