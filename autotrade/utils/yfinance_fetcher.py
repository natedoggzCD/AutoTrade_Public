"""
yfinance-based data fetcher for the Financial DB.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _infer_time_of_day(ts) -> str:
    if not ts:
        return "TBD"
    try:
        hour = ts.hour
        if hour == 0:
            return "TBD"
        if hour < 12:
            return "BMO"
        if hour >= 16:
            return "AMC"
        return "TBD"
    except Exception:
        return "TBD"


class YFinanceFetcher:
    def __init__(self):
        import yfinance as yf

        self.yf = yf

    def fetch_earnings_calendar(self, ticker: str, limit: int = 8) -> List[Dict]:
        """Fetch upcoming/past earnings dates with estimates/actuals."""
        results: List[Dict] = []
        t = self.yf.Ticker(ticker)

        df = None
        if hasattr(t, "get_earnings_dates"):
            try:
                df = t.get_earnings_dates(limit=limit)
            except Exception:
                df = None
        if df is None and hasattr(t, "earnings_dates"):
            try:
                df = t.earnings_dates
            except Exception:
                df = None

        if df is None or len(df) == 0:
            return results

        for idx, row in df.iterrows():
            try:
                dt = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
                earnings_date = dt.date().isoformat()
                time_of_day = _infer_time_of_day(dt)
            except Exception:
                earnings_date = None
                time_of_day = "TBD"

            results.append(
                {
                    "ticker": ticker.upper(),
                    "earnings_date": earnings_date,
                    "time_of_day": time_of_day,
                    "eps_estimate": _safe_float(row.get("EPS Estimate")),
                    "eps_actual": _safe_float(row.get("Reported EPS")),
                    "revenue_estimate": _safe_float(row.get("Revenue Estimate")),
                    "revenue_actual": _safe_float(row.get("Revenue")),
                    "surprise_pct": _safe_float(row.get("Surprise(%)")),
                    "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
                }
            )

        return results

    def fetch_financial_events(self, ticker: str, max_rows: int = 50) -> List[Dict]:
        """Fetch dividends/splits as financial events."""
        results: List[Dict] = []
        t = self.yf.Ticker(ticker)

        try:
            dividends = t.dividends
        except Exception:
            dividends = None
        if dividends is not None and len(dividends) > 0:
            for idx, val in dividends.tail(max_rows).items():
                try:
                    dt = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
                    results.append(
                        {
                            "ticker": ticker.upper(),
                            "event_type": "dividend",
                            "event_date": dt.date().isoformat(),
                            "details": f"dividend={_safe_float(val)}",
                            "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
                        }
                    )
                except Exception:
                    continue

        try:
            splits = t.splits
        except Exception:
            splits = None
        if splits is not None and len(splits) > 0:
            for idx, val in splits.tail(max_rows).items():
                try:
                    dt = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
                    results.append(
                        {
                            "ticker": ticker.upper(),
                            "event_type": "split",
                            "event_date": dt.date().isoformat(),
                            "details": f"split={_safe_float(val)}",
                            "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
                        }
                    )
                except Exception:
                    continue

        return results

    def fetch_analyst_ratings(self, ticker: str, max_rows: int = 200) -> List[Dict]:
        """Fetch analyst recommendations (firm/rating)."""
        results: List[Dict] = []
        t = self.yf.Ticker(ticker)

        df = None
        try:
            df = t.recommendations
        except Exception:
            df = None

        if df is None or len(df) == 0:
            return results

        for idx, row in df.tail(max_rows).iterrows():
            try:
                dt = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
                firm = row.get("Firm") or row.get("Firm Name")
                rating = row.get("To Grade") or row.get("ToGrade") or row.get("Rating")
                results.append(
                    {
                        "ticker": ticker.upper(),
                        "firm": firm,
                        "rating": rating,
                        "price_target": _safe_float(row.get("Price Target")),
                        "date": dt.date().isoformat(),
                        "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
                    }
                )
            except Exception:
                continue

        return results

    def fetch_key_stats(self, ticker: str) -> Optional[Dict]:
        """Fetch key stats from yfinance info."""
        t = self.yf.Ticker(ticker)
        try:
            info = t.info or {}
        except Exception:
            info = {}

        if not info:
            return None

        return {
            "ticker": ticker.upper(),
            "market_cap": _safe_float(info.get("marketCap")),
            "pe_ratio": _safe_float(info.get("trailingPE")),
            "forward_pe": _safe_float(info.get("forwardPE")),
            "peg_ratio": _safe_float(info.get("pegRatio")),
            "beta": _safe_float(info.get("beta")),
            "short_pct": _safe_float(info.get("shortPercentOfFloat")),
            "inst_ownership_pct": _safe_float(info.get("heldPercentInstitutions")),
            "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
