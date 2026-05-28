"""
Overnight / premarket financial checks using financial.db.

Provides helpers that the main agent workflows can call to check
upcoming earnings, balance sheet health, cash flow risks,
dividend stability, valuation sanity, and unusual options activity.
"""

import logging
from typing import Dict, List

from autotrade.utils.financial_db import FinancialDB

logger = logging.getLogger(__name__)


def check_earnings_risk(tickers: List[str], days: int = 5) -> Dict[str, List[Dict]]:
    """Return tickers that have earnings within `days` trading days.

    Returns dict mapping ticker -> list of earnings rows.
    Used by overnight research to flag positions to reduce/avoid.
    """
    db = FinancialDB()
    upcoming = db.get_upcoming_earnings(days=days, tickers=tickers)
    risk: Dict[str, List[Dict]] = {}
    for row in upcoming:
        risk.setdefault(row["ticker"], []).append(row)
    if risk:
        logger.info(f"Earnings risk: {len(risk)} tickers have earnings in next {days}d: {sorted(risk.keys())}")
    return risk


def flag_earnings_today(tickers: List[str]) -> List[str]:
    """Return tickers that have earnings TODAY (premarket check).

    Used by premarket validation to alert on same-day earnings.
    """
    db = FinancialDB()
    flagged = [t for t in tickers if db.has_earnings_soon(t, days=0)]
    if flagged:
        logger.warning(f"EARNINGS TODAY: {flagged}")
    return flagged


def check_balance_sheet_health(tickers: List[str]) -> Dict[str, Dict]:
    """Flag tickers with weak balance sheets (high leverage, low current ratio).

    Returns dict mapping ticker -> {leverage_ratio, current_ratio, flag}.
    """
    db = FinancialDB()
    warnings: Dict[str, Dict] = {}
    for ticker in tickers:
        total_debt = db.get_latest_metric(ticker, "balance_sheet", "Total Debt")
        total_assets = db.get_latest_metric(ticker, "balance_sheet", "Total Assets")
        current_assets = db.get_latest_metric(ticker, "balance_sheet", "Current Assets")
        current_liabilities = db.get_latest_metric(ticker, "balance_sheet", "Current Liabilities")

        info: Dict = {"ticker": ticker}
        flags: List[str] = []

        if total_debt and total_assets and total_assets["value"]:
            ratio = total_debt["value"] / total_assets["value"]
            info["leverage_ratio"] = round(ratio, 2)
            if ratio > 0.6:
                flags.append(f"high_leverage={ratio:.2f}")

        if current_assets and current_liabilities and current_liabilities["value"]:
            ratio = current_assets["value"] / current_liabilities["value"]
            info["current_ratio"] = round(ratio, 2)
            if ratio < 1.0:
                flags.append(f"low_current_ratio={ratio:.2f}")

        if flags:
            info["flags"] = flags
            warnings[ticker] = info
            logger.warning(f"Balance sheet risk {ticker}: {', '.join(flags)}")

    return warnings


def check_cash_flow_health(tickers: List[str]) -> Dict[str, Dict]:
    """Flag tickers with negative free cash flow.

    Returns dict mapping ticker -> {free_cash_flow, flag}.
    """
    db = FinancialDB()
    warnings: Dict[str, Dict] = {}
    for ticker in tickers:
        fcf = db.get_latest_metric(ticker, "cash_flow", "Free Cash Flow")
        if fcf and fcf["value"] is not None and fcf["value"] < 0:
            warnings[ticker] = {
                "ticker": ticker,
                "free_cash_flow": fcf["value"],
                "period": fcf["period_end"],
                "flags": [f"negative_fcf={fcf['value']:,.0f}"],
            }
            logger.warning(f"Cash flow risk {ticker}: negative FCF = {fcf['value']:,.0f}")
    return warnings


def check_dividend_stability(tickers: List[str]) -> Dict[str, Dict]:
    """Flag tickers with irregular or cut dividends.

    Returns dict mapping ticker -> {summary, flags}.
    """
    db = FinancialDB()
    warnings: Dict[str, Dict] = {}
    for ticker in tickers:
        summary = db.get_dividend_summary(ticker)
        if not summary or summary["count"] < 2:
            continue
        payments = summary["last_4_payments"]
        amounts = [p["amount"] for p in payments if p["amount"] is not None]
        if not amounts:
            continue
            
        flags: List[str] = []
        if amounts[0] == 0:
            flags.append("dividend_suspended")
        elif len(amounts) >= 2 and amounts[0] < amounts[-1]:
            flags.append("dividend_cut")
            
        if flags:
            warnings[ticker] = {
                "ticker": ticker,
                "latest_amount": amounts[0],
                "previous_amount": amounts[-1] if len(amounts) >= 2 else 0,
                "flags": flags,
            }
            logger.warning(f"Dividend risk {ticker}: {', '.join(flags)}")
    return warnings


def check_valuation_sanity(tickers: List[str]) -> Dict[str, Dict]:
    """Flag tickers with extreme valuation metrics.

    Returns dict mapping ticker -> {metrics, flags}.
    """
    db = FinancialDB()
    warnings: Dict[str, Dict] = {}
    for ticker in tickers:
        val = db.get_valuation(ticker)
        if not val:
            continue
        flags: List[str] = []
        if val.get("trailing_pe") and val["trailing_pe"] > 100:
            flags.append(f"extreme_pe={val['trailing_pe']:.1f}")
        if val.get("debt_to_equity") and val["debt_to_equity"] > 200:
            flags.append(f"high_debt_equity={val['debt_to_equity']:.1f}")
        if val.get("earnings_growth") and val["earnings_growth"] < -0.2:
            flags.append(f"earnings_decline={val['earnings_growth']:.1%}")
        if flags:
            warnings[ticker] = {"ticker": ticker, "flags": flags}
            logger.warning(f"Valuation risk {ticker}: {', '.join(flags)}")
    return warnings


def check_options_positioning(tickers: List[str]) -> Dict[str, Dict]:
    """Flag tickers with unusual put/call ratios (bearish sentiment).

    Returns dict mapping ticker -> {put_call_ratio, flags}.
    """
    db = FinancialDB()
    warnings: Dict[str, Dict] = {}
    for ticker in tickers:
        opts = db.get_options_summary(ticker, limit=3)
        if not opts:
            continue
        # Check nearest expiration
        nearest = opts[0]
        pc_ratio = nearest.get("put_call_oi_ratio")
        if pc_ratio and pc_ratio > 1.5:
            warnings[ticker] = {
                "ticker": ticker,
                "expiration": nearest["expiration"],
                "put_call_oi_ratio": pc_ratio,
                "flags": [f"high_put_call_ratio={pc_ratio:.2f}"],
            }
            logger.warning(f"Options risk {ticker}: put/call OI ratio = {pc_ratio:.2f}")
    return warnings
