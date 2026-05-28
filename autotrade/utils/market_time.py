"""
Market time utilities for trading workflows.

Keeps weekend logic and next-trading-day rules in one place so that
PM workflows and schedulers stay consistent.
"""

from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta
from typing import Dict, Optional, Union

from autotrade.utils.mcp_client import mcp_time, mcp_alpaca

try:
    import pytz
except ImportError:  # pragma: no cover - optional dependency
    pytz = None

# Centralized US Market Holidays for 2026
US_HOLIDAYS_2026 = [
    date(2026, 1, 1),   # New Year's Day
    date(2026, 1, 19),  # MLK Day
    date(2026, 2, 16),  # Presidents Day
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 6, 19),  # Juneteenth
    date(2026, 7, 3),   # Independence Day (observed)
    date(2026, 9, 7),   # Labor Day
    date(2026, 11, 26), # Thanksgiving
    date(2026, 12, 25), # Christmas Day
]

MARKET_CLOSE_LOCAL = time(hour=16, minute=0)


def get_market_now() -> datetime:
    """Return current time in US/Eastern if pytz is available, else local time.
    Returns NAIVE datetime (tz-neutral) to avoid offset subtraction errors.
    """
    use_mcp_time = os.environ.get("AUTOTRADE_USE_MCP_TIME", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if use_mcp_time:
        try:
            # Keep a strict timeout to avoid blocking startup workflows.
            now_data = mcp_time(
                "get_current_time",
                timezone="America/New_York",
                _timeout_seconds=1.0,
            )
            if isinstance(now_data, dict) and "datetime" in now_data:
                dt = datetime.fromisoformat(now_data["datetime"])
                return dt.replace(tzinfo=None)
        except Exception:
            # Fallback to local implementation
            pass

    if pytz:
        # Get localized time but strip tzinfo to return naive
        return datetime.now(pytz.timezone("US/Eastern")).replace(tzinfo=None)
    return datetime.now()


def is_weekend(value: Union[datetime, date]) -> bool:
    """True if the given date/datetime falls on Saturday/Sunday."""
    if isinstance(value, datetime):
        return value.weekday() >= 5
    return value.weekday() >= 5


def is_trading_day(value: Union[datetime, date]) -> bool:
    """True if the given date is a weekday AND not a market holiday."""
    d = value.date() if isinstance(value, datetime) else value
    if d.weekday() >= 5:
        return False
    return d not in US_HOLIDAYS_2026


def next_trading_day(value: Optional[Union[datetime, date]] = None) -> date:
    """Get the next valid trading date (skips weekends and holidays)."""
    if value is None:
        value = get_market_now()
    if isinstance(value, datetime):
        current = value.date()
    else:
        current = value

    target = current + timedelta(days=1)
    while not is_trading_day(target):
        target += timedelta(days=1)
    return target


def last_trading_day(value: Optional[Union[datetime, date]] = None) -> date:
    """Get the most recent valid trading date (skips weekends and holidays)."""
    if value is None:
        value = get_market_now()
    if isinstance(value, datetime):
        current = value.date()
    else:
        current = value

    target = current
    while not is_trading_day(target):
        target -= timedelta(days=1)
    return target


def resolve_session_review_date(
    now: Optional[Union[datetime, date]] = None,
    *,
    after_close_hour: int = 16,
) -> date:
    """Resolve which session date a "post-market" artifact should describe.

    Use this for any artifact that summarizes a *completed* trading session —
    EOD review, daily review, end-of-day feedback. The rule:

      - Weekend or market holiday: most recent completed trading day.
      - Trading day before market close (default 16:00 ET): the *prior*
        trading day, because today's session has not yet completed. This
        matters when the agent boots overnight (e.g., 12:15 AM): the
        relevant review is yesterday's, not today's not-yet-existed session.
      - Trading day at/after close: today's date — the session has completed.

    Always defers to is_trading_day / last_trading_day, so US_HOLIDAYS_2026
    holidays (MLK Day, Presidents Day, Good Friday, Memorial Day, Juneteenth,
    Independence Day, Labor Day, Thanksgiving, Christmas) and weekends roll
    back to the previous trading session correctly.

    The default `after_close_hour=16` matches NYSE close. Override only for
    testing.
    """
    now = now or get_market_now()
    if isinstance(now, date) and not isinstance(now, datetime):
        if not is_trading_day(now):
            return last_trading_day(now)
        return now
    today = now.date()
    if not is_trading_day(today):
        # Weekend or holiday — most recent completed session.
        return last_trading_day(today)
    if now.hour >= after_close_hour:
        # Trading day at/after close — today's session has completed.
        return today
    # Trading day, before close — yesterday's session is the latest completed.
    return last_trading_day((now - timedelta(days=1)).date())


def expected_core_market_data_date(
    reference_dt: Optional[datetime] = None,
    market_close_local: time = MARKET_CLOSE_LOCAL,
) -> str:
    """
    Return the market-data date that must be present for trading to proceed.

    If today is a trading day and it is after market close, the required dataset
    is today's data. Otherwise, it is the most recent trading day BEFORE today.
    """
    as_of = reference_dt if isinstance(reference_dt, datetime) else get_market_now()
    current_day = as_of.date()

    # If it's a trading day and market has closed, expect today's data.
    if is_trading_day(current_day) and as_of.time() >= market_close_local:
        expected = current_day
    else:
        # If it's a holiday, weekend, or before market close, 
        # we expect the data from the MOST RECENT trading day that finished.
        expected = last_trading_day(current_day - timedelta(days=1))

    return expected.isoformat()


def is_weekend_daytime(
    now: Optional[datetime] = None,
    start_hour: int = 8,
    end_hour: int = 20,
) -> bool:
    """Weekend daytime window used to treat PM mode as after-hours."""
    now = now or get_market_now()
    if now.weekday() < 5:
        return False
    return start_hour <= now.hour < end_hour


def get_pm_plan_date(
    now: Optional[Union[datetime, date]] = None,
    after_close_hour: int = 16,
) -> date:
    """
    Determine the trading date a PM plan should be saved under.

    PM plans are target-date keyed, not session-date keyed. A plan written
    after the close on April 14 is saved as `pm_plan_2026-04-15.json`, while
    session-keyed artifacts such as `eod_review_YYYY-MM-DD.json`,
    `signals_YYYY-MM-DD.json`, and `morning_game_plan_YYYYMMDD.json` keep the
    session date.

    - If it's the weekend, always use the next trading day (Monday).
    - If it's after market close, use the next trading day.
    - Otherwise, use today's date (useful for manual/testing runs).
    """
    now = now or get_market_now()
    if isinstance(now, date) and not isinstance(now, datetime):
        if not is_trading_day(now):
            return next_trading_day(now)
        return now
    today = now.date()
    if not is_trading_day(today):
        # Weekend or market holiday — PM plan targets the next trading day.
        return next_trading_day(now)
    if now.hour >= after_close_hour:
        # Trading day at/after close — PM plan targets the next session.
        return next_trading_day(now)
    return today


def is_market_open() -> bool:
    """Authoritative market status check using Alpaca MCP."""
    try:
        clock = mcp_alpaca("get_market_clock", _timeout_seconds=1.5)
        if isinstance(clock, dict) and "is_open" in clock:
            return bool(clock["is_open"])
    except Exception:
        pass
    
    # Fallback: simple hour check (not calendar aware)
    now = get_market_now()
    if now.weekday() >= 5:
        return False
    return 9.5 <= (now.hour + now.minute/60) < 16.0


def get_market_times_today() -> Dict:
    """Get today's exact open/close times from Alpaca MCP."""
    try:
        now = get_market_now()
        date_str = now.strftime("%Y-%m-%d")
        calendar = mcp_alpaca(
            "get_market_calendar",
            start=date_str,
            end=date_str,
            _timeout_seconds=1.5,
        )
        if isinstance(calendar, list) and len(calendar) > 0:
            return calendar[0]
    except Exception:
        pass
    return {}
