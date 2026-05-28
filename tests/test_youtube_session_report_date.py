"""Holiday/weekend handling for youtube_daily_scanner._session_report_date.

Regression: at 12:15 AM on a market holiday the scanner used to key its
report under the orphan holiday date instead of the next trading session,
so it disagreed with PM-plan / EOD pipelines that already roll holidays
forward via market_time.get_pm_plan_date.
"""

from datetime import datetime

from tools.youtube_daily_scanner import _session_report_date


def test_weekday_midnight_returns_today():
    ts = datetime(2026, 5, 6, 0, 15)  # Wed 12:15 AM, normal trading day
    assert _session_report_date(ts) == "2026-05-06"


def test_weekday_after_close_rolls_to_next_session():
    ts = datetime(2026, 5, 6, 17, 0)  # Wed 5 PM
    assert _session_report_date(ts) == "2026-05-07"  # Thursday


def test_friday_after_close_rolls_to_monday():
    ts = datetime(2026, 5, 8, 17, 0)  # Fri 5 PM
    assert _session_report_date(ts) == "2026-05-11"  # Monday


def test_saturday_rolls_to_monday():
    ts = datetime(2026, 5, 9, 10, 0)  # Saturday morning
    assert _session_report_date(ts) == "2026-05-11"


def test_mlk_day_midnight_rolls_to_tuesday():
    # 2026-01-19 is MLK Day (Monday). At 12:15 AM Mon, the legacy code
    # returned 2026-01-19 (the holiday itself); we now want Tuesday.
    ts = datetime(2026, 1, 19, 0, 15)
    assert _session_report_date(ts) == "2026-01-20"


def test_good_friday_midnight_rolls_to_monday():
    # 2026-04-03 is Good Friday. 12:15 AM Friday should target Mon Apr 6.
    ts = datetime(2026, 4, 3, 0, 15)
    assert _session_report_date(ts) == "2026-04-06"


def test_memorial_day_midnight_rolls_to_tuesday():
    # 2026-05-25 is Memorial Day (Monday).
    ts = datetime(2026, 5, 25, 0, 15)
    assert _session_report_date(ts) == "2026-05-26"


def test_thursday_before_good_friday_after_close_rolls_to_monday():
    # Thu 2026-04-02 17:00 — Good Friday is Apr 3, so next session is Mon Apr 6.
    ts = datetime(2026, 4, 2, 17, 0)
    assert _session_report_date(ts) == "2026-04-06"
