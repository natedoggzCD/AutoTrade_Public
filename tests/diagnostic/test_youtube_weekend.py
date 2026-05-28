import pytest
from datetime import datetime, timedelta
from tools.youtube_daily_scanner import _weekend_adjusted_max_age, _session_report_date

def test_weekend_adjusted_max_age():
    # Regular weekday (Wednesday)
    # Note: _weekend_adjusted_max_age uses datetime.now() internally
    # We can't easily mock datetime.now() without a library like freezegun
    # But we can verify the logic if we know today's day
    
    current_day = datetime.now().weekday()
    configured_age = 36
    
    adjusted = _weekend_adjusted_max_age(configured_age)
    
    if current_day >= 5: # Saturday or Sunday
        assert adjusted >= 96
    else:
        assert adjusted == 36

def test_session_report_date_logic():
    # Friday 10 AM -> Today
    fri_am = datetime(2026, 2, 27, 10, 0, 0)
    assert _session_report_date(fri_am) == "2026-02-27"
    
    # Friday 5 PM -> Next Monday
    fri_pm = datetime(2026, 2, 27, 17, 0, 0)
    assert _session_report_date(fri_pm) == "2026-03-02"
    
    # Saturday -> Next Monday
    sat = datetime(2026, 2, 28, 12, 0, 0)
    assert _session_report_date(sat) == "2026-03-02"
    
    # Sunday -> Next Monday
    sun = datetime(2026, 3, 1, 12, 0, 0)
    assert _session_report_date(sun) == "2026-03-02"
