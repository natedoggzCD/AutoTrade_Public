import pytest
from datetime import datetime, date
from autotrade.utils.market_time import is_weekend

def test_is_weekend_datetime():
    # 2026-03-07 is Saturday
    saturday = datetime(2026, 3, 7, 12, 0, 0)
    # 2026-03-08 is Sunday
    sunday = datetime(2026, 3, 8, 12, 0, 0)
    # 2026-03-06 is Friday
    friday = datetime(2026, 3, 6, 12, 0, 0)
    # 2026-03-09 is Monday
    monday = datetime(2026, 3, 9, 12, 0, 0)

    assert is_weekend(saturday) is True
    assert is_weekend(sunday) is True
    assert is_weekend(friday) is False
    assert is_weekend(monday) is False

def test_is_weekend_date():
    saturday = date(2026, 3, 7)
    sunday = date(2026, 3, 8)
    friday = date(2026, 3, 6)
    monday = date(2026, 3, 9)

    assert is_weekend(saturday) is True
    assert is_weekend(sunday) is True
    assert is_weekend(friday) is False
    assert is_weekend(monday) is False
