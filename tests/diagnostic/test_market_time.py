from datetime import datetime, date
from autotrade.utils.market_time import is_weekend

def test_is_weekend_saturday():
    # 2026-02-28 is a Saturday
    dt = datetime(2026, 2, 28, 12, 0, 0)
    assert is_weekend(dt) is True
    assert is_weekend(dt.date()) is True

def test_is_weekend_sunday():
    # 2026-03-01 is a Sunday
    dt = datetime(2026, 3, 1, 12, 0, 0)
    assert is_weekend(dt) is True
    assert is_weekend(dt.date()) is True

def test_is_weekend_monday():
    # 2026-03-02 is a Monday
    dt = datetime(2026, 3, 2, 12, 0, 0)
    assert is_weekend(dt) is False
    assert is_weekend(dt.date()) is False

def test_is_weekend_friday():
    # 2026-02-27 is a Friday
    dt = datetime(2026, 2, 27, 12, 0, 0)
    assert is_weekend(dt) is False
    assert is_weekend(dt.date()) is False
