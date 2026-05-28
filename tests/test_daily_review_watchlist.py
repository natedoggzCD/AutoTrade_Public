from datetime import datetime

import pandas as pd
import pytest

from autotrade.core import daily_review as daily_review_mod
from autotrade.core.daily_review import DailyReview


def test_get_yesterday_close_reads_from_bars_df():
    review = DailyReview.__new__(DailyReview)

    idx = pd.MultiIndex.from_tuples(
        [
            ("TEST", pd.Timestamp("2026-02-12")),
            ("TEST", pd.Timestamp("2026-02-13")),
            ("TEST", pd.Timestamp("2026-02-17")),
        ],
        names=["symbol", "timestamp"],
    )
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=idx)

    class _Bars:
        def __init__(self, data_frame):
            self.df = data_frame

    class _Client:
        def __init__(self, bars):
            self._bars = bars

        def get_stock_bars(self, _request):
            return self._bars

    review.data_client = _Client(_Bars(df))

    # Reference date is 2026-02-18; method should return the most recent close
    # available before that day from the bars response.
    close = review._get_yesterday_close("TEST", datetime(2026, 2, 18))
    assert close == 102.0


@pytest.fixture
def review_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(daily_review_mod, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(daily_review_mod, "PLANS_DIR", tmp_path / "plans")
    monkeypatch.setattr(daily_review_mod, "LOG_DIR", tmp_path / "logs")
    (tmp_path / "research").mkdir(parents=True, exist_ok=True)
    (tmp_path / "plans").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _new_review_stub() -> DailyReview:
    review = DailyReview.__new__(DailyReview)
    review.review_date = datetime(2026, 3, 9).date()
    review.review_date_str = "2026-03-09"
    return review


def test_get_overnight_readiness_reports_ready(review_paths):
    review = _new_review_stub()
    state_path = review_paths / "research" / "overnight_state.json"
    plan_path = review_paths / "plans" / "morning_game_plan_20260309.json"
    state_path.write_text(
        """
{
  "date": "2026-03-09",
  "research_complete": true,
  "workflow_completion": {
    "game_plan_generated": true,
    "target_trade_date": "2026-03-09"
  }
}
""".strip(),
        encoding="utf-8",
    )
    plan_path.write_text('{"date": "2026-03-09"}', encoding="utf-8")

    readiness = review._get_overnight_readiness()

    assert readiness["ran"] is True
    assert readiness["ready"] is True
    assert readiness["status"] == "ran_and_ready"


def test_get_overnight_readiness_reports_incomplete(review_paths):
    review = _new_review_stub()
    state_path = review_paths / "research" / "overnight_state.json"
    state_path.write_text(
        """
{
  "date": "2026-03-09",
  "research_complete": true,
  "workflow_completion": {
    "game_plan_generated": false,
    "target_trade_date": "2026-03-09"
  }
}
""".strip(),
        encoding="utf-8",
    )

    readiness = review._get_overnight_readiness()

    assert readiness["ran"] is True
    assert readiness["ready"] is False
    assert readiness["status"] == "ran_but_incomplete"
    assert "game_plan_generated=false" in readiness["detail"]


def test_get_overnight_readiness_reports_missing_run(review_paths):
    review = _new_review_stub()

    readiness = review._get_overnight_readiness()

    assert readiness["ran"] is False
    assert readiness["ready"] is False
    assert readiness["status"] == "not_run"
