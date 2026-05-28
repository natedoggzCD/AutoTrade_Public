from datetime import date, timedelta

from autotrade.utils.financial_db import FinancialDB
from tools.trade_reviewer import app as trade_app


def test_summarize_options_rolls_up_volume_and_oi():
    rows = [
        {
            "expiration": "2026-06-19",
            "calls_volume": 1200,
            "puts_volume": 300,
            "calls_oi": 5000,
            "puts_oi": 2500,
            "updated_at": "2026-05-21T14:00:00",
        },
        {
            "expiration": "2026-07-17",
            "calls_volume": 800,
            "puts_volume": 100,
            "calls_oi": 3000,
            "puts_oi": 500,
            "updated_at": "2026-05-21T15:00:00",
        },
    ]

    summary = trade_app._summarize_options(rows)

    assert summary["expiration_count"] == 2
    assert summary["nearest_expiration"] == "2026-06-19"
    assert summary["calls_volume"] == 2000
    assert summary["puts_volume"] == 400
    assert summary["call_put_volume_ratio"] == 5.0
    assert summary["put_call_oi_ratio"] == 0.375
    assert summary["updated_at"] == "2026-05-21T15:00:00"


def test_pivot_statement_rows_groups_metrics_by_period():
    rows = [
        {
            "period_end": "2026-03-31",
            "metric": "Total Revenue",
            "value": 1000,
            "updated_at": "2026-05-01T00:00:00",
        },
        {
            "period_end": "2026-03-31",
            "metric": "Net Income",
            "value": 100,
            "updated_at": "2026-05-01T00:00:00",
        },
        {
            "period_end": "2025-12-31",
            "metric": "Total Revenue",
            "value": 900,
            "updated_at": "2026-04-01T00:00:00",
        },
    ]

    periods = trade_app._pivot_statement_rows(rows)

    assert periods == [
        {
            "period_end": "2026-03-31",
            "Total Revenue": 1000.0,
            "updated_at": "2026-05-01T00:00:00",
            "Net Income": 100.0,
        },
        {
            "period_end": "2025-12-31",
            "Total Revenue": 900.0,
            "updated_at": "2026-04-01T00:00:00",
        },
    ]


def test_earnings_context_splits_recent_and_upcoming(tmp_path):
    db = FinancialDB(db_path=tmp_path / "financial.db")
    today = date.today()
    db.upsert_earnings(
        [
            {
                "ticker": "TEST",
                "earnings_date": (today - timedelta(days=10)).isoformat(),
                "time_of_day": "amc",
                "eps_estimate": 0.1,
                "eps_actual": 0.2,
                "revenue_estimate": None,
                "revenue_actual": None,
                "surprise_pct": 10.0,
                "updated_at": "2026-05-01T00:00:00",
            },
            {
                "ticker": "TEST",
                "earnings_date": (today + timedelta(days=10)).isoformat(),
                "time_of_day": "bmo",
                "eps_estimate": 0.3,
                "eps_actual": None,
                "revenue_estimate": None,
                "revenue_actual": None,
                "surprise_pct": None,
                "updated_at": "2026-05-01T00:00:00",
            },
        ]
    )

    context = trade_app._get_earnings_context(db, "TEST")

    assert context["recent"][0]["eps_actual"] == 0.2
    assert context["upcoming"][0]["eps_estimate"] == 0.3
