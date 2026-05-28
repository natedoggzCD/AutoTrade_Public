from __future__ import annotations

from datetime import datetime

from autotrade.utils.financial_db import FinancialDB


def test_upsert_and_query_balance_sheet(tmp_path):
    db_path = tmp_path / "financial.db"
    db = FinancialDB(db_path=db_path)

    ticker = "AAPL"
    metric = "Total Assets"
    period_end = "2023-12-31"
    value = 1_000_000
    frequency = "annual"

    db.upsert_statements(
        [
            {
                "ticker": ticker,
                "statement_type": "balance_sheet",
                "frequency": frequency,
                "metric": metric,
                "period_end": period_end,
                "value": value,
                "updated_at": datetime.utcnow().isoformat(),
            }
        ]
    )

    result = db.get_latest_metric(ticker, "balance_sheet", metric, frequency)
    assert result is not None
    assert result["value"] == value
