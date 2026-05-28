from __future__ import annotations

from datetime import datetime

import pandas as pd

from autotrade.replay.overnight_signal_historical import _ensure_db
from autotrade.replay.overnight_signal_historical import (
    compute_outcomes_from_local_bars,
)


def test_compute_outcomes_from_local_bars_joins_daily_bars_without_mutating_input_columns(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "overnight_daily_bars.duckdb"
    conn = _ensure_db(db_path)
    try:
        conn.execute(
            """
            INSERT INTO daily_bars (
                ticker,
                trade_date,
                open,
                high,
                low,
                close,
                volume,
                source,
                fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "TEST",
                "2026-04-08",
                10.0,
                10.5,
                9.8,
                10.2,
                100000.0,
                "unit_test",
                datetime(2026, 4, 8, 16, 0, 0),
            ],
        )
    finally:
        conn.close()

    monkeypatch.setattr(
        "autotrade.replay.overnight_signal_historical.ensure_daily_bars_cached",
        lambda **_: None,
    )

    signals_df = pd.DataFrame(
        [
            {
                "ticker": "TEST",
                "sig_date": "2026-04-08",
                "score": 87.5,
            }
        ]
    )

    enriched = compute_outcomes_from_local_bars(
        signals_df,
        db_path=db_path,
        fetch_missing=False,
    )

    assert list(enriched["ticker"]) == ["TEST"]
    assert list(enriched["score"]) == [87.5]
    assert float(enriched.loc[0, "open"]) == 10.0
    assert float(enriched.loc[0, "high"]) == 10.5
    assert float(enriched.loc[0, "open_to_high_pct"]) == 5.0
    assert "_row_id" not in enriched.columns
