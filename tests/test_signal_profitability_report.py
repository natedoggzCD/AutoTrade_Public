import json
from datetime import date

from tools.signal_profitability_report import build_report


def test_signal_profitability_report_groups_legacy_and_attributed_rows(tmp_path):
    journal_path = tmp_path / "trade_journal.json"
    reports_dir = tmp_path / "reports"
    journal_path.write_text(
        json.dumps(
            {
                "trades": [
                    {
                        "id": "a",
                        "symbol": "AAA",
                        "trade_type": "entry",
                        "entry_status": "filled",
                        "entry_time": "2026-05-14T10:00:00",
                        "entry_price": 10.0,
                        "quantity": 100,
                        "position_size": 1000.0,
                        "exit_time": "2026-05-15T10:00:00",
                        "pnl_dollars": 120.0,
                        "signals": {
                            "regime_at_entry": "DISPERSION",
                            "conviction_tier_at_entry": "high",
                            "sizing_multiplier_at_entry": 0.75,
                            "scoring_source": "DecisionClaw",
                        },
                    },
                    {
                        "id": "b",
                        "symbol": "BBB",
                        "trade_type": "entry",
                        "entry_status": "filled",
                        "entry_time": "2026-05-15T10:00:00",
                        "entry_price": 20.0,
                        "quantity": 50,
                        "position_size": 1000.0,
                        "exit_time": None,
                        "pnl_dollars": None,
                        "entry_source": "legacy_source",
                        "signals": {},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    outputs = build_report(
        journal_path=journal_path,
        reports_dir=reports_dir,
        since=date(2026, 5, 4),
        report_date=date(2026, 5, 19),
    )

    markdown = outputs["markdown"].read_text(encoding="utf-8")
    csv_text = outputs["csv"].read_text(encoding="utf-8")
    assert "Thursday Cohort" in markdown
    assert "DecisionClaw" in markdown
    assert "120.00" in markdown
    assert "legacy_source" in csv_text
    assert "open_or_unknown_pnl_trades" in csv_text
