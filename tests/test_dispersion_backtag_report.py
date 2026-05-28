from __future__ import annotations

from tools.dispersion_backtag_report import (
    build_dispersion_backtag_report,
    build_eod_dispersion_backtag_report,
)


def test_dispersion_backtag_report_flags_missing_trade_metadata():
    report = build_dispersion_backtag_report(
        [{"backtest_results": {"total_trades": 55, "win_rate": 0.56}}]
    )

    assert report["sample_count"] == 0
    assert report["hypothesis_confirmed"] is False
    assert "insufficient per-trade rows" in report["evidence_gap"]


def test_dispersion_backtag_report_reads_unrealized_pl():
    report = build_dispersion_backtag_report(
        {"trades": [{"dispersion_score": 0.9, "unrealized_pl": 12.5}]}
    )

    assert report["buckets"]["high"]["total_pnl"] == 12.5


def test_dispersion_backtag_report_confirms_high_bucket_edge():
    rows = []
    rows.extend({"dispersion_score": 0.9, "pnl": 10.0} for _ in range(12))
    rows.extend({"dispersion_score": 0.9, "pnl": -2.0} for _ in range(2))
    rows.extend({"dispersion_score": 0.1, "pnl": 3.0} for _ in range(8))
    rows.extend({"dispersion_score": 0.1, "pnl": -7.0} for _ in range(12))

    report = build_dispersion_backtag_report({"trades": rows})

    assert report["sample_count"] == 34
    assert report["buckets"]["high"]["trades"] == 14
    assert report["hypothesis_confirmed"] is True


def test_eod_dispersion_backtag_report_uses_trade_rows_and_market_context():
    reviews = [
        {
            "date": "2026-05-15",
            "trades": [
                {"symbol": "WIN", "unrealized_pl": 10.0},
                {"symbol": "LOSE", "unrealized_pl": -2.0},
            ],
        },
        {
            "date": "2026-05-16",
            "trades": [
                {"symbol": "LOW", "unrealized_pl": -3.0},
            ],
        },
    ]
    market = {
        "2026-05-15": {"spy_pct": -0.10, "vix": 27.1},
        "2026-05-16": {"spy_pct": 2.0, "vix": 12.0},
    }

    report = build_eod_dispersion_backtag_report(reviews, market)

    assert report["source"] == "eod_review_trades_with_session_market_context"
    assert report["sample_count"] == 3
    assert report["buckets"]["high"]["trades"] == 2
    assert report["buckets"]["low"]["trades"] == 1
    assert report["bucket_examples"]["high"][0]["date"] == "2026-05-15"
