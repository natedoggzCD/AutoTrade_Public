from datetime import date, datetime

from autotrade.data_ingestion.bootstrap import get_expected_latest_date
from autotrade.utils.data_quality import DataQualityGate
from autotrade.utils.data_sync import DataSyncManager


def test_data_sync_freshness_uses_previous_trading_day_during_monday_session(
    monkeypatch,
):
    sync = DataSyncManager()
    monkeypatch.setattr(
        sync,
        "check_sync_status",
        lambda force_refresh=False: {
            "primary_date": "2026-03-13",
            "primary_tickers": 4809,
            "all_synced": True,
        },
    )

    result = sync.get_data_freshness(reference_dt=datetime(2026, 3, 16, 12, 0))

    assert result["status"] == "OK"
    assert result["is_current"] is True
    assert result["primary_date"] == "2026-03-13"
    assert result["expected_date"] == "2026-03-13"


def test_data_sync_freshness_requires_same_day_after_market_close(monkeypatch):
    sync = DataSyncManager()
    monkeypatch.setattr(
        sync,
        "check_sync_status",
        lambda force_refresh=False: {
            "primary_date": "2026-03-13",
            "primary_tickers": 4809,
            "all_synced": True,
        },
    )

    result = sync.get_data_freshness(reference_dt=datetime(2026, 3, 16, 15, 30))

    assert result["status"] == "STALE"
    assert result["is_current"] is False
    assert result["expected_date"] == "2026-03-16"


def test_data_quality_expected_latest_date_is_session_aware():
    gate = DataQualityGate()

    assert gate.get_expected_latest_date(datetime(2026, 3, 16, 12, 0)) == date(
        2026, 3, 13
    )
    assert gate.get_expected_latest_date(datetime(2026, 3, 16, 15, 30)) == date(
        2026, 3, 16
    )


def test_bootstrap_expected_latest_date_is_session_aware():
    assert get_expected_latest_date(datetime(2026, 3, 16, 12, 0)) == date(2026, 3, 13)
    assert get_expected_latest_date(datetime(2026, 3, 16, 15, 30)) == date(2026, 3, 16)
