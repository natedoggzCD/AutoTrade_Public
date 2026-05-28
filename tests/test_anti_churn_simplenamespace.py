from datetime import datetime, timezone
from types import SimpleNamespace

import autotrade.signals.trade_learner as trade_learner
from autotrade.core.day_manager import DayManager


def _make_dm_with_trade_journal(trade_journal):
    dm = object.__new__(DayManager)
    dm.trade_journal = trade_journal
    dm._now_utc = lambda: datetime(2026, 4, 18, 15, 0, tzinfo=timezone.utc)
    return dm


def test_anti_churn_with_simplenamespace_journal_does_not_log_error(caplog):
    dm = _make_dm_with_trade_journal(SimpleNamespace())

    with caplog.at_level("ERROR", logger="autotrade.core.day_manager"):
        result = dm._check_anti_churn_block("SYM", 10.0)

    assert result is None
    assert not any(
        "ANTI-CHURN" in rec.message for rec in caplog.records
    ), "Anti-churn guard must not log ERROR for missing-trades mock journals"


def test_anti_churn_blocks_repeat_buyback_above_last_exit_price():
    dm = _make_dm_with_trade_journal(
        SimpleNamespace(
            trades=[
                {
                    "symbol": "SYM",
                    "action": "exit",
                    "timestamp": "2026-04-18T11:00:00+00:00",
                    "price": 10.0,
                }
            ]
        )
    )

    result = dm._check_anti_churn_block("SYM", 11.0)

    assert result is not None
    assert "ANTI-CHURN: Blocked buyback of SYM" in result


def test_anti_churn_blocks_real_trade_journal_trim_records(tmp_path, monkeypatch):
    monkeypatch.setattr(trade_learner, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(trade_learner, "JOURNAL_FILE", tmp_path / "trade_journal.json")
    monkeypatch.setattr(
        trade_learner, "SIGNAL_FAMILY_FILE", tmp_path / "learned_signal_families.json"
    )
    journal = trade_learner.TradeJournal()
    journal.trades = []
    journal.record_trim(
        "TBI",
        shares_sold=40,
        price=4.41,
        original_size=176.4,
        new_size=0.0,
        reason="mean_reversion_trim",
    )
    dm = object.__new__(DayManager)
    dm.trade_journal = journal
    dm._now_utc = lambda: datetime.now(timezone.utc)

    result = dm._check_anti_churn_block("TBI", 4.61)

    assert result is not None
    assert "ANTI-CHURN: Blocked buyback of TBI" in result
