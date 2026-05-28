import json

import autotrade.signals.trade_learner as trade_learner


def _build_journal(tmp_path, monkeypatch) -> trade_learner.TradeJournal:
    monkeypatch.setattr(trade_learner, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(trade_learner, "JOURNAL_FILE", tmp_path / "trade_journal.json")
    monkeypatch.setattr(
        trade_learner, "SIGNAL_FAMILY_FILE", tmp_path / "learned_signal_families.json"
    )
    return trade_learner.TradeJournal()


def test_record_exit_tolerates_ghost_open_missing_entry_price(tmp_path, monkeypatch):
    """Regression: legacy open trades missing entry_price must not abort exit writes."""
    journal = _build_journal(tmp_path, monkeypatch)
    journal.trades = [
        {
            "id": "VICR_GHOST",
            "symbol": "VICR",
            "entry_time": "2026-03-01T10:00:00",
            "exit_time": None,
            "quantity": 50,
        }
    ]
    result = journal.record_exit(symbol="VICR", price=12.34, quantity=50, reason="test")

    assert result is not None
    journal_path = tmp_path / "trade_journal.json"
    on_disk = json.loads(journal_path.read_text(encoding="utf-8"))
    assert on_disk.get("last_updated") is not None


def test_record_exit_tolerates_ghost_open_missing_entry_time(tmp_path, monkeypatch):
    """Regression: legacy open trades missing entry_time must not abort exit writes."""
    journal = _build_journal(tmp_path, monkeypatch)
    journal.trades = [
        {
            "id": "VICR_PARTIAL_GHOST",
            "symbol": "VICR",
            "entry_price": 12.0,
            "exit_time": None,
            "quantity": 50,
        }
    ]
    result = journal.record_exit(symbol="VICR", price=13.0, quantity=50, reason="test")

    assert result is not None
    journal_path = tmp_path / "trade_journal.json"
    on_disk = json.loads(journal_path.read_text(encoding="utf-8"))
    open_for_vicr = [
        trade
        for trade in on_disk["trades"]
        if trade.get("symbol") == "VICR" and trade.get("exit_time") is None
    ]
    assert len(open_for_vicr) == 0
    assert on_disk.get("last_updated") is not None


def test_record_exit_falls_back_to_observed_when_no_open_entry(tmp_path, monkeypatch):
    """H11: direct record_exit callers (day_manager, autonomous_agent) must not silently drop
    sell-side journal entries when no matching open trade exists. Broker remains source of
    truth; journal mirrors via observed_exit so daily-review tooling sees the sell."""
    journal = _build_journal(tmp_path, monkeypatch)
    journal.trades = []

    result = journal.record_exit(
        symbol="FRO",
        price=21.5,
        quantity=100,
        reason="stop_loss",
        order_id="abc-123",
    )

    assert result is not None
    on_disk = json.loads((tmp_path / "trade_journal.json").read_text(encoding="utf-8"))
    fro_trades = [t for t in on_disk["trades"] if t.get("symbol") == "FRO"]
    assert len(fro_trades) == 1
    assert fro_trades[0]["trade_type"] == "observed_exit"
    assert fro_trades[0]["reconciliation_source"] == "record_exit_fallback"
    assert fro_trades[0]["reconciled_without_entry"] is True


def test_trade_journal_load_repairs_legacy_ghosts(tmp_path, monkeypatch):
    journal_path = tmp_path / "trade_journal.json"
    seed = {
        "trades": [
            {
                "id": "VICR_GHOST",
                "symbol": "VICR",
                "entry_time": "2026-03-01T10:00:00",
                "exit_time": None,
                "quantity": 50,
            },
            {
                "id": "VICR_REAL",
                "symbol": "VICR",
                "entry_time": "2026-03-02T10:00:00",
                "entry_price": 11.11,
                "exit_time": None,
                "quantity": 50,
            },
        ],
        "last_updated": "2026-03-01T10:00:00",
    }
    journal_path.write_text(json.dumps(seed), encoding="utf-8")

    journal = _build_journal(tmp_path, monkeypatch)

    ghost, real = journal.trades
    assert ghost["exit_time"] is not None
    assert ghost["exit_reason"] == "legacy_ghost_cleanup"
    assert ghost["outcome"] == "breakeven"
    assert real["exit_time"] is None
