from pathlib import Path
from datetime import date

import autotrade.signals.trade_learner as trade_learner


def _make_submitted_trade(order_id: str = "order-1") -> dict:
    return {
        "id": "ABC_20260217_120000",
        "symbol": "ABC",
        "entry_time": "2026-02-17T12:00:00+00:00",
        "entry_price": 10.0,
        "quantity": 100,
        "filled_quantity": 0,
        "submitted_price": 10.1,
        "entry_status": "submitted",
        "order_id": order_id,
        "position_size": 1010.0,
        "exit_time": None,
        "exit_price": None,
        "pnl_dollars": None,
        "pnl_percent": None,
        "pnl_per_hour": None,
        "r_multiple": None,
        "exit_reason": None,
        "outcome": None,
        "hold_time_hours": None,
        "trade_type": "entry",
    }


def _build_journal(tmp_path: Path, monkeypatch) -> trade_learner.TradeJournal:
    monkeypatch.setattr(trade_learner, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(trade_learner, "JOURNAL_FILE", tmp_path / "trade_journal.json")
    monkeypatch.setattr(
        trade_learner, "SIGNAL_FAMILY_FILE", tmp_path / "learned_signal_families.json"
    )
    return trade_learner.TradeJournal()


def test_reconcile_submitted_entry_promotes_to_filled(tmp_path, monkeypatch):
    journal = _build_journal(tmp_path, monkeypatch)
    journal.trades = [_make_submitted_trade("order-filled")]

    result = journal.reconcile_submitted_entries(
        {
            "order-filled": {
                "status": "filled",
                "filled_qty": "34",
                "filled_avg_price": "10.5",
                "filled_at": "2026-02-17T12:05:00+00:00",
            }
        },
        now_ts="2026-02-17T12:06:00+00:00",
    )

    trade = journal.trades[0]
    assert result["updated"] == 1
    assert result["filled"] == 1
    assert trade["entry_status"] == "filled"
    assert trade["quantity"] == 34
    assert trade["filled_quantity"] == 34
    assert trade["entry_price"] == 10.5
    assert trade["entry_time"] == "2026-02-17T12:05:00+00:00"
    assert trade["position_size"] == 357.0
    assert trade["exit_time"] is None


def test_reconcile_submitted_entry_closes_unfilled_terminal(tmp_path, monkeypatch):
    journal = _build_journal(tmp_path, monkeypatch)
    journal.trades = [_make_submitted_trade("order-canceled")]

    result = journal.reconcile_submitted_entries(
        {
            "order-canceled": {
                "status": "canceled",
                "filled_qty": 0,
                "filled_avg_price": 0,
                "filled_at": None,
            }
        },
        now_ts="2026-02-17T12:10:00+00:00",
    )

    trade = journal.trades[0]
    assert result["updated"] == 1
    assert result["closed_unfilled"] == 1
    assert trade["entry_status"] == "canceled"
    assert trade["quantity"] == 0
    assert trade["filled_quantity"] == 0
    assert trade["exit_time"] == "2026-02-17T12:10:00+00:00"
    assert trade["exit_reason"] == "order_canceled"


def test_get_completed_trades_excludes_canceled_entries(tmp_path, monkeypatch):
    journal = _build_journal(tmp_path, monkeypatch)
    filled = _make_submitted_trade("filled-x")
    filled["entry_status"] = "filled"
    filled["exit_time"] = "2026-02-17T15:00:00+00:00"

    canceled = _make_submitted_trade("canceled-x")
    canceled["entry_status"] = "canceled"
    canceled["exit_time"] = "2026-02-17T15:01:00+00:00"

    journal.trades = [filled, canceled]
    completed = journal.get_completed_trades()

    assert len(completed) == 1
    assert completed[0]["order_id"] == "filled-x"


def test_record_exit_tolerates_missing_exit_time_key_and_aware_entry_time(
    tmp_path, monkeypatch
):
    journal = _build_journal(tmp_path, monkeypatch)
    trade = _make_submitted_trade("filled-x")
    trade["entry_status"] = "filled"
    trade["quantity"] = 10
    trade["filled_quantity"] = 10
    trade["entry_price"] = 10.0
    trade["entry_time"] = "2026-02-17T12:00:00+00:00"
    trade.pop("exit_time")
    journal.trades = [trade]

    trade_id = journal.record_exit("ABC", price=11.0, quantity=10, reason="hard_stop")

    assert trade_id == trade["id"]
    assert trade["exit_reason"] == "hard_stop"
    assert trade["exit_time"] is not None


def test_record_trim_persists_execution_metadata(tmp_path, monkeypatch):
    journal = _build_journal(tmp_path, monkeypatch)

    journal.record_trim(
        "CALM",
        shares_sold=2,
        price=87.37,
        original_size=4636.0,
        new_size=4461.26,
        reason="legacy_oversized_trim",
        order_id="ord-trim",
        filled_qty=2,
        submitted_price=87.47,
        execution_status="filled",
        urgency_tier="normal",
        intended_price=87.47,
        decision_price=87.47,
        slippage_bps=-11.4,
        time_to_fill_ms=795,
        replace_count=0,
    )

    trade = journal.trades[-1]
    assert trade["trade_type"] == "trim"
    assert trade["order_id"] == "ord-trim"
    assert trade["filled_quantity"] == 2
    assert trade["submitted_price"] == 87.47
    assert trade["entry_status"] == "filled"
    assert trade["slippage_bps"] == -11.4
    assert trade["time_to_fill_ms"] == 795


def test_record_exit_persists_execution_metadata(tmp_path, monkeypatch):
    journal = _build_journal(tmp_path, monkeypatch)
    trade = _make_submitted_trade("filled-x")
    trade["entry_status"] = "filled"
    trade["quantity"] = 10
    trade["filled_quantity"] = 10
    trade["entry_price"] = 10.0
    trade["entry_time"] = "2026-02-17T12:00:00+00:00"
    journal.trades = [trade]

    trade_id = journal.record_exit(
        "ABC",
        price=11.0,
        quantity=10,
        reason="hard_stop",
        order_id="ord-exit",
        filled_qty=10,
        submitted_price=10.9,
        execution_status="filled",
        slippage_bps=9.2,
        time_to_fill_ms=810,
        replace_count=1,
    )

    assert trade_id == trade["id"]
    assert trade["order_id"] == "ord-exit"
    assert trade["filled_quantity"] == 10
    assert trade["submitted_price"] == 10.9
    assert trade["entry_status"] == "filled"
    assert trade["slippage_bps"] == 9.2
    assert trade["time_to_fill_ms"] == 810


def test_record_exit_prefers_quantity_match_then_newest_lot(tmp_path, monkeypatch):
    journal = _build_journal(tmp_path, monkeypatch)
    older = {
        "id": "ARX_older",
        "symbol": "ARX",
        "entry_time": "2026-05-21T14:00:00+00:00",
        "entry_price": 16.71,
        "quantity": 259,
        "filled_quantity": 259,
        "submitted_price": 16.71,
        "entry_status": "filled",
        "order_id": "arx-old-buy",
        "position_size": 4328.0,
        "exit_time": None,
        "trade_type": "entry",
    }
    newer = {
        "id": "ARX_newer",
        "symbol": "ARX",
        "entry_time": "2026-05-26T15:00:00+00:00",
        "entry_price": 17.48,
        "quantity": 259,
        "filled_quantity": 259,
        "submitted_price": 17.48,
        "entry_status": "filled",
        "order_id": "arx-new-buy",
        "position_size": 4527.0,
        "exit_time": None,
        "trade_type": "entry",
    }
    journal.trades = [older, newer]

    trade_id = journal.record_exit(
        "ARX",
        price=16.90,
        quantity=259,
        reason="manual_cut",
        order_id="arx-sell",
        filled_qty=259,
    )

    assert trade_id == "ARX_newer"
    assert newer["exit_time"] is not None
    assert newer["order_id"] == "arx-sell"
    assert newer["pnl_dollars"] == (16.90 - 17.48) * 259
    assert older["exit_time"] is None
    assert older["order_id"] == "arx-old-buy"


def test_record_entry_rejects_filled_entry_without_valid_price(tmp_path, monkeypatch):
    journal = _build_journal(tmp_path, monkeypatch)

    trade_id = journal.record_entry(
        "VG",
        price=0,
        quantity=151,
        reason="filled_without_price",
        entry_status="filled",
        order_id="vg-buy",
    )

    assert trade_id == ""
    assert journal.trades == []


def test_record_scale_persists_execution_metadata(tmp_path, monkeypatch):
    journal = _build_journal(tmp_path, monkeypatch)

    journal.record_scale(
        "NOG",
        shares_added=35,
        price=29.09,
        new_size=3097.55,
        gain_pct=4.1,
        reason="scale_into_winner",
        order_id="ord-scale",
        filled_qty=35,
        submitted_price=29.09,
        execution_status="filled",
        urgency_tier="normal",
        intended_price=29.09,
        decision_price=29.09,
        slippage_bps=0.0,
        time_to_fill_ms=2293,
        replace_count=0,
    )

    trade = journal.trades[-1]
    assert trade["trade_type"] == "scale"
    assert trade["order_id"] == "ord-scale"
    assert trade["filled_quantity"] == 35
    assert trade["submitted_price"] == 29.09
    assert trade["entry_status"] == "filled"
    assert trade["time_to_fill_ms"] == 2293


def test_record_entry_persists_signal_family_metadata(tmp_path, monkeypatch):
    journal = _build_journal(tmp_path, monkeypatch)
    journal.signal_capture.capture = lambda *args, **kwargs: trade_learner.TradeSignals(
        symbol="ABC",
        timestamp="2026-03-06T10:00:00+00:00",
        trade_type="entry",
        price=10.0,
    )

    trade_id = journal.record_entry(
        "ABC",
        price=10.0,
        quantity=5,
        reason="test",
        signal_context={
            "setup_type": "new_high_breakout",
            "strategy_name": "strat-x",
            "strategy_id": "strat-x-id",
            "max_hold_days": 15,
            "final_score": 76.5,
            "plan_score_source": "pm_plan_2026-03-06.json",
            "strategy_params": {"max_hold_days": 15},
        },
    )

    trade = next(t for t in journal.trades if t["id"] == trade_id)
    assert trade["signal_family"] == "new_high_breakout"
    assert trade["setup_type"] == "new_high_breakout"
    assert trade["strategy_name"] == "strat-x"
    assert trade["strategy_id"] == "strat-x-id"
    assert trade["max_hold_days"] == 15
    assert trade["plan_score"] == 76.5
    assert trade["plan_source"] == "pm_plan_2026-03-06.json"


def test_record_entry_persists_attribution_metadata(tmp_path, monkeypatch):
    journal = _build_journal(tmp_path, monkeypatch)
    journal.signal_capture.capture = lambda *args, **kwargs: trade_learner.TradeSignals(
        symbol="ABC",
        timestamp="2026-03-06T10:00:00+00:00",
        trade_type="entry",
        price=10.0,
    )

    trade_id = journal.record_entry(
        "ABC",
        price=10.0,
        quantity=5,
        reason="test",
        entry_score=82.0,
        signal_context={
            "entry_source": "overnight_plan_full_watchlist",
            "resolved_regime": {"regime": "DISPERSION"},
            "sizing_multiplier": 0.75,
            "conviction_priority_score": 78.5,
        },
    )

    trade = next(t for t in journal.trades if t["id"] == trade_id)
    assert trade["regime_at_entry"] == "DISPERSION"
    assert trade["sizing_multiplier_at_entry"] == 0.75
    assert trade["conviction_at_entry"] == 78.5
    assert trade["conviction_tier_at_entry"] == "high"
    assert trade["scoring_source"] == "overnight_plan_full_watchlist"
    assert trade["signals"]["regime_at_entry"] == "DISPERSION"
    assert trade["signals"]["sizing_multiplier_at_entry"] == 0.75
    assert trade["signals"]["conviction_tier_at_entry"] == "high"


def test_record_entry_preserves_missing_entry_score(tmp_path, monkeypatch):
    journal = _build_journal(tmp_path, monkeypatch)

    trade_id = journal.record_entry(
        "ABC",
        price=10.0,
        quantity=5,
        reason="test",
        entry_score=None,
    )

    trade = next(t for t in journal.trades if t["id"] == trade_id)
    assert trade["entry_score"] is None
    assert trade["entry_score_available"] is False


def test_record_entry_upserts_existing_order_id_instead_of_duplicate(
    tmp_path, monkeypatch
):
    journal = _build_journal(tmp_path, monkeypatch)
    journal.signal_capture.capture = lambda *args, **kwargs: trade_learner.TradeSignals(
        symbol="ABC",
        timestamp="2026-03-06T10:00:00+00:00",
        trade_type="entry",
        price=10.0,
    )
    journal.trades = [_make_submitted_trade("buy-123")]

    trade_id = journal.record_entry(
        "ABC",
        price=10.25,
        quantity=10,
        reason="broker_fill",
        entry_status="filled",
        order_id="buy-123",
        filled_qty=10,
        submitted_price=10.2,
    )

    assert len(journal.trades) == 1
    trade = journal.trades[0]
    assert trade_id == trade["id"]
    assert trade["order_id"] == "buy-123"
    assert trade["entry_status"] == "filled"
    assert trade["entry_price"] == 10.25
    assert trade["quantity"] == 10
    assert trade["filled_quantity"] == 10
    assert trade["side"] == "buy"
    assert trade["action"] == "buy"
    assert trade["qty"] == 10
    assert trade["filled_avg_price"] == 10.25
    assert trade["plan_source"] == "unavailable"
    assert trade["timestamp"]


def test_record_entry_persists_hedge_authority_metadata(tmp_path, monkeypatch):
    journal = _build_journal(tmp_path, monkeypatch)

    trade_id = journal.record_entry(
        "PSQ",
        price=31.5,
        quantity=10,
        reason="hedge_entry",
        entry_score=None,
        signal_context={
            "setup_type": "hedge_entry",
            "entry_source": "hedge_entry",
            "source_bucket": "hedge",
            "trade_classification": "hedge",
            "execution_policy_snapshot_id": "runtime_execution_policy",
            "hedge_authority": {
                "symbol": "PSQ",
                "context": "hedge_entry",
                "decision_source": "hedge_entry",
            },
        },
    )

    trade = next(t for t in journal.trades if t["id"] == trade_id)
    assert trade["trade_classification"] == "hedge"
    assert trade["source_bucket"] == "hedge"
    assert trade["entry_source"] == "hedge_entry"
    assert trade["setup_type"] == "hedge_entry"
    assert trade["execution_policy_snapshot_id"] == "runtime_execution_policy"
    assert trade["hedge_authority"]["symbol"] == "PSQ"


def test_sync_with_broker_orders_backfills_missing_buy(tmp_path, monkeypatch):
    journal = _build_journal(tmp_path, monkeypatch)

    result = journal.sync_with_broker_orders(
        [
            {
                "id": "buy-1",
                "symbol": "ABC",
                "side": "buy",
                "status": "filled",
                "qty": "10",
                "filled_qty": "10",
                "filled_avg_price": "10.25",
                "limit_price": "10.20",
            }
        ]
    )

    assert result["backfilled_buys"] == 1
    trade = journal.trades[-1]
    assert trade["order_id"] == "buy-1"
    assert trade["entry_status"] == "filled"
    assert trade["entry_price"] == 10.25
    assert trade["entry_score"] is None
    assert trade["side"] == "buy"
    assert trade["action"] == "buy"
    assert trade["qty"] == 10
    assert trade["filled_avg_price"] == 10.25
    assert trade["plan_source"] == "unavailable"


def test_sync_with_broker_orders_links_blank_order_id_buy(tmp_path, monkeypatch):
    journal = _build_journal(tmp_path, monkeypatch)
    journal.trades = [
        {
            "id": "EQNR_add",
            "symbol": "EQNR",
            "timestamp": "2026-04-29T11:02:53",
            "entry_time": "2026-04-29T11:02:53",
            "entry_price": 40.085,
            "quantity": 17,
            "qty": 17,
            "filled_quantity": 17,
            "side": "buy",
            "action": "add",
            "order_id": "",
            "entry_status": "filled",
            "exit_time": None,
            "trade_type": "entry",
        }
    ]

    result = journal.sync_with_broker_orders(
        [
            {
                "id": "eqnr-buy-17",
                "symbol": "EQNR",
                "side": "buy",
                "status": "filled",
                "qty": "17",
                "filled_qty": "17",
                "filled_avg_price": "40.06",
            }
        ]
    )

    assert result["linked_existing"] == 1
    assert result["backfilled_buys"] == 0
    assert result["orders_accounted"] == 1
    assert len(journal.trades) == 1
    assert journal.trades[0]["order_id"] == "eqnr-buy-17"
    assert journal.trades[0]["reconciliation_source"] == "broker_sync_link"


def test_sync_with_broker_orders_backfills_missing_sell_exit(tmp_path, monkeypatch):
    journal = _build_journal(tmp_path, monkeypatch)
    journal.trades = [
        {
            "id": "ABC_entry",
            "symbol": "ABC",
            "entry_time": "2026-02-17T12:00:00+00:00",
            "entry_price": 10.0,
            "quantity": 10,
            "filled_quantity": 10,
            "submitted_price": 10.0,
            "entry_status": "filled",
            "order_id": "buy-0",
            "position_size": 100.0,
            "exit_time": None,
            "exit_price": None,
            "pnl_dollars": None,
            "pnl_percent": None,
            "pnl_per_hour": None,
            "r_multiple": None,
            "exit_reason": None,
            "outcome": None,
            "hold_time_hours": None,
            "trade_type": "entry",
        }
    ]

    result = journal.sync_with_broker_orders(
        [
            {
                "id": "sell-1",
                "symbol": "ABC",
                "side": "sell",
                "status": "filled",
                "qty": "10",
                "filled_qty": "10",
                "filled_avg_price": "11.00",
                "limit_price": "10.95",
            }
        ]
    )

    assert result["backfilled_sells"] == 1
    trade = journal.trades[0]
    assert trade["exit_time"] is not None
    assert trade["exit_price"] == 11.0
    assert trade["exit_reason"] == "broker_sync:filled"
    assert trade["trade_type"] == "exit"
    assert trade["side"] == "sell"
    assert trade["action"] == "sell"
    assert trade["qty"] == 10
    assert trade["filled_avg_price"] == 11.0


def test_sync_with_broker_orders_records_observed_exit_without_open_entry(
    tmp_path, monkeypatch
):
    journal = _build_journal(tmp_path, monkeypatch)

    result = journal.sync_with_broker_orders(
        [
            {
                "id": "sell-observed-1",
                "symbol": "RUN",
                "side": "sell",
                "status": "filled",
                "qty": "17",
                "filled_qty": "17",
                "filled_avg_price": "12.34",
                "limit_price": "12.30",
            }
        ]
    )

    assert result["observed_exits"] == 1
    trade = journal.trades[-1]
    assert trade["trade_type"] == "observed_exit"
    assert trade["order_id"] == "sell-observed-1"
    assert trade["reconciled_without_entry"] is True
    assert trade["reconciliation_source"] == "broker_sync"
    assert trade["exit_price"] == 12.34


def test_sync_with_broker_orders_does_not_leak_sell_metadata_between_symbols(
    tmp_path, monkeypatch
):
    journal = _build_journal(tmp_path, monkeypatch)
    journal.trades = [
        {
            "id": "ARX_entry",
            "symbol": "ARX",
            "entry_time": "2026-05-26T15:00:00+00:00",
            "entry_price": 17.48,
            "quantity": 259,
            "filled_quantity": 259,
            "submitted_price": 17.48,
            "entry_status": "filled",
            "order_id": "arx-buy",
            "position_size": 4527.0,
            "exit_time": None,
            "trade_type": "entry",
        },
        {
            "id": "FIG_entry",
            "symbol": "FIG",
            "entry_time": "2026-05-26T15:01:00+00:00",
            "entry_price": 22.75,
            "quantity": 199,
            "filled_quantity": 199,
            "submitted_price": 22.75,
            "entry_status": "filled",
            "order_id": "fig-buy",
            "position_size": 4527.25,
            "exit_time": None,
            "trade_type": "entry",
        },
    ]

    result = journal.sync_with_broker_orders(
        [
            {
                "id": "arx-sell",
                "symbol": "ARX",
                "side": "sell",
                "status": "filled",
                "qty": "259",
                "filled_qty": "259",
                "filled_avg_price": "16.90",
            },
            {
                "id": "fig-sell",
                "symbol": "FIG",
                "side": "sell",
                "status": "filled",
                "qty": "199",
                "filled_qty": "199",
                "filled_avg_price": "22.16",
            },
        ]
    )

    arx, fig = journal.trades
    assert result["backfilled_sells"] == 2
    assert arx["order_id"] == "arx-sell"
    assert arx["filled_quantity"] == 259
    assert arx["exit_price"] == 16.90
    assert fig["order_id"] == "fig-sell"
    assert fig["filled_quantity"] == 199
    assert fig["exit_price"] == 22.16


def test_sync_with_broker_orders_ignores_unfilled_orders(tmp_path, monkeypatch):
    journal = _build_journal(tmp_path, monkeypatch)

    result = journal.sync_with_broker_orders(
        [
            {
                "id": "buy-submitted-1",
                "symbol": "ABC",
                "side": "buy",
                "status": "new",
                "qty": "10",
                "filled_qty": "0",
                "limit_price": "10.20",
            }
        ]
    )

    assert result["ignored_unfilled"] == 1
    assert result["backfilled_buys"] == 0
    assert len(journal.trades) == 0


def test_sync_with_broker_orders_ignores_out_of_session_orders(tmp_path, monkeypatch):
    journal = _build_journal(tmp_path, monkeypatch)

    result = journal.sync_with_broker_orders(
        [
            {
                "id": "sell-yesterday-1",
                "symbol": "RUN",
                "side": "sell",
                "status": "filled",
                "qty": "7",
                "filled_qty": "7",
                "filled_avg_price": "9.80",
                "filled_at": "2026-04-30T20:00:00+00:00",
            }
        ],
        session_date=date(2026, 5, 1),
    )

    assert result["ignored_out_of_session"] == 1
    assert result["observed_exits"] == 0
    assert len(journal.trades) == 0


def test_save_signal_family_performance_includes_open_and_closed_trades(
    tmp_path, monkeypatch
):
    journal = _build_journal(tmp_path, monkeypatch)
    journal.trades = [
        {
            "id": "AAA_1",
            "symbol": "AAA",
            "entry_time": "2026-03-03T14:00:00+00:00",
            "entry_price": 10.0,
            "quantity": 100,
            "entry_status": "filled",
            "trade_type": "entry",
            "setup_type": "new_high_breakout",
            "signal_family": "new_high_breakout",
            "plan_score": 70.0,
            "pnl_percent": 12.0,
            "exit_time": "2026-03-05T14:00:00+00:00",
            "outcome": "win",
            "signals": {"setup_type": "new_high_breakout"},
        },
        {
            "id": "BBB_1",
            "symbol": "BBB",
            "entry_time": "2026-03-05T14:00:00+00:00",
            "entry_price": 20.0,
            "current_price": 21.0,
            "quantity": 50,
            "entry_status": "filled",
            "trade_type": "entry",
            "setup_type": "new_high_breakout",
            "signal_family": "new_high_breakout",
            "plan_score": 65.0,
            "exit_time": None,
            "signals": {"setup_type": "new_high_breakout"},
        },
    ]

    artifact = journal.save_signal_family_performance(
        as_of=trade_learner.datetime(2026, 3, 6, tzinfo=trade_learner.timezone.utc)
    )

    family = artifact["families"][0]
    assert family["signal_family"] == "new_high_breakout"
    assert family["completed_count"] == 1
    assert family["open_count"] == 1
    assert family["closed_win"] == 1
    assert family["open_positive"] == 1


def test_save_signal_family_performance_backfills_legacy_entry_family(
    tmp_path, monkeypatch
):
    journal = _build_journal(tmp_path, monkeypatch)
    journal.trades = [
        {
            "id": "LEGACY_1",
            "symbol": "LEG",
            "entry_time": "2026-03-03T14:00:00+00:00",
            "entry_price": 10.0,
            "current_price": 10.8,
            "quantity": 100,
            "entry_status": "filled",
            "exit_time": None,
            "signals": {
                "entry_reason": "Active entry | score=78 | phase=core_trading",
                "breaking_out": True,
            },
        }
    ]

    artifact = journal.save_signal_family_performance(
        as_of=trade_learner.datetime(2026, 3, 6, tzinfo=trade_learner.timezone.utc)
    )

    family = artifact["families"][0]
    assert family["signal_family"] == "breakout"
    assert family["open_count"] == 1


def test_save_signal_family_performance_excludes_trim_and_scale_rows(
    tmp_path, monkeypatch
):
    journal = _build_journal(tmp_path, monkeypatch)
    journal.trades = [
        {
            "id": "ENTRY_1",
            "symbol": "AAA",
            "entry_time": "2026-03-03T14:00:00+00:00",
            "entry_price": 10.0,
            "current_price": 11.0,
            "quantity": 100,
            "entry_status": "filled",
            "trade_type": "entry",
            "signal_family": "new_high_breakout",
            "setup_type": "new_high_breakout",
            "exit_time": None,
            "signals": {"setup_type": "new_high_breakout"},
        },
        {
            "id": "TRIM_1",
            "symbol": "AAA",
            "trade_type": "trim",
            "shares_sold": 10,
            "entry_status": "filled",
            "reason": "mean_reversion_trim",
            "signals": {"entry_reason": "mean_reversion_trim"},
        },
        {
            "id": "SCALE_1",
            "symbol": "AAA",
            "trade_type": "scale",
            "shares_added": 5,
            "entry_status": "filled",
            "reason": "scale_into_winner",
            "signals": {"entry_reason": "scale_into_winner"},
        },
    ]

    artifact = journal.save_signal_family_performance(
        as_of=trade_learner.datetime(2026, 3, 6, tzinfo=trade_learner.timezone.utc)
    )

    assert artifact["family_count"] == 1
    family = artifact["families"][0]
    assert family["signal_family"] == "new_high_breakout"
    assert family["total_count"] == 1


def test_save_signal_family_performance_excludes_legacy_scale_rows_without_trade_type(
    tmp_path, monkeypatch
):
    journal = _build_journal(tmp_path, monkeypatch)
    journal.trades = [
        {
            "id": "LEGACY_SCALE_1",
            "symbol": "AAA",
            "entry_time": "2026-03-03T14:00:00+00:00",
            "entry_price": 10.0,
            "quantity": 25,
            "entry_status": "filled",
            "signals": {"entry_reason": "scale_into_winner"},
        }
    ]

    artifact = journal.save_signal_family_performance(
        as_of=trade_learner.datetime(2026, 3, 6, tzinfo=trade_learner.timezone.utc)
    )

    assert artifact["family_count"] == 0


def test_save_signal_family_performance_maps_execute_entry_limit_to_active_entry(
    tmp_path, monkeypatch
):
    journal = _build_journal(tmp_path, monkeypatch)
    journal.trades = [
        {
            "id": "LIMIT_1",
            "symbol": "AAA",
            "entry_time": "2026-03-03T14:00:00+00:00",
            "entry_price": 10.0,
            "quantity": 25,
            "current_price": 10.2,
            "entry_status": "filled",
            "signals": {"entry_reason": "execute_entry_limit"},
        }
    ]

    artifact = journal.save_signal_family_performance(
        as_of=trade_learner.datetime(2026, 3, 6, tzinfo=trade_learner.timezone.utc)
    )

    assert artifact["family_count"] == 1
    assert artifact["families"][0]["signal_family"] == "active_entry"


def test_save_signal_family_performance_excludes_hedge_entries(tmp_path, monkeypatch):
    journal = _build_journal(tmp_path, monkeypatch)
    journal.trades = [
        {
            "id": "HEDGE_1",
            "symbol": "PSQ",
            "entry_time": "2026-03-13T14:00:00+00:00",
            "entry_price": 31.5,
            "current_price": 31.7,
            "quantity": 100,
            "entry_status": "filled",
            "trade_type": "entry",
            "setup_type": "hedge_entry",
            "signal_family": "hedge_entry",
            "entry_source": "hedge_entry",
            "source_bucket": "hedge",
            "trade_classification": "hedge",
            "signals": {"setup_type": "hedge_entry"},
            "exit_time": None,
        }
    ]

    artifact = journal.save_signal_family_performance(
        as_of=trade_learner.datetime(2026, 3, 14, tzinfo=trade_learner.timezone.utc)
    )

    assert artifact["family_count"] == 0


def test_sync_with_broker_orders_recovers_entry_score_from_signals_log(
    tmp_path, monkeypatch
):
    """Broker reconciliation should restore entry_score and source provenance
    by reading logs/signals_<fill_date>.json when the main pipeline missed
    the fill (the 2026-05-05+ regression that left 95 entries blank)."""
    import json as _json

    journal = _build_journal(tmp_path, monkeypatch)
    project_dir = tmp_path / "repo"
    (project_dir / "logs").mkdir(parents=True)
    signals_payload = {
        "date": "2026-05-12",
        "signals": [
            {
                "symbol": "XYZ",
                "entry_score": 76.5,
                "final_score": 76.5,
                "has_catalyst": True,
                "entry_source": "overnight_plan_full_watchlist",
                "plan_score_source": "morning_game_plan_20260512.json",
                "source_bucket": "watchlist",
                "scan_type": "momentum_pullback",
            }
        ],
    }
    (project_dir / "logs" / "signals_2026-05-12.json").write_text(
        _json.dumps(signals_payload), encoding="utf-8"
    )
    monkeypatch.setattr(trade_learner, "PROJECT_DIR", project_dir)

    result = journal.sync_with_broker_orders(
        [
            {
                "id": "buy-xyz",
                "symbol": "XYZ",
                "side": "buy",
                "status": "filled",
                "qty": "100",
                "filled_qty": "100",
                "filled_avg_price": "12.50",
                "filled_at": "2026-05-12T14:00:00+00:00",
            }
        ]
    )

    assert result["backfilled_buys"] == 1
    trade = journal.trades[-1]
    assert trade["entry_score"] == 76.5
    assert trade["entry_score_available"] is True
    assert trade["plan_source"] == "morning_game_plan_20260512.json"
    assert trade["entry_source"] == "overnight_plan_full_watchlist"
    assert trade["plan_score_source"] == "morning_game_plan_20260512.json"
    assert trade["source_bucket"] == "watchlist"


def test_sync_with_broker_orders_missing_signals_log_preserves_legacy_behavior(
    tmp_path, monkeypatch
):
    """If no signals_*.json exists for the fill date, the legacy null-score
    path must still hold so we don't fabricate scores."""
    journal = _build_journal(tmp_path, monkeypatch)
    monkeypatch.setattr(trade_learner, "PROJECT_DIR", tmp_path / "empty_repo")

    result = journal.sync_with_broker_orders(
        [
            {
                "id": "buy-no-log",
                "symbol": "ABC",
                "side": "buy",
                "status": "filled",
                "qty": "10",
                "filled_qty": "10",
                "filled_avg_price": "5.00",
                "filled_at": "2026-05-12T14:00:00+00:00",
            }
        ]
    )

    assert result["backfilled_buys"] == 1
    trade = journal.trades[-1]
    assert trade["entry_score"] is None
    assert trade["entry_score_available"] is False
    assert trade["plan_source"] == "unavailable"
