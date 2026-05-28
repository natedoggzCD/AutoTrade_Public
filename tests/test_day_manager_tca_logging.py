from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

pytest.importorskip("alpaca.trading.client")

from autotrade.core import day_manager as day_manager_mod
from autotrade.core.day_manager import DayManager


def _new_dm_stub() -> DayManager:
    dm = DayManager.__new__(DayManager)
    dm._entry_order_lifecycle = {}
    dm.execution_accounting = None
    dm._now_utc = lambda: datetime(2026, 2, 28, 10, 0, 0)
    dm._safe_float = DayManager._safe_float
    dm._safe_int = DayManager._safe_int
    dm._can_marketable_escalate_entry = lambda **kwargs: True
    return dm


def test_register_entry_order_lifecycle_stores_tca_fields():
    dm = _new_dm_stub()
    dm._register_entry_order_lifecycle(
        order_id="ord-1",
        symbol="aapl",
        planned_entry=100.0,
        urgency_tier="high",
        entry_score=82.0,
        signal_data={},
        replacement_count=0,
        arrival_price=100.1,
        nbbo_snapshot={"bid_price": 100.0, "ask_price": 100.2, "mid_price": 100.1},
        ack_latency_ms=34,
    )
    row = dm._entry_order_lifecycle["ord-1"]
    assert row["arrival_price"] == pytest.approx(100.1)
    assert row["nbbo_bid_price"] == pytest.approx(100.0)
    assert row["nbbo_ask_price"] == pytest.approx(100.2)
    assert row["nbbo_mid_price"] == pytest.approx(100.1)
    assert row["ack_latency_ms"] == 34


def test_record_execution_success_emits_tca_fields(monkeypatch):
    dm = _new_dm_stub()
    emitted = {}

    class _Collector:
        def emit_execution_quality(self, **kwargs):
            emitted.update(kwargs)

    monkeypatch.setattr(day_manager_mod, "MONITORING_AVAILABLE", True)
    monkeypatch.setattr(day_manager_mod, "get_metrics_collector", lambda: _Collector())

    order = SimpleNamespace(
        id="ord-2",
        status="filled",
        filled_qty=10,
        filled_avg_price=100.3,
        metadata={
            "order_type": "limit",
            "arrival_price": 100.0,
            "nbbo_bid_price": 99.9,
            "nbbo_ask_price": 100.1,
            "nbbo_mid_price": 100.0,
            "ack_latency_ms": 27,
        },
    )
    dm._record_execution_success(
        "entry",
        "AAPL",
        10,
        order_id="ord-2",
        context="execute_entry_limit",
        order=order,
    )
    assert emitted["arrival_price"] == pytest.approx(100.0)
    assert emitted["bid_price_at_arrival"] == pytest.approx(99.9)
    assert emitted["ask_price_at_arrival"] == pytest.approx(100.1)
    assert emitted["execution_latency_ms"] == pytest.approx(27.0)
