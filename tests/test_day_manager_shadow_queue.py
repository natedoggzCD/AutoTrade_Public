from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

pytest.importorskip("alpaca.trading.client")

from autotrade.core.day_manager import DayManager


class _Order:
    def __init__(self, order_id: str):
        self.id = order_id
        self.filled_qty = 5
        self.filled_avg_price = 101.25
        self.filled_at = datetime(2026, 2, 27, 10, 30, 0)


def _new_shadow_stub(tmp_path):
    dm = DayManager.__new__(DayManager)
    dm._sequential_shadow_enabled = True
    dm._sequential_shadow_pending = []
    dm._sequential_shadow_queue_path = tmp_path / "shadow_queue.jsonl"
    dm._sequential_shadow_ready_path = tmp_path / "shadow_ready.jsonl"
    dm._sequential_shadow_day = "2026-02-27"
    dm.cycle_count = 12
    dm.youtube_context = {"regime": "RISK_ON", "sizing_multiplier": 1.15}
    dm._effective_market_regime = lambda: "BULLISH"
    dm.get_current_phase = lambda: SimpleNamespace(value="CORE_TRADING")
    dm._refresh_sequential_shadow_paths = lambda: None
    return dm


@pytest.mark.parametrize("event_type", ["buy", "exit", "trim", "add"])
def test_queue_event_captures_full_context_metadata(tmp_path, event_type):
    dm = _new_shadow_stub(tmp_path)
    dm._queue_sequential_shadow_event(
        symbol="aapl",
        event_type=event_type,
        qty=5,
        order=_Order(f"{event_type}-1"),
        reason=f"{event_type}-reason",
        context=f"{event_type}-context",
        trade_id="trade-123",
        metadata={"source": "unit_test"},
    )

    assert len(dm._sequential_shadow_pending) == 1
    row = dm._sequential_shadow_pending[0]
    assert row["event_type"] == event_type
    assert row["symbol"] == "AAPL"
    assert row["metadata"]["source"] == "unit_test"
    # Full-context metadata should be present for all action types.
    assert row["metadata"]["context"] == f"{event_type}-context"
    assert row["metadata"]["phase"] == "CORE_TRADING"
    assert row["metadata"]["cycle_count"] == 12
    assert row["metadata"]["market_regime"] == "BULLISH"
    assert row["metadata"]["youtube_regime"] == "RISK_ON"
