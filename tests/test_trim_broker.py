from __future__ import annotations

from autotrade.risk.trim_broker import TrimBroker


class FakeSessionState:
    def __init__(self):
        self.data = {"trim_history": {}, "trim_count_today": {}}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def update_many(self, values):
        self.data.update(values)


def test_trim_broker_blocks_duplicate_within_cooldown():
    state = FakeSessionState()
    broker = TrimBroker(state, cooldown_min=240, max_per_day=2, aggregate_cap_pct=0.5)

    allowed, reason = broker.propose_trim(
        "ABCD", qty=10, source="risk_gate", reason="profit_take", current_qty=100
    )
    assert allowed
    assert reason == "approved"

    allowed, reason = broker.propose_trim(
        "ABCD", qty=5, source="advisor", reason="fallback", current_qty=90
    )
    assert not allowed
    assert reason.startswith("cooldown_active:")


def test_trim_broker_enforces_max_per_day_and_aggregate_cap():
    state = FakeSessionState()
    broker = TrimBroker(state, cooldown_min=0, max_per_day=2, aggregate_cap_pct=0.5)

    assert broker.propose_trim("ABCD", 20, "a", "first", current_qty=100)[0]
    assert broker.propose_trim("ABCD", 20, "b", "second", current_qty=80)[0]

    allowed, reason = broker.propose_trim(
        "ABCD", 1, "c", "third", urgency="critical", current_qty=60
    )
    assert not allowed
    assert reason == "max_per_day_exceeded:2>=2"

    state = FakeSessionState()
    broker = TrimBroker(state, cooldown_min=0, max_per_day=10, aggregate_cap_pct=0.5)
    assert broker.propose_trim("WXYZ", 40, "a", "first", current_qty=100)[0]
    allowed, reason = broker.propose_trim("WXYZ", 11, "b", "second", current_qty=60)
    assert not allowed
    assert reason == "aggregate_cap_exceeded:0.51>0.50"


def test_trim_broker_raises_aggregate_cap_on_red_book_day():
    state = FakeSessionState()
    broker = TrimBroker(
        state,
        cooldown_min=0,
        max_per_day=10,
        aggregate_cap_pct=0.5,
        aggregate_cap_red_day_max_pct=0.8,
        aggregate_cap_red_day_loss_scale_pct=5.0,
    )

    assert broker.propose_trim("RED", 40, "a", "first", current_qty=100)[0]
    allowed, reason = broker.propose_trim(
        "RED",
        11,
        "b",
        "second",
        current_qty=60,
        book_pnl_pct=-5.0,
    )

    assert allowed, reason


def test_trim_broker_keeps_normal_day_aggregate_cap_flat():
    state = FakeSessionState()
    broker = TrimBroker(
        state,
        cooldown_min=0,
        max_per_day=10,
        aggregate_cap_pct=0.5,
        aggregate_cap_red_day_max_pct=0.8,
        aggregate_cap_red_day_loss_scale_pct=5.0,
    )

    assert broker.propose_trim("FLAT", 40, "a", "first", current_qty=100)[0]
    allowed, reason = broker.propose_trim(
        "FLAT",
        11,
        "b",
        "second",
        current_qty=60,
        book_pnl_pct=0.25,
    )

    assert not allowed
    assert reason == "aggregate_cap_exceeded:0.51>0.50"


def test_trim_broker_profit_take_override_allows_winner_trim_above_aggregate_cap():
    state = FakeSessionState()
    broker = TrimBroker(state, cooldown_min=0, max_per_day=10, aggregate_cap_pct=0.5)

    allowed, reason = broker.propose_trim(
        "ASAN",
        72,
        "risk_gate",
        "profit_take_13pct",
        current_qty=100,
        position_pnl_pct=13.7,
    )

    assert allowed, reason


def test_trim_broker_profit_take_override_requires_profit_threshold():
    state = FakeSessionState()
    broker = TrimBroker(state, cooldown_min=0, max_per_day=10, aggregate_cap_pct=0.5)

    allowed, reason = broker.propose_trim(
        "ASAN",
        72,
        "risk_gate",
        "profit_take_8pct",
        current_qty=100,
        position_pnl_pct=8.0,
    )

    assert not allowed
    assert reason == "aggregate_cap_exceeded:0.72>0.50"
