from datetime import datetime
from pathlib import Path

from autotrade.analysis.sequential_shadow_schema import (
    SequentialShadowQueueEvent,
    append_jsonl,
    prediction_path_for_day,
    queue_path_for_day,
    read_jsonl,
    report_path_for_day,
)


def test_queue_event_creation_and_jsonl_roundtrip(tmp_path: Path):
    event = SequentialShadowQueueEvent.new(
        symbol="AAPL",
        event_type="buy",
        qty=10,
        order_id="abc123",
        due_cycle=5,
        reason="test_reason",
    )
    assert event.symbol == "AAPL"
    assert event.event_type == "buy"
    assert event.qty == 10
    assert event.order_id == "abc123"

    path = tmp_path / "q.jsonl"
    append_jsonl(path, event.to_dict())
    rows = read_jsonl(path)
    assert len(rows) == 1
    assert rows[0]["event_id"] == event.event_id
    assert rows[0]["due_cycle"] == 5


def test_queue_event_from_dict_normalizes_fields():
    payload = {
        "event_id": "evt-1",
        "symbol": "aapl",
        "event_type": "BUY",
        "qty": "10",
        "order_id": "oid-1",
        "due_cycle": "3",
        "fill_confirmed": "true",
        "fill_price": "101.5",
        "fill_qty": "10",
        "metadata": {"source": "test"},
    }
    event = SequentialShadowQueueEvent.from_dict(payload)
    assert event.symbol == "AAPL"
    assert event.event_type == "buy"
    assert event.qty == 10
    assert event.due_cycle == 3
    assert event.fill_qty == 10
    assert event.fill_price == 101.5


def test_path_builders_normalize_datetime_input():
    dt = datetime(2026, 2, 27, 15, 45, 30)
    assert queue_path_for_day(dt).name == "sequential_shadow_queue_2026-02-27.jsonl"
    assert prediction_path_for_day(dt).name == "sequential_shadow_predictions_2026-02-27.jsonl"
    assert report_path_for_day(dt).name == "sequential_shadow_accuracy_2026-02-27.json"
