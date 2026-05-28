from __future__ import annotations

from datetime import datetime, timezone

from autotrade.core.microstructure import MicrostructureEventDetector
from autotrade.data_ingestion.stream_bridge import QuoteStreamEvent


def _quote(*, bid_size: float, ask_size: float, bid_price: float = 100.0, ask_price: float = 100.1, second: int = 0):
    return QuoteStreamEvent(
        symbol="AAPL",
        bid_price=bid_price,
        ask_price=ask_price,
        bid_size=bid_size,
        ask_size=ask_size,
        timestamp=datetime(2026, 3, 7, 15, 10, second, tzinfo=timezone.utc),
    )


def test_microstructure_event_detector_detects_bid_imbalance_spike():
    detector = MicrostructureEventDetector(window_size=4, imbalance_spike_threshold=0.15)

    assert detector.ingest_quote(_quote(bid_size=100, ask_size=100, second=0)) == []
    assert detector.ingest_quote(_quote(bid_size=110, ask_size=100, second=1)) == []
    assert detector.ingest_quote(_quote(bid_size=105, ask_size=100, second=2)) == []

    events = detector.ingest_quote(_quote(bid_size=300, ask_size=80, second=3))

    assert len(events) == 1
    assert events[0].event_type == "bid_imbalance_spike"
    assert events[0].imbalance > events[0].baseline_imbalance


def test_microstructure_event_detector_detects_ask_imbalance_spike():
    detector = MicrostructureEventDetector(window_size=4, imbalance_spike_threshold=0.15)

    detector.ingest_quote(_quote(bid_size=100, ask_size=100, second=0))
    detector.ingest_quote(_quote(bid_size=100, ask_size=110, second=1))
    detector.ingest_quote(_quote(bid_size=100, ask_size=105, second=2))

    events = detector.ingest_quote(_quote(bid_size=80, ask_size=320, second=3))

    assert len(events) == 1
    assert events[0].event_type == "ask_imbalance_spike"
    assert events[0].imbalance < events[0].baseline_imbalance


def test_microstructure_event_detector_requires_minimum_spread():
    detector = MicrostructureEventDetector(
        window_size=4,
        imbalance_spike_threshold=0.15,
        min_spread_bps=15.0,
    )

    detector.ingest_quote(_quote(bid_size=100, ask_size=100, bid_price=100.0, ask_price=100.05, second=0))
    detector.ingest_quote(_quote(bid_size=110, ask_size=100, bid_price=100.0, ask_price=100.05, second=1))
    detector.ingest_quote(_quote(bid_size=105, ask_size=100, bid_price=100.0, ask_price=100.05, second=2))
    events = detector.ingest_quote(_quote(bid_size=300, ask_size=80, bid_price=100.0, ask_price=100.05, second=3))

    assert events == []


def test_microstructure_event_detector_ignores_non_spike_quote_sequence():
    detector = MicrostructureEventDetector(window_size=4, imbalance_spike_threshold=0.15)

    detector.ingest_quote(_quote(bid_size=100, ask_size=100, second=0))
    detector.ingest_quote(_quote(bid_size=110, ask_size=100, second=1))
    detector.ingest_quote(_quote(bid_size=115, ask_size=100, second=2))
    events = detector.ingest_quote(_quote(bid_size=120, ask_size=100, second=3))

    assert events == []
