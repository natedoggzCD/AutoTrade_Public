from pathlib import Path
from types import SimpleNamespace

from autotrade.utils.position_cache import (
    fetch_positions_with_fallback,
    load_positions_cache,
    save_positions_cache,
)


def test_save_and_load_positions_cache_roundtrip(tmp_path):
    cache_path = tmp_path / "positions_cache.json"
    positions = [
        SimpleNamespace(
            symbol="AAA",
            qty=10,
            avg_entry_price=20.0,
            current_price=21.0,
            unrealized_pl=10.0,
            unrealized_plpc=0.05,
            market_value=210.0,
            cost_basis=200.0,
            change_today=0.01,
        )
    ]

    save_positions_cache(positions, cache_path=cache_path, source="unit-test")
    payload = load_positions_cache(cache_path)

    assert payload["meta"]["source"] == "unit-test"
    assert payload["positions"][0]["symbol"] == "AAA"
    assert payload["positions"][0]["qty"] == 10.0


def test_save_positions_cache_preserves_position_timestamps(tmp_path):
    cache_path = tmp_path / "positions_cache.json"
    positions = [
        SimpleNamespace(
            symbol="AAA",
            qty=10,
            current_price=21.0,
            created_at="2026-04-02T14:30:00+00:00",
            entry_time="2026-04-02T14:25:00+00:00",
        )
    ]

    save_positions_cache(positions, cache_path=cache_path, source="unit-test")
    payload = load_positions_cache(cache_path)

    assert payload["positions"][0]["created_at"] == "2026-04-02T14:30:00+00:00"
    assert payload["positions"][0]["entry_time"] == "2026-04-02T14:25:00+00:00"


def test_fetch_positions_with_fallback_uses_cache(tmp_path):
    cache_path = tmp_path / "positions_cache.json"
    cached_positions = [
        SimpleNamespace(symbol="BBB", qty=5, current_price=30.0),
    ]
    save_positions_cache(cached_positions, cache_path=cache_path, source="unit-test")

    class FailingClient:
        def get_all_positions(self):
            raise RuntimeError("DNS failure")

    result = fetch_positions_with_fallback(
        FailingClient(),
        cache_path=cache_path,
        retries=2,
        retry_delay_seconds=0.0,
        backoff_multiplier=1.0,
        use_cache=True,
    )

    assert result["used_cache"] is True
    assert result["positions"][0].symbol == "BBB"


def test_fetch_positions_with_fallback_live_success(tmp_path):
    cache_path = tmp_path / "positions_cache.json"

    class LiveClient:
        def get_all_positions(self):
            return [SimpleNamespace(symbol="CCC", qty=7, current_price=12.0)]

    result = fetch_positions_with_fallback(
        LiveClient(),
        cache_path=cache_path,
        retries=1,
        retry_delay_seconds=0.0,
        backoff_multiplier=1.0,
        use_cache=True,
    )

    assert result["used_cache"] is False
    assert result["positions"][0].symbol == "CCC"
    assert cache_path.exists()
