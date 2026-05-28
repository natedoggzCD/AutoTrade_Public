from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timedelta
from types import SimpleNamespace

from autotrade.utils import local_data_provider
from autotrade.utils.local_data_provider import DelistedTombstones, LocalDataProvider


def test_delisted_tombstones_mark_skip_and_expire(tmp_path):
    path = tmp_path / "delisted_symbols.json"
    tombstones = DelistedTombstones(path=path, recheck_days=30)

    tombstones.mark_delisted("vvpr", "possibly delisted")
    assert tombstones.is_delisted("VVPR")

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["VVPR"]["ts"] = (datetime.now() - timedelta(days=31)).isoformat()
    path.write_text(json.dumps(payload), encoding="utf-8")

    refreshed = DelistedTombstones(path=path, recheck_days=30)
    assert not refreshed.is_delisted("VVPR")
    assert "VVPR" not in json.loads(path.read_text(encoding="utf-8"))


def test_live_price_does_not_tombstone_when_alpaca_fallback_succeeds(
    tmp_path, monkeypatch
):
    tombstones = DelistedTombstones(path=tmp_path / "delisted_symbols.json")
    monkeypatch.setattr(local_data_provider, "DELISTED_TOMBSTONES", tombstones)

    provider = LocalDataProvider.__new__(LocalDataProvider)
    provider._yf_available = True
    provider._yf = SimpleNamespace(
        Ticker=lambda _ticker: SimpleNamespace(fast_info=SimpleNamespace())
    )

    broker = SimpleNamespace(
        api=SimpleNamespace(
            get_latest_trade=lambda _ticker: SimpleNamespace(price=12.34)
        )
    )
    broker_module = types.ModuleType("autotrade.core.broker")
    broker_module.get_broker = lambda: broker
    monkeypatch.setitem(sys.modules, "autotrade.core.broker", broker_module)

    price = provider._get_live_price("live", mark_empty_as_delisted=True)

    assert price == 12.34
    assert not tombstones.is_delisted("LIVE")
