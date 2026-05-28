from __future__ import annotations

import types
from types import SimpleNamespace

from autotrade.core.day_manager import DayManager


def _bind_stub(client):
    stub = SimpleNamespace()
    stub.client = client
    stub._session_inactive_symbols = set()
    stub._last_entry_rejection_reason = ""
    stub._asset_status_rejection_reason = types.MethodType(
        DayManager._asset_status_rejection_reason, stub
    )
    stub._vwap_unavailable_for_preflight = types.MethodType(
        DayManager._vwap_unavailable_for_preflight, stub
    )
    stub._remember_inactive_symbol_reject = types.MethodType(
        DayManager._remember_inactive_symbol_reject, stub
    )
    stub._is_asset_inactive_reject_exception = (
        DayManager._is_asset_inactive_reject_exception
    )
    return stub


def test_asset_status_preflight_blocks_confirmed_inactive_symbol():
    class _Client:
        def get_asset(self, symbol):
            return SimpleNamespace(status="inactive", tradable=False)

    stub = _bind_stub(_Client())

    assert stub._asset_status_rejection_reason("DEAD") == "asset_inactive_candidate"


def test_asset_status_preflight_allows_active_thin_symbol():
    class _Client:
        def get_asset(self, symbol):
            return SimpleNamespace(status="active", tradable=True)

    stub = _bind_stub(_Client())

    assert stub._asset_status_rejection_reason("THIN") == ""


def test_inactive_broker_reject_populates_session_set():
    stub = _bind_stub(SimpleNamespace())

    stub._remember_inactive_symbol_reject("DEAD", RuntimeError("asset is not active"))

    assert "DEAD" in stub._session_inactive_symbols


def test_vwap_preflight_only_triggers_when_vwap_unavailable():
    stub = _bind_stub(SimpleNamespace())

    assert stub._vwap_unavailable_for_preflight({"vwap_unavailable": True}) is True
    assert stub._vwap_unavailable_for_preflight({"vwap": 12.34}) is False
