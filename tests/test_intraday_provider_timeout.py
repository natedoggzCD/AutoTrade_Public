"""H1 upstream hardening: wall-clock timeout wrapper for provider calls.

Validates `_call_with_timeout` in autotrade/utils/intraday_data_provider.py
— the upstream guard that bounds Alpaca/yfinance/MCP calls so a hung
provider can't block the day_manager candidate-scoring sweep.
"""

from __future__ import annotations

import time

import pytest

from autotrade.utils.intraday_data_provider import (
    ProviderCallTimeout,
    _call_with_timeout,
)


def test_returns_result_when_call_completes_in_budget():
    def quick(x):
        return x * 2

    assert _call_with_timeout(quick, 7, _timeout_s=2.0, _label="quick") == 14


def test_raises_timeout_when_call_exceeds_budget():
    def slow():
        time.sleep(5.0)
        return "should-not-return"

    started = time.monotonic()
    with pytest.raises(ProviderCallTimeout):
        _call_with_timeout(slow, _timeout_s=1.0, _label="slow")
    elapsed = time.monotonic() - started
    # Must surface the timeout near the 1s budget, not the 5s sleep.
    assert elapsed < 2.5, (
        f"timeout did not bound the call: elapsed={elapsed:.2f}s"
    )


def test_propagates_inner_exceptions():
    def boom():
        raise ValueError("upstream failure")

    with pytest.raises(ValueError, match="upstream failure"):
        _call_with_timeout(boom, _timeout_s=1.0, _label="boom")


def test_kwargs_passthrough():
    def add(a, b=0):
        return a + b

    assert (
        _call_with_timeout(add, 5, b=3, _timeout_s=1.0, _label="add") == 8
    )
