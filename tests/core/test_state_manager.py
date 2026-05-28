from __future__ import annotations

import asyncio

import pytest

from autotrade.core.state_manager import TradingStateManager


@pytest.mark.asyncio
async def test_trading_state_manager_serializes_concurrent_position_updates():
    state = TradingStateManager(initial_state={"position_size": 0.0})

    async def add_unit():
        await state.mutate("position_size", lambda current: float(current or 0.0) + 1.0)

    await asyncio.gather(*(add_unit() for _ in range(25)))

    assert await state.get("position_size") == 25.0


@pytest.mark.asyncio
async def test_trading_state_manager_snapshot_is_isolated_copy():
    state = TradingStateManager(initial_state={"positions": {"AAPL": {"qty": 10}}})

    snapshot = await state.snapshot()
    snapshot["positions"]["AAPL"]["qty"] = 999

    live_snapshot = await state.snapshot()

    assert live_snapshot["positions"]["AAPL"]["qty"] == 10


@pytest.mark.asyncio
async def test_trading_state_manager_updates_multiple_keys_atomically():
    state = TradingStateManager()

    updated = await state.update(
        {
            "account_equity": 100_000.0,
            "fast_loop_symbols": ["AAPL", "MSFT"],
        }
    )

    assert updated["account_equity"] == 100_000.0
    assert updated["fast_loop_symbols"] == ["AAPL", "MSFT"]
    assert await state.get("account_equity") == 100_000.0
