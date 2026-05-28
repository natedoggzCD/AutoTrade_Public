from __future__ import annotations

import asyncio

import pytest

from autotrade.core.orchestrator import AsyncTradingOrchestrator


@pytest.mark.asyncio
async def test_async_trading_orchestrator_runs_both_loops():
    events: list[str] = []

    async def housekeeping_loop():
        await asyncio.sleep(0.02)
        events.append("housekeeping")

    async def fast_track_loop():
        await asyncio.sleep(0.01)
        events.append("fast_track")

    orchestrator = AsyncTradingOrchestrator(
        housekeeping_loop=housekeeping_loop,
        fast_track_loop=fast_track_loop,
    )

    await orchestrator.run_once()

    assert sorted(events) == ["fast_track", "housekeeping"]


@pytest.mark.asyncio
async def test_async_trading_orchestrator_loops_run_independently():
    started = asyncio.Event()
    fast_finished = asyncio.Event()

    async def housekeeping_loop():
        started.set()
        await asyncio.sleep(0.05)

    async def fast_track_loop():
        await started.wait()
        fast_finished.set()
        await asyncio.sleep(0.01)

    orchestrator = AsyncTradingOrchestrator(
        housekeeping_loop=housekeeping_loop,
        fast_track_loop=fast_track_loop,
    )

    task = asyncio.create_task(orchestrator.start())
    await asyncio.wait_for(fast_finished.wait(), timeout=0.2)
    assert orchestrator.running_tasks() >= 1
    await task
