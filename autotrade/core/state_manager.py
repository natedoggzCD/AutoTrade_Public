from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any, Callable, Mapping, Optional


StateMutator = Callable[[Any], Any]


class TradingStateManager:
    """Async-safe shared state container for orchestration loops."""

    def __init__(self, initial_state: Optional[Mapping[str, Any]] = None) -> None:
        self._lock = asyncio.Lock()
        self._state: dict[str, Any] = deepcopy(dict(initial_state or {}))

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return deepcopy(self._state)

    async def get(self, key: str, default: Any = None) -> Any:
        async with self._lock:
            return deepcopy(self._state.get(key, default))

    async def set(self, key: str, value: Any) -> Any:
        async with self._lock:
            self._state[key] = deepcopy(value)
            return deepcopy(self._state[key])

    async def update(self, values: Mapping[str, Any]) -> dict[str, Any]:
        async with self._lock:
            for key, value in values.items():
                self._state[str(key)] = deepcopy(value)
            return deepcopy(self._state)

    async def mutate(self, key: str, mutator: StateMutator, default: Any = None) -> Any:
        async with self._lock:
            current = deepcopy(self._state.get(key, default))
            updated = mutator(current)
            self._state[key] = deepcopy(updated)
            return deepcopy(updated)

    async def remove(self, key: str, default: Any = None) -> Any:
        async with self._lock:
            value = self._state.pop(key, default)
            return deepcopy(value)
