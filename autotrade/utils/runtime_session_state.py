"""Thread-safe runtime session state persisted across agent restarts."""

import json
import os
import threading
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any, Dict


class RuntimeSessionStateStore:
    """Small JSON state store keyed by the current trading session date."""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._lock = threading.RLock()
        self._state = self._load()

    def _fresh(self) -> Dict[str, Any]:
        return {
            "session_date": date.today().isoformat(),
            "trim_count_today": {},
            "trim_history": {},
            "add_blocked_symbols_today": [],
            "exited_symbols_today": [],
            "scalp_loser_symbols_today": [],
            "scalp_orders_today": 0,
            "scalp_positions": {},
            "cycle_counter": 0,
            "last_exit_times": {},
        }

    def _load(self) -> Dict[str, Any]:
        if not self._path.exists():
            return self._fresh()
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return self._fresh()
        if data.get("session_date") != date.today().isoformat():
            return self._fresh()
        fresh = self._fresh()
        fresh.update(data)
        return fresh

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return deepcopy(self._state.get(key, default))

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._state[key] = deepcopy(value)
            self._flush_locked()

    def update_many(self, values: Dict[str, Any]) -> None:
        with self._lock:
            for key, value in values.items():
                self._state[key] = deepcopy(value)
            self._flush_locked()

    def _flush_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._state, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self._path)
