import json
from datetime import datetime

import autotrade.core.day_manager as day_manager_mod
from autotrade.core.day_manager import DayManager


class _LogCapture:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.errors = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(str(msg) % args if args else str(msg))

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(str(msg) % args if args else str(msg))

    def error(self, msg, *args, **kwargs):
        self.errors.append(str(msg) % args if args else str(msg))


def test_run_cycle_persists_trace_and_trips_observe_only(monkeypatch, tmp_path):
    monkeypatch.setattr(day_manager_mod, "LOG_DIR", tmp_path)

    dm = DayManager.__new__(DayManager)
    dm.logger = _LogCapture()
    dm.cycle_count = 78
    dm.current_regime = None
    dm.execution_accounting = None
    dm.signals = []
    dm.dry_run = False
    dm._cycle_recursion_error_streak = 0
    dm._cycle_recursion_breaker_threshold = 2
    dm._cycle_recursion_observe_only = False
    dm._cycle_recursion_last_trace_path = None

    def _fake_inner():
        dm.cycle_count += 1
        if dm.dry_run:
            return {
                "entries": 0,
                "exits": 0,
                "errors": 0,
                "observe_only": True,
                "mode": "observe_only",
            }
        raise RecursionError("maximum recursion depth exceeded while calling a Python object")

    dm._run_cycle_inner = _fake_inner

    first = dm.run_cycle()
    second = dm.run_cycle()
    third = dm.run_cycle()

    assert first["error"].startswith("RecursionError in run_cycle")
    assert second["error"].startswith("RecursionError in run_cycle")
    assert third["observe_only"] is True
    assert third["mode"] == "observe_only"
    assert dm.dry_run is True
    assert dm._cycle_recursion_observe_only is True
    assert dm._cycle_recursion_error_streak == 0

    trace_path = tmp_path / f"recursion_trace_{datetime.now():%Y-%m-%d}.jsonl"
    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 2
    assert rows[0]["cycle"] == 79
    assert rows[1]["cycle"] == 80
    assert rows[0]["frames"]
    assert rows[1]["frames"]
    assert "RecursionError" in rows[0]["traceback_text"]
    assert "RecursionError" in rows[1]["traceback_text"]

