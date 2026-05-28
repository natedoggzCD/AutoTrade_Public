import time
from types import SimpleNamespace

from autotrade.core.day_manager import DayManager


def _manager_for_parallel_eval(monkeypatch, workers="4"):
    monkeypatch.setenv("AUTOTRADE_POSITION_EVAL_WORKERS", workers)
    dm = DayManager.__new__(DayManager)
    dm.position_health = {}
    dm._apply_failsafe_triage = lambda pos, health, conviction: health
    dm._estimate_position_conviction = lambda pos, health: float(health["score"])
    return dm


def test_parallel_position_eval_populates_health_in_symbol_order(monkeypatch):
    dm = _manager_for_parallel_eval(monkeypatch, workers="4")
    calls = []

    def calculate_position_health(pos):
        calls.append(pos.symbol)
        if pos.symbol == "BETA":
            time.sleep(0.05)
        return {"action": "hold", "score": len(pos.symbol), "pnl_pct": 1.0}

    dm.calculate_position_health = calculate_position_health
    positions = [
        SimpleNamespace(symbol="BETA"),
        SimpleNamespace(symbol="ALFA"),
        SimpleNamespace(symbol="CASH"),
        SimpleNamespace(symbol="DELT"),
    ]

    results = dm._evaluate_position_health_batch(positions)

    assert sorted(calls) == ["ALFA", "BETA", "CASH", "DELT"]
    assert [pos.symbol for pos, _health, _conviction in results] == [
        "ALFA",
        "BETA",
        "CASH",
        "DELT",
    ]
    assert set(dm.position_health) == {"ALFA", "BETA", "CASH", "DELT"}
    assert dm.position_health["BETA"]["action"] == "hold"


def test_parallel_position_eval_skips_worker_exception(monkeypatch):
    dm = _manager_for_parallel_eval(monkeypatch, workers="3")

    def calculate_position_health(pos):
        if pos.symbol == "FAIL":
            raise RuntimeError("boom")
        return {"action": "watch", "score": 3, "pnl_pct": 0.0}

    dm.calculate_position_health = calculate_position_health
    positions = [
        SimpleNamespace(symbol="OK1"),
        SimpleNamespace(symbol="FAIL"),
        SimpleNamespace(symbol="OK2"),
    ]

    results = dm._evaluate_position_health_batch(positions)

    assert [pos.symbol for pos, _health, _conviction in results] == ["OK1", "OK2"]
    assert set(dm.position_health) == {"OK1", "OK2"}


def test_position_eval_worker_knob_allows_serial(monkeypatch):
    dm = _manager_for_parallel_eval(monkeypatch, workers="1")
    calls = []

    def calculate_position_health(pos):
        calls.append(pos.symbol)
        return {"action": "hold", "score": 1, "pnl_pct": 0.0}

    dm.calculate_position_health = calculate_position_health
    positions = [SimpleNamespace(symbol="B"), SimpleNamespace(symbol="A")]

    results = dm._evaluate_position_health_batch(positions)

    assert calls == ["B", "A"]
    assert [pos.symbol for pos, _health, _conviction in results] == ["A", "B"]
