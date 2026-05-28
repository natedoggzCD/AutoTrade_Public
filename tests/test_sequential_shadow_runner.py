from __future__ import annotations

from types import SimpleNamespace

from autotrade.analysis import sequential_shadow_runner as runner


class _Pred:
    def __init__(self, event_id: str):
        self.event_id = event_id

    def to_dict(self):
        return {
            "event_id": self.event_id,
            "symbol": "AAPL",
            "event_type": "buy",
            "recommended_action": "hold",
            "confidence": 0.5,
            "summary_reasoning": "ok",
        }


class _Outcome:
    def __init__(self, event_id: str):
        self.event_id = event_id

    def to_dict(self):
        return {
            "event_id": self.event_id,
            "event_type": "buy",
            "sequential_more_accurate": False,
            "score_delta": 0.0,
        }


def _mk_cfg(batch_size: int = 50, retry_attempts: int = 0):
    return SimpleNamespace(
        llm=SimpleNamespace(model_medium="deepseek-r1:8b", num_ctx=2048),
        sequential_shadow_eval=SimpleNamespace(
            model="deepseek-r1:8b",
            batch_size=batch_size,
            retry_attempts=retry_attempts,
        ),
    )


def test_run_for_day_processes_events_in_batches(monkeypatch):
    ready_rows = [
        {
            "event_id": f"e{i}",
            "symbol": "AAPL",
            "event_type": "buy",
            "baseline_action": "buy",
            "context_bundle": {},
        }
        for i in range(5)
    ]
    appended = []

    def _fake_read_jsonl(path):
        if "ready" in str(path):
            return ready_rows
        return []

    monkeypatch.setattr(runner, "get_config", lambda: _mk_cfg(batch_size=2, retry_attempts=0))
    monkeypatch.setattr(runner, "read_jsonl", _fake_read_jsonl)
    monkeypatch.setattr(runner, "append_jsonl", lambda path, row: appended.append(row))
    monkeypatch.setattr(
        runner,
        "run_sequential_shadow_inference",
        lambda row, engine_cfg: _Pred(row["event_id"]),
    )
    monkeypatch.setattr(
        runner,
        "score_event_outcome",
        lambda event, prediction, horizon_minutes=120: _Outcome(event["event_id"]),
    )

    report = runner.run_for_day(
        day_str="2026-02-27",
        max_workers=1,
        max_events=0,
        timeout_seconds=5,
    )

    assert report["new_predictions"] == 5
    assert report["summary"]["evaluated_events"] == 5
    assert report["batches_processed"] == 3
    assert len(appended) == 5


def test_run_for_day_retries_llm_failure(monkeypatch):
    ready_rows = [
        {
            "event_id": "e1",
            "symbol": "AAPL",
            "event_type": "buy",
            "baseline_action": "buy",
            "context_bundle": {},
        }
    ]
    calls = {"n": 0}
    appended = []

    def _fake_infer(row, engine_cfg):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return _Pred(row["event_id"])

    monkeypatch.setattr(runner, "get_config", lambda: _mk_cfg(batch_size=10, retry_attempts=1))
    monkeypatch.setattr(
        runner,
        "read_jsonl",
        lambda path: ready_rows if "ready" in str(path) else [],
    )
    monkeypatch.setattr(runner, "append_jsonl", lambda path, row: appended.append(row))
    monkeypatch.setattr(runner, "run_sequential_shadow_inference", _fake_infer)
    monkeypatch.setattr(
        runner,
        "score_event_outcome",
        lambda event, prediction, horizon_minutes=120: _Outcome(event["event_id"]),
    )

    report = runner.run_for_day(
        day_str="2026-02-27",
        max_workers=1,
        max_events=0,
        timeout_seconds=5,
    )

    assert calls["n"] == 2
    assert report["new_predictions"] == 1
    assert report["errors"] == 0
    assert len(appended) == 1
