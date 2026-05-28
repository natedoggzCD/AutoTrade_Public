from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from autotrade.analysis import sequential_shadow_runner as runner


class _Pred:
    def __init__(self, event_id: str, action: str):
        self._payload = {
            "event_id": event_id,
            "symbol": "AAPL",
            "event_type": "buy",
            "recommended_action": action,
            "confidence": 0.6,
            "summary_reasoning": "integration-test",
        }

    def to_dict(self):
        return dict(self._payload)


def test_shadow_pipeline_runner_to_report_integration(monkeypatch, tmp_path: Path):
    ready_path = tmp_path / "ready.jsonl"
    pred_path = tmp_path / "predictions.jsonl"
    rpt_path = tmp_path / "report.json"

    ready_rows = [
        {
            "event_id": "evt-buy",
            "symbol": "AAPL",
            "event_type": "buy",
            "baseline_action": "buy",
            "fill_price": 100.0,
            "fill_time": "2026-02-27T10:00:00",
            "trade_id": "trade-buy",
            "context_bundle": {},
        },
        {
            "event_id": "evt-exit",
            "symbol": "AAPL",
            "event_type": "exit",
            "baseline_action": "exit",
            "fill_price": 100.0,
            "fill_time": "2026-02-27T10:30:00",
            "context_bundle": {},
        },
    ]
    for row in ready_rows:
        runner.append_jsonl(ready_path, row)

    monkeypatch.setattr(
        runner,
        "get_config",
        lambda: SimpleNamespace(
            llm=SimpleNamespace(model_medium="deepseek-r1:8b", num_ctx=2048),
            sequential_shadow_eval=SimpleNamespace(
                model="deepseek-r1:8b",
                batch_size=10,
                retry_attempts=1,
                horizon_minutes=120,
            ),
        ),
    )
    monkeypatch.setattr(runner, "ready_path_for_day", lambda day: ready_path)
    monkeypatch.setattr(runner, "prediction_path_for_day", lambda day: pred_path)
    monkeypatch.setattr(runner, "report_path_for_day", lambda day: rpt_path)

    def _fake_infer(row, engine_cfg):
        if row["event_type"] == "buy":
            return _Pred(row["event_id"], "avoid")
        return _Pred(row["event_id"], "hold")

    monkeypatch.setattr(runner, "run_sequential_shadow_inference", _fake_infer)
    monkeypatch.setattr(
        "autotrade.analysis.sequential_outcome_scorer._find_trade_by_id",
        lambda trade_id: {"id": trade_id, "outcome": "loss"}
        if trade_id == "trade-buy"
        else None,
    )
    monkeypatch.setattr(
        "autotrade.analysis.sequential_outcome_scorer._future_profile",
        lambda symbol, start_time_iso, start_price, horizon_minutes: {
            "ret_pct": 1.2,
            "vol_pct": 0.25,
            "max_favorable_pct": 1.4,
            "max_adverse_pct": -0.2,
        },
    )

    report = runner.run_for_day(
        day_str="2026-02-27",
        max_workers=1,
        max_events=0,
        timeout_seconds=5,
    )

    assert report["new_predictions"] == 2
    assert report["summary"]["evaluated_events"] == 2
    assert report["comparative_report"]["total_events"] == 2
    assert "buy" in report["comparative_report"]["by_event_type"]
    assert "exit" in report["comparative_report"]["by_event_type"]

    assert pred_path.exists()
    assert rpt_path.exists()
    saved = json.loads(rpt_path.read_text(encoding="utf-8"))
    assert "comparative_report" in saved
    assert saved["comparative_report"]["total_events"] == 2
