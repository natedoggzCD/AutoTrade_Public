import requests

from autotrade.reasoning.sequential_engine import (
    SequentialEngineConfig,
    run_sequential_shadow_inference,
)


class _DummyResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_qwen3_sets_think_false(monkeypatch):
    seen = {}

    class _Cfg:
        ollama_url = "http://test.local/api/chat"
        model_medium = "deepseek-r1:8b"
        num_ctx = 2048

    def _fake_cfg():
        return _Cfg()

    def _fake_post(url, json=None, timeout=None):
        seen["url"] = url
        seen["json"] = json or {}
        return _DummyResp(
            {
                "message": {
                    "content": (
                        '{"recommended_action":"hold","confidence":0.7,'
                        '"summary_reasoning":"ok","thoughts":[{"thought":"a","thoughtNumber":1,"totalThoughts":2,"nextThoughtNeeded":true},{"thought":"b","thoughtNumber":2,"totalThoughts":2,"nextThoughtNeeded":false}]}'
                    )
                }
            }
        )

    monkeypatch.setattr(
        "autotrade.reasoning.sequential_engine.get_llm_config", _fake_cfg
    )
    monkeypatch.setattr("autotrade.reasoning.sequential_engine.requests.post", _fake_post)

    pred = run_sequential_shadow_inference(
        {
            "event_id": "e1",
            "symbol": "AAPL",
            "event_type": "buy",
            "baseline_action": "buy",
            "context_bundle": {"foo": "bar"},
        },
        SequentialEngineConfig(model="qwen3:8b", timeout_seconds=10),
    )
    assert pred.event_id == "e1"
    assert pred.recommended_action in {"buy", "hold", "avoid"}
    assert seen["json"].get("think") is False


def test_timeout_returns_fail_open(monkeypatch):
    class _Cfg:
        ollama_url = "http://test.local/api/chat"
        model_medium = "deepseek-r1:8b"
        num_ctx = 2048

    def _fake_cfg():
        return _Cfg()

    def _fake_post(url, json=None, timeout=None):
        raise requests.Timeout("timeout")

    monkeypatch.setattr(
        "autotrade.reasoning.sequential_engine.get_llm_config", _fake_cfg
    )
    monkeypatch.setattr("autotrade.reasoning.sequential_engine.requests.post", _fake_post)

    pred = run_sequential_shadow_inference(
        {
            "event_id": "e2",
            "symbol": "MSFT",
            "event_type": "exit",
            "baseline_action": "exit",
            "context_bundle": {},
        },
        SequentialEngineConfig(model="deepseek-r1:8b", timeout_seconds=1),
    )
    assert pred.event_id == "e2"
    assert pred.timed_out is True
    assert pred.recommended_action == "hold"
