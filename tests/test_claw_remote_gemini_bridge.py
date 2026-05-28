import asyncio
from types import SimpleNamespace

from tools.claw_remote.gemini_bridge import GeminiBridge


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


class _FakeModels:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def generate_content(self, *, model, contents):
        self.calls.append(model)
        response = self.responses[model]
        if isinstance(response, Exception):
            raise response
        return _FakeResponse(response)


def _build_bridge(*, fast_model="gemini-3.1-flash-lite", code_model="gemini-3.1-pro", responses):
    bridge = GeminiBridge.__new__(GeminiBridge)
    bridge.config = SimpleNamespace(
        gemini_model_fast=fast_model,
        gemini_model_code=code_model,
    )
    bridge.client = SimpleNamespace(models=_FakeModels(responses))
    return bridge


def test_classify_intent_falls_back_from_stale_fast_model():
    bridge = _build_bridge(
        responses={
            "gemini-3.1-flash-lite": RuntimeError(
                "404 NOT_FOUND. models/gemini-3.1-flash-lite is not found for API version v1beta"
            ),
            "gemini-2.5-flash-lite": '```json\n{"target_agent":"DIAGNOSTIC","params":{"query":"failure"},"confidence":0.9}\n```',
        }
    )

    result = asyncio.run(bridge.classify_intent("why did remote claw fail?"))

    assert result["target_agent"] == "DIAGNOSTIC"
    assert result["params"]["query"] == "failure"
    assert bridge.client.models.calls == [
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash-lite",
    ]


def test_classify_intent_routes_morning_signal_brief_without_model_call():
    bridge = _build_bridge(
        responses={
            "gemini-3.1-flash-lite": '{"target_agent":"CHAT","params":{},"confidence":0.1}',
        }
    )

    result = asyncio.run(bridge.classify_intent("show me the top 10 signals this morning"))

    assert result["target_agent"] == "SIGNALS"
    assert result["params"]["limit"] == 10
    assert bridge.client.models.calls == []


def test_generate_response_short_circuits_signal_brief_payload():
    bridge = _build_bridge(
        responses={
            "gemini-3.1-flash-lite": "unused",
        }
    )
    bridge.client = None

    raw_result = {
        "id": "cmd-1",
        "status": "completed",
        "result": {
            "status": "ok",
            "result_type": "signal_brief",
            "summary": {"date": "2026-04-16", "returned": 2},
            "signals": [{"symbol": "CENX"}, {"symbol": "PL"}],
        },
    }

    result = asyncio.run(bridge.generate_response({}, raw_result))

    assert "Top 2 signals for 2026-04-16" in result
    assert "CENX" in result


def test_generate_code_change_falls_back_from_stale_code_model():
    bridge = _build_bridge(
        responses={
            "gemini-3.1-pro": RuntimeError(
                "404 NOT_FOUND. models/gemini-3.1-pro is not found for API version v1beta"
            ),
            "gemini-2.5-pro": "print('fixed')\n",
        }
    )

    result = asyncio.run(
        bridge.generate_code_change(
            request="fix the bug",
            file_path="tools/claw_remote/example.py",
            file_content="print('broken')\n",
        )
    )

    assert result == "print('fixed')"
    assert bridge.client.models.calls == [
        "gemini-3.1-pro",
        "gemini-2.5-pro",
    ]
