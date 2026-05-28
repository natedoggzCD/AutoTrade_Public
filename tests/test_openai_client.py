from autotrade.utils.openai_client import OpenAIClient


def test_openai_client_retries_with_max_completion_tokens(monkeypatch):
    client = OpenAIClient(api_key="test-key")
    client._available = True

    calls = []

    class _Choice:
        def __init__(self):
            self.message = type("_Msg", (), {"content": "{\"ok\":true}"})()

    class _Usage:
        prompt_tokens = 10
        completion_tokens = 5

    class _Response:
        choices = [_Choice()]
        usage = _Usage()

    class _Completions:
        @staticmethod
        def create(**kwargs):
            calls.append(dict(kwargs))
            if "max_tokens" in kwargs:
                raise Exception(
                    "Unsupported parameter: 'max_tokens' is not supported with this model. "
                    "Use 'max_completion_tokens' instead."
                )
            return _Response()

    fake_client = type(
        "_FakeClient",
        (),
        {"chat": type("_Chat", (), {"completions": _Completions()})()},
    )()
    monkeypatch.setattr(client, "_get_client", lambda: fake_client)

    response = client.chat(
        "hello",
        system="system",
        model="gpt-4.1-mini",
        temperature=0.1,
        max_tokens=123,
    )

    assert response.success is True
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 123
    assert "max_tokens" not in calls[1]
    assert calls[1]["max_completion_tokens"] == 123


def test_openai_client_retries_without_temperature_when_model_rejects_it(monkeypatch):
    client = OpenAIClient(api_key="test-key")
    client._available = True

    calls = []

    class _Choice:
        def __init__(self):
            self.message = type("_Msg", (), {"content": "{\"ok\":true}"})()

    class _Usage:
        prompt_tokens = 10
        completion_tokens = 5

    class _Response:
        choices = [_Choice()]
        usage = _Usage()

    class _Completions:
        @staticmethod
        def create(**kwargs):
            calls.append(dict(kwargs))
            if "temperature" in kwargs:
                raise Exception(
                    "Unsupported value: 'temperature' does not support 0.1 with this model. "
                    "Only the default (1) value is supported."
                )
            return _Response()

    fake_client = type(
        "_FakeClient",
        (),
        {"chat": type("_Chat", (), {"completions": _Completions()})()},
    )()
    monkeypatch.setattr(client, "_get_client", lambda: fake_client)

    response = client.chat(
        "hello",
        system="system",
        model="gpt-4.1-mini",
        temperature=0.1,
        max_tokens=123,
    )

    assert response.success is True
    assert len(calls) == 2
    assert calls[0]["temperature"] == 0.1
    assert "temperature" not in calls[1]


def test_openai_client_adapts_multiple_unsupported_parameters_in_sequence(monkeypatch):
    client = OpenAIClient(api_key="test-key")
    client._available = True

    calls = []

    class _Choice:
        def __init__(self):
            self.message = type("_Msg", (), {"content": "{\"ok\":true}"})()

    class _Usage:
        prompt_tokens = 10
        completion_tokens = 5

    class _Response:
        choices = [_Choice()]
        usage = _Usage()

    class _Completions:
        @staticmethod
        def create(**kwargs):
            calls.append(dict(kwargs))
            if "temperature" in kwargs:
                raise Exception(
                    "Unsupported value: 'temperature' does not support 0.1 with this model. "
                    "Only the default (1) value is supported."
                )
            if "max_tokens" in kwargs:
                raise Exception(
                    "Unsupported parameter: 'max_tokens' is not supported with this model. "
                    "Use 'max_completion_tokens' instead."
                )
            return _Response()

    fake_client = type(
        "_FakeClient",
        (),
        {"chat": type("_Chat", (), {"completions": _Completions()})()},
    )()
    monkeypatch.setattr(client, "_get_client", lambda: fake_client)

    response = client.chat(
        "hello",
        system="system",
        model="gpt-4.1-mini",
        temperature=0.1,
        max_tokens=123,
    )

    assert response.success is True
    assert len(calls) == 3
    assert calls[0]["temperature"] == 0.1
    assert calls[0]["max_tokens"] == 123
    assert "temperature" not in calls[1]
    assert calls[1]["max_tokens"] == 123
    assert "temperature" not in calls[2]
    assert "max_tokens" not in calls[2]
    assert calls[2]["max_completion_tokens"] == 123


def test_openai_client_passes_prompt_cache_options(monkeypatch):
    client = OpenAIClient(api_key="test-key")
    client._available = True

    calls = []

    class _Choice:
        def __init__(self):
            self.message = type("_Msg", (), {"content": "{\"ok\":true}"})()

    class _PromptDetails:
        cached_tokens = 7

    class _Usage:
        prompt_tokens = 10
        completion_tokens = 5
        prompt_tokens_details = _PromptDetails()

    class _Response:
        choices = [_Choice()]
        usage = _Usage()

    class _Completions:
        @staticmethod
        def create(**kwargs):
            calls.append(dict(kwargs))
            return _Response()

    fake_client = type(
        "_FakeClient",
        (),
        {"chat": type("_Chat", (), {"completions": _Completions()})()},
    )()
    monkeypatch.setattr(client, "_get_client", lambda: fake_client)

    response = client.chat(
        "hello",
        system="system",
        model="gpt-4.1-mini",
        prompt_cache_key="decision_claw:market_state:v1",
        prompt_cache_retention="24h",
    )

    assert response.success is True
    assert calls[0]["prompt_cache_key"] == "decision_claw:market_state:v1"
    assert calls[0]["prompt_cache_retention"] == "24h"
    assert response.input_tokens == 10
    assert response.output_tokens == 5
    assert response.cached_input_tokens == 7
