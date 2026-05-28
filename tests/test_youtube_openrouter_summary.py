import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from tools.youtube_daily_scanner import (
    _build_synthesis_prompt,
    _extract_and_validate_report_json,
    _resolve_synthesis_input_budget_tokens,
    extract_with_llm,
    generate_daily_report,
    ReportSchemaValidationError,
)


def _write_sample_extraction(rag_dir: Path, date_str: str) -> None:
    date_dir = rag_dir / "by_date" / date_str
    date_dir.mkdir(parents=True, exist_ok=True)
    payload = {"channel": "test", "notes": "sample extraction"}
    (date_dir / "test_channel_abc123.json").write_text(json.dumps(payload), encoding="utf-8")


def test_report_schema_defaults_make_sector_bias_optional() -> None:
    parsed = _extract_and_validate_report_json(
        json.dumps({"market_regime": "NEUTRAL", "trading_signals": {}}),
        required_keys=["trading_signals.sector_bias"],
    )

    assert parsed is not None
    assert parsed["trading_signals"]["sector_bias"] == []


def test_generate_daily_report_openrouter_missing_key(tmp_path: Path, monkeypatch) -> None:
    date_str = "2026-02-11"
    rag_dir = tmp_path / "rag"
    _write_sample_extraction(rag_dir, date_str)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("tools.youtube_daily_scanner._get_env_var", lambda _name: None)

    config = {
        "scanner": {
            "extraction": {
                "synthesis_provider": "openrouter",
                "report_model": "stepfun/step-3.5-flash:free",
                "openrouter_api_key_env": "TEST_MISSING_OPENROUTER_KEY",
            }
        }
    }
    result = generate_daily_report(date_str, rag_dir, config)
    assert result is None


def test_generate_daily_report_openrouter_success(tmp_path: Path, monkeypatch) -> None:
    date_str = "2026-02-11"
    rag_dir = tmp_path / "rag"
    _write_sample_extraction(rag_dir, date_str)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "date": date_str,
                                    "executive_summary": "Synthetic summary",
                                    "market_regime": "NEUTRAL",
                                    "trading_signals": {"sector_bias": []},
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }

    def fake_post(*args, **kwargs):  # noqa: ANN002, ANN003
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)

    config = {
        "scanner": {
            "extraction": {
                "synthesis_provider": "openrouter",
                "report_model": "stepfun/step-3.5-flash:free",
                "openrouter_api_key_env": "OPENROUTER_API_KEY",
                "openrouter_url": "https://openrouter.ai/api/v1/chat/completions",
                "openrouter_max_tokens": 512,
            }
        }
    }
    result = generate_daily_report(
        date_str,
        rag_dir,
        config,
        provider_override="openrouter",
        model_override="stepfun/step-3.5-flash:free",
        output_filename=f"{date_str}_consolidated_openrouter_test.json",
    )
    assert result is not None
    assert result.get("market_regime") == "NEUTRAL"
    assert result.get("_meta", {}).get("provider") == "openrouter"
    assert "token_usage" in result.get("_meta", {})


def test_generate_daily_report_defaults_missing_sector_bias(
    tmp_path: Path, monkeypatch
) -> None:
    date_str = "2026-02-11"
    rag_dir = tmp_path / "rag"
    _write_sample_extraction(rag_dir, date_str)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "date": date_str,
                                    "executive_summary": "Synthetic summary",
                                    "market_regime": "NEUTRAL",
                                    "trading_signals": {
                                        "sizing_multiplier": 0.5,
                                        "sizing_rationale": "limited coverage",
                                    },
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }

    monkeypatch.setattr("requests.post", lambda *args, **kwargs: FakeResponse())

    result = generate_daily_report(
        date_str,
        rag_dir,
        {
            "scanner": {
                "extraction": {
                    "synthesis_provider": "openrouter",
                    "report_model": "stepfun/step-3.5-flash:free",
                    "openrouter_api_key_env": "OPENROUTER_API_KEY",
                    "openrouter_url": "https://openrouter.ai/api/v1/chat/completions",
                    "openrouter_max_tokens": 512,
                }
            }
        },
        save_report=False,
    )

    assert result is not None
    assert result["trading_signals"]["sector_bias"] == []


def test_generate_daily_report_skips_failed_extractions(tmp_path: Path, monkeypatch) -> None:
    date_str = "2026-02-11"
    rag_dir = tmp_path / "rag"
    date_dir = rag_dir / "by_date" / date_str
    date_dir.mkdir(parents=True, exist_ok=True)
    (date_dir / "rta_trading_good123.json").write_text(
        json.dumps({"channel": "rta_trading", "notes": "usable extraction"}),
        encoding="utf-8",
    )
    (date_dir / "trade_brigade_bad456.json").write_text(
        json.dumps(
            {
                "raw_extraction": "",
                "transcript_context": "bad transcript",
                "_meta": {"extraction_failed": True, "error": "401 unauthorized"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "date": date_str,
                                    "executive_summary": "Synthetic summary",
                                    "market_regime": "NEUTRAL",
                                    "trading_signals": {"sector_bias": []},
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
            }

    monkeypatch.setattr("requests.post", lambda *args, **kwargs: FakeResponse())

    config = {
        "channels": {"rta_trading": {}, "trade_brigade": {}},
        "scanner": {
            "extraction": {
                "synthesis_provider": "openrouter",
                "report_model": "stepfun/step-3.5-flash:free",
                "openrouter_api_key_env": "OPENROUTER_API_KEY",
                "openrouter_url": "https://openrouter.ai/api/v1/chat/completions",
                "openrouter_max_tokens": 512,
            }
        },
    }
    result = generate_daily_report(date_str, rag_dir, config, save_report=False)

    assert result is not None
    assert result["_meta"]["video_count"] == 1
    assert result["_meta"]["skipped_failed_extractions"] == 1
    assert result["_meta"]["channels_included"] == ["rta_trading"]


def test_generate_daily_report_raises_on_markdown_only_synthesis(
    tmp_path: Path, monkeypatch
) -> None:
    date_str = "2026-02-11"
    rag_dir = tmp_path / "rag"
    _write_sample_extraction(rag_dir, date_str)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [{"message": {"content": "# Daily Report\nNot JSON"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
            }

    monkeypatch.setattr("requests.post", lambda *args, **kwargs: FakeResponse())
    config = {
        "scanner": {
            "extraction": {
                "synthesis_provider": "openrouter",
                "report_model": "stepfun/step-3.5-flash:free",
                "openrouter_api_key_env": "OPENROUTER_API_KEY",
                "openrouter_url": "https://openrouter.ai/api/v1/chat/completions",
                "openrouter_max_tokens": 512,
            }
        }
    }

    try:
        generate_daily_report(date_str, rag_dir, config, save_report=False)
    except ReportSchemaValidationError:
        return
    raise AssertionError("markdown-only report synthesis should fail loudly")


def test_generate_daily_report_includes_prior_evening_excludes_prior_morning(
    tmp_path: Path, monkeypatch
) -> None:
    date_str = "2026-04-24"
    rag_dir = tmp_path / "rag"
    today_dir = rag_dir / "by_date" / date_str
    prior_dir = rag_dir / "by_date" / "2026-04-23"
    today_dir.mkdir(parents=True)
    prior_dir.mkdir(parents=True)
    (today_dir / "mike_market_today456.json").write_text(
        json.dumps({"raw_extraction": "today", "_meta": {"extracted_at": "2026-04-24T09:00:00"}}),
        encoding="utf-8",
    )
    (prior_dir / "trade_brigade_evening123.json").write_text(
        json.dumps({"raw_extraction": "evening", "_meta": {"extracted_at": "2026-04-23T17:30:00"}}),
        encoding="utf-8",
    )
    (prior_dir / "click_capital_morning123.json").write_text(
        json.dumps({"raw_extraction": "morning", "_meta": {"extracted_at": "2026-04-23T10:30:00"}}),
        encoding="utf-8",
    )

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "message": {
                    "content": json.dumps(
                        {"market_regime": "NEUTRAL", "trading_signals": {"sector_bias": []}}
                    )
                }
            }

    monkeypatch.setattr("requests.post", lambda *args, **kwargs: FakeResponse())
    result = generate_daily_report(
        date_str,
        rag_dir,
        {
            "channels": {"mike_market": {}, "trade_brigade": {}, "click_capital": {}},
            "scanner": {
                "extraction": {
                    "synthesis_provider": "ollama",
                    "report_model": "test-model",
                    "synthesis_chunking_enabled": False,
                }
            },
        },
        save_report=False,
    )

    assert result is not None
    assert sorted(result["_meta"]["channels_included"]) == ["mike_market", "trade_brigade"]
    assert result["_meta"]["video_count"] == 2


def test_generate_daily_report_routes_native_gemini_provider(
    tmp_path: Path, monkeypatch
) -> None:
    date_str = "2026-02-11"
    rag_dir = tmp_path / "rag"
    _write_sample_extraction(rag_dir, date_str)
    monkeypatch.setattr("tools.youtube_daily_scanner._get_env_var", lambda _name: "test-key")

    calls = []

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            calls.append({"model": model, "contents": contents, "config": config})
            return SimpleNamespace(
                text=json.dumps(
                    {"market_regime": "NEUTRAL", "trading_signals": {"sector_bias": []}}
                ),
                usage_metadata=SimpleNamespace(
                    prompt_token_count=10,
                    candidates_token_count=5,
                    total_token_count=15,
                ),
            )

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.models = FakeModels()

    monkeypatch.setattr(
        "tools.youtube_daily_scanner.genai",
        SimpleNamespace(Client=FakeClient),
    )

    result = generate_daily_report(
        date_str,
        rag_dir,
        {
            "scanner": {
                "extraction": {
                    "synthesis_provider": "gemini",
                    "report_model": "gemini-pro-latest",
                    "openrouter_max_tokens": 512,
                    "synthesis_chunking_enabled": False,
                }
            }
        },
        save_report=False,
    )

    assert result is not None
    assert result["_meta"]["provider"] == "gemini"
    assert result["_meta"]["model"] == "gemini-pro-latest"
    assert calls and calls[0]["model"] == "gemini-pro-latest"


def test_generate_daily_report_gemini_schema_failure_uses_conservative_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    date_str = "2026-02-11"
    rag_dir = tmp_path / "rag"
    _write_sample_extraction(rag_dir, date_str)
    monkeypatch.setattr("tools.youtube_daily_scanner._get_env_var", lambda _name: "test-key")

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            return SimpleNamespace(
                text="# Daily Report\nGemini returned prose instead of JSON.",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=10,
                    candidates_token_count=5,
                    total_token_count=15,
                ),
            )

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.models = FakeModels()

    monkeypatch.setattr(
        "tools.youtube_daily_scanner.genai",
        SimpleNamespace(Client=FakeClient),
    )

    result = generate_daily_report(
        date_str,
        rag_dir,
        {
            "scanner": {
                "extraction": {
                    "synthesis_provider": "gemini",
                    "report_model": "gemini-pro-latest",
                    "openrouter_max_tokens": 512,
                    "synthesis_chunking_enabled": False,
                }
            }
        },
        save_report=False,
    )

    assert result is not None
    assert result["_schema_fallback"] is True
    assert result["market_regime"] == "NEUTRAL"
    assert result["trading_signals"]["sector_bias"] == []
    assert result["_meta"]["provider"] == "gemini_schema_fallback"


def test_extract_with_llm_routes_native_gemini_provider(monkeypatch) -> None:
    monkeypatch.setattr("tools.youtube_daily_scanner._get_env_var", lambda _name: "test-key")

    calls = []

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            calls.append({"model": model, "contents": contents, "config": config})
            return SimpleNamespace(
                text=json.dumps(
                    {"market_regime": "BULLISH", "notes": "Gemini extraction ok"}
                ),
                usage_metadata=SimpleNamespace(
                    prompt_token_count=11,
                    candidates_token_count=7,
                    total_token_count=18,
                ),
            )

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.models = FakeModels()

    monkeypatch.setattr(
        "tools.youtube_daily_scanner.genai",
        SimpleNamespace(Client=FakeClient),
    )

    out = extract_with_llm(
        transcript="hello world " * 100,
        template_prompt="{transcript}",
        channel_name="Arete Trading",
        video_title="Test Video",
        video_date="20260427",
        config={
            "scanner": {
                "extraction": {
                    "extraction_provider": "gemini",
                    "model": "gemini-flash-latest",
                    "openrouter_max_tokens": 512,
                }
            }
        },
    )

    assert out is not None
    assert out.get("market_regime") == "BULLISH"
    meta = out.get("_meta", {})
    assert meta.get("provider") == "gemini"
    assert meta.get("model") == "gemini-flash-latest"
    assert calls and calls[0]["model"] == "gemini-flash-latest"


def test_extract_with_llm_falls_back_to_openai_after_primary_failures(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def _fail_requests(*args, **kwargs):  # noqa: ANN001, ANN003
        raise RuntimeError("local model unavailable")

    monkeypatch.setattr("requests.post", _fail_requests)

    class _FakeResponse:
        def model_dump(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "market_regime": "RISK_OFF",
                                    "notes": "Recovered via OpenAI",
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            }

    class _FakeCompletions:
        def create(self, **kwargs):  # noqa: ANN003
            return _FakeResponse()

    class _FakeChat:
        def __init__(self) -> None:
            self.completions = _FakeCompletions()

    class _FakeOpenAI:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.chat = _FakeChat()

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    config = {
        "scanner": {
            "extraction": {
                "extraction_provider": "openrouter",
                "model": "stepfun/step-3.5-flash:free",
                "fallback_model": "glm-4.7-flash",
                "local_fallback_model": "glm-4.7-flash",
                "openai_fallback_enabled": True,
                "openai_fallback_extraction_model": "gpt-4.1",
                "timeout": 1,
            }
        }
    }

    out = extract_with_llm(
        transcript="hello world " * 1000,
        template_prompt="{transcript}",
        channel_name="Test Channel",
        video_title="Test Video",
        video_date="20260213",
        config=config,
    )

    assert out is not None
    assert out.get("market_regime") == "RISK_OFF"
    meta = out.get("_meta", {})
    assert meta.get("provider") == "openai"
    assert meta.get("requested_provider") == "openrouter"


def test_resolve_synthesis_input_budget_tokens_openrouter() -> None:
    ext_config = {
        "openrouter_default_input_tokens": 32000,
        "openrouter_max_tokens": 4000,
        "synthesis_input_safety_margin_tokens": 1000,
    }
    budget = _resolve_synthesis_input_budget_tokens(ext_config, "openrouter", "unknown:model")
    assert budget == 27000


def test_resolve_synthesis_input_budget_tokens_openai() -> None:
    ext_config = {
        "openai_default_input_tokens": 120000,
        "openai_fallback_max_tokens": 8000,
        "synthesis_input_safety_margin_tokens": 1000,
    }
    budget = _resolve_synthesis_input_budget_tokens(ext_config, "openai", "gpt-4.1")
    assert budget == 111000


def test_build_synthesis_prompt_applies_truncation() -> None:
    extractions = {"channel_a": {"text": "x" * 5000}}
    prompt = _build_synthesis_prompt(
        extractions=extractions,
        date_str="2026-02-11",
        prior_report={"a": "b" * 5000},
        max_chars_per_channel=120,
        max_prior_chars=120,
    )
    assert "truncated" in prompt

