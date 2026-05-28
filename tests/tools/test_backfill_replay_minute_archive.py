from tools.backfill_replay_minute_archive import _extract_from_json, _extract_from_text


def test_extract_from_json_ignores_free_text_noise_tokens():
    payload = {
        "notes": "HELD POSITION ERROR JSONL YAML",
        "missing_symbols": ["HELD", "ERROR", "JSONL"],
        "sample_symbols": [{"symbol": "AAPL"}, {"symbol": "MSFT"}],
    }

    symbols = _extract_from_json(payload)

    assert symbols == {"AAPL", "MSFT"}


def test_extract_from_json_reads_explicit_symbol_fields_only():
    payload = {
        "held_symbols": ["AAPL", "MSFT"],
        "executed_symbols": "NVDA, AMD",
        "metadata": {
            "summary": "this text should not be mined for uppercase words like HELD or FAIL"
        },
    }

    symbols = _extract_from_json(payload)

    assert {"AAPL", "MSFT", "NVDA", "AMD"}.issubset(symbols)
    assert "HELD" not in symbols
    assert "FAIL" not in symbols


def test_extract_from_text_requires_explicit_ticker_patterns():
    text = "Held AAPL and MSFT; symbol: NVDA ticker=TSLA $SPY $HELD"

    symbols = _extract_from_text(text)

    assert symbols == {"NVDA", "TSLA", "SPY"}
