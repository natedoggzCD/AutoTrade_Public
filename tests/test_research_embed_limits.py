import httpx
import pytest

from tools.mcp import research_mcp_server as rms


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        request = httpx.Request("POST", "http://localhost:11434/api/embed")
        response = httpx.Response(self.status_code, request=request, text=self.text)
        raise httpx.HTTPStatusError("fake embed failure", request=request, response=response)


def test_chunk_text_force_code_splits_minified_source_under_embed_cap():
    minified = "function x(){return 1;}" * 1200

    chunks = rms.chunk_text(minified, lang="javascript", force_code=True)

    assert len(chunks) > 1
    assert all(chunk.startswith("[CODE:javascript]\n") for chunk in chunks)
    assert all(len(chunk) <= rms.EMBED_INPUT_MAX_CHARS for chunk in chunks)


def test_embed_texts_recursively_splits_oversized_batches(monkeypatch):
    calls: list[tuple[int, int]] = []

    def fake_post(_url, json=None, timeout=None):  # noqa: A002
        inputs = list(json["input"])
        total_chars = sum(len(text) for text in inputs)
        calls.append((len(inputs), total_chars))
        if len(inputs) > 1 and total_chars > 50:
            return _FakeResponse(
                400, text='{"error":"the input length exceeds the context length"}'
            )
        return _FakeResponse(
            200,
            payload={"embeddings": [[float(idx + 1)] for idx, _ in enumerate(inputs)]},
        )

    monkeypatch.setattr(rms.httpx, "post", fake_post)

    embeds = rms.embed_texts(["a" * 30, "b" * 30, "c" * 30], model="fake")

    assert len(embeds) == 3
    assert all(embeds)
    assert any(size > 1 for size, _ in calls)
    assert any(size == 1 for size, _ in calls)


def test_embed_texts_fails_closed_for_single_oversized_input(monkeypatch):
    def fake_post(_url, json=None, timeout=None):  # noqa: A002
        return _FakeResponse(
            400, text='{"error":"the input length exceeds the context length"}'
        )

    monkeypatch.setattr(rms.httpx, "post", fake_post)

    with pytest.raises(RuntimeError, match="too large"):
        rms.embed_texts(["x" * 500], model="fake")
