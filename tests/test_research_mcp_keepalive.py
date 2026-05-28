import asyncio
import time

from tools.mcp import recall_mcp_server as rcs
from tools.mcp import research_mcp_server as rms


class _Ctx:
    def __init__(self):
        self.progress = []

    async def report_progress(self, step: int, total: int, message: str):
        self.progress.append((step, total, message))


def test_search_github_extract_emits_keepalive_progress(monkeypatch):
    monkeypatch.setattr(rms, "LONG_MCP_KEEPALIVE_INTERVAL_S", 0.01)

    def _fake_impl(
        query: str, search_type: str = "repositories", progress_cb=None
    ) -> str:
        assert query == "weather radar viewer"
        assert search_type == "repositories"
        if progress_cb is not None:
            progress_cb({"step": 0, "total": 1, "message": "indexing repo archive"})
        time.sleep(0.05)
        return "ok"

    monkeypatch.setattr(rms, "_search_github_extract_impl", _fake_impl)

    ctx = _Ctx()
    result = asyncio.run(
        rms.search_github_extract(
            query="weather radar viewer",
            ctx=ctx,
            search_type="repositories",
        )
    )

    assert result == "ok"
    assert len(ctx.progress) >= 2
    assert any("indexing repo archive" in message for _, _, message in ctx.progress)


def test_recall_expand_research_github_uses_sync_impl(monkeypatch):
    calls = []

    def _fake_impl(
        query: str, search_type: str = "repositories", progress_cb=None
    ) -> str:
        calls.append((query, search_type, progress_cb))
        return (
            "# GitHub Search: weather\n\n"
            "## [1] Repo\n"
            "**URL:** https://github.com/example/weather\n\n"
            "Useful content.\n"
        )

    monkeypatch.setattr(rcs, "_search_github_extract_impl", _fake_impl)

    result = rcs.recall_expand_research(
        query="weather radar viewer",
        mode="github",
        github_search_type="repositories",
    )

    assert calls
    assert calls[0][2] is None
    assert "SUCCESS" in result
    assert "https://github.com/example/weather" in result
