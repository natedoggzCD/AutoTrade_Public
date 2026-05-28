from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from autotrade.utils import mcp_client


def test_get_server_params_reads_command_args(tmp_path, monkeypatch):
    config_path = tmp_path / "opencode.json"
    config_path.write_text(
        json.dumps(
            {
                "mcp": {
                    "alpaca": {
                        "enabled": True,
                        "command": ["uvx", "alpaca-mcp-server"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    client = mcp_client.MCPClient()
    monkeypatch.setattr(client, "opencode_config_path", config_path)

    params = asyncio.run(client._get_server_params("alpaca"))

    assert params is not None
    assert params.command == "uvx"
    assert params.args == ["alpaca-mcp-server"]
    assert params.env["PYTHONUTF8"] == "1"
    assert params.env["PYTHONIOENCODING"] == "utf-8"


def test_call_mcp_tool_works_inside_running_event_loop(monkeypatch):
    async def _fake_call_tool(server, tool, **kwargs):
        return {"server": server, "tool": tool, "kwargs": kwargs}

    monkeypatch.setattr(mcp_client, "_client", SimpleNamespace(call_tool=_fake_call_tool))

    async def _run():
        return mcp_client.call_mcp_tool("alpaca", "get_account_info", foo="bar")

    result = asyncio.run(_run())

    assert result == {
        "server": "alpaca",
        "tool": "get_account_info",
        "kwargs": {"foo": "bar"},
    }
