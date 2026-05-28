"""
MCP Client — Unified interface for calling MCP server tools.

Uses the official 'mcp' Python SDK to communicate with servers via stdio.
"""

import asyncio
import json
import logging
import os
import threading
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from config.config_loader import get_config

MCP_SDK_AVAILABLE = False
MCP_SDK_IMPORT_ERROR: Optional[str] = None

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    MCP_SDK_AVAILABLE = True
except Exception as e:  # pragma: no cover - depends on local SDK/env health
    ClientSession = Any  # type: ignore[assignment]
    StdioServerParameters = Any  # type: ignore[assignment]
    stdio_client = None
    MCP_SDK_IMPORT_ERROR = str(e)

logger = logging.getLogger("AutoTrade.MCPClient")
DEFAULT_MCP_CALL_TIMEOUT_SECONDS = float(
    os.environ.get("AUTOTRADE_MCP_CALL_TIMEOUT_SECONDS", "5.0")
)

class MCPClient:
    """
    Client for interacting with Model Context Protocol (MCP) servers.
    Manages server lifecycles and provides a unified tool-calling interface.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to reuse server connections."""
        if cls._instance is None:
            cls._instance = super(MCPClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.config = get_config().mcp if hasattr(get_config(), 'mcp') else None
        self.project_root = Path(__file__).resolve().parents[2]
        self.opencode_config_path = self.project_root / "opencode.json"
        
        self._initialized = True
        logger.info("MCPClient foundation initialized")
        if not MCP_SDK_AVAILABLE and MCP_SDK_IMPORT_ERROR:
            logger.warning("MCP SDK unavailable; MCP calls will fail-open: %s", MCP_SDK_IMPORT_ERROR)

    async def _get_server_params(self, server_name: str) -> Optional[StdioServerParameters]:
        """Load server parameters from opencode.json."""
        if not MCP_SDK_AVAILABLE:
            return None
        if not self.opencode_config_path.exists():
            logger.error(f"opencode.json not found at {self.opencode_config_path}")
            return None
            
        try:
            with open(self.opencode_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            mcp_config = config.get("mcp", {})
            if server_name not in mcp_config:
                logger.error(f"Server '{server_name}' not defined in opencode.json")
                return None
                
            server_data = mcp_config[server_name]
            if not server_data.get("enabled", True):
                logger.warning(f"Server '{server_name}' is disabled")
                return None
                
            cmd_list = server_data.get("command", [])
            if not cmd_list:
                logger.error(f"No command defined for server '{server_name}'")
                return None
                
            env = {**os.environ, **server_data.get("environment", {})}
            if server_name == "alpaca":
                env.setdefault("PYTHONUTF8", "1")
                env.setdefault("PYTHONIOENCODING", "utf-8")

            return StdioServerParameters(
                command=cmd_list[0],
                args=cmd_list[1:],
                env=env,
            )
        except Exception as e:
            logger.error(f"Error loading server params for '{server_name}': {e}")
            return None

    async def call_tool(self, server_name: str, tool_name: str, **kwargs) -> Any:
        """Call a tool on a specific MCP server."""
        params = await self._get_server_params(server_name)
        if not params:
            return {"error": f"Could not establish session with {server_name}"}

        try:
            logger.debug(f"Calling tool {server_name}.{tool_name} with args: {kwargs}")
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments=kwargs)

            if hasattr(result, 'content'):
                text_parts = [c.text for c in result.content if hasattr(c, 'text')]
                if text_parts:
                    combined = "\n".join(text_parts)
                    try:
                        return json.loads(combined)
                    except json.JSONDecodeError:
                        return combined
                return result.content
            return result
        except Exception as e:
            logger.error(f"Error calling tool {server_name}.{tool_name}: {e}")
            return {"error": str(e)}

    async def read_resource(self, server_name: str, uri: str) -> Any:
        """Read a resource from an MCP server."""
        params = await self._get_server_params(server_name)
        if not params:
            return {"error": f"Could not establish session with {server_name}"}

        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await session.read_resource(uri)
        except Exception as e:
            logger.error(f"Error reading resource {uri} from {server_name}: {e}")
            return {"error": str(e)}

    async def close_all(self):
        """Close all active sessions."""
        return None

# Synchronous wrappers for easier integration
_client = MCPClient()


def _run_coro_sync(coro: Any, timeout_seconds: float) -> Any:
    """Run async work in a dedicated event loop thread for task-safe MCP lifecycles."""

    result: Dict[str, Any] = {}

    def _worker() -> None:
        try:
            result["value"] = asyncio.run(asyncio.wait_for(coro, timeout=timeout_seconds))
        except BaseException as exc:  # pragma: no cover - exercised via caller behavior
            result["error"] = exc

    thread = threading.Thread(target=_worker, name="mcp-client-call", daemon=True)
    thread.start()
    thread.join(timeout_seconds + 1.0)

    if thread.is_alive():
        raise asyncio.TimeoutError()
    if "error" in result:
        raise result["error"]
    return result.get("value")

def call_mcp_tool(server: str, tool: str, **kwargs):
    """Sync wrapper for calling an MCP tool."""
    timeout_seconds = kwargs.pop("_timeout_seconds", DEFAULT_MCP_CALL_TIMEOUT_SECONDS)
    if not MCP_SDK_AVAILABLE:
        return {
            "error": (
                f"MCP SDK unavailable for {server}.{tool}: "
                f"{MCP_SDK_IMPORT_ERROR or 'unknown import failure'}"
            )
        }
    try:
        return _run_coro_sync(_client.call_tool(server, tool, **kwargs), timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(
            "MCP tool call timed out: %s.%s (timeout=%.2fs)",
            server,
            tool,
            timeout_seconds,
        )
        return {
            "error": (
                f"timeout calling MCP tool {server}.{tool} "
                f"after {timeout_seconds:.2f}s"
            )
        }

def mcp_alpaca(tool: str, **kwargs):
    return call_mcp_tool("alpaca", tool, **kwargs)

def mcp_sqlite(tool: str, **kwargs):
    return call_mcp_tool("sqlite", tool, **kwargs)

def mcp_fetch(tool: str, **kwargs):
    # The 'fetch' server often uses the tool name 'fetch'
    actual_tool = tool if tool != "fetch_url" else "fetch"
    return call_mcp_tool("fetch", actual_tool, **kwargs)

def mcp_time(tool: str, **kwargs):
    return call_mcp_tool("time", tool, **kwargs)


def _serialize_mcp_arg(value: Any) -> Any:
    """Best-effort conversion of SDK objects to MCP-safe JSON values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return _serialize_mcp_arg(value.value)
    if isinstance(value, dict):
        return {str(k): _serialize_mcp_arg(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_mcp_arg(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(exclude_none=True)
        except TypeError:
            dumped = value.model_dump()
        return _serialize_mcp_arg(dumped)
    if hasattr(value, "value"):
        try:
            return _serialize_mcp_arg(value.value)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        raw = {k: v for k, v in vars(value).items() if not str(k).startswith("_")}
        if raw:
            return _serialize_mcp_arg(raw)
    return str(value)


def _extract_positional_kwargs(args: tuple[Any, ...]) -> Dict[str, Any]:
    """Convert common positional request objects into kwargs."""
    if not args:
        return {}
    if len(args) == 1:
        first = args[0]
        if isinstance(first, dict):
            return _serialize_mcp_arg(first)
        if hasattr(first, "model_dump") or hasattr(first, "__dict__"):
            serialized = _serialize_mcp_arg(first)
            if isinstance(serialized, dict):
                return serialized
    return {}


def _normalize_alpaca_kwargs(name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Map SDK-style Alpaca request kwargs to MCP tool kwargs."""
    if name in {"get_stock_bars", "get_stock_latest_quote", "get_stock_latest_trade"}:
        symbol_or_symbols = kwargs.get("symbol_or_symbols")
        if symbol_or_symbols is not None and "symbol" not in kwargs and "symbols" not in kwargs:
            if isinstance(symbol_or_symbols, (list, tuple, set)):
                symbols = [str(s).upper() for s in symbol_or_symbols if s]
                if len(symbols) == 1:
                    kwargs["symbol"] = symbols[0]
                elif symbols:
                    kwargs["symbols"] = symbols
            else:
                kwargs["symbol"] = str(symbol_or_symbols).upper()
    return kwargs


class MCPClientProxy:
    """
    A proxy that mimics an SDK client but routes calls to an MCP server.
    Used for drop-in replacement of Alpaca/SQLite SDKs.
    """
    def __init__(self, server_name: str):
        self.server_name = server_name

    def __getattr__(self, name: str):
        def wrapper(*args, **kwargs):
            # Convert SDK-style positional request objects to MCP kwargs.
            positional_kwargs = _extract_positional_kwargs(args)
            for key, value in positional_kwargs.items():
                kwargs.setdefault(key, value)
             
            # Special case for Alpaca methods: map common SDK names to MCP tool names
            tool_name = name
            if self.server_name == "alpaca":
                # get_account() -> get_account_info
                if name == "get_account":
                    tool_name = "get_account_info"
                # get_order_by_id(id) -> get_order(order_id=id)
                elif name == "get_order_by_id":
                    kwargs["order_id"] = args[0] if args else kwargs.get("id")
                    tool_name = "get_order"
                # cancel_order_by_id(id) -> cancel_order(order_id=id)
                elif name == "cancel_order_by_id":
                    kwargs["order_id"] = args[0] if args else kwargs.get("id")
                    tool_name = "cancel_order"
                kwargs = _normalize_alpaca_kwargs(name, kwargs)

            serialized_kwargs = _serialize_mcp_arg(kwargs)
            if not isinstance(serialized_kwargs, dict):
                serialized_kwargs = {}
            return call_mcp_tool(self.server_name, tool_name, **serialized_kwargs)
        return wrapper

