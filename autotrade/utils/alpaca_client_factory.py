"""Shared Alpaca credential and client initialization helpers."""

from __future__ import annotations

import os
import time
import socket
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Callable

# Suppress Pydantic v2 shadowing warnings from alpaca-py (known harmless SDK issue)
warnings.filterwarnings("ignore", message='Field name "stop_price" shadows an attribute in parent')

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

from config.config_loader import get_config

ALPACA_KEY_ENV_VARS = ("ALPACA_API_KEY", "APCA_API_KEY_ID")
ALPACA_SECRET_ENV_VARS = (
    "ALPACA_SECRET_KEY",
    "APCA_API_SECRET_KEY",
    "ALPACA_API_SECRET",
)
ALPACA_PAPER_ENV_VARS = ("ALPACA_PAPER", "APCA_API_PAPER")


@dataclass(frozen=True)
class AlpacaCredentials:
    api_key: str
    secret_key: str
    paper: bool = True


def _first_non_empty_env(var_names: tuple[str, ...]) -> Optional[str]:
    for name in var_names:
        value = os.environ.get(name)
        if value and str(value).strip():
            return str(value).strip()
    return None


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def resolve_alpaca_credentials(
    *,
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    paper: Optional[bool] = None,
    allow_config_fallback: bool = True,
    require: bool = False,
) -> Optional[AlpacaCredentials]:
    """
    Resolve Alpaca credentials from explicit args, env vars, then config.

    Supports both project env names and Alpaca-native APCA_* aliases.
    """
    resolved_key = api_key or _first_non_empty_env(ALPACA_KEY_ENV_VARS)
    resolved_secret = secret_key or _first_non_empty_env(ALPACA_SECRET_ENV_VARS)
    env_paper = _first_non_empty_env(ALPACA_PAPER_ENV_VARS)
    resolved_paper = (
        _parse_bool(paper, True)
        if paper is not None
        else _parse_bool(env_paper, True)
    )

    if allow_config_fallback and (
        not resolved_key or not resolved_secret or paper is None
    ):
        try:
            cfg = get_config().alpaca
            resolved_key = resolved_key or getattr(cfg, "api_key", None)
            resolved_secret = resolved_secret or getattr(cfg, "secret_key", None)
            if paper is None and env_paper is None:
                resolved_paper = _parse_bool(getattr(cfg, "paper", resolved_paper), True)
        except Exception:
            pass

    if not resolved_key or not resolved_secret:
        if require:
            keys = ", ".join(ALPACA_KEY_ENV_VARS)
            secrets = ", ".join(ALPACA_SECRET_ENV_VARS)
            raise ValueError(
                "Missing Alpaca credentials. Set one of "
                f"[{keys}] and one of [{secrets}] in env or config."
            )
        return None

    return AlpacaCredentials(
        api_key=str(resolved_key),
        secret_key=str(resolved_secret),
        paper=bool(resolved_paper),
    )


def create_trading_client(
    *,
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    paper: Optional[bool] = None,
    validate_connection: bool = False,
    retries: int = 1,
    retry_delay_seconds: float = 1.0,
    backoff_multiplier: float = 2.0,
    max_retry_delay_seconds: float = 30.0,
    logger: Any = None,
    require_credentials: bool = False,
    client_factory: Optional[Callable[..., TradingClient]] = None,
    sleep_fn: Optional[Callable[[float], None]] = None,
) -> Optional[TradingClient]:
    """Create TradingClient with optional connection validation + retries."""
    creds = resolve_alpaca_credentials(
        api_key=api_key, secret_key=secret_key, paper=paper, require=require_credentials
    )
    if creds is None:
        return None

    attempts = max(1, int(retries))
    last_error: Optional[Exception] = None
    factory = client_factory or TradingClient
    sleeper = sleep_fn or time.sleep
    base_delay = max(0.0, float(retry_delay_seconds))
    max_delay = max(base_delay, float(max_retry_delay_seconds))
    backoff = max(1.0, float(backoff_multiplier))

    for attempt in range(1, attempts + 1):
        try:
            client = factory(
                api_key=creds.api_key,
                secret_key=creds.secret_key,
                paper=bool(creds.paper),
            )
            if validate_connection:
                client.get_account()
            return client
        except Exception as exc:
            last_error = exc
            is_dns = _is_dns_error(exc)
            if logger is not None:
                label = "dns_error" if is_dns else "connection_error"
                logger.warning(
                    "TradingClient init attempt %s/%s failed (%s): %s",
                    attempt,
                    attempts,
                    label,
                    exc,
                )
            if attempt < attempts:
                if is_dns:
                    delay = min(base_delay * (backoff ** (attempt - 1)), max_delay)
                else:
                    delay = base_delay
                sleeper(delay)

    if require_credentials and last_error is not None:
        raise RuntimeError(f"Unable to initialize TradingClient: {last_error}") from last_error
    return None


def _is_dns_error(exc: Exception) -> bool:
    if isinstance(exc, socket.gaierror):
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "name or service not known",
            "temporary failure in name resolution",
            "nodename nor servname provided",
            "getaddrinfo failed",
        )
    )


def create_data_client(
    *,
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    paper: Optional[bool] = None,
    require_credentials: bool = False,
) -> Optional[StockHistoricalDataClient]:
    """Create StockHistoricalDataClient from resolved credentials."""
    creds = resolve_alpaca_credentials(
        api_key=api_key, secret_key=secret_key, paper=paper, require=require_credentials
    )
    if creds is None:
        return None
    return StockHistoricalDataClient(
        api_key=creds.api_key,
        secret_key=creds.secret_key,
    )
