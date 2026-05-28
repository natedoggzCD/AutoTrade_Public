import socket
import types

import pytest

from autotrade.utils import alpaca_client_factory


def _clear_env(monkeypatch):
    for key in (
        alpaca_client_factory.ALPACA_KEY_ENV_VARS
        + alpaca_client_factory.ALPACA_SECRET_ENV_VARS
        + alpaca_client_factory.ALPACA_PAPER_ENV_VARS
    ):
        monkeypatch.delenv(key, raising=False)


def test_create_trading_client_retries_on_dns_failure():
    attempts = {"count": 0}
    sleeps = []

    def failing_factory(*args, **kwargs):
        attempts["count"] += 1
        raise socket.gaierror("Temporary failure in name resolution")

    def record_sleep(delay):
        sleeps.append(delay)

    client = alpaca_client_factory.create_trading_client(
        api_key="test",
        secret_key="secret",
        paper=True,
        validate_connection=True,
        retries=3,
        retry_delay_seconds=0.5,
        backoff_multiplier=2.0,
        max_retry_delay_seconds=5.0,
        client_factory=failing_factory,
        sleep_fn=record_sleep,
    )

    assert client is None
    assert attempts["count"] == 3
    assert sleeps == [0.5, 1.0]


def test_create_trading_client_retries_constant_on_non_dns_error():
    attempts = {"count": 0}
    sleeps = []

    def failing_factory(*args, **kwargs):
        attempts["count"] += 1
        raise RuntimeError("boom")

    def record_sleep(delay):
        sleeps.append(delay)

    client = alpaca_client_factory.create_trading_client(
        api_key="test",
        secret_key="secret",
        paper=True,
        validate_connection=True,
        retries=3,
        retry_delay_seconds=0.5,
        backoff_multiplier=3.0,
        max_retry_delay_seconds=5.0,
        client_factory=failing_factory,
        sleep_fn=record_sleep,
    )

    assert client is None
    assert attempts["count"] == 3
    assert sleeps == [0.5, 0.5]


def test_parse_bool_variants():
    parse_bool = alpaca_client_factory._parse_bool
    assert parse_bool(None, True) is True
    assert parse_bool("false", True) is False
    assert parse_bool("yes", False) is True
    assert parse_bool("junk", False) is False


def test_resolve_alpaca_credentials_env(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("ALPACA_API_KEY", "k1")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s1")
    monkeypatch.setenv("ALPACA_PAPER", "false")

    creds = alpaca_client_factory.resolve_alpaca_credentials(
        allow_config_fallback=False
    )

    assert creds.api_key == "k1"
    assert creds.secret_key == "s1"
    assert creds.paper is False


def test_resolve_alpaca_credentials_config_fallback(monkeypatch):
    _clear_env(monkeypatch)
    fake_cfg = types.SimpleNamespace(
        alpaca=types.SimpleNamespace(api_key="k2", secret_key="s2", paper=False)
    )
    monkeypatch.setattr(alpaca_client_factory, "get_config", lambda: fake_cfg)

    creds = alpaca_client_factory.resolve_alpaca_credentials()

    assert creds.api_key == "k2"
    assert creds.secret_key == "s2"
    assert creds.paper is False


def test_resolve_alpaca_credentials_require_raises(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(alpaca_client_factory, "get_config", lambda: (_ for _ in ()).throw(RuntimeError("no cfg")))

    with pytest.raises(ValueError):
        alpaca_client_factory.resolve_alpaca_credentials(
            allow_config_fallback=False, require=True
        )


def test_create_trading_client_success_with_factory(monkeypatch):
    _clear_env(monkeypatch)

    class StubClient:
        def __init__(self, *args, **kwargs):
            self.checked = False

        def get_account(self):
            self.checked = True

    client = alpaca_client_factory.create_trading_client(
        api_key="k3",
        secret_key="s3",
        paper=True,
        validate_connection=True,
        client_factory=lambda **kwargs: StubClient(),
    )

    assert isinstance(client, StubClient)
    assert client.checked is True


def test_create_data_client_missing_creds(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(alpaca_client_factory, "get_config", lambda: (_ for _ in ()).throw(RuntimeError("no cfg")))

    client = alpaca_client_factory.create_data_client(require_credentials=False)
    assert client is None


def test_create_data_client_with_creds(monkeypatch):
    _clear_env(monkeypatch)

    client = alpaca_client_factory.create_data_client(
        api_key="k4",
        secret_key="s4",
        paper=True,
    )

    assert client is not None
