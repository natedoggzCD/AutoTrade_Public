from types import SimpleNamespace

from autotrade.utils import alpaca_client_factory as acf
from config.config_loader import TradingConfig


def test_resolve_alpaca_credentials_supports_apca_alias_env(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("ALPACA_PAPER", raising=False)
    monkeypatch.setenv("APCA_API_KEY_ID", "alias-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "alias-secret")
    monkeypatch.setenv("APCA_API_PAPER", "false")

    creds = acf.resolve_alpaca_credentials(require=True)

    assert creds.api_key == "alias-key"
    assert creds.secret_key == "alias-secret"
    assert creds.paper is False


def test_resolve_alpaca_credentials_falls_back_to_config(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)

    cfg = SimpleNamespace(
        alpaca=SimpleNamespace(api_key="cfg-key", secret_key="cfg-secret", paper=True)
    )
    monkeypatch.setattr(acf, "get_config", lambda: cfg, raising=True)

    creds = acf.resolve_alpaca_credentials(require=True)
    assert creds.api_key == "cfg-key"
    assert creds.secret_key == "cfg-secret"
    assert creds.paper is True


def test_create_trading_client_retries_and_validates(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "env-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "env-secret")

    calls = {"init": 0, "account": 0}

    class _DummyTradingClient:
        def __init__(self, api_key, secret_key, paper=True):
            calls["init"] += 1
            if calls["init"] == 1:
                raise RuntimeError("transient init failure")
            self.api_key = api_key
            self.secret_key = secret_key
            self.paper = paper

        def get_account(self):
            calls["account"] += 1
            return SimpleNamespace(status="ACTIVE")

    monkeypatch.setattr(acf, "TradingClient", _DummyTradingClient, raising=True)

    client = acf.create_trading_client(
        validate_connection=True,
        retries=2,
        retry_delay_seconds=0.0,
        require_credentials=True,
    )

    assert isinstance(client, _DummyTradingClient)
    assert calls["init"] == 2
    assert calls["account"] == 1


def test_config_loader_reads_apca_alias_env(monkeypatch, tmp_path):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("ALPACA_PAPER", raising=False)
    monkeypatch.setenv("APCA_API_KEY_ID", "cfg-alias-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "cfg-alias-secret")
    monkeypatch.setenv("APCA_API_PAPER", "false")

    cfg_file = tmp_path / "trading_config.yaml"
    cfg_file.write_text("{}", encoding="utf-8")

    cfg = TradingConfig.from_yaml(cfg_file)
    assert cfg.alpaca.api_key == "cfg-alias-key"
    assert cfg.alpaca.secret_key == "cfg-alias-secret"
    assert cfg.alpaca.paper is False


def test_config_loader_exposes_backward_compatible_market_data_alias():
    cfg = TradingConfig()

    assert hasattr(cfg, "market_data")
    assert cfg.market_data is cfg.data
    assert cfg.market_data.market_data_duckdb == cfg.data.market_data_duckdb
