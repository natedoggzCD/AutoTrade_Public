from types import SimpleNamespace

from autotrade.analysis.searxng_client import SearXNGClient


def test_searxng_client_loads_configured_backup_hosts(monkeypatch):
    cfg = SimpleNamespace(
        search=SimpleNamespace(
            primary_host="http://primary.test",
            backup_hosts=["https://backup-a.test", "https://backup-b.test"],
            timeout=7,
        )
    )
    monkeypatch.setattr("config.config_loader.get_config", lambda: cfg)

    client = SearXNGClient()

    assert client.host == "http://primary.test"
    assert client.backup_hosts == ["https://backup-a.test", "https://backup-b.test"]
    assert client.all_hosts == [
        "http://primary.test",
        "https://backup-a.test",
        "https://backup-b.test",
    ]
