from types import SimpleNamespace
from datetime import datetime
from zoneinfo import ZoneInfo

from autotrade.core.autonomous_agent import AutonomousAgent
import autotrade.core.autonomous_agent as autonomous_agent_module
from autotrade.utils import market_data_cache as mdc


class _LogCapture:
    def __init__(self):
        self.warnings = []

    def info(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None

    def warning(self, msg, *args, **kwargs):
        if args:
            self.warnings.append(str(msg) % args)
        else:
            self.warnings.append(str(msg))

    def error(self, *args, **kwargs):
        return None


def test_post_market_cycle_uses_class_research_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(AutonomousAgent, "RESEARCH_DIR", tmp_path, raising=False)

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LogCapture()
    agent.plan_generator = SimpleNamespace(
        get_account_info=lambda: {"day_pnl": 0.0, "equity": 100000.0}
    )
    agent.task_router = None
    agent._reflect_done_today = False
    agent.decision_claw = None
    agent._minute_replay_archive_done_today = True

    calls = {"self_debug": 0}

    def _fake_self_debug(day_pnl):
        calls["self_debug"] += 1
        assert day_pnl == 0.0

    agent._run_self_debugging = _fake_self_debug

    interval = agent._run_post_market_cycle(cycle_count=1)

    assert interval == 120
    assert calls["self_debug"] == 1
    assert not any("RESEARCH_DIR" in w for w in agent.logger.warnings)


def test_market_data_cache_handles_missing_config_without_spam(monkeypatch, tmp_path):
    monkeypatch.setattr(
        mdc, "resolve_alpaca_credentials", lambda **kwargs: None, raising=True
    )

    cache = mdc.MarketDataCache(cache_path=str(tmp_path / "market_context.db"))

    assert cache.client is None
    assert cache.needs_update(max_age_hours=0) is False
    assert cache.update() is False
    assert cache._warned_no_client_update is True


def test_market_data_cache_uses_env_credentials_when_config_unavailable(
    monkeypatch, tmp_path
):
    from autotrade.utils.alpaca_client_factory import AlpacaCredentials

    monkeypatch.setattr(
        mdc,
        "resolve_alpaca_credentials",
        lambda **kwargs: AlpacaCredentials(
            api_key="env-key", secret_key="env-secret", paper=True
        ),
        raising=True,
    )

    created = {}

    class _DummyClient:
        def __init__(self, api_key, secret_key):
            created["api_key"] = api_key
            created["secret_key"] = secret_key

    monkeypatch.setattr(mdc, "StockHistoricalDataClient", _DummyClient)

    cache = mdc.MarketDataCache(cache_path=str(tmp_path / "market_context.db"))

    assert isinstance(cache.client, _DummyClient)
    assert created == {"api_key": "env-key", "secret_key": "env-secret"}


def test_post_market_cycle_runs_replay_minute_archive_once():
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LogCapture()
    agent._broker_sync_done = True
    agent._eod_review_done_today = True
    agent._minute_replay_archive_done_today = False
    agent.decision_claw = None
    agent.scheduler = SimpleNamespace(
        get_current_time=lambda: datetime(2026, 3, 27, 16, 5)
    )
    agent._run_reflect_phase = lambda force=False: {
        "success": True,
        "review_session_date": "2026-03-27",
    }

    calls = {"archive": 0}

    def _fake_archive(session_date=None):
        calls["archive"] += 1
        assert session_date == "2026-03-27"
        return {
            "success": True,
            "archived_symbols": 4,
            "total_symbols_requested": 5,
            "premarket_covered_symbols": 3,
            "regular_session_only_symbols": 1,
        }

    agent._capture_replay_minute_archive = _fake_archive

    assert agent._run_post_market_cycle(cycle_count=1) == 120
    assert agent._run_post_market_cycle(cycle_count=2) == 120
    assert calls["archive"] == 1


def test_post_market_cycle_retries_replay_archive_until_success():
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LogCapture()
    agent._broker_sync_done = True
    agent._eod_review_done_today = True
    agent._minute_replay_archive_done_today = False
    agent.decision_claw = None
    agent.scheduler = SimpleNamespace(
        get_current_time=lambda: datetime(2026, 3, 27, 16, 5)
    )
    agent._run_reflect_phase = lambda force=False: {
        "success": True,
        "review_session_date": "2026-03-27",
    }

    calls = {"archive": 0}

    def _fake_archive(session_date=None):
        calls["archive"] += 1
        if calls["archive"] == 1:
            return {
                "success": False,
                "error": "archive_empty:0/5",
                "archived_symbols": 0,
                "total_symbols_requested": 5,
                "premarket_covered_symbols": 0,
                "regular_session_only_symbols": 0,
            }
        return {
            "success": True,
            "archived_symbols": 4,
            "total_symbols_requested": 5,
            "premarket_covered_symbols": 3,
            "regular_session_only_symbols": 1,
        }

    agent._capture_replay_minute_archive = _fake_archive

    assert agent._run_post_market_cycle(cycle_count=1) == 120
    assert agent._minute_replay_archive_done_today is False
    assert agent._run_post_market_cycle(cycle_count=2) == 120
    assert agent._minute_replay_archive_done_today is True
    assert calls["archive"] == 2


def test_capture_replay_minute_archive_marks_empty_capture_as_failure(
    monkeypatch, tmp_path
):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LogCapture()
    agent.market_data_client = object()
    agent._collect_runtime_replay_symbol_sources = lambda *_args, **_kwargs: {
        "SPY": {"benchmark"}
    }

    cfg = SimpleNamespace(
        enabled=True,
        duckdb_path="data/replay_minute_bars.duckdb",
        preferred_start_et="04:00",
        preferred_end_et="16:00",
        fallback_regular_session_only=True,
        allow_live_fallback_if_incomplete=False,
    )
    monkeypatch.setattr(
        autonomous_agent_module,
        "get_config",
        lambda: SimpleNamespace(data=SimpleNamespace(replay_minute_archive=cfg)),
    )
    monkeypatch.setattr(autonomous_agent_module, "LOG_DIR", tmp_path)
    monkeypatch.setattr(
        autonomous_agent_module,
        "resolve_archive_path",
        lambda *args, **kwargs: tmp_path / "replay_minute_bars.duckdb",
    )
    monkeypatch.setattr(
        autonomous_agent_module,
        "archive_session_minute_bars",
        lambda **kwargs: {
            "session_date": kwargs["session_date"],
            "db_path": str(kwargs["db_path"]),
            "total_symbols_requested": len(kwargs["symbol_sources"]),
            "archived_symbols": 0,
            "missing_symbols_count": len(kwargs["symbol_sources"]),
            "premarket_covered_symbols": 0,
            "regular_session_only_symbols": 0,
            "status_counts": {"archived": 0, "archived_regular_session_only": 0, "missing": 1},
            "manifest_rows": [],
        },
    )

    result = agent._capture_replay_minute_archive(session_date="2026-03-27")

    assert result["success"] is False
    assert str(result.get("error", "")).startswith("archive_empty:")


def test_capture_replay_minute_archive_retries_empty_capture_once(
    monkeypatch, tmp_path
):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LogCapture()
    agent.market_data_client = object()
    agent._collect_runtime_replay_symbol_sources = lambda *_args, **_kwargs: {
        "SPY": {"benchmark"}
    }

    cfg = SimpleNamespace(
        enabled=True,
        duckdb_path="data/replay_minute_bars.duckdb",
        preferred_start_et="04:00",
        preferred_end_et="16:00",
        fallback_regular_session_only=True,
        allow_live_fallback_if_incomplete=True,
    )
    monkeypatch.setattr(
        autonomous_agent_module,
        "get_config",
        lambda: SimpleNamespace(data=SimpleNamespace(replay_minute_archive=cfg)),
    )
    monkeypatch.setattr(autonomous_agent_module, "LOG_DIR", tmp_path)
    monkeypatch.setattr(
        autonomous_agent_module,
        "resolve_archive_path",
        lambda *args, **kwargs: tmp_path / "replay_minute_bars.duckdb",
    )

    calls = {"n": 0}

    def _fake_archive(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            archived = 0
            missing = 1
        else:
            archived = 1
            missing = 0
        return {
            "session_date": kwargs["session_date"],
            "db_path": str(kwargs["db_path"]),
            "total_symbols_requested": len(kwargs["symbol_sources"]),
            "archived_symbols": archived,
            "missing_symbols_count": missing,
            "premarket_covered_symbols": archived,
            "regular_session_only_symbols": 0,
            "status_counts": {
                "archived": archived,
                "archived_regular_session_only": 0,
                "missing": missing,
            },
            "manifest_rows": [],
        }

    monkeypatch.setattr(
        autonomous_agent_module, "archive_session_minute_bars", _fake_archive
    )

    result = agent._capture_replay_minute_archive(session_date="2026-03-27")

    assert result["success"] is True
    assert int(result.get("archived_symbols", 0) or 0) == 1
    assert calls["n"] == 2


def test_sync_trade_journal_with_broker_prefers_active_day_manager():
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LogCapture()

    captured = {"called": False, "session_date": None, "after": None}

    class _DummyClient:
        def get_orders(self, req):
            captured["after"] = str(getattr(req, "after", ""))
            return [
                {
                    "id": "buy-1",
                    "symbol": "ABC",
                    "side": "buy",
                    "status": "filled",
                    "qty": "5",
                    "filled_qty": "5",
                    "filled_avg_price": "10.0",
                    "filled_at": "2026-05-01T15:00:00+00:00",
                }
            ]

    class _DummyJournal:
        def sync_with_broker_orders(self, orders, *, session_date=None):
            captured["called"] = True
            captured["session_date"] = session_date
            return {"backfilled_buys": 0, "backfilled_sells": 0, "backfilled_trims": 0}

    agent.alpaca_client = _DummyClient()
    agent._day_manager_instance = SimpleNamespace(trade_journal=_DummyJournal())

    agent._sync_trade_journal_with_broker()

    assert captured["called"] is True
    assert captured["session_date"] == datetime.now(ZoneInfo("America/Chicago")).date()
    assert "00:00:00" in captured["after"]


def test_sync_trade_journal_with_broker_skips_when_day_manager_missing():
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LogCapture()

    class _DummyClient:
        def get_orders(self, _req):
            return [
                {
                    "id": "sell-1",
                    "symbol": "XYZ",
                    "side": "sell",
                    "status": "filled",
                    "qty": "3",
                    "filled_qty": "3",
                    "filled_avg_price": "7.5",
                }
            ]

    agent.alpaca_client = _DummyClient()
    agent._day_manager_instance = None

    agent._sync_trade_journal_with_broker()

    assert any(
        "standalone backfill disabled for live safety" in msg
        for msg in agent.logger.warnings
    )
