import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from autotrade.data_ingestion.stream_bridge import QuoteStreamEvent, TradeStreamEvent
from autotrade.monitoring.dashboard import (
    DashboardLiveFeed,
    PerformanceDashboard,
    calculate_drawdown,
    calculate_risk_of_ruin,
    get_alpaca_account_info,
    get_trade_journal_stats,
    get_recent_trade_activity,
    get_trade_logs_from_journal,
    get_trade_logs_from_sqlite,
    plot_trade_replay,
)
from autotrade.monitoring.reporting import DashboardArtifact


def test_performance_dashboard_defaults_to_local_sqlite_dashboard() -> None:
    dashboard = PerformanceDashboard()

    assert dashboard.sqlite_path.name == "financial.db"
    assert dashboard.is_local_only is True
    assert dashboard.port == 8501
    assert dashboard.get_streamlit_server_config()["server.address"] == "127.0.0.1"


def test_performance_dashboard_builds_context_and_account_snapshot() -> None:
    dashboard = PerformanceDashboard(
        sqlite_path=Path("custom.db"),
        alpaca_account_loader=lambda: {"equity": 123456.78, "buying_power": 25000.0},
    )

    payload = dashboard.render()

    assert payload["context"]["sqlite_path"].endswith("custom.db")
    assert payload["context"]["is_local_only"] is True
    assert payload["context"]["streamlit_server"]["server.address"] == "127.0.0.1"
    assert payload["account"]["equity"] == 123456.78


def test_performance_dashboard_forces_non_local_host_back_to_localhost() -> None:
    dashboard = PerformanceDashboard(host="0.0.0.0", port=9001)

    assert dashboard.host == "127.0.0.1"
    assert dashboard.is_local_only is True
    assert dashboard.get_streamlit_server_config()["server.port"] == 9001


def test_performance_dashboard_includes_reporting_artifact() -> None:
    class DummyReportingEngine:
        def generate_dashboard_artifact(self, date=None):
            return DashboardArtifact(
                generated_at=datetime(2026, 3, 7, 16, 0, tzinfo=timezone.utc),
                kpis={"trade_conversion_rate": 0.5},
                signal_funnel={"evaluated": 10},
                execution_quality={"total_orders": 2},
                alpha_contribution={"families": {}},
                risk_alarms={"active_alerts": 1},
            )

    dashboard = PerformanceDashboard(
        reporting_engine=DummyReportingEngine(),
        alpaca_account_loader=lambda: {},
    )

    payload = dashboard.render()

    assert payload["artifact"]["kpis"]["trade_conversion_rate"] == 0.5
    assert payload["artifact"]["execution_quality"]["total_orders"] == 2


def test_get_trade_logs_from_sqlite_reads_trade_log_rows(tmp_path) -> None:
    db_path = tmp_path / "financial.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE trade_logs (
                symbol TEXT,
                side TEXT,
                qty REAL,
                fill_price REAL,
                pnl REAL,
                strategy TEXT,
                status TEXT,
                executed_at TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO trade_logs(symbol, side, qty, fill_price, pnl, strategy, status, executed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("AAPL", "buy", 100, 185.25, 42.5, "momentum", "filled", "2026-03-07T09:31:00"),
        )
        conn.commit()

    rows = get_trade_logs_from_sqlite(db_path)

    assert len(rows) == 1
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["status"] == "filled"


def test_get_trade_logs_from_journal_reads_completed_trades(tmp_path) -> None:
    journal_path = tmp_path / "trade_journal.json"
    journal_path.write_text(
        json.dumps(
            {
                "trades": [
                    {
                        "symbol": "MSFT",
                        "entry_time": "2026-03-07T09:31:00",
                        "entry_price": 401.0,
                        "quantity": 10,
                        "filled_quantity": 10,
                        "entry_status": "filled",
                        "exit_time": "2026-03-07T10:15:00",
                        "exit_price": 405.5,
                        "pnl_dollars": 45.0,
                        "outcome": "win",
                        "reason": "momentum_breakout",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = get_trade_logs_from_journal(journal_path)

    assert len(rows) == 1
    assert rows[0]["symbol"] == "MSFT"
    assert rows[0]["source"] == "trade_journal"
    assert rows[0]["pnl"] == 45.0


def test_get_recent_trade_activity_falls_back_to_journal(tmp_path) -> None:
    db_path = tmp_path / "financial.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE unrelated(id INTEGER)")
        conn.commit()

    journal_path = tmp_path / "trade_journal.json"
    journal_path.write_text(
        json.dumps(
            {
                "trades": [
                    {
                        "symbol": "NVDA",
                        "entry_time": "2026-03-07T11:00:00",
                        "entry_price": 900.0,
                        "quantity": 5,
                        "filled_quantity": 5,
                        "entry_status": "filled",
                        "reason": "ai_momentum",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = get_recent_trade_activity(db_path, journal_path=journal_path)

    assert len(rows) == 1
    assert rows[0]["symbol"] == "NVDA"
    assert rows[0]["source"] == "trade_journal"


def test_get_trade_journal_stats_reads_completed_trade_summary(monkeypatch) -> None:
    class DummyJournal:
        def get_stats(self):
            return {
                "total_trades": 12,
                "wins": 7,
                "losses": 3,
                "win_rate": 0.58,
                "total_pnl": 345.5,
                "avg_hold_time": 5.5,
                "avg_pnl_per_hour": 0.8,
            }

    monkeypatch.setattr("autotrade.monitoring.dashboard.TradeJournal", DummyJournal)

    stats = get_trade_journal_stats()

    assert stats["total_trades"] == 12
    assert stats["win_rate"] == 0.58
    assert stats["total_pnl"] == 345.5


def test_get_alpaca_account_info_normalizes_snapshot() -> None:
    class DummyClient:
        def get_account(self):
            return SimpleNamespace(
                account_number="PA123",
                equity="100500.25",
                buying_power="25000.00",
                cash="1500.50",
                portfolio_value="100500.25",
                status="AccountStatus.ACTIVE",
            )

    info = get_alpaca_account_info(client_factory=lambda **_: DummyClient())

    assert info["account_number"] == "PA123"
    assert info["equity"] == 100500.25
    assert info["buying_power"] == 25000.0
    assert info["status"] == "ACTIVE"


def test_performance_dashboard_render_uses_metric_layout_with_streamlit_stub() -> None:
    class _Column:
        def __init__(self, sink):
            self.sink = sink

        def metric(self, label, value):
            self.sink.append((label, value))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Tab:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class StreamlitStub:
        def __init__(self):
            self.metrics = []
            class _ColumnConfig:
                @staticmethod
                def TextColumn(*_args, **_kwargs):
                    return None

                @staticmethod
                def NumberColumn(*_args, **_kwargs):
                    return None

            self.column_config = _ColumnConfig()

        def set_page_config(self, **kwargs):
            self.config = kwargs

        def title(self, *_args, **_kwargs):
            return None

        def markdown(self, *args, **kwargs):
            return None

        def columns(self, count):
            actual = count if isinstance(count, int) else len(count)
            return [_Column(self.metrics) for _ in range(actual)]

        def tabs(self, names):
            self.tab_names = names
            return [_Tab() for _ in names]

        def expander(self, *_args, **_kwargs):
            return _Tab()

        def subheader(self, *_args, **_kwargs):
            return None

        def caption(self, *_args, **_kwargs):
            return None

        def dataframe(self, *_args, **_kwargs):
            return None

        def json(self, *_args, **_kwargs):
            return None

    class DummyReportingEngine:
        def generate_dashboard_artifact(self, date=None):
            return DashboardArtifact(
                generated_at=datetime(2026, 3, 7, 16, 0, tzinfo=timezone.utc),
                kpis={"trade_conversion_rate": 0.25, "active_alerts": 2},
                signal_funnel={"evaluated": 12, "accepted": 5, "executed": 3},
                execution_quality={"total_orders": 4, "avg_slippage_bps": 3.2, "avg_fill_rate": 0.75},
                alpha_contribution={},
                risk_alarms={"drawdown_alarm": False, "critical_drawdown": False, "data_staleness": False, "execution_failures": 0},
            )

    stub = StreamlitStub()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "autotrade.monitoring.dashboard.get_trade_journal_stats",
        lambda: {
            "total_trades": 12,
            "wins": 7,
            "losses": 3,
            "win_rate": 0.58,
            "total_pnl": 345.5,
            "avg_hold_time": 5.5,
            "avg_pnl_per_hour": 0.8,
        },
    )
    dashboard = PerformanceDashboard(
        alpaca_account_loader=lambda: {"equity": 100000.0, "buying_power": 50000.0, "cash": 12000.0, "status": "ACTIVE"},
        reporting_engine=DummyReportingEngine(),
    )

    payload = dashboard.render(stub)
    monkeypatch.undo()

    assert payload["account"]["status"] == "ACTIVE"
    assert ("Tracked Trades", 12) in stub.metrics
    assert ("Conversion", "25.00%") in stub.metrics
    assert ("Active Alerts", 2) in stub.metrics


def test_calculate_drawdown_reports_current_and_max_drawdown() -> None:
    metrics = calculate_drawdown([100000.0, 105000.0, 101000.0, 99000.0])

    assert round(metrics["current_drawdown_pct"], 4) == round(((105000.0 - 99000.0) / 105000.0) * 100.0, 4)
    assert metrics["max_drawdown_pct"] == metrics["current_drawdown_pct"]
    assert metrics["peak_equity"] == 105000.0


def test_calculate_risk_of_ruin_increases_with_negative_edge() -> None:
    positive = calculate_risk_of_ruin(
        win_rate=0.58,
        payoff_ratio=1.4,
        risk_per_trade_pct=1.0,
    )
    negative = calculate_risk_of_ruin(
        win_rate=0.35,
        payoff_ratio=0.9,
        risk_per_trade_pct=2.0,
    )

    assert positive["edge"] > 0
    assert 0.0 <= positive["risk_of_ruin_pct"] < 100.0
    assert negative["edge"] <= 0
    assert negative["risk_of_ruin_pct"] == 100.0


def test_plot_trade_replay_maps_entry_and_exit_to_price_chart() -> None:
    price_bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-03-07T09:30:00",
                    "2026-03-07T09:31:00",
                    "2026-03-07T09:32:00",
                ]
            ),
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.5, 100.5, 101.5],
            "close": [100.8, 101.8, 102.6],
        }
    )
    trade_log = {
        "symbol": "AAPL",
        "entry_time": "2026-03-07T09:31:00",
        "entry_price": 101.25,
        "exit_time": "2026-03-07T09:32:00",
        "exit_price": 102.5,
    }

    figure = plot_trade_replay(price_bars, trade_log)

    assert len(figure.data) == 3
    assert figure.data[1].name == "Entry"
    assert figure.data[1].y[0] == 101.25
    assert figure.data[2].name == "Exit"
    assert figure.data[2].y[0] == 102.5


@pytest.mark.asyncio
async def test_dashboard_live_feed_returns_cached_snapshot() -> None:
    feed = DashboardLiveFeed()
    quote = QuoteStreamEvent(
        symbol="AAPL",
        bid_price=100.0,
        ask_price=100.2,
        bid_size=10.0,
        ask_size=12.0,
        timestamp=datetime(2026, 3, 7, 15, 30, tzinfo=timezone.utc),
    )
    trade = TradeStreamEvent(
        symbol="AAPL",
        price=100.15,
        size=200.0,
        exchange="V",
        conditions=("@",),
        timestamp=datetime(2026, 3, 7, 15, 30, 1, tzinfo=timezone.utc),
    )

    await feed.cache_store.update_quote(quote)
    await feed.cache_store.update_trade(trade)
    snapshot = await feed.latest_snapshot("AAPL")

    assert snapshot["symbol"] == "AAPL"
    assert snapshot["quote"]["bid_price"] == 100.0
    assert snapshot["trade"]["price"] == 100.15
    assert snapshot["updated_at"] == "2026-03-07T15:30:01+00:00"


@pytest.mark.asyncio
async def test_dashboard_live_feed_subscribe_forwards_symbols_to_bridge() -> None:
    captured = {}

    class DummyBridge:
        def subscribe(self, symbols):
            captured["symbols"] = tuple(symbols)
            return tuple(symbols)

    feed = DashboardLiveFeed(stream_bridge=DummyBridge())
    subscribed = await feed.subscribe(["aapl", "msft"])

    assert subscribed == ("AAPL", "MSFT")
    assert captured["symbols"] == ("AAPL", "MSFT")
