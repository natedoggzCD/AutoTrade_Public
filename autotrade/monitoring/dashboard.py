from __future__ import annotations

import sqlite3
import json
from dataclasses import asdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd

from autotrade.data_ingestion.fast_cache import FastMarketDataCache
from autotrade.data_ingestion.stream_bridge import AlpacaStreamBridge
from autotrade.monitoring.reporting import ReportingEngine
from autotrade.signals.trade_learner import TradeJournal
from autotrade.signals.trade_learner import JOURNAL_FILE
from autotrade.utils.alpaca_client_factory import create_trading_client


def _normalize_display_value(value: Any) -> str:
    text = str(value or "").strip()
    if "." in text:
        return text.split(".")[-1]
    return text


def _format_currency(value: Any) -> str:
    try:
        return f"${float(value or 0.0):,.2f}"
    except (TypeError, ValueError):
        return "$0.00"


def _format_pct(value: Any) -> str:
    try:
        return f"{float(value or 0.0):.2f}%"
    except (TypeError, ValueError):
        return "0.00%"


def _format_signed_currency(value: Any) -> str:
    try:
        numeric = float(value or 0.0)
    except (TypeError, ValueError):
        numeric = 0.0
    return f"{numeric:+,.2f}"


def get_trade_logs_from_sqlite(
    sqlite_path: Path | str,
    *,
    limit: int = 250,
) -> List[Dict[str, Any]]:
    db_path = Path(sqlite_path)
    if not db_path.exists():
        return []

    limit_value = max(1, int(limit))
    query = """
        SELECT symbol, side, qty, fill_price, pnl, strategy, status, executed_at
        FROM trade_logs
        ORDER BY executed_at DESC, rowid DESC
        LIMIT ?
    """
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, (limit_value,)).fetchall()
    except sqlite3.OperationalError:
        return []
    return [dict(row) for row in rows]


def get_trade_logs_from_journal(
    journal_path: Path | str = JOURNAL_FILE,
    *,
    limit: int = 250,
) -> List[Dict[str, Any]]:
    path = Path(journal_path)
    if not path.exists():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    trades = payload.get("trades", [])
    if not isinstance(trades, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        trade_type = str(trade.get("trade_type", "entry") or "entry").lower()
        if trade_type in {"trim", "scale"}:
            continue
        normalized.append(
            {
                "symbol": str(trade.get("symbol", "") or ""),
                "side": "sell" if trade.get("exit_time") else "buy",
                "qty": int(float(trade.get("filled_quantity", trade.get("quantity", 0)) or 0)),
                "fill_price": float(
                    trade.get("exit_price")
                    or trade.get("entry_price")
                    or 0.0
                ),
                "pnl": float(trade.get("pnl_dollars", 0.0) or 0.0),
                "strategy": str(
                    trade.get("reason")
                    or trade.get("exit_reason")
                    or trade.get("signals", {}).get("entry_family", "")
                    or ""
                ),
                "status": str(
                    trade.get("outcome")
                    or trade.get("entry_status")
                    or "open"
                ),
                "executed_at": str(
                    trade.get("exit_time")
                    or trade.get("entry_time")
                    or ""
                ),
                "source": "trade_journal",
            }
        )

    normalized.sort(key=lambda row: row.get("executed_at", ""), reverse=True)
    return normalized[: max(1, int(limit))]


def get_recent_trade_activity(
    sqlite_path: Path | str,
    *,
    limit: int = 250,
    journal_path: Path | str = JOURNAL_FILE,
) -> List[Dict[str, Any]]:
    rows = get_trade_logs_from_sqlite(sqlite_path, limit=limit)
    if rows:
        for row in rows:
            row.setdefault("source", "financial_db")
        return rows
    return get_trade_logs_from_journal(journal_path, limit=limit)


def summarize_trade_activity(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {
            "total_trades": 0,
            "realized_pnl": 0.0,
            "wins": 0,
            "losses": 0,
            "open_positions": 0,
            "win_rate": 0.0,
        }

    realized = 0.0
    wins = 0
    losses = 0
    open_positions = 0
    closed = 0
    for trade in trades:
        status = str(trade.get("status", "") or "").lower()
        pnl = float(trade.get("pnl", 0.0) or 0.0)
        if status in {"open", "submitted"}:
            open_positions += 1
            continue
        realized += pnl
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1
        closed += 1

    return {
        "total_trades": len(trades),
        "realized_pnl": realized,
        "wins": wins,
        "losses": losses,
        "open_positions": open_positions,
        "win_rate": (wins / closed) if closed > 0 else 0.0,
    }


def get_trade_journal_stats() -> Dict[str, Any]:
    try:
        journal = TradeJournal()
        stats = journal.get_stats()
    except Exception:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_hold_time": 0.0,
            "avg_pnl_per_hour": 0.0,
        }

    return {
        "total_trades": int(stats.get("total_trades", 0) or 0),
        "wins": int(stats.get("wins", 0) or 0),
        "losses": int(stats.get("losses", 0) or 0),
        "win_rate": float(stats.get("win_rate", 0.0) or 0.0),
        "total_pnl": float(stats.get("total_pnl", 0.0) or 0.0),
        "avg_hold_time": float(stats.get("avg_hold_time", 0.0) or 0.0),
        "avg_pnl_per_hour": float(stats.get("avg_pnl_per_hour", 0.0) or 0.0),
    }


def get_alpaca_account_info(
    client_factory: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    client = create_trading_client(
        validate_connection=False,
        require_credentials=False,
        client_factory=client_factory,
    )
    if client is None:
        return {}

    account = client.get_account()
    return {
        "account_number": str(getattr(account, "account_number", "") or ""),
        "equity": float(getattr(account, "equity", 0.0) or 0.0),
        "buying_power": float(getattr(account, "buying_power", 0.0) or 0.0),
        "cash": float(getattr(account, "cash", 0.0) or 0.0),
        "portfolio_value": float(getattr(account, "portfolio_value", 0.0) or 0.0),
        "status": _normalize_display_value(getattr(account, "status", "")),
    }


def calculate_drawdown(equity_curve: Iterable[float]) -> Dict[str, float]:
    values = [float(v) for v in equity_curve]
    if not values:
        return {
            "current_drawdown_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "peak_equity": 0.0,
            "current_equity": 0.0,
        }

    peak = values[0]
    max_drawdown_pct = 0.0
    current_drawdown_pct = 0.0
    for value in values:
        peak = max(peak, value)
        drawdown_pct = ((peak - value) / peak) * 100.0 if peak > 0 else 0.0
        max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
        current_drawdown_pct = drawdown_pct

    return {
        "current_drawdown_pct": current_drawdown_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "peak_equity": peak,
        "current_equity": values[-1],
    }


def calculate_risk_of_ruin(
    *,
    win_rate: float,
    payoff_ratio: float,
    risk_per_trade_pct: float,
    capital_fraction_at_risk: float = 1.0,
) -> Dict[str, float]:
    win = min(max(float(win_rate), 0.0), 1.0)
    payoff = max(float(payoff_ratio), 0.0)
    risk_pct = max(float(risk_per_trade_pct), 0.0) / 100.0
    capital_frac = min(max(float(capital_fraction_at_risk), 0.0), 1.0)
    if risk_pct <= 0 or capital_frac <= 0:
        return {
            "edge": 0.0,
            "risk_of_ruin_pct": 0.0,
        }

    loss_rate = 1.0 - win
    expectancy = (win * payoff) - loss_rate
    if expectancy <= 0:
        return {
            "edge": expectancy,
            "risk_of_ruin_pct": 100.0,
        }

    normalized_risk = min(1.0, risk_pct * capital_frac * 10.0)
    ruin_probability = min(1.0, max(0.0, normalized_risk / max(expectancy, 1e-9)))
    return {
        "edge": expectancy,
        "risk_of_ruin_pct": ruin_probability * 100.0,
    }


def plot_trade_replay(
    price_bars: pd.DataFrame,
    trade_log: Dict[str, Any],
) -> Any:
    import plotly.graph_objects as go

    frame = price_bars.copy()
    if frame.empty:
        raise ValueError("price_bars must not be empty")

    if "timestamp" in frame.columns:
        x_axis = pd.to_datetime(frame["timestamp"])
    elif frame.index.name or isinstance(frame.index, pd.DatetimeIndex):
        x_axis = pd.to_datetime(frame.index)
    else:
        raise ValueError("price_bars must provide a timestamp column or DatetimeIndex")

    figure = go.Figure()
    figure.add_trace(
        go.Candlestick(
            x=x_axis,
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="OHLCV",
        )
    )

    entry_time = pd.to_datetime(trade_log.get("entry_time"))
    exit_time = pd.to_datetime(trade_log.get("exit_time")) if trade_log.get("exit_time") else None
    entry_price = float(trade_log.get("entry_price", 0.0) or 0.0)
    exit_price = float(trade_log.get("exit_price", 0.0) or 0.0)
    symbol = str(trade_log.get("symbol", "") or "")

    figure.add_trace(
        go.Scatter(
            x=[entry_time],
            y=[entry_price],
            mode="markers",
            name="Entry",
            marker={"symbol": "triangle-up", "size": 12, "color": "green"},
            text=[f"{symbol} entry"],
        )
    )
    if exit_time is not None and exit_price > 0:
        figure.add_trace(
            go.Scatter(
                x=[exit_time],
                y=[exit_price],
                mode="markers",
                name="Exit",
                marker={"symbol": "triangle-down", "size": 12, "color": "red"},
                text=[f"{symbol} exit"],
            )
        )

    figure.update_layout(
        title=f"Trade Replay: {symbol}".strip(),
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
    )
    return figure


class DashboardLiveFeed:
    def __init__(
        self,
        *,
        cache_store: Optional[FastMarketDataCache] = None,
        stream_bridge: Optional[AlpacaStreamBridge] = None,
    ) -> None:
        self.cache_store = cache_store or FastMarketDataCache()
        self.stream_bridge = stream_bridge

    async def subscribe(self, symbols: Iterable[str]) -> tuple[str, ...]:
        normalized = tuple(str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip())
        if self.stream_bridge is None:
            return normalized
        return self.stream_bridge.subscribe(normalized)

    async def latest_snapshot(self, symbol: str) -> Dict[str, Any]:
        snapshot = await self.cache_store.get_snapshot(symbol)
        return {
            "symbol": snapshot.symbol,
            "quote": asdict(snapshot.quote) if snapshot.quote is not None else None,
            "trade": asdict(snapshot.trade) if snapshot.trade is not None else None,
            "updated_at": snapshot.updated_at.isoformat() if snapshot.updated_at else None,
        }


@dataclass
class PerformanceDashboard:
    sqlite_path: Path | str = field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "financial.db"
    )
    host: str = "127.0.0.1"
    port: int = 8501
    page_title: str = "AutoTrade Performance Dashboard"
    refresh_seconds: int = 15
    alpaca_account_loader: Optional[Callable[[], Dict[str, Any]]] = None
    reporting_engine: Optional[ReportingEngine] = None

    def __post_init__(self) -> None:
        self.sqlite_path = Path(self.sqlite_path)
        if str(self.host).strip() in {"0.0.0.0", "::"}:
            self.host = "127.0.0.1"

    @property
    def is_local_only(self) -> bool:
        return self.host in {"127.0.0.1", "localhost"}

    def get_streamlit_server_config(self) -> Dict[str, Any]:
        return {
            "server.address": "127.0.0.1" if not self.is_local_only else self.host,
            "server.port": int(self.port),
            "browser.serverAddress": "127.0.0.1" if not self.is_local_only else self.host,
        }

    def build_context(self) -> Dict[str, Any]:
        return {
            "page_title": self.page_title,
            "sqlite_path": str(self.sqlite_path),
            "host": self.host,
            "port": self.port,
            "refresh_seconds": self.refresh_seconds,
            "is_local_only": self.is_local_only,
            "streamlit_server": self.get_streamlit_server_config(),
        }

    def get_alpaca_account_snapshot(self) -> Dict[str, Any]:
        if self.alpaca_account_loader is None:
            return get_alpaca_account_info()
        return dict(self.alpaca_account_loader() or {})

    def get_trade_logs(self, *, limit: int = 250) -> List[Dict[str, Any]]:
        return get_recent_trade_activity(self.sqlite_path, limit=limit)

    def get_dashboard_artifact(self, *, date: Optional[str] = None) -> Dict[str, Any]:
        if self.reporting_engine is None:
            return {}
        artifact = self.reporting_engine.generate_dashboard_artifact(date=date)
        payload = {
            "generated_at": artifact.generated_at.isoformat(),
            "kpis": dict(artifact.kpis),
            "signal_funnel": dict(artifact.signal_funnel),
            "execution_quality": dict(artifact.execution_quality),
            "alpha_contribution": dict(artifact.alpha_contribution),
            "risk_alarms": dict(artifact.risk_alarms),
        }
        return self._apply_journal_fallbacks(payload)

    def _apply_journal_fallbacks(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        journal_stats = get_trade_journal_stats()
        kpis = dict(artifact.get("kpis", {}))
        signal_funnel = dict(artifact.get("signal_funnel", {}))
        execution_quality = dict(artifact.get("execution_quality", {}))

        if int(kpis.get("signal_funnel_total", 0) or 0) <= 0:
            kpis["signal_funnel_total"] = int(journal_stats.get("total_trades", 0) or 0)
        if float(kpis.get("trade_conversion_rate", 0.0) or 0.0) <= 0 and int(journal_stats.get("total_trades", 0) or 0) > 0:
            kpis["trade_conversion_rate"] = float(journal_stats.get("win_rate", 0.0) or 0.0)

        if not signal_funnel and int(journal_stats.get("total_trades", 0) or 0) > 0:
            signal_funnel = {
                "evaluated": int(journal_stats.get("total_trades", 0) or 0),
                "accepted": int(journal_stats.get("total_trades", 0) or 0),
                "executed": int(journal_stats.get("wins", 0) or 0) + int(journal_stats.get("losses", 0) or 0),
            }

        if not execution_quality and int(journal_stats.get("total_trades", 0) or 0) > 0:
            execution_quality = {
                "total_orders": int(journal_stats.get("total_trades", 0) or 0),
                "avg_slippage_bps": 0.0,
                "avg_fill_rate": 1.0,
            }

        artifact["kpis"] = kpis
        artifact["signal_funnel"] = signal_funnel
        artifact["execution_quality"] = execution_quality
        artifact["journal_stats"] = journal_stats
        return artifact

    def render(self, streamlit_module: Optional[Any] = None) -> Dict[str, Any]:
        context = self.build_context()
        account = self.get_alpaca_account_snapshot()
        trades = self.get_trade_logs(limit=25)
        artifact = self.get_dashboard_artifact()
        if streamlit_module is None:
            return {
                "context": context,
                "account": account,
                "trades": trades,
                "artifact": artifact,
            }

        streamlit_module.set_page_config(
            page_title=self.page_title,
            layout="wide",
        )
        self._render_streamlit_dashboard(streamlit_module, account, trades, artifact)
        return {
            "context": context,
            "account": account,
            "trades": trades,
            "artifact": artifact,
        }

    def _render_streamlit_dashboard(
        self,
        streamlit_module: Any,
        account: Dict[str, Any],
        trades: List[Dict[str, Any]],
        artifact: Dict[str, Any],
    ) -> None:
        trade_summary = dict(artifact.get("journal_stats", {})) or summarize_trade_activity(trades)
        streamlit_module.title(self.page_title)
        streamlit_module.caption(
            f"Read-only local monitor | account {account.get('account_number', 'unavailable')} | "
            f"binding {self.host}:{self.port} | source {self.sqlite_path.name}"
        )

        kpis = dict(artifact.get("kpis", {}))
        account_cols = streamlit_module.columns(4)
        account_cols[0].metric("Equity", _format_currency(account.get("equity")))
        account_cols[1].metric("Buying Power", _format_currency(account.get("buying_power")))
        account_cols[2].metric("Cash", _format_currency(account.get("cash")))
        account_cols[3].metric(
            "Status",
            _normalize_display_value(account.get("status", "offline")) or "OFFLINE",
        )

        summary_cols = streamlit_module.columns(4)
        summary_cols[0].metric(
            "Tracked Trades",
            int(trade_summary.get("total_trades", 0) or 0),
        )
        summary_cols[1].metric(
            "Win Rate",
            _format_pct(float(trade_summary.get("win_rate", 0.0) or 0.0) * 100.0),
        )
        summary_cols[2].metric(
            "Realized P&L",
            _format_signed_currency(
                trade_summary.get("realized_pnl", trade_summary.get("total_pnl", 0.0))
            ),
        )
        summary_cols[3].metric(
            "Open Positions",
            int(trade_summary.get("open_positions", 0) or 0),
        )

        telemetry_cols = streamlit_module.columns(4)
        telemetry_cols[0].metric(
            "Conversion",
            _format_pct(kpis.get("trade_conversion_rate", 0.0) * 100.0),
        )
        telemetry_cols[1].metric(
            "Drawdown",
            _format_pct(kpis.get("total_drawdown_pct", 0.0)),
        )
        telemetry_cols[2].metric(
            "Slippage Gap",
            f"{float(kpis.get('realized_vs_expected_slippage_bps', 0.0) or 0.0):.2f} bps",
        )
        telemetry_cols[3].metric(
            "Active Alerts",
            int(kpis.get("active_alerts", 0) or 0),
        )

        with streamlit_module.expander("Environment", expanded=False):
            streamlit_module.json(
                {
                    "streamlit_server": self.get_streamlit_server_config(),
                    "refresh_seconds": self.refresh_seconds,
                    "sqlite_path": str(self.sqlite_path),
                    "trade_journal_fallback": str(JOURNAL_FILE),
                }
            )

        overview_tab, trades_tab, risk_tab, raw_tab = streamlit_module.tabs(
            ["Overview", "Trades", "Risk", "Raw"]
        )

        with overview_tab:
            funnel = dict(artifact.get("signal_funnel", {}))
            exec_quality = dict(artifact.get("execution_quality", {}))
            streamlit_module.subheader("Recent Trades Snapshot")
            streamlit_module.caption(
                f"Journal-backed rows: {len(trades)} | Execution telemetry rows: {int(exec_quality.get('total_orders', 0) or 0)}"
            )
            if trades:
                preview = pd.DataFrame(trades[:10])
                streamlit_module.dataframe(
                    preview,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "symbol": streamlit_module.column_config.TextColumn("Symbol"),
                        "side": streamlit_module.column_config.TextColumn("Side"),
                        "qty": streamlit_module.column_config.NumberColumn("Qty", format="%d"),
                        "fill_price": streamlit_module.column_config.NumberColumn("Fill", format="$%.2f"),
                        "pnl": streamlit_module.column_config.NumberColumn("P&L", format="$%.2f"),
                        "strategy": streamlit_module.column_config.TextColumn("Strategy"),
                        "status": streamlit_module.column_config.TextColumn("Status"),
                        "executed_at": streamlit_module.column_config.TextColumn("Executed"),
                        "source": streamlit_module.column_config.TextColumn("Source"),
                    },
                )
            else:
                streamlit_module.markdown(
                    "No trade activity was found in either `financial.db` or `logs/trade_journal.json`."
                )
            funnel_cols = streamlit_module.columns(3)
            funnel_cols[0].metric("Evaluated", int(funnel.get("evaluated", 0) or 0))
            funnel_cols[1].metric("Accepted", int(funnel.get("accepted", 0) or 0))
            funnel_cols[2].metric("Executed", int(funnel.get("executed", 0) or 0))
            exec_cols = streamlit_module.columns(3)
            exec_cols[0].metric("Orders", int(exec_quality.get("total_orders", 0) or 0))
            exec_cols[1].metric(
                "Avg Slippage",
                f"{float(exec_quality.get('avg_slippage_bps', 0.0) or 0.0):.2f} bps",
            )
            exec_cols[2].metric(
                "Avg Fill Rate",
                _format_pct(float(exec_quality.get("avg_fill_rate", 0.0) or 0.0) * 100.0),
            )
            if not funnel and not exec_quality:
                streamlit_module.info(
                    "Telemetry is still sparse. The dashboard is falling back to the trade journal for visibility, but the monitoring collector is not yet publishing populated signal and execution summaries."
                )

        with trades_tab:
            streamlit_module.subheader("Recent Trades")
            if trades:
                trade_frame = pd.DataFrame(trades)
                streamlit_module.dataframe(
                    trade_frame,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "qty": streamlit_module.column_config.NumberColumn("Qty", format="%d"),
                        "fill_price": streamlit_module.column_config.NumberColumn("Fill Price", format="$%.2f"),
                        "pnl": streamlit_module.column_config.NumberColumn("P&L", format="$%.2f"),
                    },
                )
            else:
                streamlit_module.info(
                    "No rows were found in `trade_logs`. The dashboard expects either `financial.db` trade rows or `logs/trade_journal.json` fallback data."
                )

        with risk_tab:
            risk_alarms = dict(artifact.get("risk_alarms", {}))
            risk_cols = streamlit_module.columns(4)
            risk_cols[0].metric("Drawdown Alarm", "ON" if risk_alarms.get("drawdown_alarm") else "OFF")
            risk_cols[1].metric("Critical DD", "ON" if risk_alarms.get("critical_drawdown") else "OFF")
            risk_cols[2].metric("Stale Data", "ON" if risk_alarms.get("data_staleness") else "OFF")
            risk_cols[3].metric("Exec Fails", int(risk_alarms.get("execution_failures", 0) or 0))
            streamlit_module.json(risk_alarms or {"active_alerts": 0})

        with raw_tab:
            streamlit_module.subheader("Raw Payloads")
            streamlit_module.json(
                {
                    "account": account or {},
                    "artifact": artifact or {},
                    "context": self.build_context(),
                }
            )
