from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    eq = equity.astype(float)
    running_max = eq.cummax()
    dd = (eq - running_max) / running_max.replace(0, np.nan)
    dd = dd.fillna(0.0)
    return float(dd.min())  # negative


def _sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float) -> float:
    if daily_returns.empty:
        return 0.0
    rf_daily = float(risk_free_rate) / 252.0
    excess = daily_returns.astype(float) - rf_daily
    std = float(excess.std(ddof=0))
    if std <= 0:
        return 0.0
    return float((excess.mean() / std) * np.sqrt(252.0))


def _sortino_ratio(daily_returns: pd.Series, risk_free_rate: float) -> float:
    if daily_returns.empty:
        return 0.0
    rf_daily = float(risk_free_rate) / 252.0
    excess = daily_returns.astype(float) - rf_daily
    downside = excess[excess < 0]
    if downside.empty:
        return 0.0
    downside_std = float(downside.std(ddof=0))
    if downside_std <= 0:
        return 0.0
    return float((excess.mean() / downside_std) * np.sqrt(252.0))


def _calmar_ratio(annual_return_pct: float, max_drawdown_pct: float) -> float:
    if max_drawdown_pct <= 0:
        return 0.0
    return float(annual_return_pct / max_drawdown_pct)


def _calculate_turnover(
    trades: List[Dict[str, Any]], initial_cash: float, periods: int
) -> float:
    if not trades or initial_cash <= 0 or periods <= 0:
        return 0.0
    total_volume = sum(
        abs(float(t.get("pnl_dollars", 0) or t.get("entry_value", 0))) for t in trades
    )
    return float(total_volume / (initial_cash * periods))


def _calculate_capacity_proxy(
    avg_position_value: float, avg_daily_volume: float
) -> float:
    if avg_daily_volume <= 0:
        return 0.0
    return float(min(1.0, avg_position_value / avg_daily_volume))


def _calculate_hit_rate(trades: List[Dict[str, Any]]) -> float:
    if not trades:
        return 0.0
    hits = sum(1 for t in trades if float(t.get("pnl_dollars", 0) or 0) > 0)
    return float(hits / len(trades))


def _calculate_expectancy(trades: List[Dict[str, Any]]) -> float:
    if not trades:
        return 0.0
    pnl_values = [float(t.get("pnl_pct", 0) or 0) for t in trades]
    return float(np.mean(pnl_values))


def _calculate_precision_recall(
    trades: List[Dict[str, Any]], predictions: List[bool]
) -> tuple[float, float]:
    if not trades or len(trades) != len(predictions):
        return 0.0, 0.0

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    for trade, pred in zip(trades, predictions):
        actual_positive = float(trade.get("pnl_dollars", 0) or 0) > 0

        if pred and actual_positive:
            true_positives += 1
        elif pred and not actual_positive:
            false_positives += 1
        elif not pred and actual_positive:
            false_negatives += 1
        else:
            true_negatives += 1

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    return float(precision), float(recall)


def _annualize_returns(total_pnl_pct: float, days: int) -> float:
    if days <= 0:
        return 0.0
    years = days / 252.0
    if years <= 0:
        return 0.0
    return float((1 + total_pnl_pct / 100.0) ** (1 / years) - 1) * 100.0


def _annualize_volatility(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return 0.0
    return float(daily_returns.std(ddof=0) * np.sqrt(252.0) * 100.0)


def summarize_backtest(
    trades: List[Dict[str, Any]],
    equity_curve: pd.DataFrame,
    initial_cash: float,
    risk_free_rate: float = 0.05,
) -> Dict[str, Any]:
    total_trades = int(len(trades))
    pnl_values = [float(t.get("pnl_dollars", 0.0) or 0.0) for t in trades]

    wins = [p for p in pnl_values if p > 0]
    losses = [p for p in pnl_values if p < 0]

    win_rate = (len(wins) / total_trades) if total_trades else 0.0
    profit_factor = (
        (sum(wins) / abs(sum(losses))) if losses else (float("inf") if wins else 0.0)
    )

    total_pnl = float(sum(pnl_values))
    avg_trade = (total_pnl / total_trades) if total_trades else 0.0

    equity = pd.Series(dtype=float)
    daily_returns = pd.Series(dtype=float)
    max_drawdown = 0.0
    sharpe = 0.0
    sortino = 0.0
    annual_return = 0.0
    annual_volatility = 0.0

    if not equity_curve.empty and "equity" in equity_curve.columns:
        equity = equity_curve.set_index("date")["equity"].astype(float).sort_index()
        daily_returns = equity.pct_change().dropna()
        max_drawdown = abs(_max_drawdown_pct(equity))
        sharpe = _sharpe_ratio(daily_returns, risk_free_rate=risk_free_rate)
        sortino = _sortino_ratio(daily_returns, risk_free_rate=risk_free_rate)

        total_days = len(equity)
        total_pnl_pct_val = (
            (total_pnl / float(initial_cash)) * 100.0 if initial_cash else 0.0
        )
        annual_return = _annualize_returns(total_pnl_pct_val, total_days)
        annual_volatility = _annualize_volatility(daily_returns)

    calmar = _calmar_ratio(annual_return, max_drawdown * 100.0)
    hit_rate = _calculate_hit_rate(trades)
    expectancy = _calculate_expectancy(trades)

    periods = max(1, len(daily_returns))
    turnover = _calculate_turnover(trades, initial_cash, periods)

    avg_position_value = total_pnl / total_trades if total_trades else 0.0
    capacity = _calculate_capacity_proxy(avg_position_value, initial_cash * 0.1)

    stats = {
        "total_trades": total_trades,
        "winning_trades": int(len(wins)),
        "losing_trades": int(len(losses)),
        "win_rate_pct": float(win_rate * 100.0),
        "total_pnl_dollars": float(total_pnl),
        "total_pnl_pct": float((total_pnl / float(initial_cash)) * 100.0)
        if initial_cash
        else 0.0,
        "avg_trade_dollars": float(avg_trade),
        "avg_trade_pct": float(
            np.mean([t.get("pnl_pct", 0.0) for t in trades]) if trades else 0.0
        ),
        "profit_factor": float(profit_factor if np.isfinite(profit_factor) else 999.0),
        "max_drawdown_pct": float(max_drawdown * 100.0),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "calmar_ratio": float(calmar),
        "annual_return_pct": float(annual_return),
        "annual_volatility_pct": float(annual_volatility),
        "hit_rate": float(hit_rate),
        "expectancy_pct": float(expectancy),
        "turnover": float(turnover),
        "capacity_proxy": float(capacity),
    }

    return {
        "total_trades": total_trades,
        "win_rate": float(win_rate),
        "profit_factor": float(stats["profit_factor"]),
        "total_pnl": float(total_pnl),
        "avg_trade": float(avg_trade),
        "max_drawdown": float(max_drawdown),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "calmar_ratio": float(calmar),
        "stats": stats,
    }


def compute_extended_metrics(
    trades: List[Dict[str, Any]],
    equity_curve: pd.DataFrame,
    initial_cash: float,
    risk_free_rate: float = 0.05,
    predictions: Optional[List[bool]] = None,
) -> Dict[str, Any]:
    """Compute extended metrics bundle including all Phase 3 requirements."""
    summary = summarize_backtest(trades, equity_curve, initial_cash, risk_free_rate)
    stats = summary.get("stats", {})

    precision = 0.0
    recall = 0.0
    if predictions is not None and len(predictions) == len(trades):
        precision, recall = _calculate_precision_recall(trades, predictions)

    return {
        "sharpe_ratio": stats.get("sharpe_ratio", 0.0),
        "sortino_ratio": stats.get("sortino_ratio", 0.0),
        "calmar_ratio": stats.get("calmar_ratio", 0.0),
        "max_drawdown_pct": stats.get("max_drawdown_pct", 0.0),
        "win_rate": stats.get("win_rate_pct", 0.0) / 100.0,
        "profit_factor": stats.get("profit_factor", 0.0),
        "hit_rate": stats.get("hit_rate", 0.0),
        "expectancy": stats.get("expectancy_pct", 0.0),
        "precision": precision,
        "recall": recall,
        "turnover": stats.get("turnover", 0.0),
        "capacity_proxy": stats.get("capacity_proxy", 0.0),
        "annual_return_pct": stats.get("annual_return_pct", 0.0),
        "annual_volatility_pct": stats.get("annual_volatility_pct", 0.0),
        "total_trades": stats.get("total_trades", 0),
        "winning_trades": stats.get("winning_trades", 0),
        "losing_trades": stats.get("losing_trades", 0),
    }
