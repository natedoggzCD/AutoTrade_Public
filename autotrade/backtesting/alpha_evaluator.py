"""Per-symbol alpha evaluator for the strategy lab."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from autotrade.backtesting.alpha_catalog.base import (
    AlphaContext,
    AlphaDefinition,
    require_bar_columns,
)
from autotrade.backtesting.cost_model import CostModel
from autotrade.backtesting.leakage_detector import LeakageDetector
from autotrade.backtesting.metrics import _sharpe_ratio
from autotrade.backtesting.purged_walk_forward import (
    FoldEvaluation,
    PurgedKFoldConfig,
    fold_boundaries_for_leakage_check,
    generate_purged_folds,
)
from autotrade.backtesting.statistical_controls import compute_deflated_sharpe


@dataclass(frozen=True)
class TradeEvaluation:
    entry_date: object
    exit_date: object
    side: str
    gross_return_pct: float
    net_return_pct: float


@dataclass
class SymbolAlphaEvaluation:
    symbol: str
    alpha_id: str
    n_trades: int
    raw_mean_return: float
    raw_std_return: float
    raw_pf: float
    raw_sharpe: float
    deflated_sharpe: float
    buy_and_hold_return: float
    beats_bnh: bool
    per_fold: list[FoldEvaluation] = field(default_factory=list)
    leakage_passed: bool = True


def _empty_result(
    symbol: str,
    alpha_id: str,
    buy_and_hold_return: float = 0.0,
    leakage_passed: bool = True,
) -> SymbolAlphaEvaluation:
    return SymbolAlphaEvaluation(
        symbol=symbol,
        alpha_id=alpha_id,
        n_trades=0,
        raw_mean_return=0.0,
        raw_std_return=0.0,
        raw_pf=0.0,
        raw_sharpe=0.0,
        deflated_sharpe=0.0,
        buy_and_hold_return=buy_and_hold_return,
        beats_bnh=False,
        per_fold=[],
        leakage_passed=leakage_passed,
    )


def _profit_factor(returns_pct: list[float]) -> float:
    wins = [r for r in returns_pct if r > 0.0]
    losses = [r for r in returns_pct if r < 0.0]
    if not wins:
        return 0.0
    avg_loss = abs(float(np.mean(losses))) if losses else abs(float(np.mean(wins)))
    # A one-trade pseudo-loss prevents tiny all-winner samples from producing
    # effectively infinite PF while preserving strong multi-trade ledgers.
    gross_losses = abs(sum(losses)) + avg_loss
    return float(sum(wins) / gross_losses)


def _trade_sharpe(returns_pct: list[float]) -> float:
    if len(returns_pct) < 2:
        return 0.0
    series = pd.Series([r / 100.0 for r in returns_pct], dtype=float)
    return _sharpe_ratio(series, risk_free_rate=0.0)


def _date_array(values: pd.Series) -> np.ndarray:
    return pd.to_datetime(values).dt.date.to_numpy()


def _next_bar_index(
    date_to_next_index: dict[object, int], signal_date: object
) -> Optional[int]:
    return date_to_next_index.get(pd.to_datetime(signal_date).date())


def _signals_to_trades(
    signals: pd.DataFrame,
    bars: pd.DataFrame,
    symbol: str,
    cost_model: CostModel,
) -> list[TradeEvaluation]:
    if signals.empty:
        return []

    df = bars.sort_values("date").reset_index(drop=True).copy()
    df_dates = _date_array(df["date"])
    date_to_next_index = {df_dates[i]: i + 1 for i in range(len(df_dates) - 1)}

    sig = signals.sort_values("date").reset_index(drop=True)
    trades: list[TradeEvaluation] = []
    pending: Optional[tuple[object, str]] = None
    for row in sig.itertuples(index=False):
        if bool(row.entry) and pending is None:
            pending = (row.date, str(row.side or "long").lower())
            continue
        if bool(row.exit) and pending is not None:
            entry_signal_date, side = pending
            entry_idx = _next_bar_index(date_to_next_index, entry_signal_date)
            exit_idx = _next_bar_index(date_to_next_index, row.date)
            pending = None
            if entry_idx is None or exit_idx is None or exit_idx <= entry_idx:
                continue
            entry_price = float(df.iloc[entry_idx]["open"])
            exit_price = float(df.iloc[exit_idx]["open"])
            if entry_price <= 0:
                continue
            gross_decimal = (exit_price / entry_price) - 1.0
            if side == "short":
                gross_decimal *= -1.0
            gross_pct = gross_decimal * 100.0
            net_pct = cost_model.apply_round_trip(gross_pct, symbol)
            trades.append(
                TradeEvaluation(
                    entry_date=df.iloc[entry_idx]["date"],
                    exit_date=df.iloc[exit_idx]["date"],
                    side=side,
                    gross_return_pct=float(gross_pct),
                    net_return_pct=float(net_pct),
                )
            )
    return trades


def _fold_eval(
    fold_index: int, train_returns: list[float], test_returns: list[float]
) -> FoldEvaluation:
    return FoldEvaluation(
        fold_index=fold_index,
        n_train_trades=len(train_returns),
        n_test_trades=len(test_returns),
        train_sharpe=_trade_sharpe(train_returns),
        test_sharpe=_trade_sharpe(test_returns),
        train_pf=_profit_factor(train_returns),
        test_pf=_profit_factor(test_returns),
        test_returns=list(test_returns),
    )


def evaluate_alpha_on_symbol(
    alpha: AlphaDefinition,
    symbol: str,
    bars: pd.DataFrame,
    cost_model: CostModel,
    wf_config: PurgedKFoldConfig,
    ctx: Optional[AlphaContext] = None,
    n_trials: int = 1,
) -> SymbolAlphaEvaluation:
    """Evaluate one alpha on one symbol using next-open trade realization."""

    require_bar_columns(bars)
    df = bars.sort_values("date").reset_index(drop=True).copy()
    buy_and_hold_return = 0.0
    if len(df) >= 2 and float(df["close"].iloc[0]) > 0:
        buy_and_hold_return = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1.0)

    folds = generate_purged_folds(df["date"].tolist(), wf_config)
    leakage_check = LeakageDetector(fail_on_leakage=False).check_fold_boundaries(
        fold_boundaries_for_leakage_check(folds)
    )
    if not leakage_check.passed:
        return _empty_result(
            symbol, alpha.alpha_id, buy_and_hold_return, leakage_passed=False
        )

    signals = alpha.generate(df, ctx)
    trades = _signals_to_trades(signals, df, symbol, cost_model)
    returns = [t.net_return_pct for t in trades]
    if not returns:
        return _empty_result(
            symbol, alpha.alpha_id, buy_and_hold_return, leakage_passed=True
        )

    entry_dates = np.array(pd.to_datetime([t.entry_date for t in trades]).date)
    per_fold: list[FoldEvaluation] = []
    fold_sharpes: list[float] = []
    for fold in folds:
        train_returns = [
            returns[i]
            for i in range(len(returns))
            if fold.train_start <= entry_dates[i] <= fold.train_end
        ]
        test_returns = [
            returns[i]
            for i in range(len(returns))
            if fold.test_start <= entry_dates[i] <= fold.test_end
        ]
        fold_result = _fold_eval(fold.fold_index, train_returns, test_returns)
        per_fold.append(fold_result)
        fold_sharpes.append(fold_result.test_sharpe)

    raw_sharpe = _trade_sharpe(returns)
    dsr_input = (
        fold_sharpes if any(abs(v) > 0.0 for v in fold_sharpes) else [raw_sharpe]
    )
    dsr = compute_deflated_sharpe(dsr_input, n_trials=max(1, int(n_trials))).dsr
    deflated_sharpe = min(float(dsr), raw_sharpe)
    strategy_total_return = sum(returns) / 100.0

    return SymbolAlphaEvaluation(
        symbol=symbol,
        alpha_id=alpha.alpha_id,
        n_trades=len(returns),
        raw_mean_return=float(np.mean(returns)),
        raw_std_return=float(np.std(returns, ddof=0)),
        raw_pf=_profit_factor(returns),
        raw_sharpe=raw_sharpe,
        deflated_sharpe=deflated_sharpe,
        buy_and_hold_return=buy_and_hold_return,
        beats_bnh=bool(strategy_total_return > buy_and_hold_return),
        per_fold=per_fold,
        leakage_passed=True,
    )
