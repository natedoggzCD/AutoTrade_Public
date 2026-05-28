from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from config.config_loader import BacktestConfig, get_config
from autotrade.backtesting.data import (
    get_latest_prediction_date,
    load_benchmark_close,
    load_daily_bars,
    load_predictions,
    resolve_backtest_paths,
)
from autotrade.backtesting.execution import ExecutionModel
from autotrade.backtesting.metrics import summarize_backtest
from autotrade.signals.unified_strategy import PARTIAL_AT_R1, UnifiedStrategy

logger = logging.getLogger(__name__)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _to_date(value) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return pd.to_datetime(value).date()


@dataclass
class Position:
    ticker: str
    entry_date: date
    entry_price: float
    qty: int

    stop_price: float
    r1_price: float
    r2_price: float
    target_price: float
    r2_in_range: bool
    risk_reward: float

    partial_taken: bool = False
    remaining_qty: int = 0
    days_held: int = 0

    realized_pnl: float = 0.0
    realized_commission: float = 0.0
    partial_exits: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.remaining_qty <= 0:
            self.remaining_qty = int(self.qty)


class BacktestEngine:
    """Local-data backtest engine with lightweight execution modeling."""

    def __init__(
        self,
        config_override: Optional[Dict[str, Any]] = None,
        strategy_config: Optional[Any] = None,
        parent_logger: Optional[logging.Logger] = None,
    ):
        cfg = get_config().backtest
        if strategy_config is not None and hasattr(strategy_config, "to_backtest_override"):
            try:
                strategy_override = strategy_config.to_backtest_override()
                config_override = _deep_merge(strategy_override, config_override or {})
            except Exception:
                pass
        if config_override:
            cfg = BacktestConfig(**_deep_merge(cfg.model_dump(), config_override))

        self.config = cfg
        self.portfolio_cfg = get_config().portfolio
        self.risk_limits_cfg = get_config().risk_limits

        self.logger = parent_logger or logging.getLogger("backtest_engine")

        paths = resolve_backtest_paths()
        self.daily_features_path = paths.daily_features_path

        self.exec_model = ExecutionModel(
            commission_pct=cfg.commission_pct,
            slippage_pct=cfg.slippage_pct,
            spread_pct=getattr(cfg, "spread_pct", 0.001),
            partial_fill_min=getattr(cfg, "partial_fill_min", 1.0),
            partial_fill_max=getattr(cfg, "partial_fill_max", 1.0),
            seed=getattr(cfg, "seed", 42),
        )

    def _position_value_cap(self, equity: float) -> float:
        max_pct = float(self.risk_limits_cfg.max_position_pct) / 100.0
        cap_pct = equity * max_pct if equity > 0 else float("inf")

        cap_cfg = float(getattr(self.config, "max_position_value", float("inf")) or float("inf"))
        cap_live = float(self.portfolio_cfg.position_size_max or float("inf"))

        return min(cap_pct, cap_cfg, cap_live)

    def _desired_position_value(self, equity: float) -> float:
        cap = self._position_value_cap(equity)
        target = float(self.portfolio_cfg.position_size_target)
        return min(target, cap)

    def _max_positions(self) -> int:
        return int(min(int(self.config.max_positions), int(self.portfolio_cfg.max_positions)))

    def _score_signals(self, preds: pd.DataFrame, strategy: UnifiedStrategy) -> pd.DataFrame:
        if preds.empty:
            return preds

        min_bullish = float(self.config.min_bullish)
        preds = preds[preds["bullish_score"].fillna(0) >= min_bullish].copy()
        if preds.empty:
            return preds

        lessons_pass: list[bool] = []
        lesson_score: list[float] = []

        for _, row in preds.iterrows():
            candidate = {
                "bullish_score": row.get("bullish_score", 0),
                "s1_strength": row.get("s1_strength", 0),
                "r1_strength": row.get("r1_strength", 0),
                "atr_percent": row.get("atr_percent", 0),
                "distance_to_s1_pct": row.get("distance_to_s1_pct", 0),
                "distance_to_r1_pct": row.get("distance_to_r1_pct", 0),
                "spread_pct": row.get("spread_pct"),
                "momentum_score": row.get("momentum_score"),
                "poc_distance_pct": row.get("poc_distance_pct"),
                "in_value_area": row.get("in_value_area"),
            }
            passes, score, _rules = strategy.passes_filter(candidate)
            lessons_pass.append(bool(passes))
            lesson_score.append(float(score))

        preds["lessons_pass"] = lessons_pass
        preds["lesson_score"] = lesson_score
        preds = preds[preds["lessons_pass"]].copy()

        if preds.empty:
            return preds

        preds["rank_score"] = (
            preds["bullish_score"].fillna(0).astype(float) * 100.0
            + preds["rr_ratio"].fillna(0).astype(float) * 10.0
            + preds["lesson_score"].fillna(0).astype(float)
        )
        return preds.sort_values(["prediction_date", "rank_score"], ascending=[True, False])

    def run(
        self,
        symbols: Sequence[str],
        lookback_days: Optional[int] = 90,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
        save_path: Optional[Path] = None,
        walk_forward: bool = False,
        optimize_thresholds: bool = False,
        log_summary: bool = True,
    ) -> Dict[str, Any]:
        symbols = [str(s).upper().strip() for s in (symbols or []) if str(s).strip()]
        symbols = sorted(set(symbols))

        if not symbols:
            return {"error": "No symbols provided"}

        if not self.daily_features_path.exists():
            return {"error": "Daily features parquet not found", "daily_features_path": str(self.daily_features_path)}

        if end_date is None and lookback_days:
            latest = get_latest_prediction_date(self.daily_features_path)
            if not latest:
                return {"error": "No market dates found in daily parquet"}
            end_d = latest
            start_d = end_d - timedelta(days=int(lookback_days))
        else:
            start_d = _to_date(start_date) if start_date else None
            end_d = _to_date(end_date) if end_date else None
            if not start_d or not end_d:
                return {"error": "start_date and end_date are required when lookback_days is not provided"}

        if start_d > end_d:
            start_d, end_d = end_d, start_d

        if walk_forward:
            return self._run_walk_forward(
                symbols=symbols,
                start_date=start_d,
                end_date=end_d,
                optimize_thresholds=optimize_thresholds,
                save_path=save_path,
                log_summary=log_summary,
            )

        return self._run_single_period(
            symbols=symbols,
            start_date=start_d,
            end_date=end_d,
            save_path=save_path,
            log_summary=log_summary,
        )

    def run_walk_forward(
        self,
        symbols: Sequence[str],
        lookback_days: Optional[int] = 360,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
        optimize_thresholds: bool = False,
        save_path: Optional[Path] = None,
        log_summary: bool = True,
    ) -> Dict[str, Any]:
        """Public walk-forward helper with explicit intent and fold reporting output."""
        return self.run(
            symbols=symbols,
            lookback_days=lookback_days,
            start_date=start_date,
            end_date=end_date,
            save_path=save_path,
            walk_forward=True,
            optimize_thresholds=optimize_thresholds,
            log_summary=log_summary,
        )

    def _run_single_period(
        self,
        symbols: Sequence[str],
        start_date: date,
        end_date: date,
        save_path: Optional[Path],
        log_summary: bool,
    ) -> Dict[str, Any]:
        cfg = self.config
        end_extended = end_date + timedelta(days=int(cfg.max_hold_days) + 3)

        preds = load_predictions(
            daily_features_path=self.daily_features_path,
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
        )

        if preds.empty:
            return {
                "error": "No candidate signals found for requested window",
                "start_date": str(start_date),
                "end_date": str(end_date),
            }

        benchmark_symbol = str(getattr(cfg, "benchmark_symbol", "SPY") or "SPY").upper()
        bars = load_daily_bars(
            daily_features_path=self.daily_features_path,
            start_date=start_date,
            end_date=end_extended,
            symbols=list(symbols) + [benchmark_symbol],
        )

        if bars.empty:
            return {"error": "No daily bars loaded", "start_date": str(start_date), "end_date": str(end_date)}

        bars = bars.sort_values(["ticker", "date"]).copy()
        bars["next_date"] = bars.groupby("ticker")["date"].shift(-1)
        bars["next_open"] = bars.groupby("ticker")["open"].shift(-1)

        join_cols = bars[["ticker", "date", "next_date", "next_open"]].rename(columns={"date": "prediction_date"})
        merged = preds.merge(join_cols, on=["ticker", "prediction_date"], how="left")
        merged = merged.dropna(subset=["next_date", "next_open"]).copy()
        merged["entry_date"] = pd.to_datetime(merged["next_date"]).dt.date
        merged["entry_open"] = merged["next_open"].astype(float)
        merged = merged[merged["entry_date"] <= end_date].copy()

        if merged.empty:
            return {"error": "No entries possible in requested window (missing next day bars)", "start_date": str(start_date), "end_date": str(end_date)}

        strategy = UnifiedStrategy(min_score=float(cfg.min_lesson_score))
        merged = self._score_signals(merged, strategy)
        if merged.empty:
            return {"error": "No signals passed filters", "start_date": str(start_date), "end_date": str(end_date)}

        signals_per_day = int(cfg.signals_per_day)
        by_day: Dict[date, List[Dict[str, Any]]] = {}
        for entry_d, day_df in merged.groupby("entry_date"):
            day_sorted = day_df.sort_values("rank_score", ascending=False).head(signals_per_day)
            by_day[_to_date(entry_d)] = day_sorted.to_dict(orient="records")

        bars_idx = bars.set_index(["ticker", "date"]).sort_index()
        benchmark_close = load_benchmark_close(
            daily_features_path=self.daily_features_path,
            benchmark_symbol=benchmark_symbol,
            start_date=start_date,
            end_date=end_date,
        )

        rng = self.exec_model.rng()
        initial_cash = float(getattr(cfg, "initial_cash", 100_000.0))
        cash = initial_cash
        positions: Dict[str, Position] = {}
        trades: List[Dict[str, Any]] = []
        equity_rows: List[Dict[str, Any]] = []

        all_dates = sorted({d for d in bars["date"].unique() if start_date <= d <= end_date})
        max_positions = self._max_positions()

        for current_date in all_dates:
            # --- exits ---
            for ticker in list(positions.keys()):
                pos = positions.get(ticker)
                if not pos:
                    continue

                key = (ticker, current_date)
                if key not in bars_idx.index:
                    continue

                bar = bars_idx.loc[key]
                high = float(bar.get("high", 0) or 0)
                low = float(bar.get("low", 0) or 0)
                close = float(bar.get("close", 0) or 0)

                exit_reason = None
                exit_price = 0.0
                exit_qty = 0

                # Stop loss (priority)
                if pos.stop_price > 0 and low > 0 and low <= pos.stop_price:
                    exit_reason = "stop_loss"
                    exit_price = float(pos.stop_price)
                    exit_qty = int(pos.remaining_qty)
                else:
                    # Partial at R1 (only if targeting R2)
                    if pos.r2_in_range and (not pos.partial_taken) and pos.r1_price > 0 and high >= pos.r1_price:
                        partial_qty = int(max(0, round(pos.qty * float(PARTIAL_AT_R1))))
                        partial_qty = min(partial_qty, int(pos.remaining_qty))
                        if partial_qty > 0:
                            sell_px = self.exec_model.effective_price(pos.r1_price, side="sell")
                            notional = sell_px * partial_qty
                            comm = self.exec_model.commission(notional)
                            cash += notional - comm

                            pnl = (sell_px - pos.entry_price) * partial_qty
                            pos.realized_pnl += pnl
                            pos.realized_commission += comm
                            pos.remaining_qty -= partial_qty
                            pos.partial_taken = True
                            pos.partial_exits.append(
                                {
                                    "date": str(current_date),
                                    "qty": partial_qty,
                                    "price": sell_px,
                                    "reason": "partial_r1",
                                    "pnl_dollars": pnl - comm,
                                }
                            )

                    # Final target
                    if pos.target_price > 0 and high >= pos.target_price and pos.remaining_qty > 0:
                        exit_reason = "target_hit"
                        exit_price = float(pos.target_price)
                        exit_qty = int(pos.remaining_qty)

                # Time exit
                if exit_reason is None:
                    pos.days_held += 1
                    if pos.days_held >= int(cfg.max_hold_days) and pos.remaining_qty > 0:
                        exit_reason = "time_exit"
                        exit_price = close
                        exit_qty = int(pos.remaining_qty)

                if exit_reason and exit_qty > 0 and exit_price > 0:
                    sell_px = self.exec_model.effective_price(exit_price, side="sell")
                    notional = sell_px * exit_qty
                    comm = self.exec_model.commission(notional)
                    cash += notional - comm

                    pnl = (sell_px - pos.entry_price) * exit_qty
                    pos.realized_pnl += pnl
                    pos.realized_commission += comm
                    pos.remaining_qty -= exit_qty

                    total_comm = pos.realized_commission
                    total_pnl = pos.realized_pnl - total_comm
                    hold_days = int((current_date - pos.entry_date).days) + 1

                    trades.append(
                        {
                            "ticker": pos.ticker,
                            "entry_date": str(pos.entry_date),
                            "entry_price": float(pos.entry_price),
                            "qty": int(pos.qty),
                            "exit_date": str(current_date),
                            "exit_price": float(sell_px),
                            "pnl_dollars": float(total_pnl),
                            "pnl_pct": float((total_pnl / (pos.entry_price * pos.qty)) * 100.0) if pos.entry_price > 0 and pos.qty > 0 else 0.0,
                            "hold_days": hold_days,
                            "exit_reason": exit_reason,
                            "risk_reward": float(pos.risk_reward),
                            "partial_exits": pos.partial_exits,
                        }
                    )

                    del positions[ticker]

            # --- entries ---
            available_slots = max_positions - len(positions)
            if available_slots > 0:
                candidates = by_day.get(current_date, [])
                if candidates:
                    # Mark-to-market equity for sizing
                    equity = cash
                    for tkr, pos in positions.items():
                        k = (tkr, current_date)
                        if k in bars_idx.index:
                            equity += float(bars_idx.loc[k].get("close", 0) or 0) * pos.remaining_qty
                    desired_value = self._desired_position_value(equity)

                    for cand in candidates:
                        if available_slots <= 0:
                            break
                        tkr = str(cand.get("ticker", "")).upper()
                        if not tkr or tkr in positions:
                            continue

                        entry_open = float(cand.get("entry_open", 0) or 0)
                        buy_px = self.exec_model.effective_price(entry_open, side="buy")
                        if buy_px <= 0:
                            continue

                        shares_target = int(desired_value / buy_px) if desired_value > 0 else 0
                        if shares_target <= 0:
                            continue

                        fill_frac = self.exec_model.sample_fill_fraction(rng)
                        shares = max(1, int(shares_target * fill_frac))

                        # Fit to cash
                        notional = buy_px * shares
                        comm = self.exec_model.commission(notional)
                        total_cost = notional + comm
                        if total_cost > cash:
                            shares = int(cash / max(buy_px * (1 + self.exec_model.commission_pct), 1e-9))
                            shares = max(0, shares)
                            if shares <= 0:
                                continue
                            notional = buy_px * shares
                            comm = self.exec_model.commission(notional)
                            total_cost = notional + comm
                            if total_cost > cash:
                                continue

                        cash -= total_cost

                        levels = strategy.calculate_levels(
                            entry_price=buy_px,
                            s1_price=float(cand.get("s1_price", 0) or 0),
                            s1_strength=float(cand.get("s1_strength", 0) or 0),
                            r1_price=float(cand.get("r1_price", 0) or 0),
                            r1_strength=float(cand.get("r1_strength", 0) or 0),
                            s2_price=float(cand.get("s2_price", 0) or 0),
                            r2_price=float(cand.get("r2_price", 0) or 0),
                            atr=float(cand.get("atr_14", 0) or 0),
                        )
                        atr_val = float(cand.get("atr_14", 0) or 0)
                        if atr_val > 0:
                            stop_atr = float(getattr(cfg, "stop_atr", 0.0) or 0.0)
                            target_atr = float(getattr(cfg, "target_atr", 0.0) or 0.0)
                            if stop_atr > 0:
                                atr_stop = buy_px - (atr_val * stop_atr)
                                if atr_stop > 0:
                                    if levels.stop_price > 0:
                                        levels.stop_price = max(float(levels.stop_price), float(atr_stop))
                                    else:
                                        levels.stop_price = float(atr_stop)
                            if target_atr > 0:
                                atr_target = buy_px + (atr_val * target_atr)
                                if atr_target > 0:
                                    levels.target_price = float(atr_target)
                                    if levels.r1_price <= 0:
                                        levels.r1_price = float(atr_target)
                            risk = buy_px - float(levels.stop_price or 0.0)
                            reward = float(levels.target_price or 0.0) - buy_px
                            if risk > 0:
                                levels.risk_reward = reward / risk

                        positions[tkr] = Position(
                            ticker=tkr,
                            entry_date=_to_date(cand.get("entry_date")),
                            entry_price=float(buy_px),
                            qty=int(shares),
                            stop_price=float(levels.stop_price),
                            r1_price=float(levels.r1_price or 0),
                            r2_price=float(levels.r2_price or 0),
                            target_price=float(levels.target_price or 0),
                            r2_in_range=bool(levels.r2_in_range),
                            risk_reward=float(levels.risk_reward or 0),
                            realized_commission=float(comm),
                        )

                        available_slots -= 1

            # --- equity curve ---
            equity = cash
            positions_value = 0.0
            for tkr, pos in positions.items():
                k = (tkr, current_date)
                if k in bars_idx.index:
                    px = float(bars_idx.loc[k].get("close", 0) or 0)
                    positions_value += px * pos.remaining_qty
            equity += positions_value
            equity_rows.append(
                {
                    "date": str(current_date),
                    "cash": float(cash),
                    "positions_value": float(positions_value),
                    "equity": float(equity),
                    "open_positions": int(len(positions)),
                }
            )

        equity_df = pd.DataFrame(equity_rows)
        summary = summarize_backtest(
            trades=trades,
            equity_curve=equity_df,
            initial_cash=initial_cash,
            risk_free_rate=float(cfg.risk_free_rate),
        )

        benchmark = {}
        if not benchmark_close.empty and len(benchmark_close) >= 2:
            b0 = float(benchmark_close.iloc[0])
            b1 = float(benchmark_close.iloc[-1])
            benchmark = {
                "symbol": benchmark_symbol,
                "start_close": b0,
                "end_close": b1,
                "return_pct": float(((b1 - b0) / b0) * 100.0) if b0 > 0 else 0.0,
            }

        payload: Dict[str, Any] = {
            "backtest_date": datetime.now().isoformat(),
            "start_date": str(start_date),
            "end_date": str(end_date),
            "symbols": list(symbols),
            "config": {
                "backtest": self.config.model_dump(),
                "portfolio": self.portfolio_cfg.model_dump(),
                "risk_limits": self.risk_limits_cfg.model_dump(),
            },
            "benchmark": benchmark,
            "trades": trades,
            "equity_curve": equity_rows,
        }
        payload.update(summary)

        if log_summary:
            self.logger.info(
                "[BACKTEST] trades=%s win_rate=%.1f%% PF=%.2f pnl=$%.2f dd=%.1f%% sharpe=%.2f bench=%s",
                payload.get("total_trades", 0),
                payload.get("win_rate", 0.0) * 100.0,
                payload.get("profit_factor", 0.0),
                payload.get("total_pnl", 0.0),
                payload.get("max_drawdown", 0.0) * 100.0,
                payload.get("sharpe_ratio", 0.0),
                f"{benchmark_symbol} {benchmark.get('return_pct', 0.0):.1f}%" if benchmark else "n/a",
            )

        if save_path:
            save_path.parent.mkdir(exist_ok=True)
            save_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            self.logger.info(f"[BACKTEST] Saved results: {save_path}")

        return payload

    def _run_walk_forward(
        self,
        symbols: Sequence[str],
        start_date: date,
        end_date: date,
        optimize_thresholds: bool,
        save_path: Optional[Path],
        log_summary: bool,
    ) -> Dict[str, Any]:
        cfg = self.config
        train_days = int(cfg.walk_forward_train_days)
        test_days = int(cfg.walk_forward_test_days)

        preds = load_predictions(
            daily_features_path=self.daily_features_path,
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
        )
        if preds.empty:
            return {
                "error": "No candidate signals found for requested window",
                "start_date": str(start_date),
                "end_date": str(end_date),
            }

        unique_dates = sorted(set(pd.to_datetime(preds["prediction_date"]).dt.date))
        if len(unique_dates) < (train_days + test_days):
            return {"error": "Not enough dates for walk-forward", "available_days": len(unique_dates)}

        windows: List[Dict[str, Any]] = []
        idx = 0
        window_no = 0
        while idx + train_days + test_days <= len(unique_dates):
            window_no += 1
            train_start = unique_dates[idx]
            train_end = unique_dates[idx + train_days - 1]
            test_start = unique_dates[idx + train_days]
            test_end = unique_dates[idx + train_days + test_days - 1]

            min_bullish = float(cfg.min_bullish)
            if optimize_thresholds:
                min_bullish = self._calibrate_min_bullish(
                    preds=preds,
                    train_start=train_start,
                    train_end=train_end,
                    target_per_day=int(cfg.signals_per_day),
                )

            override = {"min_bullish": min_bullish}
            engine = BacktestEngine(config_override=override, parent_logger=self.logger)
            train_res = engine._run_single_period(
                symbols=symbols,
                start_date=train_start,
                end_date=train_end,
                save_path=None,
                log_summary=False,
            )
            test_res = engine._run_single_period(
                symbols=symbols,
                start_date=test_start,
                end_date=test_end,
                save_path=None,
                log_summary=False,
            )

            train_pf = float(train_res.get("profit_factor", 0.0) or 0.0)
            test_pf = float(test_res.get("profit_factor", 0.0) or 0.0)
            degradation_pct = ((train_pf - test_pf) / train_pf * 100.0) if train_pf > 0 else 0.0

            windows.append(
                {
                    "window": window_no,
                    "train_start": str(train_start),
                    "train_end": str(train_end),
                    "test_start": str(test_start),
                    "test_end": str(test_end),
                    "min_bullish": float(min_bullish),
                    "train_metrics": {
                        "total_trades": train_res.get("total_trades", 0),
                        "win_rate": train_res.get("win_rate", 0.0),
                        "profit_factor": train_res.get("profit_factor", 0.0),
                        "total_pnl": train_res.get("total_pnl", 0.0),
                        "max_drawdown": train_res.get("max_drawdown", 0.0),
                        "sharpe_ratio": train_res.get("sharpe_ratio", 0.0),
                    },
                    "test_metrics": {
                        "total_trades": test_res.get("total_trades", 0),
                        "win_rate": test_res.get("win_rate", 0.0),
                        "profit_factor": test_res.get("profit_factor", 0.0),
                        "total_pnl": test_res.get("total_pnl", 0.0),
                        "max_drawdown": test_res.get("max_drawdown", 0.0),
                        "sharpe_ratio": test_res.get("sharpe_ratio", 0.0),
                        "benchmark_return_pct": (test_res.get("benchmark") or {}).get("return_pct", 0.0),
                    },
                    # Backward compatibility with existing consumers expecting test-window metrics.
                    "metrics": {
                        "total_trades": test_res.get("total_trades", 0),
                        "win_rate": test_res.get("win_rate", 0.0),
                        "profit_factor": test_res.get("profit_factor", 0.0),
                        "total_pnl": test_res.get("total_pnl", 0.0),
                        "max_drawdown": test_res.get("max_drawdown", 0.0),
                        "sharpe_ratio": test_res.get("sharpe_ratio", 0.0),
                        "benchmark_return_pct": (test_res.get("benchmark") or {}).get("return_pct", 0.0),
                    },
                    "degradation_pct": float(degradation_pct),
                }
            )

            idx += test_days

        def _avg(section: str, key: str) -> float:
            vals = [float((w.get(section) or {}).get(key, 0.0) or 0.0) for w in windows]
            return float(sum(vals) / len(vals)) if vals else 0.0

        avg_train_pf = _avg("train_metrics", "profit_factor")
        avg_test_pf = _avg("test_metrics", "profit_factor")
        avg_degradation = ((avg_train_pf - avg_test_pf) / avg_train_pf * 100.0) if avg_train_pf > 0 else 0.0

        payload = {
            "backtest_date": datetime.now().isoformat(),
            "walk_forward": True,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "symbols": list(symbols),
            "config": {"backtest": self.config.model_dump()},
            "windows": windows,
            "avg_metrics": {
                "total_trades": float(sum(float((w.get("test_metrics") or {}).get("total_trades", 0) or 0) for w in windows)),
                "train_win_rate": _avg("train_metrics", "win_rate"),
                "test_win_rate": _avg("test_metrics", "win_rate"),
                "win_rate": _avg("test_metrics", "win_rate"),
                "train_profit_factor": avg_train_pf,
                "test_profit_factor": avg_test_pf,
                "profit_factor": avg_test_pf,
                "train_total_pnl": _avg("train_metrics", "total_pnl"),
                "test_total_pnl": _avg("test_metrics", "total_pnl"),
                "total_pnl": _avg("test_metrics", "total_pnl"),
                "max_drawdown": _avg("test_metrics", "max_drawdown"),
                "sharpe_ratio": _avg("test_metrics", "sharpe_ratio"),
                "benchmark_return_pct": _avg("test_metrics", "benchmark_return_pct"),
                "degradation_pct": float(avg_degradation),
            },
        }

        if log_summary:
            self.logger.info(
                "[WALK_FORWARD] windows=%s avg_win_rate=%.1f%% avg_PF=%.2f avg_pnl=$%.2f",
                len(windows),
                float(payload["avg_metrics"]["win_rate"]) * 100.0,
                float(payload["avg_metrics"]["profit_factor"]),
                float(payload["avg_metrics"]["total_pnl"]),
            )

        if save_path:
            save_path.parent.mkdir(exist_ok=True)
            save_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            self.logger.info(f"[WALK_FORWARD] Saved results: {save_path}")

        return payload

    def _calibrate_min_bullish(
        self,
        preds: pd.DataFrame,
        train_start: date,
        train_end: date,
        target_per_day: int,
    ) -> float:
        df = preds[(preds["prediction_date"] >= train_start) & (preds["prediction_date"] <= train_end)].copy()
        if df.empty:
            return float(self.config.min_bullish)

        thresholds = [2.0, 3.0, 4.0, 5.0, 6.0]
        best = float(self.config.min_bullish)
        best_diff = float("inf")

        dates = sorted(set(df["prediction_date"]))
        for thr in thresholds:
            counts = []
            for d in dates:
                counts.append(int((df[df["prediction_date"] == d]["bullish_score"].fillna(0) >= thr).sum()))
            avg_daily = float(sum(counts) / len(counts)) if counts else 0.0
            diff = abs(avg_daily - float(target_per_day))
            if diff < best_diff:
                best = thr
                best_diff = diff

        return float(best)
