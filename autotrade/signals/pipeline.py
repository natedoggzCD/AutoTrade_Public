"""
Modular signal generation pipeline.

This module centralizes staged signal generation with explicit boundaries:
1) feature intake
2) alpha family generation
3) baseline generation
4) optional regime routing
5) calibration + blending
6) action normalization
7) schema enforcement
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.config_loader import get_config
from autotrade.signals.contracts import (
    RegimeLabel,
    SignalAction,
    SignalBatch,
    SignalContext,
    SignalDecision,
    SignalFamily,
    validate_signal_dict,
)
from autotrade.signals.registry import register_signal_zoo_from_config
from autotrade.signals.regime_router import RegimeRouter
from autotrade.signals.strategy_pool import load_validated_strategies

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Container for pipeline outputs."""

    batch: SignalBatch
    legacy_candidates: List[Dict]


class SignalGenerationPipeline:
    """Staged signal generation pipeline with compatibility helpers."""

    def __init__(self, parent_logger: Optional[logging.Logger] = None):
        cfg = get_config()
        self._cfg = cfg.signal_generation
        self._logger = parent_logger or logger
        self._registry = register_signal_zoo_from_config()
        self._seed = int(getattr(self._cfg, "determinism_seed", 42))
        self._regime_router = RegimeRouter(parent_logger)
        self._alpha_min_win_rate = float(
            getattr(self._cfg, "alpha_zoo_min_win_rate", 0.48)
        )
        self._alpha_family_win_rate = self._resolve_alpha_family_win_rates()
        np.random.seed(self._seed)

    @staticmethod
    def _normalize_win_rate(raw_value: object) -> float:
        """Normalize win-rate values from either 0-1 or 0-100 representations."""
        try:
            value = float(raw_value or 0.0)
        except Exception:
            return 0.0
        if value > 1.0:
            value = value / 100.0
        return max(0.0, min(1.0, value))

    @staticmethod
    def _setup_type_to_family(setup_type: str) -> Optional[SignalFamily]:
        label = str(setup_type or "").strip().lower()
        if not label:
            return None
        if "mean_reversion" in label or "vwap" in label:
            return SignalFamily.MEAN_REVERSION
        if "pullback" in label or "ma_bounce" in label or "rsi_cool" in label:
            return SignalFamily.PULLBACK
        if "breakout" in label:
            return SignalFamily.BREAKOUT
        if (
            "momentum" in label
            or "trend_follow" in label
            or "new_high" in label
            or "continuation" in label
        ):
            return SignalFamily.TS_MOMENTUM
        return None

    @staticmethod
    def _resolve_legacy_candidate_family(row: Dict) -> Optional[SignalFamily]:
        """
        Best-effort family inference for legacy screener candidates.

        Priority:
        1) explicit family tags on the candidate
        2) setup/scan type hints
        3) source/reason hints (e.g., volume divergence)
        """
        if not isinstance(row, dict):
            return None

        def _normalize(value: object) -> str:
            return str(value or "").strip().lower()

        def _map_label(label: str) -> Optional[SignalFamily]:
            if not label:
                return None
            if label in {"baseline_a", "signal_a", "a"}:
                return SignalFamily.BASELINE_A
            if label in {"baseline_b", "signal_b", "b"}:
                return SignalFamily.BASELINE_B
            if (
                "mean_reversion" in label
                or "volume_divergence" in label
                or "vwap" in label
            ):
                return SignalFamily.MEAN_REVERSION
            if "pullback" in label or "ma_bounce" in label or "rsi_cool" in label:
                return SignalFamily.PULLBACK
            if "breakout" in label or "donchian" in label or "squeeze" in label:
                return SignalFamily.BREAKOUT
            if "xs_momentum" in label or "cross_section" in label:
                return SignalFamily.XS_MOMENTUM
            if "ts_momentum" in label or "trend_follow" in label or "momentum" in label:
                return SignalFamily.TS_MOMENTUM
            return None

        direct_fields = (
            row.get("signal_family"),
            row.get("family"),
            row.get("alpha_family"),
            row.get("setup_type"),
            row.get("scan_type"),
            row.get("alpha_source"),
        )
        for value in direct_fields:
            resolved = _map_label(_normalize(value))
            if resolved is not None:
                return resolved

        fallback_blob = " ".join(
            _normalize(v) for v in (row.get("reason"), row.get("source"))
        )
        return _map_label(fallback_blob)

    def _resolve_alpha_family_win_rates(self) -> Dict[SignalFamily, float]:
        """
        Build best-known win-rate estimates per alpha family from validated strategies.

        Families with no mapped strategy data are left unconstrained.
        """
        family_scores: Dict[SignalFamily, List[float]] = defaultdict(list)
        try:
            strategies = load_validated_strategies()
        except Exception as e:
            self._logger.debug(
                f"[SignalPipeline] strategy win-rate map unavailable: {e}"
            )
            return {}

        for row in strategies:
            if not isinstance(row, dict):
                continue
            strategy_def = row.get("strategy_definition", {}) or {}
            setup_type = str(
                row.get("setup_type")
                or (strategy_def.get("entry", {}) or {}).get("setup_type")
                or ""
            )
            family = self._setup_type_to_family(setup_type)
            if family is None:
                continue

            metrics = row.get("metrics", {}) or {}
            bt_metrics = strategy_def.get("backtest_results", {}) or {}
            win_rate = self._normalize_win_rate(
                metrics.get("win_rate", bt_metrics.get("win_rate", 0.0))
            )
            if win_rate > 0:
                family_scores[family].append(win_rate)

        resolved = {fam: max(values) for fam, values in family_scores.items() if values}
        if resolved:
            self._logger.info(
                "[SignalPipeline] alpha quality gate active (min_wr=%.0f%%): %s",
                self._alpha_min_win_rate * 100.0,
                {fam.value: round(wr * 100.0, 1) for fam, wr in resolved.items()},
            )
        return resolved

    def _alpha_family_allowed(self, family: SignalFamily) -> bool:
        observed_wr = self._alpha_family_win_rate.get(family)
        if observed_wr is None:
            return True
        return observed_wr >= self._alpha_min_win_rate

    def build_context(
        self,
        tickers: Optional[Iterable[str]] = None,
        price_data: Optional[pd.DataFrame] = None,
        sr_data: Optional[pd.DataFrame] = None,
        feature_data: Optional[pd.DataFrame] = None,
        exclude_tickers: Optional[Iterable[str]] = None,
        regime: Optional[RegimeLabel] = None,
        nbbo_data: Optional[Dict[str, Tuple[float, float]]] = None,
        **kwargs,
    ) -> SignalContext:
        ticker_list = [str(t).upper() for t in (tickers or []) if str(t).strip()]
        return SignalContext(
            tickers=ticker_list,
            price_data=price_data if price_data is not None else pd.DataFrame(),
            sr_data=sr_data,
            feature_data=feature_data,
            exclude_tickers=list(exclude_tickers) if exclude_tickers else None,
            regime=regime,
            nbbo_data=nbbo_data or {},
            **kwargs,
        )

    # ---------- Stage 1 ----------
    def stage_feature_intake(self, context: SignalContext) -> SignalContext:
        """Normalize incoming tabular features into expected OHLCV shape."""
        if isinstance(context.price_data, pd.DataFrame):
            price_data = context.price_data.copy()
            rename_map = {
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
            price_data = price_data.rename(columns=rename_map)
            context.price_data = price_data
        return context

    # ---------- Stage 2 ----------
    def stage_alpha_generation(self, context: SignalContext) -> List[SignalDecision]:
        signals: List[SignalDecision] = []
        quality_pruned = 0
        alpha_models = []
        for model in self._registry.get_enabled_models():
            if model.family in (SignalFamily.BASELINE_A, SignalFamily.BASELINE_B):
                continue
            if not self._alpha_family_allowed(model.family):
                quality_pruned += 1
                continue
            issues = model.validate_input(context)
            if issues:
                continue
            alpha_models.append(model)

        max_workers = int(
            max(
                1,
                getattr(
                    self._cfg,
                    "alpha_generation_max_workers",
                    os.cpu_count() or 4,
                )
                or 1,
            )
        )
        if max_workers <= 1 or len(alpha_models) <= 1:
            for model in alpha_models:
                try:
                    signals.extend(model.generate(context))
                except Exception as e:
                    self._logger.warning(
                        "[SignalPipeline] alpha model failed %s: %s", model.name, e
                    )
        else:
            model_results: List[Tuple[int, List[SignalDecision]]] = []
            worker_count = min(max_workers, len(alpha_models))
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="alpha-zoo",
            ) as executor:
                future_to_model = {
                    executor.submit(model.generate, context): (idx, model)
                    for idx, model in enumerate(alpha_models)
                }
                for future in as_completed(future_to_model):
                    idx, model = future_to_model[future]
                    try:
                        model_results.append((idx, list(future.result() or [])))
                    except Exception as e:
                        self._logger.warning(
                            "[SignalPipeline] alpha model failed %s: %s",
                            model.name,
                            e,
                        )
            for _, generated in sorted(model_results, key=lambda item: item[0]):
                signals.extend(generated)
        if quality_pruned > 0:
            self._logger.info(
                "[SignalPipeline] pruned %d alpha model(s) below %.0f%% win-rate floor",
                quality_pruned,
                self._alpha_min_win_rate * 100.0,
            )
        return signals

    # ---------- Stage 3 ----------
    def stage_baseline_generation(self, context: SignalContext) -> List[SignalDecision]:
        signals: List[SignalDecision] = []
        for model in self._registry.get_enabled_models():
            if model.family not in (SignalFamily.BASELINE_A, SignalFamily.BASELINE_B):
                continue
            issues = model.validate_input(context)
            if issues:
                continue
            try:
                signals.extend(model.generate(context))
            except Exception as e:
                self._logger.warning(
                    "[SignalPipeline] baseline model failed %s: %s", model.name, e
                )
        return signals

    # ---------- Stage 4 ----------
    def stage_regime_routing(
        self, signals: List[SignalDecision], context: SignalContext
    ) -> List[SignalDecision]:
        if not bool(getattr(self._cfg, "enable_regime_router", True)):
            for s in signals:
                if not s.regime:
                    s.regime = context.regime or RegimeLabel.NEUTRAL
            return signals

        regime = context.regime
        if not regime or regime == RegimeLabel.NEUTRAL:
            regime_result = self._regime_router.get_current_regime(
                price_data=context.price_data,
                feature_data=context.feature_data,
            )
            regime = regime_result.regime
            context.regime = regime
            context.regime_confidence = regime_result.confidence

        try:
            accepted, filtered = self._regime_router.filter_signals_by_regime(
                signals, regime, context.price_data
            )
        except Exception as e:
            self._logger.warning(
                "[SignalPipeline] Regime routing failed; preserving signals (including baselines): %s",
                e,
            )
            for s in signals:
                if not s.regime:
                    s.regime = regime or RegimeLabel.NEUTRAL
            return signals

        if filtered:
            self._logger.info(
                "[SignalPipeline] Regime routing: %d accepted, %d filtered for regime %s",
                len(accepted),
                len(filtered),
                regime.value,
            )

        return accepted

    # ---------- Stage 5 ----------
    def stage_calibrate_and_blend(
        self,
        alpha_signals: List[SignalDecision],
        baseline_signals: List[SignalDecision],
        legacy_signals: Optional[List[SignalDecision]] = None,
    ) -> List[SignalDecision]:
        legacy_signals = legacy_signals or []
        by_ticker: Dict[str, List[SignalDecision]] = defaultdict(list)
        for s in baseline_signals + legacy_signals:
            by_ticker[s.ticker].append(s)
        alpha_by_ticker: Dict[str, List[SignalDecision]] = defaultdict(list)
        for s in alpha_signals:
            alpha_by_ticker[s.ticker].append(s)

        blended: List[SignalDecision] = []
        base_weight = 0.8
        alpha_weight = 0.2
        if (
            str(getattr(self._cfg, "combine_method", "weighted_sum")).strip().lower()
            != "weighted_sum"
        ):
            base_weight = 1.0
            alpha_weight = 0.0

        tickers = sorted(set(list(by_ticker.keys()) + list(alpha_by_ticker.keys())))
        for ticker in tickers:
            base_group = by_ticker.get(ticker, [])
            alpha_group = alpha_by_ticker.get(ticker, [])

            if base_group and alpha_group:
                # Combined case: use weighted blend
                anchor = max(base_group, key=lambda s: float(s.score or 0.0))
                base_score = float(np.mean([float(s.score or 0.0) for s in base_group]))
                alpha_score = float(
                    np.mean(
                        [float((s.signal_strength + 1.0) * 50.0) for s in alpha_group]
                    )
                )
                blended_score = float(
                    np.clip(
                        base_score * base_weight + alpha_score * alpha_weight,
                        0.0,
                        100.0,
                    )
                )
            elif base_group:
                # Baseline only: use baseline average
                anchor = max(base_group, key=lambda s: float(s.score or 0.0))
                blended_score = float(
                    np.mean([float(s.score or 0.0) for s in base_group])
                )
                alpha_score = 50.0  # Neutral
            elif alpha_group:
                # Alpha only: use pure alpha average (don't wash out with 50.0 baseline)
                anchor = max(alpha_group, key=lambda s: float(s.score or 0.0))
                alpha_score = float(
                    np.mean(
                        [float((s.signal_strength + 1.0) * 50.0) for s in alpha_group]
                    )
                )
                blended_score = alpha_score
                base_score = 50.0  # Neutral
            else:
                continue

            blended_strength = float(np.clip((blended_score - 50.0) / 50.0, -1.0, 1.0))

            anchor.score = blended_score
            anchor.signal_strength = blended_strength
            anchor.confidence = float(np.clip(abs(blended_strength), 0.0, 1.0))
            anchor.factor_scores["base_score"] = (
                base_score if "base_score" in locals() else 50.0
            )
            anchor.factor_scores["alpha_score"] = alpha_score
            anchor.factor_scores["composite_score"] = blended_score
            blended.append(anchor)

        return blended

    # ---------- Stage 6 ----------
    def stage_action_normalization(
        self, signals: List[SignalDecision]
    ) -> List[SignalDecision]:
        for s in signals:
            action_value = (
                s.action.value if isinstance(s.action, SignalAction) else str(s.action)
            )
            action = SignalAction.from_string(action_value)
            # Keep compatibility with long-only downstream consumers.
            if action in (SignalAction.SELL, SignalAction.EXIT, SignalAction.AVOID):
                action = SignalAction.WATCH
            elif action in (
                SignalAction.BUY,
                SignalAction.BUY_OPEN,
                SignalAction.BUY_DIP,
            ):
                action = SignalAction.BUY_OPEN
            s.action = action
        return signals

    # ---------- Stage 7 ----------
    def stage_schema_enforcement(
        self, signals: List[SignalDecision]
    ) -> List[SignalDecision]:
        out: List[SignalDecision] = []
        fail_fast = bool(getattr(self._cfg, "fail_fast_on_schema_violation", True))
        for s in signals:
            issues = validate_signal_dict(s.to_dict())
            if not issues:
                out.append(s)
                continue
            if fail_fast:
                raise ValueError(f"Signal schema violation for {s.ticker}: {issues}")
            self._logger.warning(
                "[SignalPipeline] schema violation %s: %s", s.ticker, issues
            )
        return out

    def _to_legacy_candidates(self, signals: List[SignalDecision]) -> List[Dict]:
        min_score = float(getattr(self._cfg, "min_composite_score", 35.0))
        max_signals = int(getattr(self._cfg, "max_signals_per_batch", 200))
        ranked = sorted(
            [s for s in signals if float(s.score or 0.0) >= min_score],
            key=lambda s: (float(s.score or 0.0), s.ticker),
            reverse=True,
        )[:max_signals]

        out = []
        for s in ranked:
            out.append(
                {
                    "ticker": s.ticker,
                    "action": s.action.value,
                    "price": float(s.entry_price or 0.0),
                    "entry_price": float(s.entry_price or 0.0),
                    "arrival_price": float(s.arrival_price or 0.0),
                    "nbbo": s.nbbo,
                    "stop_price": float(s.stop_price or 0.0),
                    "target_price": float(s.target_price or 0.0),
                    "partial_target_price": float(s.partial_target_price or 0.0),
                    "score": float(s.score or 0.0),
                    "normalized_score": float(s.score or 0.0),
                    "signal_strength": float(s.signal_strength or 0.0),
                    "confidence": float(s.confidence or 0.0),
                    "family": s.family.value,
                    "source": s.source,
                    "atr_14": float(s.atr_14 or 0.0),
                    "risk_reward": float(s.risk_reward or 0.0),
                    "reason": s.reason,
                    "factor_scores": dict(s.factor_scores or {}),
                    "validation": {
                        "schema_ok": True,
                        "generation_time_ms": float(
                            s.diagnostics.generation_time_ms or 0.0
                        ),
                    },
                }
            )
        return out

    def run(
        self,
        context: SignalContext,
        legacy_candidates: Optional[List[Dict]] = None,
        include_baselines: bool = True,
    ) -> PipelineResult:
        context = self.stage_feature_intake(context)
        alpha_signals = self.stage_alpha_generation(context)
        baseline_signals = (
            self.stage_baseline_generation(context) if include_baselines else []
        )
        legacy_signals = []
        for row in legacy_candidates or []:
            if not isinstance(row, dict):
                continue
            family = self._resolve_legacy_candidate_family(row)
            if family is None:
                family = SignalFamily.BASELINE_A
            legacy_signals.append(
                SignalDecision.from_candidate_dict(row, family=family)
            )
        blended = self.stage_calibrate_and_blend(
            alpha_signals=alpha_signals,
            baseline_signals=baseline_signals,
            legacy_signals=legacy_signals,
        )
        routed = self.stage_regime_routing(blended, context)
        normalized = self.stage_action_normalization(routed)
        enforced = self.stage_schema_enforcement(normalized)

        # Capture arrival price and NBBO for each signal
        for s in enforced:
            s.arrival_price = self._capture_arrival_price(s.ticker, context)
            s.nbbo = context.nbbo_data.get(s.ticker)

        batch = SignalBatch(
            signals=enforced,
            generated_at=datetime.now(),
            total_generated=len(enforced),
            regime=context.regime or RegimeLabel.NEUTRAL,
        )
        legacy_out = self._to_legacy_candidates(enforced)
        return PipelineResult(batch=batch, legacy_candidates=legacy_out)

    def _capture_arrival_price(self, ticker: str, context: SignalContext) -> float:
        """Capture the arrival price benchmark (Bar-Close Benchmark)."""
        if not isinstance(context.price_data, pd.DataFrame) or context.price_data.empty:
            return 0.0

        # Handle multi-index (symbol, timestamp) vs single index
        if isinstance(context.price_data.index, pd.MultiIndex):
            try:
                ticker_data = context.price_data.xs(ticker, level=0)
            except (KeyError, ValueError):
                return 0.0
        else:
            ticker_data = context.price_data

        if ticker_data.empty or "close" not in ticker_data.columns:
            return 0.0

        return float(ticker_data["close"].iloc[-1])
