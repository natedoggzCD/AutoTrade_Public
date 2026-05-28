"""Strategy lab orchestrator (alpha-catalog edition).

Replaces the grid-sweep factory. For each (symbol, alpha) pair we evaluate
on purged walk-forward folds with a realized-fills cost model, apply
hierarchical shrinkage across symbols within a cluster, run BH-FDR per
family on deflated-Sharpe p-values, and promote a per-symbol winner to the
challenger JSON (separate file from the live champion).

Design notes:
- Workers receive picklable args only (AlphaContext is optional but should
  be picklable; the alpha generate callables are module-level functions).
- Cluster key uses sector (best-effort lookup) + ATR percentile band + cap
  bucket. Sector is not currently in key_stats; UNK is acceptable here
  because shrinkage degrades gracefully with sparse clusters.
- Buy-and-hold check is per symbol, not portfolio, matching the per-symbol
  promotion gate the spec requires.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import erf, sqrt
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from autotrade.backtesting.alpha_catalog import (
    AlphaContext,
    AlphaDefinition,
    get_alpha,
    iter_alphas,
)
from autotrade.backtesting.alpha_evaluator import (
    SymbolAlphaEvaluation,
    evaluate_alpha_on_symbol,
)
from autotrade.backtesting.cost_model import CostModel, build_cost_model
from autotrade.backtesting.shared_cache import (
    rehydrate_cache,
    release_handles,
    share_cache,
)
from autotrade.backtesting.hierarchical_shrinkage import (
    DEFAULT_K,
    ShrinkageResult,
    SymbolAlphaEstimate,
    assign_cluster_key,
    shrink_estimates,
)
from autotrade.backtesting.purged_walk_forward import PurgedKFoldConfig
from autotrade.backtesting.statistical_controls import benjamini_hochberg_correction

logger = logging.getLogger(__name__)

CHALLENGER_PATH = Path(
    "data/strategy_lab/validated_strategies_by_symbol_challenger.json"
)
LAB_LOG_DIR = Path("logs")
BH_ALPHA = 0.10
MIN_PROMOTION_TRADES = 20


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _deflated_sharpe_pvalue(dsr: float) -> float:
    """One-sided upper-tail p-value treating DSR as a z-statistic."""
    if not np.isfinite(dsr):
        return 1.0
    return float(max(0.0, min(1.0, 1.0 - _normal_cdf(float(dsr)))))


def _atr_pct_from_bars(bars: pd.DataFrame, window: int = 14) -> Optional[float]:
    if bars is None or bars.empty or len(bars) < window + 1:
        return None
    df = bars.sort_values("date").tail(window * 4).copy()
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window).mean().iloc[-1]
    last_close = close.iloc[-1]
    if last_close and last_close > 0 and pd.notna(atr):
        return float(atr / last_close * 100.0)
    return None


def _cap_bucket(market_cap: Optional[float]) -> str:
    if market_cap is None or not np.isfinite(market_cap) or market_cap <= 0:
        return "UNK"
    if market_cap < 300e6:
        return "MICRO"
    if market_cap < 2e9:
        return "SMALL"
    if market_cap < 10e9:
        return "MID"
    if market_cap < 200e9:
        return "LARGE"
    return "MEGA"


def _lookup_symbol_context(symbol: str) -> Dict[str, Any]:
    """Best-effort sector + market_cap lookup from data/financial.db."""
    try:
        from autotrade.utils.financial_db import FinancialDB

        db = FinancialDB()
        stats = db.get_key_stats(symbol) or {}
        return {
            "market_cap": stats.get("market_cap"),
            "sector": None,  # not stored in key_stats; placeholder
        }
    except Exception:
        return {"market_cap": None, "sector": None}


def _data_requirements_satisfied(
    alpha: AlphaDefinition, ctx: Optional[AlphaContext]
) -> bool:
    reqs = set(alpha.data_requirements or ())
    if not reqs:
        return True
    if "daily_only" in reqs:
        return True
    if ctx is None:
        return False
    if "needs_universe" in reqs and ctx.universe_bars is None:
        return False
    if "needs_spy" in reqs and ctx.spy_bars is None:
        return False
    if "needs_earnings" in reqs and ctx.earnings_calendar is None:
        return False
    if "needs_sector_map" in reqs and not ctx.sector_map:
        return False
    if "needs_analyst_ratings" in reqs:
        # not gated by ctx; alpha will fall back if data is absent
        return True
    if "needs_financial_db" in reqs:
        return True
    return True


def _empty_gate_counts() -> Dict[str, int]:
    return {
        "evaluated": 0,
        "no_trades": 0,
        "leakage_failed": 0,
        "beats_bnh_failed": 0,
        "fdr_failed": 0,
        "min_trades_failed": 0,
        "missing_shrinkage": 0,
        "eligible": 0,
        "promoted": 0,
    }


def _build_gate_diagnostics(
    evaluations: Dict[Tuple[str, str], SymbolAlphaEvaluation],
    alpha_lookup: Dict[str, AlphaDefinition],
    alpha_ids: Sequence[str],
    symbols: Sequence[str],
    significant_pairs: set[Tuple[str, str]],
    shrunk_lookup: Dict[Tuple[str, str], ShrinkageResult],
    promotions: Dict[str, "SymbolPromotion"],
) -> Dict[str, Any]:
    overall = Counter()
    by_family: Dict[str, Counter] = defaultdict(Counter)
    top_rejections: List[Dict[str, Any]] = []

    for sym in symbols:
        for aid in alpha_ids:
            ev = evaluations.get((sym, aid))
            alpha = alpha_lookup.get(aid)
            family = alpha.family if alpha else "unknown"
            overall["evaluated"] += 1
            by_family[family]["evaluated"] += 1

            if ev is None or ev.n_trades <= 0:
                reason = "no_trades"
            elif not ev.leakage_passed:
                reason = "leakage_failed"
            elif not ev.beats_bnh:
                reason = "beats_bnh_failed"
            elif ev.n_trades < MIN_PROMOTION_TRADES:
                reason = "min_trades_failed"
            elif (sym, aid) not in significant_pairs:
                reason = "fdr_failed"
            elif (sym, aid) not in shrunk_lookup:
                reason = "missing_shrinkage"
            else:
                reason = "eligible"

            overall[reason] += 1
            by_family[family][reason] += 1

            if reason not in {"eligible", "no_trades"} and ev is not None:
                top_rejections.append(
                    {
                        "symbol": sym,
                        "alpha_id": aid,
                        "family": family,
                        "reason": reason,
                        "n_trades": int(ev.n_trades),
                        "raw_pf": float(ev.raw_pf),
                        "raw_sharpe": float(ev.raw_sharpe),
                        "deflated_sharpe": float(ev.deflated_sharpe),
                        "beats_bnh": bool(ev.beats_bnh),
                    }
                )

    for promo in promotions.values():
        overall["promoted"] += 1
        by_family[promo.family]["promoted"] += 1

    keys = _empty_gate_counts()
    return {
        "overall": {key: int(overall.get(key, 0)) for key in keys},
        "by_family": {
            family: {key: int(counts.get(key, 0)) for key in keys}
            for family, counts in sorted(by_family.items())
        },
        "top_rejections": sorted(
            top_rejections,
            key=lambda row: (
                row["deflated_sharpe"],
                row["raw_pf"],
                row["n_trades"],
            ),
            reverse=True,
        )[:20],
    }


@dataclass
class SymbolPromotion:
    symbol: str
    alpha_id: str
    posterior_pf: float
    posterior_se: float
    n_trades: int
    raw_sharpe: float
    deflated_sharpe: float
    buy_and_hold_return: float
    family: str
    cluster_key: str


@dataclass
class LabRunReport:
    started_at: str
    finished_at: str
    runtime_sec: float
    n_symbols: int
    n_alphas: int
    n_evaluations: int
    n_promotions: int
    promotions_per_family: Dict[str, int] = field(default_factory=dict)
    top_promotions: List[Dict[str, Any]] = field(default_factory=list)
    gate_diagnostics: Dict[str, Any] = field(default_factory=dict)
    output_path: str = ""
    log_path: str = ""

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "runtime_sec": self.runtime_sec,
            "n_symbols": self.n_symbols,
            "n_alphas": self.n_alphas,
            "n_evaluations": self.n_evaluations,
            "n_promotions": self.n_promotions,
            "promotions_per_family": dict(self.promotions_per_family),
            "top_promotions": list(self.top_promotions),
            "gate_diagnostics": dict(self.gate_diagnostics),
            "output_path": self.output_path,
            "log_path": self.log_path,
        }


def _evaluate_worker(
    alpha_id: str,
    symbol: str,
    bars: pd.DataFrame,
    cost_model: CostModel,
    wf_config: PurgedKFoldConfig,
    ctx: Optional[AlphaContext],
    n_trials: int,
) -> Optional[SymbolAlphaEvaluation]:
    try:
        alpha = get_alpha(alpha_id)
        return evaluate_alpha_on_symbol(
            alpha=alpha,
            symbol=symbol,
            bars=bars,
            cost_model=cost_model,
            wf_config=wf_config,
            ctx=ctx,
            n_trials=n_trials,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "evaluate_alpha_on_symbol failed for %s/%s: %s", symbol, alpha_id, exc
        )
        return None


def _evaluate_symbol_batch(
    symbol: str,
    bars: pd.DataFrame,
) -> Dict[str, SymbolAlphaEvaluation]:
    """Evaluate every alpha against a single symbol within one worker.

    Per-run constants (alpha list, ctx, cost model, wf config, n_trials)
    are read from worker-process globals populated by `_worker_init` via
    the ProcessPoolExecutor `initargs`. This keeps the per-job payload
    down to just `(symbol, bars)` -- the alternative pickles the
    multi-GB AlphaContext.cache on every single submit.
    """
    ctx = _WORKER_CTX
    cost_model = _WORKER_COST_MODEL
    wf_config = _WORKER_WF_CONFIG
    alpha_ids = _WORKER_ALPHA_IDS or []
    n_trials = _WORKER_N_TRIALS

    if "symbol" not in bars.columns:
        bars = bars.copy()
        bars["symbol"] = symbol
    results: Dict[str, SymbolAlphaEvaluation] = {}
    for alpha_id in alpha_ids:
        try:
            alpha = get_alpha(alpha_id)
            res = evaluate_alpha_on_symbol(
                alpha=alpha,
                symbol=symbol,
                bars=bars,
                cost_model=cost_model,
                wf_config=wf_config,
                ctx=ctx,
                n_trials=n_trials,
            )
            if res is not None:
                results[alpha_id] = res
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "evaluate_alpha_on_symbol failed for %s/%s: %s", symbol, alpha_id, exc
            )
    return results


def _build_strategy_row(
    promotion: SymbolPromotion,
    evaluation: SymbolAlphaEvaluation,
    alpha: AlphaDefinition,
    generated_at_iso: str,
) -> dict:
    metrics = {
        "profit_factor": float(evaluation.raw_pf),
        "win_rate": float(
            sum(1 for f in evaluation.per_fold for r in f.test_returns if r > 0)
            / max(1, sum(len(f.test_returns) for f in evaluation.per_fold))
        )
        if any(f.test_returns for f in evaluation.per_fold)
        else 0.0,
        "sharpe_ratio": float(evaluation.raw_sharpe),
        "deflated_sharpe": float(evaluation.deflated_sharpe),
        "total_trades": int(evaluation.n_trades),
        "posterior_pf": float(promotion.posterior_pf),
        "posterior_se": float(promotion.posterior_se),
        "buy_and_hold_return": float(evaluation.buy_and_hold_return),
        "beats_bnh": bool(evaluation.beats_bnh),
        "n_folds": int(len(evaluation.per_fold)),
    }
    return {
        "strategy_name": f"alpha_catalog__{alpha.alpha_id}",
        "setup_type": alpha.alpha_id,
        "config_patch": {},
        "backtest_profit_factor": float(evaluation.raw_pf),
        "backtest_win_rate": float(metrics["win_rate"]),
        "final_score": float(promotion.posterior_pf - promotion.posterior_se),
        "metrics": metrics,
        "promoted_at": generated_at_iso,
        "promotion_basis": "posterior_pf_minus_se_with_bh_fdr",
        "alpha_metadata": {
            "alpha_id": alpha.alpha_id,
            "family": alpha.family,
            "hypothesis": alpha.hypothesis,
            "variant": alpha.variant,
            "params": dict(alpha.params),
            "regime_compatibility": list(alpha.regime_compatibility),
            "data_requirements": list(alpha.data_requirements),
            "cluster_key": promotion.cluster_key,
        },
    }


def run_lab(
    symbols: Sequence[str],
    bars_loader: Callable[[str], pd.DataFrame],
    output_path: Path = CHALLENGER_PATH,
    ctx_factory: Optional[Callable[[], AlphaContext]] = None,
    wf_config: Optional[PurgedKFoldConfig] = None,
    parallel_workers: int = 8,
    max_runtime_sec: Optional[float] = None,
) -> LabRunReport:
    """Run the lab end-to-end and write a challenger JSON file."""

    started_at = datetime.now(timezone.utc)
    wf = wf_config or PurgedKFoldConfig()
    cost_model = build_cost_model()
    ctx = ctx_factory() if ctx_factory else None

    # Move the precomputed CS cache into shared memory before fanning out to
    # workers. Without this, every worker on Windows-spawn would unpickle its
    # own private multi-GB copy and the run would OOM at scale.
    _shm_handles: List[Any] = []
    if ctx is not None and ctx.cache:
        cache_count = len(ctx.cache)
        t_share = time.monotonic()
        shared, _shm_handles = share_cache(ctx.cache)
        ctx.cache = shared
        elapsed_share = time.monotonic() - t_share
        total_mb = sum(
            getattr(h, "size", 0) for h in _shm_handles
        ) / (1024 * 1024)
        logger.info(
            "shared %d cache frames (%.1f MB total) in %.1fs",
            cache_count,
            total_mb,
            elapsed_share,
        )
        print(
            f"  shared_memory: hosted {cache_count} cache frames "
            f"({total_mb:.0f} MB total) in {elapsed_share:.1f}s",
            flush=True,
        )

    alphas: List[AlphaDefinition] = [
        a for a in iter_alphas() if _data_requirements_satisfied(a, ctx)
    ]
    alpha_ids = [a.alpha_id for a in alphas]
    alpha_lookup = {a.alpha_id: a for a in alphas}

    # Strip heavy fields from ctx before workers pickle it. The CS cache (now
    # in shared memory) holds everything cross-sectional alphas actually use;
    # the long-form universe_bars panel was only a fallback. Without this, 20
    # workers each unpickle their own ~800 MB copy and the run OOMs.
    if ctx is not None:
        ctx.universe_bars = None

    symbols = [s.upper() for s in symbols]
    n_trials = max(1, len(alphas) * max(1, len(symbols)))

    bars_cache: Dict[str, pd.DataFrame] = {}
    symbol_contexts: Dict[str, Dict[str, Any]] = {}
    for sym in symbols:
        try:
            bars = bars_loader(sym)
        except Exception as exc:
            logger.warning("bars_loader failed for %s: %s", sym, exc)
            continue
        if bars is None or len(bars) < 60:
            logger.info("skip %s: insufficient bars", sym)
            continue
        bars_cache[sym] = bars.sort_values("date").reset_index(drop=True).copy()
        info = _lookup_symbol_context(sym)
        info["atr_pct"] = _atr_pct_from_bars(bars_cache[sym])
        info["cap_bucket"] = _cap_bucket(info.get("market_cap"))
        info["cluster_key"] = assign_cluster_key(
            sym,
            info.get("sector"),
            info.get("atr_pct"),
            info.get("cap_bucket"),
        )
        symbol_contexts[sym] = info

    evaluations: Dict[Tuple[str, str], SymbolAlphaEvaluation] = {}
    deadline = (time.monotonic() + max_runtime_sec) if max_runtime_sec else None
    n_evals_requested = 0

    workers = max(1, int(parallel_workers))
    if workers == 1:
        for sym in bars_cache:
            for alpha in alphas:
                if deadline and time.monotonic() > deadline:
                    break
                result = _evaluate_worker(
                    alpha.alpha_id, sym, bars_cache[sym], cost_model, wf, ctx, n_trials
                )
                n_evals_requested += 1
                if result is not None:
                    evaluations[(sym, alpha.alpha_id)] = result
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(ctx, cost_model, wf, list(alpha_ids), int(n_trials)),
        ) as pool:
            futures = {}
            for sym in bars_cache:
                if deadline and time.monotonic() > deadline:
                    break
                fut = pool.submit(_evaluate_symbol_batch, sym, bars_cache[sym])
                futures[fut] = sym
                n_evals_requested += len(alpha_ids)
            n_jobs = len(futures)
            print(
                f"  submitted {n_jobs} per-symbol jobs covering {n_evals_requested} (sym, alpha) evals to {workers} workers",
                flush=True,
            )
            t_start = time.monotonic()
            last_report = t_start
            report_every_jobs = max(10, n_jobs // 100)
            n_jobs_done = 0
            n_evals_done = 0
            for fut in as_completed(futures):
                sym = futures[fut]
                if deadline and time.monotonic() > deadline:
                    fut.cancel()
                    continue
                try:
                    sym_results = fut.result()
                except Exception as exc:
                    logger.warning("worker failed %s: %s", sym, exc)
                    sym_results = {}
                for aid, result in sym_results.items():
                    if result is not None:
                        evaluations[(sym, aid)] = result
                n_jobs_done += 1
                n_evals_done += len(alpha_ids)  # counted whether the alpha returned a result or not
                now = time.monotonic()
                if n_jobs_done % report_every_jobs == 0 or (now - last_report) >= 30.0:
                    elapsed = now - t_start
                    job_rate = n_jobs_done / max(0.001, elapsed)
                    eval_rate = n_evals_done / max(0.001, elapsed)
                    pct = 100.0 * n_jobs_done / max(1, n_jobs)
                    eta_min = ((n_jobs - n_jobs_done) / max(0.001, job_rate)) / 60.0
                    print(
                        f"  [progress] {n_jobs_done}/{n_jobs} symbols ({pct:.1f}%) "
                        f"job_rate={job_rate:.1f}/s eval_rate={eval_rate:.1f}/s "
                        f"elapsed={elapsed/60:.1f}min ETA={eta_min:.1f}min",
                        flush=True,
                    )
                    last_report = now

    # --- Hierarchical shrinkage per alpha across symbols ---
    estimates: List[SymbolAlphaEstimate] = []
    for (sym, aid), ev in evaluations.items():
        if ev.n_trades <= 0:
            continue
        cluster_key = symbol_contexts.get(sym, {}).get("cluster_key", "UNK|atrUNK|UNK")
        estimates.append(
            SymbolAlphaEstimate(
                symbol=sym,
                alpha_id=aid,
                cluster_key=cluster_key,
                n_trades=ev.n_trades,
                raw_mean_return=ev.raw_mean_return,
                raw_std_return=ev.raw_std_return,
                raw_pf=ev.raw_pf,
            )
        )
    shrunk = shrink_estimates(estimates, k=DEFAULT_K)
    shrunk_lookup: Dict[Tuple[str, str], ShrinkageResult] = {
        (s.symbol, s.alpha_id): s for s in shrunk
    }

    # --- BH-FDR per family on deflated Sharpe p-values ---
    family_rows: Dict[
        str, List[Tuple[str, str, float]]
    ] = {}  # family -> [(sym, aid, p)]
    for (sym, aid), ev in evaluations.items():
        alpha = alpha_lookup.get(aid)
        if not alpha or ev.n_trades <= 0:
            continue
        p = _deflated_sharpe_pvalue(ev.deflated_sharpe)
        family_rows.setdefault(alpha.family, []).append((sym, aid, p))

    significant_pairs: set[Tuple[str, str]] = set()
    for family, rows in family_rows.items():
        pvals = [r[2] for r in rows]
        result = benjamini_hochberg_correction(pvals, alpha=BH_ALPHA)
        for idx in result.significant_indices:
            sym, aid, _ = rows[idx]
            significant_pairs.add((sym, aid))

    # --- Per-symbol promotion selection ---
    promotions: Dict[str, SymbolPromotion] = {}
    for sym in bars_cache:
        best: Optional[SymbolPromotion] = None
        for aid in alpha_ids:
            ev = evaluations.get((sym, aid))
            if ev is None or ev.n_trades <= 0:
                continue
            if not ev.beats_bnh:
                continue
            if ev.n_trades < MIN_PROMOTION_TRADES:
                continue
            if (sym, aid) not in significant_pairs:
                continue
            shrunk_row = shrunk_lookup.get((sym, aid))
            if shrunk_row is None:
                continue
            score = shrunk_row.posterior_pf - shrunk_row.posterior_se
            alpha = alpha_lookup[aid]
            candidate = SymbolPromotion(
                symbol=sym,
                alpha_id=aid,
                posterior_pf=float(shrunk_row.posterior_pf),
                posterior_se=float(shrunk_row.posterior_se),
                n_trades=ev.n_trades,
                raw_sharpe=float(ev.raw_sharpe),
                deflated_sharpe=float(ev.deflated_sharpe),
                buy_and_hold_return=float(ev.buy_and_hold_return),
                family=alpha.family,
                cluster_key=symbol_contexts.get(sym, {}).get("cluster_key", ""),
            )
            if best is None or score > (best.posterior_pf - best.posterior_se):
                best = candidate
        if best is not None:
            promotions[sym] = best

    generated_at_iso = started_at.isoformat()
    symbols_out: Dict[str, List[dict]] = {}
    promotions_per_family: Dict[str, int] = {}
    for sym, promo in promotions.items():
        alpha = alpha_lookup[promo.alpha_id]
        ev = evaluations[(sym, promo.alpha_id)]
        symbols_out[sym] = [_build_strategy_row(promo, ev, alpha, generated_at_iso)]
        promotions_per_family[promo.family] = (
            promotions_per_family.get(promo.family, 0) + 1
        )

    payload = {
        "generated_at": generated_at_iso,
        "promotion_mode": "alpha_catalog_purged_wf_shrinkage",
        "source_run_id": f"lab_orchestrator_{started_at.strftime('%Y%m%d_%H%M%S')}",
        "symbols": symbols_out,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp_path, output_path)

    finished_at = datetime.now(timezone.utc)
    runtime = (finished_at - started_at).total_seconds()

    top_promotions = sorted(
        (
            {
                "symbol": p.symbol,
                "alpha_id": p.alpha_id,
                "posterior_pf": p.posterior_pf,
                "posterior_se": p.posterior_se,
                "n_trades": p.n_trades,
                "deflated_sharpe": p.deflated_sharpe,
            }
            for p in promotions.values()
        ),
        key=lambda r: r["posterior_pf"],
        reverse=True,
    )[:5]

    gate_diagnostics = _build_gate_diagnostics(
        evaluations=evaluations,
        alpha_lookup=alpha_lookup,
        alpha_ids=alpha_ids,
        symbols=list(bars_cache),
        significant_pairs=significant_pairs,
        shrunk_lookup=shrunk_lookup,
        promotions=promotions,
    )

    report = LabRunReport(
        started_at=generated_at_iso,
        finished_at=finished_at.isoformat(),
        runtime_sec=runtime,
        n_symbols=len(bars_cache),
        n_alphas=len(alphas),
        n_evaluations=len(evaluations),
        n_promotions=len(promotions),
        promotions_per_family=promotions_per_family,
        top_promotions=top_promotions,
        gate_diagnostics=gate_diagnostics,
        output_path=str(output_path),
    )

    LAB_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = (
        LAB_LOG_DIR / f"strategy_lab_run_{started_at.strftime('%Y%m%d_%H%M%S')}.json"
    )
    log_path.write_text(
        json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    report.log_path = str(log_path)

    logger.info(
        "lab_orchestrator run complete: %d symbols, %d alphas, %d evals, %d promotions (%.1fs)",
        report.n_symbols,
        report.n_alphas,
        report.n_evaluations,
        report.n_promotions,
        runtime,
    )
    # Release shared-memory blocks now that all workers have exited (the
    # ProcessPoolExecutor `with` block has already returned).
    if _shm_handles:
        release_handles(_shm_handles)
    return report


# Per-worker globals populated by `_worker_init`. Pickling these once at
# worker startup (vs. on every submit) is the difference between a 5-minute
# full-universe run and a 20-hour one: at full scale the AlphaContext.cache
# alone is several GB.
_WORKER_CTX: Optional[AlphaContext] = None
_WORKER_COST_MODEL: Optional[CostModel] = None
_WORKER_WF_CONFIG: Optional[PurgedKFoldConfig] = None
_WORKER_ALPHA_IDS: Optional[List[str]] = None
_WORKER_N_TRIALS: int = 1


def _worker_init(
    ctx: Optional[AlphaContext] = None,
    cost_model: Optional[CostModel] = None,
    wf_config: Optional[PurgedKFoldConfig] = None,
    alpha_ids: Optional[List[str]] = None,
    n_trials: int = 1,
) -> None:
    """ProcessPool initializer.

    Silences noisy third-party warnings AND stashes the per-run constants
    (ctx, cost_model, wf_config, alpha_ids, n_trials) into module-level
    globals so per-job payloads stay tiny.

    The ``ctx.cache`` arrives populated with ``SharedFrameRef`` handles
    instead of full DataFrames; we rehydrate them here into zero-copy
    DataFrame views over the parent-owned shared-memory blocks. This keeps
    total RAM usage at ~one copy of the cache regardless of worker count.
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    if ctx is not None and ctx.cache:
        ctx.cache = rehydrate_cache(ctx.cache)
    global _WORKER_CTX, _WORKER_COST_MODEL, _WORKER_WF_CONFIG
    global _WORKER_ALPHA_IDS, _WORKER_N_TRIALS
    _WORKER_CTX = ctx
    _WORKER_COST_MODEL = cost_model
    _WORKER_WF_CONFIG = wf_config
    _WORKER_ALPHA_IDS = alpha_ids
    _WORKER_N_TRIALS = int(n_trials)


def bulk_bars_loader(
    parquet_path: Path, symbols: Sequence[str]
) -> Callable[[str], pd.DataFrame]:
    """Single-query bulk loader for the lab CLI.

    Runs one DuckDB query covering every requested symbol, groups by ticker,
    and returns a closure that resolves cached frames in O(1). Replaces the
    per-symbol `default_bars_loader` for production-scale runs where loading
    bars sequentially leaves CPU workers starved.
    """
    parquet_path = Path(parquet_path)
    syms = sorted({str(s).strip().upper() for s in symbols if str(s).strip()})
    if not syms:
        return lambda _s: pd.DataFrame()
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("duckdb required for bulk_bars_loader") from exc

    placeholders = ",".join("?" for _ in syms)
    query = (
        'SELECT UPPER(ticker) AS symbol, "Date" AS date, "Open" AS open, "High" AS high, '
        '"Low" AS low, "Close" AS close, "Volume" AS volume '
        f"FROM read_parquet('{parquet_path.as_posix()}') "
        f"WHERE UPPER(ticker) IN ({placeholders}) ORDER BY symbol, date"
    )
    t0 = time.monotonic()
    con = duckdb.connect()
    try:
        df = con.execute(query, syms).df()
    finally:
        con.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    cache: Dict[str, pd.DataFrame] = {
        sym: g.drop(columns=["symbol"]).reset_index(drop=True)
        for sym, g in df.groupby("symbol", sort=False)
    }
    elapsed = time.monotonic() - t0
    logger.info(
        "bulk_bars_loader: loaded %d/%d symbols in %.1fs",
        len(cache),
        len(syms),
        elapsed,
    )
    print(
        f"  bulk_bars_loader: loaded {len(cache)}/{len(syms)} symbols in {elapsed:.1f}s",
        flush=True,
    )

    def _load(symbol: str) -> pd.DataFrame:
        return cache.get(str(symbol).upper(), pd.DataFrame())

    return _load


def default_bars_loader(parquet_path: Path) -> Callable[[str], pd.DataFrame]:
    """DuckDB-backed loader from DownDay parquet."""
    parquet_path = Path(parquet_path)

    def _load(symbol: str) -> pd.DataFrame:
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("duckdb required for default_bars_loader") from exc
        sym = symbol.upper()
        # DownDay parquet uses title-cased OHLCV with 'Date'. Normalize here.
        query = (
            'SELECT UPPER(ticker) AS symbol, "Date" AS date, "Open" AS open, "High" AS high, '
            '"Low" AS low, "Close" AS close, "Volume" AS volume '
            f"FROM read_parquet('{parquet_path.as_posix()}') "
            "WHERE upper(ticker) = ? ORDER BY date"
        )
        con = duckdb.connect()
        try:
            df = con.execute(query, [sym]).df()
        finally:
            con.close()
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    return _load


def build_default_alpha_context(
    parquet_path: Path,
    symbols: Sequence[str],
    *,
    include_spy: bool = True,
) -> AlphaContext:
    """Build cross-sectional/SPY/earnings context for the default lab CLI.

    The context is scoped to the requested run symbols to keep ProcessPool
    payloads bounded. Production runs should pass a broad symbols file so
    cross-sectional ranks have enough peers to be meaningful.
    """
    parquet_path = Path(parquet_path)
    requested = {str(s).strip().upper() for s in symbols if str(s).strip()}
    if include_spy:
        requested.add("SPY")
    if not requested:
        return AlphaContext()

    import duckdb

    placeholders = ",".join("?" for _ in requested)
    query = (
        'SELECT UPPER(ticker) AS symbol, "Date" AS date, "Open" AS open, '
        '"High" AS high, "Low" AS low, "Close" AS close, "Volume" AS volume '
        f"FROM read_parquet('{parquet_path.as_posix()}') "
        f"WHERE UPPER(ticker) IN ({placeholders}) ORDER BY symbol, date"
    )
    con = duckdb.connect()
    try:
        universe = con.execute(query, sorted(requested)).df()
    finally:
        con.close()

    if not universe.empty:
        universe["date"] = pd.to_datetime(universe["date"]).dt.date

    spy_bars: Optional[pd.DataFrame] = None
    if not universe.empty:
        spy_rows = universe[universe["symbol"] == "SPY"].copy()
        if not spy_rows.empty:
            spy_bars = spy_rows.drop(columns=["symbol"]).reset_index(drop=True)
        else:
            spy_bars = _build_market_proxy_bars(universe)

    sector_map: Dict[str, str] = {}
    try:
        from autotrade.utils.security_metadata import (
            load_security_metadata,
            normalize_sector_label,
        )

        metadata = load_security_metadata()
        if not metadata.empty and {"ticker", "sector"}.issubset(metadata.columns):
            for row in metadata.itertuples(index=False):
                ticker = str(getattr(row, "ticker", "") or "").upper()
                sector = normalize_sector_label(getattr(row, "sector", None))
                if ticker and sector:
                    sector_map[ticker] = sector
    except Exception as exc:
        logger.info("security metadata unavailable for alpha context: %s", exc)

    for symbol in requested:
        sector_map.setdefault(symbol, "unknown")

    earnings_calendar = _load_earnings_calendar_for_context(requested)

    # --- Cross-sectional precomputation ---
    # Universe-wide rank tables are symbol-agnostic. Building them once here
    # (vs. inside every alpha generator on every (symbol, alpha) call) is the
    # difference between an O(N) full run and an O(N^2) one. The CS alpha
    # generators consult `ctx.cache` and only fall back to recomputing when
    # the cache is missing (e.g., contexts built outside this builder).
    cs_cache: Dict[str, Any] = {}
    if not universe.empty and {"symbol", "date", "close"}.issubset(universe.columns):
        uni = universe.copy()
        uni["date_key"] = pd.to_datetime(uni["date"]).dt.date
        uni["symbol"] = uni["symbol"].astype(str)
        close_matrix = uni.pivot_table(
            index="date_key",
            columns="symbol",
            values="close",
            aggfunc="last",
        ).sort_index().astype(float)
        cs_cache["close_matrix"] = close_matrix

        t_cs = time.monotonic()
        # Universe-wide return + rank by lookback (used by xs_momentum_universe_*).
        for lookback in (20, 60):
            ret = close_matrix.pct_change(lookback)
            cs_cache[f"universe_returns_{lookback}"] = ret
            cs_cache[f"universe_rank_{lookback}"] = ret.rank(axis=1, pct=True)

        # SPY-relative return + rank (used by relative_strength_leader_*).
        if spy_bars is not None and {"date", "close"}.issubset(spy_bars.columns):
            spy_keyed = spy_bars.copy()
            spy_keyed["date_key"] = pd.to_datetime(spy_keyed["date"]).dt.date
            spy_series = spy_keyed.set_index("date_key")["close"].astype(float)
            for lookback in (20,):
                spy_ret = spy_series.pct_change(lookback)
                rel = close_matrix.pct_change(lookback).sub(spy_ret, axis=0)
                cs_cache[f"rel_spy_returns_{lookback}"] = rel
                cs_cache[f"rel_spy_rank_{lookback}"] = rel.rank(axis=1, pct=True)

        # Volatility rank (used by low_vol_anomaly).
        daily_returns = close_matrix.pct_change()
        for vol_window in (60,):
            realized_vol = daily_returns.rolling(vol_window, min_periods=vol_window).std()
            cs_cache[f"vol_rank_{vol_window}"] = realized_vol.rank(axis=1, pct=True)
        for mom_window in (126,):
            cs_cache[f"universe_returns_{mom_window}"] = close_matrix.pct_change(
                mom_window
            )

        # Sector-restricted rank (used by xs_momentum_sector_rank_*).
        # Group symbols by their sector_map label, then pre-rank within each
        # sector slice. Sector keys are normalized lowercase strings; unknown
        # sectors are bucketed under "unknown" (same as live).
        by_sector: Dict[str, List[str]] = defaultdict(list)
        for sym in close_matrix.columns:
            by_sector[sector_map.get(sym, "unknown")].append(sym)
        for sector, sector_syms in by_sector.items():
            sec_close = close_matrix[sector_syms]
            for lookback in (20, 60):
                sec_ret = sec_close.pct_change(lookback)
                cs_cache[f"sector_returns_{sector}_{lookback}"] = sec_ret
                cs_cache[f"sector_rank_{sector}_{lookback}"] = sec_ret.rank(
                    axis=1, pct=True
                )

        elapsed_cs = time.monotonic() - t_cs
        logger.info(
            "build_default_alpha_context: precomputed %d CS matrices in %.1fs (universe=%dx%d, %d sectors)",
            len(cs_cache),
            elapsed_cs,
            close_matrix.shape[0],
            close_matrix.shape[1],
            len(by_sector),
        )
        print(
            f"  cs_cache: precomputed {len(cs_cache)} matrices in {elapsed_cs:.1f}s "
            f"(universe={close_matrix.shape[0]}x{close_matrix.shape[1]}, {len(by_sector)} sectors)",
            flush=True,
        )

    return AlphaContext(
        spy_bars=spy_bars,
        sector_map=sector_map,
        universe_bars=universe if not universe.empty else None,
        earnings_calendar=earnings_calendar,
        cache=cs_cache,
    )


def _build_market_proxy_bars(universe: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Fallback benchmark when ETF bars are absent from the small/mid-cap parquet."""
    if universe.empty or not {"symbol", "date", "close"}.issubset(universe.columns):
        return None
    close = universe.pivot_table(
        index="date",
        columns="symbol",
        values="close",
        aggfunc="last",
    ).sort_index()
    if close.empty:
        return None
    returns = close.pct_change().mean(axis=1, skipna=True).fillna(0.0)
    proxy_close = 100.0 * (1.0 + returns).cumprod()
    return pd.DataFrame(
        {
            "date": list(proxy_close.index),
            "open": proxy_close.to_numpy(),
            "high": proxy_close.to_numpy(),
            "low": proxy_close.to_numpy(),
            "close": proxy_close.to_numpy(),
            "volume": 0,
        }
    )


def _load_earnings_calendar_for_context(symbols: set[str]) -> Optional[pd.DataFrame]:
    db_path = Path("data/financial.db")
    if not symbols or not db_path.exists():
        return None
    try:
        import sqlite3

        placeholders = ",".join("?" for _ in symbols)
        with sqlite3.connect(db_path) as conn:
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='earnings_calendar'",
                conn,
            )
            if tables.empty:
                return None
            events = pd.read_sql_query(
                "SELECT * FROM earnings_calendar WHERE UPPER(ticker) IN "
                f"({placeholders})",
                conn,
                params=sorted(symbols),
            )
    except Exception as exc:
        logger.info("earnings calendar unavailable for alpha context: %s", exc)
        return None
    if events.empty:
        return None
    events = events.rename(columns={"ticker": "symbol", "earnings_date": "date"})
    return events
