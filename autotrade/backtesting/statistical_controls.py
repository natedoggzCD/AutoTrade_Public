"""
Statistical controls for backtesting: multiple testing, selection bias, and robust inference.

Implements:
- Multiple testing adjustment (Benjamini-Hochberg, Bonferroni)
- Selection bias controls: PBO (Probability of Backtest Overfitting), Deflated Sharpe Ratio
- Robust Sharpe inference with bootstrap and HAC-style tests
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, log, sqrt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _normal_pdf(x: float) -> float:
    return np.exp(-0.5 * x * x) / sqrt(2.0 * np.pi)


@dataclass
class MultipleTestingResult:
    """Result from multiple testing correction."""

    method: str
    original_pvalues: List[float]
    adjusted_pvalues: List[float]
    significant_indices: List[int]
    alpha: float


def bonferroni_correction(
    pvalues: List[float], alpha: float = 0.05
) -> MultipleTestingResult:
    """Apply Bonferroni correction for multiple comparisons."""
    n = len(pvalues)
    adjusted = [min(1.0, p * n) for p in pvalues]
    significant = [i for i, p in enumerate(adjusted) if p < alpha]
    return MultipleTestingResult(
        method="bonferroni",
        original_pvalues=list(pvalues),
        adjusted_pvalues=adjusted,
        significant_indices=significant,
        alpha=alpha,
    )


def benjamini_hochberg_correction(
    pvalues: List[float], alpha: float = 0.05
) -> MultipleTestingResult:
    """Apply Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    if n == 0:
        return MultipleTestingResult(
            method="benjamini_hochberg",
            original_pvalues=[],
            adjusted_pvalues=[],
            significant_indices=[],
            alpha=alpha,
        )

    sorted_indices = sorted(range(n), key=lambda i: pvalues[i])
    sorted_pvalues = [pvalues[i] for i in sorted_indices]

    adjusted = [0.0] * n
    for rank, p in enumerate(sorted_pvalues):
        bh_value = (p * n) / (rank + 1)
        adjusted[rank] = min(1.0, bh_value)

    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    original_order_adjusted = [0.0] * n
    for sorted_idx, orig_idx in enumerate(sorted_indices):
        original_order_adjusted[orig_idx] = adjusted[sorted_idx]

    significant = [i for i, p in enumerate(original_order_adjusted) if p < alpha]

    return MultipleTestingResult(
        method="benjamini_hochberg",
        original_pvalues=list(pvalues),
        adjusted_pvalues=original_order_adjusted,
        significant_indices=significant,
        alpha=alpha,
    )


@dataclass
class PBOResult:
    """Probability of Backtest Overfitting result."""

    pbo_estimate: float
    n_candidates: int
    n_folds: int
    in_sample_sharpes: List[float]
    out_of_sample_sharpes: List[float]
    bootstrap_samples: int


def compute_pbo(
    candidate_sharpes: List[Dict[str, float]],
    n_folds: int = 8,
    bootstrap_samples: int = 1000,
    random_seed: int = 42,
) -> PBOResult:
    """
    Estimate Probability of Backtest Overfitting using cross-validation.

    Compares in-sample vs out-of-sample performance to estimate the
    probability that a strategy's apparent performance is due to overfitting.
    """
    np.random.seed(random_seed)

    n_candidates = len(candidate_sharpes)
    if n_candidates == 0 or n_folds < 2:
        return PBOResult(
            pbo_estimate=0.0,
            n_candidates=0,
            n_folds=n_folds,
            in_sample_sharpes=[],
            out_of_sample_sharpes=[],
            bootstrap_samples=bootstrap_samples,
        )

    in_sample = []
    out_of_sample = []

    for candidate in candidate_sharpes:
        is_sharpe = candidate.get("in_sample_sharpe", 0.0)
        oos_sharpe = candidate.get("out_of_sample_sharpe", 0.0)
        in_sample.append(is_sharpe)
        out_of_sample.append(oos_sharpe)

    in_sample = np.array(in_sample)
    out_of_sample = np.array(out_of_sample)

    rank_correlations = []
    for _ in range(bootstrap_samples):
        indices = np.random.choice(n_candidates, size=n_candidates, replace=True)
        is_ranked = in_sample[indices]
        oos_ranked = out_of_sample[indices]

        if len(set(is_ranked)) > 1 and len(set(oos_ranked)) > 1:
            corr = np.corrcoef(
                np.argsort(np.argsort(is_ranked)), np.argsort(np.argsort(oos_ranked))
            )[0, 1]
            if not np.isnan(corr):
                rank_correlations.append(corr)

    if rank_correlations:
        avg_corr = np.mean(rank_correlations)
        pbo = max(0.0, 1.0 - avg_corr)
    else:
        pbo = 0.5

    return PBOResult(
        pbo_estimate=float(pbo),
        n_candidates=n_candidates,
        n_folds=n_folds,
        in_sample_sharpes=in_sample.tolist(),
        out_of_sample_sharpes=out_of_sample.tolist(),
        bootstrap_samples=bootstrap_samples,
    )


@dataclass
class DeflatedSharpeResult:
    """Deflated Sharpe Ratio result."""

    dsr: float
    original_sharpe: float
    n_trials: int
    avg_sharpe: float
    std_sharpe: float
    skewness: float
    kurtosis: float


def compute_deflated_sharpe(
    sharpe_values: List[float],
    n_trials: int,
    rf: float = 0.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    random_seed: int = 42,
) -> DeflatedSharpeResult:
    """
    Compute Deflated Sharpe Ratio (DSR) accounting for selection bias.

    Based on Bailey and Lopez de Prado (2014) "The Deflated Sharpe Ratio".
    """
    np.random.seed(random_seed)

    if not sharpe_values or n_trials <= 0:
        return DeflatedSharpeResult(
            dsr=0.0,
            original_sharpe=0.0,
            n_trials=n_trials,
            avg_sharpe=0.0,
            std_sharpe=0.0,
            skewness=0.0,
            kurtosis=0.0,
        )

    sharpes = np.array(sharpe_values)
    original_sharpe = float(np.max(sharpes))
    avg_sharpe = float(np.mean(sharpes))
    std_sharpe = float(np.std(sharpes))

    if std_sharpe <= 0:
        return DeflatedSharpeResult(
            dsr=0.0,
            original_sharpe=original_sharpe,
            n_trials=n_trials,
            avg_sharpe=avg_sharpe,
            std_sharpe=std_sharpe,
            skewness=skewness,
            kurtosis=kurtosis,
        )

    z = (original_sharpe - rf) / std_sharpe

    expected_max_sharpe = avg_sharpe + std_sharpe * (
        (1 - skewness * z + (kurtosis - 1) * (z * z - 1)) / 6
    )

    n = n_trials
    pvalue = (1 - _normal_cdf(z)) ** n

    if pvalue > 0 and pvalue < 1:
        dsr = z - _normal_cdf(pvalue) / (1 - pvalue)
    else:
        dsr = z

    dsr_adjusted = max(0.0, dsr)

    return DeflatedSharpeResult(
        dsr=float(dsr_adjusted),
        original_sharpe=original_sharpe,
        n_trials=n_trials,
        avg_sharpe=avg_sharpe,
        std_sharpe=std_sharpe,
        skewness=skewness,
        kurtosis=kurtosis,
    )


@dataclass
class RobustSharpeResult:
    """Robust Sharpe inference result."""

    sharpe: float
    ci_lower: float
    ci_upper: float
    pvalue: float
    test_type: str
    bootstrap_samples: int


def bootstrap_sharpe_inference(
    returns: List[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    rf: float = 0.0,
    random_seed: int = 42,
) -> RobustSharpeResult:
    """
    Compute robust Sharpe ratio with bootstrap confidence intervals.

    Uses bootstrap resampling to account for non-normality in returns.
    """
    np.random.seed(random_seed)

    if not returns or len(returns) < 2:
        return RobustSharpeResult(
            sharpe=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            pvalue=1.0,
            test_type="bootstrap",
            bootstrap_samples=n_bootstrap,
        )

    returns_arr = np.array(returns)
    mean_ret = np.mean(returns_arr) - rf
    std_ret = np.std(returns_arr, ddof=1)

    if std_ret <= 0:
        return RobustSharpeResult(
            sharpe=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            pvalue=1.0,
            test_type="bootstrap",
            bootstrap_samples=n_bootstrap,
        )

    sharpe = float(mean_ret / std_ret * sqrt(252.0))

    bootstrap_sharpes = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(returns_arr, size=len(returns_arr), replace=True)
        sample_mean = np.mean(sample) - rf
        sample_std = np.std(sample, ddof=1)

        if sample_std > 0:
            boot_sharpe = sample_mean / sample_std * sqrt(252.0)
            bootstrap_sharpes.append(boot_sharpe)

    if bootstrap_sharpes:
        bootstrap_sharpes = np.array(bootstrap_sharpes)
        ci_lower = float(np.percentile(bootstrap_sharpes, (alpha / 2) * 100))
        ci_upper = float(np.percentile(bootstrap_sharpes, (1 - alpha / 2) * 100))

        pvalue = float(np.mean(bootstrap_sharpes <= 0))
    else:
        ci_lower = ci_upper = sharpe
        pvalue = 1.0

    return RobustSharpeResult(
        sharpe=sharpe,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        pvalue=pvalue,
        test_type="bootstrap",
        bootstrap_samples=n_bootstrap,
    )


def hac_adjusted_sharpe(
    returns: List[float],
    rf: float = 0.0,
    bandwidth: Optional[int] = None,
) -> RobustSharpeResult:
    """
    Compute HAC (Heteroskedasticity-Autocorrelation Consistent) adjusted Sharpe.

    Uses Newey-West HAC estimator for robust inference with autocorrelated returns.
    """
    if not returns or len(returns) < 2:
        return RobustSharpeResult(
            sharpe=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            pvalue=1.0,
            test_type="hac",
            bootstrap_samples=0,
        )

    returns_arr = np.array(returns)
    n = len(returns_arr)

    excess_returns = returns_arr - rf
    mean_excess = np.mean(excess_returns)
    var_excess = np.var(excess_returns, ddof=1)

    if var_excess <= 0:
        return RobustSharpeResult(
            sharpe=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            pvalue=1.0,
            test_type="hac",
            bootstrap_samples=0,
        )

    if bandwidth is None:
        bandwidth = int(4 * (n / 100) ** (2 / 9))
    bandwidth = min(bandwidth, n - 1)

    hac_var = var_excess
    for lag in range(1, bandwidth + 1):
        gamma = 2 * np.mean(excess_returns[lag:] * excess_returns[:-lag])
        weight = 1 - lag / (bandwidth + 1)
        hac_var += weight * gamma

    hac_var = max(hac_var, var_excess * 0.5)

    sharpe = float(mean_excess / sqrt(hac_var) * sqrt(252.0))

    se = 1.0 / sqrt(n)
    ci_lower = sharpe - 1.96 * se
    ci_upper = sharpe + 1.96 * se

    pvalue = 2 * (1 - _normal_cdf(abs(sharpe)))

    return RobustSharpeResult(
        sharpe=sharpe,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        pvalue=pvalue,
        test_type="hac",
        bootstrap_samples=0,
    )


@dataclass
class WhiteRealityCheckResult:
    """White's Reality Check result."""

    pvalue: float
    best_strategy_sharpe: float
    mean_shuffle_sharpe: float
    n_bootstraps: int


def whites_reality_check(
    strategy_returns: List[float],
    benchmark_returns: List[float],
    n_bootstraps: int = 1000,
    random_seed: int = 42,
) -> WhiteRealityCheckResult:
    """
    Perform White's Reality Check for data-snooping correction.

    Compares best strategy to shuffled benchmarks to assess if outperformance
    is due to genuine signal or data snooping.
    """
    np.random.seed(random_seed)

    if not strategy_returns or not benchmark_returns:
        return WhiteRealityCheckResult(
            pvalue=1.0,
            best_strategy_sharpe=0.0,
            mean_shuffle_sharpe=0.0,
            n_bootstraps=n_bootstraps,
        )

    strat_arr = np.array(strategy_returns)
    bench_arr = np.array(benchmark_returns)

    strat_sharpe = (
        float(np.mean(strat_arr) / np.std(strat_arr, ddof=1))
        if np.std(strat_arr) > 0
        else 0.0
    )

    shuffle_sharpes = []
    for _ in range(n_bootstraps):
        shuffled = np.random.choice(bench_arr, size=len(bench_arr), replace=True)
        boot_sharpe = (
            float(np.mean(shuffled) / np.std(shuffled, ddof=1))
            if np.std(shuffled) > 0
            else 0.0
        )
        shuffle_sharpes.append(boot_sharpe)

    mean_shuffle = float(np.mean(shuffle_sharpes)) if shuffle_sharpes else 0.0
    pvalue = (
        float(np.mean([s >= strat_sharpe for s in shuffle_sharpes]))
        if shuffle_sharpes
        else 1.0
    )

    return WhiteRealityCheckResult(
        pvalue=pvalue,
        best_strategy_sharpe=strat_sharpe,
        mean_shuffle_sharpe=mean_shuffle,
        n_bootstraps=n_bootstraps,
    )


@dataclass
class SPAResult:
    """Hansen's Superior Predictive Ability (SPA) test result."""

    pvalue: float
    best_candidate_sharpe: float
    adjusted_critical_value: float
    n_candidates: int
    n_bootstraps: int


def hansens_spa_test(
    candidate_returns: List[List[float]],
    benchmark_returns: List[float],
    alpha: float = 0.05,
    n_bootstraps: int = 1000,
    random_seed: int = 42,
) -> SPAResult:
    """
    Perform Hansen's SPA (Superior Predictive Ability) test.

    More powerful than White's Reality Check for comparing multiple strategies.
    """
    np.random.seed(random_seed)

    if not candidate_returns or not benchmark_returns:
        return SPAResult(
            pvalue=1.0,
            best_candidate_sharpe=0.0,
            adjusted_critical_value=0.0,
            n_candidates=0,
            n_bootstraps=n_bootstraps,
        )

    bench_arr = np.array(benchmark_returns)
    n_candidates = len(candidate_returns)

    candidate_sharpes = []
    for returns in candidate_returns:
        ret_arr = np.array(returns)
        shr = (
            float(np.mean(ret_arr) / np.std(ret_arr, ddof=1))
            if np.std(ret_arr) > 0
            else 0.0
        )
        candidate_sharpes.append(shr)

    best_idx = np.argmax(candidate_sharpes)
    best_sharpe = candidate_sharpes[best_idx]

    test_statistics = []
    for _ in range(n_bootstraps):
        boot_bench = np.random.choice(bench_arr, size=len(bench_arr), replace=True)
        bench_mean = np.mean(boot_bench)

        stat = max(0, best_sharpe - bench_mean)
        test_statistics.append(stat)

    test_statistics = np.array(test_statistics)

    critical_value = np.percentile(test_statistics, (1 - alpha) * 100)

    observed_stat = max(0, best_sharpe - np.mean(bench_arr))
    pvalue = float(np.mean(test_statistics >= observed_stat))

    return SPAResult(
        pvalue=pvalue,
        best_candidate_sharpe=best_sharpe,
        adjusted_critical_value=float(critical_value),
        n_candidates=n_candidates,
        n_bootstraps=n_bootstraps,
    )


def apply_statistical_controls(
    candidate_metrics: List[Dict[str, Any]],
    method: str = "dsr+spa",
    min_candidates: int = 20,
    bootstrap_samples: int = 1000,
    alpha: float = 0.05,
    enable_pbo: bool = True,
    pbo_min_folds: int = 8,
    robust_sharpe_test: str = "bootstrap",
) -> Dict[str, Any]:
    """
    Apply comprehensive statistical controls to backtest candidates.

    Returns adjusted metrics and decision flags.
    """
    results = {
        "multiple_testing": None,
        "pbo": None,
        "deflated_sharpe": None,
        "spa": None,
        "robust_sharpe": None,
        "passed_controls": False,
        "reason_codes": [],
    }

    n_candidates = len(candidate_metrics)

    if n_candidates >= min_candidates and method in [
        "bonferroni",
        "bh",
        "benjamini_hochberg",
    ]:
        pvalues = [m.get("pvalue", 0.5) for m in candidate_metrics]

        if method == "bonferroni":
            mt_result = bonferroni_correction(pvalues, alpha)
        else:
            mt_result = benjamini_hochberg_correction(pvalues, alpha)

        results["multiple_testing"] = {
            "method": mt_result.method,
            "n_significant": len(mt_result.significant_indices),
            "significant_indices": mt_result.significant_indices,
        }

        if len(mt_result.significant_indices) > 0:
            results["reason_codes"].append("multiple_testing_passed")

    if enable_pbo and n_candidates >= pbo_min_folds:
        candidate_sharpes = [
            {
                "in_sample_sharpe": m.get("in_sample_sharpe", 0.0),
                "out_of_sample_sharpe": m.get("out_of_sample_sharpe", 0.0),
            }
            for m in candidate_metrics
        ]

        pbo_result = compute_pbo(
            candidate_sharpes,
            n_folds=pbo_min_folds,
            bootstrap_samples=bootstrap_samples,
        )

        results["pbo"] = {
            "pbo_estimate": pbo_result.pbo_estimate,
            "n_candidates": pbo_result.n_candidates,
        }

        if pbo_result.pbo_estimate < 0.6:
            results["reason_codes"].append("pbo_acceptable")

    if "dsr" in method.lower() and n_candidates >= 2:
        sharpe_values = [m.get("sharpe_ratio", 0.0) for m in candidate_metrics]
        dsr_result = compute_deflated_sharpe(
            sharpe_values,
            n_trials=n_candidates,
        )

        results["deflated_sharpe"] = {
            "dsr": dsr_result.dsr,
            "original_sharpe": dsr_result.original_sharpe,
            "n_trials": dsr_result.n_trials,
        }

        if dsr_result.dsr > 0.5:
            results["reason_codes"].append("dsr_acceptable")

    if "spa" in method.lower() and n_candidates >= 2:
        candidate_returns = [m.get("returns", []) for m in candidate_metrics]
        benchmark_returns = candidate_returns[0] if candidate_returns else []

        spa_result = hansens_spa_test(
            candidate_returns,
            benchmark_returns,
            alpha=alpha,
            n_bootstraps=bootstrap_samples,
        )

        results["spa"] = {
            "pvalue": spa_result.pvalue,
            "best_candidate_sharpe": spa_result.best_candidate_sharpe,
            "n_candidates": spa_result.n_candidates,
        }

        if spa_result.pvalue < alpha:
            results["reason_codes"].append("spa_significant")

    all_returns = []
    for m in candidate_metrics:
        rets = m.get("returns", [])
        if rets:
            all_returns.extend(rets)

    if robust_sharpe_test == "bootstrap":
        if all_returns:
            rs_result = bootstrap_sharpe_inference(
                all_returns,
                n_bootstrap=bootstrap_samples,
                alpha=alpha,
            )

            results["robust_sharpe"] = {
                "sharpe": rs_result.sharpe,
                "ci_lower": rs_result.ci_lower,
                "ci_upper": rs_result.ci_upper,
                "pvalue": rs_result.pvalue,
                "test_type": rs_result.test_type,
            }

            if rs_result.pvalue < alpha and rs_result.sharpe > 0:
                results["reason_codes"].append("robust_sharpe_significant")
    elif robust_sharpe_test == "hac" and all_returns:
        rs_result = hac_adjusted_sharpe(all_returns)

        results["robust_sharpe"] = {
            "sharpe": rs_result.sharpe,
            "ci_lower": rs_result.ci_lower,
            "ci_upper": rs_result.ci_upper,
            "pvalue": rs_result.pvalue,
            "test_type": rs_result.test_type,
        }

        if rs_result.pvalue < alpha and rs_result.sharpe > 0:
            results["reason_codes"].append("robust_sharpe_significant")

    results["passed_controls"] = len(results["reason_codes"]) > 0

    return results
