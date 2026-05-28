from autotrade.backtesting.statistical_controls import apply_statistical_controls


def _candidate(idx: int) -> dict:
    base = 0.02 + idx * 0.001
    returns = [base, -0.005 + idx * 0.0002, 0.01 + idx * 0.0003, -0.002]
    sharpe = 0.3 + idx * 0.05
    return {
        "pvalue": max(0.001, 0.08 - idx * 0.002),
        "in_sample_sharpe": sharpe + 0.15,
        "out_of_sample_sharpe": sharpe,
        "sharpe_ratio": sharpe,
        "returns": returns,
    }


def test_apply_statistical_controls_dsr_spa_path() -> None:
    candidate_metrics = [_candidate(i) for i in range(20)]
    out = apply_statistical_controls(
        candidate_metrics,
        method="dsr+spa",
        min_candidates=20,
        bootstrap_samples=200,
        alpha=0.05,
        enable_pbo=True,
        pbo_min_folds=8,
        robust_sharpe_test="bootstrap",
    )

    assert "multiple_testing" in out
    assert out["pbo"] is not None
    assert out["deflated_sharpe"] is not None
    assert out["spa"] is not None
    assert out["robust_sharpe"] is not None
    assert isinstance(out["reason_codes"], list)


def test_apply_statistical_controls_bonferroni_path() -> None:
    candidate_metrics = [_candidate(i) for i in range(20)]
    out = apply_statistical_controls(
        candidate_metrics,
        method="bonferroni",
        min_candidates=20,
        bootstrap_samples=100,
        alpha=0.05,
        enable_pbo=False,
        robust_sharpe_test="bootstrap",
    )

    mt = out["multiple_testing"]
    assert mt is not None
    assert mt["method"] == "bonferroni"
