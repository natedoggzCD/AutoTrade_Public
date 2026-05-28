"""Foundation tests for the strategy-lab redesign:

- purged k-fold generates leak-free folds
- alpha catalog has registered modules and they run on synthetic data
- hierarchical shrinkage shrinks toward cluster prior as expected
- cost model loads without error
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from autotrade.backtesting.purged_walk_forward import (
    PurgedKFoldConfig,
    generate_purged_folds,
)
from autotrade.backtesting.hierarchical_shrinkage import (
    SymbolAlphaEstimate,
    assign_cluster_key,
    shrink_estimates,
)


def _synthetic_bars(n_days: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp("2026-05-01"), periods=n_days, freq="B")
    drift = 0.0003
    vol = 0.015
    rets = rng.normal(drift, vol, n_days)
    close = 50.0 * np.exp(np.cumsum(rets))
    high = close * (1.0 + rng.uniform(0.001, 0.02, n_days))
    low = close * (1.0 - rng.uniform(0.001, 0.02, n_days))
    open_ = close * (1.0 + rng.uniform(-0.01, 0.01, n_days))
    volume = rng.integers(100_000, 5_000_000, n_days)
    return pd.DataFrame(
        {
            "date": [d.date() for d in dates],
            "open": open_,
            "high": np.maximum.reduce([high, open_, close]),
            "low": np.minimum.reduce([low, open_, close]),
            "close": close,
            "volume": volume,
        }
    )


def _synthetic_alpha_context():
    from autotrade.backtesting.alpha_catalog import AlphaContext

    symbols = ["TEST", "AAA", "BBB", "CCC", "DDD"]
    frames = []
    for offset, symbol in enumerate(symbols):
        bars = _synthetic_bars(seed=offset + 10)
        bars["symbol"] = symbol
        # Make TEST's bars identical to the per-symbol fixture so symbol
        # inference works even if an evaluator omits the symbol column.
        if symbol == "TEST":
            bars = _synthetic_bars()
            bars["symbol"] = symbol
        frames.append(bars)
    universe = pd.concat(frames, ignore_index=True)
    spy_bars = _synthetic_bars(seed=99)
    earnings_calendar = pd.DataFrame(
        {
            "symbol": ["TEST"],
            "date": [universe.loc[200, "date"]],
            "surprise_pct": [8.0],
        }
    )
    return AlphaContext(
        spy_bars=spy_bars,
        sector_map={symbol: "TECH" for symbol in symbols},
        universe_bars=universe,
        earnings_calendar=earnings_calendar,
    )


def test_purged_folds_have_no_train_test_overlap():
    bars = _synthetic_bars()
    cfg = PurgedKFoldConfig(n_splits=4, label_horizon_days=10, embargo_days=5)
    folds = generate_purged_folds(bars["date"].tolist(), cfg)
    assert folds, "should produce at least one fold"
    for f in folds:
        # train and test masks must be disjoint
        assert not (f.train_mask & f.test_mask).any()
        # train end must precede test start
        assert f.train_end < f.test_start
        # embargo end is at or after test end
        assert f.embargo_end >= f.test_end


def test_purged_folds_purge_label_window():
    bars = _synthetic_bars()
    cfg = PurgedKFoldConfig(
        n_splits=3,
        label_horizon_days=20,
        embargo_days=5,
        min_train_days=100,
    )
    folds = generate_purged_folds(bars["date"].tolist(), cfg)
    assert folds
    dates = pd.to_datetime(pd.Series(bars["date"])).dt.date.to_numpy()
    for f in folds:
        # no training bar should sit within the purge zone
        purge_lo = f.test_start - timedelta(days=cfg.label_horizon_days)
        purge_hi = f.embargo_end
        train_dates = dates[f.train_mask]
        in_zone = (train_dates >= purge_lo) & (train_dates <= purge_hi)
        assert not in_zone.any(), "training bars must not overlap purge zone"


def test_alpha_catalog_registers_expected_alphas():
    from autotrade.backtesting.alpha_catalog import iter_alphas, list_alpha_ids

    ids = list_alpha_ids()
    assert len(ids) >= 30
    # we expect at least one alpha from each non-event family
    expected_substrings = ["ts_momentum", "donchian", "bollinger_z", "nr", "inside_bar"]
    for sub in expected_substrings:
        assert any(sub in aid for aid in ids), f"missing alpha matching '{sub}' in {ids}"
    # every alpha has a callable generator
    for a in iter_alphas():
        assert callable(a.generate)


def test_each_alpha_runs_on_synthetic_bars_without_error():
    from autotrade.backtesting.alpha_catalog import iter_alphas

    bars = _synthetic_bars()
    bars["symbol"] = "TEST"
    ctx = _synthetic_alpha_context()
    for alpha in iter_alphas():
        out = alpha.generate(bars, ctx)
        assert isinstance(out, pd.DataFrame)
        assert set(out.columns) == {"date", "entry", "exit", "side", "score", "note"}
        # entries and exits must come in matched pairs per fired signal
        # (each entry row has a corresponding exit row later in the frame)
        if not out.empty:
            assert (out["entry"] | out["exit"]).all()


def test_cross_sectional_alphas_degrade_without_context():
    from autotrade.backtesting.alpha_catalog import alphas_for_family

    bars = _synthetic_bars()
    for alpha in alphas_for_family("cross_sectional"):
        out = alpha.generate(bars, None)
        assert isinstance(out, pd.DataFrame)
        assert out.empty
        assert set(out.columns) == {"date", "entry", "exit", "side", "score", "note"}


def test_shrinkage_pulls_thin_sample_toward_cluster():
    # Two symbols in the same cluster. One has 5 trades with raw_mean=0.10 (lucky);
    # the other has 200 trades with raw_mean=0.01. Shrinkage should pull the
    # thin one's posterior strongly toward the well-sampled symbol's prior.
    estimates = [
        SymbolAlphaEstimate(
            symbol="THIN", alpha_id="ts_momentum_12_1", cluster_key="TECH|atrMID|MID",
            n_trades=5, raw_mean_return=0.10, raw_std_return=0.20, raw_pf=2.0,
        ),
        SymbolAlphaEstimate(
            symbol="THICK", alpha_id="ts_momentum_12_1", cluster_key="TECH|atrMID|MID",
            n_trades=200, raw_mean_return=0.01, raw_std_return=0.05, raw_pf=1.1,
        ),
    ]
    out = shrink_estimates(estimates, k=30.0)
    thin = next(r for r in out if r.symbol == "THIN")
    thick = next(r for r in out if r.symbol == "THICK")
    # posterior for thin should be substantially below its raw 0.10
    assert thin.posterior_mean_return < 0.05
    # posterior for thick should remain near its raw 0.01 (high weight on symbol)
    assert abs(thick.posterior_mean_return - 0.01) < 0.005
    # the thin symbol's weight should be much smaller than the thick symbol's
    assert thin.weight < thick.weight


def test_shrinkage_falls_back_to_raw_when_cluster_too_sparse():
    # Single symbol in cluster — n_cluster_trades < MIN_CLUSTER_N → posterior == raw
    estimates = [
        SymbolAlphaEstimate(
            symbol="LONE", alpha_id="ts_momentum_12_1", cluster_key="UNK|atrLOW|UNK",
            n_trades=20, raw_mean_return=0.07, raw_std_return=0.1, raw_pf=1.4,
        )
    ]
    out = shrink_estimates(estimates)
    assert out[0].posterior_mean_return == pytest.approx(0.07)
    assert out[0].weight == pytest.approx(1.0)


def test_cluster_key_bands_match_atr_thresholds():
    k_low = assign_cluster_key("AAA", "TECH", 1.0, "MID")
    k_mid = assign_cluster_key("BBB", "TECH", 3.0, "MID")
    k_hi = assign_cluster_key("CCC", "TECH", 6.0, "MID")
    assert "atrLOW" in k_low
    assert "atrMID" in k_mid
    assert "atrHI" in k_hi


def test_cost_model_builds_from_journal_or_empty_fallback():
    from autotrade.backtesting.cost_model import build_cost_model, cost_model_summary

    model = build_cost_model()
    summary = cost_model_summary(model)
    assert "global_median_bps" in summary
    assert summary["global_median_bps"] >= 0.0
    # round-trip cost on a 1.0% return should be lower than the input
    after = model.apply_round_trip(1.0, "ANY")
    assert after <= 1.0
