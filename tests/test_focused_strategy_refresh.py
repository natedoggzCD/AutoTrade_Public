import json
from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd

from tools import focused_strategy_refresh as refresh
from tools import strategy_lab


def test_family_evidence_selects_historical_winners_and_excludes_failed_families(
    tmp_path,
):
    evidence_path = tmp_path / "strategy_factory_20260324_194307.json"
    evidence_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-03-24T19:43:07",
                "candidates": [
                    {
                        "name": "trend_follow_factory_20260324_194307",
                        "setup_type": "trend_follow",
                        "metrics": {
                            "profit_factor": 1.72,
                            "win_rate": 0.53,
                            "total_trades": 132,
                        },
                        "walkforward": {"passed": True},
                    },
                    {
                        "name": "stoch_oversold_bounce_artifact",
                        "setup_type": "stoch_bounce",
                        "metrics": {
                            "profit_factor": 999.0,
                            "win_rate": 1.0,
                            "total_trades": 3,
                        },
                        "walkforward": {"passed": False},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    evidence = refresh.load_strategy_evidence([evidence_path])
    summaries = {
        row.setup_type: row for row in refresh.summarize_family_evidence(evidence)
    }

    assert summaries["trend_follow"].selected is True
    assert summaries["trend_follow"].robust_rows == 1
    assert summaries["trend_follow"].walk_forward_passes == 1
    assert summaries["stoch_bounce"].selected is False
    assert summaries["stoch_bounce"].reason == "excluded_standalone_family"
    assert summaries["stoch_bounce"].suspicious_rows == 1


def test_build_focused_candidates_only_uses_selected_successful_families():
    candidates = refresh.build_focused_candidates(
        ["trend_follow", "pullback_support", "stoch_bounce"],
        max_candidates=1000,
    )

    setup_types = {candidate.entry.setup_type for candidate in candidates}

    assert setup_types == {"trend_follow", "pullback_support"}
    assert all(candidate.name.startswith("focused_") for candidate in candidates)
    assert all(candidate.exit.trailing_stop is False for candidate in candidates)
    assert len({candidate.name for candidate in candidates}) == len(candidates)
    assert len(
        {refresh._strategy_signature(candidate) for candidate in candidates}
    ) == len(candidates)


def test_select_walkforward_pool_limits_duplicate_family_dominance():
    rows = [{"setup_type": "ma_bounce", "rank_score": 100 - i} for i in range(10)] + [
        {"setup_type": "trend_follow", "rank_score": 50 - i} for i in range(3)
    ]

    selected = refresh._select_walkforward_pool(
        rows,
        total_limit=6,
        per_family_limit=3,
    )

    assert [row["setup_type"] for row in selected].count("ma_bounce") == 3
    assert [row["setup_type"] for row in selected].count("trend_follow") == 3


def test_resolve_expanded_universe_splits_old_and_new_cohorts(tmp_path):
    start = date(2026, 1, 1)
    rows = []
    symbols = {
        "OLD": {"close": 100.0, "cap": 50_000_000_000},
        "NEWPX": {"close": 250.0, "cap": 100_000_000_000},
        "NEWCAP": {"close": 150.0, "cap": 500_000_000_000},
        "SMALL": {"close": 25.0, "cap": 100_000_000},
        "THIN": {"close": 30.0, "cap": 100_000_000},
        "TOOBIG": {"close": 150.0, "cap": 1_500_000_000_000},
        "SPY": {"close": 500.0, "cap": 600_000_000_000},
    }
    for i in range(25):
        current = start + timedelta(days=i)
        for symbol, meta in symbols.items():
            rows.append(
                {
                    "ticker": symbol,
                    "Date": current,
                    "Close": meta["close"],
                    "Volume": 30_000 if symbol == "THIN" else 300_000,
                }
            )

    parquet_path = tmp_path / "daily_features.parquet"
    pd.DataFrame(rows).to_parquet(parquet_path)
    metadata_path = tmp_path / "nasdaq_screener.csv"
    pd.DataFrame(
        [
            {"Symbol": symbol, "Market Cap": meta["cap"]}
            for symbol, meta in symbols.items()
        ]
    ).to_csv(metadata_path, index=False)

    cohorts = refresh.resolve_expanded_universe(
        start_date=str(start),
        end_date=str(start + timedelta(days=24)),
        daily_features_path=parquet_path,
        metadata_path=metadata_path,
    )

    assert cohorts.combined == ["NEWPX", "NEWCAP", "OLD", "SMALL"]
    assert cohorts.legacy_eligible == ["OLD", "SMALL"]
    assert cohorts.expanded_additions == ["NEWPX", "NEWCAP"]
    assert cohorts.counts == {
        "combined": 4,
        "legacy_eligible": 2,
        "expanded_additions": 2,
    }


def test_shared_strategy_lab_candidates_honor_symbol_subset(monkeypatch):
    rows = []
    start = date(2026, 1, 1)
    for i in range(120):
        current = start + timedelta(days=i)
        for symbol, base_close in (("AAA", 10.0), ("BBB", 20.0)):
            close = base_close + i * 0.1
            rows.append(
                {
                    "ticker": symbol,
                    "date": current,
                    "open": close,
                    "close": close,
                    "volume": 300_000,
                    "rsi": 50.0,
                    "atr_14": close * 0.03,
                    "sma20": close - 0.5,
                    "adx": 30.0,
                    "macd_hist": 1.0,
                    "ema_10": close,
                    "ema_20": close - 0.1,
                    "bb_squeeze_flag": 0.0,
                    "price_pos_bb": 0.5,
                    "roc_5": 3.0,
                    "roc_10": 5.0,
                    "stoch_k": 50.0,
                    "cci_val": 0.0,
                    "ichi_tenkan": close,
                    "ichi_kijun": close - 0.1,
                    "ichi_senkou_a": close - 0.2,
                    "ichi_senkou_b": close - 0.3,
                    "dist_sma20": 1.0,
                    "atr_ratio": 1.0,
                }
            )

    monkeypatch.setattr(strategy_lab, "_SHARED_DF", pd.DataFrame(rows))
    monkeypatch.setattr(strategy_lab, "_SHARED_PARQUET", "unused.parquet")
    strategy = SimpleNamespace(
        entry=SimpleNamespace(
            rsi_min=40.0,
            rsi_max=60.0,
            min_atr_pct=1.0,
            max_atr_pct=5.0,
            min_volume_ratio=1.0,
            require_above_sma20=False,
            require_sma5_curl_positive=False,
            min_adx=20.0,
            require_positive_macd_hist=True,
            require_ema_10_above_20=True,
        )
    )

    result = strategy_lab._load_market_signal_candidates(
        strategy=strategy,
        start_date=str(start + timedelta(days=30)),
        end_date=str(start + timedelta(days=60)),
        symbols=["AAA"],
        top_n=25,
    )

    assert set(result["ticker"]) == {"AAA"}
