import json
from argparse import Namespace

import pandas as pd

from tools.short_engine_signal_backtest import run_backtest


def test_short_engine_backtest_validates_feature_and_ledger_candidates(tmp_path):
    features = tmp_path / "features.parquet"
    logs = tmp_path / "logs"
    logs.mkdir()
    rows = [
        {
            "ticker": "BEAR",
            "Date": "2026-01-02",
            "Open": 10.0,
            "High": 10.2,
            "Low": 9.7,
            "Close": 10.0,
            "Close_lag_1": 10.5,
            "EMA_20": 10.7,
            "SMA_20": 10.9,
            "RSI_14": 35.0,
            "atr_14": 0.4,
            "volume_ratio": 2.0,
            "ROC_5": -3.0,
        },
        {
            "ticker": "BULL",
            "Date": "2026-01-02",
            "Open": 10.0,
            "High": 10.6,
            "Low": 9.9,
            "Close": 10.4,
            "Close_lag_1": 10.0,
            "EMA_20": 10.0,
            "SMA_20": 9.8,
            "RSI_14": 65.0,
            "atr_14": 0.3,
            "volume_ratio": 1.0,
            "ROC_5": 2.0,
        },
        {
            "ticker": "BEAR",
            "Date": "2026-01-05",
            "Open": 9.8,
            "High": 9.9,
            "Low": 9.1,
            "Close": 9.2,
            "Close_lag_1": 10.0,
            "EMA_20": 10.4,
            "SMA_20": 10.6,
            "RSI_14": 30.0,
            "atr_14": 0.4,
            "volume_ratio": 2.5,
            "ROC_5": -4.0,
        },
        {
            "ticker": "BULL",
            "Date": "2026-01-05",
            "Open": 10.5,
            "High": 10.8,
            "Low": 10.4,
            "Close": 10.7,
            "Close_lag_1": 10.4,
            "EMA_20": 10.1,
            "SMA_20": 9.9,
            "RSI_14": 67.0,
            "atr_14": 0.3,
            "volume_ratio": 1.0,
            "ROC_5": 2.0,
        },
    ]
    pd.DataFrame(rows).to_parquet(features)
    (logs / "signals_2026-01-02.json").write_text(
        json.dumps(
            {
                "signals": [
                    {
                        "symbol": "BEAR",
                        "price": 10.0,
                        "confidence": 0.40,
                        "atr_14": 0.4,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = run_backtest(
        Namespace(
            start="2026-01-02",
            end="2026-01-05",
            features=str(features),
            logs=str(logs),
            mode="both",
            max_breadth_pct=50.0,
            max_up_probability=0.45,
            min_score=40.0,
            top_per_day=5,
            hold_days=1,
            slippage_bps=0.0,
            min_breadth_universe=2,
            output_json=str(tmp_path / "out.json"),
        )
    )

    assert payload["low_breadth_dates"] == 2
    assert payload["summary"]["trade_count"] >= 1
    assert payload["summary"]["win_rate"] == 1.0
    assert payload["summary"]["profit_factor"] == 999.0
    assert (tmp_path / "out.json").exists()
