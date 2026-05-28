from datetime import date

from autotrade.backtesting.contracts import (
    BacktestRequest,
    BacktestResultArtifact,
    FoldResult,
)


def test_backtest_request_hash_is_deterministic() -> None:
    req = BacktestRequest(
        strategy_id="demo",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 3, 1),
        metadata={"note": "x"},
    )
    assert req.to_config_hash() == req.to_config_hash()


def test_backtest_artifact_round_trip(tmp_path) -> None:
    artifact = BacktestResultArtifact(
        schema_version="1.0",
        config_hash="abc123",
        run_id="demo_abc123_20260101",
        run_timestamp="2026-01-01T00:00:00",
        strategy_id="demo",
        start_date="2025-01-01",
        end_date="2025-02-01",
        request=BacktestRequest(
            strategy_id="demo",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 2, 1),
        ).to_dict(),
        folds=[
            FoldResult(
                fold_index=1,
                train_start=date(2024, 10, 1),
                train_end=date(2024, 12, 31),
                test_start=date(2025, 1, 1),
                test_end=date(2025, 1, 31),
                trades_count=5,
                pnl_dollars=123.45,
            )
        ],
    )

    out = tmp_path / "artifact.json"
    artifact.save(out)
    loaded = BacktestResultArtifact.load(out)

    assert loaded.strategy_id == "demo"
    assert len(loaded.folds) == 1
    assert isinstance(loaded.folds[0], FoldResult)
    assert loaded.folds[0].test_start == date(2025, 1, 1)
