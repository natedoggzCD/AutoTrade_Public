from datetime import date

from autotrade.backtesting.contracts import BacktestRequest
from autotrade.backtesting.protocol import BacktestProtocol, ProtocolAdapter, create_protocol_from_config


class DummyConfig:
    initial_cash = 100_000.0
    walk_forward_train_days = 20
    walk_forward_test_days = 10
    commission_pct = 0.001
    slippage_pct = 0.0005
    spread_pct = 0.001
    seed = 42
    max_positions = 10
    max_position_value = 10_000.0
    max_hold_days = 5


class DummyEngine:
    def __init__(self) -> None:
        self.config = DummyConfig()

    def _run_single_period(self, symbols, start_date, end_date, save_path, log_summary):
        _ = symbols, save_path, log_summary
        sharpe = float((end_date - start_date).days) / 10.0
        return {
            "total_trades": 3,
            "total_pnl": 120.0,
            "total_return_pct": 1.2,
            "sharpe_ratio": sharpe,
            "sortino_ratio": 1.1,
            "max_drawdown": 0.05,
            "calmar_ratio": 0.9,
            "win_rate": 0.67,
            "profit_factor": 1.4,
            "avg_trade_pnl": 40.0,
            "turnover": 0.2,
            "trades": [{"pnl_dollars": 10.0}],
            "config": {"backtest": {"min_bullish": 4.0}},
        }

    def run(self, symbols, lookback_days, start_date, end_date, walk_forward):
        _ = symbols, lookback_days, start_date, end_date, walk_forward
        return {"legacy": True}


def _request(nested: bool = False) -> BacktestRequest:
    return BacktestRequest(
        strategy_id="demo",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 4, 1),
        train_days=20,
        test_days=10,
        nested_validation_enabled=nested,
        inner_folds=3,
    )


def test_protocol_walk_forward_builds_oos_folds() -> None:
    protocol = BacktestProtocol(request=_request(), engine=DummyEngine())
    artifact = protocol.run_walk_forward(["AAPL"])

    assert artifact.folds
    assert all(fold.is_oos for fold in artifact.folds)
    assert artifact.metrics is not None
    assert artifact.metrics.get("total_trades", 0) > 0


def test_protocol_nested_validation_generates_inner_results() -> None:
    protocol = BacktestProtocol(request=_request(nested=True), engine=DummyEngine())
    artifact = protocol.run_walk_forward(["AAPL"])

    assert artifact.folds
    assert protocol._nested_results
    assert len(protocol._nested_results) == len(artifact.folds)
    assert protocol._nested_results[0].metrics["evaluated_inner_folds"] >= 1


def test_protocol_adapter_routes_explicit_dates_through_protocol() -> None:
    payload = ProtocolAdapter.run_backtest_legacy(
        engine=DummyEngine(),
        symbols=["AAPL"],
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        walk_forward=False,
    )

    assert payload.get("schema_version") == "1.0"
    assert payload.get("folds")


def test_protocol_adapter_falls_back_for_lookback_only_calls() -> None:
    payload = ProtocolAdapter.run_backtest_legacy(
        engine=DummyEngine(),
        symbols=["AAPL"],
        lookback_days=30,
        start_date=None,
        end_date=None,
        walk_forward=False,
    )

    assert payload == {"legacy": True}


def test_create_protocol_from_nested_config_uses_backtest_protocol_values() -> None:
    config = {
        "strategy_id": "cfg_demo",
        "backtest": {
            "initial_cash": 250_000.0,
            "commission_pct": 0.002,
            "slippage_pct": 0.0007,
            "spread_pct": 0.0011,
            "walk_forward_train_days": 60,
            "walk_forward_test_days": 15,
        },
        "backtest_protocol": {
            "walk_forward": {"mode": "rolling", "train_days": 90, "test_days": 30},
            "nested_validation": {"enabled": True, "inner_folds": 4},
        },
    }
    protocol = create_protocol_from_config(
        config,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
    )

    assert protocol.request.train_days == 90
    assert protocol.request.test_days == 30
    assert protocol.request.nested_validation_enabled is True
    assert protocol.request.inner_folds == 4
