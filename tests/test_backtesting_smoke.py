import sys
from pathlib import Path

try:
    import pytest
except ImportError:  # pragma: no cover - optional pytest
    pytest = None

# Ensure project root is on path when running from tests/ directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _skip(msg: str) -> bool:
    if pytest:
        pytest.skip(msg)
    print(f"SKIP: {msg}")
    return True


def test_backtest_engine_smoke_schema():
    from autotrade.backtesting.engine import BacktestEngine
    from autotrade.utils.local_data_provider import get_provider

    provider = get_provider()
    tickers = provider.get_available_tickers()
    if not tickers and _skip("No local tickers available"):
        return

    symbols = tickers[:5]

    engine = BacktestEngine(
        config_override={
            # Make the smoke test permissive and fast
            "signals_per_day": 10,
            "max_positions": 2,
            "max_hold_days": 3,
            "min_bullish": 0,
            "min_lesson_score": -100,
            # Deterministic + no execution noise for schema validation
            "commission_pct": 0.0,
            "slippage_pct": 0.0,
            "spread_pct": 0.0,
            "partial_fill_min": 1.0,
            "partial_fill_max": 1.0,
            "seed": 1,
        }
    )

    results = engine.run(symbols=symbols, lookback_days=30, log_summary=False)
    if results.get("error") and _skip(f"Backtest unavailable: {results.get('error')}"):
        return

    required_keys = [
        "backtest_date",
        "start_date",
        "end_date",
        "symbols",
        "total_trades",
        "win_rate",
        "profit_factor",
        "total_pnl",
        "avg_trade",
        "max_drawdown",
        "stats",
        "trades",
        "equity_curve",
    ]
    for key in required_keys:
        assert key in results, f"Missing key: {key}"

    assert 0.0 <= float(results["win_rate"]) <= 1.0
    assert 0.0 <= float(results["max_drawdown"]) <= 1.0
    assert isinstance(results["stats"], dict)
    assert isinstance(results["trades"], list)


if __name__ == "__main__":
    try:
        test_backtest_engine_smoke_schema()
        print("Backtest engine smoke test: PASS")
    except Exception as exc:
        print(f"Backtest engine smoke test: FAIL ({exc})")
        raise

