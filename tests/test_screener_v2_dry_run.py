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


def test_screener_v2_dry_run_schema():
    """Dry-run screener v2 on a small subset and validate schema."""
    from autotrade.utils.local_data_provider import get_provider
    from autotrade.signals.screener_v2 import ScreenerV2

    provider = get_provider()
    tickers = provider.get_available_tickers()
    if not tickers and _skip("No local tickers available"):
        return

    symbols = tickers[:50]

    screener = ScreenerV2(config_override={
        'min_composite_score': 0,
        'min_atr_pct': 0,
        'max_atr_pct': 100,
        'rsi_min': 0,
        'rsi_max': 100,
        'prefer_bullish_regime': False,
        'enforce_min_candidates': False,
    })

    results = screener.screen(symbols=symbols, max_candidates=20, log_samples=False)
    if not results and _skip("Screener returned no results (data missing or insufficient history)"):
        return

    required_keys = [
        'ticker', 'price', 'score', 'composite_score', 'sr_bonus', 'rsi',
        's1_price', 's1_strength', 'r1_price', 'r1_strength',
        'atr_14', 'risk_reward', 'stop_price', 'target_price',
        'score_breakdown', 'factor_scores',
        'sma5_slope_pct', 'sma5_accel', 'sma20_trend_pct', 'gap_pct'
    ]

    sample = results[0]
    for key in required_keys:
        assert key in sample, f"Missing key: {key}"

    assert isinstance(sample['score'], (int, float))
    assert isinstance(sample['score_breakdown'], dict)
    assert isinstance(sample['factor_scores'], dict)

    # Validate expected factor scores exist
    for factor in ['sma5_curl', 'ema_alignment', 'regime', 'rsi', 'macd', 'stoch', 'bb_position']:
        assert factor in sample['factor_scores'], f"Missing factor score: {factor}"


if __name__ == '__main__':
    try:
        test_screener_v2_dry_run_schema()
        print("Screener v2 dry-run test: PASS")
    except Exception as exc:
        print(f"Screener v2 dry-run test: FAIL ({exc})")
        raise
