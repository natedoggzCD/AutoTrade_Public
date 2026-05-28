import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autotrade.utils.vl_chart_validator import VLChartValidator
from autotrade.utils.local_data_provider import get_provider
from tools.screenipy_mini import detect_patterns


def _find_chart_for_symbol(symbol: str, charts_dir: Path) -> Path | None:
    if not charts_dir.exists():
        return None
    matches = sorted(charts_dir.glob(f"{symbol}_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _matches_pattern(vl_payload: Dict, screeni_patterns: List[str]) -> bool:
    if not screeni_patterns:
        return True
    haystack = " ".join([
        str(vl_payload.get('pattern_primary', '')),
        str(vl_payload.get('pattern_secondary', '')),
        str(vl_payload.get('pattern', '')),
        str(vl_payload.get('reasoning', '')),
        " ".join(vl_payload.get('key_observations', []) if isinstance(vl_payload.get('key_observations'), list) else []),
    ]).lower()

    for pat in screeni_patterns:
        token = pat.lower().split('(')[0].strip()
        if token and token in haystack:
            return True
    return False


def _resolve_symbols(symbols_arg: str, limit: int) -> List[str]:
    if symbols_arg:
        return [s.strip().upper() for s in symbols_arg.split(',') if s.strip()]
    provider = get_provider()
    available = provider.get_available_tickers()
    if not available:
        return []
    if limit > 0:
        return [s.upper() for s in available[:limit]]
    return [s.upper() for s in available[:10]]


def main() -> int:
    parser = argparse.ArgumentParser(description='Compare VL model patterns vs Screeni-py mini detector')
    parser.add_argument('--symbols', type=str, default='', help='Comma-separated symbols (optional)')
    parser.add_argument('--limit', type=int, default=10, help='Auto-pick N symbols from local dataset if --symbols omitted')
    parser.add_argument('--days', type=int, default=250, help='Lookback days for Screeni-py mini')
    parser.add_argument('--charts-dir', type=str, default='charts', help='Charts directory')
    parser.add_argument('--model', type=str, default='bespoke-minichart:7b', help='VL model')
    parser.add_argument('--skip-model', action='store_true', help='Skip VL model call')
    parser.add_argument('--allow-generate-chart', action='store_true', help='Allow yfinance chart generation if chart missing')
    parser.add_argument('--fail-on-mismatch', action='store_true', help='Exit non-zero if mismatch detected')
    args = parser.parse_args()

    symbols = _resolve_symbols(args.symbols, args.limit)
    if not symbols:
        print("SKIP: no symbols available from local dataset")
        return 0
    charts_dir = Path(args.charts_dir)

    validator = VLChartValidator(vl_model=args.model)
    mismatches = 0

    for symbol in symbols:
        screeni = detect_patterns(symbol, days=args.days)
        if screeni.notes.get('error') == 'no data':
            print(f"{symbol}: SKIP (no local data)")
            continue
        chart_path = _find_chart_for_symbol(symbol, charts_dir)

        if args.skip_model:
            print(f"{symbol}: Screeni-py patterns={screeni.patterns} | VL=SKIPPED")
            continue

        if not chart_path and args.allow_generate_chart:
            chart_path = validator.generate_chart(symbol)

        if not chart_path:
            print(f"{symbol}: SKIP (no chart available)")
            continue

        vl_result = validator.validate_chart(chart_path, symbol, sr_data=None)

        match = _matches_pattern(vl_result, screeni.patterns)
        if not match:
            mismatches += 1

        print(json.dumps({
            'symbol': symbol,
            'screeni_patterns': screeni.patterns,
            'vl_pattern_primary': vl_result.get('pattern') or vl_result.get('pattern_primary'),
            'vl_direction': vl_result.get('direction'),
            'vl_confidence': vl_result.get('confidence'),
            'match': match,
        }, indent=2))

    if args.fail_on_mismatch and mismatches:
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
