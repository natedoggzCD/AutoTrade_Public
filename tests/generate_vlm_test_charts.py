#!/usr/bin/env python
"""
Generate VLM-Optimized Test Charts (Option-2 Layout: No Volume)

4-Panel Layout: Price + MACD + RSI + Stochastic
- 1920x1200 @ 150dpi with tight layout
- GridSpec ratios [68, 14, 9, 9]
- NO VOLUME panel
- MACD(12,26,9) with cross detection
- RSI(14) with 30/50/70 lines
- Fast+Slow Stochastic with 20/80 lines
- Event tags with (t-n) age and ACTIVE/STALE gating

Usage:
    python tests/generate_vlm_test_charts.py --count 5
    python tests/generate_vlm_test_charts.py AAPL NVDA TSLA
"""

import sys
import random
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from autotrade.utils.vlm_chart_generator import VLMChartGenerator, CHARTS_DIR


# Popular tickers for random selection
POPULAR_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "AMD",
    "NFLX",
    "CRM",
    "ORCL",
    "ADBE",
    "INTC",
    "QCOM",
    "AVGO",
    "TXN",
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    "ARKK",
    "XLK",
    "XLF",
    "XLE",
    "JPM",
    "BAC",
    "GS",
    "MS",
    "V",
    "MA",
    "PYPL",
    "SQ",
    "WMT",
    "TGT",
    "COST",
    "HD",
    "LOW",
    "NKE",
    "SBUX",
    "MCD",
]


def main():
    parser = argparse.ArgumentParser(description="Generate VLM-optimized 4-panel charts (No Volume)")
    parser.add_argument("symbols", nargs="*", help="Specific symbols to generate")
    parser.add_argument("--count", "-n", type=int, default=5, help="Number of random charts to generate (default: 5)")
    parser.add_argument("--days", "-d", type=int, default=60, help="Days of history (default: 60)")
    parser.add_argument("--force", "-f", action="store_true", help="Force regeneration even if cached")
    args = parser.parse_args()

    # Determine symbols
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        symbols = random.sample(POPULAR_TICKERS, min(args.count, len(POPULAR_TICKERS)))

    print(f"\n{'=' * 70}")
    print("  VLM-OPTIMIZED CHART GENERATOR (Option-2: No Volume)")
    print(f"{'=' * 70}")
    print("  Layout: Price (68%) + MACD (14%) + RSI (9%) + STOCH (9%)")
    print("  Size: 1920x1200 @ 150dpi")
    print("  Indicators: SMA20, SMA50, AVWAP(H), MACD, RSI, Fast/Slow Stoch")
    print("  Signals: MACD_BULL_CROSS, AVWAP_RECLAIM, STOCH_BULL_LT20")
    print(f"{'=' * 70}")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Days: {args.days}")
    print(f"  Output: {CHARTS_DIR}")
    print(f"{'=' * 70}\n")

    # Generate charts
    generator = VLMChartGenerator()
    results = []

    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i}/{len(symbols)}] {symbol}...", end=" ", flush=True)

        try:
            chart_path = generator.generate_chart(
                symbol=symbol,
                days=args.days,
                force_regenerate=True,
                output_prefix="VL_TEST",
            )

            if chart_path and chart_path.exists():
                print(f"✓ {chart_path.name}")
                results.append((symbol, str(chart_path), True))
            else:
                print("✗ No chart generated")
                results.append((symbol, None, False))

        except Exception as e:
            print(f"✗ Error: {e}")
            results.append((symbol, None, False))

    # Summary
    success_count = sum(1 for _, _, ok in results if ok)
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: {success_count}/{len(symbols)} charts generated")
    print(f"{'=' * 70}")

    if success_count > 0:
        print("\n  Generated charts:")
        for symbol, path, ok in results:
            if ok:
                print(f"    - {path}")

    print()
    return 0 if success_count == len(symbols) else 1


if __name__ == "__main__":
    sys.exit(main())
