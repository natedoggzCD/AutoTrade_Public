#!/usr/bin/env python3
"""
VL Setup Classifier Backtest
============================
Tests the new VL Setup Classifier prompt against generated charts.
Compares hardcoded signal detection with VL model visual classification.

Usage:
    conda run -n gpu-stocks python tests/backtest_vl_classifier.py
    conda run -n gpu-stocks python tests/backtest_vl_classifier.py AAPL NVDA TSLA
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from autotrade.utils.vlm_chart_generator import (
    VLMChartGenerator,
    compute_indicators,
    detect_events,
)
from autotrade.utils.vl_chart_validator import VLChartValidator


# Test symbols covering different market conditions
TEST_SYMBOLS = ["AAPL", "NVDA", "TSLA", "MSFT", "META", "AMD", "GOOGL", "AMZN"]


def get_hardcoded_classification(df, events: Dict) -> Dict:
    """
    Apply STRICT decision rules matching user's technical analysis.
    Key rules:
    - Below both zones = BEARISH
    - All indicators falling = BEARISH even if at support
    - In support zone but no bounce = BEARISH/NEUTRAL
    - Need ACTUAL bounce confirmation, not just proximity
    """
    import pandas as pd

    n = len(df)

    # Extract current values
    close = df["Close"].iloc[-1]
    sma20 = df["SMA20"].iloc[-1] if pd.notna(df["SMA20"].iloc[-1]) else None
    sma50 = df["SMA50"].iloc[-1] if pd.notna(df["SMA50"].iloc[-1]) else None
    avwap = df["AVWAP_high"].iloc[-1] if pd.notna(df["AVWAP_high"].iloc[-1]) else None
    rsi = df["RSI"].iloc[-1] if pd.notna(df["RSI"].iloc[-1]) else 50
    macd = df["MACD"].iloc[-1] if pd.notna(df["MACD"].iloc[-1]) else 0
    macd_sig = df["MACD_Signal"].iloc[-1] if pd.notna(df["MACD_Signal"].iloc[-1]) else 0
    macd_hist = df["MACD_Hist"].iloc[-1] if pd.notna(df["MACD_Hist"].iloc[-1]) else 0
    fast_k = df["Fast_K"].iloc[-1] if pd.notna(df["Fast_K"].iloc[-1]) else 50
    slow_d = df["Slow_D"].iloc[-1] if pd.notna(df["Slow_D"].iloc[-1]) else 50

    # Previous values for direction detection
    macd_hist_prev = df["MACD_Hist"].iloc[-2] if n > 1 and pd.notna(df["MACD_Hist"].iloc[-2]) else 0
    macd_hist_prev2 = df["MACD_Hist"].iloc[-3] if n > 2 and pd.notna(df["MACD_Hist"].iloc[-3]) else 0
    rsi_prev = df["RSI"].iloc[-2] if n > 1 and pd.notna(df["RSI"].iloc[-2]) else 50
    fast_k_prev = df["Fast_K"].iloc[-2] if n > 1 and pd.notna(df["Fast_K"].iloc[-2]) else 50
    close_prev = df["Close"].iloc[-2] if n > 1 else close

    atr = df["ATR"].iloc[-1] if pd.notna(df["ATR"].iloc[-1]) else close * 0.02

    # Determine price position vs zones
    supports = events.get("support_zones", [])
    resistances = events.get("resistance_zones", [])

    support_level = supports[0] if supports else close - atr * 3
    resistance_level = resistances[0] if resistances else close + atr * 3

    below_support = close < support_level - atr * 0.5
    in_support = abs(close - support_level) < atr * 1.5
    above_support = close > support_level + atr * 0.5

    below_resistance = close < resistance_level - atr * 0.5
    in_resistance = abs(close - resistance_level) < atr * 1.5
    above_resistance = close > resistance_level + atr * 0.5

    # Price position
    below_avwap = avwap is not None and close < avwap
    above_avwap = avwap is not None and close > avwap
    below_sma20 = sma20 is not None and close < sma20
    above_sma20 = sma20 is not None and close > sma20

    # Indicator directions (are they ACTUALLY turning?)
    macd_falling = macd_hist < macd_hist_prev and macd_hist_prev < macd_hist_prev2
    macd_rising = macd_hist > macd_hist_prev and macd_hist_prev > macd_hist_prev2
    macd_curling_up = macd_hist > macd_hist_prev and macd_hist_prev <= macd_hist_prev2
    macd_bearish_cross = macd < macd_sig and df["MACD"].iloc[-2] >= df["MACD_Signal"].iloc[-2] if n > 1 else False
    macd_bullish_cross = macd > macd_sig and df["MACD"].iloc[-2] <= df["MACD_Signal"].iloc[-2] if n > 1 else False

    rsi_falling = rsi < rsi_prev
    rsi_rising = rsi > rsi_prev

    stoch_falling = fast_k < fast_k_prev
    stoch_rising = fast_k > fast_k_prev
    stoch_curling_from_oversold = fast_k > fast_k_prev and fast_k_prev < 20

    # Check for actual bounce (green candle after being at support)
    recent_bounce = False
    if in_support or below_support:
        # Look for reversal candle in last 3 bars
        for i in range(-3, 0):
            if n + i >= 1:
                c = df["Close"].iloc[i]
                o = df["Open"].iloc[i]
                if c > o and c > df["Close"].iloc[i - 1]:
                    recent_bounce = True
                    break

    risk_flags = []

    # ===== STRICT CLASSIFICATION LOGIC =====

    # RULE 1: Below BOTH zones = BEARISH (breakdown)
    if below_support and below_resistance:
        return {
            "label": "BEARISH",
            "setup_type": "breakdown",
            "confidence": 0.80,
            "risk_flags": ["below_both_zones", "breakdown"],
        }

    # RULE 2: MACD bearish cross = BEARISH
    if macd_bearish_cross:
        risk_flags.append("macd_bearish_cross")
        if below_avwap and below_sma20:
            return {
                "label": "BEARISH",
                "setup_type": "bear_trend",
                "confidence": 0.75,
                "risk_flags": risk_flags + ["below_avwap", "below_sma20"],
            }

    # RULE 3: All indicators falling = BEARISH
    all_falling = macd_falling and rsi_falling and stoch_falling
    if all_falling:
        risk_flags.append("all_indicators_falling")
        if below_avwap:
            return {
                "label": "BEARISH",
                "setup_type": "bear_trend",
                "confidence": 0.70,
                "risk_flags": risk_flags + ["below_avwap"],
            }

    # RULE 4: In support zone but no bounce = NEUTRAL (not bullish yet)
    if in_support and not recent_bounce and (macd_falling or stoch_falling):
        return {
            "label": "NEUTRAL_RANGE",
            "setup_type": "range_chop",
            "confidence": 0.50,
            "risk_flags": ["at_support_no_bounce", "indicators_still_falling"],
        }

    # RULE 5: Below AVWAP + below SMA20 + negative MACD = BEARISH
    if below_avwap and below_sma20 and macd_hist < 0 and macd_falling:
        return {
            "label": "BEARISH",
            "setup_type": "bear_trend",
            "confidence": 0.65,
            "risk_flags": ["below_avwap", "below_sma20", "macd_negative_falling"],
        }

    # RULE 6: Above BOTH zones = potential breakout (STRONG_BULLISH candidate)
    if above_support and above_resistance:
        if above_avwap and macd_rising and rsi > 50:
            return {
                "label": "STRONG_BULLISH",
                "setup_type": "breakout_above_resistance",
                "confidence": 0.80,
                "risk_flags": [],
            }
        if above_avwap:
            return {
                "label": "BULLISH",
                "setup_type": "breakout_above_resistance",
                "confidence": 0.65,
                "risk_flags": ["needs_momentum_confirmation"],
            }

    # RULE 7: Above AVWAP + MACD bullish + RSI >50 = BULLISH
    if above_avwap and above_sma20 and macd > macd_sig and rsi > 50:
        if macd_rising and stoch_rising:
            return {
                "label": "STRONG_BULLISH",
                "setup_type": "vwap_reclaim",
                "confidence": 0.75,
                "risk_flags": [],
            }
        return {
            "label": "BULLISH",
            "setup_type": "vwap_reclaim",
            "confidence": 0.60,
            "risk_flags": ["momentum_not_accelerating"],
        }

    # RULE 8: At support with actual bounce + indicators curling = WATCH_BOUNCE
    if in_support and recent_bounce and (stoch_curling_from_oversold or macd_curling_up):
        return {
            "label": "WATCH_BOUNCE",
            "setup_type": "support_bounce",
            "confidence": 0.55,
            "risk_flags": ["below_avwap"] if below_avwap else [],
        }

    # RULE 9: Improving but below AVWAP = WATCH_RECLAIM
    if below_avwap and (macd_rising or stoch_rising) and close > close_prev:
        return {
            "label": "WATCH_RECLAIM",
            "setup_type": "potential_reclaim",
            "confidence": 0.45,
            "risk_flags": ["below_avwap", "needs_confirmation"],
        }

    # DEFAULT: NEUTRAL_RANGE
    return {
        "label": "NEUTRAL_RANGE",
        "setup_type": "range_chop",
        "confidence": 0.40,
        "risk_flags": ["mixed_signals"],
    }


def run_backtest(symbols: List[str], save_results: bool = True) -> Dict:
    """Run backtest comparing hardcoded signals vs VL model classification."""
    print("=" * 70)
    print("VL Setup Classifier Backtest")
    print("=" * 70)
    print(f"Symbols: {symbols}")
    print("Model: bespoke-minichart:7b")
    print()

    generator = VLMChartGenerator()
    validator = VLChartValidator()

    results = []

    for symbol in symbols:
        print(f"\n{'=' * 50}")
        print(f"Testing: {symbol}")
        print("=" * 50)

        # Generate chart
        print("  Generating chart...", end=" ")
        chart_path = generator.generate_chart(symbol, force_regenerate=True, output_prefix="BACKTEST")

        if not chart_path:
            print("FAILED")
            results.append({"symbol": symbol, "error": "Chart generation failed"})
            continue
        print(f"OK ({chart_path.name})")

        # Get hardcoded classification
        print("  Computing hardcoded signals...", end=" ")
        import yfinance as yf

        stock = yf.Ticker(symbol)
        hist = stock.history(period="120d").tail(60)
        df = compute_indicators(hist)
        events = detect_events(df)
        hardcoded = get_hardcoded_classification(df, events)
        print(f"OK ({hardcoded['label']}, {hardcoded['confidence']:.0%})")

        # Get VL model classification
        print("  Running VL model...", end=" ")
        start_time = time.time()
        vl_result = validator.validate_chart(chart_path, symbol)
        vl_time = time.time() - start_time

        if not vl_result.get("success", False):
            print(f"FAILED ({vl_result.get('error', 'unknown')})")
            results.append({
                "symbol": symbol,
                "hardcoded": hardcoded,
                "vl_error": vl_result.get("error"),
            })
            continue

        print(f"OK ({vl_result.get('label', 'N/A')}, {vl_time:.1f}s)")

        # Compare
        match = hardcoded["label"] == vl_result.get("label", "")

        result = {
            "symbol": symbol,
            "chart_path": str(chart_path),
            "hardcoded": hardcoded,
            "vl_result": {
                "label": vl_result.get("label"),
                "setup_type": vl_result.get("setup_type"),
                "confidence": vl_result.get("confidence"),
                "confidence_score": vl_result.get("confidence_score"),
                "signals_detected": vl_result.get("signals_detected", []),
                "key_evidence": vl_result.get("key_evidence", []),
                "triggers_next": vl_result.get("triggers_next", []),
                "risk_flags": vl_result.get("risk_flags", []),
                "recommendation": vl_result.get("recommendation"),
            },
            "vl_time": vl_time,
            "label_match": match,
        }
        results.append(result)

        # Print comparison
        print("\n  Comparison:")
        print(f"    Hardcoded: {hardcoded['label']} ({hardcoded['setup_type']}, {hardcoded['confidence']:.0%})")
        print(
            f"    VL Model:  {vl_result.get('label', 'N/A')} "
            f"({vl_result.get('setup_type', 'N/A')}, {vl_result.get('confidence_score', 0):.0%})"
        )
        print(f"    Match: {'✓' if match else '✗'}")

        # VL model's reasoning (to debug hallucinations)
        print("\n    VL MODEL SEES:")
        if vl_result.get("price_position"):
            pp = vl_result["price_position"]
            print(f"      vs_avwap: {pp.get('vs_avwap', '?')}, vs_green: {pp.get('vs_green_zone', '?')}, vs_red: {pp.get('vs_red_zone', '?')}")
        if vl_result.get("indicator_direction"):
            ind = vl_result["indicator_direction"]
            print(f"      MACD: {ind.get('macd', '?')}, RSI: {ind.get('rsi', '?')}, STOCH: {ind.get('stoch', '?')}")
        if vl_result.get("key_evidence"):
            print("      Evidence:")
            for ev in vl_result.get("key_evidence", [])[:3]:
                print(f"        - {ev[:70]}...")
        if vl_result.get("why_this_label"):
            print(f"      Why: {vl_result.get('why_this_label', '')[:80]}...")
        if vl_result.get("risk_flags"):
            print(f"      Risk Flags: {', '.join(str(f) for f in vl_result['risk_flags'][:3])}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    valid_results = [r for r in results if "vl_result" in r]
    matches = sum(1 for r in valid_results if r.get("label_match", False))
    total = len(valid_results)

    print(f"Total tested: {len(symbols)}")
    print(f"Successful VL calls: {total}")
    accuracy_pct = (matches / total * 100) if total else 0
    print(f"Label matches: {matches}/{total} ({accuracy_pct:.0f}%)")

    # Label distribution
    hardcoded_labels = {}
    vl_labels = {}
    for r in valid_results:
        h_label = r["hardcoded"]["label"]
        v_label = r["vl_result"]["label"]
        hardcoded_labels[h_label] = hardcoded_labels.get(h_label, 0) + 1
        vl_labels[v_label] = vl_labels.get(v_label, 0) + 1

    print(f"\nHardcoded distribution: {hardcoded_labels}")
    print(f"VL Model distribution: {vl_labels}")

    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = Path(__file__).parent.parent / "logs" / f"vl_backtest_{timestamp}.json"

        with open(results_path, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "symbols": symbols,
                    "summary": {
                        "total": len(symbols),
                        "successful": total,
                        "matches": matches,
                        "accuracy": matches / total if total else 0,
                    },
                    "results": results,
                },
                f,
                indent=2,
                default=str,
            )

        print(f"\nResults saved: {results_path}")

    return {
        "results": results,
        "accuracy": matches / total if total else 0,
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        symbols = [s.upper() for s in sys.argv[1:]]
    else:
        symbols = TEST_SYMBOLS[:5]

    run_backtest(symbols)
