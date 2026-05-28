"""Compare VL models on all 8 test symbols."""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from autotrade.utils.vl_chart_validator import VLChartValidator
from autotrade.utils.vlm_chart_generator import VLMChartGenerator

SYMBOLS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "GOOGL", "AMZN"]

# User's manual analysis (ground truth)
USER_LABELS = {
    "AAPL": "BULLISH",
    "MSFT": "BEARISH",
    "NVDA": "BEARISH",
    "TSLA": "NEUTRAL",
    "AMD": "BEARISH",
    "META": "BEARISH",
    "GOOGL": "WATCH",
    "AMZN": "BEARISH",
}


def normalize_label(label):
    """Normalize labels for comparison."""
    if label in ["STRONG_BULLISH", "BULLISH"]:
        return "BULLISH"
    if label in ["WATCH_BOUNCE", "WATCH_RECLAIM"]:
        return "WATCH"
    if label == "NEUTRAL_RANGE":
        return "NEUTRAL"
    return label


def test_model(model_name: str, symbols: list):
    """Test a specific model on all symbols."""
    print(f"\n{'=' * 70}")
    print(f"TESTING MODEL: {model_name}")
    print(f"{'=' * 70}")

    validator = VLChartValidator(vl_model=model_name)
    generator = VLMChartGenerator()

    results = []

    for symbol in symbols:
        print(f"\n{symbol}: ", end="", flush=True)

        # Generate fresh chart
        chart_path = generator.generate_chart(symbol, force_regenerate=True, output_prefix="COMPARE")
        if not chart_path:
            print("CHART FAILED")
            continue

        # Validate with model
        start = time.time()
        result = validator.validate_chart(chart_path, symbol)
        elapsed = time.time() - start

        # Check if model worked
        if not result.get("success", False):
            print(f"MODEL FAILED: {result.get('error', 'unknown')}")
            results.append({
                "symbol": symbol,
                "label": "ERROR",
                "error": result.get("error"),
            })
            continue

        label = result.get("label", "?")
        conf = result.get("confidence_score", 0)
        tokens = result.get("prompt_tokens", 0)

        # Check against user's labels
        expected = USER_LABELS.get(symbol, "?")
        normalized = normalize_label(label)
        match = normalized == expected or (normalized == "BEARISH" and expected == "BEARISH")

        print(f"{label:15s} | conf={conf:.0%} | {elapsed:.1f}s | tokens={tokens} | vs expected {expected}: {'✓' if match else '✗'}")

        if result.get("key_evidence"):
            for ev in result.get("key_evidence", [])[:2]:
                print(f"   - {ev[:60]}")

        results.append({
            "symbol": symbol,
            "label": label,
            "normalized": normalized,
            "expected": expected,
            "match": match,
            "confidence": conf,
            "key_evidence": result.get("key_evidence", []),
            "prompt_tokens": tokens,
        })

    # Summary
    valid_results = [r for r in results if r.get("label") != "ERROR"]
    matches = sum(1 for r in valid_results if r.get("match", False))
    total = len(valid_results)
    errors = len([r for r in results if r.get("label") == "ERROR"])

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {model_name}")
    print(f"{'=' * 70}")
    print(f"Successful: {total}/{len(symbols)}, Errors: {errors}")
    if total > 0:
        print(f"Accuracy: {matches}/{total} ({matches / total * 100:.0f}%)")

        # Distribution
        from collections import Counter

        labels = Counter(r["label"] for r in valid_results)
        print(f"Label distribution: {dict(labels)}")

    return results


if __name__ == "__main__":
    models_to_test = ["qwen2.5vl:7b", "qwen3-vl:8b", "bespoke-minichart:7b"]

    all_results = {}

    for model in models_to_test:
        all_results[model] = test_model(model, SYMBOLS)

    # Final comparison
    print(f"\n\n{'#' * 70}")
    print("FINAL COMPARISON")
    print(f"{'#' * 70}")

    print(f"\n{'Symbol':8s} | {'Expected':10s} | {'qwen2.5vl':15s} | {'qwen3-vl':15s} | {'bespoke':15s}")
    print("-" * 80)

    for i, symbol in enumerate(SYMBOLS):
        expected = USER_LABELS.get(symbol, "?")

        labels = []
        for model in models_to_test:
            if i < len(all_results[model]):
                lbl = all_results[model][i].get("label", "?")
                match = all_results[model][i].get("match", False)
                labels.append(f"{lbl:12s} {'✓' if match else '✗'}")
            else:
                labels.append("?")

        print(f"{symbol:8s} | {expected:10s} | {' | '.join(labels)}")
