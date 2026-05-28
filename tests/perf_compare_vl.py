"""
Fast VL Model Comparison with Performance Profiling
Tests: llama3.2-vision vs qwen2.5vl with timing breakdown
"""
import sys
import time
import base64
import json
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from autotrade.utils.vlm_chart_generator import VLMChartGenerator

SYMBOLS = ["MSFT", "AAPL", "NVDA", "TSLA"]  # Mix of bearish and one bullish


def get_image_stats(chart_path):
    """Get image size info."""
    size_kb = chart_path.stat().st_size / 1024
    with open(chart_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    return size_kb, len(img_b64), img_b64


def test_model_quick(model: str, img_b64: str, prompt: str):
    """Quick test with timing breakdown."""
    print(f"\n{'=' * 50}")
    print(f"MODEL: {model}")
    print(f"{'=' * 50}")

    # Time the request
    start = time.time()

    try:
        payload = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 256},
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=300) as response:
            data = json.loads(response.read().decode("utf-8"))

        total_time = time.time() - start

        # Timing breakdown from Ollama
        load_time = data.get("load_duration", 0) / 1e9
        prompt_time = data.get("prompt_eval_duration", 0) / 1e9
        eval_time = data.get("eval_duration", 0) / 1e9
        prompt_tokens = data.get("prompt_eval_count", 0)
        eval_tokens = data.get("eval_count", 0)

        print(f"Total time: {total_time:.1f}s")
        print(f"  - Model load: {load_time:.1f}s")
        print(f"  - Prompt eval: {prompt_time:.1f}s ({prompt_tokens} tokens)")
        print(f"  - Generation: {eval_time:.1f}s ({eval_tokens} tokens)")
        print(f"  - Tokens/sec: {eval_tokens / eval_time:.1f}" if eval_time > 0 else "")

        content = data.get("message", {}).get("content", "")

        # Extract key info
        is_bearish = "BEARISH" in content.upper()
        is_bullish = "BULLISH" in content.upper() and not is_bearish

        print(f"\nClassification: {'BEARISH' if is_bearish else 'BULLISH' if is_bullish else 'UNCLEAR'}")
        print(f"Response preview: {content[:200]}...")

        return {
            "model": model,
            "total_time": total_time,
            "prompt_tokens": prompt_tokens,
            "classification": "BEARISH" if is_bearish else "BULLISH" if is_bullish else "UNCLEAR",
            "content": content,
        }

    except urllib.error.URLError as e:
        print(f"TIMEOUT/ERROR: {e}")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def main():
    # Generate charts
    print("Generating charts...")
    gen = VLMChartGenerator()

    models = ["qwen2.5vl:7b", "llama3.2-vision:latest"]
    all_results = []

    for symbol in SYMBOLS:
        chart_path = gen.generate_chart(symbol, force_regenerate=False, output_prefix="PERF")

        # Get image stats
        size_kb, b64_len, img_b64 = get_image_stats(chart_path)
        print(f"\n{'#' * 60}")
        print(f"# {symbol} - {size_kb:.0f}KB")
        print(f"{'#' * 60}")

        # Simple prompt
        prompt = (
            f"Look at this {symbol} chart. Is it BULLISH or BEARISH?\n\n"
            "Check:\n"
            "1. Is price above or below the purple AVWAP line?\n"
            "2. Is MACD histogram red or green?\n"
            "3. Is RSI above or below 50?\n\n"
            "Answer: [BULLISH/BEARISH] because [one sentence reason]"
        )

        for model in models:
            result = test_model_quick(model, img_b64, prompt)
            if result:
                result["symbol"] = symbol
                all_results.append(result)

    # Summary
    print(f"\n{'=' * 70}")
    print("FULL COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Symbol':<8} | {'Model':<25} | {'Time':>6} | Classification")
    print("-" * 70)
    for r in all_results:
        print(f"{r['symbol']:<8} | {r['model']:<25} | {r['total_time']:>5.1f}s | {r['classification']}")

    # Model accuracy summary
    print(f"\n{'=' * 70}")
    print("MODEL SUMMARY")
    print(f"{'=' * 70}")
    for model in models:
        model_results = [r for r in all_results if r["model"] == model]
        avg_time = sum(r["total_time"] for r in model_results) / len(model_results)
        classifications = [r["classification"] for r in model_results]
        print(f"{model}: avg {avg_time:.1f}s | {classifications}")


if __name__ == "__main__":
    main()
