import argparse
import base64
import json
import re
from pathlib import Path
from typing import Dict, List

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "bespoke-minichart:7b"
REQUIRED_FIELDS = [
    "pattern_primary",
    "pattern_secondary",
    "trend",
    "trend_strength",
    "direction",
    "confidence",
    "entry_quality",
    "signal_validation",
    "visual_evidence",
    "alt_confirmation",
    "invalidating_evidence",
    "recommendation",
    "reasoning",
]

PROMPT_TEMPLATE = """You are a strict technical analyst. Use only what you can SEE in the chart.
The chart includes candles, SMA20 (blue), SMA50 (orange), volume bars, and RSI panel.

TASK (2-PASS VALIDATION):
1) Identify the PRIMARY pattern and current trend.
2) Confirm or invalidate the signal using at least TWO independent visual checks:
   - Swing structure (higher highs/higher lows or lower highs/lower lows)
   - Moving average alignment and slope (SMA20 vs SMA50)
   - RSI regime or divergence
   - Volume behavior around recent moves

If evidence conflicts or is unclear, set direction to NEUTRAL and signal_validation to UNCONFIRMED.
Do NOT guess. Be conservative.

RESPOND IN JSON ONLY:
{
  "pattern_primary": "...",
  "pattern_secondary": "...",
  "trend": "UPTREND | DOWNTREND | SIDEWAYS",
  "trend_strength": "STRONG | MODERATE | WEAK",
  "direction": "BULLISH | BEARISH | NEUTRAL",
  "confidence": "HIGH | MEDIUM | LOW",
  "entry_quality": "EXCELLENT | GOOD | FAIR | POOR",
  "signal_validation": "CONFIRMED | UNCONFIRMED | CONTRADICTED",
  "visual_evidence": ["...", "..."],
  "alt_confirmation": ["...", "..."],
  "invalidating_evidence": ["..."],
  "recommendation": "BUY | HOLD | AVOID | SELL",
  "reasoning": "one sentence"
}
"""


def _find_latest_chart(charts_dir: Path) -> Path:
    if not charts_dir.exists():
        raise FileNotFoundError(f"Charts directory not found: {charts_dir}")
    pngs = list(charts_dir.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"No chart images found in {charts_dir}")
    return max(pngs, key=lambda p: p.stat().st_mtime)


def _infer_symbol(chart_path: Path) -> str:
    name = chart_path.stem
    if "_" in name:
        return name.split("_")[0].upper()
    return name.upper()


def _encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _extract_json(content: str) -> Dict:
    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        raise ValueError("No JSON object found in model response")
    return json.loads(match.group())


def _validate_fields(payload: Dict, required: List[str]) -> List[str]:
    missing = [k for k in required if k not in payload]
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run VL prompt candidate against a chart image")
    parser.add_argument("--chart", type=str, help="Path to chart image (PNG)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama VL model")
    parser.add_argument("--skip-model", action="store_true", help="Skip calling the model")
    args = parser.parse_args()

    charts_dir = Path(__file__).resolve().parents[1] / "charts"
    chart_path = Path(args.chart) if args.chart else _find_latest_chart(charts_dir)
    if not chart_path.exists():
        raise FileNotFoundError(f"Chart not found: {chart_path}")

    symbol = _infer_symbol(chart_path)
    prompt = PROMPT_TEMPLATE.replace("{symbol}", symbol)

    print(f"Chart: {chart_path}")
    print(f"Symbol: {symbol}")
    print(f"Model: {args.model}")

    if args.skip_model:
        print("SKIP: model call disabled (--skip-model)")
        return 0

    image_b64 = _encode_image(chart_path)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": args.model,
                "messages": [
                    {"role": "user", "content": prompt, "images": [image_b64]}
                ],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 512},
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"SKIP: model request failed ({exc})")
        return 0

    content = response.json().get("message", {}).get("content", "")
    if not content:
        print("FAIL: empty response from model")
        return 1

    try:
        payload = _extract_json(content)
    except Exception as exc:
        print(f"FAIL: could not parse JSON ({exc})")
        print(content[:500])
        return 1

    missing = _validate_fields(payload, REQUIRED_FIELDS)
    if missing:
        print(f"FAIL: missing fields: {missing}")
        print(json.dumps(payload, indent=2)[:800])
        return 1

    print("PASS: required fields present")
    print("Summary:")
    print(f"  direction={payload.get('direction')} confidence={payload.get('confidence')} validation={payload.get('signal_validation')}")
    print(f"  pattern_primary={payload.get('pattern_primary')} pattern_secondary={payload.get('pattern_secondary')}")
    print(f"  recommendation={payload.get('recommendation')} entry_quality={payload.get('entry_quality')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
