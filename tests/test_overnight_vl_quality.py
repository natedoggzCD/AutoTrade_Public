from types import SimpleNamespace

from autotrade.core.autonomous_agent import AutonomousAgent, OvernightResearchEngine


def _make_engine() -> OvernightResearchEngine:
    engine = OvernightResearchEngine.__new__(OvernightResearchEngine)
    return engine


def _make_agent() -> AutonomousAgent:
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.overnight_engine = SimpleNamespace(
        vl_min_schema_pass_rate=0.85,
        vl_min_unique_combos=6,
        vl_max_dominant_combo_ratio=0.70,
        vl_max_duplicate_reasoning_ratio=0.75,
        vl_max_stage_runtime_seconds=180,
    )
    return agent


def test_normalize_vl_analysis_payload_valid():
    engine = _make_engine()
    normalized, errors = engine._normalize_vl_analysis_payload(
        {
            "trend": "bullish",
            "pattern": "bull flag",
            "entry_signal": "bounce",
            "recommendation": "buy",
            "confidence": 85,
            "reasoning": "Price holds above VWAP with bullish MACD crossover and support bounce.",
            "macd_status": "bullish_cross",
            "rsi_status": "oversold",
            "vwap_position": "above",
        }
    )
    assert errors == []
    assert normalized["trend"] == "bullish"
    assert normalized["entry_signal"] == "bounce"
    assert normalized["recommendation"] == "buy"
    assert normalized["confidence"] == 85.0
    assert normalized["macd_status"] == "bullish_cross"
    assert normalized["rsi_status"] == "oversold"
    assert normalized["vwap_position"] == "above"


def test_normalize_vl_analysis_payload_invalid():
    engine = _make_engine()
    normalized, errors = engine._normalize_vl_analysis_payload(
        {
            "trend": "mystery",
            "entry_signal": "something odd",
            "recommendation": "unknown",
            "confidence": "n/a",
            "reasoning": "short",
        }
    )
    assert len(errors) >= 4
    assert normalized["trend"] == "neutral"
    assert normalized["entry_signal"] == "none"
    assert normalized["recommendation"] == "wait"
    assert normalized["confidence"] == 50.0


def test_vl_quality_quarantines_collapsed_outputs():
    agent = _make_agent()
    rows = []
    for idx in range(50):
        rows.append(
            {
                "symbol": f"SYM{idx}",
                "vl_analysis": {
                    "analyzed": True,
                    "schema_valid": True,
                    "trend": "bullish",
                    "pattern": "bull flag",
                    "entry_signal": "bounce",
                    "recommendation": "buy",
                    "confidence": 85,
                    "reasoning": "Price above VWAP with MACD crossover and EMA support bounce.",
                },
            }
        )
    quality = agent._evaluate_top50_vl_quality(rows, processed=50, stage_runtime_seconds=40.0)
    assert quality["vl_reliable"] is False
    assert quality["unique_combo_count"] == 1
    assert "low_unique_combos" in quality["quarantine_reason"] or "dominant_combo_ratio_too_high" in quality["quarantine_reason"]


def test_vl_quality_passes_diverse_valid_outputs():
    agent = _make_agent()
    rows = []
    combos = [
        ("bullish", "bull flag", "bounce", "buy", 85, "Bull setup off EMA20 with strengthening momentum."),
        ("bullish", "ascending triangle", "breakout", "buy", 90, "Coiled range with breakout and volume expansion."),
        ("neutral", "tight consolidation", "none", "wait", 55, "Range-bound structure with mixed momentum."),
        ("bearish", "double top", "reversal", "sell", 75, "Rejection at resistance with bearish MACD turn."),
        ("bullish", "cup-handle", "breakout", "buy", 88, "Handle resolved upward above VWAP and rising RSI."),
        ("bearish", "bear flag", "momentum", "sell", 80, "Lower highs and downside momentum continuation."),
    ]
    for idx in range(48):
        trend, pattern, entry, rec, conf, reason = combos[idx % len(combos)]
        rows.append(
            {
                "symbol": f"SYM{idx}",
                "vl_analysis": {
                    "analyzed": True,
                    "schema_valid": True,
                    "trend": trend,
                    "pattern": pattern,
                    "entry_signal": entry,
                    "recommendation": rec,
                    "confidence": conf,
                    "reasoning": f"{reason} id={idx}",
                },
            }
        )
    quality = agent._evaluate_top50_vl_quality(rows, processed=48, stage_runtime_seconds=65.0)
    assert quality["vl_reliable"] is True
    assert quality["schema_pass_rate"] >= 0.85
    assert quality["unique_combo_count"] >= 6
