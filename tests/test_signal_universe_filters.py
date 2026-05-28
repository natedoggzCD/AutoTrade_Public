import json
import logging
from types import SimpleNamespace

from autotrade.signals.agentic_signal_generator import AgenticSignalGenerator
from autotrade.signals.overnight_signal_rebuild import (
    select_constrained_overnight_candidates,
)
from autotrade.signals.universe_filters import (
    HARD_BLOCK_MEGA_CAP,
    signal_universe_rejection_reason,
)


def _row(symbol: str, **overrides):
    base = {
        "symbol": symbol,
        "ranking_score": 75,
        "confidence": 75,
        "overnight_expected_open_to_high_pct": 2.5,
        "overnight_expected_profit_proxy": 1.7,
        "overnight_expected_open_to_close_pct": 0.5,
        "overnight_stickiness_score": 0.5,
        "overnight_hit_2pct_prob": 0.5,
        "overnight_actionability_score": 70,
        "overnight_bad_close_prob": 0.05,
        "overnight_trap_risk": 0.1,
        "overnight_fade_risk": 0.1,
        "overnight_execution_intent": "hold_candidate",
        "setup_type": "pullback_support",
        "market_cap": 5_000_000_000,
        "volume_ratio": 1.2,
        "High": 10.5,
        "Low": 9.8,
        "Close": 10.0,
        "atr_14": 0.5,
    }
    base.update(overrides)
    return base


def test_hard_block_mega_cap_list_contains_audited_leaks():
    assert {"SNOW", "PANW", "MRVL", "VEEV", "WDAY", "ZS", "OXY"}.issubset(
        HARD_BLOCK_MEGA_CAP
    )
    assert signal_universe_rejection_reason(_row("SNOW")) == "hard_block_mega_cap"


def test_signal_universe_filter_rejects_post_spike_long_candidate():
    row = _row("FGI", volume_ratio=6.0, High=14.0, Low=10.0, Close=12.0, atr_14=1.0)

    assert signal_universe_rejection_reason(row) == "post_spike_long_exclusion"


def test_overnight_selector_excludes_post_spike_longs_before_ranking():
    rows = [
        _row(
            "FGI",
            ranking_score=99,
            volume_ratio=6.0,
            High=14.0,
            Low=10.0,
            Close=12.0,
            atr_14=1.0,
        ),
        _row("KEEP", ranking_score=70),
    ]

    selected, diagnostics = select_constrained_overnight_candidates(rows, top_n=2)

    assert [row["symbol"] for row in selected] == ["KEEP"]
    assert diagnostics["post_spike_skipped"] == 1


def test_agentic_save_signals_filters_mega_cap_and_post_spike(tmp_path, monkeypatch):
    monkeypatch.setattr("autotrade.signals.agentic_signal_generator.LOG_DIR", tmp_path)
    generator = object.__new__(AgenticSignalGenerator)
    generator.logger = logging.getLogger("test_agentic_save_signals_filters")
    generator.config = SimpleNamespace(
        universe_scanner=SimpleNamespace(
            min_market_cap=2_000_000_000,
            max_market_cap=10_000_000_000,
            post_spike_volume_threshold=5.0,
            post_spike_range_atr_threshold=1.5,
        )
    )
    rows = [
        _row("SNOW", ticker="SNOW"),
        _row(
            "FGI",
            ticker="FGI",
            volume_ratio=6.0,
            High=14.0,
            Low=10.0,
            Close=12.0,
            atr_14=1.0,
        ),
        _row("KEEP", ticker="KEEP", ranking_score=102.0, final_score=72.5),
    ]

    path = generator.save_signals(
        rows,
        target_date="2026-05-20",
        allow_overwrite=True,
        min_count=1,
        enforce_filters=True,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [row["symbol"] for row in payload["signals"]] == ["KEEP"]
    assert payload["signals"][0]["final_score"] == 72.5
    assert payload["signals"][0]["ranking_score"] == 72.5
    assert payload["signal_manifest"]["filtered_out_total"] == 2
