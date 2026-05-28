import json
import logging
from pathlib import Path
from types import SimpleNamespace

from autotrade.signals.agentic_signal_generator import AgenticSignalGenerator


def _load_signals(path: str = "logs/signals_2026-05-20.json") -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload.get("signals") or [])


def _float_or_none(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def test_20260520_buy_signals_saved_by_generator_have_long_side_entry_geometry(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("autotrade.signals.agentic_signal_generator.LOG_DIR", tmp_path)
    generator = object.__new__(AgenticSignalGenerator)
    generator.logger = logging.getLogger("test_signal_direction_invariants")
    generator.config = SimpleNamespace(
        universe_scanner=SimpleNamespace(
            min_market_cap=2_000_000_000,
            max_market_cap=10_000_000_000,
            post_spike_volume_threshold=99.0,
            post_spike_range_atr_threshold=99.0,
        )
    )
    path = generator.save_signals(
        _load_signals(),
        target_date="2026-05-20",
        allow_overwrite=True,
        min_count=1,
        enforce_filters=True,
    )
    rows = json.loads(path.read_text(encoding="utf-8"))["signals"]

    malformed = []
    for row in rows:
        recommendation = str(row.get("recommendation") or "").upper()
        if "BUY" not in recommendation or "SHORT" in recommendation:
            continue
        entry = _float_or_none(row.get("entry_price") or row.get("price"))
        stop = _float_or_none(row.get("stop_loss") or row.get("stop"))
        target = _float_or_none(row.get("target") or row.get("target_price"))
        if not entry or not stop or not target or not (stop < entry < target):
            malformed.append(row.get("symbol"))

    assert malformed == []


def test_short_style_long_input_without_atr_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("autotrade.signals.agentic_signal_generator.LOG_DIR", tmp_path)
    generator = object.__new__(AgenticSignalGenerator)
    generator.logger = logging.getLogger("test_signal_direction_invariants")
    generator.config = SimpleNamespace(
        universe_scanner=SimpleNamespace(
            min_market_cap=2_000_000_000,
            max_market_cap=10_000_000_000,
            post_spike_volume_threshold=99.0,
            post_spike_range_atr_threshold=99.0,
        )
    )

    path = generator.save_signals(
        [
            {
                "symbol": "SHORTSTYLE",
                "ticker": "SHORTSTYLE",
                "recommendation": "BUY",
                "entry_price": 100,
                "stop_loss": 50,
                "target": 60,
                "market_cap": 5_000_000_000,
            },
            {
                "symbol": "KEEP",
                "ticker": "KEEP",
                "recommendation": "BUY",
                "entry_price": 100,
                "stop_loss": 95,
                "target": 110,
                "market_cap": 5_000_000_000,
            },
        ],
        target_date="2026-05-20",
        allow_overwrite=True,
        min_count=1,
        enforce_filters=True,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [row["symbol"] for row in payload["signals"]] == ["KEEP"]
    assert payload["signal_manifest"]["malformed_geometry_filtered_total"] == 1
