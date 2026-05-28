from pathlib import Path
from types import SimpleNamespace

from autotrade.core.day_manager import DayManager


def test_boot_step_records_elapsed_and_delta():
    dm = DayManager.__new__(DayManager)
    dm._boot_timing = []
    dm._boot_started_perf = 10.0
    dm._boot_last_perf = 10.0

    DayManager._record_boot_step(dm, "unit_test")

    assert dm._boot_timing
    assert dm._boot_timing[-1]["step"] == "unit_test"
    assert dm._boot_timing[-1]["elapsed_sec"] >= 0.0
    assert dm._boot_timing[-1]["delta_sec"] >= 0.0


def test_signal_enrichment_cache_reuses_output_when_plan_unchanged(
    tmp_path, monkeypatch
):
    dm = DayManager.__new__(DayManager)
    dm.signal_generation_cfg = SimpleNamespace(
        day_manager_enrichment_cache_enabled=True,
        day_manager_alpha_blend_weight=0.3,
    )
    dm.signal_pipeline = object()
    dm.alpha_zoo_active = False
    dm.alpha_zoo_signal_count = 0
    dm.last_alpha_zoo_run = None

    signal_path = tmp_path / "pm_plan_2026-05-15.json"
    signal_path.write_text('{"watchlist":[]}', encoding="utf-8")
    monkeypatch.setattr(
        dm,
        "_signal_enrichment_cache_dir",
        lambda: Path(tmp_path) / "cache",
    )

    calls = {"count": 0}

    def _fake_enrich(candidates, regime=None):
        calls["count"] += 1
        enriched = [dict(row) for row in candidates]
        enriched[0]["alpha_zoo_score"] = 88.0
        enriched[0]["alpha_family"] = "ts_momentum"
        return enriched

    monkeypatch.setattr(dm, "_enrich_signals_with_alpha_zoo", _fake_enrich)

    candidates = [{"ticker": "ABCD", "score": 60.0}]
    first = DayManager._maybe_enrich_signals_with_alpha_zoo(
        dm,
        [dict(row) for row in candidates],
        signal_path=signal_path,
    )
    second = DayManager._maybe_enrich_signals_with_alpha_zoo(
        dm,
        [dict(row) for row in candidates],
        signal_path=signal_path,
    )

    assert calls["count"] == 1
    assert first == second
    assert second[0]["alpha_zoo_score"] == 88.0
    assert dm.alpha_zoo_active is True
    assert dm.alpha_zoo_signal_count == 1
