from pathlib import Path
from types import SimpleNamespace

from autotrade.core.autonomous_agent import OvernightResearchEngine


def test_generate_chart_restores_file_output_without_vl(tmp_path, monkeypatch):
    engine = OvernightResearchEngine.__new__(OvernightResearchEngine)
    engine.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )

    chart_path = tmp_path / "charts" / "AAA_20260419_VLM.png"
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    chart_path.write_bytes(b"fake chart")

    registered = {}

    def _register_chart(symbol, path, vl_analysis=None):
        registered["symbol"] = symbol
        registered["path"] = Path(path)
        registered["vl_analysis"] = vl_analysis

    engine._register_chart = _register_chart

    monkeypatch.setattr(
        "autotrade.core.autonomous_agent.generate_vlm_chart",
        lambda symbol, days=60, sr_data=None, force=False, prefix="": chart_path,
    )

    result_path, vl_analysis = engine._generate_chart("AAA")

    assert result_path == chart_path
    assert vl_analysis is None
    assert registered["symbol"] == "AAA"
    assert registered["path"] == chart_path
    assert registered["vl_analysis"] is None
