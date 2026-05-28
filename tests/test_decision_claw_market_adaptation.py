import json
from pathlib import Path

from autotrade.core.decision_claw import DecisionClaw
from config.config_loader import DecisionClawConfig


class _LoggerStub:
    def info(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


def test_decision_claw_compaction_includes_market_adaptation_context(tmp_path):
    cfg = DecisionClawConfig()
    cfg.state_path = "data/decision_claw_state.json"
    data_dir = Path(tmp_path) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "intraday_analysis.json").write_text(
        json.dumps(
            {
                "market_context": {
                    "regime_label": "DISPERSION",
                    "dispersion_score": 1.2873,
                    "spy_pct": -0.10,
                    "vix": 27.1,
                    "sizing_multiplier": 1.15,
                    "sizing_rationale": "DISPERSION score 1.29: flat index.",
                },
                "execution_diagnostics": {
                    "replacement_rejection_reasons": {"vwap_overextended": 2}
                },
                "tape_inference": "High dispersion tape: flat index with elevated VIX.",
                "anomalies": ["replacement_pressure_without_attempts"],
            }
        ),
        encoding="utf-8",
    )

    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))
    packet = claw._compact_payload(
        {
            "phase": "market_state",
            "candidate_count": 1,
            "candidates": [{"symbol": "GLNG", "score": 81.0}],
        },
        phase_agent="market_state",
        phase_cfg=cfg.market_state,
    )

    adaptation = packet["market_adaptation"]
    assert adaptation["regime_label"] == "DISPERSION"
    assert adaptation["sizing_multiplier"] > 1.0
    assert adaptation["replacement_rejection_reasons"]["vwap_overextended"] == 2
