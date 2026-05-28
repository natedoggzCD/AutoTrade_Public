import json
from types import SimpleNamespace
from pathlib import Path

from autotrade.core.claw_command_watcher import CommandWatcher


def _build_watcher(tmp_path: Path) -> CommandWatcher:
    config = SimpleNamespace(claw_remote=SimpleNamespace(command_dir="data/claw_commands"))
    service_health = SimpleNamespace(get_overall_status=lambda: "OK")
    journal = SimpleNamespace()
    return CommandWatcher(
        service_health=service_health,
        journal=journal,
        config=config,
        project_root=tmp_path,
    )


def test_execute_command_returns_top_signal_brief(tmp_path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": "2026-04-16",
        "generated_at": "2026-04-16T08:30:43",
        "total_signals": 3,
        "signals": [
            {
                "symbol": "AAA",
                "recommendation": "BUY",
                "confidence": 91.0,
                "entry_price": 10.0,
                "stop_loss": 9.0,
                "target": 12.0,
                "ranking_score": 100.0,
                "final_score": 80.0,
                "technical_score": 55.0,
                "conviction_priority_score": 88.0,
                "overnight_actionability_score": 77.0,
                "rsi_14": 54.0,
                "atr_percent": 4.2,
                "volume_ratio": 2.1,
                "sector": "Industrials",
                "catalyst_note": "earnings tomorrow",
            },
            {
                "symbol": "BBB",
                "recommendation": "BUY",
                "confidence": 78.0,
                "entry_price": 20.0,
                "stop_loss": 18.0,
                "target": 24.0,
                "ranking_score": 90.0,
                "final_score": 72.0,
                "technical_score": 50.0,
                "conviction_priority_score": 80.0,
                "overnight_actionability_score": 70.0,
                "rsi_14": 48.0,
                "atr_percent": 3.1,
                "volume_ratio": 1.5,
                "sector": "Tech",
                "catalyst_note": "",
            },
            {
                "symbol": "CCC",
                "recommendation": "WATCH",
                "confidence": 66.0,
                "entry_price": 30.0,
                "stop_loss": 27.0,
                "target": 36.0,
                "ranking_score": 50.0,
                "final_score": 60.0,
                "technical_score": 40.0,
                "conviction_priority_score": 60.0,
                "overnight_actionability_score": 55.0,
                "rsi_14": 42.0,
                "atr_percent": 2.1,
                "volume_ratio": 1.1,
                "sector": "Energy",
                "catalyst_note": "",
            },
        ],
    }
    (logs_dir / "signals_2026-04-16.json").write_text(json.dumps(payload), encoding="utf-8")

    watcher = _build_watcher(tmp_path)
    result = watcher._execute_command(
        {
            "id": "cmd-1",
            "intent": {"target_agent": "SIGNALS", "params": {"limit": 2, "as_of": "2026-04-16"}},
        }
    )

    assert result["status"] == "ok"
    assert result["result_type"] == "signal_brief"
    assert result["summary"]["returned"] == 2
    assert [signal["symbol"] for signal in result["signals"]] == ["AAA", "BBB"]
    assert result["signals"][0]["sparkline"]
