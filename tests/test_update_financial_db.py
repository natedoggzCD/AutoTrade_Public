import json
from pathlib import Path

from tools import update_financial_db


def test_load_held_tickers_reads_dict_wrapped_plan_and_signal_payloads(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    plans_dir = repo_root / "plans"
    logs_dir = repo_root / "logs"
    tools_dir = repo_root / "tools"
    plans_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    tools_dir.mkdir(parents=True)

    (tools_dir / "update_financial_db.py").write_text("# stub\n", encoding="utf-8")
    (plans_dir / "morning_game_plan_20260330.json").write_text(
        json.dumps(
            {
                "signals": [
                    {"ticker": "AAPL"},
                    {"symbol": "MSFT"},
                ],
                "buy_signals": [
                    {"symbol": "NVDA"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (logs_dir / "signals_2026-03-30.json").write_text(
        json.dumps(
            {
                "signals": [
                    {"ticker": "TSLA"},
                    {"symbol": "AAPL"},
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(update_financial_db, "__file__", str(tools_dir / "update_financial_db.py"))

    tickers = update_financial_db._load_candidate_tickers()

    assert tickers == ["AAPL", "MSFT", "NVDA", "TSLA"]


def test_load_live_held_tickers_uses_alpaca_positions(monkeypatch):
    class _Position:
        def __init__(self, symbol):
            self.symbol = symbol

    class _Client:
        def get_all_positions(self):
            return [_Position("CF"), _Position("NVDA")]

    monkeypatch.setattr(
        update_financial_db,
        "create_trading_client",
        lambda require_credentials=False: _Client(),
    )

    tickers = update_financial_db._load_live_held_tickers()

    assert tickers == ["CF", "NVDA"]
