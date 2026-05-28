import json
from pathlib import Path

from autotrade.replay.decision_claw_overnight_hold_eval import (
    DecisionClawOvernightHoldEvaluator,
    _derive_risk_reward,
    _next_trading_day,
    _resolve_benchmark_set,
    _select_dates,
    _score_hold_outcome,
)


def test_next_trading_day_skips_weekend():
    assert _next_trading_day("2026-04-03") == "2026-04-06"


def test_score_hold_outcome_rewards_good_hold_and_trim():
    good_hold = _score_hold_outcome(
        approved_hold=True,
        next_close_return_pct=2.2,
        first_hour_min_drawdown_pct=-1.1,
    )
    assert good_hold["decision_quality"] == "correct_hold"
    assert good_hold["score"] == 1.0

    good_trim = _score_hold_outcome(
        approved_hold=False,
        next_close_return_pct=-1.8,
        first_hour_min_drawdown_pct=-3.4,
    )
    assert good_trim["decision_quality"] == "correct_trim"
    assert good_trim["score"] == 1.0


def test_derive_risk_reward_fills_missing_value_from_prices():
    assert (
        _derive_risk_reward(
            current_price=43.31,
            target_price=47.5923,
            stop_price=40.4858,
            raw_value=0.0,
        )
        > 1.5
    )


def test_overnight_hold_evaluator_builds_candidate_rows(tmp_path: Path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "plans").mkdir()

    (tmp_path / "data" / "eod_review_2026-04-01.json").write_text(
        json.dumps(
            {
                "trades": [
                    {
                        "symbol": "APA",
                        "qty": 45,
                        "avg_entry_price": 98.0,
                        "current_price": 100.0,
                        "unrealized_plpc": 0.05,
                    },
                    {
                        "symbol": "WEAK",
                        "qty": 20,
                        "avg_entry_price": 50.0,
                        "current_price": 49.0,
                        "unrealized_plpc": -0.02,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "logs" / "signals_2026-04-01.json").write_text(
        json.dumps(
            [
                {
                    "symbol": "APA",
                    "target": 108.0,
                    "stop_loss": 96.0,
                    "risk_reward": 2.0,
                    "relative_strength": 1.7,
                    "atr_14": 3.5,
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "autotrade.replay.decision_claw_overnight_hold_eval.fetch_symbol_minute_bars",
        lambda *args, **kwargs: __import__("pandas").DataFrame(
            {
                "open": [101.0, 102.0],
                "high": [103.0, 104.0],
                "low": [100.0, 101.0],
                "close": [102.0, 103.0],
            },
            index=__import__("pandas").to_datetime(
                ["2026-04-02 14:30:00+00:00", "2026-04-02 15:30:00+00:00"]
            ),
        ),
    )

    evaluator = DecisionClawOvernightHoldEvaluator(project_dir=tmp_path)
    scenario = evaluator.build_scenario(session_date="2026-04-01")

    assert scenario.next_session_date == "2026-04-02"
    assert len(scenario.candidates) == 1
    assert scenario.candidates[0]["symbol"] == "APA"
    assert scenario.payload["wind_down_oversized_winners"][0]["symbol"] == "APA"


def test_overnight_hold_evaluator_derives_missing_risk_reward(tmp_path: Path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "plans").mkdir()

    (tmp_path / "data" / "eod_review_2026-03-24.json").write_text(
        json.dumps(
            {
                "trades": [
                    {
                        "symbol": "BTSG",
                        "qty": 99,
                        "current_price": 43.31,
                        "market_value": 4287.69,
                        "unrealized_plpc": 0.00807,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "logs" / "signals_2026-03-24.json").write_text(
        json.dumps(
            [
                {
                    "symbol": "BTSG",
                    "target": 47.5923,
                    "stop_loss": 40.4858,
                    "risk_reward": 0.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "autotrade.replay.decision_claw_overnight_hold_eval.fetch_symbol_minute_bars",
        lambda *args, **kwargs: __import__("pandas").DataFrame(
            {
                "open": [43.81, 44.10],
                "high": [44.56, 44.85],
                "low": [43.735, 43.90],
                "close": [44.4696, 44.85],
            },
            index=__import__("pandas").to_datetime(
                ["2026-03-25 14:30:00+00:00", "2026-03-25 15:30:00+00:00"]
            ),
        ),
    )

    evaluator = DecisionClawOvernightHoldEvaluator(project_dir=tmp_path)
    scenario = evaluator.build_scenario(session_date="2026-03-24")

    assert scenario.candidates[0]["risk_reward"] > 1.5
    assert scenario.candidates[0]["overnight_hold_bias"] == "favorable"


def test_select_dates_uses_available_range(tmp_path: Path):
    (tmp_path / "data").mkdir()
    for value in ("2026-03-19", "2026-03-20", "2026-03-24", "2026-04-01"):
        (tmp_path / "data" / f"eod_review_{value}.json").write_text("{}", encoding="utf-8")

    selected = _select_dates(
        project_dir=tmp_path,
        raw_dates=[],
        start_date="2026-03-20",
        end_date="2026-03-24",
    )

    assert selected == ["2026-03-20", "2026-03-24"]


def test_resolve_benchmark_set_recent_two_weeks_uses_latest_available_dates(tmp_path: Path):
    (tmp_path / "data").mkdir()
    values = [
        "2026-03-08",
        "2026-03-09",
        "2026-03-10",
        "2026-03-11",
        "2026-03-12",
        "2026-03-13",
        "2026-03-15",
        "2026-03-16",
        "2026-03-17",
        "2026-03-18",
        "2026-03-19",
        "2026-03-20",
        "2026-03-22",
        "2026-03-23",
    ]
    for value in values:
        (tmp_path / "data" / f"eod_review_{value}.json").write_text("{}", encoding="utf-8")

    selected = _resolve_benchmark_set(
        project_dir=tmp_path,
        benchmark_set="recent_two_weeks",
    )

    assert selected == values[-12:]


def test_overnight_hold_evaluator_scores_review(tmp_path: Path, monkeypatch):
    evaluator = DecisionClawOvernightHoldEvaluator(project_dir=tmp_path)

    class _Result:
        decision = "manage_positions"
        confidence = 0.8
        reasoning_summary = "approve APA"
        actions = [
            type(
                "_Action",
                (),
                {
                    "__dict__": {
                        "action_type": "hold_position",
                        "symbol": "APA",
                        "reason": "carry",
                        "metadata": {
                            "approve_overnight_oversize": True,
                            "max_size_multiplier": 2.0,
                        },
                    },
                    "action_type": "hold_position",
                    "symbol": "APA",
                    "reason": "carry",
                    "metadata": {
                        "approve_overnight_oversize": True,
                        "max_size_multiplier": 2.0,
                    },
                },
            )()
        ]

    monkeypatch.setattr(
        evaluator,
        "build_scenario",
        lambda session_date: type(
            "_Scenario",
            (),
            {
                "session_date": session_date,
                "next_session_date": "2026-04-02",
                "payload": {"wind_down_oversized_winners": [{"symbol": "APA"}]},
                "legacy_recommendation": {"legacy_choice": []},
                "candidates": [
                    {
                        "symbol": "APA",
                        "position_size_multiple": 2.25,
                        "pnl_pct": 5.0,
                        "next_day_metrics": {
                            "available": True,
                            "next_close_return_pct": 2.0,
                            "first_hour_min_drawdown_pct": -1.0,
                        },
                    }
                ],
            },
        )(),
    )
    monkeypatch.setattr(evaluator.controller, "review", lambda **kwargs: _Result())

    report = evaluator.evaluate_dates(dates=["2026-04-01"], persist=False)

    assert report["summary"]["approved_holds_total"] == 1
    assert report["summary"]["decisive_accuracy"] == 1.0
    assert (
        report["dates"][0]["candidates"][0]["evaluation"]["decision_quality"]
        == "correct_hold"
    )
