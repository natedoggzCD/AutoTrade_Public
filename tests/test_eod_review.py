import json
from types import SimpleNamespace
from unittest.mock import patch
from autotrade.core.eod_review import EODReview
from autotrade.core.overnight_agent import OvernightAgent
from autotrade.risk.position_state import PortfolioState
from autotrade.core.autonomous_agent import AutonomousAgent


def test_overnight_agent_threshold_adjustment():
    """
    Verify that OvernightAgent adjusts thresholds based on performance.
    """
    agent = OvernightAgent()
    initial_threshold = agent.min_signal_score

    # Simulate poor performance
    poor_results = {"win_rate": 0.2, "avg_pnl": -200.0}

    # Calling generate_next_day_plan with results should trigger adjustment
    portfolio = PortfolioState(cash_available=100000)

    # We'll mock the internal components to avoid actual screening
    with (
        patch.object(agent, "analyze_portfolio") as mock_analyze,
        patch.object(agent, "scan_watchlist") as mock_scan,
        patch.object(agent, "save_plan"),
    ):
        mock_analyze.return_value = {
            "positions": {},
            "rotation_candidates": [],
            "add_candidates": [],
            "critical": [],
        }
        mock_scan.return_value = []

        agent.generate_next_day_plan(portfolio, eod_results=poor_results)

        assert agent.min_signal_score > initial_threshold
        assert agent.min_signal_score == 70.0  # 65 + 5


def test_eod_review_instantiation():
    """
    Verify that EODReview can be instantiated.
    """
    review = EODReview()
    assert review is not None


def test_eod_review_run_dry_run():
    """
    Verify that EODReview.run() returns a valid structure in dry_run mode.
    """
    review = EODReview(dry_run=True)
    results = review.run()

    assert "date" in results
    assert "win_rate" in results
    assert "avg_pnl" in results
    assert "total_trades" in results
    assert "score_buckets" in results
    assert "best_trade" in results
    assert "worst_trade" in results


def test_eod_review_signal_matching():
    """
    Verify that EODReview correctly matches symbols with morning game plan scores.
    """
    # Use today's date where we know morning_game_plan_20260227.json exists
    # and contains ADEA with score 78.75
    date_str = "2026-02-27"
    review = EODReview(dry_run=True, date_str=date_str)

    results = review.run()

    # In mock data, ADEA has unrealized_pl 150.0
    adea_trade = next((t for t in results["trades"] if t["symbol"] == "ADEA"), None)
    assert adea_trade is not None
    assert adea_trade["entry_score"] == 78.75

    # Check score buckets
    assert results["score_buckets"]["65-80"]["count"] >= 1


def test_eod_review_falls_back_to_pm_plan_scores(tmp_path, monkeypatch):
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    (plans_dir / "pm_plan_2026-03-09.json").write_text(
        json.dumps(
            {
                "signals": [
                    {"symbol": "ADEA", "final_score": 71.5, "entry_source": "pm_plan"}
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    review = EODReview(dry_run=True, date_str="2026-03-09")
    matched = review._match_with_signals([{"symbol": "ADEA", "unrealized_pl": 1.0}])

    assert matched[0]["entry_score"] == 71.5
    assert matched[0]["plan_score_source"] == "pm_plan_2026-03-09.json"


def test_eod_review_falls_back_to_adjusted_plan_scores(tmp_path, monkeypatch):
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    (plans_dir / "adjusted_plan_20260310_0829.json").write_text(
        json.dumps(
            {
                "buy_signals": [
                    {"ticker": "ADEA", "confidence": 84.25, "entry_source": "adjusted"}
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    review = EODReview(dry_run=True, date_str="2026-03-10")
    matched = review._match_with_signals([{"symbol": "ADEA", "unrealized_pl": 1.0}])

    assert matched[0]["entry_score"] == 84.25
    assert matched[0]["plan_score_source"] == "adjusted_plan_20260310_0829.json"


def test_eod_review_marks_missing_scores_unavailable(tmp_path, monkeypatch):
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    review = EODReview(dry_run=True, date_str="2026-03-10")
    matched = review._match_with_signals([{"symbol": "MISS", "unrealized_pl": 1.0}])

    assert matched[0]["entry_score"] is None
    assert matched[0]["entry_score_available"] is False


def test_eod_review_separates_broker_orders_from_open_positions(monkeypatch):
    review = EODReview(dry_run=True, date_str="2026-04-29")
    review._fetch_positions = lambda: [
        {"symbol": "EQNR", "unrealized_pl": 54.0, "current_price": 40.11}
    ]
    review._match_with_signals = lambda positions: [
        {
            "symbol": "EQNR",
            "unrealized_pl": 54.0,
            "entry_score": 71.5,
            "entry_score_available": True,
        }
    ]
    review._load_watchlist_causality = lambda: {"summary": {}}
    review._summarize_missed_watchlist_opportunities = lambda: []

    class _Client:
        def get_orders(self, filter=None):
            return [
                SimpleNamespace(
                    id="buy-1",
                    symbol="EQNR",
                    side="buy",
                    status="filled",
                    qty="17",
                    filled_qty="17",
                    filled_avg_price="40.06",
                    submitted_at=None,
                    filled_at=None,
                ),
                SimpleNamespace(
                    id="sell-1",
                    symbol="EQNR",
                    side="sell",
                    status="filled",
                    qty="8",
                    filled_qty="8",
                    filled_avg_price="40.08",
                    submitted_at=None,
                    filled_at=None,
                ),
            ]

        def get_account(self):
            return SimpleNamespace(
                equity="88680.27",
                last_equity="88976.19",
                cash="84656.88",
                buying_power="300803.92",
                status="ACTIVE",
            )

    review.client = _Client()

    results = review.run()

    assert results["total_trades"] == 1
    assert results["open_positions_count"] == 1
    assert results["same_day_filled_orders_count"] == 2
    assert results["broker_orders_by_side"] == {"buy": 1, "sell": 1}
    assert results["broker_day_pnl"] == -295.92


def test_eod_review_uses_full_watchlist_when_signals_missing(tmp_path, monkeypatch):
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    (plans_dir / "pm_plan_2026-03-10.json").write_text(
        json.dumps(
            {
                "signals": [],
                "actionable_top50": [],
                "full_watchlist": [
                    {
                        "symbol": "MISS",
                        "final_score": 83.5,
                        "entry_source": "pm_report",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    review = EODReview(dry_run=True, date_str="2026-03-10")
    matched = review._match_with_signals([{"symbol": "MISS", "unrealized_pl": 1.0}])

    assert matched[0]["entry_score"] == 83.5
    assert matched[0]["entry_score_available"] is True
    assert matched[0]["score_source"] == "full_watchlist"


def test_eod_review_summarizes_missed_watchlist_opportunities(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "watchlist_causality_2026-03-10.json").write_text(
        json.dumps(
            {
                "summary": {"blocked": 2, "executed": 1},
                "symbols": [
                    {
                        "ticker": "ADEA",
                        "executed": False,
                        "was_evaluated": True,
                        "current_status": "blocked",
                        "blocking_reason": "below_min_score",
                        "blocking_rule": "below_min_score",
                        "entry_source": "overnight_plan",
                        "raw_score": 72.0,
                        "validated_score": 72.0,
                        "ranking_position": 2,
                        "displaced_by": "DISC",
                        "displaced_by_score": 77.0,
                        "missing_inputs": [],
                        "phases_seen": ["core_trading"],
                    },
                    {
                        "ticker": "DISC",
                        "executed": True,
                        "was_evaluated": True,
                        "current_status": "executed",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    review = EODReview(dry_run=True, date_str="2026-03-10")
    results = review.run()

    assert results["watchlist_causality_summary"]["blocked"] == 2
    assert results["missed_watchlist_opportunities"][0]["symbol"] == "ADEA"
    assert (
        results["missed_watchlist_opportunities"][0]["blocking_reason"]
        == "below_min_score"
    )


def test_eod_review_integration_in_agent():
    """
    Verify that AutonomousAgent triggers EODReview in run_pm_workflow.
    """
    agent = AutonomousAgent()

    # We'll mock EODReview.run to avoid actual execution
    with patch("autotrade.core.eod_review.EODReview.run") as mock_run:
        mock_run.return_value = {"win_rate": 0.8, "date": "2026-02-27"}

        # We need to mock PostMarketWorkflow and DailyReview too to avoid network calls
        with (
            patch(
                "autotrade.execution.post_market_workflow.PostMarketWorkflow.run"
            ) as mock_pm_run,
            patch("os.path.getmtime", return_value=0),
            patch("autotrade.core.daily_review.DailyReview.run") as mock_review_run,
            patch(
                "autotrade.core.daily_lessons_analyzer.DailyLessonsAnalyzer.run"
            ) as mock_lessons_run,
        ):
            mock_pm_run.return_value = {"status": "ok"}
            mock_review_run.return_value = {"review": "good"}
            mock_lessons_run.return_value = ["lesson 1"]

            result = agent.run_pm_workflow(dry_run=True)

            # Check if EOD review is in result
            assert "eod_review" in result
            assert result["eod_review"]["win_rate"] == 0.8
            assert mock_run.called
