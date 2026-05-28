"""
Daily Review - Lightweight Post-Market Performance Analysis
============================================================
Runs at market close to answer:
  1. How did my positions perform today?
  2. How did stocks on my watchlist perform?
  3. Did the workflow execute properly today?

This is a QUICK review, NOT the heavy overnight research.
The heavy scanning/signal generation happens in the Overnight Research Cycle.

Usage:
  python -m autotrade.core.daily_review
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import autotrade.signals.trade_learner as trade_learner
from autotrade.utils.market_time import get_market_now, last_trading_day
from autotrade.utils.alpaca_client_factory import (
    create_data_client,
    create_trading_client,
    resolve_alpaca_credentials,
)
from autotrade.utils.daily_learning_state import build_learning_state

# Try to import ReportingEngine for release gate status
try:
    from autotrade.monitoring.reporting import get_reporting_engine

    REPORTING_AVAILABLE = True
except ImportError:
    REPORTING_AVAILABLE = False

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = Path(
    os.environ.get("AUTOTRADE_ROOT", Path(__file__).resolve().parents[2])
)
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
PLANS_DIR = PROJECT_DIR / "plans"
PLANS_DIR.mkdir(exist_ok=True)

# Import safe logging utilities
try:
    from autotrade.utils.safe_logging import get_safe_logger

    logger = get_safe_logger("daily_review")
except ImportError:
    # Fallback if safe_logging not available
    def setup_logging():
        log_format = "%(asctime)s | %(levelname)s | %(message)s"
        logger = logging.getLogger("daily_review")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console)

        file_handler = logging.FileHandler(
            LOG_DIR / f"daily_review_{datetime.now().strftime('%Y-%m-%d')}.log",
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

        return logger

    logger = setup_logging()

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Load environment
env_path = PROJECT_DIR / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, val = line.split("=", 1)
            os.environ[key.strip()] = val.strip()


def _prior_trading_day(now: datetime):
    """Return the most recent completed trading session date."""
    return last_trading_day((now - timedelta(days=1)).date())


def _resolve_review_date(now: datetime):
    """Resolve which session date the review should describe.

    Delegates to autotrade.utils.market_time.resolve_session_review_date
    so EODReview, DailyReview, and any pm_workflow EOD lookups all agree
    on what "today's completed session" means at any time of day.
    """
    from autotrade.utils.market_time import resolve_session_review_date

    return resolve_session_review_date(now)


def _classify_review_phase(now: datetime) -> str:
    """Classify when the review was generated relative to the market session."""
    if now.weekday() >= 5:
        return "weekend_catchup"
    hour_value = now.hour + (now.minute / 60.0)
    if hour_value >= 16.0:
        return "post_market"
    if hour_value >= 9.5:
        return "intraday"
    if hour_value >= 4.0:
        return "premarket_catchup"
    return "overnight_catchup"


class DailyReview:
    """
    Lightweight post-market review.

    Answers:
    - How did my positions do today?
    - How did my watchlist perform?
    - Did my workflows execute properly?
    """

    def __init__(self):
        self.market_now = get_market_now()
        self.review_generated_at = datetime.now()
        self.review_phase = _classify_review_phase(self.market_now)
        self.review_date = _resolve_review_date(self.market_now)
        self.review_date_str = self.review_date.strftime("%Y-%m-%d")
        self.review_context = {
            "generated_at": self.review_generated_at.isoformat(),
            "market_now": self.market_now.isoformat(),
            "phase": self.review_phase,
            "is_post_market_session_review": self.review_phase == "post_market",
            "is_catchup": self.review_phase != "post_market",
            "source_session_date": self.review_date_str,
        }

        creds = resolve_alpaca_credentials(require=True)
        self.api_key = creds.api_key
        self.secret_key = creds.secret_key
        self.paper = bool(creds.paper)

        # Initialize Alpaca clients
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        self.client = create_trading_client(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper,
            validate_connection=True,
            retries=3,
            retry_delay_seconds=2.0,
            logger=logger,
            require_credentials=True,
        )
        self.data_client = create_data_client(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper,
            require_credentials=True,
        )
        self.trade_journal = trade_learner.get_journal()
        self.GetOrdersRequest = GetOrdersRequest
        self.QueryOrderStatus = QueryOrderStatus

        logger.info(
            "DailyReview initialized for session %s (%s)",
            self.review_date_str,
            self.review_phase,
        )

    def run(self, learn: bool = False) -> Dict:
        """
        Run the daily review.

        Args:
            learn: If True, also run pattern analysis and generate feedback.

        Returns summary with:
        - position_performance: How each position did today
        - watchlist_performance: How watchlist stocks did today
        - workflow_status: Did scheduled workflows run?
        - day_summary: Overall day summary
        """
        logger.info("=" * 60)
        logger.info("DAILY REVIEW - How Did Today Go?")
        logger.info(
            "Time: %s | Review session: %s | Mode: %s",
            self.review_generated_at.strftime("%Y-%m-%d %H:%M:%S"),
            self.review_date_str,
            self.review_phase,
        )
        logger.info("=" * 60)

        review = {
            "generated_at": self.review_generated_at.isoformat(),
            "review_date": self.review_date_str,
            "review_context": dict(self.review_context),
            "position_performance": [],
            "watchlist_performance": [],
            "workflow_status": {},
            "orders_today": [],
            "score_inversion_diagnostics": {},
            "day_summary": {},
        }

        # 1. Position Performance
        logger.info("\n[POSITION PERFORMANCE]")
        logger.info("-" * 50)
        review["position_performance"] = self._analyze_position_performance()

        # 2. Watchlist Performance
        logger.info("\n[WATCHLIST PERFORMANCE]")
        logger.info("-" * 50)
        review["watchlist_performance"] = self._analyze_watchlist_performance()
        review["score_inversion_diagnostics"] = self._analyze_score_inversion(
            review["watchlist_performance"]
        )

        # 3. Orders Executed Today
        logger.info("\n[ORDERS TODAY]")
        logger.info("-" * 50)
        review["orders_today"] = self._get_todays_orders()
        review["short_side_activity"] = self._load_short_side_activity(
            review["orders_today"]
        )
        review["market_adaptation"] = self._load_market_adaptation_summary()

        # 4. Workflow Execution Status
        logger.info("\n[WORKFLOW STATUS]")
        logger.info("-" * 50)
        review["workflow_status"] = self._check_workflow_execution()

        # 5. Release Gates Status (New)
        if REPORTING_AVAILABLE:
            logger.info("\n[RELEASE GATES]")
            logger.info("-" * 50)
            review["release_gates"] = self._check_release_gates()

        # 6. Generate Day Summary
        review["day_summary"] = self._generate_day_summary(review)

        # 7. Post-Market Learning (New)
        if learn:
            logger.info("\n[LEARNING]")
            logger.info("-" * 50)
            review["learning"] = self.learn_from_today(review)

        # Print final summary
        self._print_summary(review)

        # Save review
        self._save_review(review)

        return review

    def learn_from_today(self, review: Dict) -> Dict:
        """
        Analyze today's trades and generate lessons for tomorrow.
        """
        logger.info("   Starting post-market learning process...")

        try:
            # 0. Sync outcomes from EOD review so journal knows who won/lost
            # This fixes the 0.0 win-rate / insufficient data issues
            self.trade_journal.update_outcomes_from_eod(
                {"trades": review.get("position_performance", [])}
            )

            # 1. Run pattern analysis and generate new lessons
            lessons = trade_learner.analyze_and_learn()

            # 2. Extract feedback metrics
            feedback = _build_feedback_from_review(review)
            feedback["date"] = getattr(
                self, "review_date_str", datetime.now().strftime("%Y-%m-%d")
            )
            feedback["lessons_updated"] = False
            feedback["underperforming_sector_details"] = (
                self._get_underperforming_sector_details(review)
            )
            feedback["underperforming_sectors"] = [
                row["sector"] for row in feedback["underperforming_sector_details"]
            ]
            feedback["score_bucket_performance"] = self._analyze_score_buckets(review)
            feedback["score_inversion_diagnostics"] = review.get(
                "score_inversion_diagnostics", {}
            )
            signal_family_payload = self._extract_signal_family_payload(lessons)
            feedback["learning_context"] = {
                "generated_at": signal_family_payload.get("generated_at"),
                "family_count": int(signal_family_payload.get("family_count", 0) or 0),
            }
            feedback["best_signal_families"] = self._summarize_signal_families(
                signal_family_payload, positive=True
            )
            feedback["weak_signal_families"] = self._summarize_signal_families(
                signal_family_payload, positive=False
            )

            report_path = (
                PROJECT_DIR / "reports" / f"daily_lessons_{feedback['date']}.md"
            )
            lessons_status = {
                "analyzer_invoked": False,
                "report_path": str(report_path),
                "report_exists": report_path.exists(),
                "mode": "",
                "success": False,
                "error": "",
            }

            # 3. Generate markdown lesson files (Phase 7: lesson pipeline fix)
            try:
                from autotrade.core.daily_lessons_analyzer import DailyLessonsAnalyzer

                analyzer = DailyLessonsAnalyzer()
                analyzer_result = analyzer.run(date_str=feedback["date"])
                lessons_status["analyzer_invoked"] = True
                if isinstance(analyzer_result, dict):
                    lessons_status["mode"] = str(analyzer_result.get("mode") or "")
                    lessons_status["success"] = bool(
                        analyzer_result.get("success", False)
                    )
                    report_path = Path(
                        analyzer_result.get("report_path") or str(report_path)
                    )
                lessons_status["report_path"] = str(report_path)
                lessons_status["report_exists"] = report_path.exists()
                feedback["lessons_updated"] = bool(report_path.exists())
                logger.info("   ✅ Daily markdown lessons generated")
            except Exception as analyzer_err:
                lessons_status["error"] = str(analyzer_err)
                lessons_status["report_exists"] = report_path.exists()
                feedback["lessons_updated"] = bool(report_path.exists())
                logger.warning(
                    f"   ⚠️ Could not generate lesson markdown: {analyzer_err}"
                )

            feedback["lessons_status"] = lessons_status

            learning_state = build_learning_state(
                review=review,
                feedback=feedback,
                lessons_payload=lessons,
                reports_dir=PROJECT_DIR / "reports",
                state_path=PROJECT_DIR / "data" / "daily_learning_state.json",
                as_of_date=feedback["date"],
            )
            feedback["learning_state_path"] = "data/daily_learning_state.json"
            feedback["learning_state_summary"] = (
                learning_state.get("learning_digest", {}) if learning_state else {}
            )
            feedback["learning_artifact_status"] = (
                learning_state.get("learning_artifact_status", {})
                if learning_state
                else {}
            )

            # 4. Save as latest feedback for research bias after lesson status is known
            feedback_path = PROJECT_DIR / "data" / "eod_feedback_latest.json"
            feedback_path.parent.mkdir(exist_ok=True)
            with open(feedback_path, "w", encoding="utf-8") as f:
                json.dump(feedback, f, indent=2)

            logger.info(f"   ✅ Feedback loop complete. Saved to {feedback_path.name}")
            return feedback

        except Exception as e:
            logger.error(f"   âŒ Learning failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _get_underperforming_sectors(self, review: Dict) -> List[str]:
        """Identify sectors that dragged down watchlist performance today."""
        return [
            row["sector"] for row in self._get_underperforming_sector_details(review)
        ]

    def _get_underperforming_sector_details(self, review: Dict) -> List[Dict[str, Any]]:
        watchlist = (
            review.get("watchlist_performance", []) if isinstance(review, dict) else []
        )
        if not watchlist:
            return []

        sector_changes: Dict[str, List[float]] = {}
        for row in watchlist:
            sector = str(row.get("sector", "") or "").strip()
            change_pct = row.get("change_pct")
            if not sector or change_pct is None:
                continue
            sector_changes.setdefault(sector, []).append(float(change_pct))

        details: List[Dict[str, Any]] = []
        for sector, changes in sector_changes.items():
            avg_change = sum(changes) / len(changes)
            if avg_change < 0:
                details.append(
                    {
                        "sector": sector,
                        "avg_change_pct": round(avg_change, 2),
                        "sample_size": len(changes),
                    }
                )

        details.sort(key=lambda row: (row["avg_change_pct"], -row["sample_size"]))
        return details

    def _analyze_score_buckets(self, review: Dict) -> Dict:
        """Analyze how different signal score ranges performed."""
        watchlist = (
            review.get("watchlist_performance", []) if isinstance(review, dict) else []
        )
        bucket_specs = (
            ("0-34", 0.0, 35.0),
            ("35-49", 35.0, 50.0),
            ("50-64", 50.0, 65.0),
            ("65-79", 65.0, 80.0),
            ("80+", 80.0, None),
        )
        buckets: Dict[str, Dict[str, Any]] = {
            label: {
                "count": 0,
                "avg_change_pct": 0.0,
                "positive_rate": 0.0,
                "missed_opportunities": 0,
                "top_sector": "",
                "top_setup_type": "",
            }
            for label, _, _ in bucket_specs
        }
        raw_changes: Dict[str, List[float]] = {
            label: [] for label, _, _ in bucket_specs
        }
        sector_counts: Dict[str, Dict[str, int]] = {
            label: {} for label, _, _ in bucket_specs
        }
        setup_counts: Dict[str, Dict[str, int]] = {
            label: {} for label, _, _ in bucket_specs
        }

        for row in watchlist:
            score = row.get("final_score", row.get("confidence"))
            change_pct = row.get("change_pct")
            if score is None or change_pct is None:
                continue
            score_value = float(score)
            change_value = float(change_pct)
            label = "80+"
            for candidate_label, floor, ceiling in bucket_specs:
                if ceiling is None:
                    if score_value >= floor:
                        label = candidate_label
                        break
                elif floor <= score_value < ceiling:
                    label = candidate_label
                    break

            buckets[label]["count"] += 1
            raw_changes[label].append(change_value)
            if row.get("missed_opportunity"):
                buckets[label]["missed_opportunities"] += 1
            sector = str(row.get("sector", "") or "").strip()
            if sector:
                sector_counts[label][sector] = sector_counts[label].get(sector, 0) + 1
            setup_type = str(row.get("setup_type", "") or "").strip()
            if setup_type:
                setup_counts[label][setup_type] = (
                    setup_counts[label].get(setup_type, 0) + 1
                )

        for label, bucket in buckets.items():
            changes = raw_changes[label]
            if changes:
                positives = sum(1 for value in changes if value > 0)
                bucket["avg_change_pct"] = round(sum(changes) / len(changes), 2)
                bucket["positive_rate"] = round(positives / len(changes), 3)
            if sector_counts[label]:
                bucket["top_sector"] = max(
                    sector_counts[label].items(), key=lambda item: (item[1], item[0])
                )[0]
            if setup_counts[label]:
                bucket["top_setup_type"] = max(
                    setup_counts[label].items(), key=lambda item: (item[1], item[0])
                )[0]
        return buckets

    def _analyze_score_inversion(
        self, watchlist: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Detect when top-ranked signals underperform the broader signal pool."""
        valid_rows: List[Dict[str, float]] = []
        for row in watchlist or []:
            if not isinstance(row, dict):
                continue
            score = row.get("final_score", row.get("confidence", row.get("score")))
            change = row.get("change_pct")
            if score is None or change is None:
                continue
            try:
                valid_rows.append(
                    {
                        "score": float(score),
                        "change_pct": float(change),
                    }
                )
            except Exception:
                continue

        if not valid_rows:
            return {
                "available": False,
                "reason": "no_scored_watchlist_rows",
                "rows": 0,
            }

        ranked = sorted(valid_rows, key=lambda item: item["score"], reverse=True)
        top_n = min(10, max(1, len(ranked)))
        top_rows = ranked[:top_n]

        def _avg(rows: List[Dict[str, float]]) -> float:
            return round(sum(item["change_pct"] for item in rows) / len(rows), 4)

        overall_avg = _avg(valid_rows)
        top_avg = _avg(top_rows)
        top_positive_rate = round(
            sum(1 for item in top_rows if item["change_pct"] > 0) / len(top_rows), 4
        )
        overall_positive_rate = round(
            sum(1 for item in valid_rows if item["change_pct"] > 0) / len(valid_rows),
            4,
        )
        inverted = top_avg < overall_avg and top_positive_rate <= overall_positive_rate

        # Enhanced Bucket Analysis
        buckets: Dict[str, List[float]] = {
            "STRONG_BUY": [],  # 75+
            "BUY": [],  # 55-75
            "WATCH": [],  # < 55
        }
        for row in valid_rows:
            sc = row["score"]
            ch = row["change_pct"]
            if sc >= 75:
                buckets["STRONG_BUY"].append(ch)
            elif sc >= 55:
                buckets["BUY"].append(ch)
            else:
                buckets["WATCH"].append(ch)

        bucket_avgs = {k: (sum(v) / len(v)) if v else 0.0 for k, v in buckets.items()}
        bucket_counts = {k: len(v) for k, v in buckets.items()}

        # Check for bucket inversion (STRONG_BUY worse than BUY or WATCH)
        bucket_inversion = False
        if bucket_counts["STRONG_BUY"] >= 3:
            if (
                bucket_avgs["STRONG_BUY"] < bucket_avgs["BUY"]
                or bucket_avgs["STRONG_BUY"] < bucket_avgs["WATCH"]
            ):
                bucket_inversion = True

        return {
            "available": True,
            "rows": len(valid_rows),
            "top_n": top_n,
            "top_avg_change_pct": top_avg,
            "overall_avg_change_pct": overall_avg,
            "top_positive_rate": top_positive_rate,
            "overall_positive_rate": overall_positive_rate,
            "inverted": inverted,
            "bucket_inversion": bucket_inversion,
            "bucket_avgs": bucket_avgs,
            "bucket_counts": bucket_counts,
            "severity": "warning" if (inverted or bucket_inversion) else "info",
        }

    def _extract_signal_family_payload(self, lessons: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(lessons, dict):
            return {"generated_at": None, "family_count": 0, "families": []}

        payload = lessons.get("signal_families")
        if isinstance(payload, dict):
            families = payload.get("families", [])
            return {
                "generated_at": payload.get("generated_at"),
                "family_count": payload.get("family_count", len(families)),
                "families": families,
            }

        return {"generated_at": None, "family_count": 0, "families": []}

    def _summarize_signal_families(
        self, family_payload: Dict[str, Any], positive: bool
    ) -> List[Dict[str, Any]]:
        families = (
            family_payload.get("families", [])
            if isinstance(family_payload, dict)
            else []
        )
        summarized: List[Dict[str, Any]] = []

        for family in families:
            adjustment = float(family.get("score_adjustment", 0.0) or 0.0)
            if positive and adjustment <= 0:
                continue
            if not positive and adjustment >= 0:
                continue
            summarized.append(
                {
                    "signal_family": family.get("signal_family"),
                    "score_adjustment": adjustment,
                    "confidence": float(family.get("confidence", 0.0) or 0.0),
                    "sample_size": int(family.get("total_count", 0) or 0),
                    "avg_realized_pct": float(
                        family.get("avg_realized_pct", 0.0) or 0.0
                    ),
                    "avg_open_pct": float(family.get("avg_open_pct", 0.0) or 0.0),
                    "reason": family.get("reason", ""),
                }
            )

        summarized.sort(
            key=lambda row: abs(float(row.get("score_adjustment", 0.0) or 0.0)),
            reverse=True,
        )
        return summarized

    def _analyze_position_performance(self) -> List[Dict]:
        """Analyze how each current position performed today."""
        from autotrade.utils.position_cache import fetch_positions_with_fallback

        result = fetch_positions_with_fallback(
            self.client,
            logger=logger,
            retries=2,
            retry_delay_seconds=1.0,
            backoff_multiplier=2.0,
            use_cache=True,
        )
        positions = list(result.get("positions") or [])

        if not positions:
            logger.info("   No positions currently held")
            return []

        performance = []

        for pos in positions:
            symbol = pos.symbol
            entry_price = float(pos.avg_entry_price)
            current_price = float(pos.current_price)
            qty = int(pos.qty)
            market_value = float(pos.market_value)

            # Total P&L
            total_pnl_pct = float(pos.unrealized_plpc) * 100
            total_pnl_dollars = float(pos.unrealized_pl)

            # Today's change (approximate from change_today)
            try:
                today_pnl_pct = (
                    float(pos.change_today) * 100 if hasattr(pos, "change_today") else 0
                )
            except Exception:
                today_pnl_pct = 0

            # Determine performance grade (ASCII-safe for Windows console)
            if today_pnl_pct >= 3:
                grade = "[++] GREAT"
            elif today_pnl_pct >= 1:
                grade = "[+] GOOD"
            elif today_pnl_pct >= 0:
                grade = "[=] FLAT"
            elif today_pnl_pct >= -2:
                grade = "[-] DOWN"
            else:
                grade = "[X] BAD"

            logger.info(
                f"   {symbol}: {grade} | Today: {today_pnl_pct:+.2f}% | Total: {total_pnl_pct:+.2f}% (${total_pnl_dollars:+,.2f})"
            )

            performance.append(
                {
                    "symbol": symbol,
                    "qty": qty,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "market_value": market_value,
                    "today_pnl_pct": today_pnl_pct,
                    "total_pnl_pct": total_pnl_pct,
                    "total_pnl_dollars": total_pnl_dollars,
                    "grade": grade,
                }
            )

        return performance

    def _analyze_watchlist_performance(self) -> List[Dict]:
        """Check how stocks on the watchlist performed today."""
        from alpaca.data.requests import StockLatestQuoteRequest

        # Load today's watchlist/signals
        watchlist = self._load_watchlist()

        if not watchlist:
            logger.info("   No watchlist found for today")
            return []

        performance = []

        # Get quotes for watchlist symbols
        symbols = [
            w.get("symbol", w) if isinstance(w, dict) else w for w in watchlist[:20]
        ]

        try:
            quotes = self.data_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=symbols)
            )

            for symbol in symbols:
                quote_mid: Optional[float] = None
                if symbol in quotes:
                    quote = quotes[symbol]
                    bid = float(getattr(quote, "bid_price", 0.0) or 0.0)
                    ask = float(getattr(quote, "ask_price", 0.0) or 0.0)
                    if bid > 0 and ask > 0:
                        quote_mid = (ask + bid) / 2
                    elif ask > 0:
                        quote_mid = ask
                    elif bid > 0:
                        quote_mid = bid

                reference_dt = datetime.combine(self.review_date, datetime.min.time())
                yesterday_close = self._get_yesterday_close(symbol, reference_dt)
                prev_close, latest_close = self._get_recent_daily_closes(
                    symbol, reference_dt
                )
                if yesterday_close is None:
                    yesterday_close = prev_close

                current_price = (
                    quote_mid if quote_mid and quote_mid > 0 else latest_close
                )
                change_pct: Optional[float] = None
                if (
                    current_price is not None
                    and yesterday_close is not None
                    and yesterday_close > 0
                ):
                    change_pct = (
                        (current_price - yesterday_close) / yesterday_close
                    ) * 100

                # Check if we should have entered
                watchlist_entry = next(
                    (
                        w
                        for w in watchlist
                        if (w.get("symbol") if isinstance(w, dict) else w) == symbol
                    ),
                    None,
                )
                entry_price_raw = (
                    watchlist_entry.get("entry_price")
                    if isinstance(watchlist_entry, dict)
                    else None
                )
                try:
                    entry_price = (
                        float(entry_price_raw) if entry_price_raw is not None else None
                    )
                except Exception:
                    entry_price = None

                missed_opportunity = bool(
                    entry_price and current_price and current_price > entry_price * 1.02
                )  # 2%+ above entry

                if change_pct is None:
                    status = "[?] N/A"
                else:
                    status = "[^] UP" if change_pct >= 0 else "[v] DOWN"
                if missed_opportunity:
                    status = "[!] MISSED"

                if change_pct is None:
                    if current_price is not None:
                        logger.info(
                            f"   {symbol}: {status} (missing prior close) @ ${current_price:.2f}"
                        )
                    else:
                        logger.info(f"   {symbol}: {status} (missing price/close data)")
                else:
                    logger.info(
                        f"   {symbol}: {status} {change_pct:+.2f}% @ ${current_price:.2f}"
                    )

                # Carry signal scores so _analyze_score_buckets can bucket them
                signal_score = None
                if isinstance(watchlist_entry, dict):
                    for _sk in ("final_score", "confidence", "score"):
                        _sv = watchlist_entry.get(_sk)
                        if _sv is not None:
                            try:
                                signal_score = float(_sv)
                            except (TypeError, ValueError):
                                continue
                            break

                perf_entry = {
                    "symbol": symbol,
                    "current_price": current_price,
                    "yesterday_close": yesterday_close,
                    "change_pct": change_pct,
                    "missed_opportunity": missed_opportunity,
                    "status": status,
                }
                if signal_score is not None:
                    perf_entry["final_score"] = signal_score
                performance.append(perf_entry)

        except Exception as e:
            logger.warning(f"   Could not fetch quotes: {e}")

        return performance

    def _get_yesterday_close(
        self, symbol: str, reference_date: Optional[datetime] = None
    ) -> Optional[float]:
        """Get yesterday's closing price for a symbol."""
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        try:
            if reference_date is None:
                reference_date = datetime.now()
            target = reference_date - timedelta(days=1)
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=target - timedelta(days=5),  # Buffer for weekends
                end=target,
            )
            bars = self.data_client.get_stock_bars(request)

            closes = self._extract_daily_closes(bars, symbol)
            if closes:
                return float(closes[-1])
        except Exception:
            pass

        return None

    def _get_recent_daily_closes(
        self, symbol: str, reference_date: datetime
    ) -> tuple[Optional[float], Optional[float]]:
        """Return (previous_close, latest_close) around review date."""
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=reference_date - timedelta(days=7),
                end=reference_date + timedelta(days=1),
            )
            bars = self.data_client.get_stock_bars(request)
            closes = self._extract_daily_closes(bars, symbol)
            if len(closes) >= 2:
                return float(closes[-2]), float(closes[-1])
            if len(closes) == 1:
                return None, float(closes[-1])
        except Exception:
            pass
        return None, None

    def _extract_daily_closes(self, bars: object, symbol: str) -> List[float]:
        """Extract close prices from Alpaca bars response (df or symbol mapping)."""
        closes: List[float] = []

        try:
            df = getattr(bars, "df", None)
            if df is not None and not df.empty:
                symbols_idx = df.index.get_level_values(0)
                if symbol in symbols_idx:
                    symbol_df = df.loc[symbol]
                    if "close" in symbol_df:
                        return [
                            float(v)
                            for v in symbol_df["close"].tolist()
                            if v is not None
                        ]
        except Exception:
            pass

        try:
            symbol_bars = bars[symbol]
            for bar in symbol_bars:
                close_val = getattr(bar, "close", None)
                if close_val is not None:
                    closes.append(float(close_val))
        except Exception:
            pass

        return closes

    def _load_watchlist(self) -> List:
        """Load today's watchlist from signals or plans."""
        review_date = self.review_date_str

        # Try signals file first
        signals_path = LOG_DIR / f"signals_{review_date}.json"
        if signals_path.exists():
            with open(signals_path) as f:
                data = json.load(f)
            # signals files can be a dict with 'signals' key or a plain list
            if isinstance(data, dict):
                return data.get("signals", [])
            return data

        # Try PM plan for review date
        plan_path = PLANS_DIR / f"pm_plan_{review_date}.json"
        if plan_path.exists():
            with open(plan_path) as f:
                plan = json.load(f)
            return plan.get("entry_candidates", plan.get("signals", []))

        # Try morning game plan for review date
        morning_plan_path = (
            PLANS_DIR / f"morning_game_plan_{review_date.replace('-', '')}.json"
        )
        if morning_plan_path.exists():
            with open(morning_plan_path) as f:
                plan = json.load(f)
            return plan.get("entry_candidates", plan.get("signals", []))

        # Try yesterday's plan (might be looking for today's trades)
        yesterday = (self.review_date - timedelta(days=1)).strftime("%Y-%m-%d")
        plan_path = PLANS_DIR / f"pm_plan_{yesterday}.json"
        if plan_path.exists():
            with open(plan_path) as f:
                plan = json.load(f)
            return plan.get("entry_candidates", plan.get("signals", []))

        return []

    def _get_todays_orders(self) -> List[Dict]:
        """Get all orders placed today."""
        review_start = datetime.combine(self.review_date, datetime.min.time())

        orders = []
        try:
            review_end = review_start + timedelta(days=1)
            request = self.GetOrdersRequest(
                status=self.QueryOrderStatus.ALL,
                after=review_start,
                until=review_end,
                limit=500,
            )
            all_orders = self.client.get_orders(filter=request)

            for order in all_orders:
                status = (
                    (
                        str(order.status.value)
                        if hasattr(order.status, "value")
                        else str(order.status)
                    )
                    .lower()
                    .split(".")[-1]
                )
                side = (
                    (
                        str(order.side.value)
                        if hasattr(order.side, "value")
                        else str(order.side)
                    )
                    .lower()
                    .split(".")[-1]
                )

                order_info = {
                    "id": str(getattr(order, "id", "") or ""),
                    "symbol": order.symbol,
                    "side": side,
                    "qty": int(order.qty),
                    "status": status,
                    "filled_qty": int(order.filled_qty) if order.filled_qty else 0,
                    "avg_fill_price": float(order.filled_avg_price)
                    if order.filled_avg_price
                    else None,
                    "submitted_at": order.submitted_at.isoformat()
                    if order.submitted_at
                    else None,
                    "filled_at": order.filled_at.isoformat()
                    if getattr(order, "filled_at", None)
                    else None,
                }
                orders.append(order_info)

                indicator = (
                    "[OK]"
                    if status == "filled"
                    else "[..]"
                    if status == "new"
                    else "[X]"
                )
                logger.info(
                    f"   {indicator} {side.upper()} {order.qty} {order.symbol} - {status}"
                )

        except Exception as e:
            logger.warning(f"   Could not fetch orders: {e}")

        if not orders:
            logger.info("   No orders placed today")

        return orders

    def _load_json_if_exists(self, path: Path) -> Optional[Dict]:
        """Read a JSON file if it exists and is valid."""
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _get_overnight_readiness(self) -> Dict[str, object]:
        """Resolve overnight readiness from current workflow artifacts."""
        target_trade_date = self.review_date
        target_trade_date_str = target_trade_date.strftime("%Y-%m-%d")
        target_trade_date_compact = target_trade_date.strftime("%Y%m%d")
        research_dir = PROJECT_DIR / "research"
        state_path = research_dir / "overnight_state.json"
        plan_path = PLANS_DIR / f"morning_game_plan_{target_trade_date_compact}.json"
        legacy_report_date = (target_trade_date - timedelta(days=1)).strftime("%Y%m%d")
        legacy_report_files = list(
            research_dir.glob(f"overnight_report_{legacy_report_date}*.json")
        )

        state = self._load_json_if_exists(state_path) or {}
        completion = state.get("workflow_completion", {})
        if not isinstance(completion, dict):
            completion = {}

        state_date = str(state.get("date") or "").strip()
        target_matches = state_date == target_trade_date_str
        research_complete = bool(state.get("research_complete"))
        game_plan_generated = bool(completion.get("game_plan_generated"))
        target_from_completion = str(completion.get("target_trade_date") or "").strip()
        completion_matches = (
            not target_from_completion
            or target_from_completion == target_trade_date_str
        )
        plan_exists = plan_path.exists()
        ready = bool(
            target_matches
            and completion_matches
            and research_complete
            and game_plan_generated
            and plan_exists
        )
        ran = bool(target_matches or plan_exists or legacy_report_files)

        if ready:
            return {
                "ran": True,
                "ready": True,
                "status": "ran_and_ready",
                "detail": f"state={state_path.name}, plan={plan_path.name}",
            }
        if ran:
            reasons: List[str] = []
            if not target_matches:
                reasons.append(f"state_date={state_date or 'missing'}")
            if target_matches and not research_complete:
                reasons.append("research_complete=false")
            if target_matches and not game_plan_generated:
                reasons.append("game_plan_generated=false")
            if not plan_exists:
                reasons.append(f"missing_plan={plan_path.name}")
            if legacy_report_files and not reasons:
                reasons.append("legacy_report_only")
            return {
                "ran": True,
                "ready": False,
                "status": "ran_but_incomplete",
                "detail": ", ".join(reasons),
            }
        return {
            "ran": False,
            "ready": False,
            "status": "not_run",
            "detail": f"missing_state_and_plan_for={target_trade_date_str}",
        }

    def _check_workflow_execution(self) -> Dict:
        """Check if scheduled workflows ran successfully today."""
        today = self.review_date_str

        status = {
            "premarket_ran": False,
            "market_hours_ran": False,
            "pm_workflow_ran": False,
            "overnight_research_ran": False,
            "review_phase": self.review_phase,
            "source_session_date": today,
            "is_post_market_session_review": self.review_phase == "post_market",
            "issues_found": [],
            "all_healthy": True,
        }

        if self.review_phase != "post_market":
            status["issues_found"].append(
                f"Review generated during {self.review_phase}; treating as catch-up for {today}"
            )
            status["all_healthy"] = False
            logger.info(
                f"   [INFO] Catch-up review for {today}; skipping live workflow health checks"
            )
            return status

        agent_log = (
            LOG_DIR / f"autonomous_agent_{self.review_date.strftime('%Y-%m-%d')}.log"
        )

        def _read_recent_agent_log() -> str:
            if not agent_log.exists():
                return ""
            try:
                with open(agent_log, "r", encoding="utf-8", errors="ignore") as f:
                    try:
                        f.seek(max(0, agent_log.stat().st_size - 500_000))
                    except Exception:
                        pass
                    return f.read()
            except Exception:
                return ""

        # Check for log files indicating workflows ran
        premarket_log = LOG_DIR / f"premarket_{today}.log"
        if premarket_log.exists():
            status["premarket_ran"] = True
            logger.info("   [OK] Premarket workflow ran")
        else:
            content = _read_recent_agent_log()
            if "PREMARKET" in content and (
                "MARKET_OPEN" in content or "CYCLE" in content
            ):
                status["premarket_ran"] = True
                logger.info("   [OK] Premarket ran (via autonomous agent)")
            else:
                logger.info("   [WARN] Premarket workflow did NOT run")
                status["issues_found"].append("Premarket did not run")
                status["all_healthy"] = False

        # Check day manager log
        day_manager_log = LOG_DIR / f"day_manager_{today}.log"
        if day_manager_log.exists():
            status["market_hours_ran"] = True
            logger.info("   [OK] Market hours workflow ran")
        else:
            content = _read_recent_agent_log()
            if "MARKET_HOURS" in content or "[DAY MANAGER]" in content:
                status["market_hours_ran"] = True
                logger.info("   [OK] Market hours ran (via autonomous agent)")
            else:
                logger.info("   [WARN] Market hours workflow did NOT run")
                status["issues_found"].append("Market hours did not run")
                status["all_healthy"] = False

        # Check for PM workflow
        pm_log = LOG_DIR / f"pm_workflow_{today}.log"
        if pm_log.exists():
            status["pm_workflow_ran"] = True
            logger.info("   [OK] PM workflow ran")

        overnight = self._get_overnight_readiness()
        if overnight["ready"]:
            status["overnight_research_ran"] = True
            logger.info(
                f"   [OK] Overnight research ran and is ready ({overnight['detail']})"
            )
        elif overnight["ran"]:
            logger.info(
                f"   [WARN] Overnight research ran but is incomplete ({overnight['detail']})"
            )
            status["issues_found"].append("Overnight research ran but is incomplete")
            status["all_healthy"] = False
        else:
            logger.info(
                f"   [WARN] Overnight research did NOT run ({overnight['detail']})"
            )
            status["issues_found"].append("Overnight research did not run")
            status["all_healthy"] = False

        # Check for errors in today's logs
        errors_found = self._scan_logs_for_errors(today)
        if errors_found:
            status["issues_found"].extend(errors_found)
            status["all_healthy"] = False
            for err in errors_found[:3]:
                logger.info(f"   [ERROR] {err}")

        if status["all_healthy"] and self.review_phase == "post_market":
            logger.info("   [OK] All workflows healthy!")

        return status

    def _scan_logs_for_errors(self, date_str: str) -> List[str]:
        """Scan today's logs for ERROR entries."""
        errors = []

        for log_file in LOG_DIR.glob(f"*_{date_str}*.log"):
            try:
                content = log_file.read_text(encoding="utf-8", errors="ignore")
                for line in content.split("\n"):
                    if "| ERROR |" in line:
                        # Extract error message
                        parts = line.split("| ERROR |")
                        if len(parts) > 1:
                            error_msg = parts[1].strip()[:100]
                            if error_msg not in errors:
                                errors.append(f"{log_file.stem}: {error_msg}")
            except Exception:
                pass

        return errors[:5]  # Limit to 5 errors

    def _check_release_gates(self) -> Dict:
        """Check the status of release gates via ReportingEngine."""
        status = {}
        if not REPORTING_AVAILABLE:
            return {"status": "unavailable"}

        try:
            # Import on demand to avoid circular deps if any
            from autotrade.monitoring.contracts import GateStatus

            engine = get_reporting_engine(output_dir=LOG_DIR / "reports")
            # Generate the report - this checks the gates
            gate_report = engine.generate_release_gate_report()

            # Extract key info
            status = {
                "overall_status": gate_report.overall_status,
                "test_status": gate_report.test_status,
                "backtest_status": gate_report.backtest_status,
                "paper_trading_status": gate_report.paper_trading_status,
                "canary_ready": gate_report.canary_ready,
                "rollback_ready": gate_report.rollback_ready,
                "pass_reason": gate_report.pass_reason,
                "fail_reason": gate_report.fail_reason,
            }

            indicator = (
                "âœ… PASS"
                if gate_report.overall_status == GateStatus.PASS.value
                else "âŒ FAIL"
            )
            if gate_report.overall_status == GateStatus.PASS.value:
                indicator = "[OK] PASS"
            elif gate_report.overall_status == GateStatus.FAIL.value:
                indicator = "[FAIL] FAIL"
            else:
                indicator = f"[PENDING] {gate_report.overall_status.upper()}"
            logger.info(f"   Overall Status: {indicator}")

            if gate_report.fail_reason:
                for reason in gate_report.fail_reason.split("; "):
                    logger.info(f"   Reason: {reason}")
                fail_list = []
                for reason in fail_list:
                    logger.info(f"   âš ï¸ Reason: {reason}")
            elif gate_report.pass_reason:
                logger.info(f"   Reason: {gate_report.pass_reason}")

        except Exception as e:
            logger.warning(f"   Could not check release gates: {e}")
            status = {"error": str(e)}

        return status

    def _generate_day_summary(self, review: Dict) -> Dict:
        """Generate an overall summary of the day."""
        positions = review["position_performance"]
        orders = review["orders_today"]
        workflow = review["workflow_status"]
        release_gates = review.get("release_gates", {})

        # Calculate totals
        total_positions = len(positions)
        total_value = sum(p["market_value"] for p in positions)
        total_pnl_today = sum(
            p.get("today_pnl_pct", 0) * p["market_value"] / 100 for p in positions
        )
        total_pnl_overall = sum(p["total_pnl_dollars"] for p in positions)

        winners = len([p for p in positions if p.get("today_pnl_pct", 0) > 0])
        losers = len([p for p in positions if p.get("today_pnl_pct", 0) < 0])

        # Orders summary
        filled_orders = len([o for o in orders if o["status"] == "filled"])
        total_orders = len(orders)

        # Release Gates
        gates_status = release_gates.get("overall_status", "unknown")

        # Determine overall grade
        if total_pnl_today >= 100:
            grade = "ðŸš€ EXCELLENT DAY"
        elif total_pnl_today >= 0:
            grade = "âœ… GOOD DAY"
        elif total_pnl_today >= -100:
            grade = "âž¡ï¸ FLAT DAY"
        elif total_pnl_today >= -300:
            grade = "âš ï¸ DOWN DAY"
        else:
            grade = "âŒ BAD DAY"

        summary = {
            "grade": grade,
            "total_positions": total_positions,
            "total_value": total_value,
            "today_pnl_dollars": total_pnl_today,
            "overall_pnl_dollars": total_pnl_overall,
            "winners": winners,
            "losers": losers,
            "orders_filled": filled_orders,
            "orders_total": total_orders,
            "workflows_healthy": workflow["all_healthy"],
            "issues_count": len(workflow["issues_found"]),
            "release_gates_status": gates_status,
        }

        return summary

    def _print_summary(self, review: Dict):
        """Print the final summary."""
        summary = review["day_summary"]
        context = review.get("review_context", {})

        logger.info("\n" + "=" * 60)
        logger.info("DAILY SUMMARY")
        logger.info("=" * 60)
        logger.info(
            "   Session: %s | Mode: %s",
            review.get("review_date", self.review_date_str),
            context.get("phase", "unknown"),
        )
        logger.info(f"   Grade: {summary['grade']}")
        logger.info(
            f"   Positions: {summary['total_positions']} | Value: ${summary['total_value']:,.2f}"
        )
        logger.info(f"   Today P&L: ${summary['today_pnl_dollars']:+,.2f}")
        logger.info(f"   Overall P&L: ${summary['overall_pnl_dollars']:+,.2f}")
        logger.info(f"   Winners/Losers: {summary['winners']}/{summary['losers']}")
        logger.info(
            f"   Orders: {summary['orders_filled']}/{summary['orders_total']} filled"
        )
        short_side = review.get("short_side_activity", {}) or {}
        logger.info(
            "   Short-side: inverse_screens=%s inverse_candidates=%s "
            "single_name_generated=%s single_name_executed=%s",
            short_side.get("inverse_etf_screens", 0),
            short_side.get("inverse_etf_candidates", 0),
            short_side.get("single_name_shorts_generated", 0),
            short_side.get("single_name_shorts_executed", 0),
        )
        market_adaptation = review.get("market_adaptation", {}) or {}
        if market_adaptation:
            logger.info(
                "   Market adaptation: %s | dispersion=%.2f | sizing_hint=%.2fx | %s",
                market_adaptation.get("regime_label", "UNKNOWN"),
                float(market_adaptation.get("dispersion_score", 0.0) or 0.0),
                float(market_adaptation.get("sizing_multiplier", 1.0) or 1.0),
                market_adaptation.get("summary_line", ""),
            )

        workflow_status = (
            "All Healthy"
            if summary["workflows_healthy"]
            else f"WARN {summary['issues_count']} issues"
        )
        logger.info(f"   Workflows: {workflow_status}")

        gates_indicator = (
            "PASS"
            if summary.get("release_gates_status") == "pass"
            else f"PENDING {summary.get('release_gates_status', 'Unknown').upper()}"
        )
        logger.info(f"   Release Gates: {gates_indicator}")

        logger.info("=" * 60)

    def _save_review(self, review: Dict):
        """Save the review to disk."""
        date_str = str(review.get("review_date") or self.review_date_str)
        review_path = LOG_DIR / f"daily_review_{date_str}.json"

        with open(review_path, "w", encoding="utf-8") as f:
            json.dump(review, f, indent=2, default=str)

        logger.info(f">>> Review saved: {review_path}")

    def _load_short_side_activity(
        self, orders_today: Optional[List[Dict]] = None
    ) -> Dict:
        """Summarize inverse and single-name short diagnostics for daily review."""
        date_str = self.review_date_str
        telemetry_path = LOG_DIR / f"short_engine_telemetry_{date_str}.jsonl"
        intraday_path = LOG_DIR / f"intraday_analysis_{date_str}.jsonl"

        single_name_generated = 0
        telemetry_records = 0
        if telemetry_path.exists():
            for line in telemetry_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                telemetry_records += 1
                single_name_generated += int(row.get("signals_generated", 0) or 0)

        inverse_screens = 0
        inverse_candidates = 0
        if intraday_path.exists():
            for line in intraday_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                diagnostics = row.get("execution_diagnostics") or {}
                inverse_screens += int(diagnostics.get("inverse_etf_screens", 0) or 0)
                inverse_candidates += int(
                    diagnostics.get("inverse_etf_candidates", 0) or 0
                )

        short_exec_sides = {"short", "sell_short"}
        single_name_executed = sum(
            1
            for order in orders_today or []
            if str(order.get("side", "")).lower() in short_exec_sides
            and str(order.get("status", "")).lower() == "filled"
        )

        all_zero = (
            inverse_screens == 0
            and inverse_candidates == 0
            and single_name_generated == 0
            and single_name_executed == 0
        )
        return {
            "inverse_etf_screens": inverse_screens,
            "inverse_etf_candidates": inverse_candidates,
            "single_name_shorts_generated": single_name_generated,
            "single_name_shorts_executed": single_name_executed,
            "short_engine_telemetry_records": telemetry_records,
            "warning": "short_side_silent" if all_zero else "",
        }

    def _load_market_adaptation_summary(self) -> Dict:
        """Return a one-line intraday market adaptation summary for EOD review."""
        date_str = self.review_date_str
        intraday_path = LOG_DIR / f"intraday_analysis_{date_str}.jsonl"
        if not intraday_path.exists():
            return {}

        latest: Dict = {}
        for line in intraday_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                latest = row
        market = latest.get("market_context") if isinstance(latest, dict) else {}
        if not isinstance(market, dict) or not market:
            return {}

        regime = str(market.get("regime_label") or "NEUTRAL").upper()
        dispersion_score = float(market.get("dispersion_score", 0.0) or 0.0)
        sizing = float(market.get("sizing_multiplier", 1.0) or 1.0)
        if regime == "DISPERSION":
            summary = (
                "today was a DISPERSION regime; system played to its selection edge"
            )
        elif regime == "CORRELATION_HIGH":
            summary = "today was a CORRELATION_HIGH regime; broad-index risk dominated"
        else:
            summary = f"today was a {regime} regime"
        return {
            "regime_label": regime,
            "dispersion_score": dispersion_score,
            "sizing_multiplier": sizing,
            "summary_line": summary,
            "tape_inference": str(latest.get("tape_inference") or ""),
        }


def _build_feedback_from_review(review: Dict) -> Dict:
    """Build feedback metrics aligned to EOD review PnL semantics."""
    summary = review.get("day_summary", {}) if isinstance(review, dict) else {}
    positions = (
        review.get("position_performance", []) if isinstance(review, dict) else []
    )
    winners = sum(
        1 for p in positions if float(p.get("total_pnl_dollars", 0.0) or 0.0) > 0.0
    )
    losers = sum(
        1 for p in positions if float(p.get("total_pnl_dollars", 0.0) or 0.0) < 0.0
    )
    win_rate = winners / max(1, winners + losers)

    return {
        "date": str(review.get("review_date") or datetime.now().strftime("%Y-%m-%d")),
        "win_rate": win_rate,
        "total_pnl": float(summary.get("overall_pnl_dollars", 0.0) or 0.0),
    }


def main():
    """Run the daily review."""
    import argparse

    parser = argparse.ArgumentParser(description="AutoTrade Daily Review")
    parser.add_argument("--learn", action="store_true", help="Run EOD learning loop")
    args = parser.parse_args()

    logger.info("Starting Daily Review...")
    review = DailyReview()
    result = review.run(learn=args.learn)
    return result


if __name__ == "__main__":
    main()
