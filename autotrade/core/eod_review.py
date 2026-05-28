"""
End-of-Day Trade Review Module
==============================
Analyzes daily trade performance and generates feedback for research.
"""

import json
import logging
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Iterable

from autotrade.utils.alpaca_client_factory import (
    create_trading_client,
    resolve_alpaca_credentials,
)

logger = logging.getLogger(__name__)


class EODReview:
    """
    Automates the analysis of daily trade performance.
    """

    def __init__(self, dry_run: bool = False, date_str: str = None):
        self.dry_run = dry_run
        if date_str:
            self.date_str = date_str
        else:
            # Use the resolved session date — at 12:15 AM the relevant
            # "completed session" is yesterday's, not today's not-yet-
            # existed session. The same helper backs DailyReview's date
            # resolution.
            from autotrade.utils.market_time import resolve_session_review_date

            self.date_str = resolve_session_review_date().strftime("%Y-%m-%d")

        # Initialize Alpaca client
        creds = resolve_alpaca_credentials(require=False)
        if creds and not dry_run:
            self.client = create_trading_client(
                api_key=creds.api_key,
                secret_key=creds.secret_key,
                paper=bool(creds.paper),
            )
        else:
            self.client = None

    def run(self) -> Dict[str, Any]:
        """
        Run the full EOD review process.
        """
        logger.info(f"Starting EOD review for {self.date_str}")

        # 1. Fetch trades/positions
        positions = self._fetch_positions()
        broker_orders = self._fetch_broker_orders()
        broker_account = self._fetch_broker_account_snapshot()

        # 2. Match with signals
        matches = self._match_with_signals(positions)

        # 3. Calculate metrics
        metrics = self._calculate_metrics(matches)

        # 4. Score-bucket analysis
        buckets = self._analyze_score_buckets(matches)
        watchlist_causality = self._load_watchlist_causality()
        missed_watchlist = self._summarize_missed_watchlist_opportunities()

        results = {
            "date": self.date_str,
            "total_trades": len(matches),
            "open_positions_count": len(matches),
            "same_day_filled_orders_count": len(
                [order for order in broker_orders if order.get("status") == "filled"]
            ),
            "broker_orders_by_side": dict(
                Counter(order.get("side", "") for order in broker_orders)
            ),
            "broker_orders_by_status": dict(
                Counter(order.get("status", "") for order in broker_orders)
            ),
            "broker_day_pnl": broker_account.get("day_pnl_equity_delta"),
            "broker_equity": broker_account.get("equity"),
            "open_position_unrealized_pnl": round(
                sum(float(row.get("unrealized_pl", 0.0) or 0.0) for row in matches), 4
            ),
            "realized_day_pnl": round(
                (broker_account.get("day_pnl_equity_delta") or 0.0) - 
                sum(float(row.get("unrealized_pl", 0.0) or 0.0) for row in matches), 4
            ),
            "win_rate": metrics.get("win_rate", 0.0),
            "avg_pnl": metrics.get("avg_pnl", 0.0),
            "score_buckets": buckets,
            "best_trade": metrics.get("best_trade"),
            "worst_trade": metrics.get("worst_trade"),
            "watchlist_causality_summary": watchlist_causality.get("summary", {}),
            "missed_watchlist_opportunities": missed_watchlist,
            "broker_orders": broker_orders,
            "broker_account": broker_account,
            "trades": matches,
        }

        if not self.dry_run:
            self._save_results(results)

        return results

    def _fetch_positions(self) -> List[Dict[str, Any]]:
        """Fetch current positions from Alpaca."""
        if not self.client:
            # Mock data for dry_run or missing credentials
            return [
                {"symbol": "ADEA", "unrealized_pl": 150.0, "current_price": 21.0},
                {"symbol": "TSLA", "unrealized_pl": -50.0, "current_price": 240.0},
            ]

        try:
            from autotrade.utils.position_cache import fetch_positions_with_fallback

            result = fetch_positions_with_fallback(
                self.client,
                logger=logger,
                retries=2,
                retry_delay_seconds=1.0,
                backoff_multiplier=2.0,
                use_cache=True,
            )
            positions = result.get("positions") or []
            return [
                {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                }
                for p in positions
            ]
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    def _fetch_broker_account_snapshot(self) -> Dict[str, Any]:
        """Fetch account equity fields used for day-P/L truth separation."""
        if not self.client:
            return {}
        try:
            account = self.client.get_account()
            equity = self._safe_float(getattr(account, "equity", None))
            last_equity = self._safe_float(getattr(account, "last_equity", None))
            return {
                "equity": equity,
                "last_equity": last_equity,
                "day_pnl_equity_delta": (
                    round(equity - last_equity, 4)
                    if equity is not None and last_equity is not None
                    else None
                ),
                "cash": self._safe_float(getattr(account, "cash", None)),
                "buying_power": self._safe_float(
                    getattr(account, "buying_power", None)
                ),
                "status": str(getattr(account, "status", "") or ""),
            }
        except Exception as e:
            logger.warning(f"Failed to fetch broker account snapshot: {e}")
            return {"error": str(e)}

    def _fetch_broker_orders(self) -> List[Dict[str, Any]]:
        """Fetch same-day broker orders separately from open-position performance."""
        if not self.client:
            return []
        try:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest

            session_start = datetime.strptime(self.date_str, "%Y-%m-%d")
            session_end = session_start + timedelta(days=1)
            request = GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                after=session_start,
                until=session_end,
                limit=500,
            )
            orders = list(self.client.get_orders(filter=request) or [])
        except TypeError:
            request = GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                after=session_start,
                limit=500,
            )
            orders = list(self.client.get_orders(filter=request) or [])
        except Exception as e:
            logger.warning(f"Failed to fetch broker orders: {e}")
            return []

        rows: List[Dict[str, Any]] = []
        for order in orders:
            side = str(getattr(order, "side", "") or "").lower().split(".")[-1]
            status = str(getattr(order, "status", "") or "").lower().split(".")[-1]
            submitted_at = getattr(order, "submitted_at", None)
            filled_at = getattr(order, "filled_at", None)
            rows.append(
                {
                    "id": str(getattr(order, "id", "") or ""),
                    "symbol": str(getattr(order, "symbol", "") or "").upper(),
                    "side": side,
                    "status": status,
                    "qty": self._safe_float(getattr(order, "qty", None)),
                    "filled_qty": self._safe_float(getattr(order, "filled_qty", None)),
                    "filled_avg_price": self._safe_float(
                        getattr(order, "filled_avg_price", None)
                    ),
                    "submitted_at": submitted_at.isoformat()
                    if submitted_at is not None
                    else None,
                    "filled_at": filled_at.isoformat()
                    if filled_at is not None
                    else None,
                }
            )
        return rows

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _score_from_signal(signal: Dict[str, Any]) -> float:
        for key in ("final_score", "confidence", "score"):
            try:
                value = signal.get(key)
                if value is None:
                    continue
                return float(value)
            except Exception:
                continue
        return 0.0

    def _candidate_plan_paths(self) -> List[Path]:
        plans_dir = Path("plans")
        logs_dir = Path("logs")
        plan_date = self.date_str.replace("-", "")
        current_pm_date = self.date_str
        try:
            prev_date = (
                datetime.strptime(self.date_str, "%Y-%m-%d")
                .date()
                .fromordinal(
                    datetime.strptime(self.date_str, "%Y-%m-%d").date().toordinal() - 1
                )
            )
            prev_pm_date = prev_date.isoformat()
        except Exception:
            prev_pm_date = ""

        paths = [
            plans_dir / f"morning_game_plan_{plan_date}.json",
            plans_dir / f"pm_plan_{current_pm_date}.json",
        ]
        if prev_pm_date:
            paths.append(plans_dir / f"pm_plan_{prev_pm_date}.json")
        paths.extend(
            sorted(
                plans_dir.glob(f"adjusted_plan_{plan_date}*.json"),
                key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
                reverse=True,
            )
        )
        # Include signal log — covers symbols entered autonomously outside plans
        signals_path = logs_dir / f"signals_{current_pm_date}.json"
        if signals_path.exists():
            paths.append(signals_path)
        return paths

    def _load_scores_from_plan(self, plan_path: Path) -> Dict[str, Dict[str, Any]]:
        try:
            with open(plan_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load plan {plan_path}: {e}")
            return {}

        scores: Dict[str, Dict[str, Any]] = {}

        def _iter_signal_groups(
            raw_payload: Any,
        ) -> Iterable[tuple[str, List[Dict[str, Any]]]]:
            if isinstance(raw_payload, list):
                yield plan_path.name, raw_payload
                return
            if not isinstance(raw_payload, dict):
                return
            groups = [
                ("signals", raw_payload.get("signals")),
                ("buy_signals", raw_payload.get("buy_signals")),
                ("entry_candidates", raw_payload.get("entry_candidates")),
                ("actionable_top50", raw_payload.get("actionable_top50")),
                ("full_watchlist", raw_payload.get("full_watchlist")),
                (
                    "pm_report.top_entries",
                    (raw_payload.get("pm_report") or {}).get("top_entries"),
                ),
            ]
            for label, rows in groups:
                if isinstance(rows, list) and rows:
                    yield label, rows

        for score_source, signals in _iter_signal_groups(payload):
            for sig in signals:
                if not isinstance(sig, dict):
                    continue
                symbol = str(sig.get("symbol", sig.get("ticker", "")) or "").upper()
                if not symbol or symbol in scores:
                    continue
                scores[symbol] = {
                    "entry_score": self._score_from_signal(sig),
                    "plan_source": plan_path.name,
                    "entry_source": sig.get("entry_source", ""),
                    "score_source": score_source,
                }
        return scores

    def _match_with_signals(
        self, positions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Match positions with the best available plan signal scores."""
        scores: Dict[str, Dict[str, Any]] = {}
        used_sources: List[str] = []

        for plan_path in self._candidate_plan_paths():
            if not plan_path.exists():
                continue
            loaded_scores = self._load_scores_from_plan(plan_path)
            if loaded_scores:
                scores.update(
                    {k: v for k, v in loaded_scores.items() if k not in scores}
                )
                used_sources.append(plan_path.name)

        for p in positions:
            symbol = str(p.get("symbol", "") or "").upper()
            match = scores.get(symbol, {})
            score = match.get("entry_score")
            p["entry_score"] = float(score) if score is not None else None
            p["entry_score_available"] = score is not None
            p["plan_score_source"] = match.get("plan_source", "")
            p["signal_entry_source"] = match.get("entry_source", "")
            p["score_source"] = match.get("score_source", "")

        if used_sources:
            logger.info(
                "EOD score matching sources: %s",
                ", ".join(used_sources[:6]),
            )

        return positions

    def _watchlist_causality_path(self) -> Path:
        file_name = f"watchlist_causality_{self.date_str}.json"
        local_path = Path("logs") / file_name
        if local_path.exists():
            return local_path
        try:
            from config.config_loader import get_config

            return Path(get_config().logs_dir) / file_name
        except Exception:
            return local_path

    def _load_watchlist_causality(self) -> Dict[str, Any]:
        path = self._watchlist_causality_path()
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f) or {}
            return payload if isinstance(payload, dict) else {}
        except Exception as e:
            logger.error(f"Failed to load watchlist causality {path}: {e}")
            return {}

    def _summarize_missed_watchlist_opportunities(self) -> List[Dict[str, Any]]:
        payload = self._load_watchlist_causality()
        rows = payload.get("symbols", [])
        if not isinstance(rows, list):
            return []

        missed: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("executed"):
                continue
            status = str(row.get("current_status", "") or "")
            if status == "never_became_candidate" and not row.get("was_evaluated"):
                continue
            missed.append(
                {
                    "symbol": str(row.get("ticker", "") or "").upper(),
                    "status": status,
                    "blocking_reason": row.get("blocking_reason", ""),
                    "blocking_rule": row.get("blocking_rule", ""),
                    "entry_source": row.get("entry_source", ""),
                    "raw_score": float(row.get("raw_score", 0.0) or 0.0),
                    "validated_score": float(row.get("validated_score", 0.0) or 0.0),
                    "ranking_position": row.get("ranking_position"),
                    "displaced_by": row.get("displaced_by", ""),
                    "displaced_by_score": row.get("displaced_by_score"),
                    "missing_inputs": row.get("missing_inputs", []),
                    "phases_seen": row.get("phases_seen", []),
                    "was_evaluated": bool(row.get("was_evaluated", False)),
                }
            )

        missed.sort(
            key=lambda row: (
                1 if row.get("was_evaluated") else 0,
                1 if row.get("status") == "blocked" else 0,
                float(row.get("validated_score", 0.0) or 0.0),
                float(row.get("raw_score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        return missed[:15]

    def _calculate_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate performance metrics."""
        if not trades:
            return {}

        winning_trades = [t for t in trades if t.get("unrealized_pl", 0) > 0]
        win_rate = len(winning_trades) / len(trades)

        pnl_values = [t.get("unrealized_pl", 0) for t in trades]
        avg_pnl = sum(pnl_values) / len(trades)

        best_trade = max(trades, key=lambda x: x.get("unrealized_pl", 0))
        worst_trade = min(trades, key=lambda x: x.get("unrealized_pl", 0))

        return {
            "win_rate": round(win_rate, 3),
            "avg_pnl": round(avg_pnl, 2),
            "best_trade": best_trade.get("symbol"),
            "worst_trade": worst_trade.get("symbol"),
        }

    def _analyze_score_buckets(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Group trade outcomes by entry signal score ranges."""
        buckets = {
            "unavailable": {"count": 0, "win_rate": 0.0, "pnl_sum": 0.0},
            "0-35": {"count": 0, "win_rate": 0.0, "pnl_sum": 0.0},
            "35-50": {"count": 0, "win_rate": 0.0, "pnl_sum": 0.0},
            "50-65": {"count": 0, "win_rate": 0.0, "pnl_sum": 0.0},
            "65-80": {"count": 0, "win_rate": 0.0, "pnl_sum": 0.0},
            "80+": {"count": 0, "win_rate": 0.0, "pnl_sum": 0.0},
        }

        for t in trades:
            score = t.get("entry_score", 0)
            pnl = t.get("unrealized_pl", 0)

            if score is None:
                key = "unavailable"
            elif score < 35:
                key = "0-35"
            elif score < 50:
                key = "35-50"
            elif score < 65:
                key = "50-65"
            elif score < 80:
                key = "65-80"
            else:
                key = "80+"

            buckets[key]["count"] += 1
            buckets[key]["pnl_sum"] += pnl
            if pnl > 0:
                buckets[key]["win_rate"] += 1  # Will divide later

        for key in buckets:
            if buckets[key]["count"] > 0:
                buckets[key]["win_rate"] = round(
                    buckets[key]["win_rate"] / buckets[key]["count"], 3
                )
                buckets[key]["avg_pnl"] = round(
                    buckets[key]["pnl_sum"] / buckets[key]["count"], 2
                )

        return buckets

    def _save_results(self, results: Dict[str, Any]):
        """Save EOD review results to a JSON file."""
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        file_path = data_dir / f"eod_review_{self.date_str}.json"
        try:
            with open(file_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"EOD review saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save EOD review: {e}")
