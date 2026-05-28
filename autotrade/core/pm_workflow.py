"""
Post-Market Workflow - Agentic Analysis
========================================
Runs at 6:30 PM after market close.

COMPLETE POST-MARKET WORKFLOW:
1. Analyze current positions -> Identify exits for tomorrow
2. Generate new entry signals -> Agentic signal generation
3. Create next-day trading plan

Features:
- Fetches REAL positions from Alpaca
- Calculates conviction scores for each position
- Identifies exits for next morning
- Generates new entry watchlist (agentic)
- Saves complete plan for day_manager

Usage:
  python -m autotrade.execution.post_market_workflow              # Dry run (analysis only)
  python -m autotrade.execution.post_market_workflow --execute    # Save plan for tomorrow
"""

import json
import logging
import sys
import os
import time
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.config_loader import get_config
from autotrade.core.position_caps import resolve_position_caps
from autotrade.utils.market_time import (
    get_market_now,
    get_pm_plan_date,
    last_trading_day,
)
from autotrade.backtesting.strategy_validator import StrategyValidator
from autotrade.utils.alpaca_client_factory import (
    create_data_client,
    create_trading_client,
    resolve_alpaca_credentials,
)
from autotrade.signals.support_resistance import estimate_sr_levels

try:
    from autotrade.utils.intraday_data_provider import get_intraday_bars

    INTRADAY_PROVIDER_AVAILABLE = True
except Exception:
    get_intraday_bars = None
    INTRADAY_PROVIDER_AVAILABLE = False

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = Path(
    os.environ.get("AUTOTRADE_ROOT", Path(__file__).resolve().parents[2])
)
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
PLANS_DIR = PROJECT_DIR / "plans"
PLANS_DIR.mkdir(exist_ok=True)

# Setup safe logging (Windows console compatible)
try:
    from autotrade.utils.safe_logging import get_safe_logger, safe_exception_text

    logger = get_safe_logger(
        "post_market_workflow",
        LOG_DIR / f"post_market_workflow_{datetime.now().strftime('%Y-%m-%d')}.log",
    )
except ImportError:

    def safe_exception_text(exc, max_len=500):
        return str(exc)

    def setup_logging():
        log_format = "%(asctime)s | %(levelname)s | %(message)s"
        logger = logging.getLogger("post_market_workflow")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console)

        file_handler = logging.FileHandler(
            LOG_DIR / f"post_market_workflow_{datetime.now().strftime('%Y-%m-%d')}.log",
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


class PostMarketWorkflow:
    """Post-market agentic workflow for position analysis."""

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        creds = resolve_alpaca_credentials(require=True)
        self.api_key = creds.api_key
        self.secret_key = creds.secret_key
        self.paper = bool(creds.paper)

        # Parallelization config
        self.max_workers = int(os.environ.get("PM_WORKFLOW_MAX_WORKERS", "4"))
        self.enable_parallel = (
            os.environ.get("PM_WORKFLOW_ENABLE_PARALLEL", "true").lower() == "true"
        )

        if self.enable_parallel:
            logger.info(f"Parallelization enabled (max_workers={self.max_workers})")
        else:
            logger.info("Parallelization disabled (sequential mode)")

        # Initialize Alpaca clients
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

        # Initialize components
        from autotrade.core.overnight_agent import OvernightAgent
        from autotrade.signals.conviction_engine import ConvictionEngine
        from autotrade.risk.risk_gate import RiskGate
        from autotrade.risk.strategy_failsafe import StrategyFailsafeManager
        from autotrade.risk.day_trade_tracker import DayTradeTracker
        from autotrade.signals.unified_strategy import UnifiedStrategy, MIN_LESSON_SCORE

        self.overnight_agent = OvernightAgent()
        self.conviction_engine = ConvictionEngine()

        # Unified strategy for entry/exit using multi-level approach
        # Backtested: +52% better P&L, +51% better avg trade, PF 1.66
        self.strategy = UnifiedStrategy(min_score=MIN_LESSON_SCORE)
        logger.info(f"UnifiedStrategy loaded (min_score={MIN_LESSON_SCORE})")
        self.risk_gate = RiskGate()
        self.strategy_failsafe = StrategyFailsafeManager()
        self.strategy_failsafe_snapshot = self.strategy_failsafe.load_snapshot()
        self.day_tracker = DayTradeTracker()

        self.youtube_context = {}  # Populated in run() by _ensure_youtube_intelligence()
        self.regime_detector = None
        self.current_regime_analysis = None
        self.regime_strategy_overrides = {}
        self.regime_router = None
        self.regime_router_output: Dict[str, Any] = {
            "available": False,
            "regime": "NEUTRAL",
            "confidence": 0.0,
            "method": "disabled",
            "error": None,
        }
        self.resolved_regime_output: Dict[str, Any] = {}
        try:
            from autotrade.analysis.market_regime import MarketRegimeDetector

            self.regime_detector = MarketRegimeDetector()
            logger.info("Quantitative Regime Detection initialized")
        except Exception as e:
            logger.warning(f"Could not initialize Market Regime Detector: {e}")
        if bool(getattr(get_config().signal_generation, "enable_regime_router", True)):
            try:
                from autotrade.signals.regime_router import RegimeRouter

                self.regime_router = RegimeRouter(parent_logger=logger)
                self.regime_router_output["method"] = "router"
                logger.info("Signal RegimeRouter initialized")
            except Exception as e:
                self.regime_router = None
                self.regime_router_output["method"] = "unavailable"
                self.regime_router_output["error"] = str(e)
                logger.warning(f"Signal RegimeRouter unavailable: {e}")
        self.position_levels: Dict[str, Dict[str, float]] = {}
        self._last_validated_strategy_count: int = 0
        self._last_merged_candidate_count: int = 0
        self._last_strategy_routing_diagnostics: Dict[str, Any] = {}
        self._asset_eligibility_cache: Dict[str, Dict[str, Any]] = {}
        logger.info(f"PM Workflow initialized (dry_run={dry_run})")

    def get_positions(self) -> List:
        """Get current Alpaca positions."""
        from autotrade.utils.position_cache import fetch_positions_with_fallback

        result = fetch_positions_with_fallback(
            self.client,
            logger=logger,
            retries=2,
            retry_delay_seconds=1.0,
            backoff_multiplier=2.0,
            use_cache=True,
        )
        return list(result.get("positions") or [])

    def get_account(self):
        """Get account info."""
        return self.client.get_account()

    def _load_overnight_watchlist_context(
        self, open_slots: int, log: logging.Logger
    ) -> Dict[str, Any]:
        """Load a fresh overnight watchlist and return ranked restrict-ticker context."""
        state_path = PROJECT_DIR / "research" / "overnight_state.json"
        context: Dict[str, Any] = {
            "used": False,
            "reason": "not_checked",
            "state_path": str(state_path),
            "state_date": None,
            "symbols_considered": 0,
            "symbols_loaded": [],
        }

        if open_slots <= 0:
            context["reason"] = "no_open_slots"
            return context

        if not state_path.exists():
            context["reason"] = "missing_state_file"
            return context

        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as e:
            context["reason"] = "state_load_failed"
            context["error"] = safe_exception_text(e)
            return context

        if not bool(state.get("research_complete")):
            context["reason"] = "research_incomplete"
            return context

        watchlist = state.get("watchlist") or []
        if not isinstance(watchlist, list) or not watchlist:
            context["reason"] = "empty_watchlist"
            return context

        state_date_raw = str(state.get("date") or "").strip()
        context["state_date"] = state_date_raw or None
        try:
            state_date = (
                datetime.fromisoformat(state_date_raw).date()
                if state_date_raw
                else None
            )
        except ValueError:
            state_date = None

        plan_date = get_pm_plan_date()
        # Build a window of recent acceptable dates (handles weekends + holidays).
        acceptable_dates = set()
        cursor = plan_date
        for _ in range(5):
            acceptable_dates.add(last_trading_day(cursor))
            cursor -= timedelta(days=1)
        if state_date is None or state_date not in acceptable_dates:
            context["reason"] = "stale_watchlist"
            context["acceptable_dates"] = sorted(
                d.isoformat() for d in acceptable_dates
            )
            return context

        actionable_recommendations = {"BUY", "WEAK BUY", "STRONG BUY"}
        ranked_watchlist = []
        for item in watchlist:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol") or item.get("ticker") or "").strip().upper()
            if not symbol:
                continue
            if (
                str(item.get("recommendation") or "").strip().upper()
                not in actionable_recommendations
            ):
                continue
            if bool(item.get("validation_gated")):
                continue
            ranked_watchlist.append(item)

        if not ranked_watchlist:
            context["reason"] = "no_actionable_watchlist_symbols"
            return context

        ranked_watchlist.sort(
            key=lambda item: (
                float(item.get("conviction_priority_score", 0.0) or 0.0),
                float(item.get("ranking_score", 0.0) or 0.0),
                float(item.get("confidence", 0.0) or 0.0),
            ),
            reverse=True,
        )

        # Bridge cap. Historically max(25, open_slots * 4) — with 6 positions
        # held and 50 max, that capped the bridge at 176 symbols even when the
        # overnight produced 400+. The user wants the full actionable watchlist
        # to flow through; downstream gates (conviction, posture, position cap,
        # failsafe, day manager) constrain *actual entries*, so it's safe to
        # surface the entire BUY/WEAK BUY/STRONG BUY pool here.
        # Override via AUTOTRADE_PM_BRIDGE_LIMIT (integer, "0"/"all" = no cap).
        bridge_limit_cfg = int(
            getattr(get_config().plan_caps, "pm_bridge_limit", 0) or 0
        )
        bridge_limit_env = (
            os.environ.get("AUTOTRADE_PM_BRIDGE_LIMIT", "").strip().lower()
        )
        bridge_limit_raw = bridge_limit_env or str(bridge_limit_cfg)
        if bridge_limit_raw.lower() in {"0", "all", "none", "unlimited"}:
            symbol_limit = len(ranked_watchlist)
        else:
            try:
                symbol_limit = min(
                    len(ranked_watchlist),
                    max(int(bridge_limit_raw), open_slots * 4),
                )
            except ValueError:
                symbol_limit = len(ranked_watchlist)
        symbols: List[str] = []
        seen = set()
        for item in ranked_watchlist[:symbol_limit]:
            symbol = str(item.get("symbol") or item.get("ticker") or "").strip().upper()
            if symbol and symbol not in seen:
                seen.add(symbol)
                symbols.append(symbol)

        if not symbols:
            context["reason"] = "no_ranked_symbols"
            return context

        context.update(
            {
                "used": True,
                "reason": "fresh_overnight_watchlist",
                "symbols_considered": len(ranked_watchlist),
                "symbols_loaded": symbols,
            }
        )
        log.info(
            "[PM WORKFLOW] Using %d ranked overnight watchlist symbols from %s (state_date=%s, actionable=%d)",
            len(symbols),
            state_path.name,
            state_date_raw or "unknown",
            len(ranked_watchlist),
        )
        return context

    def run(self) -> Dict:
        """
        Run the PM workflow.

        Returns next-day plan with:
        - positions: current positions with analysis
        - morning_exits: positions to exit at open
        - morning_adds: positions to add to
        - watchlist: new positions to watch
        - rotation_plan: capital rotation suggestions
        """
        logger.info("=" * 60)
        logger.info("PM WORKFLOW - Post-Market Analysis")
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        self._last_validated_strategy_count = 0
        self._last_merged_candidate_count = 0
        self._last_strategy_routing_diagnostics = {}

        # === STEP -1: YouTube Intelligence Scan ===
        # Ensure we have fresh YouTube channel data for next-day planning
        self.youtube_context = self._ensure_youtube_intelligence()
        self._refresh_quantitative_regime()
        regime_router_snapshot = self._detect_signal_router_regime()
        quantitative_label = self._effective_market_regime().upper()
        router_label = str(
            regime_router_snapshot.get("regime", "NEUTRAL") or "NEUTRAL"
        ).upper()
        regime_divergence = {
            "labels_match": quantitative_label == router_label,
            "quantitative_regime": quantitative_label,
            "regime_router_regime": router_label,
        }
        if (
            regime_router_snapshot.get("available")
            and quantitative_label != router_label
        ):
            logger.warning(
                "[REGIME DIVERGENCE] quantitative=%s vs regime_router=%s (diagnostic only)",
                quantitative_label,
                router_label,
            )

        # Get account state
        account = self.get_account()
        equity = float(account.equity)
        buying_power = float(account.buying_power)
        day_trades_remaining = self.day_tracker.get_remaining()

        portfolio_cfg = get_config().portfolio

        logger.info("\n[ACCOUNT STATUS]")
        logger.info(f"   Equity: ${equity:,.2f}")
        logger.info(f"   Buying Power: ${buying_power:,.2f}")
        logger.info(f"   Day Trades Remaining: {day_trades_remaining}")

        # Strategy-level health check before signal generation
        strategy_health = None
        strategy_optimal = None
        try:
            strat_validator = StrategyValidator(parent_logger=logger)
            strategy_health = strat_validator.validate_from_strategy_lab()
            # Quick parameter suggestion (non-blocking, lightweight grid)
            strategy_optimal = strat_validator.find_optimal_parameters(lookback_days=40)
        except Exception as e:
            logger.warning(f"Strategy validation skipped: {e}")

        # Update and persist failsafe state from latest validation + equity curve.
        self.strategy_failsafe_snapshot = (
            self.strategy_failsafe.update_from_strategy_validation(
                strategy_validation=strategy_health,
                equity=equity,
                source="pm_workflow",
                as_of_date=datetime.now().strftime("%Y-%m-%d"),
            )
        )
        logger.info(
            "[FAILSAFE] level=%s | halt_entries=%s | max_positions=%s | stop_mult=%.2f | session_dd=%.1f%% | peak_dd=%.1f%%",
            self.strategy_failsafe_snapshot.level.upper(),
            self.strategy_failsafe_snapshot.halt_new_entries,
            self.strategy_failsafe_snapshot.max_positions,
            self.strategy_failsafe_snapshot.stop_multiplier,
            float(
                getattr(
                    self.strategy_failsafe_snapshot, "session_drawdown_pct", 0.0
                )
                or 0.0
            ),
            float(
                getattr(
                    self.strategy_failsafe_snapshot,
                    "peak_drawdown_pct",
                    getattr(self.strategy_failsafe_snapshot, "drawdown_pct", 0.0),
                )
                or 0.0
            ),
        )

        # Get positions
        positions = self.get_positions()

        if not positions:
            logger.info(
                "\n[!] No positions to analyze - continuing with flat-book signal generation"
            )

        # === STEP 0: Nightly DB Update (Held Positions) ===
        if not self.dry_run:
            logger.info(
                f"\n[DB UPDATE] Refreshing financial data for {len(positions)} positions..."
            )
            try:
                subprocess.run(
                    [sys.executable, "tools/update_financial_db.py", "--held-only"],
                    check=False,  # Don't crash if update fails
                    timeout=300,  # 5 min max
                )
            except Exception as e:
                logger.warning(f"   DB update failed: {e}")

        logger.info(f"\n[ANALYZING {len(positions)} POSITIONS]")
        logger.info("-" * 50)

        # Analyze each position (parallelized or sequential)
        position_analyses = []
        morning_exits = []
        morning_adds = []
        holds = []

        start_time = time.time()

        if self.enable_parallel and len(positions) > 1:
            # Parallel execution
            logger.info(f"   [PARALLEL MODE] Using {self.max_workers} workers")

            # Submit all positions for parallel analysis
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit tasks and maintain order
                future_to_pos = {
                    executor.submit(self._analyze_position, pos): pos
                    for pos in positions
                }

                # Collect results in original order
                for future in as_completed(future_to_pos):
                    try:
                        analysis = future.result()
                        position_analyses.append(analysis)
                    except Exception as e:
                        pos = future_to_pos[future]
                        logger.error(f"   Error analyzing {pos.symbol}: {e}")

            # Sort to maintain deterministic order by symbol
            position_analyses.sort(key=lambda x: x["symbol"])

        else:
            # Sequential execution (fallback or single position)
            if not self.enable_parallel:
                logger.info("   [SEQUENTIAL MODE] Parallelization disabled")

            for pos in positions:
                try:
                    analysis = self._analyze_position(pos)
                    position_analyses.append(analysis)
                except Exception as e:
                    logger.error(f"   Error analyzing {pos.symbol}: {e}")

        elapsed = time.time() - start_time
        logger.info(f"   Analysis completed in {elapsed:.2f}s")

        # Categorize by action
        for analysis in position_analyses:
            action = analysis["recommended_action"]
            if action in ["exit", "exit_immediately"]:
                morning_exits.append(analysis)
            elif action == "add":
                morning_adds.append(analysis)
            else:
                holds.append(analysis)

        # Generate overnight analysis
        logger.info("\n[OVERNIGHT ANALYSIS]")
        logger.info("-" * 50)

        # Simple overnight plan based on our analysis
        overnight_plan = {
            "exit_at_open": [p["symbol"] for p in morning_exits],
            "add_at_open": [p["symbol"] for p in morning_adds],
            "watch_closely": [p["symbol"] for p in holds if p["conviction_score"] < 50],
        }

        # === STEP 2: Generate new entry signals ===
        logger.info("\n[GENERATING NEW ENTRY SIGNALS]")
        logger.info("-" * 50)

        entry_candidates = []
        heatmap_results: List[Dict] = []
        portfolio_full = False
        pm_screener_override: Dict[str, Any] = self._build_pm_screener_override()
        overnight_watchlist_context: Dict[str, Any] = {
            "used": False,
            "reason": "signal_generation_not_started",
            "symbols_considered": 0,
            "symbols_loaded": [],
        }
        signal_pipeline_trace: Dict[str, Any] = {
            "generated_raw": 0,
            "after_strategy_merge": 0,
            "after_validation": 0,
            "after_asset_filter": 0,
            "after_staleness_filter": 0,
            "final_published": 0,
            "asset_dropped_symbols": [],
            "staleness_dropped_symbols": [],
        }

        # === DATA FRESHNESS: Rebuild parquet from DownDay if source .h5 is newer ===
        # DownDay runs at 3:30 PM daily; PM Workflow runs at 6:30 PM, so .h5 is always fresh.
        # ensure_parquet_current() is mtime-gated — safe to call every run (no-ops if current).
        try:
            _downday_dir = os.getenv("DOWNDAY_ROOT", "data/downday")
            if os.path.exists(_downday_dir):
                _rebuild = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "from full_rebuild import ensure_parquet_current; ensure_parquet_current()",
                    ],
                    cwd=_downday_dir,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=120,
                )
                _stdout = (_rebuild.stdout or "").strip()
                _stderr = (_rebuild.stderr or "").strip()
                _rebuild_msg = (
                    (_stdout or "OK")
                    if _rebuild.returncode == 0
                    else (_stderr[:300] or "non-zero exit")
                )
                logger.info(f"   Parquet freshness check: {_rebuild_msg}")
            else:
                logger.warning(
                    f"   DownDay directory not found at {_downday_dir} — skipping parquet rebuild"
                )
        except Exception as _rebuild_err:
            logger.warning(
                f"   Parquet rebuild check failed (non-fatal): {_rebuild_err}"
            )

        # === DATA QUALITY GATE: Validate data before generating signals ===
        try:
            from autotrade.utils.data_quality import validate_data_for_signals

            should_proceed, quality_report = validate_data_for_signals()

            if not should_proceed:
                logger.warning("DATA QUALITY GATE BLOCKED signal generation")
                logger.warning(
                    f"  Latest data: {quality_report.latest_date}, expected: {quality_report.expected_date}"
                )
                logger.warning(
                    f"  Quality score: {quality_report.quality_score:.0f}/100, issues: {len(quality_report.issues)}"
                )
                for issue in quality_report.issues[:3]:
                    logger.warning(f"  - {issue}")
                # Don't abort completely - still try to generate.
            else:
                if quality_report.recommendation == "CAUTION":
                    logger.warning(f"Data quality caution: {quality_report.issues[:2]}")
                else:
                    logger.info(
                        f"Data quality OK: {quality_report.quality_score:.0f}/100, {quality_report.ticker_count} tickers"
                    )

        except Exception as e:
            logger.warning(f"Data quality check failed: {e}, proceeding without gate")

        # PM evening signal generation is OFF by default — the overnight
        # research workflow (8 PM - 4 AM) produces a much richer plan
        # (morning_game_plan_*.json, ~88 buy_signals + ~400 watchlist) that
        # supersedes anything the PM screen could compile from yesterday's
        # post-close data. Setting AUTOTRADE_PM_GENERATE_SIGNALS=1 re-enables
        # the legacy 5-pick PM screen for diagnostic comparison.
        pm_generate_signals = str(
            os.environ.get("AUTOTRADE_PM_GENERATE_SIGNALS", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        if not pm_generate_signals:
            logger.info(
                "[PM WORKFLOW] PM signal generation disabled — deferring to "
                "overnight research + morning_game_plan. Set "
                "AUTOTRADE_PM_GENERATE_SIGNALS=1 to re-enable."
            )
            overnight_watchlist_context["reason"] = "pm_signal_generation_disabled"

        try:
            from autotrade.signals.agentic_signal_generator import (
                AgenticSignalGenerator,
            )

            # Calculate how many slots we'll have after exits
            exits_planned = len(morning_exits)
            current_positions = len(positions)
            max_positions = self._effective_max_positions()
            open_slots = max_positions - current_positions + exits_planned

            if self.strategy_failsafe_snapshot.halt_new_entries:
                logger.warning(
                    "[FAILSAFE] New entries halted (level=%s). Signals generated for monitoring only.",
                    self.strategy_failsafe_snapshot.level.upper(),
                )
                # Still generate signals for monitoring/planning, but tag them as non-actionable
                # open_slots stays positive so signal generation runs

            logger.info(
                f"   Current: {current_positions} | Exits: {exits_planned} | "
                f"Open slots: {open_slots} | Max positions cap: {max_positions}"
            )

            # Always load overnight watchlist and generate signals, even when
            # portfolio is full.  When open_slots <= 0 these become replacement
            # candidates the Day Manager can rotate into as positions are exited.
            effective_slots = max(open_slots, 5)  # at least 5 replacement candidates
            portfolio_full = open_slots <= 0

            if portfolio_full:
                logger.info(
                    "   Portfolio full (%d positions, max %d) — generating replacement candidates",
                    current_positions,
                    max_positions,
                )

            if pm_generate_signals:
                screener_enabled = get_config().screener_v2.enabled
                overnight_watchlist_context = self._load_overnight_watchlist_context(
                    open_slots=effective_slots,
                    log=logger,
                )
                restrict_tickers = (
                    overnight_watchlist_context.get("symbols_loaded") or None
                )
                max_candidates = min(effective_slots + 5, 15)
                if restrict_tickers:
                    max_candidates = min(
                        max(len(restrict_tickers), effective_slots + 10), 30
                    )
                else:
                    logger.info(
                        "[PM WORKFLOW] Overnight watchlist bridge unavailable (%s); falling back to fresh universe screen",
                        overnight_watchlist_context.get("reason", "unknown"),
                    )
                signal_gen = AgenticSignalGenerator(
                    max_candidates=max_candidates,
                    use_llm=True,
                    use_lessons=True,  # Apply learned patterns from backtest
                    use_screener_v2=screener_enabled,
                    parent_logger=logger,
                    restrict_tickers=restrict_tickers,
                    screener_config_override=pm_screener_override,
                )
                logger.info(
                    "   [BACKTEST HOOKS] strategy_validator=ON | lessons_filter=ON | "
                    "signal_validator=ON | multi_strategy_merge=ON"
                )
                entry_candidates = signal_gen.run()
                signal_pipeline_trace["generated_raw"] = len(entry_candidates or [])

                # === Multi-strategy screening: add signals from validated strategies ===
                entry_candidates = self._merge_multi_strategy_signals(
                    entry_candidates, open_slots, logger
                )
                signal_pipeline_trace["after_strategy_merge"] = len(
                    entry_candidates or []
                )

                # === Backtest validation: reject signals with poor historical lookalikes ===
                try:
                    val_cfg = get_config().signal_validation
                    if val_cfg.enabled and entry_candidates:
                        from autotrade.backtesting.signal_validator import (
                            SignalValidator,
                        )

                        validator = SignalValidator(parent_logger=logger)
                        entry_candidates, val_meta = validator.validate_and_filter(
                            entry_candidates,
                            logger=logger,
                            adjust_scores=True,
                            weight_signal=getattr(val_cfg, "score_weight_signal", 1.0),
                            weight_backtest=getattr(
                                val_cfg, "score_weight_backtest", 0.3
                            ),
                            portfolio_cfg=portfolio_cfg,
                            build_heatmap=True,
                            return_details=True,
                        )
                        rejected = val_meta.get("rejected", 0)
                        heatmap_results = val_meta.get("heatmap", [])
                        logger.info(
                            f"   Signal validation: {len(entry_candidates)} passed, {rejected} rejected "
                            f"(min_score={val_cfg.min_backtest_score}, min_matches={val_cfg.min_similar_signals})"
                        )
                except Exception as e:
                    logger.warning(f"   Signal validation failed (non-fatal): {e}")
                signal_pipeline_trace["after_validation"] = len(entry_candidates or [])

                if entry_candidates:
                    # === SIGNAL BATCH VALIDATION: Detect flat scores ===
                    try:
                        from autotrade.utils.data_quality import validate_signal_batch

                        signals_for_validation = [
                            c.to_dict() if hasattr(c, "to_dict") else c
                            for c in entry_candidates
                        ]
                        is_valid, batch_report = validate_signal_batch(
                            signals_for_validation
                        )

                        if not is_valid:
                            logger.error(
                                f"Signal batch validation FAILED: {batch_report.issues[:3]}"
                            )
                        else:
                            logger.info(
                                f"Signal batch valid: {batch_report.stats.get('count', 0)} signals, "
                                f"score range: {batch_report.stats.get('score_min', 0):.1f}-{batch_report.stats.get('score_max', 0):.1f}"
                            )

                    except Exception as e:
                        logger.warning(f"Signal batch validation failed: {e}")

                    self._log_signal_candidate_trace(entry_candidates, logger)

                    # Save signals BEFORE printing so a display crash can't block persistence
                    if not self.dry_run:
                        signal_gen.save_signals(
                            entry_candidates,
                            source="pm_workflow",
                        )

                    try:
                        signal_gen.print_summary(entry_candidates)
                    except Exception as _print_err:
                        logger.warning(
                            f"Signal summary display failed (non-fatal): {_print_err}"
                        )
            # (open_slots gate removed — signals always generated above)

        except Exception as e:
            logger.warning("   Signal generation failed: %s", safe_exception_text(e))

        # Build final plan with CANONICAL SCHEMA
        # Primary key: 'signals' (not 'entry_candidates' or 'buy_signals')
        if entry_candidates:
            entry_candidates.sort(
                key=lambda c: getattr(c, "final_score", 0), reverse=True
            )
            # Tag replacement candidates so execution layer knows context
            if portfolio_full:
                for c in entry_candidates:
                    if hasattr(c, "metadata"):
                        c.metadata = {
                            **(c.metadata or {}),
                            "replacement_candidate": True,
                        }
                    elif isinstance(c, dict):
                        c["replacement_candidate"] = True
        # Publish cap. Historical default was 10, which starved the morning
        # plan even when 30+ validated signals were available. Default lifted
        # to 75; override via AUTOTRADE_PM_PLAN_SIGNALS_CAP ("0"/"all" = no cap).
        cap_cfg = int(getattr(get_config().plan_caps, "pm_plan_signals_cap", 0) or 0)
        cap_env = os.environ.get("AUTOTRADE_PM_PLAN_SIGNALS_CAP", "").strip().lower()
        cap_raw = cap_env or str(cap_cfg)
        if cap_raw.lower() in {"0", "all", "none", "unlimited"}:
            cap = len(entry_candidates)
        else:
            try:
                cap = max(1, int(cap_raw))
            except ValueError:
                cap = 75
        if entry_candidates and len(entry_candidates) > cap:
            logger.info(
                f"   [PLAN] Capping signals_list to {cap} (had "
                f"{len(entry_candidates)} validated candidates). Override via "
                f"AUTOTRADE_PM_PLAN_SIGNALS_CAP."
            )
        signals_list = [
            c.to_dict() if hasattr(c, "to_dict") else c for c in entry_candidates[:cap]
        ]

        # Ensure scores and priorities are properly ranked (no flat scores/99s)
        for sig in signals_list:
            if sig.get("score") is None:
                sig["score"] = sig.get("final_score", sig.get("confidence", 0))
        signals_list.sort(
            key=lambda s: (
                s.get("score", 0),
                s.get("volume_ratio", 0),
                s.get("atr_percent", 0),
                s.get("risk_reward", 0),
            ),
            reverse=True,
        )
        signals_list, dropped_symbols = self._filter_plan_signals_by_asset_status(
            signals_list, log=logger
        )
        signal_pipeline_trace["after_asset_filter"] = len(signals_list or [])
        signal_pipeline_trace["asset_dropped_symbols"] = list(dropped_symbols or [])

        # === STALENESS FILTER: Purge symbols appearing 4+ days without execution ===
        # Note: pass `positions` (list), not `current_positions` (int count
        # that shadows it on line 646). Filter iterates the list.
        signals_list, staleness_dropped = self._update_and_filter_stale_signals(
            signals_list, positions, log=logger
        )
        signal_pipeline_trace["after_staleness_filter"] = len(signals_list or [])
        signal_pipeline_trace["staleness_dropped_symbols"] = list(
            staleness_dropped or []
        )
        if staleness_dropped:
            dropped_symbols = list(set(dropped_symbols + staleness_dropped))

        plan_score_source = (
            f"morning_game_plan_{get_pm_plan_date().strftime('%Y%m%d')}.json"
        )
        for i, sig in enumerate(signals_list):
            sig["priority"] = i + 1
            sig.setdefault("score", sig.get("final_score", sig.get("confidence", 0)))
            sig.setdefault("entry_source", "overnight_plan")
            sig.setdefault("source_bucket", "watchlist")
            sig.setdefault("plan_score_source", plan_score_source)
        signal_pipeline_trace["final_published"] = len(signals_list or [])

        quantitative_regime = {
            "available": bool(self.current_regime_analysis),
            "regime": self._effective_market_regime(),
            "confidence": 0.0,
            "breadth_pct_positive": None,
            "pattern_detected": None,
            "strategy_overrides": dict(self.regime_strategy_overrides or {}),
            "detected_at": None,
        }
        if self.current_regime_analysis:
            quantitative_regime.update(
                {
                    "regime": self.current_regime_analysis.regime.value.upper(),
                    "confidence": float(self.current_regime_analysis.confidence),
                    "breadth_pct_positive": float(
                        self.current_regime_analysis.breadth_pct_positive
                    ),
                    "pattern_detected": self.current_regime_analysis.pattern_detected,
                    "detected_at": (
                        self.current_regime_analysis.detection_timestamp.isoformat()
                        if getattr(
                            self.current_regime_analysis, "detection_timestamp", None
                        )
                        else None
                    ),
                }
            )

        entry_constraints = self._build_entry_constraints()
        entry_constraints.update(
            {
                "position_size_multiplier": self._regime_override_float(
                    "position_size_multiplier", 1.0
                ),
                "stop_multiplier": self._effective_stop_multiplier(fallback=2.0),
            }
        )

        plan = {
            "generated_at": datetime.now().isoformat(),
            "market_intelligence": {
                "regime": self.youtube_context.get("regime", "NEUTRAL"),
                "regime_confidence": self.youtube_context.get("regime_confidence", 0),
                "sizing_multiplier": self.youtube_context.get("sizing_multiplier", 1.0),
                "avoid_sectors": self.youtube_context.get("avoid_sectors", []),
                "favor_sectors": self.youtube_context.get("favor_sectors", []),
                "smallcap_ok": self.youtube_context.get("smallcap_ok", True),
                "directives": self.youtube_context.get("directives", []),
                "report_date": self.youtube_context.get("report_date", "none"),
                "available": self.youtube_context.get("available", False),
            },
            "quantitative_regime": quantitative_regime,
            "regime_router": regime_router_snapshot,
            "regime_divergence": regime_divergence,
            "entry_constraints": entry_constraints,
            "account": {
                "equity": equity,
                "buying_power": buying_power,
                "day_trades_remaining": day_trades_remaining,
            },
            "strategy_validation": strategy_health.__dict__
            if strategy_health
            else None,
            "strategy_failsafe": self.strategy_failsafe_snapshot.to_dict()
            if self.strategy_failsafe_snapshot
            else None,
            "strategy_optimal_params": strategy_optimal,
            "strategy_routing_diagnostics": dict(
                self._last_strategy_routing_diagnostics or {}
            ),
            "signal_heat_map": heatmap_results,
            "signal_pipeline_trace": signal_pipeline_trace,
            "pm_screener_override": dict(pm_screener_override or {}),
            "overnight_watchlist_bridge": overnight_watchlist_context,
            "positions": position_analyses,
            "morning_exits": morning_exits,
            "morning_adds": morning_adds,
            "holds": holds,
            "overnight_plan": overnight_plan,
            "signals": signals_list,  # CANONICAL: Use 'signals' key
            "entry_candidates": signals_list,  # Legacy support (backward compat)
            "summary": self._generate_summary(
                position_analyses, morning_exits, morning_adds
            ),
        }
        if dropped_symbols:
            plan["summary"] = dict(plan.get("summary") or {})
            plan["summary"]["asset_hygiene_dropped_symbols"] = dropped_symbols
            if staleness_dropped:
                plan["summary"]["staleness_purged_symbols"] = staleness_dropped

        # Validate schema
        try:
            from autotrade.utils.plan_schema_validator import validate_plan_schema

            is_valid, errors = validate_plan_schema(plan, strict=False)
            if not is_valid:
                logger.warning(f"Plan validation warnings: {'; '.join(errors)}")
        except Exception as e:
            logger.debug(f"Schema validation skipped: {e}")

        # Print summary
        self._print_plan(plan)

        # Save plan
        plan_artifact_path = None
        plan_artifact_status = "dry_run_not_saved"
        if not self.dry_run:
            plan_artifact_path = self._save_plan(plan)
            plan_artifact_status = (
                "saved"
                if plan_artifact_path is not None and Path(plan_artifact_path).exists()
                else "save_failed"
            )

        merged_count = (
            self._last_merged_candidate_count
            if self._last_merged_candidate_count
            else len(signals_list)
            if isinstance(signals_list, list)
            else 0
        )
        logger.info(
            "[PM WORKFLOW][COMPLETE] validated_strategies=%d | merged_candidates=%d | "
            "plan_artifact=%s | status=%s",
            self._last_validated_strategy_count,
            merged_count,
            str(plan_artifact_path) if plan_artifact_path else "none",
            plan_artifact_status,
        )

        shadow_result = self._run_sequential_shadow_eval_batch()
        if isinstance(shadow_result, dict):
            plan["sequential_shadow_eval"] = shadow_result

        return plan

    def _update_and_filter_stale_signals(
        self,
        signals: List[Dict[str, Any]],
        current_positions: List,
        log: logging.Logger,
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        Track consecutive days symbols appear on the watchlist without execution.
        Purge symbols reaching the 4-day threshold as recommended in daily_trades.
        """
        staleness_file = PROJECT_DIR / "data" / "watchlist_staleness.json"
        staleness_data = {}
        if staleness_file.exists():
            try:
                with open(staleness_file, "r", encoding="utf-8") as f:
                    staleness_data = json.load(f)
            except Exception as e:
                log.warning(f"Could not load watchlist_staleness.json: {e}")

        # Current symbols in plan
        plan_symbols = {
            str(s.get("symbol") or s.get("ticker", "")).upper() for s in signals
        }
        plan_symbols.discard("")

        # Current symbols held (don't increment staleness for these)
        held_symbols = {
            str(
                getattr(p, "symbol", p.get("symbol", "") if isinstance(p, dict) else "")
            ).upper()
            for p in current_positions
        }
        held_symbols.discard("")

        today_str = datetime.now().strftime("%Y-%m-%d")

        filtered_signals = []
        dropped_symbols = []

        # 1. Update counts for everything in the plan
        for symbol in plan_symbols:
            if symbol in held_symbols:
                # If we hold it, reset its staleness counter
                if symbol in staleness_data:
                    del staleness_data[symbol]
                continue

            entry = staleness_data.get(symbol, {"count": 0, "last_seen": None})

            # If we haven't seen it today yet
            if entry["last_seen"] != today_str:
                entry["count"] += 1
                entry["last_seen"] = today_str
                staleness_data[symbol] = entry

        # 2. Filter out symbols that have reached the threshold
        # Recommendation 6 says 4+ days
        threshold = 4

        for sig in signals:
            symbol = str(sig.get("symbol") or sig.get("ticker", "")).upper()
            count = staleness_data.get(symbol, {}).get("count", 0)

            if count >= threshold:
                dropped_symbols.append(symbol)
                log.warning(
                    f"[STALENESS] Purging {symbol} - appeared {count} days without execution"
                )
            else:
                filtered_signals.append(sig)

        # 3. Cleanup: Remove symbols from staleness_data if they are NOT in today's plan
        # and weren't seen today (i.e., they dropped off naturally)
        to_delete = []
        for symbol, entry in staleness_data.items():
            if symbol not in plan_symbols and entry["last_seen"] != today_str:
                to_delete.append(symbol)

        for symbol in to_delete:
            del staleness_data[symbol]

        # 4. Save updated staleness data (only if not dry run)
        if not self.dry_run:
            try:
                staleness_file.parent.mkdir(parents=True, exist_ok=True)
                with open(staleness_file, "w", encoding="utf-8") as f:
                    json.dump(staleness_data, f, indent=2)
            except Exception as e:
                log.warning(f"Could not save watchlist_staleness.json: {e}")

        return filtered_signals, dropped_symbols

    def _run_sequential_shadow_eval_batch(self) -> Dict[str, Any]:
        """
        Run offline sequential shadow evaluation as a separate process.

        This is review-only analytics and never affects live order decisions.
        """
        cfg = getattr(get_config(), "sequential_shadow_eval", None)
        if cfg is None or not bool(getattr(cfg, "enabled", False)):
            return {"ran": False, "reason": "disabled"}
        if self.dry_run:
            return {"ran": False, "reason": "dry_run"}
        if str(getattr(cfg, "schedule", "pm_daily")).lower() != "pm_daily":
            return {"ran": False, "reason": "schedule_not_pm_daily"}

        plan_date = get_pm_plan_date()
        day_str = plan_date.strftime("%Y-%m-%d")
        cmd = [
            sys.executable,
            "-m",
            "autotrade.analysis.sequential_shadow_runner",
            "--date",
            day_str,
            "--workers",
            str(int(max(1, getattr(cfg, "max_workers", 2)))),
            "--max-events",
            str(int(max(0, getattr(cfg, "max_events_per_run", 0)))),
            "--timeout-seconds",
            str(int(max(1, getattr(cfg, "timeout_seconds_per_event", 10)))),
        ]
        timeout_s = int(max(60, getattr(cfg, "runner_timeout_seconds", 900)))
        started = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_DIR),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout_s,
                check=False,
            )
            elapsed = int((time.time() - started) * 1000)
            if proc.returncode == 0:
                logger.info(
                    "[SEQUENTIAL SHADOW] Batch complete for %s in %.1fs",
                    day_str,
                    elapsed / 1000.0,
                )
                return {
                    "ran": True,
                    "success": True,
                    "date": day_str,
                    "elapsed_ms": elapsed,
                    "return_code": int(proc.returncode),
                }
            stderr = (proc.stderr or "").strip()
            logger.warning(
                "[SEQUENTIAL SHADOW] Batch failed (rc=%s): %s",
                proc.returncode,
                stderr[:300],
            )
            return {
                "ran": True,
                "success": False,
                "date": day_str,
                "elapsed_ms": elapsed,
                "return_code": int(proc.returncode),
                "error": stderr[:500],
            }
        except subprocess.TimeoutExpired:
            logger.warning(
                "[SEQUENTIAL SHADOW] Batch timed out after %ss for %s",
                timeout_s,
                day_str,
            )
            return {
                "ran": True,
                "success": False,
                "date": day_str,
                "error": f"timeout>{timeout_s}s",
            }
        except Exception as e:
            logger.warning(f"[SEQUENTIAL SHADOW] Batch execution error: {e}")
            return {
                "ran": True,
                "success": False,
                "date": day_str,
                "error": str(e),
            }

    def _ensure_youtube_intelligence(self) -> Dict:
        """Scan YouTube channels and load intelligence for next-day planning."""
        try:
            from autotrade.utils.youtube_readiness import (
                ensure_youtube_ready,
                get_intelligence_context,
                format_readiness_log,
                check_readiness,
            )

            # Check current state
            status = check_readiness()
            logger.info(format_readiness_log(status))

            # Auto-scan if needed (not in dry run — scanning is read-only and always safe)
            if status["needs_scan"] or status["needs_report"]:
                logger.info("[YOUTUBE] Scanning channels for latest videos...")
                ensure_youtube_ready(timeout=1800)

            # Load intelligence context
            ctx = get_intelligence_context()
            if ctx["available"]:
                logger.info(
                    f"[YOUTUBE] Intelligence loaded: regime={ctx['regime']}, "
                    f"sizing={ctx['sizing_multiplier']}x, "
                    f"report_date={ctx['report_date']}"
                )
                if ctx["avoid_sectors"]:
                    logger.info(
                        f"[YOUTUBE] Avoid sectors: {', '.join(ctx['avoid_sectors'])}"
                    )
                if ctx["directives"]:
                    for i, d in enumerate(ctx["directives"][:3], 1):
                        logger.info(f"[YOUTUBE] Directive {i}: {d[:100]}")
            else:
                logger.info("[YOUTUBE] No intelligence available — using defaults")

            return ctx

        except ImportError as e:
            logger.warning(f"[YOUTUBE] Module not available: {e}")
            return {
                "regime": "NEUTRAL",
                "sizing_multiplier": 1.0,
                "available": False,
                "avoid_sectors": [],
                "favor_sectors": [],
                "smallcap_ok": True,
                "directives": [],
                "regime_confidence": 0,
                "report_date": "none",
            }
        except Exception as e:
            logger.warning(f"[YOUTUBE] Intelligence loading failed: {e}")
            return {
                "regime": "NEUTRAL",
                "sizing_multiplier": 1.0,
                "available": False,
                "avoid_sectors": [],
                "favor_sectors": [],
                "smallcap_ok": True,
                "directives": [],
                "regime_confidence": 0,
                "report_date": "none",
            }

    def _refresh_quantitative_regime(self) -> None:
        """Load quantitative regime for PM planning."""
        if not self.regime_detector:
            self.current_regime_analysis = None
            self.regime_strategy_overrides = {}
            return

        try:
            analysis = self.regime_detector.detect_regime(use_cache=True)
            self.current_regime_analysis = analysis
            self.regime_strategy_overrides = analysis.recommended_strategy or {}
            logger.info(
                "[REGIME] %s (confidence %.0f%%, breadth %.1f%%)",
                analysis.regime.value.upper(),
                analysis.confidence * 100.0,
                analysis.breadth_pct_positive,
            )
        except Exception as e:
            logger.warning(f"[REGIME] Detection failed: {e}")
            self.current_regime_analysis = None
            self.regime_strategy_overrides = {}

    def _detect_signal_router_regime(self) -> Dict[str, Any]:
        """Best-effort alpha-zoo regime-router snapshot (diagnostic-only)."""
        out = dict(self.regime_router_output or {})
        if self.regime_router is None:
            out.setdefault("available", False)
            out.setdefault("regime", "NEUTRAL")
            out.setdefault("confidence", 0.0)
            out.setdefault("method", "disabled")
            self.regime_router_output = out
            return out
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            now = datetime.now()
            start = now - timedelta(days=90)
            req = StockBarsRequest(
                symbol_or_symbols="SPY",
                timeframe=TimeFrame.Day,
                start=start,
                end=now,
            )
            bars = self.data_client.get_stock_bars(req)
            if bars is not None and hasattr(bars, "df") and not bars.df.empty:
                spy_df = bars.df.loc["SPY"].reset_index()
                spy_df = spy_df.rename(
                    columns={"close": "close", "high": "high", "low": "low"}
                )
            else:
                spy_df = None

            regime_result = self.regime_router.get_current_regime(price_data=spy_df)
            out.update(
                {
                    "available": True,
                    "regime": str(
                        getattr(
                            getattr(regime_result, "regime", None), "value", "NEUTRAL"
                        )
                    ).upper(),
                    "confidence": float(
                        getattr(regime_result, "confidence", 0.0) or 0.0
                    ),
                    "method": str(
                        getattr(regime_result, "method", "router") or "router"
                    ),
                    "error": None,
                }
            )
        except Exception as e:
            out.update(
                {
                    "available": False,
                    "regime": "NEUTRAL",
                    "confidence": 0.0,
                    "error": str(e),
                }
            )
            logger.warning(f"[REGIME ROUTER] Detection failed: {e}")
        self.regime_router_output = out
        return out

    def _effective_market_regime(self) -> str:
        """Return effective regime label for downstream calls."""
        if self.current_regime_analysis is not None:
            return str(self.current_regime_analysis.regime.value).upper()
        return str(self.youtube_context.get("regime", "NEUTRAL") or "NEUTRAL").upper()

    def _regime_override_float(self, key: str, default: float) -> float:
        """Safely read a float regime override value."""
        overrides = (
            self.regime_strategy_overrides
            if isinstance(self.regime_strategy_overrides, dict)
            else {}
        )
        raw = overrides.get(key, default)
        try:
            return float(raw)
        except Exception:
            return float(default)

    def _regime_override_int(self, key: str, default: int) -> int:
        """Safely read an int regime override value."""
        overrides = (
            self.regime_strategy_overrides
            if isinstance(self.regime_strategy_overrides, dict)
            else {}
        )
        raw = overrides.get(key, default)
        try:
            return int(raw)
        except Exception:
            return int(default)

    def _merge_multi_strategy_signals(
        self, baseline_candidates: List, open_slots: int, log: logging.Logger
    ) -> List:
        """Augment baseline signals with candidates from all validated strategies.

        Each strategy's screener config is used to run get_entry_candidates().
        Results are merged and deduplicated (highest composite_score per ticker wins).
        Strategy metadata (strategy_name, setup_type, max_hold_days, etc.) is tagged
        onto each signal dict for downstream traceability.
        """
        merge_started = time.perf_counter()
        try:
            from autotrade.signals.strategy_pool import (
                load_validated_strategies,
                load_validated_strategies_by_symbol,
            )
            from autotrade.signals.screener_v2 import get_entry_candidates
        except ImportError as e:
            log.warning(f"Multi-strategy imports unavailable: {e}")
            self._last_validated_strategy_count = 0
            self._last_merged_candidate_count = len(baseline_candidates or [])
            self._last_strategy_routing_diagnostics = {}
            return baseline_candidates

        strategies = load_validated_strategies()
        self._last_validated_strategy_count = len(strategies)
        if not strategies:
            self._last_merged_candidate_count = len(baseline_candidates or [])
            self._last_strategy_routing_diagnostics = {}
            return baseline_candidates

        lab_cfg = get_config().strategy_lab
        per_symbol_enabled = bool(getattr(lab_cfg, "per_symbol_strategy_enabled", True))
        per_symbol_fallback = bool(
            getattr(lab_cfg, "per_symbol_fallback_to_global", True)
        )
        signal_cfg = getattr(get_config(), "signal_generation", None)
        pm_screener_override = self._build_pm_screener_override()
        confluence_boost_per = float(
            getattr(signal_cfg, "confluence_boost_per_strategy", 0.03)
            if signal_cfg is not None
            else 0.03
        )
        confluence_boost_max = float(
            getattr(signal_cfg, "confluence_boost_max", 0.15)
            if signal_cfg is not None
            else 0.15
        )
        confluence_boost_per = max(0.0, confluence_boost_per)
        confluence_boost_max = max(0.0, confluence_boost_max)
        strategy_hits: Dict[str, set] = defaultdict(set)

        per_symbol_map: Dict[str, List[Dict[str, Any]]] = {}
        if per_symbol_enabled:
            per_symbol_map = load_validated_strategies_by_symbol(
                fallback_to_global=False
            )

        symbol_strategy_rank: Dict[str, Dict[tuple, int]] = {}
        for symbol, rows in per_symbol_map.items():
            symbol_key = str(symbol or "").strip().upper()
            if not symbol_key or not isinstance(rows, list):
                continue
            ranked: Dict[tuple, int] = {}
            for idx, row in enumerate(rows, 1):
                if not isinstance(row, dict):
                    continue
                key = (
                    str(row.get("strategy_name") or "").strip(),
                    str(row.get("setup_type") or "").strip(),
                )
                if key[0] and key[1] and key not in ranked:
                    ranked[key] = idx
            if ranked:
                symbol_strategy_rank[symbol_key] = ranked

        routing_stats: Dict[str, Any] = {
            "per_symbol_enabled": per_symbol_enabled,
            "per_symbol_fallback_to_global": per_symbol_fallback,
            "per_symbol_symbol_count": len(symbol_strategy_rank),
            "strategy_signals_considered": 0,
            "strategy_signals_kept": 0,
            "kept_per_symbol": 0,
            "kept_global_fallback": 0,
            "dropped_not_in_top_k": 0,
            "dropped_no_symbol_history": 0,
            "symbols_with_ranked_history": set(),
            "symbols_with_insufficient_history": set(),
        }

        log.info(f"   Multi-strategy screening: {len(strategies)} validated strategies")

        routing_decision_started = time.perf_counter()
        # Collect all signals keyed by ticker (baseline first)
        best_by_ticker: Dict[str, Any] = {}
        for c in baseline_candidates:
            d = (
                c.to_dict()
                if hasattr(c, "to_dict")
                else (c if isinstance(c, dict) else {})
            )
            ticker = str(d.get("ticker") or d.get("symbol") or "").strip().upper()
            if ticker:
                d["ticker"] = ticker
                d.setdefault("symbol", ticker)
                d.setdefault("strategy_name", "baseline")
                d.setdefault("setup_type", "default")
                d.setdefault("symbol_strategy_rank", None)
                d.setdefault("symbol_strategy_source", "baseline")
                best_by_ticker[ticker] = d

        for strat in strategies:
            strat_name = strat.get("strategy_name", "unknown")
            setup_type = strat.get("setup_type", "unknown")
            screener_patch = dict(
                strat.get("config_patch", {}).get("screener_v2") or {}
            )
            if pm_screener_override:
                screener_patch.update(pm_screener_override)
            if not screener_patch:
                log.debug(f"   Skipping {strat_name}: no screener_v2 config_patch")
                continue

            # Extract exit params for tagging
            backtest_patch = strat.get("config_patch", {}).get("backtest", {})
            max_hold_days = backtest_patch.get("max_hold_days")
            stop_atr_mult = backtest_patch.get("stop_atr")
            target_atr_mult = backtest_patch.get("target_atr")

            try:
                strat_candidates = get_entry_candidates(
                    max_candidates=50,
                    config_override=screener_patch,
                    parent_logger=log,
                    log_samples=False,
                )
                log.info(
                    f"   [{strat_name}] ({setup_type}): {len(strat_candidates)} candidates"
                )
            except Exception:
                log.exception("   [%s] screener failed", strat_name)
                continue

            for cand in strat_candidates:
                d = (
                    cand
                    if isinstance(cand, dict)
                    else (cand.to_dict() if hasattr(cand, "to_dict") else {})
                )
                ticker = str(d.get("ticker") or d.get("symbol") or "").strip().upper()
                if not ticker:
                    continue
                d["ticker"] = ticker
                d.setdefault("symbol", ticker)

                routing_stats["strategy_signals_considered"] += 1

                symbol_rank = None
                symbol_source = "global_fallback"
                eligible = True
                rank_map = symbol_strategy_rank.get(ticker)
                if per_symbol_enabled:
                    if rank_map:
                        routing_stats["symbols_with_ranked_history"].add(ticker)
                        symbol_rank = rank_map.get((str(strat_name), str(setup_type)))
                        if symbol_rank is not None:
                            symbol_source = "per_symbol"
                            routing_stats["kept_per_symbol"] += 1
                        else:
                            eligible = False
                            routing_stats["dropped_not_in_top_k"] += 1
                    else:
                        routing_stats["symbols_with_insufficient_history"].add(ticker)
                        if per_symbol_fallback:
                            symbol_source = "global_fallback"
                            routing_stats["kept_global_fallback"] += 1
                        else:
                            eligible = False
                            routing_stats["dropped_no_symbol_history"] += 1
                else:
                    routing_stats["kept_global_fallback"] += 1

                if not eligible:
                    continue

                strategy_hits[ticker].add((strat_name, setup_type))
                confluence_count = max(1, len(strategy_hits[ticker]))

                def _apply_confluence_boost(
                    candidate: Dict[str, Any], count: int
                ) -> None:
                    base_score = float(
                        candidate.get(
                            "base_composite_score",
                            candidate.get("composite_score", candidate.get("score", 0)),
                        )
                        or 0.0
                    )
                    multiplier = 1.0
                    if count > 1 and confluence_boost_per > 0.0:
                        multiplier = 1.0 + min(
                            confluence_boost_max,
                            confluence_boost_per * float(count - 1),
                        )
                    boosted = base_score * multiplier
                    candidate["confluence_count"] = int(count)
                    candidate["confluence_multiplier"] = round(float(multiplier), 4)
                    candidate["base_composite_score"] = base_score
                    candidate["confluence_score"] = boosted
                    candidate["composite_score"] = boosted
                    candidate["score"] = boosted
                    candidate["final_score"] = boosted

                # Tag with strategy metadata
                d["strategy_name"] = strat_name
                d["setup_type"] = setup_type
                d["symbol_strategy_rank"] = symbol_rank
                d["symbol_strategy_source"] = symbol_source
                if max_hold_days is not None:
                    d["max_hold_days"] = max_hold_days
                if stop_atr_mult is not None:
                    d["stop_atr_mult"] = stop_atr_mult
                if target_atr_mult is not None:
                    d["target_atr_mult"] = target_atr_mult

                routing_stats["strategy_signals_kept"] += 1

                # Keep highest composite_score per ticker
                existing = best_by_ticker.get(ticker)
                _apply_confluence_boost(d, confluence_count)
                if existing is not None:
                    _apply_confluence_boost(existing, confluence_count)
                new_score = float(d.get("composite_score", d.get("score", 0)) or 0)
                if existing is None:
                    best_by_ticker[ticker] = d
                else:
                    old_score = float(
                        existing.get("composite_score", existing.get("score", 0)) or 0
                    )
                    if new_score > old_score:
                        best_by_ticker[ticker] = d

        merged = sorted(
            best_by_ticker.values(),
            key=lambda s: float(s.get("composite_score", s.get("score", 0)) or 0),
            reverse=True,
        )

        # Cap at reasonable limit
        cap = max(open_slots + 10, 30)
        merged = merged[:cap]

        baseline_count = sum(1 for s in merged if s.get("strategy_name") == "baseline")
        strat_count = len(merged) - baseline_count
        self._last_merged_candidate_count = len(merged)

        routing_filter_sec = time.perf_counter() - routing_decision_started
        total_merge_sec = max(1e-9, time.perf_counter() - merge_started)
        strategy_symbols_considered = {
            str(row.get("ticker") or row.get("symbol") or "").strip().upper()
            for row in merged
            if isinstance(row, dict) and row.get("strategy_name") != "baseline"
        }
        strategy_symbols_considered.discard("")
        ranked_symbols_seen = set(routing_stats["symbols_with_ranked_history"])
        coverage_pct = (
            (len(ranked_symbols_seen) / len(strategy_symbols_considered) * 100.0)
            if strategy_symbols_considered
            else 0.0
        )
        kept_total = max(1, int(routing_stats["strategy_signals_kept"]))
        per_symbol_pct = routing_stats["kept_per_symbol"] / kept_total * 100.0
        fallback_pct = routing_stats["kept_global_fallback"] / kept_total * 100.0
        routing_runtime_pct = (routing_filter_sec / total_merge_sec) * 100.0

        diagnostics = {
            "per_symbol_enabled": per_symbol_enabled,
            "per_symbol_fallback_to_global": per_symbol_fallback,
            "per_symbol_symbol_count": int(routing_stats["per_symbol_symbol_count"]),
            "strategy_signals_considered": int(
                routing_stats["strategy_signals_considered"]
            ),
            "strategy_signals_kept": int(routing_stats["strategy_signals_kept"]),
            "kept_per_symbol": int(routing_stats["kept_per_symbol"]),
            "kept_global_fallback": int(routing_stats["kept_global_fallback"]),
            "dropped_not_in_top_k": int(routing_stats["dropped_not_in_top_k"]),
            "dropped_no_symbol_history": int(
                routing_stats["dropped_no_symbol_history"]
            ),
            "per_symbol_coverage_pct": float(coverage_pct),
            "per_symbol_usage_pct": float(per_symbol_pct),
            "global_fallback_usage_pct": float(fallback_pct),
            "routing_filter_runtime_ms": float(routing_filter_sec * 1000.0),
            "routing_filter_runtime_pct_of_merge": float(routing_runtime_pct),
            "confluence_boost_per_strategy": float(confluence_boost_per),
            "confluence_boost_max": float(confluence_boost_max),
            "symbols_with_insufficient_history": sorted(
                set(routing_stats["symbols_with_insufficient_history"])
            ),
        }
        self._last_strategy_routing_diagnostics = diagnostics

        if routing_stats["strategy_signals_considered"] > 0:
            log.info(
                "   Strategy routing: considered=%d kept=%d | per_symbol=%d (%.1f%%) | "
                "global_fallback=%d (%.1f%%) | dropped_not_in_top_k=%d dropped_no_history=%d",
                routing_stats["strategy_signals_considered"],
                routing_stats["strategy_signals_kept"],
                routing_stats["kept_per_symbol"],
                per_symbol_pct,
                routing_stats["kept_global_fallback"],
                fallback_pct,
                routing_stats["dropped_not_in_top_k"],
                routing_stats["dropped_no_symbol_history"],
            )

        if per_symbol_enabled and strategy_symbols_considered and coverage_pct < 40.0:
            log.warning(
                "   [STRATEGY ROUTING] Low per-symbol coverage %.1f%% (%d/%d symbols ranked)",
                coverage_pct,
                len(ranked_symbols_seen),
                len(strategy_symbols_considered),
            )

        if routing_runtime_pct > 15.0 and routing_filter_sec > 0.5:
            log.warning(
                "   [STRATEGY ROUTING] Runtime guardrail exceeded: routing filter used %.1f%% "
                "of merge time (target <= 15%%)",
                routing_runtime_pct,
            )

        log.info(
            f"   Multi-strategy merge: {len(merged)} total "
            f"({baseline_count} baseline + {strat_count} from strategies)"
        )

        from autotrade.signals.agentic_signal_generator import EntryCandidate

        merged_objects = []
        for d in merged:
            try:
                ec = EntryCandidate(
                    symbol=d.get("symbol", d.get("ticker", "UNKNOWN")),
                    price=float(
                        d.get("price", d.get("entry_price", d.get("close", 0.0)))
                    ),
                )
                for k, v in d.items():
                    if hasattr(ec, k) and k not in ["symbol", "price"]:
                        setattr(ec, k, v)
                ec.strategy_name = d.get("strategy_name", "baseline")
                ec.setup_type = d.get("setup_type", "default")
                ec.symbol_strategy_rank = d.get("symbol_strategy_rank")
                ec.symbol_strategy_source = d.get("symbol_strategy_source")
                merged_objects.append(ec)
            except Exception as e:
                log.warning(f"Failed to reconstruct EntryCandidate: {e}")

        return merged_objects

    def _position_cap_resolution(self):
        """Resolve the active position-cap contract for PM planning."""
        regime = dict(self.resolved_regime_output or {})
        if not regime:
            regime = {
                "regime": self._effective_market_regime(),
                "strategy_overrides": dict(self.regime_strategy_overrides or {}),
            }
        alpha_router = dict(self.regime_router_output or {})
        return resolve_position_caps(
            regime=regime,
            failsafe=self.strategy_failsafe_snapshot,
            config=get_config(),
            alpha_router=alpha_router,
        )

    def _effective_max_positions(self) -> int:
        """Total positions after core + reserve capacity resolution."""
        return int(self._position_cap_resolution().total_cap)

    def _build_entry_constraints(self) -> Dict[str, Any]:
        caps = self._position_cap_resolution()
        entry_constraints = caps.to_entry_constraints()
        entry_constraints["source"] = "pm_workflow"
        return entry_constraints

    def _effective_stop_multiplier(self, fallback: float = 2.0) -> float:
        """
        Resolve stop multiplier with safety-first precedence.
        The tighter of failsafe/regime stop multipliers is applied.
        """
        failsafe_mult = float(
            getattr(self.strategy_failsafe_snapshot, "stop_multiplier", fallback)
            or fallback
        )
        regime_mult = self._regime_override_float("stop_multiplier", failsafe_mult)
        return max(0.5, min(failsafe_mult, regime_mult))

    def _build_pm_screener_override(self) -> Dict[str, Any]:
        """Align PM screener gating with the resolved regime contract."""
        resolved_state = getattr(self, "resolved_regime_output", {})
        resolved = resolved_state if isinstance(resolved_state, dict) else {}
        if bool(resolved.get("allow_new_longs", True)):
            return {
                "prefer_bullish_regime": False,
                "momentum_roc_min_score": 25.0,
                "rsi_pullback_min_score": 25.0,
            }
        return {}

    def _build_fallback_position_analysis(
        self,
        *,
        symbol: str,
        entry_price: float,
        current_price: float,
        pnl_pct: float,
        market_value: float,
        qty: int,
        error_text: str,
    ) -> Dict[str, Any]:
        """Return a conservative fallback analysis when deeper helpers fail."""
        atr = entry_price * 0.02 if entry_price > 0 else max(current_price * 0.02, 0.01)
        stop_price = max(0.0, entry_price - (atr * 2.0))
        target_price = entry_price + (atr * 2.0)

        action = "hold"
        reason = "fallback_neutral"
        conviction_score = max(20.0, min(80.0, 50.0 + (pnl_pct * 2.0)))

        if pnl_pct <= -7.0:
            action = "exit"
            reason = "fallback_large_loss"
        elif pnl_pct <= -3.0:
            action = "watch"
            reason = "fallback_moderate_loss"
        elif pnl_pct >= 5.0:
            action = "trim"
            reason = "fallback_profit_lock"

        logger.warning(
            "    [FALLBACK] %s position analysis degraded: %s",
            symbol,
            error_text,
        )

        return {
            "symbol": symbol,
            "entry_price": entry_price,
            "current_price": current_price,
            "pnl_pct": pnl_pct,
            "market_value": market_value,
            "qty": qty,
            "conviction_score": conviction_score,
            "conviction_factors": {},
            "conviction_reasons": [f"fallback:{error_text}"],
            "risk_action": "fallback",
            "risk_rules": [error_text],
            "recommended_action": action,
            "action_reason": reason,
            "levels": {
                "atr": atr,
                "atr_pct": ((atr / entry_price) * 100.0) if entry_price > 0 else 2.0,
                "stop_price": stop_price,
                "target_price": target_price,
            },
        }

    def _analyze_position(self, pos) -> Dict:
        """Analyze a single position using unified strategy + conviction engine."""
        symbol = pos.symbol
        entry_price = float(pos.avg_entry_price)
        current_price = float(pos.current_price)
        pnl_pct = float(pos.unrealized_plpc) * 100
        market_value = float(pos.market_value)
        qty = int(pos.qty)

        logger.info(f"\n  {symbol}")
        logger.info(f"    Entry: ${entry_price:.2f} -> Current: ${current_price:.2f}")
        logger.info(f"    P&L: {pnl_pct:+.1f}% | Value: ${market_value:,.2f}")

        # Estimate hold_minutes (assume at least 1 day for PM analysis)
        hold_minutes = 24 * 60

        try:
            levels_info = self._get_position_levels(symbol, entry_price)

            conviction_result = self.conviction_engine.compute_conviction(
                symbol=symbol,
                pnl_pct=pnl_pct,
                hold_minutes=hold_minutes,
                atr_pct=levels_info.get("atr_pct", 2.0),
                support_price=levels_info.get("s1_price", 0),
                resistance_price=levels_info.get("r1_price", 0),
                support_strength=levels_info.get("s1_strength", 0),
                resistance_strength=levels_info.get("r1_strength", 0),
                current_price=current_price,
                entry_price=entry_price,
                relative_strength=levels_info.get("relative_strength_5d", 0),
                news_sentiment=levels_info.get("news_sentiment", 0),
                has_catalyst=bool(levels_info.get("has_catalyst", False)),
                sector=str(levels_info.get("sector", "")),
                market_regime=self._effective_market_regime().lower(),
            )

            conviction_score = conviction_result[0]
            conviction_factors = conviction_result[1]
            conviction_reasons = conviction_result[2]

            logger.info(f"    Conviction: {conviction_score:.0f}/100")

            levels = self.strategy.calculate_levels(
                entry_price=entry_price,
                s1_price=levels_info.get("s1_price", 0),
                s1_strength=levels_info.get("s1_strength", 0),
                r1_price=levels_info.get("r1_price", 0),
                r1_strength=levels_info.get("r1_strength", 0),
                r2_price=levels_info.get("r2_price", 0),
                atr=levels_info.get("atr", entry_price * 0.02),
            )

            logger.info(
                f"    Levels: ATR stop=${levels.stop_price:.2f} | ATR target=${levels.target_price:.2f}"
            )

            atr_pct = levels_info.get("atr_pct", 2.0)
            risk_result = self.risk_gate.evaluate(
                symbol=symbol,
                entry_price=entry_price,
                current_price=current_price,
                qty=qty,
                atr_pct=atr_pct,
                time_in_trade_minutes=hold_minutes,
                stop_multiplier=self._effective_stop_multiplier(fallback=2.0),
            )

            action = "hold"
            reason = ""

            if risk_result.action.value == "exit":
                action = "exit"
                reason = (
                    risk_result.triggered_rules[0]
                    if risk_result.triggered_rules
                    else "risk gate"
                )
            elif levels.stop_price and current_price <= levels.stop_price:
                action = "exit"
                reason = f"ATR stop breached at ${levels.stop_price:.2f}"
            elif levels.target_price and current_price >= levels.target_price * 1.02:
                action = "exit"
                reason = f"ATR target exceeded at ${levels.target_price:.2f}"
            elif levels.target_price and current_price >= levels.target_price:
                action = "trim"
                reason = (
                    f"ATR target reached at ${levels.target_price:.2f} - take partial"
                )
            elif pnl_pct >= 4.0 and conviction_score >= 65:
                action = "add"
                reason = f"+{pnl_pct:.1f}% gain, conviction {conviction_score}"
            elif conviction_score < 40:
                action = "watch"
                reason = f"low conviction ({conviction_score})"
            else:
                action = "hold"
                reason = f"conviction {conviction_score}"

            if (
                self.strategy_failsafe_snapshot
                and self.strategy_failsafe_snapshot.level in ("failing", "critical")
            ):
                if (
                    conviction_score
                    < self.strategy_failsafe_snapshot.min_conviction_exit
                ):
                    action = "exit"
                    reason = (
                        f"failsafe_exit: conviction {conviction_score:.0f} < "
                        f"{self.strategy_failsafe_snapshot.min_conviction_exit:.0f}"
                    )
                elif pnl_pct < self.strategy_failsafe_snapshot.loser_exit_pnl_pct:
                    action = "exit"
                    reason = (
                        f"failsafe_exit: pnl {pnl_pct:+.1f}% < "
                        f"{self.strategy_failsafe_snapshot.loser_exit_pnl_pct:+.1f}%"
                    )
                elif action == "hold":
                    action = "watch"
                    reason = (
                        "failsafe_tight_trail: hold only with tight trailing stop "
                        f"({self.strategy_failsafe_snapshot.stop_multiplier:.1f}x ATR)"
                    )

            tag = {
                "exit": "[EXIT]",
                "add": "[ADD]",
                "hold": "[HOLD]",
                "watch": "[WATCH]",
                "trim": "[TRIM]",
            }.get(action, "[?]")
            logger.info(f"    {tag} Action: {action.upper()} - {reason}")

            return {
                "symbol": symbol,
                "entry_price": entry_price,
                "current_price": current_price,
                "pnl_pct": pnl_pct,
                "market_value": market_value,
                "qty": qty,
                "conviction_score": conviction_score,
                "conviction_factors": conviction_factors,
                "conviction_reasons": conviction_reasons,
                "risk_action": risk_result.action.value,
                "risk_rules": risk_result.triggered_rules,
                "recommended_action": action,
                "action_reason": reason,
                "levels": levels.to_dict() if hasattr(levels, "to_dict") else {},
            }
        except Exception as exc:
            return self._build_fallback_position_analysis(
                symbol=symbol,
                entry_price=entry_price,
                current_price=current_price,
                pnl_pct=pnl_pct,
                market_value=market_value,
                qty=qty,
                error_text=safe_exception_text(exc),
            )

    def _get_position_levels(self, symbol: str, entry_price: float) -> Dict:
        """Build ATR-first levels with internal S/R overlay from intraday bars."""
        key = str(symbol).upper()
        if key in self.position_levels:
            return dict(self.position_levels[key])

        atr = entry_price * 0.02
        levels: Dict[str, float] = {
            "s1_price": 0.0,
            "s1_strength": 0.0,
            "s2_price": 0.0,
            "r1_price": 0.0,
            "r1_strength": 0.0,
            "r2_price": 0.0,
            "support_dist_atr": 0.0,
            "resistance_dist_atr": 0.0,
            "sr_quality_score": 0.0,
            "atr": atr,
            "atr_pct": (atr / entry_price * 100) if entry_price else 2.0,
        }

        try:
            if INTRADAY_PROVIDER_AVAILABLE and get_intraday_bars is not None:
                bars_df = get_intraday_bars(
                    key,
                    self.data_client,
                    minutes_back=780,  # ~2 trading days
                    interval="5m",
                )
                if bars_df is not None and len(bars_df) >= 30:
                    atr_col = bars_df.get("atr_14")
                    if atr_col is not None:
                        try:
                            atr_candidate = float(atr_col.iloc[-1] or 0.0)
                            if atr_candidate > 0:
                                levels["atr"] = atr_candidate
                                levels["atr_pct"] = (
                                    (atr_candidate / entry_price * 100)
                                    if entry_price
                                    else 2.0
                                )
                        except Exception:
                            pass

                    sr = estimate_sr_levels(
                        bars_df,
                        lookback_bars=min(180, len(bars_df)),
                        pivot_window=2,
                        cluster_atr_mult=0.45,
                    )
                    if sr:
                        levels["s1_price"] = float(sr.get("s1_price", 0.0) or 0.0)
                        levels["s1_strength"] = float(sr.get("s1_strength", 0.0) or 0.0)
                        levels["r1_price"] = float(sr.get("r1_price", 0.0) or 0.0)
                        levels["r1_strength"] = float(sr.get("r1_strength", 0.0) or 0.0)
                        levels["support_dist_atr"] = float(
                            sr.get("support_dist_atr", 0.0) or 0.0
                        )
                        levels["resistance_dist_atr"] = float(
                            sr.get("resistance_dist_atr", 0.0) or 0.0
                        )
                        levels["sr_quality_score"] = float(
                            sr.get("sr_quality_score", 0.0) or 0.0
                        )
                        if levels["r1_price"] > levels["s1_price"] > 0:
                            span = levels["r1_price"] - levels["s1_price"]
                            levels["r2_price"] = levels["r1_price"] + span * 0.6

            self.position_levels[key] = dict(levels)
            return dict(levels)
        except Exception as e:
            logger.debug(f"Could not get levels for {symbol}: {e}")

        self.position_levels[key] = dict(levels)
        return dict(levels)

    def _generate_summary(self, all_positions, exits, adds) -> Dict:
        """Generate summary statistics."""
        total_value = sum(p["market_value"] for p in all_positions)
        total_pnl = sum(p["pnl_pct"] * p["market_value"] / 100 for p in all_positions)
        avg_conviction = (
            sum(p["conviction_score"] for p in all_positions) / len(all_positions)
            if all_positions
            else 0
        )

        return {
            "total_positions": len(all_positions),
            "total_value": total_value,
            "total_pnl_dollars": total_pnl,
            "avg_conviction": avg_conviction,
            "exits_planned": len(exits),
            "adds_planned": len(adds),
        }

    @staticmethod
    def _signal_to_dict(signal: Any) -> Dict[str, Any]:
        """Normalize signal object/dict into a plain dictionary."""
        if isinstance(signal, dict):
            return signal
        if hasattr(signal, "to_dict"):
            try:
                return signal.to_dict()
            except Exception:
                return {}
        return {}

    @staticmethod
    def _fmt_num(value: Any, pattern: str = "{:.1f}", default: str = "na") -> str:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return pattern.format(float(value))
        return default

    def _log_signal_candidate_trace(
        self, signals: List[Any], log: logging.Logger
    ) -> None:
        """Log concise per-candidate trace with backtest strategy context."""
        if not signals:
            return

        rows = [self._signal_to_dict(s) for s in signals]
        rows = [r for r in rows if isinstance(r, dict) and r]
        if not rows:
            return

        source_counts: Dict[str, int] = {}
        for row in rows:
            src = str(row.get("strategy_name") or "baseline")
            source_counts[src] = source_counts.get(src, 0) + 1

        source_summary = ", ".join(
            [f"{name}:{count}" for name, count in sorted(source_counts.items())]
        )
        log.info(
            f"\n[PM SIGNAL TRACE] {len(rows)} candidates | sources: {source_summary}"
        )

        for row in rows:
            symbol = str(row.get("symbol") or row.get("ticker") or "?")
            action = str(row.get("action") or row.get("recommendation") or "watch")
            strategy_name = str(row.get("strategy_name") or "baseline")
            setup_type = str(row.get("setup_type") or "default")

            score = (
                row.get("score")
                if row.get("score") is not None
                else row.get("final_score")
                if row.get("final_score") is not None
                else row.get("confidence")
            )

            validation = row.get("validation", {})
            if not isinstance(validation, dict):
                validation = {}

            bt_score = (
                row.get("backtest_score")
                if row.get("backtest_score") is not None
                else validation.get("backtest_score")
            )
            hist_win_rate = (
                row.get("historical_win_rate")
                if row.get("historical_win_rate") is not None
                else validation.get("historical_win_rate")
            )
            similar_matches = (
                row.get("similar_signals_found")
                if row.get("similar_signals_found") is not None
                else validation.get("similar_signals_found")
            )

            win_rate_pct = None
            if isinstance(hist_win_rate, (int, float)):
                win_rate_pct = (
                    hist_win_rate * 100.0
                    if hist_win_rate <= 1.0
                    else float(hist_win_rate)
                )

            risk_reward = row.get("risk_reward")
            volume_ratio = row.get("volume_ratio")

            log.info(
                "   %s | src=%s/%s | act=%s | score=%s | bt=%s wr=%s%% n=%s | rr=%s vol=%s",
                symbol,
                strategy_name[:16],
                setup_type[:12],
                action,
                self._fmt_num(score, "{:.1f}"),
                self._fmt_num(bt_score, "{:.0f}"),
                self._fmt_num(win_rate_pct, "{:.1f}"),
                self._fmt_num(similar_matches, "{:.0f}"),
                self._fmt_num(risk_reward, "{:.2f}"),
                self._fmt_num(volume_ratio, "{:.2f}"),
            )

    def _print_plan(self, plan: Dict):
        """Print the next-day plan."""
        logger.info("\n" + "=" * 60)
        logger.info("NEXT-DAY TRADING PLAN")
        logger.info("=" * 60)

        summary = plan["summary"]
        logger.info("\n[PORTFOLIO SUMMARY]")
        logger.info(f"   Positions: {summary['total_positions']}")
        logger.info(f"   Total Value: ${summary['total_value']:,.2f}")
        logger.info(f"   Day P&L: ${summary['total_pnl_dollars']:+,.2f}")
        logger.info(f"   Avg Conviction: {summary['avg_conviction']:.0f}/100")

        if plan["morning_exits"]:
            logger.info(f"\n[MORNING EXITS] ({len(plan['morning_exits'])})")
            for p in plan["morning_exits"]:
                logger.info(
                    f"   * {p['symbol']}: {p['pnl_pct']:+.1f}% - {p['action_reason']}"
                )

        if plan["morning_adds"]:
            logger.info(f"\n[MORNING ADDS] ({len(plan['morning_adds'])})")
            for p in plan["morning_adds"]:
                logger.info(
                    f"   * {p['symbol']}: {p['pnl_pct']:+.1f}% - {p['action_reason']}"
                )

        if plan["holds"]:
            logger.info(f"\n[HOLDS] ({len(plan['holds'])})")
            for p in plan["holds"]:
                action = p["recommended_action"].upper()
                logger.info(
                    f"   * {p['symbol']}: {p['pnl_pct']:+.1f}% | Conv: {p['conviction_score']:.0f} | {action}"
                )

        overnight = plan.get("overnight_plan", {})
        if overnight.get("watchlist"):
            logger.info("\n[WATCHLIST FOR TOMORROW]")
            for w in overnight["watchlist"][:5]:
                logger.info(f"   * {w.get('symbol', w)}")

        logger.info("\n" + "=" * 60)

    def _save_plan(self, plan: Dict):
        """Save plan for tomorrow's execution, merging with existing plan if present."""
        plan_date = get_pm_plan_date()
        if not hasattr(plan_date, "strftime"):
            raise RuntimeError(
                f"PM plan save cannot resolve target date: {plan_date!r}"
            )
        date_str = plan_date.strftime("%Y-%m-%d")
        plan_path = PLANS_DIR / f"pm_plan_{date_str}.json"

        # Store the intended trading date for clarity (backward compatible).
        # Overwrite any stale fallback value so the filename and payload stay in sync.
        plan["plan_date"] = date_str
        current_signals, dropped_symbols = self._filter_plan_signals_by_asset_status(
            plan.get("signals", []), log=logger
        )
        plan["signals"] = current_signals
        plan["entry_candidates"] = list(current_signals)
        if dropped_symbols:
            plan["summary"] = dict(plan.get("summary") or {})
            plan["summary"]["asset_hygiene_dropped_symbols"] = dropped_symbols

        # MERGE MODE: If plan already exists, merge signals instead of replacing
        existing_plan = None
        if plan_path.exists():
            try:
                with open(plan_path, "r") as f:
                    existing_plan = json.load(f)
                logger.info(
                    f"[MERGE] Found existing plan from {existing_plan.get('generated_at', 'unknown')}"
                )
            except Exception as e:
                logger.warning(
                    f"[MERGE] Could not load existing plan: {e}, will overwrite"
                )

        if existing_plan:
            # Merge signals: deduplicate by symbol, keep highest score
            new_signals = plan.get("signals", [])
            existing_signals, existing_dropped = (
                self._filter_plan_signals_by_asset_status(
                    existing_plan.get("signals", []), log=logger
                )
            )
            if existing_dropped:
                logger.info(
                    "[MERGE] Dropped %d inactive/non-tradable symbols from existing plan",
                    len(existing_dropped),
                )

            # Build merged signal list keyed by symbol
            signals_by_symbol = {}

            # Add existing signals first
            for sig in existing_signals:
                symbol = sig.get("symbol") or sig.get("ticker")
                if symbol:
                    signals_by_symbol[symbol] = sig

            # Overlay new signals (higher scores win)
            merged_count = 0
            added_count = 0
            for sig in new_signals:
                symbol = sig.get("symbol") or sig.get("ticker")
                if not symbol:
                    continue

                new_score = sig.get(
                    "score", sig.get("final_score", sig.get("confidence", 0))
                )

                if symbol in signals_by_symbol:
                    existing_score = signals_by_symbol[symbol].get(
                        "score",
                        signals_by_symbol[symbol].get(
                            "final_score",
                            signals_by_symbol[symbol].get("confidence", 0),
                        ),
                    )
                    if new_score > existing_score:
                        signals_by_symbol[symbol] = sig
                        merged_count += 1
                    else:
                        existing = signals_by_symbol[symbol]
                        for key in (
                            "strategy_name",
                            "setup_type",
                            "symbol_strategy_rank",
                            "symbol_strategy_source",
                            "max_hold_days",
                            "stop_atr_mult",
                            "target_atr_mult",
                        ):
                            if existing.get(key) in (None, "") and sig.get(key) not in (
                                None,
                                "",
                            ):
                                existing[key] = sig.get(key)
                else:
                    signals_by_symbol[symbol] = sig
                    added_count += 1

            # Convert back to list and re-rank by priority
            merged_signals = list(signals_by_symbol.values())
            merged_signals.sort(
                key=lambda s: s.get(
                    "score", s.get("final_score", s.get("confidence", 0))
                ),
                reverse=True,
            )
            merged_signals, merged_dropped = self._filter_plan_signals_by_asset_status(
                merged_signals, log=logger
            )
            for i, sig in enumerate(merged_signals):
                sig["priority"] = i + 1

            plan["signals"] = merged_signals
            plan["entry_candidates"] = merged_signals  # Legacy compat
            if merged_dropped:
                plan["summary"] = dict(plan.get("summary") or {})
                all_dropped = list(
                    dict.fromkeys(
                        list(plan["summary"].get("asset_hygiene_dropped_symbols", []))
                        + merged_dropped
                    )
                )
                plan["summary"]["asset_hygiene_dropped_symbols"] = all_dropped

            logger.info(
                f"[MERGE] Signals: {len(existing_signals)} existing + {len(new_signals)} new "
                f"= {len(merged_signals)} total ({merged_count} updated, {added_count} added)"
            )

            # Preserve position analysis from first run if not in current run
            if not plan.get("positions") and existing_plan.get("positions"):
                plan["positions"] = existing_plan["positions"]
                plan["morning_exits"] = existing_plan.get("morning_exits", [])
                plan["morning_adds"] = existing_plan.get("morning_adds", [])
                plan["holds"] = existing_plan.get("holds", [])
                logger.info("[MERGE] Preserved position analysis from first run")

        final_signals = plan.get("signals") or plan.get("entry_candidates") or []
        if not final_signals and existing_plan:
            existing_signals = (
                existing_plan.get("signals")
                or existing_plan.get("entry_candidates")
                or existing_plan.get("buy_signals")
                or []
            )
            if existing_signals:
                logger.warning(
                    "[MERGE] Refusing to overwrite %s with 0 signals; preserving existing non-empty plan",
                    plan_path.name,
                )
                return plan_path

        with open(plan_path, "w") as f:
            json.dump(plan, f, indent=2, default=str)

        logger.info(f">>> Plan saved: {plan_path}")
        return plan_path

    def _filter_plan_signals_by_asset_status(
        self, signals: List[Dict[str, Any]], log: Optional[logging.Logger] = None
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """Drop symbols that Alpaca marks inactive, non-tradable, or missing."""
        filtered: List[Dict[str, Any]] = []
        dropped: List[str] = []
        logger_obj = log or logger

        for signal in signals or []:
            symbol = (
                str(signal.get("symbol") or signal.get("ticker") or "").strip().upper()
            )
            if not symbol:
                filtered.append(signal)
                continue

            verdict = self._get_asset_eligibility(symbol)
            if verdict.get("allowed", True):
                filtered.append(signal)
                continue

            dropped.append(symbol)
            logger_obj.warning(
                "[PM WORKFLOW] Dropping symbol %s from plan generation (%s)",
                symbol,
                verdict.get("reason", "asset_ineligible"),
            )

        return filtered, dropped

    @staticmethod
    def _market_is_closed_for_plan_generation() -> bool:
        """Treat after-hours as the closed window for plan-save asset hygiene."""
        now = get_market_now()
        if now.weekday() >= 5:
            return True
        return now.hour >= 16 or now.hour < 7

    def _get_asset_eligibility(self, symbol: str) -> Dict[str, Any]:
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return {"allowed": False, "reason": "missing_symbol"}

        cache = getattr(self, "_asset_eligibility_cache", None)
        if cache is None:
            cache = {}
            self._asset_eligibility_cache = cache

        cached = cache.get(symbol)
        if cached is not None:
            return cached

        if self._market_is_closed_for_plan_generation():
            verdict = {
                "allowed": True,
                "reason": "market_closed_asset_lookup_deferred",
            }
            cache[symbol] = verdict
            return verdict

        client = getattr(self, "client", None)
        if client is None or not hasattr(client, "get_asset"):
            verdict = {"allowed": True, "reason": "asset_lookup_unavailable"}
            cache[symbol] = verdict
            return verdict

        try:
            asset = client.get_asset(symbol)
        except Exception as exc:
            message = str(exc).lower()
            if any(
                token in message for token in ("404", "not found", "asset not found")
            ):
                verdict = {"allowed": False, "reason": "asset_not_found"}
            else:
                verdict = {"allowed": True, "reason": "asset_lookup_failed"}
            cache[symbol] = verdict
            return verdict

        status = str(getattr(asset, "status", "") or "").strip().lower()
        if "." in status:
            status = status.split(".")[-1]
        tradable = getattr(asset, "tradable", None)
        reasons: List[str] = []
        if status and status != "active":
            reasons.append(f"status={status}")
        if tradable is False:
            reasons.append("tradable=false")

        verdict = {
            "allowed": not reasons,
            "reason": ", ".join(reasons) if reasons else "eligible",
        }
        cache[symbol] = verdict
        return verdict

    def _empty_plan(self) -> Dict:
        """Return empty plan when no positions."""
        return {
            "generated_at": datetime.now().isoformat(),
            "market_intelligence": self.youtube_context or {},
            "quantitative_regime": {
                "available": False,
                "regime": self._effective_market_regime(),
                "confidence": 0.0,
                "strategy_overrides": {},
            },
            "regime_router": dict(self.regime_router_output or {}),
            "regime_divergence": {
                "labels_match": True,
                "quantitative_regime": self._effective_market_regime(),
                "regime_router_regime": str(
                    (self.regime_router_output or {}).get(
                        "regime", self._effective_market_regime()
                    )
                ),
            },
            "entry_constraints": dict(
                self._build_entry_constraints(),
                **{
                    "position_size_multiplier": self._regime_override_float(
                        "position_size_multiplier", 1.0
                    ),
                    "stop_multiplier": self._effective_stop_multiplier(fallback=2.0),
                },
            ),
            "strategy_routing_diagnostics": dict(
                self._last_strategy_routing_diagnostics or {}
            ),
            "positions": [],
            "morning_exits": [],
            "morning_adds": [],
            "holds": [],
            "summary": {
                "total_positions": 0,
                "total_value": 0,
                "exits_planned": 0,
                "adds_planned": 0,
            },
        }


# Backward-compatible alias used by tests and legacy imports.
PMWorkflow = PostMarketWorkflow


def main():
    dry_run = "--execute" not in sys.argv

    if dry_run:
        logger.info("[DRY RUN] Analysis only, no plan saved")
    else:
        logger.info("[EXECUTE] Will save plan for tomorrow")

    workflow = PMWorkflow(dry_run=dry_run)
    plan = workflow.run()

    return plan


if __name__ == "__main__":
    main()
