"""
Agentic Signal Generator - Post-Market Analysis for Next-Day Entries
=====================================================================
Replaces basic signal_generator.py with intelligent, agentic workflow.

MODULE ARCHITECTURE MAP

PHASE 1 – CANDIDATE SCREENING (Ingestion)
    _get_momentum_candidates()
    _run_lessons_screener_batched()
    _apply_signal_pipeline()

PHASE 2 – QUALITY GATING (Scoring & Filters)
    _score_entry_quality()
    _filter_by_sma200()
    _apply_lessons_filter()
    _calculate_entry_score()

PHASE 3 – CONTEXT ENRICHMENT (Sentiment & News)
    _analyze_sentiment()
    (Integration with MomentumEngine)

PHASE 4 – INTELLIGENT SYNTHESIS (LLM)
    _llm_synthesis()
    _call_llm_for_candidate()

PHASE 5 – PRIORITIZATION & PERSISTENCE (Output)
    _final_ranking()
    save_signals()
    print_summary()

Workflow:
1. Get momentum candidates from momentum_picker (base universe)
2. Filter through conviction engine (entry quality scoring)
3. Analyze news/sentiment for each candidate
4. LLM synthesis for final ranking
5. Generate prioritized watchlist with entry plans

Run at 6:30 PM after market close.
Generates signals for next trading day.

Usage:
  python -m autotrade.signals.agentic_signal_generator
"""

import json
import logging
import os
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

from config.config_loader import get_config
from autotrade.utils.strategy_params import (
    sanitize_atr_multiplier,
    sanitize_strategy_params,
)
from autotrade.signals.universe_filters import (
    DEFAULT_MAX_MARKET_CAP,
    DEFAULT_MIN_MARKET_CAP,
    HARD_BLOCK_MEGA_CAP,
    signal_universe_rejection_reason,
)
from autotrade.signals.llm_signal_overlay import build_signal_packet
from autotrade.signals.llm_signal_overlay import normalize_overlay_label
from autotrade.signals.llm_signal_overlay import query_local_signal_classifier

try:
    from autotrade.monitoring.collector import get_metrics_collector

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

# Import LessonBook for learned filter rules
from autotrade.analysis.financial_checks import (
    check_earnings_risk,
    check_balance_sheet_health,
    check_cash_flow_health,
    check_dividend_stability,
    check_valuation_sanity,
    check_options_positioning,
)

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

    _safe_logger_available = True
except ImportError:
    _safe_logger_available = False

    def safe_exception_text(exc, max_len=500):
        return str(exc)


def setup_logging():
    if _safe_logger_available:
        return get_safe_logger(
            "agentic_signals",
            LOG_DIR / f"agentic_signals_{datetime.now().strftime('%Y-%m-%d')}.log",
        )

    log_format = "%(asctime)s | %(levelname)s | %(message)s"
    logger = logging.getLogger("agentic_signals")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console)

    file_handler = logging.FileHandler(
        LOG_DIR / f"agentic_signals_{datetime.now().strftime('%Y-%m-%d')}.log",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)

    return logger


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_long_recommendation(row: Dict[str, Any]) -> bool:
    recommendation = str(row.get("recommendation") or "").upper()
    return "BUY" in recommendation and "SHORT" not in recommendation


def _repair_or_reject_long_geometry(
    row: Dict[str, Any],
) -> tuple[Dict[str, Any] | None, str | None]:
    if not _is_long_recommendation(row):
        return row, None

    entry = _as_float(row.get("entry_price", row.get("price")), 0.0)
    stop = _as_float(row.get("stop_loss", row.get("stop", row.get("stop_price"))), 0.0)
    target = _as_float(row.get("target", row.get("target_price")), 0.0)
    if entry <= 0:
        return None, "no_entry"
    if stop > 0 and target > 0 and stop < entry < target:
        risk = entry - stop
        reward = target - entry
        rr = reward / risk if risk > 0 else 0.0
        if rr >= 1.2 and target / entry <= 1.40 and risk / entry <= 0.25:
            return row, None

    atr = _as_float(row.get("atr_14"), 0.0)
    atr_percent = _as_float(row.get("atr_percent", row.get("atr_pct")), 0.0)
    if atr <= 0 and atr_percent > 0:
        atr = entry * (atr_percent / 100.0 if atr_percent > 1.0 else atr_percent)
    if atr <= 0:
        return None, "missing_atr_for_long_geometry"

    stop_mult = _as_float(row.get("stop_atr_mult"), 2.0)
    target_mult = _as_float(row.get("target_atr_mult"), 3.0)
    repaired_stop = max(entry * 0.75, entry - (atr * stop_mult))
    repaired_target = min(entry + (atr * target_mult), entry * 1.40)
    if not (0 < repaired_stop < entry < repaired_target):
        return None, "unrepairable_long_geometry"
    repaired_rr = (repaired_target - entry) / (entry - repaired_stop)
    if repaired_rr < 1.2:
        return None, "unrepairable_long_geometry"

    repaired = dict(row)
    repaired["stop_loss"] = round(repaired_stop, 4)
    repaired["stop"] = round(repaired_stop, 4)
    repaired["target"] = round(repaired_target, 4)
    repaired["target_price"] = round(repaired_target, 4)
    repaired["risk_reward"] = round(repaired_rr, 4)
    repaired["long_geometry_repaired"] = True
    return repaired, "repaired_long_geometry"


# Only setup logging when run directly, not when imported
if __name__ == "__main__":
    logger = setup_logging()
else:
    # When imported, use parent's logger or a simple one
    logger = logging.getLogger("agentic_signals")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        # Don't add handlers - let parent module control output

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.WARNING)

try:
    from autotrade.utils.local_data_provider import DELISTED_TOMBSTONES
except Exception:
    DELISTED_TOMBSTONES = None

# Load environment
env_path = PROJECT_DIR / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, val = line.split("=", 1)
            os.environ[key.strip()] = val.strip()


# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TIMEOUT = 30


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass
class EntryCandidate:
    """A candidate for next-day entry."""

    symbol: str
    price: float

    # Momentum scores
    momentum_score: float = 0.0
    rsi: float = 50.0
    weekly_return: float = 0.0

    # S/R levels
    s1_price: float = 0.0
    s1_strength: float = 0.0
    r1_price: float = 0.0
    r1_strength: float = 0.0  # Added for lessons

    # Entry quality
    entry_score: float = 0.0
    risk_reward: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    stop_atr_mult: float = 0.0
    target_atr_mult: float = 0.0

    # Raw data for lesson evaluation
    atr_14: float = 0.0
    atr_percent: float = 0.0
    support_dist_atr: float = 0.0
    resistance_dist_atr: float = 0.0
    volume_ratio: float = 0.0
    scan_type: str = ""

    # Sentiment
    news_sentiment: float = 0.0
    has_catalyst: bool = False
    catalyst_note: str = ""

    # LLM analysis
    llm_score: float = 50.0
    llm_reasoning: str = ""

    # Backtest validation
    backtest_score: float = 0.0
    historical_win_rate: float = 0.0
    similar_signals_found: int = 0
    avg_5d_return: float = 0.0

    # Final
    final_score: float = 0.0
    priority: int = 0
    action: str = "watch"  # watch, buy_open, buy_dip
    position_size: float = 0.0

    # Extensible annotations from enrichment stages (news momentum, routing context, etc).
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Strategy metadata (Phase 1 alpha overhaul)
    strategy_id: str = ""
    strategy_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        # Canonical signal format with backward-compatible fields
        entry_price = self.price
        if self.action == "buy_dip" and self.s1_price > 0:
            entry_price = self.s1_price

        data = asdict(self)
        normalized_final = max(0.0, min(100.0, float(self.final_score or 0.0)))
        data["strategy_params"] = sanitize_strategy_params(
            data.get("strategy_params"),
            fallback_stop=self.stop_atr_mult or 2.0,
            fallback_target=self.target_atr_mult or 3.0,
        )

        # Map action to recommendation for backward compatibility
        rec_map = {"buy_open": "BUY", "buy_dip": "WEAK BUY", "watch": "WATCH"}
        recommendation = rec_map.get(self.action, "WATCH")
        if recommendation == "BUY" and normalized_final >= 85:
            recommendation = "STRONG BUY"

        data.update(
            {
                "ticker": self.symbol,
                "entry_price": entry_price,
                "score": normalized_final,
                "confidence": normalized_final,
                "recommendation": recommendation,
                "normalized_score": normalized_final,
                "bullish_score": self.momentum_score,
                "atr_percent": self.atr_percent,
                "volume_ratio": self.volume_ratio,
                "scan_type": self.scan_type,
                "position_size": self.position_size,
                "catalyst": self.catalyst_note,
                "news_sentiment": self.news_sentiment,
                "backtest_win_rate": self.historical_win_rate
                * 100.0,  # Legacy expected % format
                "avg_5d_return": self.avg_5d_return,
            }
        )
        # Always include validation block so downstream consumers get backtest context
        data["validation"] = {
            "backtest_score": self.backtest_score,
            "historical_win_rate": self.historical_win_rate,
            "similar_signals_found": self.similar_signals_found,
            "avg_5d_return": self.avg_5d_return,
        }
        return data


class AgenticSignalGenerator:
    """
    Agentic signal generator for next-day entries.

    Uses multi-step analysis:
    1. Momentum screening (quantity)
    2. Conviction filtering (quality)
    2.5. LEARNED LESSONS FILTER (from backtested patterns - min_score 10)
    3. News/sentiment check (risk)
    4. LLM synthesis (final ranking)

    Integrated with unified_strategy.py for consistent lesson filtering.
    Backtested: +52% better P&L, +51% better avg trade, PF 1.66
    """

    def __init__(
        self,
        max_candidates: int = 50,
        use_llm: bool = True,
        use_lessons: bool = True,
        use_screener_v2: bool = True,
        parent_logger=None,
        min_score: Optional[float] = None,
        restrict_tickers: Optional[List[str]] = None,
        screener_config_override: Optional[Dict[str, Any]] = None,
    ):
        """Initialize with larger candidate pool (50) to leverage 4448-stock universe."""
        self.max_candidates = max_candidates
        self.use_llm = use_llm
        self.use_lessons = use_lessons
        self.logger = parent_logger or logger
        self.config = get_config()
        self.llm_overlay_cfg = getattr(
            self.config.signal_generation, "llm_overlay", None
        )
        self.min_score_override = min_score
        self.restrict_tickers = {
            str(t).upper() for t in (restrict_tickers or []) if str(t).strip()
        } or None
        self.screener_config_override = dict(screener_config_override or {})
        self.signal_pipeline = None
        self.pipeline_diagnostics: Dict[str, Any] = {
            "enabled": False,
            "last_input_count": 0,
            "last_output_count": 0,
            "last_alpha_signal_count": 0,
            "last_error": None,
        }
        try:
            if bool(getattr(self.config.signal_generation, "enabled", True)):
                from autotrade.signals.pipeline import SignalGenerationPipeline

                self.signal_pipeline = SignalGenerationPipeline(
                    parent_logger=self.logger
                )
                self.pipeline_diagnostics["enabled"] = True
                self.logger.info("SignalGenerationPipeline enabled")
        except Exception as e:
            self.logger.warning(
                "Signal pipeline unavailable: %s", safe_exception_text(e)
            )

        # Initialize unified strategy for consistent lesson filtering
        from autotrade.signals.unified_strategy import UnifiedStrategy, MIN_LESSON_SCORE

        self.strategy = UnifiedStrategy(min_score=MIN_LESSON_SCORE)
        self.logger.info(f"Unified Strategy loaded (min_score={MIN_LESSON_SCORE})")

        # Initialize candidate source based on lessons mode
        self.universe_scanner = None
        self.lessons_screener = None
        self.momentum_picker = None

        # Primary: UniverseScanner (DuckDB full universe)
        try:
            from autotrade.signals.universe_scanner import UniverseScanner

            self.universe_scanner = UniverseScanner(parent_logger=self.logger)
            self.logger.info("Using UniverseScanner (DuckDB full-universe scan)")
        except Exception as e:
            self.logger.warning(
                "Universe scanner unavailable: %s", safe_exception_text(e)
            )

        if use_lessons:
            if use_screener_v2:
                try:
                    from autotrade.signals.screener_v2 import get_entry_candidates

                    self.lessons_screener = lambda **kwargs: get_entry_candidates(
                        parent_logger=self.logger,
                        config_override=self.screener_config_override or None,
                        **kwargs,
                    )
                    self.logger.info("Using SCREENER V2 (local data + SR context)")
                except Exception as e:
                    self.logger.warning(
                        "Screener v2 failed: %s, trying lessons screener",
                        safe_exception_text(e),
                    )

            if self.lessons_screener is None:
                try:
                    from autotrade.signals.lessons_screener import get_entry_candidates

                    self.lessons_screener = get_entry_candidates
                    self.logger.info(
                        "Using lessons screener compatibility shim (parquet-only)"
                    )
                except Exception as e:
                    self.logger.warning(
                        "Lessons screener failed: %s, falling back to momentum",
                        safe_exception_text(e),
                    )
                    use_lessons = False

        if not use_lessons or self.lessons_screener is None:
            # Fallback to momentum picker
            from momentum_picker import MomentumPicker

            self.momentum_picker = MomentumPicker(
                max_candidates=max_candidates,
                analyze_sentiment=True,
                strategy="pullback",
            )
            self.logger.info("Using MomentumPicker (pullback strategy)")

        # LessonBook for additional filtering (deprecated - now using unified_strategy)
        self.lesson_book = None

        # Load news analyzer properly (v2 with freshness weighting)
        self.news_analyzer = None
        try:
            from autotrade.analysis.news_sentiment import NewsSentimentAnalyzer

            # Fetch TTL from global config (default 4 hours)
            market_cfg = getattr(self.config, "market_data", None)
            news_ttl = int(
                getattr(market_cfg, "news_cache_ttl_hours", 4) if market_cfg else 4
            )

            # 90 days = ~1.5 quarters of news, freshness-weighted
            self.news_analyzer = NewsSentimentAnalyzer(
                max_news_age_days=90, cache_ttl_hours=news_ttl
            )
            self.logger.debug(
                f"News sentiment analyzer v2 loaded (90-day window, {news_ttl}h TTL)"
            )
        except Exception as e:
            self.logger.warning(
                "News sentiment not available: %s", safe_exception_text(e)
            )

        # Initialize News Momentum Engine (Phase 1 Quick Win)
        self.momentum_engine = None
        try:
            from autotrade.signals.momentum_engine import MomentumEngine

            self.momentum_engine = MomentumEngine()
            self.logger.info("News Momentum Engine initialized")
        except Exception as e:
            self.logger.warning(
                "Could not initialize MomentumEngine: %s",
                safe_exception_text(e),
            )

        self.logger.debug(
            f"Signal Generator initialized (max={max_candidates}, llm={use_llm})"
        )
        if self.restrict_tickers:
            self.logger.info(
                "Restricting agentic signal generation to %d tickers",
                len(self.restrict_tickers),
            )

    def _filter_restricted_tickers(self, signals: List[Dict]) -> List[Dict]:
        if not self.restrict_tickers or not signals:
            return signals
        filtered: List[Dict] = []
        for row in signals:
            symbol = str(row.get("ticker") or row.get("symbol") or "").upper()
            if symbol and symbol in self.restrict_tickers:
                filtered.append(row)
        if filtered:
            self.logger.info(
                "Restricted candidate filter kept %d/%d signals",
                len(filtered),
                len(signals),
            )
        else:
            self.logger.warning("Restricted candidate filter removed all signals")
        return filtered

    def _normalize_strength(self, raw_strength: float) -> float:
        """
        Normalize s1/r1 strength to match rebuild_parallel_v2.py formula.

        The mega_batch produces values in ~19-28 range.
        The lessons expect values in 30-100 range.

        Formula: Map the input range to 30-100 scale.
        Old range: 0-30 (observed max ~28)
        New range: 30-100
        """
        if raw_strength <= 0:
            return 30.0  # Base minimum

        # Map 0-30 input to 30-100 output
        # normalized = 30 + (raw / 30) * 70
        normalized = 30.0 + (raw_strength / 30.0) * 70.0
        return min(100.0, max(30.0, normalized))

    # ============================================================
    # CORE ORCHESTRATION
    # ============================================================

    def run(self) -> List[EntryCandidate]:
        """
        Run the full agentic signal generation pipeline.

        Returns prioritized list of entry candidates.
        """
        import random

        self.logger.info("=" * 60)
        self.logger.info("AGENTIC SIGNAL GENERATOR")
        self.logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 60)

        # Step 1: Get momentum candidates
        self.logger.info("\n[STEP 1] Candidate Screening")
        self.logger.info("-" * 40)
        raw_signals = self._get_momentum_candidates()
        self.logger.info(f"   Found {len(raw_signals)} candidates")

        if not raw_signals:
            self.logger.warning("No candidates found!")
            return []

        # Step 2: Convert to EntryCandidate and score (process all to leverage 4448-stock universe)
        self.logger.info("\n[STEP 2] Entry Quality Scoring")
        self.logger.info("-" * 40)
        candidates = self._score_entry_quality(raw_signals)  # Score all candidates
        self.logger.info(f"   Scored {len(candidates)} candidates")

        # Step 2.5: Apply learned lessons filter (skip if using lessons_screener - already filtered)
        if self.use_lessons and not self.lessons_screener:
            self.logger.info("\n[STEP 2.5] Learned Lessons Filter")
            self.logger.info("-" * 40)
            candidates = self._apply_lessons_filter(candidates)
        elif self.lessons_screener:
            self.logger.info(
                "\n[STEP 2.5] Lessons filter SKIPPED (already applied in screening)"
            )

        # Step 2.6: Apply SMA 200 filter BEFORE news (saves API calls on downtrending stocks)
        self.logger.info("\n[STEP 2.6] SMA 200 Filter (Above 200-day MA)")
        self.logger.info("-" * 40)
        candidates = self._filter_by_sma200(candidates)
        self.logger.info(f"   {len(candidates)} candidates passed SMA 200 filter")

        # Step 2.6.1: Apply dynamic min_score bias filter
        if self.min_score_override is not None:
            prev_count = len(candidates)
            candidates = [
                c for c in candidates if c.entry_score >= self.min_score_override
            ]
            self.logger.info(
                f"   [BIAS] Filtered out {prev_count - len(candidates)} candidates "
                f"below min_score {self.min_score_override:.1f}"
            )

        if not candidates:
            self.logger.warning("No candidates passed quality filters!")
            return []

        # Step 2.6.2: News-Driven Momentum Discovery (Phase 1 Quick Win)
        if self.momentum_engine:
            self.logger.info("\n[STEP 2.6.2] News Momentum Analysis")
            self.logger.info("-" * 40)
            symbols = [c.symbol for c in candidates]
            momentum_setups = self.momentum_engine.detect_setups(symbols)

            # Map setups back to candidates
            setup_map = {s["ticker"]: s for s in momentum_setups}
            for c in candidates:
                if c.symbol in setup_map:
                    setup = setup_map[c.symbol]
                    c.metadata["setup_type"] = setup["setup_type"]
                    c.metadata["setup_confidence"] = setup["setup_confidence"]
                    c.metadata["momentum_catalyst"] = setup["catalyst"]
                    # Boost score for news momentum
                    c.entry_score += 10.0

            self.logger.info(f"   Detected {len(momentum_setups)} news momentum setups")

        # Step 2.7: Shuffle candidates before news analysis (avoid A-Z bias)
        random.shuffle(candidates)
        self.logger.info(
            f"   Shuffled candidates (first 5: {[c.symbol for c in candidates[:5]]})"
        )

        # Step 3: News/sentiment analysis (top 100 for news check)
        self.logger.info("\n[STEP 3] News & Sentiment Analysis")
        self.logger.info("-" * 40)
        candidates = self._analyze_sentiment(candidates[:100])

        # Step 4: LLM synthesis (optional - top 30 for LLM)
        if self.use_llm and self._llm_overlay_enabled():
            self.logger.info("\n[STEP 4] LLM Synthesis")
            self.logger.info("-" * 40)
            candidates = self._llm_synthesis(candidates)
        elif self.use_llm:
            self.logger.info(
                "\n[STEP 4] LLM Synthesis skipped (overlay disabled; rules-only ranking active)"
            )

        # H1 D2 (2026-05-22): overnight-cut recovery boost. No-op while
        # overnight_cut_policy.enabled=false (scaffold for future cutoff PR).
        self._apply_overnight_recovery_boost(candidates)

        # Step 5: Final ranking
        self.logger.info("\n[STEP 5] Final Ranking")
        self.logger.info("-" * 40)
        candidates = self._final_ranking(candidates)

        # Take top N
        final_candidates = candidates[: self.max_candidates]

        # Assign priorities by final score (tie-break by volume_ratio, ATR%, risk_reward)
        for i, c in enumerate(final_candidates):
            c.priority = i + 1

        return final_candidates

    def _apply_overnight_recovery_boost(
        self, candidates: List[EntryCandidate]
    ) -> None:
        """H1 D2 scaffold. Reads yesterday's overnight-cuts file and boosts
        rank for any candidate re-appearing on today's signals. No-op when
        the policy flag is off.
        """
        try:
            from datetime import date as _date, timedelta as _td

            from autotrade.signals.overnight_cuts import (
                DEFAULT_RECOVERY_BOOST,
                apply_recovery_boost,
                load_cuts,
            )

            policy = getattr(self.config, "overnight_cut_policy", None)
            enabled = bool(getattr(policy, "enabled", False)) if policy else False
            boost = (
                float(getattr(policy, "recovery_boost", DEFAULT_RECOVERY_BOOST))
                if policy
                else DEFAULT_RECOVERY_BOOST
            )
            cuts = load_cuts(_date.today() - _td(days=1))
            apply_recovery_boost(
                candidates,
                cuts,
                enabled=enabled,
                boost=boost,
            )
        except Exception as e:
            self.logger.debug(f"overnight recovery boost skipped: {e}")

    def _filter_by_sma200(
        self, candidates: List[EntryCandidate]
    ) -> List[EntryCandidate]:
        """
        Filter candidates to only include stocks above their 200-day SMA.

        This is applied BEFORE news analysis to avoid wasting API calls
        on stocks that are in a long-term downtrend.

        Uses LocalDataProvider for speed (local parquet data).
        """
        if not candidates:
            return []

        entry_cfg = self.config.entry_quality
        relax_if_zero = bool(getattr(entry_cfg, "sma200_relax_if_zero_pass", True))
        relax_max_below_pct = float(
            getattr(entry_cfg, "sma200_relax_max_below_pct", 3.0)
        )

        try:
            from autotrade.utils.local_data_provider import bulk_sma_check

            # Extract symbols

            symbols = [c.symbol for c in candidates]

            # Bulk SMA check (very fast for local data)
            results = bulk_sma_check(symbols, period=200)

            passed = []
            near_sma = []
            filtered_count = 0

            for c in candidates:
                sma_data = results.get(c.symbol, {})

                if sma_data.get("sma") is None:
                    # No SMA data - keep candidate (will be filtered later if needed)
                    passed.append(c)
                    continue

                if sma_data.get("above_sma"):
                    passed.append(c)
                else:
                    filtered_count += 1
                    pct_below = (
                        (sma_data["sma"] - sma_data["price"]) / sma_data["sma"]
                    ) * 100
                    if pct_below <= relax_max_below_pct:
                        near_sma.append(c)
                    self.logger.debug(
                        f"   {c.symbol}: FILTERED - ${sma_data['price']:.2f} < SMA200 ${sma_data['sma']:.2f} "
                        f"(-{pct_below:.1f}%)"
                    )

            if filtered_count > 0:
                self.logger.info(
                    f"   Filtered {filtered_count} stocks below SMA 200 (strict)"
                )

            if passed:
                return passed

            if relax_if_zero:
                if near_sma:
                    self.logger.warning(
                        "   SMA 200 strict filter produced 0/%d passes; relaxing to %d stocks within %.1f%% of SMA 200",
                        len(candidates),
                        len(near_sma),
                        relax_max_below_pct,
                    )
                    return near_sma
                self.logger.warning(
                    "   SMA 200 strict filter produced 0/%d passes; bypassing filter for this cycle",
                    len(candidates),
                )
                return candidates

            return passed

        except ImportError:
            self.logger.warning(
                "   LocalDataProvider not available - skipping SMA 200 filter"
            )
            return candidates
        except Exception as e:
            self.logger.warning(
                f"   SMA 200 filter error: {e} - returning all candidates"
            )
            return candidates

    @staticmethod
    def _is_oom_error(error: Exception) -> bool:
        """Detect OOM-like failures across DuckDB/pandas/runtime paths."""
        message = str(error).lower()
        oom_markers = (
            "out of memory",
            "allocation failure",
            "memoryerror",
            "cannot allocate",
            "oom",
        )
        return any(marker in message for marker in oom_markers)

    # ============================================================
    # PHASE 1 – CANDIDATE SCREENING
    # ============================================================

    def _run_lessons_screener_batched(
        self, *, total_candidates: int = 200, initial_batch_size: int = 80
    ) -> List[Dict]:
        """
        Pull screener candidates in bounded batches to avoid one-shot OOM failures.
        """
        if not self.lessons_screener:
            return []

        target = max(1, int(total_candidates))
        batch_size = max(10, int(initial_batch_size))
        collected: List[Dict] = []
        seen_symbols = set()
        iterations = 0
        max_iterations = max(4, target // 10)

        while len(collected) < target and iterations < max_iterations:
            iterations += 1
            to_fetch = min(batch_size, target - len(collected))
            kwargs: Dict[str, Any] = {"max_candidates": to_fetch}
            if seen_symbols:
                kwargs["exclude_symbols"] = sorted(seen_symbols)

            try:
                batch = self.lessons_screener(**kwargs) or []
            except TypeError:
                # Compatibility path if exclude_symbols is unsupported.
                batch = self.lessons_screener(max_candidates=to_fetch) or []
            except Exception as e:
                if self._is_oom_error(e) and batch_size > 10:
                    next_batch = max(10, batch_size // 2)
                    self.logger.warning(
                        "   Lessons screener OOM at batch=%d; retrying with batch=%d",
                        batch_size,
                        next_batch,
                    )
                    batch_size = next_batch
                    continue
                raise

            if not batch:
                break

            added = 0
            for row in batch:
                symbol = str(row.get("ticker") or row.get("symbol") or "").upper()
                if not symbol or symbol in seen_symbols:
                    continue
                if DELISTED_TOMBSTONES and DELISTED_TOMBSTONES.is_delisted(symbol):
                    continue
                seen_symbols.add(symbol)
                collected.append(row)
                added += 1
                if len(collected) >= target:
                    break

            if added == 0:
                break

        return collected

    def _get_momentum_candidates(self) -> List[Dict]:
        """Get raw candidates from lessons screener or momentum picker."""
        try:
            if self.restrict_tickers and self.lessons_screener:
                symbols = sorted(self.restrict_tickers)
                max_candidates = min(self.max_candidates, len(symbols))
                signals = self.lessons_screener(
                    max_candidates=max_candidates,
                    symbols=symbols,
                    exclude_symbols=None,
                )
                self.logger.info(
                    "   Restricted screener returned %d candidates",
                    len(signals),
                )
                if signals:
                    if DELISTED_TOMBSTONES:
                        signals = [
                            row
                            for row in signals
                            if not DELISTED_TOMBSTONES.is_delisted(
                                str(row.get("ticker") or row.get("symbol") or "")
                            )
                        ]
                    return self._apply_signal_pipeline(signals)

            if self.universe_scanner:
                scan_cfg = self.config.universe_scanner
                scan_limits = [
                    int(scan_cfg.max_candidates),
                    min(int(scan_cfg.max_candidates), 120),
                    80,
                    50,
                ]
                # Keep order and uniqueness.
                scan_limits = list(dict.fromkeys(v for v in scan_limits if v > 0))

                for limit in scan_limits:
                    try:
                        df = self.universe_scanner.combined_scan(max_candidates=limit)
                    except Exception as e:
                        if self._is_oom_error(e):
                            self.logger.warning(
                                "   Universe scanner OOM at max_candidates=%d; retrying smaller batch",
                                limit,
                            )
                            continue
                        self.logger.warning(f"   Universe scanner failed: {e}")
                        break

                    if not df.empty:
                        signals = df.to_dict(orient="records")
                        if DELISTED_TOMBSTONES:
                            signals = [
                                row
                                for row in signals
                                if not DELISTED_TOMBSTONES.is_delisted(
                                    str(row.get("ticker") or row.get("symbol") or "")
                                )
                            ]
                        self.logger.info(
                            f"   Universe scanner returned {len(signals)} candidates"
                        )
                        signals = self._filter_restricted_tickers(signals)
                        return self._apply_signal_pipeline(signals)

                self.logger.warning(
                    "   Universe scanner returned no candidates, falling back"
                )

            if self.lessons_screener:
                # Fallback: lessons-based screening in bounded batches.
                signals = self._run_lessons_screener_batched(
                    total_candidates=200, initial_batch_size=80
                )
                self.logger.info(
                    f"   Lessons screener returned {len(signals)} candidates (batched)"
                )
                if signals:
                    signals = self._filter_restricted_tickers(signals)
                    return self._apply_signal_pipeline(signals)
            if self.momentum_picker:
                signals = self.momentum_picker.run()
                signals = self._filter_restricted_tickers(signals)
                return self._apply_signal_pipeline(signals)
            else:
                self.logger.error("No candidate source configured!")
                return []
        except Exception as e:
            if self._is_oom_error(e):
                self.logger.error(
                    "Candidate screening failed due to OOM after backoff retries: %s",
                    e,
                )
            else:
                self.logger.error(f"Candidate screening failed: {e}")
            return []

    def _apply_signal_pipeline(self, signals: List[Dict]) -> List[Dict]:
        """Optional stage-3 pipeline pass over candidate dictionaries."""
        if not signals or self.signal_pipeline is None:
            return signals
        self.pipeline_diagnostics["last_input_count"] = len(signals)
        try:
            tickers = [s.get("ticker") for s in signals if s.get("ticker")]
            if not tickers:
                return signals
            try:
                from autotrade.signals.screener_v2 import ScreenerV2

                price_df = ScreenerV2(parent_logger=self.logger)._load_price_data(
                    symbols=tickers
                )
            except Exception:
                price_df = None

            context = self.signal_pipeline.build_context(
                tickers=tickers,
                price_data=price_df,
            )
            result = self.signal_pipeline.run(
                context=context,
                legacy_candidates=signals,
                include_baselines=False,
            )
            alpha_count = len(
                getattr(getattr(result, "batch", None), "signals", []) or []
            )
            self.pipeline_diagnostics["last_alpha_signal_count"] = alpha_count
            if result.legacy_candidates:
                self.pipeline_diagnostics["last_output_count"] = len(
                    result.legacy_candidates
                )
                self.pipeline_diagnostics["last_error"] = None
                self.logger.info(
                    "Signal pipeline merged legacy=%d with alpha=%d -> output=%d",
                    len(signals),
                    alpha_count,
                    len(result.legacy_candidates),
                )
                return result.legacy_candidates
            self.pipeline_diagnostics["last_output_count"] = len(signals)
            self.pipeline_diagnostics["last_error"] = None
            return signals
        except Exception as e:
            self.pipeline_diagnostics["last_error"] = str(e)
            self.logger.warning(
                f"Signal pipeline pass failed, keeping legacy candidates: {e}"
            )
            return signals

    # ============================================================
    # PHASE 2 – QUALITY GATING
    # ============================================================

    def _score_entry_quality(self, signals: List[Dict]) -> List[EntryCandidate]:
        """Score entry quality for each candidate with financial DB checks."""
        candidates = []

        # Pre-fetch financial risks for all candidates
        tickers = [s["ticker"] for s in signals if "ticker" in s]

        try:
            earnings_risks = check_earnings_risk(tickers, days=5)
            balance_sheet_risks = check_balance_sheet_health(tickers)
            cash_flow_risks = check_cash_flow_health(tickers)
            dividend_risks = check_dividend_stability(tickers)
            valuation_risks = check_valuation_sanity(tickers)
            options_risks = check_options_positioning(tickers)
        except Exception as e:
            self.logger.warning(f"Financial DB check failed: {e}")
            earnings_risks = {}
            balance_sheet_risks = {}
            cash_flow_risks = {}
            dividend_risks = {}
            valuation_risks = {}
            options_risks = {}

        entry_cfg = self.config.entry_quality
        min_rr = float(getattr(entry_cfg, "min_risk_reward", 1.5))
        min_entry_score = float(getattr(entry_cfg, "min_entry_score", 35.0))

        for sig in signals:
            try:
                # Recalculate s1_strength using CONSISTENT formula (base 30 + 10 per confluence)
                # This matches rebuild_parallel_v2.py so lessons thresholds work
                raw_s1_strength = sig.get("s1_strength", 0) or 0
                s1_strength = self._normalize_strength(raw_s1_strength)

                raw_r1_strength = sig.get("r1_strength", 0) or 0
                r1_strength = self._normalize_strength(raw_r1_strength)

                price_val = sig.get("price", sig.get("Close", sig.get("close", 0))) or 0
                rsi_val = sig.get("rsi", sig.get("rsi_14", sig.get("RSI_14", 50)))
                momentum_val = sig.get("score", sig.get("bullish_score", 0))
                _strat = self.config.strategy

                c = EntryCandidate(
                    symbol=sig["ticker"],
                    price=price_val,
                    momentum_score=momentum_val,
                    rsi=rsi_val,
                    weekly_return=sig.get("weekly_return", 0),
                    s1_price=sig.get("s1_price", 0) or 0,
                    s1_strength=s1_strength,
                    r1_price=sig.get("r1_price", 0) or 0,
                    r1_strength=r1_strength,
                    # Store raw data for lessons filter
                    atr_14=sig.get("atr_14", 0) or 0,
                    support_dist_atr=sig.get("support_dist_atr", 0) or 0,
                    resistance_dist_atr=sig.get("resistance_dist_atr", 0) or 0,
                    stop_atr_mult=sanitize_atr_multiplier(
                        sig.get("stop_atr_mult"),
                        _strat.fallback_stop_atr,
                    ),
                    target_atr_mult=sanitize_atr_multiplier(
                        sig.get("target_atr_mult"),
                        _strat.fallback_target_atr,
                    ),
                )

                # Additional scanner fields
                c.scan_type = sig.get("scan_type", "") or sig.get("scan", "")
                c.volume_ratio = (
                    sig.get("volume_ratio", sig.get("volume_ratio_20d", 0)) or 0
                )

                # ATR percent
                if sig.get("atr_percent"):
                    c.atr_percent = sig.get("atr_percent", 0) or 0
                elif c.atr_14 > 0 and c.price > 0:
                    c.atr_percent = (c.atr_14 / c.price) * 100
                else:
                    c.atr_percent = 0.0

                # ATR-based position sizing
                c.position_size = self._get_position_size(c.atr_percent)

                # Pure ATR-based stop/target (S/R data excluded — unreliable)
                atr = sig.get("atr_14", 0) or (
                    c.price * 0.03
                )  # Fallback to 3% if no ATR

                c.stop_price = c.price - (atr * c.stop_atr_mult)
                c.target_price = c.price + (atr * c.target_atr_mult)

                # Calculate R:R
                risk = c.price - c.stop_price
                reward = c.target_price - c.price
                if risk > 0:
                    c.risk_reward = reward / risk
                else:
                    c.risk_reward = 1.0  # Default if calculation fails

                # Calculate entry score AFTER R:R is known.
                c.entry_score = self._calculate_entry_score(c, sig)

                # === FINANCIAL DB CHECKS ===
                # Apply penalties for financial risks

                # 1. Earnings Risk (Hard filter or massive penalty)
                if c.symbol in earnings_risks:
                    self.logger.info(
                        f"   {c.symbol} has EARNINGS SOON - applying penalty"
                    )
                    c.entry_score -= 50  # Basically kills it
                    c.catalyst_note += " [EARNINGS RISK]"

                # 2. Balance Sheet Risk (-15 pts)
                if c.symbol in balance_sheet_risks:
                    flags = balance_sheet_risks[c.symbol].get("flags", [])
                    self.logger.info(f"   {c.symbol} weak balance sheet: {flags}")
                    c.entry_score -= 15
                    c.catalyst_note += f" [WEAK BS: {','.join(flags)}]"

                # 3. Cash Flow Risk (-10 pts)
                if c.symbol in cash_flow_risks:
                    flags = cash_flow_risks[c.symbol].get("flags", [])
                    self.logger.info(f"   {c.symbol} negative cash flow: {flags}")
                    c.entry_score -= 10
                    c.catalyst_note += " [NEG FCF]"

                # 4. Dividend Risk (-20 pts, hard filter for suspensions)
                if c.symbol in dividend_risks:
                    flags = dividend_risks[c.symbol].get("flags", [])
                    self.logger.info(f"   {c.symbol} dividend risk: {flags}")
                    if "dividend_suspended" in flags:
                        c.entry_score -= 50  # Basically kills it
                    else:
                        c.entry_score -= 20
                    c.catalyst_note += f" [DIVIDEND RISK: {','.join(flags)}]"

                # 5. Valuation Sanity Checks
                if c.symbol in valuation_risks:
                    flags = valuation_risks[c.symbol].get("flags", [])
                    self.logger.info(f"   {c.symbol} valuation risk: {flags}")
                    for f in flags:
                        if "extreme_pe" in f:
                            c.entry_score -= 15
                        elif "high_debt_equity" in f:
                            c.entry_score -= 20
                        elif "earnings_decline" in f:
                            c.entry_score -= 10
                    c.catalyst_note += f" [VALUATION RISK: {','.join(flags)}]"

                # 6. Options Positioning Checks
                if c.symbol in options_risks:
                    flags = options_risks[c.symbol].get("flags", [])
                    self.logger.info(f"   {c.symbol} options risk: {flags}")
                    pc_ratio = options_risks[c.symbol].get("put_call_oi_ratio", 0.0)
                    if pc_ratio > 2.0:
                        c.entry_score -= 20
                    elif pc_ratio > 1.5:
                        c.entry_score -= 10
                    c.catalyst_note += f" [OPTIONS RISK: {','.join(flags)}]"

                # FILTER: Reject weak setups (config-driven thresholds)
                if c.risk_reward < min_rr:
                    self.logger.debug(
                        f"   Skipping {c.symbol}: R:R < {min_rr:.2f} ({c.risk_reward:.2f})"
                    )
                    continue
                if c.entry_score < min_entry_score:
                    self.logger.debug(
                        f"   Skipping {c.symbol}: entry_score < {min_entry_score:.0f} ({c.entry_score:.0f})"
                    )
                    continue

                candidates.append(c)

            except Exception as e:
                self.logger.debug(f"Error processing {sig.get('ticker', '?')}: {e}")

        # Sort by entry score
        candidates.sort(key=lambda x: x.entry_score, reverse=True)
        return candidates

    def _get_position_size(self, atr_percent: float) -> float:
        """ATR-based position sizing per risk tier."""
        base = float(self.config.portfolio.position_size_target)
        if atr_percent <= 0:
            return base
        if atr_percent > self.config.universe_scanner.high_risk_atr:
            return round(base * 0.5, 0)  # ~1500 when base=3000
        if atr_percent >= 6.0:
            return round(base * 0.67, 0)  # ~2000 when base=3000
        return base

    def _calculate_entry_score(self, c: EntryCandidate, sig: Dict) -> float:
        """Calculate entry quality score (0-100) using trade_learner v0.12 rebalancing.

        Prioritizes S/R quality and R:R over RSI pullbacks.
        Added ATR and Spread quality as minor confirmation factors.
        """
        score = 0.0

        # 1. S/R quality component (30 points max) - Granular
        s1_str = float(sig.get("s1_strength", 40))
        r1_str = float(sig.get("r1_strength", 40))
        dist_s1 = float(sig.get("distance_to_s1_pct", 0))
        dist_r1 = float(sig.get("distance_to_r1_pct", 0))
        spread = dist_s1 + dist_r1
        sr_gen_qual = float(sig.get("sr_quality_score", 50))

        # S1 strength (max 10), R1 strength (max 10), Spread (max 5), General (max 5)
        score += np.clip(s1_str / 70.0 * 10.0, 0, 10.0)
        score += np.clip(r1_str / 65.0 * 10.0, 0, 10.0)
        score += np.clip(spread / 10.8 * 5.0, 0, 5.0)
        score += np.clip(sr_gen_qual / 100.0 * 5.0, 0, 5.0)

        # 2. R:R component (25 points max)
        if c.risk_reward >= 3.0:
            score += 25
        elif c.risk_reward >= 2.5:
            score += 20
        elif c.risk_reward >= 2.0:
            score += 15
        elif c.risk_reward >= 1.5:
            score += 5

        # 3. Near support bonus (20 points max)
        support_dist_atr = float(sig.get("support_dist_atr", 5))
        if support_dist_atr <= 0.5:
            score += 20
        elif support_dist_atr <= 1.0:
            score += 15
        elif support_dist_atr <= 1.5:
            score += 10
        elif support_dist_atr <= 2.0:
            score += 5

        # 4. RSI component (15 points max) - PULLBACKS
        if 45 <= c.rsi <= 60:
            score += 15
        elif 40 <= c.rsi < 45 or 60 < c.rsi <= 65:
            score += 10
        elif 65 < c.rsi <= 70:
            score += 5

        # 5. ATR quality (NEW, 5 points max) - trade_learner: 1.5-4.0% is sweet spot
        atr_pct = float(sig.get("atr_pct", 2.0))
        if 1.5 <= atr_pct <= 4.0:
            score += 5
        elif 1.0 <= atr_pct < 1.5 or 4.0 < atr_pct <= 6.0:
            score += 3

        # 6. Spread quality (NEW, 5 points max) - trade_learner: > 10.8%
        if spread >= 10.8:
            score += 5
        elif spread >= 8.0:
            score += 3

        return float(score)

    def _apply_lessons_filter(
        self, candidates: List[EntryCandidate]
    ) -> List[EntryCandidate]:
        """
        Apply learned lessons from backtested trade patterns.

        Uses UnifiedStrategy (min_score 10) to:
        1. Filter candidates below threshold (AVOID patterns)
        2. Score based on learned preferences (PREFER patterns)

        Backtested Results (min_score 10):
        - +52% better P&L vs baseline
        - +51% better avg trade (+0.53% vs +0.35%)
        - Profit Factor 1.66 vs 1.47
        """
        filtered = []
        avoided_count = 0
        boosted_count = 0
        penalized_count = 0

        for c in candidates:
            # Compute derived fields from available data
            atr_percent = 0.0
            distance_to_s1_pct = 0.0
            distance_to_r1_pct = 0.0

            if c.atr_14 > 0 and c.price > 0:
                atr_percent = (c.atr_14 / c.price) * 100

            if c.s1_price > 0 and c.price > 0:
                distance_to_s1_pct = ((c.price - c.s1_price) / c.price) * 100

            if c.r1_price > 0 and c.price > 0:
                distance_to_r1_pct = ((c.r1_price - c.price) / c.price) * 100

            # Build candidate dict for unified strategy
            candidate_dict = {
                "bullish_score": c.momentum_score / 100,  # Normalize
                "s1_strength": c.s1_strength,
                "r1_strength": c.r1_strength,
                "atr_percent": atr_percent,
                "distance_to_s1_pct": distance_to_s1_pct,
                "distance_to_r1_pct": distance_to_r1_pct,
            }

            # Check against unified strategy
            passes, score, rules = self.strategy.passes_filter(candidate_dict)

            if not passes:
                # Get first AVOID rule that matched
                avoid_reason = next(
                    (r for r in rules if "AVOID" in r), f"lesson score {score:.0f}"
                )
                self.logger.debug(f"   âŒ {c.symbol}: {avoid_reason}")
                avoided_count += 1
                continue

            # Apply score adjustment
            c.entry_score = c.entry_score + (score * 0.5)  # Scale lesson score

            if score > 10:
                boosted_count += 1
                self.logger.debug(f"   â¬†ï¸ {c.symbol}: lesson score +{score:.0f}")
            elif score < -10:
                penalized_count += 1
                self.logger.debug(f"   â¬‡ï¸ {c.symbol}: lesson score {score:.0f}")

            filtered.append(c)

        # Log summary
        self.logger.info(
            f"   Results: {avoided_count} avoided | {boosted_count} boosted | {penalized_count} penalized"
        )
        self.logger.info(
            f"   Remaining: {len(filtered)} candidates (from {len(candidates)})"
        )

        # Re-sort by adjusted entry score
        filtered.sort(key=lambda x: x.entry_score, reverse=True)
        return filtered

    # ============================================================
    # PHASE 3 – CONTEXT ENRICHMENT
    # ============================================================

    def _analyze_sentiment(
        self, candidates: List[EntryCandidate]
    ) -> List[EntryCandidate]:
        """Analyze news sentiment for each candidate with freshness weighting."""
        if not self.news_analyzer:
            self.logger.warning(
                "   [!] News analyzer not loaded - this should not happen!"
            )
            return candidates

        self.logger.info(f"   Analyzing news for {len(candidates)} candidates...")
        analyzed = 0
        for c in candidates:
            try:
                # Use v2 analyzer with freshness weighting
                result = self.news_analyzer.analyze_ticker(
                    c.symbol, current_price=c.price
                )

                # Use weighted_score (accounts for freshness + priced-in factor)
                c.news_sentiment = result.get("weighted_score", 0.0)

                # Get the signal for logging
                signal, score, reason = self.news_analyzer.get_sentiment_signal(
                    c.symbol
                )

                # Check for catalysts in headlines (prioritize recent ones)
                if result and result.get("news_count", 0) > 0:
                    for h in result.get("headlines", []):
                        headline = h.get("headline", "").lower()
                        age_days = h.get("age_days", 99)
                        # Only flag catalysts from last 7 days
                        if age_days < 7 and any(
                            kw in headline
                            for kw in [
                                "earnings",
                                "fda",
                                "approval",
                                "contract",
                                "deal",
                                "upgrade",
                                "acquisition",
                                "partner",
                                "launch",
                                "breakthrough",
                            ]
                        ):
                            c.has_catalyst = True
                            c.catalyst_note = h.get("headline", "")[:100]
                            break

                # Log with freshness info
                freshness = result.get("freshness_summary", "")
                if c.has_catalyst:
                    self.logger.info(
                        f"   {c.symbol}: [CATALYST] {c.catalyst_note[:50]}"
                    )
                elif score > 0.3:
                    self.logger.info(
                        f"   {c.symbol}: Bullish news ({score:+.2f}) - {freshness}"
                    )
                elif score < -0.3:
                    self.logger.info(
                        f"   {c.symbol}: [WARNING] Bearish news ({score:+.2f})"
                    )

                analyzed += 1
            except Exception as e:
                self.logger.debug(f"   News error for {c.symbol}: {e}")

        self.logger.info(f"   Analyzed sentiment for {analyzed} candidates")
        return candidates

    # ============================================================
    # PHASE 4 – INTELLIGENT SYNTHESIS
    # ============================================================

    def _llm_overlay_enabled(self) -> bool:
        return bool(
            self.use_llm
            and self.llm_overlay_cfg is not None
            and bool(getattr(self.llm_overlay_cfg, "enabled", False))
        )

    def _llm_overlay_mode(self) -> str:
        mode = str(
            getattr(self.llm_overlay_cfg, "mode", "single_model") or "single_model"
        )
        mode = mode.strip().lower()
        return (
            mode
            if mode in {"single_model", "reject_veto_only", "two_model_router"}
            else "single_model"
        )

    def _llm_overlay_top_slice_size(self) -> int:
        return max(1, int(getattr(self.llm_overlay_cfg, "top_slice_size", 30) or 30))

    def _llm_overlay_timeout_seconds(self) -> float:
        configured = getattr(self.llm_overlay_cfg, "timeout_seconds", None)
        if configured is not None:
            return max(1.0, float(configured))
        return max(
            1.0, float(getattr(self.config.llm, "ollama_timeout", OLLAMA_TIMEOUT))
        )

    def _llm_overlay_override_threshold(self) -> float:
        return max(
            0.0,
            min(
                100.0,
                float(
                    getattr(self.llm_overlay_cfg, "min_confidence_to_override", 70.0)
                    or 70.0
                ),
            ),
        )

    def _llm_overlay_num_ctx(self) -> int:
        return max(1024, int(getattr(self.config.llm, "num_ctx", 4096) or 4096))

    def _apply_overlay_decision(
        self,
        candidate: EntryCandidate,
        *,
        decision: Dict[str, Any],
        role: str,
        allow_promotion: bool = True,
    ) -> None:
        label = normalize_overlay_label(decision.get("action"))
        confidence = float(decision.get("confidence", 50.0) or 50.0)
        candidate.llm_score = float(
            decision.get("score", candidate.llm_score) or candidate.llm_score
        )
        candidate.llm_reasoning = str(decision.get("reasoning") or "").strip()
        candidate.metadata["llm_overlay_last_decision"] = {
            "role": role,
            "model": str(decision.get("model") or ""),
            "label": label,
            "confidence": round(confidence, 2),
            "score": round(float(candidate.llm_score), 2),
            "timed_out": bool(decision.get("timed_out", False)),
            "parse_ok": bool(decision.get("parse_ok", False)),
            "error": str(decision.get("error") or ""),
        }

        override_threshold = self._llm_overlay_override_threshold()
        if label == "reject_no_buy" and confidence >= override_threshold:
            candidate.metadata["llm_overlay_reject_veto"] = True
            candidate.action = "watch"
        elif (
            allow_promotion
            and label == "promote_buy"
            and confidence >= override_threshold
        ):
            candidate.metadata["llm_overlay_reject_veto"] = False
            candidate.action = "buy_open"
        else:
            candidate.metadata.setdefault("llm_overlay_reject_veto", False)

    def _llm_synthesis(self, candidates: List[EntryCandidate]) -> List[EntryCandidate]:
        """Use the configured LLM overlay on the top rules-ranked candidate slice."""
        analyzed = 0
        vetoed = 0
        promoted = 0

        for c in candidates[: self._llm_overlay_top_slice_size()]:
            try:
                if self._llm_overlay_mode() == "two_model_router":
                    reject_model = str(
                        getattr(self.llm_overlay_cfg, "reject_model", "") or ""
                    ).strip()
                    if reject_model:
                        reject_result = self._call_llm_for_candidate(
                            c,
                            model=reject_model,
                        )
                        if reject_result:
                            self._apply_overlay_decision(
                                c,
                                decision=reject_result,
                                role="reject_model",
                            )
                            analyzed += 1
                            if bool(c.metadata.get("llm_overlay_reject_veto")):
                                vetoed += 1
                                self.logger.info(
                                    "   %s: WATCH (reject veto via %s, %.0f)",
                                    c.symbol,
                                    reject_model,
                                    float(
                                        reject_result.get("confidence", 50.0) or 50.0
                                    ),
                                )
                                continue

                    promote_model = str(
                        getattr(self.llm_overlay_cfg, "promote_model", "") or ""
                    ).strip()
                    if promote_model:
                        promote_result = self._call_llm_for_candidate(
                            c,
                            model=promote_model,
                        )
                        if promote_result:
                            self._apply_overlay_decision(
                                c,
                                decision=promote_result,
                                role="promote_model",
                            )
                            analyzed += 1
                            if c.action == "buy_open":
                                promoted += 1
                            self.logger.info(
                                "   %s: %s (%s %.0f)",
                                c.symbol,
                                c.action.upper(),
                                promote_model,
                                float(promote_result.get("confidence", 50.0) or 50.0),
                            )
                else:
                    result = self._call_llm_for_candidate(c)
                    if result:
                        reject_only_mode = (
                            self._llm_overlay_mode() == "reject_veto_only"
                        )
                        self._apply_overlay_decision(
                            c,
                            decision=result,
                            role=(
                                "reject_veto_only"
                                if reject_only_mode
                                else "single_model"
                            ),
                            allow_promotion=not reject_only_mode,
                        )
                        analyzed += 1
                        if bool(c.metadata.get("llm_overlay_reject_veto")):
                            vetoed += 1
                        elif c.action == "buy_open" and not reject_only_mode:
                            promoted += 1
                        self.logger.info(
                            "   %s: %s (%s %.0f)",
                            c.symbol,
                            c.action.upper(),
                            str(result.get("model") or ""),
                            float(result.get("confidence", 50.0) or 50.0),
                        )
            except Exception as e:
                self.logger.debug(f"   LLM error for {c.symbol}: {e}")

        self.logger.info(
            "   LLM overlay analyzed %d calls | promoted=%d | vetoed=%d",
            analyzed,
            promoted,
            vetoed,
        )
        return candidates

    def _call_llm_for_candidate(
        self,
        c: EntryCandidate,
        *,
        model: Optional[str] = None,
    ) -> Optional[Dict]:
        """Call the configured local classifier for a single candidate."""
        model_name = str(
            model or getattr(self.llm_overlay_cfg, "single_model", "") or "qwen2.5:3b"
        ).strip()
        result = query_local_signal_classifier(
            signal_packet=build_signal_packet(c),
            model=model_name,
            ollama_url=str(getattr(self.config.llm, "ollama_url", OLLAMA_URL)),
            timeout_seconds=self._llm_overlay_timeout_seconds(),
            num_ctx=self._llm_overlay_num_ctx(),
        )
        return {
            "action": result.label,
            "confidence": result.confidence,
            "score": result.score,
            "reasoning": result.reasoning,
            "model": result.model,
            "parse_ok": result.parse_ok,
            "timed_out": result.timed_out,
            "error": result.error,
        }

    # ============================================================
    # PHASE 5 – PRIORITIZATION & PERSISTENCE
    # ============================================================

    def _final_ranking(self, candidates: List[EntryCandidate]) -> List[EntryCandidate]:
        """Calculate final score and rank candidates using trade_learner v0.12 weights."""
        for c in candidates:
            # Weighted final score (rebalanced v0.12)
            # Sentiment term reduced from 25 to 7.5 (max 15 points)
            c.final_score = (
                c.entry_score * 0.55  # Entry quality: 55%
                + c.momentum_score * 0.10  # Raw momentum: 10%
                + (c.news_sentiment + 1.0) * 7.5  # Sentiment: 7.5% (scaled to 0-15)
                + c.llm_score * 0.20  # LLM synthesis: 20%
            )

            # Signal family demotions (based on historical family win rates)
            family = str(c.metadata.get("signal_family") or c.strategy_id or "").upper()
            if "TREND_CONTINUATION" in family:
                c.final_score *= 0.60
            elif "BREAKOUT" in family:
                c.final_score *= 0.65
            elif "ACTIVE_ENTRY" in family:
                c.final_score *= 0.85

            # Penalty for negative news
            if c.news_sentiment < -0.3:
                c.final_score *= 0.7

            # Bonus for catalyst
            if c.has_catalyst and c.news_sentiment >= 0:
                c.final_score *= 1.2

            # Add tiny deterministic jitter to avoid flat scores
            c.final_score += self._stable_jitter(c.symbol)
            c.final_score = max(0.0, min(100.0, c.final_score))

            # Score-based action promotion: if the composite score is strong
            # enough, promote from "watch" to actionable regardless of LLM
            # opinion.  The LLM is advisory — the score is the final arbiter.
            vetoed = bool(c.metadata.get("llm_overlay_reject_veto"))
            if vetoed:
                c.action = "watch"
            elif c.action == "watch":
                if c.final_score >= 60:
                    c.action = "buy_open"
                elif c.final_score >= 50 and c.risk_reward >= 1.5:
                    c.action = "buy_dip"

        # Sort by final score
        candidates.sort(
            key=lambda x: (
                x.final_score,
                x.volume_ratio or 0,
                x.atr_percent or 0,
                x.risk_reward or 0,
            ),
            reverse=True,
        )

        # Log top candidates
        self.logger.info("\n   Top 10 Final Rankings:")
        for i, c in enumerate(candidates[:10], 1):
            action_tag = {
                "buy_open": "[BUY]",
                "buy_dip": "[DIP]",
                "watch": "[   ]",
            }.get(c.action, "[   ]")
            self.logger.info(
                f"   {i}. {action_tag} {c.symbol}: Score={c.final_score:.0f} | RSI={c.rsi:.0f} | R:R={c.risk_reward:.1f}"
            )

        return candidates

    @staticmethod
    def _stable_jitter(symbol: str) -> float:
        """Stable small jitter to prevent ties (0.00 - 0.09)."""
        if not symbol:
            return 0.0
        return (sum(ord(c) for c in symbol) % 100) / 1000.0

    def save_signals(
        self,
        candidates: List[EntryCandidate],
        target_date: Optional[str] = None,
        *,
        source: str = "unknown",
        allow_overwrite: bool = False,
        min_count: int = 10,
        enforce_filters: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save signals for day_manager.

        Args:
            target_date: Trading date these signals are for (YYYY-MM-DD).
                If None, auto-detects: after 6 PM uses next trading day.
            source: Logical origin of this signals batch (diagnostic only).
            allow_overwrite: When false, refuses to overwrite an existing file.
            min_count: Minimum signal count required to persist.
            enforce_filters: When false, preserve the provided signal universe exactly.
            metadata: Optional diagnostic manifest fields to persist with the file.
        """
        if target_date:
            date_str = str(target_date)
        else:
            now = datetime.now()
            if now.hour >= 18:
                from autotrade.utils.market_time import next_trading_day

                date_str = next_trading_day(now.date()).strftime("%Y-%m-%d")
            else:
                date_str = now.strftime("%Y-%m-%d")

        def _sanitize_saved_signal(signal: Dict[str, Any]) -> Dict[str, Any]:
            row = dict(signal or {})
            clean_strategy_params = sanitize_strategy_params(
                row.get("strategy_params", {}),
                fallback_stop=2.0,
                fallback_target=3.0,
            )
            row["strategy_params"] = clean_strategy_params
            if row.get("entry_score") is None:
                fallback_entry_score = row.get("final_score")
                if fallback_entry_score is None:
                    fallback_entry_score = row.get("confidence")
                if fallback_entry_score is not None:
                    row["entry_score"] = fallback_entry_score
            final_score = _as_float(row.get("final_score"), None)
            ranking_score = _as_float(row.get("ranking_score"), None)
            if final_score is not None:
                row["final_score"] = round(max(0.0, min(100.0, final_score)), 2)
                row["ranking_score"] = row["final_score"]
            elif ranking_score is not None:
                row["ranking_score"] = round(max(0.0, min(100.0, ranking_score)), 2)
                row["final_score"] = row["ranking_score"]
            row["stop_atr_mult"] = sanitize_atr_multiplier(
                row.get("stop_atr_mult"),
                clean_strategy_params.get("stop_atr_mult", 2.0),
            )
            row["target_atr_mult"] = sanitize_atr_multiplier(
                row.get("target_atr_mult"),
                clean_strategy_params.get("target_atr_mult", 3.0),
            )
            return row

        # Canonical format signals
        signals = [
            _sanitize_saved_signal(c.to_dict() if hasattr(c, "to_dict") else c)
            for c in candidates
        ]

        output = {
            "date": date_str,
            "generated_at": datetime.now().isoformat(),
            "generator": "agentic",
            "total_signals": len(signals),
            "signals": signals,
            "source": source,
            "signal_manifest": {
                "source": source,
                "input_total": len(signals),
                "enforce_filters": bool(enforce_filters),
            },
        }
        if isinstance(metadata, dict) and metadata:
            output["signal_manifest"].update(
                {str(key): value for key, value in metadata.items() if str(key).strip()}
            )

        output_path = LOG_DIR / f"signals_{date_str}.json"
        excluded = {
            "AAPL",
            "MSFT",
            "NVDA",
            "TSLA",
            "AMZN",
            "GOOGL",
            "META",
            "SPY",
            "QQQ",
            "DIA",
            "IWM",
            "VXX",
            "UVXY",
            "NEE",
            "SO",
        }
        excluded.update(HARD_BLOCK_MEGA_CAP)

        # Market Cap Enforcer ($2B - $10B) plus post-spike long exclusion.
        universe_cfg = getattr(getattr(self, "config", None), "universe_scanner", None)
        min_cap = float(getattr(universe_cfg, "min_market_cap", DEFAULT_MIN_MARKET_CAP))
        max_cap = float(getattr(universe_cfg, "max_market_cap", DEFAULT_MAX_MARKET_CAP))
        post_spike_volume = float(
            getattr(universe_cfg, "post_spike_volume_threshold", 5.0) or 5.0
        )
        post_spike_range_atr = float(
            getattr(universe_cfg, "post_spike_range_atr_threshold", 1.5) or 1.5
        )

        filtered_signals = []
        malformed_geometry_total = 0
        repaired_geometry_total = 0
        if enforce_filters:
            for s in signals:
                sym = str(s.get("symbol") or s.get("ticker") or "").upper()
                if sym in excluded:
                    self.logger.info(f"   [FILTER] Skipping {sym}: hard exclusion list")
                    continue

                rejection = signal_universe_rejection_reason(
                    s,
                    min_market_cap=min_cap,
                    max_market_cap=max_cap,
                    post_spike_volume_threshold=post_spike_volume,
                    post_spike_range_atr_threshold=post_spike_range_atr,
                )
                if rejection:
                    self.logger.info(
                        "   [FILTER] Skipping %s: %s", sym or "UNKNOWN", rejection
                    )
                    continue

                repaired, geometry_reason = _repair_or_reject_long_geometry(s)
                if geometry_reason == "repaired_long_geometry":
                    repaired_geometry_total += 1
                    s = repaired or s
                elif geometry_reason:
                    malformed_geometry_total += 1
                    self.logger.info(
                        "   [FILTER] Skipping %s: %s", sym or "UNKNOWN", geometry_reason
                    )
                    continue

                filtered_signals.append(s)
        else:
            filtered_signals = list(signals)

        if len(filtered_signals) < int(min_count):
            self.logger.warning(
                "Refusing to save signals: count %d < min_count %d (source=%s)",
                len(filtered_signals),
                int(min_count),
                source,
            )
            return output_path

        output["signals"] = filtered_signals
        output["total_signals"] = len(filtered_signals)
        output["signal_manifest"]["saved_total"] = len(filtered_signals)
        output["signal_manifest"]["filtered_out_total"] = len(signals) - len(
            filtered_signals
        )
        output["signal_manifest"]["malformed_geometry_filtered_total"] = (
            malformed_geometry_total
        )
        output["signal_manifest"]["long_geometry_repaired_total"] = (
            repaired_geometry_total
        )

        if output_path.exists() and not allow_overwrite:
            self.logger.warning(
                "Refusing to overwrite existing signals file: %s (source=%s)",
                output_path,
                source,
            )
            return output_path

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        self.logger.info(f"\n>>> Saved {len(signals)} signals to {output_path}")

        if MONITORING_AVAILABLE:
            try:
                collector = get_metrics_collector()
                collector.aggregate_signal_lifecycle(
                    family="agentic",
                    signals_evaluated=len(signals),
                    signals_accepted=len(signals),
                )
            except Exception:
                pass

        return output_path

    def print_summary(self, candidates: List[EntryCandidate]):
        """Print final summary. Handles both EntryCandidate objects and plain dicts."""

        def _get(c, attr, default=None):
            """Get attribute from either a dataclass/object or a plain dict."""
            if isinstance(c, dict):
                return c.get(attr, default)
            return getattr(c, attr, default)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("NEXT-DAY ENTRY WATCHLIST")
        self.logger.info("=" * 60)

        buy_open = [c for c in candidates if _get(c, "action") == "buy_open"]
        buy_dip = [c for c in candidates if _get(c, "action") == "buy_dip"]
        watch = [c for c in candidates if _get(c, "action") == "watch"]

        if buy_open:
            self.logger.info(f"\n[BUY AT OPEN] ({len(buy_open)})")
            for c in buy_open[:5]:
                price = _get(c, "price") or 0.0
                stop = _get(c, "stop_price") or 0.0
                target = _get(c, "target_price") or 0.0
                self.logger.info(
                    f"   {_get(c, 'priority', '?')}. {_get(c, 'symbol', '?')}: "
                    f"${price:.2f} | Stop ${stop:.2f} | Target ${target:.2f}"
                )
                reasoning = _get(c, "llm_reasoning")
                if reasoning:
                    self.logger.info(f"      -> {str(reasoning)[:60]}")

        if buy_dip:
            self.logger.info(f"\n[BUY ON DIP] ({len(buy_dip)})")
            for c in buy_dip[:5]:
                s1 = _get(c, "s1_price") or _get(c, "price") or 0.0
                self.logger.info(
                    f"   {_get(c, 'priority', '?')}. {_get(c, 'symbol', '?')}: "
                    f"Wait for pullback to ~${s1:.2f}"
                )

        if watch:
            self.logger.info(f"\n[WATCH LIST] ({len(watch)})")
            for c in watch[:5]:
                price = _get(c, "price") or 0.0
                rsi = _get(c, "rsi") or 0.0
                self.logger.info(
                    f"   {_get(c, 'priority', '?')}. {_get(c, 'symbol', '?')}: "
                    f"${price:.2f} | RSI={rsi:.0f}"
                )

        self.logger.info("\n" + "=" * 60)


def main():
    use_llm = "--no-llm" not in sys.argv
    use_lessons = "--no-lessons" not in sys.argv
    save_plan = "--save" in sys.argv or "--execute" in sys.argv

    generator = AgenticSignalGenerator(
        max_candidates=20, use_llm=use_llm, use_lessons=use_lessons
    )
    candidates = generator.run()

    if not candidates:
        generator.logger.warning("No candidates generated!")
        return 1

    try:
        val_cfg = get_config().signal_validation
        if val_cfg.enabled and candidates:
            from autotrade.backtesting.signal_validator import SignalValidator

            validator = SignalValidator(parent_logger=generator.logger)
            candidates, val_meta = validator.validate_and_filter(
                candidates,
                logger=generator.logger,
                adjust_scores=True,
                weight_signal=getattr(val_cfg, "score_weight_signal", 1.0),
                weight_backtest=getattr(val_cfg, "score_weight_backtest", 0.3),
                return_details=True,
            )
            rejected = val_meta.get("rejected", 0)
            generator.logger.info(
                f"Signal validation: {len(candidates)} passed, {rejected} rejected "
                f"(min_score={val_cfg.min_backtest_score}, min_matches={val_cfg.min_similar_signals})"
            )
    except Exception as e:
        generator.logger.warning(f"Signal validation failed (non-fatal): {e}")

    if not candidates:
        generator.logger.warning("No candidates passed validation")
        return 1

    generator.print_summary(candidates)

    if save_plan:
        generator.save_signals(candidates)
        generator.logger.info("[OK] Signals saved for tomorrow's trading")
    else:
        generator.logger.info("\n[TIP] Run with --save to save signals for day_manager")

    return 0


if __name__ == "__main__":
    sys.exit(main())
