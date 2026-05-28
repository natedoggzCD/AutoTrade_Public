"""
Premarket Manager
=================
Dedicated premarket orchestration (4:00 AM - 9:30 AM ET) that prepares a
standardized morning intelligence handoff for Day Manager.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from dataclasses import dataclass, asdict
from zoneinfo import ZoneInfo

from autotrade.utils.news_aggregator import NewsAggregator
from autotrade.utils.momentum_scanner import load_momentum_watchlist
from autotrade.utils.premarket_vwap import (
    PremarketVWAPTracker,
    STATE_MIXED,
    STATE_STRONG_ABOVE,
    STATE_STRONG_BELOW,
    STATE_WAIT,
)
from autotrade.utils.stocktwits_scraper import StocktwitsScraper
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetStatus, AssetClass
from autotrade.utils.alpaca_client_factory import (
    resolve_alpaca_credentials,
    create_data_client,
    create_trading_client,
)
from autotrade.monitoring.liquidity_gate import LiquidityGate
from autotrade.signals.support_resistance import estimate_sr_levels
from autotrade.core.research_artifacts import load_latest_research_artifact_bundle
from autotrade.utils.research_freshness import check_research_freshness
from autotrade.utils.market_time import get_pm_plan_date

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from autotrade.utils.intraday_data_provider import get_intraday_bars

    INTRADAY_PROVIDER_AVAILABLE = True
except Exception:
    get_intraday_bars = None
    INTRADAY_PROVIDER_AVAILABLE = False

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
PROJECT_DIR = Path(
    os.environ.get("AUTOTRADE_ROOT", Path(__file__).resolve().parents[2])
)

PHASE_MARKET_CONTEXT = "MARKET_CONTEXT"  # 4:00-6:00 ET
PHASE_POSITION_CHECKS = "POSITION_CHECKS"  # 6:00-7:00 ET
PHASE_WATCHLIST_SCAN = "WATCHLIST_SCAN"  # 7:00-8:30 ET
PHASE_OPEN_PREP = "OPEN_PREP"  # 8:30-9:30 ET
PHASE_OUTSIDE_WINDOW = "OUTSIDE_WINDOW"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


class _DisabledNewsAggregator:
    def collect(self, symbol: str) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "available": False,
            "sentiment_score": 0.0,
            "headline_count": 0,
            "coverage": "none",
            "confidence": 0.0,
            "headlines": [],
            "source_status": {"news": "disabled"},
        }


class _DisabledStocktwitsScraper:
    def fetch(self, symbol: str) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "available": False,
            "sentiment_score": 0.0,
            "bull_bear_ratio": 1.0,
            "message_velocity": 0.0,
            "is_trending": False,
            "coverage": "none",
            "confidence": 0.0,
            "source_status": "disabled",
        }


@dataclass
class PreMarketData:
    """Pre-market data for a single ticker."""

    ticker: str

    # Prices
    prev_close: float = 0.0
    premarket_price: float = 0.0
    premarket_high: float = 0.0
    premarket_low: float = 0.0
    premarket_open: float = 0.0

    # Gap analysis
    gap_pct: float = 0.0
    gap_direction: str = "flat"  # up, down, flat

    # Volume
    premarket_volume: int = 0
    avg_volume: int = 0
    volume_ratio: float = 0.0  # premarket vs avg daily

    # Trend
    premarket_trend: str = "flat"  # bullish, bearish, flat
    premarket_range_pct: float = 0.0

    # Quality
    has_data: bool = False
    liquidity_score: float = 0.0  # 0-100
    spread_pct: float = 0.0
    is_tradable: bool = True
    liquidity_block_reason: str = ""

    # Timestamps
    last_update: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


class PreMarketAnalyzer:
    """
    Analyzes pre-market data for watchlist stocks.

    Pre-market hours: 4:00 AM - 9:30 AM ET
    """

    def __init__(self, api_key: str = None, api_secret: str = None):
        """Initialize with Alpaca credentials."""
        creds = resolve_alpaca_credentials(
            api_key=api_key,
            secret_key=api_secret,
            require=False,
        )
        self.api_key = creds.api_key if creds else None
        self.api_secret = creds.secret_key if creds else None
        self.is_paper = bool(creds.paper) if creds else True

        # Alpaca clients
        self.api = None
        self.data_api = None
        self.liquidity_gate = LiquidityGate()
        self._init_clients()
        self._cached_youtube_intel: Dict[str, Any] = {}
        self._last_adjust_snapshot: Dict[str, Dict[str, Any]] = {}

    def _init_clients(self):
        """Initialize Alpaca API clients."""
        try:
            self.api = create_trading_client(
                api_key=self.api_key,
                secret_key=self.api_secret,
                paper=self.is_paper,
                validate_connection=True,
                retries=3,
                retry_delay_seconds=2.0,
                logger=logger,
                require_credentials=True,
            )

            self.data_api = create_data_client(
                api_key=self.api_key,
                secret_key=self.api_secret,
                paper=self.is_paper,
                require_credentials=True,
            )

            logger.debug("Alpaca clients initialized for pre-market data")

        except Exception as e:
            logger.error(f"Failed to initialize Alpaca clients: {e}")
            raise

    def get_premarket_quote(self, ticker: str) -> Optional[Dict]:
        """Get latest pre-market quote for a ticker."""
        try:
            from alpaca.data.requests import StockLatestQuoteRequest

            request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
            quote = self.data_api.get_stock_latest_quote(request)

            if ticker in quote:
                q = quote[ticker]
                return {
                    "bid": float(q.bid_price) if q.bid_price else 0,
                    "ask": float(q.ask_price) if q.ask_price else 0,
                    "mid": (float(q.bid_price) + float(q.ask_price)) / 2
                    if q.bid_price and q.ask_price
                    else 0,
                    "bid_size": int(q.bid_size) if q.bid_size else 0,
                    "ask_size": int(q.ask_size) if q.ask_size else 0,
                    "timestamp": str(q.timestamp),
                }
            return None
        except Exception as e:
            logger.debug(f"Quote error for {ticker}: {e}")
            return None

    def get_premarket_bars(self, ticker: str) -> pd.DataFrame:
        """Get pre-market bars for today."""
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            # Pre-market starts 4 AM ET
            now = datetime.now()
            today_start = now.replace(hour=4, minute=0, second=0, microsecond=0)

            request = StockBarsRequest(
                symbol_or_symbols=ticker, timeframe=TimeFrame.Minute, start=today_start
            )

            bars = self.data_api.get_stock_bars(request)

            if ticker in bars and len(bars[ticker]) > 0:
                data = []
                for bar in bars[ticker]:
                    data.append(
                        {
                            "timestamp": bar.timestamp,
                            "open": float(bar.open),
                            "high": float(bar.high),
                            "low": float(bar.low),
                            "close": float(bar.close),
                            "volume": int(bar.volume),
                        }
                    )
                return pd.DataFrame(data)

            return pd.DataFrame()

        except Exception as e:
            logger.debug(f"Bars error for {ticker}: {e}")
            return pd.DataFrame()

    def get_previous_close(self, ticker: str) -> float:
        """Get previous day's closing price."""
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            # Get yesterday's daily bar
            end = datetime.now().replace(hour=0, minute=0)
            start = end - timedelta(days=5)

            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )

            bars = self.data_api.get_stock_bars(request)

            if ticker in bars and len(bars[ticker]) > 0:
                return float(bars[ticker][-1].close)

            return 0.0

        except Exception as e:
            logger.debug(f"Previous close error for {ticker}: {e}")
            return 0.0

    def get_avg_volume(self, ticker: str, days: int = 20) -> int:
        """Get average daily volume."""
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            end = datetime.now()
            start = end - timedelta(days=days + 5)  # Extra buffer

            request = StockBarsRequest(
                symbol_or_symbols=ticker, timeframe=TimeFrame.Day, start=start, end=end
            )

            bars = self.data_api.get_stock_bars(request)

            if ticker in bars and len(bars[ticker]) > 0:
                volumes = [int(bar.volume) for bar in bars[ticker]]
                return int(sum(volumes) / len(volumes))

            return 0

        except Exception as e:
            logger.debug(f"Avg volume error for {ticker}: {e}")
            return 0

    def analyze_ticker(self, ticker: str) -> PreMarketData:
        """
        Full pre-market analysis for a single ticker.

        Returns PreMarketData with all metrics.
        """
        pm = PreMarketData(ticker=ticker)

        try:
            # Get previous close
            pm.prev_close = self.get_previous_close(ticker)

            # Get current quote (pre-market or extended hours)
            quote = self.get_premarket_quote(ticker)
            if quote and quote["mid"] > 0:
                pm.premarket_price = quote["mid"]
                pm.has_data = True
                pm.last_update = quote["timestamp"]

            # Get pre-market bars for OHLC
            bars = self.get_premarket_bars(ticker)
            if not bars.empty:
                pm.premarket_open = float(bars.iloc[0]["open"])
                pm.premarket_high = float(bars["high"].max())
                pm.premarket_low = float(bars["low"].min())
                pm.premarket_volume = int(bars["volume"].sum())

                # Use latest bar close if no quote
                if pm.premarket_price == 0:
                    pm.premarket_price = float(bars.iloc[-1]["close"])
                    pm.has_data = True

                # Pre-market range
                if pm.premarket_low > 0:
                    pm.premarket_range_pct = (
                        (pm.premarket_high - pm.premarket_low) / pm.premarket_low * 100
                    )

                # Pre-market trend (compare open to latest)
                if pm.premarket_open > 0:
                    pm_change = (
                        (pm.premarket_price - pm.premarket_open)
                        / pm.premarket_open
                        * 100
                    )
                    if pm_change > 0.5:
                        pm.premarket_trend = "bullish"
                    elif pm_change < -0.5:
                        pm.premarket_trend = "bearish"
                    else:
                        pm.premarket_trend = "flat"

            # Gap calculation
            if pm.prev_close > 0 and pm.premarket_price > 0:
                pm.gap_pct = (pm.premarket_price - pm.prev_close) / pm.prev_close * 100
                if pm.gap_pct > 0.5:
                    pm.gap_direction = "up"
                elif pm.gap_pct < -0.5:
                    pm.gap_direction = "down"
                else:
                    pm.gap_direction = "flat"

            # Volume analysis
            pm.avg_volume = self.get_avg_volume(ticker)
            if pm.avg_volume > 0 and pm.premarket_volume > 0:
                # Pre-market is ~10% of day volume on average
                expected_pm_vol = pm.avg_volume * 0.1
                pm.volume_ratio = pm.premarket_volume / expected_pm_vol

            # Liquidity score (0-100)
            # Based on pre-market volume and bid-ask spread
            if pm.premarket_volume > 0:
                # Volume component (up to 50 points)
                vol_score = min(pm.volume_ratio * 25, 50)

                # Spread component (up to 50 points) - needs quote
                spread_score = 50  # Default
                if quote and quote["bid"] > 0 and quote["ask"] > 0:
                    spread_pct = (quote["ask"] - quote["bid"]) / quote["mid"] * 100
                    pm.spread_pct = float(spread_pct)
                    if spread_pct < 0.1:
                        spread_score = 50
                    elif spread_pct < 0.3:
                        spread_score = 40
                    elif spread_pct < 0.5:
                        spread_score = 30
                    elif spread_pct < 1.0:
                        spread_score = 20
                    else:
                        spread_score = 10

                pm.liquidity_score = vol_score + spread_score

            decision = self.liquidity_gate.evaluate(
                price=pm.premarket_price or pm.prev_close,
                bid_price=(quote or {}).get("bid", 0.0),
                ask_price=(quote or {}).get("ask", 0.0),
                avg_volume=pm.avg_volume,
                session_volume=pm.premarket_volume,
            )
            pm.is_tradable = bool(decision.tradable)
            pm.liquidity_block_reason = str(decision.reason)
            if decision.spread_pct > 0:
                pm.spread_pct = float(decision.spread_pct)

        except Exception as e:
            logger.error(f"Pre-market analysis failed for {ticker}: {e}")

        return pm

    def get_active_symbols(self) -> set[str]:
        """Fetch set of active and tradable symbols from Alpaca."""
        if not self.api:
            return set()
        try:
            req = GetAssetsRequest(
                status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY
            )
            assets = self.api.get_all_assets(req)
            return {a.symbol.upper() for a in assets if a.tradable}
        except Exception as e:
            logger.debug(f"Failed to fetch active assets: {e}")
            return set()

    def analyze_watchlist(self, tickers: List[str]) -> List[PreMarketData]:
        """Analyze pre-market data for a list of tickers."""
        results = []
        for ticker in tickers:
            pm = self.analyze_ticker(ticker)
            results.append(pm)

            # Log significant gaps
            if pm.has_data:
                if abs(pm.gap_pct) > 2:
                    direction = "GAP UP" if pm.gap_pct > 0 else "GAP DOWN"
                    logger.info(
                        f"   [{direction}] {ticker}: {pm.gap_pct:+.1f}%, PM trend: {pm.premarket_trend}"
                    )

        return results


class PremarketManager:
    """Premarket lifecycle runner with standardized Day Manager handoff."""

    PHASE_WINDOWS: Tuple[Tuple[str, time, time], ...] = (
        (PHASE_MARKET_CONTEXT, time(hour=4, minute=0), time(hour=6, minute=0)),
        (PHASE_POSITION_CHECKS, time(hour=6, minute=0), time(hour=7, minute=0)),
        (PHASE_WATCHLIST_SCAN, time(hour=7, minute=0), time(hour=8, minute=30)),
        (PHASE_OPEN_PREP, time(hour=8, minute=30), time(hour=9, minute=30)),
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        output_dir: Optional[Path] = None,
        premarket_analyzer: Optional[Any] = None,
        news_aggregator: Optional[NewsAggregator] = None,
        stocktwits_scraper: Optional[StocktwitsScraper] = None,
        max_watchlist_symbols: int = 40,
        now_provider: Optional[Any] = None,
    ) -> None:
        creds = resolve_alpaca_credentials(
            api_key=api_key,
            secret_key=api_secret,
            require=False,
        )
        self.api_key = creds.api_key if creds else None
        self.api_secret = creds.secret_key if creds else None
        self.output_dir = Path(output_dir or (PROJECT_DIR / "plans"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_watchlist_symbols = max(1, int(max_watchlist_symbols))
        self.now_provider = now_provider

        self.premarket_analyzer = premarket_analyzer or self._build_analyzer()
        self.news_aggregator = news_aggregator or NewsAggregator()
        self.stocktwits_scraper = stocktwits_scraper or StocktwitsScraper()
        self.last_handoff: Dict[str, Any] = {}
        self._cached_youtube_intel: Dict[str, Any] = {}
        self._last_adjust_snapshot: Dict[str, Dict[str, Any]] = {}
        self.momentum_scanner_cfg = None
        self.momentum_scanner_state: Dict[str, Any] = {}
        self._active_symbols_cache: Optional[set[str]] = None

        # Optional config wiring, with safe defaults when config is unavailable.
        self.weights = {
            "gap": 1.0,
            "volume": 1.0,
            "news": 1.0,
            "stocktwits": 0.75,
            "vwap": 1.0,
            "sr": 0.8,
        }
        self._load_config_weights()

    def resolve_phase(self, now_et: Optional[datetime] = None) -> str:
        now = self._to_et(now_et)
        current_t = now.time()
        for phase, start_t, end_t in self.PHASE_WINDOWS:
            if start_t <= current_t < end_t:
                return phase
        return PHASE_OUTSIDE_WINDOW

    def _load_market_intelligence(self) -> Optional[dict]:
        """Load today's YouTube market intelligence report."""
        rag_dir = PROJECT_DIR / "data" / "youtube" / "rag" / "daily_reports"

        # Try today first, then yesterday
        for offset in [0, 1]:
            date_str = (datetime.now() - timedelta(days=offset)).strftime("%Y-%m-%d")
            report_path = rag_dir / f"{date_str}_consolidated.json"
            if report_path.exists():
                try:
                    with open(report_path, "r", encoding="utf-8") as f:
                        report = json.load(f)
                    return report
                except Exception as e:
                    logger.warning(f"Failed to parse YouTube report {report_path}: {e}")
        return None

    def _apply_intelligence_to_context(self, context: dict, report: dict) -> dict:
        """Enrich market context with YouTube intelligence signals."""
        report = report if isinstance(report, dict) else {}
        flattened = context.get("youtube_intelligence")
        flattened = flattened if isinstance(flattened, dict) else {}

        regime = (
            report.get("regime")
            if isinstance(report.get("regime"), str)
            else report.get("market_regime")
        )
        if not regime and isinstance(report.get("regime"), dict):
            regime = report.get("regime", {}).get("classification")
        if not regime:
            regime = flattened.get("regime", "NEUTRAL")

        signals = (
            report.get("trading_signals")
            if isinstance(report.get("trading_signals"), dict)
            else {}
        )

        sizing = signals.get("sizing_multiplier")
        if sizing is None:
            sizing = signals.get("position_sizing_multiplier")
        if sizing is None:
            sizing = report.get("sizing_multiplier", flattened.get("sizing_multiplier", 1.0))

        context["youtube_regime"] = str(regime or "NEUTRAL").upper()
        context["position_sizing_multiplier"] = sizing

        # Adjust premarket thresholds based on regime
        # (These can be used by _rank_watchlist and _assess_positions)
        if str(regime or "").upper() in ("RISK-OFF", "RISK-OFF-LIGHT", "CRASH"):
            context["gap_threshold_tightened"] = True
            logger.info(
                f"[YOUTUBE-PM] {str(regime or 'NEUTRAL').upper()} regime: applying defensive gap filters"
            )

        # Sector bias
        avoid = [
            s["sector"]
            for s in signals.get("sector_bias", [])
            if s.get("bias") == "AVOID"
        ]
        if not avoid:
            avoid = list(flattened.get("avoid_sectors") or [])
        if avoid:
            context["avoid_sectors"] = avoid

        favor = [
            s["sector"]
            for s in signals.get("sector_bias", [])
            if s.get("bias") == "OVERWEIGHT"
        ]
        if not favor:
            favor = list(flattened.get("favor_sectors") or [])
        if favor:
            context["favor_sectors"] = favor

        return context

    def run_cycle(
        self,
        watchlist: Sequence[Any],
        holdings: Optional[Sequence[str]] = None,
        now_et: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Run a premarket cycle for the current ET window and produce handoff.
        """
        timestamp = self._to_et(now_et)
        phase = self.resolve_phase(timestamp)
        overnight_gate = self._overnight_gate_status(timestamp)
        log = getattr(self, "logger", logger)
        if overnight_gate.get("should_wait"):
            log.warning(
                "[PREMARKET] Overnight workflow not complete yet "
                "(reason=%s, age=%s, deadline=%s ET) - delaying premarket handoff",
                overnight_gate.get("workflow_reason", "unknown"),
                overnight_gate.get("age_hours", "unknown"),
                overnight_gate.get("fallback_deadline_et", "08:15"),
            )
            self.last_handoff = {
                "generated_at_et": timestamp.isoformat(),
                "phase": phase,
                "status": "waiting_for_overnight",
                "error": "overnight_workflow_incomplete",
                "retry_after_seconds": int(
                    overnight_gate.get("retry_after_seconds", 300)
                ),
                "overnight_gate": overnight_gate,
                "degraded_mode": True,
            }
            return self.last_handoff

        holdings_symbols = [str(s).upper() for s in (holdings or []) if str(s).strip()]
        research_bundle = load_latest_research_artifact_bundle(self.output_dir)
        momentum_state = self._load_live_momentum_watchlist(timestamp)
        merged_watchlist = self._merge_research_bundle_watchlist(
            watchlist, research_bundle
        )
        merged_watchlist = self._merge_live_momentum_watchlist(
            merged_watchlist, momentum_state.get("symbols", [])
        )

        # Filter out delisted/non-tradable tickers to prevent noise and fetch errors
        merged_watchlist = self._filter_delisted_tickers(merged_watchlist)

        market_context = self._collect_market_context()

        # Integrate YouTube Intelligence (Phase 1B)
        intel = market_context.get("youtube_intelligence")
        if not isinstance(intel, dict) or not intel:
            intel = self._load_market_intelligence()
        if intel:
            market_context = self._apply_intelligence_to_context(market_context, intel)

        position_assessments = (
            self._assess_positions(holdings_symbols)
            if phase in {PHASE_POSITION_CHECKS, PHASE_WATCHLIST_SCAN, PHASE_OPEN_PREP}
            else []
        )
        ranked_watchlist = (
            self._rank_watchlist(merged_watchlist)
            if phase in {PHASE_WATCHLIST_SCAN, PHASE_OPEN_PREP}
            else []
        )
        if ranked_watchlist:
            ranked_watchlist = self._apply_rvol_momentum(ranked_watchlist, timestamp)
        adjust_triggered, adjust_reasons, scalp_watchlist = (
            self._compute_dynamic_adjust(ranked_watchlist, timestamp)
        )
        open_alerts = (
            self._build_open_alerts(
                ranked_watchlist, position_assessments, market_context
            )
            if phase == PHASE_OPEN_PREP
            else []
        )

        coverage = self._summarize_coverage(
            ranked_watchlist=ranked_watchlist,
            position_assessments=position_assessments,
            market_context=market_context,
        )

        handoff = {
            "generated_at_et": timestamp.isoformat(),
            "phase": phase,
            "phase_windows_et": [
                {
                    "phase": p,
                    "start": start.isoformat(timespec="minutes"),
                    "end": end.isoformat(timespec="minutes"),
                }
                for p, start, end in self.PHASE_WINDOWS
            ],
            "market_bias": market_context.get("futures_direction", "unknown"),
            "volatility_context": market_context.get("volatility_context", "unknown"),
            "market_context": market_context,
            "position_assessments": position_assessments,
            "ranked_watchlist": ranked_watchlist,
            "open_alerts": open_alerts,
            "coverage": coverage,
            "degraded_mode": bool(coverage.get("degraded_mode", False)),
            "schema_version": 1,
            "adjust_plan_triggered": adjust_triggered,
            "adjust_plan_reasons": adjust_reasons,
            "scalp_watchlist": scalp_watchlist,
            "research_bundle": {
                "loaded": research_bundle is not None,
                "trade_date": research_bundle.trade_date if research_bundle else None,
                "top_pick_symbols": [
                    str(row.get("symbol", "")).upper()
                    for row in (research_bundle.top_picks if research_bundle else [])
                    if str(row.get("symbol", "")).strip()
                ],
            },
            "momentum_scanner": {
                "loaded": bool(momentum_state.get("loaded", False)),
                "stale": bool(momentum_state.get("stale", False)),
                "scan_count": _as_int(momentum_state.get("scan_count"), 0),
                "symbol_count": len(momentum_state.get("symbols", []) or []),
                "generated_at_et": str(momentum_state.get("generated_at_et", "") or ""),
                "artifact_path": str(momentum_state.get("artifact_path", "") or ""),
            },
            "overnight_gate": overnight_gate,
            "overnight_fallback_used": bool(overnight_gate.get("fallback_ready")),
            "fallback_reason": "",
        }

        if overnight_gate.get("fallback_ready"):
            handoff["degraded_mode"] = True
            handoff["fallback_reason"] = (
                f"overnight_incomplete:{overnight_gate.get('workflow_reason', 'unknown')}"
            )

        artifact = self._persist_handoff(handoff, timestamp)
        handoff["artifact_path"] = str(artifact)
        self.last_handoff = handoff
        return handoff

    def _filter_delisted_tickers(
        self, watchlist: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove delisted or non-tradable tickers from the watchlist."""
        if not self.premarket_analyzer or not hasattr(
            self.premarket_analyzer, "get_active_symbols"
        ):
            return watchlist

        # Use cache if available to avoid multiple API calls in the same manager lifetime
        if self._active_symbols_cache is None:
            self._active_symbols_cache = self.premarket_analyzer.get_active_symbols()

        if not self._active_symbols_cache:
            return watchlist

        filtered = []
        for item in watchlist:
            symbol = str(item.get("ticker") or item.get("symbol") or "").strip().upper()
            if not symbol or symbol in self._active_symbols_cache:
                filtered.append(item)
            else:
                logger.warning(
                    f"[PREMARKET] Filtering delisted or non-tradable ticker: {symbol}"
                )

        removed = len(watchlist) - len(filtered)
        if removed > 0:
            logger.info(
                f"[PREMARKET] Filtered {removed} delisted or non-tradable tickers"
            )

        return filtered

    def _merge_research_bundle_watchlist(
        self,
        watchlist: Sequence[Any],
        research_bundle: Optional[Any],
    ) -> List[Any]:
        merged: List[Any] = []
        seen: set[str] = set()

        def _append_rows(rows: Sequence[Any]) -> None:
            for item in rows or []:
                if isinstance(item, dict):
                    symbol = (
                        str(item.get("ticker") or item.get("symbol") or "")
                        .strip()
                        .upper()
                    )
                else:
                    symbol = str(item).strip().upper()
                if not symbol or symbol in seen:
                    continue
                row = dict(item) if isinstance(item, dict) else {"symbol": symbol}
                row["symbol"] = symbol
                merged.append(row)
                seen.add(symbol)

        if research_bundle is not None:
            _append_rows(research_bundle.top_picks)
            _append_rows(research_bundle.full_watchlist)
        _append_rows(watchlist)
        return merged

    def _load_live_momentum_watchlist(self, timestamp: datetime) -> Dict[str, Any]:
        cfg = self.momentum_scanner_cfg
        if cfg is None or not bool(getattr(cfg, "enabled", False)):
            return {"loaded": False, "symbols": []}
        state = load_momentum_watchlist(
            now_et=timestamp,
            path=getattr(cfg, "artifact_path", None),
        )
        self.momentum_scanner_state = state
        return state

    def _merge_live_momentum_watchlist(
        self,
        watchlist: Sequence[Any],
        momentum_symbols: Sequence[Any],
    ) -> List[Any]:
        merged: List[Any] = []
        seen: Dict[str, Dict[str, Any]] = {}

        def _upsert(item: Any, *, prefer_existing: bool) -> None:
            if isinstance(item, dict):
                symbol = (
                    str(item.get("ticker") or item.get("symbol") or "").strip().upper()
                )
                payload = dict(item)
            else:
                symbol = str(item).strip().upper()
                payload = {"symbol": symbol}
            if not symbol:
                return
            payload["symbol"] = symbol
            if symbol in seen:
                current = seen[symbol]
                for key, value in payload.items():
                    if key in {"symbol", "ticker"}:
                        current[key] = symbol
                    elif prefer_existing and current.get(key) not in (None, "", [], {}):
                        continue
                    elif value not in (None, "", [], {}):
                        current[key] = value
                return
            seen[symbol] = payload
            merged.append(seen[symbol])

        for item in watchlist or []:
            _upsert(item, prefer_existing=False)
        for item in momentum_symbols or []:
            momentum_row = (
                dict(item) if isinstance(item, dict) else {"symbol": str(item)}
            )
            momentum_row.setdefault("entry_source", "momentum_scanner")
            momentum_row.setdefault("source_bucket", "watchlist")
            momentum_row.setdefault("intraday_reserve", True)
            momentum_row.setdefault("reserved_slot_class", "momentum_scanner")
            momentum_row["momentum_scanner_active"] = True
            momentum_row["momentum_scanner_score"] = _as_float(
                momentum_row.get("score"), 0.0
            )
            momentum_row["momentum_scanner_catalyst"] = str(
                momentum_row.get("catalyst_summary")
                or momentum_row.get("catalyst_note")
                or ""
            )
            _upsert(momentum_row, prefer_existing=True)
        return merged

    def _rank_watchlist(self, watchlist: Sequence[Any]) -> List[Dict[str, Any]]:
        symbols = self._extract_symbols(watchlist)[: self.max_watchlist_symbols]

        # Build sector lookup from watchlist items (overnight research provides sector)
        sector_map: Dict[str, str] = {}
        metadata_map: Dict[str, Dict[str, Any]] = {}
        for item in watchlist or []:
            if isinstance(item, dict):
                sym = (
                    str(item.get("ticker") or item.get("symbol") or "").strip().upper()
                )
                sector = str(item.get("sector", "")).strip()
                if sym and sector:
                    sector_map[sym] = sector
                if sym:
                    metadata_map[sym] = {
                        "conviction_priority": _as_int(item.get("conviction_priority")),
                        "conviction_priority_score": _as_float(
                            item.get("conviction_priority_score")
                        ),
                        "target_allocation_pct": _as_float(
                            item.get("target_allocation_pct")
                        ),
                        "allocation_weight": _as_float(item.get("allocation_weight")),
                        "discovery_families": list(
                            item.get("discovery_families", []) or []
                        ),
                        "entry_source": str(item.get("entry_source", "") or ""),
                        "source_bucket": str(item.get("source_bucket", "") or ""),
                        "intraday_reserve": bool(item.get("intraday_reserve", False)),
                        "reserved_slot_class": str(
                            item.get("reserved_slot_class", "") or ""
                        ),
                        "momentum_scanner_active": bool(
                            item.get("momentum_scanner_active", False)
                            or str(item.get("entry_source", "") or "")
                            == "momentum_scanner"
                        ),
                        "momentum_scanner_score": _as_float(
                            item.get("momentum_scanner_score", item.get("score"))
                        ),
                        "momentum_scanner_catalyst": str(
                            item.get("momentum_scanner_catalyst")
                            or item.get("catalyst_summary")
                            or item.get("catalyst_note")
                            or ""
                        ),
                    }

        ranked: List[Dict[str, Any]] = []
        for symbol in symbols:
            row = self._evaluate_symbol(symbol)
            row["sector"] = sector_map.get(symbol, "")
            overnight = metadata_map.get(symbol, {})
            if overnight:
                row.update(
                    {
                        "conviction_priority": overnight.get("conviction_priority", 0),
                        "conviction_priority_score": round(
                            _as_float(overnight.get("conviction_priority_score")), 2
                        ),
                        "target_allocation_pct": round(
                            _as_float(overnight.get("target_allocation_pct")), 2
                        ),
                        "allocation_weight": round(
                            _as_float(overnight.get("allocation_weight")), 4
                        ),
                        "discovery_families": overnight.get("discovery_families", []),
                    }
                )

                priority_rank = max(0, _as_int(overnight.get("conviction_priority")))
                priority_score = _as_float(overnight.get("conviction_priority_score"))
                overnight_boost = _clamp((priority_score - 50.0) * 0.2, 0.0, 12.0)
                if priority_rank > 0:
                    overnight_boost += _clamp((6 - priority_rank) * 1.5, 0.0, 6.0)
                    row["rationale"].insert(
                        0, f"Overnight conviction priority #{priority_rank}"
                    )
                live_score = _as_float(row.get("score"))
                row["score"] = _clamp(live_score + overnight_boost, 0.0, 100.0)
                priority_floor_ratio = 0.0
                if priority_rank == 1:
                    priority_floor_ratio = 1.0
                elif priority_rank <= 3:
                    priority_floor_ratio = 0.85
                elif priority_rank <= 5:
                    priority_floor_ratio = 0.7
                if priority_floor_ratio > 0.0:
                    priority_floor = _clamp(
                        priority_score * priority_floor_ratio, 0.0, 100.0
                    )
                    if priority_floor > _as_float(row.get("score")):
                        row["score"] = priority_floor
                        row["overnight_priority_floor"] = round(priority_floor, 2)
                row["overnight_priority_boost"] = round(overnight_boost, 2)
                if overnight.get("momentum_scanner_active"):
                    scanner_score = _as_float(
                        overnight.get("momentum_scanner_score"), 0.0
                    )
                    scanner_boost = _clamp((scanner_score - 55.0) * 0.12, 0.0, 8.0)
                    row["score"] = _clamp(
                        _as_float(row.get("score")) + scanner_boost, 0.0, 100.0
                    )
                    row["entry_source"] = "momentum_scanner"
                    row["source_bucket"] = "watchlist"
                    row["intraday_reserve"] = bool(
                        overnight.get("intraday_reserve", True)
                    )
                    row["reserved_slot_class"] = str(
                        overnight.get("reserved_slot_class") or "momentum_scanner"
                    )
                    row["momentum_scanner_score"] = round(scanner_score, 2)
                    row["rationale"].insert(0, "Live momentum scanner qualified")
                    catalyst_note = str(
                        overnight.get("momentum_scanner_catalyst") or ""
                    ).strip()
                    if catalyst_note:
                        row["rationale"].insert(
                            1, f"Scanner catalyst: {catalyst_note[:50]}"
                        )
            ranked.append(row)

        # Apply YouTube sector bias after base scoring
        yt = getattr(self, "_cached_youtube_intel", {}) or {}
        if yt.get("available") and (yt.get("avoid_sectors") or yt.get("favor_sectors")):
            avoid = [s.lower() for s in (yt.get("avoid_sectors") or [])]
            favor = [s.lower() for s in (yt.get("favor_sectors") or [])]
            boosted = 0
            penalized = 0
            for row in ranked:
                sector_lower = row.get("sector", "").lower()
                if not sector_lower:
                    continue
                if any(a in sector_lower for a in avoid if a):
                    row["score"] = max(0.0, row["score"] - 8.0)
                    row["rationale"].append(f"YouTube: avoid {sector_lower}")
                    penalized += 1
                elif any(f in sector_lower for f in favor if f):
                    row["score"] = min(100.0, row["score"] + 5.0)
                    row["rationale"].append(f"YouTube: favor {sector_lower}")
                    boosted += 1
            if boosted or penalized:
                logger.info(
                    f"[PREMARKET][YOUTUBE] Watchlist bias: {boosted} boosted, {penalized} penalized"
                )

        ranked.sort(key=lambda row: row.get("score", 0.0), reverse=True)
        for i, row in enumerate(ranked, start=1):
            row["rank"] = i
        return ranked

    def _is_rvol_window(self, ts: datetime) -> bool:
        """True during 6:30-8:30 ET pre-open monitoring window."""
        if ts is None or ts.tzinfo is None:
            return False
        t = ts.timetz()
        start = time(6, 30, tzinfo=ET)
        end = time(8, 30, tzinfo=ET)
        return start <= t < end

    def _apply_rvol_momentum(
        self, ranked_watchlist: Sequence[Dict[str, Any]], ts: datetime
    ) -> List[Dict[str, Any]]:
        """Annotate and adjust scores for high RVOL + momentum in early window."""
        is_window = self._is_rvol_window(ts)
        updated: List[Dict[str, Any]] = []
        for row in ranked_watchlist:
            vol_ratio = _as_float(row.get("volume_ratio"))
            pm_volume = _as_int(row.get("premarket_volume"))
            rvol_flag = vol_ratio >= 2.0 and pm_volume >= 150_000

            vwap_state = str(row.get("vwap_state", STATE_WAIT))
            trend = str(row.get("premarket_trend", "flat")).lower()
            if vwap_state == STATE_STRONG_ABOVE or trend == "bullish":
                momentum = "bullish"
            elif vwap_state == STATE_STRONG_BELOW or trend == "bearish":
                momentum = "bearish"
            else:
                momentum = "mixed"

            adjustment = 0.0
            rationale = list(row.get("rationale", []))
            if is_window and rvol_flag:
                if momentum == "bullish":
                    adjustment = 6.0
                    rationale.insert(
                        0, "RVOL window boost: high volume with bullish momentum"
                    )
                elif momentum == "bearish":
                    adjustment = -4.0
                    rationale.insert(
                        0, "RVOL window caution: high volume with bearish momentum"
                    )
                else:
                    adjustment = 2.0
                    rationale.insert(
                        0, "RVOL window noted: high volume, mixed momentum"
                    )

            row["high_rvol"] = rvol_flag
            row["rvol_multiple"] = round(vol_ratio, 3)
            row["rvol_window"] = is_window
            row["momentum_signal"] = momentum
            if adjustment != 0.0:
                row["score"] = _clamp(_as_float(row.get("score")), 0.0, 100.0)
                row["score"] = _clamp(row["score"] + adjustment, 0.0, 100.0)
                row["rvol_adjustment"] = adjustment
            row["rationale"] = rationale[:6]
            updated.append(row)

        # Re-rank after adjustments
        updated.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        for i, row in enumerate(updated, start=1):
            row["rank"] = i
        return updated

    def _compute_dynamic_adjust(
        self, ranked_watchlist: Sequence[Dict[str, Any]], ts: datetime
    ) -> Tuple[bool, List[str], List[Dict[str, Any]]]:
        """Decide if adjust_plan should fire based on RVOL/catalyst deltas; derive scalp list."""
        reasons: List[str] = []
        if not ranked_watchlist:
            return False, reasons, []

        snapshot: Dict[str, Dict[str, Any]] = {}
        for row in ranked_watchlist:
            sym = row.get("symbol")
            if not sym:
                continue
            snapshot[sym] = {
                "rvol": _as_float(row.get("volume_ratio")),
                "high_rvol": bool(row.get("high_rvol")),
                "has_catalyst": bool(row.get("has_catalyst")),
                "momentum": str(row.get("momentum_signal", "")),
            }

        triggered = False
        if not self._last_adjust_snapshot:
            triggered = True
            reasons.append("Initial premarket scan")
        else:
            for sym, curr in snapshot.items():
                prev = self._last_adjust_snapshot.get(sym, {})
                if curr["high_rvol"] and not prev.get("high_rvol", False):
                    triggered = True
                    reasons.append(f"RVOL delta: {sym} became high-RVOL")
                if curr["has_catalyst"] and not prev.get("has_catalyst", False):
                    triggered = True
                    reasons.append(f"New catalyst detected for {sym}")
                if abs(curr["rvol"] - _as_float(prev.get("rvol"))) >= 0.5:
                    triggered = True
                    reasons.append(f"RVOL delta: {sym} changed by >=0.5x")

        self._last_adjust_snapshot = snapshot

        scalp_watchlist: List[Dict[str, Any]] = []
        is_window = self._is_rvol_window(ts)
        if is_window:
            for row in ranked_watchlist:
                if (
                    row.get("high_rvol")
                    and row.get("has_catalyst")
                    and _as_float(row.get("score")) >= 60.0
                ):
                    scalp_watchlist.append(
                        {
                            "symbol": row.get("symbol"),
                            "score": row.get("score"),
                            "rvol_multiple": row.get("rvol_multiple"),
                            "momentum": row.get("momentum_signal"),
                            "reason": "; ".join(row.get("rationale", [])[:3]),
                        }
                    )
            scalp_watchlist = sorted(
                scalp_watchlist, key=lambda r: _as_float(r.get("score")), reverse=True
            )[:8]

        return triggered, reasons, scalp_watchlist

    def _evaluate_symbol(self, symbol: str) -> Dict[str, Any]:
        pm = self._safe_analyze(symbol)
        news = self._safe_news(symbol)
        stocktwits = self._safe_stocktwits(symbol)
        vwap = self._compute_vwap_state(symbol)
        sr = self._estimate_sr_context(symbol)

        has_pm_data = bool(pm.get("has_data", False))
        gap_pct = _as_float(pm.get("gap_pct"))
        volume_ratio = _as_float(pm.get("volume_ratio"))
        trend = str(pm.get("premarket_trend", "flat")).lower()
        liquidity_score = _as_float(pm.get("liquidity_score"), 50.0)

        score = 50.0
        rationale: List[str] = []

        # Gap behavior
        gap_score = _clamp(gap_pct * 3.0, -18.0, 18.0)
        score += gap_score * self.weights["gap"]
        if has_pm_data:
            rationale.append(f"Gap {gap_pct:+.2f}%")
        else:
            score -= 8.0
            rationale.append("Premarket quote unavailable")

        # Volume context
        vol_score = _clamp((volume_ratio - 1.0) * 10.0, -12.0, 20.0)
        score += vol_score * self.weights["volume"]
        if volume_ratio > 0:
            rationale.append(f"Volume ratio {volume_ratio:.2f}x")

        # Liquidity guardrail
        if has_pm_data:
            if liquidity_score < 40:
                score -= 10.0
                rationale.append(f"Low liquidity (score {liquidity_score:.0f})")
            elif liquidity_score > 80:
                score += 5.0
                rationale.append("High premarket liquidity")

        # Trend
        if trend == "bullish":
            score += 8.0
            rationale.append("Premarket trend bullish")
        elif trend == "bearish":
            score -= 8.0
            rationale.append("Premarket trend bearish")

        # VWAP state
        vwap_state = str(vwap.get("state", STATE_WAIT))
        if vwap_state == STATE_STRONG_ABOVE:
            score += 10.0 * self.weights["vwap"]
            rationale.append("Price holding above premarket VWAP")
        elif vwap_state == STATE_STRONG_BELOW:
            score -= 10.0 * self.weights["vwap"]
            rationale.append("Price weak below premarket VWAP")
        elif vwap_state == STATE_MIXED:
            rationale.append("VWAP relationship mixed")
        else:
            score -= 2.0
            rationale.append("VWAP state waiting for signal")

        # Local technical S/R context (intraday + premarket bars fallback).
        sr_adjustment = _as_float(sr.get("score_adjustment"))
        score += sr_adjustment * self.weights["sr"]
        for item in sr.get("rationale", []):
            rationale.append(str(item))

        # News + social sentiment
        news_score = _as_float(news.get("sentiment_score"))
        stocktwits_score = _as_float(stocktwits.get("sentiment_score"))
        catalyst_score = _as_float(news.get("catalyst_score"))
        score += news_score * 10.0 * self.weights["news"]
        score += stocktwits_score * 8.0 * self.weights["stocktwits"]
        score += (catalyst_score / 10.0) * self.weights["news"]
        if news.get("available"):
            rationale.append(f"News sentiment {news_score:+.2f}")
        if news.get("has_catalyst"):
            rationale.append(
                f"Catalyst: {str(news.get('catalyst_note') or '').strip()[:50]}"
            )
        if stocktwits.get("available"):
            rationale.append(f"Stocktwits sentiment {stocktwits_score:+.2f}")

        score = _clamp(score, 0.0, 100.0)
        confidence = _clamp(
            (
                _as_float(news.get("confidence"))
                + _as_float(stocktwits.get("confidence"))
                + (0.5 if has_pm_data else 0.0)
            )
            / 3.0,
            0.0,
            1.0,
        )

        coverage = self._merge_coverage(
            [
                "full" if has_pm_data else "none",
                str(news.get("coverage", "none")),
                str(stocktwits.get("coverage", "none")),
                str(sr.get("coverage", "none")),
            ]
        )

        return {
            "symbol": symbol,
            "score": round(score, 2),
            "prev_close": _as_float(pm.get("prev_close")),
            "premarket_gap_pct": gap_pct if has_pm_data else None,
            "premarket_volume": _as_int(pm.get("premarket_volume")),
            "volume_ratio": round(volume_ratio, 3),
            "liquidity_score": round(liquidity_score, 2),
            "premarket_trend": trend,
            "vwap_state": vwap_state,
            "vwap_distance_pct": round(_as_float(vwap.get("distance_pct")), 3),
            "news": news,
            "has_catalyst": bool(news.get("has_catalyst")),
            "catalyst_score": round(catalyst_score, 2),
            "catalyst_tags": list(news.get("catalyst_tags", []) or []),
            "catalyst_note": str(news.get("catalyst_note", "") or ""),
            "stocktwits": stocktwits,
            "s1_price": round(_as_float(sr.get("s1_price")), 4),
            "s1_strength": round(_as_float(sr.get("s1_strength")), 2),
            "r1_price": round(_as_float(sr.get("r1_price")), 4),
            "r1_strength": round(_as_float(sr.get("r1_strength")), 2),
            "support_dist_atr": round(_as_float(sr.get("support_dist_atr")), 3),
            "resistance_dist_atr": round(_as_float(sr.get("resistance_dist_atr")), 3),
            "sr_quality_score": round(_as_float(sr.get("sr_quality_score")), 2),
            "coverage": coverage,
            "confidence": round(confidence, 3),
            "rationale": rationale[:6],
        }

    def _assess_positions(self, holdings: Sequence[str]) -> List[Dict[str, Any]]:
        assessments: List[Dict[str, Any]] = []
        for symbol in holdings:
            pm = self._safe_analyze(symbol)
            vwap = self._compute_vwap_state(symbol)
            has_pm_data = bool(pm.get("has_data", False))
            gap_pct = _as_float(pm.get("gap_pct"))
            trend = str(pm.get("premarket_trend", "flat")).lower()
            vwap_state = str(vwap.get("state", STATE_WAIT))

            status = "WATCH"
            reason = "Insufficient premarket data"
            if has_pm_data:
                if (
                    gap_pct <= -4.0
                    or (gap_pct < -2.0 and trend == "bearish")
                    or vwap_state == STATE_STRONG_BELOW
                ):
                    status = "CONCERN"
                    reason = f"Weak premarket: gap {gap_pct:+.2f}%, trend={trend}, vwap={vwap_state}"
                elif (
                    gap_pct >= 1.5
                    and trend == "bullish"
                    and vwap_state in {STATE_STRONG_ABOVE, STATE_MIXED}
                ):
                    status = "HOLD"
                    reason = f"Constructive premarket: gap {gap_pct:+.2f}%, trend={trend}, vwap={vwap_state}"
                else:
                    status = "WATCH"
                    reason = f"Mixed premarket: gap {gap_pct:+.2f}%, trend={trend}, vwap={vwap_state}"

            assessments.append(
                {
                    "symbol": symbol,
                    "status": status,
                    "reason": reason,
                    "premarket_gap_pct": gap_pct if has_pm_data else None,
                    "premarket_trend": trend,
                    "vwap_state": vwap_state,
                }
            )
        return assessments

    def _build_open_alerts(
        self,
        ranked_watchlist: Sequence[Dict[str, Any]],
        position_assessments: Sequence[Dict[str, Any]],
        market_context: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        alerts: List[Dict[str, Any]] = []

        for row in ranked_watchlist[:8]:
            symbol = row.get("symbol")
            if not symbol:
                continue
            score = _as_float(row.get("score"))
            state = str(row.get("vwap_state", STATE_WAIT))
            gap = row.get("premarket_gap_pct")
            if score >= 72 and state == STATE_STRONG_ABOVE:
                alerts.append(
                    {
                        "symbol": symbol,
                        "severity": "high",
                        "type": "OPEN_ENTRY_FAVORABLE",
                        "message": f"{symbol} score={score:.1f} with strong VWAP hold",
                    }
                )
            elif state == STATE_STRONG_BELOW or (
                isinstance(gap, (int, float)) and float(gap) <= -4.0
            ):
                alerts.append(
                    {
                        "symbol": symbol,
                        "severity": "medium",
                        "type": "OPEN_ENTRY_CAUTION",
                        "message": f"{symbol} showing weak open setup (gap={gap}, vwap={state})",
                    }
                )

        for row in position_assessments:
            if row.get("status") == "CONCERN":
                alerts.append(
                    {
                        "symbol": row.get("symbol"),
                        "severity": "high",
                        "type": "POSITION_CONCERN",
                        "message": str(row.get("reason", "Premarket concern")),
                    }
                )

        bias = str(market_context.get("futures_direction", "unknown"))
        vol = str(market_context.get("volatility_context", "unknown"))
        alerts.append(
            {
                "symbol": "MARKET",
                "severity": "info",
                "type": "MARKET_CONTEXT",
                "message": f"Market bias={bias}, volatility={vol}",
            }
        )
        return alerts

    def _collect_market_context(self) -> Dict[str, Any]:
        spy = self._safe_analyze("SPY")
        qqq = self._safe_analyze("QQQ")

        spy_gap = _as_float(spy.get("gap_pct"))
        qqq_gap = _as_float(qqq.get("gap_pct"))
        proxy_values = [v for v in (spy_gap, qqq_gap) if abs(v) > 0]
        proxy_avg = sum(proxy_values) / len(proxy_values) if proxy_values else 0.0

        if proxy_values:
            if proxy_avg >= 0.4:
                futures_direction = "bullish"
            elif proxy_avg <= -0.4:
                futures_direction = "bearish"
            else:
                futures_direction = "neutral"
        else:
            futures_direction = "unknown"

        vix_level = self._safe_vix_level()
        if vix_level is None:
            volatility = "unknown"
        elif vix_level >= 28:
            volatility = "high"
        elif vix_level >= 20:
            volatility = "elevated"
        else:
            volatility = "normal"

        sector_cues = self._sector_cues()

        # Load YouTube market intelligence for regime + sector context
        youtube_intel = self._load_youtube_market_context()
        self._cached_youtube_intel = youtube_intel  # Cache for _evaluate_symbol

        return {
            "futures_direction": futures_direction,
            "futures_proxy": {
                "spy_gap_pct": round(spy_gap, 3) if proxy_values else None,
                "qqq_gap_pct": round(qqq_gap, 3) if proxy_values else None,
            },
            "vix_level": round(vix_level, 3)
            if isinstance(vix_level, (int, float))
            else None,
            "volatility_context": volatility,
            "sector_cues": sector_cues,
            "youtube_intelligence": youtube_intel,
        }

    def _overnight_gate_status(self, timestamp: datetime) -> Dict[str, Any]:
        """Return overnight completion status plus a premarket fallback policy."""
        state_path = PROJECT_DIR / "research" / "overnight_state.json"
        freshness = check_research_freshness(
            state_path=state_path,
            now_et=timestamp,
            persist_metadata=False,
        )
        workflow_complete = bool(freshness.get("workflow_complete"))
        fallback_deadline = time(hour=8, minute=15)
        should_wait = (not workflow_complete) and timestamp.time() < fallback_deadline
        fallback_ready = (not workflow_complete) and not should_wait
        return {
            "workflow_complete": workflow_complete,
            "workflow_reason": str(freshness.get("workflow_reason", "unknown")),
            "age_hours": freshness.get("age_hours"),
            "max_age_hours": freshness.get("max_age_hours"),
            "is_fresh": bool(freshness.get("is_fresh")),
            "should_wait": should_wait,
            "fallback_ready": fallback_ready,
            "fallback_deadline_et": fallback_deadline.strftime("%H:%M"),
            "retry_after_seconds": 300 if should_wait else 0,
            "freshness": freshness,
        }

    def _safe_analyze(self, symbol: str) -> Dict[str, Any]:
        if self.premarket_analyzer is None:
            return {"symbol": symbol, "has_data": False}
        try:
            pm = self.premarket_analyzer.analyze_ticker(symbol)
            return {
                "symbol": symbol,
                "has_data": bool(getattr(pm, "has_data", False)),
                "prev_close": _as_float(getattr(pm, "prev_close", 0.0)),
                "gap_pct": _as_float(getattr(pm, "gap_pct", 0.0)),
                "volume_ratio": _as_float(getattr(pm, "volume_ratio", 0.0)),
                "liquidity_score": _as_float(getattr(pm, "liquidity_score", 0.0)),
                "premarket_volume": _as_int(getattr(pm, "premarket_volume", 0)),
                "premarket_trend": str(getattr(pm, "premarket_trend", "flat")),
            }
        except Exception as exc:
            logger.debug("Premarket analysis failed for %s: %s", symbol, exc)
            return {"symbol": symbol, "has_data": False}

    def _safe_news(self, symbol: str) -> Dict[str, Any]:
        try:
            return self.news_aggregator.collect(symbol)
        except Exception as exc:
            logger.debug("News aggregation failed for %s: %s", symbol, exc)
            return _DisabledNewsAggregator().collect(symbol)

    def _safe_stocktwits(self, symbol: str) -> Dict[str, Any]:
        try:
            return self.stocktwits_scraper.fetch(symbol)
        except Exception as exc:
            logger.debug("Stocktwits scrape failed for %s: %s", symbol, exc)
            return _DisabledStocktwitsScraper().fetch(symbol)

    def _compute_vwap_state(self, symbol: str) -> Dict[str, Any]:
        if self.premarket_analyzer is None:
            return {"state": STATE_WAIT, "distance_pct": 0.0}
        try:
            bars = self.premarket_analyzer.get_premarket_bars(symbol)
        except Exception:
            bars = None

        records = self._bar_records(bars)
        if not records:
            return {"state": STATE_WAIT, "distance_pct": 0.0}

        tracker = PremarketVWAPTracker()
        snapshot = tracker.update_many(records)
        return {
            "state": snapshot.state,
            "distance_pct": snapshot.distance_pct,
            "vwap": snapshot.vwap,
            "last_price": snapshot.last_price,
            "bars_seen": snapshot.bars_seen,
        }

    def _estimate_sr_context(self, symbol: str) -> Dict[str, Any]:
        base = {
            "s1_price": 0.0,
            "s1_strength": 0.0,
            "r1_price": 0.0,
            "r1_strength": 0.0,
            "support_dist_atr": 0.0,
            "resistance_dist_atr": 0.0,
            "sr_quality_score": 0.0,
            "score_adjustment": 0.0,
            "coverage": "none",
            "rationale": [],
        }
        bars_df = None

        # Primary source: intraday bars via Alpaca/yfinance provider (when client is available).
        data_client = getattr(self.premarket_analyzer, "data_api", None)
        if (
            INTRADAY_PROVIDER_AVAILABLE
            and get_intraday_bars is not None
            and data_client is not None
        ):
            try:
                bars_df = get_intraday_bars(
                    symbol,
                    data_client,
                    minutes_back=780,
                    interval="5m",
                )
            except Exception:
                bars_df = None

        # Fallback: use available premarket bars if they have enough structure.
        if (
            (bars_df is None or len(bars_df) < 20)
            and pd is not None
            and self.premarket_analyzer is not None
        ):
            try:
                pm_bars = self.premarket_analyzer.get_premarket_bars(symbol)
                records = self._bar_records(pm_bars)
                if len(records) >= 10:
                    bars_df = pd.DataFrame.from_records(records)
            except Exception:
                bars_df = None

        if bars_df is None or len(bars_df) < 10:
            return base

        try:
            sr = estimate_sr_levels(
                bars_df,
                lookback_bars=min(160, len(bars_df)),
                pivot_window=2,
                cluster_atr_mult=0.45,
            )
        except Exception:
            sr = None
        if not sr:
            return base

        out = dict(base)
        out.update(
            {
                "s1_price": _as_float(sr.get("s1_price")),
                "s1_strength": _as_float(sr.get("s1_strength")),
                "r1_price": _as_float(sr.get("r1_price")),
                "r1_strength": _as_float(sr.get("r1_strength")),
                "support_dist_atr": _as_float(sr.get("support_dist_atr")),
                "resistance_dist_atr": _as_float(sr.get("resistance_dist_atr")),
                "sr_quality_score": _as_float(sr.get("sr_quality_score")),
                "coverage": "partial",
            }
        )

        support_dist = out["support_dist_atr"]
        resistance_dist = out["resistance_dist_atr"]
        quality = out["sr_quality_score"]

        adjustment = 0.0
        rationale: List[str] = []
        if 0 < support_dist <= 0.6:
            adjustment += 8.0
            rationale.append("Price near technical support")
        elif 0 < support_dist <= 1.0:
            adjustment += 4.0
            rationale.append("Price within 1 ATR of support")

        if 0 < resistance_dist <= 0.8:
            adjustment -= 8.0
            rationale.append("Tight overhead resistance")
        elif 0 < resistance_dist <= 1.2:
            adjustment -= 4.0
            rationale.append("Moderate overhead resistance")
        elif resistance_dist >= 2.0:
            adjustment += 3.0
            rationale.append("Room to resistance")

        if quality >= 65.0:
            adjustment += 4.0
            rationale.append("High-quality S/R structure")
        elif quality < 40.0:
            adjustment -= 2.0

        out["score_adjustment"] = _clamp(adjustment, -12.0, 12.0)
        out["rationale"] = rationale[:2]
        return out

    def _bar_records(self, bars: Any) -> List[Dict[str, Any]]:
        if bars is None:
            return []
        if isinstance(bars, list):
            return [b for b in bars if isinstance(b, dict)]
        if hasattr(bars, "empty") and bool(getattr(bars, "empty")):
            return []
        if hasattr(bars, "to_dict"):
            try:
                recs = bars.to_dict("records")
                return [r for r in recs if isinstance(r, dict)]
            except Exception:
                return []
        return []

    def _load_youtube_market_context(self) -> Dict[str, Any]:
        """Load YouTube intelligence for premarket market context."""
        try:
            from autotrade.utils.youtube_readiness import get_intelligence_context

            ctx = get_intelligence_context()
            if ctx.get("available"):
                logger.info(
                    f"[PREMARKET][YOUTUBE] regime={ctx['regime']}, "
                    f"sizing={ctx['sizing_multiplier']}x, "
                    f"avoid={ctx.get('avoid_sectors', [])}, "
                    f"favor={ctx.get('favor_sectors', [])}"
                )
                return {
                    "available": True,
                    "regime": ctx.get("regime", "NEUTRAL"),
                    "regime_confidence": ctx.get("regime_confidence", 0),
                    "sizing_multiplier": ctx.get("sizing_multiplier", 1.0),
                    "avoid_sectors": ctx.get("avoid_sectors", []),
                    "favor_sectors": ctx.get("favor_sectors", []),
                    "executive_summary": str(ctx.get("executive_summary", ""))[:300],
                    "directives": (ctx.get("directives") or [])[:3],
                    "trigger_levels": ctx.get("trigger_levels", {}),
                    "report_date": ctx.get("report_date", "none"),
                }
            return {"available": False}
        except Exception as e:
            logger.debug(f"[PREMARKET][YOUTUBE] Could not load: {e}")
            return {"available": False}

    def _sector_cues(self) -> Dict[str, Any]:
        sectors = ["XLK", "XLF", "XLE", "XLV", "XLI"]
        rows: List[Tuple[str, float]] = []
        for sym in sectors:
            pm = self._safe_analyze(sym)
            if pm.get("has_data"):
                rows.append((sym, _as_float(pm.get("gap_pct"))))
        if not rows:
            return {"leaders": [], "laggards": []}

        rows.sort(key=lambda x: x[1], reverse=True)
        leaders = [f"{sym}:{gap:+.2f}%" for sym, gap in rows[:2]]
        laggards = [f"{sym}:{gap:+.2f}%" for sym, gap in rows[-2:]]
        return {"leaders": leaders, "laggards": laggards}

    def _safe_vix_level(self) -> Optional[float]:
        try:
            import yfinance as yf

            hist = yf.Ticker("^VIX").history(period="2d", interval="1d")
            if hist is None or hist.empty:
                return None
            return float(hist["Close"].iloc[-1])
        except Exception:
            return None

    def _summarize_coverage(
        self,
        ranked_watchlist: Sequence[Dict[str, Any]],
        position_assessments: Sequence[Dict[str, Any]],
        market_context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        market_ok = market_context.get("futures_direction") != "unknown"
        item_coverages = [str(r.get("coverage", "none")) for r in ranked_watchlist]
        if not item_coverages:
            watchlist_cov = "none"
        else:
            watchlist_cov = self._merge_coverage(item_coverages)

        degraded = (not market_ok) or watchlist_cov == "none"
        return {
            "market_context": "full" if market_ok else "partial",
            "watchlist": watchlist_cov,
            "positions_checked": len(position_assessments),
            "degraded_mode": degraded,
        }

    def _persist_handoff(self, handoff: Dict[str, Any], now_et: datetime) -> Path:
        plan_date = get_pm_plan_date(now_et)
        handoff = dict(handoff)
        handoff.setdefault("trade_date", plan_date.isoformat())
        handoff.setdefault("report_date", plan_date.isoformat())
        handoff.setdefault("generated_at_et", now_et.isoformat())
        stamped = (
            self.output_dir
            / f"morning_intelligence_{plan_date.strftime('%Y%m%d')}_{now_et.strftime('%H%M')}.json"
        )
        latest = self.output_dir / "morning_intelligence_latest.json"
        for path in (stamped, latest):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(handoff, f, indent=2, default=str)

        try:
            from autotrade.utils.phase_sync import (
                normalize_watchlist,
                record_decisions,
                update_phase_snapshot,
            )

            ranked_norm = normalize_watchlist(
                handoff.get("ranked_watchlist", []),
                source="premarket",
                phase=handoff.get("phase", ""),
            )
            scalp_norm = normalize_watchlist(
                handoff.get("scalp_watchlist", []),
                source="premarket_scalp",
                phase=handoff.get("phase", ""),
            )

            update_phase_snapshot(
                plan_date,
                plans_dir=self.output_dir,
                premarket={
                    "handoff_path": str(latest),
                    "ranked_watchlist": ranked_norm,
                    "scalp_watchlist": scalp_norm,
                    "adjust_plan_reasons": handoff.get("adjust_plan_reasons", []),
                },
            )

            record_decisions(
                [
                    {
                        "symbol": item.get("symbol"),
                        "action": "ranked_watchlist",
                        "score": item.get("score"),
                        "has_catalyst": item.get("has_catalyst"),
                        "source": item.get("source"),
                    }
                    for item in ranked_norm[:40]
                ],
                phase="premarket",
                day=plan_date,
            )

            if scalp_norm:
                record_decisions(
                    [
                        {
                            "symbol": item.get("symbol"),
                            "action": "scalp_watchlist",
                            "score": item.get("score"),
                            "source": item.get("source"),
                            "reason": "; ".join(
                                handoff.get("adjust_plan_reasons") or []
                            ).strip()
                            or "dynamic_adjust_trigger",
                        }
                        for item in scalp_norm
                    ],
                    phase="premarket",
                    day=plan_date,
                )
        except Exception as sync_exc:
            logger.debug(f"[SYNC] Premarket phase snapshot skipped: {sync_exc}")
        return stamped

    def _extract_symbols(self, watchlist: Sequence[Any]) -> List[str]:
        out: List[str] = []
        seen = set()
        for item in watchlist or []:
            symbol = ""
            if isinstance(item, dict):
                symbol = str(item.get("ticker") or item.get("symbol") or "").strip()
            else:
                symbol = str(item).strip()
            symbol = symbol.upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            out.append(symbol)
        return out

    def _to_et(self, dt: Optional[datetime]) -> datetime:
        if dt is None:
            if callable(self.now_provider):
                dt = self.now_provider()
            else:
                dt = datetime.now(ET)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=ET)
        return dt.astimezone(ET)

    @staticmethod
    def _merge_coverage(values: Sequence[str]) -> str:
        cleaned = [str(v).lower() for v in values if str(v).strip()]
        if not cleaned:
            return "none"
        if all(v == "full" for v in cleaned):
            return "full"
        if any(v in {"full", "partial"} for v in cleaned):
            return "partial"
        return "none"

    def _build_analyzer(self) -> Optional[Any]:
        try:
            return PreMarketAnalyzer(api_key=self.api_key, api_secret=self.api_secret)
        except Exception as exc:
            logger.debug("PreMarketAnalyzer unavailable: %s", exc)
            return None

    def _load_config_weights(self) -> None:
        try:
            from config.config_loader import get_config

            all_cfg = get_config()
            cfg = all_cfg.premarket_manager
            self.momentum_scanner_cfg = getattr(all_cfg, "momentum_scanner", None)
            self.weights["gap"] = _as_float(getattr(cfg, "weight_gap", 1.0), 1.0)
            self.weights["volume"] = _as_float(getattr(cfg, "weight_volume", 1.0), 1.0)
            self.weights["news"] = _as_float(getattr(cfg, "weight_news", 1.0), 1.0)
            self.weights["stocktwits"] = _as_float(
                getattr(cfg, "weight_stocktwits", 0.75), 0.75
            )
            self.weights["vwap"] = _as_float(getattr(cfg, "weight_vwap", 1.0), 1.0)
            self.weights["sr"] = _as_float(getattr(cfg, "weight_sr", 0.8), 0.8)
            self.max_watchlist_symbols = int(
                max(
                    1, getattr(cfg, "max_watchlist_symbols", self.max_watchlist_symbols)
                )
            )
            if not bool(getattr(cfg, "use_news", True)):
                self.news_aggregator = _DisabledNewsAggregator()
            if not bool(getattr(cfg, "use_stocktwits", True)):
                self.stocktwits_scraper = _DisabledStocktwitsScraper()
        except Exception:
            return


def load_latest_morning_intelligence(
    output_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Load the latest saved morning intelligence handoff if present."""
    base = Path(output_dir or (PROJECT_DIR / "plans"))
    latest = base / "morning_intelligence_latest.json"
    if not latest.exists():
        return None
    try:
        with open(latest, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
