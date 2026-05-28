"""
Pre-Market Agent
================
Early morning agent (5:30 AM - 9:30 AM) that:
1. Drops candidates if pre-market shows weakness
2. Identifies strong pre-market opportunities (rare, but considered)
3. Updates watchlist priorities based on PM action
4. Can initiate pre-market orders if opportunity is exceptional
5. Checks for same-day earnings on held positions
6. Validates candidates against financial DB health checks

Integrates with day_manager for morning handoff.
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = Path(os.environ.get("AUTOTRADE_ROOT", Path(__file__).resolve().parents[2]))
LOG_DIR = PROJECT_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# Load environment
env_path = PROJECT_DIR / '.env'
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            key, val = line.split('=', 1)
            os.environ[key.strip()] = val.strip()

logger = logging.getLogger('premarket_agent')

# Import financial DB checks
from autotrade.analysis.financial_checks import (
    flag_earnings_today,
    check_balance_sheet_health,
    check_cash_flow_health,
    check_valuation_sanity,
    check_options_positioning,
)

@dataclass
class PreMarketDecision:
    """Decision made by pre-market agent."""
    ticker: str
    action: str  # 'drop', 'watch', 'consider_entry', 'premarket_buy'
    reason: str
    gap_pct: float = 0.0
    pm_trend: str = "flat"
    pm_volume_ratio: float = 0.0
    liquidity_score: float = 0.0
    priority_change: int = 0  # -2 to +2
    
    # For pre-market orders
    order_price: float = 0.0
    order_qty: int = 0
    order_type: str = ""  # 'limit', 'market'


@dataclass
class PreMarketPlan:
    """Complete pre-market plan for the day."""
    timestamp: str
    drops: List[PreMarketDecision] = field(default_factory=list)
    watches: List[PreMarketDecision] = field(default_factory=list)
    strong_candidates: List[PreMarketDecision] = field(default_factory=list)
    premarket_orders: List[PreMarketDecision] = field(default_factory=list)
    earnings_alerts: List[str] = field(default_factory=list)  # Holdings with same-day earnings
    
    # Summary
    total_analyzed: int = 0
    total_dropped: int = 0
    total_upgraded: int = 0


class PreMarketAgent:
    """
    Pre-market analysis agent.
    
    Runs early morning (5:30 AM - 9:30 AM ET) to:
    1. Filter out weak candidates based on PM action
    2. Identify exceptional PM opportunities
    3. Update watchlist priorities
    4. Optionally place pre-market orders
    
    Pre-market order criteria (VERY selective):
    - Gap up 3%+ with strong PM trend
    - Above-average PM volume (2x+)
    - High liquidity score (70+)
    - Strong fundamental catalyst
    """
    
    # Thresholds
    DROP_GAP_THRESHOLD = -2.0      # Drop if gapping down more than 2%
    DROP_TREND_THRESHOLD = -1.0    # Drop if PM trend down 1%+ from PM open
    UPGRADE_GAP_THRESHOLD = 2.0    # Upgrade priority if gap 2%+
    PM_ORDER_GAP_THRESHOLD = 3.0   # Consider PM order if gap 3%+
    PM_ORDER_VOLUME_THRESHOLD = 2.0  # Require 2x normal PM volume
    PM_ORDER_LIQUIDITY_THRESHOLD = 70  # Require good liquidity
    
    def __init__(self, 
                 api_key: str = None,
                 api_secret: str = None,
                 allow_premarket_orders: bool = False,
                 max_pm_orders: int = 1,
                 pm_order_size: float = 1500):
        """
        Initialize pre-market agent.
        
        Args:
            allow_premarket_orders: Whether to allow pre-market buy orders
            max_pm_orders: Maximum number of PM orders per day
            pm_order_size: Dollar amount per PM order
        """
        self.allow_premarket_orders = allow_premarket_orders
        self.max_pm_orders = max_pm_orders
        self.pm_order_size = pm_order_size
        
        # API credentials
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.api_secret = api_secret or os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_API_SECRET')
        
        # Initialize pre-market analyzer
        self.pm_analyzer = None
        try:
            from autotrade.core.premarket_analyzer import PreMarketAnalyzer
            self.pm_analyzer = PreMarketAnalyzer(
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            logger.debug("Pre-market analyzer initialized")
        except Exception as e:
            logger.warning(f"Could not initialize pre-market analyzer: {e}")
        
        # Initialize trading client for orders
        self.trading_client = None
        if allow_premarket_orders:
            try:
                from alpaca.trading.client import TradingClient
                self.trading_client = TradingClient(
                    api_key=self.api_key,
                    secret_key=self.api_secret,
                    paper=True  # Always paper for now
                )
            except Exception as e:
                logger.warning(f"Could not initialize trading client: {e}")
        
        # Track PM orders placed today
        self.pm_orders_today = 0
        self.last_order_date = None
    
    def analyze_candidate(self, ticker: str, signal_data: Dict = None) -> PreMarketDecision:
        """
        Analyze a single candidate in pre-market.
        
        Includes financial DB validation:
        - Check for same-day earnings (drop if found)
        - Check balance sheet health (down-priority if weak)
        
        Args:
            ticker: Stock ticker
            signal_data: Optional signal data from watchlist
            
        Returns:
            PreMarketDecision with action and reasoning
        """
        if not self.pm_analyzer:
            return PreMarketDecision(
                ticker=ticker,
                action='watch',
                reason='Pre-market analyzer not available'
            )
        
        try:
            pm = self.pm_analyzer.analyze_ticker(ticker)
            
            if not pm.has_data:
                return PreMarketDecision(
                    ticker=ticker,
                    action='watch',
                    reason='No pre-market data available'
                )
            
            # Extract metrics
            gap_pct = pm.gap_pct
            pm_trend = pm.premarket_trend
            pm_vol_ratio = pm.volume_ratio
            liquidity = pm.liquidity_score
            
            # Calculate PM price change from PM open
            pm_change = 0.0
            if pm.premarket_open > 0:
                pm_change = (pm.premarket_price - pm.premarket_open) / pm.premarket_open * 100
            
            # Decision logic
            action = 'watch'
            reason = ''
            priority_change = 0
            
            # ================================================================
            # FINANCIAL DB CHECKS - Early hard filters
            # ================================================================
            
            # Check for same-day earnings (HARD BLOCK)
            earnings_today = flag_earnings_today([ticker])
            if earnings_today:
                return PreMarketDecision(
                    ticker=ticker,
                    action='drop',
                    reason=f"[EARNINGS] Earnings today - removing from candidates",
                    priority_change=-2
                )
            
            # Check balance sheet health (SOFT: down-priority)
            bs_warnings = check_balance_sheet_health([ticker])
            bs_penalty = 0
            if bs_warnings:
                warning = bs_warnings[ticker]
                bs_penalty = 1  # Down-priority if weak BS
                logger.warning(f"[BALANCE SHEET] {ticker}: {warning.get('flags', [])} (down-priority)")
            
            # Check cash flow health (SOFT: down-priority)
            fcf_warnings = check_cash_flow_health([ticker])
            fcf_penalty = 0
            if fcf_warnings:
                fcf_penalty = 1  # Down-priority if negative FCF
                logger.warning(f"[CASH FLOW] {ticker}: negative FCF (down-priority)")
            
            # ================================================================
            # PREMARKET TECHNICAL ANALYSIS (existing logic)
            # ================================================================
            
            # 1. DROP: Significant gap down or weak PM
            if gap_pct <= self.DROP_GAP_THRESHOLD:
                action = 'drop'
                reason = f"Gap down {gap_pct:.1f}% - removing from watchlist"
                priority_change = -2
            
            elif pm_change <= self.DROP_TREND_THRESHOLD and gap_pct < 0:
                action = 'drop'
                reason = f"Weak PM trend ({pm_change:.1f}%) with negative gap"
                priority_change = -2
            
            elif gap_pct < -1.0 and pm_trend == "bearish":
                action = 'drop'
                reason = f"Bearish PM action: gap {gap_pct:.1f}%, trend down"
                priority_change = -2
            
            # 2. STRONG: Significant gap up with bullish PM
            elif gap_pct >= self.PM_ORDER_GAP_THRESHOLD:
                if (pm_vol_ratio >= self.PM_ORDER_VOLUME_THRESHOLD and 
                    liquidity >= self.PM_ORDER_LIQUIDITY_THRESHOLD and
                    pm_trend == "bullish"):
                    action = 'consider_entry'
                    reason = f"Strong PM: +{gap_pct:.1f}% gap, bullish trend, {pm_vol_ratio:.1f}x volume"
                    priority_change = 2 - (bs_penalty + fcf_penalty)  # Reduce uplift if financial warnings
                elif pm_trend == "bullish":
                    action = 'consider_entry'
                    reason = f"Gap up {gap_pct:.1f}% with bullish trend"
                    priority_change = 1 - (bs_penalty + fcf_penalty)
                else:
                    action = 'watch'
                    reason = f"Gap up {gap_pct:.1f}% but PM trend is {pm_trend}"
                    priority_change = -bs_penalty - fcf_penalty
            
            # 3. UPGRADE: Moderate gap up
            elif gap_pct >= self.UPGRADE_GAP_THRESHOLD:
                if pm_trend == "bullish":
                    action = 'watch'
                    reason = f"Positive gap {gap_pct:.1f}% with bullish PM"
                    priority_change = 1 - (bs_penalty + fcf_penalty)
                else:
                    action = 'watch'
                    reason = f"Moderate gap {gap_pct:.1f}%"
                    priority_change = -bs_penalty - fcf_penalty
            
            # 4. NEUTRAL: Small moves
            elif abs(gap_pct) < 1.0:
                action = 'watch'
                reason = f"Flat PM ({gap_pct:+.1f}%)"
                priority_change = -bs_penalty - fcf_penalty
            
            # 5. SLIGHT NEGATIVE: Minor gap down but not critical
            else:
                action = 'watch'
                reason = f"Minor gap {gap_pct:.1f}%"
                if gap_pct < -1.0:
                    priority_change = -1
                priority_change -= bs_penalty + fcf_penalty
            
            return PreMarketDecision(
                ticker=ticker,
                action=action,
                reason=reason,
                gap_pct=gap_pct,
                pm_trend=pm_trend,
                pm_volume_ratio=pm_vol_ratio,
                liquidity_score=liquidity,
                priority_change=priority_change
            )
            
        except Exception as e:
            logger.error(f"Pre-market analysis error for {ticker}: {e}")
            return PreMarketDecision(
                ticker=ticker,
                action='watch',
                reason=f'Analysis error: {e}'
            )
    
    def check_held_positions_earnings(self, holdings: List[str]) -> List[str]:
        """
        Check all held positions for same-day earnings.
        
        Returns:
            List of tickers with earnings today
        """
        if not holdings:
            return []
        
        earnings_today = flag_earnings_today(holdings)
        
        if earnings_today:
            logger.warning("")
            logger.warning("=" * 70)
            logger.warning("[EARNINGS ALERT] Same-day earnings on held positions:")
            for ticker in earnings_today:
                logger.warning(f"  - {ticker}: Monitor for gap/volatility")
            logger.warning("=" * 70)
        
        return earnings_today
    
    def run(self, watchlist: List[Dict], holdings: List[str] = None) -> PreMarketPlan:
        """
        Run pre-market analysis on the full watchlist.
        
        Also checks held positions for same-day earnings.
        
        Args:
            watchlist: List of signal dicts from PM workflow
            holdings: List of held tickers to check for earnings
            
        Returns:
            PreMarketPlan with drops, upgrades, and optional orders
        """
        plan = PreMarketPlan(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        if not watchlist:
            logger.warning("Empty watchlist provided")
            return plan
        
        logger.info("=" * 60)
        logger.info("PRE-MARKET AGENT")
        logger.info(f"Time: {plan.timestamp}")
        logger.info("=" * 60)
        
        # Check held positions for same-day earnings (FIRST THING)
        if holdings:
            earnings_tickers = self.check_held_positions_earnings(holdings)
            plan.earnings_alerts = earnings_tickers
        
        # Reset daily order count if new day
        today = datetime.now().date()
        if self.last_order_date != today:
            self.pm_orders_today = 0
            self.last_order_date = today
        
        # Analyze each candidate
        for signal in watchlist:
            ticker = signal.get('ticker') or signal.get('symbol')
            if not ticker:
                continue
            
            decision = self.analyze_candidate(ticker, signal)
            plan.total_analyzed += 1
            
            # Categorize
            if decision.action == 'drop':
                plan.drops.append(decision)
                plan.total_dropped += 1
                logger.info(f"  [DROP] {ticker}: {decision.reason}")
            
            elif decision.action == 'consider_entry':
                plan.strong_candidates.append(decision)
                plan.total_upgraded += 1
                logger.info(f"  [STRONG] {ticker}: {decision.reason}")
            
            else:
                plan.watches.append(decision)
                if decision.priority_change != 0:
                    logger.debug(f"  [WATCH] {ticker}: {decision.reason} (priority {decision.priority_change:+d})")
        
        # Consider pre-market orders for exceptional opportunities
        if self.allow_premarket_orders and plan.strong_candidates:
            self._consider_pm_orders(plan)
        
        # Summary
        logger.info("")
        logger.info(f"[SUMMARY] Analyzed: {plan.total_analyzed}")
        logger.info(f"          Dropped: {plan.total_dropped}")
        logger.info(f"          Strong: {len(plan.strong_candidates)}")
        if plan.premarket_orders:
            logger.info(f"          PM Orders: {len(plan.premarket_orders)}")
        logger.info("=" * 60)
        
        return plan
    
    def _consider_pm_orders(self, plan: PreMarketPlan):
        """
        Consider placing pre-market orders for exceptional opportunities.
        
        Very selective - only for the best setups.
        """
        if self.pm_orders_today >= self.max_pm_orders:
            logger.info(f"[PM ORDER] Daily limit reached ({self.pm_orders_today}/{self.max_pm_orders})")
            return
        
        # Sort by gap (strongest first)
        sorted_candidates = sorted(
            plan.strong_candidates,
            key=lambda x: x.gap_pct,
            reverse=True
        )
        
        for candidate in sorted_candidates[:3]:  # Only consider top 3
            # Extra validation for PM orders
            if candidate.gap_pct < self.PM_ORDER_GAP_THRESHOLD:
                continue
            if candidate.pm_volume_ratio < self.PM_ORDER_VOLUME_THRESHOLD:
                continue
            if candidate.liquidity_score < self.PM_ORDER_LIQUIDITY_THRESHOLD:
                continue
            if candidate.pm_trend != "bullish":
                continue
            
            # This is an exceptional opportunity
            logger.info(f"[PM ORDER] Exceptional opportunity: {candidate.ticker}")
            logger.info(f"           Gap: +{candidate.gap_pct:.1f}%")
            logger.info(f"           Volume: {candidate.pm_volume_ratio:.1f}x")
            logger.info(f"           Liquidity: {candidate.liquidity_score:.0f}/100")
            
            # In production, would place order here
            # For now, just flag it
            candidate.action = 'premarket_buy'
            plan.premarket_orders.append(candidate)
            self.pm_orders_today += 1
            
            if self.pm_orders_today >= self.max_pm_orders:
                break
    
    def get_updated_watchlist(self, watchlist: List[Dict], plan: PreMarketPlan) -> List[Dict]:
        """
        Return updated watchlist with drops removed and priorities adjusted.
        
        Args:
            watchlist: Original watchlist
            plan: PreMarketPlan from run()
            
        Returns:
            Updated watchlist with drops removed
        """
        # Get dropped tickers
        dropped_tickers = {d.ticker for d in plan.drops}
        
        # Get priority adjustments
        priority_adj = {}
        for d in plan.strong_candidates + plan.watches:
            if d.priority_change != 0:
                priority_adj[d.ticker] = d.priority_change
        
        # Filter and adjust
        updated = []
        for signal in watchlist:
            ticker = signal.get('ticker') or signal.get('symbol')
            if ticker in dropped_tickers:
                continue  # Skip dropped
            
            # Adjust priority/score if needed
            if ticker in priority_adj:
                adj = priority_adj[ticker]
                if 'priority' in signal:
                    signal['priority'] = max(1, signal['priority'] - adj)  # Lower number = higher priority
                if 'score' in signal:
                    signal['score'] = signal['score'] + (adj * 5)  # Bump score
            
            updated.append(signal)
        
        return updated


def demo():
    """Demo the pre-market agent (logging-only)."""
    logger.info("PreMarketAgent demo disabled; use CLI/tests for execution.")


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info('premarket_agent.py loaded; use pytest/CLI for tests.')

