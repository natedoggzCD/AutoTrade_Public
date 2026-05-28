"""
Portfolio Rotator - Active Capital Rotation System
===================================================
The core of active trading: rotate capital from weak to strong positions.

Key Principles:
1. Limited capital, limited positions (7 max)
2. Every slot should be earning - no dead money
3. Conviction/risk rules for exits, adds, and rotations
4. Conviction-based: need reasons TO hold, not just reasons to exit

This module coordinates:
- DayManager: Intraday position monitoring
- OvernightAgent: After-market analysis & planning
- RiskGate: Fast path for critical decisions
- ConvictionEngine: Scoring positions for rotation
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from autotrade.risk.position_state import (
    PositionState, PortfolioState, PDTStatus,
    HoldPhase,
)
from autotrade.signals.conviction_engine import ConvictionEngine
from autotrade.risk.risk_gate import RiskGate, RiskGateConfig, RiskAction, ActionPlan
from autotrade.risk.strategy_failsafe import StrategyFailsafeManager
from autotrade.core.overnight_agent import OvernightAgent, WatchlistCandidate, NextDayPlan

logger = logging.getLogger(__name__)
PROJECT_DIR = Path(os.environ.get("AUTOTRADE_ROOT", Path(__file__).resolve().parents[2]))


@dataclass
class RotationDecision:
    """A decision to rotate one position for another."""
    exit_symbol: str
    exit_qty: int
    exit_price: float
    exit_conviction: float
    exit_reasons: List[str]
    
    enter_symbol: str
    enter_price: float
    enter_qty: int
    enter_signal_score: float
    enter_reasons: List[str]
    
    improvement_score: float  # How much better is new position?
    capital_amount: float     # Capital being rotated
    
    def summary(self) -> str:
        return (
            f"ROTATE: {self.exit_symbol} ({self.exit_conviction:.0f} conviction) -> "
            f"{self.enter_symbol} ({self.enter_signal_score:.0f} signal) | "
            f"+{self.improvement_score:.0f} improvement | ${self.capital_amount:,.0f}"
        )


class PortfolioRotator:
    """
    Coordinates all trading decisions for the portfolio.
    
    Intraday Flow:
    1. Check critical positions (hard stops, ATR stops)
    2. Check rotation candidates (unlocked + low conviction)
    3. Check ADD candidates (unlocked + high conviction + profit)
    4. Execute decisions via DayManager
    
    Overnight Flow:
    1. Score all positions on conviction
    2. Identify rotation candidates for next day
    3. Scan watchlist for entry candidates
    4. Generate next-day trading plan
    """
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        dry_run: bool = True,
        trading_client=None,
        data_client=None,
    ):
        self.config_path = config_path or (PROJECT_DIR / 'config' / 'trading_config.yaml')
        self.dry_run = dry_run
        self.trading_client = trading_client
        self.data_client = data_client
        
        # Initialize components
        self.risk_gate = RiskGate(RiskGateConfig.from_yaml(self.config_path) if self.config_path.exists() else None)
        self.conviction_engine = ConvictionEngine()
        self.overnight_agent = OvernightAgent()
        self.strategy_failsafe = StrategyFailsafeManager()
        
        # Load config
        self.config = self._load_config()
        
        # State tracking
        self.portfolio = PortfolioState()
        self.pending_rotations: List[RotationDecision] = []
        self.executed_today: Dict[str, List[str]] = {'exits': [], 'entries': [], 'adds': []}
        
        logger.info(f"PortfolioRotator initialized (dry_run={dry_run})")

    def sync_positions(
        self,
        positions: List[Any],
        position_entries: Dict[str, datetime],
        remaining_day_trades: int,
    ) -> PortfolioState:
        """
        Synchronize PortfolioState with live Alpaca positions.

        Args:
            positions: List of Alpaca position objects
            position_entries: Dict of symbol -> entry_time (from DayManager state)
            remaining_day_trades: Ignored legacy argument; PDT is disabled

        Returns:
            Updated PortfolioState
        """
        portfolio_cfg = self.config.get('portfolio', {})
        pdt_status = PDTStatus(
            trades_used_today=0,
            trades_remaining=999,
            can_burn_day_trade=True,
            emergency_only=False,
        )

        portfolio = PortfolioState(
            pdt_status=pdt_status,
            max_positions=portfolio_cfg.get('max_positions', 7),
            position_size_target=portfolio_cfg.get('position_size_target', 3000.0),
            position_size_max=portfolio_cfg.get('position_size_max', 6000.0),
        )

        total_value = 0.0
        for pos in positions:
            symbol = getattr(pos, 'symbol', None)
            if not symbol:
                continue

            entry_price = float(getattr(pos, 'avg_entry_price', 0) or 0)
            current_price = float(getattr(pos, 'current_price', 0) or 0)
            qty = int(getattr(pos, 'qty', 0) or 0)
            market_value = float(getattr(pos, 'market_value', 0) or 0)

            entry_time = position_entries.get(symbol)
            if entry_time is None:
                # Default to "locked" when we don't know entry time.
                entry_time = datetime.now()
                logger.debug(f"[sync_positions] Missing entry time for {symbol}; defaulting to now")

            state = PositionState(
                symbol=symbol,
                entry_price=entry_price,
                current_price=current_price,
                qty=qty,
                entry_time=entry_time,
            )

            # Populate basic context if available
            state.pnl_pct = float(getattr(pos, 'unrealized_plpc', 0) or 0) * 100
            state.pnl_dollars = (state.pnl_pct / 100) * (entry_price * qty) if entry_price else 0.0

            portfolio.add_position(state)
            total_value += abs(market_value)

        portfolio.positions_value = total_value

        # Attempt to enrich with account context when client is available.
        if self.trading_client:
            try:
                account = self.trading_client.get_account()
                portfolio.total_equity = float(account.equity)
                portfolio.cash_available = float(account.cash)
                portfolio.buying_power = float(account.buying_power)
            except Exception as e:
                logger.debug(f"[sync_positions] Account fetch failed: {e}")

        self.update_portfolio(portfolio)
        return portfolio
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML."""
        if self.config_path.exists():
            import yaml
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def update_portfolio(self, portfolio: PortfolioState):
        """Update portfolio state."""
        self.portfolio = portfolio
    
    def evaluate_position(
        self,
        pos: PositionState,
        day_trades_remaining: int = 10
    ) -> ActionPlan:
        """
        Evaluate a single position and return action plan.
        
        Uses both risk gate and conviction engine.
        """
        # First compute conviction
        conviction_score, factors, conviction_reasons = self.conviction_engine.compute_conviction(
            symbol=pos.symbol,
            pnl_pct=pos.pnl_pct,
            hold_minutes=pos.hold_minutes,
            atr_pct=pos.atr_pct,
            rsi=pos.rsi,
            current_price=pos.current_price,
            entry_price=pos.entry_price,
            high_since_entry=pos.high_since_entry,
            support_price=pos.s1_price,
            resistance_price=pos.r1_price,
            sector_momentum=pos.sector_momentum,
            relative_strength=pos.relative_strength,
            news_sentiment=pos.sentiment_score
        )
        
        # Update position with conviction
        pos.set_conviction(conviction_score, conviction_reasons)
        
        # Run risk gate; PDT limits are disabled.
        failsafe_snapshot = self.strategy_failsafe.load_snapshot()
        action_plan = self.risk_gate.evaluate_with_pdt(
            symbol=pos.symbol,
            entry_price=pos.entry_price,
            current_price=pos.current_price,
            qty=pos.qty,
            atr_pct=pos.atr_pct,
            time_in_trade_minutes=pos.hold_minutes,
            high_since_entry=pos.high_since_entry,
            day_trades_remaining=day_trades_remaining,
            conviction_score=conviction_score,
            rsi=pos.rsi,
            relative_strength=pos.relative_strength,
            stop_multiplier=failsafe_snapshot.stop_multiplier,
        )
        
        # If no risk gate trigger, check conviction-based actions
        if action_plan.action == RiskAction.NONE:
            min_hold_conviction = max(
                self.config.get('conviction_engine', {}).get('hold_conviction_min', 40),
                failsafe_snapshot.min_conviction_hold
            )
            # Low conviction + unlocked = recommend rotation
            if (pos.hold_phase == HoldPhase.UNLOCKED 
                    and conviction_score < min_hold_conviction):
                action_plan.action = RiskAction.EXIT
                action_plan.reasons.append(
                    f"Low conviction ({conviction_score:.0f}) < min_hold {min_hold_conviction:.0f} - rotation candidate"
                )
                action_plan.triggered_rules.append("low_conviction")
                action_plan.conviction_score = conviction_score
        
        return action_plan
    
    def evaluate_all_positions(self) -> Dict[str, ActionPlan]:
        """
        Evaluate all positions in portfolio.
        
        Returns dict of symbol -> ActionPlan
        """
        results = {}
        
        for symbol, pos in self.portfolio.positions.items():
            action_plan = self.evaluate_position(
                pos, 
                self.portfolio.pdt_status.trades_remaining
            )
            results[symbol] = action_plan
            
            # Log significant actions
            if action_plan.action != RiskAction.NONE:
                logger.info(f"[{symbol}] {action_plan.action.value.upper()}: {action_plan.reasons[0] if action_plan.reasons else 'N/A'}")
        
        return results
    
    def find_rotation_candidates(self) -> List[Tuple[PositionState, float]]:
        """
        Find positions eligible for rotation.
        
        Returns list of (position, conviction_score) sorted by conviction (lowest first).
        """
        candidates = []
        
        for symbol, pos in self.portfolio.positions.items():
            # Must be unlocked (past 1D hold)
            if pos.hold_phase not in (HoldPhase.UNLOCKED, HoldPhase.UNLOCKING):
                continue
            
            # Get conviction score
            conviction_score = pos.conviction_score
            
            # Low conviction = rotation candidate
            if conviction_score < self.config.get('conviction_engine', {}).get('hold_conviction_min', 40):
                candidates.append((pos, conviction_score))
        
        # Sort by conviction (lowest first = rotate first)
        candidates.sort(key=lambda x: x[1])
        
        return candidates
    
    def find_add_candidates(self) -> List[Tuple[PositionState, float]]:
        """
        Find positions eligible for adding.
        
        Returns list of (position, conviction_score) sorted by conviction (highest first).
        """
        candidates = []
        
        for symbol, pos in self.portfolio.positions.items():
            # Must have profit >= threshold
            add_threshold = self.config.get('risk_gate', {}).get('add_profit_threshold_pct', 4.0)
            if pos.pnl_pct < add_threshold:
                continue
            
            # Get conviction score
            conviction_score = pos.conviction_score
            
            # High conviction = add candidate
            add_min = self.config.get('risk_gate', {}).get('add_conviction_min', 65.0)
            if conviction_score >= add_min:
                candidates.append((pos, conviction_score))
        
        # Sort by conviction (highest first = add first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates
    
    def generate_rotation_plan(
        self,
        rotation_candidates: List[Tuple[PositionState, float]],
        entry_candidates: List[WatchlistCandidate]
    ) -> List[RotationDecision]:
        """
        Match rotation candidates with entry candidates.
        
        Only rotates if entry is significantly better than exit.
        """
        rotations = []
        min_improvement = self.config.get('rotation', {}).get('min_improvement_score', 15.0)
        max_rotations = self.config.get('rotation', {}).get('max_rotations_per_day', 2)
        
        available_entries = entry_candidates.copy()
        
        for pos, conviction in rotation_candidates:
            if len(rotations) >= max_rotations:
                break
            
            if not available_entries:
                break
            
            # Find best available entry
            best_entry = available_entries[0]
            improvement = best_entry.signal_score - conviction
            
            if improvement >= min_improvement:
                capital = pos.current_price * pos.qty
                entry_qty = int(capital / best_entry.current_price) if best_entry.current_price > 0 else 0
                
                rotation = RotationDecision(
                    exit_symbol=pos.symbol,
                    exit_qty=pos.qty,
                    exit_price=pos.current_price,
                    exit_conviction=conviction,
                    exit_reasons=pos.conviction_reasons,
                    enter_symbol=best_entry.symbol,
                    enter_price=best_entry.current_price,
                    enter_qty=entry_qty,
                    enter_signal_score=best_entry.signal_score,
                    enter_reasons=best_entry.catalysts,
                    improvement_score=improvement,
                    capital_amount=capital
                )
                
                rotations.append(rotation)
                available_entries.pop(0)
                
                logger.info(rotation.summary())
        
        return rotations
    
    def run_intraday_cycle(self) -> Dict[str, Any]:
        """
        Run one intraday evaluation cycle.
        
        Returns summary of actions taken.
        """
        logger.info("=" * 60)
        logger.info("INTRADAY CYCLE")
        logger.info("=" * 60)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'positions_evaluated': 0,
            'critical_exits': [],
            'rotation_candidates': [],
            'add_candidates': [],
            'actions_taken': []
        }
        
        # Evaluate all positions
        action_plans = self.evaluate_all_positions()
        results['positions_evaluated'] = len(action_plans)
        
        for symbol, plan in action_plans.items():
            pos = self.portfolio.positions.get(symbol)
            if not pos:
                continue
            
            # Handle critical exits (hard stop, ATR stop)
            # Check if it's a critical exit by looking at triggered rules
            is_critical = (
                plan.action == RiskAction.EXIT and 
                ('hard_stop' in plan.triggered_rules or 
                 plan.hold_phase.value == 'critical' or
                 pos.is_critical)
            )
            
            if is_critical:
                results['critical_exits'].append({
                    'symbol': symbol,
                    'pnl_pct': pos.pnl_pct,
                    'reason': plan.triggered_rules[0] if plan.triggered_rules else 'unknown'
                })
                results['actions_taken'].append(f"EXIT {symbol} (critical)")
                self.executed_today['exits'].append(symbol)
            
            # Track rotation candidates
            elif plan.action == RiskAction.EXIT and 'low_conviction' in plan.triggered_rules:
                results['rotation_candidates'].append({
                    'symbol': symbol,
                    'conviction': plan.conviction_score,
                    'pnl_pct': pos.pnl_pct
                })
            
            # Handle ADD signals
            elif plan.action == RiskAction.ADD:
                results['add_candidates'].append({
                    'symbol': symbol,
                    'conviction': plan.conviction_score,
                    'pnl_pct': pos.pnl_pct,
                    'add_qty': plan.size_delta
                })
                if not self.dry_run:
                    results['actions_taken'].append(f"ADD {plan.size_delta} {symbol}")
                    self.executed_today['adds'].append(symbol)
        
        # Log summary
        logger.info(f"Evaluated {results['positions_evaluated']} positions")
        if results['critical_exits']:
            logger.warning(f"Critical exits: {[e['symbol'] for e in results['critical_exits']]}")
        if results['add_candidates']:
            logger.info(f"ADD candidates: {[a['symbol'] for a in results['add_candidates']]}")
        
        return results
    
    def run_overnight_analysis(self) -> NextDayPlan:
        """
        Run overnight analysis and generate next-day plan.
        """
        logger.info("=" * 60)
        logger.info("OVERNIGHT ANALYSIS")
        logger.info("=" * 60)
        
        # Generate plan using overnight agent
        plan = self.overnight_agent.generate_next_day_plan(self.portfolio)
        
        return plan
    
    def get_portfolio_summary(self) -> str:
        """Get human-readable portfolio summary."""
        lines = [
            "=" * 60,
            "PORTFOLIO SUMMARY",
            "=" * 60,
            ""
        ]
        
        # Overall stats
        lines.append(self.portfolio.summary())
        lines.append("")
        
        # Position details
        lines.append("POSITIONS:")
        for symbol, pos in self.portfolio.positions.items():
            phase_icon = {
                HoldPhase.LOCKED: "[LOCK]",
                HoldPhase.UNLOCKING: "[UNLK]",
                HoldPhase.UNLOCKED: "[OK]",
                HoldPhase.CRITICAL: "[!]"
            }.get(pos.hold_phase, "[?]")
            
            lines.append(
                f"  {phase_icon} {symbol:6} | "
                f"P&L: {pos.pnl_pct:+5.1f}% | "
                f"Conv: {pos.conviction_score:3.0f} | "
                f"Action: {pos.recommended_action.value}"
            )
        
        # Rotation/Add candidates
        rotation_cands = self.find_rotation_candidates()
        add_cands = self.find_add_candidates()
        
        if rotation_cands:
            lines.append("")
            lines.append("ROTATION CANDIDATES (low conviction, unlocked):")
            for pos, score in rotation_cands[:3]:
                lines.append(f"  [OUT] {pos.symbol}: conviction {score:.0f}, P&L {pos.pnl_pct:+.1f}%")
        
        if add_cands:
            lines.append("")
            lines.append("ADD CANDIDATES (high conviction, profitable):")
            for pos, score in add_cands[:3]:
                lines.append(f"  [ADD] {pos.symbol}: conviction {score:.0f}, P&L {pos.pnl_pct:+.1f}%")
        
        return "\n".join(lines)


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info('portfolio_rotator.py loaded; use pytest/CLI for tests.')

