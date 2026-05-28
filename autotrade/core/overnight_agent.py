"""
Overnight Agent - After-Market Analysis & Next-Day Planning
=============================================================
Runs after market close to:
1. Analyze current positions for next-day action
2. Scan watchlist for potential new entries
3. Identify rotation candidates (weak positions to close)
4. Queue up next-day orders
5. Generate trading plan for next session

Key insight: Positions bought at 2:30 PM can sell at next market open
(past 1D hold restriction), so overnight we identify who to rotate.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import json

from autotrade.risk.position_state import PositionState, PortfolioState, HoldPhase, PositionAction
from autotrade.signals.conviction_engine import ConvictionEngine
from autotrade.analysis.financial_checks import (
    check_earnings_risk,
    check_balance_sheet_health,
    check_cash_flow_health,
    check_valuation_sanity,
    check_options_positioning,
)

logger = logging.getLogger(__name__)
PROJECT_DIR = Path(os.environ.get("AUTOTRADE_ROOT", Path(__file__).resolve().parents[2]))


@dataclass
class WatchlistCandidate:
    """A stock being considered for next-day entry."""
    symbol: str
    
    # Signal strength
    signal_score: float = 0.0      # From your stage1/stage2 pipeline
    signal_type: str = ""          # e.g., "support_bounce", "breakout"
    
    # S/R levels
    s1_price: float = 0.0
    r1_price: float = 0.0
    
    # Technical context
    rsi: float = 50.0
    atr_pct: float = 2.0
    current_price: float = 0.0
    
    # Why we're watching
    catalysts: List[str] = field(default_factory=list)
    
    # Priority for entry (higher = enter first)
    priority: float = 0.0
    
    # Entry plan
    target_entry_price: float = 0.0
    stop_loss_price: float = 0.0
    position_size: float = 3000.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'signal_score': self.signal_score,
            'signal_type': self.signal_type,
            'priority': self.priority,
            'target_entry': self.target_entry_price,
            'stop_loss': self.stop_loss_price,
            'catalysts': self.catalysts
        }


@dataclass
class RotationCandidate:
    """A current position being considered for rotation (exit)."""
    position: PositionState
    conviction_score: float
    reasons: List[str]
    capital_to_free: float  # How much capital we'd recover
    priority: float = 0.0   # Higher = rotate first
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.position.symbol,
            'pnl_pct': self.position.pnl_pct,
            'conviction': self.conviction_score,
            'capital': self.capital_to_free,
            'priority': self.priority,
            'reasons': self.reasons
        }


@dataclass
class NextDayPlan:
    """Complete trading plan for next session."""
    created_at: datetime = field(default_factory=datetime.now)
    
    # Positions to exit at open
    exit_at_open: List[str] = field(default_factory=list)
    exit_reasons: Dict[str, List[str]] = field(default_factory=dict)
    
    # Positions to potentially add to
    add_candidates: List[str] = field(default_factory=list)
    add_targets: Dict[str, float] = field(default_factory=dict)  # symbol -> target add %
    
    # New entries to make
    entry_candidates: List[WatchlistCandidate] = field(default_factory=list)
    
    # Rotation plan (exit weak, enter strong)
    rotations: List[Tuple[str, str]] = field(default_factory=list)  # (exit, enter)
    
    # Capital allocation
    capital_to_deploy: float = 0.0
    capital_from_exits: float = 0.0
    
    # Notes for human review
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'created_at': self.created_at.isoformat(),
            'exit_at_open': self.exit_at_open,
            'exit_reasons': self.exit_reasons,
            'add_candidates': self.add_candidates,
            'add_targets': self.add_targets,
            'entry_candidates': [c.to_dict() for c in self.entry_candidates],
            'rotations': self.rotations,
            'capital_to_deploy': self.capital_to_deploy,
            'capital_from_exits': self.capital_from_exits,
            'notes': self.notes
        }
    
    def summary(self) -> str:
        lines = [
            f"=== Next Day Trading Plan ({self.created_at.strftime('%Y-%m-%d')}) ===",
            ""
        ]
        
        if self.exit_at_open:
            lines.append("EXIT AT OPEN:")
            for sym in self.exit_at_open:
                reasons = self.exit_reasons.get(sym, [])
                lines.append(f"   {sym}: {'; '.join(reasons[:2])}")
            lines.append("")
        
        if self.add_candidates:
            lines.append("ADD TO WINNERS:")
            for sym in self.add_candidates:
                target = self.add_targets.get(sym, 0)
                lines.append(f"   {sym}: +{target:.0%} position")
            lines.append("")
        
        if self.entry_candidates:
            lines.append("NEW ENTRIES:")
            for c in self.entry_candidates[:5]:
                lines.append(f"   {c.symbol}: {c.signal_type} (score: {c.signal_score:.0f})")
            lines.append("")
        
        if self.rotations:
            lines.append("ROTATIONS:")
            for exit_sym, enter_sym in self.rotations:
                lines.append(f"   {exit_sym} -> {enter_sym}")
            lines.append("")
        
        lines.append(f"Capital: ${self.capital_from_exits:,.0f} from exits, ${self.capital_to_deploy:,.0f} to deploy")
        
        if self.notes:
            lines.append("")
            lines.append("Notes:")
            for note in self.notes:
                lines.append(f"   - {note}")
        
        return "\n".join(lines)


class OvernightAgent:
    """
    Runs after market close to plan next day's trading.
    
    Workflow:
    1. Score all current positions on conviction
    2. Identify positions unlocking tomorrow (past 1D hold)
    3. Find rotation candidates (low conviction + unlocked)
    4. Scan watchlist for entry candidates
    5. Match rotations: exit weak, enter strong
    6. Generate next-day trading plan
    """
    
    def __init__(
        self,
        data_dir: Path = None,
        max_entries_per_day: int = 3,
        min_rotation_improvement: float = 15.0  # New position must be 15% better score
    ):
        if data_dir is not None:
            self.data_dir = data_dir
        else:
            down_root = os.environ.get("DOWNDAY_ROOT")
            self.data_dir = (Path(down_root) / "data") if down_root else (PROJECT_DIR / "data")
        self.conviction_engine = ConvictionEngine()
        self.max_entries = max_entries_per_day
        self.min_rotation_improvement = min_rotation_improvement
        
        # Adaptive thresholds
        self.min_signal_score = 65.0
        self.min_conviction_score = 60.0
        
        # Plan storage
        self.plan_dir = PROJECT_DIR / 'plans'
        self.plan_dir.mkdir(exist_ok=True)

    def _apply_performance_feedback(self, eod_results: Dict[str, Any]):
        """
        Adjust thresholds based on EOD performance results.
        """
        if not eod_results:
            return
            
        win_rate = eod_results.get("win_rate", 0.5)
        avg_pnl = eod_results.get("avg_pnl", 0.0)
        
        # If performance is poor, raise the bar
        if win_rate < 0.4 or avg_pnl < -100:
            self.min_signal_score += 5.0
            self.min_conviction_score += 5.0
            logger.info(f"Raising thresholds due to poor performance: min_signal={self.min_signal_score}")
        elif win_rate > 0.6 and avg_pnl > 100:
            self.min_signal_score = max(60.0, self.min_signal_score - 2.0)
            self.min_conviction_score = max(55.0, self.min_conviction_score - 2.0)
            logger.info(f"Lowering thresholds due to good performance: min_signal={self.min_signal_score}")
    
    def analyze_portfolio(self, portfolio: PortfolioState) -> Dict[str, Any]:
        """
        Analyze current portfolio and score all positions.
        
        Returns:
            Analysis dict with scored positions and recommendations
        """
        analysis = {
            'positions': {},
            'rotation_candidates': [],
            'add_candidates': [],
            'critical': []
        }
        
        for symbol, pos in portfolio.positions.items():
            # Compute conviction
            score, factors, reasons = self.conviction_engine.compute_conviction(
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
            pos.set_conviction(score, reasons)
            
            # Get action recommendation
            action, confidence, action_reasons = self.conviction_engine.get_action_recommendation(
                conviction_score=score,
                pnl_pct=pos.pnl_pct,
                hold_phase=pos.hold_phase.value,
                is_critical=pos.is_critical
            )
            pos.set_action(PositionAction(action), confidence, action_reasons)
            
            analysis['positions'][symbol] = {
                'conviction': score,
                'factors': factors.to_dict(),
                'reasons': reasons,
                'action': action,
                'hold_phase': pos.hold_phase.value
            }
            
            # Critical positions
            if pos.is_critical:
                analysis['critical'].append(pos)
            
            # Rotation candidates: Low conviction + unlocked
            if (pos.hold_phase in (HoldPhase.UNLOCKED, HoldPhase.UNLOCKING) 
                    and score < self.min_conviction_score - 20):
                capital = pos.current_price * pos.qty
                analysis['rotation_candidates'].append(RotationCandidate(
                    position=pos,
                    conviction_score=score,
                    reasons=reasons,
                    capital_to_free=capital,
                    priority=100 - score  # Lower conviction = higher priority to rotate
                ))
            
            # ADD candidates: High conviction + profitable
            if score >= self.min_conviction_score + 5 and pos.pnl_pct >= 4.0:
                analysis['add_candidates'].append(pos)
        
        # Sort rotation candidates by priority
        analysis['rotation_candidates'].sort(key=lambda x: x.priority, reverse=True)
        
        return analysis
    
    def scan_watchlist(self, max_candidates: int = 10) -> List[WatchlistCandidate]:
        """
        Scan for next-day entry candidates from the local price-data screener.
        
        Applies financial DB checks to filter and score candidates:
        1. Filter out tickers with earnings in next 5 days
        2. Apply score penalties for weak balance sheets
        3. Apply score penalties for negative free cash flow
        4. Flag valuation and options risks (soft, not hard blocks)
        """
        candidates = []
        try:
            from autotrade.signals.screener_v2 import get_entry_candidates

            def _to_float(value, default: float = 0.0) -> float:
                try:
                    out = float(value)
                    return out if out == out else default
                except Exception:
                    return default

            raw = get_entry_candidates(
                max_candidates=max(max_candidates * 3, 30),
                exclude_symbols=None,
                symbols=None,
                log_samples=False,
                scoring_mode="momentum_pullback",
            )

            for row in raw:
                symbol = str(row.get("ticker", "")).upper().strip()
                if not symbol:
                    continue

                score = _to_float(row.get("score", row.get("composite_score", 0)), 0.0)
                if score < self.min_signal_score:
                    continue

                candidates.append(
                    WatchlistCandidate(
                        symbol=symbol,
                        signal_score=score,
                        signal_type=str(row.get("strategy_profile", "momentum_pullback")),
                        s1_price=_to_float(row.get("s1_price", 0), 0.0),
                        r1_price=_to_float(row.get("r1_price", 0), 0.0),
                        current_price=_to_float(row.get("price", 0), 0.0),
                        target_entry_price=_to_float(row.get("price", 0), 0.0),
                        stop_loss_price=_to_float(row.get("stop_price", 0), 0.0),
                        priority=score,
                    )
                )
                if len(candidates) >= max_candidates:
                    break
        except Exception as e:
            logger.warning(f"Could not load screener candidates: {e}")
        
        # ================================================================
        # FINANCIAL DB CHECKS - Filter and score candidates
        # ================================================================
        if candidates:
            logger.info("")
            logger.info("=" * 70)
            logger.info("FINANCIAL DB CHECKS: Filtering entry candidates")
            logger.info("=" * 70)
            
            candidate_symbols = [c.symbol for c in candidates]
            
            # 1. CHECK EARNINGS RISK - Hard filter (next 5 days)
            earnings_risk = check_earnings_risk(candidate_symbols, days=5)
            if earnings_risk:
                logger.info(f"[EARNINGS] Filtering out {len(earnings_risk)} tickers with earnings in next 5 days")
                candidates = [c for c in candidates if c.symbol not in earnings_risk]
                logger.info(f"  Removed: {sorted(earnings_risk.keys())}")
            
            # 2. CHECK BALANCE SHEET HEALTH - Scoring penalty
            if candidates:
                bs_warnings = check_balance_sheet_health([c.symbol for c in candidates])
                if bs_warnings:
                    logger.info(f"[BALANCE SHEET] Applying penalties to {len(bs_warnings)} tickers")
                    for candidate in candidates:
                        if candidate.symbol in bs_warnings:
                            warning = bs_warnings[candidate.symbol]
                            penalty = 15.0  # Score penalty
                            logger.warning(f"  {candidate.symbol}: {warning.get('flags', [])} (penalty: -{penalty:.0f})")
                            candidate.priority -= penalty
                            candidate.signal_score -= penalty
            
            # 3. CHECK CASH FLOW HEALTH - Scoring penalty
            if candidates:
                fcf_warnings = check_cash_flow_health([c.symbol for c in candidates])
                if fcf_warnings:
                    logger.info(f"[CASH FLOW] Applying penalties to {len(fcf_warnings)} tickers")
                    for candidate in candidates:
                        if candidate.symbol in fcf_warnings:
                            warning = fcf_warnings[candidate.symbol]
                            penalty = 10.0  # Lighter penalty than BS
                            logger.warning(f"  {candidate.symbol}: negative FCF {warning.get('free_cash_flow', 'N/A')} (penalty: -{penalty:.0f})")
                            candidate.priority -= penalty
                            candidate.signal_score -= penalty
            
            # 4. CHECK VALUATION SANITY - Scoring penalty
            if candidates:
                val_warnings = check_valuation_sanity([c.symbol for c in candidates])
                if val_warnings:
                    logger.info(f"[VALUATION] Applying penalties to {len(val_warnings)} tickers")
                    for candidate in candidates:
                        if candidate.symbol in val_warnings:
                            warning = val_warnings[candidate.symbol]
                            flags = warning.get('flags', [])
                            penalty = 0.0
                            for f in flags:
                                if "extreme_pe" in f: penalty += 15.0
                                elif "high_debt_equity" in f: penalty += 20.0
                                elif "earnings_decline" in f: penalty += 10.0
                            
                            if penalty > 0:
                                logger.warning(f"  {candidate.symbol}: {flags} (penalty: -{penalty:.0f})")
                                candidate.priority -= penalty
                                candidate.signal_score -= penalty
            
            # 5. CHECK OPTIONS POSITIONING - Scoring penalty
            if candidates:
                opts_warnings = check_options_positioning([c.symbol for c in candidates])
                if opts_warnings:
                    logger.info(f"[OPTIONS] Applying penalties to {len(opts_warnings)} tickers")
                    for candidate in candidates:
                        if candidate.symbol in opts_warnings:
                            warning = opts_warnings[candidate.symbol]
                            pc_ratio = warning.get('put_call_oi_ratio', 0.0)
                            penalty = 0.0
                            if pc_ratio > 2.0: penalty = 20.0
                            elif pc_ratio > 1.5: penalty = 10.0
                            
                            if penalty > 0:
                                logger.warning(f"  {candidate.symbol}: put/call ratio {pc_ratio:.2f} (penalty: -{penalty:.0f})")
                                candidate.priority -= penalty
                                candidate.signal_score -= penalty

            # 6. CHECK DIVIDEND STABILITY - Scoring penalty / Hard filter
            if candidates:
                div_warnings = check_dividend_stability([c.symbol for c in candidates])
                if div_warnings:
                    logger.info(f"[DIVIDEND] Applying penalties to {len(div_warnings)} tickers")
                    orig_count = len(candidates)
                    
                    # We might need to filter some out entirely (suspensions)
                    new_candidates = []
                    for candidate in candidates:
                        if candidate.symbol in div_warnings:
                            warning = div_warnings[candidate.symbol]
                            flags = warning.get('flags', [])
                            if "dividend_suspended" in flags:
                                logger.warning(f"  {candidate.symbol}: DIVIDEND SUSPENDED (FILTERED)")
                                continue # Hard filter
                            
                            penalty = 20.0 # Standard cut penalty
                            logger.warning(f"  {candidate.symbol}: {flags} (penalty: -{penalty:.0f})")
                            candidate.priority -= penalty
                            candidate.signal_score -= penalty
                        
                        new_candidates.append(candidate)
                    
                    candidates = new_candidates
                    if len(candidates) < orig_count:
                        logger.info(f"  Removed {orig_count - len(candidates)} tickers due to dividend suspension")
            
            logger.info("=" * 70)
        
        # Re-sort by (adjusted) priority
        candidates.sort(key=lambda x: x.priority, reverse=True)
        
        return candidates[:max_candidates]
    
    def create_rotation_plan(
        self,
        rotation_candidates: List[RotationCandidate],
        entry_candidates: List[WatchlistCandidate],
        open_slots: int
    ) -> List[Tuple[str, str]]:
        """
        Match rotation candidates with entry candidates.
        
        Only rotate if new entry is significantly better than current position.
        """
        rotations = []
        
        for rotation in rotation_candidates:
            if not entry_candidates:
                break
            
            # Find best entry candidate
            best_entry = entry_candidates[0]
            
            # Check if rotation makes sense
            # Entry signal score must be at least min_rotation_improvement better
            improvement = best_entry.signal_score - rotation.conviction_score
            
            if improvement >= self.min_rotation_improvement:
                rotations.append((rotation.position.symbol, best_entry.symbol))
                entry_candidates.pop(0)
                
                logger.info(f"Rotation: {rotation.position.symbol} ({rotation.conviction_score:.0f}) -> {best_entry.symbol} ({best_entry.signal_score:.0f})")
        
        return rotations
    
    def generate_next_day_plan(
        self,
        portfolio: PortfolioState,
        eod_results: Optional[Dict[str, Any]] = None
    ) -> NextDayPlan:
        """
        Generate complete next-day trading plan.
        
        This is the main entry point for overnight analysis.
        """
        logger.info("=" * 60)
        logger.info("OVERNIGHT AGENT: Generating next-day plan")
        logger.info("=" * 60)
        
        # 0. Apply performance feedback
        feedback = eod_results or self._load_latest_eod_review()
        if feedback:
            self._apply_performance_feedback(feedback)
        
        plan = NextDayPlan()
        
        # 1. Analyze current portfolio
        analysis = self.analyze_portfolio(portfolio)
        
        # 2. Handle critical positions (exit regardless)
        for pos in analysis['critical']:
            plan.exit_at_open.append(pos.symbol)
            plan.exit_reasons[pos.symbol] = pos.risk_triggers
            plan.capital_from_exits += pos.current_price * pos.qty
            plan.notes.append(f"[!] {pos.symbol}: Critical exit ({pos.risk_triggers})")
        
        # 3. Identify rotation candidates
        rotation_candidates = analysis['rotation_candidates']
        if rotation_candidates:
            plan.notes.append(f"Found {len(rotation_candidates)} rotation candidates")
        
        # 4. Scan for entry candidates
        entry_candidates = self.scan_watchlist(max_candidates=10)
        if entry_candidates:
            plan.entry_candidates = entry_candidates
            plan.notes.append(f"Found {len(entry_candidates)} entry candidates")
        
        # 5. Create rotation plan
        rotations = self.create_rotation_plan(
            rotation_candidates,
            entry_candidates.copy(),
            portfolio.open_slots
        )
        plan.rotations = rotations
        
        # Add non-critical rotation exits
        for rotation in rotation_candidates:
            if rotation.position.symbol not in plan.exit_at_open:
                # Check if it's part of a rotation pair
                if any(r[0] == rotation.position.symbol for r in rotations):
                    plan.exit_at_open.append(rotation.position.symbol)
                    plan.exit_reasons[rotation.position.symbol] = rotation.reasons
                    plan.capital_from_exits += rotation.capital_to_free
        
        # 6. Identify ADD candidates
        for pos in analysis['add_candidates']:
            plan.add_candidates.append(pos.symbol)
            # Target 25-50% add based on conviction
            if pos.conviction_score >= 80:
                plan.add_targets[pos.symbol] = 0.50
            else:
                plan.add_targets[pos.symbol] = 0.25
        
        # 7. Calculate capital to deploy
        plan.capital_to_deploy = plan.capital_from_exits + portfolio.cash_available
        
        # 8. Save plan
        self.save_plan(plan)
        
        logger.info("\n" + plan.summary())
        
        return plan
    
    def save_plan(self, plan: NextDayPlan):
        """Save plan to disk."""
        filename = self.plan_dir / f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(plan.to_dict(), f, indent=2, default=str)
        logger.info(f"Plan saved to {filename}")
    
    def load_latest_plan(self) -> Optional[NextDayPlan]:
        """Load most recent plan."""
        plans = sorted(self.plan_dir.glob("plan_*.json"), reverse=True)
        if plans:
            with open(plans[0]) as f:
                data = json.load(f)
            # Reconstruct plan (simplified)
            plan = NextDayPlan()
            plan.exit_at_open = data.get('exit_at_open', [])
            plan.add_candidates = data.get('add_candidates', [])
            plan.notes = data.get('notes', [])
            return plan
        return None

    def _load_latest_eod_review(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recent EOD performance review results.
        """
        data_dir = Path("data")
        if not data_dir.exists():
            return None
            
        reviews = sorted(data_dir.glob("eod_review_*.json"), reverse=True)
        if reviews:
            try:
                with open(reviews[0], 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load latest EOD review: {e}")
        return None


# ============================================================
# STANDALONE RUNNER
# ============================================================

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info('overnight_agent.py loaded; use pytest/CLI for tests.')

