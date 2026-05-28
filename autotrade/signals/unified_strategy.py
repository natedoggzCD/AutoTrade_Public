"""
UNIFIED TRADING STRATEGY
========================
Core strategy module used by ALL trading components.

Based on backtested results (min-score 10):
- +1542% Total P&L (vs +1015% baseline = +52% better)
- +0.53% avg trade (vs +0.35% baseline = +51% better)
- 1.66 Profit Factor (vs 1.47 baseline)
- 793 R2 hits at +3.06% each

Strategy:
- ENTRY: Long on bullish signals filtered by lessons (min-score 10)
- TARGET 1: R1 (resistance level) - Take 50% profit
- TARGET 2: R2 (if within 3 ATR) - Take remaining 50%
- STOP: S1 break (support level)

This module provides:
1. Lesson scoring for entry filtering
2. Stop/target calculation using actual S/R levels
3. Exit logic using multi-level approach
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

from config.config_loader import get_logging_config, get_strategy_config
from autotrade.utils.logging_utils import configure_logging

logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = Path(os.environ.get("AUTOTRADE_ROOT", Path(__file__).resolve().parents[2]))
LESSONS_FILE = PROJECT_DIR / 'logs' / 'lessons_learned.json'

_STRATEGY_CONFIG = get_strategy_config()

# Strategy Constants (from backtest optimization)
MIN_LESSON_SCORE = _STRATEGY_CONFIG.min_lesson_score
R2_MAX_ATR = _STRATEGY_CONFIG.r2_max_atr
PARTIAL_AT_R1 = _STRATEGY_CONFIG.partial_at_r1
MIN_BULLISH = _STRATEGY_CONFIG.min_bullish
S1_STOP_BUFFER_PCT = _STRATEGY_CONFIG.s1_stop_buffer_pct
FALLBACK_STOP_ATR = _STRATEGY_CONFIG.fallback_stop_atr
FALLBACK_TARGET_ATR = _STRATEGY_CONFIG.fallback_target_atr


def _configure_logging() -> None:
    """Configure JSON logging based on config."""
    try:
        log_cfg = get_logging_config()
        configure_logging(
            PROJECT_DIR / "logs",
            level=log_cfg.level,
            filename=log_cfg.json_filename,
            max_bytes=log_cfg.max_bytes,
            backup_count=log_cfg.backup_count,
            console=log_cfg.console,
        )
    except Exception:
        logging.basicConfig(level=logging.INFO)


@dataclass
class TradeLevels:
    """Complete level structure for a trade."""
    entry_price: float
    
    # Support levels
    s1_price: float
    s1_strength: float
    
    # Resistance levels
    r1_price: float
    r1_strength: float
    
    # Optional levels
    s2_price: Optional[float] = None
    r2_price: Optional[float] = None
    
    # ATR for context
    atr: float = 0.0
    
    # Distances (calculated)
    s1_dist_pct: float = 0.0
    r1_dist_pct: float = 0.0
    r2_dist_pct: float = 0.0
    r2_in_range: bool = False
    
    # Risk/reward
    stop_price: float = 0.0
    target_price: float = 0.0
    risk_reward: float = 0.0
    
    def __post_init__(self):
        """Calculate derived fields — defaults are informational only."""
        if self.entry_price > 0:
            # Distances (informational only, not used for stop/target)
            if self.s1_price:
                self.s1_dist_pct = (self.entry_price - self.s1_price) / self.entry_price * 100
            if self.r1_price:
                self.r1_dist_pct = (self.r1_price - self.entry_price) / self.entry_price * 100
            if self.r2_price:
                self.r2_dist_pct = (self.r2_price - self.entry_price) / self.entry_price * 100

            # R2 in range check (informational only)
            if self.r2_price and self.atr > 0:
                self.r2_in_range = (self.r2_price - self.entry_price) <= (R2_MAX_ATR * self.atr)

            # Risk/Reward defaults if not already set by calculate_levels
            if self.stop_price <= 0:
                if self.atr > 0:
                    self.stop_price = self.entry_price - (self.atr * FALLBACK_STOP_ATR)
                else:
                    self.stop_price = self.entry_price * 0.95
            
            if self.target_price <= 0:
                if self.atr > 0:
                    self.target_price = self.entry_price + (self.atr * FALLBACK_TARGET_ATR)
                else:
                    self.target_price = self.entry_price * 1.08

            # Risk/Reward
            risk = self.entry_price - self.stop_price
            reward = self.target_price - self.entry_price
            if risk > 0:
                self.risk_reward = reward / risk
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExitSignal:
    """Exit signal with reason and price."""
    action: str  # 'exit', 'partial', 'hold', 'stop'
    reason: str
    exit_price: float
    pnl_pct: float
    partial_pct: float = 0.0  # What % of position to exit (0.5 = 50%)
    
    def to_dict(self) -> Dict:
        return asdict(self)


class UnifiedStrategy:
    """
    Unified trading strategy using multi-level approach.
    
    This is THE source of truth for:
    - Lesson scoring
    - Entry filtering
    - Stop/target calculation
    - Exit logic
    """
    
    def __init__(self, min_score: float = MIN_LESSON_SCORE):
        _configure_logging()
        self.min_score = min_score
        self.lessons = self._load_lessons()
        
        avoid_count = len(self.lessons.get('AVOID', []))
        prefer_count = len(self.lessons.get('PREFER', []))
        logger.info(f"UnifiedStrategy initialized: {avoid_count} AVOID, {prefer_count} PREFER, min_score={min_score}")
    
    def _load_lessons(self) -> Dict:
        """Load lesson rules from file."""
        if not LESSONS_FILE.exists():
            logger.warning(f"Lessons file not found: {LESSONS_FILE}")
            return {'AVOID': [], 'PREFER': []}
        
        with open(LESSONS_FILE) as f:
            data = json.load(f)
        
        avoid_rules = []
        prefer_rules = []
        
        # Format 1: AVOID/PREFER keys
        if 'AVOID' in data:
            avoid_rules = data.get('AVOID', [])
        if 'PREFER' in data:
            prefer_rules = data.get('PREFER', [])
        
        # Format 2: prefer_conditions/avoid_conditions keys
        if 'prefer_conditions' in data:
            for c in data['prefer_conditions']:
                prefer_rules.append({
                    'field': c.get('field'),
                    'operator': c.get('op'),
                    'threshold': c.get('value'),
                    'weight': 10,
                })
        if 'avoid_conditions' in data:
            for c in data['avoid_conditions']:
                avoid_rules.append({
                    'field': c.get('field'),
                    'operator': c.get('op'),
                    'threshold': c.get('value'),
                    'weight': 10,
                })
        
        return {'AVOID': avoid_rules, 'PREFER': prefer_rules}
    
    def score_entry(self, candidate: Dict) -> Tuple[float, List[str]]:
        """
        Score an entry candidate using learned lessons.
        
        Args:
            candidate: Dict with entry-time fields:
                - bullish_score, s1_strength, r1_strength
                - atr_percent, distance_to_s1_pct, distance_to_r1_pct
                - spread_pct, momentum_score (optional)
        
        Returns:
            (score, list of matched rules)
        """
        score = 0.0
        matched_rules = []
        
        # Map candidate fields
        entry_fields = {
            'bullish_score': candidate.get('bullish_score', 0),
            's1_strength': candidate.get('s1_strength', 0),
            'r1_strength': candidate.get('r1_strength', 0),
            'atr_percent': candidate.get('atr_percent', 0),
            'distance_to_s1_pct': candidate.get('distance_to_s1_pct', 0),
            'distance_to_r1_pct': candidate.get('distance_to_r1_pct', 0),
            'spread_pct': candidate.get('spread_pct'),
            'momentum_score': candidate.get('momentum_score'),
            'poc_distance_pct': candidate.get('poc_distance_pct'),
            'in_value_area': candidate.get('in_value_area'),
        }
        
        # Check AVOID rules
        for rule in self.lessons.get('AVOID', []):
            field = rule.get('field', '')
            if field not in entry_fields:
                continue
            val = entry_fields.get(field)
            if val is None:
                continue
            
            op = rule.get('operator', '')
            threshold = rule.get('threshold', 0)
            weight = rule.get('weight', 10)
            
            matches = False
            if op == '<' and val < threshold:
                matches = True
            elif op == '>' and val > threshold:
                matches = True
            elif op == '<=' and val <= threshold:
                matches = True
            elif op == '>=' and val >= threshold:
                matches = True
            
            if matches:
                score -= weight
                matched_rules.append(f"AVOID: {field} {op} {threshold} (weight -{weight})")
        
        # Check PREFER rules
        for rule in self.lessons.get('PREFER', []):
            field = rule.get('field', '')
            if field not in entry_fields:
                continue
            val = entry_fields.get(field)
            if val is None:
                continue
            
            op = rule.get('operator', '')
            threshold = rule.get('threshold', 0)
            weight = rule.get('weight', 10)
            
            matches = False
            if op == '<' and val < threshold:
                matches = True
            elif op == '>' and val > threshold:
                matches = True
            elif op == '<=' and val <= threshold:
                matches = True
            elif op == '>=' and val >= threshold:
                matches = True
            
            if matches:
                score += weight
                matched_rules.append(f"PREFER: {field} {op} {threshold} (weight +{weight})")
        
        return score, matched_rules
    
    def passes_filter(self, candidate: Dict) -> Tuple[bool, float, List[str]]:
        """
        Check if candidate passes lesson filter.
        
        Returns:
            (passes, score, matched_rules)
        """
        score, rules = self.score_entry(candidate)
        passes = score >= self.min_score
        return passes, score, rules
    
    def calculate_levels(
        self,
        entry_price: float,
        s1_price: float = 0,
        s1_strength: float = 0,
        r1_price: float = 0,
        r1_strength: float = 0,
        s2_price: float = 0,
        r2_price: float = 0,
        atr: float = 0,
        strategy_params: Optional[Dict] = None,
        bid: float = 0,
        ask: float = 0,
    ) -> TradeLevels:
        """
        Calculate microstructure-aware trade levels.

        This uses RiskGate logic to calculate volatility-and-spread-aware stops.
        Prefers strategy_params when present.
        """
        from autotrade.risk.risk_gate import get_risk_gate
        risk_gate = get_risk_gate()
        
        # Calculate levels via microstructure-aware RiskGate logic (Phase 1 Risk track)
        risk_levels = risk_gate.compute_entry_levels(
            entry_price=entry_price,
            atr_14=atr,
            s1_price=s1_price,
            s1_strength=s1_strength,
            r1_price=r1_price,
            r1_strength=r1_strength,
            strategy_params=strategy_params,
            bid=bid,
            ask=ask
        )

        # Handle missing levels with defaults if compute_entry_levels failed
        if not s1_price and atr > 0:
            s1_price = entry_price - (atr * 1.5)
        if not r1_price and atr > 0:
            r1_price = entry_price + (atr * 2.0)
        if not s2_price and s1_price > 0:
            s2_price = s1_price - (atr * 1.0 if atr else s1_price * 0.03)
        if not r2_price and r1_price > 0:
            r2_price = r1_price + (atr * 1.0 if atr else r1_price * 0.03)

        levels = TradeLevels(
            entry_price=entry_price,
            s1_price=s1_price,
            s1_strength=s1_strength,
            r1_price=r1_price,
            r1_strength=r1_strength,
            s2_price=s2_price,
            r2_price=r2_price,
            atr=atr,
            stop_price=risk_levels["stop_price"],
            target_price=risk_levels["target_price"],
            risk_reward=risk_levels["risk_reward"]
        )

        return levels
    
    def evaluate_exit(
        self,
        levels: TradeLevels,
        current_price: float,
        current_high: float,
        current_low: float,
        s1_tested: bool = False,
        s1_held: bool = True,
        r1_tested: bool = False,
        r1_broken: bool = False,
    ) -> ExitSignal:
        """
        Evaluate exit using multi-level logic.
        
        Priority:
        1. S1 BROKEN â†’ Hard stop at S1
        2. R2 HIT â†’ Full profit at R2
        3. R1 BROKEN â†’ Ride continuation
        4. R1 TESTED but rejected â†’ Partial at R1
        5. Otherwise â†’ HOLD
        """
        entry = levels.entry_price
        pnl_pct = (current_price - entry) / entry * 100
        
        # Priority 1: S1 BROKEN = hard stop
        if s1_tested and not s1_held:
            return ExitSignal(
                action='stop',
                reason='S1_BROKEN - Support failed',
                exit_price=levels.s1_price,
                pnl_pct=(levels.s1_price - entry) / entry * 100,
                partial_pct=1.0  # Full exit
            )
        
        # Priority 2: R2 HIT (if in range)
        if levels.r2_in_range and levels.r2_price and current_high >= levels.r2_price:
            return ExitSignal(
                action='exit',
                reason='R2_HIT - Full target reached',
                exit_price=levels.r2_price,
                pnl_pct=(levels.r2_price - entry) / entry * 100,
                partial_pct=1.0  # Full exit
            )
        
        # Priority 3: R1 BROKEN (continuation)
        if r1_tested and r1_broken:
            return ExitSignal(
                action='hold',
                reason='R1_BROKEN - Riding continuation to R2',
                exit_price=current_price,
                pnl_pct=pnl_pct,
                partial_pct=0.0  # Keep holding
            )
        
        # Priority 4: R1 TESTED but rejected
        if r1_tested and not r1_broken and current_high >= levels.r1_price * 0.998:
            return ExitSignal(
                action='partial',
                reason='R1_PARTIAL - Take partial at R1',
                exit_price=levels.r1_price,
                pnl_pct=(levels.r1_price - entry) / entry * 100,
                partial_pct=PARTIAL_AT_R1  # 50% exit
            )
        
        # Default: HOLD
        return ExitSignal(
            action='hold',
            reason='Levels not triggered - continue holding',
            exit_price=current_price,
            pnl_pct=pnl_pct,
            partial_pct=0.0
        )
    
    def get_entry_plan(self, levels: TradeLevels) -> Dict:
        """
        Generate entry plan with stops and targets.
        
        Returns dict suitable for order placement.
        """
        return {
            'entry_price': levels.entry_price,
            'stop_price': levels.stop_price,
            'target_1': levels.r1_price,
            'target_1_pct': PARTIAL_AT_R1,
            'target_2': levels.r2_price if levels.r2_in_range else levels.r1_price,
            'target_2_pct': 1.0 - PARTIAL_AT_R1,
            'risk_reward': levels.risk_reward,
            'r2_in_range': levels.r2_in_range,
            's1_dist_pct': levels.s1_dist_pct,
            'r1_dist_pct': levels.r1_dist_pct,
            'r2_dist_pct': levels.r2_dist_pct if levels.r2_price else 0,
        }


# Convenience functions for other modules
_strategy_instance = None

def get_strategy(min_score: float = MIN_LESSON_SCORE) -> UnifiedStrategy:
    """Get singleton strategy instance."""
    global _strategy_instance
    if _strategy_instance is None or _strategy_instance.min_score != min_score:
        _strategy_instance = UnifiedStrategy(min_score)
    return _strategy_instance


def score_candidate(candidate: Dict, min_score: float = MIN_LESSON_SCORE) -> Tuple[bool, float, List[str]]:
    """Score a candidate and check if it passes filter."""
    strategy = get_strategy(min_score)
    return strategy.passes_filter(candidate)


def get_trade_levels(
    entry_price: float,
    s1_price: float = 0,
    s1_strength: float = 0,
    r1_price: float = 0,
    r1_strength: float = 0,
    atr: float = 0,
    strategy_params: Optional[Dict] = None,
    **kwargs
) -> TradeLevels:
    """Calculate trade levels for an entry."""
    strategy = get_strategy()
    return strategy.calculate_levels(
        entry_price=entry_price,
        s1_price=s1_price,
        s1_strength=s1_strength,
        r1_price=r1_price,
        r1_strength=r1_strength,
        atr=atr,
        strategy_params=strategy_params,
        **kwargs
    )

