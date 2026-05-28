"""
Pattern Playbooks - Actionable trading strategies triggered by market patterns.

Each playbook defines detection criteria and specific actions to take when
patterns like short squeezes, V-bottom recoveries, and sector rotations occur.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


class PatternType(Enum):
    SHORT_SQUEEZE_SETUP = "short_squeeze_setup"
    V_BOTTOM_RECOVERY = "v_bottom_recovery"
    MOMENTUM_CONTINUATION = "momentum_continuation"
    SECTOR_ROTATION = "sector_rotation"
    DISTRIBUTION_WARNING = "distribution_warning"
    CAPITULATION_PANIC = "capitulation_panic"
    BREAKOUT_CONFIRMATION = "breakout_confirmation"


@dataclass
class Playbook:
    """A trading playbook with detection criteria and actions."""
    name: str
    pattern_type: PatternType
    detection_criteria: str
    confidence_threshold: float
    actions: List[str] = field(default_factory=list)
    scan_filters: Dict[str, Any] = field(default_factory=dict)
    position_sizing: Dict[str, Any] = field(default_factory=dict)
    exit_rules: Dict[str, Any] = field(default_factory=dict)
    max_positions: int = 10
    priority: int = 1  # Higher = execute first


# Pattern Playbooks Dictionary
PLAYBOOKS: Dict[str, Playbook] = {
    "short_squeeze_setup": Playbook(
        name="Short Squeeze Snapback",
        pattern_type=PatternType.SHORT_SQUEEZE_SETUP,
        detection_criteria="3+ down days -> breadth flip > 60%",
        confidence_threshold=0.75,
        actions=[
            "Scan for most beaten-down stocks (weekly_return < -5%)",
            "Filter: still has solid fundamentals (not broken company)",
            "Filter: volume surge on reversal day (>2x average)",
            "Enter with tight stops (1.5x ATR)",
            "Target: +2-3% (snapback, not long-term hold)",
            "Exit by end of next day if not profitable",
        ],
        scan_filters={
            "weekly_return_max": -5.0,
            "volume_surge_min": 2.0,
            "min_atr_percent": 3.0,
            "scan_type": "mean_reversion"
        },
        position_sizing={
            "multiplier": 0.8,
            "max_positions": 10,
            "concentration": "moderate"
        },
        exit_rules={
            "stop_multiplier": 1.5,
            "target_multiplier": 1.5,
            "time_stop_days": 2,
            "trim_at": [2.0, 3.0]
        },
        max_positions=10,
        priority=10
    ),
    
    "v_bottom_recovery": Playbook(
        name="V-Bottom Recovery",
        pattern_type=PatternType.V_BOTTOM_RECOVERY,
        detection_criteria="5%+ universe decline over 3-5 days -> high-volume reversal",
        confidence_threshold=0.70,
        actions=[
            "Aggressive buying: double normal position count",
            "Focus on quality names that sold off WITH the market",
            "Wider stops (2.5x ATR, reversal is volatile)",
            "Target: +5-8% (recovery typically overshoots)",
            "Hold 3-5 days minimum",
        ],
        scan_filters={
            "market_correlation_min": 0.7,
            "volume_surge_min": 1.5,
            "quality_filter": True,
            "scan_type": "mean_reversion"
        },
        position_sizing={
            "multiplier": 1.3,
            "max_positions": 30,
            "concentration": "high"
        },
        exit_rules={
            "stop_multiplier": 2.5,
            "target_multiplier": 3.0,
            "time_stop_days": 5,
            "trim_at": [4.0, 6.0, 8.0]
        },
        max_positions=30,
        priority=9
    ),
    
    "momentum_continuation": Playbook(
        name="Momentum Continuation",
        pattern_type=PatternType.MOMENTUM_CONTINUATION,
        detection_criteria="3+ up days with expanding breadth and volume",
        confidence_threshold=0.75,
        actions=[
            "Buy breakouts, ride momentum",
            "Use momentum scan only (not mean reversion)",
            "Hold 3-5 days minimum",
            "Trail stops aggressively",
        ],
        scan_filters={
            "sma5_slope_min": 0.1,
            "rsi_min": 50,
            "rsi_max": 75,
            "volume_trend": "increasing",
            "scan_type": "momentum_breakout"
        },
        position_sizing={
            "multiplier": 1.1,
            "max_positions": 25,
            "concentration": "moderate"
        },
        exit_rules={
            "stop_multiplier": 2.0,
            "target_multiplier": 3.5,
            "trailing_stop": True,
            "trim_at": [5.0, 10.0]
        },
        max_positions=25,
        priority=8
    ),
    
    "sector_rotation": Playbook(
        name="Sector Rotation",
        pattern_type=PatternType.SECTOR_ROTATION,
        detection_criteria="Sector spread > 5% (leaders vs laggards)",
        confidence_threshold=0.65,
        actions=[
            "Exit positions in lagging sectors",
            "Enter positions in leading sectors",
            "Use momentum scan only (not mean reversion)",
            "Hold period: 3-5 days",
        ],
        scan_filters={
            "sector_focus": "leading_only",
            "avoid_sectors": "lagging",
            "scan_type": "momentum_breakout"
        },
        position_sizing={
            "multiplier": 0.9,
            "max_positions": 20,
            "concentration": "moderate"
        },
        exit_rules={
            "stop_multiplier": 2.0,
            "target_multiplier": 2.5,
            "time_stop_days": 5,
            "trim_at": [4.0, 6.0]
        },
        max_positions=20,
        priority=7
    ),
    
    "distribution_warning": Playbook(
        name="Distribution Warning",
        pattern_type=PatternType.DISTRIBUTION_WARNING,
        detection_criteria="Breadth narrowing, volume on down moves increasing",
        confidence_threshold=0.70,
        actions=[
            "Tighten all stops to 1.5x ATR",
            "Take 25% profit on all winners > +3%",
            "No new entries until breadth improves",
            "Increase cash reserve to 40%",
        ],
        scan_filters={
            "new_entries": False,
            "scan_type": None
        },
        position_sizing={
            "multiplier": 0.6,
            "max_positions": 15,
            "concentration": "defensive"
        },
        exit_rules={
            "stop_multiplier": 1.5,
            "profit_take_immediate": True,
            "profit_take_threshold": 3.0,
            "profit_take_pct": 0.25
        },
        max_positions=15,
        priority=10  # High priority - defensive action
    ),
    
    "capitulation_panic": Playbook(
        name="Capitulation Panic",
        pattern_type=PatternType.CAPITULATION_PANIC,
        detection_criteria="Breadth < 30% with volume surge",
        confidence_threshold=0.80,
        actions=[
            "DO NOT ADD new positions",
            "Exit all losing positions immediately",
            "Hold only strong winners",
            "Increase cash reserve to 80%",
            "Wait for stabilization before re-entry",
        ],
        scan_filters={
            "new_entries": False,
            "scan_type": None
        },
        position_sizing={
            "multiplier": 0.0,
            "max_positions": 0,
            "concentration": "none"
        },
        exit_rules={
            "exit_all_losers": True,
            "preserve_cash": True
        },
        max_positions=0,
        priority=10
    ),
    
    "breakout_confirmation": Playbook(
        name="Breakout Confirmation",
        pattern_type=PatternType.BREAKOUT_CONFIRMATION,
        detection_criteria="Stock breaking above resistance with market support",
        confidence_threshold=0.70,
        actions=[
            "Enter on confirmed breakout",
            "Use standard position sizing",
            "Set stop below breakout level",
            "Target measured move",
        ],
        scan_filters={
            "breakout_above_resistance": True,
            "volume_confirmation": True,
            "market_breadth_min": 55,
            "scan_type": "momentum_breakout"
        },
        position_sizing={
            "multiplier": 1.0,
            "max_positions": 20,
            "concentration": "moderate"
        },
        exit_rules={
            "stop_below_breakout": True,
            "target_measured_move": True,
            "trim_at": [3.0, 6.0]
        },
        max_positions=20,
        priority=6
    ),
}


def get_playbook(pattern_name: str) -> Optional[Playbook]:
    """
    Get a playbook by name.
    
    Args:
        pattern_name: Name of the pattern/playbook
        
    Returns:
        Playbook if found, None otherwise
    """
    return PLAYBOOKS.get(pattern_name)


def get_playbooks_for_regime(regime: str) -> List[Playbook]:
    """
    Get relevant playbooks for a market regime.
    
    Args:
        regime: Market regime string (e.g., "short_squeeze", "selloff")
        
    Returns:
        List of relevant playbooks
    """
    regime_playbook_map = {
        "short_squeeze": ["short_squeeze_setup"],
        "recovery": ["v_bottom_recovery", "momentum_continuation"],
        "strong_rally": ["momentum_continuation", "breakout_confirmation"],
        "rotation": ["sector_rotation"],
        "distribution": ["distribution_warning"],
        "selloff": ["distribution_warning"],
        "capitulation": ["capitulation_panic"],
        "crash": ["capitulation_panic"],
    }
    
    playbook_names = regime_playbook_map.get(regime, [])
    return [PLAYBOOKS[name] for name in playbook_names if name in PLAYBOOKS]


def evaluate_pattern_match(pattern_name: str, context: Dict[str, Any]) -> float:
    """
    Evaluate how well current market context matches a pattern.
    
    Args:
        pattern_name: Name of pattern to evaluate
        context: Dict with current market metrics
        
    Returns:
        Confidence score 0-1
    """
    playbook = get_playbook(pattern_name)
    if not playbook:
        return 0.0
    
    confidence = 0.0
    
    if pattern_name == "short_squeeze_setup":
        # Check for consecutive down days and breadth reversal
        down_days = context.get('consecutive_down_days', 0)
        breadth = context.get('breadth_pct', 50)
        
        if down_days >= 3 and breadth > 60:
            confidence = min(0.95, 0.7 + (down_days - 3) * 0.05 + (breadth - 60) * 0.01)
    
    elif pattern_name == "v_bottom_recovery":
        # Check for sharp decline followed by reversal
        avg_return_5d = context.get('avg_return_5d', 0)
        breadth = context.get('breadth_pct', 50)
        
        if avg_return_5d < -5 and breadth > 55:
            confidence = min(0.95, 0.7 + abs(avg_return_5d) * 0.02 + (breadth - 55) * 0.01)
    
    elif pattern_name == "momentum_continuation":
        # Check for sustained up move
        up_days = context.get('consecutive_up_days', 0)
        breadth = context.get('breadth_pct', 50)
        volume = context.get('volume_trend', 'stable')
        
        if up_days >= 3 and breadth > 55 and volume == 'increasing':
            confidence = min(0.95, 0.75 + (up_days - 3) * 0.05)
    
    elif pattern_name == "sector_rotation":
        # Check for high dispersion
        dispersion = context.get('dispersion', 0)
        
        if dispersion > 3.0:
            confidence = min(0.90, 0.65 + (dispersion - 3.0) * 0.05)
    
    elif pattern_name == "distribution_warning":
        # Check for narrowing breadth during uptrend
        breadth_trend = context.get('breadth_trend', 'stable')
        volume = context.get('volume_trend', 'stable')
        
        if breadth_trend == 'deteriorating' and volume == 'increasing':
            confidence = 0.80
    
    return confidence


def get_active_playbooks(context: Dict[str, Any], min_confidence: float = 0.65) -> List[Dict[str, Any]]:
    """
    Get all playbooks that match current market context.
    
    Args:
        context: Current market metrics dict
        min_confidence: Minimum confidence threshold
        
    Returns:
        List of active playbooks with confidence scores
    """
    active = []
    
    for pattern_name, playbook in PLAYBOOKS.items():
        confidence = evaluate_pattern_match(pattern_name, context)
        
        if confidence >= min_confidence and confidence >= playbook.confidence_threshold:
            active.append({
                "playbook": playbook,
                "confidence": confidence,
                "pattern_name": pattern_name
            })
    
    # Sort by priority (highest first), then confidence
    active.sort(key=lambda x: (-x["playbook"].priority, -x["confidence"]))
    
    return active


def format_playbook_for_execution(playbook: Playbook, confidence: float) -> Dict[str, Any]:
    """
    Format a playbook into actionable execution parameters.
    
    Args:
        playbook: The playbook to format
        confidence: Confidence score
        
    Returns:
        Dict with execution parameters
    """
    return {
        "pattern": playbook.name,
        "pattern_type": playbook.pattern_type.value,
        "confidence": confidence,
        "actions": playbook.actions,
        "scan_filters": playbook.scan_filters,
        "position_sizing": playbook.position_sizing,
        "exit_rules": playbook.exit_rules,
        "max_positions": playbook.max_positions,
        "priority": playbook.priority
    }
