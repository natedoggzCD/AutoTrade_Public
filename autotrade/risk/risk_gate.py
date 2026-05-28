"""
Risk Gate Module - Deterministic FAST PATH
===========================================
Runs BEFORE any LLM/agent calls. If triggered, skips the expensive agent pipeline.

Rules checked (in order):
1. Hard stop breach -> EXIT
2. ATR stop breach -> EXIT
3. Trailing ATR stop -> EXIT
4. Drawdown trailing trim -> TRIM/EXIT
5. Trim threshold breach -> TRIM
6. Time stop (flat too long) -> TRIM or EXIT
7. Profit taking levels -> TRIM
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml

from config.config_loader import get_risk_gate_config, get_logging_config
from autotrade.utils.logging_utils import configure_logging
from autotrade.utils.strategy_params import sanitize_strategy_params

logger = logging.getLogger(__name__)
PROJECT_DIR = Path(
    os.environ.get("AUTOTRADE_ROOT", Path(__file__).resolve().parents[2])
)

# Minimum position value - trims that would leave less than this should either:
# 1. Do nothing (skip the trim)
# 2. Do a full exit instead
MIN_POSITION_VALUE = 100.0  # $100 minimum to prevent stub positions


class RiskAction(Enum):
    """Possible risk gate actions."""

    NONE = "none"  # No fast path trigger - run full agent pipeline
    EXIT = "exit"  # Exit entire position immediately
    TRIM = "trim"  # Trim position by configured fraction
    PROFIT_TAKE = "profit_take"  # Take partial profits
    ADD = "add"  # Scale into winning position


class HoldPhase(Enum):
    """Legacy hold phases retained for serialized risk-plan compatibility."""

    LOCKED = "locked"
    UNLOCKING = "unlocking"
    UNLOCKED = "unlocked"
    CRITICAL = "critical"


@dataclass
class ActionPlan:
    """Result of risk gate evaluation."""

    action: RiskAction
    size_delta: float  # Negative = reduce, 0 = hold, positive = add
    confidence: float  # 0-1
    reasons: List[str] = field(default_factory=list)
    triggered_rules: List[str] = field(default_factory=list)
    skip_agents: bool = False  # If True, skip LLM pipeline
    hold_phase: HoldPhase = HoldPhase.UNLOCKED
    pdt_burn_required: bool = False
    conviction_score: float = 50.0  # For ADD decisions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "size_delta": self.size_delta,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "triggered_rules": self.triggered_rules,
            "skip_agents": self.skip_agents,
            "hold_phase": self.hold_phase.value,
            "pdt_burn_required": self.pdt_burn_required,
            "conviction_score": self.conviction_score,
        }


@dataclass
class RiskGateConfig:
    """Configuration for risk gate thresholds."""

    # Hard stop
    hard_stop_pct: float = -8.0

    # Trim threshold
    trim_threshold_pct: float = -5.0
    trim_fraction: float = 0.5

    # ATR-based stop
    atr_k: float = 2.0
    atr_trail_k: float = 1.5
    target_atr: float = 3.0
    hard_atr_stop_k: float = 2.2

    # Dynamic stop distance
    bid_ask_spread_k: float = 2.0
    min_stop_distance_pct: float = 0.5

    # ATR regime sizing
    low_atr_percent_threshold: float = 2.0
    high_atr_percent_threshold: float = 5.0
    low_atr_size_multiplier: float = 1.2
    high_atr_size_multiplier: float = 0.6

    # Profit guard
    profit_guard_activation_pct: float = 4.0
    profit_guard_trail_atr_k: float = 1.0

    # Time stop
    time_stop_minutes: float = 240
    time_stop_flat_threshold_pct: float = 1.0
    multi_day_time_stop_days: float = 3.0
    multi_day_time_stop_min_gain_pct: float = 0.5

    # Profit taking
    profit_take_levels: List[Dict[str, float]] = field(
        default_factory=lambda: [
            {"pct": 5.0, "trim": 0.25},
            {"pct": 10.0, "trim": 0.25},
            {"pct": 15.0, "trim": 0.50},
        ]
    )

    # Drawdown-based trailing trims (from high since entry)
    drawdown_trim_levels: List[Dict[str, float]] = field(
        default_factory=lambda: [
            {"drawdown_pct": 3.0, "trim": 0.33},
            {"drawdown_pct": 6.0, "trim": 0.45},
            {"drawdown_pct": 10.0, "trim": 1.0},
        ]
    )
    drawdown_trail_atr_k: float = 0.5

    # Hold period constraints
    min_hold_days: float = 0.0  # Minimum calendar days before non-critical trim/exit (0 = no hold guard)
    min_hold_minutes_for_trim: int = 120  # Legacy intraday guard (minutes)

    # Legacy PDT compatibility fields. Runtime PDT gating is disabled.
    pdt_unlock_hours: float = 24.0
    pdt_buffer_trades: int = 2

    # ADD thresholds (scaling into winners)
    add_profit_threshold_pct: float = 4.0  # Min profit to consider ADD
    add_conviction_min: float = 65.0  # Min conviction score to ADD
    add_max_position_pct: float = 0.50  # Max 50% position increase

    # Minimum hold time before allowing trims (does not affect hard stops)
    min_hold_minutes_for_trim: int = 120

    @classmethod
    def from_yaml(cls, path: Path) -> "RiskGateConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        rg = data.get("risk_gate", {})
        return cls(
            hard_stop_pct=rg.get("hard_stop_pct", -8.0),
            trim_threshold_pct=rg.get("trim_threshold_pct", -5.0),
            trim_fraction=rg.get("trim_fraction", 0.5),
            atr_k=rg.get("atr_k", 2.0),
            atr_trail_k=rg.get("atr_trail_k", 1.5),
            target_atr=rg.get("target_atr", 3.0),
            hard_atr_stop_k=rg.get("hard_atr_stop_k", 2.2),
            bid_ask_spread_k=rg.get("bid_ask_spread_k", 2.0),
            min_stop_distance_pct=rg.get("min_stop_distance_pct", 0.5),
            low_atr_percent_threshold=rg.get("low_atr_percent_threshold", 2.0),
            high_atr_percent_threshold=rg.get("high_atr_percent_threshold", 5.0),
            low_atr_size_multiplier=rg.get("low_atr_size_multiplier", 1.2),
            high_atr_size_multiplier=rg.get("high_atr_size_multiplier", 0.6),
            profit_guard_activation_pct=rg.get("profit_guard_activation_pct", 4.0),
            profit_guard_trail_atr_k=rg.get("profit_guard_trail_atr_k", 1.0),
            time_stop_minutes=rg.get("time_stop_minutes", 240),
            time_stop_flat_threshold_pct=rg.get("time_stop_flat_threshold_pct", 1.0),
            multi_day_time_stop_days=rg.get("multi_day_time_stop_days", 3.0),
            multi_day_time_stop_min_gain_pct=rg.get(
                "multi_day_time_stop_min_gain_pct", 0.5
            ),
            profit_take_levels=rg.get(
                "profit_take_levels",
                [
                    {"pct": 5.0, "trim": 0.25},
                    {"pct": 10.0, "trim": 0.25},
                    {"pct": 15.0, "trim": 0.50},
                ],
            ),
            drawdown_trim_levels=rg.get(
                "drawdown_trim_levels",
                [
                    {"drawdown_pct": 3.0, "trim": 0.33},
                    {"drawdown_pct": 6.0, "trim": 0.45},
                    {"drawdown_pct": 10.0, "trim": 1.0},
                ],
            ),
            drawdown_trail_atr_k=rg.get("drawdown_trail_atr_k", 0.5),
            pdt_unlock_hours=rg.get("pdt_unlock_hours", 24.0),
            pdt_buffer_trades=rg.get("pdt_buffer_trades", 2),
            add_profit_threshold_pct=rg.get("add_profit_threshold_pct", 4.0),
            add_conviction_min=rg.get("add_conviction_min", 65.0),
            add_max_position_pct=rg.get("add_max_position_pct", 0.50),
            min_hold_minutes_for_trim=rg.get("min_hold_minutes_for_trim", 120),
        )

    @classmethod
    def from_trading_config(cls) -> "RiskGateConfig":
        """Load config from centralized config loader."""
        cfg = get_risk_gate_config()
        return cls(
            hard_stop_pct=cfg.hard_stop_pct,
            trim_threshold_pct=cfg.trim_threshold_pct,
            trim_fraction=cfg.trim_fraction,
            atr_k=cfg.atr_k,
            atr_trail_k=cfg.atr_trail_k,
            target_atr=getattr(cfg, "target_atr", 3.0),
            hard_atr_stop_k=getattr(cfg, "hard_atr_stop_k", 2.2),
            bid_ask_spread_k=getattr(cfg, "bid_ask_spread_k", 2.0),
            min_stop_distance_pct=getattr(cfg, "min_stop_distance_pct", 0.5),
            low_atr_percent_threshold=getattr(cfg, "low_atr_percent_threshold", 2.0),
            high_atr_percent_threshold=getattr(cfg, "high_atr_percent_threshold", 5.0),
            low_atr_size_multiplier=getattr(cfg, "low_atr_size_multiplier", 1.2),
            high_atr_size_multiplier=getattr(cfg, "high_atr_size_multiplier", 0.6),
            profit_guard_activation_pct=getattr(
                cfg, "profit_guard_activation_pct", 4.0
            ),
            profit_guard_trail_atr_k=getattr(cfg, "profit_guard_trail_atr_k", 1.0),
            time_stop_minutes=cfg.time_stop_minutes,
            time_stop_flat_threshold_pct=cfg.time_stop_flat_threshold_pct,
            multi_day_time_stop_days=getattr(cfg, "multi_day_time_stop_days", 3.0),
            multi_day_time_stop_min_gain_pct=getattr(
                cfg, "multi_day_time_stop_min_gain_pct", 0.5
            ),
            profit_take_levels=cfg.profit_take_levels,
            drawdown_trim_levels=getattr(
                cfg,
                "drawdown_trim_levels",
                [
                    {"drawdown_pct": 3.0, "trim": 0.33},
                    {"drawdown_pct": 6.0, "trim": 0.45},
                    {"drawdown_pct": 10.0, "trim": 1.0},
                ],
            ),
            drawdown_trail_atr_k=getattr(cfg, "drawdown_trail_atr_k", 0.5),
            pdt_unlock_hours=cfg.pdt_unlock_hours,
            pdt_buffer_trades=cfg.pdt_buffer_trades,
            add_profit_threshold_pct=cfg.add_profit_threshold_pct,
            add_conviction_min=cfg.add_conviction_min,
            add_max_position_pct=cfg.add_max_position_pct,
            min_hold_minutes_for_trim=getattr(cfg, "min_hold_minutes_for_trim", 120),
        )


class RiskGate:
    """
    Deterministic risk gate that runs BEFORE any LLM/agent calls.

    This is the FAST PATH - if any rule triggers, we skip the expensive
    agent pipeline and act immediately based on hard rules.
    """

    def __init__(self, config: Optional[RiskGateConfig] = None):
        self.config = config or RiskGateConfig()
        self._profit_levels_hit: Dict[
            str, List[float]
        ] = {}  # Track which profit levels already hit per symbol
        self._drawdown_levels_hit: Dict[str, List[float]] = {}

    def _normalize_drawdown_levels(self) -> List[Dict[str, float]]:
        levels = []
        for level in self.config.drawdown_trim_levels or []:
            try:
                drawdown_pct = float(level.get("drawdown_pct", level.get("pct", 0.0)))
                trim_frac = float(level.get("trim", level.get("trim_fraction", 0.0)))
            except Exception:
                continue
            if drawdown_pct <= 0 or trim_frac <= 0:
                continue
            levels.append({"drawdown_pct": drawdown_pct, "trim": trim_frac})
        levels.sort(key=lambda x: x["drawdown_pct"])
        return levels

    def _evaluate_drawdown_trims(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        qty: int,
        atr_pct: float,
        high_since_entry: Optional[float],
    ) -> Optional[ActionPlan]:
        levels = self._normalize_drawdown_levels()
        if not levels:
            return None
        if current_price <= 0:
            return None
        high_ref = float(high_since_entry or 0.0)
        if high_ref <= 0:
            high_ref = max(float(entry_price or 0.0), float(current_price))
        else:
            high_ref = max(high_ref, float(current_price))
        if high_ref <= 0:
            return None

        drawdown_pct = ((high_ref - current_price) / high_ref) * 100.0
        atr_adj = max(0.0, float(atr_pct or 0.0)) * float(
            self.config.drawdown_trail_atr_k
        )

        triggered_level = None
        for level in levels:
            trigger_pct = level["drawdown_pct"] + atr_adj
            if drawdown_pct >= trigger_pct:
                triggered_level = level

        if not triggered_level:
            return None

        level_pct = float(triggered_level["drawdown_pct"])
        already_hit = self._drawdown_levels_hit.get(symbol, [])
        if level_pct in already_hit:
            return None

        newly_hit = [
            lvl["drawdown_pct"] for lvl in levels if lvl["drawdown_pct"] <= level_pct
        ]
        self._drawdown_levels_hit[symbol] = sorted(set(already_hit + newly_hit))

        trim_frac = float(triggered_level["trim"])
        trigger_pct = level_pct + atr_adj
        if trim_frac >= 1.0:
            reasons = [
                f"DRAWDOWN EXIT: {drawdown_pct:.1f}% off high {high_ref:.2f} "
                f">= {trigger_pct:.1f}% (base {level_pct:.1f}%, atr_adj {atr_adj:.1f}%)"
            ]
            return ActionPlan(
                action=RiskAction.EXIT,
                size_delta=-qty,
                confidence=0.90,
                reasons=reasons,
                triggered_rules=[f"drawdown_trim_{level_pct:g}"],
                skip_agents=True,
            )

        trim_qty = int(qty * trim_frac)
        remaining_qty = qty - trim_qty
        remaining_value = remaining_qty * current_price
        if remaining_value < MIN_POSITION_VALUE and remaining_qty > 0:
            reasons = [
                f"DRAWDOWN TRIM->EXIT: {drawdown_pct:.1f}% off high {high_ref:.2f} "
                f">= {trigger_pct:.1f}% (base {level_pct:.1f}%, atr_adj {atr_adj:.1f}%)"
            ]
            return ActionPlan(
                action=RiskAction.EXIT,
                size_delta=-qty,
                confidence=0.85,
                reasons=reasons,
                triggered_rules=[f"drawdown_trim_{level_pct:g}"],
                skip_agents=True,
            )

        if trim_qty > 0:
            reasons = [
                f"DRAWDOWN TRIM: {drawdown_pct:.1f}% off high {high_ref:.2f} "
                f">= {trigger_pct:.1f}% (base {level_pct:.1f}%, atr_adj {atr_adj:.1f}%), "
                f"trimming {trim_frac * 100:.0f}%"
            ]
            return ActionPlan(
                action=RiskAction.TRIM,
                size_delta=-trim_qty,
                confidence=0.85,
                reasons=reasons,
                triggered_rules=[f"drawdown_trim_{level_pct:g}"],
                skip_agents=True,
            )
        return None

    def get_position_size_multiplier_for_atr(
        self, atr_percent: float
    ) -> tuple[float, str]:
        """
        Return ATR-regime size multiplier.

        LOW ATR  -> larger size (default 1.2x)
        HIGH ATR -> defensive size (default 0.6x)
        """
        try:
            atr_pct = float(atr_percent or 0.0)
        except Exception:
            atr_pct = 0.0

        low_threshold = float(self.config.low_atr_percent_threshold)
        high_threshold = float(self.config.high_atr_percent_threshold)
        if atr_pct > 0 and atr_pct <= low_threshold:
            return float(self.config.low_atr_size_multiplier), "low_atr"
        if atr_pct >= high_threshold:
            return float(self.config.high_atr_size_multiplier), "high_atr"
        return 1.0, "normal_atr"

    def compute_entry_levels(
        self,
        entry_price: float,
        atr_14: float,
        s1_price: float = 0.0,
        s1_strength: float = 0.0,
        r1_price: float = 0.0,
        r1_strength: float = 0.0,
        min_rr: float = 2.0,
        strategy_params: Optional[Dict] = None,
        bid: float = 0.0,
        ask: float = 0.0,
    ) -> Dict[str, float]:
        """
        Compute microstructure-aware stop/target levels.

        When strategy_params are provided (from validated Strategy Lab configs),
        uses strategy-specific ATR multipliers. Otherwise falls back to global
        config.

        Microstructure Logic (Phase 1 Risk track):
        - stop_distance = max(atr_k * ATR, spread_k * spread)
        - ensures a minimum Liquidity Floor (min_stop_distance_pct)
        """
        entry = max(0.0, float(entry_price or 0.0))
        atr = max(0.0, float(atr_14 or 0.0))
        if entry <= 0 or atr <= 0:
            return {
                "stop_price": 0.0,
                "target_price": 0.0,
                "partial_target_price": 0.0,
                "risk_reward": 0.0,
            }

        # Use strategy-specific multipliers when available
        atr_k = self.config.atr_k
        target_atr = self.config.target_atr
        if strategy_params:
            clean_params = sanitize_strategy_params(
                strategy_params,
                fallback_stop=atr_k,
                fallback_target=target_atr,
            )
            atr_k = float(clean_params.get("stop_atr_mult", atr_k))
            target_atr = float(clean_params.get("target_atr_mult", target_atr))

        # 1. Base ATR-based stop distance
        stop_dist_atr = atr * atr_k

        # 2. Spread-based stop distance (Microstructure-aware)
        spread = ask - bid if ask > bid else 0.0
        stop_dist_spread = spread * self.config.bid_ask_spread_k

        # 3. Dynamic Stop Distance: max(N*ATR, K*Spread)
        stop_dist = max(stop_dist_atr, stop_dist_spread)

        # 4. Liquidity Floor: minimum stop distance pct (to avoid noise stops)
        min_dist_floor = entry * (self.config.min_stop_distance_pct / 100.0)
        stop_dist = max(stop_dist, min_dist_floor)

        stop_price = entry - stop_dist
        target_price = entry + (atr * target_atr)

        # Optional S/R stop tightening only when S1 is near ATR stop and strong.
        if (
            s1_price > 0
            and s1_strength >= 50
            and abs(float(s1_price) - stop_price) <= atr
        ):
            stop_price = max(stop_price, float(s1_price) * 0.99)

        partial_target_price = 0.0
        if r1_price > entry and r1_strength >= 60 and r1_price <= target_price:
            partial_target_price = float(r1_price)

        risk = entry - stop_price
        reward = target_price - entry
        rr = reward / risk if risk > 0 else 0.0
        if risk > 0 and rr < min_rr:
            target_price = entry + (risk * min_rr)
            reward = target_price - entry
            rr = reward / risk if risk > 0 else 0.0

        return {
            "stop_price": float(stop_price),
            "target_price": float(target_price),
            "partial_target_price": float(partial_target_price),
            "risk_reward": float(rr),
        }

    def evaluate(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        qty: int,
        atr_pct: float,
        time_in_trade_minutes: float,
        high_since_entry: Optional[float] = None,
        pnl_improving: bool = False,
        stop_multiplier: Optional[float] = None,
    ) -> ActionPlan:
        """
        Evaluate position against risk gate rules.

        Returns ActionPlan with action, size_delta, confidence, and triggered rules.
        If any hard rule triggers, skip_agents=True to bypass LLM pipeline.
        """
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        atr_k = (
            self.config.atr_k * stop_multiplier
            if stop_multiplier
            else self.config.atr_k
        )
        atr_trail_k = (
            self.config.atr_trail_k * stop_multiplier
            if stop_multiplier
            else self.config.atr_trail_k
        )

        reasons = []
        triggered_rules = []

        # ============================================================
        # RULE 1: Hard Stop - EXIT immediately
        # ============================================================
        if pnl_pct <= self.config.hard_stop_pct:
            reasons.append(
                f"HARD STOP: Loss {pnl_pct:.1f}% exceeds limit {self.config.hard_stop_pct}%"
            )
            triggered_rules.append("hard_stop")
            logger.warning(
                f"[{symbol}] RISK GATE: HARD STOP triggered at {pnl_pct:.1f}%"
            )
            return ActionPlan(
                action=RiskAction.EXIT,
                size_delta=-qty,  # Exit all
                confidence=0.95,
                reasons=reasons,
                triggered_rules=triggered_rules,
                skip_agents=True,
            )

        # ============================================================
        # RULE 2: ATR Stop - EXIT if price below entry - k*ATR
        # ============================================================
        if atr_pct > 0:
            hard_atr_k = min(float(self.config.hard_atr_stop_k), float(atr_k))
            atr_stop_pct = -hard_atr_k * atr_pct
            if pnl_pct <= atr_stop_pct:
                reasons.append(
                    f"ATR STOP: Loss {pnl_pct:.1f}% exceeds {hard_atr_k:.2f}x ATR ({atr_stop_pct:.1f}%)"
                )
                triggered_rules.append("atr_stop")
                logger.warning(
                    f"[{symbol}] RISK GATE: ATR STOP triggered at {pnl_pct:.1f}% (limit: {atr_stop_pct:.1f}%)"
                )
                return ActionPlan(
                    action=RiskAction.EXIT,
                    size_delta=-qty,
                    confidence=0.90,
                    reasons=reasons,
                    triggered_rules=triggered_rules,
                    skip_agents=True,
                )

        # ============================================================
        # RULE 3: Trailing ATR Stop (if we have high since entry)
        # ============================================================
        if high_since_entry and high_since_entry > entry_price and atr_pct > 0:
            profit_guard_active = pnl_pct >= float(
                self.config.profit_guard_activation_pct
            )
            active_trail_k = (
                float(self.config.profit_guard_trail_atr_k)
                if profit_guard_active
                else float(atr_trail_k)
            )
            trail_stop_price = high_since_entry * (1 - active_trail_k * atr_pct / 100)
            if current_price < trail_stop_price:
                reasons.append(
                    f"TRAILING STOP: Price {current_price:.2f} below trail {trail_stop_price:.2f} "
                    f"(from high {high_since_entry:.2f}, k={active_trail_k:.2f}x ATR)"
                )
                triggered_rules.append(
                    "profit_guard_trailing_stop"
                    if profit_guard_active
                    else "trailing_stop"
                )
                logger.warning(f"[{symbol}] RISK GATE: TRAILING STOP triggered")
                return ActionPlan(
                    action=RiskAction.EXIT,
                    size_delta=-qty,
                    confidence=0.85,
                    reasons=reasons,
                    triggered_rules=triggered_rules,
                    skip_agents=True,
                )

        # ============================================================
        # RULE 4: Drawdown Trailing Trim - TRIM/EXIT from high since entry
        # ============================================================
        drawdown_plan = self._evaluate_drawdown_trims(
            symbol=symbol,
            entry_price=entry_price,
            current_price=current_price,
            qty=qty,
            atr_pct=atr_pct,
            high_since_entry=high_since_entry,
        )
        if drawdown_plan:
            logger.info(f"[{symbol}] RISK GATE: DRAWDOWN TRIM triggered")
            return drawdown_plan

        # ============================================================
        # RULE 5: Trim Threshold - TRIM position
        # ============================================================
        if pnl_pct <= self.config.trim_threshold_pct:
            trim_qty = int(qty * self.config.trim_fraction)
            remaining_qty = qty - trim_qty
            remaining_value = remaining_qty * current_price

            # Guard: Don't create stub positions
            if remaining_value < MIN_POSITION_VALUE and remaining_qty > 0:
                logger.info(
                    f"[{symbol}] RISK GATE: Converting trim to full exit (would leave ${remaining_value:.0f} stub)"
                )
                # Convert to full exit instead of creating a stub
                return ActionPlan(
                    action=RiskAction.EXIT,
                    size_delta=-qty,
                    confidence=0.85,
                    reasons=[
                        f"TRIM->EXIT: Loss {pnl_pct:.1f}% - full exit to avoid stub position"
                    ],
                    triggered_rules=["trim_to_exit"],
                    skip_agents=True,
                )

            if trim_qty > 0:
                reasons.append(
                    f"TRIM THRESHOLD: Loss {pnl_pct:.1f}% exceeds {self.config.trim_threshold_pct}%, trimming {self.config.trim_fraction * 100:.0f}%"
                )
                triggered_rules.append("trim_threshold")
                logger.info(f"[{symbol}] RISK GATE: TRIM triggered at {pnl_pct:.1f}%")
                return ActionPlan(
                    action=RiskAction.TRIM,
                    size_delta=-trim_qty,
                    confidence=0.80,
                    reasons=reasons,
                    triggered_rules=triggered_rules,
                    skip_agents=True,
                )

        # ============================================================
        # RULE 6: Time Stop - too long in flat trade
        # ============================================================
        if time_in_trade_minutes >= self.config.time_stop_minutes:
            is_flat = abs(pnl_pct) <= self.config.time_stop_flat_threshold_pct

            if is_flat and not pnl_improving:
                # Flat for too long with no improvement - TRIM
                trim_qty = int(qty * 0.5)  # Trim 50%
                remaining_qty = qty - trim_qty
                remaining_value = remaining_qty * current_price

                # Guard: Don't create stub positions
                if remaining_value < MIN_POSITION_VALUE and remaining_qty > 0:
                    logger.info(
                        f"[{symbol}] RISK GATE: TIME STOP - full exit (would leave ${remaining_value:.0f} stub)"
                    )
                    return ActionPlan(
                        action=RiskAction.EXIT,
                        size_delta=-qty,
                        confidence=0.75,
                        reasons=[
                            f"TIME STOP->EXIT: {time_in_trade_minutes:.0f} min flat - full exit to avoid stub"
                        ],
                        triggered_rules=["time_stop_to_exit"],
                        skip_agents=True,
                    )

                reasons.append(
                    f"TIME STOP: {time_in_trade_minutes:.0f} min in trade, P&L flat at {pnl_pct:+.1f}%"
                )
                triggered_rules.append("time_stop_flat")
                logger.info(f"[{symbol}] RISK GATE: TIME STOP (flat) triggered")
                return ActionPlan(
                    action=RiskAction.TRIM,
                    size_delta=-trim_qty,
                    confidence=0.70,
                    reasons=reasons,
                    triggered_rules=triggered_rules,
                    skip_agents=True,
                )

            elif pnl_pct < -1.0 and not pnl_improving:
                # In meaningful loss for too long with no improvement - TRIM (give chance to recover)
                trim_qty = int(qty * 0.5)
                remaining_qty = qty - trim_qty
                remaining_value = remaining_qty * current_price
                if remaining_value < MIN_POSITION_VALUE and remaining_qty > 0:
                    # Would leave stub - full exit instead
                    reasons.append(
                        f"TIME STOP: {time_in_trade_minutes:.0f} min losing at {pnl_pct:.1f}% - full exit (stub)"
                    )
                    triggered_rules.append("time_stop_loss_to_exit")
                    logger.info(
                        f"[{symbol}] RISK GATE: TIME STOP (loss->exit stub) triggered"
                    )
                    return ActionPlan(
                        action=RiskAction.EXIT,
                        size_delta=-qty,
                        confidence=0.75,
                        reasons=reasons,
                        triggered_rules=triggered_rules,
                        skip_agents=True,
                    )
                reasons.append(
                    f"TIME STOP: {time_in_trade_minutes:.0f} min in trade, losing at {pnl_pct:.1f}% - trimming"
                )
                triggered_rules.append("time_stop_loss")
                logger.info(f"[{symbol}] RISK GATE: TIME STOP (loss) triggered - TRIM")
                return ActionPlan(
                    action=RiskAction.TRIM,
                    size_delta=-trim_qty,
                    confidence=0.70,
                    reasons=reasons,
                    triggered_rules=triggered_rules,
                    skip_agents=True,
                )

        # Multi-day stale-trade enforcement:
        # exit positions held for multiple trading days without minimum progress.
        hold_days = time_in_trade_minutes / (60.0 * 6.5)
        if hold_days >= float(self.config.multi_day_time_stop_days) and pnl_pct < float(
            self.config.multi_day_time_stop_min_gain_pct
        ):
            reasons.append(
                f"MULTI-DAY TIME STOP: held {hold_days:.1f}d with weak progress ({pnl_pct:+.1f}%)"
            )
            triggered_rules.append("multi_day_time_stop")
            logger.info(f"[{symbol}] RISK GATE: MULTI-DAY TIME STOP triggered")
            return ActionPlan(
                action=RiskAction.EXIT,
                size_delta=-qty,
                confidence=0.78,
                reasons=reasons,
                triggered_rules=triggered_rules,
                skip_agents=True,
            )

        # ============================================================
        # RULE 7: Profit Taking - TRIM at profit levels
        # ============================================================
        if pnl_pct > 0:
            # Track which levels we've already hit for this symbol
            if symbol not in self._profit_levels_hit:
                self._profit_levels_hit[symbol] = []

            for level in self.config.profit_take_levels:
                level_pct = level["pct"]
                trim_frac = level["trim"]

                if (
                    pnl_pct >= level_pct
                    and level_pct not in self._profit_levels_hit[symbol]
                ):
                    trim_qty = int(qty * trim_frac)
                    remaining_qty = qty - trim_qty
                    remaining_value = remaining_qty * current_price

                    # Guard: Don't trim if it would leave a stub position
                    if remaining_value < MIN_POSITION_VALUE and remaining_qty > 0:
                        logger.info(
                            f"[{symbol}] RISK GATE: Skipping profit take - would leave ${remaining_value:.0f} stub"
                        )
                        continue  # Try next level instead

                    if trim_qty > 0:
                        self._profit_levels_hit[symbol].append(level_pct)
                        reasons.append(
                            f"PROFIT TAKE: +{pnl_pct:.1f}% hit {level_pct}% target, taking {trim_frac * 100:.0f}%"
                        )
                        triggered_rules.append(f"profit_take_{level_pct}")
                        logger.info(
                            f"[{symbol}] RISK GATE: PROFIT TAKE at +{pnl_pct:.1f}%"
                        )
                        return ActionPlan(
                            action=RiskAction.PROFIT_TAKE,
                            size_delta=-trim_qty,
                            confidence=0.85,
                            reasons=reasons,
                            triggered_rules=triggered_rules,
                            skip_agents=False,  # Still run agents for remaining position
                        )

        # ============================================================
        # NO RULE TRIGGERED - proceed to agent pipeline
        # ============================================================
        return ActionPlan(
            action=RiskAction.NONE,
            size_delta=0,
            confidence=0.0,
            reasons=["No risk gate triggered - proceeding to agent analysis"],
            triggered_rules=[],
            skip_agents=False,
        )

    def evaluate_with_pdt(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        qty: int,
        atr_pct: float,
        time_in_trade_minutes: float,
        high_since_entry: Optional[float] = None,
        pnl_improving: bool = False,
        day_trades_remaining: int = 10,
        conviction_score: float = 50.0,
        rsi: float = 50.0,
        relative_strength: float = 0.0,
        stop_multiplier: Optional[float] = None,
    ) -> ActionPlan:
        """
        Evaluate risk rules with ADD signal support.

        Key differences from evaluate():
        1. Generates ADD signals for strong winners
        2. Returns conviction score for decision making
        3. Preserves legacy hold-phase fields without PDT gating
        """
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        atr_k = (
            self.config.atr_k * stop_multiplier
            if stop_multiplier
            else self.config.atr_k
        )
        atr_trail_k = (
            self.config.atr_trail_k * stop_multiplier
            if stop_multiplier
            else self.config.atr_trail_k
        )

        # PDT/day-trade gates are disabled. Hold phase remains serialized for
        # old reports but should not suppress exits, trims, or time stops.
        hold_phase = HoldPhase.UNLOCKED
        is_hold_locked = False
        pdt_burn_required = False

        reasons = []
        triggered_rules = []

        # ============================================================
        # CRITICAL RULES (Hard Stop / ATR Stop)
        # ============================================================

        # Hard stop - ALWAYS exit
        if pnl_pct <= self.config.hard_stop_pct:
            reasons.append(
                f"CRITICAL: Hard stop {pnl_pct:.1f}% <= {self.config.hard_stop_pct}%"
            )
            triggered_rules.append("hard_stop")
            logger.warning(f"[{symbol}] CRITICAL EXIT: Hard stop at {pnl_pct:.1f}%")
            return ActionPlan(
                action=RiskAction.EXIT,
                size_delta=-qty,
                confidence=0.95,
                reasons=reasons,
                triggered_rules=triggered_rules,
                skip_agents=True,
                hold_phase=HoldPhase.CRITICAL,
                pdt_burn_required=False,
                conviction_score=0,
            )

        # ATR stop - ALWAYS exit
        if atr_pct > 0:
            hard_atr_k = min(float(self.config.hard_atr_stop_k), float(atr_k))
            atr_stop_pct = -hard_atr_k * atr_pct
            if pnl_pct <= atr_stop_pct:
                reasons.append(
                    f"ATR STOP: {pnl_pct:.1f}% <= {atr_stop_pct:.1f}% ({hard_atr_k:.2f}x ATR)"
                )
                triggered_rules.append("atr_stop")

                return ActionPlan(
                    action=RiskAction.EXIT,
                    size_delta=-qty,
                    confidence=0.90,
                    reasons=reasons,
                    triggered_rules=triggered_rules,
                    skip_agents=True,
                    hold_phase=HoldPhase.CRITICAL,
                    pdt_burn_required=False,
                    conviction_score=10,
                )

        # ============================================================
        # ACTIVE RISK RULES (Trailing Stop / Trims)
        # ============================================================

        # Trailing stop - execute immediately.
        if high_since_entry and high_since_entry > entry_price and atr_pct > 0:
            profit_guard_active = pnl_pct >= float(
                self.config.profit_guard_activation_pct
            )
            active_trail_k = (
                float(self.config.profit_guard_trail_atr_k)
                if profit_guard_active
                else float(atr_trail_k)
            )
            trail_stop_price = high_since_entry * (1 - active_trail_k * atr_pct / 100)
            if current_price < trail_stop_price:
                reasons.append(
                    f"TRAILING STOP: Price {current_price:.2f} < trail {trail_stop_price:.2f} "
                    f"(k={active_trail_k:.2f}x ATR)"
                )
                triggered_rules.append(
                    "profit_guard_trailing_stop"
                    if profit_guard_active
                    else "trailing_stop"
                )

                return ActionPlan(
                    action=RiskAction.EXIT,
                    size_delta=-qty,
                    confidence=0.85,
                    reasons=reasons,
                    triggered_rules=triggered_rules,
                    skip_agents=True,
                    hold_phase=hold_phase,
                    pdt_burn_required=False,
                    conviction_score=20,
                )

        # Drawdown trailing trims are active; PDT does not lock them.
        if not is_hold_locked:
            drawdown_plan = self._evaluate_drawdown_trims(
                symbol=symbol,
                entry_price=entry_price,
                current_price=current_price,
                qty=qty,
                atr_pct=atr_pct,
                high_since_entry=high_since_entry,
            )
            if drawdown_plan:
                drawdown_plan.hold_phase = hold_phase
                drawdown_plan.pdt_burn_required = False
                return drawdown_plan

        # Trim threshold - trigger exit immediately.
        if not is_hold_locked and pnl_pct <= self.config.trim_threshold_pct:
            # We treat trims as full exits if PDT is not a concern and loss is meaningful
            is_full_exit = pnl_pct <= -5.0
            trim_qty = int(qty) if is_full_exit else int(qty * self.config.trim_fraction)
            
            if trim_qty > 0:
                reasons.append(f"TRIM/EXIT: Loss {pnl_pct:.1f}% exceeded threshold")
                triggered_rules.append("trim_threshold")
                return ActionPlan(
                    action=RiskAction.EXIT if is_full_exit else RiskAction.TRIM,
                    size_delta=-trim_qty,
                    confidence=0.80,
                    reasons=reasons,
                    triggered_rules=triggered_rules,
                    skip_agents=True,
                    hold_phase=HoldPhase.CRITICAL if is_full_exit else hold_phase,
                    pdt_burn_required=False,
                    conviction_score=30,
                )

        # Time stop.
        if (
            not is_hold_locked
            and time_in_trade_minutes >= self.config.time_stop_minutes
            and hold_phase == HoldPhase.UNLOCKED
        ):
            is_flat = abs(pnl_pct) <= self.config.time_stop_flat_threshold_pct

            if is_flat and not pnl_improving:
                trim_qty = int(qty * 0.5)
                reasons.append(f"TIME STOP: Flat for {time_in_trade_minutes:.0f} min")
                triggered_rules.append("time_stop_flat")
                return ActionPlan(
                    action=RiskAction.TRIM,
                    size_delta=-trim_qty,
                    confidence=0.70,
                    reasons=reasons,
                    triggered_rules=triggered_rules,
                    skip_agents=True,
                    hold_phase=hold_phase,
                    pdt_burn_required=False,
                    conviction_score=40,
                )

            elif pnl_pct < 0:
                reasons.append(f"TIME STOP: Losing for {time_in_trade_minutes:.0f} min")
                triggered_rules.append("time_stop_loss")
                return ActionPlan(
                    action=RiskAction.EXIT,
                    size_delta=-qty,
                    confidence=0.75,
                    reasons=reasons,
                    triggered_rules=triggered_rules,
                    skip_agents=True,
                    hold_phase=hold_phase,
                    pdt_burn_required=False,
                    conviction_score=25,
                )

        # Multi-day stale-trade enforcement.
        hold_days_calc = time_in_trade_minutes / (60.0 * 6.5)
        if (
            hold_days_calc >= float(self.config.multi_day_time_stop_days)
            and pnl_pct < float(self.config.multi_day_time_stop_min_gain_pct)
        ):
            reasons.append(
                f"MULTI-DAY TIME STOP: held {hold_days_calc:.1f}d with weak progress ({pnl_pct:+.1f}%)"
            )
            triggered_rules.append("multi_day_time_stop")
            return ActionPlan(
                action=RiskAction.EXIT,
                size_delta=-qty,
                confidence=0.78,
                reasons=reasons,
                triggered_rules=triggered_rules,
                skip_agents=True,
                hold_phase=hold_phase,
                pdt_burn_required=pdt_burn_required,
                conviction_score=20,
            )

        # ============================================================
        # ADD SIGNAL - Scale into winners
        # ============================================================
        if pnl_pct >= self.config.add_profit_threshold_pct:
            # Winner! Check if we should add
            add_score = conviction_score

            # Boost score for strong technicals
            if rsi > 50 and rsi < 70:
                add_score += 10
                reasons.append(f"RSI {rsi:.0f} supportive")

            if relative_strength > 2:
                add_score += 10
                reasons.append(f"Outperforming SPY by {relative_strength:.1f}%")

            # Strong winner momentum
            if pnl_pct >= 6:
                add_score += 10
                reasons.append(f"Strong winner +{pnl_pct:.1f}%")

            if add_score >= self.config.add_conviction_min:
                add_qty = int(qty * self.config.add_max_position_pct)
                reasons.insert(
                    0, f"ADD SIGNAL: +{pnl_pct:.1f}% gain, conviction {add_score:.0f}"
                )
                triggered_rules.append("add_signal")
                logger.info(
                    f"[{symbol}] ADD SIGNAL: +{pnl_pct:.1f}%, conviction {add_score:.0f}"
                )
                return ActionPlan(
                    action=RiskAction.ADD,
                    size_delta=add_qty,  # Positive = add
                    confidence=min(0.85, add_score / 100),
                    reasons=reasons,
                    triggered_rules=triggered_rules,
                    skip_agents=False,  # Run agents for confirmation
                    hold_phase=hold_phase,
                    pdt_burn_required=False,
                    conviction_score=add_score,
                )

        # ============================================================
        # PROFIT TAKING (same as before, but hold guard aware)
        # ============================================================
        if not is_hold_locked and pnl_pct > 0:
            if symbol not in self._profit_levels_hit:
                self._profit_levels_hit[symbol] = []

            for level in self.config.profit_take_levels:
                level_pct = level["pct"]
                trim_frac = level["trim"]

                if (
                    pnl_pct >= level_pct
                    and level_pct not in self._profit_levels_hit[symbol]
                ):
                    trim_qty = int(qty * trim_frac)
                    if trim_qty > 0:
                        self._profit_levels_hit[symbol].append(level_pct)
                        reasons.append(
                            f"PROFIT TAKE: +{pnl_pct:.1f}% hit {level_pct}% target"
                        )
                        triggered_rules.append(f"profit_take_{level_pct}")
                        return ActionPlan(
                            action=RiskAction.PROFIT_TAKE,
                            size_delta=-trim_qty,
                            confidence=0.85,
                            reasons=reasons,
                            triggered_rules=triggered_rules,
                            skip_agents=False,
                            hold_phase=hold_phase,
                            pdt_burn_required=False,
                            conviction_score=conviction_score,
                        )

        # ============================================================
        # NO TRIGGER - return with hold phase info
        # ============================================================
        return ActionPlan(
            action=RiskAction.NONE,
            size_delta=0,
            confidence=0.0,
            reasons=[f"No trigger - {hold_phase.value}, P&L {pnl_pct:+.1f}%"],
            triggered_rules=[],
            skip_agents=False,
            hold_phase=hold_phase,
            pdt_burn_required=pdt_burn_required,
            conviction_score=conviction_score,
        )

    def reset_profit_levels(self, symbol: str):
        """Reset profit and drawdown level tracking for a symbol (call when position closed)."""
        if symbol in self._profit_levels_hit:
            del self._profit_levels_hit[symbol]
        if symbol in self._drawdown_levels_hit:
            del self._drawdown_levels_hit[symbol]


# Singleton for use across modules
_default_gate: Optional[RiskGate] = None


def get_risk_gate(config_path: Optional[Path] = None) -> RiskGate:
    """Get or create the default risk gate."""
    global _default_gate
    if _default_gate is None:
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

        if config_path and config_path.exists():
            config = RiskGateConfig.from_yaml(config_path)
        else:
            config = RiskGateConfig.from_trading_config()
        _default_gate = RiskGate(config)
    return _default_gate
