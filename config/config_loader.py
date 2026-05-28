"""
Centralized Configuration Loader
=================================
Single source of truth for all configuration values.

Features:
- Loads from YAML config file
- Environment variable overrides
- Pydantic validation
- Type-safe access
- Path resolution for external resources

Usage:
    from config.config_loader import get_config
    config = get_config()

    # Access typed config sections
    risk_config = config.risk_gate
    llm_config = config.llm
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

_ENV_LOADED = False


def _load_env_file(env_path: Optional[Path] = None) -> None:
    """Load .env into os.environ without overwriting existing values."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_path = env_path or (Path(__file__).parent.parent / ".env")
    if not env_path.exists():
        _ENV_LOADED = True
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if key and key not in os.environ:
            os.environ[key] = val
    _ENV_LOADED = True


def _env_first(*names: str) -> Optional[str]:
    """Return first non-empty environment value for provided keys."""
    for name in names:
        value = os.environ.get(name)
        if value and str(value).strip():
            return str(value).strip()
    return None


# Load .env early so config defaults resolve correctly (esp. DownDay paths).
_load_env_file()


# ============================================================
# Configuration Models (Pydantic for validation)
# ============================================================


class RiskGateConfig(BaseModel):
    """Risk gate threshold configuration."""

    hard_stop_pct: float = Field(
        default=-8.0, description="Exit if loss exceeds this %"
    )
    early_fade_pnl_pct: float = Field(
        default=-2.0,
        description="Fast-fail PnL%% threshold for fresh intraday entries",
    )
    early_fade_max_hold_minutes: int = Field(
        default=60,
        ge=5,
        le=240,
        description="Window during which early-fade rule is active after entry",
    )
    trim_threshold_pct: float = Field(
        default=-5.0, description="Trim if loss exceeds this %"
    )
    trim_fraction: float = Field(default=0.5, ge=0.0, le=1.0)

    # ATR-based stops
    atr_k: float = Field(default=2.0, ge=0.5, le=5.0)
    atr_trail_k: float = Field(default=1.5, ge=0.5, le=5.0)
    target_atr: float = Field(
        default=3.0,
        ge=1.0,
        le=8.0,
        description="ATR multiple for initial profit target",
    )
    bid_ask_spread_k: float = Field(
        default=2.0, description="Multiplier for spread in stop calculation"
    )
    min_stop_distance_pct: float = Field(
        default=0.5,
        description="Minimum stop distance as % of entry price (liquidity floor)",
    )

    # Time-based stops
    time_stop_minutes: float = Field(default=240, ge=30)
    time_stop_flat_threshold_pct: float = Field(default=1.0)
    multi_day_time_stop_days: float = Field(default=3.0, ge=1.0, le=20.0)
    multi_day_time_stop_min_gain_pct: float = Field(default=0.5, ge=-5.0, le=10.0)

    # Profit taking levels
    profit_take_levels: List[Dict[str, float]] = Field(
        default_factory=lambda: [
            {"pct": 5.0, "trim": 0.10},
            {"pct": 8.0, "trim": 0.10},
            {"pct": 12.0, "trim": 0.15},
            {"pct": 15.0, "trim": 0.20},
            {"pct": 20.0, "trim": 0.25},
        ]
    )

    # Drawdown-based trailing trims (from high since entry)
    drawdown_trim_levels: List[Dict[str, float]] = Field(
        default_factory=lambda: [
            {"drawdown_pct": 3.0, "trim": 0.33},
            {"drawdown_pct": 6.0, "trim": 0.45},
            {"drawdown_pct": 10.0, "trim": 1.0},
        ]
    )
    drawdown_trail_atr_k: float = Field(default=0.5, ge=0.0, le=5.0)

    # PDT constraints
    pdt_unlock_hours: float = Field(default=24.0)
    pdt_buffer_trades: int = Field(default=2, ge=0, le=3)

    # ADD thresholds
    add_profit_threshold_pct: float = Field(default=4.0)
    add_conviction_min: float = Field(default=65.0)
    add_max_position_pct: float = Field(default=0.50)

    # Add-block and midday trim controls
    add_block_after_trim: bool = Field(default=True)
    midday_revalidation_trim_fraction: float = Field(default=0.33, ge=0.05, le=1.0)
    min_hold_minutes_for_trim: int = Field(
        default=120,
        ge=0,
        le=480,
        description="Minimum minutes a position must be held before any trim (except hard stops)",
    )


class ConvictionConfig(BaseModel):
    """Conviction engine configuration."""

    technical_weight: float = Field(default=0.30)
    fundamental_weight: float = Field(default=0.15)
    relative_weight: float = Field(default=0.15)
    position_weight: float = Field(default=0.40)

    # Aggressive Time Decay Settings (v0.12.1)
    flat_threshold_pct: float = Field(default=0.5)
    flat_decay_rate: float = Field(default=4.0)
    flat_decay_max: float = Field(default=55.0)
    flat_decay_aggressive_hours: float = Field(default=16.0)
    winner_decay_start_hours: float = Field(default=8.0)
    winner_decay_rate: float = Field(default=1.5)
    loser_decay_rate: float = Field(default=5.0)
    loser_decay_max: float = Field(default=60.0)
    stagnation_penalty: float = Field(default=15.0)
    stagnation_hours_threshold: float = Field(default=3.0)

    # Legacy aliases
    time_decay_start_hours: float = Field(default=4.0)
    time_decay_rate_per_hour: float = Field(default=2.0)

    add_conviction_min: float = Field(default=65.0)
    hold_conviction_min: float = Field(default=40.0)

    @model_validator(mode="after")
    def validate_weights(self):
        total = (
            self.technical_weight
            + self.fundamental_weight
            + self.relative_weight
            + self.position_weight
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Conviction weights must sum to 1.0, got {total}")
        return self


class RotationConfig(BaseModel):
    """Portfolio rotation configuration."""

    min_improvement_score: float = Field(default=15.0)
    max_rotations_per_day: int = Field(default=2)
    scan_top_candidates: int = Field(default=10)
    min_signal_score: float = Field(default=70.0)


class DecisionPolicyConfig(BaseModel):
    """Decision policy configuration."""

    min_candidates: int = Field(default=3)

    # Scoring weights
    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "risk_score": 0.35,
            "technical_score": 0.25,
            "sentiment_score": 0.15,
            "momentum_score": 0.15,
            "time_score": 0.10,
        }
    )

    exit_score_threshold: float = Field(default=70.0)
    trim_score_threshold: float = Field(default=50.0)
    add_score_threshold: float = Field(default=65.0)

    max_add_pct: float = Field(default=0.50)
    min_trim_pct: float = Field(default=0.20)


class PlanCapsConfig(BaseModel):
    """Plan publishing and handoff caps. A value of 0 means unlimited."""

    pm_bridge_limit: int = Field(default=0, ge=0)
    pm_plan_signals_cap: int = Field(default=0, ge=0)
    premarket_exec_cap: int = Field(default=0, ge=0)
    premarket_input_cap: int = Field(default=0, ge=0)


class TrimGovernanceConfig(BaseModel):
    """Cross-path trim controls for live position management."""

    cooldown_minutes: int = Field(default=60, ge=0)
    max_per_symbol_per_day: int = Field(default=2, ge=0)
    aggregate_cap_pct: float = Field(default=0.50, ge=0.0, le=1.0)
    aggregate_cap_red_day_max_pct: float = Field(default=0.80, ge=0.0, le=1.0)
    aggregate_cap_red_day_loss_scale_pct: float = Field(default=5.0, ge=0.1, le=50.0)
    disable_auto_trim_paths: bool = Field(default=True)


class LLMAdvisoryEscalationConfig(BaseModel):
    """Advisory-only LLM trim/exit escalation ladder."""

    enabled: bool = Field(default=True)
    min_gap_minutes: int = Field(default=30, ge=0, le=390)
    advisor_cooldown_minutes: int = Field(default=30, ge=0, le=390)
    ladder_pct: List[float] = Field(default_factory=lambda: [25.0, 50.0, 100.0])
    confidence_floor: float = Field(default=0.70, ge=0.0, le=1.0)
    full_exit_reentry_cooldown_minutes: int = Field(default=120, ge=0, le=1440)
    max_steps_per_symbol_per_day: int = Field(default=3, ge=1, le=10)
    allow_risk_gate_profit_take_full_exit: bool = Field(default=False)


class LossFloorTierConfig(BaseModel):
    """Single cumulative-P&L loss-floor tier."""

    max_pnl_pct: float = Field(default=-5.0, le=0.0)
    min_held_minutes: float = Field(default=0.0, ge=0.0)


class LossFloorConfig(BaseModel):
    """Always-on cumulative loss-floor exits for held positions."""

    enabled: bool = Field(default=False)
    ladder: List[LossFloorTierConfig] = Field(
        default_factory=lambda: [
            LossFloorTierConfig(max_pnl_pct=-2.0, min_held_minutes=60.0),
            LossFloorTierConfig(max_pnl_pct=-3.5, min_held_minutes=30.0),
            LossFloorTierConfig(max_pnl_pct=-5.0, min_held_minutes=0.0),
        ]
    )
    telemetry_path: str = Field(default="logs/loss_floor_telemetry.jsonl")


class PortfolioConfig(BaseModel):
    """Portfolio limits configuration."""

    max_positions: int = Field(default=7, ge=1, le=60)
    core_max_positions: Optional[int] = Field(
        default=None,
        ge=1,
        le=60,
        description="Core max positions for standard entries (reserved slots handled separately)",
    )
    late_scan_reserved_positions: int = Field(
        default=0,
        ge=0,
        le=20,
        description="Reserved slots for intraday reserve scans (additive to core cap)",
    )
    position_size_target: float = Field(default=3000.0)
    position_size_max: float = Field(default=6000.0)
    max_position_pct_of_equity: float = Field(default=10.0, ge=0.0, le=100.0)
    min_hold_minutes: int = Field(default=30)

    # Sector exposure limits
    max_sector_exposure_pct: float = Field(
        default=40.0, description="Max % of portfolio in one sector"
    )
    max_correlation_exposure: float = Field(
        default=0.7, description="Max avg correlation between positions"
    )


class StrategyConfig(BaseModel):
    """Core strategy thresholds used across modules."""

    min_lesson_score: float = Field(default=10.0)
    r2_max_atr: float = Field(default=3.0)
    partial_at_r1: float = Field(default=0.5, ge=0.0, le=1.0)
    min_bullish: float = Field(default=4.0)
    s1_stop_buffer_pct: float = Field(
        default=0.2, description="Buffer below S1 in percent"
    )
    fallback_stop_atr: float = Field(default=1.5)
    fallback_target_atr: float = Field(default=2.5)


class EntryQualityConfig(BaseModel):
    """Entry quality controls for intraday execution."""

    enabled: bool = Field(default=True)
    min_entry_score: float = Field(default=25.0, ge=0.0, le=100.0)
    min_risk_reward: float = Field(default=1.4, ge=0.5, le=5.0)
    off_watchlist_validation_enabled: bool = Field(default=True)
    off_watchlist_min_score: float = Field(default=60.0, ge=0.0, le=100.0)
    off_watchlist_min_risk_reward: float = Field(default=1.6, ge=0.5, le=5.0)
    off_watchlist_min_volume_ratio: float = Field(default=1.15, ge=0.0, le=10.0)
    off_watchlist_require_backtest_validation: bool = Field(default=True)
    watchlist_source_score_bonus: float = Field(default=4.0, ge=0.0, le=25.0)
    overnight_full_watchlist_edge_enabled: bool = Field(default=True)
    overnight_full_watchlist_edge_min_score: float = Field(
        default=60.0, ge=0.0, le=100.0
    )
    overnight_full_watchlist_edge_max_score: float = Field(
        default=75.0, ge=0.0, le=100.0
    )
    overnight_full_watchlist_edge_bonus: float = Field(default=6.0, ge=0.0, le=25.0)
    unknown_source_score_band_block_enabled: bool = Field(default=True)
    unknown_source_block_min_score: float = Field(default=75.0, ge=0.0, le=100.0)
    unknown_source_block_max_score: float = Field(default=85.0, ge=0.0, le=100.0)
    high_score_no_catalyst_confirmation_enabled: bool = Field(default=True)
    high_score_no_catalyst_min_score: float = Field(default=75.0, ge=0.0, le=100.0)
    overnight_watchlist_validation_carry_min_plan_score: float = Field(
        default=70.0, ge=0.0, le=100.0
    )
    overnight_watchlist_validation_carry_fraction: float = Field(
        default=0.25, ge=0.0, le=1.0
    )
    overnight_watchlist_validation_carry_cap: float = Field(
        default=12.0, ge=0.0, le=25.0
    )
    off_watchlist_score_penalty: float = Field(default=8.0, ge=0.0, le=25.0)

    # Intraday momentum gate
    momentum_gate_enabled: bool = Field(default=True)
    momentum_penalty_points: float = Field(default=15.0, ge=0.0, le=50.0)
    momentum_insufficient_data_penalty_points: float = Field(
        default=5.0, ge=0.0, le=50.0
    )
    momentum_min_bars: int = Field(default=3, ge=2, le=20)
    momentum_min_volume_ratio: float = Field(default=0.8, ge=0.1, le=10.0)
    momentum_require_above_vwap: bool = Field(default=True)
    momentum_require_trend_up: bool = Field(default=True)
    momentum_hard_block: bool = Field(default=False)
    block_duplicate_open_orders: bool = Field(default=True)

    # Watchlist breakout override (promote select WATCH signals to entries)
    watch_breakout_override_enabled: bool = Field(default=True)
    watch_breakout_min_score: float = Field(default=65.0, ge=0.0, le=100.0)
    watch_breakout_min_volume_ratio: float = Field(default=2.0, ge=0.5, le=20.0)
    breakout_tight_stop_atr: float = Field(default=1.35, ge=0.5, le=5.0)

    # Candidate backtest validation in day-manager entry flow
    candidate_backtest_validation_enabled: bool = Field(default=True)
    candidate_backtest_top_n: int = Field(default=12, ge=1, le=100)
    candidate_backtest_refresh_minutes: int = Field(default=30, ge=1, le=240)
    candidate_backtest_boost_cap: float = Field(default=10.0, ge=0.0, le=50.0)
    overnight_watchlist_backtest_reject_enabled: bool = Field(default=False)
    overnight_watchlist_backtest_penalty_cap: float = Field(
        default=8.0, ge=0.0, le=25.0
    )

    # Signal freshness
    stale_signal_max_age_days: int = Field(default=1, ge=0, le=10)
    stale_signal_penalty_points: float = Field(default=10.0, ge=0.0, le=40.0)

    # Wave-based entry throttling
    wave_entry_enabled: bool = Field(default=True)
    wave1_allow_observation_entries: bool = Field(default=True)
    early_runner_enabled: bool = Field(default=True)
    early_runner_max_positions: int = Field(default=10, ge=1, le=50)
    early_runner_post_open_rescan_enabled: bool = Field(default=True)
    overnight_recheck_after_first_hour_enabled: bool = Field(default=True)
    observation_period_minutes: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Minutes after market open (9:30 ET) when observation period ends and trading begins",
    )
    strong_candidate_fast_lane_enabled: bool = Field(
        default=True,
        description="When true, candidates with score >= strong_candidate_fast_lane_min_score "
        "bypass the OBSERVATION-phase no-entry gate after a minimum watch window.",
    )
    strong_candidate_fast_lane_min_score: float = Field(
        default=80.0,
        ge=0.0,
        le=100.0,
        description="Minimum candidate score to qualify for the fast lane.",
    )
    strong_candidate_fast_lane_min_minutes_after_open: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Minimum minutes after 09:30 ET before the fast lane can submit. "
        "Acts as the always-on observation guard for high-conviction entries.",
    )
    wave_breakout_rescue_enabled: bool = Field(default=True)
    wave_breakout_rescue_min_score: float = Field(default=75.0, ge=0.0, le=100.0)
    wave_breakout_rescue_max_gap_pct: float = Field(default=9.0, ge=0.0, le=20.0)
    wave_breakout_rescue_size_multiplier: float = Field(default=0.40, ge=0.05, le=1.0)
    wave_breakout_rescue_limit_offset_bps: float = Field(default=15.0, ge=0.0, le=200.0)
    pm_wave_gap_rescue_enabled: bool = Field(default=True)
    pm_wave_gap_rescue_min_score: float = Field(default=80.0, ge=0.0, le=100.0)
    pm_wave_gap_rescue_max_gap_pct: float = Field(default=18.0, ge=0.0, le=25.0)
    wave_hard_reject_gap_pct: float = Field(default=12.0, ge=1.0, le=50.0)
    quick_turnover_continuation_enabled: bool = Field(default=True)
    quick_turnover_continuation_min_score: float = Field(default=66.0, ge=0.0, le=100.0)
    quick_turnover_continuation_max_gap_pct: float = Field(
        default=18.0, ge=0.0, le=25.0
    )
    quick_turnover_continuation_min_volume_ratio: float = Field(
        default=1.0, ge=0.0, le=10.0
    )
    wave_max_entries: int = Field(default=10, ge=1, le=50)
    wave3_auto_advance: bool = Field(default=True)
    wave_capacity_slots_multiplier: float = Field(default=0.75, ge=0.25, le=1.0)
    wave_capacity_bypass_min_score: float = Field(default=75.0, ge=0.0, le=100.0)
    wave2_min_wave1_avg_pnl: float = Field(default=-0.5)
    wave3_min_overall_avg_pnl: float = Field(default=-0.3)
    wave4_min_overall_avg_pnl: float = Field(default=-0.2)
    entry_gap_reject_pct: float = Field(default=7.0, ge=1.0, le=25.0)
    strict_plan_authority_enabled: bool = Field(
        default=True,
        description=(
            "When enabled, non-plan discretionary long entries are blocked until "
            "plan-origin candidates are exhausted."
        ),
    )
    strict_plan_authority_min_plan_candidates: int = Field(
        default=1,
        ge=0,
        le=500,
        description="Minimum plan candidates required for pre-open execution contract readiness.",
    )
    preopen_execution_contract_required: bool = Field(
        default=True,
        description=(
            "Fail closed for new entries at session start when execution-control "
            "invariants are missing (for example no executable plan candidates)."
        ),
    )
    max_buy_orders_per_day: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Hard cap on buy submissions per session day across all entry lanes.",
    )
    enforce_buying_power_gate: bool = Field(
        default=True,
        description=(
            "Require per-order buying-power validation before any buy submission."
        ),
    )
    buy_order_min_cash_floor: float = Field(
        default=1000.0,
        ge=0.0,
        le=5_000_000.0,
        description=(
            "Cash-account floor for buy submissions. Margin accounts are gated by "
            "buying power instead of raw cash."
        ),
    )
    buy_order_guard_fail_closed: bool = Field(
        default=True,
        description=(
            "Fail closed for buy submissions when capital guard cannot resolve account data."
        ),
    )
    same_day_reentry_cooldown_minutes: int = Field(default=60, ge=0, le=240)
    entry_conviction_floor: float = Field(default=60.0, ge=0.0, le=100.0)
    fresh_entry_exit_cooldown_minutes: int = Field(default=120, ge=0, le=240)
    stale_entry_reprice_streak: int = Field(default=2, ge=0, le=10)
    stale_entry_gap_reject_streak: int = Field(default=2, ge=0, le=10)
    stale_entry_temp_evict_sessions: int = Field(default=1, ge=0, le=10)
    early_runner_empty_streak_observe_only: bool = Field(default=True)
    early_runner_observe_only_streak: int = Field(default=3, ge=1, le=20)
    core_trading_adaptive_min_score: float = Field(default=30.0, ge=0.0, le=100.0)
    core_trading_adaptive_top_n: int = Field(default=12, ge=1, le=100)
    replacement_score_gap: float = Field(default=15.0, ge=0.0, le=100.0)
    replacement_weak_score_gap: float = Field(default=10.0, ge=0.0, le=100.0)
    replacement_weak_position_score_ceiling: float = Field(
        default=5.0, ge=-100.0, le=100.0
    )
    replacement_min_candidate_score: float = Field(default=45.0, ge=0.0, le=100.0)
    replacement_max_attempts_per_position: int = Field(default=10, ge=1, le=50)
    replacement_prefilter_enabled: bool = Field(default=True)
    replacement_prefilter_intraday_move_pct: float = Field(
        default=12.0, ge=0.0, le=100.0
    )
    watchlist_prune_tolerance_min_score: float = Field(default=35.0, ge=0.0, le=100.0)
    watchlist_prune_orb_tolerance_pct: float = Field(default=1.0, ge=0.0, le=10.0)
    watchlist_prune_vwap_tolerance_pct: float = Field(default=1.0, ge=0.0, le=10.0)

    # Power Hour (3:00-4:00 PM ET) institutional volume spike detection
    power_hour_volume_enabled: bool = Field(
        default=True,
        description="Enable institutional volume spike detection during power hour (3:00-4:00 PM ET)",
    )
    power_hour_volume_threshold: float = Field(
        default=2.5,
        ge=1.5,
        le=5.0,
        description="Volume surge ratio threshold to trigger power hour entries (2.5x = institutional level)",
    )
    power_hour_min_position_pct: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Only trigger power hour entries if below this percentage of max positions (room to add)",
    )
    power_hour_max_entries_per_day: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of power hour entries allowed per day",
    )

    # Intraday strategy overlays for day-manager candidate scoring/sizing.
    strategy_profiles_enabled: bool = Field(default=True)
    opening_range_window_minutes: int = Field(default=30, ge=10, le=90)
    opening_range_breakout_score_bonus: float = Field(default=12.0, ge=0.0, le=40.0)
    opening_range_breakout_size_multiplier: float = Field(default=1.15, ge=0.25, le=3.0)

    # VWAP mean reversion: lunch-hour reversion setup (times in CT minutes).
    vwap_mean_reversion_window_start_ct: int = Field(default=630, ge=0, le=1439)
    vwap_mean_reversion_window_end_ct: int = Field(default=750, ge=0, le=1439)
    vwap_mean_reversion_std_threshold: float = Field(default=1.6, ge=0.5, le=5.0)
    vwap_mean_reversion_score_bonus: float = Field(default=16.0, ge=0.0, le=50.0)
    vwap_mean_reversion_size_multiplier: float = Field(default=1.10, ge=0.25, le=3.0)
    vwap_mean_reversion_momentum_penalty_multiplier: float = Field(
        default=0.35, ge=0.0, le=1.0
    )

    # Gap-fill setup controls.
    gap_fill_window_end_ct: int = Field(default=630, ge=0, le=1439)
    gap_fill_min_abs_gap_pct: float = Field(default=1.5, ge=0.5, le=20.0)
    gap_fill_score_bonus: float = Field(default=10.0, ge=0.0, le=40.0)
    gap_fill_size_multiplier: float = Field(default=0.95, ge=0.25, le=3.0)

    # Conviction-based position sizing
    conviction_sizing_enabled: bool = Field(default=True)
    conviction_size_tiers: List[Dict[str, float]] = Field(
        default_factory=lambda: [
            {"min_score": 70.0, "multiplier": 1.5},
            {"min_score": 55.0, "multiplier": 1.0},
            {"min_score": 35.0, "multiplier": 0.5},
        ]
    )

    # SMA 200 zero-pass fallback
    sma200_relax_if_zero_pass: bool = Field(default=True)
    sma200_relax_max_below_pct: float = Field(default=3.0, ge=0.0, le=10.0)

    # --- Daily drawdown circuit breaker ---
    daily_max_drawdown_pct: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Maximum daily loss (%) before halting ALL new entries for the day",
    )
    daily_drawdown_tiers: List[Dict[str, float]] = Field(
        default_factory=lambda: [
            {"drawdown_pct": 1.0, "size_multiplier": 0.5},
            {"drawdown_pct": 1.5, "halt_entries": 1.0, "trail_atr": 1.5},
            {"drawdown_pct": 2.0, "trail_atr": 1.2, "trim_bottom_quartile": 1.0},
            {"drawdown_pct": 3.0, "force_close_losers": 1.0, "halt_next_open": 1.0},
        ],
        description="Tiered defensive actions fired as daily loss deepens.",
    )

    # --- VWAP Universe Scanner (full-universe SMA-100 scan) ---
    vwap_universe_scan_enabled: bool = Field(
        default=True,
        description="Enable full-universe VWAP mean reversion scanning during market hours",
    )
    vwap_universe_scan_interval_cycles: int = Field(
        default=5,
        ge=2,
        le=30,
        description="Run VWAP universe scan every N cycles (~N minutes)",
    )
    vwap_universe_max_entries_per_day: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum VWAP universe entries per day (cap on new positions)",
    )
    vwap_universe_min_score: float = Field(
        default=50.0,
        ge=20.0,
        le=90.0,
        description="Minimum VWAP setup score to pass validation gate",
    )
    vwap_universe_min_risk_reward: float = Field(
        default=1.5,
        ge=0.5,
        le=5.0,
        description="Minimum risk/reward ratio for VWAP universe entries",
    )
    vwap_universe_sma_period: int = Field(
        default=100,
        ge=20,
        le=200,
        description="SMA period for universe pre-filter (stocks must be above this SMA)",
    )
    vwap_universe_min_price: float = Field(
        default=2.0,
        ge=1.0,
        le=50.0,
        description="Minimum stock price for VWAP universe",
    )
    vwap_universe_max_price: float = Field(
        default=200.0,
        ge=50.0,
        le=1000.0,
        description="Maximum stock price for VWAP universe",
    )
    vwap_universe_min_avg_volume: float = Field(
        default=500000.0,
        ge=50000.0,
        le=10000000.0,
        description="Minimum average daily volume for VWAP universe",
    )
    vwap_universe_min_atr_pct: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Minimum ATR as % of price for VWAP universe (filters out dead stocks)",
    )
    vwap_universe_max_scan_tickers: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Max tickers to fetch intraday bars for per scan cycle (API budget)",
    )

    # Intraday reserve scan (SMA-filtered, PM-style analysis) for late entries.
    intraday_reserve_scan_enabled: bool = Field(default=True)
    intraday_reserve_scan_delay_minutes: int = Field(
        default=60,
        ge=10,
        le=240,
        description="Minutes after market open before intraday reserve scan runs",
    )
    intraday_reserve_min_score: float = Field(
        default=55.0,
        ge=0.0,
        le=100.0,
        description="Minimum score for intraday reserve candidates",
    )
    intraday_reserve_size_multiplier: float = Field(
        default=1.0,
        ge=0.25,
        le=3.0,
        description="Position size multiplier for intraday reserve entries",
    )
    intraday_reserve_stocktwits_score_weight: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Score weight applied to Stocktwits sentiment for reserve candidates",
    )
    intraday_reserve_signal_validation_enabled: bool = Field(default=True)
    intraday_reserve_fallback_enabled: bool = Field(default=True)
    intraday_reserve_recent_days: int = Field(default=10, ge=1, le=60)
    intraday_reserve_seed_cap: int = Field(default=60, ge=5, le=250)
    strength_reentry_enabled: bool = Field(
        default=True,
        description="Enable live relative-strength reclaim entries during bearish/choppy sessions",
    )
    strength_reentry_active_regimes: List[str] = Field(
        default_factory=lambda: [
            "CHOP",
            "SELLOFF",
            "SELL_OFF",
            "CRISIS",
            "RISK_OFF",
            "BEARISH",
            "DISTRIBUTION",
            "LEAN_BEARISH",
        ]
    )
    strength_reentry_scan_interval_cycles: int = Field(default=3, ge=1, le=30)
    strength_reentry_first_hour_minutes: int = Field(default=60, ge=15, le=120)
    strength_reentry_min_open_to_hour_gain_pct: float = Field(
        default=1.0, ge=0.1, le=20.0
    )
    strength_reentry_min_relative_strength_pct: float = Field(
        default=1.25, ge=0.1, le=20.0
    )
    strength_reentry_held_min_open_to_hour_gain_pct: float = Field(
        default=-1.0, ge=-10.0, le=20.0
    )
    strength_reentry_held_min_current_relative_strength_pct: float = Field(
        default=2.0, ge=0.1, le=30.0
    )
    strength_reentry_held_min_day_return_pct: float = Field(
        default=0.25, ge=-10.0, le=20.0
    )
    strength_reentry_held_near_reclaim_tolerance_pct: float = Field(
        default=0.2, ge=0.0, le=2.0
    )
    strength_reentry_follow_through_minutes: int = Field(default=3, ge=0, le=30)
    strength_reentry_follow_through_hold_above_reclaim_pct: float = Field(
        default=0.0, ge=0.0, le=5.0
    )
    strength_reentry_learning_block_override_enabled: bool = Field(default=True)
    strength_reentry_max_pullback_pct: float = Field(default=3.25, ge=0.25, le=15.0)
    strength_reentry_min_reclaim_volume_ratio: float = Field(
        default=1.15, ge=0.5, le=10.0
    )
    claw_long_exception_require_above_vwap: bool = Field(default=True)
    claw_long_exception_min_volume_ratio: float = Field(default=1.15, ge=0.5, le=10.0)
    claw_long_exception_min_relative_strength_pct: float = Field(
        default=1.25, ge=0.1, le=20.0
    )
    strength_reentry_new_entry_size_multiplier: float = Field(
        default=0.55, ge=0.1, le=2.0
    )
    strength_reentry_add_size_multiplier: float = Field(default=0.5, ge=0.1, le=2.0)
    strength_reentry_max_new_entries_per_day: int = Field(default=3, ge=1, le=20)
    strength_reentry_max_adds_per_day: int = Field(default=3, ge=1, le=20)
    bearish_session_min_deployed_pct: float = Field(default=0.50, ge=0.0, le=1.0)
    bearish_session_floor_counts_inverse_exposure: bool = Field(default=True)
    no_fill_watchdog_cycles: int = Field(default=2, ge=1, le=20)
    no_fill_escalation_cycles: int = Field(default=4, ge=1, le=40)
    no_fill_recheck_top_n: int = Field(default=5, ge=1, le=20)
    no_fill_override_top_n: int = Field(default=3, ge=1, le=20)
    no_fill_override_max_entries: int = Field(default=2, ge=1, le=20)
    replacement_min_hold_minutes: int = Field(default=120, ge=0, le=1440)


class ResearchFreshnessConfig(BaseModel):
    """Freshness controls for overnight research artifacts."""

    enabled: bool = Field(default=True)
    weekday_max_age_hours: float = Field(default=18.0, ge=1.0, le=72.0)
    monday_max_age_hours: float = Field(default=60.0, ge=12.0, le=120.0)
    premarket_max_age_hours: float = Field(default=18.0, ge=1.0, le=72.0)
    premarket_previous_day_cutoff_hour_et: float = Field(default=18.5, ge=0.0, le=23.99)
    premarket_previous_day_cutoff_override: bool = Field(default=True)
    strict_weekday_max_age_hours: float = Field(default=6.0, ge=1.0, le=24.0)
    strict_weekday_start_hour_et: float = Field(default=9.5, ge=0.0, le=23.99)
    warning_age_hours: float = Field(default=24.0, ge=1.0, le=120.0)

    # Sunday evening auto-refresh policy (ET).
    sunday_refresh_hour_et: int = Field(default=18, ge=0, le=23)
    sunday_refresh_min_age_hours: float = Field(default=48.0, ge=1.0, le=168.0)

    # Risk degradation once research drifts beyond warning threshold.
    stale_penalty_age_hours: float = Field(default=24.0, ge=1.0, le=120.0)
    stale_score_threshold_penalty: float = Field(default=5.0, ge=0.0, le=40.0)
    stale_position_size_multiplier: float = Field(default=0.85, ge=0.1, le=1.0)

    # Premarket behavior when freshness guard fails.
    premarket_block_if_stale: bool = Field(default=True)


class PremarketGapPolicyConfig(BaseModel):
    """Gap handling policy before market-open execution."""

    enabled: bool = Field(default=True)

    # Hard rejects.
    extreme_gap_up_pct: float = Field(default=10.0, ge=0.0, le=50.0)
    extreme_gap_down_pct: float = Field(default=-5.0, ge=-50.0, le=0.0)

    # Moderate gaps are repriced, not dropped.
    moderate_gap_up_pct: float = Field(default=3.0, ge=0.0, le=30.0)
    moderate_gap_down_pct: float = Field(default=-2.0, ge=-30.0, le=0.0)
    reprice_up_multiplier: float = Field(default=0.99, ge=0.8, le=1.2)
    reprice_down_multiplier: float = Field(default=1.01, ge=0.8, le=1.2)


class PremarketScalpConfig(BaseModel):
    """Premarket scalp lane: opportunistic scalps on extreme-gap-up names.

    Hard rule: every position is liquidated before 09:30 ET. Nothing
    held into the regular session. Paper account by intent.
    """

    enabled: bool = Field(default=True)
    paper_only: bool = Field(default=True)
    max_concurrent_positions: int = Field(default=5, ge=0, le=20)
    max_orders_per_day: int = Field(default=10, ge=0, le=50)
    size_usd: float = Field(default=1500.0, ge=100.0, le=20000.0)
    min_gap_pct: float = Field(default=5.0, ge=0.0, le=50.0)
    min_volume_ratio: float = Field(default=3.0, ge=0.0, le=20.0)
    min_liquidity_score: float = Field(default=70.0, ge=0.0, le=100.0)
    min_entry_minutes_before_open: int = Field(default=60, ge=1, le=180)
    max_spread_pct: float = Field(default=1.0, ge=0.0, le=10.0)
    trailing_stop_pct: float = Field(default=1.5, ge=0.0, le=20.0)
    profit_take_pct: float = Field(default=5.0, ge=0.0, le=50.0)
    require_sector_alignment: bool = Field(default=True)
    # 3-tier escalating exit before regular open (09:30 ET):
    # force >= aggressive >= emergency. Each is the *minutes before
    # open* at which that tier becomes active.
    force_exit_minutes_before_open: int = Field(default=15, ge=1, le=30)
    aggressive_exit_minutes_before_open: int = Field(default=10, ge=1, le=20)
    emergency_exit_minutes_before_open: int = Field(default=1, ge=0, le=10)
    # day-manager 2026-05-19 (H1): redesigned entry window + dynamic exits.
    # Entry window is T-90 .. T-45 minutes before open (no entries inside T-45).
    entry_window_start_minutes_before_open: int = Field(default=90, ge=15, le=240)
    entry_window_end_minutes_before_open: int = Field(default=45, ge=5, le=180)
    # Combined momentum gate: ALL must pass for entry.
    momentum_min_consec_up_bars: int = Field(default=3, ge=0, le=10)
    momentum_min_volume_ratio: float = Field(default=1.5, ge=0.0, le=20.0)
    momentum_max_vwap_distance_pct: float = Field(default=1.5, ge=0.0, le=20.0)
    # Per-position dynamic exits (in addition to the timer ladder above).
    hard_stop_pct: float = Field(default=1.0, ge=0.0, le=20.0)
    trailing_arm_at_pct: float = Field(default=1.5, ge=0.0, le=20.0)
    max_hold_minutes: int = Field(default=25, ge=1, le=120)
    # Daily circuit-breaker (disables lane for the session when fired).
    circuit_breaker_consecutive_losses: int = Field(default=3, ge=1, le=20)
    circuit_breaker_daily_loss_dollars: float = Field(default=-200.0, le=0.0)
    telemetry_path: str = Field(default="logs/premarket_scalp_telemetry.jsonl")


class OvernightCutPolicyConfig(BaseModel):
    """T-15 weak-position overnight-risk trim policy."""

    enabled: bool = Field(default=True)
    recovery_boost: float = Field(default=5.0, ge=0.0, le=50.0)
    trigger_minutes_before_close: int = Field(default=15, ge=1, le=120)
    latest_minutes_before_close: int = Field(default=2, ge=0, le=30)
    weak_entry_plus_minutes: int = Field(default=30, ge=1, le=240)
    weak_entry_plus_threshold_pct: float = Field(default=-1.0, ge=-50.0, le=50.0)
    trim_fraction: float = Field(default=0.50, ge=0.0, le=1.0)
    min_hold_conviction_score: float = Field(default=90.0, ge=0.0, le=100.0)
    min_remaining_position_value: float = Field(default=100.0, ge=0.0, le=10000.0)
    decision_claw_override_mode: str = Field(default="shadow_only")


class PremarketManagerConfig(BaseModel):
    """Dedicated premarket manager configuration."""

    enabled: bool = Field(default=True)
    max_watchlist_symbols: int = Field(default=400, ge=5, le=500)
    use_news: bool = Field(default=True)
    use_stocktwits: bool = Field(default=True)
    report_basename: str = Field(default="morning_intelligence")

    # Watchlist score component weights.
    weight_gap: float = Field(default=1.0, ge=0.0, le=5.0)
    weight_volume: float = Field(default=1.0, ge=0.0, le=5.0)
    weight_news: float = Field(default=1.0, ge=0.0, le=5.0)
    weight_stocktwits: float = Field(default=0.75, ge=0.0, le=5.0)
    weight_vwap: float = Field(default=1.0, ge=0.0, le=5.0)
    weight_sr: float = Field(default=0.8, ge=0.0, le=5.0)


class MomentumScannerConfig(BaseModel):
    """Continuous live momentum watchlist builder."""

    enabled: bool = Field(default=True)
    artifact_path: str = Field(default="data/momentum_watchlist_live.json")
    start_time_ct: str = Field(default="06:00")
    premarket_interval_seconds: int = Field(default=90, ge=15, le=900)
    intraday_interval_seconds: int = Field(default=180, ge=15, le=1800)
    sleep_outside_window_seconds: int = Field(default=300, ge=30, le=3600)
    max_candidates: int = Field(default=100, ge=1, le=400)
    max_live_scan_symbols: int = Field(default=400, ge=25, le=5000)
    max_news_candidates: int = Field(default=25, ge=0, le=200)
    reserved_slot_cap: int = Field(default=10, ge=0, le=50)
    dynamic_reserved_slots_enabled: bool = Field(default=True)
    max_reserved_slot_cap: int = Field(default=15, ge=0, le=50)
    empty_artifact_fallback_minutes: int = Field(default=15, ge=1, le=240)
    min_price: float = Field(default=2.0, ge=0.1, le=500.0)
    max_price: float = Field(default=200.0, ge=1.0, le=1000.0)
    min_avg_volume: float = Field(default=200000.0, ge=0.0)
    min_premarket_gain_pct: float = Field(default=2.5, ge=0.0, le=100.0)
    min_intraday_gain_pct: float = Field(default=2.5, ge=0.0, le=100.0)
    # Scanner internal min_score is a permissive floor; the entry-quality
    # stack (unknown_source_score_band_block, catalyst-required at 75+)
    # handles the upper-band loser zone surfaced in the broker+signals
    # analysis (2026-05-15 EOD review, final_score 75-85 = -$3,252).
    min_score: float = Field(default=60.0, ge=0.0, le=100.0)
    min_short_momentum_pct: float = Field(default=0.4, ge=-20.0, le=50.0)
    min_rel_volume: float = Field(default=1.5, ge=0.1, le=20.0)
    min_trend_strength: float = Field(default=0.55, ge=0.0, le=1.0)
    min_premarket_volume: int = Field(default=100000, ge=0)
    fade_reject_pct: float = Field(default=4.0, ge=0.0, le=50.0)
    extreme_gap_pct: float = Field(default=12.0, ge=0.0, le=100.0)
    require_catalyst_for_extreme_gap_names: bool = Field(default=True)
    stale_after_seconds: int = Field(default=420, ge=30, le=7200)
    scan_minutes_back_premarket: int = Field(default=390, ge=30, le=1440)
    scan_minutes_back_intraday: int = Field(default=390, ge=30, le=1440)
    alpaca_batch_size: int = Field(default=25, ge=1, le=200)
    alpaca_pause_seconds: float = Field(default=1.0, ge=0.0, le=60.0)


class ScreenerV2Config(BaseModel):
    """Multi-factor screener v2 configuration."""

    enabled: bool = Field(default=True)
    default_scoring_mode: str = Field(default="momentum_pullback")
    max_candidates: int = Field(default=200)
    min_composite_score: float = Field(default=60.0)
    max_composite_score: float = Field(
        default=0.0,
        ge=0.0,
        description="Optional high-score trap veto; 0 disables the upper bound.",
    )
    max_composite_score_exempt_regimes: List[str] = Field(
        default_factory=lambda: ["STRONG_RALLY"],
        description="Market/regime labels exempt from the high-score trap veto.",
    )
    max_price: float = Field(default=200.0)
    min_atr_pct: float = Field(default=2.5)
    max_atr_pct: float = Field(default=8.0)
    min_price: float = Field(default=5.0)
    min_market_cap: float = Field(default=2_000_000_000.0)
    max_market_cap: float = Field(default=10_000_000_000.0)
    post_spike_volume_threshold: float = Field(default=5.0, ge=0.0)
    post_spike_range_atr_threshold: float = Field(default=1.5, ge=0.0)
    min_avg_dollar_volume: float = Field(default=500_000.0)
    min_avg_volume: int = Field(default=100_000)
    volume_lookback_days: int = Field(default=20)
    five_day_range_min_pct: float = Field(default=3.0)
    sr_data_max_age_days: int = Field(default=5)
    sr_round_number_tolerance: float = Field(default=0.01)

    sma5_min_slope: float = Field(default=0.05)
    sma5_min_accel: float = Field(default=0.0)
    rsi_min: float = Field(default=30.0)
    rsi_max: float = Field(default=60.0)
    momentum_roc_min_score: float = Field(default=40.0)
    rsi_pullback_min_score: float = Field(default=35.0)
    min_adx_for_trend_bonus: float = Field(default=20.0, ge=0.0, le=60.0)
    prefer_bullish_regime: bool = Field(default=True)

    history_days: int = Field(
        default=260, description="Lookback window for indicator calculations"
    )
    min_history_days: int = Field(
        default=80, description="Minimum rows per ticker to score"
    )
    enforce_min_candidates: bool = Field(default=True)

    # Factor weights (sum to 1.0)
    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "sma5_curl": 0.15,
            "momentum_roc": 0.15,
            "relative_strength": 0.05,
            "rs_vs_sector": 0.05,
            "volume_surge": 0.10,
            "ema_alignment": 0.10,
            "rsi_pullback": 0.05,
            "regime": 0.05,
            "sr_alignment": 0.05,
            "adx_trend_quality": 0.15,
            "volume_divergence": 0.05,
            "mean_reversion": 0.05,
        }
    )
    sector_relative_strength_lookbacks: List[int] = Field(
        default_factory=lambda: [5, 20]
    )
    sector_etf_map: Dict[str, str] = Field(
        default_factory=lambda: {
            "basic_materials": "XLB",
            "communication_services": "XLC",
            "communications": "XLC",
            "consumer_cyclical": "XLY",
            "consumer_discretionary": "XLY",
            "consumer_staples": "XLP",
            "energy": "XLE",
            "financial": "XLF",
            "financials": "XLF",
            "health_care": "XLV",
            "healthcare": "XLV",
            "industrials": "XLI",
            "real_estate": "XLRE",
            "technology": "XLK",
            "utilities": "XLU",
        }
    )

    # Scoring scales
    sma5_curl_scale: float = Field(default=1.5)
    sma20_trend_scale: float = Field(default=8.0)
    macd_hist_scale: float = Field(default=1000.0)
    momentum_scale: float = Field(default=5.0)
    gap_penalty_per_pct: float = Field(default=5.0)
    gap_negative_penalty: float = Field(default=5.0)

    # SR bonus thresholds
    sr_support_strength_min: float = Field(default=50.0)
    sr_distance_to_resistance_min_pct: float = Field(default=4.0)
    sr_rr_ratio_min: float = Field(default=2.0)
    sr_rr_discount_factor: float = Field(default=0.6, ge=0.1, le=1.0)
    sr_rr_effective_min: float = Field(default=1.2, ge=0.5, le=5.0)
    sr_support_distance_atr_max: float = Field(default=1.5, ge=0.1, le=5.0)
    sr_max_weight: float = Field(
        default=0.10,
        ge=0.0,
        le=0.50,
        description="Hard cap for S/R contribution weight",
    )
    sr_action_plan_keywords: List[str] = Field(
        default_factory=lambda: ["bullish", "buy", "accumulate", "trend", "breakout"]
    )
    sr_bonus_cap: float = Field(default=10.0)
    min_atr_risk_reward: float = Field(
        default=2.0, ge=1.0, le=5.0, description="Minimum ATR-based R:R"
    )


class UniverseScannerConfig(BaseModel):
    """DuckDB universe scanner configuration."""

    max_candidates: int = Field(default=200)
    min_atr_percent: float = Field(default=1.5)
    preferred_atr_range: List[float] = Field(default_factory=lambda: [2.5, 6.0])
    high_risk_atr: float = Field(default=8.0)
    min_avg_volume: int = Field(default=200_000)
    min_price: float = Field(default=2.0)
    max_price: float = Field(default=200.0)
    max_data_age_days: int = Field(default=3)

    # Dynamic signal parameters (exposed for autonomous improvement)
    min_volume_ratio: float = Field(default=1.5)
    rsi_min: float = Field(default=40.0)
    rsi_max: float = Field(default=75.0)
    min_weekly_return: float = Field(default=3.0)
    min_gap_percent: float = Field(default=3.0)
    min_market_cap: float = Field(default=2_000_000_000.0)
    max_market_cap: float = Field(default=10_000_000_000.0)
    post_spike_volume_threshold: float = Field(default=5.0, ge=0.0)
    post_spike_range_atr_threshold: float = Field(default=1.5, ge=0.0)

    scan_types: List[str] = Field(
        default_factory=lambda: [
            "momentum_breakout",
            "mean_reversion",
            "earnings_momentum",
        ]
    )
    score_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "momentum": 0.30,
            "volume": 0.25,
            "volatility": 0.20,
            "trend": 0.15,
            "pullback": 0.10,
        }
    )


class StrategyBacktesterConfig(BaseModel):
    """Strategy backtester calibration and runtime thresholds."""

    cache_staleness_hours: float = Field(
        default=24.0, ge=1.0, description="Cache staleness window for OHLCV data"
    )
    min_trades_for_validity: int = Field(
        default=12, ge=1, description="Minimum trades to accept backtest metrics"
    )
    gate_win_rate: float = Field(
        default=40.0, description="Gate threshold for win rate percentage"
    )
    gate_min_trades: int = Field(
        default=12,
        ge=1,
        description="Minimum trades required before win-rate gating applies",
    )
    boost_win_rate: float = Field(
        default=55.0, description="Boost threshold for win rate percentage"
    )
    boost_min_trades: int = Field(
        default=20,
        ge=1,
        description="Minimum trades required before win-rate boost applies",
    )
    batch_prefetch_size: int = Field(
        default=200, ge=1, description="Tickers per prefetch batch"
    )
    yfinance_rate_limit: int = Field(
        default=5, ge=1, description="Max tickers per second for yfinance"
    )

    # Deprecated PF/Sharpe gates (kept for backward-compatible config loading)
    gate_profit_factor: float = Field(
        default=0.8, description="Deprecated: profit-factor gate"
    )
    gate_sharpe: float = Field(default=-0.5, description="Deprecated: Sharpe gate")
    boost_profit_factor: float = Field(
        default=2.0, description="Deprecated: profit-factor boost"
    )

    # Scoring thresholds for MultiModelScorer backtest component
    score_win_rate_high: float = Field(
        default=60.0, description="Win-rate high threshold"
    )
    score_win_rate_mid: float = Field(
        default=50.0, description="Win-rate mid threshold"
    )
    score_win_rate_low: float = Field(
        default=40.0, description="Win-rate low threshold"
    )
    score_win_rate_weak: float = Field(
        default=30.0, description="Win-rate weak threshold"
    )

    score_avg_gain_high: float = Field(
        default=2.0, description="Avg gain high threshold"
    )
    score_avg_gain_mid: float = Field(default=1.0, description="Avg gain mid threshold")
    score_avg_gain_low: float = Field(default=0.0, description="Avg gain low threshold")
    score_avg_gain_penalty: float = Field(
        default=-1.0, description="Avg gain penalty threshold"
    )
    score_backtest_weight_no_vl: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Backtest score weight when VL is not used",
    )
    score_backtest_weight_with_vl: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Backtest score weight when VL is used",
    )


class BacktestConfig(BaseModel):
    """Backtesting configuration."""

    bar_size: str = Field(default="1h")
    commission_pct: float = Field(default=0.001)
    slippage_pct: float = Field(default=0.0005)
    spread_pct: float = Field(
        default=0.001, description="Total bid-ask spread (fraction, e.g., 0.001 = 0.1%)"
    )
    partial_fill_min: float = Field(default=1.0, ge=0.0, le=1.0)
    partial_fill_max: float = Field(default=1.0, ge=0.0, le=1.0)
    seed: int = Field(default=42)
    initial_cash: float = Field(
        default=100_000.0, description="Starting cash for portfolio-level backtests"
    )
    benchmark_symbol: str = Field(
        default="SPY",
        description="Benchmark buy-and-hold symbol (must exist in local data)",
    )

    # Fast backtest defaults
    signals_per_day: int = Field(default=100)
    target_atr: float = Field(default=3.0)
    stop_atr: float = Field(default=2.0)
    min_lesson_score: float = Field(default=20.0)
    min_bullish: float = Field(default=4.0)

    # Realistic backtest defaults
    max_positions: int = Field(default=10)
    max_position_value: float = Field(
        default=10_000.0, description="Cap per-position notional value"
    )
    max_hold_days: int = Field(default=5)
    use_news: bool = Field(default=True)
    enable_weekly_signal_audit: bool = Field(
        default=True, description="Run weekly signal audit automatically"
    )

    # Risk metrics
    risk_free_rate: float = Field(
        default=0.05, description="Annual risk-free rate for Sharpe"
    )
    min_sharpe_ratio: float = Field(
        default=1.0, description="Minimum acceptable Sharpe"
    )
    max_drawdown_pct: float = Field(
        default=20.0, description="Maximum acceptable drawdown"
    )

    # Walk-forward testing
    walk_forward_train_days: int = Field(default=60)
    walk_forward_test_days: int = Field(default=20)

    # Strategy backtester calibration thresholds
    strategy_backtester: StrategyBacktesterConfig = Field(
        default_factory=StrategyBacktesterConfig
    )


class WalkForwardConfig(BaseModel):
    """Walk-forward testing configuration."""

    mode: str = Field(default="rolling", description="rolling|expanding")
    train_days: int = Field(default=90, ge=30, le=365)
    test_days: int = Field(default=30, ge=5, le=90)


class NestedValidationConfig(BaseModel):
    """Nested cross-validation configuration."""

    enabled: bool = Field(default=True)
    inner_folds: int = Field(default=3, ge=2, le=10)


class TransactionCostSensitivityConfig(BaseModel):
    """Transaction cost sensitivity analysis configuration."""

    commission_pct: List[float] = Field(default_factory=lambda: [0.0005, 0.001, 0.002])
    slippage_pct: List[float] = Field(default_factory=lambda: [0.0002, 0.0005, 0.001])
    spread_pct: List[float] = Field(default_factory=lambda: [0.0005, 0.001, 0.002])


class LeakageGuardConfig(BaseModel):
    """Anti-leakage guard configuration."""

    fail_on_detected_leakage: bool = Field(default=True)


class PaperGateConfig(BaseModel):
    """Paper trading gate configuration."""

    min_out_of_sample_trades: int = Field(default=30, ge=10)


class MultipleTestingControlConfig(BaseModel):
    """Multiple testing correction configuration."""

    method: str = Field(default="dsr+spa", description="bonferroni|bh|dsr+spa")
    min_candidate_count: int = Field(default=20, ge=5)
    bootstrap_samples: int = Field(default=1000, ge=100)
    alpha: float = Field(default=0.05, ge=0.01, le=0.2)


class SelectionBiasControlConfig(BaseModel):
    """Selection bias control configuration."""

    enable_pbo: bool = Field(default=True)
    pbo_min_folds: int = Field(default=8, ge=4, le=20)


class InferenceConfig(BaseModel):
    """Statistical inference configuration."""

    robust_sharpe_test: str = Field(default="bootstrap", description="bootstrap|hac")
    robust_sharpe_min_pvalue: float = Field(default=0.05, ge=0.01, le=0.2)


class BacktestProtocolConfig(BaseModel):
    """Canonical backtest protocol configuration."""

    enabled: bool = Field(default=True)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    nested_validation: NestedValidationConfig = Field(
        default_factory=NestedValidationConfig
    )
    transaction_cost_sensitivity: TransactionCostSensitivityConfig = Field(
        default_factory=TransactionCostSensitivityConfig
    )
    leakage_guard: LeakageGuardConfig = Field(default_factory=LeakageGuardConfig)
    paper_gate: PaperGateConfig = Field(default_factory=PaperGateConfig)
    multiple_testing_control: MultipleTestingControlConfig = Field(
        default_factory=MultipleTestingControlConfig
    )
    selection_bias_control: SelectionBiasControlConfig = Field(
        default_factory=SelectionBiasControlConfig
    )
    inference: InferenceConfig = Field(default_factory=InferenceConfig)


class ExecutionSimConfig(BaseModel):
    """Simulator execution configuration."""

    seed: int = Field(default=42)
    latency_ms_min: int = Field(default=50, ge=0)
    latency_ms_max: int = Field(default=500, ge=0)
    partial_fill_enabled: bool = Field(default=True)
    cannot_fill_probability: float = Field(default=0.02, ge=0.0, le=1.0)


class ExecutionCostConfig(BaseModel):
    """Execution-cost assumptions for sim/live parity checks."""

    commission_pct: float = Field(default=0.001, ge=0.0)
    slippage_pct: float = Field(default=0.0005, ge=0.0)
    spread_pct: float = Field(default=0.001, ge=0.0)


class ExecutionUrgencyTierConfig(BaseModel):
    """Per-tier entry urgency limits."""

    max_chase_bps: int = Field(default=15, ge=0, le=500)
    max_slippage_bps: int = Field(default=20, ge=0, le=1000)


class ExecutionEntryPolicyConfig(BaseModel):
    """Entry execution policy for limit-first with optional urgency escalation."""

    enabled: bool = Field(default=True)
    high_urgency_min_score: float = Field(default=72.0, ge=0.0, le=100.0)
    critical_urgency_min_score: float = Field(default=84.0, ge=0.0, le=100.0)
    replace_schedule_seconds: List[int] = Field(default_factory=lambda: [20, 45, 90])
    max_replacements: int = Field(default=3, ge=0, le=12)
    min_reprice_gap_pct: float = Field(default=0.2, ge=0.0, le=10.0)
    stale_after_seconds: int = Field(default=300, ge=30, le=3600)
    escalate_to_marketable_min_score: float = Field(default=85.0, ge=0.0, le=100.0)
    escalate_to_marketable_min_dollar_vol: float = Field(default=2500000.0, ge=0.0)
    normal: ExecutionUrgencyTierConfig = Field(
        default_factory=lambda: ExecutionUrgencyTierConfig(
            max_chase_bps=8,
            max_slippage_bps=12,
        )
    )
    high: ExecutionUrgencyTierConfig = Field(
        default_factory=lambda: ExecutionUrgencyTierConfig(
            max_chase_bps=18,
            max_slippage_bps=20,
        )
    )
    critical: ExecutionUrgencyTierConfig = Field(
        default_factory=lambda: ExecutionUrgencyTierConfig(
            max_chase_bps=30,
            max_slippage_bps=30,
        )
    )

    @field_validator("replace_schedule_seconds")
    @classmethod
    def validate_replace_schedule(cls, value: List[int]) -> List[int]:
        cleaned = sorted({int(v) for v in (value or []) if int(v) > 0})
        if not cleaned:
            return [20, 45, 90]
        return cleaned


class ExecutionConfig(BaseModel):
    """Unified execution adapter configuration."""

    mode: str = Field(default="paper", description="live|paper|sim")
    default_order_type: str = Field(default="market", description="market|limit")
    sim: ExecutionSimConfig = Field(default_factory=ExecutionSimConfig)
    cost: ExecutionCostConfig = Field(default_factory=ExecutionCostConfig)
    entry: ExecutionEntryPolicyConfig = Field(
        default_factory=ExecutionEntryPolicyConfig
    )
    fail_fast_on_unhandled_order_state: bool = Field(default=True)

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"live", "paper", "sim"}:
            raise ValueError("execution.mode must be one of: live, paper, sim")
        return normalized

    @field_validator("default_order_type")
    @classmethod
    def validate_order_type(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"market", "limit"}:
            raise ValueError(
                "execution.default_order_type must be one of: market, limit"
            )
        return normalized

    @model_validator(mode="after")
    def validate_latency_window(self):
        if self.sim.latency_ms_max < self.sim.latency_ms_min:
            raise ValueError("execution.sim.latency_ms_max must be >= latency_ms_min")
        return self


class StrategyLabMonteCarloConfig(BaseModel):
    """Bootstrap confidence-interval controls for walk-forward validation."""

    enabled: bool = Field(default=True)
    n_simulations: int = Field(default=1000, ge=100, le=20000)
    confidence_level: float = Field(default=0.80, ge=0.50, le=0.99)
    min_ci_profit_factor: float = Field(default=0.80, ge=0.0, le=5.0)
    random_seed: int = Field(default=42)


class StrategyLabConfig(BaseModel):
    """Interactive strategy-lab defaults and guardrails."""

    enabled: bool = Field(default=True)
    quick_scan_days: int = Field(default=90, ge=30)
    default_lookback_days: int = Field(default=180, ge=60)
    default_universe_size: int = Field(default=0, ge=0, le=5000)
    default_model: str = Field(default="phi4:14b-q4_K_M")
    min_profit_factor_export: float = Field(default=1.2)
    min_trades_export: int = Field(default=50, ge=1)
    walk_forward_min_test_pf: float = Field(default=0.90)
    walk_forward_max_degradation_pct: float = Field(default=40.0)
    default_train_pct: float = Field(default=0.70, ge=0.4, le=0.9)
    default_folds: int = Field(default=3, ge=2, le=10)
    auto_max_full_wf_candidates: int = Field(default=30, ge=5, le=500)
    auto_strict_validation_enabled: bool = Field(default=True)
    auto_strict_top_n: int = Field(default=5, ge=1, le=20)
    auto_walkforward_pf_floor: float = Field(default=0.85, ge=0.0, le=5.0)
    auto_early_reject_pf: float = Field(default=0.70, ge=0.0, le=5.0)
    auto_rich_search_enabled: bool = Field(default=True)
    auto_entry_combo_cap: int = Field(default=4, ge=1, le=12)
    auto_exit_combo_cap: int = Field(default=3, ge=1, le=12)
    validated_strategy_pool_top_n: int = Field(default=5, ge=1, le=20)
    validated_strategy_min_profit_factor: float = Field(default=1.05, ge=0.0, le=5.0)
    per_symbol_strategy_enabled: bool = Field(default=True)
    per_symbol_top_k: int = Field(default=5, ge=1, le=20)
    per_symbol_min_trades: int = Field(default=8, ge=1, le=5000)
    per_symbol_shrinkage_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    per_symbol_fallback_to_global: bool = Field(default=True)
    strict_require_walkforward_pass: bool = Field(default=True)
    strict_min_profit_factor: float = Field(default=1.00)
    strict_min_win_rate: float = Field(default=0.50, ge=0.0, le=1.0)
    strict_min_sharpe: float = Field(default=0.5)
    strict_max_drawdown_pct: float = Field(default=25.0, ge=0.0, le=100.0)
    strict_min_trades: int = Field(default=100, ge=1)
    strict_min_total_pnl: float = Field(default=0.0)
    monte_carlo: StrategyLabMonteCarloConfig = Field(
        default_factory=StrategyLabMonteCarloConfig
    )


class ReplayMinuteArchiveConfig(BaseModel):
    """Local minute-bar archive settings for runtime replay."""

    enabled: bool = Field(
        default=True,
        description="Persist end-of-day minute bars for replay",
    )
    duckdb_path: str = Field(
        default="data/replay_minute_bars.duckdb",
        description="DuckDB archive for replay minute bars",
    )
    preferred_start_et: str = Field(
        default="04:00",
        description="Preferred ET archive start time",
    )
    preferred_end_et: str = Field(
        default="16:00",
        description="Preferred ET archive end time",
    )
    fallback_regular_session_only: bool = Field(
        default=True,
        description="Allow 09:30-16:00 fallback when premarket is unavailable",
    )
    max_batch_symbols: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Max symbols processed per archive batch",
    )
    prefer_local_archive: bool = Field(
        default=True,
        description="Prefer local archive before live fetch during replay",
    )
    allow_live_fallback_if_incomplete: bool = Field(
        default=True,
        description="Allow live fetch when archive coverage is incomplete",
    )


class LLMConfig(BaseModel):
    """LLM configuration for agentic workflow."""

    provider: str = Field(default="local", description="LLM backend: codex or local")
    codex_command: str = Field(default="codex", description="Codex CLI command")
    codex_timeout: int = Field(default=120, description="Codex CLI timeout (seconds)")
    codex_use_stdin: bool = Field(
        default=False, description="Send prompt via stdin to Codex CLI"
    )
    codex_extra_args: List[str] = Field(
        default_factory=list, description="Extra args for Codex CLI"
    )
    codex_json: bool = Field(
        default=True, description="Stream Codex JSON events for debug output"
    )
    repair_openai_enabled: bool = Field(
        default=False, description="Allow OpenAI as a code-repair fallback"
    )
    repair_codex_enabled: bool = Field(
        default=False, description="Allow Codex CLI as a final code-repair fallback"
    )
    repair_qwen3_enabled: bool = Field(
        default=False,
        description="Allow qwen3-coder-next as an opt-in local code-repair fallback",
    )

    # Project Chat provider (OpenAI for conversation/coding)
    chat_provider: str = Field(
        default="openai", description="Project Chat provider: openai or local"
    )
    openai_fast_model: str = Field(
        default="gpt-4.1-mini", description="Fast general chat model"
    )
    openai_code_model: str = Field(
        default="gpt-4.1", description="Code/chat model for improvements/debug"
    )
    openai_reasoning_model: str = Field(
        default="o4-mini", description="Reasoning model for complex analysis"
    )
    openai_merge_model: str = Field(
        default="gpt-4.1", description="Merge/synthesis model"
    )

    use_opencode_for_fixes: bool = Field(default=True)
    ollama_url: str = Field(default="http://localhost:11434/api/chat")
    ollama_timeout: int = Field(default=60)

    # Task-specific model assignments (OPTIMIZED)
    model_technical: str = Field(
        default="qwen2.5-coder:7b", description="Technical analysis & JSON output"
    )
    model_news: str = Field(default="gemma4:e4b", description="News summarization")
    model_risk: str = Field(default="phi4:14b-q4_K_M", description="Risk reasoning")
    model_decision: str = Field(
        default="deepseek-coder-v2:16b", description="Final synthesis"
    )

    # Legacy compatibility (map to new task-specific models)
    model_small: str = Field(
        default="qwen2.5-coder:7b", description="Fast model for technical tasks"
    )
    model_medium: str = Field(
        default="gemma4:e4b", description="Medium model for risk assessment"
    )
    model_large: str = Field(
        default="deepseek-coder-v2:16b", description="Large model for final synthesis"
    )

    # Context settings
    num_ctx: int = Field(default=8192, description="Context window size")
    temperature: float = Field(default=0.1)

    # Parallelization
    max_concurrent_calls: int = Field(default=2, description="Max parallel LLM calls")

    # Caching
    enable_cache: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=1800)


class SearchConfig(BaseModel):
    """Web search provider failover configuration."""

    primary_host: str = Field(default="http://localhost:8080")
    backup_hosts: List[str] = Field(default_factory=list)
    timeout: int = Field(default=10, ge=1, le=120)
    max_retries_per_host: int = Field(default=1, ge=1, le=5)


class DataConfig(BaseModel):
    """Data sources configuration."""

    # Market data priority: local files first, API as fallback
    use_local_data_primary: bool = Field(
        default=True, description="Use DownDay local files as primary data source"
    )
    use_alpaca_data: bool = Field(
        default=True, description="Use Alpaca for current positions and real-time data"
    )
    yfinance_cache_ttl: int = Field(
        default=300, description="yfinance cache TTL in seconds"
    )
    live_price_provider_priority: List[str] = Field(
        default_factory=lambda: ["alpaca", "yfinance", "local"],
        description="Ordered provider priority for live price lookups",
    )
    asset_class: str = Field(
        default="equities_us", description="Primary traded asset class"
    )
    primary_timeframe: str = Field(default="1d", description="Primary bar timeframe")
    intraday_timeframe: str = Field(default="1m", description="Intraday bar timeframe")
    max_staleness_days: int = Field(
        default=1,
        ge=0,
        le=10,
        description="Allowed trading-day staleness for core data",
    )
    bootstrap_from_h5_enabled: bool = Field(
        default=True, description="Allow automatic H5 -> parquet bootstrap"
    )
    fail_fast_on_missing_core_data: bool = Field(
        default=True,
        description="Raise hard errors when core data is missing after bootstrap attempts",
    )
    use_ingestion_gateway: bool = Field(
        default=True,
        description="Enable startup bootstrap + runtime health checks via data-ingestion gateway",
    )

    # Backtest data paths. A fresh clone downloads public data into data/downday/.
    downday_root: str = Field(
        default="data/downday",
        description="Root directory for local backtest data",
    )
    nasdaq_screener_csv: str = Field(
        default="nasdaq_screener.csv",
        description="Canonical NASDAQ screener CSV with market cap and sector metadata",
    )
    daily_features_parquet: str = Field(
        default="daily_features.parquet", description="Daily features parquet file"
    )
    hourly_prices_parquet: str = Field(
        default="prices_hourly.parquet", description="Hourly prices parquet file"
    )
    daily_features_h5: str = Field(
        default="daily_features.h5", description="Daily features HDF5 file"
    )
    market_data_duckdb: str = Field(
        default="market_data.duckdb", description="DuckDB market data"
    )
    replay_minute_archive: ReplayMinuteArchiveConfig = Field(
        default_factory=ReplayMinuteArchiveConfig
    )

    # DuckDB for caching and metrics
    duckdb_path: str = Field(default="data/trading_cache.duckdb")

    # News caching
    news_cache_ttl_hours: int = Field(default=4, description="News cache TTL in hours")
    news_cache_db: str = Field(
        default="data/news_cache.db", description="News cache SQLite database"
    )

    # RAG sources
    rag_documents_dir: Optional[str] = Field(
        default=None, description="Directory with RAG documents"
    )
    rag_embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    @field_validator("live_price_provider_priority")
    @classmethod
    def validate_live_price_provider_priority(cls, value: List[str]) -> List[str]:
        cleaned = [str(v).strip().lower() for v in value if str(v).strip()]
        if not cleaned:
            return ["alpaca", "yfinance", "local"]
        deduped = []
        seen = set()
        for item in cleaned:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped


class RuntimeRecoveryConfig(BaseModel):
    """Guarded runtime recovery controls for supervised launches."""

    enabled: bool = Field(default=True)
    use_supervisor_wrapper: bool = Field(default=True)
    preflight_enabled: bool = Field(default=True)
    post_restart_smoke_enabled: bool = Field(default=True)
    allow_live_local_repair: bool = Field(default=False)
    allow_live_openai_repair: bool = Field(default=False)
    allow_live_codex_repair: bool = Field(default=False)
    restart_backoff_seconds: int = Field(default=20, ge=1, le=900)
    max_restarts_per_hour: int = Field(default=3, ge=0, le=20)
    max_restarts_per_session: int = Field(default=6, ge=0, le=50)
    fatal_signature_threshold: int = Field(default=3, ge=1, le=20)
    smoke_py_compile_changed_files: bool = Field(default=True)
    smoke_pytest_blocking: bool = Field(default=False)
    smoke_pytest_recent_commit_window: int = Field(default=12, ge=0, le=100)
    smoke_pytest_timeout_seconds: int = Field(default=600, ge=30, le=1800)
    smoke_pytest_modules: List[str] = Field(
        default_factory=lambda: [
            "tests/test_momentum_scanner.py",
            "tests/test_premarket_manager.py",
            "tests/test_day_manager_execution_policy.py",
            "tests/diagnostic/test_master_supervisor_validation.py",
            "tests/test_autonomous_agent_pm_healing_hook.py",
            "tests/test_autonomous_agent_workflow_state.py",
        ]
    )
    status_json_path: str = Field(default="logs/runtime_recovery_latest.json")
    status_jsonl_path: str = Field(default="logs/runtime_recovery.jsonl")


class ClawRemoteConfig(BaseModel):
    """Configuration for CLAW Remote mobile interface."""

    enabled: bool = Field(default=True)
    port: int = Field(default=8778, ge=1024, le=65535)
    command_dir: str = Field(default="data/claw_commands")
    poll_interval_sec: float = Field(default=2.0, ge=0.5, le=60.0)
    command_timeout_sec: int = Field(default=30, ge=5, le=300)
    max_history: int = Field(default=200, ge=10, le=1000)
    gemini_model_fast: str = Field(default="gemini-2.5-flash-lite")
    gemini_model_code: str = Field(default="gemini-2.5-pro")
    allowed_code_paths: List[str] = Field(
        default_factory=lambda: ["autotrade/", "config/", "tools/"]
    )


class WorkflowClawConfig(BaseModel):
    """LLM-powered autonomous workflow manager (WorkflowClaw)."""

    enabled: bool = Field(default=True)
    journal_path: str = Field(default="logs/workflow_claw_journal.jsonl")
    state_path: str = Field(default="data/workflow_claw_state.json")
    briefing_dir: str = Field(default="logs/morning_briefings")

    # LLM config
    primary_model: str = Field(
        default="phi4:14b-q4_K_M", description="Local Ollama model for reasoning"
    )
    fallback_model: str = Field(
        default="qwen2.5-coder:14b", description="Local fallback model"
    )
    openrouter_model: str = Field(
        default="anthropic/claude-3.5-sonnet",
        description="Cloud fallback when Ollama is down",
    )
    openai_model: str = Field(
        default="gpt-4.1-mini", description="Last resort cloud model"
    )
    max_agent_turns: int = Field(default=15, ge=3, le=50)
    num_ctx: int = Field(default=32768, ge=4096, le=131072)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)

    # Monitoring
    supervisor_poll_interval_sec: int = Field(default=10, ge=1, le=60)
    output_stale_timeout_sec: int = Field(
        default=3600, ge=120, le=7200, description="No output for this long = stalled"
    )
    artifact_check_interval_sec: int = Field(default=300, ge=60, le=1800)

    # Services
    services: Dict[str, Any] = Field(
        default_factory=lambda: {
            "ollama": {
                "health_url": "http://localhost:11434/api/tags",
                "restart_command": ["ollama", "serve"],
                "kill_command": ["taskkill", "/F", "/IM", "ollama.exe"],
                "restart_cooldown_sec": 30,
                "max_restart_attempts": 5,
                "health_timeout_sec": 10,
                "critical": True,
            },
            "searxng": {
                "health_url": "http://localhost:8080/healthz",
                "restart_command": ["docker", "start", "searxng"],
                "restart_cooldown_sec": 60,
                "max_restart_attempts": 3,
                "health_timeout_sec": 10,
                "critical": False,
            },
        }
    )

    # Safety
    max_recovery_attempts_per_hour: int = Field(default=10, ge=1, le=50)
    max_total_recovery_attempts: int = Field(default=50, ge=5, le=200)


class DecisionClawPhaseConfig(BaseModel):
    """Phase-specific DecisionClaw controls."""

    enabled: bool = Field(default=True)
    provider: str = Field(default="local")
    model: str = Field(default="phi4:14b-q4_K_M")
    fallback_provider: str = Field(default="local")
    fallback_model: str = Field(default="qwen2.5-coder:14b")
    max_symbols: int = Field(default=8, ge=1, le=50)
    max_output_tokens: int = Field(default=800, ge=128, le=8192)
    cooldown_seconds: int = Field(default=900, ge=0, le=86400)
    scheduled_interval_minutes: int = Field(default=30, ge=1, le=1440)
    paid_enabled: bool = Field(default=True)
    reasoning_effort: Optional[str] = Field(default=None)
    confidence_floor: float = Field(default=0.60, ge=0.0, le=1.0)


class DecisionClawBudgetConfig(BaseModel):
    """Budget and cache controls for DecisionClaw."""

    daily_max_calls: int = Field(default=24, ge=0, le=1000)
    daily_max_cost_usd: float = Field(default=10.0, ge=0.0, le=1000.0)
    packet_cache_ttl_seconds: int = Field(default=900, ge=0, le=86400)


class DecisionClawConfig(BaseModel):
    """Agentic phase controller with legacy-advisory inputs."""

    enabled: bool = Field(default=True)
    authority_mode: str = Field(default="full")
    prompt_profile_path: str = Field(
        default="config/decision_claw_prompt_profiles.json"
    )
    state_path: str = Field(default="data/decision_claw_state.json")
    decisions_log_path: str = Field(default="logs/decision_claw_decisions_{date}.jsonl")
    actions_log_path: str = Field(default="logs/decision_claw_actions_{date}.jsonl")
    cost_log_path: str = Field(default="logs/decision_claw_cost_{date}.jsonl")
    phase_snapshot_path: str = Field(
        default="plans/decision_claw_phase_snapshot_{date}.json"
    )
    final_watchlist_path: str = Field(
        default="plans/decision_claw_final_watchlist_{date}.json"
    )
    legacy_advisory_enabled: bool = Field(default=True)
    openai_primary: bool = Field(default=False)
    idle_capital_threshold_pct: float = Field(default=50.0, ge=0.0, le=100.0)
    redeployment_interval_minutes: int = Field(default=60, ge=5, le=240)
    redeployment_min_no_fill_streak: int = Field(default=2, ge=0, le=20)
    redeployment_candidate_limit: int = Field(default=3, ge=1, le=10)
    redeployment_entry_cap: int = Field(default=2, ge=1, le=5)
    top_n_floor: int = Field(default=3, ge=0, le=10)
    post_market: DecisionClawPhaseConfig = Field(
        default_factory=lambda: DecisionClawPhaseConfig(
            provider="local",
            model="qwen3:14b-q4_K_M",
            fallback_provider="local",
            fallback_model="phi4:14b-q4_K_M",
            max_symbols=10,
            max_output_tokens=500,
            scheduled_interval_minutes=1440,
        )
    )
    overnight: DecisionClawPhaseConfig = Field(
        default_factory=lambda: DecisionClawPhaseConfig(
            provider="openai",
            model="gpt-5",
            fallback_provider="local",
            fallback_model="qwen3:14b-q4_K_M",
            max_symbols=48,
            max_output_tokens=1400,
            scheduled_interval_minutes=360,
        )
    )
    premarket: DecisionClawPhaseConfig = Field(
        default_factory=lambda: DecisionClawPhaseConfig(
            provider="local",
            model="qwen3:14b-q4_K_M",
            fallback_provider="local",
            fallback_model="phi4:14b-q4_K_M",
            max_symbols=20,
            max_output_tokens=900,
            scheduled_interval_minutes=30,
        )
    )
    market_open: DecisionClawPhaseConfig = Field(
        default_factory=lambda: DecisionClawPhaseConfig(
            provider="openai",
            model="gpt-5",
            fallback_provider="local",
            fallback_model="qwen3:14b-q4_K_M",
            max_symbols=10,
            max_output_tokens=700,
            cooldown_seconds=600,
            scheduled_interval_minutes=15,
        )
    )
    market_state: DecisionClawPhaseConfig = Field(
        default_factory=lambda: DecisionClawPhaseConfig(
            provider="openai",
            model="gpt-5",
            fallback_provider="local",
            fallback_model="qwen3:14b-q4_K_M",
            max_symbols=10,
            max_output_tokens=600,
            cooldown_seconds=900,
            scheduled_interval_minutes=30,
        )
    )
    wave_checkpoint: DecisionClawPhaseConfig = Field(
        default_factory=lambda: DecisionClawPhaseConfig(
            provider="openai",
            model="gpt-5",
            fallback_provider="local",
            fallback_model="qwen3:14b-q4_K_M",
            max_symbols=8,
            max_output_tokens=450,
            cooldown_seconds=600,
            scheduled_interval_minutes=15,
        )
    )
    symbol_validation: DecisionClawPhaseConfig = Field(
        default_factory=lambda: DecisionClawPhaseConfig(
            provider="local",
            model="qwen3:14b-q4_K_M",
            fallback_provider="openai",
            fallback_model="gpt-5",
            max_symbols=1,
            max_output_tokens=500,
            cooldown_seconds=0,
            scheduled_interval_minutes=1,
        )
    )
    budget_controls: DecisionClawBudgetConfig = Field(
        default_factory=DecisionClawBudgetConfig
    )


class AlpacaConfig(BaseModel):
    """Alpaca API configuration."""

    api_key: Optional[str] = Field(default=None)
    secret_key: Optional[str] = Field(default=None)
    paper: bool = Field(default=True)

    # Advanced order types
    use_bracket_orders: bool = Field(
        default=True, description="Use OCO/bracket orders for stops"
    )
    use_trailing_stops: bool = Field(
        default=True, description="Use server-side trailing stops"
    )

    # Rate limiting
    max_orders_per_minute: int = Field(default=30)


class RiskMetricsConfig(BaseModel):
    """Portfolio risk metrics configuration."""

    # VaR settings
    var_confidence: float = Field(default=0.95, description="VaR confidence level")
    var_lookback_days: int = Field(default=252, description="VaR lookback period")
    var_method: str = Field(
        default="historical", description="historical, parametric, or monte_carlo"
    )

    # Position sizing
    max_var_per_position_pct: float = Field(
        default=2.0, description="Max VaR per position as % of portfolio"
    )
    max_portfolio_var_pct: float = Field(
        default=5.0, description="Max total portfolio VaR"
    )

    # Drawdown limits
    max_daily_loss_pct: float = Field(default=3.0)
    max_weekly_loss_pct: float = Field(default=5.0)

    # Volatility regime
    high_volatility_vix_threshold: float = Field(default=25.0)
    position_size_reduction_high_vol: float = Field(
        default=0.5, description="Reduce position size by this factor in high vol"
    )


class SentimentConfig(BaseModel):
    """Sentiment analysis configuration."""

    # News sentiment
    max_news_age_days: int = Field(default=90)
    finbert_model: str = Field(default="ProsusAI/finbert")

    # Stocktwits (optional)
    use_stocktwits: bool = Field(default=False)
    stocktwits_api_url: Optional[str] = Field(default=None)

    # Sentiment thresholds
    bullish_threshold: float = Field(default=0.3)
    bearish_threshold: float = Field(default=-0.3)


class SignalValidationConfig(BaseModel):
    """Signal validation via historical similar-signal backtesting."""

    enabled: bool = Field(
        default=True, description="Enable backtest validation of signals"
    )
    lookback_days: int = Field(
        default=60, description="Days of history to search for similar setups"
    )
    tolerance: float = Field(
        default=0.15, description="Tolerance for RSI/ATR matching (±15%)"
    )
    min_backtest_score: float = Field(
        default=40.0, description="Minimum score to pass validation"
    )
    min_similar_signals: int = Field(
        default=10, description="Minimum matches for confident validation"
    )
    score_weight_signal: float = Field(
        default=0.6, description="Weight for original signal score"
    )
    score_weight_backtest: float = Field(
        default=0.4, description="Weight for backtest score"
    )
    win_threshold_pct: float = Field(
        default=2.0,
        description="Absolute win threshold in percent if ATR-relative is disabled",
    )
    win_threshold_atr_relative: bool = Field(
        default=True, description="Use ATR%% of candidate as win hurdle"
    )
    win_threshold_atr_multiple: float = Field(
        default=1.0,
        description="Multiplier on ATR%% when win_threshold_atr_relative is true",
    )
    win_rate_full_credit: float = Field(
        default=0.337, description="Win-rate percentile for full credit (e.g., P80)"
    )
    avg_return_full_credit: float = Field(
        default=1.45, description="Avg 5D return percentile for full credit (e.g., P80)"
    )
    confidence_full_credit: int = Field(
        default=3876,
        description="Matches needed for full confidence credit after dedup",
    )
    gl_ratio_full_credit: float = Field(
        default=2.83, description="Gain/loss ratio for full credit"
    )
    penalty_win_rate_threshold: float = Field(
        default=0.25, description="Win-rate threshold that triggers a penalty"
    )
    penalty_win_rate_points: float = Field(
        default=10.0, description="Points deducted when win-rate penalty triggers"
    )
    penalty_avg_return_threshold: float = Field(
        default=-1.0, description="Avg 5D return threshold that triggers a penalty"
    )
    penalty_avg_return_points: float = Field(
        default=10.0, description="Points deducted when avg return penalty triggers"
    )
    penalty_gl_ratio_threshold: float = Field(
        default=0.5, description="Gain/loss ratio threshold that triggers a penalty"
    )
    penalty_gl_ratio_points: float = Field(
        default=5.0, description="Points deducted when gain/loss penalty triggers"
    )
    adaptive_tolerance_enabled: bool = Field(
        default=True, description="Shrink tolerance when match counts are high"
    )
    adaptive_tolerance_trigger_matches: int = Field(
        default=4000, description="Matches threshold to start shrinking tolerance"
    )
    adaptive_tolerance_min: float = Field(
        default=0.05, description="Minimum tolerance after shrinking"
    )
    adaptive_tolerance_shrink_factor: float = Field(
        default=0.5, description="Multiplier applied to base tolerance when shrinking"
    )
    regime_filter_enabled: bool = Field(
        default=False, description="Require market regime alignment for matches"
    )
    regime_lookback_days: int = Field(
        default=20, description="Lookback window for regime metric"
    )
    regime_min_trend: float = Field(
        default=0.0,
        description="Minimum trend/return threshold to consider regime bullish/neutral",
    )
    liquidity_filter_enabled: bool = Field(
        default=True, description="Exclude illiquid/low-priced names from matches"
    )
    min_dollar_volume: float = Field(
        default=2_000_000.0, description="Minimum dollar volume filter for matches"
    )
    min_price: float = Field(default=2.0, description="Minimum close price for matches")
    atr_filter_enabled: bool = Field(
        default=True, description="Bound ATR%% range for matches"
    )
    min_atr_pct: float = Field(
        default=1.0, description="Minimum ATR%% for matches when enabled"
    )
    max_atr_pct: float = Field(
        default=8.0, description="Maximum ATR%% for matches when enabled"
    )
    recency_filter_enabled: bool = Field(
        default=True, description="Limit matches to recent history"
    )
    recency_max_days: int = Field(
        default=365,
        description="Maximum age in days for matches when recency filter enabled",
    )
    gl_ratio_filter_enabled: bool = Field(
        default=True, description="Drop matches with poor gain/loss ratio"
    )
    gl_ratio_filter_min: float = Field(
        default=1.2, description="Minimum gain/loss ratio for a match to count"
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO")
    json_filename: str = Field(default="app.jsonl")
    max_bytes: int = Field(default=5_000_000)
    backup_count: int = Field(default=5)
    console: bool = Field(default=True)

    # Optional verbosity toggles used by some modules
    show_rule_triggers: bool = Field(default=True)
    show_score_breakdown: bool = Field(default=True)
    show_pdt_status: bool = Field(default=True)


class RiskLimitsConfig(BaseModel):
    """Workflow risk limits used by agentic state."""

    max_position_pct: float = Field(default=15.0)
    max_daily_loss_pct: float = Field(default=3.0)
    max_loss_to_atr: float = Field(default=2.0)
    min_hold_minutes: int = Field(default=30)
    offering_triggers_exit: bool = Field(default=True)
    dilution_triggers_exit: bool = Field(default=True)


class StrategyFailsafeLevelConfig(BaseModel):
    """Failsafe profile for a specific strategy-health level."""

    max_positions: int = Field(default=20, ge=0, le=60)
    min_conviction_hold: float = Field(default=50.0, ge=0.0, le=100.0)
    min_conviction_exit: float = Field(default=55.0, ge=0.0, le=100.0)
    position_size_pct: float = Field(
        default=3.0, ge=0.0, le=100.0, description="Target position size as % equity"
    )
    position_size_multiplier: float = Field(default=1.0, ge=0.0, le=2.0)
    stop_multiplier: float = Field(default=2.0, ge=0.1, le=5.0)
    halt_new_entries: bool = Field(default=False)
    message: Optional[str] = Field(default=None)


class StrategyFailsafeConfig(BaseModel):
    """Portfolio safety throttles driven by strategy validation + drawdown."""

    normal: StrategyFailsafeLevelConfig = Field(
        default_factory=StrategyFailsafeLevelConfig
    )
    degraded: StrategyFailsafeLevelConfig = Field(
        default_factory=lambda: StrategyFailsafeLevelConfig(
            max_positions=12,
            min_conviction_hold=55.0,
            min_conviction_exit=55.0,
            position_size_pct=2.0,
            position_size_multiplier=0.67,
            stop_multiplier=1.5,
            halt_new_entries=False,
        )
    )
    failing: StrategyFailsafeLevelConfig = Field(
        default_factory=lambda: StrategyFailsafeLevelConfig(
            max_positions=8,
            min_conviction_hold=60.0,
            min_conviction_exit=55.0,
            position_size_pct=1.5,
            position_size_multiplier=0.5,
            stop_multiplier=1.0,
            halt_new_entries=True,
        )
    )
    critical: StrategyFailsafeLevelConfig = Field(
        default_factory=lambda: StrategyFailsafeLevelConfig(
            max_positions=0,
            min_conviction_hold=70.0,
            min_conviction_exit=65.0,
            position_size_pct=0.0,
            position_size_multiplier=0.0,
            stop_multiplier=1.0,
            halt_new_entries=True,
            message="Strategy critically failing. All positions should be liquidated.",
        )
    )

    # Health thresholds
    degraded_win_rate: float = Field(default=0.35)
    degraded_profit_factor: float = Field(default=1.0)
    failing_win_rate: float = Field(default=0.30)
    failing_profit_factor: float = Field(default=0.8)
    critical_win_rate: float = Field(default=0.20)

    # Drawdown thresholds (percent from peak equity)
    degraded_drawdown_pct: float = Field(default=5.0)
    failing_drawdown_pct: float = Field(default=7.0)
    critical_drawdown_pct: float = Field(default=10.0)

    # Recovery conditions
    recovery_win_rate: float = Field(default=0.35)
    recovery_profit_factor: float = Field(default=1.0)
    recovery_days: int = Field(default=3, ge=1)

    # Emergency triage thresholds (failing/critical)
    loser_exit_pnl_pct: float = Field(default=-1.0)
    profit_take_pnl_pct: float = Field(default=10.0)

    # Persistence
    state_file: str = Field(default="plans/strategy_failsafe_state.json")
    no_trade_eval_minutes: int = Field(default=60, ge=1, le=240)
    no_trade_override_enabled: bool = Field(default=True)
    no_trade_override_cooldown_minutes: int = Field(default=60, ge=1, le=240)
    crash_regime_reduced_caps_enabled: bool = Field(default=True)
    error_cascade_cooldown_minutes: int = Field(default=15, ge=1, le=1440)
    error_cascade_stepdown_minutes: int = Field(default=15, ge=1, le=1440)
    error_cascade_min_quiet_cycles: int = Field(default=2, ge=1, le=100)


class RegimeStrategyConfig(BaseModel):
    """Strategy parameters for a specific market regime."""

    scan_types: List[str] = Field(
        default_factory=lambda: ["momentum_breakout", "mean_reversion"]
    )
    min_atr_percent: float = Field(default=3.0)
    position_size_multiplier: float = Field(default=1.0)
    max_positions: int = Field(default=25)
    hold_period_target: str = Field(default="3 days")
    stop_multiplier: float = Field(default=2.0)
    profit_target_multiplier: float = Field(default=2.5)
    entry_preference: str = Field(default="standard")
    cash_reserve_pct: int = Field(default=20)
    action: Optional[str] = Field(default=None)
    prefer_beaten_down: bool = Field(default=False)


class RiskManagementKillSwitchConfig(BaseModel):
    """Kill-switch thresholds nested under risk_management.kill_switch."""

    max_consecutive_failures: int = Field(default=3)
    max_realized_daily_loss_pct: float = Field(default=3.0)


class RiskManagementConfig(BaseModel):
    """Risk management policy configuration for fail-fast invariants and kill-switch."""

    enabled: bool = Field(default=True)
    fail_fast_on_limit_breach: bool = Field(default=True)

    # Exposure limits
    max_portfolio_exposure_pct: float = Field(default=100.0)
    max_symbol_exposure_pct: float = Field(default=15.0)
    max_sector_exposure_pct: float = Field(default=40.0)
    max_correlation_cluster_exposure_pct: float = Field(default=35.0)
    max_gross_exposure_pct: float = Field(default=150.0)
    max_net_exposure_pct: float = Field(default=100.0)

    # Drawdown
    entry_halt_drawdown_pct: float = Field(default=10.0)

    # Rotation
    rotation_cooldown_minutes: int = Field(default=30)
    cross_sectional_rebalance_frequency: str = Field(default="weekly")

    # Signal family budgets
    signal_family_risk_budgets: Dict[str, float] = Field(
        default_factory=lambda: {
            "ts_momentum": 0.25,
            "xs_momentum": 0.20,
            "mean_reversion": 0.15,
            "breakout": 0.15,
            "pullback": 0.10,
            "pairs": 0.15,
        }
    )

    # Kill switch (canonical nested form).
    kill_switch: RiskManagementKillSwitchConfig = Field(
        default_factory=RiskManagementKillSwitchConfig
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_flat_kill_switch_keys(cls, data):
        """
        Backward-compatibility shim:
        Accept legacy flat keys and map them into kill_switch.
        """
        if not isinstance(data, dict):
            return data

        ks = dict(data.get("kill_switch") or {})
        if "max_consecutive_failures" in data and "max_consecutive_failures" not in ks:
            ks["max_consecutive_failures"] = data["max_consecutive_failures"]
        if (
            "max_realized_daily_loss_pct" in data
            and "max_realized_daily_loss_pct" not in ks
        ):
            ks["max_realized_daily_loss_pct"] = data["max_realized_daily_loss_pct"]
        data["kill_switch"] = ks
        return data

    @property
    def max_consecutive_failures(self) -> int:
        """Legacy accessor retained for backward compatibility."""
        return int(self.kill_switch.max_consecutive_failures)

    @property
    def max_realized_daily_loss_pct(self) -> float:
        """Legacy accessor retained for backward compatibility."""
        return float(self.kill_switch.max_realized_daily_loss_pct)


class FeatureEngineeringConfig(BaseModel):
    """Feature engineering module configuration."""

    enabled: bool = Field(default=True)
    cache_enabled: bool = Field(default=True)
    cache_ttl_minutes: int = Field(default=60)
    primary_timeframe: str = Field(default="1d")
    secondary_timeframes: List[str] = Field(default_factory=lambda: ["1h"])
    lookback_bars: int = Field(default=252)
    min_history_bars: int = Field(default=80)
    alpha_families_enabled: List[str] = Field(
        default_factory=lambda: [
            "ts_momentum",
            "xs_momentum",
            "mean_reversion",
            "breakout",
            "pullback",
            "pairs",
        ]
    )

    trend: Dict[str, Any] = Field(
        default_factory=lambda: {
            "ma_slope_window": 20,
            "dmi_window": 14,
            "efficiency_window": 20,
        }
    )

    momentum: Dict[str, Any] = Field(
        default_factory=lambda: {
            "exclude_recent_bars": 21,
        }
    )

    reversal: Dict[str, Any] = Field(
        default_factory=lambda: {
            "rsi_short_window": 2,
        }
    )

    breakout: Dict[str, Any] = Field(
        default_factory=lambda: {
            "donchian_window": 20,
            "squeeze_percentile_window": 120,
        }
    )

    pairs: Dict[str, Any] = Field(
        default_factory=lambda: {
            "zscore_window": 60,
            "min_correlation": 0.7,
        }
    )

    breadth_symbols: List[str] = Field(default_factory=lambda: ["SPY", "QQQ"])
    fail_fast_on_missing_columns: bool = Field(default=True)


class SignalGenerationConfig(BaseModel):
    """Signal generation pipeline configuration."""

    enabled: bool = Field(default=True)
    baseline_signal_a_name: str = Field(default="SignalA_v0")
    baseline_signal_b_name: str = Field(default="SignalB_v0")
    enable_regime_router: bool = Field(default=True)
    regime_router_default: str = Field(default="neutral")
    combine_method: str = Field(default="weighted_sum")
    min_composite_score: float = Field(default=35.0)
    max_signals_per_batch: int = Field(default=200)
    determinism_seed: int = Field(default=42)
    fail_fast_on_schema_violation: bool = Field(default=True)
    signal_zoo_min_count: int = Field(default=10)
    signal_zoo_target_count: int = Field(default=20)
    day_manager_pipeline_enabled: bool = Field(default=True)
    day_manager_alpha_blend_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    day_manager_enrichment_cache_enabled: bool = Field(default=True)
    alpha_generation_max_workers: int = Field(default=4, ge=1, le=16)
    llm_overlay: "SignalGenerationLLMOverlayConfig" = Field(
        default_factory=lambda: SignalGenerationLLMOverlayConfig()
    )

    families_enabled: List[str] = Field(
        default_factory=lambda: [
            "ts_momentum",
            "xs_momentum",
            "mean_reversion",
            "breakout",
            "pullback",
            "pairs",
        ]
    )

    ts_momentum: Dict[str, Any] = Field(
        default_factory=lambda: {
            "lookback_long_bars": 252,
            "lookback_short_bars": 126,
            "exclude_recent_bars": 21,
        }
    )

    breakout: Dict[str, Any] = Field(
        default_factory=lambda: {
            "donchian_window": 20,
            "squeeze_percentile_window": 120,
        }
    )

    mean_reversion: Dict[str, Any] = Field(
        default_factory=lambda: {
            "rsi_window": 2,
            "rsi_entry_threshold": 10,
        }
    )

    pairs: Dict[str, Any] = Field(
        default_factory=lambda: {
            "zscore_entry_threshold": 2.0,
        }
    )

    ensemble: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enable_ev_calibration": True,
            "calibration_lookback_bars": 252,
            "output_range": [-1.0, 1.0],
        }
    )

    regime: Dict[str, Any] = Field(
        default_factory=lambda: {
            "rule_labels": ["trend", "chop", "crisis"],
            "probabilistic_model": "hmm_optional",
            "family_enable_policy": {
                "ts_momentum": ["trend"],
                "xs_momentum": ["trend"],
                "mean_reversion": ["chop", "crisis"],
                "breakout": ["trend"],
                "pullback": ["trend"],
                "pairs": ["chop"],
                "baseline_a": ["trend", "chop", "crisis", "neutral"],
                "baseline_b": ["trend", "chop", "crisis", "neutral"],
            },
        }
    )


class SignalGenerationLLMOverlayConfig(BaseModel):
    """Optional LLM overlay for top-slice signal promotion and rejection."""

    enabled: bool = Field(default=False)
    mode: str = Field(
        default="reject_veto_only",
        description="single_model|reject_veto_only|two_model_router",
    )
    single_model: str = Field(default="qwen3.5:9b-q4_K_M")
    promote_model: str = Field(default="")
    reject_model: str = Field(default="")
    top_slice_size: int = Field(default=30, ge=1, le=200)
    timeout_seconds: int = Field(default=10, ge=1, le=120)
    min_confidence_to_override: float = Field(default=70.0, ge=0.0, le=100.0)
    min_benchmark_improvement_margin: float = Field(default=3.0, ge=0.0, le=50.0)
    benchmark_report_path: str = Field(
        default="reports/llm_signal_layer_benchmark_latest.json"
    )


# ============================================================
# Main Configuration Class
# ============================================================


class AlpacaMCPServerConfig(BaseModel):
    enabled: bool = Field(default=True)
    paper_mode: bool = Field(default=True)


class SQLiteMCPServerConfig(BaseModel):
    enabled: bool = Field(default=True)
    db_path: str = Field(default="data/downday/predictions_db.sqlite")


class FetchMCPServerConfig(BaseModel):
    enabled: bool = Field(default=True)
    max_length_default: int = Field(default=5000)
    ignore_robots_txt: bool = Field(default=False)


class TimeMCPServerConfig(BaseModel):
    enabled: bool = Field(default=True)
    local_timezone: str = Field(default="America/New_York")


class MCPServersConfig(BaseModel):
    alpaca: AlpacaMCPServerConfig = Field(default_factory=AlpacaMCPServerConfig)
    sqlite: SQLiteMCPServerConfig = Field(default_factory=SQLiteMCPServerConfig)
    fetch: FetchMCPServerConfig = Field(default_factory=FetchMCPServerConfig)
    time: TimeMCPServerConfig = Field(default_factory=TimeMCPServerConfig)


class MCPConfig(BaseModel):
    """MCP client integration configuration."""

    enabled: bool = Field(default=True)
    timeout: int = Field(default=30)
    retry_attempts: int = Field(default=3)
    fallback_to_direct: bool = Field(default=True)
    log_calls: bool = Field(default=True)
    servers: MCPServersConfig = Field(default_factory=MCPServersConfig)


class SequentialShadowEvalConfig(BaseModel):
    """Offline executed-trade shadow evaluation settings."""

    enabled: bool = Field(default=False)
    mode: str = Field(default="executed_trade_shadow")
    schedule: str = Field(default="pm_daily")
    model: str = Field(default="deepseek-r1:8b")
    timeout_seconds_per_event: int = Field(default=10)
    max_workers: int = Field(default=2)
    batch_size: int = Field(default=50)
    max_events_per_run: int = Field(default=0)
    horizon_minutes: int = Field(default=120)
    runner_timeout_seconds: int = Field(default=900)
    persist_summary_only: bool = Field(default=True)
    materialize_ttl_cycles: int = Field(default=20)


class InverseETFTriggersConfig(BaseModel):
    vix_above: float = Field(default=25.0)
    spy_below_sma20: bool = Field(default=True)
    market_regime: str = Field(default="BEARISH")
    consecutive_red_days: int = Field(default=3)


class InverseETFExitTriggersConfig(BaseModel):
    vix_below: float = Field(default=18.0)
    spy_above_sma20: bool = Field(default=True)
    market_regime: str = Field(default="BULLISH")
    profit_target_pct: float = Field(default=10.0)


class InverseETFConfig(BaseModel):
    """Inverse ETF hedging configuration."""

    enabled: bool = Field(default=True)
    max_allocation_pct: float = Field(default=20.0)
    max_positions: int = Field(default=3)
    position_size: float = Field(default=3000.0)
    leveraged_max_hold_days: int = Field(default=3)
    entry_triggers: InverseETFTriggersConfig = Field(
        default_factory=InverseETFTriggersConfig
    )
    exit_triggers: InverseETFExitTriggersConfig = Field(
        default_factory=InverseETFExitTriggersConfig
    )


class MonitoringAlertsConfig(BaseModel):
    drawdown_alarm_pct: float = Field(default=5.0)
    data_staleness_hours: int = Field(default=24)
    realized_vs_expected_slippage_bps: int = Field(default=20)
    trade_conversion_rate_floor: float = Field(default=0.05)
    alpha_family_degradation_pct: float = Field(default=30.0)
    regime_drift_psi_threshold: float = Field(default=0.2)


class MonitoringReportingConfig(BaseModel):
    daily_report_enabled: bool = Field(default=True)
    weekly_rollup_enabled: bool = Field(default=True)


class MonitoringReleaseGatesConfig(BaseModel):
    paper_trading_min_days: int = Field(default=10)
    canary_max_risk_pct: float = Field(default=10.0)


class MonitoringConfig(BaseModel):
    enabled: bool = Field(default=True)
    metrics_emit_interval_sec: int = Field(default=60)
    alerts: MonitoringAlertsConfig = Field(default_factory=MonitoringAlertsConfig)
    reporting: MonitoringReportingConfig = Field(
        default_factory=MonitoringReportingConfig
    )
    release_gates: MonitoringReleaseGatesConfig = Field(
        default_factory=MonitoringReleaseGatesConfig
    )


class AdvisorControlConfig(BaseModel):
    """Controls for throttling expensive advisor reevaluation."""

    enabled: bool = Field(default=True)
    min_recheck_seconds: int = Field(default=300, ge=30, le=3600)
    force_recheck_seconds: int = Field(default=900, ge=60, le=7200)
    price_move_bps_trigger: int = Field(default=35, ge=1, le=1000)
    pnl_change_pct_trigger: float = Field(default=0.5, ge=0.05, le=20.0)
    hold_minutes_trigger: int = Field(default=30, ge=1, le=600)


class PostureConfig(BaseModel):
    """Intraday posture decay and lockout controls."""

    stress_window_minutes: int = Field(default=90, ge=1, le=390)
    hard_stop_lockout_minutes: int = Field(default=90, ge=1, le=390)


class ThesisCacheConfig(BaseModel):
    """Multi-cycle thesis cache settings."""

    enabled: bool = Field(default=True)
    persist_to_disk: bool = Field(default=True)
    persist_path: str = Field(default="data/position_theses.json")
    max_history_entries: int = Field(default=10, ge=1, le=50)
    archive_path: str = Field(default="data/thesis_archive.jsonl")


class PositionSchedulerTierConfig(BaseModel):
    """Single tier configuration."""

    interval_seconds: int = Field(default=60, ge=10, le=3600)


class PositionSchedulerConfig(BaseModel):
    """Tier-based position monitoring scheduler."""

    enabled: bool = Field(default=True)
    tiers: Dict[str, PositionSchedulerTierConfig] = Field(
        default_factory=lambda: {
            "critical": PositionSchedulerTierConfig(interval_seconds=30),
            "active": PositionSchedulerTierConfig(interval_seconds=60),
            "stable": PositionSchedulerTierConfig(interval_seconds=180),
            "dormant": PositionSchedulerTierConfig(interval_seconds=300),
        }
    )
    force_eval_price_move_bps: float = Field(default=50.0, ge=1.0, le=1000.0)
    force_eval_pnl_change_pct: float = Field(default=1.0, ge=0.1, le=20.0)


class OvernightSecondaryConfig(BaseModel):
    """Secondary overnight tasks that run only after primary workflow completion."""

    enabled: bool = Field(default=True)
    run_when_research_complete: bool = Field(default=True)

    # VL model-vs-signal benchmark
    vl_benchmark_enabled: bool = Field(default=True)
    vl_benchmark_cycle_interval: int = Field(default=45, ge=1, le=720)
    vl_benchmark_symbols: int = Field(default=8, ge=1, le=100)
    vl_benchmark_max_runtime_seconds: int = Field(default=420, ge=30, le=3600)

    # Deep research refresh / catalyst monitor on current top picks
    top_pick_research_enabled: bool = Field(default=True)
    top_pick_research_cycle_interval: int = Field(default=20, ge=1, le=720)
    top_pick_research_symbols: int = Field(default=6, ge=1, le=100)
    top_pick_research_max_runtime_seconds: int = Field(default=240, ge=30, le=3600)
    top_pick_research_universe_symbols: int = Field(default=200, ge=1, le=250)
    top_pick_research_rotation_enabled: bool = Field(default=True)
    top_pick_research_actionable_only: bool = Field(default=True)
    top_pick_research_include_deep_dive: bool = Field(default=True)
    top_pick_research_include_web_research: bool = Field(default=True)
    top_pick_research_update_researched_state: bool = Field(default=True)

    # Historical revisit of recent plans for bullish resurgence
    historical_revisit_enabled: bool = Field(default=True)
    historical_revisit_cycle_interval: int = Field(default=60, ge=1, le=720)
    historical_revisit_days: int = Field(default=5, ge=1, le=30)
    historical_revisit_symbols: int = Field(default=20, ge=1, le=200)
    historical_revisit_max_runtime_seconds: int = Field(default=360, ge=30, le=3600)

    # Idle-slot signal mining / discovery registry
    signal_mining_enabled: bool = Field(default=True)
    signal_mining_cycle_interval: int = Field(default=30, ge=1, le=720)
    signal_mining_universe_symbols: int = Field(default=1600, ge=1, le=5000)
    signal_mining_max_new_registry_entries: int = Field(default=25, ge=1, le=500)
    signal_mining_gap_threshold_pct: float = Field(default=5.0, ge=0.5, le=50.0)
    signal_mining_breadth_threshold_pct: float = Field(default=65.0, ge=1.0, le=100.0)
    signal_mining_volatility_atr_threshold: float = Field(default=6.0, ge=0.5, le=50.0)
    signal_mining_volatility_range_threshold: float = Field(
        default=8.0, ge=0.5, le=100.0
    )


class OvernightWorkflowConfig(BaseModel):
    """Primary overnight completion and quality controls."""

    target_watchlist_size: int = Field(default=200, ge=50, le=500)
    completion_min_watchlist_size: int = Field(default=100, ge=10, le=500)
    breadth_min_symbols: int = Field(default=2000, ge=100, le=10000)
    enforce_top_news_count: int = Field(default=200, ge=0, le=500)
    news_stage_required: bool = Field(default=True)
    news_stage_max_workers: int = Field(default=8, ge=1, le=64)
    news_stage_timeout_seconds: int = Field(default=300, ge=30, le=3600)


class ShortEngineConfig(BaseModel):
    """Short-side engine rollout and risk limits."""

    enabled: bool = Field(default=False)
    dry_run: bool = Field(default=True)
    paper_trading: bool = Field(default=False)
    paper_size_multiplier: float = Field(default=0.5, ge=0.0, le=1.0)
    gross_cap_pct: float = Field(default=0.15, ge=0.0, le=0.40)
    hard_to_borrow_allowed: bool = Field(default=False)
    min_promotion_sessions: int = Field(default=0, ge=0)
    paper_sanity_sessions: int = Field(default=1, ge=0)
    min_promotion_profit_factor: float = Field(default=1.2, ge=0.0)
    min_promotion_win_rate: float = Field(default=0.60, ge=0.0, le=1.0)
    historical_min_trades: int = Field(default=30, ge=1)
    historical_min_profit_factor: float = Field(default=1.2, ge=0.0)
    historical_min_win_rate: float = Field(default=0.52, ge=0.0, le=1.0)
    historical_max_breadth_pct: float = Field(default=45.0, ge=0.0, le=100.0)
    historical_max_up_probability: float = Field(default=0.65, ge=0.0, le=1.0)
    historical_hold_days: int = Field(default=3, ge=1, le=10)


class TradingConfig(BaseSettings):
    """
    Main trading configuration.

    Loads from:
    1. config/trading_config.yaml
    2. Environment variables (override YAML)
    3. .env file
    """

    # Sub-configurations
    risk_gate: RiskGateConfig = Field(default_factory=RiskGateConfig)
    conviction: ConvictionConfig = Field(default_factory=ConvictionConfig)
    rotation: RotationConfig = Field(default_factory=RotationConfig)
    decision_policy: DecisionPolicyConfig = Field(default_factory=DecisionPolicyConfig)
    plan_caps: PlanCapsConfig = Field(default_factory=PlanCapsConfig)
    trim_governance: TrimGovernanceConfig = Field(default_factory=TrimGovernanceConfig)
    llm_advisory_escalation: LLMAdvisoryEscalationConfig = Field(
        default_factory=LLMAdvisoryEscalationConfig
    )
    loss_floor: LossFloorConfig = Field(default_factory=LossFloorConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    entry_quality: EntryQualityConfig = Field(default_factory=EntryQualityConfig)
    research_freshness: ResearchFreshnessConfig = Field(
        default_factory=ResearchFreshnessConfig
    )
    premarket_gap_policy: PremarketGapPolicyConfig = Field(
        default_factory=PremarketGapPolicyConfig
    )
    premarket_scalp: PremarketScalpConfig = Field(default_factory=PremarketScalpConfig)
    overnight_cut_policy: "OvernightCutPolicyConfig" = Field(
        default_factory=lambda: OvernightCutPolicyConfig()
    )
    premarket_manager: PremarketManagerConfig = Field(
        default_factory=PremarketManagerConfig
    )
    momentum_scanner: MomentumScannerConfig = Field(
        default_factory=MomentumScannerConfig
    )
    screener_v2: ScreenerV2Config = Field(default_factory=ScreenerV2Config)
    universe_scanner: UniverseScannerConfig = Field(
        default_factory=UniverseScannerConfig
    )
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    backtest_protocol: BacktestProtocolConfig = Field(
        default_factory=BacktestProtocolConfig
    )
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    advisor_control: AdvisorControlConfig = Field(default_factory=AdvisorControlConfig)
    posture: PostureConfig = Field(default_factory=PostureConfig)
    thesis_cache: ThesisCacheConfig = Field(default_factory=ThesisCacheConfig)
    position_scheduler: PositionSchedulerConfig = Field(
        default_factory=PositionSchedulerConfig
    )
    strategy_lab: StrategyLabConfig = Field(default_factory=StrategyLabConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    runtime_recovery: RuntimeRecoveryConfig = Field(
        default_factory=RuntimeRecoveryConfig
    )
    data: DataConfig = Field(default_factory=DataConfig)
    alpaca: AlpacaConfig = Field(default_factory=AlpacaConfig)
    risk_metrics: RiskMetricsConfig = Field(default_factory=RiskMetricsConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    overnight_secondary: OvernightSecondaryConfig = Field(
        default_factory=OvernightSecondaryConfig
    )
    overnight_workflow: OvernightWorkflowConfig = Field(
        default_factory=OvernightWorkflowConfig
    )
    risk_limits: RiskLimitsConfig = Field(default_factory=RiskLimitsConfig)
    strategy_failsafe: StrategyFailsafeConfig = Field(
        default_factory=StrategyFailsafeConfig
    )
    short_engine: ShortEngineConfig = Field(default_factory=ShortEngineConfig)
    signal_validation: SignalValidationConfig = Field(
        default_factory=SignalValidationConfig
    )
    feature_engineering: FeatureEngineeringConfig = Field(
        default_factory=FeatureEngineeringConfig
    )
    signal_generation: SignalGenerationConfig = Field(
        default_factory=SignalGenerationConfig
    )
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    sequential_shadow_eval: SequentialShadowEvalConfig = Field(
        default_factory=SequentialShadowEvalConfig
    )
    inverse_etf: InverseETFConfig = Field(default_factory=InverseETFConfig)
    risk_management: RiskManagementConfig = Field(default_factory=RiskManagementConfig)
    claw_remote: ClawRemoteConfig = Field(default_factory=ClawRemoteConfig)
    workflow_claw: WorkflowClawConfig = Field(default_factory=WorkflowClawConfig)
    decision_claw: DecisionClawConfig = Field(default_factory=DecisionClawConfig)
    regime_strategies: Dict[str, RegimeStrategyConfig] = Field(default_factory=dict)

    # Paths (resolved at runtime)
    config_dir: Path = Field(default_factory=lambda: Path(__file__).parent)
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    logs_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "logs"
    )
    plans_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "plans"
    )
    data_dir: Path = Field(
        default_factory=lambda: (
            (Path(os.environ["DOWNDAY_ROOT"]) / "data")
            if os.environ.get("DOWNDAY_ROOT")
            else Path(__file__).parent.parent / "data"
        )
    )

    model_config = {"env_prefix": "TRADING_", "env_file": ".env", "extra": "ignore"}

    @classmethod
    def from_yaml(cls, config_path: Optional[Path] = None) -> "TradingConfig":
        """Load configuration from YAML file with env overrides."""
        _load_env_file()
        if config_path is None:
            config_path = Path(__file__).parent / "trading_config.yaml"

        yaml_data = {}
        if config_path.exists():
            with open(config_path) as f:
                yaml_data = yaml.safe_load(f) or {}
            logger.info(f"Loaded config from {config_path}")
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")

        # Load environment variables for sensitive data
        alpaca_config = yaml_data.get("alpaca", {})
        env_api_key = _env_first("ALPACA_API_KEY", "APCA_API_KEY_ID")
        env_secret_key = _env_first(
            "ALPACA_SECRET_KEY", "APCA_API_SECRET_KEY", "ALPACA_API_SECRET"
        )
        env_paper = _env_first("ALPACA_PAPER", "APCA_API_PAPER")
        if env_api_key:
            alpaca_config["api_key"] = env_api_key
        else:
            alpaca_config["api_key"] = alpaca_config.get("api_key")
        if env_secret_key:
            alpaca_config["secret_key"] = env_secret_key
        else:
            alpaca_config["secret_key"] = alpaca_config.get("secret_key")
        if env_paper is not None:
            text = str(env_paper).strip().lower()
            alpaca_config["paper"] = text in {"1", "true", "yes", "on"}
        yaml_data["alpaca"] = alpaca_config

        # Resolve data paths
        data_config = yaml_data.get("data", {})
        if os.environ.get("DOWNDAY_ROOT"):
            data_config.setdefault("downday_root", os.environ["DOWNDAY_ROOT"])

        # Override from environment
        if os.environ.get("DUCKDB_PATH"):
            data_config["duckdb_path"] = os.environ["DUCKDB_PATH"]

        yaml_data["data"] = data_config

        return cls(**yaml_data)

    def get_predictions_db_path(self) -> Optional[Path]:
        """Legacy compatibility shim: predictions DB workflow is removed."""
        return None

    @property
    def market_data(self) -> DataConfig:
        """Backward-compatible alias for the nested data configuration."""
        return self.data

    def get_duckdb_path(self) -> Path:
        """Get resolved path to DuckDB cache."""
        path = Path(self.data.duckdb_path)
        if path.is_absolute():
            return path
        return self.project_root / path

    def to_dict(self) -> Dict[str, Any]:
        """Export config as dictionary."""
        return {
            "risk_gate": self.risk_gate.model_dump(),
            "conviction": self.conviction.model_dump(),
            "rotation": self.rotation.model_dump(),
            "decision_policy": self.decision_policy.model_dump(),
            "plan_caps": self.plan_caps.model_dump(),
            "trim_governance": self.trim_governance.model_dump(),
            "llm_advisory_escalation": self.llm_advisory_escalation.model_dump(),
            "portfolio": self.portfolio.model_dump(),
            "strategy": self.strategy.model_dump(),
            "entry_quality": self.entry_quality.model_dump(),
            "research_freshness": self.research_freshness.model_dump(),
            "premarket_gap_policy": self.premarket_gap_policy.model_dump(),
            "overnight_cut_policy": self.overnight_cut_policy.model_dump(),
            "premarket_manager": self.premarket_manager.model_dump(),
            "momentum_scanner": self.momentum_scanner.model_dump(),
            "screener_v2": self.screener_v2.model_dump(),
            "universe_scanner": self.universe_scanner.model_dump(),
            "backtest": self.backtest.model_dump(),
            "backtest_protocol": self.backtest_protocol.model_dump(),
            "llm": self.llm.model_dump(),
            "search": self.search.model_dump(),
            "runtime_recovery": self.runtime_recovery.model_dump(),
            "data": self.data.model_dump(),
            "alpaca": {
                k: v
                for k, v in self.alpaca.model_dump().items()
                if k not in ("api_key", "secret_key")
            },
            "risk_metrics": self.risk_metrics.model_dump(),
            "sentiment": self.sentiment.model_dump(),
            "logging": self.logging.model_dump(),
            "overnight_secondary": self.overnight_secondary.model_dump(),
            "overnight_workflow": self.overnight_workflow.model_dump(),
            "risk_limits": self.risk_limits.model_dump(),
            "risk_management": self.risk_management.model_dump(),
            "strategy_failsafe": self.strategy_failsafe.model_dump(),
            "signal_validation": self.signal_validation.model_dump(),
        }


# ============================================================
# Global Config Accessor
# ============================================================

_config_instance: Optional[TradingConfig] = None


def get_config(reload: bool = False) -> TradingConfig:
    """
    Get the global configuration instance.

    Args:
        reload: If True, force reload from disk

    Returns:
        TradingConfig instance
    """
    global _config_instance

    if _config_instance is None or reload:
        config_path = Path(__file__).parent / "trading_config.yaml"
        _config_instance = TradingConfig.from_yaml(config_path)

    return _config_instance


def get_risk_gate_config() -> RiskGateConfig:
    """Get risk gate configuration."""
    return get_config().risk_gate


def get_llm_config() -> LLMConfig:
    """Get LLM configuration."""
    return get_config().llm


def get_portfolio_config() -> PortfolioConfig:
    """Get portfolio configuration."""
    return get_config().portfolio


def get_backtest_config() -> BacktestConfig:
    """Get backtest configuration."""
    return get_config().backtest


def get_backtest_protocol_config() -> BacktestProtocolConfig:
    """Get backtest protocol configuration."""
    return get_config().backtest_protocol


def get_execution_config() -> ExecutionConfig:
    """Get unified execution adapter configuration."""
    return get_config().execution


def get_strategy_config() -> StrategyConfig:
    """Get strategy configuration."""
    return get_config().strategy


def get_logging_config() -> LoggingConfig:
    """Get logging configuration."""
    return get_config().logging


def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration."""
    return get_config().monitoring


def get_risk_limits_config() -> RiskLimitsConfig:
    """Get workflow risk limits configuration."""
    return get_config().risk_limits


def get_risk_management_config() -> RiskManagementConfig:
    """Get risk management fail-fast and kill-switch configuration."""
    return get_config().risk_management


def get_strategy_failsafe_config() -> StrategyFailsafeConfig:
    """Get strategy failsafe configuration."""
    return get_config().strategy_failsafe


def get_alpaca_config() -> AlpacaConfig:
    """Get Alpaca configuration."""
    return get_config().alpaca


def get_feature_engineering_config() -> FeatureEngineeringConfig:
    """Get feature engineering configuration."""
    return get_config().feature_engineering


def get_sequential_shadow_eval_config() -> SequentialShadowEvalConfig:
    """Get offline sequential shadow evaluation configuration."""
    return get_config().sequential_shadow_eval


# ============================================================
# Config validation helper
# ============================================================


def validate_config() -> List[str]:
    """
    Validate the current configuration.

    Returns list of warnings/errors.
    """
    warnings = []
    config = get_config()

    # Check Alpaca credentials
    if not config.alpaca.api_key or not config.alpaca.secret_key:
        warnings.append("Alpaca API credentials not configured")

    # Check conviction weights
    try:
        _ = config.conviction  # Triggers validation
    except Exception as e:
        warnings.append(f"Conviction config error: {e}")

    # Check portfolio limits
    if config.portfolio.position_size_max < config.portfolio.position_size_target:
        warnings.append("position_size_max should be >= position_size_target")

    return warnings


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info(
        "config_loader.py loaded; use pytest/CLI for tests."
    )
