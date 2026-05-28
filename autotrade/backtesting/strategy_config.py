from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


BACKTEST_RESULTS_DIR = Path("data") / "backtest_results"
STRATEGY_DEFINITIONS_DIR = Path("data") / "strategies"


def ensure_strategy_lab_dirs() -> None:
    BACKTEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    STRATEGY_DEFINITIONS_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_name(value: str) -> str:
    s = (value or "strategy").strip().lower()
    s = re.sub(r"[^a-z0-9_\\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "strategy"


def make_strategy_name(setup_type: str, suffix: str = "") -> str:
    base = _sanitize_name(setup_type or "strategy")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{suffix}_{ts}" if suffix else f"{base}_{ts}"


@dataclass
class EntryRules:
    setup_type: str = "pullback_support"
    rsi_min: float = 35.0
    rsi_max: float = 65.0
    min_atr_pct: float = 2.5
    max_atr_pct: float = 10.0
    min_volume_ratio: float = 1.0
    require_above_sma20: bool = False
    require_sma5_curl_positive: bool = False
    min_composite_score: float = 70.0
    # Trend filters
    min_adx: float = 0.0
    require_positive_macd_hist: bool = False
    require_ema_10_above_20: bool = False
    # Bollinger / Squeeze
    require_bb_squeeze: bool = False
    require_bb_width_compression: bool = False
    max_bb_width_percentile: float = 1.0
    bb_width_percentile_lookback: int = 60
    require_nr7: bool = False
    volume_avg_deviation_pct: float = 0.0
    max_price_position_bb: float = 999.0
    min_price_position_bb: float = -999.0
    # Momentum
    min_roc_5: float = -999.0
    min_roc_10: float = -999.0
    # Oscillators
    max_stoch_k: float = 100.0
    min_stoch_k: float = 0.0
    min_cci: float = -999.0
    max_cci: float = 999.0
    # Ichimoku
    require_above_ichimoku_cloud: bool = False
    require_tenkan_above_kijun: bool = False
    # Distance / volatility ratio
    max_distance_from_sma20: float = 999.0
    min_atr_ratio_5_14: float = 0.0
    # Scoring
    scoring_profile: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "setup_type": self.setup_type,
            "rsi_min": float(self.rsi_min),
            "rsi_max": float(self.rsi_max),
            "min_atr_pct": float(self.min_atr_pct),
            "max_atr_pct": float(self.max_atr_pct),
            "min_volume_ratio": float(self.min_volume_ratio),
            "require_above_sma20": bool(self.require_above_sma20),
            "require_sma5_curl_positive": bool(self.require_sma5_curl_positive),
            "min_composite_score": float(self.min_composite_score),
            "min_adx": float(self.min_adx),
            "require_positive_macd_hist": bool(self.require_positive_macd_hist),
            "require_ema_10_above_20": bool(self.require_ema_10_above_20),
            "require_bb_squeeze": bool(self.require_bb_squeeze),
            "require_bb_width_compression": bool(self.require_bb_width_compression),
            "max_bb_width_percentile": float(self.max_bb_width_percentile),
            "bb_width_percentile_lookback": int(self.bb_width_percentile_lookback),
            "require_nr7": bool(self.require_nr7),
            "volume_avg_deviation_pct": float(self.volume_avg_deviation_pct),
            "max_price_position_bb": float(self.max_price_position_bb),
            "min_price_position_bb": float(self.min_price_position_bb),
            "min_roc_5": float(self.min_roc_5),
            "min_roc_10": float(self.min_roc_10),
            "max_stoch_k": float(self.max_stoch_k),
            "min_stoch_k": float(self.min_stoch_k),
            "min_cci": float(self.min_cci),
            "max_cci": float(self.max_cci),
            "require_above_ichimoku_cloud": bool(self.require_above_ichimoku_cloud),
            "require_tenkan_above_kijun": bool(self.require_tenkan_above_kijun),
            "max_distance_from_sma20": float(self.max_distance_from_sma20),
            "min_atr_ratio_5_14": float(self.min_atr_ratio_5_14),
            "scoring_profile": str(self.scoring_profile),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EntryRules":
        d = data or {}
        return cls(
            setup_type=str(d.get("setup_type", "pullback_support")),
            rsi_min=float(d.get("rsi_min", 35.0)),
            rsi_max=float(d.get("rsi_max", 65.0)),
            min_atr_pct=float(d.get("min_atr_pct", 2.5)),
            max_atr_pct=float(d.get("max_atr_pct", 10.0)),
            min_volume_ratio=float(d.get("min_volume_ratio", 1.0)),
            require_above_sma20=bool(d.get("require_above_sma20", False)),
            require_sma5_curl_positive=bool(d.get("require_sma5_curl_positive", False)),
            min_composite_score=float(d.get("min_composite_score", 70.0)),
            min_adx=float(d.get("min_adx", 0.0)),
            require_positive_macd_hist=bool(d.get("require_positive_macd_hist", False)),
            require_ema_10_above_20=bool(d.get("require_ema_10_above_20", False)),
            require_bb_squeeze=bool(d.get("require_bb_squeeze", False)),
            require_bb_width_compression=bool(
                d.get("require_bb_width_compression", False)
            ),
            max_bb_width_percentile=float(d.get("max_bb_width_percentile", 1.0)),
            bb_width_percentile_lookback=int(d.get("bb_width_percentile_lookback", 60)),
            require_nr7=bool(d.get("require_nr7", False)),
            volume_avg_deviation_pct=float(d.get("volume_avg_deviation_pct", 0.0)),
            max_price_position_bb=float(d.get("max_price_position_bb", 999.0)),
            min_price_position_bb=float(d.get("min_price_position_bb", -999.0)),
            min_roc_5=float(d.get("min_roc_5", -999.0)),
            min_roc_10=float(d.get("min_roc_10", -999.0)),
            max_stoch_k=float(d.get("max_stoch_k", 100.0)),
            min_stoch_k=float(d.get("min_stoch_k", 0.0)),
            min_cci=float(d.get("min_cci", -999.0)),
            max_cci=float(d.get("max_cci", 999.0)),
            require_above_ichimoku_cloud=bool(
                d.get("require_above_ichimoku_cloud", False)
            ),
            require_tenkan_above_kijun=bool(d.get("require_tenkan_above_kijun", False)),
            max_distance_from_sma20=float(d.get("max_distance_from_sma20", 999.0)),
            min_atr_ratio_5_14=float(d.get("min_atr_ratio_5_14", 0.0)),
            scoring_profile=str(d.get("scoring_profile", "default")),
        )


@dataclass
class ExitRules:
    stop_atr_mult: float = 2.0
    target_atr_mult: float = 3.0
    trailing_stop: bool = True
    trailing_atr_mult: float = 1.5
    max_hold_days: int = 5
    target_entry_range_mult: float = 0.0
    time_stop_if_flat_days: int = 3
    profit_take_levels: List[float] = field(default_factory=lambda: [5.0, 10.0])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stop_atr_mult": float(self.stop_atr_mult),
            "target_atr_mult": float(self.target_atr_mult),
            "trailing_stop": bool(self.trailing_stop),
            "trailing_atr_mult": float(self.trailing_atr_mult),
            "max_hold_days": int(self.max_hold_days),
            "target_entry_range_mult": float(self.target_entry_range_mult),
            "time_stop_if_flat_days": int(self.time_stop_if_flat_days),
            "profit_take_levels": [float(x) for x in self.profit_take_levels],
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ExitRules":
        d = data or {}
        levels = d.get("profit_take_levels") or [5.0, 10.0]
        return cls(
            stop_atr_mult=float(d.get("stop_atr_mult", 2.0)),
            target_atr_mult=float(d.get("target_atr_mult", 3.0)),
            trailing_stop=bool(d.get("trailing_stop", True)),
            trailing_atr_mult=float(d.get("trailing_atr_mult", 1.5)),
            max_hold_days=int(d.get("max_hold_days", 5)),
            target_entry_range_mult=float(d.get("target_entry_range_mult", 0.0)),
            time_stop_if_flat_days=int(d.get("time_stop_if_flat_days", 3)),
            profit_take_levels=[float(x) for x in levels],
        )


@dataclass
class PortfolioRules:
    max_positions: int = 30
    position_size_usd: float = 3333.0
    max_sector_pct: float = 25.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_positions": int(self.max_positions),
            "position_size_usd": float(self.position_size_usd),
            "max_sector_pct": float(self.max_sector_pct),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PortfolioRules":
        d = data or {}
        position_size = d.get("position_size_usd", d.get("position_size_pct", 3333.0))
        return cls(
            max_positions=int(d.get("max_positions", 30)),
            position_size_usd=float(position_size),
            max_sector_pct=float(d.get("max_sector_pct", 25.0)),
        )


@dataclass
class BacktestSummary:
    period: str = ""
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    total_pnl: float = 0.0
    walk_forward_validated: bool = False
    walk_forward_degradation: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period,
            "total_trades": int(self.total_trades),
            "win_rate": float(self.win_rate),
            "profit_factor": float(self.profit_factor),
            "sharpe": float(self.sharpe),
            "max_drawdown": float(self.max_drawdown),
            "total_pnl": float(self.total_pnl),
            "walk_forward_validated": bool(self.walk_forward_validated),
            "walk_forward_degradation": float(self.walk_forward_degradation),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "BacktestSummary":
        d = data or {}
        return cls(
            period=str(d.get("period", "")),
            total_trades=int(d.get("total_trades", 0)),
            win_rate=float(d.get("win_rate", 0.0)),
            profit_factor=float(d.get("profit_factor", 0.0)),
            sharpe=float(d.get("sharpe", 0.0)),
            max_drawdown=float(d.get("max_drawdown", 0.0)),
            total_pnl=float(d.get("total_pnl", 0.0)),
            walk_forward_validated=bool(d.get("walk_forward_validated", False)),
            walk_forward_degradation=float(d.get("walk_forward_degradation", 0.0)),
        )


@dataclass
class LabStrategyConfig:
    name: str
    description: str
    version: int = 1
    created: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    author: str = "strategy_lab"
    entry: EntryRules = field(default_factory=EntryRules)
    exit: ExitRules = field(default_factory=ExitRules)
    portfolio: PortfolioRules = field(default_factory=PortfolioRules)
    backtest_results: BacktestSummary = field(default_factory=BacktestSummary)

    def copy(self) -> "LabStrategyConfig":
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": int(self.version),
            "created": self.created,
            "author": self.author,
            "entry": self.entry.to_dict(),
            "exit": self.exit.to_dict(),
            "portfolio": self.portfolio.to_dict(),
            "backtest_results": self.backtest_results.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LabStrategyConfig":
        return cls(
            name=str(data.get("name") or make_strategy_name("strategy")),
            description=str(data.get("description", "Strategy Lab definition")),
            version=int(data.get("version", 1)),
            created=str(data.get("created", datetime.now().strftime("%Y-%m-%d"))),
            author=str(data.get("author", "strategy_lab")),
            entry=EntryRules.from_dict(data.get("entry")),
            exit=ExitRules.from_dict(data.get("exit")),
            portfolio=PortfolioRules.from_dict(data.get("portfolio")),
            backtest_results=BacktestSummary.from_dict(data.get("backtest_results")),
        )

    def with_parameter(self, param_name: str, value: Any) -> "LabStrategyConfig":
        updated = self.copy()
        for section in (updated.entry, updated.exit, updated.portfolio):
            if hasattr(section, param_name):
                setattr(section, param_name, value)
                return updated
        if hasattr(updated, param_name):
            setattr(updated, param_name, value)
        return updated

    def to_backtest_override(self) -> Dict[str, Any]:
        min_bullish = max(2.0, min(10.0, float(self.entry.min_composite_score) / 20.0))
        return {
            "min_bullish": float(min_bullish),
            "max_hold_days": int(self.exit.max_hold_days),
            "max_positions": int(self.portfolio.max_positions),
            "stop_atr": float(self.exit.stop_atr_mult),
            "target_atr": float(self.exit.target_atr_mult),
        }

    def to_trading_config_patch(self) -> Dict[str, Any]:
        patch: Dict[str, Any] = {
            "screener_v2": {
                "rsi_min": float(self.entry.rsi_min),
                "rsi_max": float(self.entry.rsi_max),
                "min_atr_pct": float(self.entry.min_atr_pct),
                "max_atr_pct": float(self.entry.max_atr_pct),
                "min_composite_score": float(self.entry.min_composite_score),
            },
            "backtest": {
                "max_hold_days": int(self.exit.max_hold_days),
                "stop_atr": float(self.exit.stop_atr_mult),
                "target_atr": float(self.exit.target_atr_mult),
                "max_positions": int(self.portfolio.max_positions),
            },
            "portfolio": {
                "max_positions": int(self.portfolio.max_positions),
                "position_size_target": float(self.portfolio.position_size_usd),
            },
            "strategy": {
                "min_lesson_score": max(
                    1.0, float(self.entry.min_composite_score) * 0.3
                ),
                "fallback_stop_atr": float(self.exit.stop_atr_mult),
                "fallback_target_atr": float(self.exit.target_atr_mult),
            },
        }
        # For time-only strategies (stop >= 10x ATR), widen risk_gate to 10x ATR
        # so the proven strategy isn't overridden by the default 2x stop.
        # The hard_stop_pct (-8%) absolute floor remains untouched.
        stop_mult = float(self.exit.stop_atr_mult)
        target_mult = float(self.exit.target_atr_mult)
        if stop_mult >= 10.0:
            patch["risk_gate"] = {
                "atr_k": 10.0,
                "target_atr": min(target_mult, 10.0),
            }
        elif stop_mult > 3.0:
            patch["risk_gate"] = {
                "atr_k": min(stop_mult, 10.0),
                "target_atr": min(target_mult, 10.0),
            }
        return patch

    def save(self, path: Optional[Path] = None) -> Path:
        ensure_strategy_lab_dirs()
        p = path or (STRATEGY_DEFINITIONS_DIR / f"{_sanitize_name(self.name)}.json")
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path) -> "LabStrategyConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(raw)


def list_strategy_files(limit: int = 100) -> List[Path]:
    ensure_strategy_lab_dirs()
    files = sorted(
        STRATEGY_DEFINITIONS_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[:limit]


def load_strategy_definition(path: Path) -> LabStrategyConfig:
    return LabStrategyConfig.load(path)


def save_strategy_definition(
    strategy: LabStrategyConfig, path: Optional[Path] = None
) -> Path:
    return strategy.save(path=path)
