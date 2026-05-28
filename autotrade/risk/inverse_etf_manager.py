"""
Inverse ETF Manager - Manages hedging positions during market downturns.

Fixes applied:
- Added SDS (2x SPY) to hardcoded universe
- SPY < SMA20 trigger now implemented (was dead code)
- Consecutive red days trigger now implemented (was dead code)
- Profit target exit now implemented (was dead code)
- Default hedge is RWM (Russell 2000) for small/mid-cap alignment
- Universe lookup uses SQLite DB as primary, dict as cache fallback
"""

import logging
from typing import Dict, List, Optional, Any

import pandas as pd

from config.config_loader import get_config
from autotrade.analysis.market_regime import MarketRegimeDetector, MarketRegime

logger = logging.getLogger("AutoTrade.InverseETFManager")

# Hardcoded cache — primary source is now the DB (inverse_etf_universe table).
# This dict is used as fallback when DB is unavailable.
INVERSE_ETF_UNIVERSE = {
    "SH": {"name": "ProShares Short S&P 500", "leverage": 1, "index": "SPY"},
    "SDS": {"name": "ProShares UltraShort S&P 500", "leverage": 2, "index": "SPY"},
    "PSQ": {"name": "ProShares Short QQQ", "leverage": 1, "index": "QQQ"},
    "QID": {"name": "ProShares UltraShort QQQ", "leverage": 2, "index": "QQQ"},
    "RWM": {"name": "ProShares Short Russell 2000", "leverage": 1, "index": "IWM"},
    "TWM": {"name": "ProShares UltraShort Russell 2000", "leverage": 2, "index": "IWM"},
    "DOG": {"name": "ProShares Short Dow30", "leverage": 1, "index": "DIA"},
    "SDD": {"name": "ProShares UltraShort Dow30", "leverage": 2, "index": "DIA"},
    "MZZ": {"name": "ProShares UltraShort MidCap400", "leverage": 2, "index": "MDY"},
    "SPXU": {"name": "ProShares UltraPro Short S&P500", "leverage": 3, "index": "SPY"},
    "SQQQ": {"name": "ProShares UltraPro Short QQQ", "leverage": 3, "index": "QQQ"},
    "SDOW": {"name": "ProShares UltraPro Short Dow30", "leverage": 3, "index": "DIA"},
    "SRTY": {
        "name": "ProShares UltraPro Short Russell 2000",
        "leverage": 3,
        "index": "IWM",
    },
    "TZA": {"name": "Direxion Daily Small Cap Bear 3X", "leverage": 3, "index": "IWM"},
    "FAZ": {"name": "Direxion Daily Financial Bear 3X", "leverage": 3, "index": "XLF"},
    "SOXS": {
        "name": "Direxion Daily Semiconductor Bear 3X",
        "leverage": 3,
        "index": "SOXX",
    },
}

# All known inverse ETF tickers (populated from DB on first access)
_db_ticker_cache: Optional[set] = None


def _load_db_tickers() -> set:
    """Load inverse ETF tickers from DB, cache for session."""
    global _db_ticker_cache
    if _db_ticker_cache is not None:
        return _db_ticker_cache
    try:
        from autotrade.utils.financial_db import FinancialDB

        db = FinancialDB()
        _db_ticker_cache = db.get_inverse_etf_tickers(active_only=True)
        if _db_ticker_cache:
            return _db_ticker_cache
    except Exception:
        pass
    _db_ticker_cache = set(INVERSE_ETF_UNIVERSE.keys())
    return _db_ticker_cache


def is_inverse_etf(ticker: str) -> bool:
    """Check if a ticker is a known inverse ETF (DB-backed with dict fallback)."""
    t = str(ticker).upper()
    if t in INVERSE_ETF_UNIVERSE:
        return True
    return t in _load_db_tickers()


def get_inverse_etf_info(ticker: str) -> Optional[Dict]:
    """Get ETF metadata from dict cache or DB."""
    t = str(ticker).upper()
    if t in INVERSE_ETF_UNIVERSE:
        return INVERSE_ETF_UNIVERSE[t]
    try:
        from autotrade.utils.financial_db import FinancialDB

        row = FinancialDB().get_inverse_etf(t)
        if row:
            return {
                "name": row.get("name", ""),
                "leverage": row.get("leverage", 1),
                "index": row.get("underlying", "SPY"),
            }
    except Exception:
        pass
    return None


class InverseETFManager:
    """
    Allocates to inverse ETFs when market conditions are bearish.
    """

    def __init__(self):
        self.full_config = get_config()
        self.config = getattr(
            self.full_config,
            "inverse_etf_hedging",
            getattr(self.full_config, "inverse_etf", None),
        )
        self.regime_detector = MarketRegimeDetector()

    @staticmethod
    def _get_previous_close(symbol: str) -> float:
        try:
            from autotrade.utils.local_data_provider import get_ticker_data

            df = get_ticker_data(
                str(symbol or "").upper(),
                days=5,
                include_features=False,
            )
            if df is None or df.empty:
                return 0.0
            close_col = next(
                (col for col in df.columns if str(col).lower() == "close"),
                None,
            )
            if close_col is None:
                return 0.0
            return float(df[close_col].iloc[-1] or 0.0)
        except Exception:
            return 0.0

    def get_instrument_profile(self, symbol: str) -> Dict[str, Any]:
        """Return leverage-aware execution defaults for an inverse ETF."""
        info = get_inverse_etf_info(symbol) or {}
        leverage = int(info.get("leverage", 1) or 1)
        profile = {
            "symbol": str(symbol or "").upper(),
            "underlying": str(info.get("index", "SPY") or "SPY").upper(),
            "leverage": leverage,
            "entry_size_multiplier": 1.0,
            "profit_target_pct": 6.0,
            "stop_loss_pct": -4.0,
            "scale_out_fraction": 0.25,
            "reversal_distance_pct": 0.004,
            "max_hold_days": int(getattr(self.config, "max_hold_days", 3) or 3),
        }
        if leverage == 2:
            profile.update(
                {
                    "entry_size_multiplier": 0.7,
                    "profit_target_pct": 4.5,
                    "stop_loss_pct": -3.0,
                    "scale_out_fraction": 0.33,
                    "reversal_distance_pct": 0.003,
                }
            )
        elif leverage >= 3:
            profile.update(
                {
                    "entry_size_multiplier": 0.45,
                    "profit_target_pct": 3.0,
                    "stop_loss_pct": -2.0,
                    "scale_out_fraction": 0.5,
                    "reversal_distance_pct": 0.002,
                    "max_hold_days": int(
                        getattr(self.config, "leveraged_max_hold_days", 3) or 3
                    ),
                }
            )
        return profile

    def evaluate_intraday_reversal(
        self,
        symbol: str,
        bars: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Detect strong intraday reversal using VWAP and short moving averages."""
        profile = self.get_instrument_profile(symbol)
        if bars is None or len(bars) < 5:
            return {
                "should_scale_out": False,
                "should_exit": False,
                "reasons": [],
                "profile": profile,
            }

        price_series = bars["close"].astype(float)
        current_price = float(price_series.iloc[-1])
        sma5 = float(price_series.tail(5).mean())
        sma9 = float(price_series.tail(min(9, len(price_series))).mean())
        typical = (
            bars["high"].astype(float)
            + bars["low"].astype(float)
            + bars["close"].astype(float)
        ) / 3.0
        volume = bars["volume"].astype(float).replace(0.0, pd.NA).ffill().fillna(0.0)
        total_vol = float(volume.sum())
        vwap = current_price
        if total_vol > 0:
            vwap = float((typical * volume).sum() / total_vol)
        last_return = 0.0
        if len(price_series) >= 2 and float(price_series.iloc[-2]) > 0:
            last_return = (current_price - float(price_series.iloc[-2])) / float(
                price_series.iloc[-2]
            )

        reversal_distance = float(profile["reversal_distance_pct"])
        reasons: List[str] = []
        if current_price < vwap * (1 - reversal_distance):
            reasons.append("price_below_vwap")
        if current_price < sma5 * (1 - reversal_distance):
            reasons.append("price_below_sma5")
        if current_price < sma9 * (1 - reversal_distance):
            reasons.append("price_below_sma9")
        if last_return <= max(-0.015, -reversal_distance * 5):
            reasons.append("sharp_negative_bar")

        strong_reversal = len(reasons) >= 2
        return {
            "should_scale_out": bool(reasons),
            "should_exit": strong_reversal,
            "reasons": reasons,
            "profile": profile,
            "current_price": current_price,
            "vwap": vwap,
            "sma5": sma5,
            "sma9": sma9,
        }

    def check_hedge_conditions(
        self, vix_level: float = 0.0, spy_price: float = 0.0, spy_sma20: float = 0.0
    ) -> Dict[str, Any]:
        """
        Evaluate if we should be in hedge positions.
        Now implements SPY < SMA20 and consecutive red days triggers.
        """
        if not self.config or not getattr(self.config, "enabled", False):
            return {
                "should_enter": False,
                "should_exit": False,
                "reason": "Disabled in config",
            }

        analysis = self.regime_detector.detect_regime(use_cache=True)
        regime = analysis.regime
        regime_label = (
            regime.value.lower() if hasattr(regime, "value") else str(regime).lower()
        )

        reasons = []
        should_enter = False
        should_exit = False

        allowed_regimes = [
            str(r).lower()
            for r in getattr(self.config, "allowed_regimes", ["crisis", "risk_off"])
        ]

        # 1. VIX trigger
        vix_trigger = 25.0
        if hasattr(self.config, "entry_triggers") and hasattr(
            self.config.entry_triggers, "vix_above"
        ):
            vix_trigger = self.config.entry_triggers.vix_above

        if vix_level >= vix_trigger:
            reasons.append(f"VIX level {vix_level:.1f} >= {vix_trigger}")
            should_enter = True

        # 2. Regime trigger
        if regime_label in allowed_regimes:
            reasons.append(f"Bearish market regime detected: {regime_label.upper()}")
            should_enter = True

        spy_prev_close = self._get_previous_close("SPY") if spy_price > 0 else 0.0
        spy_intraday_change = (
            ((float(spy_price) - spy_prev_close) / spy_prev_close) * 100.0
            if spy_price > 0 and spy_prev_close > 0
            else 0.0
        )
        if spy_intraday_change > 0.5:
            reasons.append(
                f"SPY intraday strength {spy_intraday_change:.2f}% suppresses inverse entry"
            )
            should_enter = False

        # 3. SPY < SMA20 trigger (was dead code — now implemented)
        if spy_price > 0 and spy_sma20 > 0 and spy_price < spy_sma20:
            pct_below = ((spy_sma20 - spy_price) / spy_sma20) * 100
            reasons.append(
                f"SPY ${spy_price:.2f} below SMA20 ${spy_sma20:.2f} ({pct_below:.1f}%)"
            )
            should_enter = True

        # 4. Consecutive red days trigger (was dead code — now implemented)
        consecutive_red_threshold = 3
        if hasattr(self.config, "entry_triggers") and hasattr(
            self.config.entry_triggers, "consecutive_red_days"
        ):
            consecutive_red_threshold = int(
                self.config.entry_triggers.consecutive_red_days
            )
        red_days = self._count_consecutive_red_days()
        if red_days >= consecutive_red_threshold:
            reasons.append(
                f"SPY has {red_days} consecutive red days (threshold: {consecutive_red_threshold})"
            )
            should_enter = True

        # 5. Exit Triggers
        bullish_regimes = [
            "strong_rally",
            "recovery",
            "grinding_up",
            "neutral",
            "trend",
        ]
        if regime_label in bullish_regimes and regime_label not in allowed_regimes:
            reasons.append(f"Bullish market regime detected: {regime_label.upper()}")
            should_exit = True

        if should_exit:
            return {
                "should_enter": False,
                "should_exit": True,
                "reason": "; ".join(reasons),
                "recommended_etfs": [],
            }

        return {
            "should_enter": should_enter,
            "should_exit": False,
            "reason": "; ".join(reasons) if reasons else "Neutral conditions",
            "recommended_etfs": self._select_etfs(regime) if should_enter else [],
        }

    def _count_consecutive_red_days(self) -> int:
        """Count consecutive red (down) days for SPY from local data."""
        try:
            from autotrade.utils.local_data_provider import get_ticker_data

            df = get_ticker_data("SPY", days=10)
            if df is None or len(df) < 2:
                return 0
            # Ensure sorted by date ascending
            if "date" in df.columns:
                df = df.sort_values("date")
            count = 0
            closes = df["close"].values
            for i in range(len(closes) - 1, 0, -1):
                if closes[i] < closes[i - 1]:
                    count += 1
                else:
                    break
            return count
        except Exception as e:
            logger.debug("Failed to count red days: %s", e)
            return 0

    def _select_etfs(self, regime: MarketRegime) -> List[str]:
        """Choose appropriate ETFs based on severity of downturn.
        Default to RWM (Russell 2000 inverse) for small/mid-cap portfolio alignment.
        """
        if regime == MarketRegime.CRASH:
            return ["SQQQ", "SPXU", "SRTY"]  # Aggressive 3x for severe crashes
        elif regime in (MarketRegime.SELLOFF, MarketRegime.CAPITULATION):
            return ["RWM", "PSQ", "SH"]  # RWM first for small/mid-cap alignment
        return ["RWM"]  # Default to Russell 2000 inverse for small/mid-cap portfolio

    def select_hedge_instrument(
        self,
        *,
        primary_index: str = "SPY",
        regime: Optional[MarketRegime] = None,
    ) -> str:
        """Pick a single hedge instrument using configured symbols and index preference."""
        configured = [
            str(symbol).upper()
            for symbol in getattr(self.config, "symbols", []) or []
            if str(symbol).upper() in INVERSE_ETF_UNIVERSE
        ]
        if not configured:
            configured = ["RWM"]  # Default to RWM for small/mid-cap alignment

        target_index = str(primary_index or "SPY").upper()
        matches = [
            symbol
            for symbol in configured
            if INVERSE_ETF_UNIVERSE.get(symbol, {}).get("index") == target_index
        ]
        candidates = matches or configured

        if regime == MarketRegime.CRASH:
            leveraged = [
                symbol
                for symbol in candidates
                if INVERSE_ETF_UNIVERSE.get(symbol, {}).get("leverage", 1) >= 3
            ]
            if leveraged:
                return leveraged[0]

        conservative = [
            symbol
            for symbol in candidates
            if INVERSE_ETF_UNIVERSE.get(symbol, {}).get("leverage", 1) == 1
        ]
        return (conservative or candidates)[0]

    def calculate_hedge_size(
        self,
        *,
        portfolio_value: float,
        net_exposure_pct: float = 100.0,
        hedge_symbol: str = "RWM",
        max_allocation_pct: Optional[float] = None,
    ) -> float:
        """
        Size hedge notional from current portfolio exposure and hedge leverage.
        """
        symbol = str(hedge_symbol).upper()
        instrument = INVERSE_ETF_UNIVERSE.get(
            symbol, INVERSE_ETF_UNIVERSE.get("RWM", {"leverage": 1})
        )
        leverage = float(instrument.get("leverage", 1) or 1)
        gross_portfolio = max(float(portfolio_value or 0.0), 0.0)
        net_exposure = max(float(net_exposure_pct or 0.0), 0.0) / 100.0
        if gross_portfolio <= 0 or net_exposure <= 0:
            return 0.0

        raw_target = gross_portfolio * net_exposure / leverage
        cap_pct = (
            float(max_allocation_pct)
            if max_allocation_pct is not None
            else float(getattr(self.config, "max_allocation_pct", 20.0) or 20.0)
        )
        hard_cap = gross_portfolio * max(cap_pct, 0.0) / 100.0
        return max(0.0, min(raw_target, hard_cap))

    def calculate_current_allocation(self, positions: List[Any]) -> float:
        """Calculate current % exposure to inverse ETFs (DB-aware)."""
        total_value = sum(abs(float(getattr(p, "market_value", 0))) for p in positions)
        if total_value == 0:
            return 0.0

        inverse_value = sum(
            abs(float(getattr(p, "market_value", 0)))
            for p in positions
            if is_inverse_etf(str(getattr(p, "symbol", "")))
        )

        return (inverse_value / total_value) * 100.0

    def rebalance_hedge(
        self,
        *,
        portfolio_value: float,
        positions: List[Any],
        net_exposure_pct: float = 100.0,
        primary_index: str = "SPY",
        regime: Optional[MarketRegime] = None,
    ) -> Dict[str, Any]:
        """
        Compare current hedge allocation to target allocation and emit a rebalance plan.
        """
        hedge_symbol = self.select_hedge_instrument(
            primary_index=primary_index,
            regime=regime,
        )
        current_allocation_pct = self.calculate_current_allocation(positions)
        target_notional = self.calculate_hedge_size(
            portfolio_value=portfolio_value,
            net_exposure_pct=net_exposure_pct,
            hedge_symbol=hedge_symbol,
        )
        gross_portfolio = max(float(portfolio_value or 0.0), 0.0)
        target_allocation_pct = (
            (target_notional / gross_portfolio) * 100.0 if gross_portfolio > 0 else 0.0
        )
        rebalance_delta_pct = target_allocation_pct - current_allocation_pct
        action = "hold"
        if rebalance_delta_pct > 0.25:
            action = "increase"
        elif rebalance_delta_pct < -0.25:
            action = "decrease"

        return {
            "hedge_symbol": hedge_symbol,
            "current_allocation_pct": current_allocation_pct,
            "target_allocation_pct": target_allocation_pct,
            "target_notional": target_notional,
            "rebalance_delta_pct": rebalance_delta_pct,
            "action": action,
        }

    def check_hedge_exit(
        self,
        *,
        current_regime: str,
        vix_level: float,
        hedge_return_pct: float = 0.0,
        symbol: str = "",
        intraday_signal: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Determine whether a hedge should be deactivated.
        Now implements profit target exit (was dead code).
        """
        profile = self.get_instrument_profile(symbol)
        exit_reasons: List[str] = []
        regime_label = str(current_regime or "neutral").lower()
        vix_below = float(
            getattr(
                getattr(self.config, "exit_triggers", None),
                "vix_below",
                18.0,
            )
            or 18.0
        )
        allowed_regimes = {
            str(r).lower()
            for r in getattr(self.config, "allowed_regimes", ["crisis", "risk_off"])
        }

        if regime_label not in allowed_regimes:
            exit_reasons.append("regime_recovery")

        if float(vix_level or 0.0) <= vix_below:
            exit_reasons.append("volatility_mean_reversion")

        # Profit target exit (was dead code — config has profit_target_pct but was unchecked)
        profit_target_pct = float(
            getattr(self.config, "profit_target_pct", profile["profit_target_pct"])
            or profile["profit_target_pct"]
        )
        pnl = float(hedge_return_pct or 0.0)
        if pnl >= profit_target_pct:
            exit_reasons.append(
                f"profit_target_hit ({pnl:.1f}% >= {profit_target_pct:.1f}%)"
            )

        # Stop loss
        stop_loss = self.manage_hedge_stop_loss(
            hedge_return_pct=hedge_return_pct,
            symbol=symbol,
        )
        if stop_loss["should_exit"]:
            exit_reasons.append(stop_loss["reason"])
        if isinstance(intraday_signal, dict):
            if intraday_signal.get("should_exit"):
                exit_reasons.append(
                    "intraday_reversal:" + ",".join(intraday_signal.get("reasons", []))
                )

        return {
            "should_exit": bool(exit_reasons),
            "reasons": exit_reasons,
            "profile": profile,
        }

    def manage_hedge_stop_loss(
        self,
        *,
        hedge_return_pct: float,
        symbol: str = "",
    ) -> Dict[str, Any]:
        """Fail closed on adverse hedge performance using a simple stop-loss threshold."""
        profile = self.get_instrument_profile(symbol)
        stop_loss_pct = float(
            getattr(self.config, "stop_loss_pct", profile["stop_loss_pct"])
            or profile["stop_loss_pct"]
        )
        pnl = float(hedge_return_pct or 0.0)
        if pnl <= stop_loss_pct:
            return {"should_exit": True, "reason": "hedge_stop_loss"}
        return {"should_exit": False, "reason": ""}
