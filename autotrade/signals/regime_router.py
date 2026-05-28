"""
Regime-Adaptive Strategy Router.
Routes strategy families by market regime and applies sector bias.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from config.config_loader import get_config
from autotrade.signals.contracts import SignalFamily, RegimeLabel

logger = logging.getLogger(__name__)


class RegimeRouter:
    """
    Router that decides which strategies are allowed in the current market regime.
    """

    def __init__(self, parent_logger=None):
        self.config = get_config()
        self.logger = parent_logger or logger
        self.regime_cfg = getattr(self.config.signal_generation, "regime", None)
        # family_enable_policy is a dict: {family_name: [regime1, regime2]}
        default_policy = {
            SignalFamily.TS_MOMENTUM.value: [RegimeLabel.TREND.value],
            SignalFamily.MEAN_REVERSION.value: [
                RegimeLabel.CHOP.value,
                RegimeLabel.CRISIS.value,
            ],
        }
        cfg_policy = (
            getattr(self.regime_cfg, "family_enable_policy", {})
            if self.regime_cfg
            else {}
        )
        self.policy = dict(default_policy)
        if isinstance(cfg_policy, dict):
            for family, regimes in cfg_policy.items():
                self.policy[str(family)] = (
                    list(regimes)
                    if isinstance(regimes, (list, tuple, set))
                    else regimes
                )

    @property
    def name(self) -> str:
        """Unique identifier for this router."""
        return "regime_router"

    @property
    def default_regime(self) -> RegimeLabel:
        """Default regime when routing fails."""
        return RegimeLabel.NEUTRAL

    def filter_signals_by_regime(
        self,
        signals: List[Any],
        regime: Any,
        price_data: Optional[Any] = None,
    ) -> Tuple[List[Any], List[Any]]:
        """Compatibility API expected by SignalGenerationPipeline.

        Returns a tuple of (accepted, filtered) while preserving input order.
        """
        _ = price_data  # Reserved for future trigger-aware filtering.
        accepted = self.filter_allowed_strategies(regime, signals)

        # Preserve stable ordering and object identity semantics.
        accepted_ordered: List[Any] = []
        filtered: List[Any] = []
        accepted_iter = iter(accepted)
        sentinel = object()
        next_accepted = next(accepted_iter, sentinel)
        for signal in signals:
            if signal is next_accepted:
                accepted_ordered.append(signal)
                next_accepted = next(accepted_iter, sentinel)
            else:
                filtered.append(signal)
        return accepted_ordered, filtered

    def _get_regime_label(self, value: Any) -> RegimeLabel:
        """Helper to safely get a RegimeLabel from various input types."""
        if isinstance(value, RegimeLabel):
            return value
        val_str = str(getattr(value, "value", value) or "").strip()
        return RegimeLabel.from_string(val_str)

    def filter_allowed_strategies(
        self, regime: Any, candidates: List[Any]
    ) -> List[Any]:
        """
        Filter out candidates whose strategy family is not allowed in the current regime.

        Args:
            regime: Current market regime (trend, chop, crisis, neutral)
            candidates: List of signal objects or dicts

        Returns:
            Filtered list of candidates
        """
        if not regime:
            return candidates

        regime_label_obj = self._get_regime_label(regime)
        regime_label = regime_label_obj.value
        allowed = []

        for cand in candidates:
            # Check if candidate has a family field
            family = None
            if isinstance(cand, dict):
                family = cand.get("family")
            else:
                family = getattr(cand, "family", None)
                if hasattr(family, "value"):
                    family = family.value

            if not family:
                # If no family, allow it (legacy or generic)
                allowed.append(cand)
                continue

            # Check policy
            # Policy uses family names as keys (e.g., 'ts_momentum')
            supported_regimes = self.policy.get(family, [])

            # If no policy for this family, allow it (opt-out vs opt-in)
            if not supported_regimes:
                allowed.append(cand)
                continue

            # Normalize supported regimes to strings
            supported_strs = [str(r).lower() for r in supported_regimes]

            if regime_label in supported_strs:
                allowed.append(cand)
            else:
                if isinstance(cand, dict):
                    symbol = cand.get("ticker") or cand.get("symbol") or "?"
                else:
                    symbol = (
                        getattr(cand, "symbol", None)
                        or getattr(cand, "ticker", None)
                        or "?"
                    )
                self.logger.debug(
                    f"[ROUTER] Filtering {symbol}: family '{family}' not allowed in regime '{regime_label}'"
                )

        return allowed

    def resolve_youtube_divergence(
        self,
        quantitative_regime: Any,
        quantitative_confidence: float,
        youtube_regime: Any,
        youtube_confidence: float,
        *,
        low_confidence_threshold: float = 0.55,
        youtube_weight_on_low: float = 0.4,
    ) -> Dict[str, Any]:
        """
        Resolve divergent regimes by applying a YouTube tiebreaker when
        quantitative confidence is low.

        Returns a dict with resolved regime, confidence, method, and weights.
        """
        qt_label = self._get_regime_label(quantitative_regime)
        yt_label = self._get_regime_label(youtube_regime)

        def _normalize_conf(value: Any) -> float:
            try:
                conf = float(value or 0.0)
            except Exception:
                return 0.0
            if conf > 1.0:
                conf = conf / 100.0
            return max(0.0, min(1.0, conf))

        qt_conf = _normalize_conf(quantitative_confidence)
        yt_conf = _normalize_conf(youtube_confidence)

        if yt_label == RegimeLabel.NEUTRAL and qt_label != RegimeLabel.NEUTRAL:
            return {
                "regime": qt_label,
                "confidence": qt_conf,
                "method": "quantitative",
                "weights": {"quantitative": 1.0, "youtube": 0.0},
            }

        if qt_label == yt_label:
            return {
                "regime": qt_label,
                "confidence": max(qt_conf, yt_conf),
                "method": "aligned",
                "weights": {"quantitative": 1.0, "youtube": 0.0},
            }

        if qt_conf >= float(low_confidence_threshold):
            return {
                "regime": qt_label,
                "confidence": qt_conf,
                "method": "quantitative",
                "weights": {"quantitative": 1.0, "youtube": 0.0},
            }

        youtube_weight = max(0.0, min(1.0, float(youtube_weight_on_low)))
        quant_weight = max(0.0, 1.0 - youtube_weight)
        qt_score = qt_conf * quant_weight
        yt_score = yt_conf * youtube_weight

        if yt_score > qt_score:
            chosen = yt_label
            method = "youtube_tiebreaker"
            chosen_conf = yt_conf
        else:
            chosen = qt_label
            method = "quantitative"
            chosen_conf = qt_conf

        return {
            "regime": chosen,
            "confidence": chosen_conf,
            "method": method,
            "weights": {"quantitative": quant_weight, "youtube": youtube_weight},
        }

    def apply_sector_bias(
        self, regime_report: Dict, candidates: List[Any]
    ) -> List[Any]:
        """
        Adjust scores based on sector bias from market intelligence (e.g. YouTube).

        Args:
            regime_report: Intelligence report containing avoid_sectors and favor_sectors
            candidates: List of signal objects or dicts

        Returns:
            Candidates with adjusted scores
        """
        avoid = [s.lower() for s in (regime_report.get("avoid_sectors") or [])]
        favor = [s.lower() for s in (regime_report.get("favor_sectors") or [])]

        if not avoid and not favor:
            return candidates

        boosted = 0
        penalized = 0

        for cand in candidates:
            sector = ""
            if isinstance(cand, dict):
                sector = str(cand.get("sector", "")).lower()
            else:
                sector = str(getattr(cand, "sector", "")).lower()

            if not sector:
                continue

            score_adj = 0.0
            if any(a in sector for a in avoid if a):
                score_adj -= 10.0
                penalized += 1
            elif any(f in sector for f in favor if f):
                score_adj += 8.0
                boosted += 1

            if score_adj != 0:
                if isinstance(cand, dict):
                    # Check for various score fields
                    for field in ["final_score", "score", "composite_score"]:
                        if field in cand:
                            cand[field] = float(cand[field] or 0.0) + score_adj
                    cand["_sector_adj"] = score_adj
                else:
                    if hasattr(cand, "final_score"):
                        cand.final_score = float(cand.final_score or 0.0) + score_adj
                    if hasattr(cand, "score"):
                        cand.score = float(cand.score or 0.0) + score_adj
                    cand._sector_adj = score_adj

        if boosted or penalized:
            self.logger.info(
                f"[ROUTER] Sector bias applied: {boosted} boosted, {penalized} penalized"
            )

        return candidates

    def get_current_regime(self, price_data=None, feature_data=None) -> Any:
        """
        Detect current regime based on intraday data and triggers.
        """

        # Define return structure locally for consistency
        class RegimeResult:
            def __init__(self, regime, confidence, method):
                self.regime = regime
                self.confidence = confidence
                self.method = method
                self.timestamp = datetime.now()

        # Stage-2 feature-driven regime if available.
        try:
            if (
                feature_data is not None
                and hasattr(feature_data, "empty")
                and not feature_data.empty
                and hasattr(feature_data, "columns")
                and "regime_label" in feature_data.columns
            ):
                latest_regime = str(
                    feature_data.iloc[-1].get("regime_label", "")
                ).strip()
                if latest_regime:
                    return RegimeResult(
                        regime=RegimeLabel.from_string(latest_regime),
                        confidence=0.7,
                        method="feature_label",
                    )
        except Exception as e:
            self.logger.debug(f"[ROUTER] Feature regime parse failed: {e}")

        # Fallback to neutral if no data
        if price_data is None or (hasattr(price_data, "empty") and price_data.empty):
            return RegimeResult(
                regime=RegimeLabel.NEUTRAL, confidence=0.5, method="router_fallback"
            )

        # Basic trigger logic (Phase 3 step 4):
        # If SPY is down significantly from open, force CRISIS/CHOP
        try:

            def _row_value(row: Any, key: str, default: float = 0.0) -> float:
                try:
                    if isinstance(row, dict):
                        return float(row.get(key, default) or default)
                    return float(row[key] or default)
                except Exception:
                    return float(default)

            def _series_values(bars: Any, key: str) -> List[float]:
                values: List[float] = []
                if isinstance(bars, list):
                    for row in bars:
                        value = _row_value(row, key, 0.0)
                        if value > 0:
                            values.append(value)
                    return values
                if hasattr(bars, "columns") and key in getattr(bars, "columns", []):
                    try:
                        return [
                            float(v)
                            for v in bars[key].dropna().tolist()
                            if float(v) > 0
                        ]
                    except Exception:
                        return []
                return values

            def _context_value(keys: Tuple[str, ...], default: float = 0.0) -> float:
                containers = [price_data]
                if spy_bars is not price_data:
                    containers.append(spy_bars)
                for container in containers:
                    if isinstance(container, dict):
                        for key in keys:
                            if key in container:
                                return _row_value(container, key, default)
                    if hasattr(container, "columns"):
                        for key in keys:
                            if key in getattr(container, "columns", []):
                                try:
                                    return float(container[key].dropna().iloc[-1])
                                except Exception:
                                    continue
                return float(default)

            # Handle both DataFrame and dict of bars
            spy_bars = None
            if isinstance(price_data, dict):
                spy_bars = price_data.get("SPY")
            elif hasattr(price_data, "index") and getattr(
                price_data.index, "names", None
            ):
                if "symbol" in list(price_data.index.names):
                    try:
                        spy_bars = price_data.xs("SPY", level="symbol")
                    except Exception:
                        spy_bars = None
                elif hasattr(price_data, "columns") and "symbol" in list(
                    price_data.columns
                ):
                    spy_bars = price_data.query("symbol == 'SPY'")
                else:
                    # Caller already provided SPY-only bars.
                    spy_bars = price_data
            elif hasattr(price_data, "columns") and "symbol" in list(
                price_data.columns
            ):
                spy_bars = price_data.query("symbol == 'SPY'")
            else:
                spy_bars = price_data

            if spy_bars is not None and not (
                hasattr(spy_bars, "empty") and spy_bars.empty
            ):
                current = 0.0
                opening = 0.0

                if isinstance(spy_bars, list):
                    current = spy_bars[-1].get("close", 0)
                    opening = spy_bars[0].get("open", 0)
                elif hasattr(spy_bars, "iloc"):
                    close_col = (
                        "close" if "close" in getattr(spy_bars, "columns", []) else None
                    )
                    open_col = (
                        "open" if "open" in getattr(spy_bars, "columns", []) else None
                    )
                    if close_col and open_col:
                        current = float(spy_bars.iloc[-1][close_col] or 0.0)
                        opening = float(spy_bars.iloc[0][open_col] or 0.0)

                if opening > 0:
                    change = (current - opening) / opening

                    if change < -0.025:
                        return RegimeResult(
                            regime=RegimeLabel.CRISIS,
                            confidence=0.9,
                            method="intraday_trigger",
                        )
                closes = _series_values(spy_bars, "close")
                market_return_5d = _context_value(
                    ("market_return_5d", "market_5d_return", "spy_return_5d"),
                    0.0,
                )
                if len(closes) >= 5:
                    start = closes[-6] if len(closes) >= 6 else closes[0]
                    if start > 0:
                        market_return_5d = (closes[-1] - start) / start
                breadth_negative_flag = bool(_context_value(("breadth_negative",), 0.0))
                breadth_pct_positive = _context_value(
                    ("breadth_pct_positive", "breadth_positive_pct"),
                    50.0,
                )
                breadth_negative = breadth_negative_flag or breadth_pct_positive <= 45.0
                realized_vol = _context_value(
                    ("realized_volatility", "realized_vol", "volatility"),
                    0.0,
                )
                realized_vol_p70 = _context_value(
                    ("realized_vol_p70", "volatility_p70", "realized_volatility_p70"),
                    0.0,
                )
                elevated_vol = realized_vol > 0.0 and (
                    (realized_vol_p70 > 0.0 and realized_vol >= realized_vol_p70)
                    or (realized_vol_p70 <= 0.0 and realized_vol >= 0.025)
                )
                if market_return_5d <= -0.02:
                    return RegimeResult(
                        regime=RegimeLabel.BEARISH,
                        confidence=0.75,
                        method="sustained_bearish_5d",
                    )
                if breadth_negative and elevated_vol:
                    return RegimeResult(
                        regime=RegimeLabel.BEARISH,
                        confidence=0.7,
                        method="breadth_vol_bearish",
                    )
                if opening > 0:
                    if change < -0.015:
                        return RegimeResult(
                            regime=RegimeLabel.CHOP,
                            confidence=0.7,
                            method="intraday_trigger",
                        )
        except Exception as e:
            self.logger.debug(f"[ROUTER] Intraday trigger check failed: {e}")

        return RegimeResult(
            regime=RegimeLabel.NEUTRAL, confidence=0.5, method="router_default"
        )


def create_regime_router(parent_logger=None) -> RegimeRouter:
    """Factory helper retained for callers expecting a create_* API."""
    return RegimeRouter(parent_logger=parent_logger)
