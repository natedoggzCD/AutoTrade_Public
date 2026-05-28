"""
Multi-Model Signal Scorer - Combines All Analysis for Final Decision

This module orchestrates multiple analysis models to produce a comprehensive
trading signal with confidence scoring:

1. SR Validation (SRbatch) - Support/Resistance levels + regime classification
2. VL Chart Analysis - Vision-language pattern recognition
3. LLM Synthesis - Final reasoning combining all signals
4. ML Ensemble - Probability predictions from DownDay ensemble

Scoring Breakdown:
- Component weights are dynamically normalized from base ratios.
- Backtest weight is config-driven (`score_backtest_weight_*`) and can be
  reduced when calibration shows weak predictive lift.

Usage:
    from autotrade.utils.multimodel_scorer import MultiModelScorer
    scorer = MultiModelScorer()
    result = scorer.score_symbol('AAPL', current_price=150.0)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime
from dataclasses import dataclass
from config.config_loader import get_config

logger = logging.getLogger(__name__)


@dataclass
class SignalScore:
    """Final scored trading signal."""
    symbol: str
    final_score: float  # 0-100
    direction: str  # BULLISH, BEARISH, NEUTRAL
    confidence: str  # HIGH, MEDIUM, LOW
    
    # Component scores (0-100 each)
    technical_score: float
    sr_score: float
    vl_score: float
    ml_score: float
    sentiment_score: float
    
    # Entry/Exit levels
    entry_price: float
    stop_loss: float
    target1: float
    target2: Optional[float]
    risk_reward: float
    
    # Flags and recommendations
    action: str  # BUY, SELL, HOLD, AVOID
    flags: List[str]
    reasoning: str
    
    # Raw data
    sr_data: Optional[Dict]
    vl_data: Optional[Dict]
    llm_synthesis: Optional[str]


class MultiModelScorer:
    """
    Combines multiple analysis models for comprehensive signal scoring.
    """
    
    def __init__(self):
        self._vl_validator = None
        self._llm_helper = None
        self._bt_config = None

    def _get_backtest_config(self):
        """Lazy load backtest strategy config."""
        if self._bt_config is None:
            try:
                self._bt_config = get_config().backtest.strategy_backtester
            except Exception:
                self._bt_config = None
        return self._bt_config
    
    def _get_vl_validator(self):
        """Runtime visual validation is disabled in production."""
        if self._vl_validator is None:
            logger.info(
                "[MULTIMODEL] Visual validation disabled; VL validator will not load"
            )
        self._vl_validator = None
        return None
    
    def _get_llm_helper(self):
        """Lazy load LLM helper."""
        if self._llm_helper is None:
            try:
                from langgraph_workflow.llm_helper import LLMHelper
                self._llm_helper = LLMHelper()
            except Exception as e:
                logger.warning(f"Failed to load LLM helper: {e}")
        return self._llm_helper
    
    def _score_technical(self, sr_data: Optional[Dict], symbol: str = None) -> float:
        """Score technical indicators enriched with DownDay parquet."""
        sr_data = sr_data or {}

        # Enrich sr_data with DownDay parquet if technical fields are missing
        needs_enrichment = not sr_data.get('rsi_14') or sr_data.get('rsi_14') == 50
        if needs_enrichment and symbol:
            try:
                import duckdb
                parquet_path = str(Path("data/downday/daily_features.parquet"))
                # Use per-call connection — duckdb.execute() is NOT thread-safe
                conn = duckdb.connect(database=':memory:')
                row = conn.execute(
                    f"SELECT RSI_14, MACD_hist, ADX, SMA_20, Close "
                    f"FROM '{parquet_path}' "
                    f"WHERE ticker = ? ORDER BY date DESC LIMIT 1",
                    [symbol]
                ).fetchone()
                conn.close()
                if row:
                    rsi_14, macd_hist, adx, sma_20, close = row
                    if rsi_14 is not None:
                        sr_data['rsi_14'] = float(rsi_14)
                    if macd_hist is not None:
                        sr_data['macd_hist'] = float(macd_hist)
                    if adx is not None:
                        sr_data['adx'] = float(adx)
                    if sma_20 is not None:
                        sr_data['ma_20'] = float(sma_20)
                    if close is not None:
                        sr_data.setdefault('close_price', float(close))
                    logger.debug(f"  Enriched {symbol} technicals: RSI={rsi_14}, MACD_hist={macd_hist}, ADX={adx}")
            except Exception as e:
                logger.debug(f"  DownDay enrichment failed for {symbol}: {e}")

        score = 50  # Start neutral

        # RSI scoring (use 'or' to handle None values from dict.get)
        rsi = sr_data.get('rsi_14') or 50
        if rsi is not None and rsi < 30:
            score += 15  # Oversold = bullish
        elif rsi is not None and rsi < 40:
            score += 10
        elif rsi is not None and rsi > 70:
            score -= 15  # Overbought = bearish
        elif rsi is not None and rsi > 60:
            score -= 5

        # MACD scoring
        macd_hist = sr_data.get('macd_hist') or 0
        if macd_hist is not None and macd_hist > 0.5:
            score += 10  # Strong bullish momentum
        elif macd_hist is not None and macd_hist > 0:
            score += 5
        elif macd_hist is not None and macd_hist < -0.5:
            score -= 10  # Strong bearish momentum
        elif macd_hist is not None and macd_hist < 0:
            score -= 5

        # ADX trend strength
        adx = sr_data.get('adx') or 20
        if adx is not None and adx > 25:
            # Strong trend - amplify direction
            if sr_data.get('regime') == 'BULLISH':
                score += 5
            else:
                score -= 5
        
        # Moving average position
        close = sr_data.get('close_price', 0)
        ma20 = sr_data.get('ma_20', 0)
        ma50 = sr_data.get('ma_50', 0)
        ma200 = sr_data.get('ma_200', 0)
        
        if close and ma20 and close > ma20:
            score += 5
        if close and ma50 and close > ma50:
            score += 5
        if close and ma200 and close > ma200:
            score += 10  # Above 200 SMA is important
        
        # Relative strength
        rs_5d = sr_data.get('relative_strength_5d', 0)
        if rs_5d > 2:
            score += 10  # Strong short-term momentum
        elif rs_5d > 0:
            score += 5
        elif rs_5d < -2:
            score -= 10
        
        return max(0, min(100, score))

    def _score_sr_quality(self, sr_data: Dict, current_price: float) -> float:
        """Score based on SR levels and entry quality."""
        if not sr_data:
            return 50
        
        import math
        
        def safe_val(v, default=0):
            """Handle NaN and None values."""
            if v is None:
                return default
            if isinstance(v, float) and math.isnan(v):
                return default
            return v
        
        score = 50
        
        # Entry score from SR (0-10)
        entry_score = safe_val(sr_data.get('entry_score'), 5)
        score += (entry_score - 5) * 5  # -25 to +25
        
        # Risk/Reward ratio (apply realism discount to predictions_db rr_ratio).
        try:
            sc_cfg = get_config().screener_v2
            rr_discount = float(getattr(sc_cfg, 'sr_rr_discount_factor', 0.6))
            rr_effective_min = float(getattr(sc_cfg, 'sr_rr_effective_min', 1.2))
        except Exception:
            rr_discount = 0.6
            rr_effective_min = 1.2
        rr_raw = safe_val(sr_data.get('rr_ratio'), 1.0)
        rr = rr_raw * rr_discount
        if rr >= max(2.0 * rr_discount, rr_effective_min + 0.8):
            score += 15
        elif rr >= max(1.5 * rr_discount, rr_effective_min + 0.3):
            score += 10
        elif rr >= rr_effective_min:
            score += 5
        elif rr < 1.0:
            score -= 15
        
        # Distance to support (want close to S1)
        s1 = safe_val(sr_data.get('s1_price'))
        if s1 and current_price:
            dist_pct = ((current_price - s1) / s1) * 100
            if 0 <= dist_pct <= 2:
                score += 15  # Right at support
            elif 2 < dist_pct <= 5:
                score += 5
            elif dist_pct > 10:
                score -= 10  # Too far from support
        
        # Regime scoring
        if sr_data.get('regime') == 'BULLISH':
            score += 10
        elif sr_data.get('regime') == 'BEARISH':
            score -= 15
        
        # Check for warning flags
        caution = sr_data.get('caution_flags', '')
        if 'overbought' in caution.lower():
            score -= 10
        if 'near_highs' in caution.lower():
            score -= 5
        
        # Check for opportunity flags
        opportunity = sr_data.get('opportunity_flags', '')
        if 'at_support' in opportunity.lower():
            score += 10
        if 'healthy_pullback' in opportunity.lower():
            score += 10
        
        return max(0, min(100, score))
    
    def _score_vl_analysis(self, vl_data: Dict) -> float:
        """Score based on VL chart analysis."""
        if not vl_data or not vl_data.get('success'):
            return 50  # Neutral if no VL data
        
        score = 50
        
        # Direction scoring
        direction = vl_data.get('direction', 'NEUTRAL')
        if direction == 'BULLISH':
            score += 20
        elif direction == 'BEARISH':
            score -= 20
        
        # Confidence adjustment
        confidence = vl_data.get('confidence', 'LOW')
        multiplier = {'HIGH': 1.0, 'MEDIUM': 0.7, 'LOW': 0.3}.get(confidence, 0.5)
        score = 50 + (score - 50) * multiplier
        
        # Entry quality
        entry_quality = vl_data.get('entry_quality', 'FAIR')
        if entry_quality == 'EXCELLENT':
            score += 15
        elif entry_quality == 'GOOD':
            score += 10
        elif entry_quality == 'POOR':
            score -= 15
        
        # Trend alignment
        trend = vl_data.get('trend', 'SIDEWAYS')
        if trend == 'UPTREND':
            score += 10
        elif trend == 'DOWNTREND':
            score -= 10
        
        # Recommendation boost/penalty
        rec = vl_data.get('recommendation', 'HOLD')
        if rec == 'BUY':
            score += 10
        elif rec == 'AVOID' or rec == 'SELL':
            score -= 15
        
        return max(0, min(100, score))
    
    def _score_ml_ensemble(self, sr_data: Dict) -> float:
        """Score based on ML ensemble predictions."""
        if not sr_data:
            return 50
        
        import math
        
        # Get ML probabilities with NaN handling
        prob_1d = sr_data.get('ensemble_prob_1d', 0.5)
        prob_1w = sr_data.get('ensemble_prob_1w', 0.5)
        
        # Handle NaN/None values
        if prob_1d is None or (isinstance(prob_1d, float) and math.isnan(prob_1d)):
            prob_1d = 0.5
        if prob_1w is None or (isinstance(prob_1w, float) and math.isnan(prob_1w)):
            prob_1w = 0.5
        
        # Convert to score (0.5 = neutral 50, 1.0 = bullish 100, 0.0 = bearish 0)
        score_1d = prob_1d * 100
        score_1w = prob_1w * 100
        
        # Weight 1-week more heavily
        score = score_1d * 0.3 + score_1w * 0.7
        
        # Boost if both agree
        if prob_1d > 0.6 and prob_1w > 0.6:
            score = min(100, score + 10)
        elif prob_1d < 0.4 and prob_1w < 0.4:
            score = max(0, score - 10)
        
        return score
    
    def _llm_synthesize(self, symbol: str, sr_data: Dict, vl_data: Dict,
                        scores: Dict) -> Dict:
        """Use LLM to synthesize all signals and provide final recommendation."""
        llm = self._get_llm_helper()
        if not llm:
            return {'recommendation': 'HOLD', 'reasoning': 'LLM unavailable'}
        
        prompt = f"""You are an expert trading analyst. Synthesize the following analysis for {symbol}:

SCORES (0-100, 50=neutral):
- Technical Score: {scores['technical']:.0f}
- S/R Quality Score: {scores['sr']:.0f}
- VL Chart Score: {scores['vl']:.0f}
- ML Ensemble Score: {scores['ml']:.0f}
- WEIGHTED FINAL: {scores['final']:.0f}

SR ANALYSIS:
- Regime: {sr_data.get('regime', 'N/A') if sr_data else 'N/A'}
- Entry Score: {sr_data.get('bullish_score', 'N/A') if sr_data else 'N/A'}/10
- R:R Ratio: {sr_data.get('rr_ratio', 'N/A') if sr_data else 'N/A'}
- S1: {f"${sr_data.get('s1_price', 0):.2f}" if (sr_data and sr_data.get('s1_price')) else 'N/A'}
- R1: {f"${sr_data.get('r1_price', 0):.2f}" if (sr_data and sr_data.get('r1_price')) else 'N/A'}
- Stop Loss: {f"${sr_data.get('stop_loss', 0):.2f}" if (sr_data and sr_data.get('stop_loss')) else 'N/A'}
- Target1: {f"${sr_data.get('target1', 0):.2f}" if (sr_data and sr_data.get('target1')) else 'N/A'}

VL CHART ANALYSIS:
- Direction: {vl_data.get('direction', 'N/A') if vl_data else 'N/A'}
- Pattern: {vl_data.get('pattern', 'N/A') if vl_data else 'N/A'}
- Entry Quality: {vl_data.get('entry_quality', 'N/A') if vl_data else 'N/A'}
- Recommendation: {vl_data.get('recommendation', 'N/A') if vl_data else 'N/A'}

Provide your final verdict in JSON:
{{
    "action": "BUY | HOLD | AVOID",
    "confidence": "HIGH | MEDIUM | LOW",
    "primary_reason": "most important factor",
    "risk_factors": ["risk1", "risk2"],
    "entry_guidance": "specific entry advice"
}}"""

        try:
            result = llm.call(
                prompt=prompt,
                task='decision',
                temperature=0.1,
                max_tokens=300,
                use_cache=False
            )
            
            if result.success and result.parsed_json:
                return result.parsed_json
            else:
                # Parse from text
                return {
                    'action': 'HOLD',
                    'confidence': 'LOW',
                    'reasoning': result.content[:200] if result.content else 'No response'
                }
        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}")
            return {'action': 'HOLD', 'reasoning': str(e)}
    
    def _score_sentiment(self, sentiment_data: Optional[Dict]) -> float:
        """Score based on StockTwits + news sentiment data.

        Args:
            sentiment_data: Dict with 'stocktwits' and/or 'news' keys

        Returns:
            Score 0-100 (50 = neutral)
        """
        if not sentiment_data:
            return 50

        score = 50

        # StockTwits component
        st = sentiment_data.get('stocktwits', {})
        if st:
            bull_bear = st.get('bull_bear_ratio', 1.0)
            velocity = st.get('velocity', 0)
            trending = st.get('trending', False)
            st_score = st.get('score', 0.5)

            # Bull/bear ratio: 2.0+ is bullish, 0.5- is bearish
            if bull_bear >= 2.0:
                score += 15
            elif bull_bear >= 1.5:
                score += 10
            elif bull_bear >= 1.2:
                score += 5
            elif bull_bear <= 0.5:
                score -= 15
            elif bull_bear <= 0.7:
                score -= 10

            # Trending boost
            if trending:
                if bull_bear > 1.0:
                    score += 5  # Trending + bullish
                else:
                    score -= 5  # Trending + bearish

            # High velocity = strong interest
            if velocity > 10:
                score += 3

        # News component
        news = sentiment_data.get('news', [])
        if news and isinstance(news, list) and len(news) > 0:
            score += 5  # Presence of recent news is mildly positive (catalyst)

        return max(0, min(100, score))

    def _score_backtest(self, backtest_data: Optional[Dict]) -> float:
        """Score backtest quality — PRIMARY signal for overnight research.
        
        Backtest is our best predictive edge. This score dominates the final
        weighted calculation (50% weight). Generous scoring for winning 
        backtests; only penalize clearly bad historical performance.
        """
        if not backtest_data:
            return 50  # Neutral — no data, doesn't help or hurt

        cfg = self._get_backtest_config()

        win_rate = backtest_data.get('win_rate', 50)
        avg_gain = backtest_data.get('avg_gain', 0)
        total_trades = backtest_data.get('total_trades', 0)

        # Not enough data - require minimum trades for confidence
        min_trades = cfg.min_trades_for_validity if cfg else 12
        if total_trades < min_trades:
            return 50  # Neutral - not enough data to score

        score = 50

        # Win rate scoring — generous for winners, harsh for losers
        if win_rate >= 60:
            score += 25   # Strong edge
        elif win_rate >= 55:
            score += 18   # Good edge  
        elif win_rate >= 50:
            score += 12   # Decent edge
        elif win_rate >= 45:
            score += 5    # Marginal
        elif win_rate >= 40:
            score += 0    # Neutral
        elif win_rate >= 30:
            score -= 15   # Poor
        else:
            score -= 25   # Very poor

        # Average gain — amplifies winners, penalizes losers
        if avg_gain > 2.0:
            score += 15
        elif avg_gain > 1.0:
            score += 10
        elif avg_gain > 0.5:
            score += 5
        elif avg_gain > 0.0:
            score += 2
        elif avg_gain < -1.0:
            score -= 10

        # Reliability bump for larger sample size
        if total_trades >= max(min_trades + 8, 20):
            score += 5

        return max(0, min(100, score))

    def score_symbol(self, symbol: str, current_price: float = None,
                     include_vl: bool = False, include_llm: bool = True,
                     sentiment_data: Optional[Dict] = None,
                     backtest_data: Optional[Dict] = None) -> SignalScore:
        """
        Full multi-model scoring for a symbol.

        Args:
            symbol: Stock ticker
            current_price: Current price
            include_vl: Legacy flag only. Visual validation is disabled at runtime.
            include_llm: Whether to run LLM synthesis (slower)
            sentiment_data: Pre-fetched sentiment data (StockTwits + news)
            backtest_data: Pre-fetched backtest results

        Returns:
            SignalScore with comprehensive analysis
        """
        logger.info(f"[MULTIMODEL] Scoring {symbol}...")
        sr_data: Optional[Dict] = None
        
        try:
            # 1. Get Price
            # Fallback: pull price from local data provider (uses local cache or live API)
            if not current_price:
                try:
                    from autotrade.utils.local_data_provider import LocalDataProvider
                    ldp = LocalDataProvider()
                    price, source = ldp.get_current_price(symbol)
                    if price:
                        current_price = price
                        logger.debug(f"  Price from LocalDataProvider ({source})")
                except Exception as e:
                    logger.debug(f"  LocalDataProvider price fetch failed: {e}")

            # If still no price, skip this symbol
            if not current_price:
                logger.warning(f"  No price data for {symbol}, skipping")
                return self._create_default_score(symbol, "No price data available")
            
            vl_data = None
            if include_vl:
                logger.info(
                    f"[MULTIMODEL] {symbol}: include_vl requested but visual validation is disabled"
                )
            
            # 3. Calculate component scores
            # Note: sr_data is now None as the legacy provider is retired.
            sr_data = None
            technical_score = self._score_technical(None, symbol=symbol)
            sr_score = self._score_sr_quality(None, current_price)
            vl_score = self._score_vl_analysis(vl_data) if vl_data else 50
            ml_score = self._score_ml_ensemble(None)
            sentiment_score = self._score_sentiment(sentiment_data)
            backtest_score = self._score_backtest(backtest_data)

            # 4. Weighted final score
            # BACKTEST-PRIMARY scoring model (v0.22.0):
            # Backtest is our best predictive edge — dominates scoring.
            # Technical (RSI/MACD/MA) confirms setup quality.
            # Sentiment (news/StockTwits) catches disqualifying bad news.
            # SR quality and ML ensemble REMOVED — not predictive enough.
            final_score = (
                backtest_score * 0.50 +       # Primary signal — historical edge
                technical_score * 0.25 +       # RSI/MACD/MA confirmation
                sentiment_score * 0.25         # News/StockTwits veto gate
            )
            
            scores_dict = {
                'technical': technical_score,
                'sr': sr_score,
                'vl': vl_score,
                'ml': ml_score,
                'sentiment': sentiment_score,
                'backtest': backtest_score,
                'final': final_score
            }

            logger.debug(f"  Scores: BT={backtest_score:.0f}(50%) Tech={technical_score:.0f}(25%) Sent={sentiment_score:.0f}(25%) [unused: SR={sr_score:.0f} ML={ml_score:.0f}] -> Final={final_score:.0f}")
            
            # 5. LLM Synthesis
            llm_synthesis = None
            llm_result = None
            if include_llm:
                if not sr_data:
                    logger.info("  Skipping LLM synthesis (no SR data available)")
                    include_llm = False
            if include_llm:
                llm_result = self._llm_synthesize(symbol, sr_data, vl_data, scores_dict)
                llm_synthesis = json.dumps(llm_result, indent=2)
            
            # 6. Determine direction and confidence
            # v0.12.1: Tightened — require >= 65 for BULLISH (was >= 50)
            if final_score >= 65:
                direction = 'BULLISH'
                confidence = 'HIGH' if final_score >= 75 else 'MEDIUM'
                action = 'BUY'
            elif final_score >= 55:
                direction = 'NEUTRAL'
                confidence = 'LOW'
                action = 'HOLD'
            elif final_score <= 35:
                direction = 'BEARISH'
                confidence = 'HIGH' if final_score <= 25 else 'MEDIUM'
                action = 'AVOID'
            else:
                direction = 'NEUTRAL'
                confidence = 'LOW'
                action = 'HOLD'
            
            # 7. Entry/Exit levels — pure ATR-based (S/R data excluded — unreliable)
            entry_price = current_price or (sr_data.get('close_price') if sr_data else 0)
            atr_val = sr_data.get('atr_14', 0) if sr_data else 0
            if not atr_val and entry_price:
                atr_val = entry_price * 0.03  # 3% fallback
            _strat_cfg = get_config().strategy
            stop_loss = entry_price - (atr_val * _strat_cfg.fallback_stop_atr) if entry_price else 0
            target1 = entry_price + (atr_val * _strat_cfg.fallback_target_atr) if entry_price else 0
            target2 = None

            # Calculate R:R
            risk = entry_price - stop_loss if entry_price and stop_loss else 0
            reward = target1 - entry_price if target1 and entry_price else 0
            risk_reward = reward / risk if risk > 0 else 1.0
            
            # 8. Collect flags
            flags = []
            if sr_data:
                if sr_data.get('caution_flags'):
                    flags.extend(sr_data['caution_flags'].split(','))
                if sr_data.get('opportunity_flags'):
                    flags.extend([f"+{f}" for f in sr_data['opportunity_flags'].split(',')])
            if vl_data and vl_data.get('key_observations'):
                flags.extend(vl_data['key_observations'])
            
            # 9. Build reasoning
            reasoning = llm_result.get('primary_reason', '') if llm_result else ''
            if not reasoning:
                reasoning = f"Score {final_score:.0f}/100: BT={backtest_score:.0f}, Tech={technical_score:.0f}, Sent={sentiment_score:.0f}"
            
            logger.info(f"  Final: {final_score:.0f}/100 -> {direction} ({confidence}) -> {action}")
            
            return SignalScore(
                symbol=symbol,
                final_score=final_score,
                direction=direction,
                confidence=confidence,
                technical_score=technical_score,
                sr_score=sr_score,
                vl_score=vl_score,
                ml_score=ml_score,
                sentiment_score=sentiment_score,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target1=target1,
                target2=target2,
                risk_reward=risk_reward,
                action=action,
                flags=flags,
                reasoning=reasoning,
                sr_data=sr_data,
                vl_data=vl_data,
                llm_synthesis=llm_synthesis,
            )
        
        except Exception as e:
            logger.exception(f"[MULTIMODEL] Scoring failed for {symbol}: {e}")
            return self._create_default_score(symbol, f"Scoring error: {str(e)}")
    
    def _create_default_score(self, symbol: str, reason: str) -> SignalScore:
        """Create a neutral default score when analysis fails."""
        return SignalScore(
            symbol=symbol,
            final_score=50.0,
            direction='NEUTRAL',
            confidence='LOW',
            technical_score=50.0,
            sr_score=50.0,
            vl_score=50.0,
            ml_score=50.0,
            sentiment_score=50.0,
            entry_price=0.0,
            stop_loss=0.0,
            target1=0.0,
            target2=None,
            risk_reward=0.0,
            action='HOLD',
            flags=[],
            reasoning=reason,
            sr_data=None,
            vl_data=None,
            llm_synthesis=None,
        )
    
    def score_batch(self, symbols: List[str], include_vl: bool = False,
                    include_llm: bool = True) -> List[SignalScore]:
        """Score multiple symbols."""
        results = []
        for symbol in symbols:
            try:
                score = self.score_symbol(symbol, include_vl=include_vl, include_llm=include_llm)
                results.append(score)
            except Exception as e:
                logger.error(f"Failed to score {symbol}: {e}")
        
        # Sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results


# Singleton
_scorer: Optional[MultiModelScorer] = None


def get_scorer() -> MultiModelScorer:
    """Get singleton scorer instance."""
    global _scorer
    if _scorer is None:
        _scorer = MultiModelScorer()
    return _scorer


def score_symbol(symbol: str, **kwargs) -> SignalScore:
    """Convenience function to score a symbol."""
    return get_scorer().score_symbol(symbol, **kwargs)


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    scorer = MultiModelScorer()
    
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
        result = scorer.score_symbol(symbol)
        
        print(f"\n{'='*60}")
        print(f"MULTI-MODEL ANALYSIS: {symbol}")
        print(f"{'='*60}")
        print(f"\nFINAL SCORE: {result.final_score:.0f}/100")
        print(f"Direction: {result.direction} ({result.confidence})")
        print(f"Action: {result.action}")
        print(f"\nComponent Scores:")
        print(f"  Technical: {result.technical_score:.0f}")
        print(f"  S/R Quality: {result.sr_score:.0f}")
        print(f"  VL Chart: {result.vl_score:.0f}")
        print(f"  ML Ensemble: {result.ml_score:.0f}")
        print(f"\nEntry Levels:")
        print(f"  Entry: ${result.entry_price:.2f}")
        print(f"  Stop: ${result.stop_loss:.2f}")
        print(f"  Target 1: ${result.target1:.2f}")
        print(f"  R:R Ratio: {result.risk_reward:.2f}")
        print(f"\nReasoning: {result.reasoning}")
        if result.flags:
            print(f"\nFlags: {', '.join(result.flags[:5])}")
    else:
        print("Usage: python multimodel_scorer.py SYMBOL")
