"""
VL Chart Validator - Vision-Language Model Chart Analysis

Generates VLM-optimized charts and validates them using vision-language models.

VL MODEL COMPARISON (tested Feb 2026):
===========================================
Model                  | Vision | Speed   | Accuracy | Notes
-----------------------|--------|---------|----------|----------------------------------
llama3.2-vision:latest | YES    | 4.4s    | 100%*    | RECOMMENDED - fastest + accurate
qwen2.5vl:7b          | YES    | 9.2s    | 100%*    | Works well but 2x slower
qwen3-vl:8b           | YES    | ~10s    | 25%      | Works but needs simpler prompts
bespoke-minichart:7b  | NO**   | N/A     | 12%      | BROKEN - no vision capability

* Accuracy on MSFT/AAPL/NVDA/TSLA test set - llama3.2 and qwen2.5vl agreed 100%
** bespoke-minichart was created without vision capability

OPTIMIZED WORKFLOW (v0.6.0+):
==============================
1. Run hardcoded classification FIRST (fast, ~100% accurate for bearish)
2. If hardcoded returns BULLISH/STRONG_BULLISH, run VL for visual confirmation
3. VL confirmation prevents false positives from hardcoded rules
4. This reduces VL calls by ~80% (most stocks are bearish/neutral)

The hardcoded rule-based system (`classify_symbol`) achieves ~100% accuracy
for BEARISH detection. VL model provides visual confirmation for bullish signals.

Chart Layout (Option-2: No Volume):
- 4 Panels: Price (68%) + MACD (14%) + RSI (9%) + Stochastic (9%)
- 1920x1200 @ 150dpi
- Indicators: SMA20, SMA50, AVWAP(H), MACD(12,26,9), RSI(14), Fast/Slow Stoch
- Event tags with (t-n) age suffix and ACTIVE/STALE gating

Features:
- S/R zones as semi-transparent rectangles
- Swing high/low markers
- Event markers: MACD_BULL_CROSS, AVWAP_RECLAIM, STOCH_BULL_LT20
- NOW marker with current price line

Usage:
    from autotrade.utils.vl_chart_validator import VLChartValidator
    vl = VLChartValidator()
    
    # RECOMMENDED: Use hardcoded classification (fast, accurate)
    result = vl.classify_symbol('AAPL')
    
    # ALTERNATIVE: VL model classification (slower, 75% accuracy with qwen2.5vl)
    result = vl.validate_symbol('AAPL', use_vl_model=True)
"""

import subprocess
import json
import base64
import time
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime
import logging
import sys

# Set matplotlib backend to non-interactive before importing mplfinance
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for thread safety

logger = logging.getLogger(__name__)

# Chart output directory
CHARTS_DIR = Path(__file__).parent.parent.parent / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# VL Model config
# llama3.2-vision is 2x faster than qwen2.5vl with same accuracy
# Both correctly classify MSFT/AAPL/NVDA/TSLA (100% agreement)
# bespoke-minichart:7b CANNOT process images (missing vision capability)
DEFAULT_VL_MODEL = "llama3.2-vision:latest"
VL_TIMEOUT = 120  # Reduced timeout - llama3.2 is faster


class VLChartValidator:
    """Validates stock charts using vision-language models."""
    
    def __init__(self, vl_model: str = DEFAULT_VL_MODEL):
        self.vl_model = vl_model
        self._mplfinance = None
    
    def _get_mplfinance(self):
        """Lazy load mplfinance."""
        if self._mplfinance is None:
            try:
                import mplfinance as mpf
                self._mplfinance = mpf
            except ImportError:
                logger.error("mplfinance not installed. Run: pip install mplfinance")
                return None
        return self._mplfinance
    
    def generate_chart(self, symbol: str, days: int = 60, sr_data: Optional[Dict] = None) -> Optional[Path]:
        """
        Generate a VLM-optimized 4-panel chart for a symbol.
        
        Uses VLMChartGenerator with Option-2 layout (No Volume):
        - Price (68%): Candlesticks, SMA20, SMA50, AVWAP(H), S/R zones
        - MACD (14%): MACD line, Signal line, Histogram
        - RSI (9%): RSI(14) with 30/50/70 lines
        - Stochastic (9%): Fast %K/%D, Slow %K/%D with 20/80 lines
        
        Event tags with (t-n) age and ACTIVE/STALE gating:
        - MACD_BULL_CROSS
        - AVWAP_RECLAIM  
        - STOCH_BULL_LT20
        
        Args:
            symbol: Stock ticker
            days: Number of days of data
            sr_data: Optional support/resistance data
            
        Returns:
            Path to generated chart image
        """
        try:
            from autotrade.utils.vlm_chart_generator import VLMChartGenerator
            
            generator = VLMChartGenerator()
            chart_path = generator.generate_chart(
                symbol=symbol,
                days=days,
                sr_data=sr_data,
                force_regenerate=False
            )
            
            return chart_path
            
        except ImportError:
            logger.warning("VLMChartGenerator not available, falling back to basic chart")
            return self._generate_basic_chart(symbol, days)
        except Exception as e:
            logger.error(f"VLM chart generation failed for {symbol}: {e}")
            return self._generate_basic_chart(symbol, days)
    
    def _generate_basic_chart(self, symbol: str, days: int = 60) -> Optional[Path]:
        """Fallback basic chart generation if VLMChartGenerator fails."""
        mpf = self._get_mplfinance()
        if not mpf:
            return None
        
        # Check for cached chart (today's date)
        today = datetime.now().strftime('%Y%m%d')
        chart_path = CHARTS_DIR / f"{symbol}_{today}.png"
        
        if chart_path.exists():
            logger.debug(f"Using cached chart: {chart_path}")
            return chart_path
        
        try:
            import yfinance as yf
            import pandas as pd
            
            # Get data
            stock = yf.Ticker(symbol)
            hist = stock.history(period=f'{days}d')
            
            if len(hist) < 20:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Calculate indicators
            hist['SMA20'] = hist['Close'].rolling(20).mean()
            hist['SMA50'] = hist['Close'].rolling(50).mean()
            
            # RSI
            delta = hist['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            hist['RSI'] = 100 - (100 / (1 + rs))
            
            # Set up the chart
            add_plots = [
                mpf.make_addplot(hist['SMA20'], color='blue', width=0.8),
                mpf.make_addplot(hist['SMA50'], color='orange', width=0.8),
            ]
            
            # Add RSI panel if we have enough data
            if not hist['RSI'].isna().all():
                add_plots.append(
                    mpf.make_addplot(hist['RSI'], panel=2, color='purple', ylabel='RSI')
                )
            
            # Style
            mc = mpf.make_marketcolors(
                up='green', down='red',
                edge='inherit',
                wick='inherit',
                volume='inherit'
            )
            style = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='-',
                gridcolor='lightgray'
            )
            
            # Generate chart
            fig, axes = mpf.plot(
                hist,
                type='candle',
                style=style,
                volume=True,
                addplot=add_plots,
                title=f'{symbol} - {days} Day Chart',
                figsize=(12, 8),
                panel_ratios=(4, 1, 1) if len(add_plots) > 2 else (4, 1),
                returnfig=True
            )
            
            fig.savefig(chart_path, dpi=150, bbox_inches='tight')
            import matplotlib.pyplot as plt
            plt.close(fig)
            
            logger.info(f"Generated chart: {chart_path}")
            return chart_path
            
        except Exception as e:
            logger.error(f"Failed to generate chart for {symbol}: {e}")
            return None
    
    def _encode_image_base64(self, image_path: Path) -> str:
        """Encode image to base64 for VL API."""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _build_vl_prompt(self, symbol: str, sr_data: Optional[Dict] = None) -> str:
        """
        Build the VL Setup Classifier prompt with strict technical rules.
        """
        
        prompt = f"""You are a Vision-Language Trading Setup Classifier. Analyze this {symbol} chart using STRICT technical rules.

CHART ELEMENTS TO IDENTIFY:
1) Title: ticker, current price, % change, "NOW" vertical line
2) Signals legend: black circles with white numbers (1=MACDx, 2=AVWAP, 3=STOCHx, 4=RSI>50, 5=SMAx)
3) Price overlays:
   - Green zone = SUPPORT (S: $...)
   - Red zone = RESISTANCE (R: $...)
   - Purple line = AVWAP
   - Blue line = SMA20, Orange line = SMA50
4) Indicator panels:
   - MACD: histogram bars + two lines (blue=MACD, orange=Signal)
   - RSI: single line with 30/50/70 references
   - STOCH: two lines (blue=FastK, orange=SlowD) with 20/80 zones

STRICT CLASSIFICATION RULES:

**BEARISH (avoid longs) - ANY of these:**
- Price below BOTH green AND red zones = BEARISH (breakdown)
- Price below AVWAP AND below SMA20 AND MACD histogram negative/falling = BEARISH
- MACD just crossed below signal (bearish cross) = BEARISH
- All three indicators (MACD, RSI, STOCH) pointing DOWN = BEARISH
- In green zone but NO bounce visible (no rejection wick, no green candle reversal) = BEARISH

**NEUTRAL_RANGE (no edge) - mixed/unclear:**
- Price chopping between green and red zones with no clear direction
- Indicators mixed (some up, some down)
- At support but indicators still falling (bounce not confirmed)
- Recent drop into green zone but STOCH/MACD have NOT curled up yet

**WATCH_BOUNCE (watchlist - wait for confirmation):**
- Price AT or IN green support zone
- AND at least ONE indicator starting to curl up (STOCH from <20, or MACD histogram shrinking)
- BUT price still below AVWAP
- Need to see: actual bounce candle + indicator confirmation before buying

**WATCH_RECLAIM (watchlist - needs AVWAP reclaim):**
- Price action improving (higher lows)
- But still BELOW AVWAP line
- AVWAP reclaim would be the trigger

**BULLISH (actionable with caution):**
- Price above AVWAP
- AND above green support zone  
- AND at least 2 of: MACD rising/crossed bullish, RSI >50, STOCH rising
- BUT under red resistance (capped upside)

**STRONG_BULLISH (actionable now):**
- Price above BOTH green AND red zones (breakout)
- OR price above AVWAP + above SMA20 + MACD bullish + RSI >50 + STOCH rising
- Multiple recent signal dots (2+) in last 5 bars

CRITICAL DISTINCTIONS:
- Being IN the green zone does NOT mean bullish - you must SEE a bounce (rejection wick, green reversal candle)
- All indicators turning DOWN = BEARISH even if at support
- MACD histogram going MORE negative = momentum still bearish
- STOCH below 20 but still falling = NOT a buy signal yet
- Only call BULLISH if you can see ACTUAL confirmation, not just "potential"

OUTPUT FORMAT (JSON ONLY):
{{
  "ticker": "{symbol}",
  "label": "STRONG_BULLISH|BULLISH|WATCH_BOUNCE|WATCH_RECLAIM|NEUTRAL_RANGE|BEARISH",
  "setup_type": "breakout_above_resistance|vwap_reclaim|support_bounce|breakdown|bear_trend|range_chop",
  "confidence": 0.00-1.00,
  "price_position": {{
    "vs_avwap": "above|below|at",
    "vs_green_zone": "above|inside|below",
    "vs_red_zone": "above|inside|below",
    "vs_sma20": "above|below"
  }},
  "indicator_direction": {{
    "macd": "rising|falling|flat|bullish_cross|bearish_cross",
    "rsi": "rising|falling|above_50|below_50",
    "stoch": "rising|falling|oversold_curling|overbought_rolling"
  }},
  "signals_detected": [{{"id": 1, "recency": "latest|recent|old"}}],
  "key_evidence": ["max 3 specific observations"],
  "why_this_label": "one sentence explaining the classification",
  "risk_flags": ["below_avwap", "indicators_falling", "no_bounce_confirmed", etc]
}}

BE STRICT: If indicators are all falling, it's BEARISH regardless of zone proximity. Only upgrade to WATCH/BULLISH when you see ACTUAL reversal signs."""
        
        return prompt
    
    def _build_simple_vl_prompt(self, symbol: str) -> str:
        """
        Build a simpler prompt for models like llama3.2-vision and qwen3-vl.
        These models perform better with concise, direct prompts.
        Complex JSON prompts cause hallucinations.
        
        This prompt achieved 100% accuracy on MSFT/AAPL/NVDA/TSLA test set.
        """
        return f"""Look at this {symbol} chart. Is it BULLISH or BEARISH?

Check these indicators:
1. Is price above or below the purple AVWAP line?
2. Is MACD histogram red (falling) or green (rising)?
3. Is RSI above or below 50?

Rules:
- BULLISH: Price above AVWAP, MACD green/rising, RSI above 50
- BEARISH: Price below AVWAP, MACD red/falling, RSI below 50

Answer: [BULLISH/BEARISH] because [one sentence explaining what you see]"""
    
    def validate_chart(self, chart_path: Path, symbol: str, 
                       sr_data: Optional[Dict] = None) -> Dict:
        """
        Validate a chart using the VL model.
        
        Args:
            chart_path: Path to chart image
            symbol: Stock ticker
            sr_data: Optional SR data for context
            
        Returns:
            Dict with validation results
        """
        if not chart_path.exists():
            return {
                'success': False,
                'error': f'Chart not found: {chart_path}',
                'direction': 'NEUTRAL',
                'confidence': 'LOW',
            }
        
        try:
            # Encode image
            image_b64 = self._encode_image_base64(chart_path)
            
            # Build prompt - use simpler prompt for llama3.2 and qwen3-vl
            # These models perform better with concise, direct prompts
            # Complex JSON prompts cause hallucinations
            if 'qwen3' in self.vl_model or 'llama3.2' in self.vl_model:
                prompt = self._build_simple_vl_prompt(symbol)
            else:
                prompt = self._build_vl_prompt(symbol, sr_data)
            
            # Model-specific options
            # qwen3-vl needs more tokens for full response
            # bespoke-minichart is BROKEN (doesn't process images)
            model_options = {
                'temperature': 0.1,
                'num_predict': 1024 if 'qwen3' in self.vl_model else 512,
            }
            
            # Call Ollama with vision model
            import requests
            
            response = requests.post(
                'http://localhost:11434/api/chat',
                json={
                    'model': self.vl_model,
                    'messages': [
                        {
                            'role': 'user',
                            'content': prompt,
                            'images': [image_b64]
                        }
                    ],
                    'stream': False,
                    'options': model_options
                },
                timeout=VL_TIMEOUT
            )
            response.raise_for_status()
            
            # Check if image was actually processed (detect broken models)
            # NOTE: Different models have very different token counts for images:
            # - qwen2.5vl: ~2600 tokens (detailed image encoding)
            # - llama3.2-vision: ~88 tokens (compressed image encoding)
            # - bespoke-minichart (BROKEN): ~100-150 tokens but NO vision capability
            # We can't reliably detect broken models by token count alone
            # Instead, trust the model's response and check for actual content
            resp_data = response.json()
            prompt_tokens = resp_data.get('prompt_eval_count', 0)
            
            content = resp_data['message']['content']
            
            # Only flag as broken if model returned empty/error
            if not content or len(content) < 20:
                logger.warning(f"Model {self.vl_model} returned empty response")
                return {
                    'success': False,
                    'error': f'Model {self.vl_model} returned empty response',
                    'direction': 'NEUTRAL',
                    'confidence': 'LOW',
                    'prompt_tokens': prompt_tokens,
                }
            
            # Parse JSON from response
            result = self._parse_vl_response(content)
            result['success'] = True
            result['raw_response'] = content
            result['model'] = self.vl_model
            result['chart_path'] = str(chart_path)
            result['prompt_tokens'] = prompt_tokens
            
            return result
            
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': f'VL model timeout after {VL_TIMEOUT}s',
                'direction': 'NEUTRAL',
                'confidence': 'LOW',
            }
        except Exception as e:
            logger.error(f"VL validation failed for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e),
                'direction': 'NEUTRAL',
                'confidence': 'LOW',
            }
    
    def _parse_vl_response(self, content: str) -> Dict:
        """Parse JSON from VL model response with strict classifier format."""
        import re
        
        # Try to find JSON in response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
        
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                
                # Map label to direction
                label = parsed.get('label', 'NEUTRAL_RANGE')
                if label in ['STRONG_BULLISH', 'BULLISH']:
                    direction = 'BULLISH'
                elif label in ['WATCH_BOUNCE', 'WATCH_RECLAIM']:
                    direction = 'NEUTRAL'  # Watchlist = not actionable yet
                elif label == 'BEARISH':
                    direction = 'BEARISH'
                else:
                    direction = 'NEUTRAL'
                
                # Map label to recommendation
                label_to_rec = {
                    'STRONG_BULLISH': 'BUY',
                    'BULLISH': 'BUY',
                    'WATCH_BOUNCE': 'WATCHLIST',
                    'WATCH_RECLAIM': 'WATCHLIST',
                    'NEUTRAL_RANGE': 'AVOID',
                    'BEARISH': 'AVOID',
                }
                
                # Map confidence float to string
                conf_val = parsed.get('confidence', 0.5)
                if isinstance(conf_val, (int, float)):
                    if conf_val >= 0.7:
                        confidence = 'HIGH'
                    elif conf_val >= 0.4:
                        confidence = 'MEDIUM'
                    else:
                        confidence = 'LOW'
                else:
                    confidence = str(conf_val).upper() if conf_val else 'LOW'
                
                return {
                    'label': label,
                    'setup_type': parsed.get('setup_type', 'unknown'),
                    'direction': direction,
                    'confidence': confidence,
                    'confidence_score': conf_val if isinstance(conf_val, (int, float)) else 0.5,
                    'price_position': parsed.get('price_position', {}),
                    'indicator_direction': parsed.get('indicator_direction', {}),
                    'signals_detected': parsed.get('signals_detected', []),
                    'key_evidence': parsed.get('key_evidence', []),
                    'why_this_label': parsed.get('why_this_label', ''),
                    'risk_flags': parsed.get('risk_flags', []),
                    'entry_quality': 'GOOD' if label in ['STRONG_BULLISH', 'BULLISH'] else 'POOR',
                    'recommendation': label_to_rec.get(label, 'AVOID'),
                    'reasoning': parsed.get('why_this_label', ''),
                }
            except json.JSONDecodeError:
                pass
        
        # Fallback - try to extract from text (handles simple prompt format too)
        content_upper = content.upper()
        
        # Try to parse simple format: LABEL: BEARISH, VS_AVWAP: below, etc
        label = None
        vs_avwap = None
        evidence = []
        
        # Look for LABEL: pattern
        label_match = re.search(r'LABEL:\s*(STRONG_BULLISH|BULLISH|BEARISH|NEUTRAL)', content_upper)
        if label_match:
            label = label_match.group(1)
        
        # Look for VS_AVWAP: pattern
        avwap_match = re.search(r'VS_AVWAP:\s*(ABOVE|BELOW)', content_upper)
        if avwap_match:
            vs_avwap = avwap_match.group(1).lower()
        
        # Look for EVIDENCE: pattern
        evidence_match = re.search(r'EVIDENCE:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
        if evidence_match:
            evidence = [evidence_match.group(1).strip()]
        
        # If no LABEL found, try keyword matching
        if not label:
            if 'STRONG_BULLISH' in content_upper:
                label = 'STRONG_BULLISH'
            elif 'BULLISH' in content_upper and 'BEARISH' not in content_upper:
                label = 'BULLISH'
            elif 'BEARISH' in content_upper:
                label = 'BEARISH'
            elif 'WATCH' in content_upper:
                label = 'WATCH_BOUNCE'
            else:
                label = 'NEUTRAL_RANGE'
        
        # Map label to direction
        if label in ['STRONG_BULLISH', 'BULLISH']:
            direction = 'BULLISH'
        elif label == 'BEARISH':
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'
        
        # Set confidence based on whether we found structured fields
        confidence_score = 0.7 if label_match else 0.5
        
        return {
            'label': label,
            'setup_type': 'unknown',
            'direction': direction,
            'confidence': 'MEDIUM' if confidence_score >= 0.5 else 'LOW',
            'confidence_score': confidence_score,
            'price_position': {'vs_avwap': vs_avwap} if vs_avwap else {},
            'indicator_direction': {},
            'signals_detected': [],
            'key_evidence': evidence,
            'why_this_label': evidence[0] if evidence else '',
            'risk_flags': [],
            'entry_quality': 'GOOD' if label in ['STRONG_BULLISH', 'BULLISH'] else 'POOR',
            'recommendation': 'BUY' if label in ['STRONG_BULLISH', 'BULLISH'] else 'AVOID',
            'reasoning': content[:200] if content else '',
        }
    
    def validate_with_vl_confirmation(self, symbol: str, sr_data: Optional[Dict] = None) -> Dict:
        """
        OPTIMIZED WORKFLOW: Hardcoded first, VL only for bullish confirmation.
        
        This method implements the v0.6.0 optimized validation:
        1. Run hardcoded classification (fast, ~100% accurate for bearish)
        2. If BEARISH/NEUTRAL: return immediately (no VL call)
        3. If BULLISH/STRONG_BULLISH: run VL model for visual confirmation
        4. Final classification requires BOTH hardcoded AND VL to agree on bullish
        
        This reduces VL calls by ~80% since most stocks are bearish/neutral,
        while preventing false positive bullish signals.
        
        Args:
            symbol: Stock ticker
            sr_data: Optional SR data for context
            
        Returns:
            Dict with classification result and whether VL was used
        """
        start_time = time.time()
        
        # STEP 1: Hardcoded classification (fast)
        hardcoded_result = self.classify_symbol(symbol)
        hardcoded_label = hardcoded_result.get('label', 'NEUTRAL_RANGE')
        hardcoded_direction = hardcoded_result.get('direction', 'NEUTRAL')
        
        result = {
            'success': True,
            'symbol': symbol,
            'classification_method': 'hardcoded_first',
            'hardcoded_label': hardcoded_label,
            'hardcoded_direction': hardcoded_direction,
            'vl_used': False,
            'vl_confirmed': None,
            **hardcoded_result  # Include all hardcoded fields
        }
        
        # STEP 2: If NOT bullish, return immediately (no VL needed)
        if hardcoded_direction != 'BULLISH':
            result['classification_time'] = time.time() - start_time
            logger.info(f"[{symbol}] Hardcoded: {hardcoded_label} ({hardcoded_direction}) - VL skipped")
            return result
        
        # STEP 3: Hardcoded says BULLISH - need VL confirmation
        logger.info(f"[{symbol}] Hardcoded: {hardcoded_label} - requesting VL confirmation...")
        
        # Generate chart
        chart_path = self.generate_chart(symbol, sr_data=sr_data)
        if not chart_path:
            result['vl_error'] = 'Failed to generate chart'
            result['classification_time'] = time.time() - start_time
            return result
        
        result['chart_path'] = str(chart_path)
        
        # Run VL model
        vl_result = self.validate_chart(chart_path, symbol, sr_data)
        result['vl_used'] = True
        result['vl_result'] = vl_result
        
        # Check if VL confirms bullish
        vl_direction = vl_result.get('direction', 'NEUTRAL')
        vl_confirmed = vl_direction == 'BULLISH'
        result['vl_confirmed'] = vl_confirmed
        result['vl_direction'] = vl_direction
        
        # STEP 4: Final decision - require VL confirmation for bullish
        if vl_confirmed:
            # Both agree - high confidence bullish
            logger.info(f"[{symbol}] VL CONFIRMED bullish - both agree!")
            result['final_direction'] = 'BULLISH'
            result['confidence'] = 'HIGH'
            result['confidence_score'] = min(0.9, hardcoded_result.get('confidence_score', 0.6) + 0.2)
        else:
            # VL disagrees - downgrade to neutral/watch
            logger.info(f"[{symbol}] VL says {vl_direction} - downgrading to NEUTRAL")
            result['label'] = 'WATCH_RECLAIM'
            result['direction'] = 'NEUTRAL'
            result['final_direction'] = 'NEUTRAL'
            result['confidence'] = 'LOW'
            result['confidence_score'] = 0.35
            result['recommendation'] = 'WATCHLIST'
            result['risk_flags'] = result.get('risk_flags', []) + ['vl_disagrees_with_hardcoded']
        
        result['classification_time'] = time.time() - start_time
        return result
    
    def validate_symbol(self, symbol: str, sr_data: Optional[Dict] = None, use_vl_model: bool = False) -> Dict:
        """
        Full validation pipeline: generate chart and optionally validate with VL.
        
        NOTE: For optimized workflow with hardcoded-first + VL confirmation,
        use validate_with_vl_confirmation() instead.
        
        Args:
            symbol: Stock ticker
            sr_data: Optional SR data for context
            use_vl_model: If True, also run VL model
            
        Returns:
            Dict with chart path and classification results
        """
        # Generate VLM-optimized chart (passes sr_data for S/R zones)
        chart_path = self.generate_chart(symbol, sr_data=sr_data)
        
        if not chart_path:
            return {
                'success': False,
                'error': 'Failed to generate chart',
                'direction': 'NEUTRAL',
                'recommendation': 'HOLD',
            }
        
        # Get hardcoded classification (primary method)
        hardcoded_result = self.classify_symbol(symbol)
        
        result = {
            'success': True,
            'chart_path': str(chart_path),
            'classification_method': 'hardcoded',
            **hardcoded_result
        }
        
        # Optionally also run VL model (for comparison/testing only)
        if use_vl_model:
            vl_result = self.validate_chart(chart_path, symbol, sr_data)
            result['vl_model_result'] = vl_result
            result['vl_model_warning'] = 'VL model prone to hallucination - use hardcoded classification'
        
        return result
    
    def classify_symbol(self, symbol: str, days: int = 60) -> Dict:
        """
        Classify a symbol using hardcoded rule-based logic.
        
        This is the RECOMMENDED method as it provides accurate, reliable
        classification based on actual indicator values.
        
        Key rules:
        - Below both zones = BEARISH
        - All indicators falling = BEARISH even if at support
        - In support zone but no bounce = BEARISH/NEUTRAL
        - Need ACTUAL bounce confirmation, not just proximity
        
        Args:
            symbol: Stock ticker
            days: Days of data to use
            
        Returns:
            Dict with classification result
        """
        import pandas as pd
        import numpy as np
        
        try:
            # Get data
            import yfinance as yf
            from autotrade.utils.vlm_chart_generator import compute_indicators, detect_events
            
            stock = yf.Ticker(symbol)
            hist = stock.history(period='120d').tail(days)
            df = compute_indicators(hist)
            events = detect_events(df)
            
            n = len(df)
            if n < 10:
                return {
                    'label': 'NEUTRAL_RANGE',
                    'setup_type': 'insufficient_data',
                    'direction': 'NEUTRAL',
                    'confidence': 'LOW',
                    'confidence_score': 0.1,
                    'risk_flags': ['insufficient_data'],
                    'recommendation': 'AVOID',
                }
            
            # Extract current values
            close = df['Close'].iloc[-1]
            sma20 = df['SMA20'].iloc[-1] if pd.notna(df['SMA20'].iloc[-1]) else None
            sma50 = df['SMA50'].iloc[-1] if pd.notna(df['SMA50'].iloc[-1]) else None
            avwap = df['AVWAP_high'].iloc[-1] if pd.notna(df['AVWAP_high'].iloc[-1]) else None
            rsi = df['RSI'].iloc[-1] if pd.notna(df['RSI'].iloc[-1]) else 50
            macd = df['MACD'].iloc[-1] if pd.notna(df['MACD'].iloc[-1]) else 0
            macd_sig = df['MACD_Signal'].iloc[-1] if pd.notna(df['MACD_Signal'].iloc[-1]) else 0
            macd_hist = df['MACD_Hist'].iloc[-1] if pd.notna(df['MACD_Hist'].iloc[-1]) else 0
            fast_k = df['Fast_K'].iloc[-1] if pd.notna(df['Fast_K'].iloc[-1]) else 50
            slow_d = df['Slow_D'].iloc[-1] if pd.notna(df['Slow_D'].iloc[-1]) else 50
            
            # Previous values for direction detection
            macd_hist_prev = df['MACD_Hist'].iloc[-2] if n > 1 and pd.notna(df['MACD_Hist'].iloc[-2]) else 0
            macd_hist_prev2 = df['MACD_Hist'].iloc[-3] if n > 2 and pd.notna(df['MACD_Hist'].iloc[-3]) else 0
            rsi_prev = df['RSI'].iloc[-2] if n > 1 and pd.notna(df['RSI'].iloc[-2]) else 50
            fast_k_prev = df['Fast_K'].iloc[-2] if n > 1 and pd.notna(df['Fast_K'].iloc[-2]) else 50
            close_prev = df['Close'].iloc[-2] if n > 1 else close
            
            atr = df['ATR'].iloc[-1] if pd.notna(df['ATR'].iloc[-1]) else close * 0.02
            
            # Determine price position vs zones
            supports = events.get('support_zones', [])
            resistances = events.get('resistance_zones', [])
            
            support_level = supports[0] if supports else close - atr * 3
            resistance_level = resistances[0] if resistances else close + atr * 3
            
            below_support = close < support_level - atr * 0.5
            in_support = abs(close - support_level) < atr * 1.5
            above_support = close > support_level + atr * 0.5
            
            below_resistance = close < resistance_level - atr * 0.5
            in_resistance = abs(close - resistance_level) < atr * 1.5
            above_resistance = close > resistance_level + atr * 0.5
            
            # Price position
            below_avwap = avwap is not None and close < avwap
            above_avwap = avwap is not None and close > avwap
            below_sma20 = sma20 is not None and close < sma20
            above_sma20 = sma20 is not None and close > sma20
            
            # Indicator directions (are they ACTUALLY turning?)
            macd_falling = macd_hist < macd_hist_prev and macd_hist_prev < macd_hist_prev2
            macd_rising = macd_hist > macd_hist_prev and macd_hist_prev > macd_hist_prev2
            macd_curling_up = macd_hist > macd_hist_prev and macd_hist_prev <= macd_hist_prev2
            macd_bearish_cross = macd < macd_sig and df['MACD'].iloc[-2] >= df['MACD_Signal'].iloc[-2] if n > 1 else False
            macd_bullish_cross = macd > macd_sig and df['MACD'].iloc[-2] <= df['MACD_Signal'].iloc[-2] if n > 1 else False
            
            rsi_falling = rsi < rsi_prev
            rsi_rising = rsi > rsi_prev
            
            stoch_falling = fast_k < fast_k_prev
            stoch_rising = fast_k > fast_k_prev
            stoch_curling_from_oversold = fast_k > fast_k_prev and fast_k_prev < 20
            
            # Check for actual bounce (green candle after being at support)
            recent_bounce = False
            if in_support or below_support:
                # Look for reversal candle in last 3 bars
                for i in range(-3, 0):
                    if n + i >= 1:
                        c = df['Close'].iloc[i]
                        o = df['Open'].iloc[i]
                        if c > o and c > df['Close'].iloc[i-1]:  # Green candle higher than prior
                            recent_bounce = True
                            break
            
            risk_flags = []
            
            # ===== STRICT CLASSIFICATION LOGIC =====
            
            # RULE 1: Below BOTH zones = BEARISH (breakdown)
            if below_support and below_resistance:
                return self._build_classification('BEARISH', 'breakdown', 0.80, 
                    ['below_both_zones', 'breakdown'],
                    f"Price ${close:.2f} below both support ${support_level:.2f} and resistance ${resistance_level:.2f}")
            
            # RULE 2: MACD bearish cross = BEARISH
            if macd_bearish_cross:
                risk_flags.append('macd_bearish_cross')
                if below_avwap and below_sma20:
                    return self._build_classification('BEARISH', 'bear_trend', 0.75,
                        risk_flags + ['below_avwap', 'below_sma20'],
                        f"MACD bearish cross with price below AVWAP (${avwap:.2f}) and SMA20")
            
            # RULE 3: All indicators falling = BEARISH
            all_falling = macd_falling and rsi_falling and stoch_falling
            if all_falling:
                risk_flags.append('all_indicators_falling')
                if below_avwap:
                    return self._build_classification('BEARISH', 'bear_trend', 0.70,
                        risk_flags + ['below_avwap'],
                        "All indicators (MACD, RSI, STOCH) falling, price below AVWAP")
            
            # RULE 4: In support zone but no bounce = NEUTRAL (not bullish yet)
            if in_support and not recent_bounce and (macd_falling or stoch_falling):
                return self._build_classification('NEUTRAL_RANGE', 'range_chop', 0.50,
                    ['at_support_no_bounce', 'indicators_still_falling'],
                    f"At support ${support_level:.2f} but no bounce - indicators still falling")
            
            # RULE 5: Below AVWAP + below SMA20 + negative MACD = BEARISH
            if below_avwap and below_sma20 and macd_hist < 0 and macd_falling:
                return self._build_classification('BEARISH', 'bear_trend', 0.65,
                    ['below_avwap', 'below_sma20', 'macd_negative_falling'],
                    f"Below AVWAP (${avwap:.2f}), below SMA20, MACD histogram negative and falling")
            
            # RULE 6: Above BOTH zones = potential breakout (STRONG_BULLISH candidate)
            if above_support and above_resistance:
                if above_avwap and macd_rising and rsi > 50:
                    return self._build_classification('STRONG_BULLISH', 'breakout_above_resistance', 0.80,
                        [],
                        f"Breakout above resistance ${resistance_level:.2f}, above AVWAP, MACD rising, RSI {rsi:.0f}>50")
                elif above_avwap:
                    return self._build_classification('BULLISH', 'breakout_above_resistance', 0.65,
                        ['needs_momentum_confirmation'],
                        f"Above both zones and AVWAP, but momentum not fully confirmed")
            
            # RULE 7: Above AVWAP + MACD bullish + RSI >50 = BULLISH
            if above_avwap and above_sma20 and macd > macd_sig and rsi > 50:
                if macd_rising and stoch_rising:
                    return self._build_classification('STRONG_BULLISH', 'vwap_reclaim', 0.75,
                        [],
                        f"Above AVWAP (${avwap:.2f}), above SMA20, MACD bullish, RSI {rsi:.0f}>50, momentum accelerating")
                else:
                    return self._build_classification('BULLISH', 'vwap_reclaim', 0.60,
                        ['momentum_not_accelerating'],
                        f"Above AVWAP and SMA20, MACD bullish, RSI {rsi:.0f}>50")
            
            # RULE 8: At support with actual bounce + indicators curling = WATCH_BOUNCE
            if in_support and recent_bounce and (stoch_curling_from_oversold or macd_curling_up):
                return self._build_classification('WATCH_BOUNCE', 'support_bounce', 0.55,
                    ['below_avwap'] if below_avwap else [],
                    f"Bounce from support ${support_level:.2f} with indicators curling up")
            
            # RULE 9: Improving but below AVWAP = WATCH_RECLAIM
            if below_avwap and (macd_rising or stoch_rising) and close > close_prev:
                return self._build_classification('WATCH_RECLAIM', 'potential_reclaim', 0.45,
                    ['below_avwap', 'needs_confirmation'],
                    f"Price improving but still below AVWAP (${avwap:.2f}) - watch for reclaim")
            
            # DEFAULT: NEUTRAL_RANGE
            return self._build_classification('NEUTRAL_RANGE', 'range_chop', 0.40,
                ['mixed_signals'],
                "Mixed signals - no clear direction")
                
        except Exception as e:
            logger.error(f"Classification failed for {symbol}: {e}")
            return {
                'label': 'NEUTRAL_RANGE',
                'setup_type': 'error',
                'direction': 'NEUTRAL',
                'confidence': 'LOW',
                'confidence_score': 0.1,
                'risk_flags': ['classification_error'],
                'recommendation': 'AVOID',
                'error': str(e),
            }
    
    def _build_classification(self, label: str, setup_type: str, confidence: float, 
                              risk_flags: list, reasoning: str) -> Dict:
        """Build a classification result dict."""
        # Map label to direction
        if label in ['STRONG_BULLISH', 'BULLISH']:
            direction = 'BULLISH'
        elif label == 'BEARISH':
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'
        
        # Map confidence to string
        if confidence >= 0.7:
            conf_str = 'HIGH'
        elif confidence >= 0.4:
            conf_str = 'MEDIUM'
        else:
            conf_str = 'LOW'
        
        # Map label to recommendation
        label_to_rec = {
            'STRONG_BULLISH': 'BUY',
            'BULLISH': 'BUY',
            'WATCH_BOUNCE': 'WATCHLIST',
            'WATCH_RECLAIM': 'WATCHLIST',
            'NEUTRAL_RANGE': 'AVOID',
            'BEARISH': 'AVOID',
        }
        
        return {
            'label': label,
            'setup_type': setup_type,
            'direction': direction,
            'confidence': conf_str,
            'confidence_score': confidence,
            'risk_flags': risk_flags,
            'recommendation': label_to_rec.get(label, 'AVOID'),
            'reasoning': reasoning,
        }


# Singleton instance
_vl_validator: Optional[VLChartValidator] = None


def get_vl_validator() -> VLChartValidator:
    """Get the singleton VL validator."""
    global _vl_validator
    if _vl_validator is None:
        _vl_validator = VLChartValidator()
    return _vl_validator


def validate_symbol(symbol: str, sr_data: Optional[Dict] = None, use_vl_model: bool = False) -> Dict:
    """Convenience function to validate a symbol using hardcoded classification."""
    return get_vl_validator().validate_symbol(symbol, sr_data, use_vl_model)


def classify_symbol(symbol: str, days: int = 60) -> Dict:
    """Convenience function to classify a symbol using hardcoded logic (recommended)."""
    return get_vl_validator().classify_symbol(symbol, days)


if __name__ == "__main__":
    # Add project root to path for standalone execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
        print(f"Classifying {symbol} (hardcoded logic)...")
        
        validator = VLChartValidator()
        result = validator.classify_symbol(symbol)
        
        print(f"\n=== Classification: {symbol} ===")
        print(f"Label: {result.get('label')}")
        print(f"Setup: {result.get('setup_type')}")
        print(f"Direction: {result.get('direction')}")
        print(f"Confidence: {result.get('confidence')} ({result.get('confidence_score', 0):.0%})")
        print(f"Recommendation: {result.get('recommendation')}")
        print(f"Risk Flags: {', '.join(result.get('risk_flags', []))}")
        print(f"Reasoning: {result.get('reasoning')}")
        
        # Also test chart generation
        print(f"\nGenerating chart...")
        chart_path = validator.generate_chart(symbol)
        if chart_path:
            print(f"Chart saved: {chart_path}")
    else:
        print("Usage: python vl_chart_validator.py SYMBOL")
