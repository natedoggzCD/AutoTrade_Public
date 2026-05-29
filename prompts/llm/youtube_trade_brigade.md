# Trade Brigade (Matt) - YouTube Transcript Analyzer

## Producer Profile

**Channel**: Trade Brigade  
**Host**: Matt  
**Show**: Midweek Market Update (Wed), Saturday Deep Dive, Daily Pre-Market Prep (8am ET), Live close streams (3:50pm ET)  
**Style**: Institutional-grade technical analysis, market profile, orderflow, extremely methodical  
**Typical Length**: 30-60 minutes (very thorough)  
**Upload Schedule**: Multiple daily touchpoints - pre-market, midweek, Saturday

## Why This Channel Matters

Matt is the most technically rigorous of all four channels:
- **Market profile analysis** - value area, point of control, volume flows. Professional-level tools
- **S&P vs NASDAQ divergence** - consistently separates the two. When SPY holds but QQQ breaks, he catches that rotation signal immediately
- **IWM / Small-cap coverage** - HE ACTUALLY COVERS SMALL CAPS. This is unique among our channels and directly relevant to our 1600-stock universe
- **Market internals** - advance/decline, cumulative volume, tick, index score. Real breadth data
- **Level-by-level precision** - doesn't generalize. SPY 686, QQQ 613, QQQ 607.25, QQQ 595.25. Exact entries/exits
- **Both bull AND bear cases** - always presents both scenarios with trigger levels. No bias
- **Trade ideas at the end** - 8+ specific setups with entry/stop/target logic
- **Live sessions** - pre-market prep and close streams provide real-time context

## How We Use This (CRITICAL)

**Matt gives us what no other channel does: small-cap analysis AND institutional-grade levels.**

His value to our trading system:
1. **IWM/Russell analysis** - DIRECT read on our small-cap universe health
2. **SPY vs QQQ divergence** - when S&P holds but NASDAQ breaks, small-caps often follow S&P (rotation into value). This is a BUY signal for us
3. **Exact trigger levels** - "under 686 = bearish", "over 613 = frustrated shorts". Our agent can use these as regime switches
4. **Market profile** - value area overlap/down tells you if sellers are making progress or just noise
5. **Bear flag / lower high setups** - his counter-trend timing tells our agent when to expect dead cat bounces vs real reversals
6. **Trade ideas** - he covers small/mid-cap setups (Wingstop, Etsy, AFRM, CCL). Some of these ARE in our universe
7. **"Rotation not crash" thesis** - when RSP (equal weight) looks great while QQQ bleeds, our average stock is FINE. Critical signal

## Content Structure Pattern

Very consistent and methodical:
1. **Opening summary** - names hit hardest, theme of the day
2. **SPY daily chart** - trend, 50/200 SMA, key levels, close position
3. **SPY hourly chart** - intraday structure, lower highs, range analysis
4. **SPY market internals** - advance/decline, cumulative volume, index score
5. **SPY market profile** - value area, point of control, overlap analysis
6. **QQQ daily chart** - same treatment, almost always more bearish
7. **QQQ hourly chart** - counter-trend setups, bear flag potential
8. **QQQ market internals** - volume flows, NASDAQ advance/decline
9. **QQQ market profile** - value overlap direction
10. **IWM / Russell** - small-cap health check (CRITICAL for us)
11. **Sector charts** - XLF, SMH, IGV, individual sectors
12. **VIX / volatility** - structure, term, contango
13. **Individual trade ideas** - 6-10 specific setups with levels
14. **Earnings preview** - what reports tomorrow, expected impact
15. **Close** - live stream schedule, next video topics

## Key Extraction Points

### MUST EXTRACT (Priority Order):
- **IWM / Russell 2000 status** - MOST IMPORTANT. Is small-cap "okay" or breaking? His exact words
- **SPY vs QQQ divergence** - rotation signal. SPY holding while QQQ breaks = rotation into value = our stocks survive
- **RSP (equal weight) vs QQQ** - "RSP looks great" while QQQ bleeds = breadth is healthy = our universe is fine
- **Exact trigger levels** - SPY 686, QQQ 613, QQQ 607, QQQ 595.25. These are regime switches
- **Market profile verdict** - value overlapping down (bearish) vs contained (neutral) vs overlapping up (bullish)
- **Bull case vs bear case** - he always gives BOTH with trigger levels. Extract both scenarios
- **Trade ideas** - especially any in $2-$200 range that overlap our universe
- **Counter-trend timing** - when he expects bounces and what would confirm them
- **Earnings impact** - what reports tomorrow and expected market effect

### SECONDARY:
- Individual mega-cap chart analysis (MSFT, AMZN, GOOGL) - only for index impact
- Fibonacci levels - useful but secondary to his key levels
- Anchored VWAP stacks - institutional level data, extract when spoken
- Intraday 0DTE gamma - more relevant for day trading than our swing system

### FILTER OUT:
- Visual chart references impossible to capture ("look at this", "right here")
- Channel promotion, live stream plugs
- Market profile tutorial references ("if you're not familiar, video in top right")
- Whisper transcription artifacts/repetitions

## Extraction Prompt

```
You are analyzing a Trade Brigade (Matt) YouTube transcript to extract precise 
market structure and small-cap health data for AutoTrade's trading system.

CONTEXT:
- AutoTrade trades ~1600 small/mid-cap stocks ($2-$200 price range)
- Matt is the ONLY channel that covers IWM/Russell small-caps directly
- He separates SPY from QQQ analysis - divergences between them signal rotation
- RSP (equal weight S&P) vs QQQ performance tells us if the average stock is okay
- His levels are institutional-grade precise - preserve exact numbers
- He presents both bull AND bear cases with trigger levels for each
- He provides 6-10 specific trade ideas per video, some overlapping our universe

EXTRACT THE FOLLOWING:

## 1. Small-Cap Health Check (IWM/Russell)
- Matt's assessment: healthy / neutral / breaking
- Key level and whether it held
- Comparison to SPY and QQQ performance
- His exact words on small-cap outlook
- AGENT IMPLICATION: If small-caps "look okay" → our universe is fine. If breaking → defensive.

## 2. SPY vs QQQ Divergence Signal
- Are they diverging? (SPY holds, QQQ breaks = rotation)
- Rotation thesis: value vs growth, financials bid, etc.
- What this means for breadth and our stocks
- RSP (equal weight) performance vs QQQ

## 3. Regime Trigger Levels
Format: INSTRUMENT | LEVEL | IF ABOVE | IF BELOW
These are binary switches our agent can use:
- Example: QQQ | 613 | Shorts frustrated, back to range | Lower high confirmed, target 595.25

Extract ALL spoken levels. Be precise to decimal points.

## 4. Market Profile Summary
For SPY and QQQ separately:
- Value area: overlapping down / contained / overlapping up
- Point of control position
- Volume flow direction
- His verdict: "progress for bears" vs "not enough data for bears"

## 5. Bull Case Scenario
- Trigger: specific level that must hold or break over
- Expected move and target
- Probability/conviction (his tone)
- Timeframe

## 6. Bear Case Scenario
- Trigger: specific level that must break
- Expected move and target  
- What confirmation looks like (lower high, internals, etc.)
- Timeframe

## 7. Trade Ideas (CRITICAL)
For EACH trade idea mentioned:
- Ticker
- Direction: long / short
- Setup type: breakout / breakdown / reversion / hammer / etc.
- Entry level
- Target level
- Stop/risk level (if mentioned)
- Price range (is it in our $2-$200 universe?)
- Sector

## 8. Sector Rotation Map
- Which sectors are green on a red day (rotation targets)
- Which are absolute carnage
- Software/IGV status and V-shape recovery potential
- Semiconductors vs broader market

## 9. Volatility & Market Internals
- VIX status and structure
- Advance/decline line direction
- Cumulative volume flows
- "Index score" or breadth proxy

## 10. Tomorrow's Setup
- Earnings reporting (names and expected impact)
- Pre-market prep time and focus
- Live stream schedule (for real-time updates)
- Key scenarios for next session

---
TRANSCRIPT:
{transcript}
---

OUTPUT FORMAT:
- Start with SMALL-CAP VERDICT: one line on IWM health + our universe implication
- SPY/QQQ DIVERGENCE: one line on rotation status
- REGIME: BULL above [level] / BEAR below [level] for each index
- Then structured sections above
- End with AGENT ACTIONS: 3-5 specific instructions for our trading agent
```

## Example Output Format

```json
{
  "date": "2026-02-05",
  "video_title": "Absolute Carnage",
  "video_type": "midweek_update",

  "smallcap_verdict": "IWM neutral - swept 4-day balance low but closed back inside range. Small-caps NOT participating in the carnage. Rotation keeping S&P and smalls supported while NASDAQ gets blown out.",
  
  "divergence_signal": "ACTIVE ROTATION: SPY holding 50 SMA on close, QQQ definitively broken under 613. RSP equal weight 'looks great'. Value > Growth. Small-caps following S&P, not NASDAQ.",

  "agent_actions": [
    "DO NOT go defensive on small-caps - IWM is holding, rotation is supporting breadth",
    "AVOID: high-beta tech/software names - NASDAQ in confirmed downtrend under 613",
    "WATCH: If SPY loses 686 on close → THEN go defensive on everything",
    "OPPORTUNITY: Software names with hammer reversals (CRM, ETSY, INTU) for counter-trend bounce if confirmed tomorrow",
    "SIZE: Normal position sizes for small-caps. Reduced for anything tech-adjacent."
  ],

  "regime_levels": [
    {"instrument": "SPY", "level": 686, "above": "Range-bound, neutral-to-bullish bias, rotation still healthy", "below": "Confirmed breakdown, target 680 then 675"},
    {"instrument": "SPY", "level": 683.75, "above": "Counter-trend possible to 20 SMA", "below": "Sellers prove it, bigger move down"},
    {"instrument": "QQQ", "level": 613, "above": "Shorts frustrated, back to balance range", "below": "Downtrend confirmed, target 595.25"},
    {"instrument": "QQQ", "level": 607.25, "above": "Still in overshoot zone", "below": "Lower low confirmed"},
    {"instrument": "QQQ", "level": 595.25, "above": "N/A", "below": "Range double complete, major support test"},
    {"instrument": "NQ", "level": 25550, "above": "Volume turns over, bear case weakens", "below": "Lower highs persist, bears in control"}
  ],

  "market_profile": {
    "spy": {
      "value_area": "contained_inside_tuesday",
      "poc": "inside range",
      "verdict": "Not enough data to be overly aggressive on bear side"
    },
    "qqq": {
      "value_area": "overlapping_to_down",
      "poc": "lower",
      "verdict": "Progress for the bears. Short setup valid under 613."
    }
  },

  "bull_case": {
    "trigger": "SPY holds 686, QQQ snaps over 613",
    "target": "SPY counter-trend to 20 SMA. QQQ back into frustrating balance range.",
    "conviction": "medium_low",
    "note": "Not a new bull thesis. Just 'keeping your sanity' in range."
  },

  "bear_case": {
    "trigger": "QQQ lower high under 613 confirmed, SPY closes under 686",
    "target": "QQQ 595.25 (range double). SPY 680 → 675 (11% pullback from highs = garden variety).",
    "conviction": "medium_high_on_qqq",
    "note": "Bear case proven in NASDAQ, NOT yet in S&P. Dual-track analysis required."
  },

  "trade_ideas": [
    {"ticker": "CRM", "direction": "long", "setup": "hammer closing back inside 195 range after blowout", "target": "thin structure gap above", "in_our_universe": false, "sector": "Software"},
    {"ticker": "ETSY", "direction": "long", "setup": "hammer, H&S neckline test failed. Over high = thin structure retrace", "target": "MA cluster above", "in_our_universe": true, "sector": "Software/E-commerce"},
    {"ticker": "INTU", "direction": "long", "setup": "failure to go lower, closing inside prior range, thin structure gap above", "target": "gap fill", "in_our_universe": false, "sector": "Software"},
    {"ticker": "AFRM", "direction": "long", "setup": "volume surge, failed breakdown", "target": "thin structure retrace", "in_our_universe": true, "sector": "Fintech"},
    {"ticker": "WING", "direction": "long", "setup": "rounded base, ascending triangle. Over daily 200 SMA", "target": "305-315 origin of breakdown", "in_our_universe": false, "sector": "Consumer"},
    {"ticker": "CCL", "direction": "long", "setup": "over 32.70 breakout", "target": "new record highs", "in_our_universe": true, "sector": "Travel/Leisure"},
    {"ticker": "TSN", "direction": "long", "setup": "flat top breakout at 66", "target": "breakout continuation", "in_our_universe": true, "sector": "Consumer Staples"},
    {"ticker": "TSM", "direction": "long", "setup": "pullback hold at 311.20, mirrors SMH", "target": "rebound with semis", "in_our_universe": false, "sector": "Semiconductors"}
  ],

  "sector_status": {
    "software_igv": {"status": "carnage", "recovery_signal": "hammers forming on CRM/ETSY/INTU, watch for confirmation"},
    "semiconductors": {"status": "mixed", "note": "popping on CapEx spend news, SMH holding trend support"},
    "financials": {"status": "green_rotation", "note": "+84bps on red day, 2nd heaviest S&P weight"},
    "high_beta": {"status": "absolute_carnage", "names": "RKLB, OKLO, PLTR, BLMN, KTOS all -10%+"},
    "consumer_staples": {"status": "holding", "note": "TSN breakout, WING setup"}
  },

  "volatility": {
    "vix_note": "Elevated but not panicked",
    "internals_spy": "Breadth proxy healthy - not confirming bear case",
    "internals_qqq": "Bearish - volume flows -400M, A/D line trending lower"
  },

  "cross_references": [
    {"video": "Tuesday", "context": "Big washout day. Shorts from Tuesday still comfortable with lower highs under 613."},
    {"video": "Saturday", "context": "Mentioned direct analog from 2025 consolidation. Weekly bar analysis."},
    {"video": "Tomorrow 8am", "context": "Pre-market prep. Amazon/Reddit/Bloom earnings focus."},
    {"video": "Tomorrow 3:50pm", "context": "Live close stream - Amazon earnings reaction."}
  ]
}
```

## RAG Integration Notes

### Trade Ideas Tracking (UNIQUE VALUE)
Matt gives 6-10 specific trade setups per video. Store and track:
```json
{
  "date": "2026-02-05",
  "ideas": [
    {"ticker": "CCL", "direction": "long", "trigger": "over 32.70", "target": "new highs", "status": "pending"},
    {"ticker": "ETSY", "direction": "long", "trigger": "over prior high", "target": "MA cluster", "status": "pending"}
  ]
}
```
Cross-reference with our signal pipeline - if our signals agree with Matt's setup, conviction increases.

### Agent Decision Integration
Matt's data provides the most granular overlay:

1. **IWM health → our universe health**: Direct correlation. If IWM neutral/healthy, our agent stays normal. If IWM breaks, go defensive
2. **SPY/QQQ divergence → rotation signal**: When they diverge, our value-oriented small-caps benefit. Agent can increase exposure
3. **Regime levels as binary switches**: QQQ above/below 613 changes the entire playbook. Agent checks these at open
4. **Market profile for confirmation**: Value overlapping down on QQQ = valid short thesis = reduce tech-adjacent small-caps
5. **Trade ideas overlap**: Filter his ideas for tickers in our $2-$200 universe → direct watchlist adds
6. **Counter-trend timing**: His "lower high under 613" setup timing helps our agent know when dead cat bounces end

### Complementary Channel Matrix
| Data Point | Trade Brigade | RTA Trading | Mike | Click Capital |
|------------|--------------|-------------|------|---------------|
| Small-cap analysis | **PRIMARY** | - | Mentioned | - |
| Index levels (precise) | **PRIMARY** | PRIMARY | Good | Basic |
| Market profile | **PRIMARY** | - | - | - |
| SPY/QQQ divergence | **PRIMARY** | Mentioned | Mentioned | - |
| Market breadth | Good | - | **PRIMARY** | - |
| Gamma/dealer flow | Mentioned | - | **PRIMARY** | - |
| Trade ideas (specific) | **PRIMARY** | Some | Some | Some |
| VIX deep analysis | Good | **PRIMARY** | Basic | Good |
| Bond/crash signal | - | - | **PRIMARY** | Mentioned |
| Macro/sentiment | Basic | Good | Good | **PRIMARY** |
| Earnings analysis | Good | **PRIMARY** | Good | Good |

## Matt's Technical Vocabulary

| Term | Meaning |
|------|---------|
| "Value overlapping to down" | Market profile: sellers making progress, bearish |
| "Contained inside range" | Market profile: no directional progress, neutral |
| "Point of control" | Price level with most volume in the session |
| "Thin structure retrace" | Low volume area above → fast move through it on bounce |
| "Look above and fail" | Tried to break above key level, rejected, now lower |
| "Range double" | Target = 2x the range measured from the breakdown point |
| "Brigade bolt" | His term for a strong breakout move |
| "Sweep the low" | Quick undercut of support that reclaims immediately → bullish |
| "Overshoot" | Wick beyond a level without closing through it |
| "Garden variety" | Normal pullback magnitude (~10-11% on S&P) |
| "Confirming internals" | Advance/decline, volume, ticks all agreeing with price |
| "Lower high under key level" | Bear flag setup - the short trigger |
| "Counter-trend" | A bounce within a larger downtrend, not a reversal |
| "Anchored VWAP" | Volume-weighted average price from a specific date |
| "Hammer" | Candlestick with long lower wick, small body → potential reversal |
| "Inside bar" | Trading range contained within prior bar → coiling for move |

## Long Video Addendum

This is a longer Trade Brigade video (likely a Saturday Deep Dive or extended midweek update). Extract MORE detail than usual:

- **Weekly charts**: Extract all weekly timeframe levels, trends, and chart patterns discussed (not just daily/hourly)
- **All trade ideas**: Matt typically gives 8-12+ trade setups in long videos. Capture EVERY one with full entry/stop/target details
- **Weekly value area**: Market profile weekly value area analysis — is the weekly value overlapping up, down, or contained?
- **Monday prep / forward-looking**: Any specific guidance for the upcoming week — what scenarios he expects, which levels to watch on Monday open
- **Multi-week structure**: Broader market structure analysis (head and shoulders, ranges, measured moves) that spans weeks
- **Sector deep dives**: He often goes deeper into 3-4 sectors in Saturday videos — capture each with key levels and verdict
- **Anchored VWAP stacks**: Institutional reference levels from earnings, pivots, or major events — extract all mentioned
- **Risk scenarios**: Both bull and bear cases with WEEKLY timeframe triggers (not just daily)
