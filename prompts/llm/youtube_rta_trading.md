# RTA Trading (after-hours reviews) - YouTube Transcript Analyzer

## Producer Profile

**Channel**: RTA Trading  
**Host**: Unknown first name (27+ years trading experience)  
**Show**: Daily market reviews (often post-close / after-hours)  
**Style**: Professional trader, technical-first, no fluff, direct, conversational  
**Typical Length**: 15-30 minutes  
**Upload Schedule**: Daily (often late evening post-earnings calls)  
**Saturday**: Deep-dive episodes (longer, more analysis)

## Why This Channel Matters

This is the **#1 priority** YouTube source for AutoTrade. The host is:
- A 27-year veteran trader with real skin in the game (shares live positions)
- Technically rigorous - references specific price levels, moving averages, volume
- No clickbait, no fluff - pure information density
- Builds on previous videos (references Tuesday's levels, prior calls)
- Provides forward-looking levels and targets, not just commentary
- **High conviction track record** - his 6-month forward recommendations tend to be solid
- Trades mega-caps and indices - but his macro reads apply to the entire market

## How We Use This (CRITICAL)

**He trades mega-caps. We trade small/mid-caps ($2-$200). But markets move together.**

When SPX/QQQ break key levels, our entire universe gets hit. When he says
"tomorrow's gonna be a shaky morning", our small-caps gap down too. His value
to us is:
1. **Market open expectations** - gap up/down, volatility expected
2. **Risk-on vs risk-off** - should our agent be aggressive or defensive today?
3. **Sector rotation** - what sectors are getting crushed/bid, spillover to small-caps
4. **VIX / volatility regime** - does our agent tighten stops or widen them?
5. **Multi-week direction** - his forward calls inform position sizing, not individual picks
6. **"Let it burn" signals** - when he says stay out, our agent should reduce exposure

DO NOT try to trade his specific mega-cap picks. Extract the MARKET DIRECTION.

## Critical Limitation

**~60% of content is visual** - he draws on charts, points at levels, references
things on screen that the transcript cannot capture. Price levels are sometimes
spoken ("588", "48K") but sometimes implied by pointing. This significantly
reduces the transcript's value vs watching the video. However:
- Index levels (ES, QQQ) he usually says out loud - these are the most valuable
- Sector direction (bearish/bullish) is always stated verbally
- His overall thesis and conviction come through in words alone
- Cross-reference spoken levels with actual market data to fill visual gaps

## Content Structure Pattern

NO fixed structure - stream of consciousness technical analysis:
1. **Index overview** - ES/SPX, QQQ/NQ levels, where we broke/held
2. **Key level callouts** - Specific numbers with significance
3. **Tangents** - Deep dives into whatever matters NOW (these are gold)
4. **Individual names** - Stocks he's actively trading with positions
5. **Sector themes** - What's rotating, what's breaking
6. **Tinfoil hat** - Macro/political/structural theories (marked clearly)
7. **Tomorrow setup** - Where he thinks we're heading, levels to watch

## Key Extraction Points

### MUST EXTRACT:
- **Market direction for tomorrow** - gap up/down expectation, "shaky morning", etc.
- **Index levels** - ES, QQQ, SPX numbers - these drive our entire universe
- **VIX regime** - current level, what handle he wants to see, term structure
- **Sector rotation** - which sectors breaking, which holding - spillover to small-caps
- **Risk posture** - is he adding risk or reducing? "Let it burn" = our agent goes defensive
- **Directional calls with timeframe** - "heading to 48K", "going to 588" + when
- **Earnings reactions** - mega-cap earnings move indices which move our stocks
- **Moving average status** - 200 weekly breaks = secular concern for whole market
- **Cross-references to prior videos** - level continuations, evolving narratives
- **6-month forward outlook** - his longer-term calls tend to be high quality

### SECONDARY (nice to have):
- His specific positions (short AMZN, puts on AGQ) - useful as sentiment indicator
- Individual mega-cap levels - only matters for index impact
- Volume observations - exhaustive selling, capitulation patterns

### FILTER OUT:
- Drawing/visual references meaningless without screen ("look at that", "right here")
- Channel promotion ("subscribe", "Saturday deep dive")
- Self-deprecating humor (entertaining but not actionable)
- Whisper transcription stutters/repetitions
- Detailed mega-cap trade entries/exits (we don't trade them)

### CRITICAL CONTEXT - VIDEO CHAINING:
This creator builds analysis across multiple videos. Extract:
- References to prior videos: "if you watched Tuesday's video"
- Level continuations: "the levels we gave on Tuesday, they hit"
- Theme evolution: ongoing narratives that span days/weeks

### FILTER OUT:
- Drawing/visual references that are meaningless without screen ("look at that", "right here")
- Channel promotion ("subscribe", "Saturday deep dive")
- Self-deprecating humor (entertaining but not actionable)
- Whisper transcription stutters/repetitions

## Extraction Prompt

```
You are analyzing an RTA Trading YouTube transcript to extract MARKET DIRECTION 
intelligence for AutoTrade's small/mid-cap trading system.

CONTEXT:
- AutoTrade trades ~1600 small/mid-cap stocks ($2-$200 price range)
- We DO NOT trade mega-caps - but when SPX/QQQ drop, our stocks drop too
- This host trades mega-caps and indices - we use his reads as a MARKET OVERLAY
- His value: market open expectations, risk regime, sector rotation, VIX analysis
- ~60% of his analysis is VISUAL - extract what you CAN from spoken words
- He builds on prior videos - flag cross-references for RAG chaining
- His 6-month forward calls are historically reliable

EXTRACT THE FOLLOWING:

## 1. Tomorrow's Market Open Expectation
- Gap up or gap down?
- His exact words on what to expect (quote when possible)
- Confidence: high / medium / low
- Any specific triggers (earnings after-hours, futures, etc.)

## 2. Risk Regime (for our agent)
- RISK-OFF: "let it burn", "stay out", selling not done → agent reduces exposure
- RISK-ON: "picking up pieces", "buy zone", capitulation complete → agent adds
- NEUTRAL: "watch levels", "need to see X" → agent holds current exposure
- What VIX level/behavior he wants before changing posture
- Any "if X then Y" scenarios that flip the regime

## 3. Index Levels That Matter
Format: INDEX | LEVEL | SIGNIFICANCE
These drive our entire universe. Extract ALL spoken numbers.
Note: "[VISUAL - level not captured]" for levels he points at but doesn't say.

## 4. Sector Direction Map
For each sector discussed:
- Sector name
- Direction: STRONG / WEAK / BREAKING / HOLDING / ROTATING INTO
- His reasoning (1 sentence)
- Spillover risk to small/mid-caps (high/medium/low)

This is critical - sector rotation in mega-caps leads small-caps by 1-3 days.

## 5. Volatility Assessment
- Current VIX level/handle
- Term structure (contango/backwardation)
- What he needs to see before positioning changes
- Impact: should our agent tighten or widen stops?

## 6. Forward Outlook (multi-week / 6-month)
- His longer-term thesis
- Any "I think this is heading..." calls with timeframes
- Sector or asset class calls beyond tomorrow
- These inform our agent's position SIZING, not individual picks

## 7. Earnings Impact on Market
- Which mega-cap reported and reaction
- Does the reaction drag indices? (moves our stocks)
- Management commentary that changes sector narrative
- CapEx trajectory (affects tech/growth sentiment broadly)

## 8. Cross-Video Continuity
- Prior video references with context
- Levels from earlier videos that hit/failed
- Evolving narratives spanning multiple videos
- Saturday deep dive topics previewed

## 9. Trading Wisdom / Lessons
- Risk management principles stated
- Mistakes he shares openly
- Market psychology observations
- These get stored in RAG for long-term agent learning

---
TRANSCRIPT:
{transcript}
---

OUTPUT FORMAT:
- Start with a 2-sentence MARKET DIRECTION SUMMARY our agent can act on immediately 
- Then structured sections above
- For visual references, note "[VISUAL]" - don't guess at numbers
- Preserve his exact price levels - do NOT round them
- End with AGENT ACTION: 1-3 sentences on what our trading agent should do tomorrow
```

## Example Output Format

```json
{
  "date": "2026-02-05",
  "video_title": "Everything is Crashing - The Real Reason Why",
  
  "direction_summary": "Bearish near-term. Expect shaky/gap-down Friday morning. Selling not done - needs VIX 30+ and reset event before adding risk. Great companies getting thrown out with bad - opportunities forming on 6-month horizon.",
  
  "agent_action": "DEFENSIVE. Reduce exposure at open. Do not buy dips until VIX prints 30+ handle. Tighten stops on existing positions. Software and crypto-adjacent small-caps especially vulnerable.",

  "market_open": {
    "expectation": "gap_down",
    "detail": "Amazon -11% after hours on $200B CapEx. QQQ futures sliding. 'Wouldn't surprise me to see a 5-handle' on ES.",
    "confidence": "high",
    "triggers": ["AMZN earnings miss + CapEx shock", "BTC falling through 60K", "Software sector in freefall"]
  },

  "risk_regime": {
    "current": "RISK-OFF",
    "signal_phrase": "Let it all burn and stay out of the way",
    "flip_condition": "VIX needs 30+ handle before looking to add long risk",
    "agent_instruction": "Reduce position sizes. No new longs in growth/tech sectors. Tighten trailing stops."
  },
  
  "index_levels": [
    {"index": "ES", "level": 5887, "significance": "Support from Tuesday - BROKE today"},
    {"index": "ES", "level": 5880, "significance": "Next support if 5887 fails"},
    {"index": "QQQ", "level": 588, "significance": "Key support from Tuesday - hit today"},
    {"index": "QQQ", "level": 580, "significance": "Next level down - 'if I don't hold 588, going to 580'"},
    {"index": "QQQ", "level": null, "significance": "[VISUAL] 'If you break that, you got to go' - number not spoken"}
  ],

  "sector_direction": [
    {"sector": "Software/SaaS", "direction": "BREAKING", "detail": "IGV fastest drop in history, 200wk MA test, Anthropic fear", "smallcap_spillover": "high"},
    {"sector": "Private Credit", "direction": "WEAK", "detail": "8-12% software book exposure dragging OWL/KKR/APO", "smallcap_spillover": "medium"},
    {"sector": "Crypto/Blockchain", "direction": "BREAKING", "detail": "Thesis credibility crisis, needs FTX-level reset", "smallcap_spillover": "high"},
    {"sector": "Silver/Commodities", "direction": "BREAKING", "detail": "Key level broke, heading to 55", "smallcap_spillover": "medium"},
    {"sector": "Semiconductors", "direction": "MIXED", "detail": "CapEx beneficiaries vs valuation concern. Memory (MU, SNDK) supported by demand", "smallcap_spillover": "medium"}
  ],

  "volatility": {
    "vix_handle": "20s",
    "assessment": "Finally got a 2-handle but needs 30+ for capitulation",
    "agent_stop_adjustment": "tighten",
    "note": "VIX not fully uncoiled - more selling pressure likely"
  },

  "forward_outlook": {
    "near_term": "More downside. Shaky Friday, need weekend reset.",
    "multi_week": "Chop and repair. Credibility issues in crypto/software need resolution event.",
    "six_month": "Bullish on quality names. 'Great companies getting thrown out - we will pick up the pieces.' NASDAQ will eventually break out.",
    "agent_sizing": "Reduce to 50-70% of normal position sizes until VIX regime changes"
  },

  "earnings_impact": {
    "companies": [
      {"ticker": "AMZN", "reaction": "down_11_after_hours", "issue": "$200B CapEx - market doesn't believe it pays off", "index_drag": "high"},
      {"ticker": "GOOGL", "reaction": "mixed", "issue": "Great quarter but fading after hours. Similar CapEx concern.", "index_drag": "medium"}
    ],
    "narrative": "CapEx arms race scaring market. If CapEx peaks, entire AI trade reprices."
  },

  "cross_references": [
    {"video": "Tuesday", "context": "Gave QQQ 600 level - said if breaks, going to 588. HIT TODAY."},
    {"video": "Saturday upcoming", "context": "Deep dive on Anthropic impact, software .com parallels - WATCH THIS"}
  ],

  "trading_lessons": [
    "Covered MSFT short too early at 140 - 'surely that's the end of it' - it wasn't. Let winners run in panic.",
    "Record volume bars look like capitulation but 'not so fast' - biggest volume ever on IGV and it kept dropping.",
    "'You make decisions WHEN it gets to levels' - core philosophy of level-based trading.",
    "'Fear gets overblown, so does greed' - pendulum swings both ways."
  ]
}
```

## RAG Integration Notes

### Video Chaining (CRITICAL)
This creator's videos are sequential chapters, not standalone episodes:
- **Tuesday video** sets levels → **Thursday video** confirms/adjusts → **Saturday** deep dives
- Store ALL transcripts with date context
- When querying, retrieve last 3-5 videos for full narrative arc
- Track level predictions vs outcomes across videos

### Metadata to Store Per Video
```json
{
  "channel": "rta_trading",
  "date": "2026-02-05",
  "day_of_week": "Thursday",
  "is_deep_dive": false,
  "market_context": "post_close_with_earnings",
  "key_levels_given": [5887, 588, 580, 48000, 55],
  "active_shorts": ["AMZN", "MSTR", "AGQ", "CIRC", "BTC"],
  "active_longs": [],
  "next_video_preview": "Saturday deep dive on Anthropic, software sector, .com parallels",
  "cross_refs": ["Tuesday"]
}
```

### Agent Decision Integration
The agent uses RTA Trading analysis as a **market direction overlay**:

1. **Pre-market risk check**: Before market open, check latest RTA summary for gap expectation
2. **Risk regime**: If RISK-OFF → reduce position sizes, tighten stops, skip new longs
3. **Sector avoidance**: If he says a sector is "breaking", avoid small-caps in that sector
4. **VIX-based sizing**: His VIX thresholds inform our dynamic position sizing
5. **6-month forward tilt**: His longer-term calls adjust our sector weight preferences
6. **"Let it burn" mode**: When triggered, agent pauses new entries entirely
7. **Cross-video trend**: Chain 3-5 videos to detect if his stance is shifting bull→bear or vice versa

## Host's Trading Vocabulary

| Term | Meaning |
|------|---------|
| "Going through like butter" | Key support levels failing with no bounce |
| "Gap fill for ants" | Barely held a gap fill level before breaking |
| "Let it all burn" | Don't try to catch falling knives, wait for setup |
| "Tinfoil hat hour" | Speculative macro/political thesis (still worth hearing) |
| "Mother of dragons pattern" | Distribution top pattern (his term) |
| "Dumpster fire of the day" | Stocks getting absolutely destroyed |
| "Secular break" | Breaking below major long-term support (200 weekly MA) |
| "Two/three-handle on VIX" | VIX in 20s/30s range |
| "Reset" | Capitulation event needed to clear positioning |
| "Wash, rinse, repeat" | Institutional manipulation cycle |
| "You make decisions at levels" | Core philosophy - wait for price, don't anticipate |

## Long Video Addendum

This is a longer RTA Trading video (likely a Saturday Deep Dive). Extract MORE detail than usual:

- **Weekly VIX structure**: Full VIX term structure analysis — contango/backwardation across expirations, not just spot level
- **Options flow deep dive**: Any discussion of unusual options activity, put/call ratios, skew changes, gamma exposure shifts
- **Forward earnings schedule**: Which major earnings are coming next week and his expectations for market impact
- **Multi-week outlook**: His 2-4 week directional thesis with specific levels and conditions for bull/bear scenarios
- **Sector deep dives**: Extended analysis of individual sectors with weekly chart levels and rotation signals
- **Historical analogs**: Any references to past market periods (2022, 2020, 2018) as comparison templates
- **Position management philosophy**: Risk management wisdom and portfolio construction insights shared in longer format
- **Weekend event risk**: Any geopolitical, macro, or earnings events over the weekend that could gap markets Monday
