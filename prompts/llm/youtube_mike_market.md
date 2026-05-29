# Mike (Market Education / Tech Analysis) - YouTube Transcript Analyzer

## Producer Profile

**Channel**: [Mike's channel - market education focus]  
**Host**: Mike  
**Style**: Approachable educator, data-driven, shows charts & breadth daily, reads earnings  
**Typical Length**: 15-25 minutes  
**Upload Schedule**: Daily (post-close), Saturday deep dives (Q&A from comments)  
**Extras**: Morning news briefs (referenced in videos), membership community

## Why This Channel Matters

Mike fills a different niche than RTA Trading or Click Capital:
- **Market breadth obsession** - tracks % of stocks above 20/50/200 MA daily. This is GOLD for our universe of 1600 small/mid-caps because breadth tells you what the average stock is doing, not just mega-caps
- **Negative gamma / options flow** - explains dealer positioning and why selling accelerates. This directly affects our stop-loss triggers
- **Sector rotation tracking** - shows side-by-side sector performance daily, flags when defensive sectors start selling too (true bear signal)
- **"Year firsts"** - tracks unprecedented market events, keeps you aware of regime changes
- **Practical educator** - explains WHY things move (margin calls, leveraged ETFs forced selling, dealer hedging) not just what moved
- **Bond market as crash signal** - uses bond market stress as the key differentiator between normal pullback and real crash

## How We Use This (CRITICAL)

**Mike's value is MARKET INTERNALS + BREADTH, not specific stock picks.**

When he shows 62% of S&P above 20-day MA, that tells our agent more about what
our 1600 stocks are doing than any mega-cap analysis. His value to us:

1. **Breadth readings** - % stocks above 20/50/200 MA → our agent's broad market health score
2. **Negative gamma warnings** - dealer positioning driving forced selling → tighten all stops
3. **Sector rotation status** - when defensive sectors finally join the selling = real pullback incoming
4. **Bond market stress** - his key "real crash" indicator. If bonds fine → just a correction
5. **Options flow / gamma** - explains acceleration in selling. Helps calibrate stop distances
6. **V-shape recovery patterns** - he tracks these on IGV/software. Timing dip buys for our sectors
7. **Leverage/margin call context** - when he notes 100x crypto leverage unwinding, that selling spills into small-caps via risk-off sentiment

## Content Structure Pattern

Fairly consistent format:
1. **Memes** - humor opener (skip but sometimes reveals market mood)
2. **Economic data** - jobs, claims, layoffs with actual numbers
3. **S&P / QQQ chart** - key levels, moving averages, gaps
4. **Market breadth** - % above 20/50/200 MA for SPX, QQQ, Russell
5. **Sector rotation** - side-by-side sector performance
6. **Negative gamma / dealer positioning** - when relevant
7. **Individual charts** - usually 2-4 stocks/ETFs with specific levels
8. **Crypto analysis** - BTC/ETH with cycle context
9. **Earnings preview** - what's reporting tomorrow
10. **Saturday teaser** - asks for Q&A topics in comments

## Key Extraction Points

### MUST EXTRACT:
- **Market breadth numbers** - "62% above 20 MA", "51% on Nasdaq" - core health metric
- **Gamma regime** - positive/negative gamma, how many days in negative gamma
- **Bond market status** - "nothing going on with bond market" = no crash. Bond stress = real danger
- **Sector rotation signals** - which sectors green/red, and critically: are the green ones starting to turn red?
- **Critical threshold** - "5-10% pullback on S&P" trigger: when defensive sectors sell too
- **Moving average levels** - 50, 100, 200 day on indices. Where price sits relative to them
- **RSI levels** - weekly RSI above/below 50 (bull/bear control), oversold readings
- **MACD crosses** - weekly, 2-week, monthly bearish/bullish crosses
- **V-shape recovery historical pattern** - his IGV/software recovery thesis with magnitude
- **Specific levels and gaps** - December lows, January lows, gap fills
- **Economic data** - jobs claims, layoffs, actual numbers vs expectations
- **Earnings reactions** - how after-hours moves affect tomorrow's open
- **Leverage/margin call warnings** - crypto 100x leverage, leveraged ETF forced selling

### SECONDARY:
- His individual stock levels (AMD gap, MSFT trend line) - useful as mega-cap health check
- Crypto cycle analysis (200-week MA, 4-year cycle) - sentiment indicator
- Net flows data - retail vs institutional positioning

### FILTER OUT:
- Meme section (occasionally captures mood but rarely actionable)
- Channel promotion / membership plugs
- Whisper repetition artifacts (this transcript had many)
- Saturday teaser / comment requests
- Detailed options mechanics explanations (educational but we need the conclusion not the lesson)

## Extraction Prompt

```
You are analyzing a Mike (market education) YouTube transcript to extract MARKET
INTERNALS and BREADTH data for AutoTrade's small/mid-cap trading system.

CONTEXT:
- AutoTrade trades ~1600 small/mid-cap stocks ($2-$200 price range)
- We DO NOT trade mega-caps but breadth data tells us what the AVERAGE stock is doing
- Mike's key value: market breadth, gamma regime, sector rotation, bond market heath
- His bond market stress indicator differentiates corrections from crashes
- He tracks % of stocks above moving averages daily - this IS our market health score

EXTRACT THE FOLLOWING:

## 1. Market Breadth Snapshot
For each index reported (SPX, QQQ, Russell):
- % above 20-day MA
- % above 50-day MA
- % above 200-day MA
- Direction: improving / deteriorating / stable
This is the MOST IMPORTANT section - it directly measures our universe's health.

## 2. Gamma Regime
- Current regime: POSITIVE (dealers buying = tailwind) or NEGATIVE (dealers selling = headwind)
- Days in current regime
- Impact: "selling accelerates" / "buying support" / "neutral"
- When does it flip? Any conditions mentioned

## 3. Crash vs Correction Assessment
Mike's key framework:
- Bond market status: STRESS / CALM / NEUTRAL
- If bonds calm → correction, not crash → our agent stays in with tighter stops
- If bonds stress → crash risk → our agent goes maximum defensive
- His actual words on this (quote when possible)

## 4. Sector Rotation Status
For each sector mentioned:
- Sector name
- Status: GREEN / RED / TURNING
- CRITICAL FLAG: Are the previously-green sectors (defensives, staples) joining the selling?
  If YES → "real pullback" signal (5-10% on SPX, more on QQQ)
  If NO → rotation still healthy, not systemic

## 5. Index Technical Levels
For SPX and QQQ:
- Current price area
- Key support levels (50/100/200 MA, gaps, prior lows)
- Distance to key levels in %
- RSI status (weekly above/below 50)
- MACD crosses (weekly, 2-week, monthly)

## 6. V-Shape Recovery Watch
- Any mention of historical V-shaped recoveries in sectors
- ETF magnitude ("10-20% jumps on ETF, stocks make way more")
- Conditions needed for recovery to trigger
- What needs to bottom first (e.g., "does MSFT have to bottom first?")

## 7. Economic Data
- Jobs/claims numbers vs expectations
- Layoff data and context
- Any data surprising to the upside or downside
- Tomorrow's economic calendar

## 8. Tomorrow's Setup
- Earnings reporting after-hours / pre-market
- Expected gap direction
- Key sector to watch (e.g., "if semis open green, market gets help")
- Catalyst list for next session

## 9. Leverage & Margin Call Context
- Crypto margin calls / leverage unwind status
- Leveraged ETF forced selling
- Retail flow data (net ETF flows)
- Impact on overall market selling pressure

---
TRANSCRIPT:
{transcript}
---

OUTPUT FORMAT:
- Start with BREADTH SCORE: one line summarizing market health (e.g., "62% above 20MA, breadth improving despite index weakness - internals healthier than price suggests")
- CRASH RISK: LOW/MEDIUM/HIGH based on bond market status
- Then structured sections above
- End with AGENT ACTION: 2-3 sentences on position sizing and sector exposure for tomorrow
```

## Example Output Format

```json
{
  "date": "2026-02-05",
  "video_title": "Two Big Reasons Tech & Crypto Continue to Selloff",
  
  "breadth_score": "SPX 62% above 20MA (up 7%), 50MA and 200MA also improving. Breadth healthier than price action suggests. Russell holding 0.5 fib. Nasdaq weakest at 51%.",
  "crash_risk": "LOW - Bond market showing no stress signals. This is a correction, not a crash.",
  
  "agent_action": "CAUTIOUS but not defensive. Breadth improving underneath = average stock doing okay. Tighten stops due to negative gamma (3 days), but no need to dump positions. Avoid software/crypto-adjacent names. If defensive sectors start selling, THEN go full defensive.",

  "market_breadth": {
    "spx": {"above_20ma": 62, "above_50ma": "+4%", "above_200ma": "+5%", "direction": "improving"},
    "nasdaq": {"above_20ma": 51, "direction": "deteriorating"},
    "russell": {"status": "holding 0.5 fib, not broken down", "direction": "stable"},
    "interpretation": "74% of stocks actually up. Not how bear markets start. Selling concentrated in tech/software."
  },

  "gamma_regime": {
    "current": "NEGATIVE",
    "days": 2,
    "tomorrow": "3 days negative unless strong overnight rally",
    "impact": "Dealers selling to hedge puts → accelerating downside pressure. Margin calls + leveraged ETF forced selling adding to it.",
    "agent_implication": "Wider stops needed - moves overshoot in negative gamma"
  },

  "crash_assessment": {
    "bond_market": "CALM",
    "mikes_words": "Right now there's nothing going on with the bond market that I can see... you get your real 20% crashes when the bond market is signaling stuff",
    "verdict": "Correction only. April's crash had bond stress. This doesn't.",
    "agent_response": "Stay invested but tighter risk. This is a dip, not a regime change."
  },

  "sector_rotation": {
    "status": "ROTATION_INTACT",
    "red": ["Software/IGV", "Tech/XLK", "Semiconductors/SMH", "Crypto"],
    "green_to_watch": ["S&P equal weight sectors", "Transports", "Defensives"],
    "critical_flag": false,
    "mikes_warning": "If the ones that have been green start to sell down too and you don't see rotation into tech - THEN you get 5-10% on S&P, QQQ down more."
  },

  "index_levels": {
    "spy": {
      "status": "Took out Jan 2026 lows today. Stopped at 100 MA.",
      "supports": ["100 MA (current)", "December lows (gap below)", "200 MA (~4.5% lower)"],
      "pattern": "Double top almost played out"
    },
    "qqq": {
      "status": "Below 50 and 100 MA. Already took out December lows.",
      "supports": ["200 MA (~3% lower, would be 9.2% total drawdown)"],
      "weekly_rsi": "At 50 line - bulls must hold this",
      "macd": "Weekly bearish cross confirmed. 2-week bearish cross forming. Monthly not yet."
    }
  },

  "v_shape_watch": {
    "sector": "Software/IGV",
    "pattern": "Historical V-shaped recoveries: 10-20% ETF jumps, individual stocks much more",
    "condition": "Does MSFT need to bottom first? Coming down to 0.79 fib level",
    "count": "Would be 9th V-shape recovery if pattern holds",
    "risk": "This time different? AI disruption is real. Could be structural not cyclical."
  },

  "economic_data": {
    "jobs_claims": "Higher than expected - finally breaking the declining trend",
    "layoffs": "108,435 in January - highest since 2009",
    "continuing_claims": "1.8-1.9M range, hanging in same band",
    "tomorrow": "Average hourly earnings, non-farm payrolls, unemployment rate"
  },

  "leverage_context": {
    "crypto": "100x leverage common. Margin calls galore. Accounts wiped out. Never recovered from Binance liquidation event.",
    "etfs": "2x 3x leveraged ETFs for every stock - all being forced to sell",
    "retail_flows": "January ETF net flows highest since 2015 - retail was piling in at the top",
    "impact": "Forced selling extends beyond fundamentals. Overshoots in negative gamma."
  },

  "key_watches_tomorrow": [
    "Amazon after-hours reaction → gap down? Semi response to CapEx news",
    "If semiconductors open GREEN → market gets support (17% of SPX)",
    "Jobs report (NFP, unemployment) → could flip narrative",
    "3rd day of negative gamma → selling pressure continues",
    "Defensive sectors - if they start selling, shift to full defensive"
  ]
}
```

## RAG Integration Notes

### Breadth Tracking (UNIQUE VALUE)
Mike's daily breadth readings should be stored as time series:
```json
{
  "date": "2026-02-05",
  "spx_above_20ma": 62,
  "spx_above_50ma": null,
  "spx_above_200ma": null,
  "qqq_above_20ma": 51,
  "russell_status": "holding",
  "gamma_regime": "negative",
  "gamma_days": 2,
  "bond_stress": false,
  "crash_risk": "low"
}
```
Trend the breadth over time. Rising breadth + falling index = healthy correction.
Falling breadth + falling index = real trouble.

### Agent Decision Integration
Mike's data plugs into our agent as a **market health overlay**:

1. **Breadth score** → modulate position sizing (high breadth = full size, low = reduce)
2. **Gamma regime** → adjust stop distances (negative gamma = wider stops, moves overshoot)
3. **Bond market status** → crash vs correction decision tree (calm = buy dips, stress = sell)
4. **Sector rotation flag** → if defensives join selling, agent goes max defensive
5. **V-shape recovery watch** → when software/IGV triggers, our software-adjacent small-caps will rip too
6. **Leverage/margin context** → when forced selling dominates, fundamentals don't matter. Wait.
7. **Jobs/economic data** → macro overlay for sector preferences

### Complementary to Other Channels
| Data Point | Mike | RTA Trading | Click Capital |
|------------|------|-------------|---------------|
| Market breadth | PRIMARY | - | - |
| Gamma/dealer flow | PRIMARY | - | - |
| Index levels | Good | PRIMARY | Basic |
| Sector rotation | Good | Good | Good |
| Bond/crash signal | PRIMARY | - | Mentions |
| VIX analysis | Basic | PRIMARY | Good |
| Individual trades | Basic | PRIMARY | Basic |
| Macro/sentiment | Good | Good | PRIMARY |
| Forward outlook | Good | PRIMARY | Good |

## Mike's Key Phrases & Concepts

| Term | Meaning |
|------|---------|
| "Negative gamma" | Dealers hedging by selling → selling accelerates. Rare but critical. |
| "Year first" | Unprecedented market event - regime awareness |
| "Bond market signaling" | His crash indicator. Bond stress = real crash. No stress = correction. |
| "V-shape recovery" | Historical pattern in IGV/software - 10-20% ETF bounce, stocks more |
| "Real pullback" | 5-10% on SPX - only happens when ALL sectors sell together |
| "Under the hood" | Market internals/breadth vs headline index level |
| "Margin calls galore" | Leveraged positions being liquidated regardless of fundamentals |
| "Dead cat bounce" | A temporary rally in a downtrend - "we'll find out" |
| "Hot knife through butter" | Support levels failing with zero resistance |
| "Gap fill" | Price returning to fill a prior gap - he tracks half-gap fills too |
| "Weekly RSI 50" | His bull/bear dividing line. Above 50 = bulls. Below = bears. |
| "Bearish MACD cross" | Momentum turning negative. Monthly cross = "dead certain major pullback" |

## Long Video Addendum

This is a longer Mike video (likely a Saturday Q&A or deep dive). Extract MORE detail than usual:

- **Weekly breadth trends**: Not just today's snapshot — how breadth has evolved over the past week. Is it trending up, down, or choppy?
- **Gamma evolution**: How has gamma regime changed over the week? How many consecutive days in positive/negative gamma? Dealer positioning shift
- **Bond market weekly**: Full bond market assessment — yields, credit spreads, any stress signals building or dissipating over the week
- **Economic calendar impact**: Next week's economic data releases and his expectations for market reaction
- **V-shape recovery analysis**: Deeper historical comparison — which past V-shapes match current setup? What triggered the bounce?
- **Leverage/margin weekly**: How has the leverage unwind progressed? Any signs of forced selling exhaustion?
- **Monthly MACD / RSI analysis**: Longer timeframe indicators he only covers in deep dives
- **Q&A insights**: If this is a Saturday Q&A, extract the most actionable questions and his answers
