# Click Capital (Jarrod) - YouTube Transcript Analyzer

## Producer Profile

**Channel**: Click Capital  
**Host**: Jarrod  
**Show**: The Daily Market Review  
**Style**: Australian market commentator, contrarian lean, momentum/factor investor  
**Typical Length**: 15-25 minutes  
**Upload Schedule**: Daily (market days)

## Content Structure Pattern

Jarrod follows a consistent format:
1. **Clickbait intro** - Dramatic headlines (often exaggerated for engagement)
2. **Market overview** - Fear & Greed index, sentiment shift
3. **Sector rotation** - What's getting hit, what's rotating into
4. **Deep dives** - 3-5 specific trades/stocks with data
5. **Macro context** - Jobs, Fed, GDP
6. **Volatility outlook** - VIX, term structure
7. **Philosophical close** - "Don't get scared out" wisdom

## Key Extraction Points

### MUST EXTRACT:
- **Specific tickers mentioned** with his stance (bullish/bearish/neutral)
- **Price levels** - Support/resistance, targets
- **Valuations** - PE ratios, comparisons to historical
- **Sentiment indicators** - Fear & Greed readings, VIX levels
- **Rotation signals** - Which sectors money is flowing to/from
- **Earnings reactions** - How market responded to recent reports

### FILTER OUT:
- Clickbait intro hysteria (title is usually exaggerated)
- Repetitive phrases (Whisper sometimes stutters)
- Generic market education (explaining what momentum is)
- Philosophical/motivational closers
- Self-promotion and channel plugs

## Extraction Prompt

```
You are analyzing a Click Capital (Jarrod) YouTube transcript for actionable trading intelligence.

CONTEXT:
- AutoTrade trades ~1600 small/mid-cap stocks ($2-$200 price range)
- We DO NOT trade mega-caps (AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META)
- Focus on: sector rotation signals, momentum plays, sentiment shifts

EXTRACT THE FOLLOWING:

## 1. Market Sentiment Summary
- Fear & Greed index reading and direction
- Jarrod's overall market stance (bullish/bearish/cautious)
- Key phrase that captures his thesis

## 2. Sector Rotation Signals
Format: SECTOR | DIRECTION | REASON
Example: Software | BEARISH | AI disruption narrative, oversold RSI

## 3. Individual Stock Mentions
For EACH stock mentioned, extract:
- Ticker (if identifiable)
- Jarrod's stance: BULLISH / BEARISH / NEUTRAL
- Key data point (PE, price level, % move)
- His reasoning (1 sentence)

EXCLUDE mega-caps unless discussing sector implications.

## 4. Volatility & Risk Signals
- VIX level and term structure (contango/backwardation)
- Any "sigma move" references
- Margin/leverage warnings

## 5. Macro Data Points
- Jobs data
- Fed rate expectations
- GDP/earnings trends

## 6. Contrarian Opportunities
Jarrod often identifies oversold opportunities. List any explicit "buy the dip" mentions with his reasoning.

## 7. Actionable Takeaways
3-5 bullet points for trading decisions.

---
TRANSCRIPT:
{transcript}
---

Respond in the structured format above. Be concise - extract data, not narrative.
```

## Example Output Format

```json
{
  "date": "2026-02-05",
  "sentiment": {
    "fear_greed": 12,
    "direction": "falling",
    "jarrod_stance": "cautious_bullish",
    "thesis": "momentum crash, not fundamental - expect chop then recovery"
  },
  "sector_rotation": [
    {"sector": "Software", "direction": "bearish", "reason": "AI disruption fear, 10yr RSI low"},
    {"sector": "Consumer Staples", "direction": "bullish", "reason": "defensive rotation"},
    {"sector": "Crypto", "direction": "bearish", "reason": "leverage unwind"}
  ],
  "stock_mentions": [
    {"ticker": "COIN", "stance": "bullish", "data": "PE 14", "reason": "oversold despite business strength"},
    {"ticker": "DUOL", "stance": "bullish", "data": "PE 14, double-digit rev growth", "reason": "AI beneficiary not victim"},
    {"ticker": "HOOD", "stance": "bullish", "data": "PE 33, -50% from highs", "reason": "reasonable value for growth"}
  ],
  "volatility": {
    "vix_level": "elevated",
    "term_structure": "backwardation",
    "warning": "more volatility near-term expected"
  },
  "contrarian_plays": [
    "Software sector historically bounces 3-6mo after this oversold",
    "Crypto fear index at 12 - extreme fear zone"
  ],
  "actionable": [
    "Expect continued chop 2-3 weeks",
    "Software dip-buy window opening",
    "Avoid leveraged positions",
    "Watch for VIX to uncoil before adding risk"
  ]
}
```

## Integration Notes

- **Relevant to AutoTrade**: Sector rotation signals help identify which industries to favor/avoid
- **Not directly tradeable**: Most stocks Jarrod covers are mega-caps outside our universe
- **Use for**: Overall market sentiment, sector tilt adjustments, risk management
- **Frequency**: Process daily videos for sentiment baseline

## Jarrod's Common Terms

| Term | Meaning |
|------|---------|
| Debasement trade | Long hard assets (gold, silver, crypto) + short USD |
| Fat tails | Extreme moves happen more than normal distribution suggests |
| Dry powder | Cash available for dip buying |
| Shakeout/Washout | Forced selling/capitulation |
| Sigma move | Standard deviation event (6-sigma = extremely rare) |
| K-shape economy | Divergence between wealthy and working class |
| Momentum factor | Stocks that have risen recently tend to continue |

## Long Video Addendum

This is a longer Click Capital video (likely a weekend deep dive). Extract MORE detail than usual:

- **Multi-week macro outlook**: Extended macro thesis beyond the daily recap — where does Jarrod see markets in 2-4 weeks?
- **Portfolio construction**: Any discussion of allocation, diversification, position sizing philosophy
- **Sector rotation patterns**: Deeper analysis of sector flows — which sectors are building momentum vs losing it over multiple weeks
- **Fear & Greed weekly trend**: Not just today's reading — how has sentiment evolved over the week? Direction and velocity of change
- **Contrarian deep dives**: Extended analysis of oversold opportunities with historical comparisons and recovery timelines
- **Global macro**: International markets, currencies, commodities — broader context for US small/mid-caps
- **Factor analysis**: Any discussion of value vs growth, quality vs momentum, large vs small-cap rotation trends
- **Risk scenarios**: Tail risk scenarios he's watching and what would trigger them
