# StockedUp - YouTube Transcript Analyzer

## Producer Profile

**Channel**: StockedUp
**Host**: Mike D'Antonio / StockedUp team
**Focus**: Daily stock/options watchlists, support and resistance, unusual options flow, custom indicator context, and short-term trading sentiment.
**Use in AutoTrade**: Add actionable ticker and level coverage when one of the legacy daily channels misses a session. Use market bias as confirmation, not standalone authority.

## Key Extraction Focus

1. **Ticker Watchlist**: Extract every stock or ETF with a concrete setup, level, or directional bias.
2. **Support/Resistance**: Preserve exact entry, breakout, breakdown, stop, and target levels.
3. **Options Flow / Smart Money**: Capture unusual options or flow comments, including direction and expiry if spoken.
4. **Market Bias**: Separate SPY, QQQ, and IWM commentary. Do not infer small-cap health unless IWM/Russell or breadth is mentioned.
5. **AutoTrade Fit**: Mark whether each ticker is likely inside the $2-$200 small/mid-cap universe.

## Filtering Rules

- Ignore Discord/course promotion, generic education, and repeated stream housekeeping.
- Treat options-only commentary as a directional signal only if price levels or confirmation rules are provided.
- Do not create broad avoid_sectors from generic "market is risky" statements. Require sector-specific evidence.
- If no actionable setup exists, return empty trade_ideas and a neutral market_bias.

## Extraction Prompt

```
You are analyzing a StockedUp YouTube transcript for AutoTrade, an automated small/mid-cap stock trading system.

AutoTrade trades roughly 1600 stocks priced $2-$200. It can use StockedUp for ticker setups, support/resistance levels, options-flow confirmation, and broad risk context. Keep outputs specific and machine-readable.

EXTRACT:

1. MARKET BIAS
- SPY bias and levels
- QQQ bias and levels
- IWM/Russell bias and levels if mentioned
- overall bias: bullish / bearish / neutral
- confidence: 0-100

2. TICKER SETUPS
For each mentioned ticker:
- ticker
- direction: long / short / watch
- setup type
- entry trigger
- stop or invalidation
- target(s)
- timeframe
- evidence phrase
- likely_in_autotrade_universe: true/false

3. OPTIONS FLOW
For each flow mention:
- ticker
- call_or_put
- expiry
- strike
- bullish_or_bearish
- reason

4. SECTOR OR THEME BIAS
Only include specific sectors/themes with evidence:
- sector_or_theme
- bias: overweight / neutral / underweight / avoid
- reason
- confidence

5. AGENT ACTIONS
Return 3-5 concrete instructions for AutoTrade, focused on levels, tickers, and risk sizing. Avoid blanket universe-level vetoes.

TRANSCRIPT:
{transcript}

Return structured JSON with these top-level keys:
{
  "market_bias": "",
  "market_regime": "",
  "regime_confidence": 0,
  "index_levels": {},
  "trade_ideas": [],
  "options_flow": [],
  "sector_bias": [],
  "agent_actions": []
}
```

## Long Video Addendum

For long videos, perform a deeper extraction of any "top stocks", watchlists, or live-scanned names. Preserve all repeated levels only once, with the clearest explanation.
