# FinFluential Television - YouTube Transcript Analyzer

## Producer Profile

**Channel**: FinFluential Television
**Host**: FinFluential
**Focus**: Macro context, S&P 500, Nasdaq 100, Russell 2000, treasury bonds, Bitcoin, and broad risk appetite.
**Use in AutoTrade**: Cross-check index regime and small-cap/Russell 2000 conditions without allowing macro commentary to veto our full small/mid-cap universe by itself.

## Key Extraction Focus

1. **Index Regime**: Extract the host's SPY/S&P 500, QQQ/Nasdaq 100, and IWM/Russell 2000 bias separately.
2. **Small-Cap Readthrough**: Preserve any direct comments on Russell 2000, IWM, breadth, risk appetite, or speculative growth.
3. **Macro Drivers**: Capture treasury yields, dollar, inflation, Fed, recession, liquidity, and Bitcoin/crypto risk signals only when they affect next-session equity risk.
4. **Trigger Levels**: Preserve exact index, yield, VIX, or Bitcoin levels with above/below implications.
5. **Actionability**: Convert commentary into concrete AutoTrade directives: size up/down, favor/avoid sectors, or wait for confirmation.

## Filtering Rules

- Treat mega-cap commentary as index context only unless a ticker is in the $2-$200 universe.
- Do not turn "avoid small caps/speculation" into a sector-level avoid. Flag it as a universe-level caution instead.
- Ignore channel promotion, generic education, and non-market personal finance content.
- If the transcript lacks direct trading implications, return neutral regime and explain the missing evidence.

## Extraction Prompt

```
You are analyzing a FinFluential Television YouTube transcript for AutoTrade, an automated small/mid-cap stock trading system.

AutoTrade trades roughly 1600 small/mid-cap stocks in the $2-$200 price range. It uses YouTube only as contextual intelligence. Macro warnings must be translated into specific, evidence-backed trading adjustments and must not become a blanket "avoid all small/mid-caps" instruction.

EXTRACT:

1. MARKET REGIME
- classification: RISK-ON / RISK-ON-SELECTIVE / NEUTRAL / ROTATION / RISK-OFF-LIGHT / RISK-OFF / CRASH
- confidence: 0-100
- reason: one concise paragraph

2. INDEX AND SMALL-CAP CONTEXT
- SPY/S&P 500 bias and key levels
- QQQ/Nasdaq 100 bias and key levels
- IWM/Russell 2000 bias and key levels
- whether small-caps are healthy, neutral, deteriorating, or breaking
- whether weakness is mega-cap-specific or broad

3. MACRO DRIVERS
- treasury yields / bonds
- dollar
- Fed or inflation expectations
- Bitcoin/crypto risk appetite
- recession or liquidity signals
- direct implication for next-session equity risk

4. SECTOR AND THEME BIAS
For each mentioned sector/theme:
- sector_or_theme
- bias: overweight / neutral / underweight / avoid
- reason
- confidence

5. TRIGGER LEVELS
For each specific level:
- instrument
- level
- if_above
- if_below
- source_phrase

6. TRADE IDEAS OR TICKERS
For each ticker mentioned:
- ticker
- direction
- setup
- entry
- stop
- target
- whether it is likely in AutoTrade's $2-$200 universe

7. AGENT ACTIONS
Return 3-5 concrete instructions for AutoTrade. Keep sector avoids specific. If the only caution is "small caps/speculation", mark it as universe_level_caution instead of avoid_sectors.

TRANSCRIPT:
{transcript}

Return structured JSON with these top-level keys:
{
  "market_regime": "",
  "regime_confidence": 0,
  "smallcap_health": {"status": "", "reason": ""},
  "index_context": {},
  "macro_drivers": [],
  "sector_bias": [],
  "trigger_levels": [],
  "trade_ideas": [],
  "universe_level_caution": [],
  "agent_actions": []
}
```

## Long Video Addendum

For long videos, preserve the sequence of the thesis: what changed, what level confirms it, what invalidates it, and which index or sector matters most for the next session.
