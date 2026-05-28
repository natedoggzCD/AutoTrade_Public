import pandas as pd
from autotrade.signals.contracts import SignalContext
from autotrade.signals.alpha.fundamental import PEADAlphaSource, SqueezeAlphaSource

def test_integration_fundamental_db():
    # Use very low thresholds to find SOMETHING in the existing DB
    pead = PEADAlphaSource(lookback_days=60, min_surprise=1.0)
    squeeze = SqueezeAlphaSource(min_short_pct=0.001) # 0.1% or lower
    
    context = SignalContext(tickers=["AAPL", "TSLA", "MSFT", "AMD", "NVDA"], price_data=pd.DataFrame())
    
    pead_signals = pead.generate(context)
    squeeze_signals = squeeze.generate(context)
    
    print(f"\nFound {len(pead_signals)} PEAD signals")
    for s in pead_signals:
        print(f"  {s.ticker}: {s.reason}")
        
    print(f"Found {len(squeeze_signals)} Squeeze signals")
    for s in squeeze_signals:
        print(f"  {s.ticker}: {s.reason}")

    # Based on our sqlite3 queries, we expect at least AAPL for PEAD 
    # and TSLA/AMD/NVDA for Squeeze
    assert len(pead_signals) > 0
    assert len(squeeze_signals) > 0
    print("\nIntegration test PASSED!")

if __name__ == "__main__":
    test_integration_fundamental_db()
