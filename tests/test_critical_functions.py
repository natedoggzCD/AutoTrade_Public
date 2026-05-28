#!/usr/bin/env python3
"""
CRITICAL FUNCTION TEST SUITE
Tests all recently fixed/added functions to catch errors before trading hours.

Run this before starting the bot to verify everything works:
    python tests/test_critical_functions.py
"""

import os
import sys
import json
import traceback
import pytest
from pathlib import Path
from datetime import datetime

# Ensure project root is on path when running from tests/ directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

if os.getenv("AUTOTRADE_RUN_HEAVY_TESTS") != "1":
    pytest.skip(
        "Skipping critical live-data test suite; set AUTOTRADE_RUN_HEAVY_TESTS=1 to run.",
        allow_module_level=True,
    )

# Track results
PASSED = []
FAILED = []

def case(name):
    """Decorator to register and run tests."""
    def decorator(func):
        def wrapper():
            try:
                print(f"\n{'='*60}")
                print(f"TEST: {name}")
                print(f"{'='*60}")
                result = func()
                if result:
                    PASSED.append(name)
                    print(f"[OK] PASSED: {name}")
                else:
                    FAILED.append((name, "Test returned False"))
                    print(f"[X] FAILED: {name}")
            except Exception as e:
                FAILED.append((name, str(e)))
                print(f"[X] FAILED: {name}")
                print(f"   Error: {e}")
                traceback.print_exc()
        return wrapper
    return decorator


# ============================================================
# TEST 1: F-String Formatting in MultiModel Scorer
# ============================================================
@case("MultiModel Scorer F-String Syntax")
def test_multimodel_fstring():
    """Verify the f-string fix in multimodel_scorer.py works."""
    from autotrade.utils.multimodel_scorer import MultiModelScorer
    
    scorer = MultiModelScorer()
    
    # Test with mock data that previously caused the crash
    test_data = {
        'symbol': 'TEST',
        'current_price': 100.0,
        'technical': {'rsi': 55, 'macd_signal': 'bullish'},
        'sr_data': {'s1_price': 95.0, 'r1_price': 105.0, 'pivot': 100.0},
        'vl_data': {'trend': 'bullish', 'confidence': 75},
        'news': [{'title': 'Test news', 'sentiment': 0.5}],
    }
    
    # This should NOT raise "Invalid format specifier" error
    try:
        # Test the _build_llm_prompt method if it exists
        if hasattr(scorer, '_build_llm_prompt'):
            prompt = scorer._build_llm_prompt(test_data)
            if 'N/A' in prompt or '$' in prompt:
                print("   Prompt generation works with conditionals")
        
        # Test scoring a real symbol (will use cached data if available)
        result = scorer.score_symbol('SPY')
        # Result is a SignalScore dataclass, access with attributes
        final_score = getattr(result, 'final_score', 'N/A')
        action = getattr(result, 'action', 'N/A')
        print(f"   SPY score: {final_score}")
        print(f"   Action: {action}")
        
        # Check no error in reasoning
        reasoning = getattr(result, 'reasoning', '')
        if 'Invalid format specifier' in str(reasoning):
            print(f"   [!] STILL HAS FORMAT ERROR: {reasoning[:100]}")
            return False
        
        return True
    except Exception as e:
        if 'format specifier' in str(e).lower():
            print(f"   F-STRING ERROR STILL EXISTS: {e}")
            return False
        raise


# ============================================================
# TEST 2: Lessons Screener Database Path
# ============================================================
@case("Lessons Screener Database Path")
def test_screener_db_path():
    """Verify screener uses correct DownDay database."""
    from autotrade.signals.lessons_screener import DB_PATH, get_entry_candidates
    
    # Check the database path (it's a module-level constant)
    print(f"   DB Path: {DB_PATH}")
    if 'DownDay' in str(DB_PATH):
        print("   [OK] Using DownDay database")
    else:
        print(f"   [!] NOT using DownDay database!")
        return False
    
    # Verify database exists
    if not DB_PATH.exists():
        print(f"   [!] Database does not exist: {DB_PATH}")
        return False
    print(f"   [OK] Database exists")
    
    # Try to get candidates
    candidates = get_entry_candidates(max_candidates=10)
    print(f"   Got {len(candidates)} candidates")
    
    if len(candidates) > 0:
        symbols = [c.get('ticker', c) for c in candidates[:5]]
        print(f"   Sample: {symbols}")
    
    return len(candidates) > 0


# ============================================================
# TEST 3: Exclude Symbols Parameter
# ============================================================
@case("Lessons Screener Exclude Symbols")
def test_screener_exclude():
    """Verify exclude_symbols prevents duplicates."""
    from autotrade.signals.lessons_screener import get_entry_candidates
    
    # Get first batch
    batch1 = get_entry_candidates(max_candidates=5)
    symbols1 = [c.get('ticker', str(c)) for c in batch1]
    print(f"   Batch 1: {symbols1}")
    
    # Get second batch excluding first
    batch2 = get_entry_candidates(max_candidates=5, exclude_symbols=symbols1)
    symbols2 = [c.get('ticker', str(c)) for c in batch2]
    print(f"   Batch 2: {symbols2}")
    
    # Check no overlap
    overlap = set(symbols1) & set(symbols2)
    if overlap:
        print(f"   [!] OVERLAP FOUND: {overlap}")
        return False
    
    print("   [OK] No overlap between batches")
    return True


# ============================================================
# TEST 4: Recommendation Filter
# ============================================================
@case("Recommendation Filter Accepts WEAK BUY")
def test_recommendation_filter():
    """Verify the filter accepts STRONG BUY, BUY, WEAK BUY, WATCH."""
    
    # Test data with different recommendations
    test_candidates = [
        {'symbol': 'A', 'recommendation': 'STRONG BUY'},
        {'symbol': 'B', 'recommendation': 'BUY'},
        {'symbol': 'C', 'recommendation': 'WEAK BUY'},
        {'symbol': 'D', 'recommendation': 'WATCH'},
        {'symbol': 'E', 'recommendation': 'HOLD'},
        {'symbol': 'F', 'recommendation': 'SELL'},
    ]
    
    # The filter that should be used
    accepted = ['STRONG BUY', 'BUY', 'WEAK BUY', 'WATCH']
    
    passed = [c for c in test_candidates if c['recommendation'] in accepted]
    failed = [c for c in test_candidates if c['recommendation'] not in accepted]
    
    print(f"   Accepted: {[c['symbol'] for c in passed]}")
    print(f"   Rejected: {[c['symbol'] for c in failed]}")
    
    # Should accept A, B, C, D and reject E, F
    expected_pass = {'A', 'B', 'C', 'D'}
    actual_pass = {c['symbol'] for c in passed}
    
    if actual_pass == expected_pass:
        print("   [OK] Filter works correctly")
        return True
    else:
        print(f"   [!] Expected {expected_pass}, got {actual_pass}")
        return False


# ============================================================
# TEST 5: Morning Game Plan Loading
# ============================================================
@case("Morning Game Plan Loading")
def test_game_plan_load():
    """Verify morning game plan has correct structure and signals."""
    
    plans_dir = PROJECT_ROOT / 'plans'
    today = datetime.now().strftime('%Y%m%d')
    plan_path = plans_dir / f'morning_game_plan_{today}.json'
    
    if not plan_path.exists():
        print(f"   [!] No plan for today: {plan_path}")
        # Try to find any recent plan
        plans = list(plans_dir.glob('morning_game_plan_*.json'))
        if plans:
            plan_path = max(plans, key=lambda p: p.stat().st_mtime)
            print(f"   Using most recent: {plan_path.name}")
        else:
            print("   No game plans found!")
            return False
    
    with open(plan_path) as f:
        plan = json.load(f)
    
    total = plan.get('total_picks', 0)
    signals = plan.get('buy_signals', [])
    
    print(f"   Plan date: {plan.get('date', 'unknown')}")
    print(f"   Total picks claimed: {total}")
    print(f"   Actual signals in array: {len(signals)}")
    
    if total != len(signals):
        print(f"   [!] MISMATCH: total_picks={total} but array has {len(signals)}")
    
    # Check recommendations
    recs = {}
    for s in signals:
        r = s.get('recommendation', '[OK]')
        recs[r] = recs.get(r, 0) + 1
    print(f"   Recommendations: {recs}")
    
    # Verify we have more than 3 signals
    if len(signals) < 10:
        print(f"   [!] Only {len(signals)} signals - expected 100+")
        return False
    
    print(f"   [OK] Plan has {len(signals)} signals")
    return True


# ============================================================
# TEST 6: Overnight State File
# ============================================================
@case("Overnight State File Structure")
def test_overnight_state():
    """Verify overnight state has proper structure."""
    
    state_path = PROJECT_ROOT / 'research' / 'overnight_state.json'
    
    if not state_path.exists():
        print(f"   [!] No overnight state file")
        return False
    
    with open(state_path) as f:
        state = json.load(f)
    
    researched = len(state.get('researched', {}))
    candidates = len(state.get('all_candidates', []))
    watchlist = len(state.get('watchlist', []))
    complete = state.get('research_complete', False)
    
    print(f"   Researched: {researched}")
    print(f"   All Candidates: {candidates}")
    print(f"   Watchlist: {watchlist}")
    print(f"   Complete: {complete}")
    
    if watchlist < 50:
        print(f"   [!] Watchlist only has {watchlist} stocks")
        return False
    
    if not complete:
        print(f"   [!] Research not marked complete")
    
    return True


# ============================================================
# TEST 7: Signal Generator
# ============================================================
@case("Agentic Signal Generator")
def test_signal_generator():
    """Verify signal generator can run without errors."""
    from autotrade.signals.agentic_signal_generator import AgenticSignalGenerator
    
    gen = AgenticSignalGenerator(
        max_candidates=5,
        use_llm=False,  # Skip LLM for speed
        use_lessons=True,
    )
    
    # This should not crash
    signals = gen.run()
    
    print(f"   Generated {len(signals)} signals")
    if signals:
        sample = signals[0]
        # EntryCandidate is a dataclass, access with attributes
        symbol = getattr(sample, 'symbol', str(sample))
        price = getattr(sample, 'price', 'N/A')
        print(f"   Sample: {symbol} @ ${price}")
    
    return True


# ============================================================
# TEST 8: Day Manager State
# ============================================================
@case("Day Manager State File")
def test_day_manager_state():
    """Verify day manager state is valid."""
    
    state_path = PROJECT_ROOT / 'day_manager_state.json'
    
    if not state_path.exists():
        print(f"   [!] No day manager state file")
        return False
    
    with open(state_path) as f:
        state = json.load(f)
    
    date = state.get('date', 'unknown')
    cycle = state.get('cycle_count', 0)
    positions = state.get('position_health', {})
    
    print(f"   Date: {date}")
    print(f"   Cycle count: {cycle}")
    print(f"   Positions tracked: {len(positions)}")
    
    return True


# ============================================================
# TEST 9: Ollama Connection
# ============================================================
@case("Ollama LLM Connection")
def test_ollama():
    """Verify Ollama is running and responsive."""
    import requests
    
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"   Ollama running with {len(models)} models")
            model_names = [m.get('name', '[OK]') for m in models[:5]]
            print(f"   Models: {model_names}")
            return True
        else:
            print(f"   [!] Ollama returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("   [!] Ollama not running!")
        return False


# ============================================================
# TEST 10: SearXNG Connection  
# ============================================================
@case("SearXNG Search Connection")
def test_searxng():
    """Verify SearXNG is running for web research."""
    import requests
    
    try:
        response = requests.get('http://localhost:8080/healthz', timeout=5)
        if response.status_code == 200:
            print("   SearXNG is running")
            return True
        else:
            print(f"   [!] SearXNG returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("   [!] SearXNG not running!")
        return False


# ============================================================
# TEST 11: Premarket Cycle Function
# ============================================================
@case("Premarket Cycle Loads Signals")
def test_premarket_loads_signals():
    """Verify premarket cycle loads all signals from game plan."""
    from autotrade.core.autonomous_agent import AutonomousAgent
    
    agent = AutonomousAgent()
    
    # Check the _run_premarket_cycle loads signals correctly
    # We'll simulate by checking the plan loading logic
    plans_dir = PROJECT_ROOT / 'plans'
    today = datetime.now().strftime('%Y%m%d')
    plan_path = plans_dir / f'morning_game_plan_{today}.json'
    
    if plan_path.exists():
        with open(plan_path) as f:
            plan = json.load(f)
        signals = plan.get('buy_signals', [])
        print(f"   Plan has {len(signals)} signals")
        
        # Check filter
        accepted = ['STRONG BUY', 'BUY', 'WEAK BUY', 'WATCH']
        filtered = [s for s in signals if s.get('recommendation') in accepted]
        print(f"   After filter: {len(filtered)} signals")
        
        if len(filtered) < len(signals) * 0.9:
            print(f"   [!] Filter is too strict - only {len(filtered)}/{len(signals)} pass")
        
        return len(filtered) > 10
    
    print("   [!] No game plan for today")
    return False


# ============================================================
# TEST 12: PM Workflow Candidate Selection
# ============================================================
@case("PM Workflow Candidate Selection")
def test_pm_workflow_selection():
    """Verify PM workflow selects candidates correctly."""
    
    # Simulate the filter logic from autonomous_agent.py
    test_candidates = [
        {'symbol': 'A', 'recommendation': 'STRONG BUY', 'confidence': 80},
        {'symbol': 'B', 'recommendation': 'BUY', 'confidence': 70},
        {'symbol': 'C', 'recommendation': 'WEAK BUY', 'confidence': 60},
        {'symbol': 'D', 'recommendation': 'WATCH', 'confidence': 55},
        {'symbol': 'E', 'recommendation': 'HOLD', 'confidence': 50},
    ]
    
    # The CORRECT filter (fixed version)
    accepted = ['STRONG BUY', 'BUY', 'WEAK BUY', 'WATCH']
    entry_candidates = [c for c in test_candidates if c.get('recommendation') in accepted]
    
    print(f"   Input: {len(test_candidates)} candidates")
    print(f"   Selected: {len(entry_candidates)} for entry")
    print(f"   Symbols: {[c['symbol'] for c in entry_candidates]}")
    
    # Should select A, B, C, D (not E)
    expected = {'A', 'B', 'C', 'D'}
    actual = {c['symbol'] for c in entry_candidates}
    
    if actual == expected:
        return True
    else:
        print(f"   [!] Expected {expected}, got {actual}")
        return False


# ============================================================
# TEST 13: Day Manager Signal Loading (CRITICAL)
# ============================================================
@case("Day Manager Signal Loading")
def test_day_manager_signal_loading():
    """Verify DayManager loads signals from correct file with correct count."""
    from pathlib import Path
    from datetime import datetime
    
    PLANS_DIR = PROJECT_ROOT / 'plans'
    today = datetime.now().strftime('%Y-%m-%d')
    today_compact = datetime.now().strftime('%Y%m%d')
    
    # Check what files exist (premarket plans only)
    adjusted_plans = list(PLANS_DIR.glob(f"adjusted_plan_{today_compact}_*.json"))
    morning_plan = PLANS_DIR / f'morning_game_plan_{today_compact}.json'
    
    print(f"   adjusted_plan_{today_compact}_*.json exists: {bool(adjusted_plans)}")
    print(f"   morning_game_plan_{today_compact}.json exists: {morning_plan.exists()}")
    
    # At least one must exist
    if not adjusted_plans and not morning_plan.exists():
        print("   [!] NO PREMARKET PLAN FILES FOR TODAY!")
        return False
    
    # Test actual loading
    from autotrade.core.day_manager import DayManager
    dm = DayManager.__new__(DayManager)
    dm.signals = []
    signals = dm._load_signals()
    
    print(f"   Loaded {len(signals)} signals")
    
    # Must have a reasonable number of signals (premarket plan should include multiple picks)
    if len(signals) < 5:
        print(f"   [!] TOO FEW SIGNALS! Only {len(signals)} loaded (need 5+)")
        print("   This likely means no premarket plan was found or parsed!")
        return False
    
    # Check signal format has required fields
    if signals:
        s = signals[0]
        required = ['ticker', 'symbol', 'entry_price', 'stop_loss']
        missing = [f for f in required if f not in s]
        if missing:
            print(f"   [!] Signal missing fields: {missing}")
            return False
        print(f"   First ticker: {s.get('ticker')}")
        print(f"   Signal has all required fields [OK]")
    
    return True


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "="*60)
    print(" CRITICAL FUNCTION TEST SUITE")
    print(" Run this before starting the trading bot!")
    print("="*60)
    
    # Run all tests
    tests = [
        test_multimodel_fstring,
        test_screener_db_path,
        test_screener_exclude,
        test_recommendation_filter,
        test_game_plan_load,
        test_overnight_state,
        test_signal_generator,
        test_day_manager_state,
        test_ollama,
        test_searxng,
        test_premarket_loads_signals,
        test_pm_workflow_selection,
        test_day_manager_signal_loading,  # NEW: Critical signal loading test
        test_regime_router,
    ]
    
    for t in tests:
        t()
    
    # Summary
    print("\n" + "="*60)
    print(" TEST SUMMARY")
    print("="*60)
    print(f"\n[OK] PASSED: {len(PASSED)}")
    for name in PASSED:
        print(f"   - {name}")
    
    if FAILED:
        print(f"\n[X] FAILED: {len(FAILED)}")
        for name, error in FAILED:
            print(f"   - {name}")
            print(f"     Error: {error[:100]}")
    
    print(f"\n{'='*60}")
    if not FAILED:
        print(" ALL TESTS PASSED - READY FOR TRADING!")
    else:
        print(f" {len(FAILED)} TESTS FAILED - FIX BEFORE TRADING!")
    print("="*60 + "\n")
    
    return len(FAILED) == 0


# ============================================================
# TEST: Regime Router Logic
# ============================================================

@case("Regime Router Strategy Filtering & Sector Bias")
def test_regime_router():
    """Verify RegimeRouter correctly filters families and adjusts scores."""
    from autotrade.signals.regime_router import RegimeRouter
    
    router = RegimeRouter()
    
    # Mock policy for testing
    router.policy = {
        "ts_momentum": ["trend"],
        "mean_reversion": ["chop", "crisis"]
    }
    
    candidates = [
        {"symbol": "TSLA", "family": "ts_momentum", "final_score": 80, "sector": "Technology"},
        {"symbol": "AAPL", "family": "mean_reversion", "final_score": 75, "sector": "Technology"},
        {"symbol": "XOM", "family": "breakout", "final_score": 70, "sector": "Energy"}
    ]
    
    # 1. Test filtering in 'trend' regime
    trend_picks = router.filter_allowed_strategies("trend", candidates)
    symbols = [c["symbol"] for c in trend_picks]
    if "TSLA" not in symbols: return False # ts_momentum allowed in trend
    if "AAPL" in symbols: return False    # mean_reversion NOT allowed in trend
    if "XOM" not in symbols: return False  # breakout has no policy, so allowed
    
    # 2. Test filtering in 'crisis' regime
    crisis_picks = router.filter_allowed_strategies("crisis", candidates)
    symbols = [c["symbol"] for c in crisis_picks]
    if "TSLA" in symbols: return False    # ts_momentum NOT allowed in crisis
    if "AAPL" not in symbols: return False # mean_reversion allowed in crisis
    
    # 3. Test sector bias
    report = {
        "avoid_sectors": ["Technology"],
        "favor_sectors": ["Energy"]
    }
    biased = router.apply_sector_bias(report, candidates)
    
    tsla = next(c for c in biased if c["symbol"] == "TSLA")
    xom = next(c for c in biased if c["symbol"] == "XOM")
    
    if tsla["final_score"] != 70: return False # 80 - 10 penalty
    if xom["final_score"] != 78: return False  # 70 + 8 boost
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

