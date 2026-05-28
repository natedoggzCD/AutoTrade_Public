"""
Test financial DB workflow wiring.

Tests that:
1. overnight_agent.py imports and uses check functions
2. premarket_agent.py imports and uses check functions  
3. Financial checks are applied correctly without errors
"""

import sys
import logging
from pathlib import Path

# Setup paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_overnight_agent_imports():
    """Test that overnight_agent imports financial checks."""
    logger.info("TEST 1: overnight_agent imports...")
    try:
        from autotrade.core.overnight_agent import (
            check_earnings_risk,
            check_balance_sheet_health,
            check_cash_flow_health,
            check_valuation_sanity,
            check_options_positioning,
            OvernightAgent
        )
        logger.info("  ✓ All imports successful")
        assert True
    except ImportError as e:
        logger.error(f"  ✗ Import failed: {e}")
        assert False


def test_premarket_agent_imports():
    """Test that premarket_agent imports financial checks."""
    logger.info("TEST 2: premarket_agent imports...")
    try:
        from autotrade.core.premarket_agent import (
            flag_earnings_today,
            check_balance_sheet_health,
            check_cash_flow_health,
            check_valuation_sanity,
            check_options_positioning,
            PreMarketAgent
        )
        logger.info("  ✓ All imports successful")
        assert True
    except ImportError as e:
        logger.error(f"  ✗ Import failed: {e}")
        assert False


def test_overnight_agent_scan_watchlist():
    """Test that scan_watchlist has financial DB integration."""
    logger.info("TEST 3: overnight_agent.scan_watchlist() has financial checks...")
    try:
        from autotrade.core.overnight_agent import OvernightAgent
        import inspect
        
        agent = OvernightAgent()
        source = inspect.getsource(agent.scan_watchlist)
        
        # Check for key phrases
        checks = [
            ('check_earnings_risk', 'Checks for earnings risk'),
            ('check_balance_sheet_health', 'Checks balance sheet'),
            ('check_cash_flow_health', 'Checks cash flow'),
            ('check_valuation_sanity', 'Checks valuation'),
            ('check_options_positioning', 'Checks options'),
        ]
        
        found = []
        missing = []
        for func_name, desc in checks:
            if func_name in source:
                found.append(f"✓ {desc}")
            else:
                missing.append(f"✗ {desc}")
        
        for item in found:
            logger.info(f"  {item}")
        for item in missing:
            logger.warning(f"  {item}")
        
        assert len(missing) == 0
        
    except Exception as e:
        logger.error(f"  ✗ Error: {e}")
        assert False


def test_premarket_agent_analyze_candidate():
    """Test that analyze_candidate has financial DB integration."""
    logger.info("TEST 4: premarket_agent.analyze_candidate() has financial checks...")
    try:
        from autotrade.core.premarket_agent import PreMarketAgent
        import inspect
        
        source = inspect.getsource(PreMarketAgent.analyze_candidate)
        
        # Check for key phrases
        checks = [
            ('flag_earnings_today', 'Flags same-day earnings'),
            ('check_balance_sheet_health', 'Checks balance sheet'),
            ('check_cash_flow_health', 'Checks cash flow'),
            ('[EARNINGS]', 'Earnings alert logging'),
            ('[BALANCE SHEET]', 'Balance sheet alert logging'),
        ]
        
        found = []
        missing = []
        for check_str, desc in checks:
            if check_str in source:
                found.append(f"✓ {desc}")
            else:
                missing.append(f"✗ {desc}")
        
        for item in found:
            logger.info(f"  {item}")
        for item in missing:
            logger.warning(f"  {item}")
        
        assert len(missing) == 0
        
    except Exception as e:
        logger.error(f"  ✗ Error: {e}")
        assert False


def test_premarket_agent_check_holdings():
    """Test that premarket_agent has check_held_positions_earnings method."""
    logger.info("TEST 5: premarket_agent.check_held_positions_earnings() exists...")
    try:
        from autotrade.core.premarket_agent import PreMarketAgent
        
        agent = PreMarketAgent()
        assert hasattr(agent, "check_held_positions_earnings")
        logger.info("  ✓ Method exists")

        # Test the method signature
        import inspect
        sig = inspect.signature(agent.check_held_positions_earnings)
        params = list(sig.parameters.keys())

        assert "holdings" in params
        logger.info("  ✓ Accepts 'holdings' parameter")
    except Exception as e:
        logger.error(f"  ✗ Error: {e}")
        assert False


def test_autonomous_agent_check_functions():
    """Test that autonomous_agent has all check functions."""
    logger.info("TEST 6: autonomous_agent has all check functions...")
    try:
        from autotrade.analysis.financial_checks import (
            check_earnings_risk,
            flag_earnings_today,
            check_balance_sheet_health,
            check_cash_flow_health,
            check_dividend_stability,
            check_valuation_sanity,
            check_options_positioning,
        )

        functions = [
            ("check_earnings_risk", check_earnings_risk),
            ("flag_earnings_today", flag_earnings_today),
            ("check_balance_sheet_health", check_balance_sheet_health),
            ("check_cash_flow_health", check_cash_flow_health),
            ("check_dividend_stability", check_dividend_stability),
            ("check_valuation_sanity", check_valuation_sanity),
            ("check_options_positioning", check_options_positioning),
        ]

        for name, func in functions:
            assert callable(func)
            logger.info(f"  ✓ {name}")

    except ImportError as e:
        logger.error(f"  ✗ Import failed: {e}")
        assert False
    except Exception as e:
        logger.error(f"  ✗ Error: {e}")
        assert False


def main():
    """Run all tests."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINANCIAL DB WORKFLOW WIRING - TEST SUITE")
    logger.info("=" * 70)
    logger.info("")
    
    tests = [
        test_autonomous_agent_check_functions,
        test_overnight_agent_imports,
        test_premarket_agent_imports,
        test_overnight_agent_scan_watchlist,
        test_premarket_agent_analyze_candidate,
        test_premarket_agent_check_holdings,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            logger.error(f"Test {test_func.__name__} crashed: {e}")
            results.append(False)
        logger.info("")
    
    # Summary
    logger.info("=" * 70)
    passed = sum(results)
    total = len(results)
    logger.info(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("✓ ALL TESTS PASSED")
        return 0
    else:
        logger.warning(f"✗ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit(main())












