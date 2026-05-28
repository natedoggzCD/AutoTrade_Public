import pytest
from autotrade.utils.news_aggregator import NewsAggregator
from autotrade.utils.news_cache import get_cache

def test_news_refresh_batch():
    agg = NewsAggregator()
    cache = get_cache()
    
    test_symbols = ["AAPL", "TSLA", "NVDA"]
    
    # Run batch refresh (dry-ish run, will fetch if stale)
    results = agg.refresh_batch(test_symbols, max_age_hours=1)
    
    assert results["total"] == len(test_symbols)
    assert "details" in results
    
    # Second run immediately should hit cache
    results_cached = agg.refresh_batch(test_symbols, max_age_hours=1)
    assert results_cached["refreshed"] == 0
    assert results_cached["total"] == len(test_symbols)
