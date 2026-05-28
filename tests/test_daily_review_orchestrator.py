from __future__ import annotations
import pytest
from unittest.mock import MagicMock
from autotrade.reasoning.daily_review_orchestrator import DailyReviewOrchestrator

def test_daily_review_orchestrator_initialization():
    orchestrator = DailyReviewOrchestrator()
    assert orchestrator is not None

def test_synthesize_lessons():
    # Mock logs
    logs = {
        "order_1": {
            "symbol": "AAPL",
            "arrival_price": 150.0,
            "events": {
                "submission": "2026-03-01 10:00:00",
                "fill": "2026-03-01 10:01:00"
            }
        }
    }
    
    # Mock LLM advisor
    mock_advisor = MagicMock()
    mock_advisor.get_advice.return_value = {
        "lessons": ["Execution was slightly delayed", "Slippage within acceptable bounds"],
        "sentiment": "positive"
    }
    
    orchestrator = DailyReviewOrchestrator(advisor=mock_advisor)
    lessons = orchestrator.synthesize_lessons(logs)
    
    assert len(lessons) > 0
    assert "Execution was slightly delayed" in lessons
    assert mock_advisor.get_advice.called
