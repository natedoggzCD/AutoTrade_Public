import inspect

from autotrade.core.autonomous_agent import AutonomousAgent, TomorrowsPlanGenerator


def test_get_current_positions_is_on_plan_generator_not_agent():
    assert hasattr(TomorrowsPlanGenerator, "get_current_positions")
    assert not hasattr(AutonomousAgent, "get_current_positions"), (
        "AutonomousAgent must not define get_current_positions; call it via self.plan_generator."
    )


def test_no_bare_self_get_current_positions_inside_autonomous_agent():
    src = inspect.getsource(AutonomousAgent)
    assert "self.get_current_positions()" not in src, (
        "Found bare self.get_current_positions() call inside AutonomousAgent; "
        "must route through self.plan_generator.get_current_positions()."
    )
