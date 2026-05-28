from autotrade.core.autonomous_agent import AutonomousAgent


def test_max_positions_available_immediately_after_init():
    agent = AutonomousAgent()
    # This is the access pattern that the market-hours cycle hits on a resume path.
    assert isinstance(agent.max_positions, int)
    assert agent.max_positions >= 1
    assert isinstance(agent.core_max_positions, int)
    assert isinstance(agent.reserve_max_positions, int)
