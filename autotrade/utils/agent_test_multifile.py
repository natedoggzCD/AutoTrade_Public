from autotrade.utils.agent_test_helpers import build_weekend_probe


def run_multifile_test(config: dict) -> None:
    """
    Intentional multi-file error for agent validation.
    """
    probe = build_weekend_probe(config)
    if isinstance(probe, dict) and probe["age"] > 0:  # TypeError: string indices must be integers
        pass
