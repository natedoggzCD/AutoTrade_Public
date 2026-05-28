from collections import defaultdict

from tools.session_hygiene_check import categorize, strict_failures


def test_runtime_artifacts_are_classified_as_local_artifacts():
    assert categorize(".claude/scheduled_tasks.lock") == "local_artifact"
    assert categorize("data/delisted_symbols.json") == "local_artifact"
    assert categorize("day_manager_state.bak.json") == "local_artifact"


def test_staged_local_artifact_deletions_are_allowed():
    buckets = defaultdict(list)
    buckets["local_artifact"].append(
        {"status": "D", "path": "data/delisted_symbols.json"}
    )

    assert strict_failures(buckets) == []
