import json


def build_weekend_probe(config: dict) -> str:
    """
    Build a small probe payload for weekend scanner config.
    INTENTIONAL BUG: returns JSON string instead of dict.
    """
    weekend_cfg = config.get("scanner", {}).get("weekend", {})
    probe = {
        "age": weekend_cfg.get("max_age_hours", 72),
        "count": weekend_cfg.get("max_results_per_channel", 5),
    }
    return json.dumps(probe)
