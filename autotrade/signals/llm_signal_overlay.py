"""LLM signal overlay stub - no LLM in public backtesting distribution. Functions are no-ops."""


def build_signal_packet(*_a, **_kw):
    return {}


def normalize_overlay_label(label, *_a, **_kw):
    return label


def query_local_signal_classifier(*_a, **_kw):
    return None

