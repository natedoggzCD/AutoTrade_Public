
try:
    from autotrade.utils.agentic_exceptions import install as _install_agentic_exceptions
    _install_agentic_exceptions()
except Exception:
    # Never block import; fall back to default excepthook
    pass
