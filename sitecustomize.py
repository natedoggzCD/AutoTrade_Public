"""
AutoTrade sitecustomize
Installs the agentic exception handler for all project scripts and forces UTF-8 encoding.
"""
import sys
import os
import io

# Force UTF-8 encoding for standard streams (Windows specific)
if sys.platform == 'win32':
    try:
        # Reconfigure standard streams if possible (Python 3.7+)
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stdin, 'reconfigure'):
            sys.stdin.reconfigure(encoding='utf-8', errors='replace')
        
        # Also set the environment variable for subprocesses
        os.environ['PYTHONUTF8'] = '1'
    except Exception:
        # Fallback for environments where reconfigure is not available or fails
        pass

try:
    from autotrade.utils.agentic_exceptions import install as _install_agentic_exceptions
    _install_agentic_exceptions()
except Exception:
    # Never block startup; fall back to default excepthook
    pass

try:
    from autotrade.utils.yfinance_404_filter import install as _install_yf404_filter
    _install_yf404_filter()
except Exception:
    pass
