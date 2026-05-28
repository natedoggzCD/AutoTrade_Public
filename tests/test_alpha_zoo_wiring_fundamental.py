import pytest
from autotrade.signals.registry import register_signal_zoo
from autotrade.signals.contracts import SignalFamily

def test_fundamental_signals_registered():
    registry = register_signal_zoo(include_baselines=False)
    models = registry.get_all_models()
    
    names = [m.name for m in models]
    assert "PEADAlphaSource_v1" in names
    assert "SqueezeAlphaSource_v1" in names
    assert "InverseETFAlphaSource_v1" in names
    
    # Check if they are enabled by default
    assert registry.is_enabled("PEADAlphaSource_v1")
    assert registry.is_enabled("SqueezeAlphaSource_v1")
    assert registry.is_enabled("InverseETFAlphaSource_v1")

def test_fundamental_signals_in_ts_momentum_family():
    registry = register_signal_zoo(include_baselines=False)
    models = registry.get_models_by_family(SignalFamily.TS_MOMENTUM)
    
    names = [m.name for m in models]
    assert "PEADAlphaSource_v1" in names
    assert "SqueezeAlphaSource_v1" in names
    assert "InverseETFAlphaSource_v1" in names
