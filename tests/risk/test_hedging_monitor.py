from __future__ import annotations

from types import SimpleNamespace

from autotrade.risk.hedging_monitor import HedgingTriggerMonitor


def test_hedging_trigger_monitor_flags_combined_risk_off_conditions():
    monitor = HedgingTriggerMonitor()

    snapshot = monitor.evaluate(
        vix_level=30.0,
        breadth_ratio=0.72,
        new_highs=10,
        new_lows=55,
        market_regime="crisis",
    )

    assert snapshot.should_hedge is True
    assert snapshot.risk_score >= 3.0
    assert "vix_spike" in snapshot.triggers
    assert "breadth_divergence" in snapshot.triggers
    assert "new_lows_expanding" in snapshot.triggers
    assert "bearish_regime" in snapshot.triggers


def test_hedging_trigger_monitor_stays_idle_in_neutral_conditions():
    monitor = HedgingTriggerMonitor()

    snapshot = monitor.evaluate(
        vix_level=17.0,
        breadth_ratio=1.15,
        new_highs=40,
        new_lows=12,
        market_regime="neutral",
    )

    assert snapshot.should_hedge is False
    assert snapshot.risk_score == 0.0
    assert snapshot.triggers == ()


def test_hedging_trigger_monitor_allows_partial_trigger_threshold_override():
    monitor = HedgingTriggerMonitor(risk_score_trigger=1.0)

    snapshot = monitor.evaluate(
        vix_level=27.0,
        breadth_ratio=1.0,
        new_highs=30,
        new_lows=20,
        market_regime="neutral",
    )

    assert snapshot.should_hedge is True
    assert snapshot.triggers == ("vix_spike",)


def test_hedging_trigger_monitor_reads_vix_from_market_data_provider():
    cache = SimpleNamespace(get_vix_level=lambda: {"current": 28.4, "regime": "elevated"})
    monitor = HedgingTriggerMonitor(market_data_cache=cache)

    assert monitor.check_vix_spike() is True


def test_hedging_trigger_monitor_check_vix_spike_thresholds_explicit_value():
    monitor = HedgingTriggerMonitor(vix_spike_threshold=24.0)

    assert monitor.check_vix_spike(25.5) is True
    assert monitor.check_vix_spike(19.0) is False


def test_hedging_trigger_monitor_flags_breadth_divergence_from_ad_ratio():
    monitor = HedgingTriggerMonitor(breadth_ratio_floor=0.9)

    assert monitor.check_breadth_divergence(
        breadth_ratio=0.78,
        new_highs=35,
        new_lows=40,
    ) is True


def test_hedging_trigger_monitor_flags_expanding_new_lows():
    monitor = HedgingTriggerMonitor(new_low_excess_threshold=20)

    snapshot = monitor.get_breadth_snapshot(
        breadth_ratio=1.05,
        new_highs=10,
        new_lows=35,
    )

    assert "new_lows_expanding" in snapshot.triggers
    assert snapshot.risk_score == 1.0


def test_hedging_trigger_monitor_ignores_healthy_breadth():
    monitor = HedgingTriggerMonitor()

    snapshot = monitor.get_breadth_snapshot(
        breadth_ratio=1.12,
        new_highs=45,
        new_lows=12,
    )

    assert snapshot.triggers == ()
    assert snapshot.risk_score == 0.0
