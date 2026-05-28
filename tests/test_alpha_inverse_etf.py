
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pandas as pd

from autotrade.signals.contracts import (
    SignalContext,
    SignalAction,
    SignalFamily,
    RegimeLabel,
)
from autotrade.signals.alpha.inverse_etf import InverseETFAlphaSource

class TestInverseETFAlphaSource:
    def test_name_property(self):
        source = InverseETFAlphaSource()
        assert source.name == "InverseETFAlphaSource_v1"

    def test_family_property(self):
        source = InverseETFAlphaSource()
        # In the new implementation, we'll use a specific HEDGE family if possible, 
        # or stick to TS_MOMENTUM for now as it was in the original.
        assert source.family == SignalFamily.TS_MOMENTUM

    @patch("autotrade.signals.alpha.inverse_etf.get_config")
    def test_generate_signals_crisis(self, mock_get_config):
        # Setup mock config
        mock_config = MagicMock()
        mock_config.inverse_etf_hedging.enabled = True
        mock_config.inverse_etf_hedging.allowed_regimes = ["crisis", "bear"]
        mock_config.inverse_etf_hedging.symbols = ["SH", "PSQ"]
        mock_config.inverse_etf_hedging.max_hold_days = 3
        mock_get_config.return_value = mock_config

        source = InverseETFAlphaSource()
        
        # Create a context with CRISIS regime
        context = SignalContext(
            tickers=[],
            price_data=pd.DataFrame(),
            regime=RegimeLabel.CRISIS
        )

        decisions = source.generate(context)

        assert len(decisions) == 2
        assert {d.ticker for d in decisions} == {"SH", "PSQ"}
        assert decisions[0].action == SignalAction.BUY
        assert "HEDGE" in decisions[0].reason

    @patch("autotrade.signals.alpha.inverse_etf.get_config")
    def test_generate_no_signals_trend(self, mock_get_config):
        mock_config = MagicMock()
        mock_config.inverse_etf_hedging.enabled = True
        mock_config.inverse_etf_hedging.allowed_regimes = ["crisis"]
        mock_get_config.return_value = mock_config

        source = InverseETFAlphaSource()
        context = SignalContext(tickers=[], price_data=pd.DataFrame(), regime=RegimeLabel.TREND)
        
        decisions = source.generate(context)
        assert len(decisions) == 0

    @patch("autotrade.signals.alpha.inverse_etf.get_config")
    def test_generate_disabled(self, mock_get_config):
        mock_config = MagicMock()
        mock_config.inverse_etf_hedging.enabled = False
        mock_get_config.return_value = mock_config

        source = InverseETFAlphaSource()
        context = SignalContext(tickers=[], price_data=pd.DataFrame(), regime=RegimeLabel.CRISIS)
        
        decisions = source.generate(context)
        assert len(decisions) == 0


@patch("autotrade.signals.alpha.inverse_etf.get_config")
def test_legacy_inverse_etf_wrapper_supports_old_runtime_import(mock_get_config):
    from autotrade.signals.alpha_inverse_etf import (
        InverseETFAlphaSource as LegacyInverseETFAlphaSource,
    )

    mock_config = MagicMock()
    mock_config.inverse_etf_hedging.enabled = True
    mock_config.inverse_etf_hedging.allowed_regimes = ["crisis", "bear"]
    mock_config.inverse_etf_hedging.symbols = ["SH", "PSQ"]
    mock_config.inverse_etf_hedging.max_hold_days = 3
    mock_get_config.return_value = mock_config

    source = LegacyInverseETFAlphaSource(parent_logger=MagicMock())
    decisions = source.generate_signals(RegimeLabel.CRISIS)

    assert {d.ticker for d in decisions} == {"SH", "PSQ"}
    assert all(d.action == SignalAction.BUY for d in decisions)


def test_inverse_fast_screener_scores_first_open_bar():
    from autotrade.signals.inverse_etf_screener import InverseETFScreener

    screener = InverseETFScreener(financial_db=MagicMock())
    bars = pd.DataFrame(
        {
            "open": [54.08],
            "high": [54.61],
            "low": [54.08],
            "close": [54.53],
            "volume": [9_084_471],
        }
    )
    screener._fetch_intraday_bars = lambda ticker: bars

    result = screener._score_candidate(
        {
            "ticker": "SQQQ",
            "name": "ProShares UltraPro Short QQQ",
            "leverage": 3,
            "underlying": "QQQ",
            "category": "index",
            "avg_daily_volume": 100_000_000,
            "aum_millions": 1_000,
        },
        entry_mode="inverse_fast",
        minutes_since_open=1,
    )

    assert result is not None
    assert result["signal"] == "ENTRY"


def test_inverse_fast_screener_bypasses_regime_precheck():
    from autotrade.signals.inverse_etf_screener import InverseETFScreener

    screener = InverseETFScreener(financial_db=MagicMock())
    screener._gate1_liquidity = lambda: [
        {
            "ticker": "SQQQ",
            "name": "ProShares UltraPro Short QQQ",
            "leverage": 3,
            "underlying": "QQQ",
            "category": "index",
            "avg_daily_volume": 100_000_000,
            "aum_millions": 1_000,
        }
    ]
    screener._score_candidate = lambda etf, **kwargs: {
        "ticker": etf["ticker"],
        "signal": "ENTRY",
        "entry_price": 54.53,
        "composite_score": 90,
    }

    results = screener.screen_universe(
        regime="RegimeLabel.NEUTRAL",
        portfolio_holdings=[],
        entry_mode="inverse_fast",
        minutes_since_open=1,
    )

    assert results[0]["ticker"] == "SQQQ"
    assert results[0]["signal"] == "ENTRY"


@patch("autotrade.risk.inverse_etf_manager.get_config")
@patch("autotrade.risk.inverse_etf_manager.MarketRegimeDetector")
def test_inverse_etf_manager_selects_index_matched_hedge(mock_regime_detector, mock_get_config):
    from autotrade.risk.inverse_etf_manager import InverseETFManager

    mock_config = MagicMock()
    mock_config.inverse_etf_hedging.symbols = ["SH", "PSQ", "RWM"]
    mock_get_config.return_value = mock_config
    mock_regime_detector.return_value = MagicMock()

    manager = InverseETFManager()

    assert manager.select_hedge_instrument(primary_index="QQQ") == "PSQ"
    assert manager.select_hedge_instrument(primary_index="IWM") == "RWM"


@patch("autotrade.risk.inverse_etf_manager.get_config")
@patch("autotrade.risk.inverse_etf_manager.MarketRegimeDetector")
def test_inverse_etf_manager_calculates_hedge_size_capped_by_max_allocation(mock_regime_detector, mock_get_config):
    from autotrade.risk.inverse_etf_manager import InverseETFManager

    hedge_cfg = MagicMock()
    hedge_cfg.max_allocation_pct = 20.0
    mock_config = MagicMock()
    mock_config.inverse_etf_hedging = hedge_cfg
    mock_get_config.return_value = mock_config
    mock_regime_detector.return_value = MagicMock()

    manager = InverseETFManager()

    size = manager.calculate_hedge_size(
        portfolio_value=100000.0,
        net_exposure_pct=50.0,
        hedge_symbol="SH",
    )

    assert size == 20000.0


@patch("autotrade.risk.inverse_etf_manager.get_config")
@patch("autotrade.risk.inverse_etf_manager.MarketRegimeDetector")
def test_inverse_etf_manager_leveraged_hedge_reduces_required_notional(mock_regime_detector, mock_get_config):
    from autotrade.analysis.market_regime import MarketRegime
    from autotrade.risk.inverse_etf_manager import InverseETFManager

    mock_config = MagicMock()
    mock_config.inverse_etf_hedging.symbols = ["SQQQ", "SH"]
    mock_config.inverse_etf_hedging.max_allocation_pct = 50.0
    mock_get_config.return_value = mock_config
    mock_regime_detector.return_value = MagicMock()

    manager = InverseETFManager()

    assert manager.select_hedge_instrument(primary_index="QQQ", regime=MarketRegime.CRASH) == "SQQQ"
    size = manager.calculate_hedge_size(
        portfolio_value=90000.0,
        net_exposure_pct=45.0,
        hedge_symbol="SQQQ",
    )
    assert size == 13500.0


@patch("autotrade.risk.inverse_etf_manager.get_config")
@patch("autotrade.risk.inverse_etf_manager.MarketRegimeDetector")
def test_inverse_etf_manager_rebalance_increase_when_underhedged(mock_regime_detector, mock_get_config):
    from autotrade.risk.inverse_etf_manager import InverseETFManager

    mock_config = MagicMock()
    mock_config.inverse_etf_hedging.symbols = ["SH", "PSQ", "RWM"]
    mock_config.inverse_etf_hedging.max_allocation_pct = 20.0
    mock_get_config.return_value = mock_config
    mock_regime_detector.return_value = MagicMock()

    positions = [
        MagicMock(symbol="AAPL", market_value=70000.0),
        MagicMock(symbol="MSFT", market_value=30000.0),
    ]

    manager = InverseETFManager()
    plan = manager.rebalance_hedge(
        portfolio_value=100000.0,
        positions=positions,
        net_exposure_pct=40.0,
        primary_index="SPY",
    )

    assert plan["hedge_symbol"] == "SH"
    assert plan["target_notional"] == 20000.0
    assert plan["action"] == "increase"


@patch("autotrade.risk.inverse_etf_manager.get_config")
@patch("autotrade.risk.inverse_etf_manager.MarketRegimeDetector")
def test_inverse_etf_manager_rebalance_decrease_when_overhedged(mock_regime_detector, mock_get_config):
    from autotrade.risk.inverse_etf_manager import InverseETFManager

    mock_config = MagicMock()
    mock_config.inverse_etf_hedging.symbols = ["SH", "PSQ", "RWM"]
    mock_config.inverse_etf_hedging.max_allocation_pct = 20.0
    mock_get_config.return_value = mock_config
    mock_regime_detector.return_value = MagicMock()

    positions = [
        MagicMock(symbol="AAPL", market_value=70000.0),
        MagicMock(symbol="MSFT", market_value=30000.0),
        MagicMock(symbol="SH", market_value=-20000.0),
    ]

    manager = InverseETFManager()
    plan = manager.rebalance_hedge(
        portfolio_value=100000.0,
        positions=positions,
        net_exposure_pct=10.0,
        primary_index="SPY",
    )

    assert plan["current_allocation_pct"] > plan["target_allocation_pct"]
    assert plan["action"] == "decrease"


@patch("autotrade.risk.inverse_etf_manager.get_config")
@patch("autotrade.risk.inverse_etf_manager.MarketRegimeDetector")
def test_inverse_etf_manager_rebalance_holds_when_near_target(mock_regime_detector, mock_get_config):
    from autotrade.risk.inverse_etf_manager import InverseETFManager

    mock_config = MagicMock()
    mock_config.inverse_etf_hedging.symbols = ["SH"]
    mock_config.inverse_etf_hedging.max_allocation_pct = 20.0
    mock_get_config.return_value = mock_config
    mock_regime_detector.return_value = MagicMock()

    positions = [
        MagicMock(symbol="AAPL", market_value=80000.0),
        MagicMock(symbol="SH", market_value=-19900.0),
    ]

    manager = InverseETFManager()
    plan = manager.rebalance_hedge(
        portfolio_value=100000.0,
        positions=positions,
        net_exposure_pct=20.0,
        primary_index="SPY",
    )

    assert abs(plan["rebalance_delta_pct"]) < 0.25


@patch("autotrade.risk.inverse_etf_manager.get_config")
@patch("autotrade.risk.inverse_etf_manager.MarketRegimeDetector")
def test_check_hedge_conditions_suppresses_entry_when_spy_is_strong_green(
    mock_regime_detector, mock_get_config
):
    from autotrade.analysis.market_regime import MarketRegime
    from autotrade.risk.inverse_etf_manager import InverseETFManager

    hedge_cfg = MagicMock()
    hedge_cfg.enabled = True
    hedge_cfg.allowed_regimes = ["crisis", "risk_off"]
    hedge_cfg.entry_triggers = SimpleNamespace(vix_above=25.0, consecutive_red_days=3)
    mock_config = MagicMock()
    mock_config.inverse_etf_hedging = hedge_cfg
    mock_get_config.return_value = mock_config
    mock_regime_detector.return_value = MagicMock(
        detect_regime=lambda use_cache=True: SimpleNamespace(regime=MarketRegime.SELLOFF)
    )

    manager = InverseETFManager()
    manager._count_consecutive_red_days = lambda: 0
    manager._get_previous_close = lambda symbol: 100.0

    result = manager.check_hedge_conditions(vix_level=30.0, spy_price=101.0, spy_sma20=99.0)

    assert result["should_enter"] is False
    assert "suppresses inverse entry" in result["reason"]


@patch("autotrade.risk.inverse_etf_manager.get_config")
@patch("autotrade.risk.inverse_etf_manager.MarketRegimeDetector")
def test_inverse_etf_manager_exits_on_regime_recovery_and_vix_mean_reversion(mock_regime_detector, mock_get_config):
    from autotrade.risk.inverse_etf_manager import InverseETFManager

    hedge_cfg = MagicMock()
    hedge_cfg.allowed_regimes = ["crisis", "risk_off"]
    hedge_cfg.exit_triggers.vix_below = 18.0
    hedge_cfg.stop_loss_pct = -8.0
    mock_config = MagicMock()
    mock_config.inverse_etf_hedging = hedge_cfg
    mock_get_config.return_value = mock_config
    mock_regime_detector.return_value = MagicMock()

    manager = InverseETFManager()
    decision = manager.check_hedge_exit(
        current_regime="neutral",
        vix_level=16.5,
        hedge_return_pct=2.0,
    )

    assert decision["should_exit"] is True
    assert "regime_recovery" in decision["reasons"]
    assert "volatility_mean_reversion" in decision["reasons"]


@patch("autotrade.risk.inverse_etf_manager.get_config")
@patch("autotrade.risk.inverse_etf_manager.MarketRegimeDetector")
def test_inverse_etf_manager_exits_on_hedge_stop_loss(mock_regime_detector, mock_get_config):
    from autotrade.risk.inverse_etf_manager import InverseETFManager

    hedge_cfg = MagicMock()
    hedge_cfg.allowed_regimes = ["crisis"]
    hedge_cfg.exit_triggers.vix_below = 18.0
    hedge_cfg.stop_loss_pct = -6.0
    mock_config = MagicMock()
    mock_config.inverse_etf_hedging = hedge_cfg
    mock_get_config.return_value = mock_config
    mock_regime_detector.return_value = MagicMock()

    manager = InverseETFManager()
    stop = manager.manage_hedge_stop_loss(hedge_return_pct=-7.5)
    decision = manager.check_hedge_exit(
        current_regime="crisis",
        vix_level=25.0,
        hedge_return_pct=-7.5,
    )

    assert stop["should_exit"] is True
    assert stop["reason"] == "hedge_stop_loss"
    assert decision["should_exit"] is True
    assert "hedge_stop_loss" in decision["reasons"]


@patch("autotrade.risk.inverse_etf_manager.get_config")
@patch("autotrade.risk.inverse_etf_manager.MarketRegimeDetector")
def test_inverse_etf_manager_keeps_hedge_when_risk_off_conditions_persist(mock_regime_detector, mock_get_config):
    from autotrade.risk.inverse_etf_manager import InverseETFManager

    hedge_cfg = MagicMock()
    hedge_cfg.allowed_regimes = ["crisis", "risk_off"]
    hedge_cfg.exit_triggers.vix_below = 18.0
    hedge_cfg.stop_loss_pct = -8.0
    hedge_cfg.profit_target_pct = 10.0  # Set explicitly so profit target doesn't fire
    mock_config = MagicMock()
    mock_config.inverse_etf_hedging = hedge_cfg
    mock_get_config.return_value = mock_config
    mock_regime_detector.return_value = MagicMock()

    manager = InverseETFManager()
    decision = manager.check_hedge_exit(
        current_regime="crisis",
        vix_level=27.0,
        hedge_return_pct=1.0,
    )

    assert decision["should_exit"] is False
    assert decision["reasons"] == []


@patch("autotrade.risk.inverse_etf_manager.get_config")
@patch("autotrade.risk.inverse_etf_manager.MarketRegimeDetector")
def test_inverse_etf_manager_profiles_are_stricter_for_3x(mock_regime_detector, mock_get_config):
    from autotrade.risk.inverse_etf_manager import InverseETFManager

    mock_config = MagicMock()
    mock_config.inverse_etf_hedging.max_hold_days = 4
    mock_config.inverse_etf_hedging.leveraged_max_hold_days = 2
    mock_get_config.return_value = mock_config
    mock_regime_detector.return_value = MagicMock()

    manager = InverseETFManager()

    one_x = manager.get_instrument_profile("SH")
    three_x = manager.get_instrument_profile("SQQQ")

    assert three_x["leverage"] == 3
    assert three_x["entry_size_multiplier"] < one_x["entry_size_multiplier"]
    assert three_x["profit_target_pct"] < one_x["profit_target_pct"]
    assert three_x["stop_loss_pct"] > one_x["stop_loss_pct"]
    assert three_x["scale_out_fraction"] > one_x["scale_out_fraction"]
    assert three_x["max_hold_days"] < one_x["max_hold_days"]


@patch("autotrade.risk.inverse_etf_manager.get_config")
@patch("autotrade.risk.inverse_etf_manager.MarketRegimeDetector")
def test_inverse_etf_manager_intraday_reversal_is_stricter_for_3x(mock_regime_detector, mock_get_config):
    from autotrade.risk.inverse_etf_manager import InverseETFManager

    mock_config = MagicMock()
    mock_get_config.return_value = mock_config
    mock_regime_detector.return_value = MagicMock()

    manager = InverseETFManager()
    bars = pd.DataFrame(
        {
            "high": [10.00, 10.02, 10.04, 10.14, 10.02],
            "low": [9.96, 9.98, 10.00, 10.08, 9.96],
            "close": [9.98, 10.00, 10.02, 10.12, 10.00],
            "volume": [100_000, 110_000, 120_000, 130_000, 140_000],
        }
    )

    one_x = manager.evaluate_intraday_reversal("SH", bars=bars)
    three_x = manager.evaluate_intraday_reversal("SQQQ", bars=bars)

    assert one_x["should_exit"] is False
    assert three_x["should_exit"] is True
    assert "price_below_sma5" in three_x["reasons"]
