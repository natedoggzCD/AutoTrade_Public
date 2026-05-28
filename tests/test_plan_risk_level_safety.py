from autotrade.core.autonomous_agent import TomorrowsPlanGenerator


class _Logger:
    def __init__(self):
        self.warnings = []

    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        self.warnings.append((args, kwargs))


def test_convert_signals_repairs_inverted_long_bracket_levels():
    generator = TomorrowsPlanGenerator.__new__(TomorrowsPlanGenerator)
    generator.logger = _Logger()
    generator.max_positions = 5
    generator.position_size_target = 1000.0
    generator.get_current_positions = lambda: []
    generator.get_account_info = lambda: {"buying_power": 10000.0}
    generator._plan_entry_constraints = lambda plan_data: {"max_positions": 5}
    generator._coerce_float = TomorrowsPlanGenerator._coerce_float.__get__(
        generator, TomorrowsPlanGenerator
    )

    converted = generator._convert_signals_to_plan(
        {
            "signals": [
                {
                    "symbol": "RGC",
                    "entry_price": 26.86,
                    "stop_loss": 31.69,
                    "target": 25.0,
                    "score": 90.0,
                }
            ]
        }
    )

    order = converted["entry_orders"][0]
    assert order["stop_loss"] < order["entry_price"]
    assert order["take_profit"] > order["entry_price"]
    assert order["risk_level_repaired"] is True
    assert "stop_loss_not_below_entry" in order["risk_level_repair_reasons"]
    assert "take_profit_not_above_entry" in order["risk_level_repair_reasons"]


def test_tomorrows_plan_generator_repairs_inverted_long_bracket_levels(tmp_path):
    plan_path = tmp_path / "pm_plan.json"
    plan_path.write_text(
        """
        {
          "generated_at": "2026-05-13T08:00:00",
          "signals": [
            {
              "symbol": "RGC",
              "entry_price": 26.86,
              "stop_loss": 31.69,
              "target": 25.0,
              "confidence": 90.0,
              "recommendation": "BUY"
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    generator = TomorrowsPlanGenerator.__new__(TomorrowsPlanGenerator)
    generator.logger = _Logger()
    generator.max_positions = 5
    generator.position_size_target = 1000.0
    generator.get_account_info = lambda: {"buying_power": 10000.0}
    generator.detect_odd_lots = lambda min_position_size=5: []
    generator.detect_losers_to_cut = lambda max_loss_pct=-5.0: []
    generator.get_current_positions = lambda: []
    generator._apply_plan_entry_constraints = lambda plan, source="plan": {
        "max_positions": 5
    }
    generator._coerce_float = TomorrowsPlanGenerator._coerce_float.__get__(
        generator, TomorrowsPlanGenerator
    )
    generator._coerce_int = TomorrowsPlanGenerator._coerce_int.__get__(
        generator, TomorrowsPlanGenerator
    )

    plan = generator.generate_plan(pm_plan_path=plan_path)

    # day-manager 2026-05-19 (H3): TomorrowsPlanGenerator.generate_plan now
    # rejects explicit bad source geometry at the producer instead of
    # silently repairing it. The signal above has tp=25.0 < entry=26.86,
    # so the candidate must be skipped, not repaired. The defense-in-depth
    # sanitize path remains for legitimate edge cases (e.g. missing levels
    # that fall through to the entry*0.95 / entry*1.10 defaults).
    assert plan["entry_orders"] == []
    assert any(
        "[PLAN PRODUCER REJECT]" in str(call[0][0])
        for call in generator.logger.warnings
    ), "expected [PLAN PRODUCER REJECT] warning for inverted source geometry"
