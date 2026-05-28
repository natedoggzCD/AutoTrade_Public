from autotrade.signals.inverse_etf_screener import InverseETFScreener


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows

    def get_all_inverse_etfs(self, active_only=True):
        return list(self._rows)

    def upsert_screen_result(self, payload):
        return None


def test_gate1_liquidity_keeps_leveraged_index_inverse_etfs_without_metadata():
    screener = InverseETFScreener(
        _FakeDB(
            [
                {
                    "ticker": "SQQQ",
                    "category": "index",
                    "leverage": 3,
                    "avg_daily_volume": 0,
                    "aum_millions": 0.0,
                },
                {
                    "ticker": "SPXU",
                    "category": "index",
                    "leverage": 3,
                    "avg_daily_volume": 0,
                    "aum_millions": 0.0,
                },
            ]
        )
    )

    passed = screener._gate1_liquidity()

    assert {row["ticker"] for row in passed} == {"SQQQ", "SPXU"}


def test_classify_signal_allows_crash_open_momentum_in_inverse_fast():
    screener = InverseETFScreener(_FakeDB([]))

    normal_signal = screener._classify_signal(
        composite=48.0,
        rsi=80.0,
        vwap_distance=0.025,
        momentum_return=0.008,
        volume_ratio=1.3,
        entry_mode="",
        minutes_since_open=2,
    )
    fast_signal = screener._classify_signal(
        composite=48.0,
        rsi=80.0,
        vwap_distance=0.025,
        momentum_return=0.008,
        volume_ratio=1.3,
        entry_mode="inverse_fast",
        minutes_since_open=2,
    )

    assert normal_signal == "AVOID"
    assert fast_signal == "ENTRY"
