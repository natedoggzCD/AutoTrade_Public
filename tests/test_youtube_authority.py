from autotrade.core.autonomous_agent import filter_universe_level_youtube_avoids


def test_filter_universe_level_youtube_avoids_keeps_specific_sectors() -> None:
    filtered, dropped, tokens = filter_universe_level_youtube_avoids(
        ["small/mid-cap speculatives", "semiconductors"]
    )

    assert filtered == ["semiconductors"]
    assert dropped == ["small/mid-cap speculatives"]
    assert "small/mid-cap" in tokens
    assert "speculative" in tokens
