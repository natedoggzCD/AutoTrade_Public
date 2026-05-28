import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from tools.youtube_daily_scanner import get_recent_videos, filter_new_videos

def test_filter_new_videos_terminal_states():
    # Test that 'complete' and 'complete_with_fallback' are terminal
    processed = {
        "processed": {
            "vid1": {"status": "complete"},
            "vid2": {"status": "complete_with_fallback"},
            "vid3": {"status": "transcription_failed"},
            "vid4": {"status": "extraction_failed"}
        }
    }
    
    videos = [
        {"id": "vid1", "title": "V1"},
        {"id": "vid2", "title": "V2"},
        {"id": "vid3", "title": "V3"},
        {"id": "vid4", "title": "V4"},
        {"id": "vid5", "title": "V5"}
    ]
    
    # filter_new_videos should keep vid3, vid4 (non-terminal) and vid5 (new)
    new_vids = filter_new_videos(videos, processed, "test_channel")
    new_ids = [v["id"] for v in new_vids]
    
    assert "vid1" not in new_ids
    assert "vid2" not in new_ids
    assert "vid3" in new_ids
    assert "vid4" in new_ids
    assert "vid5" in new_ids

@patch("yt_dlp.YoutubeDL")
def test_get_recent_videos_error_handling(mock_ydl):
    # Simulate a network error in yt-dlp
    mock_ydl.return_value.__enter__.return_value.extract_info.side_effect = Exception("DNS failure")
    
    # Should not raise exception, but return empty list and log error
    videos = get_recent_videos("https://youtube.com/c/test", max_age_hours=24)
    assert isinstance(videos, list)
    assert len(videos) == 0
