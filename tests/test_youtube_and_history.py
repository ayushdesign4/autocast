"""Tests for YouTube upload module, OAuth credentials, and duplicate history prevention."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from pipeline.story_history import is_duplicate_story, save_history_entry, load_history, history_prompt_block
from pipeline.youtube_upload import get_youtube_credentials, upload_short


def test_youtube_creds_reconstruction(monkeypatch):
    """Test OAuth credentials reconstruction from environment variables."""
    monkeypatch.setenv("YT_CLIENT_ID", "test-client-id.apps.googleusercontent.com")
    monkeypatch.setenv("YT_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv("YT_REFRESH_TOKEN", "test-refresh-token")

    with patch("google.oauth2.credentials.Credentials.refresh"):
        creds = get_youtube_credentials()
        assert creds.client_id == "test-client-id.apps.googleusercontent.com"
        assert creds.client_secret == "test-client-secret"
        assert creds.refresh_token == "test-refresh-token"


def test_youtube_upload_mocked_success(tmp_path, monkeypatch):
    """Test YouTube upload calls insert with correct snippet, tags, and private privacy."""
    monkeypatch.setenv("YT_CLIENT_ID", "test-id")
    monkeypatch.setenv("YT_CLIENT_SECRET", "test-secret")
    monkeypatch.setenv("YT_REFRESH_TOKEN", "test-token")

    fake_video = tmp_path / "video.mp4"
    fake_video.write_bytes(b"VIDEO_DATA" * 50)

    mock_insert_req = MagicMock()
    mock_insert_req.next_chunk.return_value = (None, {"id": "uploaded_vid_999"})

    mock_videos = MagicMock()
    mock_videos.insert.return_value = mock_insert_req

    mock_youtube = MagicMock()
    mock_youtube.videos.return_value = mock_videos

    with patch("pipeline.youtube_upload.get_youtube_credentials"):
        with patch("pipeline.youtube_upload.build", return_value=mock_youtube):
            vid = upload_short(
                fake_video,
                "1990s की सुनहरी यादें",
                "गर्मियों की छुट्टियों की कहानी।",
                privacy_status="private",
            )
            assert vid == "uploaded_vid_999"

            # Check insert body kwargs
            call_kwargs = mock_videos.insert.call_args[1]
            body = call_kwargs["body"]
            assert body["status"]["privacyStatus"] == "private"
            assert "1990s" in body["snippet"]["title"]
            assert body["snippet"]["defaultLanguage"] == "hi"


def test_duplicate_prevention_history(tmp_path, monkeypatch):
    """Test durable duplicate history blocks repeats of topic, title, and video hash."""
    test_hist_file = tmp_path / "history.json"
    test_hist_file.write_text("[]", encoding="utf-8")
    monkeypatch.setattr("pipeline.story_history.HISTORY_FILE", test_hist_file)

    # 1. Initially empty
    assert len(load_history()) == 0
    is_dup, _ = is_duplicate_story("दूरदर्शन की यादें", "रविवार की सुबह")
    assert is_dup is False

    # 2. Save an entry
    save_history_entry(
        "90s_indian_nostalgia",
        topic="दूरदर्शन की यादें",
        title="रविवार की सुबह और शक्तिमान",
        narration="एक लंबी कहानी दूरदर्शन के बारे में...",
        video_id="vid_123",
        video_hash="hash_abc123",
        duration_s=50.0,
    )

    # 3. Same topic or title should be flagged as duplicate
    is_dup1, reason1 = is_duplicate_story("दूरदर्शन की यादें", "अलग शीर्षक")
    assert is_dup1 is True
    assert "already used" in reason1

    is_dup2, reason2 = is_duplicate_story("नया टॉपिक", "रविवार की सुबह और शक्तिमान")
    assert is_dup2 is True

    is_dup3, reason3 = is_duplicate_story("नया टॉपिक", "नया शीर्षक", video_hash="hash_abc123")
    assert is_dup3 is True

    # 4. Completely new topic & title passes
    is_dup4, _ = is_duplicate_story("स्कूल का टिफिन", "पराठे और आम का अचार")
    assert is_dup4 is False

    # 5. History prompt block returns text
    block = history_prompt_block("90s_indian_nostalgia")
    assert "ANTI-REPEAT" in block
    assert "रविवार की सुबह" in block
