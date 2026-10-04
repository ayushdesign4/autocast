"""Tests for YouTube OAuth refresh, video upload stage, and notifications."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autocast.config import Config
from autocast.spine import Run, Script, StageSkipped, Thumbnail, Topic, Upload, Video
from autocast.stages import upload
from autocast.util.youtube_auth import YouTubeAuthError, get_access_token


def test_get_access_token_dry_run():
    token = get_access_token("cid", "sec", "ref", dry_run=True)
    assert token == "DRYRUN_ACCESS_TOKEN"


def test_get_access_token_missing_creds_raises():
    with pytest.raises(YouTubeAuthError, match="Missing YouTube credentials"):
        get_access_token(None, "sec", "ref", dry_run=False)


def test_get_access_token_refreshes_successfully():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"access_token": "valid_refreshed_access_token"}

    with patch("httpx.post", return_value=mock_resp) as mock_post:
        token = get_access_token("cid", "sec", "ref", dry_run=False)
        assert token == "valid_refreshed_access_token"
        mock_post.assert_called_once()


def test_get_access_token_invalid_grant_raises_clear_error():
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.json.return_value = {"error": "invalid_grant", "error_description": "Token has been expired or revoked."}

    with patch("httpx.post", return_value=mock_resp):
        with pytest.raises(YouTubeAuthError, match="Failed to refresh YouTube access token"):
            get_access_token("cid", "sec", "ref", dry_run=False)


def test_upload_dry_run_records_metadata_and_skips_network(tmp_path):
    cfg = Config(runs_dir=tmp_path)
    run = Run.new("2026-10-04")
    run.topic = Topic(title="1990s Mango Summer")
    run.script = Script(full_text="Warm summer afternoon in the village.")
    run.video = Video(final_path="video.mp4")

    out_run = upload.run(run, cfg, dry_run=True)

    assert out_run.upload is not None
    assert out_run.upload.title == "1990s Mango Summer"
    assert out_run.upload.privacy == "private"
    assert out_run.upload.youtube_video_id == "DRYRUN_VIDEO_ID"
    assert "AI-generated" in out_run.upload.description


def test_upload_without_creds_raises_stage_skipped(tmp_path):
    cfg = Config(runs_dir=tmp_path, yt_client_id=None, yt_client_secret=None, yt_refresh_token=None)
    run = Run.new("2026-10-04")
    run.topic = Topic(title="1990s Mango Summer")
    run.video = Video(final_path="video.mp4")

    with pytest.raises(StageSkipped, match="no YouTube credentials"):
        upload.run(run, cfg, dry_run=False)


def test_upload_real_resumable_and_thumbnail(tmp_path):
    cfg = Config(
        runs_dir=tmp_path,
        yt_client_id="test_cid",
        yt_client_secret="test_sec",
        yt_refresh_token="test_ref",
    )
    run_dir = cfg.run_dir("2026-10-04")
    run_dir.mkdir(parents=True, exist_ok=True)
    video_file = run_dir / "video.mp4"
    video_file.write_bytes(b"fake_video_data_mp4")

    thumb_dir = cfg.assets_dir("2026-10-04")
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb_file = thumb_dir / "thumb.jpg"
    thumb_file.write_bytes(b"fake_jpeg_thumb")

    run = Run.new("2026-10-04")
    run.topic = Topic(title="आख़िरी आम की गर्मी")
    run.script = Script(full_text="गांव की दोपहर में चीकू दौड़ा।")
    run.video = Video(final_path="video.mp4")
    run.thumbnail = Thumbnail(path="assets/thumb.jpg", width=1280, height=720)

    # Mock token refresh
    mock_token_resp = MagicMock(status_code=200)
    mock_token_resp.json.return_value = {"access_token": "ya29.fake_token"}

    # Mock video init
    mock_init_resp = MagicMock(status_code=200)
    mock_init_resp.headers = {"Location": "https://upload.youtube.com/resumable/test1234"}

    # Mock video upload
    mock_upload_resp = MagicMock(status_code=200)
    mock_upload_resp.json.return_value = {"id": "TEST_YT_VIDEO_ID_999"}

    # Mock thumbnail upload
    mock_thumb_resp = MagicMock(status_code=200)

    def mock_post(url, **kwargs):
        if "oauth2.googleapis.com" in url:
            return mock_token_resp
        if "youtube/v3/videos" in url:
            return mock_init_resp
        if "thumbnails/set" in url:
            return mock_thumb_resp
        raise ValueError(f"Unexpected url: {url}")

    def mock_put(url, **kwargs):
        if "upload.youtube.com" in url:
            return mock_upload_resp
        raise ValueError(f"Unexpected put url: {url}")

    with patch("httpx.post", side_effect=mock_post), patch("httpx.put", side_effect=mock_put):
        out_run = upload.run(run, cfg, dry_run=False)

    assert out_run.upload.youtube_video_id == "TEST_YT_VIDEO_ID_999"
    assert out_run.upload.privacy == "private"
    assert out_run.stage("upload").provider_used == "youtube-data-api-v3"


def test_upload_resumability_skips_duplicate(tmp_path):
    cfg = Config(
        runs_dir=tmp_path,
        yt_client_id="test_cid",
        yt_client_secret="test_sec",
        yt_refresh_token="test_ref",
    )
    run = Run.new("2026-10-04")
    run.topic = Topic(title="Already Uploaded Title")
    run.video = Video(final_path="video.mp4")
    run.upload = Upload(
        title="Already Uploaded Title",
        description="desc",
        tags=["t1"],
        privacy="private",
        youtube_video_id="EXISTING_YT_ID",
    )

    with patch("httpx.post") as mock_post:
        out_run = upload.run(run, cfg, dry_run=False)
        mock_post.assert_not_called()

    assert out_run.upload.youtube_video_id == "EXISTING_YT_ID"
