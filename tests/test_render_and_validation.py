"""Tests for caption generation, FFmpeg vertical render setup, and duration validation."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from pipeline.captions import build_srt
from pipeline.render_short import validate_rendered_short


def test_build_srt_devanagari(tmp_path):
    """Verify SRT generator creates valid subtitle blocks with Devanagari text."""
    timings = [
        {"text": "1990 के दशक की खूबसूरत यादें", "offset_ms": 0, "duration_ms": 4000},
        {"text": "रविवार सुबह दूरदर्शन का वो दौर", "offset_ms": 4000, "duration_ms": 5000},
    ]
    srt_file = tmp_path / "test.srt"
    build_srt(timings, srt_file, total_duration=9.0, max_words_per_line=3)

    assert srt_file.is_file()
    content = srt_file.read_text(encoding="utf-8")
    assert "00:00:00,000 -->" in content
    assert "1990" in content
    assert "दूरदर्शन" in content


def test_validate_rendered_short_success(tmp_path, monkeypatch):
    """Test validator passes when video is vertical 9:16 and has sufficient duration."""
    fake_video = tmp_path / "valid.mp4"
    fake_video.write_bytes(b"FAKE_MP4_BYTES" * 100)

    mock_metadata = {
        "streams": [
            {"codec_type": "video", "width": 1080, "height": 1920, "codec_name": "h264"},
            {"codec_type": "audio", "codec_name": "aac"},
        ],
        "format": {"duration": "48.5"},
    }

    monkeypatch.setattr("pipeline.render_short.probe_video_metadata", lambda p: mock_metadata)

    dur = validate_rendered_short(fake_video, min_duration=30.0, max_duration=90.0)
    assert dur == 48.5


def test_validate_rendered_short_rejects_horizontal(tmp_path, monkeypatch):
    """Test validator rejects landscape 16:9 video."""
    fake_video = tmp_path / "horizontal.mp4"
    fake_video.write_bytes(b"FAKE_BYTES" * 100)

    mock_metadata = {
        "streams": [
            {"codec_type": "video", "width": 1920, "height": 1080},
            {"codec_type": "audio"},
        ],
        "format": {"duration": "45.0"},
    }

    monkeypatch.setattr("pipeline.render_short.probe_video_metadata", lambda p: mock_metadata)

    with pytest.raises(ValueError, match="not vertical 9:16"):
        validate_rendered_short(fake_video)


def test_validate_rendered_short_rejects_too_short(tmp_path, monkeypatch):
    """Test validator rejects video below duration threshold."""
    fake_video = tmp_path / "short.mp4"
    fake_video.write_bytes(b"FAKE_BYTES" * 100)

    mock_metadata = {
        "streams": [
            {"codec_type": "video", "width": 1080, "height": 1920},
            {"codec_type": "audio"},
        ],
        "format": {"duration": "12.0"},  # Only 12s
    }

    monkeypatch.setattr("pipeline.render_short.probe_video_metadata", lambda p: mock_metadata)

    with pytest.raises(ValueError, match="too short"):
        validate_rendered_short(fake_video, min_duration=25.0)
