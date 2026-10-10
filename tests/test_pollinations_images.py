"""Tests for Pollinations image generation, error classification, and 9:16 vertical formatting."""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import httpx
from PIL import Image
import pytest

from pipeline.images import save_scene_image, _ensure_vertical_9_16


def _create_mock_image_bytes(w: int, h: int, color=(200, 100, 50)) -> bytes:
    img = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_ensure_vertical_9_16_resizes_and_crops_properly():
    """Verify that images are cropped and resized to exact 1080x1920 without stretching."""
    # Test horizontal image cropped to 9:16
    horiz_bytes = _create_mock_image_bytes(1920, 1080)
    vert_bytes = _ensure_vertical_9_16(horiz_bytes, target_w=1080, target_h=1920)

    out_img = Image.open(io.BytesIO(vert_bytes))
    assert out_img.size == (1080, 1920)

    # Test square image cropped to 9:16
    square_bytes = _create_mock_image_bytes(1024, 1024)
    vert_bytes2 = _ensure_vertical_9_16(square_bytes, target_w=1080, target_h=1920)
    out_img2 = Image.open(io.BytesIO(vert_bytes2))
    assert out_img2.size == (1080, 1920)


def test_pollinations_success_mocked(tmp_path, monkeypatch):
    """Test successful image generation and saving via Pollinations."""
    monkeypatch.setenv("POLLINATIONS_API_KEY", "mock-pollinations-key")

    mock_png = _create_mock_image_bytes(1080, 1920)

    def mock_get(self, url, headers=None, **kwargs):
        assert headers.get("Authorization") == "Bearer mock-pollinations-key"
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.headers = {"Content-Type": "image/png"}
        resp.content = mock_png
        return resp

    monkeypatch.setattr(httpx.Client, "get", mock_get)

    out_file = tmp_path / "scene_01.png"
    status, detail = save_scene_image(1, "A nostalgic 1990s room", out_file)

    assert status == "ok"
    assert out_file.is_file()
    assert out_file.stat().st_size > 0
    saved_img = Image.open(out_file)
    assert saved_img.size == (1080, 1920)


def test_pollinations_401_auth_error_fails_fast(tmp_path, monkeypatch):
    """Test that HTTP 401 raises an explicit authentication error immediately."""
    monkeypatch.setenv("POLLINATIONS_API_KEY", "bad-key")

    attempts = 0
    def mock_get(self, url, headers=None, **kwargs):
        nonlocal attempts
        attempts += 1
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 401
        resp.text = "Unauthorized: Invalid API Key"
        return resp

    monkeypatch.setattr(httpx.Client, "get", mock_get)

    out_file = tmp_path / "fail.png"
    with pytest.raises(RuntimeError, match="authentication failed"):
        save_scene_image(1, "test prompt", out_file)

    # Fast failure: does not endlessly retry 401
    assert attempts <= 2


def test_pollinations_402_quota_exhausted_error(tmp_path, monkeypatch):
    """Test that HTTP 402 raises an explicit quota/budget exhausted error."""
    monkeypatch.setenv("POLLINATIONS_API_KEY", "no-credits-key")

    def mock_get(self, url, headers=None, **kwargs):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 402
        resp.text = "Payment Required: Credit balance zero"
        return resp

    monkeypatch.setattr(httpx.Client, "get", mock_get)

    out_file = tmp_path / "quota_fail.png"
    with pytest.raises(RuntimeError, match="quota/budget exhausted"):
        save_scene_image(1, "test prompt", out_file)


def test_pollinations_429_rate_limit_retries_and_recovers(tmp_path, monkeypatch):
    """Test that HTTP 429 rate limit backs off and succeeds on retry."""
    monkeypatch.setenv("POLLINATIONS_API_KEY", "test-key")
    monkeypatch.setattr("time.sleep", lambda s: None)

    mock_png = _create_mock_image_bytes(1080, 1920)
    attempts = 0

    def mock_get(self, url, headers=None, **kwargs):
        nonlocal attempts
        attempts += 1
        resp = MagicMock(spec=httpx.Response)
        if attempts == 1:
            resp.status_code = 429
            resp.headers = {"Retry-After": "1"}
            resp.text = "Rate limit reached"
            return resp
        resp.status_code = 200
        resp.headers = {"Content-Type": "image/png"}
        resp.content = mock_png
        return resp

    monkeypatch.setattr(httpx.Client, "get", mock_get)

    out_file = tmp_path / "retry_success.png"
    status, detail = save_scene_image(1, "retry prompt", out_file)

    assert status == "ok"
    assert attempts == 2
    assert out_file.is_file()
