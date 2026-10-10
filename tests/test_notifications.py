"""Tests for email notifications (success & failure alerts via SMTP)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from pipeline.notifications import (
    send_upload_success_email,
    send_failure_alert_email,
    _get_smtp_config,
)


def test_smtp_config_parsing(monkeypatch):
    """Test SMTP environment variable reading and validation."""
    monkeypatch.setenv("SMTP_HOST", "smtp.gmail.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USERNAME", "bot@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret-app-password")
    monkeypatch.setenv("NOTIFICATION_EMAIL", "creator@example.com")

    cfg = _get_smtp_config()
    assert cfg is not None
    assert cfg["host"] == "smtp.gmail.com"
    assert cfg["port"] == 587
    assert cfg["user"] == "bot@example.com"
    assert cfg["recipient"] == "creator@example.com"


def test_smtp_missing_config_returns_none(monkeypatch):
    """Test missing SMTP credentials gracefully disables alerts without crashing."""
    monkeypatch.delenv("SMTP_HOST", raising=False)
    monkeypatch.delenv("NOTIFICATION_EMAIL", raising=False)

    cfg = _get_smtp_config()
    assert cfg is None
    # Neither success nor failure alert should crash
    assert send_upload_success_email(
        title="Test Title",
        video_id="vid123",
        video_url="https://youtube.com/shorts/vid123",
        topic="Test Topic",
        summary="Test Summary",
        duration_s=45.0,
    ) is False


def test_send_upload_success_email_mocked(monkeypatch):
    """Test success email composition and transmission."""
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USERNAME", "user@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")
    monkeypatch.setenv("NOTIFICATION_EMAIL", "notify@example.com")

    sent_messages = []

    class MockSMTP:
        def __init__(self, host, port, timeout=30):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def ehlo(self):
            pass
        def starttls(self):
            pass
        def login(self, u, p):
            assert p == "secret"
        def send_message(self, msg):
            sent_messages.append(msg)

    with patch("smtplib.SMTP", MockSMTP):
        ok = send_upload_success_email(
            title="1990s की यादें",
            video_id="v_12345",
            video_url="https://www.youtube.com/shorts/v_12345",
            topic="दूरदर्शन का दौर",
            summary="एक सुंदर कहानी",
            duration_s=52.3,
            run_id="2026-10-10_run_01",
        )
        assert ok is True
        assert len(sent_messages) == 1
        msg = sent_messages[0]
        assert "Upload successful" in msg["Subject"]
        body = msg.get_content()
        assert "https://www.youtube.com/shorts/v_12345" in body
        assert "52.3s" in body
        assert "Sarvam AI / shubh" in body
        assert "IST" in body


def test_send_failure_alert_email_mocked(monkeypatch):
    """Test failure email alert with error classification."""
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USERNAME", "user@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")
    monkeypatch.setenv("NOTIFICATION_EMAIL", "notify@example.com")

    sent_messages = []

    class MockSMTP:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def ehlo(self):
            pass
        def starttls(self):
            pass
        def login(self, u, p):
            pass
        def send_message(self, msg):
            sent_messages.append(msg)

    with patch("smtplib.SMTP", MockSMTP):
        ok = send_failure_alert_email(
            stage="image_generation",
            error_message="Pollinations AI quota/budget exhausted (HTTP 402): Account out of credits.",
            topic="नानी का घर",
            run_id="run_999",
        )
        assert ok is True
        assert len(sent_messages) == 1
        msg = sent_messages[0]
        assert "Pipeline failed at 'image_generation'" in msg["Subject"]
        body = msg.get_content()
        assert "HTTP 402" in body
        assert "credits exhausted" in body.lower()
