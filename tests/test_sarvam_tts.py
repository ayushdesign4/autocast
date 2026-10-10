"""Tests for Sarvam AI TTS module, speaker persona 'shubh', and error classification."""
from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock
import httpx
import pytest

from pipeline.sarvam_tts import synthesize_full, split_sentences


def test_split_sentences_hindi():
    """Verify sentence splitting handles Devanagari danda and standard punctuation."""
    text = "यह पहली पंक्ति है। दूसरी पंक्ति यहां है! क्या यह तीसरी है? चौथी पंक्ति।"
    sentences = split_sentences(text)
    assert len(sentences) == 4
    assert sentences[0] == "यह पहली पंक्ति है"
    assert sentences[1] == "दूसरी पंक्ति यहां है"


def test_sarvam_tts_success_mocked(tmp_path, monkeypatch):
    """Test successful Sarvam TTS call and sentence timings calculation."""
    monkeypatch.setenv("SARVAM_API_KEY", "mock-sarvam-key")
    monkeypatch.setattr("pipeline.sarvam_tts.probe_audio_duration", lambda p: 45.0)

    # Fake audio payload
    mock_audio_bytes = b"RIFF____WAVEfmt " + b"\x00" * 4000
    b64_audio = base64.b64decode(base64.b64encode(mock_audio_bytes)).decode("latin1")
    b64_encoded = base64.b64encode(mock_audio_bytes).decode("ascii")

    def mock_post(self, url, json=None, headers=None, **kwargs):
        assert headers.get("api-subscription-key") == "mock-sarvam-key"
        assert json.get("speaker") == "shubh"
        assert json.get("language_code") == "hi-IN"
        assert json.get("model") == "bulbul:v3"

        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {"audios": [b64_encoded]}
        return resp

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    out_file = tmp_path / "voiceover.wav"
    narration = "1990 का वो दौर बहुत प्यारा था। हम सब मिलकर दूरदर्शन देखते थे।"
    total_dur, timings = synthesize_full(narration, out_file)

    assert total_dur == 45.0
    assert len(timings) == 2
    assert out_file.is_file()
    assert timings[0]["offset_ms"] == 0
    assert timings[1]["offset_ms"] > 0


def test_sarvam_tts_401_auth_error(tmp_path, monkeypatch):
    """Test that HTTP 401 raises an explicit Sarvam authentication error."""
    monkeypatch.setenv("SARVAM_API_KEY", "invalid-key")

    def mock_post(self, url, **kwargs):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 401
        resp.text = "Unauthorized: Invalid Subscription Key"
        return resp

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    with pytest.raises(RuntimeError, match="authentication failed"):
        synthesize_full("कुछ हिंदी पंक्तियां।", tmp_path / "out.wav")


def test_sarvam_tts_402_quota_error(tmp_path, monkeypatch):
    """Test that HTTP 402 raises an explicit Sarvam quota/credits exhausted error."""
    monkeypatch.setenv("SARVAM_API_KEY", "no-balance-key")

    def mock_post(self, url, **kwargs):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 402
        resp.text = "Payment Required: Out of credits"
        return resp

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    with pytest.raises(RuntimeError, match="quota/budget exhausted"):
        synthesize_full("कुछ हिंदी पंक्तियां।", tmp_path / "out.wav")
