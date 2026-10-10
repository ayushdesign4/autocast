"""Tests for Groq script generation, validation, and 90s Indian Nostalgia fallback."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from pipeline.channel_presets import get_preset
from pipeline.groq_script import generate_short_pack, _get_fallback_pack


def test_fallback_pack_quality_and_word_count():
    """Verify that fallback script is rich, coherent, has 130+ Hindi words and 6 scene prompts."""
    preset = get_preset("90s_indian_nostalgia")
    pack = _get_fallback_pack(preset, "टेस्ट टॉपिक")

    assert pack["youtube_title"]
    assert len(pack["youtube_title"]) < 90
    words = pack["full_narration"].split()
    assert len(words) >= 130, f"Fallback script has only {len(words)} words; expected >= 130"
    assert len(pack["image_prompts"]) == 6
    for p in pack["image_prompts"]:
        assert len(p) > 20
        assert "vertical 9:16" in p.lower()


def test_groq_script_generation_mocked_success(monkeypatch):
    """Test successful Groq LLM script generation with mocked client."""
    monkeypatch.setenv("GROQ_API_KEY", "mock-groq-key")
    preset = get_preset("90s_indian_nostalgia")

    mock_resp_json = {
        "youtube_title": "1990s के रविवार और टीवी एंटीना की कहानी",
        "youtube_description": "वो दौर जब पूरा मोहल्ला एक छत पर टीवी देखने जुटता था। #90sNostalgia #Shorts",
        "full_narration": " ".join(["यादें"] * 145),  # 145 words
        "image_prompts": [
            f"Scene {i + 1}: Vintage Indian rooftop in 1990s, vertical 9:16" for i in range(6)
        ],
        "tags": ["nostalgia", "doordarshan", "shorts"],
    }

    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content=f"```json\n{json_dumps(mock_resp_json)}\n```"))
    ]

    with patch("pipeline.groq_script.Groq") as mock_groq_cls:
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = mock_completion
        mock_groq_cls.return_value = mock_instance

        pack = generate_short_pack(preset, topic_hint="रविवार दूरदर्शन")

        assert pack["youtube_title"] == "1990s के रविवार और टीवी एंटीना की कहानी"
        assert len(pack["full_narration"].split()) == 145
        assert len(pack["image_prompts"]) == 6


def test_groq_script_retries_on_short_narration_and_falls_back(monkeypatch):
    """Test that Groq retries if narration is too short and falls back gracefully."""
    monkeypatch.setenv("GROQ_API_KEY", "mock-groq-key")
    preset = get_preset("90s_indian_nostalgia")

    # Short narration (only 10 words)
    mock_short_json = {
        "youtube_title": "Too Short Title",
        "full_narration": "यह कहानी बहुत छोटी है और अस्वीकार कर दी जाएगी।",
        "image_prompts": ["scene 1"] * 6,
    }

    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content=json_dumps(mock_short_json)))
    ]

    with patch("groq.Groq") as mock_groq_cls:
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = mock_completion
        mock_groq_cls.return_value = mock_instance

        pack = generate_short_pack(preset, topic_hint="छोटा टॉपिक")
        # Should fall back to valid fallback pack
        assert len(pack["full_narration"].split()) >= 130
        assert len(pack["image_prompts"]) == 6


def json_dumps(d):
    import json
    return json.dumps(d, ensure_ascii=False)
