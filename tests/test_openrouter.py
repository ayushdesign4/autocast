"""Comprehensive tests for OpenRouter free fallback, model failover, and quality validation."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import httpx
import pytest

from pipeline.groq_script import generate_short_pack, validate_short_pack
from pipeline.openrouter import (
    is_verified_free_model,
    fetch_free_models,
    generate_openrouter_short_pack,
    _sanitize_log,
)
from pipeline.story_history import save_history_entry, is_duplicate_story


@pytest.fixture
def mock_preset():
    return {
        "label": "90s Indian Nostalgia",
        "topic_pool": ["1990s nostalgia test topic"],
        "min_words": 130,
        "max_words": 185,
        "segment_count": 6,
        "groq_system_hint": "Write engaging Shorts",
    }


@pytest.fixture
def valid_story_payload():
    return {
        "youtube_title": "ओपनराउटर की पहली सुनहरी याद",
        "youtube_description": "1990 के दौर की खूबसूरत कहानी। #90sNostalgia #Shorts",
        "full_narration": (
            "1990 का वो दौर जब हर शाम मोहल्ले की गलियों में बच्चों का मेला लगता था और चारों तरफ हंसी गूंजती थी। "
            "लकड़ी के बड़े से दरवाजे पर खड़े होकर मां का आवाज लगाना और नंगे पांव घर की तरफ दौड़ पड़ना। "
            "हाथ-पैर धोकर जब सब बच्चे दरी पर बैठते थे, तो लालटेन की मद्धम रोशनी में किताबें खुल जाती थीं। "
            "दीवार पर परछाइयों के खेल बनाना और दादाजी से पुरानी कहानियां सुनकर धीरे-धीरे सो जाना। "
            "न कोई टीवी का शोर था और न मोबाइल का पहरा, बस दिल से जुड़े रिश्ते और अपनों की सच्ची ममता थी। "
            "गर्मियों की शाम को मिट्टी के मटके का ठंडा पानी पीना और आंगन में नीम के पेड़ के नीचे बैठकर बातें करना जिंदगी का सबसे बड़ा सुख था। "
            "आज सब कुछ बदल चुका है, हाथ में बड़ी-बड़ी स्क्रीन हैं, मगर वो भोली यादें आज भी दिल के किसी कोने में मुस्कुरा रही हैं।"
        ),
        "image_prompts": [
            "A rustic Indian courtyard at dusk, children laughing and playing, vertical 9:16",
            "A warm mother standing by a weathered wooden doorway calling her children home, vertical 9:16",
            "Children sitting cross-legged on a woven durrie rug studying near a hurricane lantern, vertical 9:16",
            "Hand shadows on a lime-washed clay wall mimicking flying birds and rabbits, vertical 9:16",
            "An elderly grandfather with spectacles softly telling folk tales to sleepy grandchildren, vertical 9:16",
            "Gentle golden morning light illuminating an empty traditional verandah with quiet peace, vertical 9:16",
        ],
        "tags": ["90s nostalgia", "openrouter test", "childhood"],
    }


def test_free_model_verification():
    """Verify that only models explicitly marked free or zero-priced are verified as free."""
    # openrouter/free is the official free auto-router
    assert is_verified_free_model("openrouter/free") is True

    # Explicit :free suffix
    assert is_verified_free_model("meta-llama/llama-3.3-70b-instruct:free") is True
    assert is_verified_free_model("mistralai/mistral-7b-instruct:free") is True

    # Pricing prompt = 0, completion = 0
    assert is_verified_free_model("custom/zero-model", {"prompt": "0", "completion": "0"}) is True
    assert is_verified_free_model("custom/zero-float", {"prompt": 0.0, "completion": 0.0}) is True

    # Paid models MUST be rejected
    assert is_verified_free_model("meta-llama/llama-3.3-70b-instruct") is False
    assert is_verified_free_model("openai/gpt-4o") is False
    assert is_verified_free_model("anthropic/claude-3.5-sonnet") is False
    assert is_verified_free_model("custom/paid-model", {"prompt": "0.001", "completion": "0.002"}) is False
    assert is_verified_free_model("") is False


def test_fetch_free_models_filters_out_paid():
    """Ensure fetch_free_models filters out paid models from OpenRouter catalogue."""
    fake_catalog = {
        "data": [
            {"id": "meta-llama/llama-3.3-70b-instruct:free", "pricing": {"prompt": "0", "completion": "0"}},
            {"id": "openai/gpt-4o", "pricing": {"prompt": "0.005", "completion": "0.015"}},
            {"id": "google/gemini-pro", "pricing": {"prompt": "0.00025", "completion": "0.0005"}},
            {"id": "qwen/qwen-2.5-72b:free", "pricing": {"prompt": "0", "completion": "0"}},
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fake_catalog

    mock_client = MagicMock()
    mock_client.get.return_value = mock_resp

    free_list = fetch_free_models("https://openrouter.ai/api/v1", "fake-key", client=mock_client)
    assert len(free_list) == 2
    assert "meta-llama/llama-3.3-70b-instruct:free" in free_list
    assert "qwen/qwen-2.5-72b:free" in free_list
    assert "openai/gpt-4o" not in free_list
    assert "google/gemini-pro" not in free_list


def test_groq_succeeds_openrouter_not_called(mock_preset, valid_story_payload, monkeypatch):
    """When Groq succeeds, OpenRouter fallback is never invoked."""
    mock_openrouter = MagicMock()
    monkeypatch.setattr("pipeline.openrouter.generate_openrouter_short_pack", mock_openrouter)
    monkeypatch.setattr("pipeline.groq_script._call_groq", MagicMock(return_value=valid_story_payload))

    result = generate_short_pack(mock_preset, topic_hint="कुछ पुरानी यादें")
    assert result["youtube_title"] == valid_story_payload["youtube_title"]
    assert mock_openrouter.called is False


def test_groq_fails_openrouter_succeeds(mock_preset, valid_story_payload, monkeypatch):
    """When Groq fails, OpenRouter is called and returns valid story pack."""
    # Groq fails with rate limit or error
    monkeypatch.setattr("pipeline.groq_script._call_groq", MagicMock(side_effect=RuntimeError("Groq 429 Rate Limit")))

    # OpenRouter succeeds
    monkeypatch.setattr(
        "pipeline.openrouter.generate_openrouter_short_pack",
        MagicMock(return_value=valid_story_payload),
    )

    result = generate_short_pack(mock_preset, topic_hint="कुछ पुरानी यादें")
    assert result["youtube_title"] == valid_story_payload["youtube_title"]
    assert len(result["image_prompts"]) == 6


def test_openrouter_router_fails_and_free_candidate_succeeds(mock_preset, valid_story_payload, monkeypatch):
    """When openrouter/free fails (e.g. 503), another verified free model succeeds."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    monkeypatch.setattr("pipeline.openrouter._cached_working_free_model", None)
    monkeypatch.setattr("pipeline.openrouter._cached_free_catalog", None)

    # First call (openrouter/free) returns 503; second call (free model) returns 200 with valid story
    resp_503 = MagicMock()
    resp_503.status_code = 503
    resp_503.text = "Service Unavailable"

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "choices": [{"message": {"content": json.dumps(valid_story_payload)}}],
    }

    # Model catalog response
    resp_catalog = MagicMock()
    resp_catalog.status_code = 200
    resp_catalog.json.return_value = {
        "data": [
            {"id": "meta-llama/llama-3.3-70b-instruct:free", "pricing": {"prompt": "0", "completion": "0"}},
            {"id": "paid/gpt-4", "pricing": {"prompt": "1", "completion": "1"}},
        ]
    }

    def fake_post(url, **kwargs):
        payload = kwargs.get("json", {})
        if payload.get("model") == "openrouter/free":
            return resp_503
        if payload.get("model") == "meta-llama/llama-3.3-70b-instruct:free":
            return resp_200
        return resp_503

    def fake_get(url, **kwargs):
        if "models" in url:
            return resp_catalog
        return resp_503

    with patch("httpx.Client.post", side_effect=fake_post):
        with patch("httpx.Client.get", side_effect=fake_get):
            pack = generate_openrouter_short_pack(
                mock_preset,
                user_prompt="prompt",
                min_words=130,
                max_words=185,
                seg_count=6,
                topic="test topic",
            )
            assert pack is not None
            assert pack["youtube_title"] == valid_story_payload["youtube_title"]


def test_openrouter_401_fails_fast(mock_preset, monkeypatch):
    """When OpenRouter returns 401 Unauthorized, it fails fast without endless candidate retries."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-invalid-key")

    resp_401 = MagicMock()
    resp_401.status_code = 401
    resp_401.text = "Unauthorized: Invalid API key"

    with patch("httpx.Client.post", return_value=resp_401) as mock_post:
        pack = generate_openrouter_short_pack(
            mock_preset,
            user_prompt="prompt",
            min_words=130,
            max_words=185,
            seg_count=6,
            topic="test topic",
        )
        assert pack is None
        # Must fail fast after 1 post attempt
        assert mock_post.call_count == 1


def test_openrouter_402_credit_restriction_fails_fast(mock_preset, monkeypatch):
    """When OpenRouter returns 402, it fails fast and never switches to paid models."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")

    resp_402 = MagicMock()
    resp_402.status_code = 402
    resp_402.text = "Payment Required: Credit balance zero"

    with patch("httpx.Client.post", return_value=resp_402) as mock_post:
        pack = generate_openrouter_short_pack(
            mock_preset,
            user_prompt="prompt",
            min_words=130,
            max_words=185,
            seg_count=6,
            topic="test topic",
        )
        assert pack is None
        assert mock_post.call_count == 1


def test_all_openrouter_free_models_fail_uses_fallback_catalog(mock_preset, monkeypatch):
    """When both Groq and all OpenRouter free models fail, 10-story fallback catalog is used."""
    monkeypatch.setattr("pipeline.groq_script._call_groq", MagicMock(side_effect=RuntimeError("Groq failed")))
    monkeypatch.setattr("pipeline.openrouter.generate_openrouter_short_pack", MagicMock(return_value=None))

    pack = generate_short_pack(mock_preset, topic_hint="कोई भी टॉपिक")
    assert pack is not None
    # Must be from the pre-authored 10-story catalog
    assert len(pack["image_prompts"]) == 6
    assert 130 <= len(pack["full_narration"].split()) <= 185


def test_openrouter_duplicate_output_rejected_in_novelty_loop(mock_preset, valid_story_payload, monkeypatch, tmp_path):
    """Ensure that if OpenRouter produces a duplicate of a previously uploaded story, it is detected."""
    test_hist_file = tmp_path / "history.json"
    test_hist_file.write_text("[]", encoding="utf-8")
    monkeypatch.setattr("pipeline.story_history.HISTORY_FILE", test_hist_file)

    # Save initial entry with same title and narration
    save_history_entry(
        "90s_indian_nostalgia",
        topic="पुरानी यादें",
        title=valid_story_payload["youtube_title"],
        narration=valid_story_payload["full_narration"],
    )

    # Check that this story is flagged as duplicate
    is_dup, reason = is_duplicate_story(
        topic="नया टॉपिक",
        title=valid_story_payload["youtube_title"],
        narration=valid_story_payload["full_narration"],
        scene_prompts=valid_story_payload["image_prompts"],
    )
    assert is_dup is True
    assert "Title" in reason or "Identical script narration hash" in reason


def test_secret_is_sanitized_in_logs():
    """Verify authorization secrets and Bearer tokens are redacted in log helper."""
    secret = "Bearer sk-or-v1-abcdef1234567890abcdef1234567890"
    sanitized = _sanitize_log(f"Request failed with header: {secret}")
    assert "sk-or-v1" not in sanitized
    assert "Bearer [REDACTED]" in sanitized


def test_validate_short_pack_helper():
    """Test the shared story validation helper against edge cases."""
    # 1. Valid pack
    valid_pack = {
        "youtube_title": "Valid Title",
        "full_narration": "शब्द " * 140,
        "image_prompts": ["Detailed visual prompt description " + str(i) for i in range(1, 7)],
    }
    is_ok, _ = validate_short_pack(valid_pack, min_words=130, max_words=185, seg_count=6)
    assert is_ok is True

    # 2. Too short
    short_pack = {
        "youtube_title": "Title",
        "full_narration": "शब्द " * 20,
        "image_prompts": ["Prompt " + str(i) for i in range(1, 7)],
    }
    is_ok, err = validate_short_pack(short_pack, min_words=130, max_words=185, seg_count=6)
    assert is_ok is False
    assert "Narration too short" in err

    # 3. Wrong scene prompt count
    wrong_scenes = {
        "youtube_title": "Title",
        "full_narration": "शब्द " * 140,
        "image_prompts": ["Prompt 1", "Prompt 2"],
    }
    is_ok, err = validate_short_pack(wrong_scenes, min_words=130, max_words=185, seg_count=6)
    assert is_ok is False
    assert "Expected exactly 6 image_prompts" in err
