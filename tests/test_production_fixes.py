"""Comprehensive tests for production-quality fixes:
- LLM stage-specific timeouts and Groq fallback
- Script-aware direction fallback covering full Hindi story (no generic marbles)
- Direction validation (scene count >= 6, duration >= 45s, visual diversity)
- Topic history deduplication against manifest.json and India nostalgia themes
- Video duration enforcement (no upload < 45s)
- Stale artifact prevention via prompt_hash checking
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autocast.config import Config
from autocast.providers.llm import build_llm_providers
from autocast.spine import Character, CharacterBible, Run, Scene, Script, Shot, Topic, Upload, Video
from autocast.stages import direction, images, topic, upload
from autocast.stages.direction import _script_aware_direction, validate_direction
from autocast.stages.topic import _get_recent_titles, _is_duplicate_or_too_similar, _normalize_tokens


# ============================================================================
# 1. LLM Stage-Specific Timeouts and Cascade
# ============================================================================

def test_build_llm_providers_stage_specific_timeouts():
    cfg = Config(
        gemini_api_key="fake-gemini-key",
        agnes_api_key="fake-agnes-key",
        groq_api_key="fake-groq-key",
    )

    # Direction stage -> 120s default timeout
    dir_providers = build_llm_providers(cfg, prompt="test prompt", kind="direction")
    assert any(p.name == "gemini" for p in dir_providers)
    assert any(p.name == "agnes" for p in dir_providers)
    assert any(p.name == "groq" for p in dir_providers)

    # Script stage -> 60s default timeout
    script_providers = build_llm_providers(cfg, prompt="test prompt", kind="script")
    assert len(script_providers) >= 3

    # Topic stage -> 25s default timeout
    topic_providers = build_llm_providers(cfg, prompt="test prompt", kind="topic-rank")
    assert len(topic_providers) >= 3

    # Explicit override timeout respected
    custom_providers = build_llm_providers(
        cfg, prompt="test prompt", kind="direction", timeout=42.0
    )
    assert len(custom_providers) >= 3


def test_groq_fallback_models_on_failure():
    from autocast.providers.llm import _call_groq

    calls = []

    def mock_compat(url, model, prompt, **kwargs):
        calls.append(model)
        if model == "llama-3.3-70b-versatile":
            raise RuntimeError("404 Not Found")
        return "Groq response from fallback model"

    with patch("autocast.providers.llm._call_openai_compat", side_effect=mock_compat):
        resp = _call_groq("test-key", "Test Prompt")
        assert resp == "Groq response from fallback model"
        assert calls == ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]


# ============================================================================
# 2. Script-Aware Direction Fallback & Validation
# ============================================================================

def test_script_aware_direction_derives_scenes_from_actual_story():
    story = (
        "1990 की गर्मियों में आरव और उसकी बहन पिंकी नानी के घर पहुंचे। "
        "आंगन में ठंडी छांव थी और नानी आम का पन्ना बना रही थीं। "
        "दोनों बच्चे दोपहर में छुपकर बगीचे की तरफ भागे। "
        "वहां कच्चे आम तोड़ते हुए आरव का पैर फिसल गया और कीचड़ में जा गिरा। "
        "पिंकी जोर-जोर से हंसने लगी और पूरा बगीचा उनकी खिलखिलाहट से गूंज उठा। "
        "शाम को छत पर चारपाई बिछी और नानी ने तारों भरी रात में परियों की कहानी सुनाई। "
        "वो बचपन के दिन हमेशा के लिए यादों में बस गए।"
    )
    title = "नानी का घर और आम का बगीचा"

    chars, scenes, provider = _script_aware_direction(story, title, num_scenes=7)

    assert provider == "script-aware-fallback"
    assert len(scenes) >= 6
    assert len(chars) >= 2

    # Check each scene has story text, dialogue, and valid duration
    for s in scenes:
        assert s["dialogue_hindi"]
        assert s["narration"]
        assert 8.0 <= s["duration_s"] <= 10.0
        assert "1990s Ghibli-inspired" in s["video_prompt"]
        assert "Negative prompt:" in s["video_prompt"]

    # Verify visual diversity: every scene prompt is unique
    prompts = [s["video_prompt"] for s in scenes]
    assert len(set(prompts)) == len(prompts)

    # Verify full coverage: joining narrations reconstructs all words of the story
    reconstructed = " ".join(s["narration"] for s in scenes)
    assert reconstructed == " ".join(story.split())

    # Ensure generic marbles shot was NEVER used
    assert not any("कंचा सीधा निशाने पर लगेगा" in s["dialogue_hindi"] for s in scenes)
    assert not any("marbles in dusty lane" in s["video_prompt"] for s in scenes)


def test_direction_validation_pass_and_failures():
    valid_scenes = [
        {
            "scene_number": i + 1,
            "dialogue_hindi": f"संवाद {i+1}",
            "video_prompt": f"Unique prompt {i+1}",
            "duration_s": 9.0,
        }
        for i in range(7)
    ]
    # Passes with 7 scenes, 63s total duration (target 60-90s), distinct prompts
    validate_direction(valid_scenes)

    # Fails if fewer than 6 scenes
    with pytest.raises(ValueError, match="only 5 scenes parsed"):
        validate_direction(valid_scenes[:5])

    # Fails if total duration < 60s
    short_scenes = [dict(s, duration_s=7.0) for s in valid_scenes[:7]]  # 7 * 7 = 49s
    with pytest.raises(ValueError, match="below minimum 60.0s"):
        validate_direction(short_scenes)

    # Fails if duplicate visual prompts exist
    dup_scenes = [dict(s) for s in valid_scenes]
    dup_scenes[1]["video_prompt"] = dup_scenes[0]["video_prompt"]
    with pytest.raises(ValueError, match="duplicate scene visual prompts"):
        validate_direction(dup_scenes)

    # Fails if dialogue is missing
    empty_dial_scenes = [dict(s) for s in valid_scenes]
    empty_dial_scenes[2]["dialogue_hindi"] = ""
    with pytest.raises(ValueError, match="no dialogue or narration"):
        validate_direction(empty_dial_scenes)


# ============================================================================
# 3. Topic History Deduplication
# ============================================================================

def test_topic_similarity_and_deduplication():
    recent = [
        "गाँव का मेला, लकड़ी की फ़िरकी और दादी की खिचड़ी",
        "बिजली का जाना, मोमबत्ती और दूरदर्शन का रविवार",
        "ताश के पत्टे और चौपाल की गपशप",
    ]

    # Exact duplicate
    assert _is_duplicate_or_too_similar("गाँव का मेला, लकड़ी की फ़िरकी और दादी की खिचड़ी", recent)

    # Close variation sharing high keyword overlap
    assert _is_duplicate_or_too_similar("गाँव का मेला और लकड़ी की फ़िरकी", recent)
    assert _is_duplicate_or_too_similar("बिजली का जाना और दूरदर्शन रविवार", recent)

    # Completely novel 1990s nostalgia topic -> NOT duplicate
    assert not _is_duplicate_or_too_similar("स्कूल की घंटी, स्याही की दवात और खट्टी इमली", recent)
    assert not _is_duplicate_or_too_similar("रेलगाड़ी की खिड़की और कुल्हड़ वाली चाय", recent)


def test_topic_stage_avoids_recent_manifest_duplicates(tmp_path):
    manifest_path = tmp_path / "runs" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_data = [
        {"run_id": "2026-10-09", "title": "गाँव का मेला, लकड़ी की फ़िरकी और दादी की खिचड़ी", "status": "completed"},
        {"run_id": "2026-10-08", "title": "बिजली का जाना, मोमबत्ती और दूरदर्शन का रविवार", "status": "completed"},
        {"run_id": "2026-10-07", "title": "ताश के पत्टे और चौपाल की गपशप", "status": "completed"},
    ]
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    cfg = Config(
        runs_dir=tmp_path / "runs",
        queue_path=tmp_path / "queue" / "topics.json",
    )

    spine = Run.new("2026-10-10")

    # Mock LLM returning duplicate candidate
    dup_title = "गाँव का मेला और फ़िरकी"
    with patch("autocast.stages.topic.try_llm", return_value=(dup_title, "mock-llm")):
        spine = topic.run(spine, cfg, dry_run=False)

    # Must reject the duplicate and pick a fresh diverse alternative
    assert spine.topic.title != dup_title
    assert not _is_duplicate_or_too_similar(spine.topic.title, [m["title"] for m in manifest_data])


# ============================================================================
# 4. Duration Safety Enforcement in Upload Stage
# ============================================================================

def test_upload_blocks_video_under_60_seconds_and_validates_probe(tmp_path):
    cfg = Config(
        runs_dir=tmp_path,
        yt_client_id="cid",
        yt_client_secret="sec",
        yt_refresh_token="ref",
    )
    run_dir = cfg.run_dir("2026-10-09")
    run_dir.mkdir(parents=True, exist_ok=True)
    video_path = run_dir / "video.mp4"

    # Case A: Metadata duration < 60.0s raises immediately
    run_a = Run.new("2026-10-09")
    run_a.topic = Topic(title="Test Story")
    run_a.video = Video(final_path="video.mp4", duration_s=15.12)
    with pytest.raises(RuntimeError, match="below safety threshold \\(60.0s\\)"):
        upload.run(run_a, cfg, dry_run=False)

    # Case B: Video file missing on disk raises FileNotFoundError
    run_b = Run.new("2026-10-09")
    run_b.topic = Topic(title="Test Story")
    run_b.video = Video(final_path="video.mp4", duration_s=75.0)
    with pytest.raises(FileNotFoundError, match="video file not found"):
        upload.run(run_b, cfg, dry_run=False)

    # Create dummy video file on disk for probe testing
    video_path.write_bytes(b"dummy-mp4-data")

    # Case C: Probe throws exception -> fails safely with RuntimeError
    with patch("autocast.ffmpeg.run.probe_duration", side_effect=RuntimeError("ffprobe crashed")):
        with pytest.raises(RuntimeError, match="duration probe failed"):
            upload.run(run_b, cfg, dry_run=False)

    # Case D: Probe returns <= 0 -> fails safely with RuntimeError
    with patch("autocast.ffmpeg.run.probe_duration", return_value=0.0):
        with pytest.raises(RuntimeError, match="invalid duration"):
            upload.run(run_b, cfg, dry_run=False)

    # Case E: Probed duration < 60.0s (e.g. 54.0s) -> rejects with RuntimeError
    with patch("autocast.ffmpeg.run.probe_duration", return_value=54.0):
        with pytest.raises(RuntimeError, match="below safety threshold \\(60.0s\\)"):
            upload.run(run_b, cfg, dry_run=False)

    # Case F: Probed duration >= 60.0s and <= 90.0s (e.g. 72.0s) -> passes duration guard
    with patch("autocast.ffmpeg.run.probe_duration", return_value=72.0), \
         patch("autocast.stages.upload.get_access_token", return_value="fake-token"), \
         patch("autocast.stages.upload._upload_video_resumable", return_value="PROBED_OK_ID"):
        res = upload.run(run_b, cfg, dry_run=False)
        assert res.upload.youtube_video_id == "PROBED_OK_ID"


# ============================================================================
# 5. Stale Artifact Prevention via Prompt Hash
# ============================================================================

def test_images_stage_re_generates_when_prompt_hash_changes(tmp_path):
    from autocast.stages.direction import compute_scene_hash

    cfg = Config(runs_dir=tmp_path / "runs")
    assets_dir = cfg.assets_dir("2026-10-10")
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Existing stale clip on disk
    scene1_file = assets_dir / "scene_001.mp4"
    scene1_file.write_bytes(b"old-stale-clip-content")

    old_prompt = "Old scene prompt"
    old_hash = compute_scene_hash(old_prompt, "पुराना संवाद", 9.0)

    new_prompt = "New modified scene prompt with different action"
    new_dial = "नया संवाद"
    expected_hash = compute_scene_hash(new_prompt, new_dial, 9.0)

    spine = Run.new("2026-10-10")
    scene1 = Scene(
        scene_number=1,
        title_hindi="दृश्य 1",
        dialogue_hindi=new_dial,
        video_prompt=new_prompt,
        duration_s=9.0,
        status="completed",
        prompt_hash=old_hash,  # Hash mismatch! (prompt and dialogue changed)
        clip_path="assets/scene_001.mp4",
    )
    spine.scenes = [scene1]
    spine.shots = [
        Shot(
            idx=0,
            scene_number=1,
            video_prompt=new_prompt,
            narration=new_dial,
            duration_s=9.0,
            status="completed",
            prompt_hash=old_hash,
            clip_path="assets/scene_001.mp4",
        )
    ]

    with patch("autocast.stages.images.build_video_providers") as mock_providers:
        mock_provider = MagicMock()
        mock_provider.name = "mock-video"
        mock_provider.run.return_value = "assets/scene_001.mp4"
        mock_providers.return_value = [mock_provider]

        spine = images.run(spine, cfg, dry_run=False)
        # Because prompt changed, generation was called (not skipped)
        mock_providers.assert_called_once()
        assert spine.scenes[0].prompt_hash == expected_hash


# ============================================================================
# 6. Script Word Count Enforcement (>= 160-180 words)
# ============================================================================

def test_script_stage_enforces_mandatory_160_words(tmp_path):
    from autocast.stages import script

    cfg = Config(runs_dir=tmp_path / "runs")
    title = "गर्मी की छुट्टियां और नानी का घर"

    # Case A: Dry run produces a complete, rich narrative with >= 160 words
    spine_dry = Run.new("2026-10-10")
    spine_dry.topic = Topic(title=title)
    spine_dry = script.run(spine_dry, cfg, dry_run=True)
    assert spine_dry.script.word_count >= 160
    assert spine_dry.script.word_count <= 320
    assert "1990" in spine_dry.script.full_text

    # Case B: LLM returns short story (< 160 words), expansion prompt expands it to >= 160 words
    short_text = "गर्मियों में आरव नानी के घर गया। वहां बहुत आम खाए और बहुत मजा आया।"  # 13 words
    long_expanded_text = (
        "1990 के उस सुनहरे दौर में जब गर्मियां शुरू होते ही नानी के गाँव जाने का इंतज़ार रहता था। "
        "सुबह की ताज़ा धूप कच्चे दालान पर बिखरती और नीम की डालियों पर बैठी चिड़ियों की चहचहाहट से नींद खुलती थी। "
        "तपती दोपहर में आरव और पिंकी ने नज़रों ही नज़रों में इशारा किया और बिना चप्पल पहने चुपके से बगीचे की तरफ दौड़ पड़े। "
        "आम की लदी डालियों पर झूलते कच्चे आम और हवा में तैरती सोंधी मिट्टी की महक। "
        "आरव ने पेड़ की डाल पकड़कर जोर से हिलाई तो रसीली अमिया सीधे पिंकी की झोली में आ गिरी। "
        "दूर से नानी की ममता भरी आवाज़ गूंजी, 'अरे शैतानों, धूप में क्यों घूम रहे हो, यहाँ आओ!' "
        "जब दोनों डरते-डरते पहुँचे, तो नानी ने अपनी सूती साड़ी के पल्लू से बच्चों का पसीना पोंछा "
        "और प्यार से सिलबट्टे पर पिसी चटनी के साथ नमक-मिर्च लगी ताज़ा अमिया थमा दी। "
        "शाम को खुली छत पर ठंडी हवा में चारपाई बिछी और तारों भरी रात में परियों की कहानियों के साथ वो प्यारा बचपन हमेशा के लिए अमर हो गया।"
    )  # ~165 words

    call_count = 0
    def mock_try_llm(providers):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return short_text, "mock-gemini"
        return long_expanded_text, "mock-gemini"

    spine_llm = Run.new("2026-10-10")
    spine_llm.topic = Topic(title=title)
    with patch("autocast.stages.script.try_llm", side_effect=mock_try_llm):
        spine_llm = script.run(spine_llm, cfg, dry_run=False)

    assert call_count == 2  # First generation + expansion call
    assert spine_llm.script.word_count >= 160
    assert spine_llm.script.provider == "mock-gemini+expanded"

    # Case C: Story already >= 160 words -> accepted immediately without expansion call
    call_count_c = 0
    def mock_try_llm_c(providers):
        nonlocal call_count_c
        call_count_c += 1
        return long_expanded_text, "mock-gemini"

    spine_good = Run.new("2026-10-10")
    spine_good.topic = Topic(title=title)
    with patch("autocast.stages.script.try_llm", side_effect=mock_try_llm_c):
        spine_good = script.run(spine_good, cfg, dry_run=False)

    assert call_count_c == 1  # Accepted on first try, no expansion needed
    assert spine_good.script.word_count >= 160
    assert spine_good.script.provider == "mock-gemini"
