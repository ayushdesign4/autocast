"""Comprehensive tests for cross-run story novelty, multi-signal duplicate detection, and pre-upload gates."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from pipeline.groq_script import _get_fallback_pack, FALLBACK_CATALOG
from pipeline.story_history import (
    is_duplicate_story,
    save_history_entry,
    load_history,
    history_prompt_block,
    compute_sha256,
)
from scripts.run_short import select_unique_topic


def test_existing_oct10_oct11_story_is_rejected():
    """Verify that the actual duplicate story uploaded on Oct 10 and 11 is rejected."""
    # Narration from the Oct 10 & Oct 11 uploads
    oct10_title = "जब टीवी का मतलब सिर्फ रविवार और रामायण था"
    oct10_narration = (
        "1990 का वो दौर भी क्या कमाल का था, जब पूरे मोहल्ले में सिर्फ एक या दो घरों में रंगीन टीवी हुआ करता था। "
        "रविवार की सुबह नौ बजते ही सड़कें पूरी तरह सूनी हो जाती थीं और छत पर टीवी एंटीना हिलाने की आवाजें गूंजती थीं। "
        "नीचे से कोई जोर से चिल्लाता था, आ गया... जरा बाएं घुमाओ! "
        "ड्राइंग रूम में दरी बिछ जाती थी और पूरा मोहल्ला एक ही परिवार बनकर बैठ जाता था। "
        "रामायण और शक्तिमान देखने के बाद, नानी के हाथ का बना गरमा-गरम चूरन और आम का पना सबमें बंटता था। "
        "न कोई स्मार्टफोन था और न इंटरनेट की आपाधापी, बस सच्चे रिश्ते, मोहल्ले का अपनापन और सादगी से भरी हंसी थी। "
        "आज हाथ में स्क्रीन तो बहुत बड़ी है, लेकिन दिल को छू लेने वाली वो मासूमियत कहीं बहुत पीछे छूट गई है।"
    )

    is_dup, reason = is_duplicate_story(
        topic="नया टॉपिक जो पहले नहीं था",
        title=oct10_title,
        narration=oct10_narration,
    )
    assert is_dup is True
    assert "1f11f0f1294c36b9" in reason or "Title" in reason


def test_multi_signal_duplicate_detection_matrix(tmp_path, monkeypatch):
    """Test every duplicate detection signal: hash, title similarity, summary overlap, scene prompts, video hash."""
    test_hist_file = tmp_path / "history.json"
    test_hist_file.write_text("[]", encoding="utf-8")
    monkeypatch.setattr("pipeline.story_history.HISTORY_FILE", test_hist_file)

    sample_narration = "1990 के दौर में छत पर पानी का छिड़काव और चारपाई पर तारों भरी रात एक अलग ही सुकून देती थी।"
    sample_scenes = [
        "Scene 1: Rooftop water spray",
        "Scene 2: Charpai laid out",
        "Scene 3: Grandmother hand fan",
    ]

    save_history_entry(
        "90s_indian_nostalgia",
        topic="छत पर पानी का छिड़काव और चारपाई",
        title="छत पर पानी का छिड़काव और तारों भरी रात",
        narration=sample_narration,
        summary="1990 के दौर में छत पर पानी का छिड़काव और चारपाई पर तारों भरी रात…",
        video_id="video_abc",
        video_hash="vidhash_111",
        duration_s=42.0,
        run_id="run_001",
        scene_prompts=sample_scenes,
    )

    # 1. Exact Video Hash Duplicate
    is_dup, reason = is_duplicate_story("अनदेखा टॉपिक", "अनदेखा शीर्षक", video_hash="vidhash_111")
    assert is_dup is True
    assert "Identical video hash" in reason

    # 2. Exact Narration Hash Duplicate
    is_dup, reason = is_duplicate_story("अलग टॉपिक", "अलग शीर्षक", narration=sample_narration)
    assert is_dup is True
    assert "Identical script narration hash" in reason

    # 3. Exact Normalized Topic Duplicate
    is_dup, reason = is_duplicate_story("छत पर पानी का छिड़काव और चारपाई", "बिल्कुल अलग शीर्षक")
    assert is_dup is True
    assert "was already used previously" in reason

    # 4. Title Token Similarity Duplicate (> 70%)
    similar_title = "छत पर पानी का छिड़काव और तारों भरी वो रात"
    is_dup, reason = is_duplicate_story("नया टॉपिक", similar_title)
    assert is_dup is True
    assert "too similar" in reason

    # 5. Narration text substantial overlap (> 60%)
    overlapping_narration = "1990 के दौर में छत पर पानी का छिड़काव और चारपाई पर तारों भरी रात में दादी का हाथ का पंखा चलता था।"
    is_dup, reason = is_duplicate_story("नया टॉपिक", "नया शीर्षक", narration=overlapping_narration)
    assert is_dup is True
    assert "shares substantial content" in reason

    # 6. Scene Prompts overlap (> 50%)
    matching_scenes = [
        "Scene 1: Rooftop water spray",
        "Scene 2: Charpai laid out",
        "Scene 99: Novel scene description",
    ]
    is_dup, reason = is_duplicate_story(
        "नया टॉपिक",
        "नया शीर्षक",
        narration="एकदम अलग अनोखी कहानी जिसमें कोई पुराना शब्द नहीं है।",
        scene_prompts=matching_scenes,
    )
    assert is_dup is True
    assert "Scene prompts share" in reason


def test_novel_story_passes_duplicate_check(tmp_path, monkeypatch):
    """Ensure a genuinely novel story passes duplicate checks."""
    test_hist_file = tmp_path / "history.json"
    test_hist_file.write_text("[]", encoding="utf-8")
    monkeypatch.setattr("pipeline.story_history.HISTORY_FILE", test_hist_file)

    save_history_entry(
        "90s_indian_nostalgia",
        topic="रेडियो पर बिनाका गीतमाला",
        title="रेडियो पर बिनाका गीतमाला और अमीन सयानी की आवाज",
        narration="बुधवार की रात आठ बजे रेडियो सीलोन पर अमीन सयानी की खनकती हुई आवाज पूरे परिवार को जोड़ती थी।",
    )

    # Completely different story about cricket
    is_dup, reason = is_duplicate_story(
        topic="गली क्रिकेट और टेनिस बॉल",
        title="गली क्रिकेट, टूटी खिड़की और आखिरी गेंद पर चौका",
        narration="रविवार की दोपहर को जब बड़े सो जाते थे, तब संकरी गली में हमारा वर्ल्ड कप शुरू होता था।",
        scene_prompts=["Scene A: Batting with bricks", "Scene B: Ball breaking window"],
    )
    assert is_dup is False
    assert reason == ""


def test_fallback_catalog_never_returns_history_duplicates(tmp_path, monkeypatch):
    """Verify that _get_fallback_pack excludes any packs present in history."""
    test_hist_file = tmp_path / "history.json"
    test_hist_file.write_text("[]", encoding="utf-8")
    monkeypatch.setattr("pipeline.story_history.HISTORY_FILE", test_hist_file)

    preset = {"label": "90s Indian Nostalgia"}

    # Save the first 3 catalog packs into history
    for i in range(3):
        pack = FALLBACK_CATALOG[i]
        save_history_entry(
            "90s_indian_nostalgia",
            topic=pack.get("topic", f"topic_{i}"),
            title=pack["youtube_title"],
            narration=pack["full_narration"],
            scene_prompts=pack["image_prompts"],
        )

    # Request a fallback pack — it must NOT be any of the first 3
    chosen = _get_fallback_pack(preset, topic="nostalgia")
    used_titles = {FALLBACK_CATALOG[i]["youtube_title"] for i in range(3)}
    assert chosen["youtube_title"] not in used_titles
    assert len(chosen["image_prompts"]) == 6
    assert 130 <= len(chosen["full_narration"].split()) <= 185


def test_select_unique_topic_excludes_history_and_extra_exclude(tmp_path, monkeypatch):
    """Verify select_unique_topic avoids history topics and extra excluded topics."""
    test_hist_file = tmp_path / "history.json"
    test_hist_file.write_text("[]", encoding="utf-8")
    monkeypatch.setattr("pipeline.story_history.HISTORY_FILE", test_hist_file)

    preset = {
        "topic_pool": [
            "Topic Alpha",
            "Topic Beta",
            "Topic Gamma",
        ]
    }

    # Put Alpha in history
    save_history_entry("90s", topic="Topic Alpha", title="T1", narration="N1")

    # Exclude Beta via extra_exclude
    chosen = select_unique_topic(preset, extra_exclude={"Topic Beta"})
    assert chosen == "Topic Gamma"


def test_bounded_novelty_loop_in_orchestrator(tmp_path, monkeypatch):
    """Test run_short novelty loop: duplicate attempt is rejected, next novel attempt accepted."""
    test_hist_file = tmp_path / "history.json"
    test_hist_file.write_text("[]", encoding="utf-8")
    monkeypatch.setattr("pipeline.story_history.HISTORY_FILE", test_hist_file)

    # Pre-populate history with a known story
    save_history_entry(
        "90s_indian_nostalgia",
        topic="पुरानी कहानी का टॉपिक",
        title="पुरानी कहानी का शीर्षक",
        narration="पुरानी कहानी की पूरी नरेशन जो पहले अपलोड हो चुकी है।",
    )

    preset = {
        "label": "90s Indian Nostalgia",
        "topic_pool": ["Topic 1", "Topic 2", "Topic 3"],
        "min_words": 130,
        "max_words": 185,
        "segment_count": 6,
    }

    # Attempt 1 returns duplicate story; Attempt 2 returns novel story
    attempt1_pack = {
        "youtube_title": "पुरानी कहानी का शीर्षक",
        "youtube_description": "पुराना विवरण",
        "full_narration": "पुरानी कहानी की पूरी नरेशन जो पहले अपलोड हो चुकी है।",
        "image_prompts": ["p1", "p2", "p3", "p4", "p5", "p6"],
    }
    attempt2_pack = {
        "youtube_title": "एकदम नई अनूठी कहानी",
        "youtube_description": "नया विवरण",
        "full_narration": (
            "यह एक बिल्कुल नई कहानी है जिसमें कोई पुराना शब्द या वाक्य शामिल नहीं है। "
            "बचपन के वो दिन जब हम सब मिलकर शाम को खेलते थे और मां प्यार से बुलाती थी। "
            "न कोई चिंता थी और न कोई तनाव, बस हर पल में खुशियों का खजाना था। "
            "यह कहानी हर किसी के दिल को छू लेगी और पुरानी यादें ताजा कर देगी। "
            "आइए उस दौर को फिर से याद करें और अपने बचपन की उन गलियों में वापस चलें। "
            "न कोई स्मार्टफोन था और न कोई इंटरनेट, बस सच्चे दिल और सच्चा प्यार था जो आज कहीं खो गया है। "
            "वो मासूमियत और वो अपनापन हमेशा याद रहेगा।"
        ),
        "image_prompts": ["np1", "np2", "np3", "np4", "np5", "np6"],
    }

    mock_generate = MagicMock(side_effect=[attempt1_pack, attempt2_pack])

    with patch("pipeline.groq_script.generate_short_pack", mock_generate):
        # Verify attempt 1 was detected as duplicate
        is_dup1, _ = is_duplicate_story(
            "Topic 1",
            attempt1_pack["youtube_title"],
            attempt1_pack["full_narration"],
            scene_prompts=attempt1_pack["image_prompts"],
        )
        assert is_dup1 is True

        # Verify attempt 2 is accepted
        is_dup2, _ = is_duplicate_story(
            "Topic 2",
            attempt2_pack["youtube_title"],
            attempt2_pack["full_narration"],
            scene_prompts=attempt2_pack["image_prompts"],
        )
        assert is_dup2 is False


def test_pre_upload_gate_blocks_duplicate_video_hash(tmp_path, monkeypatch):
    """Ensure pre-upload check blocks when video_hash already exists."""
    test_hist_file = tmp_path / "history.json"
    test_hist_file.write_text("[]", encoding="utf-8")
    monkeypatch.setattr("pipeline.story_history.HISTORY_FILE", test_hist_file)

    save_history_entry(
        "90s_indian_nostalgia",
        topic="T1",
        title="Title 1",
        narration="N1",
        video_hash="deadbeef12345678",
        video_id="prior_vid_id",
    )

    # Attempting to upload another video that has the same video hash
    is_dup, reason = is_duplicate_story(
        topic="Brand new topic",
        title="Brand new title",
        narration="Brand new narration text here.",
        video_hash="deadbeef12345678",
    )
    assert is_dup is True
    assert "Identical video hash already uploaded: deadbeef12345678" in reason


def test_history_schema_has_all_required_metadata(tmp_path, monkeypatch):
    """Confirm save_history_entry persists IST/UTC timestamps, scene prompt hashes, and normalized fields."""
    test_hist_file = tmp_path / "history.json"
    test_hist_file.write_text("[]", encoding="utf-8")
    monkeypatch.setattr("pipeline.story_history.HISTORY_FILE", test_hist_file)

    scene_prompts = [
        "Prompt A: Vintage living room",
        "Prompt B: Terrace sunset",
    ]

    save_history_entry(
        "90s_indian_nostalgia",
        topic="  गर्मियों की शाम  ",
        title="  बर्फ का गोला  ",
        narration="एक लंबी कहानी बर्फ के गोले के बारे में...",
        summary="कहानी का सारांश",
        video_id="YT_XYZ123",
        video_url="https://www.youtube.com/shorts/YT_XYZ123",
        video_hash="vhash_999",
        duration_s=41.5,
        run_id="run_20261011_01",
        scene_prompts=scene_prompts,
    )

    records = load_history()
    assert len(records) == 1
    rec = records[0]

    assert rec["channel"] == "90s_indian_nostalgia"
    assert rec["run_id"] == "run_20261011_01"
    assert rec["topic"] == "गर्मियों की शाम"
    assert rec["title"] == "बर्फ का गोला"
    assert rec["topic_normalized"] == "गर्मियों की शाम"
    assert rec["title_normalized"] == "बर्फ का गोला"
    assert "timestamp_utc" in rec and "+00:00" in rec["timestamp_utc"] or "Z" in rec["timestamp_utc"]
    assert "timestamp_ist" in rec and "+05:30" in rec["timestamp_ist"]
    assert rec["video_hash"] == "vhash_999"
    assert rec["video_id"] == "YT_XYZ123"
    assert rec["duration_s"] == 41.5
    assert len(rec["scene_prompt_hashes"]) == 2
    assert rec["scene_prompt_hashes"][0] == compute_sha256(scene_prompts[0])[:16]


def test_title_variation_of_oct10_upload_is_rejected():
    """Verify title with slight variation of Oct 10 title is rejected via token similarity."""
    # Oct 10 title: "जब टीवी का मतलब सिर्फ रविवार और रामायण था"
    slight_variation = "जब टीवी का मतलब सिर्फ रविवार और रामायण"
    is_dup, reason = is_duplicate_story(
        topic="अनोखा नया विषय",
        title=slight_variation,
        narration="पूरी तरह नई कहानी जिसमें कोई पुराना शब्द नहीं है।",
    )
    assert is_dup is True
    assert "too similar" in reason or "matches" in reason


def test_orchestrator_raises_when_all_attempts_exhausted(tmp_path, monkeypatch):
    """Verify main() in run_short fails closed with failure notification if all 4 attempts are duplicates."""
    from scripts.run_short import main

    test_hist_file = tmp_path / "history.json"
    test_hist_file.write_text("[]", encoding="utf-8")
    monkeypatch.setattr("pipeline.story_history.HISTORY_FILE", test_hist_file)

    # Pre-populate history
    save_history_entry(
        "90s_indian_nostalgia",
        topic="बार-बार दोहराया जाने वाला टॉपिक",
        title="बार-बार दोहराया जाने वाला शीर्षक",
        narration="एक ही पुरानी कहानी जो हर बार जनरेट होती है...",
    )

    long_dup_narration = "पुरानी कहानी की पूरी लंबी नरेशन जो पहले अपलोड हो चुकी है और जिसे अब फिर से नहीं दोहराना चाहिए। " * 8
    dup_pack = {
        "youtube_title": "बार-बार दोहराया जाने वाला शीर्षक",
        "youtube_description": "पुराना विवरण",
        "full_narration": long_dup_narration,
        "image_prompts": ["p1", "p2", "p3", "p4", "p5", "p6"],
    }

    # Pre-populate history with this exact title
    save_history_entry(
        "90s_indian_nostalgia",
        topic="बार-बार दोहराया जाने वाला टॉपिक",
        title="बार-बार दोहराया जाने वाला शीर्षक",
        narration=long_dup_narration,
    )

    mock_send_fail = MagicMock()
    monkeypatch.setattr("scripts.run_short.send_failure_alert_email", mock_send_fail)
    monkeypatch.setattr("scripts.run_short.generate_short_pack", MagicMock(return_value=dup_pack))
    monkeypatch.setattr("sys.argv", ["run_short.py", "--channel", "90s_indian_nostalgia"])

    with pytest.raises(SystemExit):
        main()

    # Verify failure email was dispatched
    assert mock_send_fail.called
    call_kwargs = mock_send_fail.call_args[1]
    assert call_kwargs["stage"] == "script_generation"
    assert "Failed to generate a novel story after 4 attempts" in call_kwargs["error_message"]

