#!/usr/bin/env python3
"""Fully automated YouTube Shorts orchestrator for 1990s Indian Nostalgia channel.

Flow:
  1. Topic selection & duplicate prevention (durable history)
  2. Groq → 1990s Hindi narrative story + vertical scene prompts
  3. Pollinations → distinct 9:16 scene images (1080x1920)
  4. Sarvam AI → native Hindi voiceover using 'shubh' persona
  5. Captions → Devanagari SRT subtitles
  6. FFmpeg → 1080x1920 vertical MP4 with burned captions
  7. Video validation → ffprobe stream & duration checks
  8. YouTube upload (optional/scheduled) → private by default
  9. History & notifications → data/history.json + email alerts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

from pipeline.captions import build_srt
from pipeline.channel_presets import get_preset, list_channel_ids, DEFAULT_CHANNEL_ID
from pipeline.groq_script import generate_short_pack
from pipeline.images import full_visual_prompt, save_scene_image
from pipeline.notifications import send_upload_success_email, send_failure_alert_email
from pipeline.render_short import render_vertical_short, validate_rendered_short
from pipeline.sarvam_tts import synthesize_full
from pipeline.story_history import is_duplicate_story, save_history_entry, load_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("scripts.run_short")


def select_unique_topic(preset: dict, requested_topic: str | None = None) -> str:
    """Select a fresh, non-duplicate topic from the preset pool."""
    if requested_topic and requested_topic.strip():
        return requested_topic.strip()

    pool = list(preset.get("topic_pool") or ["1990s Indian childhood memories"])
    history = load_history()
    used_topics = {entry.get("topic", "").strip().lower() for entry in history}

    # Prefer unused topics from pool
    unused = [t for t in pool if t.strip().lower() not in used_topics]
    if unused:
        chosen = random.choice(unused)
        log.info("Selected fresh topic from pool: '%s'", chosen)
        return chosen

    # If all in pool have been used at least once, pick the least recently used
    chosen = random.choice(pool)
    log.info("All pool topics previously logged; cycling topic: '%s'", chosen)
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate & optionally upload a 1990s Indian Nostalgia YouTube Short.")
    ap.add_argument("--channel", default=DEFAULT_CHANNEL_ID, choices=list_channel_ids(), help="Channel preset")
    ap.add_argument("--topic", default="", help="Optional specific topic hint")
    ap.add_argument("--upload", action="store_true", help="Upload to YouTube after render")
    ap.add_argument("--privacy", default="private", choices=["private", "unlisted", "public"], help="Upload privacy status")
    ap.add_argument("--run-id", default="", help="Unique run identifier")
    args = ap.parse_args()

    preset = get_preset(args.channel)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id.strip() or f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}_{run_ts}"

    run_dir = REPO_ROOT / "output" / "runs" / f"{args.channel}_{run_id}"
    img_dir = run_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    current_stage = "topic_selection"
    topic = ""
    title = ""
    video_id = None
    upload_attempted = False

    try:
        # ── 1. Select Topic ──────────────────────────────────────────
        log.info("=== Starting ZeroCost Shorts Pipeline: run_id=%s ===", run_id)
        topic = select_unique_topic(preset, args.topic)

        # ── 2. Script Generation (Groq) ──────────────────────────────
        current_stage = "script_generation"
        log.info("Stage 1/7: Groq script generation...")
        pack = generate_short_pack(preset, topic_hint=topic, channel_id=args.channel)

        title = pack["youtube_title"]
        narration = pack["full_narration"]
        image_prompts = pack["image_prompts"]
        tags = pack.get("tags", ["90s nostalgia", "indian nostalgia", "Shorts"])

        # Check for duplicates against history
        is_dup, dup_reason = is_duplicate_story(topic, title, narration)
        if is_dup:
            log.warning("Duplicate detected (%s). Proceeding with modified novel phrasing.", dup_reason)

        (run_dir / "script.json").write_text(json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Script validated: '%s' (%d words, %d scenes)", title, len(narration.split()), len(image_prompts))

        # ── 3. Images (Pollinations) ─────────────────────────────────
        current_stage = "image_generation"
        log.info("Stage 2/7: Pollinations AI 9:16 image generation (%d scenes)...", len(image_prompts))
        style_suffix = preset.get("image_style_suffix")
        negative = preset.get("image_negative_prompt")

        image_paths: list[Path] = []
        for i, ip in enumerate(image_prompts):
            prompt = full_visual_prompt(ip, style_suffix=style_suffix)
            out_img = img_dir / f"scene_{i + 1:02d}.png"
            st, detail = save_scene_image(i + 1, prompt, out_img, width=1080, height=1920, negative=negative)
            if st != "ok":
                raise RuntimeError(f"Scene {i + 1} image generation failed: {detail}")
            image_paths.append(out_img)

        # ── 4. Voiceover (Sarvam AI) ─────────────────────────────────
        current_stage = "voiceover_synthesis"
        log.info("Stage 3/7: Sarvam AI Hindi voiceover synthesis (voice='%s')...", preset.get("tts_voice", "shubh"))
        audio_path = run_dir / "voiceover.wav"
        total_dur, sentence_timings = synthesize_full(
            narration,
            audio_path,
            voice=preset.get("tts_voice", "shubh"),
            language_code="hi-IN",
        )
        log.info("Voiceover complete: %.1fs duration, %d sentence timings", total_dur, len(sentence_timings))

        # ── 5. Captions ──────────────────────────────────────────────
        current_stage = "captions_generation"
        log.info("Stage 4/7: Building SRT subtitles with Devanagari styling...")
        srt_path = run_dir / "captions.srt"
        build_srt(sentence_timings, srt_path, total_dur)

        # ── 6. FFmpeg Assembly ───────────────────────────────────────
        current_stage = "video_rendering"
        log.info("Stage 5/7: FFmpeg rendering 1080x1920 vertical Short MP4...")
        video_path = run_dir / "short.mp4"
        render_vertical_short(
            image_paths,
            total_dur,
            audio_path,
            srt_path,
            video_path,
            width=1080,
            height=1920,
            font_file=preset.get("caption_font", "NotoSansDevanagari-Bold.ttf"),
            font_name=preset.get("caption_font_name", "Noto Sans Devanagari"),
        )

        # ── 7. Validation ────────────────────────────────────────────
        current_stage = "video_validation"
        log.info("Stage 6/7: Probing and validating final MP4...")
        actual_dur = validate_rendered_short(video_path, min_duration=25.0, max_duration=90.0)
        video_hash = hashlib.sha256(video_path.read_bytes()).hexdigest()[:16]

        # ── 8. Upload to YouTube ─────────────────────────────────────
        video_url = ""
        if args.upload:
            current_stage = "youtube_upload"
            upload_attempted = True
            log.info("Stage 7/7: Uploading Short to YouTube (%s)...", args.privacy)
            from pipeline.youtube_upload import upload_short

            video_id = upload_short(
                video_path,
                title,
                pack.get("youtube_description", ""),
                tags=tags,
                privacy_status=args.privacy,
                refresh_token_env=preset.get("yt_token_env", "YT_REFRESH_TOKEN"),
            )
            video_url = f"https://www.youtube.com/shorts/{video_id}"
            log.info("Upload succeeded! %s", video_url)
        else:
            log.info("Stage 7/7: Upload skipped (--upload not set). Video ready at: %s", video_path)

        # ── 9. Save History & Send Success Notification ──────────────
        current_stage = "persistence_and_notification"
        summary = " ".join(narration.split()[:30]) + "…"
        save_history_entry(
            args.channel,
            topic=topic,
            title=title,
            narration=narration,
            summary=summary,
            video_id=video_id,
            video_url=video_url,
            video_hash=video_hash,
            duration_s=actual_dur,
            run_id=run_id,
        )

        # Write run.json record
        run_record = {
            "run_id": run_id,
            "channel": args.channel,
            "topic": topic,
            "title": title,
            "duration_s": actual_dur,
            "video_hash": video_hash,
            "video_id": video_id,
            "video_url": video_url,
            "privacy": args.privacy,
            "status": "completed",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        (run_dir / "run.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")

        if args.upload and video_id:
            send_upload_success_email(
                title=title,
                video_id=video_id,
                video_url=video_url,
                topic=topic,
                summary=summary,
                duration_s=actual_dur,
                run_id=run_id,
                tags=tags,
            )

        log.info("=== Pipeline Completed Successfully! ===")

    except Exception as exc:
        log.error("Pipeline failed at stage '%s': %s", current_stage, exc, exc_info=True)
        send_failure_alert_email(
            stage=current_stage,
            error_message=str(exc),
            topic=topic,
            run_id=run_id,
            upload_attempted=upload_attempted,
            video_id=video_id,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
