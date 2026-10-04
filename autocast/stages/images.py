"""Stage: images / video_gen — direct AI video generation per scene.

Replaces the legacy still-image generation with direct AI video generation
(Google Gemini Veo). Each scene generates an independent, natively-voiced
MP4 clip (`assets/scene_XXX.mp4`) and an extracted preview frame.

Resumability:
Checks each scene's status and local file on disk before generation.
If generation failed previously at scene 5 of 10, already completed scenes 1-4
are skipped on rerun, and state is persisted after every scene.
"""

from __future__ import annotations

import logging
from pathlib import Path

from autocast.config import Config
from autocast.providers.cascade import run_with_fallback
from autocast.providers.video import build_video_providers
from autocast.spine import Run

log = logging.getLogger("autocast.stages.images")

STAGE = "images"


def run(spine: Run, cfg: Config, *, dry_run: bool = False) -> Run:
    items = spine.scenes or spine.shots
    if not items:
        raise ValueError("video generation stage: no scenes or shots found (run direction first)")

    assets_dir = cfg.assets_dir(spine.run_id)
    assets_dir.mkdir(parents=True, exist_ok=True)
    run_json = cfg.run_json_path(spine.run_id)
    providers_used: set[str] = set()

    for i, item in enumerate(items):
        scene_num = getattr(item, "scene_number", None) or (i + 1)
        rel_mp4 = f"assets/scene_{scene_num:03d}.mp4"
        rel_png = f"assets/scene_{scene_num:03d}.png"
        abs_mp4 = str(assets_dir / f"scene_{scene_num:03d}.mp4")
        abs_png = str(assets_dir / f"scene_{scene_num:03d}.png")

        # Resumability check: if already completed and non-empty file exists, skip
        if (
            getattr(item, "status", None) == "completed"
            and Path(abs_mp4).exists()
            and Path(abs_mp4).stat().st_size > 0
        ):
            log.info("scene %d: already completed -> skip", scene_num)
            continue

        prompt = getattr(item, "video_prompt", None) or getattr(item, "image_prompt", "")
        duration_s = getattr(item, "duration_s", 6.5)

        log.info("scene %d: generating video clip (duration=%.1fs)...", scene_num, duration_s)

        providers = build_video_providers(
            cfg,
            prompt=prompt,
            out_mp4=abs_mp4,
            out_png=abs_png,
            duration_s=duration_s,
            dry_run=dry_run,
        )
        result = run_with_fallback(providers)
        providers_used.add(result.provider_used)

        # Update scene record
        if hasattr(item, "clip_path"):
            item.clip_path = rel_mp4
        if hasattr(item, "frame_path"):
            item.frame_path = rel_png
        if hasattr(item, "image_path"):
            item.image_path = rel_png
        if hasattr(item, "status"):
            item.status = "completed"

        # Keep spine.shots synchronized
        if i < len(spine.shots):
            shot = spine.shots[i]
            shot.clip_path = rel_mp4
            shot.image_path = rel_png
            shot.status = "completed"

        # Incremental checkpoint: save spine immediately so partial progress is resilient
        spine.save(run_json)

    provider_label = ",".join(sorted(providers_used)) or "gemini-veo"
    spine.stage(STAGE).provider_used = provider_label
    log.info("video generation: completed %d scenes via %s", len(items), provider_label)
    return spine
