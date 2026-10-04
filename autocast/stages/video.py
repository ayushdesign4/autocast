"""Stage: video — multi-scene AI video clip normalization, concat, and background music mix.

Bypasses the legacy image-to-video Ken Burns pipeline.
Takes the independently generated scene MP4 clips (`assets/scene_XXX.mp4`),
normalizes them for reliable concatenation, stitches them in exact scene order
into `reel.mp4`, and mixes CC0 background ambient music under the native dialogue
audio into `final.mp4`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from autocast.config import Config
from autocast.ffmpeg.mux import (
    build_concat_cmd,
    build_normalize_clip_cmd,
    build_scored_native_mux_cmd,
    write_concat_list,
)
from autocast.ffmpeg.run import probe_duration, run_ffmpeg
from autocast.spine import Run, Video

log = logging.getLogger("autocast.stages.video")

STAGE = "video"


def _run_or_log(cmd: list[str], *, dry_run: bool, touch: Path | None = None) -> None:
    """Run the FFmpeg command, or (dry-run) log it and touch the output file."""
    if dry_run:
        log.info("ffmpeg cmd: %s", " ".join(cmd))
        if touch is not None:
            touch.parent.mkdir(parents=True, exist_ok=True)
            touch.touch()
        return
    run_ffmpeg(cmd)


def run(spine: Run, cfg: Config, *, dry_run: bool = False) -> Run:
    items = spine.scenes or spine.shots
    if not items:
        raise ValueError("video stage: no scenes/shots found (run direction and video_gen first)")

    run_dir = cfg.run_dir(spine.run_id)
    assets_dir = cfg.assets_dir(spine.run_id)
    assets_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect and inspect per-scene MP4 clips
    clip_abs_paths: list[str] = []
    for i, item in enumerate(items):
        clip_rel = getattr(item, "clip_path", None)
        if not clip_rel:
            scene_num = getattr(item, "scene_number", i + 1)
            clip_rel = f"assets/scene_{scene_num:03d}.mp4"

        clip_abs = run_dir / clip_rel
        if not dry_run and (not clip_abs.exists() or clip_abs.stat().st_size == 0):
            raise FileNotFoundError(f"video stage: missing or empty scene clip at {clip_abs}")
        elif dry_run and not clip_abs.exists():
            clip_abs.parent.mkdir(parents=True, exist_ok=True)
            clip_abs.touch()

        # If running real FFmpeg, optionally normalize clip to ensure consistent stream parameters
        normalized_abs = assets_dir / f"norm_{clip_abs.name}"
        if not dry_run:
            norm_cmd = build_normalize_clip_cmd(
                in_path=str(clip_abs),
                out_path=str(normalized_abs),
                width=cfg.width,
                height=cfg.height,
                fps=cfg.fps,
            )
            try:
                _run_or_log(norm_cmd, dry_run=False)
                clip_abs_paths.append(str(normalized_abs))
            except Exception as exc:  # noqa: BLE001
                log.warning("video: clip normalization failed for %s (%s); using original", clip_abs.name, exc)
                clip_abs_paths.append(str(clip_abs))
        else:
            clip_abs_paths.append(str(clip_abs))

    # 2. Concat all scene clips into reel.mp4
    concat_list = write_concat_list(assets_dir / "concat.txt", clip_abs_paths)
    reel_abs = assets_dir / "reel.mp4"
    concat_cmd = build_concat_cmd(concat_list_path=str(concat_list), out_path=str(reel_abs))
    _run_or_log(concat_cmd, dry_run=dry_run, touch=reel_abs)

    # 3. Mix CC0 ambient background music under native scene dialogue
    music_rel = spine.assets.music_path if (spine.assets and spine.assets.music_path) else None
    music_abs = str(run_dir / music_rel) if (music_rel and (run_dir / music_rel).exists()) else None

    final_abs = assets_dir / "final.mp4"
    mux_cmd = build_scored_native_mux_cmd(
        reel_path=str(reel_abs),
        out_path=str(final_abs),
        music_path=music_abs,
        music_gain_db=-22.0,  # Duck music well underneath spoken dialogue
    )
    _run_or_log(mux_cmd, dry_run=dry_run, touch=final_abs)

    # Calculate final stats
    size_bytes = final_abs.stat().st_size if final_abs.exists() else 0
    duration_s = probe_duration(final_abs) if (not dry_run and final_abs.exists()) else sum(getattr(s, "duration_s", 6.5) for s in items)

    spine.video = Video(
        final_path="assets/final.mp4",
        width=cfg.width,
        height=cfg.height,
        fps=cfg.fps,
        duration_s=round(duration_s, 2),
        size_bytes=size_bytes,
    )
    log.info("video: assembled final.mp4 with native dialogue (%d clips, %d bytes)", len(items), size_bytes)
    return spine
