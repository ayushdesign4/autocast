"""Stage: tts — audio pipeline coordinator.

In the AI video workflow, native video-model audio is the PRIMARY audio source:
dialogue, sound effects, and ambient atmosphere are generated directly by the video
model in sync with character animation.

Per requirement:
- A separate TTS layer is NOT added by default to prevent duplicate voiceovers.
- If fallback TTS is ever required, it is kept explicitly disabled unless configured.
- Sets spine.audio and timestamps so the data spine remains complete and coherent.
"""

from __future__ import annotations

import logging
from pathlib import Path

from autocast.config import Config
from autocast.spine import Audio, Run

log = logging.getLogger("autocast.stages.tts")

STAGE = "tts"


def _assign_shot_audio_windows(spine: Run) -> float:
    """Lay shots/scenes end-to-end on the timeline using their durations."""
    cursor = 0.0
    for shot in spine.shots:
        shot.audio_start_s = round(cursor, 3)
        cursor += shot.duration_s
        shot.audio_end_s = round(cursor, 3)
    return round(cursor, 3)


def run(spine: Run, cfg: Config, *, dry_run: bool = False) -> Run:
    if spine.script is None:
        raise ValueError("tts stage: spine.script missing (run script first)")
    if not spine.shots:
        raise ValueError("tts stage: spine.shots empty (run direction first)")

    total_len = _assign_shot_audio_windows(spine) or float(cfg.target_len_s)
    assets_dir = cfg.assets_dir(spine.run_id)
    assets_dir.mkdir(parents=True, exist_ok=True)

    voice_wav = assets_dir / "voice.wav"

    # Native video audio is the primary source: touch a silent marker wav if absent
    if not voice_wav.exists():
        voice_wav.touch()

    provider_used = "native-video-audio"
    spine.audio = Audio(
        voice_path="assets/voice.wav",
        duration_s=total_len,
        word_timings_path=None,
        provider=provider_used,
    )
    spine.stage(STAGE).provider_used = provider_used
    log.info("tts: using native video model audio (%.1fs duration, provider=%s)", total_len, provider_used)
    return spine
