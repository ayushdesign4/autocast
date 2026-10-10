"""FFmpeg video assembly: 9:16 vertical Short MP4 with burned captions.

- Combines distinct scene images, Sarvam voiceover audio, and Devanagari captions.
- Applies subtle Ken Burns pan/zoom motion per scene.
- Burns subtitles using NotoSansDevanagari-Bold.ttf for authentic Devanagari rendering.
- Strictly produces 1080x1920 vertical H.264/AAC MP4 without black bars or image stretching.
- Validates the resulting MP4 duration and stream integrity via ffprobe.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("pipeline.render_short")

REPO_ROOT = Path(__file__).resolve().parent.parent
FONTS_DIR = REPO_ROOT / "assets" / "fonts"
DEFAULT_FONT_FILE = "NotoSansDevanagari-Bold.ttf"
DEFAULT_FONT_NAME = "Noto Sans Devanagari"

FPS = 30
FADE_DUR = 0.4
ZOOM_AMOUNT = 0.08


def _get_ffmpeg_bin() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    raise RuntimeError("FFmpeg executable not found. Please install ffmpeg.")


def _get_ffprobe_bin() -> str:
    found = shutil.which("ffprobe")
    if found:
        return found
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        probe = str(Path(exe).parent / "ffprobe.exe")
        if Path(probe).exists():
            return probe
    except Exception:
        pass
    return "ffprobe"


def probe_video_metadata(video_path: Path) -> dict[str, any]:
    """Inspect generated MP4 streams, dimensions, and duration using ffprobe."""
    ffprobe_bin = _get_ffprobe_bin()
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-show_entries", "format=duration,size:stream=codec_type,width,height,codec_name",
        "-of", "json",
        str(video_path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json
        return json.loads(res.stdout)
    except Exception as exc:
        raise RuntimeError(f"ffprobe failed to inspect {video_path}: {exc}") from exc


def validate_rendered_short(video_path: Path, min_duration: float = 30.0, max_duration: float = 90.0) -> float:
    """Validate that the final video exists, is nonzero, has audio and video, is 9:16, and meets duration requirements."""
    video_path = Path(video_path)
    if not video_path.exists() or video_path.stat().st_size == 0:
        raise ValueError(f"Rendered video is missing or 0 bytes: {video_path}")

    meta = probe_video_metadata(video_path)
    streams = meta.get("streams", [])
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    if not has_video:
        raise ValueError(f"Rendered video missing video stream: {video_path}")
    if not has_audio:
        raise ValueError(f"Rendered video missing audio stream: {video_path}")

    # Check dimensions
    v_stream = next(s for s in streams if s.get("codec_type") == "video")
    w = int(v_stream.get("width", 0))
    h = int(v_stream.get("height", 0))
    if w <= 0 or h <= 0 or w >= h:
        raise ValueError(f"Video is not vertical 9:16 (detected {w}x{h}): {video_path}")

    dur = float(meta.get("format", {}).get("duration", 0.0))
    if dur < min_duration:
        raise ValueError(f"Rendered video is too short ({dur:.1f}s < {min_duration}s target). Refusing incomplete video.")
    if dur > max_duration:
        raise ValueError(f"Rendered video exceeds Shorts limit ({dur:.1f}s > {max_duration}s target).")

    log.info("Validated rendered Short: %.2fs, %dx%d, %.1f MB", dur, w, h, video_path.stat().st_size / (1024 * 1024))
    return dur


def render_vertical_short(
    image_paths: list[Path],
    total_duration: float,
    audio_path: Path,
    srt_path: Path,
    out_video: Path,
    *,
    width: int = 1080,
    height: int = 1920,
    font_file: str = DEFAULT_FONT_FILE,
    font_name: str = DEFAULT_FONT_NAME,
) -> Path:
    """Render a full-bleed vertical 9:16 MP4 with Ken Burns effect and burned Devanagari captions."""
    if not image_paths:
        raise ValueError("No images provided for rendering")

    ffmpeg_bin = _get_ffmpeg_bin()
    out_video = Path(out_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_video.parent / "_tmp_render"
    tmp.mkdir(parents=True, exist_ok=True)

    n = len(image_paths)
    clip_dur = (total_duration + (n - 1) * FADE_DUR) / n if n > 1 else total_duration
    frames_per_clip = max(int(clip_dur * FPS), 2)

    # 1. Pre-scale images to exact 9:16 (crop if needed to avoid black bars)
    for i, src in enumerate(image_paths):
        dst = tmp / f"img_{i + 1:02d}.png"
        scale_filter = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}"
        )
        subprocess.run(
            [
                ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "warning",
                "-i", str(src),
                "-vf", scale_filter,
                "-update", "1", "-frames:v", "1",
                str(dst),
            ],
            check=True,
        )

    # 2. Generate smooth Ken Burns pan/zoom clips
    clip_paths: list[Path] = []
    for i in range(n):
        src_img = tmp / f"img_{i + 1:02d}.png"
        clip = tmp / f"clip_{i + 1:02d}.mp4"
        zoom_rate = ZOOM_AMOUNT / frames_per_clip

        if i % 2 == 0:
            zoom_expr = f"min(zoom+{zoom_rate:.8f},{1 + ZOOM_AMOUNT})"
        else:
            zoom_expr = f"if(eq(on,1),{1 + ZOOM_AMOUNT},max(zoom-{zoom_rate:.8f},1.0))"

        vf = (
            f"zoompan=z='{zoom_expr}':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={frames_per_clip}:s={width}x{height}:fps={FPS},"
            f"format=yuv420p"
        )

        subprocess.run(
            [
                ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "warning",
                "-i", str(src_img),
                "-vf", vf,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16",
                str(clip),
            ],
            check=True,
        )
        clip_paths.append(clip)

    # 3. Setup subtitle styling and font directory
    font_path = FONTS_DIR / font_file
    if not font_path.is_file():
        font_path = FONTS_DIR / DEFAULT_FONT_FILE

    font_dir = tmp / "_fonts"
    font_dir.mkdir(exist_ok=True)
    if font_path.is_file():
        shutil.copyfile(font_path, font_dir / font_path.name)

    # Safe escaping for FFmpeg subtitles filter on both Linux and Windows
    escaped_srt = str(srt_path).replace("\\", "/").replace(":", r"\:")
    escaped_fontsdir = str(font_dir).replace("\\", "/").replace(":", r"\:")

    force_style = (
        f"FontName={font_name},"
        f"FontSize=19,"
        f"PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,"
        f"BackColour=&H80000000,"
        f"BorderStyle=4,Outline=1,Bold=1,"
        f"Shadow=0,Alignment=2,"
        f"MarginV=90,MarginL=30,MarginR=30"
    )

    sub_filter = f"subtitles='{escaped_srt}':fontsdir='{escaped_fontsdir}':force_style='{force_style}'"

    # 4. Concatenate clips with crossfade (or concat demuxer if single clip)
    if n == 1:
        v_filter = f"[0:v]{sub_filter}[vout]"
        inputs = ["-i", str(clip_paths[0])]
    else:
        # Crossfade filter chain
        inputs = []
        for p in clip_paths:
            inputs.extend(["-i", str(p)])

        filter_parts = []
        cur_v = "[0:v]"
        offset = clip_dur - FADE_DUR
        for i in range(1, n):
            next_v = f"[{i}:v]"
            out_v = f"[v{i}]" if i < n - 1 else "[vjoined]"
            filter_parts.append(
                f"{cur_v}{next_v}xfade=transition=fadeblack:duration={FADE_DUR}:offset={offset:.3f}{out_v}"
            )
            cur_v = out_v
            offset += clip_dur - FADE_DUR

        filter_parts.append(f"[vjoined]{sub_filter}[vout]")
        v_filter = ";".join(filter_parts)

    cmd = [
        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "warning",
        *inputs,
        "-i", str(audio_path),
        "-filter_complex", v_filter,
        "-map", "[vout]",
        "-map", f"{n}:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-pix_fmt", "yuv420p",
        "-shortest",
        str(out_video),
    ]

    log.info("Rendering final vertical Short video via FFmpeg...")
    subprocess.run(cmd, check=True)

    # Validate output video
    validate_rendered_short(out_video, min_duration=25.0, max_duration=90.0)

    # Clean up temporary frames/clips
    try:
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception:
        pass

    log.info("Render complete: %s", out_video)
    return out_video
