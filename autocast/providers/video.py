"""Dedicated AI Video Generation Provider — Agnes AI (Primary) & Gemini Veo (Alternative).

Direct text-to-video generation with natively synchronized audio/dialogue.
- Primary provider: Agnes AI (`agnes-video-2.5` via https://apihub.agnes-ai.com)
- Alternative/fallback provider: Google Gemini Veo (`veo-3.1-generate-preview`)
- Keyless/dry-run fallback: synthetic FFmpeg clip generation for offline CI and testing.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx

from autocast.config import Config
from autocast.providers.cascade import Provider
from autocast.seeds import ANTI_TEXT_PROMPT, NEGATIVE_PROMPT

log = logging.getLogger("autocast.providers.video")

DEFAULT_AGNES_MODEL = "agnes-video-2.5"
_AGNES_VIDEO_MODELS = ("agnes-video-2.5", "agnes-video-2.5-flash", "agnes-video", "agnes-video-v2.0")
DEFAULT_AGNES_API_BASE = "https://apihub.agnes-ai.com"
DEFAULT_VEO_MODEL = "veo-3.1-generate-preview"


def _get_ffmpeg_bin() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def _generate_dry_clip(out_mp4: str, out_png: str | None = None, *, duration_s: float = 6.0) -> str:
    """Generate a clean placeholder scene clip for dry-run/testing without API keys."""
    mp4_path = Path(out_mp4)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = _get_ffmpeg_bin()
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "lavfi", "-i", f"color=c=0x1f2937:s=1280x720:d={duration_s:.2f}:r=30",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration_s:.2f}:sample_rate=48000",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            str(mp4_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            log.info("video_gen[dry]: generated synthetic mp4 via FFmpeg at %s", mp4_path.name)
        except Exception:  # noqa: BLE001
            mp4_path.touch()
    else:
        mp4_path.touch()
        log.info("video_gen[dry]: touched placeholder clip %s", mp4_path.name)

    if out_png:
        png_path = Path(out_png)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image

            Image.new("RGB", (1280, 720), (31, 41, 55)).save(png_path, "PNG")
        except Exception:  # noqa: BLE001
            png_path.touch()

    return str(mp4_path)


def _extract_first_frame(video_path: str, out_png: str) -> None:
    """Extract first frame of an MP4 as a preview PNG for thumbnails and dashboard."""
    png_path = Path(out_png)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = _get_ffmpeg_bin()
    if ffmpeg_bin and Path(video_path).exists() and Path(video_path).stat().st_size > 0:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-ss", "00:00:00.000",
            "-i", video_path,
            "-vframes", "1",
            str(png_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return
        except Exception as exc:  # noqa: BLE001
            log.warning("video_gen: frame extraction failed (%s); writing fallback PNG", exc)

    try:
        from PIL import Image

        Image.new("RGB", (1280, 720), (45, 55, 72)).save(png_path, "PNG")
    except Exception:  # noqa: BLE001
        png_path.touch()


def _ensure_anti_text_prompt(prompt: str) -> str:
    """Guarantee that strict anti-text instructions are always present in the video prompt."""
    result = prompt
    if ANTI_TEXT_PROMPT not in result:
        result = f"{result} {ANTI_TEXT_PROMPT}"
    if "Negative prompt" not in result:
        result = f"{result} Negative prompt: {NEGATIVE_PROMPT}."
    return result


class AgnesRateLimiter:
    """Enforces minimum spacing between consecutive video-generation (POST /v1/videos) requests.

    Does not affect status polling requests (GET /agnesapi).
    """

    def __init__(self, min_interval_s: float = 60.0):
        self.min_interval_s = min_interval_s
        self._last_request_time: float | None = None

    def wait_if_needed(self, sleep_fn=None) -> float:
        """Wait only for the remaining time if less than min_interval_s has elapsed.

        Returns the number of seconds waited.
        """
        if sleep_fn is None:
            sleep_fn = time.sleep

        if self._last_request_time is None:
            return 0.0

        elapsed = time.monotonic() - self._last_request_time
        remaining = self.min_interval_s - elapsed
        if remaining > 0:
            log.info("video_gen[agnes]: rate-limiter waiting %.1fs to respect 60s generation spacing...", remaining)
            sleep_fn(remaining)
            return remaining
        return 0.0

    def record_request(self) -> None:
        """Record the timestamp of an accepted/submitted video generation request."""
        self._last_request_time = time.monotonic()

    def reset(self) -> None:
        """Reset timestamp (primarily for tests)."""
        self._last_request_time = None


AGNES_LIMITER = AgnesRateLimiter(min_interval_s=60.0)


def generate_agnes_video(
    cfg: Config,
    *,
    prompt: str,
    out_mp4: str,
    out_png: str | None = None,
    duration_s: float = 5.0,
    dry_run: bool = False,
    max_poll_seconds: int = 360,
    poll_interval_s: int = 5,
) -> str:
    """Generate a single scene video clip with native Hindi audio via Agnes AI.

    Primary cloud video generation provider.
    - Enforces 60-second minimum spacing between consecutive generation POST requests.
    - Does not rate limit GET status polling.
    - POST /v1/videos
    - GET /agnesapi?video_id=<ID>&model_name=<MODEL>
    - Stream MP4 download & extract preview frame
    """
    if dry_run or not cfg.agnes_api_key:
        if not dry_run and not cfg.agnes_api_key:
            log.warning("video_gen[agnes]: AGNES_API_KEY not set; using dry-run placeholder clip")
        return _generate_dry_clip(out_mp4, out_png, duration_s=duration_s)

    base_url = (getattr(cfg, "agnes_api_base", DEFAULT_AGNES_API_BASE) or DEFAULT_AGNES_API_BASE).rstrip("/")
    configured_model = (getattr(cfg, "agnes_model", DEFAULT_AGNES_MODEL) or DEFAULT_AGNES_MODEL).strip() or DEFAULT_AGNES_MODEL

    # Build candidate models starting with the configured model
    models_to_try: list[str] = [configured_model]
    for m in _AGNES_VIDEO_MODELS:
        if m not in models_to_try:
            models_to_try.append(m)

    clamped_duration = int(round(min(10.0, max(2.0, duration_s))))
    final_prompt = _ensure_anti_text_prompt(prompt)

    rate_limit_interval = getattr(cfg, "agnes_rate_limit_seconds", 60.0)
    if rate_limit_interval is not None:
        AGNES_LIMITER.min_interval_s = float(rate_limit_interval)

    headers = {
        "Authorization": f"Bearer {cfg.agnes_api_key}",
        "Content-Type": "application/json",
    }

    create_url = f"{base_url}/v1/videos"
    max_retries = 3
    resp = None
    accepted_model: str | None = None
    last_err: Exception | None = None

    with httpx.Client(timeout=60.0) as client:
        for model_name in models_to_try:
            payload = {
                "model": model_name,
                "prompt": final_prompt,
                "seconds": str(clamped_duration),
                "size": "720P",
                "aspect_ratio": "16:9",
                "mode": "text",
            }

            log.info(
                "video_gen[agnes]: submitting scene to %s (duration=%ds, native_audio=True)",
                model_name,
                clamped_duration,
            )

            backoff_s = 60.0

            for attempt in range(max_retries + 1):
                # 1. Enforce minimum 60s spacing between generation POST requests
                AGNES_LIMITER.wait_if_needed()

                try:
                    resp = client.post(create_url, headers=headers, json=payload)
                except Exception as exc:
                    last_err = RuntimeError(f"video_gen[agnes]: failed to connect to API at {create_url}: {exc}")
                    log.warning("video_gen[agnes]: connection failed for %s: %s", model_name, exc)
                    break

                # 2. Check for distributor 503 / model_not_found / channel unavailable
                is_model_unavailable = (
                    resp.status_code == 503
                    or "model_not_found" in resp.text
                    or "No available channel" in resp.text
                    or (resp.status_code == 404 and "model" in resp.text.lower())
                )
                if is_model_unavailable:
                    err_msg = resp.text[:200]
                    log.warning(
                        "video_gen[agnes]: model '%s' unavailable on distributor (HTTP %d: %s); skipping retries for this model",
                        model_name,
                        resp.status_code,
                        err_msg,
                    )
                    last_err = RuntimeError(
                        f"video_gen[agnes]: model '{model_name}' unavailable on distributor (HTTP {resp.status_code}: {err_msg})"
                    )
                    break

                # 3. Handle HTTP 429 with bounded exponential backoff
                if resp.status_code == 429:
                    if attempt < max_retries:
                        log.warning(
                            "video_gen[agnes]: rate limited (HTTP 429). Backing off for %.1fs (attempt %d/%d)...",
                            backoff_s,
                            attempt + 1,
                            max_retries,
                        )
                        time.sleep(backoff_s)
                        backoff_s = min(240.0, backoff_s * 1.5)
                        continue
                    last_err = RuntimeError(
                        f"video_gen[agnes]: create task rate-limited (HTTP 429) after {max_retries} retries: {resp.text[:200]}"
                    )
                    break

                if resp.status_code not in (200, 201, 202):
                    err_msg = resp.text[:200]
                    last_err = RuntimeError(
                        f"video_gen[agnes]: create task failed with HTTP {resp.status_code}: {err_msg}"
                    )
                    log.warning(
                        "video_gen[agnes]: model '%s' creation failed with HTTP %d: %s",
                        model_name,
                        resp.status_code,
                        err_msg,
                    )
                    break

                # Successfully accepted! Record rate limiter timestamp
                AGNES_LIMITER.record_request()
                accepted_model = model_name
                break

            if accepted_model:
                break

        if not accepted_model or resp is None or resp.status_code not in (200, 201, 202):
            raise RuntimeError(
                f"video_gen[agnes]: all candidate models failed {models_to_try}. Last error: {last_err}"
            )

        try:
            data = resp.json()
        except Exception as exc:
            raise RuntimeError(f"video_gen[agnes]: invalid JSON response from create task: {resp.text[:200]}") from exc

        video_id = data.get("video_id") or data.get("id") or data.get("task_id")
        if not video_id and isinstance(data.get("data"), dict):
            video_id = data["data"].get("video_id") or data["data"].get("id") or data["data"].get("task_id")

        if not video_id:
            raise RuntimeError(f"video_gen[agnes]: no video_id found in creation response: {data}")

        log.info("video_gen[agnes]: task created with ID: %s (model: %s)", video_id, accepted_model)

        # Polling loop
        start_time = time.monotonic()
        video_url: str | None = None

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > max_poll_seconds:
                raise TimeoutError(
                    f"video_gen[agnes]: video generation timed out after {int(elapsed)} seconds (task: {video_id})"
                )

            poll_url = f"{base_url}/agnesapi?video_id={video_id}&model_name={accepted_model}"
            try:
                poll_resp = client.get(poll_url, headers=headers)
                if poll_resp.status_code == 404:
                    poll_resp = client.get(f"{base_url}/v1/videos/{video_id}", headers=headers)
                poll_data = poll_resp.json()
            except Exception as exc:
                log.warning("video_gen[agnes]: transient polling error (elapsed %ds): %s", int(elapsed), exc)
                time.sleep(poll_interval_s)
                continue

            status = (
                poll_data.get("status")
                or (poll_data.get("data", {}).get("status") if isinstance(poll_data.get("data"), dict) else None)
                or poll_data.get("state")
            )

            log.info("video_gen[agnes]: waiting... (elapsed: %ds, status: %s)", int(elapsed), status)

            if status in ("completed", "complete", "SUCCESS", "success"):
                video_url = (
                    poll_data.get("url")
                    or poll_data.get("video_url")
                    or (poll_data.get("data", {}).get("url") if isinstance(poll_data.get("data"), dict) else None)
                    or (poll_data.get("data", {}).get("video_url") if isinstance(poll_data.get("data"), dict) else None)
                )
                break
            elif status in ("failed", "error", "FAILED", "ERROR"):
                err_detail = poll_data.get("error") or poll_data.get("message") or "task failed"
                raise RuntimeError(f"video_gen[agnes]: generation failed on server: {err_detail}")

            time.sleep(poll_interval_s)

        if not video_url:
            raise RuntimeError(f"video_gen[agnes]: completed but no video URL returned: {poll_data}")

        # Download resulting MP4
        out_path = Path(out_mp4)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("video_gen[agnes]: downloading generated video from %s", video_url[:60] + "...")

        try:
            with client.stream("GET", video_url) as stream_resp:
                if stream_resp.status_code != 200:
                    raise RuntimeError(f"video_gen[agnes]: download failed with HTTP {stream_resp.status_code}")
                with open(out_path, "wb") as f:
                    for chunk in stream_resp.iter_bytes(chunk_size=65536):
                        f.write(chunk)
        except Exception as exc:
            raise RuntimeError(f"video_gen[agnes]: failed to download video file: {exc}") from exc

        if not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError(f"video_gen[agnes]: downloaded MP4 is missing or empty at {out_path}")

        log.info("video_gen[agnes]: saved scene video to %s (%d bytes)", out_path.name, out_path.stat().st_size)

        if out_png:
            _extract_first_frame(str(out_path), out_png)

        return str(out_path)


def generate_veo_video(
    cfg: Config,
    *,
    prompt: str,
    out_mp4: str,
    out_png: str | None = None,
    duration_s: float = 6.0,
    dry_run: bool = False,
    max_poll_seconds: int = 480,
    poll_interval_s: int = 15,
) -> str:
    """Generate a single scene video clip via Google Gemini Veo (Alternative/Fallback)."""
    if dry_run or not cfg.gemini_api_key:
        if not dry_run and not cfg.gemini_api_key:
            log.warning("video_gen[veo]: GEMINI_API_KEY not set; using dry-run placeholder clip")
        return _generate_dry_clip(out_mp4, out_png, duration_s=duration_s)

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "google-genai package is required for direct Gemini Veo video generation. "
            "Install it via `uv pip install google-genai`."
        ) from exc

    clamped_duration = int(round(min(10.0, max(2.0, duration_s))))
    model_name = getattr(cfg, "veo_model", DEFAULT_VEO_MODEL) or DEFAULT_VEO_MODEL
    final_prompt = _ensure_anti_text_prompt(prompt)

    log.info(
        "video_gen[veo]: submitting scene to %s (duration=%ds, native_audio=True)",
        model_name,
        clamped_duration,
    )

    client = genai.Client(api_key=cfg.gemini_api_key)

    config = types.GenerateVideosConfig(
        aspect_ratio="16:9",
        duration_seconds=clamped_duration,
        negative_prompt=NEGATIVE_PROMPT,
    )

    operation = client.models.generate_videos(
        model=model_name,
        prompt=final_prompt,
        config=config,
    )

    log.info("video_gen[veo]: operation initiated: %s", getattr(operation, "name", "pending"))

    start_time = time.monotonic()
    while not operation.done:
        elapsed = time.monotonic() - start_time
        if elapsed > max_poll_seconds:
            raise TimeoutError(f"video_gen[veo]: timed out after {int(elapsed)} seconds")
        log.info("video_gen[veo]: waiting for video generation... (elapsed: %ds)", int(elapsed))
        time.sleep(poll_interval_s)
        operation = client.operations.get(operation)

    if not getattr(operation, "response", None) or not getattr(operation.response, "generated_videos", None):
        error_msg = getattr(operation, "error", "No generated videos returned by operation")
        raise RuntimeError(f"video_gen[veo]: generation failed: {error_msg}")

    generated_video = operation.response.generated_videos[0]
    out_path = Path(out_mp4)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if getattr(generated_video, "video", None) and getattr(generated_video.video, "video_bytes", None):
        out_path.write_bytes(generated_video.video.video_bytes)
    else:
        client.files.download(file=generated_video.video, destination=out_path)

    log.info("video_gen[veo]: successfully saved scene video to %s (%d bytes)", out_path.name, out_path.stat().st_size)

    if out_png:
        _extract_first_frame(str(out_path), out_png)

    return str(out_path)


def generate_scene_video(
    cfg: Config,
    *,
    prompt: str,
    out_mp4: str,
    out_png: str | None = None,
    duration_s: float = 5.0,
    dry_run: bool = False,
) -> str:
    """Convenience entry point dispatching to the configured primary video provider."""
    preferred = (getattr(cfg, "video_provider", "agnes") or "agnes").lower()
    if preferred == "veo" and cfg.gemini_api_key:
        return generate_veo_video(cfg, prompt=prompt, out_mp4=out_mp4, out_png=out_png, duration_s=duration_s, dry_run=dry_run)
    return generate_agnes_video(cfg, prompt=prompt, out_mp4=out_mp4, out_png=out_png, duration_s=duration_s, dry_run=dry_run)


def build_video_providers(
    cfg: Config,
    *,
    prompt: str,
    out_mp4: str,
    out_png: str | None = None,
    duration_s: float = 5.0,
    dry_run: bool = False,
) -> list[Provider[str]]:
    """Return ordered provider list for cascade integration.

    Default hierarchy:
    1. Agnes AI (`agnes-video-2.5`) — primary native-audio provider.
    2. Google Gemini Veo (`veo-3.1-generate-preview`) — alternative/fallback.
    3. Dry-stub — offline / synthetic test fallback.
    """
    if dry_run:
        return [
            Provider(
                "dry-stub",
                lambda: _generate_dry_clip(out_mp4, out_png, duration_s=duration_s),
            )
        ]

    providers: list[Provider[str]] = []
    preferred = (getattr(cfg, "video_provider", "agnes") or "agnes").lower()

    agnes_provider = Provider(
        "agnes-video",
        lambda: generate_agnes_video(
            cfg,
            prompt=prompt,
            out_mp4=out_mp4,
            out_png=out_png,
            duration_s=duration_s,
            dry_run=False,
        ),
    )

    veo_provider = Provider(
        "gemini-veo",
        lambda: generate_veo_video(
            cfg,
            prompt=prompt,
            out_mp4=out_mp4,
            out_png=out_png,
            duration_s=duration_s,
            dry_run=False,
        ),
    )

    if preferred == "veo":
        if cfg.gemini_api_key:
            providers.append(veo_provider)
        if cfg.agnes_api_key:
            providers.append(agnes_provider)
    else:
        # Default: Agnes is primary
        if cfg.agnes_api_key:
            providers.append(agnes_provider)
        if cfg.gemini_api_key:
            providers.append(veo_provider)

    # Always ensure dry-stub fallback exists if no keys are available
    if not providers:
        providers.append(
            Provider(
                "dry-stub",
                lambda: _generate_dry_clip(out_mp4, out_png, duration_s=duration_s),
            )
        )

    return providers
