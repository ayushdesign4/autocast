"""Image generation via Pollinations AI.

Generates vertical 9:16 images (1080x1920) for YouTube Shorts.
Handles authentication via POLLINATIONS_API_KEY and classifies errors (401/402/429/5xx).
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import random
import time
import urllib.parse
from pathlib import Path

import httpx
from PIL import Image

log = logging.getLogger("pipeline.images")

DEFAULT_STYLE_SUFFIX = (
    ", vertical 9:16 portrait composition, 1990s Indian nostalgic cinema aesthetic, "
    "warm earthy retro tones, soft natural lighting, emotional atmospheric storytelling, "
    "vintage 35mm film texture, authentic 1990s Indian attire, no text, no captions, no watermark, no logos"
)

DEFAULT_NEGATIVE = (
    "text, watermark, logo, captions, subtitles, words, letters, signatures, "
    "modern smartphones, modern cars, futuristic elements, distorted anatomy, "
    "extra limbs, ugly, blurry, low quality, horizontal 16:9, black bars"
)

POLLINATIONS_GEN_URL = "https://gen.pollinations.ai/image"
POLLINATIONS_IMAGE_URL = "https://image.pollinations.ai/prompt"


def full_visual_prompt(scene: str, style_suffix: str | None = None) -> str:
    """Combine the scene description with a channel-specific style suffix."""
    suffix = style_suffix if style_suffix is not None else DEFAULT_STYLE_SUFFIX
    return f"{scene.strip()}{suffix}"


def compute_image_hash(image_bytes: bytes) -> str:
    """Return SHA-256 fingerprint of the image bytes."""
    return hashlib.sha256(image_bytes).hexdigest()[:16]


def _ensure_vertical_9_16(image_bytes: bytes, target_w: int = 1080, target_h: int = 1920) -> bytes:
    """Validate image bytes and ensure exact 9:16 vertical canvas (1080x1920) without stretching."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as exc:
        raise ValueError(f"Downloaded bytes are not a valid image: {exc}") from exc

    w, h = img.size
    target_ratio = target_w / target_h
    current_ratio = w / h

    # If already exact dimensions, return PNG bytes
    if w == target_w and h == target_h:
        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()

    # Center crop to target aspect ratio (9:16)
    if current_ratio > target_ratio:
        # Image is wider than 9:16 -> crop left and right
        new_w = int(h * target_ratio)
        offset = (w - new_w) // 2
        img = img.crop((offset, 0, offset + new_w, h))
    elif current_ratio < target_ratio:
        # Image is taller than 9:16 -> crop top and bottom
        new_h = int(w / target_ratio)
        offset = (h - new_h) // 2
        img = img.crop((0, offset, w, offset + new_h))

    # Resize to exact target 1080x1920 with high-quality resampling
    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def _call_pollinations(
    prompt: str,
    *,
    api_key: str,
    width: int = 1080,
    height: int = 1920,
    model: str = "flux",
    negative: str = DEFAULT_NEGATIVE,
    max_retries: int = 3,
) -> bytes:
    """Generate image via Pollinations API using POLLINATIONS_API_KEY.

    Classifies responses:
      - 401/403: Authentication failure (invalid key)
      - 402: Account quota/budget exhausted
      - 429: Rate limited
      - 5xx: Provider service error
    """
    clean_prompt = prompt.strip()
    encoded_prompt = urllib.parse.quote(clean_prompt)
    seed = random.randint(1, 9999999)

    headers = {
        "User-Agent": "ZeroCost-Shorts/1.0",
        "Accept": "image/*",
    }
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    # Try gen.pollinations.ai endpoint first, falling back to image.pollinations.ai
    endpoints = [
        f"{POLLINATIONS_GEN_URL}/{encoded_prompt}?width={width}&height={height}&model={model}&seed={seed}&nologo=true",
        f"{POLLINATIONS_IMAGE_URL}/{encoded_prompt}?width={width}&height={height}&model={model}&seed={seed}&nologo=true",
    ]

    backoff_s = 3.0
    last_err: Exception | None = None

    with httpx.Client(timeout=90.0, follow_redirects=True) as client:
        for url in endpoints:
            for attempt in range(max_retries + 1):
                try:
                    resp = client.get(url, headers=headers)
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    last_err = exc
                    if attempt < max_retries:
                        log.warning("Pollinations network error (%s); retrying in %.1fs...", exc, backoff_s)
                        time.sleep(backoff_s)
                        backoff_s *= 2.0
                        continue
                    break

                if resp.status_code == 200:
                    content_type = resp.headers.get("Content-Type", "")
                    if "image" in content_type or len(resp.content) > 5000:
                        return resp.content
                    raise RuntimeError(f"Pollinations returned non-image payload: {content_type} ({resp.text[:150]})")

                # 401/403: Invalid authentication — fail fast
                if resp.status_code in (401, 403):
                    raise RuntimeError(
                        f"Pollinations AI authentication failed (HTTP {resp.status_code}): "
                        "Invalid or unauthorized POLLINATIONS_API_KEY. Verify GitHub Secrets."
                    )

                # 402: Payment / Quota exhausted — fail fast
                if resp.status_code == 402:
                    raise RuntimeError(
                        "Pollinations AI quota/budget exhausted (HTTP 402): "
                        "Account has run out of generation credits. Please recharge Pollinations balance."
                    )

                # 429: Rate limited — back off and retry
                if resp.status_code == 429:
                    if attempt < max_retries:
                        log.warning("Pollinations rate limited (HTTP 429); waiting %.1fs (attempt %d/%d)...", backoff_s, attempt + 1, max_retries)
                        time.sleep(backoff_s)
                        backoff_s *= 2.0
                        continue
                    raise RuntimeError(f"Pollinations rate limited (HTTP 429) after {max_retries} retries.")

                # 5xx: Server error
                if resp.status_code >= 500:
                    if attempt < max_retries:
                        log.warning("Pollinations server error HTTP %d; retrying in %.1fs...", resp.status_code, backoff_s)
                        time.sleep(backoff_s)
                        backoff_s *= 2.0
                        continue
                    last_err = RuntimeError(f"Pollinations server error HTTP {resp.status_code}: {resp.text[:200]}")
                    break

                # Other HTTP errors
                last_err = RuntimeError(f"Pollinations image generation failed HTTP {resp.status_code}: {resp.text[:200]}")
                break

    raise RuntimeError(f"Pollinations image generation failed across all endpoints: {last_err}")


def save_scene_image(
    index: int,
    prompt: str,
    out_path: Path,
    *,
    width: int = 1080,
    height: int = 1920,
    model: str = "flux",
    negative: str = DEFAULT_NEGATIVE,
) -> tuple[str, str]:
    """Generate, validate, resize, and save one scene image to out_path.

    Returns:
        (status, detail_message)
    """
    api_key = os.environ.get("POLLINATIONS_API_KEY", "").strip()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Generating scene %d image (9:16, %dx%d)...", index, width, height)
    raw_bytes = _call_pollinations(
        prompt,
        api_key=api_key,
        width=width,
        height=height,
        model=model,
        negative=negative,
    )

    # Validate and standardize to exact 1080x1920 vertical canvas
    processed_png = _ensure_vertical_9_16(raw_bytes, target_w=width, target_h=height)
    out_path.write_bytes(processed_png)

    img_hash = compute_image_hash(processed_png)
    file_size_kb = len(processed_png) // 1024
    detail = f"saved {out_path.name} ({file_size_kb}KB, sha256:{img_hash})"
    log.info("Scene %d image ready: %s", index, detail)
    return "ok", detail
