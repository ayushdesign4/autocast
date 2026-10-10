"""Text-to-Speech synthesis using Sarvam AI REST API.

Primary voice persona: `shubh` (Bulbul model, native Hindi hi-IN).
Replaces legacy Edge TTS. Never falls back to Edge TTS, Agnes, or Gemini.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import TypedDict

import httpx

log = logging.getLogger("pipeline.sarvam_tts")

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
DEFAULT_SPEAKER = "shubh"
DEFAULT_LANGUAGE_CODE = "hi-IN"
DEFAULT_MODEL = "bulbul:v3"


class SentenceTiming(TypedDict):
    text: str
    offset_ms: int
    duration_ms: int


def _get_ffprobe_bin() -> str | None:
    found = shutil.which("ffprobe")
    if found:
        return found
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        probe_exe = str(Path(exe).parent / "ffprobe.exe")
        if Path(probe_exe).exists():
            return probe_exe
    except Exception:
        pass
    return None


def probe_audio_duration(audio_path: Path) -> float:
    """Return exact duration in seconds using ffprobe or fallback estimation."""
    ffprobe_bin = _get_ffprobe_bin()
    if ffprobe_bin:
        cmd = [
            ffprobe_bin,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            val = float(res.stdout.strip())
            if val > 0:
                return val
        except Exception as e:
            log.warning("ffprobe failed on %s: %s", audio_path, e)

    # Secondary check: if WAV, parse RIFF header
    try:
        import wave
        with wave.open(str(audio_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return frames / float(rate)
    except Exception:
        pass

    # Rough fallback based on file size if probe unavailable
    size = audio_path.stat().st_size
    return max(10.0, size / 16000.0)


def split_sentences(text: str) -> list[str]:
    """Split Hindi / English text into natural sentences by punctuation."""
    # Split on Devanagari danda (।), double danda (॥), period, exclamation, question mark, newline
    raw_sentences = re.split(r"[।॥!?.\n]+", text)
    cleaned = [s.strip() for s in raw_sentences if s.strip()]
    return cleaned if cleaned else [text.strip()]


def call_sarvam_tts(
    text: str,
    *,
    api_key: str,
    speaker: str = DEFAULT_SPEAKER,
    language_code: str = DEFAULT_LANGUAGE_CODE,
    model: str = DEFAULT_MODEL,
    max_retries: int = 3,
) -> bytes:
    """Call Sarvam AI REST API and return decoded audio bytes.

    Categorizes errors explicitly:
      - 401/403: Invalid authentication
      - 402: Account quota / credits exhausted
      - 429: Rate limited
      - 5xx: Provider service error
    """
    if not api_key or not api_key.strip():
        raise RuntimeError("SARVAM_API_KEY is missing or empty")

    headers = {
        "api-subscription-key": api_key.strip(),
        "Content-Type": "application/json",
    }
    payload = {
        "text": text.strip(),
        "language_code": language_code,
        "speaker": speaker.strip().lower(),
        "model": model,
    }

    backoff_s = 2.0
    with httpx.Client(timeout=45.0) as client:
        for attempt in range(max_retries + 1):
            try:
                resp = client.post(SARVAM_TTS_URL, json=payload, headers=headers)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt < max_retries:
                    log.warning("Sarvam TTS network error (%s); retrying in %.1fs...", exc, backoff_s)
                    time.sleep(backoff_s)
                    backoff_s *= 2.0
                    continue
                raise RuntimeError(f"Sarvam AI TTS network timeout after {max_retries} retries: {exc}") from exc

            # Classify HTTP responses
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except Exception as exc:
                    raise RuntimeError(f"Sarvam AI returned invalid JSON: {resp.text[:200]}") from exc

                audios = data.get("audios")
                if not audios or not isinstance(audios, list) or not audios[0]:
                    raise RuntimeError(f"Sarvam AI response missing 'audios' payload: {list(data.keys())}")

                try:
                    audio_bytes = base64.b64decode(audios[0])
                except Exception as exc:
                    raise RuntimeError(f"Failed to base64 decode Sarvam audio response: {exc}") from exc

                if not audio_bytes:
                    raise RuntimeError("Sarvam AI returned zero audio bytes")

                return audio_bytes

            # 401 / 403: Authentication failure — fail fast
            if resp.status_code in (401, 403):
                raise RuntimeError(
                    f"Sarvam AI authentication failed (HTTP {resp.status_code}): "
                    "Invalid or unauthorized SARVAM_API_KEY. Check GitHub Secrets."
                )

            # 402: Quota / balance exhausted — fail fast
            if resp.status_code == 402:
                raise RuntimeError(
                    "Sarvam AI quota/budget exhausted (HTTP 402): "
                    "Insufficient credits on Sarvam account. Please top up API balance."
                )

            # 429: Rate limit — retry with backoff
            if resp.status_code == 429:
                if attempt < max_retries:
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after and retry_after.isdigit() else backoff_s
                    log.warning("Sarvam AI rate limited (HTTP 429); waiting %.1fs...", wait)
                    time.sleep(wait)
                    backoff_s *= 2.0
                    continue
                raise RuntimeError(f"Sarvam AI rate limited (HTTP 429) after {max_retries} retries.")

            # 5xx: Server error — retry with backoff
            if resp.status_code >= 500:
                if attempt < max_retries:
                    log.warning("Sarvam AI server error HTTP %d; retrying in %.1fs...", resp.status_code, backoff_s)
                    time.sleep(backoff_s)
                    backoff_s *= 2.0
                    continue
                raise RuntimeError(f"Sarvam AI server error HTTP {resp.status_code}: {resp.text[:200]}")

            # Other 4xx client errors
            raise RuntimeError(f"Sarvam AI TTS request failed with HTTP {resp.status_code}: {resp.text[:200]}")

    raise RuntimeError("Sarvam AI TTS failed: exhausted all attempts")


def synthesize_full(
    narration: str,
    out_audio_path: Path,
    *,
    voice: str | None = None,
    language_code: str = DEFAULT_LANGUAGE_CODE,
    model: str = DEFAULT_MODEL,
) -> tuple[float, list[SentenceTiming]]:
    """Synthesize complete narration into an audio file and calculate sentence timings.

    Returns:
        (total_duration_seconds, list_of_sentence_timings)
    """
    api_key = os.environ.get("SARVAM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing SARVAM_API_KEY environment variable / secret")

    speaker = (voice or os.environ.get("SARVAM_SPEAKER", DEFAULT_SPEAKER)).strip().lower()
    selected_model = os.environ.get("SARVAM_MODEL", model).strip()

    sentences = split_sentences(narration)
    if not sentences:
        raise ValueError("Narration text is empty")

    out_audio_path = Path(out_audio_path)
    out_audio_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Sarvam TTS: synthesizing %d sentences with speaker='%s', model='%s'", len(sentences), speaker, selected_model)
    audio_bytes = call_sarvam_tts(
        narration,
        api_key=api_key,
        speaker=speaker,
        language_code=language_code,
        model=selected_model,
    )

    out_audio_path.write_bytes(audio_bytes)
    total_dur = probe_audio_duration(out_audio_path)

    # Calculate proportional sentence timings across total audio duration
    sentence_timings: list[SentenceTiming] = []
    weights = [max(1, len(s.split())) for s in sentences]
    total_weight = sum(weights)
    curr_ms = 0

    for i, s in enumerate(sentences):
        # Calculate duration proportional to word count
        fraction = weights[i] / total_weight
        dur_ms = int(round(total_dur * 1000 * fraction))
        if i == len(sentences) - 1:
            dur_ms = max(500, int(total_dur * 1000) - curr_ms)
        sentence_timings.append({
            "text": s,
            "offset_ms": curr_ms,
            "duration_ms": dur_ms,
        })
        curr_ms += dur_ms

    log.info("Sarvam TTS: complete audio duration = %.2fs (%d sentences mapped)", total_dur, len(sentence_timings))
    return total_dur, sentence_timings
