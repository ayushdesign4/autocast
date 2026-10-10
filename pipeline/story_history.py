"""Durable history tracking to prevent duplicate topics, scripts, and video uploads.

Stores records in `data/history.json` and `output/history/<channel>.json`.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("pipeline.story_history")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
HISTORY_FILE = DATA_DIR / "history.json"
MAX_HISTORY = 300


def _normalize_text(s: str) -> str:
    """Normalize text for duplicate comparison."""
    import re
    cleaned = re.sub(r"[^\w\s]", "", s.lower())
    return " ".join(cleaned.split())


def compute_sha256(data: bytes | str) -> str:
    """Compute SHA-256 fingerprint."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def load_history() -> list[dict[str, Any]]:
    """Load durable history list from data/history.json."""
    if not HISTORY_FILE.is_file():
        # Check backward compatibility with output/history
        compat = REPO_ROOT / "output" / "history" / "90s_indian_nostalgia.json"
        if compat.is_file():
            try:
                return json.loads(compat.read_text(encoding="utf-8"))
            except Exception:
                pass
        return []

    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception as exc:
        log.warning("Failed to parse history file %s: %s", HISTORY_FILE, exc)

    return []


def is_duplicate_story(
    topic: str,
    title: str,
    narration: str = "",
    video_hash: str | None = None,
) -> tuple[bool, str]:
    """Check if the given topic, title, narration, or video is a duplicate of a prior run."""
    history = load_history()
    norm_topic = _normalize_text(topic)
    norm_title = _normalize_text(title)
    narration_hash = compute_sha256(narration)[:16] if narration else ""

    for entry in history:
        # Check video hash if available
        if video_hash and entry.get("video_hash") == video_hash:
            return True, f"Identical video hash already uploaded: {video_hash}"

        # Check script fingerprint
        if narration_hash and entry.get("narration_hash") == narration_hash:
            return True, f"Identical script narration hash: {narration_hash}"

        # Check exact normalized topic
        existing_topic = _normalize_text(entry.get("topic", ""))
        if existing_topic and (norm_topic == existing_topic or norm_topic in existing_topic):
            return True, f"Topic '{topic}' was already used previously"

        # Check normalized title
        existing_title = _normalize_text(entry.get("title", ""))
        if existing_title and (norm_title == existing_title):
            return True, f"Title '{title}' matches previous upload"

    return False, ""


def save_history_entry(
    channel_id: str,
    *,
    topic: str,
    title: str,
    narration: str,
    summary: str = "",
    video_id: str | None = None,
    video_url: str | None = None,
    video_hash: str | None = None,
    duration_s: float | None = None,
    run_id: str | None = None,
) -> None:
    """Append a completed run record to durable history."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    history = load_history()

    narration_hash = compute_sha256(narration)[:16]
    clean_summary = summary or (" ".join(narration.split()[:25]) + "…")

    entry = {
        "channel": channel_id,
        "run_id": run_id or "",
        "topic": topic.strip(),
        "title": title.strip(),
        "summary": clean_summary.strip(),
        "narration_hash": narration_hash,
        "video_hash": video_hash or "",
        "video_id": video_id or "",
        "video_url": video_url or (f"https://www.youtube.com/shorts/{video_id}" if video_id else ""),
        "duration_s": round(duration_s, 2) if duration_s else None,
    }

    history.append(entry)
    history = history[-MAX_HISTORY:]

    # Write to primary history file
    HISTORY_FILE.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    # Also write to channel-specific file for backward compat
    channel_path = REPO_ROOT / "output" / "history" / f"{channel_id}.json"
    channel_path.parent.mkdir(parents=True, exist_ok=True)
    channel_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def history_prompt_block(channel: str) -> str:
    """Return anti-repeat prompt text listing recent story topics to inject into Groq prompt."""
    history = load_history()
    if not history:
        return ""

    recent = history[-30:]
    lines = []
    for entry in recent:
        t = entry.get("title", "")
        top = entry.get("topic", "")
        label = t or top
        if label:
            lines.append(f"  - {label}")

    if not lines:
        return ""

    listing = "\n".join(lines)
    return (
        "\n\nIMPORTANT (ANTI-REPEAT RULES):\n"
        "Do NOT repeat or closely imitate any of these recently produced stories. "
        "Choose a DISTINCT nostalgic memory, setting, characters, and emotional beat:\n"
        f"{listing}\n"
    )
