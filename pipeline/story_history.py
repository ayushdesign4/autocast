"""Durable history tracking to prevent duplicate topics, scripts, and video uploads.

Stores records in `data/history.json` and `output/history/<channel>.json`.
"""
from datetime import datetime, timezone, timedelta
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("pipeline.story_history")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
HISTORY_FILE = DATA_DIR / "history.json"
OUTPUT_HISTORY_DIR = REPO_ROOT / "output" / "history"
MAX_HISTORY = 300
IST = timezone(timedelta(hours=5, minutes=30))


def _normalize_text(s: str) -> str:
    """Normalize text for duplicate comparison while preserving Devanagari characters and matras."""
    import unicodedata
    import re
    normalized = unicodedata.normalize("NFKC", s).lower()
    cleaned = re.sub(r"[!\"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~।॥,;\?\!\-\–\—\'\"“”‘’]", " ", normalized)
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
    scene_prompts: list[str] | None = None,
    history_records: list[dict[str, Any]] | None = None,
) -> tuple[bool, str]:
    """Check if the given topic, title, narration, prompts, or video is a duplicate of a prior run."""
    history = history_records if history_records is not None else load_history()
    norm_topic = _normalize_text(topic)
    norm_title = _normalize_text(title)
    narration_hash = compute_sha256(narration)[:16] if narration else ""
    script_sha = compute_sha256(narration) if narration else ""
    curr_scene_hashes = [compute_sha256(p)[:16] for p in scene_prompts] if scene_prompts else []
    words_curr = set(_normalize_text(narration).split()) if narration else set()

    for entry in history:
        # 1. Exact video hash match
        if video_hash and entry.get("video_hash") == video_hash:
            return True, f"Identical video hash already uploaded: {video_hash}"

        # 2. Exact script fingerprint match
        if narration_hash and (
            entry.get("narration_hash") == narration_hash
            or entry.get("script_sha256") == script_sha
        ):
            return True, f"Identical script narration hash: {narration_hash}"

        # 3. Exact or substring topic match
        existing_topic = _normalize_text(entry.get("topic", ""))
        if norm_topic and existing_topic:
            if norm_topic == existing_topic or (len(norm_topic) > 15 and norm_topic in existing_topic):
                return True, f"Topic '{topic}' was already used previously"

        # 4. Exact title or high title token similarity
        existing_title = _normalize_text(entry.get("title", ""))
        if norm_title and existing_title:
            if norm_title == existing_title:
                return True, f"Title '{title}' matches previous upload"
            w1 = set(norm_title.split())
            w2 = set(existing_title.split())
            if w1 and w2:
                title_jaccard = len(w1 & w2) / len(w1 | w2)
                if title_jaccard >= 0.70:
                    return True, f"Title '{title}' is too similar to previous upload '{entry.get('title')}' (similarity: {title_jaccard:.2f})"

        # 5. Narration text word-level similarity against history summary or narration
        if len(words_curr) > 10:
            existing_compare = _normalize_text(entry.get("narration", "") or entry.get("summary", ""))
            words_hist = set(existing_compare.split())
            if len(words_hist) > 5:
                overlap = len(words_curr & words_hist) / len(words_hist)
                if overlap >= 0.60:
                    return True, f"Narration text shares substantial content ({overlap:.0%}) with previous story '{entry.get('title')}'"

        # 6. Scene prompts visual similarity
        if curr_scene_hashes and entry.get("scene_prompt_hashes"):
            hist_prompt_hashes = set(entry.get("scene_prompt_hashes", []))
            common_prompts = sum(1 for h in curr_scene_hashes if h in hist_prompt_hashes)
            if len(curr_scene_hashes) > 0 and (common_prompts / len(curr_scene_hashes)) >= 0.50:
                return True, f"Scene prompts share {common_prompts}/{len(curr_scene_hashes)} visual scenes with previous upload '{entry.get('title')}'"

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
    scene_prompts: list[str] | None = None,
) -> None:
    """Append a completed run record to durable history."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    history = load_history()

    now_utc = datetime.now(timezone.utc)
    now_ist = now_utc.astimezone(IST)

    narration_hash = compute_sha256(narration)[:16]
    script_sha = compute_sha256(narration)
    clean_summary = summary or (" ".join(narration.split()[:25]) + "…")
    scene_hashes = [compute_sha256(p)[:16] for p in scene_prompts] if scene_prompts else []

    entry = {
        "channel": channel_id,
        "run_id": run_id or "",
        "timestamp_utc": now_utc.isoformat(),
        "timestamp_ist": now_ist.isoformat(),
        "topic": topic.strip(),
        "topic_normalized": _normalize_text(topic),
        "title": title.strip(),
        "title_normalized": _normalize_text(title),
        "summary": clean_summary.strip(),
        "narration_hash": narration_hash,
        "script_sha256": script_sha,
        "scene_prompt_hashes": scene_hashes,
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
    channel_dir = OUTPUT_HISTORY_DIR if HISTORY_FILE.parent == DATA_DIR else (HISTORY_FILE.parent / "channel_history")
    channel_dir.mkdir(parents=True, exist_ok=True)
    channel_path = channel_dir / f"{channel_id}.json"
    channel_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def history_prompt_block(channel: str, extra_avoid: list[str] | None = None) -> str:
    """Return anti-repeat prompt text listing recent story topics to inject into Groq prompt."""
    history = load_history()
    recent = history[-30:] if history else []
    items: list[str] = []
    for entry in recent:
        t = entry.get("title", "")
        top = entry.get("topic", "")
        if t:
            items.append(f"Title: {t}")
        if top and top != t:
            items.append(f"Topic: {top}")

    if extra_avoid:
        for x in extra_avoid:
            if x and f"Avoid: {x}" not in items:
                items.append(f"Avoid: {x}")

    if not items:
        return ""

    listing = "\n".join(f"  - {item}" for item in items[-35:])
    return (
        "\n\nIMPORTANT (ANTI-REPEAT RULES):\n"
        "Do NOT repeat or closely imitate any of these recently produced stories. "
        "Choose a COMPLETELY DISTINCT nostalgic memory, setting, characters, and emotional beat:\n"
        f"{listing}\n"
    )
