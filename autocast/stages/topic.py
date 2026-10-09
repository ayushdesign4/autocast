"""Stage: topic — pick today's video subject.

Selection priority (highest first):
1. The human-editable topic QUEUE (`queue/topics.json`) — the one-lever control
   the future UI writes to. If it has entries, today's topic is dequeued from it.
2. Google Trends RSS (per-geo, keyless) — real trending titles, faceless-filtered.
3. A static evergreen seed list — the always-available offline / dry-run fallback.

Writes `spine.topic`. In dry-run we never touch the network.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET

from autocast.config import Config
from autocast.providers.llm import build_llm_providers, try_llm
from autocast.seeds import rotated_seed_titles
from autocast.spine import Run, Topic
from autocast.util.net import get_text

log = logging.getLogger("autocast.stages.topic")

STAGE = "topic"

# Google Trends RSS (India daily). Keyless.
_TRENDS_RSS = "https://trends.google.com/trending/rss?geo=IN"

# Grounded Indian nostalgia categories to ensure wide thematic variety
_DIVERSE_NOSTALGIA_THEMES = (
    "School life: morning assembly, ink pens, wooden ruler, brown paper notebook covers, lunch box sharing, report card day",
    "Railway journeys: window seat breeze, chai in clay kulhad, passing green fields, train whistle, station vendors",
    "Cassette tapes: rewinding spool with Natraj pencil, recording favorite radio songs, Murphy radio, rooftop antenna adjustment",
    "Monsoon memories: paper boats floating in rainwater drain, mud cricket with wooden plank, hot pakoras and tea during sudden downpour",
    "Village & family: Nani's mango pickle jars drying on terrace, sleeping under stars on charpai with mosquito net, summer afternoon games",
    "Street food & childhood treats: 50-paise orange ice lolly chuski, pink cotton candy, roasted bhutta with lemon masala, roadside hot samosas",
    "Evening games: gilli-danda in dusty lane, pitthu seven stones, hopscotch stapu, hide and seek chupan-chupai until streetlights turn on",
    "Sunday television: waiting for morning cartoons, Jungle Book title song, empty streets during Sunday serial, Rangoli and Chitrahaar songs",
)


def _read_queue(cfg: Config) -> list[str]:
    """Return queued human-picked topics (may be empty). Never raises."""
    path = cfg.queue_path
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        queued = data.get("queued", []) if isinstance(data, dict) else []
        return [str(t).strip() for t in queued if str(t).strip()]
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("topic: could not read queue %s: %s", path, exc)
        return []


def _get_recent_titles(manifest_path: Path, max_days: int = 30) -> list[str]:
    """Read prior titles from manifest.json within the last max_days."""
    if not manifest_path.exists():
        return []
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        recent: list[str] = []
        for entry in data:
            if isinstance(entry, dict):
                title = str(entry.get("title", "")).strip()
                if title and title not in recent:
                    recent.append(title)
        return recent[:max_days]
    except Exception as exc:  # noqa: BLE001
        log.warning("topic: could not read recent manifest titles (%s)", exc)
        return []


def _normalize_tokens(title: str) -> set[str]:
    """Extract clean lowercase keyword tokens for overlap comparison."""
    import re
    import unicodedata
    norm_text = unicodedata.normalize("NFKC", title).lower()
    cleaned = re.sub(r'[.,!?:;।॥"\'\-–—/\\()\[\]{}]', " ", norm_text)
    stop = {"और", "का", "की", "के", "में", "से", "पर", "एक", "था", "थी", "the", "a", "an", "and", "of", "in", "to", "with"}
    return {w.strip() for w in cleaned.split() if len(w.strip()) > 1 and w.strip() not in stop}


def _is_duplicate_or_too_similar(candidate: str, recent_titles: list[str], threshold: float = 0.45) -> bool:
    """Check if candidate is an exact match or shares significant word overlap with recent titles."""
    import unicodedata
    cand_norm = unicodedata.normalize("NFKC", candidate).strip().lower()
    cand_tokens = _normalize_tokens(candidate)
    if not cand_tokens:
        return False

    for recent in recent_titles:
        rec_norm = unicodedata.normalize("NFKC", recent).strip().lower()
        if cand_norm == rec_norm or candidate == recent:
            return True
        rec_tokens = _normalize_tokens(recent)
        if not rec_tokens:
            continue
        intersection = cand_tokens & rec_tokens
        union = cand_tokens | rec_tokens
        sim = len(intersection) / len(union) if union else 0.0
        if sim >= threshold or len(intersection) >= 3 or (len(intersection) >= 2 and sim >= 0.35):
            return True
    return False


_NOSTALGIA_BACKUP_TITLES = (
    "स्कूल की घंटी, स्याही की दवात और खट्टी इमली",
    "रेलगाड़ी की खिड़की, कुल्हड़ वाली चाय और सुहाना सफ़र",
    "नटराज की पेंसिल, कैसेट का रीवाइंड और विविध भारती",
    "बारिश का पहला दिन, कागज़ की नाव और गरम पकोड़े",
    "छत पर चारपाई, नानी का पंखा और तारों भरी रात",
    "रंग-बिरंगी फ़िरकी, पचास पैसे की चुस्की और बचपन का मेला",
    "गली का क्रिकेट, लकड़ी का फट्टा और खोई हुई गेंद",
    "रविवार की सुबह, शक्तिमान और रंगोली के गाने",
)


def _fetch_trends_titles() -> list[str]:
    """Fetch trending titles from Google Trends RSS. Returns [] on any failure."""
    try:
        xml = get_text(_TRENDS_RSS, timeout=20.0)
        root = ET.fromstring(xml)
        titles = [
            (item.findtext("title") or "").strip()
            for item in root.iter("item")
        ]
        return [t for t in titles if t]
    except Exception as exc:  # noqa: BLE001 - trends is best-effort; seeds cover us
        log.warning("topic: trends RSS fetch failed (%s); falling back to seeds", exc)
        return []


def _select_candidates(cfg: Config, run_id: str, dry_run: bool) -> tuple[list[str], str]:
    """Return (candidate signals, source label) by the selection priority."""
    queued = _read_queue(cfg)
    if queued:
        return queued, "human-queue"
    seeds = rotated_seed_titles(run_id)
    if dry_run:
        return seeds, "static-seed"
    trends = _fetch_trends_titles()
    extra_anchors = list(_DIVERSE_NOSTALGIA_THEMES) + seeds
    if trends:
        return trends + extra_anchors, "google-trends-rss"
    return extra_anchors, "nostalgia-themes"


def _craft_prompt(candidates: list[str], recent_titles: list[str] | None = None) -> str:
    negative_instruction = ""
    if recent_titles:
        sample = [f"- {t}" for t in recent_titles[:15]]
        negative_instruction = (
            "\nCRITICAL: Do NOT reuse, repeat, or closely copy any of these recently published topics:\n"
            + "\n".join(sample)
            + "\n"
        )

    return (
        "You curate a YouTube channel of emotional, family-friendly short animated stories "
        "celebrating 1990s Indian nostalgia and everyday childhood memories across diverse settings "
        "(such as school days, railway journeys, cassette tapes, monsoon cricket, village festivals, street food, "
        "Sunday morning TV, Nani/Dadi house, power cuts, rooftop sleeping on charpai).\n"
        f"{negative_instruction}\n"
        "From the trending signals and ideas below, produce ONE compelling, novel, nostalgic story title "
        "(in Hindi or Hindi-English, max 70 chars).\n"
        "Return ONLY the title text — no quotes, no numbering, no explanation.\n\n"
        "Signals:\n" + "\n".join(f"- {c}" for c in candidates)
    )


def _clean_title(text: str) -> str:
    """Take the first meaningful line, strip quotes/markdown/numbering."""
    for raw in text.splitlines():
        line = raw.strip().strip("\"'").lstrip("#*-0123456789. ").strip().strip("\"'")
        if len(line) >= 8:
            return line[:100]
    return ""


def run(spine: Run, cfg: Config, *, dry_run: bool = False) -> Run:
    candidates, source = _select_candidates(cfg, spine.run_id, dry_run)
    recent_titles = _get_recent_titles(cfg.manifest_path(), max_days=30)

    prompt = _craft_prompt(candidates, recent_titles=recent_titles)
    providers = build_llm_providers(
        cfg, prompt=prompt, kind="topic-rank", dry_run=dry_run
    )
    llm_text, provider = try_llm(providers)

    # Human-queued topics are used verbatim (that's the whole point of the queue).
    if source == "human-queue" or dry_run:
        chosen = candidates[0]
    elif llm_text:
        chosen = _clean_title(llm_text)
        if not chosen or _is_duplicate_or_too_similar(chosen, recent_titles):
            log.warning("topic: candidate %r is empty or too similar to recent run; choosing fresh seed alternative", chosen)
            # Find first seed that is not a recent duplicate
            chosen = ""
            for seed_cand in rotated_seed_titles(spine.run_id):
                if not _is_duplicate_or_too_similar(seed_cand, recent_titles):
                    chosen = seed_cand
                    break
            if not chosen:
                for backup_cand in _NOSTALGIA_BACKUP_TITLES:
                    if not _is_duplicate_or_too_similar(backup_cand, recent_titles):
                        chosen = backup_cand
                        break
            if not chosen:
                chosen = rotated_seed_titles(spine.run_id)[0]
    else:
        # LLM down: find a non-duplicate rotated seed
        chosen = ""
        for seed_cand in rotated_seed_titles(spine.run_id):
            if not _is_duplicate_or_too_similar(seed_cand, recent_titles):
                chosen = seed_cand
                break
        if not chosen:
            for backup_cand in _NOSTALGIA_BACKUP_TITLES:
                if not _is_duplicate_or_too_similar(backup_cand, recent_titles):
                    chosen = backup_cand
                    break
        if not chosen:
            chosen = rotated_seed_titles(spine.run_id)[0]

    spine.topic = Topic(
        title=chosen,
        source=source,
        rank_score=0.80,
        provider=provider,
    )
    spine.stage(STAGE).provider_used = provider
    log.info("topic: chose %r (source=%s) via %s", chosen, source, provider)
    return spine
