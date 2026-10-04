"""Stage: direction — Hindi story -> Character Bible + Scene breakdown.

`character_bible` locks visual and personality consistency for recurring characters.
`scenes[]` drives direct AI video generation: each scene contains Hindi dialogue,
scene metadata, and an English Ghibli-inspired direct video-generation prompt.
Also populates `spine.shots` for backward compatibility with existing tests and UI.
"""

from __future__ import annotations

import json
import logging
import re

from autocast.config import Config
from autocast.providers.llm import build_llm_providers, try_llm
from autocast.seeds import (
    ANTI_TEXT_PROMPT,
    GHIBLI_STYLE_PREFIX,
    NEGATIVE_PROMPT,
    TEMPLATE_SEED,
    generic_shots,
    seed_for_title,
)
from autocast.spine import Character, CharacterBible, Run, Scene, Shot

log = logging.getLogger("autocast.stages.direction")

STAGE = "direction"
_MIN_SCENES = 3
_MAX_SCENES = 12


def _default_characters() -> list[dict]:
    return [
        {
            "name": "Aarav",
            "age": "9 years old",
            "gender": "male",
            "height": "4 feet 2 inches",
            "body_type": "lean energetic child",
            "skin_tone": "warm wheatish brown",
            "face_shape": "round innocent face",
            "eyes": "large expressive dark brown sparkling eyes",
            "hair": "messy jet black hair",
            "hairstyle": "short uncombed childhood crop with playful cowlick",
            "clothing": "faded yellow cotton half-sleeve t-shirt and loose navy blue shorts",
            "footwear": "dusty brown rubber flip-flops",
            "accessories": "thin black thread around right wrist",
            "unique_facial_features": "tiny beauty mark on left cheek, bright grin",
            "personality": "adventurous, curious, sweet-hearted",
            "typical_expressions": "wide-eyed wonder, suppressed giggles",
            "body_language": "restless, bouncy gait, hands on knees",
        },
        {
            "name": "Dadi",
            "age": "68 years old",
            "gender": "female",
            "height": "5 feet 0 inches",
            "body_type": "gentle elderly grandmother",
            "skin_tone": "warm dusky with soft laugh wrinkles",
            "face_shape": "oval kind grandmotherly face",
            "eyes": "warm crinkled honey-brown eyes",
            "hair": "pure silvery white hair",
            "hairstyle": "neat low traditional bun",
            "clothing": "white handwoven cotton saree with red border",
            "footwear": "simple brown leather slippers",
            "accessories": "silver bangles, small red bindi, wire-rim glasses",
            "unique_facial_features": "deep crinkles around eyes, loving smile",
            "personality": "patient, deeply nurturing matriarch",
            "typical_expressions": "soft maternal warmth",
            "body_language": "slow rhythmic hand movements with palm leaf fan",
        },
    ]


def _normalize_scene_prompt(prompt: str) -> str:
    p = prompt.strip()
    if GHIBLI_STYLE_PREFIX not in p:
        p = f"{GHIBLI_STYLE_PREFIX} {p}"
    if ANTI_TEXT_PROMPT not in p:
        if "Negative prompt:" in p:
            p = p.replace("Negative prompt:", f"{ANTI_TEXT_PROMPT} Negative prompt:")
        else:
            p = f"{p} {ANTI_TEXT_PROMPT} Negative prompt: {NEGATIVE_PROMPT}."
    return p


def _template_direction(title: str) -> tuple[list[dict], list[dict], str]:
    seed = seed_for_title(title)
    if seed is not None:
        chars = [dict(c) for c in seed.characters]
        shots = [dict(s) for s in seed.shots]
        return chars, shots, TEMPLATE_SEED

    chars = _default_characters()
    shots = list(generic_shots(title))
    return chars, shots, "template-fallback"


def _direction_prompt(script: str) -> str:
    return (
        "You are the director for a 1990s Indian nostalgic 2D animated film.\n"
        "From the Hindi story below, produce two things in a single JSON object:\n"
        "1. 'character_bible': A list of all recurring characters with exact consistent visual descriptors.\n"
        "2. 'scenes': A list of chronological scenes (approx 6 scenes per minute, each max 10 seconds).\n\n"
        "Return ONLY a valid JSON object with keys 'character_bible' and 'scenes' (no markdown fences, no prose).\n\n"
        "Each character in 'character_bible' must have:\n"
        "  name, age, gender, height, body_type, skin_tone, face_shape, eyes, hair, hairstyle, clothing, footwear, accessories, unique_facial_features, personality, typical_expressions, body_language.\n\n"
        "Each scene in 'scenes' must have:\n"
        "  scene_number: int (1, 2, ...)\n"
        "  title_hindi: str (short Hindi scene title)\n"
        "  characters_hindi: str (names of characters in this scene)\n"
        "  location_hindi: str (village courtyard, rooftop, mango orchard, etc.)\n"
        "  action_hindi: str (what happens in this scene in Hindi)\n"
        "  dialogue_hindi: str (the exact spoken Hindi dialogue for this scene)\n"
        "  duration_s: float (between 5.0 and 8.0, maximum 10.0)\n"
        "  video_prompt: str (the comprehensive English video-generation prompt)\n\n"
        "CRITICAL RULES FOR video_prompt:\n"
        "- Must start with:\n"
        f"  '{GHIBLI_STYLE_PREFIX}'\n"
        "- Explicitly describe the environment, foreground, background, time of day, weather, camera movement, character actions, and object movement.\n"
        "- NEVER use phrases like 'same character' or 'same outfit'. Repeat the exact character appearance and clothing details from the Character Bible every time.\n"
        "- Incorporate the Hindi dialogue into the video prompt: Character says in Hindi: '<dialogue_hindi>' with natural lip movement and authentic ambient sounds.\n"
        f"- Strictly enforce anti-text rules: '{ANTI_TEXT_PROMPT}'\n"
        "- Conclude with negative visual instructions:\n"
        f"  'Negative prompt: {NEGATIVE_PROMPT}.'\n\n"
        f"STORY (HINDI):\n{script}"
    )


def _extract_json_object(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in LLM output")
        candidate = text[start : end + 1]
    data = json.loads(candidate)
    if not isinstance(data, dict):
        raise ValueError("parsed JSON is not a dict")
    return data


def _extract_json_array(text: str) -> list:
    """Pull a JSON array out of an LLM response that may wrap it in fences/prose."""
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end <= start:
            raise ValueError("no JSON array found in LLM output")
        candidate = text[start : end + 1]
    data = json.loads(candidate)
    if not isinstance(data, list) or not data:
        raise ValueError("parsed JSON is not a non-empty array")
    return data


_MOTIONS = ("zoom_in", "zoom_out", "pan_left", "pan_right")


def _coerce_shots(raw: list) -> list[dict]:
    """Validate + normalize LLM shot dicts into the fields Shot needs."""
    shots: list[dict] = []
    for item in raw[:_MAX_SCENES]:
        if not isinstance(item, dict):
            continue
        narration = str(item.get("narration", "")).strip()
        image_prompt = str(item.get("image_prompt", "")).strip()
        if not narration or not image_prompt:
            continue
        motion = str(item.get("motion", "zoom_in")).strip().lower()
        if motion not in _MOTIONS:
            motion = _MOTIONS[len(shots) % len(_MOTIONS)]
        shots.append(
            {
                "narration": narration,
                "image_prompt": image_prompt,
                "motion": motion,
                "caption": str(item.get("caption", "")).strip()[:60],
                "duration_s": float(item.get("duration_s", 5.0)) or 5.0,
            }
        )
    if len(shots) < _MIN_SCENES:
        raise ValueError(f"only {len(shots)} valid shots parsed (need >= {_MIN_SCENES})")
    return shots


def _coerce_direction(raw: dict) -> tuple[list[dict], list[dict]]:
    raw_chars = raw.get("character_bible", [])
    if not isinstance(raw_chars, list) or not raw_chars:
        raw_chars = _default_characters()

    characters: list[dict] = []
    for c in raw_chars:
        if isinstance(c, dict):
            characters.append(
                {
                    "name": str(c.get("name", "Character")).strip(),
                    "age": str(c.get("age", "")).strip(),
                    "gender": str(c.get("gender", "")).strip(),
                    "height": str(c.get("height", "")).strip(),
                    "body_type": str(c.get("body_type", "")).strip(),
                    "skin_tone": str(c.get("skin_tone", "")).strip(),
                    "face_shape": str(c.get("face_shape", "")).strip(),
                    "eyes": str(c.get("eyes", "")).strip(),
                    "hair": str(c.get("hair", "")).strip(),
                    "hairstyle": str(c.get("hairstyle", "")).strip(),
                    "clothing": str(c.get("clothing", "")).strip(),
                    "footwear": str(c.get("footwear", "")).strip(),
                    "accessories": str(c.get("accessories", "")).strip(),
                    "unique_facial_features": str(c.get("unique_facial_features", "")).strip(),
                    "personality": str(c.get("personality", "")).strip(),
                    "typical_expressions": str(c.get("typical_expressions", "")).strip(),
                    "body_language": str(c.get("body_language", "")).strip(),
                }
            )

    raw_scenes = raw.get("scenes", [])
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise ValueError("no scenes array found in direction output")

    scenes: list[dict] = []
    for idx, item in enumerate(raw_scenes[:_MAX_SCENES]):
        if not isinstance(item, dict):
            continue
        scene_num = int(item.get("scene_number", idx + 1))
        title_h = str(item.get("title_hindi", f"दृश्य {scene_num}")).strip()
        chars_h = str(item.get("characters_hindi", "")).strip()
        loc_h = str(item.get("location_hindi", "")).strip()
        act_h = str(item.get("action_hindi", "")).strip()
        dial_h = str(item.get("dialogue_hindi", "")).strip()
        video_p = str(item.get("video_prompt", "")).strip()
        dur_s = min(10.0, max(2.0, float(item.get("duration_s", 6.5))))

        if not video_p:
            continue

        if GHIBLI_STYLE_PREFIX not in video_p:
            video_p = f"{GHIBLI_STYLE_PREFIX} {video_p}"
        if ANTI_TEXT_PROMPT not in video_p:
            video_p = f"{video_p} {ANTI_TEXT_PROMPT}"
        if "Negative prompt" not in video_p:
            video_p = f"{video_p} Negative prompt: {NEGATIVE_PROMPT}."

        scenes.append(
            {
                "idx": idx,
                "scene_number": scene_num,
                "title_hindi": title_h,
                "characters_hindi": chars_h,
                "location_hindi": loc_h,
                "action_hindi": act_h,
                "dialogue_hindi": dial_h,
                "video_prompt": video_p,
                "narration": dial_h or act_h or title_h,
                "image_prompt": video_p,
                "motion": "zoom_in",
                "caption": title_h[:60],
                "duration_s": dur_s,
                "has_native_audio": True,
                "status": "pending",
            }
        )

    if len(scenes) < _MIN_SCENES:
        raise ValueError(f"only {len(scenes)} valid scenes parsed (need >= {_MIN_SCENES})")

    return characters, scenes


def run(spine: Run, cfg: Config, *, dry_run: bool = False) -> Run:
    if spine.script is None:
        raise ValueError("direction stage: spine.script missing (run script first)")

    title = spine.topic.title if spine.topic else ""
    prompt = _direction_prompt(spine.script.full_text)
    providers = build_llm_providers(
        cfg, prompt=prompt, kind="direction", allow_cerebras=False, dry_run=dry_run
    )
    llm_text, provider = try_llm(providers)

    if dry_run or not llm_text:
        raw_chars, raw_scenes, provider = _template_direction(title)
    else:
        try:
            raw_chars, raw_scenes = _coerce_direction(_extract_json_object(llm_text))
            log.info("direction: parsed %d characters and %d scenes from %s", len(raw_chars), len(raw_scenes), provider)
        except Exception as exc:  # noqa: BLE001
            log.warning("direction: JSON parse failed (%s); using template direction", exc)
            raw_chars, raw_scenes, provider = _template_direction(title)

    # 1. Populate Character Bible
    spine.character_bible = CharacterBible(
        characters=[Character(**c) for c in raw_chars]
    )

    # 2. Populate Scenes
    spine.scenes = [
        Scene(
            scene_number=s.get("scene_number", i + 1),
            title_hindi=s.get("title_hindi", ""),
            characters_hindi=s.get("characters_hindi", ""),
            location_hindi=s.get("location_hindi", ""),
            action_hindi=s.get("action_hindi", ""),
            dialogue_hindi=s.get("dialogue_hindi", ""),
            video_prompt=_normalize_scene_prompt(s.get("video_prompt", "")),
            duration_s=float(s.get("duration_s", 6.5)),
            has_native_audio=True,
            status="pending",
        )
        for i, s in enumerate(raw_scenes)
    ]

    # 3. Mirror into spine.shots for backward compatibility with orchestrator / tests / dashboard
    spine.shots = [
        Shot(
            idx=i,
            scene_number=s.get("scene_number", i + 1),
            title_hindi=s.get("title_hindi", ""),
            characters_hindi=s.get("characters_hindi", ""),
            location_hindi=s.get("location_hindi", ""),
            action_hindi=s.get("action_hindi", ""),
            dialogue_hindi=s.get("dialogue_hindi", ""),
            video_prompt=_normalize_scene_prompt(s.get("video_prompt", "")),
            narration=s.get("narration", ""),
            image_prompt=_normalize_scene_prompt(s.get("image_prompt", s.get("video_prompt", ""))),
            duration_s=float(s.get("duration_s", 6.5)),
            motion=s.get("motion", "zoom_in"),
            caption=s.get("caption", s.get("title_hindi", "")),
            has_native_audio=True,
            status="pending",
        )
        for i, s in enumerate(raw_scenes)
    ]

    spine.stage(STAGE).provider_used = provider
    log.info(
        "direction: %d characters, %d scenes via %s",
        len(spine.character_bible.characters),
        len(spine.scenes),
        provider,
    )
    return spine
