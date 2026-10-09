"""Stage: direction — Hindi story -> Character Bible + Scene breakdown.

`character_bible` locks visual and personality consistency for recurring characters.
`scenes[]` drives direct AI video generation: each scene contains Hindi dialogue,
scene metadata, and an English Ghibli-inspired direct video-generation prompt.
Also populates `spine.shots` for backward compatibility with existing tests and UI.
"""

from __future__ import annotations

import hashlib
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


_SCENE_SETTINGS = (
    ("गांव का दालान और सुबह की धूप", "morning golden sunlight spilling across a rustic Indian village courtyard with clay tiles and earthen floor"),
    ("आम का बगीचा और कच्ची पगडंडी", "sun-dappled mango orchard with ancient banyan roots and dusty country path"),
    ("गांव की पुरानी चौपाल और कुआं", "traditional village gathering area under large neem tree beside stone water well"),
    ("छत और शाम का आसमान", "rooftop terrace with charpai beds, clay pots, and twilight sky full of gentle swallows"),
    ("घर का आंगन और मिट्टी का चूल्हा", "cozy home veranda with traditional brass utensils, earthen stove glowing with warm embers"),
    ("पुरानी लकड़ी की खिड़की और बारिश", "vintage wooden slatted window overlooking lush green village lanes during soft drizzle"),
    ("गांव का मेला और फिरकी", "vibrant colorful village fair with wooden stalls, paper pinwheels, and festive lanterns"),
    ("रात का आंगन और तारों भरा आसमान", "quiet village courtyard under clear starry night sky, oil lamps softly flickering"),
)


def compute_scene_hash(
    video_prompt: str, dialogue_hindi: str = "", duration_s: float = 9.0
) -> str:
    """Deterministically hash all generation-relevant inputs for a scene."""
    payload = f"prompt:{video_prompt.strip()}|dial:{dialogue_hindi.strip()}|dur:{duration_s:.1f}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _script_aware_direction(
    script_text: str, title: str, num_scenes: int = 8
) -> tuple[list[dict], list[dict], str]:
    """Derive 7-8 meaningful chronological scenes directly from the Hindi script text.

    Ensures the story is never silently replaced by generic hardcoded scenes.
    """
    words = script_text.split()
    if not words:
        words = title.split() or ["यादें"]

    n_words = len(words)
    target_count = min(8, max(7, num_scenes))
    if n_words < target_count:
        target_count = max(1, n_words)

    word_chunks: list[list[str]] = []
    for i in range(target_count):
        start = int(round(i * n_words / target_count))
        end = int(round((i + 1) * n_words / target_count))
        chunk = words[start:end]
        if chunk:
            word_chunks.append(chunk)

    if not word_chunks:
        word_chunks = [words]

    target_count = len(word_chunks)
    dur_per_scene = min(10.0, max(9.0, round(72.0 / target_count, 1)))

    scenes: list[dict] = []
    for idx, w_chunk in enumerate(word_chunks):
        chunk_text = " ".join(w_chunk)
        loc_name, loc_desc = _SCENE_SETTINGS[idx % len(_SCENE_SETTINGS)]
        scene_num = idx + 1
        title_h = f"{title[:25]} - दृश्य {scene_num}"

        video_p = (
            f"{GHIBLI_STYLE_PREFIX} In a nostalgic 1990s Indian setting depicting '{title}', scene {scene_num} of {target_count}. "
            f"Environment: {loc_desc}. "
            f"Characters in period-authentic 1990s simple Indian attire interact naturally. "
            f"Character speaks in Hindi: '{chunk_text}' with natural lip movement and authentic ambient village sounds. "
            f"Hand-drawn 2D animation aesthetic, Studio Ghibli style, lush watercolor textures, cinematic lighting. "
            f"{ANTI_TEXT_PROMPT} Negative prompt: {NEGATIVE_PROMPT}."
        )

        scenes.append(
            {
                "idx": idx,
                "scene_number": scene_num,
                "title_hindi": title_h,
                "characters_hindi": "आरव और परिवार",
                "location_hindi": loc_name,
                "action_hindi": chunk_text[:60],
                "dialogue_hindi": chunk_text,
                "video_prompt": video_p,
                "narration": chunk_text,
                "image_prompt": video_p,
                "motion": _MOTIONS[idx % len(_MOTIONS)],
                "caption": title_h[:60],
                "duration_s": dur_per_scene,
                "has_native_audio": True,
                "status": "pending",
            }
        )

    chars = _default_characters()
    return chars, scenes, "script-aware-fallback"


def _template_direction(title: str, script_text: str = "") -> tuple[list[dict], list[dict], str]:
    seed = seed_for_title(title)
    if script_text:
        chars, scenes, provider = _script_aware_direction(script_text, title)
        if seed is not None:
            if seed.characters:
                chars = [dict(c) for c in seed.characters]
            provider = TEMPLATE_SEED
        return chars, scenes, provider

    if seed is not None:
        chars = [dict(c) for c in seed.characters]
        shots = [dict(s) for s in seed.shots]
        return chars, shots, TEMPLATE_SEED

    chars = _default_characters()
    shots = list(generic_shots(title))
    return chars, shots, "template-fallback"


def validate_direction(
    scenes: list[dict | Scene],
    *,
    min_scenes: int = 6,
    min_duration_s: float = 60.0,
) -> None:
    """Validate scene count, dialogue presence, duration, and visual diversity before video gen."""
    if len(scenes) < min_scenes:
        raise ValueError(
            f"direction validation failed: only {len(scenes)} scenes parsed (need >= {min_scenes})"
        )

    total_dur = 0.0
    prompts: list[str] = []
    for i, s in enumerate(scenes):
        dur = float(
            getattr(s, "duration_s", None)
            or (s.get("duration_s") if isinstance(s, dict) else 0.0)
            or 0.0
        )
        total_dur += dur
        dial = getattr(s, "dialogue_hindi", None) or (
            s.get("dialogue_hindi") if isinstance(s, dict) else ""
        )
        narr = getattr(s, "narration", None) or (
            s.get("narration") if isinstance(s, dict) else ""
        )
        if not (dial or narr):
            raise ValueError(f"direction validation failed: scene {i+1} has no dialogue or narration")
        prompt = getattr(s, "video_prompt", None) or (
            s.get("video_prompt") if isinstance(s, dict) else ""
        )
        if not prompt:
            raise ValueError(f"direction validation failed: scene {i+1} has empty video prompt")
        prompts.append(prompt.strip())

    if total_dur < min_duration_s:
        raise ValueError(
            f"direction validation failed: total duration {total_dur:.1f}s is below minimum {min_duration_s}s"
        )

    if len(set(prompts)) != len(prompts):
        raise ValueError("direction validation failed: duplicate scene visual prompts detected")


def _direction_prompt(script: str) -> str:
    return (
        "You are the director for a 1990s Indian nostalgic 2D animated film.\n"
        "From the Hindi story below, produce two things in a single JSON object:\n"
        "1. 'character_bible': A concise list of recurring characters (name, age, clothing, appearance, personality).\n"
        "2. 'scenes': A list of 7 to 8 chronological scenes covering the entire narrative arc (target 60-90s total runtime).\n\n"
        "Return ONLY a valid JSON object with keys 'character_bible' and 'scenes' (no markdown fences, no prose).\n\n"
        "Each scene in 'scenes' must have:\n"
        "  scene_number: int (1, 2, ... up to 7-8)\n"
        "  title_hindi: str (short Hindi scene title)\n"
        "  characters_hindi: str (names of characters in this scene)\n"
        "  location_hindi: str (setting in Hindi)\n"
        "  action_hindi: str (what happens in this scene in Hindi)\n"
        "  dialogue_hindi: str (the exact spoken Hindi dialogue for this scene from the story)\n"
        "  duration_s: float (between 9.0 and 10.0 seconds)\n"
        "  video_prompt: str (the comprehensive English video-generation prompt)\n\n"
        "CRITICAL RULES FOR video_prompt:\n"
        "- Must start with:\n"
        f"  '{GHIBLI_STYLE_PREFIX}'\n"
        "- Explicitly describe the environment, lighting, camera angle, character action, and clothing.\n"
        "- Every scene must depict a distinct visual progression of the story.\n"
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
        dur_s = min(10.0, max(8.0, float(item.get("duration_s", 9.0))))

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
    script_text = spine.script.full_text
    prompt = _direction_prompt(script_text)
    providers = build_llm_providers(
        cfg, prompt=prompt, kind="direction", allow_cerebras=False, dry_run=dry_run
    )
    llm_text, provider = try_llm(providers)

    if dry_run or not llm_text:
        raw_chars, raw_scenes, provider = _template_direction(title, script_text=script_text)
    else:
        try:
            raw_chars, raw_scenes = _coerce_direction(_extract_json_object(llm_text))
            validate_direction(raw_scenes)
            log.info("direction: parsed and validated %d characters and %d scenes from %s", len(raw_chars), len(raw_scenes), provider)
        except Exception as exc:  # noqa: BLE001
            log.warning("direction: LLM direction invalid (%s); using script-aware fallback", exc)
            raw_chars, raw_scenes, provider = _template_direction(title, script_text=script_text)

    # 1. Populate Character Bible
    spine.character_bible = CharacterBible(
        characters=[Character(**c) for c in raw_chars]
    )

    # 2. Populate Scenes
    spine.scenes = []
    for i, s in enumerate(raw_scenes):
        norm_p = _normalize_scene_prompt(s.get("video_prompt", ""))
        dial_h = str(s.get("dialogue_hindi", "")).strip()
        dur_s = float(s.get("duration_s", 9.0))
        p_hash = compute_scene_hash(norm_p, dial_h, dur_s)
        spine.scenes.append(
            Scene(
                scene_number=s.get("scene_number", i + 1),
                title_hindi=s.get("title_hindi", ""),
                characters_hindi=s.get("characters_hindi", ""),
                location_hindi=s.get("location_hindi", ""),
                action_hindi=s.get("action_hindi", ""),
                dialogue_hindi=dial_h,
                video_prompt=norm_p,
                duration_s=dur_s,
                has_native_audio=True,
                status="pending",
                prompt_hash=p_hash,
            )
        )

    # 3. Mirror into spine.shots for backward compatibility with orchestrator / tests / dashboard
    spine.shots = []
    for i, s in enumerate(raw_scenes):
        norm_p = _normalize_scene_prompt(s.get("video_prompt", ""))
        dial_h = str(s.get("dialogue_hindi", "")).strip()
        dur_s = float(s.get("duration_s", 9.0))
        p_hash = compute_scene_hash(norm_p, dial_h, dur_s)
        spine.shots.append(
            Shot(
                idx=i,
                scene_number=s.get("scene_number", i + 1),
                title_hindi=s.get("title_hindi", ""),
                characters_hindi=s.get("characters_hindi", ""),
                location_hindi=s.get("location_hindi", ""),
                action_hindi=s.get("action_hindi", ""),
                dialogue_hindi=dial_h,
                video_prompt=norm_p,
                narration=s.get("narration", dial_h),
                image_prompt=_normalize_scene_prompt(s.get("image_prompt", s.get("video_prompt", ""))),
                duration_s=dur_s,
                motion=s.get("motion", "zoom_in"),
                caption=s.get("caption", s.get("title_hindi", "")),
                has_native_audio=True,
                status="pending",
                prompt_hash=p_hash,
            )
        )

    spine.stage(STAGE).provider_used = provider
    log.info(
        "direction: %d characters, %d scenes via %s",
        len(spine.character_bible.characters),
        len(spine.scenes),
        provider,
    )
    return spine
