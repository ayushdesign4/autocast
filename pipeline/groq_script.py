"""Groq LLM integration — generate Short script + image prompts as structured JSON.

Generates Devanagari Hindi narration (130-180 words) and English visual prompts for vertical 9:16 Shorts.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from groq import Groq

from pipeline.channel_presets import ChannelPreset
from pipeline.story_history import history_prompt_block

log = logging.getLogger("pipeline.groq_script")

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


def _extract_json(content: str) -> dict[str, Any]:
    """Extract valid JSON from markdown code blocks or raw response string."""
    text = content.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    elif "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        text = text[start:end].strip()

    return json.loads(text)


def _call_groq(preset: ChannelPreset, user_prompt: str, *, temperature: float = 0.75) -> dict[str, Any]:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY environment variable / secret")

    model_name = os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip()
    client = Groq(api_key=api_key)

    system_prompt = preset.get("groq_system_hint", "You write engaging YouTube Shorts scripts.")

    log.info("Calling Groq LLM (model=%s, temp=%.2f)...", model_name, temperature)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=1500,
        response_format={"type": "json_object"},
    )

    raw_text = completion.choices[0].message.content or ""
    return _extract_json(raw_text)


def _get_fallback_pack(preset: ChannelPreset, topic: str) -> dict[str, Any]:
    """Rich, validated fallback pack for 90s Indian Nostalgia when LLM is unavailable."""
    title = "जब टीवी का मतलब सिर्फ रविवार और रामायण था"
    narration = (
        "1990 का वो दौर भी क्या कमाल का था, जब पूरे मोहल्ले में सिर्फ एक या दो घरों में रंगीन टीवी हुआ करता था। "
        "रविवार की सुबह नौ बजते ही सड़कें पूरी तरह सूनी हो जाती थीं और छत पर टीवी एंटीना हिलाने की आवाजें गूंजती थीं। "
        "नीचे से कोई जोर से चिल्लाता था, आ गया... जरा बाएं घुमाओ! "
        "ड्राइंग रूम में दरी बिछ जाती थी और पूरा मोहल्ला एक ही परिवार बनकर बैठ जाता था। "
        "रामायण और शक्तिमान देखने के बाद, नानी के हाथ का बना गरमा-गरम चूरन और आम का पना सबमें बंटता था। "
        "न कोई स्मार्टफोन था और न इंटरनेट की आपाधापी, बस सच्चे रिश्ते, मोहल्ले का अपनापन और सादगी से भरी हंसी थी। "
        "आज हाथ में स्क्रीन तो बहुत बड़ी है, लेकिन दिल को छू लेने वाली वो मासूमियत कहीं बहुत पीछे छूट गई है।"
    )
    image_prompts = [
        "A quiet Indian middle-class living room in the early 1990s, vintage Onida CRT television with wooden cabinet, family and neighbors gathered eagerly on a woven carpet, morning golden sunlight streaming through wooden shutters, vertical 9:16 portrait",
        "A young Indian teenage boy standing on a terracotta brick rooftop, carefully turning a metallic Yagi TV antenna towards the sky, shouting down with joy, background of neighboring rooftops and neem trees, vertical 9:16",
        "Close-up of a crowded 1990s Indian verandah, children sitting cross-legged on the floor watching television with wide fascinated eyes, colorful steel water glasses, retro cotton clothes, vertical 9:16",
        "An elderly Indian grandmother smiling affectionately, serving sweet tangy mango panna and spicy churan wrapped in paper cones to neighborhood kids sitting on a charpai, vertical 9:16",
        "Children playing gilli-danda and marbles in a dusty Indian residential lane as evening shadows lengthen, Bajaj Chetak scooter parked beside a brick wall, warm sunset glow, vertical 9:16",
        "A solitary nostalgic adult holding an old handwritten postcard and a cassette tape on a quiet modern balcony, looking at the distant horizon with an emotional warm smile, vertical 9:16",
    ]
    return {
        "topic": topic,
        "youtube_title": title,
        "youtube_description": (
            "1990s के उस सुनहरे दौर की यादें जब रविवार का मतलब सिर्फ परिवार और मोहल्ले का साथ होता था। "
            "क्या आपको भी याद है वो टीवी एंटीना सेट करने का जमाना? कमेंट में बताएं! "
            "#90sNostalgia #IndianNostalgia #Shorts"
        ),
        "full_narration": narration,
        "image_prompts": image_prompts,
        "tags": ["90s nostalgia", "indian nostalgia", "doordarshan", "childhood memories", "retro india", "shorts"],
    }


def generate_short_pack(
    preset: ChannelPreset,
    *,
    topic_hint: str | None = None,
    channel_id: str | None = None,
) -> dict[str, Any]:
    """Generate complete validated Short pack (title, narration, image prompts, tags)."""
    topic = (topic_hint or os.environ.get("SHORT_TOPIC", "")).strip()
    if not topic:
        pool = preset.get("topic_pool", [])
        topic = pool[0] if pool else "1990s Indian childhood nostalgia"

    min_words = int(preset.get("min_words", 130))
    max_words = int(preset.get("max_words", 185))
    seg_count = int(preset.get("segment_count", 6))

    user = (
        f"Channel Preset: {preset['label']}.\n"
        f"Chosen Story Theme / Topic: {topic}.\n"
    )

    if channel_id:
        anti_repeat = history_prompt_block(channel_id)
        if anti_repeat:
            user += anti_repeat

    user += f"""
Create ONE complete, emotionally touching YouTube Short script strictly matching the topic.

Return ONLY valid JSON with this exact schema:
{{
  "youtube_title": "concise emotionally compelling title in Hindi or mixed, under 80 characters, no hashtags",
  "youtube_description": "2-3 warm sentences summarizing the story, ending with #90sNostalgia #IndianNostalgia #Shorts",
  "full_narration": "ONE continuous heartfelt narrative paragraph in natural Devanagari Hindi. MUST be between {min_words} and {max_words} words.",
  "image_prompts": [
    "visual description for scene 1: subject, action, retro setting, 1990s props, lighting. In ENGLISH. No text in image.",
    "visual description for scene 2...",
    "..."
  ],
  "tags": ["tag1", "tag2", "tag3"]
}}

MANDATORY RULES:
1. full_narration MUST be {min_words}-{max_words} Hindi words in natural Devanagari script.
2. image_prompts MUST have exactly {seg_count} distinct scene descriptions in English for vertical 9:16.
3. Every scene prompt must be story-specific and visually coherent with the characters and setting.
4. NO text, subtitles, words, logos, or captions inside image prompts.
"""

    max_attempts = 2
    last_err = ""

    for attempt in range(max_attempts):
        prompt_with_feedback = user
        if attempt > 0:
            prompt_with_feedback += (
                f"\n\nCRITICAL FIX: Your previous generation failed validation: {last_err}.\n"
                f"Please ensure full_narration has at least {min_words} words and image_prompts has exactly {seg_count} entries.\n"
            )

        try:
            data = _call_groq(preset, prompt_with_feedback, temperature=0.75 if attempt == 0 else 0.4)

            narration = str(data.get("full_narration", "")).strip()
            word_count = len(narration.split())
            if word_count < min_words:
                raise ValueError(f"Narration too short: {word_count} words (minimum required is {min_words})")

            prompts = data.get("image_prompts", [])
            if not isinstance(prompts, list) or len(prompts) != seg_count:
                raise ValueError(f"Expected exactly {seg_count} image_prompts, got {len(prompts)}")

            for i, p in enumerate(prompts):
                if not isinstance(p, str) or len(p.strip()) < 10:
                    raise ValueError(f"Scene prompt {i + 1} is empty or too short: {p}")

            title = str(data.get("youtube_title", "")).strip()
            if not title:
                raise ValueError("Missing youtube_title")

            data["topic"] = topic
            log.info("Groq script generated successfully (%d words, %d scene prompts)", word_count, len(prompts))
            return data

        except Exception as exc:
            last_err = str(exc)
            log.warning("Groq script generation attempt %d failed: %s", attempt + 1, exc)

    log.error("All Groq attempts failed. Using validated 90s nostalgia fallback pack.")
    return _get_fallback_pack(preset, topic)
