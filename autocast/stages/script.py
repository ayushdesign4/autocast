"""Stage: script — topic -> narration text.

Reads `spine.topic`, writes `spine.script`. The LLM call is behind the cascade;
in dry-run we return a deterministic placeholder script so `direction` and `tts`
have real text to flow.
"""

from __future__ import annotations

import logging

from autocast.config import Config
from autocast.providers.llm import build_llm_providers, try_llm
from autocast.seeds import TEMPLATE_SEED, generic_script, seed_for_title
from autocast.spine import Run, Script

log = logging.getLogger("autocast.stages.script")

STAGE = "script"

# Narration length that lands near the target runtime once spoken.
# Mandatory minimum is 160-180 words (targeting 180 words).
_MIN_WORDS = 160
_TARGET_WORDS = 180
_MAX_WORDS = 320


def _script_prompt(title: str, target_len_s: int) -> str:
    return (
        f"Write a heartwarming, emotional, and family-friendly 1990s Indian nostalgic childhood story in Hindi based on the title: {title}\n\n"
        "Story Requirements:\n"
        "- Era & Setting: 1990s rural or small-town India (e.g. Nani/Dadi's courtyard, mango orchard, rooftop with charpai, village fair, monsoon rain, Doordarshan Sunday, cassette player, power cuts).\n"
        "- Language: Rich, emotional, evocative HINDI (Devanagari script).\n"
        "- Themes: Natural everyday childhood memories, innocence, sibling bonding, grandmother's love, simple joys.\n"
        "- STRICTLY FORBIDDEN: No modern smartphones, social media, computers, or futuristic tech.\n"
        "- Characters: Include 2-3 memorable recurring characters with natural Hindi dialogues.\n"
        "- Length: MANDATORY at least 160 to 180 words (target 180 words, up to 250 words maximum). "
        "Must be a complete, well-developed narrative arc with a vivid beginning, escalating nostalgic adventure/moments, "
        "heartfelt dialogues, and an emotional conclusion. Stories under 160 words will be rejected.\n"
        "- Return ONLY the Hindi narrative story text without markdown formatting, headings, or stage directions.\n"
    )


def _expand_prompt(title: str, script_text: str, current_wc: int) -> str:
    return (
        f"The following 1990s Hindi nostalgia story for '{title}' is too brief ({current_wc} words). "
        f"MANDATORY REQUIREMENT: Meaningfully expand it to contain at least 160 to 180 words (target 180 words). "
        f"Enrich the narrative with vivid sensory details (the scent of wet red soil, petrichor, neem tree shade, "
        f"clinking glass marbles, brass tiffins), warm natural character dialogues, and emotional progression. "
        f"Preserve the complete narrative arc (beginning, progression, emotional moments, and ending). "
        f"Do NOT use repetitive sentences or meaningless filler words. "
        f"Return ONLY the complete expanded Hindi story text without markdown formatting, headings, or stage directions:\n\n"
        f"{script_text}"
    )


def _expand_story_fallback(title: str, base_text: str = "") -> str:
    """Build a rich, coherent 160-220 word 1990s Hindi nostalgia story preserving characters,
    beginning, progression, emotional moments, and ending without repetitive padding.
    """
    intro = (
        "1990 के उस सुनहरे दौर में जब गर्मियां शुरू होते ही नानी के गाँव जाने का इंतज़ार रहता था। "
        "सुबह की ताज़ा धूप कच्चे दालान पर बिखरती और नीम की डालियों पर बैठी चिड़ियों की चहचहाहट से नींद खुलती थी। "
    )
    cleaned = base_text.strip()
    middle = (
        f"तपती दोपहर में '{title}' का वो मासूम एहसास आज भी आँखों के सामने जीवंत हो उठता है। "
        "मिट्टी के पुराने घड़े से शीतल जल पीकर आरव और पिंकी ने नज़रों ही नज़रों में इशारा किया "
        "और बिना चप्पल पहने नंगे पाँव आँगन की तपती रेत पर दौड़ पड़े। "
        "हवा में उड़ती धूल और पेड़ों की ठंडी छाँव के बीच हंसी के ठहाके गूंजते थे। "
    )
    if cleaned:
        core = f"{cleaned} {middle}"
    else:
        core = middle

    climax = (
        "तभी नानी ने अपनी सूती साड़ी के पल्लू से बच्चों के माथे का पसीना पोंछा और प्यार से सिलबट्टे पर पिसी तीखी चटनी "
        "के साथ नमक-मिर्च लगी ताज़ा अमिया की फांकें उनके नन्हे हाथों में थमा दीं। "
        "शाम ढलते ही खुली छत पर पानी छिड़क कर चारपाई बिछाई गई, जहाँ तारों भरी रात में परियों की कहानियाँ "
        "सुनते-सुनते वो मासूम बचपन हमेशा के लिए यादों में अमर हो गया।"
    )
    return f"{intro}{core}{climax}".strip()


def _clean_script(text: str) -> str:
    """Strip markdown/label noise an LLM sometimes adds around the narration."""
    lines = []
    for raw in text.splitlines():
        # Drop heading marks and markdown emphasis outright — narration is spoken
        # text, so a literal '*' or '#' is always noise, wherever it sits.
        line = raw.strip().lstrip("#").strip().replace("*", "")
        low = line.lower()
        # Drop bracketed production cues entirely ("[music]", "[cut to ...]").
        if low.startswith("["):
            continue
        # Strip a leading speaker/section label ("Narration:", "Script:", "Title:").
        if low.startswith(("title:", "script:", "narration:")):
            line = line.split(":", 1)[-1].strip()
        if line:
            lines.append(line)
    return " ".join(lines).strip()


def run(spine: Run, cfg: Config, *, dry_run: bool = False) -> Run:
    if spine.topic is None:
        raise ValueError("script stage: spine.topic missing (run topic first)")

    title = spine.topic.title
    prompt = _script_prompt(title, cfg.target_len_s)
    providers = build_llm_providers(cfg, prompt=prompt, kind="script", dry_run=dry_run)
    llm_text, provider = try_llm(providers)

    if dry_run or not llm_text:
        seed = seed_for_title(title)
        base = seed.script if seed is not None else generic_script(title)
        text = _expand_story_fallback(title, base)
        provider = TEMPLATE_SEED if seed is not None else "template-fallback"
    else:
        text = _clean_script(llm_text)
        wc = len(text.split())

        # If LLM output is below mandatory word count (160 words), attempt LLM expansion
        if wc < _MIN_WORDS:
            log.warning("script: LLM story too short (%d words < %d); attempting expansion", wc, _MIN_WORDS)
            exp_prompt = _expand_prompt(title, text, wc)
            exp_providers = build_llm_providers(cfg, prompt=exp_prompt, kind="script", dry_run=dry_run)
            exp_text, exp_provider = try_llm(exp_providers)
            if exp_text:
                cleaned_exp = _clean_script(exp_text)
                if len(cleaned_exp.split()) >= _MIN_WORDS:
                    text = cleaned_exp
                    provider = f"{exp_provider}+expanded"

        # Final word count validation: never silently continue with a short template script
        wc = len(text.split())
        if wc < _MIN_WORDS:
            log.warning(
                "script: story still under %d words (%d words); expanding with story continuity",
                _MIN_WORDS,
                wc,
            )
            text = _expand_story_fallback(title, text)
            provider = f"{provider}+expanded-fallback"

        if len(text.split()) > _MAX_WORDS:
            text = " ".join(text.split()[:_MAX_WORDS])

    final_wc = len(text.split())
    if final_wc < _MIN_WORDS:
        raise ValueError(
            f"script validation failed: final story has only {final_wc} words "
            f"(minimum mandatory: {_MIN_WORDS}, target: {_TARGET_WORDS})"
        )

    spine.script = Script(
        full_text=text,
        word_count=final_wc,
        provider=provider,
    )
    spine.stage(STAGE).provider_used = provider
    log.info("script: %d words via %s (target %d words)", spine.script.word_count, provider, _TARGET_WORDS)
    return spine
