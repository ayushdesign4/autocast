"""Channel niches: system prompts + defaults for Groq script generation.

Primary preset: `90s_indian_nostalgia` for 1990s Indian nostalgic storytelling.
"""
from __future__ import annotations

from typing import TypedDict


class ChannelPreset(TypedDict, total=False):
    id: str
    label: str
    groq_system_hint: str
    segment_count: int  # images + script beats (typically 6-8)
    topic_pool: list[str]
    image_style_suffix: str  # appended to every image prompt
    image_negative_prompt: str  # passed as negative prompt
    language: str
    tts_voice: str  # Sarvam AI speaker persona (default: "shubh")
    caption_font: str
    caption_font_name: str
    min_words: int  # min word count for narration validation
    max_words: int
    yt_token_env: str


PRESETS: dict[str, ChannelPreset] = {
    "90s_indian_nostalgia": {
        "id": "90s_indian_nostalgia",
        "label": "1990s Indian Nostalgia (Hindi Narration, 9:16 Shorts)",
        "language": "hi",
        "tts_voice": "shubh",
        "caption_font": "NotoSansDevanagari-Bold.ttf",
        "caption_font_name": "Noto Sans Devanagari",
        "min_words": 130,
        "max_words": 185,
        "segment_count": 6,
        "groq_system_hint": (
            "You are an acclaimed 1990s Indian nostalgia storyteller who creates emotionally evocative YouTube Shorts. "
            "Your stories take viewers right back into the sights, sounds, smells, and heartfelt relationships of 1990s India. "
            "\n"
            "STRUCTURE: Hook opening beat, progression with genuine emotional conflicts or fond rituals, and an authentic touching payoff at the end. "
            "TONE: Warm, poetic, reflective, nostalgic, cinematic, culturally authentic. Never caricature or parody. "
            "\n"
            "LANGUAGE & SCRIPT RULES: "
            "- full_narration: ONE continuous paragraph in natural Devanagari Hindi. This will be read aloud by the voice artist. "
            "- Exactly 130 to 180 Hindi words (target ~150 words). Avoid bullet points, scene headings, or English words. "
            "- youtube_title: Catchy, curiosity-driven, emotionally touching title under 80 characters. "
            "- youtube_description: 2-3 warm sentences summarizing the story, inviting a comment on memories, plus hashtags: #90sNostalgia #IndianNostalgia #Shorts. "
            "- tags: 8-12 relevant search tags separated by commas. "
            "\n"
            "IMAGE PROMPT RULES: "
            "- Write image_prompts in ENGLISH ONLY (the image generator expects English). "
            "- Exactly the requested number of scene prompts. "
            "- Each prompt describes a distinct scene beat in vertical 9:16 framing. "
            "- Describe specific characters (age, 1990s clothing), lighting (golden hour, lantern, afternoon sun), props (wooden radio, Bajaj Chetak, Milton water bottle, cassette tape, Inland letter), and setting (verandah, courtyard, classroom). "
            "- STRICTLY NO text, subtitles, words, logos, or watermarks in image prompts."
        ),
        "topic_pool": [
            "रविवार सुबह दूरदर्शन पर महाभारत का प्रसारण और पूरे मोहल्ले की एक छत पर जुटने की यादें",
            "गर्मियों की छुट्टियों में नानी के घर चारपाई, आम के बगीचे और रात में छत पर तारों भरी नींद",
            "स्कूल का आखिरी दिन, रफ कॉपी के पन्नों की नाव और पहली बारिश में भीगने का भोलापन",
            "गली क्रिकेट में पुरानी टेनिस बॉल पर टेप लपेटने और 'जिसका बैट उसकी पहली बैटिंग' का नियम",
            "कैसेट की रील में पेंसिल घुमाकर रिपेयर करने और रविवार को बिनाका गीतमाला सुनने का नशा",
            "डाकिए की साइकिल की घंटी और नानी के नीले अंतर्देशीय पत्र (Inland Letter) का बेसब्री से इंतजार",
            "एसटीडी पीसीओ (STD PCO) बूथ की लाल बत्ती और 1 मिनट के सिक्के वाले फोन कॉल की धड़कन",
            "गर्मियों की शाम को मोहल्ले में बर्फ का गोला और चूरन वाले की घंटी की आवाज",
            "छत पर टीवी एंटीना हिलाने और नीचे से 'आ गया... आ गया' चिल्लाने का वो भोला दौर",
            "स्कूल के टिफिन में पराठे-आम का अचार और दोस्तों के साथ बेंच पर बैठकर बांटने की खुशी",
            "दीपावली पर मिट्टी के दीये धोकर सुखाने और छत पर मोमबत्तियों की कतार सजाने की महक",
            "किराए पर वीसीआर (VCR) लाकर पूरी रात परिवार और पड़ोसियों के साथ फिल्में देखने का रोमांच",
            "सर्दियों की गुनगुनी धूप में स्वेटर बुनती मां और दाल की बड़ी सुखाने की छत की दोपहर",
            "ट्रेन का वो स्लीपर कोच का सफर, खिड़की वाली सीट और कुल्हड़ वाली चाय की सोंधी खुशबू",
            "कंचे (गोली) और गिल्ली-डंडा का मुकाबला और जेब में भरी खनखनाती कांच की गोलियां"
        ],
        "image_style_suffix": (
            ", vertical 9:16 portrait composition, 1990s Indian nostalgic cinema aesthetic, "
            "warm earthy retro tones, soft natural lighting, emotional atmospheric storytelling, "
            "vintage 35mm film texture, authentic 1990s Indian attire and retro decor, "
            "no text, no captions, no watermark, no logos, no subtitles"
        ),
        "image_negative_prompt": (
            "text, watermark, logo, captions, subtitles, words, letters, modern smartphones, "
            "modern gadgets, modern cars, futuristic elements, distorted anatomy, extra limbs, "
            "ugly, blurry, low quality, duplicate characters, horizontal 16:9, black bars"
        ),
        "yt_token_env": "YT_REFRESH_TOKEN",
    },
    "facts": {
        "id": "facts",
        "label": "Mind-blowing facts Short (Hindi Narration)",
        "language": "hi",
        "tts_voice": "shubh",
        "caption_font": "NotoSansDevanagari-Bold.ttf",
        "caption_font_name": "Noto Sans Devanagari",
        "min_words": 100,
        "max_words": 150,
        "segment_count": 5,
        "groq_system_hint": (
            "You write punchy YouTube Shorts about surprising, verified facts in Devanagari Hindi. "
            "STRUCTURE: hook fact in opening, supporting facts in the middle, punchline + takeaway at end. "
            "TONE: energetic, curious, confident. No clickbait lies. "
            "HINDI: full_narration MUST be in Devanagari Hindi, 100-140 words. "
            "IMAGE PROMPTS: English only, realistic documentary photos, no text."
        ),
        "topic_pool": [
            "समुद्र की गहराइयों में रहने वाले रहस्यमयी जीव",
            "अंतरिक्ष में समय कैसे धीरे बीतता है",
            "प्राचीन भारत के अविश्वसनीय वैज्ञानिक आविष्कार"
        ],
        "image_style_suffix": (
            ", vertical 9:16 aspect ratio, National Geographic documentary photography, "
            "hyperrealistic, cinematic lighting, 8k, ultra detailed, no text, no logo"
        ),
        "image_negative_prompt": "text, watermark, logo, cartoon, anime, horizontal, black bars",
        "yt_token_env": "YT_REFRESH_TOKEN",
    },
    "ghost_stories": {
        "id": "ghost_stories",
        "label": "Spooky Indian Folklore & Hauntings",
        "language": "hi",
        "tts_voice": "shubh",
        "caption_font": "NotoSansDevanagari-Bold.ttf",
        "caption_font_name": "Noto Sans Devanagari",
        "min_words": 120,
        "max_words": 160,
        "segment_count": 6,
        "groq_system_hint": (
            "You write chilling Indian folklore ghost shorts in Devanagari Hindi. "
            "Eerie, suspenseful, atmospheric narration. "
            "IMAGE PROMPTS: English only, dark atmospheric eerie oil painting, vertical 9:16."
        ),
        "topic_pool": [
            "पहाड़ी हाईवे पर आधी रात को सफेद साड़ी वाली परछाई",
            "गांव के पुराने बरगद के पेड़ से आती पायल की आवाज"
        ],
        "image_style_suffix": (
            ", vertical 9:16, dark eerie atmospheric horror art, cinematic shadows, "
            "fog, moonlight, highly detailed, no text, no logo"
        ),
        "image_negative_prompt": "text, watermark, logo, cartoon, sunny, bright, horizontal, black bars",
        "yt_token_env": "YT_REFRESH_TOKEN",
    },
}

DEFAULT_CHANNEL_ID = "90s_indian_nostalgia"


def get_preset(channel_id: str | None = None) -> ChannelPreset:
    key = (channel_id or DEFAULT_CHANNEL_ID).strip().lower()
    if key not in PRESETS:
        # Fallback to 90s nostalgia if unknown
        return PRESETS[DEFAULT_CHANNEL_ID]
    return PRESETS[key]


def list_channel_ids() -> list[str]:
    return list(PRESETS.keys())
