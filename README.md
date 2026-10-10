# ZeroCost-Shorts — 1990s Indian Nostalgia

An end-to-end automated pipeline for generating and uploading high-retention, vertical (9:16) **YouTube Shorts** celebrating everyday **1990s Indian nostalgia**.

Runs fully autonomously every day via GitHub Actions at **04:00 AM IST** (22:30 UTC), requiring zero local computer resources.

---

## 🌟 Channel Niche: 1990s Indian Nostalgia

- **Storytelling:** Poetic, warm, and authentic childhood memories of 1990s India (Doordarshan Sunday mornings, cassette tape repairs, school tiffins, summer holidays at nani's house, street cricket, and handwritten inland letters).
- **Narration:** Natural, fluent Devanagari Hindi (130–180 words, ~45–75 seconds runtime).
- **Voiceover:** **Sarvam AI** Text-to-Speech using the authentic **`shubh`** voice persona (`hi-IN`).
- **Visuals:** High-aesthetic vertical (1080×1920) scene art generated via **Pollinations AI** with 1990s retro cinema styling.
- **Captions:** Clean, high-contrast burned-in subtitles with authentic Devanagari Unicode rendering via `Noto Sans Devanagari`.
- **Assembly:** Smooth Ken Burns pan/zoom and transitions compiled using **FFmpeg**.
- **Upload:** Fully automated YouTube upload with title, description, tags, and category metadata (defaults to `private` for safety).
- **Notifications:** Success summaries and failure alerts sent directly to your email via SMTP.
- **Duplicate Prevention:** Durable topic, script, and video hash tracking in `data/history.json`.

---

## ⏰ Daily Execution Schedule

| Timezone | Scheduled Time | GitHub Actions Cron |
|---|---|---|
| **India Standard Time (IST)** | **04:00 AM** daily | `30 22 * * *` |
| **UTC** | **22:30 UTC** daily | `30 22 * * *` |

*Note: 22:30 UTC corresponds to 04:00 AM IST on the following day (UTC + 5:30).*

---

## 🔑 GitHub Secrets Configuration

The pipeline utilizes the following repository secrets (configured in **Settings → Secrets and variables → Actions**):

| Secret Name | Purpose |
|---|---|
| `GROQ_API_KEY` | Script, story, scene, and metadata generation via Groq (Llama-3.3-70B) |
| `POLLINATIONS_API_KEY` | Authenticated vertical 9:16 scene image generation |
| `SARVAM_API_KEY` | Native Hindi voiceover synthesis via Sarvam AI REST API (`shubh` voice) |
| `YT_CLIENT_ID` | YouTube Data API OAuth2 Client ID |
| `YT_CLIENT_SECRET` | YouTube Data API OAuth2 Client Secret |
| `YT_REFRESH_TOKEN` | YouTube Data API OAuth2 Refresh Token |
| `NOTIFICATION_EMAIL` | Email recipient for daily upload and failure reports |
| `SMTP_HOST` | Outgoing SMTP mail server (e.g. `smtp.gmail.com`) |
| `SMTP_PORT` | SMTP port (default `587`) |
| `SMTP_USERNAME` | SMTP login username / email address |
| `SMTP_PASSWORD` | SMTP password or App Password |

---

## 🛠 Local Setup & Testing

### 1. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Test Suite (Zero Cost / Mocked)
```bash
pytest
```

### 3. Generate a Short Locally (Dry-Run / No Upload)
```bash
python scripts/run_short.py --channel 90s_indian_nostalgia
```

### 4. Generate & Upload
```bash
python scripts/run_short.py --channel 90s_indian_nostalgia --upload --privacy private
```

---

## 📁 Project Architecture

```
├── .github/
│   └── workflows/
│       └── daily_short.yml       # Production 04:00 AM IST cron workflow
├── assets/
│   └── fonts/
│       ├── NotoSansDevanagari-Bold.ttf  # Hindi subtitle font
│       └── BebasNeue-Regular.ttf
├── data/
│   └── history.json              # Durable history to prevent duplicate stories
├── pipeline/
│   ├── captions.py               # Subtitle timings & SRT generation
│   ├── channel_presets.py        # 1990s Nostalgia theme & prompt configurations
│   ├── groq_script.py            # Groq structured script generator & fallbacks
│   ├── images.py                 # Pollinations AI 9:16 vertical image generator
│   ├── notifications.py          # SMTP success and failure email dispatcher
│   ├── render_short.py           # FFmpeg vertical compiler & duration validator
│   ├── sarvam_tts.py             # Sarvam AI REST voiceover integration
│   ├── story_history.py          # Duplicate prevention and story history
│   └── youtube_upload.py         # YouTube OAuth2 upload client
├── scripts/
│   ├── run_short.py              # Main pipeline orchestrator
│   └── youtube_auth.py           # Local YouTube OAuth token generator helper
├── tests/                        # Comprehensive mock-based unit tests
├── requirements.txt
└── pyproject.toml
```

---

## 📜 License
MIT License
