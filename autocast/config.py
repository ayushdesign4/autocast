"""Typed configuration + secrets, loaded from environment variables.

Secrets come from GitHub Actions Secrets -> env -> here. NOTHING is committed.
See `.env.example` for the full list. Local dev may use a `.env` file (gitignored).

Per the CTO doc: `pydantic-settings` reading env vars. No config service.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo-relative default output roots. `runs/` is committed back by the Action.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Config(BaseSettings):
    """All runtime settings + secrets. Missing secrets are allowed at import
    time (they're only required by the stages that use them) so `--dry-run`
    works with zero keys."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- output dirs (state lives in the repo) ----
    runs_dir: Path = _PROJECT_ROOT / "runs"
    queue_path: Path = _PROJECT_ROOT / "queue" / "topics.json"

    # ---- render settings (mirrored into the spine's config_snapshot) ----
    voice: str = "af_heart"
    width: int = 1920
    height: int = 1080
    fps: int = 30
    target_len_s: int = 90

    # ---- LLM & Video providers (Agnes primary video/companion LLM, Gemini primary LLM/Veo fallback) ----
    agnes_api_key: str | None = Field(default=None)
    agnes_model: str = Field(
        default="agnes-video-2.5",
        validation_alias=AliasChoices(
            "AGNES_MODEL",
            "agnes_model",
        ),
    )
    agnes_api_base: str = Field(
        default="https://apihub.agnes-ai.com",
        validation_alias=AliasChoices(
            "AGNES_API_BASE",
            "agnes_api_base",
        ),
    )
    agnes_rate_limit_seconds: float = 60.0
    video_provider: str = "agnes"

    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "gemini_api_key",
            "google_api_key",
        ),
    )
    gemini_model: str = "gemini-3.8-flash"
    veo_model: str = "veo-3.1-generate-preview"
    groq_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "GROQ_API_KEY",
            "GROQ_KEY",
            "groq_api_key",
            "groq_key",
        ),
    )
    cerebras_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "CEREBRAS_API_KEY",
            "CEREBRAS_KEY",
            "cerebras_api_key",
            "cerebras_key",
        ),
    )

    @field_validator(
        "agnes_api_key",
        "gemini_api_key",
        "groq_api_key",
        "cerebras_api_key",
        "cloudflare_api_token",
        "cloudflare_account_id",
        mode="before",
    )
    @classmethod
    def _clean_empty_secrets(cls, v: str | None) -> str | None:
        if isinstance(v, str):
            cleaned = v.strip()
            return cleaned if cleaned else None
        return v

    @field_validator("agnes_model", mode="before")
    @classmethod
    def _clean_agnes_model(cls, v: object) -> str:
        if isinstance(v, str) and not v.strip():
            return "agnes-video-2.5"
        return str(v).strip() if v else "agnes-video-2.5"

    @field_validator("agnes_api_base", mode="before")
    @classmethod
    def _clean_agnes_api_base(cls, v: object) -> str:
        if isinstance(v, str) and not v.strip():
            return "https://apihub.agnes-ai.com"
        return str(v).strip() if v else "https://apihub.agnes-ai.com"

    # ---- Cloudflare Workers AI (LLM + image fallback). Bills on overage:
    #      cascade.py must guard it behind the budget kill-switch.
    cloudflare_api_token: str | None = Field(default=None)
    cloudflare_account_id: str | None = Field(default=None)

    # ---- image providers (gemini-image -> pollinations[keyless] -> cloudflare-flux)
    #      Pollinations needs no key.

    # ---- assets ----
    pixabay_api_key: str | None = Field(default=None)
    pexels_api_key: str | None = Field(default=None)
    freesound_api_key: str | None = Field(default=None)

    # ---- YouTube OAuth (Desktop client + one refresh token; see README gate) ----
    yt_client_id: str | None = Field(default=None)
    yt_client_secret: str | None = Field(default=None)
    yt_refresh_token: str | None = Field(default=None)

    # ---- Optional Post-Upload Email Notifications (standard library SMTP) ----
    notification_email: str | None = Field(default=None)
    smtp_host: str | None = Field(default=None)
    smtp_port: int = 587
    smtp_username: str | None = Field(default=None)
    smtp_password: str | None = Field(default=None)

    # ---- safety: Cloudflare overage budget kill-switch (USD cents). 0 = never
    #      allow paid Cloudflare fallback. Raise deliberately once a budget is set.
    cloudflare_budget_cents: int = 0

    # ---- helpers ----

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def run_json_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "run.json"

    def assets_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "assets"

    def manifest_path(self) -> Path:
        return self.runs_dir / "manifest.json"


def load_config() -> Config:
    """Single entry point so the rest of the code never touches env directly."""
    return Config()
