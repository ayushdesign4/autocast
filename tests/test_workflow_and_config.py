"""Tests for GitHub Actions workflow schedule, secret mapping, and exclusion of prohibited providers."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "daily_short.yml"


def test_workflow_exists_and_cron_is_4am_ist():
    """Verify workflow cron is 30 22 * * * (22:30 UTC = 04:00 AM IST next day)."""
    assert WORKFLOW_PATH.is_file(), f"Workflow file missing at {WORKFLOW_PATH}"
    content = WORKFLOW_PATH.read_text(encoding="utf-8")

    # Verify exact cron
    assert 'cron: "30 22 * * *"' in content or "cron: '30 22 * * *'" in content, "Cron must be '30 22 * * *'"

    # Verify UTC/IST comment is present
    assert "22:30 UTC" in content or "04:00" in content, "Workflow must contain comment explaining 22:30 UTC = 04:00 IST"

    # Verify workflow_dispatch is present
    assert "workflow_dispatch:" in content


def test_workflow_secret_mappings():
    """Verify all required GitHub secrets are mapped to environment variables."""
    content = WORKFLOW_PATH.read_text(encoding="utf-8")

    required_secrets = [
        "GROQ_API_KEY",
        "POLLINATIONS_API_KEY",
        "SARVAM_API_KEY",
        "YT_CLIENT_ID",
        "YT_CLIENT_SECRET",
        "YT_REFRESH_TOKEN",
        "NOTIFICATION_EMAIL",
        "SMTP_HOST",
        "SMTP_PORT",
        "SMTP_USERNAME",
        "SMTP_PASSWORD",
    ]

    for secret in required_secrets:
        assert f"secrets.{secret}" in content, f"Secret {secret} must be mapped in workflow"


def test_no_agnes_gemini_or_edge_tts_in_pipeline():
    """Verify that Agnes, Gemini, and Edge-TTS are not imported or required in pipeline codebase."""
    pipeline_dir = REPO_ROOT / "pipeline"
    scripts_dir = REPO_ROOT / "scripts"

    py_files = list(pipeline_dir.glob("*.py")) + list(scripts_dir.glob("*.py"))
    assert py_files, "Pipeline and script python files must exist"

    prohibited_modules = ["agnes", "gemini", "edge_tts", "kokoro", "whisperx"]

    for py_file in py_files:
        text = py_file.read_text(encoding="utf-8").lower()
        for mod in prohibited_modules:
            # Check for direct imports
            assert f"import {mod}" not in text, f"Prohibited import '{mod}' found in {py_file.name}"
            assert f"from {mod}" not in text, f"Prohibited from-import '{mod}' found in {py_file.name}"


def test_channel_presets_default_is_90s_nostalgia():
    """Verify that 90s_indian_nostalgia is the primary default preset."""
    from pipeline.channel_presets import get_preset, DEFAULT_CHANNEL_ID

    assert DEFAULT_CHANNEL_ID == "90s_indian_nostalgia"
    preset = get_preset()
    assert preset["id"] == "90s_indian_nostalgia"
    assert preset["language"] == "hi"
    assert preset["tts_voice"] == "shubh"
    assert "NotoSansDevanagari-Bold.ttf" in preset["caption_font"]
    assert len(preset["topic_pool"]) >= 10
    assert preset["min_words"] >= 120
