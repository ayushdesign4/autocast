"""Email notification service for ZeroCost Shorts pipeline.

Sends structured success and failure alerts using standard library SMTP.
Uses existing repository secrets:
  - SMTP_HOST
  - SMTP_PORT (default 587)
  - SMTP_USERNAME
  - SMTP_PASSWORD
  - NOTIFICATION_EMAIL
"""
from __future__ import annotations

import logging
import os
import smtplib
from datetime import datetime, timezone, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import Any

log = logging.getLogger("pipeline.notifications")

# IST timezone: UTC + 5:30
IST = timezone(timedelta(hours=5, minutes=30))


def _now_ist_str() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %I:%M:%S %p IST")


def _get_smtp_config() -> dict[str, str] | None:
    """Retrieve and validate SMTP configuration from environment."""
    host = os.environ.get("SMTP_HOST", "").strip()
    port_str = os.environ.get("SMTP_PORT", "587").strip()
    user = os.environ.get("SMTP_USERNAME", "").strip()
    pw = os.environ.get("SMTP_PASSWORD", "").strip()
    recipient = os.environ.get("NOTIFICATION_EMAIL", "").strip()

    if not (host and user and pw and recipient):
        log.warning("SMTP configuration incomplete; email alerts disabled.")
        return None

    try:
        port = int(port_str)
    except ValueError:
        port = 587

    return {
        "host": host,
        "port": port,
        "user": user,
        "password": pw,
        "recipient": recipient,
    }


def _send_email(subject: str, body_text: str) -> bool:
    """Send a plain text email via SMTP with TLS."""
    cfg = _get_smtp_config()
    if not cfg:
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["user"]
    msg["To"] = cfg["recipient"]
    msg.set_content(body_text)

    try:
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(cfg["user"], cfg["password"])
            server.send_message(msg)
        log.info("Email alert successfully sent to %s: '%s'", cfg["recipient"], subject)
        return True
    except Exception as exc:
        # Never log passwords or secrets
        log.warning("Failed to send email alert via %s:%s: %s", cfg["host"], cfg["port"], exc)
        return False


def send_upload_success_email(
    *,
    title: str,
    video_id: str,
    video_url: str,
    topic: str,
    summary: str,
    duration_s: float,
    run_id: str = "",
    github_run_id: str = "",
    github_repository: str = "",
    tags: list[str] | None = None,
) -> bool:
    """Send detailed success notification after confirmed YouTube upload."""
    subject = f"[ZeroCost Shorts] Upload successful — {title[:50]}"
    ist_time = _now_ist_str()

    repo = github_repository or os.environ.get("GITHUB_REPOSITORY", "ayushdesign4/autocast")
    gh_run = github_run_id or os.environ.get("GITHUB_RUN_ID", "")
    run_url = f"https://github.com/{repo}/actions/runs/{gh_run}" if gh_run else "Manual Run"

    body = f"""ZeroCost Shorts — Daily Upload Succeeded
==================================================

Title:       {title}
Status:      UPLOAD SUCCESSFUL
YouTube URL: {video_url}
Video ID:    {video_id}

Topic:       {topic}
Summary:     {summary}
Duration:    {duration_s:.1f}s
Format:      1080x1920 Vertical (9:16 Shorts)
Voice:       Sarvam AI / shubh (hi-IN)
Timestamp:   {ist_time}

Run Identity: {run_id or 'daily_run'}
Actions Run:  {run_url}
Tags:         {', '.join(tags or [])}

This Short is live/private on your channel as scheduled.
"""
    return _send_email(subject, body)


def send_failure_alert_email(
    *,
    stage: str,
    error_message: str,
    topic: str = "",
    run_id: str = "",
    github_run_id: str = "",
    github_repository: str = "",
    upload_attempted: bool = False,
    video_id: str | None = None,
) -> bool:
    """Send immediate failure alert email with sanitized context and actionable guidance."""
    subject = f"[ZeroCost Shorts Alert] Pipeline failed at '{stage}'"
    ist_time = _now_ist_str()

    repo = github_repository or os.environ.get("GITHUB_REPOSITORY", "ayushdesign4/autocast")
    gh_run = github_run_id or os.environ.get("GITHUB_RUN_ID", "")
    run_url = f"https://github.com/{repo}/actions/runs/{gh_run}" if gh_run else "Manual Run"

    # Actionable guidance by category
    advice = "Inspect the GitHub Actions workflow logs for stack trace."
    clean_err = str(error_message)[:400]

    if "402" in clean_err or "quota" in clean_err.lower() or "budget" in clean_err.lower():
        advice = "API account quota or credits exhausted. Top up your API balance on the provider platform."
    elif "401" in clean_err or "403" in clean_err or "auth" in clean_err.lower():
        advice = "API key or secret rejected. Verify the corresponding secret in GitHub repository settings."
    elif "429" in clean_err or "rate limit" in clean_err.lower():
        advice = "Provider rate limit reached. The workflow can be re-run after cooldown."
    elif "ffprobe" in clean_err.lower() or "duration" in clean_err.lower():
        advice = "Rendered video validation failed before upload (duration or stream checks). Video was NOT uploaded."

    body = f"""ZeroCost Shorts — Pipeline Execution Failure
==================================================

Status:           FAILED
Failing Stage:    {stage}
Timestamp:        {ist_time}
Topic:            {topic or 'N/A'}
Upload Attempted: {'Yes' if upload_attempted else 'No'}
Known Video ID:   {video_id or 'None'}

Error Summary:
--------------
{clean_err}

Recommended Next Action:
------------------------
{advice}

GitHub Actions Run: {run_url}
Run ID:             {run_id or 'daily_run'}
"""
    return _send_email(subject, body)
