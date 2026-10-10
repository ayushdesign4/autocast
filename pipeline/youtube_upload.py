"""YouTube Shorts upload via OAuth2.

Reconstructs credentials in memory directly from GitHub Actions secrets:
  - YT_CLIENT_ID
  - YT_CLIENT_SECRET (or YT_CLIENT_SECRET_VALUE)
  - YT_REFRESH_TOKEN
Uploads as a vertical YouTube Short with privacy default to 'private'.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

log = logging.getLogger("pipeline.youtube_upload")

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
TOKEN_URI = "https://oauth2.googleapis.com/token"


def get_youtube_credentials(refresh_token_env: str = "YT_REFRESH_TOKEN") -> Credentials:
    """Reconstruct OAuth2 credentials in memory from environment variables."""
    client_id = os.environ.get("YT_CLIENT_ID", "").strip()
    client_secret = (
        os.environ.get("YT_CLIENT_SECRET", "").strip()
        or os.environ.get("YT_CLIENT_SECRET_VALUE", "").strip()
    )
    refresh_token = os.environ.get(refresh_token_env, "").strip()
    if not refresh_token and refresh_token_env != "YT_REFRESH_TOKEN":
        refresh_token = os.environ.get("YT_REFRESH_TOKEN", "").strip()

    if not (client_id and client_secret and refresh_token):
        missing = []
        if not client_id:
            missing.append("YT_CLIENT_ID")
        if not client_secret:
            missing.append("YT_CLIENT_SECRET")
        if not refresh_token:
            missing.append(refresh_token_env)
        raise RuntimeError(f"Missing required YouTube OAuth secrets: {', '.join(missing)}")

    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri=TOKEN_URI,
        client_id=client_id,
        client_secret=client_secret,
        scopes=SCOPES,
    )

    try:
        creds.refresh(Request())
    except RefreshError as exc:
        raise RuntimeError(
            f"YouTube OAuth refresh failed: {exc}. "
            "Please ensure OAuth consent screen is published and YT_REFRESH_TOKEN is active."
        ) from exc

    return creds


def upload_short(
    video_path: Path,
    title: str,
    description: str,
    *,
    tags: list[str] | None = None,
    privacy_status: str = "private",
    category_id: str = "24",
    refresh_token_env: str = "YT_REFRESH_TOKEN",
    max_retries: int = 3,
) -> str:
    """Upload a rendered vertical Short to YouTube and return the video ID."""
    video_path = Path(video_path)
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise FileNotFoundError(f"Video file not found or empty: {video_path}")

    # Ensure title contains #Shorts or is formatted well
    clean_title = title.strip()
    clean_desc = description.strip()
    if "#Shorts" not in clean_title and "#Shorts" not in clean_desc:
        clean_desc = f"{clean_desc}\n\n#Shorts #90sNostalgia #IndianNostalgia"

    creds = get_youtube_credentials(refresh_token_env)
    youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)

    body = {
        "snippet": {
            "title": clean_title[:100],
            "description": clean_desc[:5000],
            "tags": tags or ["90s nostalgia", "indian nostalgia", "Shorts"],
            "categoryId": category_id,
            "defaultLanguage": "hi",
            "defaultAudioLanguage": "hi",
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        str(video_path),
        mimetype="video/mp4",
        chunksize=10 * 1024 * 1024,
        resumable=True,
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    log.info("Uploading '%s' (%s, %s)...", clean_title[:40], video_path.name, privacy_status)
    response = None
    backoff_s = 5.0

    for attempt in range(max_retries + 1):
        try:
            status, response = request.next_chunk()
            while response is None:
                if status:
                    log.info("Uploaded %d%%", int(status.progress() * 100))
                status, response = request.next_chunk()
            break
        except (HttpError, OSError) as exc:
            if attempt < max_retries:
                log.warning("YouTube upload transient error (%s); retrying in %.1fs...", exc, backoff_s)
                time.sleep(backoff_s)
                backoff_s *= 2.0
                continue
            raise RuntimeError(f"YouTube upload failed after {max_retries} retries: {exc}") from exc

    video_id = response.get("id") if response else None
    if not video_id:
        raise RuntimeError(f"YouTube upload response missing video ID: {response}")

    log.info("Successfully uploaded video ID: %s (https://www.youtube.com/shorts/%s)", video_id, video_id)
    return video_id
