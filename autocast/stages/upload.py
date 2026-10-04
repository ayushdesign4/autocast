"""Stage: upload — YouTube Data API v3 videos.insert (PRIVATE) + thumbnails.set.

Reads `spine.topic`, `spine.script`, `spine.video`, `spine.thumbnail`; writes
`spine.upload`. Upload metadata is DERIVED from the script/topic/scenes so nothing
drifts from the content.

Security & Compliance:
  - Privacy is hardcoded to "private" per project compliance policy.
  - Credentials are read exclusively from Config / env (never logged or hardcoded).
  - Dry-run never touches the network and assigns a stub video ID.
  - Real upload skips safely with StageSkipped if credentials are not configured.
  - Built-in post-upload notification hook triggers email alert if configured.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path

import httpx

from autocast.config import Config
from autocast.spine import Run, StageSkipped, Upload
from autocast.util.youtube_auth import YouTubeAuthError, get_access_token

log = logging.getLogger("autocast.stages.upload")

STAGE = "upload"

_VIDEOS_RESUMABLE_URL = "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status"
_THUMBNAILS_SET_URL = "https://www.googleapis.com/upload/youtube/v3/thumbnails/set"

_DISCLAIMER = (
    "\n\n---\n"
    "This video features AI-generated 1990s nostalgic animation, dialogue, and music.\n"
    "Created autonomously by AutoCast."
)


def _derive_description(spine: Run) -> str:
    """Generate SEO-rich description from script and scenes."""
    parts: list[str] = []
    if spine.topic and spine.topic.title:
        parts.append(spine.topic.title)

    if spine.script and spine.script.full_text:
        intro = spine.script.full_text[:400].strip()
        parts.append(intro)

    # Nostalgic hashtags for YouTube discovery
    parts.append("#90sIndia #Nostalgia #HindiStories #Animation #VillageLife #AutoCast")
    parts.append(_DISCLAIMER)
    return "\n\n".join(parts)


def _derive_tags(spine: Run) -> list[str]:
    """Generate curated tags for 1990s nostalgia Hindi stories."""
    base_tags = [
        "90s India",
        "nostalgia",
        "1990s Indian village",
        "Hindi animation",
        "Indian childhood",
        "Chiku",
        "mango summer",
        "village story",
        "nostalgic memories",
    ]
    if spine.topic and spine.topic.title:
        stop = {"the", "was", "why", "how", "and", "that", "still", "some", "a", "of", "in", "की", "का", "के"}
        words = [w.strip(".,!-?\"'").lower() for w in spine.topic.title.split()]
        extra = [w for w in words if w and len(w) > 2 and w not in stop]
        base_tags = extra[:4] + base_tags

    return base_tags[:15]


def _has_yt_creds(cfg: Config) -> bool:
    return bool(cfg.yt_client_id and cfg.yt_client_secret and cfg.yt_refresh_token)


def _send_upload_notification(cfg: Config, title: str, video_id: str) -> None:
    """Send optional email notification on successful upload via standard library smtplib."""
    yt_url = f"https://youtu.be/{video_id}"
    if not (cfg.notification_email and cfg.smtp_host):
        log.info("upload: notification skipped (NOTIFICATION_EMAIL or SMTP_HOST not configured). URL: %s", yt_url)
        return

    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    msg = MIMEMultipart()
    sender = cfg.smtp_username or "autocast@noreply.local"
    msg["From"] = sender
    msg["To"] = cfg.notification_email
    msg["Subject"] = f"[AutoCast] Video Uploaded: {title}"

    body = (
        f"AutoCast daily run successfully published a new video to YouTube!\n\n"
        f"Title: {title}\n"
        f"YouTube URL: {yt_url}\n"
        f"Video ID: {video_id}\n"
        f"Privacy Status: Private\n"
        f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}\n\n"
        f"-- AutoCast Autonomous Pipeline"
    )
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=20) as server:
            server.starttls()
            if cfg.smtp_username and cfg.smtp_password:
                server.login(cfg.smtp_username, cfg.smtp_password)
            server.send_message(msg)
        log.info("upload: notification email successfully sent to %s for video %s", cfg.notification_email, video_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("upload: failed to send email notification (%s), video upload succeeded", exc)


def _upload_video_resumable(
    video_path: Path,
    title: str,
    description: str,
    tags: list[str],
    access_token: str,
) -> str:
    """Upload MP4 video via YouTube Data API v3 resumable protocol."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found for upload: {video_path}")

    file_size = video_path.stat().st_size
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Length": str(file_size),
        "X-Upload-Content-Type": "video/mp4",
    }
    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "tags": tags[:50],
            "categoryId": "24",  # Entertainment
            "defaultLanguage": "hi",
            "defaultAudioLanguage": "hi",
        },
        "status": {
            "privacyStatus": "private",  # Compliance policy: always private
            "selfDeclaredMadeForKids": False,
        },
    }

    log.info("upload: initiating YouTube resumable upload session (size=%d bytes)...", file_size)
    init_resp = httpx.post(_VIDEOS_RESUMABLE_URL, headers=headers, json=body, timeout=45.0)

    if init_resp.status_code != 200:
        raise RuntimeError(
            f"YouTube API upload session init failed (HTTP {init_resp.status_code}): {init_resp.text}"
        )

    upload_url = init_resp.headers.get("Location")
    if not upload_url:
        raise RuntimeError("YouTube API did not return Location header for resumable upload.")

    log.info("upload: uploading video payload to YouTube...")
    with video_path.open("rb") as f:
        video_bytes = f.read()

    upload_headers = {
        "Content-Length": str(file_size),
        "Content-Type": "video/mp4",
    }
    upload_resp = httpx.put(upload_url, headers=upload_headers, content=video_bytes, timeout=300.0)

    if upload_resp.status_code not in (200, 201):
        raise RuntimeError(
            f"YouTube API video upload failed (HTTP {upload_resp.status_code}): {upload_resp.text}"
        )

    video_data = upload_resp.json()
    video_id = video_data.get("id")
    if not video_id:
        raise RuntimeError(f"YouTube upload response missing video ID: {upload_resp.text}")

    log.info("upload: video upload completed successfully (video_id=%s)", video_id)
    return video_id


def _upload_thumbnail(thumbnail_path: Path, video_id: str, access_token: str) -> bool:
    """Set custom thumbnail for the video via thumbnails.set."""
    if not thumbnail_path.exists():
        log.warning("upload: thumbnail file not found (%s), skipping", thumbnail_path)
        return False

    thumb_bytes = thumbnail_path.read_bytes()
    url = f"{_THUMBNAILS_SET_URL}?videoId={video_id}&uploadType=media"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "image/jpeg",
        "Content-Length": str(len(thumb_bytes)),
    }

    try:
        resp = httpx.post(url, headers=headers, content=thumb_bytes, timeout=45.0)
        if resp.status_code in (200, 201):
            log.info("upload: custom thumbnail set for video %s", video_id)
            return True
        log.warning(
            "upload: custom thumbnail upload returned HTTP %d: %s. "
            "(Channel may require phone verification for custom thumbnails; video upload remains valid)",
            resp.status_code,
            resp.text[:120],
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("upload: failed to set custom thumbnail (%s), video upload remains valid", exc)

    return False


def run(spine: Run, cfg: Config, *, dry_run: bool = False) -> Run:
    if spine.topic is None or spine.video is None:
        raise ValueError("upload stage: needs spine.topic and spine.video")

    title = spine.topic.title
    description = _derive_description(spine)
    tags = _derive_tags(spine)

    # Resumability check: if video already uploaded in a prior run attempt, don't duplicate
    if spine.upload and spine.upload.youtube_video_id and spine.upload.youtube_video_id != "DRYRUN_VIDEO_ID":
        log.info("upload: video already uploaded in spine (video_id=%s), skipping", spine.upload.youtube_video_id)
        return spine

    spine.upload = Upload(
        title=title,
        description=description,
        tags=tags,
        privacy="private",  # NEVER public until human audit
        ai_disclosure=True,
        youtube_video_id=None,
        uploaded_at=None,
    )

    if dry_run:
        get_access_token(cfg.yt_client_id, cfg.yt_client_secret, cfg.yt_refresh_token, dry_run=True)
        log.info("upload[DRY-RUN]: would insert video (privacy=private)")
        log.info("  title       = %s", title)
        log.info("  description = %s", description.replace("\n", " ")[:120] + "...")
        log.info("  tags        = %s", tags)
        log.info("  thumbnail   = %s", spine.thumbnail.path if spine.thumbnail else None)
        spine.upload.youtube_video_id = "DRYRUN_VIDEO_ID"
        spine.upload.uploaded_at = datetime.now(timezone.utc).isoformat()
        spine.stage(STAGE).provider_used = "dry-stub"
        log.info("upload: recorded video_id=%s (privacy=private)", spine.upload.youtube_video_id)
        return spine

    # Real upload: check credentials
    if not _has_yt_creds(cfg):
        raise StageSkipped(
            "no YouTube credentials (YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN) — video "
            "rendered locally, publishing skipped. Run 'uv run python -m autocast.util.youtube_auth' to authenticate."
        )

    # Obtain valid short-lived access token from refresh token
    access_token = get_access_token(
        cfg.yt_client_id,
        cfg.yt_client_secret,
        cfg.yt_refresh_token,
        dry_run=False,
    )

    video_path = cfg.run_dir(spine.run_id) / spine.video.final_path
    video_id = _upload_video_resumable(
        video_path=video_path,
        title=title,
        description=description,
        tags=tags,
        access_token=access_token,
    )

    # Upload thumbnail if available
    if spine.thumbnail and spine.thumbnail.path:
        thumb_path = cfg.run_dir(spine.run_id) / spine.thumbnail.path
        _upload_thumbnail(thumb_path, video_id, access_token)

    spine.upload.youtube_video_id = video_id
    spine.upload.uploaded_at = datetime.now(timezone.utc).isoformat()
    spine.stage(STAGE).provider_used = "youtube-data-api-v3"
    log.info("upload: published video https://youtu.be/%s (privacy=private)", video_id)

    # Optional notification hook
    _send_upload_notification(cfg, title, video_id)

    return spine
