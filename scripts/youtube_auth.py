#!/usr/bin/env python3
"""
Run this ONCE on your local machine to get a YouTube OAuth refresh token.
Then store the token file path (or its contents) as a GitHub Actions secret.

Usage:
  .venv/bin/python scripts/youtube_auth.py

Prerequisites:
  1. Google Cloud Console → create project → enable "YouTube Data API v3"
     https://console.cloud.google.com/apis/library/youtube.googleapis.com
  2. Credentials → Create OAuth client ID → Desktop app → Download JSON
  3. Save that JSON as  secrets/client_secret.json  (gitignored)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
SECRETS = REPO_ROOT / "secrets"


def main() -> None:
    client_secret_file = SECRETS / "client_secret.json"
    token_out = SECRETS / "youtube_token.json"

    env_client_id = os.environ.get("YT_CLIENT_ID", "").strip()
    env_client_secret = (
        os.environ.get("YT_CLIENT_SECRET", "").strip()
        or os.environ.get("YT_CLIENT_SECRET_VALUE", "").strip()
    )

    flow = None
    if client_secret_file.is_file():
        flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_file), SCOPES)
    elif env_client_id and env_client_secret:
        client_config = {
            "installed": {
                "client_id": env_client_id,
                "client_secret": env_client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost", "http://localhost:8080/"],
            }
        }
        flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    else:
        print(
            "❌ Missing OAuth Client Credentials.\n\n"
            "You can provide credentials in EITHER of two ways:\n\n"
            "Option A: Download JSON file:\n"
            "  1. Go to Google Cloud Console: https://console.cloud.google.com/apis/credentials\n"
            "  2. Click on your OAuth 2.0 Client ID (type 'Desktop app')\n"
            "  3. Download JSON and save it as: secrets/client_secret.json\n\n"
            "Option B: Add to .env file:\n"
            "  Add your client credentials to .env:\n"
            "    YT_CLIENT_ID=your_client_id_here\n"
            "    YT_CLIENT_SECRET=your_client_secret_here\n"
        )
        sys.exit(1)

    print("Opening browser for Google OAuth…")
    # prompt='consent' and access_type='offline' ensures Google always returns a fresh refresh token
    creds = flow.run_local_server(port=0, prompt="consent", access_type="offline")

    SECRETS.mkdir(parents=True, exist_ok=True)
    token_out.write_text(creds.to_json(), encoding="utf-8")
    print(f"\n✅ Fresh token saved locally to {token_out}")

    token_data = json.loads(creds.to_json())
    new_refresh_token = token_data.get("refresh_token")

    if not new_refresh_token:
        print("\n⚠️ Warning: No refresh token returned. Ensure prompt='consent' was used.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("── NEXT STEPS FOR GITHUB ACTIONS ──")
    print("=" * 60)
    print("1. Go to your GitHub repository:")
    print("   Settings → Secrets and variables → Actions")
    print("2. Update this Repository Secret:")
    print("   Secret Name:  YT_REFRESH_TOKEN")
    print("   Secret Value: (copy the fresh token printed below)")
    print(f"\n   YT_REFRESH_TOKEN = {new_refresh_token}\n")
    print("=" * 60)
    print("── CRITICAL: PREVENT TOKEN EXPIRING AFTER 7 DAYS ──")
    print("=" * 60)
    print("If your Google Cloud OAuth Consent Screen is in 'Testing' mode,")
    print("Google automatically expires refresh tokens after exactly 7 days!")
    print("\nTo make your token permanent (never expire):")
    print("  1. Go to: https://console.cloud.google.com/apis/credentials/consent")
    print("  2. Under 'Publishing status', click 'PUBLISH APP' (change from Testing to In Production)")
    print("  3. Confirm the dialog. (You do NOT need to complete Google verification for personal use.)")
    print("=" * 60)


if __name__ == "__main__":
    main()
