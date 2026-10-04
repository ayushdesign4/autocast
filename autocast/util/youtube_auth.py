"""YouTube OAuth: interactive authorization + refresh token exchange.

OAuth shape (per research doc §8):
  - Google Cloud OAuth 2.0 Desktop Application client.
  - Scopes: `youtube.upload` + `youtube.force-ssl` (for custom thumbnails).
  - One-time interactive authorization (access_type=offline, prompt=consent)
    via loopback redirect (http://127.0.0.1:8080).
  - Permanent refresh token stored in `.env` locally and in GitHub Repository Secrets.
  - Automatically exchanged for short-lived access tokens during pipeline runs.
"""

from __future__ import annotations

import http.server
import logging
import os
from pathlib import Path
import socketserver
import sys
import urllib.parse
import webbrowser

import httpx

log = logging.getLogger("autocast.util.youtube_auth")

_TOKEN_URL = "https://oauth2.googleapis.com/token"
_AUTH_BASE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_SCOPES = (
    "https://www.googleapis.com/auth/youtube.upload "
    "https://www.googleapis.com/auth/youtube.force-ssl"
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class YouTubeAuthError(RuntimeError):
    pass


def get_access_token(
    client_id: str | None,
    client_secret: str | None,
    refresh_token: str | None,
    *,
    dry_run: bool = False,
) -> str:
    """Return a valid access token, refreshing from the refresh token.

    In dry-run we never touch the network — return a fake token.
    In real runs, exchanges refresh_token at https://oauth2.googleapis.com/token.
    """
    if dry_run:
        return "DRYRUN_ACCESS_TOKEN"

    if not (client_id and client_secret and refresh_token):
        raise YouTubeAuthError(
            "Missing YouTube credentials (YT_CLIENT_ID, YT_CLIENT_SECRET, or YT_REFRESH_TOKEN). "
            "Run 'uv run python -m autocast.util.youtube_auth' to perform one-time authorization."
        )

    try:
        resp = httpx.post(
            _TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=30.0,
        )
    except Exception as exc:
        raise YouTubeAuthError(f"Network error refreshing YouTube access token: {exc}") from exc

    if resp.status_code != 200:
        error_detail = "unknown"
        try:
            error_data = resp.json()
            error_detail = error_data.get("error_description") or error_data.get("error", "unknown")
        except Exception:
            error_detail = resp.text[:100]

        raise YouTubeAuthError(
            f"Failed to refresh YouTube access token (HTTP {resp.status_code}): {error_detail}. "
            "The refresh token may have expired or been revoked. Re-run local authorization."
        )

    token = resp.json().get("access_token")
    if not token:
        raise YouTubeAuthError("OAuth token endpoint returned 200 OK but no access_token found in response.")

    return token


def _update_env_file(key: str, value: str, env_path: Path | None = None) -> None:
    """Update or append an environment variable in the local .env file."""
    if env_path is None:
        env_path = _PROJECT_ROOT / ".env"

    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    prefix = f"{key}="
    found = False
    new_lines: list[str] = []
    for line in lines:
        if line.strip().startswith(prefix):
            new_lines.append(f"{key}={value}")
            found = True
        else:
            new_lines.append(line)

    if not found:
        new_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def authorize_interactive(
    client_id: str | None = None,
    client_secret: str | None = None,
    port: int = 8080,
) -> str:
    """Execute one-time interactive OAuth authorization.

    1. Starts a temporary local HTTP server on http://127.0.0.1:{port}.
    2. Opens Google consent page in the user's browser.
    3. Receives authorization code upon redirect.
    4. Exchanges authorization code for a permanent refresh_token.
    5. Saves refresh_token to the local `.env` file without echoing it to stdout.
    """
    if not client_id or not client_secret:
        # Try loading from environment or .env
        from autocast.config import load_config

        cfg = load_config()
        client_id = client_id or cfg.yt_client_id or os.environ.get("YT_CLIENT_ID")
        client_secret = client_secret or cfg.yt_client_secret or os.environ.get("YT_CLIENT_SECRET")

    if not client_id or not client_secret:
        print("\n[!] YT_CLIENT_ID or YT_CLIENT_SECRET not found.")
        print("Please provide them when prompted, or configure them in your local .env file.\n")
        if not client_id:
            client_id = input("Enter Google OAuth Client ID: ").strip()
        if not client_secret:
            client_secret = input("Enter Google OAuth Client Secret: ").strip()

    if not client_id or not client_secret:
        raise YouTubeAuthError("Authorization aborted: Client ID and Client Secret are required.")

    # Save Client ID & Secret to .env if not already present
    _update_env_file("YT_CLIENT_ID", client_id)
    _update_env_file("YT_CLIENT_SECRET", client_secret)

    redirect_uri = f"http://127.0.0.1:{port}"
    auth_params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": _SCOPES,
        "access_type": "offline",
        "prompt": "consent",
    }
    auth_url = f"{_AUTH_BASE_URL}?{urllib.parse.urlencode(auth_params)}"

    auth_code_holder: dict[str, str | None] = {"code": None, "error": None}

    class OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)

            if "code" in query:
                auth_code_holder["code"] = query["code"][0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                success_html = (
                    "<!DOCTYPE html><html><body style='font-family:sans-serif;text-align:center;padding:60px;'>"
                    "<h2 style='color:#16a34a;'>AutoCast YouTube Authorization Successful!</h2>"
                    "<p>Your refresh token has been securely acquired.</p>"
                    "<p style='color:#6b7280;'>You can close this browser tab and return to your terminal.</p>"
                    "</body></html>"
                )
                self.wfile.write(success_html.encode("utf-8"))
            else:
                error_msg = query.get("error", ["Unknown error"])[0]
                auth_code_holder["error"] = error_msg
                self.send_response(400)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                fail_html = (
                    f"<!DOCTYPE html><html><body style='font-family:sans-serif;text-align:center;padding:60px;'>"
                    f"<h2 style='color:#dc2626;'>Authorization Failed</h2>"
                    f"<p>Google returned error: {error_msg}</p>"
                    f"<p>Please return to your terminal and try again.</p>"
                    f"</body></html>"
                )
                self.wfile.write(fail_html.encode("utf-8"))

        def log_message(self, format, *args):  # noqa: A002
            # Suppress HTTP server request logging so tokens/query parameters are never logged
            return

    # Attempt binding to port with socket reuse
    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    try:
        httpd = ReusableTCPServer(("127.0.0.1", port), OAuthCallbackHandler)
    except OSError as exc:
        raise YouTubeAuthError(
            f"Could not bind to local port {port} on 127.0.0.1. "
            "Ensure another application is not using port 8080."
        ) from exc

    print("\n" + "=" * 65)
    print("AutoCast YouTube OAuth Authorization")
    print("=" * 65)
    print(f"Opening browser to authorize with your YouTube channel...")
    print(f"If the browser does not open automatically, visit this URL:\n\n{auth_url}\n")
    print("Waiting for authorization callback...")

    webbrowser.open(auth_url)

    # Handle the single callback request
    try:
        httpd.handle_request()
    finally:
        httpd.server_close()

    if auth_code_holder["error"]:
        raise YouTubeAuthError(f"OAuth authorization failed with error: {auth_code_holder['error']}")

    auth_code = auth_code_holder["code"]
    if not auth_code:
        raise YouTubeAuthError("No authorization code received from Google callback.")

    print("Authorization code received. Exchanging for refresh token...")

    token_resp = httpx.post(
        _TOKEN_URL,
        data={
            "code": auth_code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=30.0,
    )

    if token_resp.status_code != 200:
        raise YouTubeAuthError(
            f"Failed to exchange code for tokens (HTTP {token_resp.status_code}): {token_resp.text}"
        )

    tokens = token_resp.json()
    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        raise YouTubeAuthError(
            "Google did not return a refresh token. "
            "Ensure access_type=offline and prompt=consent were used, or revoke app permissions in "
            "Google Account Security settings and retry."
        )

    # Securely save to .env
    _update_env_file("YT_REFRESH_TOKEN", refresh_token)

    print("\n" + "=" * 65)
    print("SUCCESS: YouTube authorization completed!")
    print(f"Saved credentials to: {_PROJECT_ROOT / '.env'}")
    print("YT_REFRESH_TOKEN has been stored securely in your local .env file.")
    print("=" * 65 + "\n")

    return refresh_token


def main():
    """CLI entrypoint for interactive authorization."""
    try:
        authorize_interactive()
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
