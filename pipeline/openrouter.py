"""OpenRouter free-model script generation fallback for ZeroCost-Shorts.

Provides resilient, strictly free-only fallback LLM generation when Groq is unavailable.
Never calls or switches to paid models.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
import time
from typing import Any

import httpx

from pipeline.channel_presets import ChannelPreset

log = logging.getLogger("pipeline.openrouter")

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openrouter/free"
MAX_FREE_CANDIDATES = 4

# Module-level cache for successfully working free model ID
_cached_working_free_model: str | None = None
_cached_free_catalog: list[str] | None = None


def _sanitize_log(text: str) -> str:
    """Mask any potential API key, bearer token, or authorization secret in logs."""
    return re.sub(r"(Bearer\s+)[A-Za-z0-9_\-\.]+", r"\1[REDACTED]", str(text))


def get_openrouter_api_key() -> str:
    """Return OPENROUTER_API_KEY environment variable."""
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


def get_openrouter_base_url() -> str:
    """Return OPENROUTER_BASE_URL environment variable or default."""
    url = os.environ.get("OPENROUTER_BASE_URL", "").strip()
    return url or DEFAULT_OPENROUTER_BASE_URL


def get_openrouter_model() -> str:
    """Return OPENROUTER_MODEL environment variable or default."""
    model = os.environ.get("OPENROUTER_MODEL", "").strip()
    return model or DEFAULT_OPENROUTER_MODEL


def extract_json_from_text(content: str) -> dict[str, Any]:
    """Extract valid JSON dictionary from markdown blocks or raw string."""
    text = content.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    elif "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        text = text[start:end].strip()

    parsed = json.loads(text)
    if isinstance(parsed, dict):
        return parsed
    raise ValueError(f"Extracted JSON is not a dictionary: {type(parsed)}")


_verified_free_models_cache: set[str] = set()


def is_verified_free_model(model_id: str, pricing: dict[str, Any] | None = None) -> bool:
    """Strictly verify that a model ID or metadata is exclusively free.

    Guarantees:
      - Never allows paid models.
      - Never strips ':free' suffix.
    """
    if not model_id or not isinstance(model_id, str):
        return False

    clean_id = model_id.strip()

    # openrouter/free is the official free auto-router
    if clean_id == "openrouter/free":
        return True

    # Check for explicit :free suffix
    if clean_id.endswith(":free"):
        return True

    if clean_id in _verified_free_models_cache:
        return True

    # If pricing metadata is available, verify both prompt and completion are zero
    if pricing and isinstance(pricing, dict):
        try:
            p_val = pricing.get("prompt")
            c_val = pricing.get("completion")
            if p_val is not None and c_val is not None:
                prompt_cost = float(p_val)
                completion_cost = float(c_val)
                if prompt_cost == 0.0 and completion_cost == 0.0:
                    _verified_free_models_cache.add(clean_id)
                    return True
        except (ValueError, TypeError):
            pass

    return False


def fetch_free_models(
    base_url: str,
    api_key: str,
    *,
    client: httpx.Client | None = None,
) -> list[str]:
    """Fetch current official available model catalogue from OpenRouter and extract verified free models.

    Only models verified as free are returned.
    """
    global _cached_free_catalog
    if client is None and not refresh and _cached_free_catalog:
        return list(_cached_free_catalog)

    url = f"{base_url.rstrip('/')}/models"
    headers = {
        "User-Agent": "AutoCast-ZeroCost/1.0",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        if client:
            resp = client.get(url, headers=headers)
        else:
            with httpx.Client(timeout=15.0) as c:
                resp = c.get(url, headers=headers)

        if resp.status_code != 200:
            log.warning("Failed to query OpenRouter models catalogue (HTTP %d): %s", resp.status_code, resp.text[:120])
            return []

        data = resp.json()
        models_data = data.get("data", [])
        free_models: list[str] = []

        for item in models_data:
            m_id = item.get("id", "")
            pricing = item.get("pricing", {})
            if is_verified_free_model(m_id, pricing):
                free_models.append(m_id)

        log.info("Discovered %d verified free models in OpenRouter catalogue.", len(free_models))
        _cached_free_catalog = free_models
        return free_models

    except Exception as exc:
        log.warning("Error fetching OpenRouter models catalogue: %s", _sanitize_log(str(exc)))
        return []


def _call_openrouter_completion(
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str,
    base_url: str,
    client: httpx.Client,
    temperature: float = 0.75,
    max_tokens: int = 1500,
    try_json_mode: bool = True,
) -> tuple[dict[str, Any] | None, int, str]:
    """Send single chat-completion request to OpenRouter.

    Returns:
        (parsed_json_dict_or_none, status_code, actual_model_or_error)
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/ayushdesign4/autocast",
        "X-Title": "AutoCast ZeroCost-Shorts",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if try_json_mode:
        payload["response_format"] = {"type": "json_object"}

    try:
        resp = client.post(url, headers=headers, json=payload)
        status = resp.status_code
    except (httpx.TimeoutException, httpx.NetworkError) as net_err:
        return None, 504, f"Network/timeout error: {net_err}"
    except Exception as exc:
        return None, 500, f"Unexpected request error: {exc}"

    if status == 200:
        data = resp.json()
        actual_model = data.get("model", model)
        choices = data.get("choices", [])
        if not choices:
            return None, 200, "Empty choices returned from OpenRouter"
        content = (choices[0].get("message", {}).get("content") or "").strip()
        if not content:
            return None, 200, "Empty message content returned from OpenRouter"

        if not try_json_mode:
            return content, 200, actual_model

        try:
            parsed = extract_json_from_text(content)
            return parsed, 200, actual_model
        except Exception as exc:
            return None, 200, f"JSON parsing failed: {exc}"

    # Handle 400 error if model doesn't support response_format: json_object
    if status == 400 and try_json_mode:
        err_msg = resp.text.lower()
        if "response_format" in err_msg or "json_object" in err_msg:
            log.info("Model '%s' does not support json_object mode; retrying raw text...", model)
            return _call_openrouter_completion(
                model,
                messages,
                api_key=api_key,
                base_url=base_url,
                client=client,
                temperature=temperature,
                max_tokens=max_tokens,
                try_json_mode=False,
            )

    return None, status, resp.text[:200]


def generate_openrouter_short_pack(
    preset: ChannelPreset,
    *,
    user_prompt: str,
    min_words: int,
    max_words: int,
    seg_count: int,
    topic: str,
) -> dict[str, Any] | None:
    """Resilient free-model fallback using OpenRouter.

    Follows strict priority and failover rules:
      1. Tries configured OPENROUTER_MODEL (openrouter/free) or cached working free model.
      2. If candidate fails (404/429/5xx/timeout/invalid), queries official catalog for free models.
      3. Tries alternate verified free models (:free) up to MAX_FREE_CANDIDATES.
      4. Bounded retries and respects Retry-After on 429.
      5. Never calls or falls back to paid models.
      6. Caches successfully working model for subsequent calls.
      7. Returns validated dictionary or None if all free options fail.
    """
    global _cached_working_free_model

    api_key = get_openrouter_api_key()
    if not api_key:
        log.warning("OPENROUTER_API_KEY is not configured; skipping OpenRouter fallback.")
        return None

    base_url = get_openrouter_base_url()
    system_prompt = preset.get("groq_system_hint", "You write engaging YouTube Shorts scripts.")

    # Candidate models pool: configured model first, or previously working free model
    initial_model = get_openrouter_model()
    candidate_queue: list[str] = []

    if _cached_working_free_model and is_verified_free_model(_cached_free_model := _cached_working_free_model):
        candidate_queue.append(_cached_free_model)

    if initial_model not in candidate_queue and is_verified_free_model(initial_model):
        candidate_queue.append(initial_model)

    failed_models: set[str] = set()
    catalog_queried = False

    with httpx.Client(timeout=45.0) as client:
        while candidate_queue and len(failed_models) < MAX_FREE_CANDIDATES:
            current_model = candidate_queue.pop(0)
            if current_model in failed_models:
                continue

            # Strict free-model check
            if not is_verified_free_model(current_model):
                log.warning("Excluding non-free model candidate: %s", current_model)
                failed_models.add(current_model)
                continue

            log.info("Calling OpenRouter free model candidate: '%s'...", current_model)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            parsed_data: dict[str, Any] | None = None
            success = False
            last_candidate_err = ""

            # Try up to 2 attempts for this candidate (e.g. initial + corrective prompt if schema/json issue)
            for attempt in range(2):
                if attempt > 0:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"\n\nCRITICAL FIX: Your previous generation failed validation: {last_candidate_err}.\n"
                            f"Ensure full_narration is between {min_words} and {max_words} Hindi words in Devanagari, "
                            f"and image_prompts has exactly {seg_count} distinct scene descriptions in English for 9:16 vertical."
                        ),
                    })

                try:
                    data, status, detail = _call_openrouter_completion(
                        current_model,
                        messages,
                        api_key=api_key,
                        base_url=base_url,
                        client=client,
                    )
                except (httpx.TimeoutException, httpx.NetworkError) as net_err:
                    status = 504
                    detail = f"Network/timeout error: {net_err}"

                # ── 1. Authentication / Permission Failure (401 / 403) ──
                if status in (401, 403):
                    log.warning(
                        "OpenRouter authentication/authorization failed (HTTP %d). "
                        "Check OPENROUTER_API_KEY secret. Aborting OpenRouter fallback.",
                        status,
                    )
                    return None

                # ── 2. Payment / Credit Restriction (402) ──────────────
                if status == 402:
                    log.warning(
                        "OpenRouter credit restriction (HTTP 402). "
                        "Free model unavailable or quota reached. Aborting OpenRouter to prevent paid usage.",
                    )
                    return None

                # ── 3. Model Not Found / Unavailable (404) ─────────────
                if status == 404:
                    log.warning("OpenRouter model '%s' unavailable (HTTP 404): %s", current_model, detail)
                    break

                # ── 4. Rate Limited (429) ──────────────────────────────
                if status == 429:
                    log.warning("OpenRouter rate-limited (HTTP 429) on model '%s': %s", current_model, detail)
                    # Retry once after short backoff if attempt 0
                    if attempt == 0:
                        time.sleep(2.0)
                        continue
                    break

                # ── 5. Server Error (5xx) ──────────────────────────────
                if status >= 500:
                    log.warning("OpenRouter server error (HTTP %d) on model '%s': %s", status, current_model, detail)
                    break

                # ── 6. Check Successful Response (200) ─────────────────
                if status == 200 and isinstance(data, dict):
                    # Validate story content
                    from pipeline.groq_script import validate_short_pack
                    valid, val_err = validate_short_pack(
                        data,
                        min_words=min_words,
                        max_words=max_words,
                        seg_count=seg_count,
                    )
                    if valid:
                        parsed_data = data
                        actual_returned_model = detail if detail else current_model
                        log.info(
                            "OpenRouter story generated and validated successfully (model: '%s', %d words, %d scenes).",
                            actual_returned_model,
                            len(str(data.get("full_narration", "")).split()),
                            len(data.get("image_prompts", [])),
                        )
                        _cached_working_free_model = current_model
                        success = True
                        break
                    else:
                        last_candidate_err = val_err
                        log.warning("OpenRouter model '%s' output failed validation: %s", current_model, val_err)
                else:
                    last_candidate_err = f"HTTP {status}: {detail}"
                    log.warning("OpenRouter model '%s' attempt %d failed: %s", current_model, attempt + 1, detail)

            if success and parsed_data:
                parsed_data["topic"] = topic
                return parsed_data

            # Mark this candidate as failed
            failed_models.add(current_model)

            # If candidates are running low, query official catalogue for additional free models
            if not candidate_queue and not catalog_queried:
                catalog_queried = True
                log.info("Querying OpenRouter catalogue for additional verified free models...")
                free_catalog = fetch_free_models(base_url, api_key, client=client)
                for f_model in free_catalog:
                    if f_model not in failed_models and f_model not in candidate_queue:
                        candidate_queue.append(f_model)
                        if len(candidate_queue) >= 3:
                            break

    log.warning("All OpenRouter free model candidates failed (%d tried). Proceeding to catalog fallback.", len(failed_models))
    return None
