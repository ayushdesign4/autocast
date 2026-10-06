"""LLM provider cascade: gemini -> agnes -> groq -> cloudflare -> cerebras -> keyless.

Keyed providers are tried in order of capability, stability, and quota:
1. Gemini: Google's flagship multimodal LLM (gemini-3.8-flash via native google-genai
   SDK with automatic fallback to OpenAI-compatible endpoint).
2. Agnes: High-speed companion LLM hosted at https://apihub.agnes-ai.com/v1/chat/completions
   using the already-configured AGNES_API_KEY (model: agnes-3.0-flash).
3. Groq: Ultra-fast open models (llama-3.3-70b-versatile).
4. Cloudflare: Workers AI (gated behind cloudflare_budget_cents > 0).
5. Cerebras: llama-3.3-70b (topic & script only; 8k context cap).
6. Pollinations: Keyless last-resort fallback.
7. Built-in template fallback: Deterministic safe harbor in try_llm.
"""

from __future__ import annotations

import logging

from autocast.config import Config
from autocast.providers.cascade import AllProvidersFailed, Provider, run_with_fallback
from autocast.util.net import post_json

log = logging.getLogger("autocast.providers.llm")

# ---- Gemini (primary keyed LLM) ----
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
_GEMINI_MODELS = ("gemini-3.8-flash", "gemini-2.5-flash")
_DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"

# ---- Agnes (companion keyed LLM) ----
_DEFAULT_AGNES_BASE = "https://apihub.agnes-ai.com"
_AGNES_CHAT_MODELS = ("agnes-3.0-flash", "agnes-2.5-flash", "agnes-2.0-flash")
_DEFAULT_AGNES_CHAT_MODEL = "agnes-3.0-flash"

# ---- Other keyed OpenAI-compatible endpoints ----
_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL = "llama-3.3-70b-versatile"
_CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
_CEREBRAS_MODEL = "llama-3.3-70b"
_CLOUDFLARE_URL = (
    "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions"
)
_CLOUDFLARE_MODEL = "@cf/meta/llama-3.1-8b-instruct"

# ---- Keyless fallback (now paywalled; kept as a fast-failing last resort) ----
_POLLINATIONS_OPENAI = "https://text.pollinations.ai/openai"
_POLLINATIONS_MODEL = "openai"
_REFERRER = "autocast"


def _dry_stub(prompt: str, kind: str) -> str:
    """Deterministic placeholder so dry-run downstream stages have real inputs."""
    return f"[DRYRUN {kind}] {prompt[:60]}"


def _call_openai_compat(
    base_url: str,
    model: str,
    prompt: str,
    *,
    api_key: str | None = None,
    temperature: float = 0.7,
    extra_body: dict | None = None,
) -> str:
    """One OpenAI-compatible chat call. Every keyed LLM provider is this shape with
    a different base_url/model/key — so the cascade is just bindings of this fn."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if extra_body:
        body.update(extra_body)
    resp = post_json(
        base_url,
        body,
        headers=headers,
        timeout=25.0,
        retries=1,
        backoff_s=0.0,
    )
    content = resp["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM returned empty content")
    return content.strip()


def _call_gemini(
    api_key: str,
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.7,
) -> str:
    """Execute Gemini chat completion via native google-genai SDK, falling back to HTTP."""
    target_model = model or _DEFAULT_GEMINI_MODEL

    # Attempt 1: Native google-genai SDK (direct, robust, supports latest models)
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(temperature=temperature)
        resp = client.models.generate_content(
            model=target_model,
            contents=prompt,
            config=config,
        )
        if resp.text and resp.text.strip():
            return resp.text.strip()
    except Exception as exc:
        log.warning("llm[gemini-sdk]: SDK call failed (%s); trying HTTP fallback", exc)

    # Attempt 2: OpenAI-compatible HTTP endpoint
    models_to_try = [target_model] + [m for m in _GEMINI_MODELS if m != target_model]
    last_err: Exception | None = None
    for m in models_to_try:
        try:
            return _call_openai_compat(
                _GEMINI_URL,
                m,
                prompt,
                api_key=api_key,
                temperature=temperature,
            )
        except Exception as exc:
            last_err = exc
            log.warning("llm[gemini-http]: model %s failed: %s", m, exc)

    raise RuntimeError(f"All Gemini generation attempts failed: {last_err}")


def _call_agnes(
    api_key: str,
    base_url: str,
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.7,
) -> str:
    """Execute chat completion via Agnes OpenAI-compatible endpoint."""
    chat_url = f"{base_url.rstrip('/')}/v1/chat/completions"
    target_model = model or _DEFAULT_AGNES_CHAT_MODEL
    models_to_try = [target_model] + [m for m in _AGNES_CHAT_MODELS if m != target_model]
    last_err: Exception | None = None
    for m in models_to_try:
        try:
            return _call_openai_compat(
                chat_url,
                m,
                prompt,
                api_key=api_key,
                temperature=temperature,
            )
        except Exception as exc:
            last_err = exc
            log.warning("llm[agnes]: model %s failed: %s", m, exc)

    raise RuntimeError(f"All Agnes chat models failed: {last_err}")


def _keyed_provider(
    name: str, base_url: str, model: str, api_key: str, prompt: str
) -> Provider[str]:
    """Bind one keyed OpenAI-compatible provider for the cascade."""
    return Provider(name, lambda: _call_openai_compat(base_url, model, prompt, api_key=api_key))


def _pollinations_openai(prompt: str, kind: str) -> str:
    log.info("llm[%s]: calling pollinations-openai (keyless, POST)", kind)
    return _call_openai_compat(
        _POLLINATIONS_OPENAI, _POLLINATIONS_MODEL, prompt, extra_body={"referrer": _REFERRER}
    )


def build_llm_providers(
    cfg: Config,
    *,
    prompt: str,
    kind: str,
    allow_cerebras: bool = True,
    dry_run: bool = False,
) -> list[Provider[str]]:
    """Return the ordered LLM providers for the cascade.

    `kind` is a label ("topic-rank" | "script" | "direction") used for logging
    and the dry stub. `allow_cerebras=False` for the direction stage (8k context cap).

    Keyed providers (Gemini -> Agnes -> Groq -> Cloudflare -> Cerebras) go FIRST
    when their key is present. Keyless Pollinations is the final attempt before
    the deterministic template safe-harbor in try_llm.
    """
    if dry_run:
        return [Provider("dry-stub", lambda: _dry_stub(prompt, kind))]

    providers: list[Provider[str]] = []

    # 1. Gemini (primary keyed LLM)
    if cfg.gemini_api_key:
        gemini_model = getattr(cfg, "gemini_model", _DEFAULT_GEMINI_MODEL) or _DEFAULT_GEMINI_MODEL
        providers.append(
            Provider(
                "gemini",
                lambda: _call_gemini(cfg.gemini_api_key, prompt, model=gemini_model),
            )
        )

    # 2. Agnes (companion keyed LLM, authenticated via AGNES_API_KEY)
    if cfg.agnes_api_key:
        agnes_base = getattr(cfg, "agnes_api_base", _DEFAULT_AGNES_BASE) or _DEFAULT_AGNES_BASE
        providers.append(
            Provider(
                "agnes",
                lambda: _call_agnes(cfg.agnes_api_key, agnes_base, prompt),
            )
        )

    # 3. Groq (fast keyed fallback)
    if cfg.groq_api_key:
        providers.append(
            _keyed_provider("groq", _GROQ_URL, _GROQ_MODEL, cfg.groq_api_key, prompt)
        )

    # 4. Cloudflare Workers AI (budget-gated)
    if (
        cfg.cloudflare_api_token
        and cfg.cloudflare_account_id
        and cfg.cloudflare_budget_cents > 0
    ):
        providers.append(
            _keyed_provider(
                "cloudflare",
                _CLOUDFLARE_URL.format(account_id=cfg.cloudflare_account_id),
                _CLOUDFLARE_MODEL,
                cfg.cloudflare_api_token,
                prompt,
            )
        )

    # 5. Cerebras (8k context cap -> topic/script only)
    if allow_cerebras and cfg.cerebras_api_key:
        providers.append(
            _keyed_provider(
                "cerebras", _CEREBRAS_URL, _CEREBRAS_MODEL, cfg.cerebras_api_key, prompt
            )
        )

    # 6. Keyless last-resort fallback before template
    providers.append(Provider("pollinations-openai", lambda: _pollinations_openai(prompt, kind)))
    return providers


# Sentinel provider name recorded when every LLM failed and the stage used its
# built-in deterministic template instead of dying.
TEMPLATE_FALLBACK = "template-fallback"


def try_llm(providers: list[Provider[str]]) -> tuple[str | None, str]:
    """Run the LLM cascade, degrading gracefully.

    Returns (text, provider_used) on success, or (None, TEMPLATE_FALLBACK) if the
    whole cascade failed — so an LLM-using stage NEVER kills the run over a flaky
    endpoint; it falls back to its template and keeps the pipeline moving.
    """
    try:
        result = run_with_fallback(providers)
        return result.value, result.provider_used
    except AllProvidersFailed as exc:
        log.warning("llm: all providers failed (%s); caller falls back to template", exc)
        return None, TEMPLATE_FALLBACK
