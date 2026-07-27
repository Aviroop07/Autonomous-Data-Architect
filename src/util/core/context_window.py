"""Resolves a provider/model's real context window size via a live API
call -- never a maintained constant table, which goes stale the moment a
provider ships a new model or changes an existing one's limit.

Verified empirically (experiments/context_window_query.py,
context_window_resolve.py) against all 6 providers this project supports:
OpenRouter, Gemini, and Groq expose context length directly via their own
free API. DeepSeek, OpenAI, and Cerebras do not (their /models endpoints
return only id/object/owned_by) -- for those, OpenRouter's own catalog is
public and unauthenticated, and it lists equivalent entries for other
vendors' models too (confirmed: 'deepseek-v4-flash' resolves via
'deepseek/deepseek-v4-flash', Cerebras' 'gpt-oss-120b' via
'openai/gpt-oss-120b') -- every one of this project's 6 default models
resolved to an exact, unambiguous match in that experiment.

Self-hosted vLLM endpoints are handled separately: they report their own
max_model_len via /v1/models, which beats any catalog lookup because
--max-model-len can cap the served window below the model's native limit.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_GEMINI_MODEL_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}"
_GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"

# In-memory caches -- one process-lifetime HTTP call per provider's
# catalog/model, not one per agent construction.
_openrouter_catalog_cache: Optional[List[Dict]] = None
_resolved_cache: Dict[Tuple[str, str], int] = {}


class ContextWindowUnresolvedError(RuntimeError):
    """Raised when a provider/model's context window cannot be determined
    from any live source. Never silently guessed or defaulted."""


def _get_openrouter_catalog() -> List[Dict]:
    global _openrouter_catalog_cache
    catalog = _openrouter_catalog_cache
    if catalog is None:
        resp = requests.get(_OPENROUTER_MODELS_URL, timeout=15.0)
        resp.raise_for_status()
        catalog = resp.json()["data"]
        _openrouter_catalog_cache = catalog
    return catalog


def _fallback_via_openrouter_catalog(model: str) -> Optional[int]:
    """Searches OpenRouter's public catalog for a model whose id's last
    path segment (vendor prefix stripped, ':free'-style suffix stripped)
    exactly matches `model`. Returns None (not a guess) on anything less
    than an unambiguous exact match."""
    catalog = _get_openrouter_catalog()
    target = model.lower()
    exact_matches = [
        m for m in catalog if m["id"].lower().split("/")[-1].split(":")[0] == target
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]["context_length"]
    if len(exact_matches) > 1:
        logger.warning(
            f"[context_window] Ambiguous OpenRouter catalog matches for "
            f"'{model}': {[m['id'] for m in exact_matches]} -- refusing to guess."
        )
    return None


def get_context_window(provider: str, model: str, *, api_key: str = "") -> int:
    """Resolves `model`'s context window (in tokens) for `provider`, live.

    Raises ContextWindowUnresolvedError if it cannot be determined from
    any live source -- callers must not fall back to a guessed constant.
    """
    cache_key = (provider, model)
    if cache_key in _resolved_cache:
        return _resolved_cache[cache_key]

    window: Optional[int] = None

    if provider == "openrouter":
        catalog = _get_openrouter_catalog()
        match = next((m for m in catalog if m["id"] == model), None)
        window = match["context_length"] if match else None

    elif provider == "gemini":
        if not api_key:
            raise ContextWindowUnresolvedError(
                "gemini requires api_key to query its models endpoint."
            )
        resp = requests.get(
            _GEMINI_MODEL_URL.format(model=model),
            params={"key": api_key},
            timeout=15.0,
        )
        if resp.status_code == 200:
            window = resp.json().get("inputTokenLimit")

    elif provider == "groq":
        if not api_key:
            raise ContextWindowUnresolvedError(
                "groq requires api_key to query its models endpoint."
            )
        resp = requests.get(
            _GROQ_MODELS_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15.0,
        )
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            match = next((m for m in models if m["id"] == model), None)
            if match:
                window = match.get("context_window") or match.get("context_length")

    elif provider == "vllm":
        # A self-hosted endpoint is its own authority here: vLLM's /v1/models
        # reports the max_model_len the server was actually launched with,
        # which is the real limit -- more accurate than any catalog entry for
        # the base model, since --max-model-len can cap it well below what the
        # weights support. Self-hosted models are also frequently absent from
        # OpenRouter's catalog entirely, so the usual fallback cannot help.
        base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
        resp = requests.get(f"{base_url}/models", timeout=15.0)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            match = next((m for m in models if m["id"] == model), None)
            if match:
                window = match.get("max_model_len") or match.get("context_length")

    else:
        # deepseek, openai, cerebras -- no context-length metadata from
        # their own APIs; fall back to OpenRouter's free public catalog.
        window = _fallback_via_openrouter_catalog(model)

    if window is None:
        raise ContextWindowUnresolvedError(
            f"Could not resolve context window for provider={provider!r}, "
            f"model={model!r} from any live source (direct API or "
            f"OpenRouter's public catalog fallback)."
        )

    _resolved_cache[cache_key] = window
    return window
