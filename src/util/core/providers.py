"""The provider registry: one row per LLM backend this project can talk to.

Everything here used to be spelled out twice as parallel if/elif ladders in
core/agent.py -- once in `_detect_provider` (which key to read, which base URL,
which default model) and once in `_build_llm` (which provider-specific
BASE_MODEL override to consult). Seven providers x two ladders meant adding one
backend touched fourteen branches, and the two ladders could disagree without
anything noticing.

Adding a provider is now one ProviderSpec.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

StructuredOutputMethod = Literal["function_calling", "json_mode", "json_schema"]


@dataclass(frozen=True)
class ProviderSpec:
    """How to reach one provider, and how to get structured output from it."""

    name: str
    key_env: str
    default_model: str
    model_env: str
    method: StructuredOutputMethod
    # None means "the OpenAI client's own default" -- only true for OpenAI.
    base_url: Optional[str] = None
    # Only vllm sets this: its base URL is not a constant, it is whatever host
    # is currently serving, so it comes from the environment per rental.
    base_url_env: Optional[str] = None
    # vLLM is unauthenticated by default but the OpenAI client demands a
    # non-empty key, so a placeholder stands in for a missing one.
    key_default: Optional[str] = None
    extra_headers_env: Dict[str, Tuple[str, str]] = field(default_factory=dict)

    def resolve_base_url(self) -> Optional[str]:
        if self.base_url_env:
            return os.getenv(self.base_url_env, self.base_url)
        return self.base_url

    def resolve_key(self) -> str:
        return os.getenv(self.key_env, "") or (self.key_default or "")

    def resolve_model(self, explicit: Optional[str], env_default: str) -> str:
        """Model precedence: explicit argument, then the generic BASE_MODEL
        override (what a --model CLI flag sets), then this provider's own
        <PROVIDER>_BASE_MODEL, then the provider default."""
        return (
            explicit
            or os.getenv("BASE_MODEL")
            or os.getenv(self.model_env)
            or env_default
        )

    def headers(self) -> Dict[str, str]:
        return {
            header: os.getenv(env_var, fallback)
            for header, (env_var, fallback) in self.extra_headers_env.items()
        }


# Registration order IS the auto-detection precedence used when several keys
# are present and PROVIDER is unset. Preserved exactly from the original
# ladder: groq, cerebras, deepseek, gemini, openai, openrouter.
PROVIDERS: Dict[str, ProviderSpec] = {
    "groq": ProviderSpec(
        name="groq",
        key_env="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        model_env="GROQ_BASE_MODEL",
        # MEASURED, 2026-07-30. Under function_calling, llama-3.3-70b returns
        # "400 Failed to call a function" for this project's real output schemas
        # -- an unconditional rejection, so groq could not run the pipeline at
        # all. json_mode clears that. It does not make groq unconditionally
        # capable: a 121-fact extraction then fails with LengthFinishReasonError
        # because the model's output limit binds (~8k tokens; a 30-fact chunk
        # completes at 8,119). But a size limit is something chunking already
        # solves, whereas the 400 was a wall.
        #
        # Unlike Gemini and DeepSeek above, this is NOT a statement about groq's
        # API -- groq offers function calling, and cerebras uses that path
        # successfully on the same schemas. It is the MODEL that cannot emit a
        # valid call for a large recursive schema, and method is per-provider
        # rather than per-model, so this is the granularity available.
        method="json_mode",
    ),
    "cerebras": ProviderSpec(
        name="cerebras",
        key_env="CEREBRAS_API_KEY",
        base_url="https://api.cerebras.ai/v1",
        default_model="gpt-oss-120b",
        model_env="CEREBRAS_BASE_MODEL",
        method="function_calling",
    ),
    "deepseek": ProviderSpec(
        name="deepseek",
        key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        default_model="deepseek-v4-flash",
        model_env="DEEPSEEK_BASE_MODEL",
        # No server-side schema enforcement, only "must be valid JSON".
        method="json_mode",
    ),
    "gemini": ProviderSpec(
        name="gemini",
        key_env="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        default_model="gemini-2.5-flash",
        model_env="GEMINI_BASE_MODEL",
        method="json_mode",
    ),
    "openai": ProviderSpec(
        name="openai",
        key_env="OPENAI_API_KEY",
        base_url=None,
        default_model="gpt-4o",
        model_env="OPENAI_BASE_MODEL",
        method="function_calling",
    ),
    "openrouter": ProviderSpec(
        name="openrouter",
        key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        default_model="openai/gpt-4o",
        model_env="OPENROUTER_BASE_MODEL",
        method="function_calling",
        extra_headers_env={
            "HTTP-Referer": ("OPENROUTER_REFERER", "https://github.com/scribbledb"),
            "X-Title": ("OPENROUTER_TITLE", "ScribbleDB"),
        },
    ),
    "vllm": ProviderSpec(
        name="vllm",
        key_env="VLLM_API_KEY",
        key_default="EMPTY",
        base_url="http://localhost:8000/v1",
        base_url_env="VLLM_BASE_URL",
        default_model="local-model",
        model_env="VLLM_BASE_MODEL",
        # The only provider that can ENFORCE the schema during sampling: vLLM
        # compiles the JSON Schema into a grammar and masks non-conforming
        # tokens, making a violation unrepresentable rather than merely
        # unlikely. Uses the response_format path, NOT tool_choice, so the
        # server does not need --tool-call-parser.
        method="json_schema",
    ),
}

# PROVIDER=local is accepted as a synonym for the self-hosted endpoint.
PROVIDER_ALIASES: Dict[str, str] = {"local": "vllm"}

# vllm is deliberately absent: VLLM_API_KEY is optional (the endpoint is
# usually unauthenticated), so its presence is not an auto-detection signal.
# Selecting it always requires PROVIDER=vllm explicitly.
_AUTODETECT_ORDER = ("groq", "cerebras", "deepseek", "gemini", "openai", "openrouter")


def resolve_provider() -> ProviderSpec:
    """Pick a provider: an explicit PROVIDER override wins, otherwise the
    first registered provider whose key is present. Raises if none is."""
    override = os.getenv("PROVIDER", "").strip().lower()
    if override:
        canonical = PROVIDER_ALIASES.get(override, override)
        spec = PROVIDERS.get(canonical)
        if spec is not None:
            return spec

    for name in _AUTODETECT_ORDER:
        spec = PROVIDERS[name]
        if os.getenv(spec.key_env):
            return spec

    raise RuntimeError(
        "No LLM API key found. Set one of: "
        + ", ".join(PROVIDERS[n].key_env for n in _AUTODETECT_ORDER)
        + " in .env (or PROVIDER=vllm with VLLM_BASE_URL for a self-hosted endpoint)."
    )
