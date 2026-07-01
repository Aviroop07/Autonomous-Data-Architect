import os
from typing import Any, Dict, Optional, Type, TypeVar, Union

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage  # type: ignore[import]
from langchain_core.runnables import Runnable  # type: ignore[import]
from langchain_openai import ChatOpenAI  # type: ignore[import]
from pydantic import BaseModel, SecretStr

from src.util.schema_ops.schema_utils import generate_hierarchical_schema_description

T = TypeVar("T", bound=BaseModel)

# Public alias used by all agent modules for their get_agent() return type
AgentType = Union["StructuredAgent", Runnable]

# ------------------------------------------------------------------
# Provider constants
# ------------------------------------------------------------------

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"
_OPENAI_DEFAULT_MODEL = "gpt-4o"
_OPENROUTER_DEFAULT_MODEL = "openai/gpt-4o"
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
_CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
_CEREBRAS_DEFAULT_MODEL = "gpt-oss-120b"


# ------------------------------------------------------------------
# Provider detection
# ------------------------------------------------------------------


def _detect_provider() -> tuple[str, str, str | None, str]:
    """Returns (provider, api_key, base_url_or_None, default_model).

    Selection rules (explicit PROVIDER override wins; otherwise key presence):
      - PROVIDER=openrouter, or only OPENROUTER_API_KEY set  -> openrouter
      - PROVIDER=openai, or only OPENAI_API_KEY set          -> openai
      - PROVIDER=gemini, or only GEMINI_API_KEY set          -> gemini
      - Multiple keys, no PROVIDER override                  -> openai > gemini > openrouter
      - No keys                                              -> RuntimeError
    """
    load_dotenv()
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    cerebras_key = os.getenv("CEREBRAS_API_KEY", "")
    provider_override = os.getenv("PROVIDER", "").lower()

    if provider_override == "cerebras":
        provider = "cerebras"
    elif provider_override == "groq":
        provider = "groq"
    elif provider_override == "openrouter":
        provider = "openrouter"
    elif provider_override == "openai":
        provider = "openai"
    elif provider_override == "gemini":
        provider = "gemini"
    elif groq_key:
        provider = "groq"
    elif cerebras_key:
        provider = "cerebras"
    elif gemini_key:
        provider = "gemini"
    elif openai_key:
        provider = "openai"
    elif openrouter_key:
        provider = "openrouter"
    else:
        raise RuntimeError(
            "No LLM API key found. Set CEREBRAS_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY in .env."
        )

    if provider == "cerebras":
        return "cerebras", cerebras_key, _CEREBRAS_BASE_URL, _CEREBRAS_DEFAULT_MODEL
    if provider == "groq":
        return "groq", groq_key, _GROQ_BASE_URL, _GROQ_DEFAULT_MODEL
    if provider == "gemini":
        return "gemini", gemini_key, _GEMINI_BASE_URL, _GEMINI_DEFAULT_MODEL
    if provider == "openrouter":
        return (
            "openrouter",
            openrouter_key,
            _OPENROUTER_BASE_URL,
            _OPENROUTER_DEFAULT_MODEL,
        )
    return "openai", openai_key, None, _OPENAI_DEFAULT_MODEL


# ------------------------------------------------------------------
# LLM builder (internal) — takes pre-detected provider info
# ------------------------------------------------------------------


def _build_llm(
    provider: str,
    api_key: str,
    base_url: str | None,
    env_default: str,
    model: Optional[str],
    use_responses_api: bool,
) -> ChatOpenAI:
    """Build a ChatOpenAI instance from pre-detected provider info.

    Model resolution order (highest to lowest priority):
      1. explicit model param
      2. BASE_MODEL env var (generic override, e.g. from --model CLI flag)
      3. GEMINI_BASE_MODEL / OPENAI_BASE_MODEL (provider-specific .env setting)
      4. provider default constant

    use_responses_api is silently ignored for Gemini (only applies to OpenAI
    Responses API). This is intentional — callers don't need to branch on
    provider when building tool-using agents.
    """
    if provider == "cerebras":
        resolved = (
            model
            or os.getenv("BASE_MODEL")
            or os.getenv("CEREBRAS_BASE_MODEL")
            or env_default
        )
    elif provider == "groq":
        resolved = (
            model
            or os.getenv("BASE_MODEL")
            or os.getenv("GROQ_BASE_MODEL")
            or env_default
        )
    elif provider == "gemini":
        resolved = (
            model
            or os.getenv("BASE_MODEL")
            or os.getenv("GEMINI_BASE_MODEL")
            or env_default
        )
    elif provider == "openrouter":
        resolved = (
            model
            or os.getenv("BASE_MODEL")
            or os.getenv("OPENROUTER_BASE_MODEL")
            or env_default
        )
    else:
        resolved = (
            model
            or os.getenv("BASE_MODEL")
            or os.getenv("OPENAI_BASE_MODEL")
            or env_default
        )

    kwargs: Dict[str, Any] = dict(api_key=SecretStr(api_key), model=resolved)
    if base_url is not None:
        kwargs["base_url"] = base_url
    if provider == "openrouter":
        kwargs["default_headers"] = {
            "HTTP-Referer": os.getenv(
                "OPENROUTER_REFERER", "https://github.com/scribbledb"
            ),
            "X-Title": os.getenv("OPENROUTER_TITLE", "ScribbleDB"),
        }
    if use_responses_api and provider == "openai":
        kwargs["use_responses_api"] = True
    return ChatOpenAI(**kwargs)


# ------------------------------------------------------------------
# Public model factory
# ------------------------------------------------------------------


def get_model(
    model: Optional[str] = None, use_responses_api: bool = False
) -> ChatOpenAI:
    """Return a ChatOpenAI instance configured for the detected provider.

    Thin public wrapper around _detect_provider + _build_llm. Use this when
    you only need the raw LLM; use get_agent_() to get a full agent.
    """
    provider, api_key, base_url, env_default = _detect_provider()
    return _build_llm(
        provider, api_key, base_url, env_default, model, use_responses_api
    )


# ------------------------------------------------------------------
# Structured-output agent wrapper
# ------------------------------------------------------------------


class StructuredAgent:
    """Wraps ChatOpenAI.with_structured_output() with the same
    ainvoke({"messages": [...]}) interface as langgraph agents.

    Response format:
        {
            "structured_response": <PydanticModel>,
            "messages": [SystemMessage, HumanMessage, AIMessage],
        }
    """

    def __init__(
        self,
        system_prompt: str,
        llm: ChatOpenAI,
        output_structure: Type[T],
        name: str = "structured_agent",
        method: str = "function_calling",
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.output_structure = output_structure
        self.chain = llm.with_structured_output(
            output_structure, include_raw=True, method=method
        )

    async def ainvoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        import asyncio as _asyncio

        messages = [SystemMessage(content=self.system_prompt)]
        messages.extend(input_dict.get("messages", []))

        _PARSE_RETRIES = 3
        last_error: Optional[str] = None
        result: Dict[str, Any] = {}
        for attempt in range(_PARSE_RETRIES):
            result = await self.chain.ainvoke(messages)
            # result = {"raw": AIMessage, "parsed": PydanticModel | None, "parsing_error": ...}
            last_error = result.get("parsing_error")
            if not last_error and result.get("parsed") is not None:
                break
            if attempt < _PARSE_RETRIES - 1:
                wait = 2.0 * (attempt + 1)
                print(
                    f"[agent] Structured output parse failure "
                    f"(attempt {attempt + 1}/{_PARSE_RETRIES}), retrying in {wait:.0f}s..."
                )
                await _asyncio.sleep(wait)

        if last_error:
            raise ValueError(f"Structured output parsing failed: {last_error}")
        if result.get("parsed") is None:
            raise ValueError(
                f"Structured output returned None for {self.output_structure.__name__} "
                f"(model produced empty or unparseable output)"
            )
        return {
            "structured_response": result["parsed"],
            "messages": messages + [result["raw"]],
        }


# ------------------------------------------------------------------
# Agent factory
# ------------------------------------------------------------------


def get_agent_(
    system_prompt: str,
    output_structure: Type[T],
    model: Optional[str] = None,
    name: Optional[str] = None,
    use_responses_api: bool = False,
) -> "StructuredAgent":
    """Create a structured-output agent. Returns a StructuredAgent.

    The OUTPUT FORMAT section is dynamically appended to the system prompt from
    the Pydantic schema. Uses json_mode for Gemini, function_calling for all others.

    Web search is handled via EvidenceStore pre-fetching before
    the agent call, not via tool-calling. See src/util/core/search_tool.py.
    """
    provider, api_key, base_url, env_default = _detect_provider()
    llm = _build_llm(provider, api_key, base_url, env_default, model, use_responses_api)

    output_format = generate_hierarchical_schema_description(output_structure)
    full_prompt = (
        f"{system_prompt}\n\n"
        f"## OUTPUT FORMAT\n"
        f"Return a JSON object matching this structure:\n{output_format}"
    )

    method = "json_mode" if provider == "gemini" else "function_calling"
    return StructuredAgent(
        system_prompt=full_prompt,
        llm=llm,
        output_structure=output_structure,
        name=name or "structured_agent",
        method=method,
    )
