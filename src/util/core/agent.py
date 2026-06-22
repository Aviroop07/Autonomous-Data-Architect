import os
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

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
_GEMINI_DEFAULT_MODEL = "gemini-3.1-flash-lite"
_OPENAI_DEFAULT_MODEL = "gpt-4o"


# ------------------------------------------------------------------
# Provider detection
# ------------------------------------------------------------------


def _detect_provider() -> tuple[str, str, str | None, str]:
    """Returns (provider, api_key, base_url_or_None, default_model).

    Selection rules:
      - Only GEMINI_API_KEY set        -> gemini
      - Only OPENAI_API_KEY set        -> openai
      - Both set, PROVIDER=openai      -> openai
      - Both set, anything else        -> gemini (free tier preferred)
      - Neither set                    -> RuntimeError
    """
    load_dotenv()
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    provider_override = os.getenv("PROVIDER", "").lower()

    if gemini_key and openai_key:
        provider = "openai" if provider_override == "openai" else "gemini"
    elif gemini_key:
        provider = "gemini"
    elif openai_key:
        provider = "openai"
    else:
        raise RuntimeError(
            "No LLM API key found. Set GEMINI_API_KEY or OPENAI_API_KEY in .env."
        )

    if provider == "gemini":
        return "gemini", gemini_key, _GEMINI_BASE_URL, _GEMINI_DEFAULT_MODEL
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
    if provider == "gemini":
        resolved = (
            model
            or os.getenv("BASE_MODEL")
            or os.getenv("GEMINI_BASE_MODEL")
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
        messages = [SystemMessage(content=self.system_prompt)]
        messages.extend(input_dict.get("messages", []))
        result = await self.chain.ainvoke(messages)
        # result = {"raw": AIMessage, "parsed": PydanticModel | None, "parsing_error": ...}
        if result.get("parsing_error"):
            raise ValueError(
                f"Structured output parsing failed: {result['parsing_error']}"
            )
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
    tools: Optional[List[Any]] = None,
    output_structure: Optional[Type[T]] = None,
    model: Optional[str] = None,
    name: Optional[str] = None,
    use_responses_api: bool = False,
) -> AgentType:
    """Create an agent. Returns an object supporting ainvoke({"messages": [...]}).

    Routing:
      - No tools  -> StructuredAgent (lightweight with_structured_output wrapper)
                     Uses json_mode for Gemini, function_calling for OpenAI.
      - With tools -> langgraph react agent via create_agent().
                     web_search_preview dict is swapped for ddg_search on
                     non-OpenAI providers (Gemini doesn't support that tool spec).

    The OUTPUT FORMAT section is dynamically appended to the system prompt from
    the Pydantic schema whenever output_structure is provided.
    """
    # Single provider detection — result shared by LLM build, tool swap, method selection.
    provider, api_key, base_url, env_default = _detect_provider()
    llm = _build_llm(provider, api_key, base_url, env_default, model, use_responses_api)

    if output_structure:
        output_format = generate_hierarchical_schema_description(output_structure)
        system_prompt = (
            f"{system_prompt}\n\n"
            f"## OUTPUT FORMAT\n"
            f"Return a JSON object matching this structure:\n{output_format}"
        )

    if tools:
        if provider != "openai":
            from src.util.core.search_tool import get_ddg_search_tool

            ddg = get_ddg_search_tool()
            swapped = [
                ddg
                if (isinstance(t, dict) and t.get("type") == "web_search_preview")
                else t
                for t in tools
            ]
            if swapped != tools:
                print(
                    f"[agent] {name or 'agent'}: web_search_preview -> ddg_search "
                    f"(provider: {provider})"
                )
            tools = swapped

        from langchain.agents import create_agent  # type: ignore[import]

        return create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            response_format=output_structure,
            name=name or "agent",
        )

    assert output_structure is not None, (
        "output_structure is required when not using tools"
    )
    method = "json_mode" if provider == "gemini" else "function_calling"
    return StructuredAgent(
        system_prompt=system_prompt,
        llm=llm,
        output_structure=output_structure,
        name=name or "structured_agent",
        method=method,
    )
