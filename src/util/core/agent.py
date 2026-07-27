import logging
from operator import itemgetter
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union

from dotenv import load_dotenv
from langchain_core.messages import (  # type: ignore[import]
    HumanMessage,
    SystemMessage,
)
from langchain_core.output_parsers.openai_tools import (  # type: ignore[import]
    PydanticToolsParser,
)
from langchain_core.runnables import (  # type: ignore[import]
    Runnable,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI  # type: ignore[import]
from pydantic import BaseModel, SecretStr

from src.util.core.providers import PROVIDERS, resolve_provider
from src.util.schema_ops.schema_utils import generate_hierarchical_schema_description

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Public alias used by all agent modules for their get_agent() return type
AgentType = Union["StructuredAgent", Runnable]

# ------------------------------------------------------------------
# Provider detection
# ------------------------------------------------------------------
#
# The per-provider constants, the detection ladder and the model-override
# ladder all live in core/providers.py now -- see that module for why.


def _detect_provider() -> tuple[str, str, str | None, str]:
    """Returns (provider, api_key, base_url_or_None, default_model).

    An explicit PROVIDER env var wins (with `local` as an alias for `vllm`);
    otherwise the first provider with a key present, in the registration order
    of providers.PROVIDERS. Raises RuntimeError when nothing is configured.

    The 4-tuple shape is kept because every caller and several tests destructure
    it; providers.resolve_provider() returns the richer ProviderSpec.

    The 4th element is the ENV-RESOLVED model, not the bare provider default.
    That matters for callers that use it directly rather than passing it back
    through _build_llm -- notably the Stage 3 auto-sharder, which sizes shards
    against the target model's real context window. Returning the bare default
    made it size against gemini-2.5-flash while the run actually used
    gemini-3.1-flash-lite (observed live), and for vllm it would have returned
    the "local-model" placeholder instead of the served model name.
    """
    load_dotenv()
    spec = resolve_provider()
    return (
        spec.name,
        spec.resolve_key(),
        spec.resolve_base_url(),
        spec.resolve_model(None, spec.default_model),
    )


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
      3. the provider's own <PROVIDER>_BASE_MODEL (.env setting)
      4. provider default

    use_responses_api applies only to the OpenAI Responses API and is silently
    ignored elsewhere -- intentional, so callers don't branch on provider when
    building tool-using agents.
    """
    spec = PROVIDERS.get(provider, PROVIDERS["openai"])
    resolved = spec.resolve_model(model, env_default)

    kwargs: Dict[str, Any] = dict(api_key=SecretStr(api_key), model=resolved)
    if base_url is not None:
        kwargs["base_url"] = base_url
    headers = spec.headers()
    if headers:
        kwargs["default_headers"] = headers
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
# Ref-preserving tool schema (function_calling bypass)
# ------------------------------------------------------------------


def _build_ref_preserving_tool(output_structure: Type[BaseModel]) -> Dict[str, Any]:
    """Hand-builds an OpenAI tool definition from Pydantic's own
    model_json_schema() -- which already uses $ref/$defs for repeated and
    recursive models -- instead of going through LangChain's
    with_structured_output(method="function_calling"), which internally
    calls convert_to_openai_tool() -> dereference_refs(). That dereference
    step only breaks true cycles; it does NOT memoize repeated non-cyclic
    $ref occurrences, so a schema with a recursive type reused across many
    sibling fields (e.g. this project's RExprUnion/RPredicate, reused
    throughout UnifiedExtractionOutput's constraint types) gets re-expanded
    in full at every occurrence -- measured directly: 33,585 chars via
    model_json_schema() vs 21,393,132 chars post-dereference for
    UnifiedExtractionOutput, a 637x blowup that exceeds every real
    provider's context window.

    convert_to_openai_tool() passes an already-dict tool whose "type" is
    "function" straight through UNCHANGED (see langchain_core.utils.
    function_calling._WellKnownOpenAITools) -- so handing bind_tools() a
    pre-built dict here, rather than the raw Pydantic class, bypasses the
    dereferencing entirely. Verified live against a real OpenAI-compatible
    endpoint (OpenRouter) that $defs/$ref inside a tool's `parameters` is
    accepted at the wire level -- this isn't just locally inert, the API
    itself supports it."""
    schema = output_structure.model_json_schema()
    name = schema.pop("title", output_structure.__name__)
    description = schema.pop("description", "") or f"Extract data as {name}."
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": schema},
    }


def _build_ref_preserving_response_format(
    output_structure: Type[BaseModel],
) -> Dict[str, Any]:
    """The json_schema-method counterpart to _build_ref_preserving_tool.

    Same hazard, same escape hatch. LangChain's
    with_structured_output(method="json_schema") calls
    _convert_to_openai_response_format(), which for a Pydantic class routes
    through convert_to_openai_function() -> dereference_refs() -- the exact
    637x blowup documented above. But that function has a pass-through
    branch: a plain dict already carrying both "name" and "schema" keys is
    wrapped as {"type": "json_schema", "json_schema": <dict>} and otherwise
    left UNTOUCHED. Handing it a pre-built dict from model_json_schema()
    therefore keeps $ref/$defs intact.

    Why this matters beyond schema size: unlike json_mode (where the prose
    OUTPUT FORMAT block is the model's only guidance and nothing enforces
    it), json_schema makes the server compile the schema into a grammar and
    mask invalid tokens during sampling. Violations become unrepresentable
    rather than merely unlikely -- measured need: a live Gemma-4-26B run
    emitted category="uniqueness"/"fanout" (sub-type names) where the enum
    only permits statistical|structural|logic|temporal|derived (family
    names), which no amount of retrying fixed because the retries were
    blind. Under grammar constraint those tokens cannot be sampled at all.

    "strict" is deliberately omitted: OpenAI's strict mode additionally
    demands additionalProperties:false on every object and every property
    listed in "required", which Pydantic's own model_json_schema() does not
    emit for models with defaults or Optional fields. Self-hosted vLLM does
    not require it, and setting it would reject schemas this project
    legitimately produces."""
    schema = output_structure.model_json_schema()
    name = schema.pop("title", output_structure.__name__)
    return {"name": name, "schema": schema}


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
        method: Literal["function_calling", "json_mode", "json_schema"] = (
            "function_calling"
        ),
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.output_structure = output_structure
        if method == "function_calling":
            # Bypass with_structured_output()'s own Pydantic->tool
            # conversion (see _build_ref_preserving_tool's docstring for
            # why) but reproduce its exact include_raw=True chain shape
            # (RunnableMap(raw=...) | assign-with-fallback) so ainvoke()'s
            # {"raw", "parsed", "parsing_error"} contract is unchanged.
            tool = _build_ref_preserving_tool(output_structure)
            tool_name = tool["function"]["name"]
            bound_llm = llm.bind_tools(
                [tool], tool_choice=tool_name, parallel_tool_calls=False
            )
            output_parser = PydanticToolsParser(
                tools=[output_structure], first_tool_only=True
            )
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser,
                parsing_error=lambda _: None,
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            self.chain: Runnable = RunnableMap(raw=bound_llm) | parser_with_fallback
        elif method == "json_schema":
            # Same $ref-preservation concern as function_calling above, so
            # hand with_structured_output() a pre-built dict rather than the
            # Pydantic class (see _build_ref_preserving_response_format).
            # Passing a dict makes LangChain skip is_pydantic_schema and use
            # JsonOutputParser, which returns a plain dict -- so re-attach
            # Pydantic validation explicitly to keep ainvoke()'s contract
            # (a validated model instance in "parsed") identical across all
            # three methods.
            response_format = _build_ref_preserving_response_format(output_structure)
            self.chain = llm.with_structured_output(
                response_format, include_raw=True, method=method
            ) | RunnablePassthrough.assign(
                parsed=lambda d: (
                    output_structure.model_validate(d["parsed"])
                    if isinstance(d.get("parsed"), dict)
                    else d.get("parsed")
                )
            ).with_fallbacks(
                [RunnablePassthrough.assign(parsed=lambda _: None)],
                exception_key="parsing_error",
            )
        else:
            self.chain = llm.with_structured_output(
                output_structure, include_raw=True, method=method
            )

    @staticmethod
    def _raw_text(raw: Any) -> str:
        """Best-effort plain text of the model's previous reply, for use as
        retry feedback. AIMessage.content is str for most providers but can
        be a list of content blocks, so handle both rather than str()-ing a
        list into something unreadable."""
        content = getattr(raw, "content", raw)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                blk.get("text", "") if isinstance(blk, dict) else str(blk)
                for blk in content
            ]
            return "".join(parts)
        return str(content)

    def _build_correction_message(self, raw: Any, error: str) -> HumanMessage:
        """Turns a parse/validation failure into corrective feedback.

        Mirrors the established "Deterministic Validation + LLM Retry"
        pattern already used at the loop level (see
        src/util/orchestration/retry_loop.py): show what was produced, show
        exactly what was wrong, demand a corrected re-emission. Without
        this the retries are blind -- the identical prompt is re-sent, so a
        deterministic schema mistake reproduces every attempt and simply
        exhausts the budget (observed live: a model repeatedly emitted an
        invalid enum literal three times running)."""
        previous = self._raw_text(raw)
        _MAX_ECHO = 4000
        if len(previous) > _MAX_ECHO:
            previous = previous[:_MAX_ECHO] + "\n...[truncated]"
        return HumanMessage(
            content=(
                "## PREVIOUS ATTEMPT REJECTED\n"
                "Your previous response did not conform to the required output "
                "schema and was discarded.\n\n"
                "### What you produced\n"
                f"{previous}\n\n"
                "### Validation errors\n"
                f"{error}\n\n"
                "### Required action\n"
                "Re-emit the COMPLETE output, corrected. Fix only what the "
                "errors identify; keep all other content identical. Pay close "
                "attention to any field whose value must be one of a fixed set "
                "of literals -- use exactly one of the permitted values, "
                "spelled exactly as listed, not a synonym or a more specific "
                "sub-type name."
            )
        )

    async def ainvoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        import asyncio as _asyncio

        base_messages: List[Any] = [SystemMessage(content=self.system_prompt)]
        base_messages.extend(input_dict.get("messages", []))

        _PARSE_RETRIES = 3
        last_error: Optional[str] = None
        result: Dict[str, Any] = {}
        messages: List[Any] = list(base_messages)
        for attempt in range(_PARSE_RETRIES):
            result = await self.chain.ainvoke(messages)
            # result = {"raw": AIMessage, "parsed": PydanticModel | None, "parsing_error": ...}
            last_error = result.get("parsing_error")
            if not last_error and result.get("parsed") is not None:
                break
            if attempt < _PARSE_RETRIES - 1:
                wait = 2.0 * (attempt + 1)
                logger.warning(
                    f"[agent] '{self.name}' structured output failed validation "
                    f"(attempt {attempt + 1}/{_PARSE_RETRIES}); retrying in "
                    f"{wait:.0f}s WITH the validation error fed back."
                )
                # Rebuild from base each time rather than appending to the
                # running list: only the most recent failure is relevant, and
                # accumulating every rejected attempt would grow the prompt
                # unboundedly across retries.
                messages = base_messages + [
                    self._build_correction_message(
                        result.get("raw"), str(last_error or "unparseable output")
                    )
                ]
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

    Method per provider:
      - json_mode      -- Gemini, DeepSeek (their APIs offer no schema
                          enforcement, only "must be valid JSON")
      - json_schema    -- self-hosted vLLM
      - function_calling -- everything else

    Self-hosted vLLM gets json_schema rather than json_mode because it is the
    only provider here that can actually ENFORCE the schema during sampling:
    vLLM compiles the JSON Schema into a grammar and masks non-conforming
    tokens, so a violation becomes unrepresentable instead of merely
    unlikely. That matters concretely -- a live Gemma-4-26B run under
    json_mode repeatedly emitted category="uniqueness"/"fanout" (sub-type
    names) where the enum only permits the family names, and blind retries
    could not fix it. Note this is the response_format path, NOT tool_choice,
    so it does not require the server to run with --tool-call-parser.

    The OUTPUT FORMAT section (a prose restatement of the Pydantic schema) is
    appended ONLY for json_mode -- that path has no schema enforcement of any
    kind, so the prose description is the model's ONLY source of truth for the
    output shape. json_schema and function_calling both transmit the real,
    machine-enforced schema (see _build_ref_preserving_response_format /
    _build_ref_preserving_tool); appending the same structure again as prose
    would be pure duplication -- telling the model the same thing twice, in
    two formats, at real prompt-length cost for zero benefit.

    Web search is handled via EvidenceStore pre-fetching before
    the agent call, not via tool-calling. See src/util/core/search_tool.py.
    """
    provider, api_key, base_url, env_default = _detect_provider()
    llm = _build_llm(provider, api_key, base_url, env_default, model, use_responses_api)

    # Which structured-output method a provider supports is a property of the
    # provider, so it lives on its ProviderSpec rather than in a third ladder
    # here (see core/providers.py).
    method = PROVIDERS.get(provider, PROVIDERS["openai"]).method
    if method == "json_mode":
        output_format = generate_hierarchical_schema_description(output_structure)
        full_prompt = (
            f"{system_prompt}\n\n"
            f"## OUTPUT FORMAT\n"
            f"Return a JSON object matching this structure:\n{output_format}"
        )
    else:
        full_prompt = system_prompt

    return StructuredAgent(
        system_prompt=full_prompt,
        llm=llm,
        output_structure=output_structure,
        name=name or "structured_agent",
        method=method,
    )
