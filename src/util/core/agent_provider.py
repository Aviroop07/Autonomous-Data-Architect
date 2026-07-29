"""The injectable, TYPED LLM-agent boundary.

Two problems, one module.

1. HIDDEN DEPENDENCY. Every agent in stages 1-4 was built by `get_agent_()`
   (agent.py), which reaches straight for a provider API key and constructs a
   live `ChatOpenAI`. Nothing between `orchestrate()` and the leaf agent module
   ever named that, so a caller could not substitute it, record it, or run the
   pipeline without a key. `AgentProvider` names it: "given a system prompt and
   an output model, give me something I can invoke". `LiveAgentProvider` is the
   production implementation and the default everywhere, forwarding verbatim to
   `get_agent_()`, so no live behavior changes.

2. UNTYPED PAYLOAD. The boundary used to be `Dict[str, Any] -> Dict[str, Any]`,
   with the two meaningful keys ("structured_response", "messages") spelled out
   as string literals at every call site and reachable only through `.get()`.
   `AgentRequest` and `AgentReply` replace that with real models, so a missing or
   misnamed field is a type error rather than a `None` discovered three frames
   later. Nothing in the pipeline handles an agent payload as a dict any more.

What the provider seam buys beyond testability: it is the natural place for a
record/replay run (a reproducibility artifact wants to re-derive results without
re-billing), for a deterministic offline run, and for pinning one stage to a
different backend than another.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Type, TypeVar, runtime_checkable

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field, SkipValidation

T = TypeVar("T", bound=BaseModel)


class AgentRequest(BaseModel):
    """What the pipeline hands an agent: the conversation to answer.

    The system prompt is NOT here -- it is fixed when the agent is built, so it
    belongs to the agent, not to a single request.
    """

    # BaseMessage is a third-party model; SkipValidation keeps the concrete
    # subclass (HumanMessage/AIMessage) intact. Without it, pydantic would
    # re-validate each entry against BaseMessage and flatten subclass-only
    # fields -- notably AIMessage.usage_metadata, which token accounting reads.
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    messages: SkipValidation[List[BaseMessage]] = Field(
        description="The conversation turns to send, in order."
    )


class AgentReply(BaseModel):
    """What an agent hands back: the validated output plus the raw exchange.

    `structured_response` is `Optional` because an agent built with no output
    model answers in prose; `get_response()` is what narrows it to a concrete
    type and raises when it is absent but required.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Annotated as the BaseModel base rather than a generic parameter: the
    # concrete type is the agent's own `output_structure`, and get_response()
    # already isinstance-narrows against it. SkipValidation is essential here --
    # validating a subclass instance against bare `BaseModel` would construct a
    # fieldless BaseModel and silently discard every field.
    structured_response: SkipValidation[Optional[BaseModel]] = Field(
        default=None,
        description="The parsed output model, when the agent has an output schema.",
    )
    messages: SkipValidation[List[BaseMessage]] = Field(
        default_factory=list,
        description="The full exchange, including the raw reply whose "
        "usage_metadata drives token accounting.",
    )


@runtime_checkable
class LLMAgent(Protocol):
    """The entire agent surface the pipeline consumes: one typed call."""

    async def ainvoke(self, request: AgentRequest) -> AgentReply: ...


class AgentProvider(Protocol):
    """Builds an `LLMAgent` for a given prompt + output model.

    The keyword names mirror `get_agent_()`'s deliberately, so the production
    implementation is a pass-through and every call site reads the same before
    and after threading.
    """

    def build(
        self,
        *,
        system_prompt: str,
        output_structure: Type[T],
        model: Optional[str] = None,
        name: Optional[str] = None,
    ) -> LLMAgent: ...


class LiveAgentProvider:
    """Production provider: a real, key-backed `StructuredAgent`."""

    __slots__ = ()

    def build(
        self,
        *,
        system_prompt: str,
        output_structure: Type[T],
        model: Optional[str] = None,
        name: Optional[str] = None,
    ) -> LLMAgent:
        # Imported here, not at module scope: agent.py imports THIS module for
        # its own type aliases, so a top-level import would be circular.
        from src.util.core.agent import get_agent_

        return get_agent_(
            system_prompt=system_prompt,
            output_structure=output_structure,
            model=model,
            name=name,
        )


#: The default provider. A module-level singleton because it is stateless.
LIVE_AGENT_PROVIDER: AgentProvider = LiveAgentProvider()


def resolve_agent_provider(provider: Optional[AgentProvider]) -> AgentProvider:
    """`provider` when given, else the live one. The single place the default is
    applied, so no call site repeats the `or` and none can drift."""
    return provider if provider is not None else LIVE_AGENT_PROVIDER
