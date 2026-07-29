"""A test `AgentProvider` that answers with canned Pydantic instances.

WHY THIS EXISTS
---------------
Every one of ScribbleDB's integration tests is `--live`-gated, so the default
suite never runs a stage end to end and never composes stage1 -> stage2 ->
stage3. This closes that gap without spending a cent and without patching
anything: `CannedAgentProvider` satisfies `src.util.core.agent_provider.
AgentProvider`, so it goes in through the front door -- the same optional
`provider=` argument production code already threads from each stage's
`orchestrate()` down to the leaf agent modules.

EVERYTHING ELSE RUNS FOR REAL: prompt.md loading, the OUTPUT-FORMAT prose
builder, `get_response()`'s structured_response extraction and its two TypeError
guards, token accounting, AgentLoop routing, every deterministic validator and
middleware, the Beta-mixture merger with its real encoder, the relational
mapper, `canonicalize()`, the DOF graph and the conflict engine.

Responses are routed by the OUTPUT MODEL CLASS. Across stages 1-3 those classes
are all distinct, so the class is a sufficient key and a test never has to care
which agent instance is asking.

Each entry is a QUEUE: successive calls for the same output model pop the next
scripted response and the last one repeats forever. That is what makes a
scripted retry expressible ("bad draft, then good draft") without touching a
single prompt.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Type

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from src.util.core.agent_provider import AgentReply, AgentRequest, LLMAgent

#: Builds the instance to hand back. Takes no arguments -- a factory rather than
#: a pre-built instance so each call gets a fresh object and a test cannot
#: accidentally assert against state a stage mutated in place.
Factory = Callable[[], BaseModel]

#: Sees the rendered query text and may return an instance, raise, or return
#: None to fall through to the queue.
Router = Callable[[str], Optional[BaseModel]]


class CannedAgent:
    """The `LLMAgent` a `CannedAgentProvider` builds.

    Reproduces the real `StructuredAgent.ainvoke` contract exactly -- the same
    typed `AgentRequest` -> `AgentReply`, with real `usage_metadata` on the
    AIMessage -- so `get_response()`'s type guards and token accounting are
    genuinely exercised rather than bypassed.
    """

    def __init__(
        self,
        registry: "CannedAgentProvider",
        output_structure: Type[BaseModel],
        name: str,
    ) -> None:
        self._registry = registry
        self.output_structure = output_structure
        self.name = name

    async def ainvoke(self, request: AgentRequest) -> AgentReply:
        query = "\n".join(str(m.content) for m in request.messages)
        parsed = self._registry.next_for(self.output_structure, query)
        return AgentReply(
            structured_response=parsed,
            messages=[
                AIMessage(
                    content="<canned>",
                    usage_metadata={
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                    },
                )
            ],
        )


class CannedAgentProvider:
    """An `AgentProvider` whose agents answer from a script.

    Pass one to `orchestrate(..., provider=...)` and no LLM call in that run
    reaches the network -- not because anything is blocked, but because no live
    agent is ever constructed.
    """

    def __init__(self) -> None:
        self._script: Dict[str, List[Factory]] = {}
        self._routers: Dict[str, Router] = {}
        #: (output_model_name, query_text) for every call, in order.
        self.call_log: List[Tuple[str, str]] = []

    # -- AgentProvider ------------------------------------------------------

    def build(
        self,
        *,
        system_prompt: str,
        output_structure: Type[BaseModel],
        model: Optional[str] = None,
        name: Optional[str] = None,
    ) -> LLMAgent:
        del system_prompt, model  # the script keys off the output model alone
        return CannedAgent(self, output_structure, name or "canned_agent")

    # -- scripting ----------------------------------------------------------

    def script(
        self, model: Type[BaseModel], *factories: Factory
    ) -> "CannedAgentProvider":
        """Queue one or more responses for `model`. The last repeats forever."""
        if not factories:
            raise ValueError(f"script({model.__name__}) needs at least one factory.")
        self._script[model.__name__] = list(factories)
        return self

    def route(self, model: Type[BaseModel], router: Router) -> "CannedAgentProvider":
        """Install a query-text-dependent router for `model`.

        Consulted BEFORE the queue. Returning None falls through to the queue;
        raising propagates out of `ainvoke`, which is how a test scripts one
        shard of a parallel fan-out to fail while its siblings succeed -- the
        script is keyed by output model, not by shard, so the rendered query is
        the only thing that distinguishes concurrent shards.
        """
        self._routers[model.__name__] = router
        return self

    # -- inspection ---------------------------------------------------------

    @property
    def calls(self) -> List[str]:
        """Output-model names, in call order."""
        return [name for name, _ in self.call_log]

    def call_count(self, model: Type[BaseModel]) -> int:
        return self.calls.count(model.__name__)

    def queries_for(self, model: Type[BaseModel]) -> List[str]:
        return [q for name, q in self.call_log if name == model.__name__]

    # -- dispatch -----------------------------------------------------------

    def next_for(self, model: Type[BaseModel], query: str = "") -> BaseModel:
        key = model.__name__
        self.call_log.append((key, query))

        router = self._routers.get(key)
        if router is not None:
            routed = router(query)
            if routed is not None:
                return routed

        if key not in self._script:
            raise AssertionError(
                f"CannedAgentProvider has no scripted response for {key}. "
                f"Calls so far: {self.calls}"
            )
        queue = self._script[key]
        factory = queue.pop(0) if len(queue) > 1 else queue[0]
        return factory()
