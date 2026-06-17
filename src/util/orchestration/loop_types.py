"""
Data types and configuration models for the AgentLoop infrastructure.

Separated from loop.py so output models, protocols, and config objects
can be imported without pulling in the executor or LLM dependencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# History delta
# ---------------------------------------------------------------------------


class HistoryEntry(BaseModel):
    """Concise per-round delta injected into subsequent rounds' context.

    Raw output artifacts are never injected wholesale -- only this summary.
    """

    round: int
    node: str
    changes_summary: str = Field(
        description="What changed vs prior output for this node"
    )
    unresolved_issues: list[str] = Field(default_factory=list)
    was_improvement: Optional[bool] = Field(
        default=None,
        description="None on round 1; True/False on subsequent rounds",
    )
    tokens: int = Field(default=0, description="Tokens consumed by this agent call")


# ---------------------------------------------------------------------------
# LoopOutputModel -- mandatory base for all loop node output models
# ---------------------------------------------------------------------------


class LoopOutputModel(BaseModel, ABC):
    """Mandatory base class for every output model used in an AgentLoop node.

    Subclasses must implement get_errors(). The infrastructure calls this
    after every invocation to populate entry.unresolved_issues and
    state.det_errors[node]. Generator nodes return [] (no intrinsic errors);
    validator nodes return the actual structural errors.
    """

    @abstractmethod
    def get_errors(self) -> list[str]:
        """Return structural errors in this output. Empty list = valid."""
        ...


# ---------------------------------------------------------------------------
# LoopAgent -- mandatory base for all agents participating in a loop
# ---------------------------------------------------------------------------


class LoopAgent(ABC):
    """Mandatory base class for every agent participating in an AgentLoop.

    Agents have no built-in LLM memory -- each invocation sees only what
    build_context injects into the query string. build_context IS the agent's
    memory system: it reads LoopContext (prior outputs, history, routed
    det_errors) and assembles a query string for the LLM call.

    Stateful cross-round data (e.g. current_schema, accumulated_accepted,
    fix_history) lives as instance fields on the LoopAgent subclass.
    """

    @abstractmethod
    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        """Call the underlying LLM or deterministic logic.

        Returns (parsed_output, tokens). Token count is 0 for deterministic
        validator nodes.
        """
        ...

    @abstractmethod
    def build_context(self, ctx: "LoopContext") -> str:
        """Construct the full query string for this invocation.

        Reads ctx.node_outputs, ctx.history, ctx.det_errors, ctx.ema_issues.
        For stateful agents: update internal state (e.g. current_schema) here
        before formatting the query.
        """
        ...

    @abstractmethod
    def emit_history(
        self,
        output: LoopOutputModel,
        prior: Optional[LoopOutputModel],
        round_num: int,
        node: str,
    ) -> HistoryEntry:
        """Summarize this round for the history log.

        Set was_improvement and changes_summary. Do NOT set unresolved_issues
        -- infrastructure always overwrites it from output.get_errors().
        """
        ...


# ---------------------------------------------------------------------------
# Edge conditions
# ---------------------------------------------------------------------------


class EdgeCondition(BaseModel):
    """Condition evaluated against the source node's output.

    Provide exactly one of: fn  OR  (field + op + value).
    Omitting both makes the edge unconditional.
    """

    fn: Optional[Callable[[BaseModel], bool]] = Field(default=None, exclude=True)
    field: Optional[str] = None
    op: Optional[Literal["eq", "neq", "gte", "lte", "in", "not_in"]] = None
    # object because this comparison value can be bool, str, int, list, etc.
    value: Optional[object] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def evaluate(self, output: BaseModel) -> bool:
        if self.fn is not None:
            return self.fn(output)
        if self.field is None:
            return True
        val = _get_nested(output, self.field)
        op = self.op
        ref = self.value
        if op == "eq":
            return val == ref
        if op == "neq":
            return val != ref
        if op == "gte":
            return val >= ref  # type: ignore[operator]
        if op == "lte":
            return val <= ref  # type: ignore[operator]
        if op == "in":
            return val in ref  # type: ignore[operator]
        if op == "not_in":
            return val not in ref  # type: ignore[operator]
        return True


def _get_nested(obj: object, path: str) -> object:
    for part in path.split("."):
        obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)  # type: ignore[index]
    return obj


# ---------------------------------------------------------------------------
# Graph edges
# ---------------------------------------------------------------------------


class GraphEdge(BaseModel):
    from_node: str
    to_node: str
    condition: Optional[EdgeCondition] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def matches(self, output: BaseModel) -> bool:
        return True if self.condition is None else self.condition.evaluate(output)


# ---------------------------------------------------------------------------
# Per-role config -- intentionally minimal; agents own context/history/errors
# ---------------------------------------------------------------------------


class AgentRoleConfig(BaseModel):
    """Configuration for a single named agent role.

    agent_factory must return a LoopAgent instance. det_error_sources
    controls which other nodes' errors are routed into ctx.det_errors for
    this agent (None = use this node's own errors only).
    """

    agent_factory: Callable[[], LoopAgent] = Field(exclude=True)
    det_error_sources: Optional[list[str]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ---------------------------------------------------------------------------
# Top-level loop config
# ---------------------------------------------------------------------------


class LoopConfig(BaseModel):
    agents: dict[str, AgentRoleConfig]
    graph: dict[str, list[GraphEdge]]  # {"edges": [...]}
    start_node: str
    max_iter: int = 5
    ema_alpha: float = 0.4
    ema_persistent_threshold: float = 0.6

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_graph(self) -> "LoopConfig":
        edges = self.graph.get("edges", [])
        valid_nodes = set(self.agents.keys()) | {"end"}
        errors: list[str] = []

        if self.start_node not in self.agents:
            errors.append(f"start_node '{self.start_node}' not in agents")

        for edge in edges:
            if edge.from_node not in self.agents and edge.from_node != "start":
                errors.append(f"Edge from_node '{edge.from_node}' not in agents")
            if edge.to_node not in valid_nodes:
                errors.append(
                    f"Edge to_node '{edge.to_node}' not in agents and not 'end'"
                )

        if errors:
            raise ValueError(
                "LoopConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )
        return self


# ---------------------------------------------------------------------------
# Context passed to agent.build_context()
# ---------------------------------------------------------------------------


@dataclass
class LoopContext:
    initial_context: str
    current_node: str
    iteration: int
    node_outputs: dict[str, LoopOutputModel]
    history: list[HistoryEntry]
    det_errors: list[str]
    det_errors_by_node: dict[str, list[str]]
    ema_issues: list[Tuple[str, float]]


# ---------------------------------------------------------------------------
# Loop result
# ---------------------------------------------------------------------------


class LoopResult(BaseModel):
    final_output: Optional[LoopOutputModel]
    final_node: str
    total_tokens: int
    tokens_by_node: dict[str, int]
    iteration_count: int
    history: list[HistoryEntry]
    converged: bool
    det_errors_exhausted: bool
    node_outputs: dict[str, LoopOutputModel] = Field(
        default_factory=dict,
        description="Last output per named node -- use this to retrieve any node's final artifact",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ---------------------------------------------------------------------------
# Internal per-run state
# ---------------------------------------------------------------------------


@dataclass
class _LoopState:
    initial_context: str
    node_outputs: dict[str, LoopOutputModel] = field(default_factory=dict)
    node_prev: dict[str, LoopOutputModel] = field(default_factory=dict)
    history: list[HistoryEntry] = field(default_factory=list)
    det_errors: dict[str, list[str]] = field(default_factory=dict)
    ema_weights: dict[str, float] = field(default_factory=dict)
    tokens_by_node: dict[str, int] = field(default_factory=dict)
    iteration: int = 0
    total_tokens: int = 0
    converged: bool = False
    det_errors_exhausted: bool = False
