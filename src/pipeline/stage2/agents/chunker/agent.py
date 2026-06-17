from pathlib import Path
from typing import List, Optional

from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage2.models.chunk import ChunkedPlan
from src.util.core.agent import AgentType, get_agent_
from src.util.core.invoke import get_response
from src.util.orchestration.loop_types import (
    HistoryEntry,
    LoopAgent,
    LoopContext,
    LoopOutputModel,
)

PROMPT_PATH = Path(__file__).parent / "prompt.txt"


class ChunkerLoopAgent(LoopAgent):
    """LoopAgent for the chunker node.

    Absorbs the old _ChunkerContextBuilder logic. Error feedback arrives via
    ctx.det_errors (routed from the chunker_validator node via det_error_sources).
    """

    def __init__(
        self,
        facts: List[AtomicFact],
        domain: Optional[str] = None,
        analytical_goal: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self._model = model
        self._agent: Optional[AgentType] = None
        self._domain = domain
        self._analytical_goal = analytical_goal
        self._formatted_facts = "\n".join(
            f"{f.id}. {f.fact} [{', '.join(f.tags)}] (refs: {f.referenced_fact_ids})"
            for f in facts
        )

    def _get_agent(self) -> AgentType:
        if self._agent is None:
            system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
            self._agent = get_agent_(
                system_prompt=system_prompt,
                output_structure=ChunkedPlan,
                model=self._model,
                name="chunker_stage2",
            )
        return self._agent

    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        parsed, tokens = await get_response(
            agent=self._get_agent(),
            output_structure=ChunkedPlan,
            query=query,
        )
        assert isinstance(parsed, ChunkedPlan)
        return parsed, tokens

    def build_context(self, ctx: LoopContext) -> str:
        parts: List[str] = []
        if self._domain:
            parts.append(f"### DOMAIN\n{self._domain}\n")
        if self._analytical_goal:
            parts.append(f"### ANALYTICAL GOAL\n{self._analytical_goal}\n")
        parts.append(
            f"### ENRICHED NL DESCRIPTION (ATOMIC FACTS):\n{self._formatted_facts}"
        )

        if ctx.det_errors:
            error_lines = "\n".join(f"- {e}" for e in ctx.det_errors)
            parts.append(
                f"### ERRORS FROM PREVIOUS ATTEMPT\n{error_lines}\n\n"
                "Please fix ONLY these issues and keep valid chunks unchanged."
            )

        return "\n\n".join(parts)

    def emit_history(
        self,
        output: LoopOutputModel,
        prior: Optional[LoopOutputModel],
        round_num: int,
        node: str,
    ) -> HistoryEntry:
        assert isinstance(output, ChunkedPlan)
        changes_summary = (
            f"{len(output.chunks)} chunks, {len(output.core_modeling_facts)} core facts"
        )
        was_improvement = None
        if isinstance(prior, ChunkedPlan):
            was_improvement = output.model_dump() != prior.model_dump()
        return HistoryEntry(
            round=round_num,
            node=node,
            changes_summary=changes_summary,
            was_improvement=was_improvement,
        )
