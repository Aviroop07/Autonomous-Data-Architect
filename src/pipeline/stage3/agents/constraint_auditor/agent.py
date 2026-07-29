"""Unified constraint-audit agent for Stage 3.

Second, independently-prompted LLM re-reading the original NL facts
against constraint_generator's structured output -- catches semantic
errors deterministic checking structurally can't (a dropped condition,
wrong-table attribution, a hallucinated bound, missed conditionality).
Replaces the 3 separate statistical/structural/logic auditors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.pipeline.stage3.agents.extraction_outputs import AuditReport, UnifiedOutput
from src.pipeline.stage3.models.shard_context import Stage3ShardContext
from src.util.core.agent import AgentType
from src.util.core.agent_provider import AgentProvider, resolve_agent_provider
from src.util.core.invoke import get_response
from src.util.orchestration.loop_types import (
    HistoryEntry,
    LoopAgent,
    LoopContext,
    LoopOutputModel,
)

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent / "prompt.md"


class ConstraintAuditorLoopAgent(LoopAgent):
    """LoopAgent for the unified audit node."""

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[AgentProvider] = None,
    ) -> None:
        self._model = model
        self._provider = provider
        self._agent: Optional[AgentType] = None

    def _get_agent(self) -> AgentType:
        if self._agent is None:
            system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
            self._agent = resolve_agent_provider(self._provider).build(
                system_prompt=system_prompt,
                output_structure=AuditReport,
                model=self._model,
                name="constraint_auditor",
            )
        return self._agent

    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        parsed, tokens = await get_response(
            agent=self._get_agent(),
            output_structure=AuditReport,
            query=query,
        )
        assert isinstance(parsed, AuditReport)
        return parsed, tokens

    def build_context(self, ctx: LoopContext[Stage3ShardContext]) -> str:
        # Reads the typed context directly. This used to json.loads() the
        # context back out of a string and then dict-index it
        # (facts_map[str(fid)]["fact"]), which meant a renamed key was
        # invisible to the type checker and failed at runtime in three separate
        # consumers. The producer now hands over the object; the only string on
        # this path is the prompt text built below.
        ctx_data = ctx.initial_context
        if not isinstance(ctx_data, Stage3ShardContext):
            # Kept deliberately: this guards a genuinely different failure --
            # a caller wiring the wrong payload into this loop -- rather than
            # the JSON round-trip damage the old ladder existed to survive.
            logger.warning(
                "[ConstraintAuditor] initial_context was %s, not "
                "Stage3ShardContext; auditing with no fact context.",
                type(ctx_data).__name__,
            )
            facts_text = ""
        else:
            facts_text = "\n".join(
                f"- [id={fid}] {ctx_data.facts_map[fid].fact}"
                for fid in sorted(ctx_data.fact_ids)
                if fid in ctx_data.facts_map
            )

        extractor_output = ctx.node_outputs.get("generator")
        if isinstance(extractor_output, UnifiedOutput):
            extracted_text = extractor_output.model_dump_json(indent=2)
        else:
            extracted_text = "(no extraction yet)"

        return (
            f"## ORIGINAL NL FACTS\n{facts_text}\n\n"
            f"## EXTRACTED CONSTRAINTS (all categories)\n{extracted_text}"
        )

    def emit_history(
        self,
        output: LoopOutputModel,
        prior: Optional[LoopOutputModel],
        round_num: int,
        node: str,
    ) -> HistoryEntry:
        assert isinstance(output, AuditReport)
        return HistoryEntry(
            round=round_num,
            node=node,
            changes_summary=(
                f"audit {'passed' if output.is_valid else 'FAILED'} "
                f"({len(output.issues)} issues)"
            ),
            was_improvement=None,
        )
