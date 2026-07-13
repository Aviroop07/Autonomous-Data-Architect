"""Logic audit agent for Stage 3.

Second, independently-prompted LLM re-reading the original NL facts
against logic_extractor's structured output (both its Constraint list and
its DerivedColumnConstraint list). See statistical_auditor/agent.py for
the shared LoopAgent pattern this mirrors.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.pipeline.stage3.agents.extraction_outputs import AuditReport, LogicOutput
from src.util.core.agent import AgentType, get_agent_
from src.util.core.invoke import get_response
from src.util.orchestration.loop_types import (
    HistoryEntry,
    LoopAgent,
    LoopContext,
    LoopOutputModel,
)

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent / "prompt.txt"


class LogicAuditorLoopAgent(LoopAgent):
    """LoopAgent for the logic audit node."""

    def __init__(self, model: Optional[str] = None) -> None:
        self._model = model
        self._agent: Optional[AgentType] = None

    def _get_agent(self) -> AgentType:
        if self._agent is None:
            system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
            self._agent = get_agent_(
                system_prompt=system_prompt,
                output_structure=AuditReport,
                model=self._model,
                name="logic_auditor",
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

    def build_context(self, ctx: LoopContext) -> str:
        context_data: Dict[str, Any] = {}
        try:
            context_data = json.loads(ctx.initial_context)
        except json.JSONDecodeError, TypeError, AttributeError:
            logger.warning("[LogicAuditor] Failed to parse initial_context as JSON.")

        fact_ids = context_data.get("fact_ids", [])
        facts_map = context_data.get("facts_map", {})
        facts_text = "\n".join(
            f"- [id={fid}] {facts_map[str(fid)]['fact']}"
            for fid in sorted(fact_ids)
            if str(fid) in facts_map
        )

        extractor_output = ctx.node_outputs.get("logic_extractor")
        if isinstance(extractor_output, LogicOutput):
            extracted_text = extractor_output.model_dump_json(indent=2)
        else:
            extracted_text = "(no extraction yet)"

        return (
            f"## ORIGINAL NL FACTS\n{facts_text}\n\n"
            f"## EXTRACTED LOGIC CONSTRAINTS AND DERIVED COLUMNS\n{extracted_text}"
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
