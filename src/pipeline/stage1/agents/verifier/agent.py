from pathlib import Path
from typing import Optional

from src.util.core.agent import AgentType, get_agent_
from src.util.core.invoke import get_response
from src.util.orchestration.loop_types import (
    HistoryEntry,
    LoopAgent,
    LoopContext,
    LoopOutputModel,
)
from src.pipeline.stage1.models.integrity_report import IntegrityReport
from src.pipeline.stage1.models.rephrased_nl import RephrasedOutput

PROMPT_PATH = Path(__file__).parent / "prompt.txt"


class VerifierLoopAgent(LoopAgent):
    """LoopAgent for the integrity verifier node."""

    def __init__(self, model: Optional[str] = None) -> None:
        self._model = model
        self._agent: Optional[AgentType] = None

    def _get_agent(self) -> AgentType:
        if self._agent is None:
            system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
            self._agent = get_agent_(
                system_prompt=system_prompt,
                output_structure=IntegrityReport,
                model=self._model,
                name="integrity_verifier",
            )
        return self._agent

    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        parsed, tokens = await get_response(
            agent=self._get_agent(),
            output_structure=IntegrityReport,
            query=query,
        )
        assert isinstance(parsed, IntegrityReport)
        return parsed, tokens

    def build_context(self, ctx: LoopContext) -> str:
        extractor_output = ctx.node_outputs.get("extractor")
        if not isinstance(extractor_output, RephrasedOutput):
            facts_text = "(none yet)"
        else:
            facts_text = "\n".join(
                f'{f.id}. {f.fact}\n   [Segment: "{f.segment_text if hasattr(f, "segment_text") else ""}"]'
                + (f" | External: {f.is_external}" if f.is_external else "")
                for f in extractor_output.flat_facts
            )
        return (
            f"## ORIGINAL DESCRIPTION\n{ctx.initial_context}\n\n"
            f"## EXTRACTED FACTS\n{facts_text}"
        )

    def emit_history(
        self,
        output: LoopOutputModel,
        prior: Optional[LoopOutputModel],
        round_num: int,
        node: str,
    ) -> HistoryEntry:
        assert isinstance(output, IntegrityReport)
        issues = [
            *[
                f"[MISSING][{i.severity.upper()}] {i.description}"
                for i in output.missing_information
            ],
            *[
                f"[INTRODUCED][{i.severity.upper()}] {i.description}"
                for i in output.introduced_information
            ],
            *[
                f"[CHANGED][{i.severity.upper()}] {i.description}"
                for i in output.changed_constraints
            ],
            *[f"[AMBIGUITY] {i.description}" for i in output.unresolved_ambiguities],
        ]
        return HistoryEntry(
            round=round_num,
            node=node,
            changes_summary="safe" if output.is_safe else f"{len(issues)} issues",
            was_improvement=output.is_safe,
        )
