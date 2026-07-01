from pathlib import Path
from typing import List, Optional

from src.pipeline.stage1.models.context_audit import (
    ContextAuditAttempt,
    ContextAuditReport,
)
from src.pipeline.stage1.models.raw_fact import RawFact
from src.pipeline.stage1.models.rephrased_nl import FactList
from src.pipeline.stage1.models.coverage_report import SpecGap
from src.util.core.search_tool import EvidenceStore
from src.util.core.agent import AgentType, get_agent_
from src.util.core.invoke import get_response
from src.util.orchestration.loop_types import (
    HistoryEntry,
    LoopAgent,
    LoopContext,
    LoopOutputModel,
)

PROMPT_PATH = Path(__file__).parent / "prompt.txt"


class ContextAuditorLoopAgent(LoopAgent):
    """LoopAgent for the context auditor node.

    Tracks audit_trail so the orchestration layer can inspect per-attempt
    audit reports after the loop completes.
    """

    def __init__(
        self,
        original_facts: List[RawFact],
        gaps: List[SpecGap],
        evidence_store: EvidenceStore,
        model: Optional[str] = None,
    ) -> None:
        self._original_facts = original_facts
        self._gaps = gaps
        self._evidence_store = evidence_store
        self._model = model
        self._agent: Optional[AgentType] = None
        self._last_proposed_count: int = 0
        self._attempt: int = 0
        self.audit_trail: List[ContextAuditAttempt] = []

    def _get_agent(self) -> AgentType:
        if self._agent is None:
            system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
            self._agent = get_agent_(
                system_prompt=system_prompt,
                output_structure=ContextAuditReport,
                model=self._model,
                name="context_auditor_stage1",
            )
        return self._agent

    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        parsed, tokens = await get_response(
            agent=self._get_agent(),
            output_structure=ContextAuditReport,
            query=query,
        )
        assert isinstance(parsed, ContextAuditReport)
        return parsed, tokens

    def build_context(self, ctx: LoopContext) -> str:
        enricher_output = ctx.node_outputs.get("enricher")
        proposed_external_facts: List[RawFact] = (
            enricher_output.facts  # type: ignore[union-attr]
            if isinstance(enricher_output, FactList)
            else []
        )
        self._last_proposed_count = len(proposed_external_facts)

        original_text = "\n".join(
            f"- id: {fact.id}\n  fact: {fact.fact}\n  is_external: {fact.is_external}"
            for fact in self._original_facts
        )
        proposed_text = "\n".join(
            f"- id: {fact.id}\n  fact: {fact.fact}\n  referenced_fact_ids: {fact.referenced_fact_ids}\n  addresses_gap: {fact.addresses_gap}\n  evidence_refs: {fact.evidence_refs}"
            for fact in proposed_external_facts
        )
        
        # Inject EVIDENCE section
        tags = set()
        for fact in proposed_external_facts:
            tags.update(fact.evidence_refs)
            
        resolved_evidence = self._evidence_store.resolve(list(tags))
        evidence_text = ""
        if resolved_evidence:
            evidence_lines = []
            for e in resolved_evidence:
                evidence_lines.append(f"[{e.tag}] {e.title}\n    Source: {e.url}\n    {e.text}")
            evidence_text = "\n".join(evidence_lines)
        else:
            evidence_text = "None cited."

        return (
            "## ORIGINAL FACTS\n"
            f"{original_text}\n\n"
            "## PROPOSED EXTERNAL CONTEXT FACTS\n"
            f"{proposed_text}\n\n"
            "## EVIDENCE (as retrieved)\n"
            f"{evidence_text}"
        )

    def emit_history(
        self,
        output: LoopOutputModel,
        prior: Optional[LoopOutputModel],
        round_num: int,
        node: str,
    ) -> HistoryEntry:
        assert isinstance(output, ContextAuditReport)
        self._attempt += 1
        self.audit_trail.append(
            ContextAuditAttempt(
                attempt=self._attempt,
                proposed_fact_count=self._last_proposed_count,
                report=output,
            )
        )
        changes_summary = (
            "acceptable"
            if output.is_acceptable
            else f"rejected {len(output.rejected_facts)} facts"
        )
        return HistoryEntry(
            round=round_num,
            node=node,
            changes_summary=changes_summary,
            was_improvement=(prior is None or output.is_acceptable),
        )
