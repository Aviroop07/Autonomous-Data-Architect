"""Deterministic structural-check node for the context-enrichment loop.

Wraps the existing `filter_external_facts` middleware as a `LoopAgent` so the
enrichment loop can validate proposed external facts each round and feed
structural problems back to the enricher constructively, rather than only
dropping them once at the end.

Policy per round:
  - SELF_REFERENCE   -> auto-repaired in place (strip the fact's own id). A fact
                        that referenced *only* itself becomes an invalid reference
                        after repair and is flagged for re-anchoring.
  - DUPLICATE        -> auto-dropped (dedup is dropping; nothing to re-generate).
  - INVALID_REFERENCE-> flagged via get_errors(), routed to the enricher as
                        feedback so it re-anchors on the next round.

Loop termination is decided here: should_exit when no invalid references remain
AND the last auditor verdict was acceptable. This node emits zero tokens.

This is the constructive first pass; the post-loop `filter_external_facts` call
in the orchestration layer remains the final hard guarantee.
"""

from __future__ import annotations

from typing import List, Optional

from src.pipeline.stage1.middleware.external_context_filter import (
    ExternalFactRejectionCode,
    filter_external_facts,
)
from src.pipeline.stage1.models.context_audit import ContextAuditReport
from src.pipeline.stage1.models.filter_report import FilterReport
from src.pipeline.stage1.models.raw_fact import RawFact
from src.pipeline.stage1.models.rephrased_nl import FactList
from src.util.orchestration.loop_types import (
    HistoryEntry,
    LoopAgent,
    LoopContext,
    LoopOutputModel,
)


class ContextFilterLoopAgent(LoopAgent):
    """Deterministic structural gate + loop-exit decision for enrichment."""

    def __init__(self, original_facts: List[RawFact]) -> None:
        self._original_facts = original_facts
        self._proposed: List[RawFact] = []
        self._last_auditor_acceptable: bool = False

    def build_context(self, ctx: LoopContext) -> str:
        enricher_output = ctx.node_outputs.get("enricher")
        # The proposed facts are the enricher's most recent output. list() copies
        # the container only -- the RawFact objects are shared with node_outputs,
        # so an in-place self-reference repair here persists for the downstream
        # accumulation step.
        self._proposed = (
            list(enricher_output.facts) if isinstance(enricher_output, FactList) else []
        )
        auditor_output = ctx.node_outputs.get("auditor")
        self._last_auditor_acceptable = (
            isinstance(auditor_output, ContextAuditReport)
            and auditor_output.is_acceptable
        )
        return "## DETERMINISTIC STRUCTURAL CHECK"

    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        # 1. Auto-repair self-references before delegating to the shared filter.
        repaired = 0
        for fact in self._proposed:
            if fact.id in fact.referenced_fact_ids:
                fact.referenced_fact_ids = [
                    ref for ref in fact.referenced_fact_ids if ref != fact.id
                ]
                repaired += 1

        # 2. Reuse the canonical filter for reference validity + dedup + bookkeeping.
        result = filter_external_facts(self._proposed, self._original_facts)

        invalid_refs = [
            r
            for r in result.rejected_facts
            if r.code == ExternalFactRejectionCode.INVALID_REFERENCE
        ]
        duplicates = [
            r
            for r in result.rejected_facts
            if r.code == ExternalFactRejectionCode.DUPLICATE_EXTERNAL_FACT
        ]

        structural_errors = [
            (
                f'External fact {r.fact.id} ("{r.fact.fact[:60]}...") references no '
                f"valid original fact (referenced_fact_ids={r.fact.referenced_fact_ids}). "
                "Re-anchor it to at least one original (non-external) fact id, or drop it."
            )
            for r in invalid_refs
        ]

        # 3. Exit only when structurally clean AND the auditor is semantically done.
        should_exit = (not invalid_refs) and self._last_auditor_acceptable

        report = FilterReport(
            should_exit=should_exit,
            accepted_facts=result.accepted_facts,
            structural_errors=structural_errors,
            self_ref_repaired=repaired,
            duplicates_dropped=len(duplicates),
            invalid_reference_count=len(invalid_refs),
        )
        return report, 0

    def emit_history(
        self,
        output: LoopOutputModel,
        prior: Optional[LoopOutputModel],
        round_num: int,
        node: str,
    ) -> HistoryEntry:
        assert isinstance(output, FilterReport)
        summary = (
            f"{len(output.accepted_facts)} accepted, "
            f"{output.invalid_reference_count} invalid-ref, "
            f"{output.duplicates_dropped} dup, "
            f"{output.self_ref_repaired} self-ref repaired; "
            f"should_exit={output.should_exit}"
        )
        return HistoryEntry(
            round=round_num,
            node=node,
            changes_summary=summary,
            was_improvement=(prior is None or output.should_exit),
        )
