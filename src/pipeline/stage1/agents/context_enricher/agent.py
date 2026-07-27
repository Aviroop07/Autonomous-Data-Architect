import re
from pathlib import Path
from typing import List, Optional, Set

from src.util.core.agent import AgentType, get_agent_
from src.util.core.invoke import get_response
from src.util.core.search_tool import EvidenceStore
from src.util.orchestration.loop_types import (
    HistoryEntry,
    LoopAgent,
    LoopContext,
    LoopOutputModel,
)
from src.pipeline.stage1.models.rephrased_nl import FactList
from src.pipeline.stage1.models.raw_fact import RawFact
from src.pipeline.stage1.models.context_audit import ContextAuditReport
from src.pipeline.stage1.models.coverage_report import SpecGap

PROMPT_PATH = Path(__file__).parent / "prompt.txt"

_ACRONYM_RE = re.compile(r"\b[A-Z]{2,5}\b")
_MAX_FACTS_IN_CONTEXT = 100  # cap facts shown to enricher; loop handles the rest
_MAX_FACT_TEXT_LEN = 200     # truncate individual fact texts
_MAX_ORIGIN_LEN = 150        # truncate segment origin texts
_MAX_OUTPUT_FACTS = 15       # max external facts per round


def build_context_enricher_agent(
    system_prompt: str,
    model: Optional[str] = None,
) -> AgentType:
    return get_agent_(
        system_prompt=system_prompt,
        output_structure=FactList,
        model=model,
        name="domain_specialist",
    )


class ContextEnricherLoopAgent(LoopAgent):
    """LoopAgent for the context enricher node.

    Runs a two-phase approach per iteration:
      1. Pre-search: derive queries from gaps or auditor directions, run them
         concurrently via DDG, inject results into context.
      2. Generation: StructuredAgent produces external facts from facts + gaps + search context.
    """

    def __init__(
        self,
        original_facts: List[RawFact],
        gaps: List[SpecGap],
        evidence_store: EvidenceStore,
        model: Optional[str] = None,
    ) -> None:
        self._facts = original_facts
        self._gaps = gaps
        self._evidence = evidence_store
        self._open_gap_ids: Set[int] = {g.id for g in gaps}
        self._auditor_next_queries: List[str] = []
        self._model = model
        self._agent: Optional[AgentType] = None
        self.accumulated_accepted: List[RawFact] = []
        # True unless this round is a structural-only retry (last auditor already
        # acceptable, loop bounced back solely to fix a filter-flagged reference).
        # Gates BOTH the outgoing edge to the auditor and phase-1 web search.
        self.needs_audit: bool = True

    def _get_agent(self) -> AgentType:
        if self._agent is None:
            system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
            self._agent = build_context_enricher_agent(
                system_prompt=system_prompt, model=self._model
            )
        return self._agent

    def _derive_search_queries(self, max_queries: int = 5) -> List[str]:
        """Build search queries from auditor directions, gaps, and acronyms as a fallback."""
        if self._auditor_next_queries:
            base = list(self._auditor_next_queries)
        else:
            base = [g.search_query for g in self._gaps if g.id in self._open_gap_ids]

        queries: List[str] = []
        seen = set()

        # Add primary queries (gaps or auditor directions)
        for q in base:
            if q.lower() not in seen:
                queries.append(q)
                seen.add(q.lower())
            if len(queries) >= max_queries:
                return queries

        # Fallback: acronym extraction
        for fact in self._facts:
            for match in _ACRONYM_RE.finditer(fact.fact):
                term = match.group()
                if term.lower() not in seen:
                    queries.append(f"{term} definition")
                    seen.add(term.lower())
                if len(queries) >= max_queries:
                    return queries

        return queries

    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        # Phase 1: run searches and inject results into the query context.
        # Skipped on structural-only retries -- re-anchoring reads the original
        # facts already in context, so no new evidence is needed and the auditor
        # (which would consume it) is bypassed this round.
        if self.needs_audit:
            queries = self._derive_search_queries()
            if queries:
                evidence = await self._evidence.fetch(queries, max_results=5)
                if evidence.formatted:
                    query = query + "\n\n" + evidence.formatted

        # Phase 2: generate external facts with the enriched context.
        parsed, tokens = await get_response(
            agent=self._get_agent(),
            output_structure=FactList,
            query=query,
        )
        assert isinstance(parsed, FactList)
        if len(parsed.facts) > _MAX_OUTPUT_FACTS:
            print(
                f"[Stage 1] Enricher produced {len(parsed.facts)} facts, "
                f"capping to {_MAX_OUTPUT_FACTS} to avoid JSON truncation."
            )
            parsed.facts = parsed.facts[:_MAX_OUTPUT_FACTS]
        for fact in parsed.facts:
            fact.is_external = True
        return parsed, tokens

    def build_context(self, ctx: LoopContext) -> str:
        audit_output = ctx.node_outputs.get("auditor")
        audit_feedback: Optional[str] = None

        # A structural-only retry is one where the last auditor already accepted
        # the context (so it emitted no new search queries) and the loop bounced
        # back solely to fix a filter-flagged reference. On such rounds we skip
        # both the auditor (edge) and web search (invoke). First round: no auditor
        # yet -> needs_audit stays True.
        self.needs_audit = not (
            isinstance(audit_output, ContextAuditReport) and audit_output.is_acceptable
        )

        if isinstance(audit_output, ContextAuditReport):
            prior_enriched = ctx.node_outputs.get("enricher")
            if isinstance(prior_enriched, FactList):
                accepted_ids = set(audit_output.accepted_fact_ids)
                existing_ids = {f.id for f in self.accumulated_accepted}
                for fact in prior_enriched.facts:
                    if fact.id in accepted_ids and fact.id not in existing_ids:
                        self.accumulated_accepted.append(fact)
                        existing_ids.add(fact.id)
            # Auditor dictates the next state
            self._open_gap_ids = set(audit_output.unresolved_gap_ids)
            self._auditor_next_queries = list(audit_output.next_search_queries)
            if self._open_gap_ids and not self._auditor_next_queries:
                print(
                    "[Stage 1] WARNING: Auditor reported open gaps but provided no next_search_queries. "
                    "Enricher will fallback to original gap queries."
                )

            if not audit_output.is_acceptable:
                audit_feedback = (
                    audit_output.retry_instructions
                    or "Context enrichment retry required."
                )

        # Build GAPS TO CLOSE section
        gaps_to_close = [g for g in self._gaps if g.id in self._open_gap_ids]
        if gaps_to_close:
            gaps_text = "\n".join(
                f"- [Gap {g.id}] {g.dimension.value.upper()}: {g.description}"
                for g in gaps_to_close
            )
            query = f"## GAPS TO CLOSE\n{gaps_text}\n\n"
        else:
            query = "## GAPS TO CLOSE\nNone.\n\n"

        def _trunc(t: str, limit: int) -> str:
            return t[:limit] + "..." if len(t) > limit else t

        facts_shown = self._facts[:_MAX_FACTS_IN_CONTEXT]
        facts_text = "\n".join(
            f"- id: {f.id}\n  fact: {_trunc(f.fact, _MAX_FACT_TEXT_LEN)}\n"
            f"  origin: {_trunc(getattr(f, 'segment_text', '') or '(none)', _MAX_ORIGIN_LEN)}"
            for f in facts_shown
        )
        total = len(self._facts)
        hidden = total - len(facts_shown)
        if hidden > 0:
            facts_text += (
                f"\n# ... and {hidden} more facts (truncated for length). "
                f"The gaps above are the priority; you do not need to see every fact to close them."
            )
        query += f"## FACTS TO ENRICH ({len(facts_shown)} shown, {total} total)\n{facts_text}"

        if self.accumulated_accepted:
            accepted_text = "\n".join(
                f"- id: {f.id}\n  fact: {f.fact}" for f in self.accumulated_accepted
            )
            query += f"\n\n## PREVIOUSLY ACCEPTED EXTERNAL FACTS (keep these)\n{accepted_text}"

        if audit_feedback:
            query += (
                f"\n\n## CONTEXT AUDIT FEEDBACK FROM PREVIOUS ATTEMPT\n{audit_feedback}"
            )

        # Structural feedback routed from the deterministic filter node (invalid
        # references it flagged last round). The enricher must re-anchor or drop
        # those facts. Routed via det_error_sources=["filter"] in the loop config.
        if ctx.det_errors:
            feedback = "\n".join(f"- {err}" for err in ctx.det_errors)
            query += (
                "\n\n## STRUCTURAL VALIDATION FEEDBACK (correct these facts and re-propose them)\n"
                f"{feedback}"
            )

        return query

    def emit_history(
        self,
        output: LoopOutputModel,
        prior: Optional[LoopOutputModel],
        round_num: int,
        node: str,
    ) -> HistoryEntry:
        assert isinstance(output, FactList)
        count = len(output.facts)
        prior_count = len(prior.facts) if isinstance(prior, FactList) else None  # type: ignore[union-attr]
        changes_summary = f"proposed {count} external facts"
        if prior_count is not None:
            delta = count - prior_count
            changes_summary += f" ({delta:+d} vs prior)"
        return HistoryEntry(
            round=round_num,
            node=node,
            changes_summary=changes_summary,
            was_improvement=(prior_count is None or count != prior_count),
        )
