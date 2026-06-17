from pathlib import Path
from typing import Any, Callable, List, Optional

from src.util.core.agent import AgentType, get_agent_
from src.util.core.invoke import get_response
from src.util.config.web_search import get_web_search_tool
from src.util.orchestration.loop_types import (
    HistoryEntry,
    LoopAgent,
    LoopContext,
    LoopOutputModel,
)
from src.pipeline.stage1.models.rephrased_nl import FactList
from src.pipeline.stage1.models.raw_fact import RawFact
from src.pipeline.stage1.models.context_audit import ContextAuditReport

PROMPT_PATH = Path(__file__).parent / "prompt.txt"
AgentFactory = Callable[..., AgentType]


def get_context_enricher_tools() -> List[dict[str, Any]]:
    return [get_web_search_tool()]


def build_context_enricher_agent(
    system_prompt: str,
    model: Optional[str] = None,
    agent_factory: AgentFactory = get_agent_,
) -> AgentType:
    return agent_factory(
        system_prompt=system_prompt,
        output_structure=FactList,
        tools=get_context_enricher_tools(),
        model=model,
        name="domain_specialist",
        use_responses_api=True,
    )


class ContextEnricherLoopAgent(LoopAgent):
    """LoopAgent for the context enricher node.

    Accumulates accepted external facts across rounds. After each auditor
    response, merges newly accepted facts into accumulated_accepted before
    building the next enrichment query.
    """

    def __init__(
        self,
        original_facts: List[RawFact],
        search_suggestions: Optional[List[str]] = None,
        model: Optional[str] = None,
    ) -> None:
        self._facts = original_facts
        self._search_suggestions = search_suggestions
        self._model = model
        self._agent: Optional[AgentType] = None
        self.accumulated_accepted: List[RawFact] = []

    def _get_agent(self) -> AgentType:
        if self._agent is None:
            system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
            self._agent = build_context_enricher_agent(
                system_prompt=system_prompt, model=self._model
            )
        return self._agent

    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        parsed, tokens = await get_response(
            agent=self._get_agent(),
            output_structure=FactList,
            query=query,
        )
        assert isinstance(parsed, FactList)
        for fact in parsed.facts:
            fact.is_external = True
        return parsed, tokens

    def build_context(self, ctx: LoopContext) -> str:
        audit_output = ctx.node_outputs.get("auditor")
        audit_feedback: Optional[str] = None

        if isinstance(audit_output, ContextAuditReport):
            prior_enriched = ctx.node_outputs.get("enricher")
            if isinstance(prior_enriched, FactList):
                accepted_ids = set(audit_output.accepted_fact_ids)
                existing_ids = {f.id for f in self.accumulated_accepted}
                for fact in prior_enriched.facts:
                    if fact.id in accepted_ids and fact.id not in existing_ids:
                        self.accumulated_accepted.append(fact)
                        existing_ids.add(fact.id)
            if not audit_output.is_acceptable:
                audit_feedback = (
                    audit_output.retry_instructions
                    or "Context enrichment retry required."
                )

        facts_text = "\n".join(
            f"- id: {f.id}\n  fact: {f.fact}\n  origin: {f.origin if f.origin else '(none)'}"
            for f in self._facts
        )
        query = f"## FACTS TO ENRICH\n{facts_text}"

        if self.accumulated_accepted:
            accepted_text = "\n".join(
                f"- id: {f.id}\n  fact: {f.fact}" for f in self.accumulated_accepted
            )
            query += f"\n\n## PREVIOUSLY ACCEPTED EXTERNAL FACTS (keep these)\n{accepted_text}"

        if audit_feedback:
            query += (
                f"\n\n## CONTEXT AUDIT FEEDBACK FROM PREVIOUS ATTEMPT\n{audit_feedback}"
            )

        if self._search_suggestions:
            suggestions_text = "\n".join(f"- {s}" for s in self._search_suggestions)
            query += (
                f"\n\n## SUGGESTED SEARCHES\nThe verifier or underspecification detector "
                f"recommends these searches:\n{suggestions_text}\n"
                "Prioritize these searches when using the web search tool."
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
