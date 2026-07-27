from typing import List, Optional, Tuple


from src.pipeline.stage1.models.rephrased_nl import AtomicFact

from src.pipeline.stage2.models.corrections import FixHistoryStep
from src.util.orchestration.loop import AgentLoop
from src.util.orchestration.loop_types import (
    AgentRoleConfig,
    EdgeCondition,
    ErrorRefreshConfig,
    GraphEdge,
    HistoryEntry,
    LoopAgent,
    LoopConfig,
    LoopContext,
    LoopOutputModel,
)

from src.pipeline.stage2.agents.er_extractor.agent import (
    get_agent as get_conceptual_extractor_agent,
)
from src.pipeline.stage2.agents.er_auditor.agent import (
    get_agent as get_conceptual_verifier_agent,
)
from src.pipeline.stage2.mapper.conceptual_model import ConceptualModel
from src.pipeline.stage2.models.conceptual_critique import ConceptualCritiqueReport


# ---------------------------------------------------------------------------
# Private types
# ---------------------------------------------------------------------------


class ERExtractorLoopAgent(LoopAgent):
    def __init__(
        self, facts: List[AtomicFact], nl_query: str, model: Optional[str] = None
    ):
        self._facts = facts
        self._nl_query = nl_query
        self._model = model
        self.agent = get_conceptual_extractor_agent(model)
        self._feedback_history: list[Tuple[int, list[str]]] = []

    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        from src.util.core.invoke import get_response

        parsed, tokens = await get_response(self.agent, ConceptualModel, query)
        return parsed, tokens

    def build_context(self, ctx: LoopContext) -> str:
        current_round_feedback = []

        # Harvest deterministic errors from the Filter
        filter_report = ctx.node_outputs.get("filter")
        if filter_report and not filter_report.is_valid:
            current_round_feedback.extend(getattr(filter_report, "det_errors", []))

        # Harvest semantic critique from the Auditor
        auditor_report = ctx.node_outputs.get("auditor")
        if (
            isinstance(auditor_report, ConceptualCritiqueReport)
            and not auditor_report.is_valid
        ):
            for fix in auditor_report.fixes:
                current_round_feedback.append(f"{fix.description}: {fix.rationale}")

        # Save this round's feedback
        if current_round_feedback:
            self._feedback_history.append((ctx.iteration, current_round_feedback))

        facts_str = "\n".join([f"- [{f.id}] {f.fact}" for f in self._facts])
        query = (
            f"## INPUT\nOriginal Description:\n{self._nl_query}\n\nFacts:\n{facts_str}"
        )

        if self._feedback_history:
            query += (
                "\n\n## PAST FEEDBACK\n"
                "You have attempted to generate this model before. Below is the historical "
                "feedback from your previous attempts. Note that some of these issues may have "
                "already been fixed in your most recent drafts, but keep them in mind to "
                "ensure you do not repeat past mistakes.\n\n"
            )
            for round_num, issues in self._feedback_history:
                query += f"### Attempt {round_num}\n"
                for issue in issues:
                    query += f"- {issue}\n"

        return query

    def emit_history(
        self,
        output: LoopOutputModel,
        prior: Optional[LoopOutputModel],
        round_num: int,
        node: str,
    ) -> HistoryEntry:
        return HistoryEntry(
            round=round_num, node=node, changes_summary="Extracted conceptual model"
        )


class ERAuditorLoopAgent(LoopAgent):
    def __init__(self, facts: List[AtomicFact], model: Optional[str] = None):
        self._facts = facts
        self._model = model
        self.agent = get_conceptual_verifier_agent(model)

    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        from src.util.core.invoke import get_response

        parsed, tokens = await get_response(self.agent, ConceptualCritiqueReport, query)
        return parsed, tokens

    def build_context(self, ctx: LoopContext) -> str:
        extracted = ctx.node_outputs.get("extractor")
        facts_str = "\n".join([f"- [{f.id}] {f.fact}" for f in self._facts])
        if isinstance(extracted, ConceptualModel):
            return f"## INPUT\nFacts:\n{facts_str}\n\nGenerated Conceptual Model:\n{extracted.model_dump_json(indent=2)}"
        return f"## INPUT\nFacts:\n{facts_str}\n\nGenerated Conceptual Model:\n{{}}"

    def emit_history(
        self,
        output: LoopOutputModel,
        prior: Optional[LoopOutputModel],
        round_num: int,
        node: str,
    ) -> HistoryEntry:
        assert isinstance(output, ConceptualCritiqueReport)
        changes_summary = (
            "valid" if output.is_valid else f"{len(output.fixes)} issues found"
        )
        return HistoryEntry(round=round_num, node=node, changes_summary=changes_summary)


# ---------------------------------------------------------------------------
# Runner functions
# ---------------------------------------------------------------------------


async def run_er_extractor_loop(
    facts: List[AtomicFact],
    nl_query: str,
    max_retries: int = 12,
    model: Optional[str] = None,
) -> Tuple[ConceptualModel, List[FixHistoryStep], int]:
    from src.pipeline.stage2.middleware.conceptual_filter_node import (
        ConceptualFilterLoopAgent,
    )

    extractor = ERExtractorLoopAgent(facts, nl_query, model)
    auditor = ERAuditorLoopAgent(facts, model)
    filter_node = ConceptualFilterLoopAgent()

    config = LoopConfig(
        agents={
            "extractor": AgentRoleConfig(
                agent_factory=lambda: extractor, det_error_sources=["filter"]
            ),
            "filter": AgentRoleConfig(agent_factory=lambda: filter_node),
            "auditor": AgentRoleConfig(agent_factory=lambda: auditor),
        },
        graph={
            "edges": [
                GraphEdge(from_node="extractor", to_node="filter"),
                GraphEdge(
                    from_node="filter",
                    to_node="extractor",
                    condition=EdgeCondition(field="is_valid", op="eq", value=False),
                ),
                GraphEdge(from_node="filter", to_node="auditor"),
                GraphEdge(
                    from_node="auditor",
                    to_node="extractor",
                    condition=EdgeCondition(field="is_valid", op="eq", value=False),
                ),
                GraphEdge(from_node="auditor", to_node="end"),
            ]
        },
        start_node="extractor",
        max_iter=max_retries,
        error_refresh=ErrorRefreshConfig(trigger_node="extractor"),
    )

    result = await AgentLoop(config).run("")
    output = result.node_outputs.get("extractor")
    if not isinstance(output, ConceptualModel):
        output = ConceptualModel(
            entities=[], relationships=[], functional_dependencies=[]
        )

    return output, [], result.total_tokens
