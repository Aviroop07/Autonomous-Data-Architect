import logging
from typing import List, Optional, Tuple


from src.pipeline.stage1.models.rephrased_nl import AtomicFact

from src.pipeline.stage2.models.corrections import FixHistoryStep
from src.util.orchestration.loop import AgentLoop
from src.util.orchestration.rounds import rounds_to_max_iter
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
    NodeOutputRecord,
)

from src.pipeline.stage2.agents.er_extractor.agent import (
    get_agent as get_conceptual_extractor_agent,
)
from src.pipeline.stage2.agents.er_auditor.agent import (
    get_agent as get_conceptual_verifier_agent,
)
from src.pipeline.stage2.mapper.conceptual_model import ConceptualModel
from src.pipeline.stage2.models.conceptual_critique import ConceptualCritiqueReport

logger = logging.getLogger(__name__)


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

        # Harvest deterministic errors from the Filter. Advisories are harvested
        # unconditionally: they never fail a model, so waiting for is_valid=False
        # to surface them would mean a correct-but-improvable draft never heard
        # about them at all.
        filter_report = ctx.node_outputs.get("filter")
        if filter_report is not None:
            if not filter_report.is_valid:
                current_round_feedback.extend(getattr(filter_report, "det_errors", []))
            current_round_feedback.extend(getattr(filter_report, "advisories", []))

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


# The shard graph is extractor -> filter -> auditor, and AgentLoop spends its
# budget once per NODE EXECUTION rather than once per pass. So a raw max_iter
# must be a multiple of this or the loop stops PART WAY through a round, leaving
# the returned model unaudited. Callers were passing 5, which buys one full pass
# plus two nodes of a second -- the extractor got the auditor's fixes once, but
# nothing ever re-checked the corrected model. Same defect class as Stage 3's
# Phase 1 loop, which had exactly this and could not retry at all.
#
# rounds * 3 is exact only while every pass reaches the auditor. A HARD filter
# rejection still routes back to the extractor and makes that pass two nodes
# long, so the conversion remains an upper bound on rounds rather than a promise
# of them. It used to be a much weaker bound: soft FK-naming advice also failed
# the model, so passes that had nothing structurally wrong with them ended at
# the filter too. Those no longer cost a round, which is why the audit is now
# reached in the common case at all.
SHARD_GRAPH_NODE_COUNT = 3


def shard_rounds_to_max_iter(rounds: int) -> int:
    """Convert "N audited rounds" into the raw per-node budget AgentLoop counts."""
    return rounds_to_max_iter(rounds, SHARD_GRAPH_NODE_COUNT)


def select_best_shard_model(
    trace: List[NodeOutputRecord],
) -> Tuple[Optional[ConceptualModel], str]:
    """Pick the extractor draft with the best MEASURED verdict, not the last one.

    The loop returns whatever the extractor produced most recently, and when the
    budget runs out mid-round that draft is by construction one no reviewer ever
    saw. A live run made the cost of that concrete: the auditor's finding count
    went 5 -> 8 -> 5 across three audits, so the drafts were not monotonically
    improving, and the model that shipped was a fourth draft with no verdict at
    all. Returning the last draft is then strictly a gamble.

    So score each draft by the reviews that actually followed it:
      - a draft the deterministic filter rejected is disqualified outright, since
        those errors are structural rather than advisory;
      - otherwise rank by how many findings the next audit raised, fewest first;
      - break ties toward the later draft, which has seen more feedback.
    A draft no reviewer reached scores as unmeasured and is used only when no
    measured draft exists at all -- which is also the single-round case, where
    this reduces to the previous behavior.
    """
    best: Optional[ConceptualModel] = None
    best_findings: Optional[int] = None
    last_unmeasured: Optional[ConceptualModel] = None
    last_any: Optional[ConceptualModel] = None

    for idx, record in enumerate(trace):
        if record.node != "extractor" or not isinstance(record.output, ConceptualModel):
            continue
        candidate = record.output
        last_any = candidate

        findings: Optional[int] = None
        disqualified = False
        # Reviews that belong to this draft are the ones before the extractor runs again.
        for later in trace[idx + 1 :]:
            if later.node == "extractor":
                break
            if later.node == "filter" and not getattr(later.output, "is_valid", True):
                disqualified = True
                break
            if isinstance(later.output, ConceptualCritiqueReport):
                findings = 0 if later.output.is_valid else len(later.output.fixes)
                break

        if disqualified:
            continue
        if findings is None:
            last_unmeasured = candidate
            continue
        if best_findings is None or findings <= best_findings:
            best, best_findings = candidate, findings

    if best is not None:
        return best, f"best audited draft, {best_findings} finding(s)"
    if last_unmeasured is None and last_any is not None:
        # Every draft was rejected by the filter. Returning nothing here would
        # lose the shard's whole contribution silently, which is worse than
        # handing on a structurally flawed model: validation downstream reports
        # the flaw loudly, and a caller can still see what was attempted.
        return last_any, "every draft failed the structural filter; using the last one"
    if last_unmeasured is not None:
        return last_unmeasured, "no draft was ever audited; using the last one"
    return None, "no usable draft was produced"


async def run_er_extractor_loop(
    facts: List[AtomicFact],
    nl_query: str,
    max_retries: int = 4,
    model: Optional[str] = None,
) -> Tuple[ConceptualModel, List[FixHistoryStep], int]:
    from src.pipeline.stage2.middleware.conceptual_filter_node import (
        ConceptualFilterLoopAgent,
    )

    extractor = ERExtractorLoopAgent(facts, nl_query, model)
    auditor = ERAuditorLoopAgent(facts, model)
    filter_node = ConceptualFilterLoopAgent([f.id for f in facts])

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
        max_iter=shard_rounds_to_max_iter(max_retries),
        error_refresh=ErrorRefreshConfig(trigger_node="extractor"),
    )

    result = await AgentLoop(config).run("")

    # The shard loop used to run completely silently: no record of what the
    # auditor said, how many rounds happened, or whether the returned model had
    # ever passed an audit. That made an ineffective audit indistinguishable from
    # a clean one -- and a live investigation into a lost entity stalled for
    # exactly that reason, since nothing on disk or in the log said whether the
    # auditor had flagged it.
    for entry in result.history:
        logger.info(
            "  [Stage 2] shard round %s: %s -> %s",
            entry.round,
            entry.node,
            entry.changes_summary,
        )
    final_audit = result.node_outputs.get("auditor")
    if isinstance(final_audit, ConceptualCritiqueReport):
        if not final_audit.is_valid:
            logger.warning(
                "  [Stage 2] shard model returned with %d UNRESOLVED auditor "
                "finding(s) after %d node execution(s); the loop ran out of budget "
                "before they were fixed. First: %s",
                len(final_audit.fixes),
                result.iteration_count,
                (final_audit.fixes[0].description[:200] if final_audit.fixes else "-"),
            )
        for fix in final_audit.fixes:
            logger.info("  [Stage 2] auditor finding: %s", fix.description[:220])
    else:
        logger.warning(
            "  [Stage 2] shard model was returned WITHOUT a final audit -- the loop "
            "ended on '%s' after %d node execution(s), so nothing verified it.",
            result.final_node,
            result.iteration_count,
        )

    output, reason = select_best_shard_model(result.output_trace)
    logger.info("  [Stage 2] shard model selected: %s", reason)
    if output is None:
        output = ConceptualModel(
            entities=[], relationships=[], functional_dependencies=[]
        )

    return output, [], result.total_tokens
