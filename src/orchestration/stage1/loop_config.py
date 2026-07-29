"""Stage 1 AgentLoop configurations."""

from __future__ import annotations

from typing import List, Optional

from src.pipeline.stage1.agents.context_auditor.agent import ContextAuditorLoopAgent
from src.pipeline.stage1.agents.context_enricher.agent import ContextEnricherLoopAgent
from src.pipeline.stage1.agents.fact_extractor.agent import FactExtractorLoopAgent
from src.pipeline.stage1.agents.verifier.agent import VerifierLoopAgent
from src.pipeline.stage1.middleware.context_filter_node import ContextFilterLoopAgent
from src.pipeline.stage1.middleware.extraction_validator_node import (
    ExtractionValidatorLoopAgent,
)
from src.pipeline.stage1.models.raw_fact import RawFact
from src.pipeline.stage1.models.coverage_report import SpecGap
from src.util.core.agent_provider import AgentProvider
from src.util.core.search_tool import EvidenceStore
from src.util.orchestration.loop_types import (
    AgentRoleConfig,
    EdgeCondition,
    ErrorRefreshConfig,
    GraphEdge,
    LoopConfig,
)
from src.util.orchestration.rounds import rounds_to_max_iter

# Both Stage 1 loops are three-node graphs and both allow three full rounds, so
# each ends up at a raw max_iter of 9. These were two bare 9s; naming them means
# the round policy and the node count are stated separately and the arithmetic is
# the shared helper's, which is the only place in the project that knows
# AgentLoop spends its budget per NODE EXECUTION rather than per pass.
EXTRACTION_GRAPH_NODE_COUNT = 3
EXTRACTION_ROUNDS = 3

# enricher -> auditor -> filter. A structural-only retry skips the auditor and so
# costs 2 rather than 3, which is why this budget affords "~3" refinement cycles
# rather than exactly 3.
ENRICHMENT_GRAPH_NODE_COUNT = 3
ENRICHMENT_ROUNDS = 3


def make_stage1_loop_config(
    nl_description: str,
    model: Optional[str] = None,
    provider: Optional[AgentProvider] = None,
) -> LoopConfig:
    """Build the AgentLoop config for Stage 1 extraction + verification."""

    extractor = FactExtractorLoopAgent(model=model, provider=provider)
    extraction_validator = ExtractionValidatorLoopAgent()
    verifier = VerifierLoopAgent(model=model, provider=provider)

    return LoopConfig(
        agents={
            "extractor": AgentRoleConfig(
                agent_factory=lambda: extractor, det_error_sources=["validator"]
            ),
            "validator": AgentRoleConfig(agent_factory=lambda: extraction_validator),
            "verifier": AgentRoleConfig(
                agent_factory=lambda: verifier,
            ),
        },
        graph={
            "edges": [
                GraphEdge(from_node="extractor", to_node="validator"),
                GraphEdge(
                    from_node="validator",
                    to_node="extractor",
                    condition=EdgeCondition(field="is_clean", op="eq", value=False),
                ),
                GraphEdge(from_node="validator", to_node="verifier"),
                GraphEdge(
                    from_node="verifier",
                    to_node="extractor",
                    condition=EdgeCondition(field="is_safe", op="eq", value=False),
                ),
                GraphEdge(from_node="verifier", to_node="end"),
            ]
        },
        start_node="extractor",
        # extractor -> validator -> verifier, so a full audited round is 3 node
        # executions. Expressed in rounds and converted, rather than as a bare
        # 9, so the number cannot silently stop matching the graph if a node is
        # added -- the failure mode would be a loop that ends PART WAY through a
        # round and returns an unaudited model.
        max_iter=rounds_to_max_iter(EXTRACTION_ROUNDS, EXTRACTION_GRAPH_NODE_COUNT),
        error_refresh=ErrorRefreshConfig(trigger_node="extractor"),
    )


def make_enrichment_loop_config(
    original_facts: List[RawFact],
    gaps: List[SpecGap],
    model: Optional[str] = None,
    provider: Optional[AgentProvider] = None,
    evidence_store: Optional[EvidenceStore] = None,
) -> tuple[
    LoopConfig,
    ContextEnricherLoopAgent,
    ContextAuditorLoopAgent,
    ContextFilterLoopAgent,
]:
    """Build the AgentLoop config for context enrichment + auditing + filtering.

    Returns the config plus the three agent instances so callers can read
    accumulated_accepted and audit_trail after the loop completes.

    Topology (loop exits from the deterministic filter):

        enricher --[needs_audit]--> auditor  (fallback)--> filter
        auditor  ----------------------------------------> filter
        filter   --[not should_exit]--> enricher (fallback)--> end

    On a structural-only retry (last auditor already acceptable, bounced back
    only to fix a filter-flagged reference) the enricher skips both web search
    and the auditor: enricher -> filter directly. The filter's invalid-reference
    errors are routed back to the enricher via det_error_sources.
    """
    # Constructed here unless the caller supplies one. The store is the ONLY
    # web-facing dependency in Stage 1, and it accumulates per-run evidence
    # tags the auditor resolves against, so a caller that needs to control or
    # observe what evidence a run saw (an offline reproduction, a cached
    # replay) has to be able to hand one in rather than have it minted here.
    if evidence_store is None:
        evidence_store = EvidenceStore()

    enricher = ContextEnricherLoopAgent(
        original_facts=original_facts,
        gaps=gaps,
        evidence_store=evidence_store,
        model=model,
        provider=provider,
    )
    auditor = ContextAuditorLoopAgent(
        original_facts=original_facts,
        gaps=gaps,
        evidence_store=evidence_store,
        model=model,
        provider=provider,
    )
    context_filter = ContextFilterLoopAgent(original_facts=original_facts)

    config = LoopConfig(
        agents={
            # The filter's invalid-reference errors flow into the enricher's
            # ctx.det_errors so it can re-anchor them next round.
            "enricher": AgentRoleConfig(
                agent_factory=lambda: enricher, det_error_sources=["filter"]
            ),
            "auditor": AgentRoleConfig(agent_factory=lambda: auditor),
            "filter": AgentRoleConfig(agent_factory=lambda: context_filter),
        },
        graph={
            "edges": [
                # Skip the auditor on structural-only retries (needs_audit False).
                GraphEdge(
                    from_node="enricher",
                    to_node="auditor",
                    condition=EdgeCondition(fn=lambda _out: enricher.needs_audit),
                ),
                GraphEdge(from_node="enricher", to_node="filter"),
                GraphEdge(from_node="auditor", to_node="filter"),
                # Loop back to the enricher unless the filter says we can exit.
                GraphEdge(
                    from_node="filter",
                    to_node="enricher",
                    condition=EdgeCondition(
                        fn=lambda out: not getattr(out, "should_exit", False)
                    ),
                ),
                GraphEdge(from_node="filter", to_node="end"),
            ]
        },
        start_node="enricher",
        # Node-executions, not cycles: a full audited cycle is 3 nodes, so this
        # allows ~3 refinement cycles (structural-only rounds cost only 2).
        max_iter=rounds_to_max_iter(ENRICHMENT_ROUNDS, ENRICHMENT_GRAPH_NODE_COUNT),
        error_refresh=ErrorRefreshConfig(trigger_node="enricher"),
    )
    return config, enricher, auditor, context_filter
