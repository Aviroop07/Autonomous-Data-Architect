"""Stage 1 AgentLoop configurations."""

from __future__ import annotations

from typing import List, Optional

from src.pipeline.stage1.agents.context_auditor.agent import ContextAuditorLoopAgent
from src.pipeline.stage1.agents.context_enricher.agent import ContextEnricherLoopAgent
from src.pipeline.stage1.agents.fact_extractor.agent import FactExtractorLoopAgent
from src.pipeline.stage1.agents.verifier.agent import VerifierLoopAgent
from src.pipeline.stage1.models.raw_fact import RawFact
from src.pipeline.stage1.models.coverage_report import SpecGap
from src.util.core.search_tool import EvidenceStore
from src.util.orchestration.loop_types import (
    AgentRoleConfig,
    EdgeCondition,
    GraphEdge,
    LoopConfig,
)


def make_stage1_loop_config(
    nl_description: str,
    model: Optional[str] = None,
) -> LoopConfig:
    """Build the AgentLoop config for Stage 1 extraction + verification."""
    return LoopConfig(
        agents={
            "extractor": AgentRoleConfig(
                agent_factory=lambda: FactExtractorLoopAgent(model=model),
            ),
            "verifier": AgentRoleConfig(
                agent_factory=lambda: VerifierLoopAgent(model=model),
            ),
        },
        graph={
            "edges": [
                GraphEdge(from_node="extractor", to_node="verifier"),
                GraphEdge(
                    from_node="verifier",
                    to_node="extractor",
                    condition=EdgeCondition(field="is_safe", op="eq", value=False),
                ),
                GraphEdge(from_node="verifier", to_node="end"),
            ]
        },
        start_node="extractor",
        max_iter=5,
    )


def make_enrichment_loop_config(
    original_facts: List[RawFact],
    gaps: List[SpecGap],
    model: Optional[str] = None,
) -> tuple[LoopConfig, ContextEnricherLoopAgent, ContextAuditorLoopAgent]:
    """Build the AgentLoop config for context enrichment + auditing.

    Returns the config plus the two agent instances so callers can read
    accumulated_accepted and audit_trail after the loop completes.
    """
    evidence_store = EvidenceStore()
    
    enricher = ContextEnricherLoopAgent(
        original_facts=original_facts,
        gaps=gaps,
        evidence_store=evidence_store,
        model=model,
    )
    auditor = ContextAuditorLoopAgent(
        original_facts=original_facts,
        gaps=gaps,
        evidence_store=evidence_store,
        model=model,
    )

    config = LoopConfig(
        agents={
            "enricher": AgentRoleConfig(agent_factory=lambda: enricher),
            "auditor": AgentRoleConfig(agent_factory=lambda: auditor),
        },
        graph={
            "edges": [
                GraphEdge(from_node="enricher", to_node="auditor"),
                GraphEdge(
                    from_node="auditor",
                    to_node="enricher",
                    condition=EdgeCondition(
                        field="is_acceptable", op="eq", value=False
                    ),
                ),
                GraphEdge(from_node="auditor", to_node="end"),
            ]
        },
        start_node="enricher",
        max_iter=5,
    )
    return config, enricher, auditor
