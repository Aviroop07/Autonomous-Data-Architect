"""Stage 3 orchestration entry point.

Runs per-shard extraction (3 agent families in parallel), merges
constraints, and runs deterministic DOF/conflict analysis.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage3.agents.logic_extractor.agent import LogicExtractorLoopAgent
from src.pipeline.stage3.agents.statistical_extractor.agent import (
    StatisticalExtractorLoopAgent,
)
from src.pipeline.stage3.agents.structural_extractor.agent import (
    StructuralExtractorLoopAgent,
)
from src.pipeline.stage3.middleware.constraint_graph import (
    analyze_cross_shard_constraints,
)
from src.pipeline.stage3.middleware.fact_allocation import (
    allocate_facts_to_shards,
    find_mentioned_tables,
)
from src.pipeline.stage3.models.cross_shard import (
    Constraint,
    DerivedColumnConstraint,
    DistributionConstraint,
    LogicExtractionOutput,
    StatisticalExtractionOutput,
    StructuralExtractionOutput,
)
from src.pipeline.stage3.models.probe import Stage3AnalysisReport
from src.pipeline.stage2.models.registry import TableFactRegistry
from src.util.config.ablation import AblationConfig
from src.util.orchestration.loop import AgentLoop
from src.util.orchestration.loop_types import (
    AgentRoleConfig,
    GraphEdge,
    LoopConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


class Stage3Output(BaseModel):
    """Complete Stage 3 output: extracted constraints + DOF analysis."""

    distributions: List[DistributionConstraint] = Field(default_factory=list)
    moment_targets: List[Constraint] = Field(default_factory=list)
    correlations: List[Constraint] = Field(default_factory=list)
    structural_constraints: List[Constraint] = Field(default_factory=list)
    logic_constraints: List[Constraint] = Field(default_factory=list)
    derived_columns: List[DerivedColumnConstraint] = Field(default_factory=list)
    analysis_report: Stage3AnalysisReport = Field(default_factory=Stage3AnalysisReport)
    token_usage: int = 0

    @property
    def total_constraints(self) -> int:
        return (
            len(self.distributions)
            + len(self.moment_targets)
            + len(self.correlations)
            + len(self.structural_constraints)
            + len(self.logic_constraints)
            + len(self.derived_columns)
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_registry(facts: List[AtomicFact], schema: Schema) -> TableFactRegistry:
    """Build a TableFactRegistry by scanning fact text for table mentions."""
    registry = TableFactRegistry()
    table_names = [t.name for t in schema.tables]
    for fact in facts:
        mentioned = find_mentioned_tables(fact.fact, table_names)
        for table_name in mentioned:
            registry.register_table_facts(table_name, [fact.id])
    return registry


def _build_shard_table_sets(shards: List[Schema]) -> List[set[str]]:
    """Extract table name sets from each schema shard."""
    return [{t.name for t in shard.tables} for shard in shards]


def _serialize_context(
    shard: Schema,
    fact_ids: List[int],
    facts_map: Dict[int, AtomicFact],
    stub_tables: List[str],
) -> str:
    """JSON-serialize the context for an extraction agent's initial_context."""
    # Convert Schema to dict for JSON serialization
    schema_dict = json.loads(shard.model_dump_json())
    return json.dumps(
        {
            "schema": schema_dict,
            "fact_ids": fact_ids,
            "facts_map": {
                str(fid): {"id": fid, "fact": facts_map[fid].fact}
                for fid in fact_ids
                if fid in facts_map
            },
            "stub_tables": stub_tables,
        }
    )


async def _run_shard_agents(
    shard: Schema,
    fact_ids: List[int],
    facts_map: Dict[int, AtomicFact],
    stub_tables: List[str],
    model: Optional[str] = None,
    max_retries: int = 5,
) -> Tuple[
    StatisticalExtractionOutput,
    StructuralExtractionOutput,
    LogicExtractionOutput,
    int,
]:
    """Run all 3 extraction agents for one shard sequentially (each gets
    its own LoopConfig with retry).

    Returns merged per-shard outputs and token count.
    """
    context_str = _serialize_context(shard, fact_ids, facts_map, stub_tables)
    total_tokens = 0

    async def _run_one(
        agent_cls: type,
        node_name: str,
        output_type: type,
    ) -> Tuple[Any, int]:
        agent = agent_cls(model=model)
        config = LoopConfig(
            agents={
                node_name: AgentRoleConfig(
                    agent_factory=lambda: agent,
                ),
            },
            graph={"edges": [GraphEdge(from_node=node_name, to_node="end")]},
            start_node=node_name,
            max_iter=max_retries,
        )
        result = await AgentLoop(config).run(context_str)
        return result.node_outputs.get(node_name), result.total_tokens

    (
        (stat_output, stat_tokens),
        (struct_output, struct_tokens),
        (
            logic_output,
            logic_tokens,
        ),
    ) = await asyncio.gather(
        _run_one(
            StatisticalExtractorLoopAgent,
            "statistical_extractor",
            StatisticalExtractionOutput,
        ),
        _run_one(
            StructuralExtractorLoopAgent,
            "structural_extractor",
            StructuralExtractionOutput,
        ),
        _run_one(LogicExtractorLoopAgent, "logic_extractor", LogicExtractionOutput),
    )

    total_tokens = stat_tokens + struct_tokens + logic_tokens

    # Normalize outputs
    if not isinstance(stat_output, StatisticalExtractionOutput):
        stat_output = StatisticalExtractionOutput()
    if not isinstance(struct_output, StructuralExtractionOutput):
        struct_output = StructuralExtractionOutput()
    if not isinstance(logic_output, LogicExtractionOutput):
        logic_output = LogicExtractionOutput()

    return stat_output, struct_output, logic_output, total_tokens


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def orchestrate(
    schema: Schema,
    shards: List[Schema],
    facts: List[AtomicFact],
    model: Optional[str] = None,
    ablation_config: Optional[AblationConfig] = None,
    max_retries: int = 5,
) -> Tuple[Stage3Output, int]:
    """Run Stage 3: extraction -> merge -> DOF analysis.

    Args:
        schema: Full merged schema (from Stage 2).
        shards: Schema shards (from Stage 2 segments).
        facts: Atomic facts (from Stage 1).
        model: LLM model name (None = default).
        ablation_config: Ablation settings (None = no ablation).
        max_retries: Max retry iterations per extraction agent.

    Returns:
        (Stage3Output, total_tokens)
    """
    logger.info(
        f"[Stage 3] Starting extraction: {len(shards)} shards, {len(facts)} facts."
    )

    # 1. Build registry and allocate facts to shards
    registry = _build_registry(facts, schema)
    shard_table_sets = _build_shard_table_sets(shards)
    facts_map = {f.id: f for f in facts}
    allocations = allocate_facts_to_shards(facts, shard_table_sets, registry)

    logger.info(
        f"[Stage 3] Fact allocation complete: "
        + ", ".join(
            f"shard {i}: {len(a.fact_ids)} facts, {len(a.stub_tables)} stubs"
            for i, a in enumerate(allocations)
        )
    )

    # 2. Run extraction agents per shard in parallel
    shard_tasks = []
    for i, (shard, allocation) in enumerate(zip(shards, allocations)):
        shard_tasks.append(
            _run_shard_agents(
                shard=shard,
                fact_ids=allocation.fact_ids,
                facts_map=facts_map,
                stub_tables=allocation.stub_tables,
                model=model,
                max_retries=max_retries,
            )
        )

    shard_results = await asyncio.gather(*shard_tasks)

    # 3. Merge per-shard outputs
    all_distributions: List[DistributionConstraint] = []
    all_moment_targets: List[Constraint] = []
    all_correlations: List[Constraint] = []
    all_structural: List[Constraint] = []
    all_logic: List[Constraint] = []
    all_derived: List[DerivedColumnConstraint] = []
    total_tokens = 0

    for stat_output, struct_output, logic_output, tokens in shard_results:
        all_distributions.extend(stat_output.distributions)
        all_moment_targets.extend(stat_output.moment_targets)
        all_correlations.extend(stat_output.correlations)
        all_structural.extend(struct_output.constraints)
        all_logic.extend(logic_output.constraints)
        total_tokens += tokens

    logger.info(
        f"[Stage 3] Merged constraints: {len(all_distributions)} distributions, "
        f"{len(all_moment_targets)} moments, {len(all_correlations)} correlations, "
        f"{len(all_structural)} structural, {len(all_logic)} logic."
    )

    # 4. Run DOF/conflict analysis
    analysis_report = analyze_cross_shard_constraints(
        distributions=all_distributions,
        structural=all_structural,
        logic=all_logic,
        derived=all_derived,
        schema=schema,
    )

    logger.info(
        f"[Stage 3] DOF analysis: {len(analysis_report.square_variables)} square, "
        f"{len(analysis_report.loose_variable_probes)} loose, "
        f"{len(analysis_report.overconstrained_blocks)} overconstrained."
    )

    output = Stage3Output(
        distributions=all_distributions,
        moment_targets=all_moment_targets,
        correlations=all_correlations,
        structural_constraints=all_structural,
        logic_constraints=all_logic,
        derived_columns=all_derived,
        analysis_report=analysis_report,
        token_usage=total_tokens,
    )

    return output, total_tokens
