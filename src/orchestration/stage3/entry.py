"""Stage 3 orchestration entry point.

Implements the 3-tier Stage 3 data flow (see PROGRESS.md for the design
discussion this followed):

  Tier 1: per-family, per-shard extractor+auditor retry loop -- each of the
          3 families (statistical/structural/logic) gets its own 2-node
          AgentLoop (extractor <-> auditor) scoped to one schema shard's
          own facts.
  Tier 2: per-shard reconciliation -- merge one shard's 3 families' outputs,
          run analyze_cross_shard_constraints scoped to that shard's own
          Schema, and reconcile any detected conflicts: MISEXTRACTION
          re-runs the specific family's loop with injected guidance,
          FALSE_POSITIVE drops the conflict (recorded, not silently
          discarded), GENUINE_CONTRADICTION is left in place. Runs for up
          to max_reconciliation_rounds per shard.
  Tier 3: global reconciliation -- once every shard's Tier 2 pass is done,
          merge ALL shards' finalized constraints and run ONE global
          analyze_cross_shard_constraints call, reconciled the same way.
          This produces the Stage3AnalysisReport that Stage 4 consumes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage3.agents.conflict_reconciler.agent import reconcile_conflict
from src.pipeline.stage3.agents.logic_auditor.agent import LogicAuditorLoopAgent
from src.pipeline.stage3.agents.logic_extractor.agent import LogicExtractorLoopAgent
from src.pipeline.stage3.agents.statistical_auditor.agent import (
    StatisticalAuditorLoopAgent,
)
from src.pipeline.stage3.agents.statistical_extractor.agent import (
    StatisticalExtractorLoopAgent,
)
from src.pipeline.stage3.agents.structural_auditor.agent import (
    StructuralAuditorLoopAgent,
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
from src.pipeline.stage3.models.probe import (
    DismissedConflict,
    MisextractionFix,
    ReconciliationVerdict,
    Stage3AnalysisReport,
)
from src.pipeline.stage2.models.registry import TableFactRegistry
from src.util.config.ablation import AblationConfig
from src.util.orchestration.loop import AgentLoop
from src.util.orchestration.loop_types import (
    AgentRoleConfig,
    EdgeCondition,
    GraphEdge,
    LoopConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


class Stage3Output(BaseModel):
    """Complete Stage 3 output: extracted constraints + global DOF analysis."""

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
# Family registry -- ties each family name to its extractor/auditor classes
# and the node names used in its 2-node LoopConfig graph.
# ---------------------------------------------------------------------------

_FAMILIES: Dict[str, Dict[str, Any]] = {
    "statistical": {
        "extractor_cls": StatisticalExtractorLoopAgent,
        "auditor_cls": StatisticalAuditorLoopAgent,
        "extractor_node": "statistical_extractor",
        "auditor_node": "statistical_auditor",
        "output_type": StatisticalExtractionOutput,
    },
    "structural": {
        "extractor_cls": StructuralExtractorLoopAgent,
        "auditor_cls": StructuralAuditorLoopAgent,
        "extractor_node": "structural_extractor",
        "auditor_node": "structural_auditor",
        "output_type": StructuralExtractionOutput,
    },
    "logic": {
        "extractor_cls": LogicExtractorLoopAgent,
        "auditor_cls": LogicAuditorLoopAgent,
        "extractor_node": "logic_extractor",
        "auditor_node": "logic_auditor",
        "output_type": LogicExtractionOutput,
    },
}


# ---------------------------------------------------------------------------
# Shard state -- mutable per-shard bookkeeping threaded through all 3 tiers
# ---------------------------------------------------------------------------


@dataclass
class _ShardState:
    index: int
    schema: Schema
    fact_ids: List[int]
    stub_tables: List[str]
    outputs: Dict[str, Any] = field(default_factory=dict)  # family -> output
    tokens: int = 0


@dataclass
class _ConflictItem:
    conflict_ref: str
    description: str
    fact_ids: List[int]


# ---------------------------------------------------------------------------
# Text rendering helpers (schema / facts / constraints -> prompt-friendly text)
# ---------------------------------------------------------------------------


def _schema_to_text(schema: Schema, stub_tables: Optional[List[str]] = None) -> str:
    """Render schema shard + stub tables into prompt-friendly text."""
    lines: List[str] = ["## SCHEMA SHARD"]
    for table in schema.tables:
        lines.append(f"### {table.name}")
        lines.append(f"  Primary key: {', '.join(table.primary_key)}")
        for col in table.columns:
            nullable = "NULL" if col.is_nullable else "NOT NULL"
            lines.append(f"  {col.name}: {col.data_type} {nullable}")
        for fk in schema.relationships or []:
            if fk.referencing_table == table.name:
                lines.append(
                    f"  FK: {fk.referencing_column} -> "
                    f"{fk.referred_table} (its primary key)"
                )

    if stub_tables:
        lines.append("\n## STUB TABLES (cross-shard, schema-only)")
        for stub in stub_tables:
            lines.append(f"### {stub} (stub)")
            lines.append("  (columns not available -- use for ON-tree references only)")

    return "\n".join(lines)


def _facts_to_text(fact_ids: List[int], facts_map: Dict[int, AtomicFact]) -> str:
    """Render allocated facts into prompt-friendly text."""
    lines: List[str] = ["## FACTS"]
    for fid in sorted(fact_ids):
        fact = facts_map.get(fid)
        if fact is not None:
            lines.append(f"- [id={fid}] {fact.fact}")
    return "\n".join(lines)


def _render_involved_constraints(
    fact_ids: List[int],
    distributions: List[DistributionConstraint],
    moment_targets: List[Constraint],
    correlations: List[Constraint],
    structural: List[Constraint],
    logic: List[Constraint],
    derived: List[DerivedColumnConstraint],
) -> str:
    """Dump every extracted constraint that references any of fact_ids, so
    the reconciliation agent can see exactly what was extracted from the
    facts it's re-examining."""
    fact_id_set = set(fact_ids)

    def _matches(c: Any) -> bool:
        return any(fid in fact_id_set for fid in c.fact_references)

    lines: List[str] = []
    for label, items in (
        ("Distribution", distributions),
        ("MomentTarget", moment_targets),
        ("Correlation", correlations),
        ("Structural", structural),
        ("Logic", logic),
        ("Derived", derived),
    ):
        for c in items:
            if _matches(c):
                lines.append(f"[{label}] {c.model_dump_json()}")
    return (
        "\n".join(lines)
        if lines
        else "(no extracted constraints reference these facts)"
    )


# ---------------------------------------------------------------------------
# Registry / allocation helpers
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
    reconciliation_guidance: Optional[str] = None,
) -> str:
    """JSON-serialize the context for an extraction agent's initial_context."""
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
            "reconciliation_guidance": reconciliation_guidance or "",
        }
    )


# ---------------------------------------------------------------------------
# Tier 1: per-family extractor+auditor loop
# ---------------------------------------------------------------------------


async def _run_family_loop(
    family: str,
    context_str: str,
    model: Optional[str],
    max_retries: int,
) -> Tuple[Any, int]:
    """Run one family's extractor+auditor retry loop.

    Graph shape (mirrors Stage 1's extractor -> validator -> [back] ->
    verifier -> [back] -> end, collapsed to 2 nodes since the extractor
    validates its own ON-tree canonicalization internally via _det_errors
    rather than through a separate deterministic node):

        extractor --[has det_errors]--> extractor   (self-loop retry)
        extractor --------------------------------> auditor   (fallback)
        auditor   --[is_valid == False]-----------> extractor (retry)
        auditor   --------------------------------> end        (fallback)

    Edge order matters -- AgentLoop._evaluate_edges is first-match-wins, so
    each conditional edge must be listed before its node's unconditional
    fallback, or the fallback would always win.
    """
    spec = _FAMILIES[family]
    extractor = spec["extractor_cls"](model=model)
    auditor = spec["auditor_cls"](model=model)
    extractor_node = spec["extractor_node"]
    auditor_node = spec["auditor_node"]

    config = LoopConfig(
        agents={
            extractor_node: AgentRoleConfig(agent_factory=lambda: extractor),
            auditor_node: AgentRoleConfig(agent_factory=lambda: auditor),
        },
        graph={
            "edges": [
                GraphEdge(
                    from_node=extractor_node,
                    to_node=extractor_node,
                    condition=EdgeCondition(
                        fn=lambda o: bool(getattr(o, "_det_errors", []))
                    ),
                ),
                GraphEdge(from_node=extractor_node, to_node=auditor_node),
                GraphEdge(
                    from_node=auditor_node,
                    to_node=extractor_node,
                    condition=EdgeCondition(field="is_valid", op="eq", value=False),
                ),
                GraphEdge(from_node=auditor_node, to_node="end"),
            ]
        },
        start_node=extractor_node,
        # Node-executions, not cycles: one full audited round costs 2 node
        # executions (extractor -> auditor), plus up to 1 extra self-loop
        # execution if canonicalize() itself found errors that round. 3x
        # max_retries budgets ~max_retries full audited rounds even in the
        # worst case (mirrors Stage 1's loop_config.py max_iter reasoning).
        max_iter=max_retries * 3,
    )
    result = await AgentLoop(config).run(context_str)
    output = result.node_outputs.get(extractor_node)
    if not isinstance(output, spec["output_type"]):
        output = spec["output_type"]()
    return output, result.total_tokens


async def _run_shard_extraction(
    shard: Schema,
    fact_ids: List[int],
    facts_map: Dict[int, AtomicFact],
    stub_tables: List[str],
    model: Optional[str],
    max_retries: int,
) -> Tuple[Dict[str, Any], int]:
    """Run all 3 family loops for one shard in parallel (Tier 1 initial pass,
    no reconciliation guidance)."""
    families = list(_FAMILIES.keys())
    context_str = _serialize_context(shard, fact_ids, facts_map, stub_tables)
    results = await asyncio.gather(
        *[
            _run_family_loop(family, context_str, model, max_retries)
            for family in families
        ]
    )
    outputs = {family: output for family, (output, _tok) in zip(families, results)}
    total_tokens = sum(tok for _output, tok in results)
    return outputs, total_tokens


async def _rerun_single_family(
    shard: Schema,
    fact_ids: List[int],
    facts_map: Dict[int, AtomicFact],
    stub_tables: List[str],
    family: str,
    guidance: str,
    model: Optional[str],
    max_retries: int,
) -> Tuple[Any, int]:
    """Re-run exactly one family's loop with reconciliation guidance
    injected -- used when a MISEXTRACTION verdict pinpoints what that
    family got wrong."""
    context_str = _serialize_context(
        shard, fact_ids, facts_map, stub_tables, reconciliation_guidance=guidance
    )
    return await _run_family_loop(family, context_str, model, max_retries)


def _merge_all(
    shard_states: List[_ShardState],
) -> Tuple[
    List[DistributionConstraint],
    List[Constraint],
    List[Constraint],
    List[Constraint],
    List[Constraint],
    List[DerivedColumnConstraint],
]:
    """Merge every shard's current per-family outputs into flat lists."""
    distributions: List[DistributionConstraint] = []
    moment_targets: List[Constraint] = []
    correlations: List[Constraint] = []
    structural: List[Constraint] = []
    logic: List[Constraint] = []
    derived: List[DerivedColumnConstraint] = []
    for ss in shard_states:
        stat = ss.outputs["statistical"]
        struct = ss.outputs["structural"]
        lg = ss.outputs["logic"]
        distributions.extend(stat.distributions)
        moment_targets.extend(stat.moment_targets)
        correlations.extend(stat.correlations)
        structural.extend(struct.constraints)
        logic.extend(lg.constraints)
        derived.extend(lg.derived)
    return distributions, moment_targets, correlations, structural, logic, derived


# ---------------------------------------------------------------------------
# Tier 2 / Tier 3: shared conflict-reconciliation loop
# ---------------------------------------------------------------------------


def _extract_conflict_items(
    report: Stage3AnalysisReport, variable_fact_map: Dict[str, List[int]]
) -> List[_ConflictItem]:
    """Turn a Stage3AnalysisReport's overconstrained_blocks (structural
    over-determination AND confirmed value-level conflicts, both routed
    through the same field -- see build_and_classify) and
    derived_cycle_conflicts into a uniform list for reconciliation."""
    items: List[_ConflictItem] = []
    for block in report.overconstrained_blocks:
        fact_ids = sorted(
            {fid for v in block.variables for fid in variable_fact_map.get(v, [])}
        )
        items.append(
            _ConflictItem(
                conflict_ref="|".join(block.variables),
                description=(
                    f"Overconstrained/contradictory variable group: "
                    f"{block.variables} (constraints: {block.constraints})"
                ),
                fact_ids=fact_ids,
            )
        )
    for cycle in report.derived_cycle_conflicts:
        items.append(
            _ConflictItem(
                conflict_ref=f"cycle::{cycle.description}",
                description=cycle.description,
                fact_ids=list(cycle.fact_references),
            )
        )
    return items


async def _reconcile_conflicts(
    items: List[_ConflictItem],
    facts_map: Dict[int, AtomicFact],
    schema_text: str,
    merged: Tuple[
        List[DistributionConstraint],
        List[Constraint],
        List[Constraint],
        List[Constraint],
        List[Constraint],
        List[DerivedColumnConstraint],
    ],
    model: Optional[str],
) -> Tuple[list, int]:
    """Call conflict_reconciler once per conflict item, in parallel."""
    distributions, moment_targets, correlations, structural, logic, derived = merged

    async def _one(item: _ConflictItem):
        involved_facts = _facts_to_text(item.fact_ids, facts_map)
        involved_constraints = _render_involved_constraints(
            item.fact_ids,
            distributions,
            moment_targets,
            correlations,
            structural,
            logic,
            derived,
        )
        verdict, tokens = await reconcile_conflict(
            conflict_ref=item.conflict_ref,
            conflict_description=item.description,
            involved_facts=involved_facts,
            involved_constraints=involved_constraints,
            schema_context=schema_text,
            model=model,
        )
        return verdict, tokens

    results = await asyncio.gather(*[_one(item) for item in items])
    verdicts = [v for v, _t in results]
    tokens = sum(t for _v, t in results)
    return verdicts, tokens


async def _apply_misextraction_fixes(
    fixes_by_family: Dict[str, List[MisextractionFix]],
    shard_states: List[_ShardState],
    fact_to_shard: Dict[int, int],
    facts_map: Dict[int, AtomicFact],
    model: Optional[str],
    max_retries: int,
) -> int:
    """Group fixes by (owning shard, family), then re-run exactly the
    affected family loops with the fix guidance injected. Mutates
    shard_states[*].outputs in place. Returns tokens spent."""
    grouped: Dict[Tuple[int, str], List[MisextractionFix]] = {}
    for family, fixes in fixes_by_family.items():
        for fix in fixes:
            shard_idx = fact_to_shard.get(fix.fact_id)
            if shard_idx is None:
                logger.warning(
                    f"[Stage 3] MisextractionFix references unknown fact_id "
                    f"{fix.fact_id} (family={family}) -- skipping."
                )
                continue
            grouped.setdefault((shard_idx, family), []).append(fix)

    if not grouped:
        return 0

    keys = list(grouped.keys())
    tasks = []
    for shard_idx, family in keys:
        ss = shard_states[shard_idx]
        fixes = grouped[(shard_idx, family)]
        guidance = "\n".join(f"- fact {f.fact_id}: {f.guidance}" for f in fixes)
        tasks.append(
            _rerun_single_family(
                ss.schema,
                ss.fact_ids,
                facts_map,
                ss.stub_tables,
                family,
                guidance,
                model,
                max_retries,
            )
        )

    results = await asyncio.gather(*tasks)
    total_tokens = 0
    for (shard_idx, family), (output, tokens) in zip(keys, results):
        shard_states[shard_idx].outputs[family] = output
        shard_states[shard_idx].tokens += tokens
        total_tokens += tokens
    return total_tokens


def _filter_dismissed(
    report: Stage3AnalysisReport, dismissed_refs: set[str]
) -> Stage3AnalysisReport:
    """Remove any conflict already judged FALSE_POSITIVE from the report's
    live conflict fields -- dismissal is tracked separately via
    dismissed_conflicts, not silently left duplicated in the active lists."""
    kept_blocks = [
        b
        for b in report.overconstrained_blocks
        if "|".join(b.variables) not in dismissed_refs
    ]
    kept_cycles = [
        c
        for c in report.derived_cycle_conflicts
        if f"cycle::{c.description}" not in dismissed_refs
    ]
    return report.model_copy(
        update={
            "overconstrained_blocks": kept_blocks,
            "derived_cycle_conflicts": kept_cycles,
        }
    )


async def _reconciliation_loop(
    shard_states: List[_ShardState],
    analysis_schema: Schema,
    facts_map: Dict[int, AtomicFact],
    fact_to_shard: Dict[int, int],
    model: Optional[str],
    max_retries: int,
    max_rounds: int,
    label: str,
) -> Tuple[Stage3AnalysisReport, int]:
    """Merge shard_states' current family outputs, analyze, reconcile any
    detected conflicts, apply MISEXTRACTION fixes by re-running the owning
    shard's specific family loop, and repeat until either no conflicts
    remain, a round produces no misextraction fixes, or max_rounds is hit.
    Mutates shard_states[*].outputs in place (via _apply_misextraction_fixes).
    Returns the final analysis report (with dismissed_conflicts populated)
    and tokens spent in this call."""
    total_tokens = 0
    dismissed: List[DismissedConflict] = []
    report = Stage3AnalysisReport()
    fixes_applied_last_round = False

    for round_num in range(1, max_rounds + 1):
        merged = _merge_all(shard_states)
        report, variable_fact_map = analyze_cross_shard_constraints(
            distributions=merged[0],
            structural=merged[3],
            logic=merged[4],
            derived=merged[5],
            schema=analysis_schema,
        )
        fixes_applied_last_round = False
        items = _extract_conflict_items(report, variable_fact_map)
        if not items:
            logger.info(f"[Stage 3][{label}] round {round_num}: no conflicts detected.")
            break

        schema_text = _schema_to_text(analysis_schema)
        verdicts, tokens = await _reconcile_conflicts(
            items, facts_map, schema_text, merged, model
        )
        total_tokens += tokens

        fixes_by_family: Dict[str, List[MisextractionFix]] = {}
        dismissed_this_round = 0
        for item, verdict in zip(items, verdicts):
            if verdict.verdict == ReconciliationVerdict.MISEXTRACTION:
                for fix in verdict.fixes:
                    fixes_by_family.setdefault(fix.family, []).append(fix)
            elif verdict.verdict == ReconciliationVerdict.FALSE_POSITIVE:
                dismissed.append(
                    DismissedConflict(
                        conflict_ref=item.conflict_ref,
                        reason=verdict.reasoning,
                        fact_references=item.fact_ids,
                    )
                )
                dismissed_this_round += 1
            else:  # GENUINE_CONTRADICTION
                logger.info(
                    f"[Stage 3][{label}] confirmed genuine contradiction "
                    f"'{item.conflict_ref}': {verdict.reasoning}"
                )

        if not fixes_by_family:
            logger.info(
                f"[Stage 3][{label}] round {round_num}: {dismissed_this_round} "
                f"dismissed as false positives, no misextractions to fix -- done."
            )
            break

        applied_tokens = await _apply_misextraction_fixes(
            fixes_by_family, shard_states, fact_to_shard, facts_map, model, max_retries
        )
        total_tokens += applied_tokens
        fixes_applied_last_round = True
        logger.info(
            f"[Stage 3][{label}] round {round_num}: applied "
            f"{sum(len(v) for v in fixes_by_family.values())} misextraction fixes "
            f"across {len(fixes_by_family)} families -- re-analyzing."
        )
    else:
        logger.warning(
            f"[Stage 3][{label}] reconciliation hit max_rounds={max_rounds} "
            f"without fully converging."
        )

    if fixes_applied_last_round:
        # The last round applied fixes but never re-analyzed afterward --
        # this is a purely deterministic re-computation (no LLM/tokens),
        # needed so the returned report reflects those fixes.
        merged = _merge_all(shard_states)
        report, _variable_fact_map = analyze_cross_shard_constraints(
            distributions=merged[0],
            structural=merged[3],
            logic=merged[4],
            derived=merged[5],
            schema=analysis_schema,
        )

    dismissed_refs = {d.conflict_ref for d in dismissed}
    final_report = _filter_dismissed(report, dismissed_refs)
    final_report.dismissed_conflicts = dismissed
    return final_report, total_tokens


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
    max_reconciliation_rounds: int = 5,
) -> Tuple[Stage3Output, int]:
    """Run Stage 3: per-shard extraction -> per-shard reconciliation ->
    global merge -> global reconciliation.

    Args:
        schema: Full merged schema (from Stage 2).
        shards: Schema shards (from Stage 2 segments).
        facts: Atomic facts (from Stage 1).
        model: LLM model name (None = default).
        ablation_config: Ablation settings (None = no ablation).
        max_retries: Max node-executions per family's extractor+auditor loop
            (see _run_family_loop's max_iter comment for how this scales).
        max_reconciliation_rounds: Max reconcile-then-reextract rounds per
            reconciliation pass (Tier 2 per shard, Tier 3 once globally).

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
        "[Stage 3] Fact allocation complete: "
        + ", ".join(
            f"shard {i}: {len(a.fact_ids)} facts, {len(a.stub_tables)} stubs"
            for i, a in enumerate(allocations)
        )
    )

    # 2. Tier 1: per-shard, per-family extraction (parallel across shards
    # and families).
    shard_states: List[_ShardState] = [
        _ShardState(
            index=i,
            schema=shard,
            fact_ids=allocation.fact_ids,
            stub_tables=allocation.stub_tables,
        )
        for i, (shard, allocation) in enumerate(zip(shards, allocations))
    ]
    tier1_results = await asyncio.gather(
        *[
            _run_shard_extraction(
                ss.schema, ss.fact_ids, facts_map, ss.stub_tables, model, max_retries
            )
            for ss in shard_states
        ]
    )
    total_tokens = 0
    for ss, (outputs, tokens) in zip(shard_states, tier1_results):
        ss.outputs = outputs
        ss.tokens = tokens
        total_tokens += tokens

    logger.info(f"[Stage 3] Tier 1 extraction complete for {len(shard_states)} shards.")

    fact_to_shard = {fid: ss.index for ss in shard_states for fid in ss.fact_ids}

    # 3. Tier 2: per-shard reconciliation, scoped to each shard's own Schema
    # (parallel across shards -- each shard's reconciliation is independent).
    tier2_results = await asyncio.gather(
        *[
            _reconciliation_loop(
                [ss],
                ss.schema,
                facts_map,
                fact_to_shard,
                model,
                max_retries,
                max_reconciliation_rounds,
                label=f"shard{ss.index}-local",
            )
            for ss in shard_states
        ]
    )
    for ss, (_local_report, tokens) in zip(shard_states, tier2_results):
        total_tokens += tokens
    # Tier 2's own per-shard reports are used only to drive shard-local
    # misextraction fixes above -- discarded here. Tier 3 re-analyzes the
    # (now shard-locally-cleaned) constraints at global scope, which is the
    # report Stage 4 actually consumes.

    logger.info("[Stage 3] Tier 2 per-shard reconciliation complete.")

    # 4. Tier 3: global reconciliation over every shard's finalized output.
    global_report, tier3_tokens = await _reconciliation_loop(
        shard_states,
        schema,
        facts_map,
        fact_to_shard,
        model,
        max_retries,
        max_reconciliation_rounds,
        label="global",
    )
    total_tokens += tier3_tokens

    distributions, moment_targets, correlations, structural, logic, derived = (
        _merge_all(shard_states)
    )

    logger.info(
        f"[Stage 3] Final merged constraints: {len(distributions)} distributions, "
        f"{len(moment_targets)} moments, {len(correlations)} correlations, "
        f"{len(structural)} structural, {len(logic)} logic, {len(derived)} derived."
    )
    logger.info(
        f"[Stage 3] Global DOF analysis: {len(global_report.square_variables)} square, "
        f"{len(global_report.loose_variable_probes)} loose, "
        f"{len(global_report.overconstrained_blocks)} overconstrained, "
        f"{len(global_report.dismissed_conflicts)} dismissed."
    )

    output = Stage3Output(
        distributions=distributions,
        moment_targets=moment_targets,
        correlations=correlations,
        structural_constraints=structural,
        logic_constraints=logic,
        derived_columns=derived,
        analysis_report=global_report,
        token_usage=total_tokens,
    )

    return output, total_tokens
