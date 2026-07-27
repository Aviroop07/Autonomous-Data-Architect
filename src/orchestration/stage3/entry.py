"""Stage 3 orchestration entry point.

Implements the redesigned Stage 3 data flow (see PROGRESS.md for the
design discussion this followed -- a rewrite of the prior 3-tier
statistical/structural/logic-family orchestration):

  Phase 1: per-shard extraction -- each schema shard gets its OWN 3-node
           AgentLoop: generator -> deterministic_checker -> auditor. One
           unified generator now extracts every constraint category in a
           single pass (replacing the 3 separate family loops); the
           deterministic canonicalize() check is its own explicit node;
           the auditor is a single unified semantic reviewer. All shards
           run in parallel.
  Phase 2: global conflict analysis -- once every shard is done, ALL
           shards' constraints are merged ONCE and analyzed together (no
           per-shard reconciliation tier). Two analyses run side by side:
             (a) the OLD cross_shard-native analyze_cross_shard_constraints
                 -- kept ONLY for what nothing else provides: square/loose
                 variable classification, MomentTarget derivation-chain
                 resolution (Q3), and derived-column cycle detection. Its
                 own overconstrained_blocks/value-conflict signal is
                 IGNORED here -- superseded by (b).
             (b) the NEW deterministic conflicts/ engine
                 (util/constraint_model/conflicts), via a bridge that
                 translates cross_shard's Constraint/ONNode/RPredicate
                 into constraint_model's richer Constraint/Relation/
                 Condition shape. This is what drives reconciliation --
                 it catches distribution/moment/correlation/population/
                 state-sequence conflicts the old engine never checked.
  Phase 3: schema-locality grouping + reconciliation -- conflicts (from
           the new engine, plus cycle issues from (a)) are grouped by
           shared table/column locality (union-find), then each group is
           sent to ONE conflict_reconciler call covering every conflict in
           it. Verdicts: MISEXTRACTION re-runs the owning shard's WHOLE
           generator loop (no more per-family isolation -- there is only
           one generator now) with injected guidance; FALSE_POSITIVE drops
           the conflict (recorded, not silently discarded);
           GENUINE_CONTRADICTION is left in place. Each conflict_ref has
           its own retry counter, surviving across rounds -- once a
           conflict has been sent to the reconciler max_constraint_retries
           times without resolving, it is auto-dismissed as a false
           positive (budget exhausted, recorded with that reason) rather
           than looping forever.

Known, deliberate scope reduction from the prior version: the old
per-shard reconciliation tier is gone (per the redesign -- one global pass
now, not per-shard-then-global), and structural constraints only get a
DOF-overconstrained check via the new engine's own bridge-based check
(util/constraint_model/variables.py), not the old constraint_graph.py
pathway's richer aggregation-DOF handling (a known pre-existing gap in
that pathway too, per its own module docstring).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from pydantic import BaseModel, Field

from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage2.models.registry import TableFactRegistry
from src.pipeline.stage2.models.schema import Schema, Table
from src.pipeline.stage3.agents.conflict_reconciler.agent import (
    ConflictItemForReconciliation,
    reconcile_conflict_group,
)
from src.pipeline.stage3.agents.constraint_auditor.agent import (
    ConstraintAuditorLoopAgent,
)
from src.pipeline.stage3.agents.constraint_generator.agent import (
    ConstraintGeneratorLoopAgent,
)
from src.pipeline.stage3.middleware.constraint_graph import (
    analyze_cross_shard_constraints,
)
from src.pipeline.stage3.middleware.deterministic_checker import (
    DeterministicCheckerLoopAgent,
)
from src.pipeline.stage3.middleware.fact_allocation import (
    allocate_facts_to_shards,
    find_mentioned_tables,
)
from src.pipeline.stage3.middleware.fact_column_mapping import build_fact_column_map
from src.pipeline.stage3.models.cross_shard import (
    Constraint,
    CorrelatedConstraint,
    DerivedColumnConstraint,
    DistributionConstraint,
    StateSequenceConstraint,
    UnifiedExtractionOutput,
)
from src.pipeline.stage3.models.probe import (
    DismissedConflict,
    GroupReconciliation,
    MisextractionFix,
    ReconciliationVerdict,
    Stage3AnalysisReport,
)
from src.util.algorithms.sharding_ilp import shard_schema_auto
from src.util.config.ablation import AblationConfig
from src.util.constraint_model.bridge.from_cross_shard import bridge_constraints
from src.util.core.agent import _detect_provider
from src.util.observability.artifact_dump import dump_artifact
from src.util.constraint_model.conflicts.evaluate import evaluate_constraints
from src.util.constraint_model.conflicts.models import Conflict
from src.util.constraint_model.relation.nodes import extract_base_tables
from src.util.orchestration.loop import AgentLoop
from src.util.orchestration.loop_types import (
    AgentRoleConfig,
    EdgeCondition,
    ErrorRefreshConfig,
    GraphEdge,
    LoopConfig,
    LoopResult,
)
from src.util.orchestration.parallel_loop import (
    ParallelLoopSpec,
    run_parallel,
    run_parallel_loops,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


class Stage3Output(BaseModel):
    """Complete Stage 3 output: extracted constraints + global DOF analysis."""

    distributions: List[DistributionConstraint] = Field(default_factory=list)
    moment_targets: List[Constraint] = Field(default_factory=list)
    correlations: List[CorrelatedConstraint] = Field(default_factory=list)
    structural_constraints: List[Constraint] = Field(default_factory=list)
    logic_constraints: List[Constraint] = Field(default_factory=list)
    derived_columns: List[DerivedColumnConstraint] = Field(default_factory=list)
    state_sequences: List[StateSequenceConstraint] = Field(default_factory=list)
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
            + len(self.state_sequences)
        )


# ---------------------------------------------------------------------------
# Shard state -- mutable per-shard bookkeeping threaded through both phases
# ---------------------------------------------------------------------------


@dataclass
class _ShardState:
    index: int
    schema: Schema
    fact_ids: List[int]
    stub_tables: List[str]
    output: UnifiedExtractionOutput = field(default_factory=UnifiedExtractionOutput)
    tokens: int = 0


@dataclass
class _ConflictItem:
    conflict_ref: str
    description: str
    fact_ids: List[int]
    tables: FrozenSet[str]


# ---------------------------------------------------------------------------
# Text rendering helpers (schema / facts / constraints -> prompt-friendly text)
# ---------------------------------------------------------------------------


def _schema_to_text(schema: Schema, stub_tables: Optional[List[str]] = None) -> str:
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
    lines: List[str] = ["## FACTS"]
    for fid in sorted(fact_ids):
        fact = facts_map.get(fid)
        if fact is not None:
            lines.append(f"- [id={fid}] {fact.fact}")
    return "\n".join(lines)


def _render_involved_constraints(fact_ids: List[int], merged: "_Merged") -> str:
    """Dump every extracted constraint that references any of fact_ids, so
    the reconciliation agent can see exactly what was extracted from the
    facts it's re-examining."""
    fact_id_set = set(fact_ids)

    def _matches(c: Any) -> bool:
        return any(fid in fact_id_set for fid in c.fact_references)

    lines: List[str] = []
    for label, items in (
        ("Distribution", merged.distributions),
        ("MomentTarget", merged.moment_targets),
        ("Correlation", merged.correlations),
        ("Structural", merged.structural),
        ("Logic", merged.logic),
        ("Derived", merged.derived),
        ("StateSequence", merged.state_sequences),
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
    registry = TableFactRegistry()
    table_names = [t.name for t in schema.tables]
    for fact in facts:
        mentioned = find_mentioned_tables(fact.fact, table_names)
        for table_name in mentioned:
            registry.register_table_facts(table_name, [fact.id])
    return registry


def _build_shard_table_sets(shards: List[Schema]) -> List[set[str]]:
    return [{t.name for t in shard.tables} for shard in shards]


# ---------------------------------------------------------------------------
# Automatic sharding (used when orchestrate() isn't given explicit shards)
# ---------------------------------------------------------------------------

# A fact whose build_fact_column_map() resolution spans more than this
# many DISTINCT tables (regardless of which of the mapper's three signals
# resolved it) is treated as a generic meta-statement, not a reliable
# cross-table join signal -- excluded from the ILP's own hard fact-
# containment constraint (empty pairs) rather than forcing that many
# tables to co-locate, which made the ILP infeasible under any reasonable
# per-shard table cap. Never dropped from `facts` itself -- orchestrate()'s
# own allocate_facts_to_shards() re-derives per-shard fact coverage
# independently of how the shards were produced.
_MAX_MENTIONED_TABLES_FOR_SHARDING = 4


def _build_ilp_inputs(
    schema: Schema, facts: List[AtomicFact]
) -> Tuple[
    List[str],
    Dict[str, List[str]],
    Dict[str, List[str]],
    List[Tuple[str, str, str, str]],
    Dict[str, List[Tuple[str, str]]],
]:
    tables = [t.name for t in schema.tables]
    columns_by_table = {t.name: [c.name for c in t.columns] for t in schema.tables}
    pks_by_table = {t.name: t.primary_key for t in schema.tables}
    table_map = schema.get_table_map()

    fks: List[Tuple[str, str, str, str]] = []
    for fk in schema.relationships or []:
        referred = table_map.get(fk.referred_table)
        if referred is None or not referred.primary_key:
            logger.warning(
                f"[Stage 3] auto-sharding: skipping FK {fk.referencing_table}."
                f"{fk.referencing_column} -> {fk.referred_table}: referred "
                f"table has no single-column PK."
            )
            continue
        fks.append(
            (
                fk.referencing_table,
                fk.referencing_column,
                fk.referred_table,
                referred.primary_key[0],
            )
        )

    fact_col_map = build_fact_column_map(schema, facts)
    ilp_facts: Dict[str, List[Tuple[str, str]]] = {}
    for fact in facts:
        cols = fact_col_map.get(fact.id, [])
        distinct_tables = {t for t, _ in cols}
        if len(distinct_tables) > _MAX_MENTIONED_TABLES_FOR_SHARDING:
            ilp_facts[str(fact.id)] = []
            continue
        ilp_facts[str(fact.id)] = cols

    return tables, columns_by_table, pks_by_table, fks, ilp_facts


def _shard_dicts_to_schemas(
    shard_dicts: List[Dict[str, List[str]]], global_schema: Schema
) -> List[Schema]:
    table_map = global_schema.get_table_map()
    shard_schemas: List[Schema] = []
    for shard_tables in shard_dicts:
        tables: List[Table] = []
        for table_name, col_names in shard_tables.items():
            src_table = table_map[table_name]
            col_set = set(col_names)
            tables.append(
                Table(
                    name=src_table.name,
                    columns=[c for c in src_table.columns if c.name in col_set],
                    primary_key=src_table.primary_key,
                )
            )
        relationships = [
            fk
            for fk in (global_schema.relationships or [])
            if fk.referencing_table in shard_tables
            and fk.referred_table in shard_tables
            and fk.referencing_column in shard_tables[fk.referencing_table]
        ]
        shard_schemas.append(Schema(tables=tables, relationships=relationships))
    return shard_schemas


def _derive_shards_from_schema(
    schema: Schema, facts: List[AtomicFact], model: Optional[str]
) -> List[Schema]:
    """The no-manual-hyperparameters entry point orchestrate() falls back to
    when callers don't supply pre-sharded `shards` themselves: derives
    max_shards/max_tables_per_shard/ILP weights automatically from the
    schema, the fact set, and the target provider/model's real context
    window (via shard_schema_auto()), never exposing those knobs here."""
    provider, api_key, _base_url, default_model = _detect_provider()
    resolved_model = model or default_model
    tables, columns_by_table, pks_by_table, fks, ilp_facts = _build_ilp_inputs(
        schema, facts
    )
    shard_dicts, _shard_facts = shard_schema_auto(
        tables,
        columns_by_table,
        pks_by_table,
        fks,
        ilp_facts,
        provider=provider,
        model=resolved_model,
        api_key=api_key,
    )
    if shard_dicts is None:
        raise RuntimeError(
            "[Stage 3] automatic sharding found no feasible solution for this "
            "schema/fact set."
        )
    logger.info(
        f"[Stage 3] Automatic sharding produced {len(shard_dicts)} shard(s): "
        + ", ".join(f"{list(st.keys())}" for st in shard_dicts)
    )
    return _shard_dicts_to_schemas(shard_dicts, schema)


def _serialize_context(
    shard: Schema,
    fact_ids: List[int],
    facts_map: Dict[int, AtomicFact],
    stub_tables: List[str],
    reconciliation_guidance: Optional[str] = None,
) -> str:
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
# Phase 1: per-shard 3-node generator/checker/auditor loop
# ---------------------------------------------------------------------------


def _build_generator_loop_config(max_iter: int, model: Optional[str]) -> LoopConfig:
    """Builds one shard's generator -> deterministic_checker -> auditor
    LoopConfig.

    Graph shape:
        generator -----------------------------------> det_checker
        det_checker --[has canonicalize errors]------> generator  (retry)
        det_checker -----------------------------------> auditor
        auditor --[is_valid == False]------------------> generator (retry)
        auditor -----------------------------------------> end

    Edge order matters -- AgentLoop._evaluate_edges is first-match-wins.

    Split out from execution (a separate function used to just call
    AgentLoop(config).run(...) directly) so Phase 1 extraction and
    misextraction reruns can batch many shards' configs through
    run_parallel_loops() in one call -- see util/orchestration/
    parallel_loop.py's module docstring for why that matters: failure
    isolation (one shard's parse hiccup no longer crashes every other
    shard's in-progress work) and cross-shard retry-budget reallocation.

    Takes the raw `max_iter` directly (not a "logical retries" count) --
    callers that want the old "N full audited rounds" semantics convert
    themselves (see _run_generator_loop's `max_retries * 4`); Phase 1's
    own construction in orchestrate() passes its auto-scaled raw value
    directly, since its policy differs from reruns/diagnostics.
    """
    generator = ConstraintGeneratorLoopAgent(model=model)
    checker = DeterministicCheckerLoopAgent()
    auditor = ConstraintAuditorLoopAgent(model=model)

    return LoopConfig(
        agents={
            "generator": AgentRoleConfig(
                agent_factory=lambda: generator,
                det_error_sources=["det_checker"],
            ),
            "det_checker": AgentRoleConfig(agent_factory=lambda: checker),
            "auditor": AgentRoleConfig(agent_factory=lambda: auditor),
        },
        graph={
            "edges": [
                GraphEdge(from_node="generator", to_node="det_checker"),
                GraphEdge(
                    from_node="det_checker",
                    to_node="generator",
                    condition=EdgeCondition(
                        fn=lambda o: bool(getattr(o, "errors", []))
                    ),
                ),
                GraphEdge(from_node="det_checker", to_node="auditor"),
                GraphEdge(
                    from_node="auditor",
                    to_node="generator",
                    condition=EdgeCondition(field="is_valid", op="eq", value=False),
                ),
                GraphEdge(from_node="auditor", to_node="end"),
            ]
        },
        start_node="generator",
        max_iter=max_iter,
        error_refresh=ErrorRefreshConfig(trigger_node="generator"),
    )


def _extract_generator_output(
    result: Optional["LoopResult"],
) -> Tuple[UnifiedExtractionOutput, int]:
    """Pulls the generator node's final output out of a completed
    LoopResult. `result` is None when run_parallel_loops() isolated a
    failure for this unit (see its own docstring) -- treated as an empty
    output rather than propagating the failure, the same way a shard that
    ran but produced nothing extractable is already handled."""
    if result is None:
        return UnifiedExtractionOutput(), 0
    output = result.node_outputs.get("generator")
    if not isinstance(output, UnifiedExtractionOutput):
        output = UnifiedExtractionOutput()
    return output, result.total_tokens


async def _run_generator_loop(
    context_str: str,
    model: Optional[str],
    max_retries: int,
) -> Tuple[UnifiedExtractionOutput, int]:
    """Runs one shard's generator loop directly (no cross-shard budget
    sharing) -- kept for any single-shard caller, e.g. isolated diagnostic
    scripts, and for shard reruns (_rerun_shard). Live Phase 1 extraction
    goes through run_parallel_loops() instead, with its own auto-scaled
    raw max_iter (see orchestrate()'s Phase 1 construction) rather than
    this function's "N full audited rounds" (max_retries * 4) policy."""
    config = _build_generator_loop_config(max_retries * 4, model)
    result = await AgentLoop(config).run(context_str)
    return _extract_generator_output(result)


async def _rerun_shard(
    shard: Schema,
    fact_ids: List[int],
    facts_map: Dict[int, AtomicFact],
    stub_tables: List[str],
    guidance: str,
    model: Optional[str],
    max_retries: int,
) -> Tuple[UnifiedExtractionOutput, int]:
    """Re-run a shard's WHOLE generator loop with reconciliation guidance
    injected -- there is only one generator now, so a MISEXTRACTION fix
    always targets the owning shard's single loop, not a specific family.

    Deliberately kept as its own function (rather than inlined into
    _build_generator_loop_config + run_parallel_loops like Phase 1's
    extraction fan-out) and run through the simpler run_parallel(), not
    run_parallel_loops() -- misextraction reruns are a small, targeted
    fan-out (only the shards a reconciliation round actually flagged),
    so cross-shard retry-budget reallocation matters far less here than
    for Phase 1's bulk extraction. Each rerun still gets its own
    independent budget. This shape also keeps _rerun_shard directly
    monkeypatchable by name, which existing tests rely on.
    """
    context_str = _serialize_context(
        shard, fact_ids, facts_map, stub_tables, reconciliation_guidance=guidance
    )
    return await _run_generator_loop(context_str, model, max_retries)


@dataclass
class _Merged:
    distributions: List[DistributionConstraint]
    moment_targets: List[Constraint]
    correlations: List[CorrelatedConstraint]
    structural: List[Constraint]
    logic: List[Constraint]
    derived: List[DerivedColumnConstraint]
    state_sequences: List[StateSequenceConstraint]


def _merge_all(shard_states: List[_ShardState]) -> _Merged:
    merged = _Merged([], [], [], [], [], [], [])
    for ss in shard_states:
        merged.distributions.extend(ss.output.distributions)
        merged.moment_targets.extend(ss.output.moment_targets)
        merged.correlations.extend(ss.output.correlations)
        merged.structural.extend(ss.output.structural_constraints)
        merged.logic.extend(ss.output.logic_constraints)
        merged.derived.extend(ss.output.derived_columns)
        merged.state_sequences.extend(ss.output.state_sequences)
    return merged


# ---------------------------------------------------------------------------
# Phase 2/3: global conflict analysis, schema-locality grouping, reconciliation
# ---------------------------------------------------------------------------


def _fact_to_tables(bridged_constraints: List[Any]) -> Dict[int, FrozenSet[str]]:
    """Maps each fact_id to the union of base tables of every bridged
    constraint that traces back to it -- used both to derive a conflict's
    own locality and (by the caller) as the reconciler's schema-slice key."""
    acc: Dict[int, set[str]] = {}
    for c in bridged_constraints:
        tables = extract_base_tables(c.relation)
        for fid in c.fact_references:
            acc.setdefault(fid, set()).update(tables)
    return {fid: frozenset(tables) for fid, tables in acc.items()}


def _conflict_items_from(
    conflicts: List[Conflict],
    cycle_issues,
    fact_to_tables: Dict[int, FrozenSet[str]],
) -> List[_ConflictItem]:
    items: List[_ConflictItem] = []
    for c in conflicts:
        tables: set[str] = set()
        for fid in c.involved_fact_references:
            tables |= fact_to_tables.get(fid, frozenset())
        items.append(
            _ConflictItem(
                conflict_ref=_conflict_ref_for(c),
                description=f"{c.summary} {c.detail}",
                fact_ids=list(c.involved_fact_references),
                tables=frozenset(tables),
            )
        )
    for cycle in cycle_issues:
        tables = set()
        for fid in cycle.fact_references:
            tables |= fact_to_tables.get(fid, frozenset())
        items.append(
            _ConflictItem(
                conflict_ref=f"cycle::{cycle.description}",
                description=cycle.description,
                fact_ids=list(cycle.fact_references),
                tables=frozenset(tables),
            )
        )
    return items


def _group_by_schema_locality(items: List[_ConflictItem]) -> List[List[_ConflictItem]]:
    """Union-find over shared table locality: two conflict items merge into
    one group if their table sets intersect. A fact-independent conflict
    (empty table set, e.g. a structural over-determination with no NL
    fact behind it) gets its own singleton group -- it cannot share
    locality with anything by construction."""
    n = len(items)
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        if not items[i].tables:
            continue
        for j in range(i + 1, n):
            if not items[j].tables:
                continue
            if items[i].tables & items[j].tables:
                _union(i, j)

    groups: Dict[int, List[_ConflictItem]] = {}
    for i in range(n):
        groups.setdefault(_find(i), []).append(items[i])
    return list(groups.values())


def _conflict_key(item: _ConflictItem) -> str:
    return item.conflict_ref


def _conflict_ref_for(c: Conflict) -> str:
    """The SAME conflict_ref formula _conflict_items_from uses -- factored
    out so the final dismissed-filter (which only has raw Conflict objects,
    not _ConflictItem) can compute the identical key without duplicating
    the string-formatting logic in two places."""
    if c.involved_fact_references:
        facts = "-".join(str(f) for f in sorted(c.involved_fact_references))
        return f"{c.kind}::{facts}"
    return f"{c.kind}::{c.summary}"


async def _reconcile_and_apply(
    shard_states: List[_ShardState],
    analysis_schema: Schema,
    facts_map: Dict[int, AtomicFact],
    fact_to_shard: Dict[int, int],
    model: Optional[str],
    max_retries: int,
    max_rounds: int,
    max_constraint_retries: int,
) -> Tuple[
    List[Conflict], List[DismissedConflict], List[str], int, Stage3AnalysisReport
]:
    """The whole Phase 2/3 loop: merge -> bridge -> evaluate -> group ->
    reconcile -> apply fixes -> repeat. Mutates shard_states[*].output in
    place via MISEXTRACTION re-extraction.

    Returns (remaining unresolved conflicts, dismissed conflicts, bridge/engine
    unsupported notes, total tokens spent, the final DOF report).

    That last element exists so orchestrate() does not run
    analyze_cross_shard_constraints() a second time on inputs this function has
    already analyzed. It used to: once per reconciliation round here, then once
    more in orchestrate(), so N+1 full DOF builds per run where the last two
    were provably identical (the final snapshot below is computed after the
    last possible mutation of shard_states). Only three fields of the report
    are ever read."""
    total_tokens = 0
    retry_count: Dict[str, int] = {}
    dismissed: List[DismissedConflict] = []
    remaining_conflicts: List[Conflict] = []
    all_unsupported: List[str] = []

    for round_num in range(1, max_rounds + 1):
        merged = _merge_all(shard_states)

        old_report, _variable_fact_map = analyze_cross_shard_constraints(
            distributions=merged.distributions,
            structural=merged.structural,
            logic=merged.logic,
            derived=merged.derived,
            schema=analysis_schema,
        )
        cycle_issues = old_report.derived_cycle_conflicts

        bridged, bridge_unsupported = bridge_constraints(
            merged.distributions,
            merged.moment_targets,
            merged.correlations,
            merged.structural,
            merged.logic,
            analysis_schema,
            state_sequences=merged.state_sequences,
        )
        conflict_report = evaluate_constraints(bridged, analysis_schema)
        fact_to_tables = _fact_to_tables(bridged)

        already_dismissed_refs = {d.conflict_ref for d in dismissed}
        items = [
            it
            for it in _conflict_items_from(
                conflict_report.conflicts, cycle_issues, fact_to_tables
            )
            if it.conflict_ref not in already_dismissed_refs
        ]

        if not items:
            logger.info(
                f"[Stage 3] round {round_num}: no conflicts remain -- converged."
            )
            break

        groups = _group_by_schema_locality(items)

        sendable_groups: List[List[_ConflictItem]] = []
        for group in groups:
            sendable = [
                it
                for it in group
                if retry_count.get(_conflict_key(it), 0) < max_constraint_retries
            ]
            if not sendable:
                # Every item in this group already exhausted its retry
                # budget -- auto-dismiss each as a false positive rather
                # than looping forever on the same unresolved dispute.
                for it in group:
                    if not any(d.conflict_ref == it.conflict_ref for d in dismissed):
                        dismissed.append(
                            DismissedConflict(
                                conflict_ref=it.conflict_ref,
                                reason=(
                                    f"Retry budget exhausted after "
                                    f"{max_constraint_retries} reconciliation attempts "
                                    "with no resolution -- auto-dismissed as a false "
                                    "positive rather than looping indefinitely."
                                ),
                                fact_references=it.fact_ids,
                            )
                        )
                continue
            sendable_groups.append(sendable)

        if not sendable_groups:
            logger.info(
                f"[Stage 3] round {round_num}: every remaining conflict already "
                "exhausted its retry budget -- done."
            )
            break

        async def _reconcile_one_group(
            group: List[_ConflictItem],
        ) -> Tuple[List[_ConflictItem], GroupReconciliation, int]:
            schema_text = _schema_to_text(analysis_schema)
            request_items = [
                ConflictItemForReconciliation(
                    conflict_ref=it.conflict_ref,
                    description=it.description,
                    involved_facts=_facts_to_text(it.fact_ids, facts_map),
                    involved_constraints=_render_involved_constraints(
                        it.fact_ids, merged
                    ),
                )
                for it in group
            ]
            verdicts, tokens = await reconcile_conflict_group(
                request_items, schema_text, model=model
            )
            return group, verdicts, tokens

        raw_results = await run_parallel(
            [_reconcile_one_group(g) for g in sendable_groups],
            labels=[f"group-{i}" for i in range(len(sendable_groups))],
        )
        failed_count = sum(1 for r in raw_results if r is None)
        if failed_count:
            logger.warning(
                f"[Stage 3] round {round_num}: {failed_count} reconciliation "
                f"group call(s) failed entirely (isolated -- siblings "
                f"unaffected) -- their conflicts stay unresolved this round "
                f"and will be retried (or eventually auto-dismissed via the "
                f"per-conflict retry budget) next round."
            )
        results = [r for r in raw_results if r is not None]

        fixes_by_shard: Dict[int, List[MisextractionFix]] = {}
        for group, group_verdicts, tokens in results:
            total_tokens += tokens
            verdict_by_ref = {v.conflict_ref: v for v in group_verdicts.verdicts}
            for it in group:
                retry_count[_conflict_key(it)] = (
                    retry_count.get(_conflict_key(it), 0) + 1
                )
                verdict = verdict_by_ref.get(it.conflict_ref)
                if verdict is None:
                    logger.warning(
                        f"[Stage 3] reconciler omitted a verdict for "
                        f"'{it.conflict_ref}' -- treating as unresolved this round."
                    )
                    continue
                if verdict.verdict == ReconciliationVerdict.MISEXTRACTION:
                    for fix in verdict.fixes:
                        shard_idx = fact_to_shard.get(fix.fact_id)
                        if shard_idx is None:
                            logger.warning(
                                f"[Stage 3] MisextractionFix references unknown "
                                f"fact_id {fix.fact_id} -- skipping."
                            )
                            continue
                        fixes_by_shard.setdefault(shard_idx, []).append(fix)
                elif verdict.verdict == ReconciliationVerdict.FALSE_POSITIVE:
                    dismissed.append(
                        DismissedConflict(
                            conflict_ref=it.conflict_ref,
                            reason=verdict.reasoning,
                            fact_references=it.fact_ids,
                        )
                    )
                else:  # GENUINE_CONTRADICTION
                    logger.info(
                        f"[Stage 3] confirmed genuine contradiction "
                        f"'{it.conflict_ref}': {verdict.reasoning}"
                    )

        if not fixes_by_shard:
            logger.info(
                f"[Stage 3] round {round_num}: no misextractions to fix -- done."
            )
            break

        shard_idxs = list(fixes_by_shard.keys())
        rerun_coros = []
        for shard_idx in shard_idxs:
            ss = shard_states[shard_idx]
            fixes = fixes_by_shard[shard_idx]
            guidance = "\n".join(f"- fact {f.fact_id}: {f.guidance}" for f in fixes)
            rerun_coros.append(
                _rerun_shard(
                    ss.schema,
                    ss.fact_ids,
                    facts_map,
                    ss.stub_tables,
                    guidance,
                    model,
                    max_retries,
                )
            )
        rerun_results = await run_parallel(
            rerun_coros, labels=[f"rerun-shard-{i}" for i in shard_idxs]
        )
        for shard_idx, result in zip(shard_idxs, rerun_results):
            if result is None:
                logger.warning(
                    f"[Stage 3] shard {shard_idx} rerun failed entirely "
                    f"(isolated -- siblings unaffected) -- keeping its "
                    f"prior output rather than crashing the whole run."
                )
                continue
            output, tokens = result
            shard_states[shard_idx].output = output
            shard_states[shard_idx].tokens += tokens
            total_tokens += tokens
        logger.info(
            f"[Stage 3] round {round_num}: re-ran {len(shard_idxs)} shard(s) after "
            f"{sum(len(v) for v in fixes_by_shard.values())} misextraction fix(es)."
        )
    else:
        logger.warning(
            f"[Stage 3] reconciliation hit max_rounds={max_rounds} without fully "
            "converging."
        )

    # Always recompute ONE final snapshot here, regardless of which branch
    # above ended the loop (converged, ran out of sendable groups, ran out
    # of misextraction fixes, or hit max_rounds) -- this is what fixes the
    # bug where remaining_conflicts/all_unsupported were only ever set
    # inside specific early-break branches and silently stayed empty
    # otherwise. The recomputation itself is pure/deterministic (no LLM
    # calls), so doing it unconditionally costs nothing but correctness.
    merged = _merge_all(shard_states)
    bridged, bridge_unsupported = bridge_constraints(
        merged.distributions,
        merged.moment_targets,
        merged.correlations,
        merged.structural,
        merged.logic,
        analysis_schema,
        state_sequences=merged.state_sequences,
    )
    final_report = evaluate_constraints(bridged, analysis_schema)
    # The DOF analysis of this same final snapshot -- returned to the caller
    # rather than recomputed there. shard_states cannot change after this
    # point, so orchestrate()'s copy would be identical by construction.
    final_dof_report, _variable_fact_map = analyze_cross_shard_constraints(
        distributions=merged.distributions,
        structural=merged.structural,
        logic=merged.logic,
        derived=merged.derived,
        schema=analysis_schema,
    )
    dismissed_refs = {d.conflict_ref for d in dismissed}
    remaining_conflicts = [
        c for c in final_report.conflicts if _conflict_ref_for(c) not in dismissed_refs
    ]
    all_unsupported = bridge_unsupported + final_report.unsupported

    return (
        remaining_conflicts,
        dismissed,
        all_unsupported,
        total_tokens,
        final_dof_report,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def orchestrate(
    schema: Schema,
    facts: List[AtomicFact],
    shards: Optional[List[Schema]] = None,
    model: Optional[str] = None,
    ablation_config: Optional[AblationConfig] = None,
    max_retries: Optional[int] = None,
    max_reconciliation_rounds: int = 5,
    max_constraint_retries: int = 3,
    artifact_dir: Optional[Path] = None,
) -> Tuple[Stage3Output, int]:
    """Run Stage 3: per-shard extraction -> global conflict analysis ->
    schema-locality grouping -> reconciliation.

    Args:
        schema: Full merged schema (from Stage 2).
        facts: Atomic facts (from Stage 1).
        shards: Pre-computed schema shards, if the caller already has them
            (e.g. reusing Stage 2's own segments, or a test needing
            deterministic shard boundaries). When omitted, shards are
            derived automatically from `schema` + `facts` via
            shard_schema_auto() -- no max_shards/max_tables_per_shard/ILP
            weight hyperparameters are exposed to callers of this
            function; everything is estimated internally from the schema's
            size, the fact set, and the target model's real context
            window.
        model: LLM model name (None = default).
        ablation_config: Ablation settings (None = no ablation).
        max_retries: Explicit override for "logical retries" (max_iter =
            max_retries * 4) used by shard reruns and reconciliation.
            When omitted (None, the default -- never manually set in
            normal use), reruns fall back to 5 (the prior hardcoded
            default), and Phase 1's OWN per-shard extraction budget is
            derived independently: each shard's raw max_iter defaults to
            a fixed constant of 3, so the total retry budget across the
            whole parallel batch is num_shards * 3 node-executions --
            shared/redistributable via run_parallel_loops()'s existing
            cross-shard reallocation (a shard that converges early
            donates its leftover budget to still-struggling siblings).
            An explicit override here still applies literally (max_iter =
            max_retries * 4) to Phase 1 too, for diagnostic/test callers
            that need a fixed, larger budget regardless of shard count.
        max_reconciliation_rounds: Max reconcile-then-reextract rounds for
            the single global reconciliation pass.
        max_constraint_retries: Max times a single conflict can be sent to
            the reconciler before it is auto-dismissed as a false positive.
        artifact_dir: If set, dumps intermediate state after each major
            phase (per-shard extraction, global reconciliation) via
            dump_artifact -- a mid-run crash still leaves everything
            computed up to that point inspectable, matching Stage 2's own
            opt-in crash-safety convention. No-ops entirely when None.

    Returns:
        (Stage3Output, total_tokens)
    """
    del ablation_config  # not yet consumed by this pathway
    if shards is None:
        shards = _derive_shards_from_schema(schema, facts, model)
    logger.info(
        f"[Stage 3] Starting extraction: {len(shards)} shards, {len(facts)} facts."
    )

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

    shard_states: List[_ShardState] = [
        _ShardState(
            index=i,
            schema=shard,
            fact_ids=allocation.fact_ids,
            stub_tables=allocation.stub_tables,
        )
        for i, (shard, allocation) in enumerate(zip(shards, allocations))
    ]
    # Reruns/reconciliation keep the prior hardcoded default (5) when the
    # caller doesn't override -- unaffected by this session's Phase 1
    # rescoping. Phase 1 gets its OWN, separately-scaled raw max_iter:
    # a fixed constant (3) per shard when not overridden, never derived
    # from or multiplied by shard count directly -- the num_shards*3
    # "total budget" is simply what emerges from every shard starting at
    # the same fixed constant, redistributable via run_parallel_loops().
    resolved_rerun_max_retries = max_retries if max_retries is not None else 5
    phase1_max_iter = max_retries * 4 if max_retries is not None else 3
    phase1_specs = [
        ParallelLoopSpec(
            config=_build_generator_loop_config(phase1_max_iter, model),
            initial_context=_serialize_context(
                ss.schema, ss.fact_ids, facts_map, ss.stub_tables
            ),
            label=f"shard-{ss.index}",
        )
        for ss in shard_states
    ]
    phase1_results = await run_parallel_loops(phase1_specs)
    total_tokens = 0
    for ss, result in zip(shard_states, phase1_results):
        if result is None:
            logger.warning(
                f"[Stage 3] shard {ss.index} extraction failed entirely "
                f"(isolated by run_parallel_loops -- siblings unaffected) "
                f"-- treated as an empty output rather than crashing the "
                f"whole run."
            )
        output, tokens = _extract_generator_output(result)
        ss.output = output
        ss.tokens = tokens
        total_tokens += tokens

    logger.info(
        f"[Stage 3] Phase 1 extraction complete for {len(shard_states)} shards."
    )
    dump_artifact(
        artifact_dir,
        "stage3_01_phase1_shard_outputs",
        [ss.output for ss in shard_states],
    )

    fact_to_shard = {fid: ss.index for ss in shard_states for fid in ss.fact_ids}

    (
        conflicts,
        dismissed,
        unsupported,
        phase23_tokens,
        old_report,
    ) = await _reconcile_and_apply(
        shard_states,
        schema,
        facts_map,
        fact_to_shard,
        model,
        resolved_rerun_max_retries,
        max_reconciliation_rounds,
        max_constraint_retries,
    )
    total_tokens += phase23_tokens

    logger.info("[Stage 3] Phase 2/3 global reconciliation complete.")
    dump_artifact(
        artifact_dir,
        "stage3_02_phase23_reconciliation",
        {
            "conflicts": [c.model_dump(mode="json") for c in conflicts],
            "dismissed": [d.model_dump(mode="json") for d in dismissed],
            "unsupported": unsupported,
        },
    )

    # square/loose/moment-target probes + cycles still come from the old,
    # cross_shard-native DOF pathway -- see module docstring for why this
    # stays alongside the new engine rather than being replaced by it.
    # `old_report` is _reconcile_and_apply's own final snapshot, not a fresh
    # analysis: shard_states is not mutated after it returns.
    merged = _merge_all(shard_states)
    dismissed_cycle_refs = {d.conflict_ref for d in dismissed}
    remaining_cycles = [
        c
        for c in old_report.derived_cycle_conflicts
        if f"cycle::{c.description}" not in dismissed_cycle_refs
    ]

    final_report = Stage3AnalysisReport(
        square_variables=old_report.square_variables,
        loose_variable_probes=old_report.loose_variable_probes,
        overconstrained_blocks=[],
        derived_cycle_conflicts=remaining_cycles,
        dismissed_conflicts=dismissed,
        conflicts=conflicts,
        unsupported=unsupported,
    )

    logger.info(
        f"[Stage 3] Final merged constraints: {len(merged.distributions)} "
        f"distributions, {len(merged.moment_targets)} moments, "
        f"{len(merged.correlations)} correlations, {len(merged.structural)} "
        f"structural, {len(merged.logic)} logic, {len(merged.derived)} derived, "
        f"{len(merged.state_sequences)} state sequences."
    )
    logger.info(
        f"[Stage 3] Global conflict analysis: {len(final_report.conflicts)} "
        f"unresolved conflicts, {len(final_report.dismissed_conflicts)} dismissed, "
        f"{len(final_report.derived_cycle_conflicts)} unresolved cycles."
    )

    output = Stage3Output(
        distributions=merged.distributions,
        moment_targets=merged.moment_targets,
        correlations=merged.correlations,
        structural_constraints=merged.structural,
        logic_constraints=merged.logic,
        derived_columns=merged.derived,
        state_sequences=merged.state_sequences,
        analysis_report=final_report,
        token_usage=total_tokens,
    )
    dump_artifact(artifact_dir, "stage3_03_final_output", output)

    return output, total_tokens
