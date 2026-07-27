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
             (b) the deterministic conflicts/ engine
                 (util/constraint_model/conflicts). Extraction output is
                 wrapped into constraint_model Constraints by bridge/
                 from_cross_shard.py -- a wrapping, not a translation,
                 since `on` and `condition` are already the right types.
                 This is what drives reconciliation: it catches
                 distribution/moment/correlation/population/state-sequence
                 conflicts the DOF pathway never checked.
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

Module layout (dependency order, acyclic):
    state.py           Stage3Output, _ShardState, _Merged, _merge_all
    context.py         schema/facts/constraints -> prompt text
    sharding.py        automatic shard derivation via the ILP
    extraction.py      Phase 1's per-shard 3-node loop
    reconciliation.py  Phases 2/3, the reconcile-then-re-extract loop
    entry.py           orchestrate() -- this file, control flow only
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage3.middleware.fact_allocation import allocate_facts_to_shards
from src.pipeline.stage3.models.probe import Stage3AnalysisReport
from src.orchestration.stage3.context import _serialize_context
from src.orchestration.stage3.extraction import (
    _build_generator_loop_config,
    _extract_generator_output,
)
from src.orchestration.stage3.reconciliation import _reconcile_and_apply
from src.orchestration.stage3.sharding import (
    _build_registry,
    _build_shard_table_sets,
    _derive_shards_from_schema,
)
from src.orchestration.stage3.state import Stage3Output, _ShardState, _merge_all
from src.util.config.ablation import AblationConfig
from src.util.observability.artifact_dump import dump_artifact
from src.util.orchestration.parallel_loop import ParallelLoopSpec, run_parallel_loops

logger = logging.getLogger(__name__)


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
