"""Phases 2 and 3: global conflict analysis, schema-locality grouping, and the
reconcile-then-re-extract loop.

This is the largest single concern in Stage 3 and the one with the most
subtle control flow (per-conflict retry budgets surviving across rounds, an
unconditional final snapshot). It gets its own module for that reason.

NOTE for tests: names patched here must be patched on THIS module, not on
entry -- e.g. monkeypatch.setattr(stage3_reconciliation, "_rerun_shard", ...).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.util.schema_model.schema import Schema
from src.pipeline.stage3.agents.conflict_reconciler.agent import (
    ConflictItemForReconciliation,
    reconcile_conflict_group,
)
from src.pipeline.stage3.middleware.constraint_graph import (
    analyze_cross_shard_constraints,
)
from src.pipeline.stage3.models.probe import (
    DismissedConflict,
    GroupReconciliation,
    MisextractionFix,
    ReconciliationVerdict,
    Stage3AnalysisReport,
)
from src.orchestration.stage3.context import (
    _facts_to_text,
    _render_involved_constraints,
    _schema_to_text,
)
from src.orchestration.stage3.extraction import _rerun_shard
from src.orchestration.stage3.state import _ShardState, _merge_all
from src.util.constraint_model.bridge.from_cross_shard import bridge_constraints
from src.util.constraint_model.conflicts.evaluate import evaluate_constraints
from src.util.constraint_model.conflicts.models import Conflict
from src.util.constraint_model.relation.nodes import extract_base_tables
from src.util.orchestration.parallel_loop import run_parallel

logger = logging.getLogger(__name__)


@dataclass
class _ConflictItem:
    conflict_ref: str
    description: str
    fact_ids: List[int]
    tables: FrozenSet[str]


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
