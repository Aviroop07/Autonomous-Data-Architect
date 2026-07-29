"""Cross-stage invariants of a whole stage1 -> stage2 -> stage3 run.

Each of these states something NO unit test can state, because each one relates
an artefact of one stage to an artefact of another. They share a single
session-scoped run (see conftest.PipelineRun).
"""

from __future__ import annotations

from typing import Iterable, List, Set, Union

from src.pipeline.stage3.models.cross_shard import (
    Constraint,
    CorrelatedConstraint,
    DerivedColumnConstraint,
    DistributionConstraint,
    StateSequenceConstraint,
)
from src.util.constraint_model.condition.expressions import (
    RArithmetic,
    RColumnRef,
    RExprUnion,
)
from src.util.constraint_model.condition.predicates import extract_columns
from src.util.constraint_model.relation.nodes import (
    Aggregate,
    BaseTable,
    Fanout,
    Join,
    RelationUnion,
    extract_base_tables,
)
from src.orchestration.stage3.state import Stage3Output
from src.util.schema_model.schema import Schema

from .conftest import PipelineRun

#: The four constraint shapes that carry an `on` Relation tree. They share no
#: base class, so the union is what keeps these helpers precisely typed instead
#: of falling back to `object`.
OnBearing = Union[
    Constraint,
    DistributionConstraint,
    CorrelatedConstraint,
    StateSequenceConstraint,
]


# ---------------------------------------------------------------------------
# Independent column-resolution helpers
# ---------------------------------------------------------------------------
#
# Deliberately a SECOND implementation of the accessible-columns rule, not a
# call into `grain.py`'s `validate_column`. A test that reuses the production
# resolver can only agree with it; this one can disagree, which is the point.


def _columns_of(schema: Schema, tables: Iterable[str]) -> Set[str]:
    wanted = set(tables)
    return {
        column.name
        for table in schema.tables
        if table.name in wanted
        for column in table.columns
    }


def _accessible_columns(on: RelationUnion, schema: Schema) -> Set[str]:
    """What an unqualified column reference may name, at this ON tree's grain."""
    if isinstance(on, Fanout):
        # A fanout is COUNT(children) GROUP BY parent.pk: the parent's own
        # columns survive, plus the synthetic `child_count`. The CHILD's columns
        # do not -- they have been aggregated away.
        return _columns_of(schema, [on.parent_table]) | {"child_count"}
    if isinstance(on, Aggregate):
        # Post-aggregation only the grouping keys and the aggregate's alias
        # remain referenceable.
        return set(on.group_by or []) | {on.alias}
    if isinstance(on, (BaseTable, Join)):
        return _columns_of(schema, extract_base_tables(on))
    raise AssertionError(f"unhandled ON node type {type(on).__name__}")


def _referenced_columns(item: OnBearing) -> Set[str]:
    """Every column name `item` names outside its own ON tree.

    Covers each family's own column-bearing fields, and reads `partition_by` /
    `order_by` reflectively so that if either is ever added to a constraint
    shape it is checked from day one rather than silently skipped.
    """
    cols: Set[str] = set()
    if isinstance(item, DistributionConstraint):
        cols.add(item.column)
        if item.if_condition is not None:
            cols |= extract_columns(item.if_condition)
    elif isinstance(item, CorrelatedConstraint):
        cols |= set(item.columns)
    elif isinstance(item, StateSequenceConstraint):
        cols.add(item.sequence_column)
    elif isinstance(item, Constraint):
        cols |= extract_columns(item.condition)
    for extra_field in ("partition_by", "order_by"):
        value = getattr(item, extra_field, None)
        if isinstance(value, str):
            cols.add(value)
        elif isinstance(value, list):
            cols |= {v for v in value if isinstance(v, str)}
    return cols


def _expression_columns(expr: RExprUnion) -> Set[str]:
    if isinstance(expr, RColumnRef):
        return {expr.name}
    if isinstance(expr, RArithmetic):
        return _expression_columns(expr.left) | _expression_columns(expr.right)
    return set()


def _all_on_bearing_constraints(stage3: Stage3Output) -> List[OnBearing]:
    return [
        *stage3.distributions,
        *stage3.moment_targets,
        *stage3.correlations,
        *stage3.structural_constraints,
        *stage3.logic_constraints,
        *stage3.state_sequences,
    ]


# ---------------------------------------------------------------------------
# The invariants
# ---------------------------------------------------------------------------


def test_stage1_fact_ids_partition_into_covered_and_uncovered(
    pipeline_run: PipelineRun,
) -> None:
    """Every Stage 1 fact id is EITHER represented in the schema Stage 2 built
    OR listed as uncovered -- never both, never neither.

    Stage 2's own `uncovered_fact_ids` is a difference of two sets it computes
    internally, so a unit test can only re-derive it the same way. This states
    the property from the other side: the three sets (registry provenance,
    schema provenance, uncovered) must exactly partition Stage 1's fact ids.

    Mutation this catches: make `_compute_uncovered_facts` intersect required
    ids with `covered` instead of subtracting (or drop the `table.source_fact_ids`
    union added for junction tables) and facts fall out of every set at once --
    the schema stops mentioning them AND the report stops flagging them.
    """
    stage1_ids = {f.id for f in pipeline_run.stage1.final_facts}
    assert stage1_ids == {1, 2, 3, 4, 5, 6}, (
        "the canned extraction defines the identities every later invariant is "
        f"stated in terms of; got {sorted(stage1_ids)}"
    )

    schema = pipeline_run.schema
    from_registry: Set[int] = set()
    for table in schema.tables:
        from_registry |= set(pipeline_run.registry.get_facts_for_tables([table.name]))

    from_schema: Set[int] = set()
    for table in schema.tables:
        from_schema |= set(table.source_fact_ids or [])
        for column in table.columns:
            from_schema |= set(column.source_fact_ids or [])
    for fk in schema.relationships or []:
        from_schema |= set(fk.source_fact_ids or [])

    covered = from_registry | from_schema
    uncovered = set(pipeline_run.stage2.uncovered_fact_ids)

    assert covered & uncovered == set(), (
        f"facts {sorted(covered & uncovered)} are reported uncovered while the "
        f"schema demonstrably traces to them"
    )
    assert covered | uncovered == stage1_ids, (
        f"facts {sorted(stage1_ids - (covered | uncovered))} vanished between "
        f"Stage 1 and Stage 2: neither represented in the schema nor reported "
        f"as uncovered"
    )
    # Every fact in this specification is genuinely modellable, so the honest
    # partition here puts all of them on the covered side. Asserting that (and
    # not merely "the sets partition") is what makes the test fail if coverage
    # silently degrades.
    assert uncovered == set(), f"unexpectedly uncovered facts: {sorted(uncovered)}"


def test_every_column_named_by_a_stage3_constraint_resolves_in_the_final_schema(
    pipeline_run: PipelineRun,
) -> None:
    """THE FLAGSHIP INVARIANT. Every column any Stage 3 constraint names must
    resolve against the schema Stage 4 will actually be handed.

    Stage 3's own deterministic checker validates columns against the SHARD
    schema it was given, before Stage 2's compliance certifier patches have been
    applied and without ever seeing the merged global schema. So a constraint can
    pass every in-stage check and still name a column that does not exist in the
    artefact downstream. Nothing in production closes that gap; this does.

    Mutation this catches: have the relational mapper name the 1:N foreign key
    anything other than the parent's PK column (`ORDER.customer_ref`), or have
    `apply_patches` drop a column the certifier flagged -- the fanout's
    `fk_column` and the constraints over that table stop resolving, and no
    existing test notices because the shard-level check already passed.
    """
    schema = pipeline_run.schema
    table_names = {t.name for t in schema.tables}
    stage3 = pipeline_run.stage3

    assert stage3.total_constraints == 7, (
        "all seven constraint families must survive to the output, or this test "
        f"silently checks fewer shapes than it claims; got "
        f"{stage3.total_constraints}"
    )

    failures: List[str] = []

    for item in _all_on_bearing_constraints(stage3):
        on = item.on
        label = f"{type(item).__name__}(facts={item.fact_references})"

        for table in extract_base_tables(on):
            if table not in table_names:
                failures.append(f"{label}: ON names table '{table}', not in schema")

        # The ON tree's own column references, which the families' fields do not
        # cover: a fanout's FK column and an aggregate's aggregated column.
        if isinstance(on, Fanout):
            child_columns = _columns_of(schema, [on.child_table])
            if on.fk_column not in child_columns:
                failures.append(
                    f"{label}: fanout fk_column '{on.fk_column}' is not a column "
                    f"of child table '{on.child_table}'"
                )
        if isinstance(on, Aggregate) and on.column != "*":
            source_columns = _columns_of(schema, extract_base_tables(on.source))
            if on.column not in source_columns:
                failures.append(
                    f"{label}: aggregate column '{on.column}' is not a column of "
                    f"{sorted(extract_base_tables(on.source))}"
                )

        accessible = _accessible_columns(on, schema)
        for column in sorted(_referenced_columns(item)):
            if column not in accessible:
                failures.append(
                    f"{label}: references column '{column}', which is not "
                    f"accessible at this grain (accessible: {sorted(accessible)})"
                )

    for derived in stage3.derived_columns:
        assert isinstance(derived, DerivedColumnConstraint)
        label = f"DerivedColumn({derived.target_table}.{derived.target_column})"
        if derived.target_table not in table_names:
            failures.append(f"{label}: target_table is not in the schema")
        for table in derived.referenced_tables:
            if table not in table_names:
                failures.append(f"{label}: referenced_tables names '{table}'")
        source_columns = _columns_of(schema, derived.referenced_tables)
        for column in sorted(_expression_columns(derived.expression)):
            if column not in source_columns:
                failures.append(
                    f"{label}: expression references '{column}', absent from "
                    f"{derived.referenced_tables}"
                )

    assert failures == [], "unresolvable column references:\n" + "\n".join(failures)


def test_stage3_fact_references_all_exist_in_stage1_output(
    pipeline_run: PipelineRun,
) -> None:
    """Fact ids are stable identities across all three stage boundaries.

    A constraint's `fact_references` is its entire provenance chain -- it is how
    reconciliation finds the NL to re-read and how a MISEXTRACTION fix targets a
    shard. An id that no longer names a Stage 1 fact makes that chain silently
    dead: `fact_to_shards.get()` returns nothing and the fix is skipped.

    Mutation this catches: renumber facts anywhere downstream -- e.g. have
    `convert_to_atomic` assign fresh sequential ids instead of carrying
    `raw.id`, or have Stage 3's fact allocation pass shard-local indices into
    `_serialize_context`'s `facts_map`. Both leave every stage internally
    consistent and break provenance across the boundary.
    """
    stage1_ids = {f.id for f in pipeline_run.stage1.final_facts}
    stage3 = pipeline_run.stage3

    cited: Set[int] = set()
    per_constraint: List[tuple[str, List[int]]] = []
    cited_by: List[Union[OnBearing, DerivedColumnConstraint]] = [
        *_all_on_bearing_constraints(stage3),
        *stage3.derived_columns,
    ]
    for item in cited_by:
        refs = list(item.fact_references)
        assert refs, f"{type(item).__name__} reached the output with no provenance"
        per_constraint.append((type(item).__name__, refs))
        cited |= set(refs)

    unknown = {
        (name, r) for name, refs in per_constraint for r in refs if r not in stage1_ids
    }
    assert unknown == set(), (
        f"Stage 3 constraints cite fact ids absent from Stage 1's final_facts: "
        f"{sorted(unknown)}; Stage 1 produced {sorted(stage1_ids)}"
    )

    # Provenance must also be non-degenerate: if every constraint cited the same
    # single fact the check above would still pass while carrying no information.
    assert len(cited) >= 5, (
        f"only {len(cited)} distinct facts are cited by any constraint "
        f"({sorted(cited)}) -- provenance has collapsed"
    )


def test_a_clean_run_never_reaches_the_conflict_reconciler(
    pipeline_run: PipelineRun,
) -> None:
    """A consistent constraint set must produce no conflicts, and therefore no
    reconciler call at all.

    This is the cross-stage counterpart to the conflict engine's unit tests: those
    feed it hand-built `Constraint` objects, whereas here the constraints have
    been through the real bridge, the real `canonicalize()`, and are evaluated
    against the real merged schema. A false positive anywhere in that chain shows
    up as an LLM call that should never have happened.

    Mutation this catches: make `evaluate_constraints` treat two distributions at
    different grains as comparable, or make the state-sequence check flag a
    linear pending->shipped->delivered graph as cyclic -- both mint a conflict on
    a clean input, and both would then burn real reconciler tokens on every run.
    """
    from src.pipeline.stage3.models.probe import GroupReconciliation

    report = pipeline_run.stage3.analysis_report
    assert report.conflicts == [], (
        "clean constraint set produced conflicts: "
        + "; ".join(f"{c.kind}: {c.summary}" for c in report.conflicts)
    )
    assert report.derived_cycle_conflicts == [], (
        "total_with_tax = total_amount * 1.08 has no cycle, yet "
        f"{len(report.derived_cycle_conflicts)} were reported"
    )
    assert report.dismissed_conflicts == [], (
        "nothing should have been raised, so nothing should have been dismissed"
    )
    assert report.is_feasible, "a clean, consistent constraint set must be feasible"
    assert pipeline_run.provider.call_count(GroupReconciliation) == 0, (
        "the reconciler was invoked despite there being no conflicts to reconcile"
    )


def test_token_accounting_survives_every_stage_boundary(
    pipeline_run: PipelineRun,
) -> None:
    """Each stage's reported token total equals 15 per agent call it made.

    Token accounting is the one number that crosses every boundary: agent ->
    get_response -> LoopResult.total_tokens -> stage Output -> caller. The canned
    reply carries exactly 15 tokens, so the expected total is fully determined by
    the call count, which makes this an equality rather than a smoke check.

    Mutation this catches: drop the `+ t_cov`/`+ t_tag` accumulations in Stage 1,
    or return `result.total_tokens` from only the last AgentLoop node instead of
    the sum -- today nothing would notice, because no test relates tokens to
    calls.
    """
    per_call = 15
    total_calls = len(pipeline_run.provider.call_log)
    s1_tokens, s2_tokens, s3_tokens = pipeline_run.tokens

    assert total_calls * per_call == s1_tokens + s2_tokens + s3_tokens, (
        f"{total_calls} agent call(s) at {per_call} tokens each should sum to "
        f"{total_calls * per_call}, but the three stages reported "
        f"{s1_tokens} + {s2_tokens} + {s3_tokens}"
    )
    assert pipeline_run.stage3.token_usage == s3_tokens, (
        "Stage3Output.token_usage must agree with the value orchestrate() returned"
    )
    for stage_name, tokens in (
        ("stage 1", s1_tokens),
        ("stage 2", s2_tokens),
        ("stage 3", s3_tokens),
    ):
        assert tokens % per_call == 0 and tokens > 0, (
            f"{stage_name} reported {tokens} tokens, not a positive multiple of "
            f"the {per_call} tokens every canned reply carries"
        )
