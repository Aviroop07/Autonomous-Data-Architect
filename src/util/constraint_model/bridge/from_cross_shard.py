"""Translates the live extraction agents' output shape (cross_shard.py's
Constraint/ONNode/RPredicate) into constraint_model's Constraint/Relation/
ConditionUnion, so the deterministic conflicts/ package (built against the
richer constraint_model representation) can evaluate real extracted data.

The CONDITION half of that translation no longer exists: the extraction
models now use constraint_model's own RPredicateUnion/RExprUnion directly
(pipeline/stage3/models/condition_nodes.py, a 17-of-20 name-for-name
duplicate of condition/{expressions,predicates}.py, has been deleted), so a
constraint's `condition` needs no conversion at all -- it is already the
right type. What remains here is genuinely a bridge: the ON tree, whose
node shapes differ, plus the four constraint-level wrappers.

That also removed this module's two largest functions, _bridge_expr and
_bridge_predicate, along with their handling of RExists/RNotExists and
RComparison's '~' operator. Those three node types were unrepresentable in
constraint_model and so were rejected here 100% of the time -- the LLM could
emit them only to have them discarded. They no longer exist in the schema
the model is given, so the failure mode is gone rather than merely reported.

DerivedColumnConstraint is still deliberately NOT bridged: it has no
constraint_model equivalent shape (no Relation+Condition at all) and stays
handled exclusively by middleware/cycles.py's cycle detector.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.pipeline.stage3.models import on_nodes as cs_on
from src.pipeline.stage3.models.cross_shard import Constraint as CSConstraint
from src.pipeline.stage3.models.cross_shard import CorrelatedConstraint
from src.pipeline.stage3.models.cross_shard import DistributionConstraint
from src.pipeline.stage3.models.cross_shard import StateSequenceConstraint
from src.util.constraint_model.condition.cohesive import Correlated
from src.util.constraint_model.condition.cohesive import Distributed
from src.util.constraint_model.condition.cohesive import (
    PairwiseCorrelation as MPairwiseCorrelation,
)
from src.util.constraint_model.condition.cohesive import StateSequence
from src.util.constraint_model.condition.cohesive import (
    StateTransition as MStateTransition,
)
from src.util.constraint_model.constraint import Constraint as MConstraint
from src.util.constraint_model.relation.nodes import Aggregate as MAggregate
from src.util.constraint_model.relation.nodes import BaseTable as MBaseTable
from src.util.constraint_model.relation.nodes import Fanout as MFanout
from src.util.constraint_model.relation.nodes import Filter as MFilter
from src.util.constraint_model.relation.nodes import Join as MJoin
from src.util.constraint_model.relation.nodes import JoinCondition as MJoinCondition
from src.util.constraint_model.relation.nodes import RelationUnion as MRelationUnion


# ---------------------------------------------------------------------------
# ON tree -> Relation tree
# ---------------------------------------------------------------------------


def _bridge_on(
    node: "cs_on.ONNode",
) -> Tuple[Optional["MRelationUnion"], Optional[str]]:
    if isinstance(node, cs_on.ONBaseTable):
        return MBaseTable(name=node.name), None
    if isinstance(node, cs_on.ONJoin):
        left, err = _bridge_on(node.left)
        if left is None:
            return None, err
        right, err = _bridge_on(node.right)
        if right is None:
            return None, err
        return (
            MJoin(
                left=left,
                right=right,
                on=[
                    MJoinCondition(left=jc.left, right=jc.right, op=jc.op)
                    for jc in node.on
                ],
                alias=node.alias,
            ),
            None,
        )
    if isinstance(node, cs_on.ONAggregate):
        source, err = _bridge_on(node.source)
        if source is None:
            return None, err
        return (
            MAggregate(
                source=source,
                fn=node.fn,
                column=node.column,
                group_by=node.group_by,
                alias=node.alias,
            ),
            None,
        )
    if isinstance(node, cs_on.ONFanout):
        return (
            MFanout(
                parent_table=node.parent_table,
                child_table=node.child_table,
                fk_column=node.fk_column,
            ),
            None,
        )
    if isinstance(node, cs_on.ONSubquery):
        return (
            None,
            "ONSubquery reached the bridge unnormalized -- normalize_on() "
            f"(deterministic_checker.py) should have replaced it before an "
            f"accepted constraint ever gets here: {node.sql}",
        )
    return None, f"Unknown ON node type: {type(node).__name__}"


# ---------------------------------------------------------------------------
# Constraint-level bridging
# ---------------------------------------------------------------------------


def _bridge_distribution(
    d: "DistributionConstraint",
) -> Tuple[Optional[MConstraint], Optional[str]]:
    base_relation, err = _bridge_on(d.on)
    if base_relation is None:
        return (
            None,
            f"DistributionConstraint(fact_references={d.fact_references}): {err}",
        )

    # if_condition is already an RPredicateUnion -- no translation step, and
    # so no "unbridgeable if_condition" failure mode any more.
    relation = base_relation
    if d.if_condition is not None:
        relation = MFilter(source=base_relation, condition=d.if_condition)

    condition = Distributed(column=d.column, family=d.family, parameters=d.parameters)
    return (
        MConstraint(
            relation=relation, condition=condition, fact_references=d.fact_references
        ),
        None,
    )


def _bridge_generic_constraint(
    c: "CSConstraint",
) -> Tuple[Optional[MConstraint], Optional[str]]:
    relation, err = _bridge_on(c.on)
    if relation is None:
        return None, f"Constraint(fact_references={c.fact_references}): {err}"
    # c.condition is already an RPredicateUnion -- passed straight through.
    return (
        MConstraint(
            relation=relation,
            condition=c.condition,
            fact_references=c.fact_references,
            severity=c.severity,
        ),
        None,
    )


def _bridge_correlated(
    c: "CorrelatedConstraint",
) -> Tuple[Optional[MConstraint], Optional[str]]:
    """Direct construction -- CorrelatedConstraint mirrors constraint_model's
    Correlated field-for-field, so no lossy translation is needed."""
    relation, err = _bridge_on(c.on)
    if relation is None:
        return None, f"CorrelatedConstraint(fact_references={c.fact_references}): {err}"
    condition = Correlated(
        columns=c.columns,
        family=c.family,
        pairwise=[
            MPairwiseCorrelation(left=p.left, right=p.right, value=p.value)
            for p in c.pairwise
        ],
        shared_parameters=dict(c.shared_parameters),
    )
    return (
        MConstraint(
            relation=relation, condition=condition, fact_references=c.fact_references
        ),
        None,
    )


def _bridge_state_sequence(
    c: "StateSequenceConstraint",
) -> Tuple[Optional[MConstraint], Optional[str]]:
    """Direct construction -- StateSequenceConstraint mirrors
    constraint_model's StateSequence field-for-field."""
    relation, err = _bridge_on(c.on)
    if relation is None:
        return (
            None,
            f"StateSequenceConstraint(fact_references={c.fact_references}): {err}",
        )
    condition = StateSequence(
        sequence_column=c.sequence_column,
        allowed_transitions=[
            MStateTransition(from_state=t.from_state, to_state=t.to_state)
            for t in c.allowed_transitions
        ],
        forbidden_transitions=[
            MStateTransition(from_state=t.from_state, to_state=t.to_state)
            for t in c.forbidden_transitions
        ],
        strict=c.strict,
    )
    return (
        MConstraint(
            relation=relation, condition=condition, fact_references=c.fact_references
        ),
        None,
    )


def bridge_constraints(
    distributions: List["DistributionConstraint"],
    moment_targets: List["CSConstraint"],
    correlations: List["CorrelatedConstraint"],
    structural: List["CSConstraint"],
    logic: List["CSConstraint"],
    schema,
    state_sequences: Optional[List["StateSequenceConstraint"]] = None,
) -> Tuple[List[MConstraint], List[str]]:
    """Bridges every distribution/moment/correlation/structural/logic/
    state-sequence Constraint into constraint_model's shape.
    DerivedColumnConstraint is never passed here -- it has no equivalent
    (see module docstring). Returns (bridged constraints, unsupported-
    reason strings) -- a skipped constraint is always reported, never
    silently dropped."""
    del schema  # not needed by this tree-shaped translation; kept for API symmetry
    bridged: List[MConstraint] = []
    unsupported: List[str] = []

    for d in distributions:
        mc, err = _bridge_distribution(d)
        if mc is not None:
            bridged.append(mc)
        else:
            unsupported.append(err or "unknown bridging failure")

    for c in correlations:
        mc, err = _bridge_correlated(c)
        if mc is not None:
            bridged.append(mc)
        else:
            unsupported.append(f"[correlation] {err or 'unknown bridging failure'}")

    for c in state_sequences or []:
        mc, err = _bridge_state_sequence(c)
        if mc is not None:
            bridged.append(mc)
        else:
            unsupported.append(f"[state_sequence] {err or 'unknown bridging failure'}")

    for label, items in (
        ("moment_target", moment_targets),
        ("structural", structural),
        ("logic", logic),
    ):
        for c in items:
            mc, err = _bridge_generic_constraint(c)
            if mc is not None:
                bridged.append(mc)
            else:
                unsupported.append(f"[{label}] {err or 'unknown bridging failure'}")

    return bridged, unsupported
