"""Translates the live extraction agents' output shape (cross_shard.py's
Constraint/ONNode/RPredicate) into constraint_model's Constraint/Relation/
ConditionUnion, so the deterministic conflicts/ package (built against the
richer constraint_model representation) can evaluate real extracted data.

Two honest tiers of coverage (see PROGRESS.md's Stage 3 redesign entry):

Tier A -- mechanical, safe: DistributionConstraint -> Distributed; a
generic Constraint's ON/RPredicate tree -> Relation/RPredicateUnion (both
taxonomies match almost 1:1, see the per-node mapping below);
DerivedColumnConstraint is deliberately NOT bridged here -- it has no
constraint_model equivalent shape (no Relation+Condition at all) and stays
handled exclusively by middleware/cycles.py's cycle detector.

Tier B (this module, now built) -- CorrelatedConstraint and
StateSequenceConstraint are typed, first-class cross_shard.py nodes
(mirroring constraint_model's Correlated/StateSequence closely enough for
DIRECT construction here -- no lossy approximation through a generic
predicate tree, unlike the old fallback of cramming a correlation/
sequencing fact into an RComparison).

Also unsupported today: RExists/RNotExists (dropped from constraint_model's
predicate taxonomy entirely -- see condition/predicates.py's module
docstring) and RComparison's '~' operator (superseded by Distributed).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.pipeline.stage3.models import condition_nodes as cs_expr
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
from src.util.constraint_model.condition.expressions import (
    RAggregateRef as MExprAggregateRef,
)
from src.util.constraint_model.condition.expressions import RArithmetic as MArithmetic
from src.util.constraint_model.condition.expressions import RColumnRef as MColumnRef
from src.util.constraint_model.condition.expressions import RExprUnion as MExprUnion
from src.util.constraint_model.condition.expressions import RLiteral as MLiteral
from src.util.constraint_model.condition.predicates import RAnd as MAnd
from src.util.constraint_model.condition.predicates import RBetween as MBetween
from src.util.constraint_model.condition.predicates import RComparison as MComparison
from src.util.constraint_model.condition.predicates import RIfThen as MIfThen
from src.util.constraint_model.condition.predicates import RInSet as MInSet
from src.util.constraint_model.condition.predicates import RNot as MNot
from src.util.constraint_model.condition.predicates import RNotInSet as MNotInSet
from src.util.constraint_model.condition.predicates import ROr as MOr
from src.util.constraint_model.condition.predicates import (
    RPredicateUnion as MPredicateUnion,
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
# Expression tree -> expression tree
# ---------------------------------------------------------------------------


def _bridge_expr(
    node: "cs_expr.RExprUnion",
) -> Tuple[Optional["MExprUnion"], Optional[str]]:
    if isinstance(node, cs_expr.RLiteral):
        return MLiteral(value=node.value), None
    if isinstance(node, cs_expr.RColumnRef):
        return MColumnRef(name=node.name), None
    if isinstance(node, cs_expr.RAggregateRef):
        return MExprAggregateRef(alias=node.alias), None
    if isinstance(node, cs_expr.RArithmetic):
        left, err = _bridge_expr(node.left)
        if left is None:
            return None, err
        right, err = _bridge_expr(node.right)
        if right is None:
            return None, err
        return MArithmetic(op=node.op, left=left, right=right), None
    return None, f"Unknown expression node type: {type(node).__name__}"


# ---------------------------------------------------------------------------
# Predicate tree -> predicate tree
# ---------------------------------------------------------------------------


def _bridge_predicate(
    node: "cs_expr.RPredicate",
) -> Tuple[Optional["MPredicateUnion"], Optional[str]]:
    if isinstance(node, cs_expr.RComparison):
        if node.op == "~":
            return (
                None,
                "RComparison op='~' has no constraint_model equivalent (superseded by Distributed).",
            )
        left, err = _bridge_expr(node.left)
        if left is None:
            return None, err
        right, err = _bridge_expr(node.right)
        if right is None:
            return None, err
        return MComparison(op=node.op, left=left, right=right), None
    if isinstance(node, cs_expr.RAnd):
        operands: List[MPredicateUnion] = []
        for op in node.operands:
            bridged, err = _bridge_predicate(op)
            if bridged is None:
                return None, err
            operands.append(bridged)
        return MAnd(operands=operands), None
    if isinstance(node, cs_expr.ROr):
        operands = []
        for op in node.operands:
            bridged, err = _bridge_predicate(op)
            if bridged is None:
                return None, err
            operands.append(bridged)
        return MOr(operands=operands), None
    if isinstance(node, cs_expr.RNot):
        operand, err = _bridge_predicate(node.operand)
        if operand is None:
            return None, err
        return MNot(operand=operand), None
    if isinstance(node, cs_expr.RBetween):
        expr, err = _bridge_expr(node.expr)
        if expr is None:
            return None, err
        low, err = _bridge_expr(node.low)
        if low is None:
            return None, err
        high, err = _bridge_expr(node.high)
        if high is None:
            return None, err
        return MBetween(expr=expr, low=low, high=high), None
    if isinstance(node, cs_expr.RInSet):
        expr, err = _bridge_expr(node.expr)
        if expr is None:
            return None, err
        return MInSet(expr=expr, values=node.values), None
    if isinstance(node, cs_expr.RNotInSet):
        expr, err = _bridge_expr(node.expr)
        if expr is None:
            return None, err
        return MNotInSet(expr=expr, values=node.values), None
    if isinstance(node, cs_expr.RIfThen):
        ante, err = _bridge_predicate(node.antecedent)
        if ante is None:
            return None, err
        cons, err = _bridge_predicate(node.consequent)
        if cons is None:
            return None, err
        return MIfThen(antecedent=ante, consequent=cons), None
    if isinstance(node, (cs_expr.RExists, cs_expr.RNotExists)):
        return (
            None,
            f"{type(node).__name__} has no constraint_model equivalent (dropped predicate).",
        )
    return None, f"Unknown predicate node type: {type(node).__name__}"


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

    relation = base_relation
    if d.if_condition is not None:
        bridged_if, err = _bridge_predicate(d.if_condition)
        if bridged_if is None:
            return None, (
                f"DistributionConstraint(fact_references={d.fact_references}) "
                f"if_condition unbridgeable: {err}"
            )
        relation = MFilter(source=base_relation, condition=bridged_if)

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
    condition, err = _bridge_predicate(c.condition)
    if condition is None:
        return None, f"Constraint(fact_references={c.fact_references}): {err}"
    return (
        MConstraint(
            relation=relation,
            condition=condition,
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
