"""Wraps the extraction agents' output (pipeline/stage3/models/cross_shard.py)
into constraint_model's Constraint, so the deterministic conflicts/ package can
evaluate real extracted data.

This module used to be a genuine bridge, ~400 lines translating three parallel
taxonomies node by node: an ON tree into a Relation tree, an R-AST predicate
tree into an RPredicateUnion, and expressions into expressions. None of that
exists any more. The extraction models use constraint_model's own Relation and
Condition types directly, so `on` IS a RelationUnion and `condition` IS an
RPredicateUnion -- both are passed through untouched.

What is left is the only thing that was ever really a translation: the four
extraction wrappers (DistributionConstraint, CorrelatedConstraint,
StateSequenceConstraint, and the generic Constraint) carry their payload in
flat, LLM-friendly fields, and those become the corresponding cohesive
Condition terms. That mapping is field-for-field by construction, so it cannot
fail -- which is why `unsupported` is now always empty. It is still returned:
callers (and the Stage3AnalysisReport) treat it as the channel for "extracted
but not evaluable", and re-adding a failure mode later should not change this
function's signature.

DerivedColumnConstraint is deliberately NOT handled here: it has no
Relation+Condition shape at all, and stays with middleware/cycles.py's cycle
detector.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.pipeline.stage3.models.cross_shard import Constraint as CSConstraint
from src.pipeline.stage3.models.cross_shard import CorrelatedConstraint
from src.pipeline.stage3.models.cross_shard import DistributionConstraint
from src.pipeline.stage3.models.cross_shard import StateSequenceConstraint
from src.util.constraint_model.condition.cohesive import (
    Correlated,
    Distributed,
    StateSequence,
)
from src.util.constraint_model.condition.cohesive import (
    PairwiseCorrelation as MPairwiseCorrelation,
)
from src.util.constraint_model.condition.cohesive import (
    StateTransition as MStateTransition,
)
from src.util.constraint_model.constraint import Constraint as MConstraint
from src.util.constraint_model.relation.nodes import Filter as MFilter


def _from_distribution(d: "DistributionConstraint") -> MConstraint:
    # A conditional distribution narrows its own population, which is exactly
    # what a Filter node means -- so if_condition becomes a Filter wrapping the
    # base relation rather than part of the Distributed term.
    relation = (
        d.on
        if d.if_condition is None
        else MFilter(source=d.on, condition=d.if_condition)
    )
    return MConstraint(
        relation=relation,
        condition=Distributed(
            column=d.column, family=d.family, parameters=d.parameters
        ),
        fact_references=d.fact_references,
    )


def _from_generic(c: "CSConstraint") -> MConstraint:
    return MConstraint(
        relation=c.on,
        condition=c.condition,
        fact_references=c.fact_references,
        severity=c.severity,
    )


def _from_correlated(c: "CorrelatedConstraint") -> MConstraint:
    return MConstraint(
        relation=c.on,
        condition=Correlated(
            columns=c.columns,
            family=c.family,
            pairwise=[
                MPairwiseCorrelation(left=p.left, right=p.right, value=p.value)
                for p in c.pairwise
            ],
            shared_parameters=dict(c.shared_parameters),
        ),
        fact_references=c.fact_references,
    )


def _from_state_sequence(c: "StateSequenceConstraint") -> MConstraint:
    return MConstraint(
        relation=c.on,
        condition=StateSequence(
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
        ),
        fact_references=c.fact_references,
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
    """Wrap every distribution/moment/correlation/structural/logic/state-
    sequence constraint as a constraint_model Constraint.

    Returns (constraints, unsupported-reason strings). `unsupported` is
    currently always empty -- see the module docstring.
    """
    del schema  # structural wrapping needs no schema; kept for API symmetry
    unsupported: List[str] = []

    bridged: List[MConstraint] = [_from_distribution(d) for d in distributions]
    bridged += [_from_correlated(c) for c in correlations]
    bridged += [_from_state_sequence(c) for c in state_sequences or []]
    for items in (moment_targets, structural, logic):
        bridged += [_from_generic(c) for c in items]

    return bridged, unsupported
