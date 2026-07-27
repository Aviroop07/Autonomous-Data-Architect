"""Column resolution and type-compatibility for a Constraint's Condition
against its Relation's synthesized schema (Section 7.3-7.4).

Defines `ConditionUnion` -- the actual type of `Constraint.condition`
(task #30): either an ordinary boolean predicate tree, OR one of the three
cohesive terms, as siblings of ONE discriminated union keyed on the shared
`node_type` field. This is what enforces Section 9.3's standalone-only
rule for real: `RAnd`/`ROr`/`RIfThen`'s own `operands`/`antecedent`/
`consequent` fields stay typed as `RPredicateUnion` (predicates.py's
8-member union, no cohesive terms) -- only the top-level `Constraint.
condition` field can ever be a `Distributed`/`Correlated`/`StateSequence`,
so nesting one inside an ordinary predicate is a type error, not just a
documented rule.

Section 7.4's "one unified resolution path" is implemented literally:
`RColumnRef.name` and `RAggregateRef.alias` are resolved through the same
membership check against the synthesized schema's columns (an Aggregate's
alias becomes an ordinary named column there) -- this doesn't yet
distinguish "this name came from a real Aggregate" from "a coincidentally
same-named ordinary column," a narrow, honestly-flagged gap rather than
one silently papered over.

Nullability-narrowing (Section 4.4) is deliberately NOT this module's
concern -- it's already fully handled inside relation/schema.py's Filter
synthesis; Section 7.3 explicitly scopes narrowing to Filter/Relation-side
only, so an ordinary top-level Condition never triggers it here.
"""

from __future__ import annotations

from typing import Annotated, Dict, List, Union

from pydantic import Field

from src.pipeline.stage2.models.data_types import DataType
from src.util.constraint_model.condition.cohesive import (
    Correlated,
    Distributed,
    StateSequence,
)
from src.util.constraint_model.condition.expressions import (
    RAggregateRef,
    RArithmetic,
    RColumnRef,
    RExprUnion,
    is_categorical_eligible,
    is_numeric,
)
from src.util.constraint_model.condition.predicates import (
    RAnd,
    RBetween,
    RComparison,
    RIfThen,
    RInSet,
    RNot,
    RNotInSet,
    ROr,
    RPredicateUnion,
    validate_predicate_types,
)
from src.util.constraint_model.relation.schema import EffectiveSchema

ConditionUnion = Annotated[
    Union[
        RComparison,
        RAnd,
        ROr,
        RNot,
        RBetween,
        RInSet,
        RNotInSet,
        RIfThen,
        Distributed,
        Correlated,
        StateSequence,
    ],
    Field(discriminator="node_type"),
]


def _collect_names_expr(node: RExprUnion, out: set[str]) -> None:
    if isinstance(node, RColumnRef):
        out.add(node.name)
    elif isinstance(node, RAggregateRef):
        out.add(node.alias)
    elif isinstance(node, RArithmetic):
        _collect_names_expr(node.left, out)
        _collect_names_expr(node.right, out)


def _collect_names_pred(node: "RPredicateUnion", out: set[str]) -> None:
    if isinstance(node, RComparison):
        _collect_names_expr(node.left, out)
        _collect_names_expr(node.right, out)
    elif isinstance(node, (RAnd, ROr)):
        for op in node.operands:
            _collect_names_pred(op, out)
    elif isinstance(node, RNot):
        _collect_names_pred(node.operand, out)
    elif isinstance(node, RBetween):
        _collect_names_expr(node.expr, out)
        _collect_names_expr(node.low, out)
        _collect_names_expr(node.high, out)
    elif isinstance(node, (RInSet, RNotInSet)):
        _collect_names_expr(node.expr, out)
    elif isinstance(node, RIfThen):
        _collect_names_pred(node.antecedent, out)
        _collect_names_pred(node.consequent, out)


def _referenced_names(condition: "ConditionUnion") -> set[str]:
    if isinstance(condition, Distributed):
        return {condition.column}
    if isinstance(condition, Correlated):
        return set(condition.columns)
    if isinstance(condition, StateSequence):
        return {condition.sequence_column}
    names = set()
    _collect_names_pred(condition, names)
    return names


def resolve_condition_columns(
    condition: "ConditionUnion", schema: EffectiveSchema
) -> List[str]:
    """Checks every column/aggregate-alias name the condition references
    exists in `schema` and isn't marked ambiguous (relation/schema.py's
    same-named-Join-merge case)."""
    errors: List[str] = []
    for name in sorted(_referenced_names(condition)):
        col = schema.columns.get(name)
        if col is None:
            errors.append(f"Condition references unknown column/alias '{name}'.")
        elif col.ambiguous:
            errors.append(
                f"Condition references '{name}', which is ambiguous in this Relation "
                "-- add a distinguishing alias."
            )
    return errors


def _validate_distributed_types(
    node: Distributed, column_types: Dict[str, DataType]
) -> List[str]:
    dtype = column_types[node.column]
    if node.family == "CATEGORICAL":
        if not is_categorical_eligible(dtype):
            return [
                f"Distributed(CATEGORICAL).column '{node.column}' has type {dtype}, "
                "not categorical-eligible."
            ]
    elif node.family == "POISSON":
        if dtype != DataType.INTEGER:
            return [
                f"Distributed(POISSON).column '{node.column}' must be INTEGER, got {dtype}."
            ]
    else:  # GAUSSIAN, LOG_NORMAL, BETA, UNIFORM
        if not is_numeric(dtype):
            return [
                f"Distributed({node.family}).column '{node.column}' must be numeric, got {dtype}."
            ]
    return []


def _validate_state_sequence_types(
    node: StateSequence, column_types: Dict[str, DataType]
) -> List[str]:
    errors: List[str] = []
    seq_type = column_types[node.sequence_column]
    if not is_categorical_eligible(seq_type):
        errors.append(
            f"StateSequence.sequence_column '{node.sequence_column}' must be "
            f"discrete/categorical, got {seq_type}."
        )
    return errors


def validate_condition_types(
    condition: "ConditionUnion", column_types: Dict[str, DataType]
) -> List[str]:
    """Type-compatibility check. Assumes every referenced name is already
    known to exist in `column_types` -- call resolve_condition_columns
    first and only proceed here once that returns no errors."""
    if isinstance(condition, Distributed):
        return _validate_distributed_types(condition, column_types)
    if isinstance(condition, Correlated):
        return []  # Section 8.2.1: numeric/categorical/mixed are all valid, one mechanism.
    if isinstance(condition, StateSequence):
        return _validate_state_sequence_types(condition, column_types)
    return validate_predicate_types(condition, column_types)


def validate_condition(
    condition: "ConditionUnion", schema: EffectiveSchema
) -> List[str]:
    """Public entry point: column resolution, then type-compatibility
    (skipped if resolution already failed -- type-checking an unresolved
    name isn't meaningful)."""
    errors = resolve_condition_columns(condition, schema)
    if errors:
        return errors
    column_types = {name: col.data_type for name, col in schema.columns.items()}
    return validate_condition_types(condition, column_types)
