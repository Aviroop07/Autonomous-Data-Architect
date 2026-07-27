"""Expression layer: RLiteral, RColumnRef, RArithmetic, RAggregateRef.

See RELATION_CONDITION_CONSTRAINT_DESIGN.md Section 7.1. This module also
carries the bottom-up TYPE-INFERENCE pass for the expression layer,
mirroring Relation's schema synthesis (Section 4) -- every expression node
synthesizes its own result type from its operands' already-inferred types.
RColumnRef/RAggregateRef cannot infer their own type in isolation (they
need the enclosing Relation's synthesized schema); infer_type() takes that
context explicitly rather than storing it on the node.
"""

from __future__ import annotations

from typing import Annotated, List, Literal, Union

from pydantic import BaseModel, Field

from src.pipeline.stage2.models.data_types import DataType

# ---------------------------------------------------------------------------
# Type classification (Section 7.2's comparison type-compatibility rules)
# ---------------------------------------------------------------------------

NUMERIC_TYPES = frozenset({DataType.INTEGER, DataType.FLOAT, DataType.DECIMAL})
TEMPORAL_TYPES = frozenset(
    {DataType.DATE, DataType.DATETIME, DataType.TIMESTAMP, DataType.TIME}
)
STRING_TYPES = frozenset({DataType.VARCHAR, DataType.TEXT})
# Orderable: MAX/MIN/MEDIAN/PERCENTILE/BETWEEN all need a real ordering.
# UUID is excluded -- comparable as a string but not meaningfully orderable.
ORDERABLE_TYPES = NUMERIC_TYPES | TEMPORAL_TYPES | STRING_TYPES
# Discrete/categorical-eligible: Distributed's CATEGORICAL family and
# StateSequence.sequence_column both need a column with a finite, named
# set of values -- not every string/int column qualifies, but this is the
# necessary (not sufficient) type-level check.
CATEGORICAL_ELIGIBLE_TYPES = frozenset(
    {DataType.VARCHAR, DataType.TEXT, DataType.BOOLEAN}
)


def is_numeric(dtype: DataType) -> bool:
    return dtype in NUMERIC_TYPES


def is_orderable(dtype: DataType) -> bool:
    return dtype in ORDERABLE_TYPES


def is_categorical_eligible(dtype: DataType) -> bool:
    return dtype in CATEGORICAL_ELIGIBLE_TYPES


class TypeMismatch(Exception):
    """Raised by infer_type()/comparison-compatibility checks when two
    expressions' types cannot be reconciled. Callers in validate.py catch
    this and turn it into an ordinary validation error string -- it is
    never allowed to propagate as an uncaught exception during validation."""


def numeric_promote(left: DataType, right: DataType) -> DataType:
    """Ordinary numeric promotion across INTEGER/FLOAT/DECIMAL subtypes.
    Section 7.2: numeric-numeric cross-subtype comparisons/arithmetic are
    allowed; the result widens to the more general of the two types."""
    if left == right:
        return left
    if DataType.DECIMAL in (left, right):
        return DataType.DECIMAL
    if DataType.FLOAT in (left, right):
        return DataType.FLOAT
    return DataType.INTEGER


def check_comparable(left: DataType, right: DataType) -> None:
    """Section 7.2's comparison type-compatibility rule. Raises
    TypeMismatch if the two types cannot be compared at all.

    - Cross-family (string vs numeric) rejected outright.
    - Numeric-numeric cross-subtype (int vs float) allowed.
    - Boolean-numeric comparison deliberately NOT allowed (no implicit
      bool-as-0/1).
    """
    if left == right:
        return
    if is_numeric(left) and is_numeric(right):
        return
    if left in TEMPORAL_TYPES and right in TEMPORAL_TYPES:
        return
    if left in STRING_TYPES and right in STRING_TYPES:
        return
    raise TypeMismatch(
        f"Cannot compare incompatible types: {left.value} vs {right.value}."
    )


# ---------------------------------------------------------------------------
# Expression nodes
# ---------------------------------------------------------------------------


class RLiteral(BaseModel):
    """A constant value used in comparisons and arithmetic."""

    node_type: Literal["literal"] = "literal"
    value: float | int | str | bool = Field(
        description="The constant value. Numeric, string, or boolean."
    )

    def _validate(self) -> List[str]:
        return []

    def inferred_type(self) -> DataType:
        if isinstance(self.value, bool):
            return DataType.BOOLEAN
        if isinstance(self.value, int):
            return DataType.INTEGER
        if isinstance(self.value, float):
            return DataType.FLOAT
        return DataType.VARCHAR


class RColumnRef(BaseModel):
    """A reference to a column by name, unqualified.

    No table qualifier -- resolved against whatever the enclosing Relation
    makes accessible at this point (Section 7.4). Its type is NOT stored
    on the node; infer_type() looks it up in the caller-supplied context.
    """

    node_type: Literal["column_ref"] = "column_ref"
    name: str = Field(description="Column name (lower_snake_case).")

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if not self.name.strip():
            errors.append("RColumnRef.name cannot be empty.")
        return errors


class RArithmetic(BaseModel):
    """Binary arithmetic expression: left op right.

    Supported ops: +, -, *, /. Both operands must be numeric (Section
    7.1) -- arithmetic on strings/booleans/temporal types is rejected.
    """

    node_type: Literal["arithmetic"] = "arithmetic"
    op: Literal["+", "-", "*", "/"] = Field(description="Arithmetic operator.")
    left: "RExprUnion" = Field(description="Left operand.")
    right: "RExprUnion" = Field(description="Right operand.")

    def _validate(self) -> List[str]:
        errors: List[str] = []
        errors.extend(f"RArithmetic.left: {e}" for e in _validate_expr(self.left))
        errors.extend(f"RArithmetic.right: {e}" for e in _validate_expr(self.right))
        return errors


class RAggregateRef(BaseModel):
    """Reference to an ONAggregate's result, by alias.

    The alias must match a real ONAggregate.alias declared somewhere in
    the enclosing Relation tree -- checked in relation/validate.py, not
    here (this node only knows its own alias string).
    """

    node_type: Literal["aggregate_ref"] = "aggregate_ref"
    alias: str = Field(description="The alias of the aggregate in the Relation tree.")

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if not self.alias.strip():
            errors.append("RAggregateRef.alias cannot be empty.")
        return errors


RExprUnion = Annotated[
    Union[RLiteral, RColumnRef, RArithmetic, RAggregateRef],
    Field(discriminator="node_type"),
]

RArithmetic.model_rebuild()


# ---------------------------------------------------------------------------
# Bottom-up type inference
# ---------------------------------------------------------------------------


def infer_type(
    expr: "RExprUnion",
    column_types: dict[str, DataType],
) -> DataType:
    """Bottom-up type inference for one expression node.

    `column_types` is the caller-supplied context mapping every name an
    RColumnRef/RAggregateRef could reference (both raw columns AND
    aggregate aliases) to its inferred DataType -- built from the
    enclosing Relation's synthesized schema (Section 4). Raises
    TypeMismatch (never a bare KeyError) if a name isn't found or an
    arithmetic operand isn't numeric.
    """
    if isinstance(expr, RLiteral):
        return expr.inferred_type()
    if isinstance(expr, RColumnRef):
        if expr.name not in column_types:
            raise TypeMismatch(f"Unknown column reference: '{expr.name}'.")
        return column_types[expr.name]
    if isinstance(expr, RAggregateRef):
        if expr.alias not in column_types:
            raise TypeMismatch(f"Unknown aggregate alias reference: '{expr.alias}'.")
        return column_types[expr.alias]
    if isinstance(expr, RArithmetic):
        left_type = infer_type(expr.left, column_types)
        right_type = infer_type(expr.right, column_types)
        if not is_numeric(left_type) or not is_numeric(right_type):
            raise TypeMismatch(
                f"RArithmetic requires numeric operands, got "
                f"{left_type.value} {expr.op} {right_type.value}."
            )
        return numeric_promote(left_type, right_type)
    raise TypeMismatch(f"Unknown expression node type: {type(expr).__name__}")


def _validate_expr(node: "RExprUnion") -> List[str]:
    """Recursively validate an expression node's own structural rules
    (NOT type-checking, which needs external context -- see infer_type)."""
    if isinstance(node, RLiteral):
        return node._validate()
    if isinstance(node, RColumnRef):
        return node._validate()
    if isinstance(node, RArithmetic):
        return node._validate()
    if isinstance(node, RAggregateRef):
        return node._validate()
    return [f"Unknown expression type: {type(node).__name__}"]
