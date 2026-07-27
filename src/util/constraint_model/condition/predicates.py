"""Predicate layer: RComparison, RAnd, ROr, RNot, RBetween, RInSet,
RNotInSet, RIfThen.

See RELATION_CONDITION_CONSTRAINT_DESIGN.md Section 7.2. RExists/
RNotExists and the '~' RComparison operator are deliberately absent --
both realistic use cases for RExists/RNotExists already reduce more
precisely to existing primitives (a non-nullable FK's guarantee; Fanout's
min_fanout >= 1), and '~' has no remaining use now that Distributed
absorbed the "distributionally pinned to" semantics as its own cohesive
term (Section 8.1).
"""

from __future__ import annotations

from typing import Annotated, Dict, List, Literal, Union

from pydantic import BaseModel, Field

from src.pipeline.stage2.models.data_types import DataType
from src.util.constraint_model.condition.expressions import (
    RExprUnion,
    TypeMismatch,
    check_comparable,
    infer_type,
)


class RComparison(BaseModel):
    """Binary comparison: left op right. Ops: <, <=, =, !=, >=, >.

    No '~' operator (see module docstring).
    """

    node_type: Literal["comparison"] = "comparison"
    op: Literal["<", "<=", "=", "!=", ">=", ">"] = Field(
        description="Comparison operator."
    )
    left: RExprUnion = Field(description="Left operand.")
    right: RExprUnion = Field(description="Right operand.")

    def _validate(self) -> List[str]:
        errors: List[str] = []
        errors.extend(f"RComparison.left: {e}" for e in _validate_expr(self.left))
        errors.extend(f"RComparison.right: {e}" for e in _validate_expr(self.right))
        return errors


class RAnd(BaseModel):
    """Logical AND of two or more predicates."""

    node_type: Literal["and"] = "and"
    operands: List["RPredicateUnion"] = Field(
        min_length=2, description="Two or more predicates that must all be true."
    )

    def _validate(self) -> List[str]:
        errors: List[str] = []
        for i, op in enumerate(self.operands):
            errors.extend(f"RAnd.operands[{i}]: {e}" for e in _validate_predicate(op))
        return errors


class ROr(BaseModel):
    """Logical OR of two or more predicates."""

    node_type: Literal["or"] = "or"
    operands: List["RPredicateUnion"] = Field(
        min_length=2, description="Two or more predicates, at least one must be true."
    )

    def _validate(self) -> List[str]:
        errors: List[str] = []
        for i, op in enumerate(self.operands):
            errors.extend(f"ROr.operands[{i}]: {e}" for e in _validate_predicate(op))
        return errors


class RNot(BaseModel):
    """Logical negation of a predicate."""

    node_type: Literal["not"] = "not"
    operand: "RPredicateUnion" = Field(description="The predicate to negate.")

    def _validate(self) -> List[str]:
        return _validate_predicate(self.operand)


class RBetween(BaseModel):
    """Range check: expr BETWEEN low AND high (inclusive on both ends)."""

    node_type: Literal["between"] = "between"
    expr: RExprUnion = Field(description="The expression to check.")
    low: RExprUnion = Field(description="Lower bound (inclusive).")
    high: RExprUnion = Field(description="Upper bound (inclusive).")

    def _validate(self) -> List[str]:
        errors: List[str] = []
        errors.extend(f"RBetween.expr: {e}" for e in _validate_expr(self.expr))
        errors.extend(f"RBetween.low: {e}" for e in _validate_expr(self.low))
        errors.extend(f"RBetween.high: {e}" for e in _validate_expr(self.high))
        return errors


class RInSet(BaseModel):
    """Set membership: expr IN (val1, val2, ...)."""

    node_type: Literal["in_set"] = "in_set"
    expr: RExprUnion = Field(description="The expression to check.")
    values: List[str | int | float] = Field(
        min_length=1, description="The set of allowed values."
    )

    def _validate(self) -> List[str]:
        errors: List[str] = []
        errors.extend(f"RInSet.expr: {e}" for e in _validate_expr(self.expr))
        if len(self.values) != len(set(self.values)):
            errors.append("RInSet.values contains duplicates.")
        return errors


class RNotInSet(BaseModel):
    """Set exclusion: expr NOT IN (val1, val2, ...)."""

    node_type: Literal["not_in_set"] = "not_in_set"
    expr: RExprUnion = Field(description="The expression to check.")
    values: List[str | int | float] = Field(
        min_length=1, description="The set of excluded values."
    )

    def _validate(self) -> List[str]:
        errors: List[str] = []
        errors.extend(f"RNotInSet.expr: {e}" for e in _validate_expr(self.expr))
        if len(self.values) != len(set(self.values)):
            errors.append("RNotInSet.values contains duplicates.")
        return errors


class RIfThen(BaseModel):
    """Conditional: IF antecedent THEN consequent.

    Recursive; practical nesting limit is 2 levels (LLM reliability) --
    beyond that, flatten into compound antecedents.
    """

    node_type: Literal["if_then"] = "if_then"
    antecedent: "RPredicateUnion" = Field(
        description="Must be true for consequent to apply."
    )
    consequent: "RPredicateUnion" = Field(
        description="Applies when antecedent is true."
    )

    def _validate(self) -> List[str]:
        errors: List[str] = []
        errors.extend(
            f"RIfThen.antecedent: {e}" for e in _validate_predicate(self.antecedent)
        )
        errors.extend(
            f"RIfThen.consequent: {e}" for e in _validate_predicate(self.consequent)
        )
        return errors


RPredicateUnion = Annotated[
    Union[RComparison, RAnd, ROr, RNot, RBetween, RInSet, RNotInSet, RIfThen],
    Field(discriminator="node_type"),
]

RAnd.model_rebuild()
ROr.model_rebuild()
RNot.model_rebuild()
RIfThen.model_rebuild()


# ---------------------------------------------------------------------------
# Recursive structural validation
# ---------------------------------------------------------------------------


def _validate_expr(node: RExprUnion) -> List[str]:
    from src.util.constraint_model.condition.expressions import _validate_expr as impl

    return impl(node)


def _validate_predicate(node: "RPredicateUnion") -> List[str]:
    if isinstance(node, RComparison):
        return node._validate()
    if isinstance(node, RAnd):
        return node._validate()
    if isinstance(node, ROr):
        return node._validate()
    if isinstance(node, RNot):
        return node._validate()
    if isinstance(node, RBetween):
        return node._validate()
    if isinstance(node, RInSet):
        return node._validate()
    if isinstance(node, RNotInSet):
        return node._validate()
    if isinstance(node, RIfThen):
        return node._validate()
    return [f"Unknown predicate type: {type(node).__name__}"]


def validate_predicate_tree(root: "RPredicateUnion") -> List[str]:
    """Public entry point for recursive structural validation (NOT
    type-checking -- see validate_predicate_types below)."""
    return _validate_predicate(root)


# ---------------------------------------------------------------------------
# Column extraction (used by nullability-narrowing and cross-checks)
# ---------------------------------------------------------------------------


def extract_columns(node: "RPredicateUnion") -> set[str]:
    """Recursively extract all column names (RColumnRef.name) referenced
    in a predicate tree. Does not descend into RAggregateRef (an aggregate
    alias is not a raw column)."""
    cols: set[str] = set()
    _collect_columns_pred(node, cols)
    return cols


def _collect_columns_expr(node: RExprUnion, out: set[str]) -> None:
    from src.util.constraint_model.condition.expressions import RArithmetic, RColumnRef

    if isinstance(node, RColumnRef):
        out.add(node.name)
    elif isinstance(node, RArithmetic):
        _collect_columns_expr(node.left, out)
        _collect_columns_expr(node.right, out)


def _collect_columns_pred(node: "RPredicateUnion", out: set[str]) -> None:
    if isinstance(node, RComparison):
        _collect_columns_expr(node.left, out)
        _collect_columns_expr(node.right, out)
    elif isinstance(node, (RAnd, ROr)):
        for op in node.operands:
            _collect_columns_pred(op, out)
    elif isinstance(node, RNot):
        _collect_columns_pred(node.operand, out)
    elif isinstance(node, RBetween):
        _collect_columns_expr(node.expr, out)
        _collect_columns_expr(node.low, out)
        _collect_columns_expr(node.high, out)
    elif isinstance(node, (RInSet, RNotInSet)):
        _collect_columns_expr(node.expr, out)
    elif isinstance(node, RIfThen):
        _collect_columns_pred(node.antecedent, out)
        _collect_columns_pred(node.consequent, out)


# ---------------------------------------------------------------------------
# Type-checking (Section 7.2) -- needs the column-type context, so lives
# as free functions rather than node methods (mirrors infer_type).
# ---------------------------------------------------------------------------


def validate_predicate_types(
    node: "RPredicateUnion", column_types: Dict[str, DataType]
) -> List[str]:
    """Recursively type-check a predicate tree against a column-type
    context (Section 7.2's comparison compatibility rules). Returns a list
    of human-readable errors; never raises -- TypeMismatch is caught here
    and turned into a message."""
    errors: List[str] = []
    if isinstance(node, RComparison):
        try:
            left_t = infer_type(node.left, column_types)
            right_t = infer_type(node.right, column_types)
            check_comparable(left_t, right_t)
        except TypeMismatch as e:
            errors.append(f"RComparison: {e}")
    elif isinstance(node, RBetween):
        try:
            expr_t = infer_type(node.expr, column_types)
            low_t = infer_type(node.low, column_types)
            high_t = infer_type(node.high, column_types)
            check_comparable(expr_t, low_t)
            check_comparable(expr_t, high_t)
        except TypeMismatch as e:
            errors.append(f"RBetween: {e}")
    elif isinstance(node, (RAnd, ROr)):
        for op in node.operands:
            errors.extend(validate_predicate_types(op, column_types))
    elif isinstance(node, RNot):
        errors.extend(validate_predicate_types(node.operand, column_types))
    elif isinstance(node, RIfThen):
        errors.extend(validate_predicate_types(node.antecedent, column_types))
        errors.extend(validate_predicate_types(node.consequent, column_types))
    # RInSet/RNotInSet: value-vs-column type checking is deliberately
    # loose here (values is a bare list of str|int|float, not RExprUnion) --
    # left to condition/validate.py if stricter checking is ever needed.
    return errors
