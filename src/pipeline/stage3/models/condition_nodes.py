"""R-AST condition nodes for the constraint representation.

The condition is a pure typed AST (Abstract Syntax Tree) -- no SQL strings
allowed. Every predicate is a node with typed children. Column references
use only the column name (no table qualifier); the ON context resolves
ambiguity.

Architecture:
    Expression layer:
        RLiteral        -> constant value (float, str, bool)
        RColumnRef      -> column name (resolved against ON context)
        RArithmetic     -> binary arithmetic (+, -, *, /)
        RAggregateRef   -> reference to an ON-defined aggregate (by alias)

    Predicate layer:
        RComparison     -> binary comparison (<, <=, =, !=, >=, >, ~)
        RAnd / ROr      -> logical connectives
        RNot            -> logical negation
        RBetween        -> range check (inclusive)
        RInSet / RNotInSet -> set membership
        RIfThen         -> conditional (recursive)
        RExists / RNotExists -> set-containment via subquery

    Supporting types:
        SubqueryRef     -> reference to a table with join condition (for EXISTS)

Design decisions:
    - RColumnRef has NO table field. Column names are unambiguous within
      the ON context. If ambiguous, the extraction agent table-qualifies
      in the ON clause, not the condition.
    - RAggregateRef references aggregates BY ALIAS (defined in ONAggregate.alias).
      The condition never re-computes aggregates.
    - RComparison.op includes '~' for distribution pins (handled by
      DistributionConstraint, but available for general use).
    - RIfThen is recursive: antecedent and consequent are both RPredicate.
    - SubqueryRef uses string column references (e.g., "PRESCRIPTION.diagnosis_id")
      because the subquery targets a different table context.
"""

from __future__ import annotations

from typing import Annotated, List, Literal, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Expression nodes (leaf values in predicates)
# ---------------------------------------------------------------------------


class RLiteral(BaseModel):
    """A constant value used in comparisons and arithmetic."""

    node_type: Literal["literal"] = "literal"
    value: float | int | str | bool = Field(
        description="The constant value. Numeric, string, or boolean."
    )

    def _validate(self) -> List[str]:
        return []


class RColumnRef(BaseModel):
    """A reference to a column by name.

    No table qualifier -- the ON context resolves which table the column
    belongs to. If the ON is a join with ambiguous column names, the
    extraction agent must use table-qualified names in the ON clause's
    SELECT list, and the condition references the unambiguous result.
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

    Supported ops: +, -, *, /
    Nesting is unlimited but practical limit is 3 levels (LLM reliability).
    """

    node_type: Literal["arithmetic"] = "arithmetic"
    op: Literal["+", "-", "*", "/"] = Field(description="Arithmetic operator.")
    left: RExprUnion = Field(description="Left operand.")
    right: RExprUnion = Field(description="Right operand.")

    def _validate(self) -> List[str]:
        errors: List[str] = []
        left_errors = _validate_expr(self.left)
        right_errors = _validate_expr(self.right)
        errors.extend(f"RArithmetic.left: {e}" for e in left_errors)
        errors.extend(f"RArithmetic.right: {e}" for e in right_errors)
        return errors


class RAggregateRef(BaseModel):
    """Reference to an aggregate column defined in the ON clause.

    The alias must match an ONAggregate.alias in the constraint's ON tree.
    The condition never re-computes aggregates -- it references them by name.
    """

    node_type: Literal["aggregate_ref"] = "aggregate_ref"
    alias: str = Field(
        description="The alias of the aggregate column in the ON clause."
    )

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if not self.alias.strip():
            errors.append("RAggregateRef.alias cannot be empty.")
        return errors


# Expression union (forward reference resolved below)
RExprUnion = Annotated[
    Union[RLiteral, RColumnRef, RArithmetic, RAggregateRef],
    Field(discriminator="node_type"),
]


# ---------------------------------------------------------------------------
# Predicate nodes (condition tree)
# ---------------------------------------------------------------------------


class RComparison(BaseModel):
    """Binary comparison: left op right.

    Ops: <, <=, =, !=, >=, > (standard), ~ (distribution pin, used by
    DistributionConstraint but available here for general use).
    """

    node_type: Literal["comparison"] = "comparison"
    op: Literal["<", "<=", "=", "!=", ">=", ">", "~"] = Field(
        description="Comparison operator."
    )
    left: RExprUnion = Field(description="Left operand.")
    right: RExprUnion = Field(description="Right operand.")

    def _validate(self) -> List[str]:
        errors: List[str] = []
        left_errors = _validate_expr(self.left)
        right_errors = _validate_expr(self.right)
        errors.extend(f"RComparison.left: {e}" for e in left_errors)
        errors.extend(f"RComparison.right: {e}" for e in right_errors)
        return errors


class RAnd(BaseModel):
    """Logical AND of multiple predicates."""

    node_type: Literal["and"] = "and"
    operands: List[RPredicate] = Field(
        min_length=2,
        description="Two or more predicates that must all be true.",
    )

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if len(self.operands) < 2:
            errors.append("RAnd must have at least 2 operands.")
        for i, op in enumerate(self.operands):
            op_errors = _validate_predicate(op)
            errors.extend(f"RAnd.operands[{i}]: {e}" for e in op_errors)
        return errors


class ROr(BaseModel):
    """Logical OR of multiple predicates."""

    node_type: Literal["or"] = "or"
    operands: List[RPredicate] = Field(
        min_length=2,
        description="Two or more predicates, at least one must be true.",
    )

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if len(self.operands) < 2:
            errors.append("ROr must have at least 2 operands.")
        for i, op in enumerate(self.operands):
            op_errors = _validate_predicate(op)
            errors.extend(f"ROr.operands[{i}]: {e}" for e in op_errors)
        return errors


class RNot(BaseModel):
    """Logical negation of a predicate."""

    node_type: Literal["not"] = "not"
    operand: RPredicate = Field(description="The predicate to negate.")

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
        min_length=1,
        description="The set of allowed values.",
    )

    def _validate(self) -> List[str]:
        errors: List[str] = []
        errors.extend(f"RInSet.expr: {e}" for e in _validate_expr(self.expr))
        if not self.values:
            errors.append("RInSet.values cannot be empty.")
        if len(self.values) != len(set(self.values)):
            errors.append("RInSet.values contains duplicates.")
        return errors


class RNotInSet(BaseModel):
    """Set exclusion: expr NOT IN (val1, val2, ...)."""

    node_type: Literal["not_in_set"] = "not_in_set"
    expr: RExprUnion = Field(description="The expression to check.")
    values: List[str | int | float] = Field(
        min_length=1,
        description="The set of excluded values.",
    )

    def _validate(self) -> List[str]:
        errors: List[str] = []
        errors.extend(f"RNotInSet.expr: {e}" for e in _validate_expr(self.expr))
        if not self.values:
            errors.append("RNotInSet.values cannot be empty.")
        if len(self.values) != len(set(self.values)):
            errors.append("RNotInSet.values contains duplicates.")
        return errors


class RIfThen(BaseModel):
    """Conditional: IF antecedent THEN consequent.

    Recursive: both antecedent and consequent are RPredicate. Practical
    limit is 2 levels of nesting (LLM reliability). Beyond that, flatten
    into compound antecedents: IF a AND b AND c THEN ...
    """

    node_type: Literal["if_then"] = "if_then"
    antecedent: RPredicate = Field(
        description="The condition (must be true for consequent to apply)."
    )
    consequent: RPredicate = Field(
        description="The rule that applies when antecedent is true."
    )

    def _validate(self) -> List[str]:
        errors: List[str] = []
        ante_errors = _validate_predicate(self.antecedent)
        cons_errors = _validate_predicate(self.consequent)
        errors.extend(f"RIfThen.antecedent: {e}" for e in ante_errors)
        errors.extend(f"RIfThen.consequent: {e}" for e in cons_errors)
        return errors


class SubqueryRef(BaseModel):
    """Reference to another table with a join condition (for EXISTS).

    Used by RExists/RNotExists to express set-containment constraints:
    "every row in the ON context must (or must not) have a matching row
    in from_table where source_table.column = target_table.column".
    """

    from_table: str = Field(
        description="The table to check for existence (UPPER_SNAKE_CASE)."
    )
    where_left: str = Field(
        description="Left side of the WHERE condition. 'TABLE.column' format."
    )
    where_right: str = Field(
        description="Right side of the WHERE condition. 'TABLE.column' format."
    )

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if not self.from_table.strip():
            errors.append("SubqueryRef.from_table cannot be empty.")
        if not self.where_left.strip():
            errors.append("SubqueryRef.where_left cannot be empty.")
        if not self.where_right.strip():
            errors.append("SubqueryRef.where_right cannot be empty.")
        if self.where_left == self.where_right:
            errors.append(
                f"SubqueryRef: where_left and where_right are identical "
                f"('{self.where_left}')."
            )
        return errors


class RExists(BaseModel):
    """Set-containment: EXISTS subquery.

    "Every row in the ON context must have at least one matching row in
    the subquery's target table."
    """

    node_type: Literal["exists"] = "exists"
    subquery: SubqueryRef = Field(description="The existence check.")

    def _validate(self) -> List[str]:
        return self.subquery._validate()


class RNotExists(BaseModel):
    """Set exclusion: NOT EXISTS subquery.

    "No row in the ON context should have a matching row in the
    subquery's target table." (Rare; usually indicates a data quality
    constraint.)
    """

    node_type: Literal["not_exists"] = "not_exists"
    subquery: SubqueryRef = Field(description="The non-existence check.")

    def _validate(self) -> List[str]:
        return self.subquery._validate()


# ---------------------------------------------------------------------------
# Predicate union (forward reference resolved below)
# ---------------------------------------------------------------------------

RPredicate = Annotated[
    Union[
        RComparison,
        RAnd,
        ROr,
        RNot,
        RBetween,
        RInSet,
        RNotInSet,
        RIfThen,
        RExists,
        RNotExists,
    ],
    Field(discriminator="node_type"),
]


# ---------------------------------------------------------------------------
# Recursive validation helpers
# ---------------------------------------------------------------------------


def _validate_expr(node: RExprUnion) -> List[str]:
    """Recursively validate an expression node."""
    if isinstance(node, RLiteral):
        return node._validate()
    elif isinstance(node, RColumnRef):
        return node._validate()
    elif isinstance(node, RArithmetic):
        return node._validate()
    elif isinstance(node, RAggregateRef):
        return node._validate()
    return [f"Unknown expression type: {type(node).__name__}"]


def _validate_predicate(node: RPredicate) -> List[str]:
    """Recursively validate a predicate node."""
    if isinstance(node, RComparison):
        return node._validate()
    elif isinstance(node, RAnd):
        return node._validate()
    elif isinstance(node, ROr):
        return node._validate()
    elif isinstance(node, RNot):
        return node._validate()
    elif isinstance(node, RBetween):
        return node._validate()
    elif isinstance(node, RInSet):
        return node._validate()
    elif isinstance(node, RNotInSet):
        return node._validate()
    elif isinstance(node, RIfThen):
        return node._validate()
    elif isinstance(node, RExists):
        return node._validate()
    elif isinstance(node, RNotExists):
        return node._validate()
    return [f"Unknown predicate type: {type(node).__name__}"]


# ---------------------------------------------------------------------------
# Column extraction -- used by constraint_graph.py to determine how many
# distinct columns a condition references (e.g. to tell a single-column
# range constraint apart from a genuine multi-table cross-column one).
# ---------------------------------------------------------------------------


def extract_columns(node: RPredicate) -> set[str]:
    """Recursively extract all column names referenced in a condition tree."""
    cols: set[str] = set()
    _collect_columns_pred(node, cols)
    return cols


def _collect_columns_expr(node: RExprUnion, out: set[str]) -> None:
    if isinstance(node, RColumnRef):
        out.add(node.name)
    elif isinstance(node, RArithmetic):
        _collect_columns_expr(node.left, out)
        _collect_columns_expr(node.right, out)
    # RLiteral, RAggregateRef: no column names


def _collect_columns_pred(node: RPredicate, out: set[str]) -> None:
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
    # RExists/RNotExists: subquery references a different table context,
    # not a column of the current ON scope -- nothing to collect.
