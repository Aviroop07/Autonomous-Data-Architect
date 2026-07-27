"""ON clause models for the constraint representation.

The ON clause defines the table/join/aggregate context for a constraint.
It is a recursive hybrid structure: at any level, it can be a structured
ONNode object or a valid SQL DML string (FROM clause). The normalize()
function recursively converts everything to pure typed objects.

Architecture:
    ONBaseTable  -> a single table from the schema
    ONJoin       -> a join between two ON nodes
    ONAggregate  -> an aggregate (SUM/COUNT/etc.) over an ON node
    ONSubquery   -> a raw SQL subquery (transitional; normalized away)

Design decisions:
    - ONJoin.on is a list of JoinCondition (equi-joins only for now).
    - ONAggregate.alias is REQUIRED (the condition references aggregates by alias).
    - ONAggregate.group_by is a list of column names (composite group-by supported).
    - ONSubquery.sql is validated as parseable SQL at construction time.
    - All names are validated (UPPER_SNAKE for tables, lower_snake for columns).
"""

from __future__ import annotations

import re
from typing import Annotated, List, Literal, Optional, Union

import sqlglot
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Name validation constants
# ---------------------------------------------------------------------------

_UPPER_SNAKE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_LOWER_SNAKE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_TABLE_ALIAS_RE = re.compile(r"^[a-z][a-z0-9_]*$")

_VALID_AGG_FNS = {"SUM", "COUNT", "AVG", "MAX", "MIN", "MEDIAN"}


def _is_upper_snake(name: str) -> bool:
    return bool(_UPPER_SNAKE_RE.fullmatch(name))


def _is_lower_snake(name: str) -> bool:
    return bool(_LOWER_SNAKE_RE.fullmatch(name))


def _is_valid_alias(name: str) -> bool:
    return bool(_TABLE_ALIAS_RE.fullmatch(name))


# ---------------------------------------------------------------------------
# JoinCondition
# ---------------------------------------------------------------------------


class JoinCondition(BaseModel):
    """An equi-join predicate: left = right.

    Both sides are column references in the form "table.column" or just
    "column" (resolved against the ON context during validation).
    """

    left: str = Field(
        description="Left side of the join condition. 'table.column' or 'column'."
    )
    right: str = Field(
        description="Right side of the join condition. 'table.column' or 'column'."
    )
    op: Literal["="] = Field(
        default="=",
        description="Join operator. Only equi-joins (=) supported for now.",
    )

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if not self.left.strip():
            errors.append("JoinCondition.left cannot be empty.")
        if not self.right.strip():
            errors.append("JoinCondition.right cannot be empty.")
        if self.left == self.right:
            errors.append(
                f"JoinCondition: left and right are identical ('{self.left}')."
            )
        return errors


# ---------------------------------------------------------------------------
# ON node types
# ---------------------------------------------------------------------------


class ONBaseTable(BaseModel):
    """A single base table from the schema."""

    type: Literal["table"] = "table"
    name: str = Field(description="Table name in UPPER_SNAKE_CASE.")

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if not self.name.strip():
            errors.append("ONBaseTable.name cannot be empty.")
        if not _is_upper_snake(self.name):
            errors.append(
                f"ONBaseTable.name must be UPPER_SNAKE_CASE, got '{self.name}'."
            )
        return errors


class ONJoin(BaseModel):
    """A join between two ON nodes.

    left and right are ONNode (recursive). on is a non-empty list of
    JoinCondition. alias is optional (used when the join result needs a
    name for the condition to reference).
    """

    type: Literal["join"] = "join"
    left: ONNode
    right: ONNode
    on: List[JoinCondition] = Field(
        min_length=1,
        description="Non-empty list of join conditions (equi-joins).",
    )
    alias: Optional[str] = Field(
        default=None,
        description="Optional alias for the join result. Must be lower_snake_case.",
    )

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if self.alias is not None and self.alias.strip():
            if not _is_valid_alias(self.alias):
                errors.append(
                    f"ONJoin.alias must be lower_snake_case, got '{self.alias}'."
                )
        for i, cond in enumerate(self.on):
            cond_errors = cond._validate()
            errors.extend(f"ONJoin.on[{i}]: {e}" for e in cond_errors)
        # Recursively validate children
        for side in ("left", "right"):
            child = getattr(self, side)
            child_errors = _validate_on_node(child)
            errors.extend(f"ONJoin.{side}: {e}" for e in child_errors)
        return errors


class ONAggregate(BaseModel):
    """An aggregate over an ON node.

    Represents: AGG(source, fn(column), GROUP_BY=group_by) AS alias

    The alias is REQUIRED because conditions reference aggregate results
    by alias (via RAggregateRef). The group_by columns define the
    per-group aggregation.
    """

    type: Literal["aggregate"] = "aggregate"
    source: ONNode = Field(description="The table/join being aggregated.")
    fn: Literal["SUM", "COUNT", "AVG", "MAX", "MIN", "MEDIAN"] = Field(
        description="Aggregate function."
    )
    column: str = Field(description="Column name to aggregate. Use '*' for COUNT(*).")
    group_by: Optional[List[str]] = Field(
        default=None,
        description="Column names to group by. None = aggregate entire source.",
    )
    alias: str = Field(
        description="Name for the resulting aggregate column. lower_snake_case. "
        "The condition references this by name."
    )

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if not self.alias.strip():
            errors.append("ONAggregate.alias cannot be empty.")
        if not _is_valid_alias(self.alias):
            errors.append(
                f"ONAggregate.alias must be lower_snake_case, got '{self.alias}'."
            )
        if self.fn not in _VALID_AGG_FNS:
            errors.append(
                f"ONAggregate.fn must be one of {_VALID_AGG_FNS}, got '{self.fn}'."
            )
        if self.column != "*" and not self.column.strip():
            errors.append("ONAggregate.column cannot be empty (use '*' for COUNT(*)).")
        if self.column != "*" and not _is_lower_snake(self.column):
            errors.append(
                f"ONAggregate.column must be lower_snake_case or '*', got '{self.column}'."
            )
        if self.group_by is not None:
            for i, gb in enumerate(self.group_by):
                if not _is_lower_snake(gb):
                    errors.append(
                        f"ONAggregate.group_by[{i}] must be lower_snake_case, got '{gb}'."
                    )
            # Check for duplicates
            if len(self.group_by) != len(set(self.group_by)):
                errors.append("ONAggregate.group_by contains duplicate columns.")
        # Recursively validate source
        source_errors = _validate_on_node(self.source)
        errors.extend(f"ONAggregate.source: {e}" for e in source_errors)
        return errors


class ONFanout(BaseModel):
    """COUNT(child rows) GROUP BY parent.pk, via LEFT JOIN semantics that
    preserve EVERY parent row -- including parents with zero matching children
    (a customer with no orders still gets a count of 0, not silently omitted).

    This is a genuinely different guarantee than ONJoin/ONAggregate composition
    provides: the FK-PK-restricted join model only guarantees the CHILD side
    is fully covered (every child has exactly one parent, via the required-FK
    assumption) -- it says nothing about the PARENT side being fully covered.
    Computing a true fanout distribution (the thing a fact like "each customer
    places 1-20 orders" is actually describing) needs the parent-side guarantee
    instead, which an ordinary INNER-JOIN-shaped ONJoin+ONAggregate would NOT
    provide (a customer with 0 orders would never appear in ORDER at all, so a
    plain GROUP BY would silently drop them, overstating the average)."""

    type: Literal["fanout"] = "fanout"
    parent_table: str = Field(description="Parent table in UPPER_SNAKE_CASE.")
    child_table: str = Field(description="Child table in UPPER_SNAKE_CASE.")
    fk_column: str = Field(
        description="FK column in the child table pointing to the parent."
    )

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if not self.parent_table.strip():
            errors.append("ONFanout.parent_table cannot be empty.")
        if not _is_upper_snake(self.parent_table):
            errors.append(
                f"ONFanout.parent_table must be UPPER_SNAKE_CASE, got '{self.parent_table}'."
            )
        if not self.child_table.strip():
            errors.append("ONFanout.child_table cannot be empty.")
        if not _is_upper_snake(self.child_table):
            errors.append(
                f"ONFanout.child_table must be UPPER_SNAKE_CASE, got '{self.child_table}'."
            )
        if not self.fk_column.strip():
            errors.append("ONFanout.fk_column cannot be empty.")
        if not _is_lower_snake(self.fk_column):
            errors.append(
                f"ONFanout.fk_column must be lower_snake_case, got '{self.fk_column}'."
            )
        return errors


class ONSubquery(BaseModel):
    """A raw SQL subquery (FROM clause).

    Transitional: the hybrid parser emits these from SQL strings, then
    normalizes them into structured ONBaseTable/ONJoin/ONAggregate nodes.
    Stored form should never contain ONSubquery after normalization.

    The sql field is validated as parseable SQL at construction time.
    """

    type: Literal["subquery"] = "subquery"
    sql: str = Field(description="Valid SQL DML (FROM clause or subquery).")

    @model_validator(mode="after")
    def _validate_sql_parseable(self) -> ONSubquery:
        if not self.sql.strip():
            raise ValueError("ONSubquery.sql cannot be empty.")
        try:
            sqlglot.parse_one(self.sql)
        except sqlglot.errors.ParseError as e:
            raise ValueError(f"ONSubquery.sql is not valid SQL: {e}") from e
        return self

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if not self.sql.strip():
            errors.append("ONSubquery.sql cannot be empty.")
        return errors


# ---------------------------------------------------------------------------
# Union type + recursive validation
# ---------------------------------------------------------------------------

ONNode = Annotated[
    Union[ONBaseTable, ONJoin, ONAggregate, ONFanout, ONSubquery],
    Field(discriminator="type"),
]


def _validate_on_node(node: ONNode) -> List[str]:
    """Recursively validate an ON node tree."""
    if isinstance(node, ONBaseTable):
        return node._validate()
    elif isinstance(node, ONJoin):
        return node._validate()
    elif isinstance(node, ONAggregate):
        return node._validate()
    elif isinstance(node, ONFanout):
        return node._validate()
    elif isinstance(node, ONSubquery):
        return node._validate()
    else:
        return [f"Unknown ON node type: {type(node).__name__}"]


# ---------------------------------------------------------------------------
# Table extraction (for sharder co-location)
# ---------------------------------------------------------------------------


def extract_tables(node: ONNode) -> set[str]:
    """Recursively extract all base table names from an ON tree.

    Used by the ILP sharder to determine which tables must co-locate
    for a constraint to be verifiable.
    """
    if isinstance(node, ONBaseTable):
        return {node.name}
    elif isinstance(node, ONJoin):
        return extract_tables(node.left) | extract_tables(node.right)
    elif isinstance(node, ONAggregate):
        return extract_tables(node.source)
    elif isinstance(node, ONFanout):
        return {node.parent_table, node.child_table}
    elif isinstance(node, ONSubquery):
        # Best-effort: parse SQL and extract table names
        try:
            parsed = sqlglot.parse_one(node.sql)
            return {
                table.name for table in parsed.find_all(sqlglot.exp.Table) if table.name
            }
        except Exception:
            return set()
    return set()
