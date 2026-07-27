"""Relation (ON) node types: BaseTable, Join, Aggregate, Filter, Project,
Fanout, plus the hybrid RawSQL escape hatch.

See RELATION_CONDITION_CONSTRAINT_DESIGN.md Section 3. Bidirectional
SQL<->object conversion (the sqlglot-backed bridge implied by RawSQL) is
NOT implemented here -- see relation/sql_bridge.py (separate, not yet
built). This module only defines the node shapes and their own
self-contained structural validation; cross-node validation (FK-PK join
checks, PK-never-dropped, alias collisions, provenance) lives in
relation/validate.py, which needs the bottom-up schema synthesis from
relation/schema.py to run at all.
"""

from __future__ import annotations

import re
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from src.util.constraint_model.condition.expressions import RExprUnion
from src.util.constraint_model.condition.predicates import RPredicateUnion

_UPPER_SNAKE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_LOWER_SNAKE_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# The locked aggregate-function catalogue (Section 3.3).
AggregateFn = Literal[
    "SUM",
    "COUNT",
    "AVG",
    "MAX",
    "MIN",
    "MEDIAN",
    "COUNT_DISTINCT",
    "STDDEV",
    "VARIANCE",
    "PERCENTILE",
    "MODE",
]


# Aggregate-function/column-type compatibility (SUM/AVG/STDDEV/VARIANCE need a
# numeric column; MAX/MIN/MEDIAN/PERCENTILE need an orderable one) requires the
# column's real type from the source's synthesized schema -- external context
# nodes.py's self-contained _validate() doesn't have. That check belongs in
# relation/validate.py (task #27), which runs after relation/schema.py exists.


def _is_upper_snake(name: str) -> bool:
    return bool(_UPPER_SNAKE_RE.fullmatch(name))


def _is_lower_snake(name: str) -> bool:
    return bool(_LOWER_SNAKE_RE.fullmatch(name))


class JoinCondition(BaseModel):
    """An equi-join predicate: left = right. Both sides are ALWAYS
    table-qualified column references ("TABLE.column"), unlike RColumnRef
    inside a Condition (Section 7.4's asymmetry)."""

    left: str = Field(description="Left side, 'TABLE.column' format.")
    right: str = Field(description="Right side, 'TABLE.column' format.")
    op: Literal["="] = Field(default="=", description="Only equi-joins supported.")

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if "." not in self.left:
            errors.append(f"JoinCondition.left '{self.left}' must be table-qualified.")
        if "." not in self.right:
            errors.append(
                f"JoinCondition.right '{self.right}' must be table-qualified."
            )
        if self.left == self.right:
            errors.append(
                f"JoinCondition: left and right are identical ('{self.left}')."
            )
        return errors


class BaseTable(BaseModel):
    """A single table from the schema."""

    type: Literal["base_table"] = "base_table"
    name: str = Field(description="Table name, UPPER_SNAKE_CASE.")
    alias: Optional[str] = Field(
        default=None, description="Optional alias (Section 3.4)."
    )

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if not _is_upper_snake(self.name):
            errors.append(
                f"BaseTable.name must be UPPER_SNAKE_CASE, got '{self.name}'."
            )
        if self.alias is not None and not _is_lower_snake(self.alias):
            errors.append(
                f"BaseTable.alias must be lower_snake_case, got '{self.alias}'."
            )
        return errors


class Join(BaseModel):
    """FK-PK-restricted join between two Relations (Section 3.1, 5).

    `on` currently must contain exactly one JoinCondition -- composite FKs
    are a deferred non-goal (Section 1); enforced in relation/validate.py,
    not here, so the node shape itself doesn't have to change later.
    """

    type: Literal["join"] = "join"
    left: "RelationUnion" = Field(description="Left operand.")
    right: "RelationUnion" = Field(description="Right operand.")
    on: List[JoinCondition] = Field(min_length=1, description="Join condition(s).")
    alias: Optional[str] = Field(default=None)

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if self.alias is not None and not _is_lower_snake(self.alias):
            errors.append(f"Join.alias must be lower_snake_case, got '{self.alias}'.")
        for i, cond in enumerate(self.on):
            errors.extend(f"Join.on[{i}]: {e}" for e in cond._validate())
        errors.extend(f"Join.left: {e}" for e in _validate_relation(self.left))
        errors.extend(f"Join.right: {e}" for e in _validate_relation(self.right))
        return errors


class Aggregate(BaseModel):
    """An aggregate over a source Relation (Section 3.1, 3.3).

    `fn_param` carries PERCENTILE's rank p (0-100); unused for every
    other function. `alias` is REQUIRED -- the Condition side references
    the aggregate's result by this name (RAggregateRef).
    """

    type: Literal["aggregate"] = "aggregate"
    source: "RelationUnion" = Field(description="The Relation being aggregated.")
    fn: AggregateFn = Field(description="Aggregate function.")
    column: str = Field(description="Column to aggregate. '*' only valid for COUNT.")
    group_by: Optional[List[str]] = Field(
        default=None, description="Columns to group by. None = aggregate whole source."
    )
    alias: str = Field(description="Required. The condition references this by name.")
    fn_param: Optional[float] = Field(
        default=None, description="PERCENTILE's rank p (0-100). Unused otherwise."
    )

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if not _is_lower_snake(self.alias):
            errors.append(
                f"Aggregate.alias must be lower_snake_case, got '{self.alias}'."
            )
        if self.column != "*" and not _is_lower_snake(self.column):
            errors.append(
                f"Aggregate.column must be lower_snake_case or '*', got '{self.column}'."
            )
        if self.column == "*" and self.fn != "COUNT":
            errors.append("Aggregate.column='*' is only valid for fn='COUNT'.")
        if self.fn == "PERCENTILE":
            if self.fn_param is None:
                errors.append(
                    "Aggregate.fn_param (the rank p) is required for PERCENTILE."
                )
            elif not (0 <= self.fn_param <= 100):
                errors.append(
                    f"Aggregate.fn_param must be in [0, 100], got {self.fn_param}."
                )
        elif self.fn_param is not None:
            errors.append(
                f"Aggregate.fn_param is only meaningful for PERCENTILE, not {self.fn}."
            )
        if self.group_by is not None:
            if len(self.group_by) != len(set(self.group_by)):
                errors.append("Aggregate.group_by contains duplicate columns.")
            for i, gb in enumerate(self.group_by):
                if not _is_lower_snake(gb):
                    errors.append(
                        f"Aggregate.group_by[{i}] must be lower_snake_case, got '{gb}'."
                    )
        errors.extend(f"Aggregate.source: {e}" for e in _validate_relation(self.source))
        return errors


class Filter(BaseModel):
    """Row-level filter over a source Relation (Section 3.1, 4.3, 4.4).

    Mints a mandatory selectivity-factor variable (relation/schema.py);
    narrows downstream nullability via the full three-valued-logic rule
    (condition/validate.py). HAVING is Filter(source=Aggregate(...)), no
    separate node.
    """

    type: Literal["filter"] = "filter"
    source: "RelationUnion" = Field(description="The Relation being filtered.")
    condition: RPredicateUnion = Field(description="The filter predicate.")
    alias: Optional[str] = Field(default=None)

    def _validate(self) -> List[str]:
        from src.util.constraint_model.condition.predicates import (
            validate_predicate_tree,
        )

        errors: List[str] = []
        if self.alias is not None and not _is_lower_snake(self.alias):
            errors.append(f"Filter.alias must be lower_snake_case, got '{self.alias}'.")
        errors.extend(
            f"Filter.condition: {e}" for e in validate_predicate_tree(self.condition)
        )
        errors.extend(f"Filter.source: {e}" for e in _validate_relation(self.source))
        return errors


class ProjectEntry(BaseModel):
    """One output column of a Project: an expression plus an alias.

    `alias` is REQUIRED whenever `expr` is not a bare RColumnRef (a
    computed/arithmetic entry) -- Section 4.5. A bare RColumnRef entry
    without an alias is a plain passthrough; with an alias, it's a rename
    (Section 4.1) that carries the source column's provenance forward.
    """

    expr: RExprUnion
    alias: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def _alias_required_for_computed_entries(self) -> "ProjectEntry":
        from src.util.constraint_model.condition.expressions import RColumnRef

        if not isinstance(self.expr, RColumnRef) and self.alias is None:
            raise ValueError(
                "ProjectEntry.alias is required for computed (non-passthrough) entries."
            )
        return self

    def output_name(self) -> str:
        """The name this entry is known by downstream: its alias if set,
        else (for a bare passthrough) the source column's own name."""
        from src.util.constraint_model.condition.expressions import RColumnRef

        if self.alias is not None:
            return self.alias
        assert isinstance(self.expr, RColumnRef)
        return self.expr.name


class Project(BaseModel):
    """Column projection over a source Relation (Section 3.1, 4.1, 4.5).

    Can NEVER drop primary-key columns -- enforced in relation/validate.py
    (needs the source's synthesized schema to know which columns are the
    PK), not here.
    """

    type: Literal["project"] = "project"
    source: "RelationUnion" = Field(description="The Relation being projected.")
    columns: List[ProjectEntry] = Field(min_length=1, description="Output column list.")
    alias: Optional[str] = Field(default=None)

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if self.alias is not None and not _is_lower_snake(self.alias):
            errors.append(
                f"Project.alias must be lower_snake_case, got '{self.alias}'."
            )
        names = [c.output_name() for c in self.columns]
        if len(names) != len(set(names)):
            errors.append("Project.columns contains duplicate output names/aliases.")
        errors.extend(f"Project.source: {e}" for e in _validate_relation(self.source))
        return errors


class Fanout(BaseModel):
    """COUNT(child rows) GROUP BY parent.pk, via LEFT JOIN semantics that
    preserve EVERY parent row -- including zero-child parents. Different
    guarantee than Join+Aggregate (Section 5): only Fanout may claim to
    literally be the parent table's own full identity.
    """

    type: Literal["fanout"] = "fanout"
    parent_table: str = Field(description="Parent table, UPPER_SNAKE_CASE.")
    child_table: str = Field(description="Child table, UPPER_SNAKE_CASE.")
    fk_column: str = Field(
        description="FK column on child_table pointing to parent_table."
    )
    alias: Optional[str] = Field(default=None)

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if not _is_upper_snake(self.parent_table):
            errors.append(
                f"Fanout.parent_table must be UPPER_SNAKE_CASE, got '{self.parent_table}'."
            )
        if not _is_upper_snake(self.child_table):
            errors.append(
                f"Fanout.child_table must be UPPER_SNAKE_CASE, got '{self.child_table}'."
            )
        if not _is_lower_snake(self.fk_column):
            errors.append(
                f"Fanout.fk_column must be lower_snake_case, got '{self.fk_column}'."
            )
        if self.alias is not None and not _is_lower_snake(self.alias):
            errors.append(f"Fanout.alias must be lower_snake_case, got '{self.alias}'.")
        return errors


class RawSQL(BaseModel):
    """A raw SQL string standing in for a structured Relation node.

    Must be a complete, valid SELECT statement -- the homogenization rule
    (Section 3.2): a base table expressed as SQL is "SELECT * FROM X",
    never a bare table name. Parsing/normalizing this into the structured
    node types above is relation/sql_bridge.py's job (NOT YET BUILT --
    this node type exists so the union has a slot for it, but nothing in
    this codebase can consume a RawSQL node yet).
    """

    type: Literal["raw_sql"] = "raw_sql"
    sql: str = Field(description="A complete SELECT statement.")

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if not self.sql.strip():
            errors.append("RawSQL.sql cannot be empty.")
        elif not self.sql.strip().upper().startswith("SELECT"):
            errors.append(
                "RawSQL.sql must be a complete SELECT statement (homogenization "
                "rule) -- a bare table name is not valid."
            )
        return errors


RelationUnion = Annotated[
    Union[BaseTable, Join, Aggregate, Filter, Project, Fanout, RawSQL],
    Field(discriminator="type"),
]

Join.model_rebuild()
Aggregate.model_rebuild()
Filter.model_rebuild()
Project.model_rebuild()


# ---------------------------------------------------------------------------
# Recursive structural validation dispatcher
# ---------------------------------------------------------------------------


def _validate_relation(node: "RelationUnion") -> List[str]:
    if isinstance(node, (BaseTable, Join, Aggregate, Filter, Project, Fanout, RawSQL)):
        return node._validate()
    return [f"Unknown Relation node type: {type(node).__name__}"]


def validate_relation_tree(root: "RelationUnion") -> List[str]:
    """Public entry point for recursive STRUCTURAL validation (naming
    conventions, node-local invariants). NOT the bottom-up schema/
    provenance/FK-PK-join validation -- see relation/validate.py."""
    return _validate_relation(root)


def extract_base_tables(node: "RelationUnion") -> set[str]:
    """Recursively extract every base table name reachable from a
    Relation tree (used for sharder co-location, mirroring on_nodes.py's
    extract_tables())."""
    if isinstance(node, BaseTable):
        return {node.name}
    if isinstance(node, Join):
        return extract_base_tables(node.left) | extract_base_tables(node.right)
    if isinstance(node, (Aggregate, Filter, Project)):
        return extract_base_tables(node.source)
    if isinstance(node, Fanout):
        return {node.parent_table, node.child_table}
    if isinstance(node, RawSQL):
        return set()  # resolved once sql_bridge.py normalizes it
    return set()
