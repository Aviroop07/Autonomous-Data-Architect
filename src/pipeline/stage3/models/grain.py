"""FK-PK-restricted join canonicalization.

Reduces any ON tree (a chain of joins/aggregates over the schema, built from
util/constraint_model's Relation nodes) to a canonical (grain, edge-set) pair,
under the deliberate
restriction that every join must be a real foreign-key-to-primary-key
relationship. See docs/design/STAGE3_PHASE2_DESIGN.md for
the full mathematical justification.

This module operates on the REAL Stage 2 schema models (Table, Column,
ForeignKey) and on constraint_model's Relation nodes (BaseTable, Join,
Aggregate, Fanout) -- not prototype dataclasses, and no longer on a
Stage-3-local copy of those node types.

Relation shapes that the ON algebra has no meaning for -- Filter, Project,
and an unnormalized RawSQL -- are not excluded by the type system (they are
members of RelationUnion, and Join/Aggregate recurse on the full union). They
are rejected HERE instead, by _canonicalize_inner's explicit branches, which
is the right layer: canonicalize() is what every extracted constraint must
pass in the deterministic-checker node, so a rejection here is a retryable
validation error with a specific message rather than a type error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Tuple, Union

from src.pipeline.stage2.models.schema import Schema, Table, ForeignKey
from src.util.constraint_model.relation.nodes import (
    Aggregate,
    BaseTable,
    Fanout,
    Filter,
    Join,
    Project,
    RawSQL,
    RelationUnion,
)

# The aggregate functions the ON algebra can canonicalize. constraint_model's
# AggregateFn is wider (COUNT_DISTINCT/STDDEV/VARIANCE/PERCENTILE/MODE); those
# have no Grain semantics defined here, so they are rejected explicitly rather
# than silently producing an agg_signature nothing downstream understands.
_CANONICALIZABLE_AGG_FNS = frozenset({"SUM", "COUNT", "AVG", "MAX", "MIN", "MEDIAN"})

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema adapter: wraps the real Schema model into the FK-lookup interface
# that canonicalize() needs.  Built once per canonicalize() call, cheap.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FKRef:
    """Frozen mirror of a ForeignKey, used as a hashable edge identity."""

    child_table: str
    fk_column: str
    parent_table: str


@dataclass(frozen=True)
class _SchemaView:
    """Read-only view over a Schema that provides O(1) FK lookups and
    nullability checks -- the only schema access canonicalize() needs."""

    tables: Dict[str, Table]
    _fk_by_child: Dict[Tuple[str, str], ForeignKey]

    @classmethod
    def from_schema(cls, schema: Schema) -> _SchemaView:
        tables = {t.name: t for t in schema.tables}
        fk_by_child: Dict[Tuple[str, str], ForeignKey] = {}
        for fk in schema.relationships or []:
            fk_by_child[(fk.referencing_table, fk.referencing_column)] = fk
        return cls(tables=tables, _fk_by_child=fk_by_child)

    def parent_has_single_column_pk(self, table: str) -> bool:
        t = self.tables.get(table)
        return t is not None and len(t.primary_key) == 1

    def is_column_nullable(self, table: str, column: str) -> bool:
        t = self.tables.get(table)
        if t is None:
            return False
        for c in t.columns:
            if c.name == column:
                return c.is_nullable
        return False

    def get_fk(self, child_table: str, fk_column: str) -> Optional[ForeignKey]:
        return self._fk_by_child.get((child_table, fk_column))


# ---------------------------------------------------------------------------
# Canonicalization result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Grain:
    """The canonical (grain table + PK, edge-multiset) identity of a joined ON tree.

    Edges are (ForeignKeyRef, occurrence) pairs, NOT a plain set of ForeignKeyRef --
    a recursive/self-referential FK (e.g. CATEGORY.parent_category_id -> CATEGORY.id)
    can legitimately be traversed more than once in a single tree (walking up two
    levels of a hierarchy). A plain set would collapse "1 hop up" and "2 hops up"
    into the identical edge, wrongly treating grandparent and parent as the same
    thing -- occurrence disambiguates hop depth for exactly this case while leaving
    genuinely distinct edges free to commute in any order."""

    table: str
    pk_columns: FrozenSet[str]
    edges: FrozenSet[Tuple[_FKRef, int]]
    # Present only if the top-level node was an ONAggregate/ONFanout; None for
    # a bare join tree. Shape for ONAggregate: (fn, column, group_by, alias,
    # source_table). Shape for ONFanout:
    # ("COUNT_CHILDREN_LEFT_JOIN", child_table, frozenset({fk_column})).
    agg_signature: Optional[tuple] = None
    # True iff reaching this grain required traversing at least one join (or
    # aggregate-regroup) over a column NOT proven required (nullable_columns).
    narrowed: bool = False

    def accessible_columns(
        self, schema: _SchemaView
    ) -> Tuple[FrozenSet[str], FrozenSet[str]]:
        """What columns can a condition actually reference at this point in
        the ON tree, and which of those names are AMBIGUOUS (present via more
        than one joined table, so an unqualified reference to them is invalid)?
        Returns (accessible, ambiguous)."""
        if (
            self.agg_signature is not None
            and self.agg_signature[0] == "COUNT_CHILDREN_LEFT_JOIN"
        ):
            parent_table = self.agg_signature[1]
            t = schema.tables.get(parent_table)
            parent_cols = frozenset(c.name for c in t.columns) if t else frozenset()
            return parent_cols | {"child_count"}, frozenset()

        if self.agg_signature is not None:
            _fn, _col, group_by, alias = self.agg_signature[:4]
            return frozenset(group_by) | {alias}, frozenset()

        # Bare table / join tree
        occurrence_counts: Dict[str, int] = {self.table: 1}
        for fk_ref, _occ in self.edges:
            occurrence_counts[fk_ref.parent_table] = (
                occurrence_counts.get(fk_ref.parent_table, 0) + 1
            )

        counts: Dict[str, int] = {}
        for t_name, n in occurrence_counts.items():
            t = schema.tables.get(t_name)
            if t is None:
                continue
            for col in t.columns:
                counts[col.name] = counts.get(col.name, 0) + n
        ambiguous = frozenset(name for name, n in counts.items() if n > 1)
        accessible = frozenset(counts.keys())

        # Synthetic presence columns for nullable FKs on the own table
        own_table = schema.tables.get(self.table)
        if own_table is not None:
            for col in own_table.columns:
                if col.is_nullable:
                    accessible = accessible | {f"{col.name}.is_present"}

        return accessible, ambiguous

    def validate_column(self, column: str, schema: _SchemaView) -> Optional[str]:
        """None if `column` is a safely-resolvable reference against this
        grain; otherwise a plain-language reason it isn't."""
        accessible, ambiguous = self.accessible_columns(schema)
        if column in ambiguous:
            tables_in_scope = sorted(
                {self.table} | {fk_ref.parent_table for fk_ref, _ in self.edges}
            )
            return (
                f"'{column}' is ambiguous at grain '{self.table}' -- more than "
                f"one reachable table ({', '.join(tables_in_scope)}) declares a "
                f"column with this name; needs table-qualification to resolve."
            )
        if column not in accessible:
            if (
                self.agg_signature is not None
                and self.agg_signature[0] != "COUNT_CHILDREN_LEFT_JOIN"
            ):
                return (
                    f"'{column}' is not accessible at grain '{self.table}' -- "
                    f"this point in the ON tree is post-aggregation, so only the "
                    f"group-by columns and the aggregate's own alias survive; "
                    f"raw source columns are no longer referenceable here."
                )
            return (
                f"'{column}' does not exist on grain '{self.table}' or any "
                f"table reachable from it."
            )
        return None

    def base_key(self) -> Tuple[str, FrozenSet[str]]:
        return (self.table, self.pk_columns)

    def reaches(self, table: str) -> bool:
        if table == self.table:
            return True
        return any(fk_ref.parent_table == table for fk_ref, _ in self.edges)

    def is_comparable_with(
        self, other: Grain, *, population_sensitive: bool = False
    ) -> bool:
        """Two grains are safe to compare if they share the same base grain
        table+PK, and one's edge set is a subset of the other's.

        If either side is an aggregate, the signatures must match exactly;
        only bare (non-aggregate) grains fall back to the edge-subset check.

        `population_sensitive`: pass True for variable kinds where the
        STATISTIC's identity depends on exactly which population it was
        computed over. When population_sensitive and either side is narrowed,
        only an EXACT same-subset match is comparable."""
        if self.base_key() != other.base_key():
            return False
        if self.agg_signature is not None or other.agg_signature is not None:
            if self.agg_signature != other.agg_signature:
                return False
            if population_sensitive and self.narrowed != other.narrowed:
                return False
            return True
        if population_sensitive and (self.narrowed or other.narrowed):
            return self.narrowed == other.narrowed and self.edges == other.edges
        return self.edges <= other.edges or other.edges <= self.edges

    def common_edges(self, other: Grain) -> FrozenSet[Tuple[_FKRef, int]]:
        return self.edges & other.edges

    def _max_occurrence(self, fk_ref: _FKRef) -> int:
        matches = [occ for edge_fk, occ in self.edges if edge_fk == fk_ref]
        return max(matches, default=0)


@dataclass(frozen=True)
class CanonicalizationFailure:
    """Returned instead of a Grain when the ON tree can't be safely reduced."""

    reason: str
    node_repr: str


CanonicalizeResult = Union[Grain, CanonicalizationFailure]


# ---------------------------------------------------------------------------
# Canonicalization logic
# ---------------------------------------------------------------------------


def _split_qualified(ref: str) -> Tuple[Optional[str], str]:
    if "." in ref:
        table, col = ref.split(".", 1)
        return table, col
    return None, ref


def canonicalize(node: "RelationUnion", schema: Schema) -> CanonicalizeResult:
    """Reduce an ON tree to its (grain, edge-set) canonical form, or explain
    why not.  Operates on constraint_model Relation nodes and the real
    Schema model."""
    view = _SchemaView.from_schema(schema)
    return _canonicalize_inner(node, view)


def _canonicalize_inner(node: "RelationUnion", view: _SchemaView) -> CanonicalizeResult:
    """Core canonicalization, operating on a pre-built _SchemaView."""

    if isinstance(node, Filter):
        return CanonicalizationFailure(
            reason=(
                "A Filter (WHERE/HAVING) has no ON-tree equivalent -- express "
                "row-level filtering in the constraint's own condition (or "
                "if_condition), not inside `on`."
            ),
            node_repr=repr(node),
        )

    if isinstance(node, Project):
        return CanonicalizationFailure(
            reason=(
                "A Project (explicit SELECT column list) has no ON-tree "
                "equivalent -- `on` describes table/join/aggregate structure "
                "only, never column projection."
            ),
            node_repr=repr(node),
        )

    if isinstance(node, RawSQL):
        return CanonicalizationFailure(
            reason=(
                "RawSQL reached canonicalize() unnormalized -- normalize_on() "
                "should have replaced it with structured nodes first: "
                f"{node.sql}"
            ),
            node_repr=repr(node),
        )

    if isinstance(node, BaseTable):
        t = view.tables.get(node.name)
        if t is None:
            return CanonicalizationFailure(
                reason=f"Table '{node.name}' not found in schema.",
                node_repr=repr(node),
            )
        return Grain(
            table=node.name,
            pk_columns=frozenset(t.primary_key),
            edges=frozenset(),
        )

    if isinstance(node, Aggregate):
        if node.fn not in _CANONICALIZABLE_AGG_FNS:
            return CanonicalizationFailure(
                reason=(
                    f"Aggregate function '{node.fn}' has no Grain semantics -- "
                    f"use one of {sorted(_CANONICALIZABLE_AGG_FNS)}."
                ),
                node_repr=repr(node),
            )
        source_result = _canonicalize_inner(node.source, view)
        if isinstance(source_result, CanonicalizationFailure):
            return source_result
        sig = (
            node.fn,
            node.column,
            frozenset(node.group_by or []),
            node.alias,
            source_result.table,
        )

        # Grain re-rooting when GROUP BY matches a parent table's FK column
        group_by_set = frozenset(node.group_by or [])
        new_table = source_result.table
        new_pk = source_result.pk_columns
        new_edges = source_result.edges
        new_narrowed = source_result.narrowed

        if group_by_set and group_by_set == source_result.pk_columns:
            pass  # already grouped by its own grain's PK
        elif len(group_by_set) == 1:
            only_col = next(iter(group_by_set))
            matching_parents = {
                fk_ref.parent_table
                for fk_ref, _ in source_result.edges
                if fk_ref.fk_column == only_col
            }
            reroot_fk_nullable = view.is_column_nullable(source_result.table, only_col)
            if not matching_parents:
                # Check schema's own FK list for direct re-rooting
                direct_fk = view.get_fk(source_result.table, only_col)
                if direct_fk is not None:
                    matching_parents = {direct_fk.referred_table}
            if len(matching_parents) == 1:
                new_root = next(iter(matching_parents))
                parent_t = view.tables.get(new_root)
                if parent_t is not None:
                    new_table = new_root
                    new_pk = frozenset(parent_t.primary_key)
                    new_edges = frozenset()
                    if reroot_fk_nullable:
                        new_narrowed = True

        return Grain(
            table=new_table,
            pk_columns=new_pk,
            edges=new_edges,
            agg_signature=sig,
            narrowed=new_narrowed,
        )

    if isinstance(node, Fanout):
        parent = view.tables.get(node.parent_table)
        child = view.tables.get(node.child_table)
        if parent is None:
            return CanonicalizationFailure(
                reason=f"Table '{node.parent_table}' not found in schema.",
                node_repr=repr(node),
            )
        if child is None:
            return CanonicalizationFailure(
                reason=f"Table '{node.child_table}' not found in schema.",
                node_repr=repr(node),
            )
        fk = view.get_fk(node.child_table, node.fk_column)
        if fk is None or fk.referred_table != node.parent_table:
            return CanonicalizationFailure(
                reason=(
                    f"'{node.child_table}.{node.fk_column}' is not a real foreign "
                    f"key to '{node.parent_table}'."
                ),
                node_repr=repr(node),
            )
        if not view.parent_has_single_column_pk(node.parent_table):
            return CanonicalizationFailure(
                reason=f"'{node.parent_table}' lacks a single-column primary key.",
                node_repr=repr(node),
            )
        return Grain(
            table=node.parent_table,
            pk_columns=frozenset(parent.primary_key),
            edges=frozenset(),
            agg_signature=(
                "COUNT_CHILDREN_LEFT_JOIN",
                node.child_table,
                frozenset({node.fk_column}),
            ),
        )

    if isinstance(node, Join):
        left_result = _canonicalize_inner(node.left, view)
        if isinstance(left_result, CanonicalizationFailure):
            return left_result
        right_result = _canonicalize_inner(node.right, view)
        if isinstance(right_result, CanonicalizationFailure):
            return right_result

        if len(node.on) != 1:
            return CanonicalizationFailure(
                reason=(
                    "Composite/multi-condition join not supported under the "
                    "FK-PK-only restriction."
                ),
                node_repr=repr(node),
            )
        cond = node.on[0]
        lt, lc = _split_qualified(cond.left)
        rt, rc = _split_qualified(cond.right)
        if lt is None or rt is None:
            return CanonicalizationFailure(
                reason="Join condition columns must be table-qualified.",
                node_repr=repr(node),
            )

        resolved: Optional[Tuple[ForeignKey, Grain, Grain]] = None

        # Direction 1: lt.lc is the FK
        fk = view.get_fk(lt, lc)
        if (
            fk is not None
            and fk.referred_table == rt
            and left_result.reaches(lt)
            and right_result.table == rt
            and view.parent_has_single_column_pk(rt)
        ):
            resolved = (fk, left_result, right_result)

        # Direction 2: rt.rc is the FK
        if resolved is None:
            fk2 = view.get_fk(rt, rc)
            if (
                fk2 is not None
                and fk2.referred_table == lt
                and right_result.reaches(rt)
                and left_result.table == lt
                and view.parent_has_single_column_pk(lt)
            ):
                resolved = (fk2, right_result, left_result)

        if resolved is None:
            return CanonicalizationFailure(
                reason=(
                    f"Join '{cond.left} = {cond.right}' is not backed by a real "
                    "foreign-key-to-primary-key relationship."
                ),
                node_repr=repr(node),
            )

        fk_obj, child_result, parent_result = resolved
        edge = _FKRef(
            child_table=fk_obj.referencing_table,
            fk_column=fk_obj.referencing_column,
            parent_table=fk_obj.referred_table,
        )
        prior_max = max(
            child_result._max_occurrence(edge), parent_result._max_occurrence(edge)
        )
        merged_edges = (
            child_result.edges | parent_result.edges | {(edge, prior_max + 1)}
        )
        edge_narrows = view.is_column_nullable(edge.child_table, edge.fk_column)
        return Grain(
            table=child_result.table,
            pk_columns=child_result.pk_columns,
            edges=merged_edges,
            narrowed=child_result.narrowed or parent_result.narrowed or edge_narrows,
        )

    return CanonicalizationFailure(
        reason=f"Unknown Relation node type: {type(node).__name__}",
        node_repr=repr(node),
    )
