"""Population identity: the narrower, Filter-aware successor to the old
`src/pipeline/stage3/models/grain.py`'s `Grain` (Section 5). Answers "do
these two Relations describe the same, or a comparable, population of
real-world rows" -- proven to need the FULL join+aggregate+filter
operation history, not just schema equality (Section 5's two worked
counter-examples: an Aggregate re-rooted to a table vs. that table's own
BaseTable report identical (table, PK) but different populations; a bare
BaseTable vs. the same table Filtered down have IDENTICAL schema but
different populations).

`compute_population` mirrors the old `grain.py`'s `canonicalize()` in
spirit, but built directly against this new module's node types and
reusing relation/schema.py's own FK-PK direction resolution
(`resolve_join_child`) rather than re-deriving it -- avoiding a second,
possibly-diverging copy of that logic.
"""

from __future__ import annotations

from typing import Any, FrozenSet, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from src.pipeline.stage2.models.schema import Schema
from src.util.constraint_model.condition.predicates import RPredicateUnion
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
from src.util.constraint_model.relation.schema import (
    resolve_join_child,
    synthesize_schema,
)


class FKEdge(BaseModel):
    """One FK hop traversed while deriving a Population, table-name based
    (not per-column provenance -- sufficient for the same purpose the old
    Grain's `_FKRef` served)."""

    model_config = ConfigDict(frozen=True)

    child_table: str
    fk_column: str
    parent_table: str


class Population(BaseModel):
    """The (grain table + PK, edge-multiset, optional agg-signature,
    narrowed-history) identity of a Relation, for cross-fact population
    comparability. Mirrors grain.py's Grain shape, extended with
    `filter_conditions` since Filter is a genuinely new node (the old
    Grain never tracked filters at all)."""

    model_config = ConfigDict(frozen=True)

    table: str
    pk_columns: FrozenSet[str]
    edges: FrozenSet[Tuple[FKEdge, int]] = frozenset()
    agg_signature: Optional[Tuple[Any, ...]] = None
    narrowed: bool = False
    filter_conditions: Tuple[RPredicateUnion, ...] = ()

    def _base_key(self) -> Tuple[str, FrozenSet[str]]:
        return (self.table, self.pk_columns)

    def is_comparable_with(
        self, other: "Population", *, population_sensitive: bool = False
    ) -> bool:
        """Two populations are safe to compare if they share the same base
        grain (table + PK) and one's edge set is a subset of the other's.

        `population_sensitive=True` is for statistics (Distributed/
        Correlated moment facts) whose own identity depends on exactly
        which population they were computed over: if either side is
        aggregated, the signatures must match exactly (and, if narrowed,
        the narrowed flags too); if either side is narrowed (via a Filter
        or a nullable-FK join), only an EXACT edges+filter_conditions
        match is comparable -- a superset relationship isn't enough,
        unlike the population-insensitive per-row-constraint case."""
        if self._base_key() != other._base_key():
            return False
        if self.agg_signature is not None or other.agg_signature is not None:
            if self.agg_signature != other.agg_signature:
                return False
            if population_sensitive and self.narrowed != other.narrowed:
                return False
            return True
        if population_sensitive and (self.narrowed or other.narrowed):
            return (
                self.narrowed == other.narrowed
                and self.edges == other.edges
                and self.filter_conditions == other.filter_conditions
            )
        return self.edges <= other.edges or other.edges <= self.edges


def compute_population(
    node: "RelationUnion", schema: Schema
) -> Tuple[Optional[Population], List[str]]:
    """Public entry point: derives `node`'s Population against the real,
    schema-declared `schema`. Follows this package's non-raising
    convention -- a None Population paired with a non-empty error list
    means the operation history couldn't be resolved (mirrors relation/
    schema.py's synthesize_schema failure shape)."""
    if isinstance(node, BaseTable):
        table = schema.get_table_map().get(node.name)
        if table is None:
            return None, [f"BaseTable: table '{node.name}' not found in schema."]
        return Population(table=table.name, pk_columns=frozenset(table.primary_key)), []

    if isinstance(node, Project):
        # Column-only transform -- doesn't touch which rows exist.
        return compute_population(node.source, schema)

    if isinstance(node, Filter):
        src_pop, errs = compute_population(node.source, schema)
        if src_pop is None:
            return None, errs
        return (
            src_pop.model_copy(
                update={
                    "narrowed": True,
                    "filter_conditions": src_pop.filter_conditions + (node.condition,),
                }
            ),
            [],
        )

    if isinstance(node, Join):
        left_eff, left_synth_errs = synthesize_schema(node.left, schema)
        right_eff, right_synth_errs = synthesize_schema(node.right, schema)
        errors = list(left_synth_errs) + list(right_synth_errs)
        if left_eff is None or right_eff is None:
            return None, errors

        direction, child_fk_col, direction_errors = resolve_join_child(
            node, left_eff, right_eff
        )
        if direction is None or child_fk_col is None:
            return None, errors + direction_errors

        if direction == "left_is_child":
            child_node, parent_node, child_eff = node.left, node.right, left_eff
        else:
            child_node, parent_node, child_eff = node.right, node.left, right_eff

        child_pop, child_errs = compute_population(child_node, schema)
        if child_pop is None:
            return None, child_errs

        parent_table = _root_table_name(parent_node)
        if parent_table is None:
            return None, ["Join: could not determine the parent side's own table name."]

        fk_nullable = child_eff.columns[child_fk_col].nullable
        edge = FKEdge(
            child_table=child_pop.table,
            fk_column=child_fk_col,
            parent_table=parent_table,
        )
        occurrence = sum(1 for e, _ in child_pop.edges if e == edge) + 1
        return (
            child_pop.model_copy(
                update={
                    "edges": child_pop.edges | {(edge, occurrence)},
                    "narrowed": child_pop.narrowed or fk_nullable,
                }
            ),
            [],
        )

    if isinstance(node, Aggregate):
        src_pop, errs = compute_population(node.source, schema)
        if src_pop is None:
            return None, errs
        group_by = tuple(sorted(node.group_by or ()))
        agg_signature = (node.fn, node.column, group_by, node.alias, src_pop.table)
        # Deliberately simpler than grain.py's old re-rooting-to-parent-table
        # logic: `table` here stays the source's own table, and `pk_columns`
        # becomes the group_by columns (each distinct combination IS one row
        # of this population). `is_comparable_with`'s aggregate branch already
        # requires an exact agg_signature match regardless of `table`/
        # `pk_columns`, so this doesn't change comparability correctness for
        # the cases Section 5 calls out (verified: an Aggregate is never
        # comparable to a bare BaseTable or Fanout over the same table, since
        # only one of the two has an agg_signature at all) -- it only affects
        # `table` as a label, which nothing here depends on yet.
        return (
            Population(
                table=src_pop.table,
                pk_columns=frozenset(group_by),
                agg_signature=agg_signature,
                narrowed=src_pop.narrowed,
                filter_conditions=src_pop.filter_conditions,
            ),
            [],
        )

    if isinstance(node, Fanout):
        table_map = schema.get_table_map()
        parent = table_map.get(node.parent_table)
        child = table_map.get(node.child_table)
        if parent is None:
            return None, [
                f"Fanout.parent_table: table '{node.parent_table}' not found in schema."
            ]
        if child is None:
            return None, [
                f"Fanout.child_table: table '{node.child_table}' not found in schema."
            ]
        agg_signature = ("FANOUT_LEFT_JOIN", node.child_table, node.fk_column)
        return (
            Population(
                table=parent.name,
                pk_columns=frozenset(parent.primary_key),
                agg_signature=agg_signature,
            ),
            [],
        )

    if isinstance(node, RawSQL):
        return None, [
            "RawSQL nodes cannot have their population computed until relation/"
            "sql_bridge.py normalizes them into structured nodes (not yet built)."
        ]

    return None, [f"Unknown Relation node type: {type(node).__name__}"]


def _root_table_name(node: "RelationUnion") -> Optional[str]:
    """The single base table name this Relation ultimately reduces to for
    edge-recording purposes -- only meaningful for a Join's parent side,
    which (by construction, via resolve_join_child) is always a relation
    whose own population is just a bare table/chain rooted at one table."""
    if isinstance(node, BaseTable):
        return node.name
    if isinstance(node, (Project, Filter)):
        return _root_table_name(node.source)
    if isinstance(node, Join):
        return None
    return None
