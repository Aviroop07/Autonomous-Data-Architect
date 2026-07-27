"""Cross-node Relation validation that needs the bottom-up schema
synthesis from relation/schema.py to run at all (Section 4.1/4.5):

- PK-never-dropped by Project -- a hard rejection, not a flagged-but-legal
  state (Section 4.5).
- Self-join/alias-collision detection -- when two sides of a Join resolve
  to the same unqualified table/alias name, it's ambiguous which side a
  JoinCondition's qualifier refers to. Disambiguating with an explicit
  alias is the LLM's responsibility; this only detects and rejects, it
  never invents one (see PROGRESS.md's 2026-07-13 22:46 entry).
- The aggregate-function/column-type compatibility check deferred from
  nodes.py (SUM/AVG/STDDEV/VARIANCE need a numeric column; MAX/MIN/MEDIAN/
  PERCENTILE need an orderable one) -- needs the source's real synthesized
  column type, which nodes.py's self-contained _validate() doesn't have.
"""

from __future__ import annotations

from typing import List

from src.pipeline.stage2.models.schema import Schema
from src.util.constraint_model.condition.expressions import is_numeric, is_orderable
from src.util.constraint_model.relation.nodes import Aggregate, Join, RelationUnion
from src.util.constraint_model.relation.schema import (
    _relation_qualifiers,
    synthesize_schema,
)

_NUMERIC_ONLY_FNS = frozenset({"SUM", "AVG", "STDDEV", "VARIANCE"})
_ORDERABLE_FNS = frozenset({"MAX", "MIN", "MEDIAN", "PERCENTILE"})


def validate_relation(node: "RelationUnion", schema: Schema) -> List[str]:
    """Public entry point: schema synthesis errors plus the cross-node
    checks that synthesis alone doesn't perform. If synthesis itself
    fails, its errors are returned as-is -- the cross-node checks below
    all need a valid synthesized schema to run."""
    _, synth_errors = synthesize_schema(node, schema)
    if synth_errors:
        return synth_errors

    errors: List[str] = []
    errors.extend(_check_project_pk_not_dropped(node, schema))
    errors.extend(_check_join_alias_collisions(node))
    errors.extend(_check_aggregate_fn_types(node, schema))
    return errors


def _child_relations(node: "RelationUnion") -> List["RelationUnion"]:
    if isinstance(node, Join):
        return [node.left, node.right]
    source = getattr(node, "source", None)
    return [source] if source is not None else []


def _check_project_pk_not_dropped(node: "RelationUnion", schema: Schema) -> List[str]:
    from src.util.constraint_model.relation.nodes import Project

    errors: List[str] = []
    if isinstance(node, Project):
        src_eff, src_errs = synthesize_schema(node.source, schema)
        proj_eff, proj_errs = synthesize_schema(node, schema)
        if src_eff is not None and proj_eff is not None:
            if len(proj_eff.primary_key) != len(src_eff.primary_key):
                errors.append(
                    "Project drops one or more primary-key columns of its source "
                    "-- a Project may never drop PK columns."
                )
        errors.extend(src_errs)
        errors.extend(proj_errs)
    for child in _child_relations(node):
        errors.extend(_check_project_pk_not_dropped(child, schema))
    return errors


def _check_join_alias_collisions(node: "RelationUnion") -> List[str]:
    errors: List[str] = []
    if isinstance(node, Join):
        left_names = _relation_qualifiers(node.left)
        right_names = _relation_qualifiers(node.right)
        shared = left_names & right_names
        if shared:
            errors.append(
                f"Join: left and right resolve to the same unqualified table/alias "
                f"name {sorted(shared)} -- this is ambiguous (e.g. a self-join); add "
                "a distinguishing alias to one side."
            )
    for child in _child_relations(node):
        errors.extend(_check_join_alias_collisions(child))
    return errors


def _check_aggregate_fn_types(node: "RelationUnion", schema: Schema) -> List[str]:
    errors: List[str] = []
    if isinstance(node, Aggregate) and node.column != "*":
        src_eff, _ = synthesize_schema(node.source, schema)
        col = src_eff.columns.get(node.column) if src_eff is not None else None
        if col is not None:
            if node.fn in _NUMERIC_ONLY_FNS and not is_numeric(col.data_type):
                errors.append(
                    f"Aggregate.fn='{node.fn}' requires a numeric column, but "
                    f"'{node.column}' is {col.data_type}."
                )
            elif node.fn in _ORDERABLE_FNS and not is_orderable(col.data_type):
                errors.append(
                    f"Aggregate.fn='{node.fn}' requires an orderable column, but "
                    f"'{node.column}' is {col.data_type}."
                )
    for child in _child_relations(node):
        errors.extend(_check_aggregate_fn_types(child, schema))
    return errors
