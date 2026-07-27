"""Normalizes an ON tree containing ONSubquery nodes into the pure
ONBaseTable/ONJoin/ONAggregate/ONFanout algebra, by reusing
constraint_model's already-built, already-tested SQL<->Relation bridge
(util/constraint_model/relation/sql_bridge.py) rather than writing a second,
bespoke SQL parser.

Two-step translation for each ONSubquery encountered:
    1. from_sql(node.sql) -> a constraint_model RelationUnion tree.
    2. _relation_to_on() walks that tree into the equivalent ON node shape,
       resolving any SQL alias (e.g. "o" in "ORDER o") back to the real
       table name -- the ON algebra qualifies JoinCondition by real table
       name only, never by a join/subquery alias (see canonicalize()'s own
       resolution logic in grain.py, which looks up FKs by real table name).

Deliberately unsupported, with a specific reason rather than a silent
partial result (mirrors sql_bridge.py's own scope discipline):
    - Filter (a WHERE/HAVING clause) -- the ON tree has no filter node;
      row-level filtering belongs in the constraint's own condition/
      if_condition, not inside its `on`.
    - Project (an explicit SELECT column list) -- the ON tree only
      describes table/join/aggregate structure, never column projection.
    - RawSQL -- an unparsed fragment; can't occur post from_sql() success,
      listed only for exhaustiveness.
    - An aggregate function with no ONAggregate equivalent (COUNT_DISTINCT,
      STDDEV, VARIANCE, PERCENTILE, MODE) or a fn_param (PERCENTILE's rank).
    - Composite (AND-ed) join conditions -- matches ONJoin's own
      single-JoinCondition restriction.

Fanout is intentionally NEVER inferred from SQL here, for the same reason
sql_bridge.py itself never infers it: recognizing the specific
"LEFT JOIN + COUNT(child.pk) GROUP BY parent.pk" shape as zero-preserving
needs a real heuristic this pass doesn't attempt. Such SQL normalizes to an
ordinary ONAggregate, not an ONFanout -- write ONFanout directly when that
guarantee is actually needed.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple, cast

from src.pipeline.stage3.models.on_nodes import (
    _VALID_AGG_FNS,
    JoinCondition,
    ONAggregate,
    ONBaseTable,
    ONFanout,
    ONJoin,
    ONNode,
    ONSubquery,
)
from src.util.constraint_model.relation.nodes import Aggregate as MAggregate
from src.util.constraint_model.relation.nodes import BaseTable as MBaseTable
from src.util.constraint_model.relation.nodes import Fanout as MFanout
from src.util.constraint_model.relation.nodes import Filter as MFilter
from src.util.constraint_model.relation.nodes import Join as MJoin
from src.util.constraint_model.relation.nodes import Project as MProject
from src.util.constraint_model.relation.nodes import RawSQL as MRawSQL
from src.util.constraint_model.relation.nodes import RelationUnion as MRelationUnion
from src.util.constraint_model.relation.sql_bridge import from_sql

# Mirrors ONAggregate.fn's own Literal exactly -- used only to narrow
# node.fn (a broader AggregateFn) after the _VALID_AGG_FNS membership check
# below, since that runtime check isn't a type guard pyright can see through.
_ONAggregateFn = Literal["SUM", "COUNT", "AVG", "MAX", "MIN", "MEDIAN"]


def _collect_aliases(node: "MRelationUnion", out: Dict[str, str]) -> None:
    """Maps every table name AND alias reachable in `node` to its real
    table name. Tolerant of unsupported node shapes (Filter/Project) --
    their presence is reported precisely, at its exact position, by
    _relation_to_on() instead of here."""
    if isinstance(node, MBaseTable):
        out[node.name] = node.name
        if node.alias:
            out[node.alias] = node.name
        return
    if isinstance(node, MJoin):
        _collect_aliases(node.left, out)
        _collect_aliases(node.right, out)
        return
    if isinstance(node, MAggregate):
        _collect_aliases(node.source, out)
        return
    if isinstance(node, MFilter):
        _collect_aliases(node.source, out)
        return
    if isinstance(node, MProject):
        _collect_aliases(node.source, out)
        return
    if isinstance(node, MFanout):
        out[node.parent_table] = node.parent_table
        out[node.child_table] = node.child_table
        return


def _resolve_qualified(
    ref: str, alias_map: Dict[str, str]
) -> Tuple[Optional[str], Optional[str]]:
    if "." not in ref:
        return None, f"JOIN ON reference '{ref}' is not table-qualified."
    qualifier, col = ref.split(".", 1)
    real = alias_map.get(qualifier)
    if real is None:
        return None, (
            f"JOIN ON reference '{ref}' uses an unrecognized table/alias '{qualifier}'."
        )
    return f"{real}.{col}", None


def _relation_to_on(
    node: "MRelationUnion", alias_map: Dict[str, str]
) -> Tuple[Optional[ONNode], Optional[str]]:
    if isinstance(node, MBaseTable):
        return ONBaseTable(name=node.name), None

    if isinstance(node, MJoin):
        left, err = _relation_to_on(node.left, alias_map)
        if left is None:
            return None, err
        right, err = _relation_to_on(node.right, alias_map)
        if right is None:
            return None, err
        if len(node.on) != 1:
            return None, (
                "Composite join conditions have no ON-tree equivalent -- "
                "ONJoin supports exactly one JoinCondition."
            )
        cond = node.on[0]
        left_ref, err = _resolve_qualified(cond.left, alias_map)
        if left_ref is None:
            return None, err
        right_ref, err = _resolve_qualified(cond.right, alias_map)
        if right_ref is None:
            return None, err
        return (
            ONJoin(
                left=left,
                right=right,
                on=[JoinCondition(left=left_ref, right=right_ref, op=cond.op)],
                alias=node.alias,
            ),
            None,
        )

    if isinstance(node, MAggregate):
        source, err = _relation_to_on(node.source, alias_map)
        if source is None:
            return None, err
        if node.fn not in _VALID_AGG_FNS or node.fn_param is not None:
            return None, (
                f"Aggregate function '{node.fn}' has no ON-tree equivalent "
                "(ONAggregate supports SUM/COUNT/AVG/MAX/MIN/MEDIAN only)."
            )
        fn = cast(_ONAggregateFn, node.fn)
        return (
            ONAggregate(
                source=source,
                fn=fn,
                column=node.column,
                group_by=node.group_by,
                alias=node.alias,
            ),
            None,
        )

    if isinstance(node, MFanout):
        return (
            ONFanout(
                parent_table=node.parent_table,
                child_table=node.child_table,
                fk_column=node.fk_column,
            ),
            None,
        )

    if isinstance(node, MFilter):
        return None, (
            "A WHERE/HAVING filter inside an ON subquery has no ON-tree "
            "equivalent -- express row-level filtering via the constraint's "
            "own condition (or if_condition), not inside the ON tree."
        )

    if isinstance(node, MProject):
        return None, (
            "An explicit SELECT column list inside an ON subquery has no "
            "ON-tree equivalent -- the ON tree only defines table/join/"
            "aggregate structure, never column projection."
        )

    if isinstance(node, MRawSQL):
        return None, "Unparsed raw SQL fragment could not be reduced to the ON algebra."

    return (
        None,
        f"Unsupported relation shape inside ON subquery: {type(node).__name__}.",
    )


def normalize_on(node: "ONNode") -> Tuple[Optional["ONNode"], Optional[str]]:
    """Recursively replaces every ONSubquery in `node` with its structured
    equivalent. Returns (normalized_tree, None) on success -- the same
    object as `node` if nothing needed changing -- or (None, reason) if
    some ONSubquery's SQL can't be reduced to the ON algebra.

    This is purely a SYNTACTIC translation (SQL text -> ON node shape); it
    does not check FK-PK validity or table existence -- run canonicalize()
    on the result for that, same as any other ON tree."""
    if isinstance(node, (ONBaseTable, ONFanout)):
        return node, None

    if isinstance(node, ONJoin):
        left, err = normalize_on(node.left)
        if left is None:
            return None, err
        right, err = normalize_on(node.right)
        if right is None:
            return None, err
        if left is node.left and right is node.right:
            return node, None
        return node.model_copy(update={"left": left, "right": right}), None

    if isinstance(node, ONAggregate):
        source, err = normalize_on(node.source)
        if source is None:
            return None, err
        if source is node.source:
            return node, None
        return node.model_copy(update={"source": source}), None

    if isinstance(node, ONSubquery):
        relation, parse_errors = from_sql(node.sql)
        if relation is None:
            return None, "ONSubquery SQL could not be parsed: " + "; ".join(
                parse_errors
            )
        alias_map: Dict[str, str] = {}
        _collect_aliases(relation, alias_map)
        return _relation_to_on(relation, alias_map)

    return None, f"Unknown ON node type: {type(node).__name__}"
