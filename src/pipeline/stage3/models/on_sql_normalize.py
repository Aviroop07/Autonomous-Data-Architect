"""Replaces RawSQL nodes in an ON tree with their structured equivalents.

This module used to be ~240 lines, almost all of it `_relation_to_on()`: a
second translation walking constraint_model's Relation tree back into a
Stage-3-local ON tree, resolving aliases and rejecting node shapes along the
way. That translation existed only because the two taxonomies were separate.
They are not any more -- `from_sql()` already returns exactly the node types
the ON tree is made of -- so normalizing is now just "swap the RawSQL subtree
for the parsed one".

What is deliberately NOT done here: validating the result. This is a purely
syntactic swap (SQL text -> node shapes). Whether the outcome is a legal ON
tree at all -- real FK-PK joins, no Filter/Project, a canonicalizable
aggregate function -- is canonicalize()'s job in grain.py, which every
extracted constraint has to pass anyway. Checking it twice, in two places
that could disagree, is what the old version did.
"""

from __future__ import annotations

from typing import Optional, Tuple

from typing import Dict

from src.util.constraint_model.relation.nodes import (
    Aggregate,
    BaseTable,
    Fanout,
    Filter,
    Join,
    JoinCondition,
    Project,
    RawSQL,
    RelationUnion,
)
from src.util.constraint_model.relation.sql_bridge import from_sql


def _collect_aliases(node: "RelationUnion", out: Dict[str, str]) -> None:
    """Map every table name AND SQL alias reachable in `node` to the real
    table name."""
    if isinstance(node, BaseTable):
        out[node.name] = node.name
        if node.alias:
            out[node.alias] = node.name
    elif isinstance(node, Join):
        _collect_aliases(node.left, out)
        _collect_aliases(node.right, out)
    elif isinstance(node, (Aggregate, Filter, Project)):
        _collect_aliases(node.source, out)
    elif isinstance(node, Fanout):
        out[node.parent_table] = node.parent_table
        out[node.child_table] = node.child_table


def _dealias(node: "RelationUnion", aliases: Dict[str, str]) -> "RelationUnion":
    """Rewrite every JoinCondition to be qualified by the REAL table name.

    SQL lets a join be written `FROM ORDER_ROW o JOIN CUSTOMER c ON
    o.customer_id = c.id`, so from_sql() faithfully produces conditions
    qualified by `o`/`c`. canonicalize() resolves foreign keys by real table
    name only, so an alias-qualified condition looks like a join against a
    table that does not exist. Resolving them here is the one genuinely
    load-bearing piece of the old, much larger translation layer.
    """
    if isinstance(node, Join):
        return node.model_copy(
            update={
                "left": _dealias(node.left, aliases),
                "right": _dealias(node.right, aliases),
                "on": [
                    JoinCondition(
                        left=_qualify(c.left, aliases),
                        right=_qualify(c.right, aliases),
                        op=c.op,
                    )
                    for c in node.on
                ],
            }
        )
    if isinstance(node, (Aggregate, Filter, Project)):
        return node.model_copy(update={"source": _dealias(node.source, aliases)})
    return node


def _qualify(ref: str, aliases: Dict[str, str]) -> str:
    """'o.customer_id' -> 'ORDER_ROW.customer_id'. Left as-is when the
    qualifier is unknown, so canonicalize() reports it rather than this
    purely syntactic pass inventing a name."""
    if "." not in ref:
        return ref
    qualifier, col = ref.split(".", 1)
    return f"{aliases.get(qualifier, qualifier)}.{col}"


def normalize_on(
    node: "RelationUnion",
) -> Tuple[Optional["RelationUnion"], Optional[str]]:
    """Recursively replace every RawSQL in `node` with its parsed structure.

    Returns (normalized_tree, None) -- the same object as `node` when nothing
    needed changing -- or (None, reason) if some RawSQL could not be parsed.
    """
    if isinstance(node, (BaseTable, Fanout)):
        return node, None

    if isinstance(node, Join):
        left, err = normalize_on(node.left)
        if left is None:
            return None, err
        right, err = normalize_on(node.right)
        if right is None:
            return None, err
        if left is node.left and right is node.right:
            return node, None
        return node.model_copy(update={"left": left, "right": right}), None

    if isinstance(node, (Aggregate, Filter, Project)):
        source, err = normalize_on(node.source)
        if source is None:
            return None, err
        if source is node.source:
            return node, None
        return node.model_copy(update={"source": source}), None

    if isinstance(node, RawSQL):
        relation, parse_errors = from_sql(node.sql)
        if relation is None:
            return None, "RawSQL could not be parsed: " + "; ".join(parse_errors)
        aliases: Dict[str, str] = {}
        _collect_aliases(relation, aliases)
        return _dealias(relation, aliases), None

    return None, f"Unknown Relation node type: {type(node).__name__}"
