"""Derived-column circular-dependency detection.

Builds a dependency graph from cross_shard.DerivedColumnConstraint
expressions, finds cycles, composes the linear expression around each
cycle, and checks whether it has a fixed point. A cycle with no fixed
point (e.g. x = x + 5) is a genuine, unresolvable contradiction.

Ported from the validated experiments/stage3_conflict_v2/cycles.py
prototype (see ISSUES.md items 1 and 7), fixing three real bugs found in
the original conflict_detection.py implementation:
1. `_linear_coeff` used to return a bare `None` on any non-linear
   sub-expression, which its own caller unconditionally 2-tuple-unpacked
   -- an unhandled TypeError crash, not the graceful "unverifiable"
   fallback the code appeared to have. Always returns a 2-tuple now.
2. A 1-node self-loop (`x = x + 1`, the simplest possible unsatisfiable
   cycle) was silently skipped via a `len(cycle_nodes) < 2` guard.
   Handled directly now via _compose_cycle_expressions' own 1-node branch.
3. A cross-table cycle (column A on table T1 depending on column B on
   table T2, and vice versa) was invisible, because the dependency graph
   always assumed a referenced column lived on the SAME table as the
   column being computed. Resolved via referenced_tables now.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import networkx as nx

from src.pipeline.stage3.models.condition_nodes import (
    RArithmetic,
    RColumnRef,
    RExprUnion,
    RLiteral,
)
from src.pipeline.stage3.models.cross_shard import DerivedColumnConstraint
from src.pipeline.stage3.models.probe import CycleIssue


def detect_derived_cycles(
    derived: List[DerivedColumnConstraint],
) -> List[CycleIssue]:
    """Detect circular dependencies in derived-column expressions.

    Returns one CycleIssue per genuine contradiction found (a cycle with
    no fixed point). An empty list means no unresolvable cycle exists --
    this does NOT mean every cycle is reported: a cycle with a valid fixed
    point (x = 0.5*x + 3, solvable at x=6) is informational, not a
    contradiction, and is not included."""
    if not derived:
        return []

    by_target: dict[str, DerivedColumnConstraint] = {
        f"{dc.target_table}.{dc.target_column}": dc for dc in derived
    }

    graph: dict[str, list[str]] = {}
    for dc in derived:
        target = f"{dc.target_table}.{dc.target_column}"
        deps = extract_columns_from_expr(dc.expression)
        for dep_col in deps:
            owner = _resolve_owning_table(
                dep_col, dc.target_table, dc.referenced_tables, derived
            )
            if owner is None:
                continue  # can't safely resolve -- skip this edge, don't guess
            graph.setdefault(f"{owner}.{dep_col}", []).append(target)

    nx_graph = nx.DiGraph()
    for src, dsts in graph.items():
        for dst in dsts:
            nx_graph.add_edge(src, dst)

    try:
        cycles = list(nx.simple_cycles(nx_graph))
    except nx.NetworkXError:
        return []

    issues: List[CycleIssue] = []
    for cycle_nodes in cycles:
        fact_refs = tuple(
            sorted(
                {
                    fid
                    for node in cycle_nodes
                    if node in by_target
                    for fid in by_target[node].fact_references
                }
            )
        )
        composed = _compose_cycle_expressions(cycle_nodes, derived)
        if composed is None:
            issues.append(
                CycleIssue(
                    description=(
                        f"Circular dependency ({' -> '.join(cycle_nodes)} -> "
                        f"{cycle_nodes[0]}): non-linear expressions, cannot "
                        f"verify whether a fixed point exists."
                    ),
                    nodes=tuple(cycle_nodes),
                    fact_references=fact_refs,
                )
            )
            continue

        coeff, const = composed
        if math.isclose(coeff, 1.0) and not math.isclose(const, 0.0):
            issues.append(
                CycleIssue(
                    description=(
                        f"Circular dependency ({' -> '.join(cycle_nodes)} -> "
                        f"{cycle_nodes[0]}): x = x + {const:.4g} has no "
                        f"solution."
                    ),
                    nodes=tuple(cycle_nodes),
                    fact_references=fact_refs,
                )
            )
        # coeff != 1.0 has a real fixed point (x = const / (1 - coeff)) --
        # informational, not a contradiction, not reported as an issue.

    return issues


def extract_columns_from_expr(expr: RExprUnion) -> set[str]:
    """Column names an arithmetic expression tree depends on."""
    cols: set[str] = set()
    _collect_expr_columns(expr, cols)
    return cols


def _collect_expr_columns(node: RExprUnion, out: set[str]) -> None:
    if isinstance(node, RColumnRef):
        out.add(node.name)
    elif isinstance(node, RArithmetic):
        _collect_expr_columns(node.left, out)
        _collect_expr_columns(node.right, out)
    # RLiteral, RAggregateRef: no column names


def _resolve_owning_table(
    column: str,
    formula_table: str,
    referenced_tables: List[str],
    all_derived: List[DerivedColumnConstraint],
) -> Optional[str]:
    """Which table does `column` actually belong to? A derived column is
    typically a NEW column not yet in the schema, so another
    DerivedColumnConstraint's own target_column within the same batch is a
    valid resolution target too, not just pre-existing schema columns."""
    all_derived_by_target = {
        f"{dc.target_table}.{dc.target_column}": dc.target_table for dc in all_derived
    }

    candidates = [
        t for t in referenced_tables if f"{t}.{column}" in all_derived_by_target
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates and f"{formula_table}.{column}" in all_derived_by_target:
        return formula_table
    return None


def _compose_cycle_expressions(
    cycle: List[str], derived: List[DerivedColumnConstraint]
) -> Optional[Tuple[float, float]]:
    """Compose linear expressions around a cycle into x = A*x + B. Returns
    None if any link is non-linear (can't verify a fixed point)."""
    if len(cycle) == 1:
        node = cycle[0]
        expr = next(
            (
                dc.expression
                for dc in derived
                if f"{dc.target_table}.{dc.target_column}" == node
            ),
            None,
        )
        if expr is None:
            return None
        var_name = node.split(".", 1)[1] if "." in node else node
        coeff, const = _linear_coeff(expr, var_name)
        if coeff is None or const is None:
            return None
        return (coeff, const)

    total_coeff, total_const = 1.0, 0.0
    for i, node in enumerate(cycle):
        next_node = cycle[(i + 1) % len(cycle)]
        expr = None
        for dc in derived:
            if f"{dc.target_table}.{dc.target_column}" != next_node:
                continue
            raw_node = node.split(".", 1)[1] if "." in node else node
            if raw_node in extract_columns_from_expr(dc.expression):
                expr = dc.expression
                break
        if expr is None:
            return None

        a, b = _linear_coeff(expr, node)
        if a is None or b is None:
            return None
        total_const = a * total_const + b
        total_coeff *= a

    return (total_coeff, total_const)


def _linear_coeff(
    expr: RExprUnion, var: str
) -> Tuple[Optional[float], Optional[float]]:
    """Coefficient and constant of a linear expression in one variable
    (`var * 3 + 2` -> (3.0, 2.0)). ALWAYS returns a 2-tuple -- (None, None)
    means "not linear / can't determine", never a bare None, which is what
    made this crash unconditionally in the original implementation.

    A reference to a DIFFERENT variable is NOT treated as constant zero --
    that would be unsound the moment a cycle formula also depends on a
    free, non-cycle variable (e.g. `ORDER.total = subtotal + tax_rate`
    where `tax_rate` isn't part of the cycle). (None, None) means
    "unverifiable", not "resolved to 0"."""
    raw_var = var.split(".", 1)[1] if "." in var else var

    if isinstance(expr, RColumnRef):
        return (1.0, 0.0) if expr.name == raw_var else (None, None)

    if isinstance(expr, RLiteral):
        val = expr.value
        return (0.0, float(val)) if isinstance(val, (int, float)) else (None, None)

    if isinstance(expr, RArithmetic):
        left_coeff, left_const = _linear_coeff(expr.left, var)
        right_coeff, right_const = _linear_coeff(expr.right, var)
        if (
            left_coeff is None
            or left_const is None
            or right_coeff is None
            or right_const is None
        ):
            return (None, None)

        if expr.op == "+":
            return (left_coeff + right_coeff, left_const + right_const)
        if expr.op == "-":
            return (left_coeff - right_coeff, left_const - right_const)
        if expr.op == "*":
            if math.isclose(left_coeff, 0.0):
                return (right_coeff * left_const, right_const * left_const)
            if math.isclose(right_coeff, 0.0):
                return (left_coeff * right_const, left_const * right_const)
            return (None, None)
        if expr.op == "/":
            if math.isclose(right_coeff, 0.0) and right_const != 0:
                return (left_coeff / right_const, left_const / right_const)
            return (None, None)

    return (None, None)
