"""Conflict detection for Stage 3 constraint representations.

A lightweight, compositional system that detects impossible combinations
of constraints. Core design: since conditions are typed R-AST trees,
compatibility checks recurse with the AST structure. Three-valued logic
("yes"/"no"/"unknown") on condition overlap ensures zero false negatives:
unknown is treated as overlap, so any uncertain case triggers a value check.

False positives are acceptable (flagging a non-conflict). False negatives
are never acceptable (missing a real conflict).

Architecture:
    Layer 1: Primitives (interval extraction, ON equivalence)
    Layer 2: Three-valued condition overlap + binary value compatibility
    Layer 3: Detection rules (column-level, structural, derived-cycle, dist-logic)
    Layer 4: Entry point (detect_all_conflicts -> ConflictReport)
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Tuple

import networkx as nx
from pydantic import BaseModel, Field

from src.pipeline.stage3.models.condition_nodes import (
    RAggregateRef,
    RAnd,
    RArithmetic,
    RBetween,
    RColumnRef,
    RComparison,
    RExists,
    RIfThen,
    RInSet,
    RLiteral,
    RNot,
    RNotExists,
    RNotInSet,
    ROr,
    RPredicate,
    RExprUnion,
    SubqueryRef,
)
from src.pipeline.stage3.models.cross_shard import (
    Constraint,
    DerivedColumnConstraint,
    DistributionConstraint,
)
from src.pipeline.stage3.models.on_nodes import (
    ONAggregate,
    ONBaseTable,
    ONJoin,
    ONNode,
    ONSubquery,
    JoinCondition,
)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Interval = Tuple[float, float]
OverlapResult = Literal["yes", "no", "unknown"]

NEG_INF = float("-inf")
POS_INF = float("inf")

_INTERVAL_OPS = frozenset({"<", "<=", ">", ">=", "="})


# ---------------------------------------------------------------------------
# Conflict / ConflictReport models
# ---------------------------------------------------------------------------


class Conflict(BaseModel):
    """A detected incompatibility between constraints."""

    conflict_type: str = Field(
        description="One of: column_value, structural, derived_cycle, dist_logic."
    )
    severity: Literal["hard", "soft", "info"] = Field(
        description="hard = true impossibility, soft = likely conflict, info = flagged for review."
    )
    description: str = Field(description="Human-readable explanation.")
    fact_refs_a: list[int] = Field(description="Fact references from first constraint.")
    fact_refs_b: list[int] = Field(
        description="Fact references from second constraint."
    )
    evidence: dict = Field(
        default_factory=dict, description="Supporting analysis data."
    )


class ConflictReport(BaseModel):
    """Result of a full conflict detection pass."""

    conflicts: list[Conflict] = Field(default_factory=list)
    checked_pairs: int = Field(default=0)
    is_consistent: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Layer 1: Primitives
# ---------------------------------------------------------------------------


def interval_of_comparison(node: RComparison) -> Optional[Interval]:
    """Extract numeric interval from a comparison predicate.

    Returns (lo, hi) representing the set of values satisfying the comparison,
    or None if the comparison can't be reduced to an interval (column-on-both-
    sides, arithmetic bounds, distribution pin, string literals, etc.).
    """
    if not isinstance(node.left, RColumnRef) and not isinstance(
        node.left, RAggregateRef
    ):
        return None
    if not isinstance(node.right, RLiteral):
        return None

    val = node.right.value
    if not isinstance(val, (int, float)):
        return None

    val = float(val)
    match node.op:
        case "<":
            return (NEG_INF, val)
        case "<=":
            return (NEG_INF, val)
        case ">":
            return (val, POS_INF)
        case ">=":
            return (val, POS_INF)
        case "=":
            return (val, val)
        case _:
            return None


def _string_eq_value(node: RComparison) -> Optional[str]:
    """Extract the string value from a string equality comparison.

    Returns the string value if the comparison is `column = 'string'` or
    `column != 'string'`, or None otherwise.
    """
    if not isinstance(node.left, RColumnRef):
        return None
    if not isinstance(node.right, RLiteral):
        return None
    if not isinstance(node.right.value, str):
        return None
    return node.right.value


def intervals_overlap(a: Interval, b: Interval) -> bool:
    """Do two intervals have any values in common?"""
    return a[0] <= b[1] and b[0] <= a[1]


def set_overlaps_interval(values: list, interval: Interval) -> bool:
    """Does any value in the list fall within the interval?"""
    for v in values:
        if isinstance(v, (int, float)) and interval[0] <= float(v) <= interval[1]:
            return True
    return False


def distribution_support(dc: DistributionConstraint) -> Optional[Interval]:
    """Estimate numeric support of a distribution.

    Returns (lo, hi) or None for non-numeric distributions (CATEGORICAL).
    Uses heuristics (3-sigma for Gaussian) -- not exact, but conservative
    enough for conflict detection.
    """
    p = dc.parameters
    match dc.family:
        case "GAUSSIAN":
            mean = float(p["mean"])
            std = float(p["std_dev"])
            return (mean - 3 * std, mean + 3 * std)
        case "UNIFORM":
            return (float(p["min_value"]), float(p["max_value"]))
        case "POISSON":
            lam = float(p["lam"])
            return (0.0, max(lam * 2, lam + 10))
        case "LOG_NORMAL":
            mean = float(p["mean"])
            std = float(p["std_dev"])
            return (0.0, math.exp(mean + 3 * std))
        case "BETA":
            return (0.0, 1.0)
        case "CATEGORICAL":
            return None
    return None


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
    match node:
        case RComparison(left=left, right=right):
            _collect_columns_expr(left, out)
            _collect_columns_expr(right, out)
        case RAnd(operands=ops) | ROr(operands=ops):
            for op in ops:
                _collect_columns_pred(op, out)
        case RNot(operand=inner):
            _collect_columns_pred(inner, out)
        case RBetween(expr=e, low=lo, high=hi):
            _collect_columns_expr(e, out)
            _collect_columns_expr(lo, out)
            _collect_columns_expr(hi, out)
        case RInSet(expr=e) | RNotInSet(expr=e):
            _collect_columns_expr(e, out)
        case RIfThen(antecedent=a, consequent=c):
            _collect_columns_pred(a, out)
            _collect_columns_pred(c, out)
        case RExists(subquery=s) | RNotExists(subquery=s):
            pass
        case _:
            pass


def on_structural_equivalence(a: ONNode, b: ONNode) -> bool:
    """Are two ON trees structurally identical after normalization?

    Two ONs are equivalent iff they reference the same base tables, same
    joins, same aggregates, in the same structure. This is a purely
    structural check -- no SQL execution, no data inspection.
    """
    if type(a) is not type(b):
        return False

    if isinstance(a, ONBaseTable) and isinstance(b, ONBaseTable):
        return a.name == b.name

    if isinstance(a, ONJoin) and isinstance(b, ONJoin):
        left_eq = on_structural_equivalence(a.left, b.left)
        right_eq = on_structural_equivalence(a.right, b.right)
        if not (left_eq and right_eq):
            return False
        if len(a.on) != len(b.on):
            return False
        a_conds = sorted([(c.left, c.right) for c in a.on], key=lambda x: (x[0], x[1]))
        b_conds = sorted([(c.left, c.right) for c in b.on], key=lambda x: (x[0], x[1]))
        return a_conds == b_conds

    if isinstance(a, ONAggregate) and isinstance(b, ONAggregate):
        return (
            on_structural_equivalence(a.source, b.source)
            and a.fn == b.fn
            and a.column == b.column
            and sorted(a.group_by or []) == sorted(b.group_by or [])
            and a.alias == b.alias
        )

    if isinstance(a, ONSubquery) and isinstance(b, ONSubquery):
        return a.sql.strip() == b.sql.strip()

    return False


# ---------------------------------------------------------------------------
# Layer 2: Condition overlap (three-valued) + value compatibility
# ---------------------------------------------------------------------------


def conditions_overlap(a: RPredicate, b: RPredicate) -> OverlapResult:
    """Can both conditions be true for the same row?

    Returns:
        "yes"    -- conditions definitely overlap (same column, compatible ranges)
        "no"     -- conditions definitely disjoint (mutually exclusive)
        "unknown" -- can't determine -> treat as overlap for conflict check

    Guarantees zero false negatives: "no" is returned ONLY when we can prove
    disjointness. Anything uncertain returns "unknown" (treated as overlap).
    """

    # --- Both are comparisons ---
    if isinstance(a, RComparison) and isinstance(b, RComparison):
        a_cols = extract_columns(a)
        b_cols = extract_columns(b)
        if a_cols != b_cols or len(a_cols) != 1:
            return "unknown"
        # Try numeric interval overlap
        ia = interval_of_comparison(a)
        ib = interval_of_comparison(b)
        if ia and ib:
            return "yes" if intervals_overlap(ia, ib) else "no"
        # Try string equality overlap
        a_str = _string_eq_value(a)
        b_str = _string_eq_value(b)
        if a_str is not None and b_str is not None:
            if a.op == "=" and b.op == "=":
                return "yes" if a_str == b_str else "no"
            if a.op == "=" or b.op == "=":
                eq_val = a_str if a.op == "=" else b_str
                neq_node = b if a.op == "=" else a
                if neq_node.op == "!=":
                    return "no" if eq_val == b_str else "yes"
            return "unknown"
        return "unknown"

    # --- Both are IN-sets on the same column ---
    if isinstance(a, RInSet) and isinstance(b, RInSet):
        a_cols = extract_columns(a)
        b_cols = extract_columns(b)
        if a_cols == b_cols and len(a_cols) == 1:
            return "yes" if set(a.values) & set(b.values) else "no"
        return "unknown"

    # --- IN-set vs Comparison on the same column ---
    if isinstance(a, RInSet) and isinstance(b, RComparison):
        a_cols = extract_columns(a)
        b_cols = extract_columns(b)
        if a_cols == b_cols and len(a_cols) == 1:
            ib = interval_of_comparison(b)
            if ib:
                return "yes" if set_overlaps_interval(a.values, ib) else "no"
        return "unknown"

    if isinstance(b, RInSet) and isinstance(a, RComparison):
        return conditions_overlap(b, a)

    # --- IN-set vs NotInSet on the same column ---
    if isinstance(a, RInSet) and isinstance(b, RNotInSet):
        a_cols = extract_columns(a)
        b_cols = extract_columns(b)
        if a_cols == b_cols and len(a_cols) == 1:
            overlap_vals = set(a.values) & set(b.values)
            allowed = set(a.values) - set(b.values)
            if not allowed:
                return "no"
            if overlap_vals:
                return "yes"
            return "no"
        return "unknown"

    if isinstance(b, RInSet) and isinstance(a, RNotInSet):
        return conditions_overlap(b, a)

    # --- AND ---
    if isinstance(a, RAnd):
        results = [conditions_overlap(op, b) for op in a.operands]
        if any(r == "no" for r in results):
            return "no"
        if all(r == "yes" for r in results):
            return "yes"
        return "unknown"

    if isinstance(b, RAnd):
        results = [conditions_overlap(a, op) for op in b.operands]
        if any(r == "no" for r in results):
            return "no"
        if all(r == "yes" for r in results):
            return "yes"
        return "unknown"

    # --- OR ---
    if isinstance(a, ROr):
        results = [conditions_overlap(op, b) for op in a.operands]
        if "yes" in results:
            return "yes"
        if all(r == "no" for r in results):
            return "no"
        return "unknown"

    if isinstance(b, ROr):
        results = [conditions_overlap(a, op) for op in b.operands]
        if "yes" in results:
            return "yes"
        if all(r == "no" for r in results):
            return "no"
        return "unknown"

    # --- NOT ---
    if isinstance(a, RNot):
        inner = conditions_overlap(a.operand, b)
        if inner == "yes":
            return "no"
        if inner == "no":
            return "yes"
        return "unknown"

    if isinstance(b, RNot):
        return conditions_overlap(b, a)

    # --- IF/THEN: decompose into OR(NOT antecedent, consequent) ---
    if isinstance(a, RIfThen):
        decomposed = ROr(operands=[RNot(operand=a.antecedent), a.consequent])
        return conditions_overlap(decomposed, b)

    if isinstance(b, RIfThen):
        decomposed = ROr(operands=[RNot(operand=b.antecedent), b.consequent])
        return conditions_overlap(a, decomposed)

    # --- Everything else ---
    return "unknown"


def values_compatible(c1: Constraint, c2: Constraint) -> bool:
    """Are the value constraints compatible, given that conditions overlap?

    Returns True if values might coexist, False if definitely incompatible.
    Conservative: returns False when uncertain (assume incompatible).
    """
    if isinstance(c1, DistributionConstraint) and isinstance(
        c2, DistributionConstraint
    ):
        if c1.family == c2.family and c1.column == c2.column:
            if c1.parameters == c2.parameters:
                return True
            return False
        return False

    if isinstance(c1, DistributionConstraint) and not isinstance(
        c2, DistributionConstraint
    ):
        sup = distribution_support(c1)
        if sup is None:
            return False
        cols2 = extract_columns(c2.condition)
        if c1.column not in cols2:
            return True
        ib = _max_interval(c2.condition)
        if ib is None:
            return False
        return intervals_overlap(sup, ib)

    if isinstance(c2, DistributionConstraint) and not isinstance(
        c1, DistributionConstraint
    ):
        return values_compatible(c2, c1)

    i1 = _max_interval(c1.condition)
    i2 = _max_interval(c2.condition)
    if i1 is None or i2 is None:
        return False
    return intervals_overlap(i1, i2)


def _max_interval(pred: RPredicate) -> Optional[Interval]:
    """Extract the tightest interval a predicate implies on its referenced column.

    For AND: intersect all operand intervals.
    For OR: union all operand intervals (widest bounds).
    For simple comparisons: direct extraction.
    """
    if isinstance(pred, RComparison):
        return interval_of_comparison(pred)

    if isinstance(pred, RBetween):
        lo = pred.low
        hi = pred.high
        if isinstance(lo, RLiteral) and isinstance(hi, RLiteral):
            lo_val = lo.value
            hi_val = hi.value
            if isinstance(lo_val, (int, float)) and isinstance(hi_val, (int, float)):
                return (float(lo_val), float(hi_val))
        return None

    if isinstance(pred, RInSet):
        numeric_vals = [v for v in pred.values if isinstance(v, (int, float))]
        if numeric_vals:
            return (
                min(float(v) for v in numeric_vals),
                max(float(v) for v in numeric_vals),
            )
        return None

    if isinstance(pred, RAnd):
        intervals = []
        for op in pred.operands:
            iv = _max_interval(op)
            if iv is not None:
                intervals.append(iv)
        if not intervals:
            return None
        lo = max(iv[0] for iv in intervals)
        hi = min(iv[1] for iv in intervals)
        if lo <= hi:
            return (lo, hi)
        return None

    if isinstance(pred, ROr):
        intervals = []
        for op in pred.operands:
            iv = _max_interval(op)
            if iv is not None:
                intervals.append(iv)
        if not intervals:
            return None
        return (min(iv[0] for iv in intervals), max(iv[1] for iv in intervals))

    if isinstance(pred, RIfThen):
        return _max_interval(pred.consequent)

    return None


# ---------------------------------------------------------------------------
# Layer 3: Detection rules
# ---------------------------------------------------------------------------


def detect_column_conflicts(
    constraints: list[Constraint],
) -> list[Conflict]:
    """Detect pairwise value-space conflicts on the same column+ON.

    Groups non-distribution constraints by (ON_signature, column). For each
    group, checks every pair:

    - If conditions are disjoint (overlap="no"): both constraints demand
      incompatible predicates on the same rows -> hard conflict.
    - If conditions overlap (overlap="yes"/"unknown") but value ranges are
      incompatible -> hard conflict.
    """
    groups: dict[tuple[frozenset[str], frozenset[str]], list[int]] = {}
    for idx, c in enumerate(constraints):
        if isinstance(c, DistributionConstraint):
            continue
        cols = extract_columns(c.condition)
        on_sig = frozenset(_on_tables(c.on))
        for col in cols:
            key = (on_sig, frozenset([col]))
            groups.setdefault(key, []).append(idx)

    conflicts: list[Conflict] = []
    seen: set[tuple[int, int]] = set()

    for (on_sig, col_set), members in groups.items():
        col = next(iter(col_set))
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a_idx, b_idx = members[i], members[j]
                pair_key = (min(a_idx, b_idx), max(a_idx, b_idx))
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                c1 = constraints[a_idx]
                c2 = constraints[b_idx]

                overlap = conditions_overlap(c1.condition, c2.condition)

                if overlap == "no":
                    conflicts.append(
                        Conflict(
                            conflict_type="column_value",
                            severity="hard",
                            description=(
                                f"Column '{col}': conditions are mutually exclusive "
                                f"(no row can satisfy both)."
                            ),
                            fact_refs_a=c1.fact_references,
                            fact_refs_b=c2.fact_references,
                            evidence={
                                "column": col,
                                "overlap": "no",
                                "on_tables": sorted(on_sig),
                            },
                        )
                    )
                    continue

                if values_compatible(c1, c2):
                    continue

                conflicts.append(
                    Conflict(
                        conflict_type="column_value",
                        severity="hard",
                        description=(
                            f"Column '{col}': conditions overlap (overlap={overlap}) "
                            f"but values are incompatible."
                        ),
                        fact_refs_a=c1.fact_references,
                        fact_refs_b=c2.fact_references,
                        evidence={
                            "column": col,
                            "overlap": overlap,
                            "on_tables": sorted(on_sig),
                        },
                    )
                )

    return conflicts


def detect_structural_conflicts(
    constraints: list[Constraint],
) -> list[Conflict]:
    """Detect structural impossibilities (cardinality ranges, fanout).

    Only considers constraints with category == "structural".
    """
    table_cards: dict[str, list[tuple[Interval, list[int]]]] = {}
    fk_fanouts: dict[tuple[str, str], list[tuple[Interval, list[int]]]] = {}

    for c in constraints:
        if not isinstance(c, Constraint) or isinstance(c, DistributionConstraint):
            continue
        if c.category != "structural":
            continue
        cols = extract_columns(c.condition)
        if len(cols) != 1:
            continue
        col = next(iter(cols))
        iv = _max_interval(c.condition)
        if iv is None:
            continue

        on_tables = _on_tables(c.on)
        if len(on_tables) == 1:
            table = next(iter(on_tables))
            table_cards.setdefault(table, []).append((iv, c.fact_references))

    conflicts: list[Conflict] = []
    for table, ranges in table_cards.items():
        for i in range(len(ranges)):
            for j in range(i + 1, len(ranges)):
                iv_a, refs_a = ranges[i]
                iv_b, refs_b = ranges[j]
                if not intervals_overlap(iv_a, iv_b):
                    conflicts.append(
                        Conflict(
                            conflict_type="structural",
                            severity="hard",
                            description=(
                                f"Table '{table}': cardinality intervals "
                                f"{iv_a} and {iv_b} do not overlap."
                            ),
                            fact_refs_a=refs_a,
                            fact_refs_b=refs_b,
                            evidence={
                                "table": table,
                                "interval_a": list(iv_a),
                                "interval_b": list(iv_b),
                            },
                        )
                    )

    return conflicts


def detect_derived_cycles(
    derived: list[DerivedColumnConstraint],
) -> list[Conflict]:
    """Detect circular dependencies in derived-column expressions.

    Builds a dependency graph, detects cycles, composes the linear
    expressions around each cycle, and checks for fixed points.
    A cycle with no fixed point is a true conflict.

    Fixes (ported from prototype experiments/stage3_conflict_v2/cycles.py):
    - 1-node self-loops (x = x + 1) are now handled, not silently skipped.
    - Cross-table dependencies are resolved via referenced_tables, not assumed
      to live on formula_table.
    - _linear_coeff always returns a 2-tuple, never bare None.
    """
    if not derived:
        return []

    graph: dict[str, list[tuple[str, RExprUnion]]] = {}
    for dc in derived:
        src = f"{dc.target_table}.{dc.target_column}"
        deps = _extract_dep_columns(dc.expression)
        for dep_col in deps:
            dep_key = _resolve_owning_table(
                dep_col, dc.target_table, dc.referenced_tables, derived
            )
            if dep_key is None:
                continue  # can't safely resolve -- skip this edge, don't guess
            full_dep_key = f"{dep_key}.{dep_col}"
            graph.setdefault(full_dep_key, []).append((src, dc.expression))

    nx_graph = nx.DiGraph()
    for src, edges in graph.items():
        for dst, _ in edges:
            nx_graph.add_edge(src, dst)

    conflicts: list[Conflict] = []
    try:
        cycles = list(nx.simple_cycles(nx_graph))
    except nx.NetworkXError:
        return []

    for cycle_nodes in cycles:
        composed = _compose_cycle_expressions(cycle_nodes, derived)
        if composed is None:
            conflicts.append(
                Conflict(
                    conflict_type="derived_cycle",
                    severity="hard",
                    description=(
                        f"Circular dependency cycle ({' -> '.join(cycle_nodes)} "
                        f"-> {cycle_nodes[0]}): non-linear expressions, "
                        f"cannot verify fixed point."
                    ),
                    fact_refs_a=[],
                    fact_refs_b=[],
                    evidence={"cycle": cycle_nodes, "type": "non_linear"},
                )
            )
            continue

        coeff, const = composed
        if math.isclose(coeff, 1.0) and not math.isclose(const, 0.0):
            conflicts.append(
                Conflict(
                    conflict_type="derived_cycle",
                    severity="hard",
                    description=(
                        f"Circular dependency ({' -> '.join(cycle_nodes)} "
                        f"-> {cycle_nodes[0]}): x = x + {const:.4g} "
                        f"has no solution."
                    ),
                    fact_refs_a=[],
                    fact_refs_b=[],
                    evidence={
                        "cycle": cycle_nodes,
                        "coefficient": coeff,
                        "constant": const,
                    },
                )
            )
        elif not math.isclose(coeff, 1.0):
            fixed = const / (1.0 - coeff)
            conflicts.append(
                Conflict(
                    conflict_type="derived_cycle",
                    severity="info",
                    description=(
                        f"Circular dependency ({' -> '.join(cycle_nodes)} "
                        f"-> {cycle_nodes[0]}): x = {coeff:.4g}*x + {const:.4g}, "
                        f"fixed point at x = {fixed:.4g}. Verify against bounds."
                    ),
                    fact_refs_a=[],
                    fact_refs_b=[],
                    evidence={
                        "cycle": cycle_nodes,
                        "coefficient": coeff,
                        "constant": const,
                        "fixed_point": fixed,
                    },
                )
            )

    return conflicts


def _extract_dep_columns(expr: RExprUnion) -> set[str]:
    """Extract column names that a derived expression depends on."""
    cols: set[str] = set()
    _collect_columns_expr(expr, cols)
    return cols


def _resolve_owning_table(
    column: str,
    formula_table: str,
    referenced_tables: tuple[str, ...],
    all_derived: list[DerivedColumnConstraint],
) -> str | None:
    """Which table does `column` actually belong to? Fixes the cross-table
    blind spot: the old code always assumed `formula_table`, silently
    mislabeling any column that lives on a different referenced table.

    A derived column is typically a NEW column not yet in the schema (Stage 3
    emits the DerivedColumnConstraint together with a schema patch), so
    another DerivedColumn's own target_column within the same batch is also a
    valid resolution target, not just pre-existing schema columns."""
    all_derived_by_target: dict[str, str] = {}
    for dc in all_derived:
        all_derived_by_target[f"{dc.target_table}.{dc.target_column}"] = dc.target_table

    candidates = []
    for t in referenced_tables:
        if f"{t}.{column}" in all_derived_by_target:
            candidates.append(t)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        # Fall back to formula_table for same-table derivations
        if f"{formula_table}.{column}" in all_derived_by_target:
            return formula_table
        return None
    return None


def _compose_cycle_expressions(
    cycle: list[str], derived: list[DerivedColumnConstraint]
) -> Optional[tuple[float, float]]:
    """Compose linear expressions around a cycle to get x = A*x + B.

    Each link in the cycle is a linear expression a*x + b (coefficient and
    constant for the next variable in terms of the current). Composition
    multiplies coefficients and accumulates constants.

    Handles 1-node self-loops directly (prototype bug #2 fix): there's
    nothing to compose, just the node's own formula's coefficient against
    itself.

    Returns (A, B) for the composed expression, or None if non-linear.
    """
    if len(cycle) == 1:
        node = cycle[0]
        expr = None
        for dc in derived:
            if f"{dc.target_table}.{dc.target_column}" == node:
                expr = dc.expression
                break
        if expr is None:
            return None
        var_name = node.split(".", 1)[1] if "." in node else node
        a0, b0 = _linear_coeff(expr, var_name)
        if a0 is None or b0 is None:
            return None
        return (a0, b0)

    total_coeff = 1.0
    total_const = 0.0

    for i, node in enumerate(cycle):
        next_node = cycle[(i + 1) % len(cycle)]

        expr = None
        for dc in derived:
            if f"{dc.target_table}.{dc.target_column}" == next_node:
                src_cols = _extract_dep_columns(dc.expression)
                raw_node = node.split(".", 1)[1] if "." in node else node
                if raw_node in src_cols:
                    expr = dc.expression
                    break

        if expr is None:
            return None

        a, b = _linear_coeff(expr, node)
        if a is None:
            return None

        total_const = a * total_const + b
        total_coeff *= a

    return (total_coeff, total_const)


def _linear_coeff(
    expr: RExprUnion, var: str
) -> Tuple[Optional[float], Optional[float]]:
    """Extract coefficient and constant of a linear expression in one variable.

    For an expression like `var * 3 + 2`, returns (3.0, 2.0).
    ALWAYS returns a 2-tuple -- (None, None) means "not linear / can't determine",
    never a bare None (crash fix from prototype bug #1).

    A reference to a different variable (not `var`) is NOT treated as constant
    zero -- returning (0.0, 0.0) for a genuinely different variable is unsound
    the moment a cycle formula also depends on a free, non-cycle variable (e.g.
    `ORDER.total = subtotal + rate` where `subtotal` isn't part of the cycle).
    Instead (None, None) means "unverifiable", not "resolved to 0." """
    raw_var = var.split(".", 1)[1] if "." in var else var

    if isinstance(expr, RColumnRef):
        if expr.name == raw_var:
            return (1.0, 0.0)
        return (None, None)

    if isinstance(expr, RLiteral):
        val = expr.value
        if isinstance(val, (int, float)):
            return (0.0, float(val))
        return (None, None)

    if isinstance(expr, RArithmetic):
        left_coeff, left_const = _linear_coeff(expr.left, var)
        right_coeff, right_const = _linear_coeff(expr.right, var)

        if left_coeff is None or right_coeff is None:
            return (None, None)

        match expr.op:
            case "+":
                return (left_coeff + right_coeff, left_const + right_const)
            case "-":
                return (left_coeff - right_coeff, left_const - right_const)
            case "*":
                if math.isclose(left_coeff, 0.0):
                    return (right_coeff * left_const, right_const * left_const)
                if math.isclose(right_coeff, 0.0):
                    return (left_coeff * right_const, left_const * right_const)
                return (None, None)
            case "/":
                if math.isclose(right_coeff, 0.0) and right_const != 0:
                    return (left_coeff / right_const, left_const / right_const)
                return (None, None)

    return (None, None)


def detect_distribution_logic(
    distributions: list[DistributionConstraint],
    logic: list[Constraint],
) -> list[Conflict]:
    """Flag distribution + logic constraint pairs on the same column.

    When a distribution and a logic constraint apply to the same column
    with overlapping conditions, we flag for LLM review. We don't attempt
    precise distribution-vs-range analysis.
    """
    conflicts: list[Conflict] = []
    seen: set[tuple[int, int]] = set()

    for di, dc in enumerate(distributions):
        for li, lc in enumerate(logic):
            pair_key = (di, li)
            if pair_key in seen:
                continue
            seen.add(pair_key)

            if dc.column not in extract_columns(lc.condition):
                continue

            dc_if = dc.if_condition or RComparison(
                op="=",
                left=RColumnRef(name="_always_true"),
                right=RLiteral(value=True),
            )

            if isinstance(lc.condition, RIfThen):
                lc_applicability = lc.condition.antecedent
            else:
                lc_applicability = RComparison(
                    op="=",
                    left=RColumnRef(name="_always_true"),
                    right=RLiteral(value=True),
                )

            overlap = conditions_overlap(dc_if, lc_applicability)

            if overlap == "no":
                continue

            conflicts.append(
                Conflict(
                    conflict_type="dist_logic",
                    severity="soft",
                    description=(
                        f"Column '{dc.column}': distribution {dc.family} "
                        f"and logic constraint may conflict (overlap={overlap}). "
                        f"Flagged for LLM review."
                    ),
                    fact_refs_a=dc.fact_references,
                    fact_refs_b=lc.fact_references,
                    evidence={
                        "column": dc.column,
                        "distribution_family": dc.family,
                        "overlap": overlap,
                    },
                )
            )

    return conflicts


# ---------------------------------------------------------------------------
# Layer 4: Entry point
# ---------------------------------------------------------------------------


def detect_all_conflicts(
    constraints: list[Constraint] | None = None,
    distributions: list[DistributionConstraint] | None = None,
    derived: list[DerivedColumnConstraint] | None = None,
) -> ConflictReport:
    """Run all conflict detection layers and aggregate results.

    Accepts the three output types from extraction agents. Each list is
    optional (defaults to empty).
    """
    constraints = constraints or []
    distributions = distributions or []
    derived = derived or []

    all_constraints: list[Constraint] = list(constraints)
    all_constraints.extend(distributions)

    col_conflicts = detect_column_conflicts(all_constraints)
    struct_conflicts = detect_structural_conflicts(all_constraints)
    cycle_conflicts = detect_derived_cycles(derived)
    dist_logic_conflicts = detect_distribution_logic(distributions, constraints)

    all_conflicts = (
        col_conflicts + struct_conflicts + cycle_conflicts + dist_logic_conflicts
    )
    total_checked = len(all_constraints) * (len(all_constraints) - 1) // 2

    return ConflictReport(
        conflicts=all_conflicts,
        checked_pairs=total_checked,
        is_consistent=len(all_conflicts) == 0,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _on_tables(node: ONNode) -> set[str]:
    """Extract all base table names from an ON tree."""
    from src.pipeline.stage3.models.on_nodes import extract_tables

    return extract_tables(node)
