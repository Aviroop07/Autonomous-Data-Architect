"""Constraint Satisfaction Rate: does generated data obey the stated rules?

The other half of data-level evaluation. Distribution Fidelity (data_eval.py)
asks whether a column's VALUES are distributed as specified; this asks whether
the RELATIONSHIPS between columns hold -- a discharge never before an admission,
a discount never above a quarter of the total, a controlled medication only
prescribed by a senior clinician.

Without it Stage 4's output cannot be scored at all, which is why this file
exists before Stage 4 does. It is a pure function of (data, constraints), so it
is testable today by construction: generate rows that satisfy a constraint and
CSR must be 1.0; violate them deliberately and it must fall by exactly the share
violated.

VACUOUS SATISFACTION IS REPORTED, NOT HIDDEN. An `ifthen` whose condition never
fires is trivially satisfied and measures nothing, so a dataset that dodges every
antecedent would otherwise score a perfect 1.0. `applicable_rows` travels with
every result for the same reason `ic_n_required_facts` travels with IC-Recall --
this project has already been bitten by a metric that returned 1.0 from an empty
set.

Grammar implemented, taken from the 1,866 constraints in the benchmark rather
than from the authoring contract's prose:

  top level   range | ifthen
  predicates  eq neq gt gte lt lte | and or | range
  left side   a column on the constraint's own table
  right side  a literal `value`
              `rhs_column` on the same table
              `rhs_column` + `rhs_table_ref` + `rhs_join` (across one join)
              `rhs_expr` -- one arithmetic step, optionally across a join
  leaves may also carry `table_ref` + `join` to compare a JOINED column
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

Row = Mapping[str, Any]

_COMPARATORS = {
    "eq": lambda a, b: a == b,
    "neq": lambda a, b: a != b,
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
}

_ARITHMETIC = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b if b else None,
}


class Unevaluable(Exception):
    """The constraint cannot be checked against this data.

    Raised rather than returning False, because "the rule is broken" and "the
    rule could not be tested" are different findings and conflating them would
    let a missing column masquerade as a violation.
    """


class GeneratedData:
    """Row-oriented view of one generated database, with join resolution.

    Rows stay as plain mappings deliberately: their columns are whatever the
    generated schema happens to have, so there is no fixed model to bind them
    to. Everything ABOUT the data that has structure -- lookups, joins, results
    -- is typed.
    """

    def __init__(self, tables: Mapping[str, Sequence[Row]]) -> None:
        self._tables = {name: list(rows) for name, rows in tables.items()}
        self._indexes: Dict[tuple, Dict[Any, Row]] = {}

    def table(self, name: str) -> List[Row]:
        if name not in self._tables:
            raise Unevaluable(f"table {name!r} is not present in the generated data")
        return self._tables[name]

    def lookup(self, table: str, key_column: str, key: Any) -> Optional[Row]:
        """First row of `table` whose `key_column` equals `key`.

        Indexed on first use per (table, column): a constraint checked over
        50,000 rows would otherwise rescan the joined table once per row, which
        turns an O(n) check into O(n*m) and made the first draft unusable on
        realistic volumes.
        """
        cache_key = (table, key_column)
        if cache_key not in self._indexes:
            index: Dict[Any, Row] = {}
            for row in self.table(table):
                if key_column not in row:
                    raise Unevaluable(
                        f"join key {table}.{key_column} is not a column of {table}"
                    )
                index.setdefault(row[key_column], row)
            self._indexes[cache_key] = index
        return self._indexes[cache_key].get(key)


@dataclass
class ConstraintResult:
    """One constraint's verdict over the whole dataset."""

    satisfied_rows: int = 0
    violated_rows: int = 0
    #: Rows the constraint actually TESTED. For an ifthen this excludes rows
    #: whose condition was false -- those are vacuously satisfied and say
    #: nothing about the data.
    applicable_rows: int = 0
    #: Rows skipped because the antecedent did not hold. Reported so a dataset
    #: that dodges every antecedent cannot look perfect.
    vacuous_rows: int = 0
    unevaluable_reason: Optional[str] = None

    @property
    def is_evaluable(self) -> bool:
        return self.unevaluable_reason is None

    @property
    def rate(self) -> Optional[float]:
        """Share of APPLICABLE rows satisfying the constraint, or None.

        None rather than 1.0 when nothing was applicable: a constraint that
        never applied has no satisfaction rate, and reporting 1.0 would let
        vacuity inflate the aggregate.
        """
        if not self.is_evaluable or self.applicable_rows == 0:
            return None
        return self.satisfied_rows / self.applicable_rows


@dataclass
class CSRReport:
    per_constraint: List[ConstraintResult] = field(default_factory=list)

    @property
    def n_evaluated(self) -> int:
        return sum(1 for r in self.per_constraint if r.rate is not None)

    @property
    def n_unevaluable(self) -> int:
        return sum(1 for r in self.per_constraint if not r.is_evaluable)

    @property
    def n_vacuous(self) -> int:
        """Evaluable, but no row ever triggered them."""
        return sum(
            1 for r in self.per_constraint if r.is_evaluable and r.applicable_rows == 0
        )

    @property
    def csr(self) -> Optional[float]:
        """Row-weighted satisfaction across every applicable row.

        Row-weighted rather than a mean of per-constraint rates, so a constraint
        that applies to three rows cannot outvote one that applies to thirty
        thousand. None when nothing was applicable at all.
        """
        applicable = sum(r.applicable_rows for r in self.per_constraint)
        if applicable == 0:
            return None
        return sum(r.satisfied_rows for r in self.per_constraint) / applicable

    def as_dict(self) -> Dict[str, Any]:
        return {
            "csr": self.csr,
            "n_constraints": len(self.per_constraint),
            "n_evaluated": self.n_evaluated,
            "n_vacuous": self.n_vacuous,
            "n_unevaluable": self.n_unevaluable,
            "applicable_rows": sum(r.applicable_rows for r in self.per_constraint),
            "violated_rows": sum(r.violated_rows for r in self.per_constraint),
        }


def _resolve_operand(node: Mapping[str, Any], row: Row, data: GeneratedData) -> Any:
    """Value of a joined-or-local column reference for one row.

    Reads ONLY the `join`/`table_ref` spelling, never `rhs_join`/`rhs_table_ref`.
    Accepting both looked harmless and was not: a predicate carries its right
    side's join on the SAME dict as its left column, so the left operand was
    resolved through the right side's join and looked for `charged` on the
    joined table. The right-hand path re-keys into this spelling before calling
    here, so the two sides stay separate by construction.
    """
    column = node["column"]
    join = node.get("join")
    table_ref = node.get("table_ref")

    if not join:
        if column not in row:
            raise Unevaluable(f"column {column!r} is not present on the row")
        return row[column]

    from_ref, to_ref = join["from"], join["to"]
    local_col = from_ref.split(".", 1)[1] if "." in from_ref else from_ref
    target_table = table_ref or (to_ref.split(".", 1)[0] if "." in to_ref else None)
    target_col = to_ref.split(".", 1)[1] if "." in to_ref else to_ref
    if target_table is None:
        raise Unevaluable(f"join {from_ref} -> {to_ref} names no target table")
    if local_col not in row:
        raise Unevaluable(f"join column {local_col!r} is not present on the row")

    joined = data.lookup(target_table, target_col, row[local_col])
    if joined is None:
        # A dangling FK is a referential-integrity defect, not a violation of
        # THIS rule; counting it as one would blame the wrong thing.
        raise Unevaluable(
            f"no {target_table} row matches {local_col}={row[local_col]!r}"
        )
    if column not in joined:
        raise Unevaluable(f"column {column!r} is not present on {target_table}")
    return joined[column]


def _right_hand_value(node: Mapping[str, Any], row: Row, data: GeneratedData) -> Any:
    if "value" in node and "rhs_expr" not in node:
        return node["value"]
    if "rhs_column" in node:
        return _resolve_operand(
            {
                "column": node["rhs_column"],
                "join": node.get("rhs_join"),
                "table_ref": node.get("rhs_table_ref"),
            },
            row,
            data,
        )
    if "rhs_expr" in node:
        expr = node["rhs_expr"]
        base = _resolve_operand(expr, row, data)
        op = expr.get("op")
        if op is None:
            return base
        if op not in _ARITHMETIC:
            raise Unevaluable(f"unsupported arithmetic operator {op!r}")
        if base is None:
            raise Unevaluable("arithmetic on a null operand")
        result = _ARITHMETIC[op](base, expr["value"])
        if result is None:
            raise Unevaluable("division by zero in rhs_expr")
        return result
    raise Unevaluable(f"predicate has no right-hand side: {sorted(node)}")


def _holds(node: Mapping[str, Any], row: Row, data: GeneratedData) -> bool:
    """Does this predicate hold for one row?"""
    kind = node.get("type")

    if kind == "and":
        return all(_holds(c, row, data) for c in node.get("conditions") or [])
    if kind == "or":
        return any(_holds(c, row, data) for c in node.get("conditions") or [])

    if kind == "range":
        left = _resolve_operand(node, row, data)
        if left is None:
            raise Unevaluable(f"column {node['column']!r} is null")
        lo, hi = node.get("min"), node.get("max")
        return (lo is None or left >= lo) and (hi is None or left <= hi)

    if kind in _COMPARATORS:
        left = _resolve_operand(node, row, data)
        right = _right_hand_value(node, row, data)
        if left is None or right is None:
            # NULL comparisons are neither satisfied nor violated in SQL, and
            # treating them as violations would penalise legitimately optional
            # columns.
            raise Unevaluable("comparison against a null value")
        try:
            return _COMPARATORS[kind](left, right)
        except TypeError as exc:
            raise Unevaluable(f"cannot compare {left!r} with {right!r}: {exc}") from exc

    raise Unevaluable(f"unsupported predicate type {kind!r}")


def evaluate_constraint(
    constraint: Mapping[str, Any], data: GeneratedData
) -> ConstraintResult:
    """Check one ground-truth constraint over every row of its table."""
    result = ConstraintResult()
    kind = constraint.get("type")
    table = constraint.get("table")
    if not table:
        result.unevaluable_reason = "constraint names no table"
        return result

    try:
        rows = data.table(table)
    except Unevaluable as exc:
        result.unevaluable_reason = str(exc)
        return result

    unevaluable_rows = 0
    last_reason: Optional[str] = None

    for row in rows:
        try:
            if kind == "ifthen":
                if not _holds(constraint["condition"], row, data):
                    result.vacuous_rows += 1
                    continue
                ok = _holds(constraint["result"], row, data)
            else:
                ok = _holds(constraint, row, data)
        except Unevaluable as exc:
            unevaluable_rows += 1
            last_reason = str(exc)
            continue

        result.applicable_rows += 1
        if ok:
            result.satisfied_rows += 1
        else:
            result.violated_rows += 1

    # Only a constraint that could not be checked ANYWHERE is unevaluable. A few
    # skipped rows (a null, a dangling FK) must not discard the verdict from the
    # rows that were checkable.
    if result.applicable_rows == 0 and unevaluable_rows > 0:
        result.unevaluable_reason = last_reason
    elif unevaluable_rows:
        logger.debug(
            "[CSR] %s constraint on %s: %d row(s) skipped as uncheckable (%s)",
            kind,
            table,
            unevaluable_rows,
            last_reason,
        )
    return result


def constraint_satisfaction_rate(
    constraints: Sequence[Mapping[str, Any]], data: GeneratedData
) -> CSRReport:
    """CSR over a whole case: every ground-truth constraint against the data."""
    report = CSRReport(
        per_constraint=[evaluate_constraint(c, data) for c in constraints]
    )
    if report.n_unevaluable:
        logger.warning(
            "[CSR] %d of %d constraint(s) could not be evaluated against this "
            "data at all; they are excluded from the rate rather than counted "
            "as satisfied.",
            report.n_unevaluable,
            len(report.per_constraint),
        )
    if report.n_vacuous:
        logger.info(
            "[CSR] %d of %d constraint(s) were never triggered by any row -- "
            "vacuously true and excluded, so they neither raise nor lower the "
            "rate.",
            report.n_vacuous,
            len(report.per_constraint),
        )
    return report
