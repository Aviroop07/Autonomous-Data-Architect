"""ScribbleDB -- deterministic validator and coverage report for a cases file.

Every check here is mechanical. Authoring benchmark ground truth by hand (or by
agent) produces exactly the class of error a machine should catch: a foreign key
pointing at a table that was renamed, a distribution parameter named for the
wrong family, a constraint referencing a column that does not exist. None of
those raise anywhere in the pipeline -- the data-level evaluator turns an
unreadable ground-truth spec into a WORST-CASE SCORE rather than an error, so a
malformed case silently reports as a terrible one.

Usage
-----
  python validate_dataset.py                          # validate the default cases file
  python validate_dataset.py --cases path/to.jsonl    # a specific file
  python validate_dataset.py --coverage               # also print the coverage report
  python validate_dataset.py --quiet                  # errors only

Exit code is non-zero when any ERROR is found, so this works as a gate.
Warnings never fail the run: they flag things worth a human look (a nominal
categorical, a table with no foreign keys) that are not necessarily wrong.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).parent
DEFAULT_CASES = PROJECT_ROOT / "dataset" / "handcrafted" / "cases.jsonl"

# The pipeline's own type vocabulary -- see src/util/schema_model/data_types.py.
# Duplicated as a literal rather than imported so this script stays runnable
# without the package installed, and mismatches are caught by a unit test.
DATA_TYPES: Set[str] = {
    "INTEGER",
    "VARCHAR",
    "FLOAT",
    "DECIMAL",
    "BOOLEAN",
    "DATE",
    "DATETIME",
    "TIMESTAMP",
    "TIME",
    "TEXT",
    "UUID",
}

NUMERIC_TYPES: Set[str] = {"INTEGER", "FLOAT", "DECIMAL"}

# Required parameter names per distribution family, in the vocabulary
# cases.jsonl is authored in (data_eval._parse_gt_dist translates these).
DIST_PARAMS: Dict[str, Set[str]] = {
    "normal": {"mean", "std"},
    "lognormal": {"mean", "variance"},
    "uniform": {"min", "max"},
    "poisson": {"lambda"},
    "exponential": {"lambda"},
    "zipf": {"a"},
    "categorical": {"weights"},
}

CONSTRAINT_TYPES: Set[str] = {"ifthen", "range"}

UPPER_SNAKE = re.compile(r"^[A-Z][A-Z0-9_]*$")
LOWER_SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")

REQUIRED_CASE_FIELDS = (
    "id",
    "domain",
    "profile",
    "nl_description",
    "ground_truth_schema",
    "ground_truth_distributions",
    "ground_truth_constraints",
)


class Findings:
    """Errors fail the run; warnings are reported and do not."""

    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def error(self, where: str, message: str) -> None:
        self.errors.append(f"{where}: {message}")

    def warn(self, where: str, message: str) -> None:
        self.warnings.append(f"{where}: {message}")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def _check_schema(where: str, schema: Any, f: Findings) -> Dict[str, Dict[str, str]]:
    """Validate ground_truth_schema; return {table: {column: data_type}}."""
    resolved: Dict[str, Dict[str, str]] = {}
    if not isinstance(schema, dict):
        f.error(where, "ground_truth_schema is not an object")
        return resolved

    tables = schema.get("tables")
    if not isinstance(tables, list) or not tables:
        f.error(where, "ground_truth_schema.tables is missing or empty")
        return resolved

    seen_tables: Counter = Counter()
    for t in tables:
        if not isinstance(t, dict):
            f.error(where, "a table entry is not an object")
            continue
        name = t.get("name")
        if not isinstance(name, str) or not name:
            f.error(where, "a table has no name")
            continue
        seen_tables[name] += 1
        if not UPPER_SNAKE.fullmatch(name):
            f.error(where, f"table '{name}' is not UPPER_SNAKE_CASE")

        cols = t.get("columns")
        if not isinstance(cols, list) or not cols:
            f.error(where, f"table '{name}' has no columns")
            continue

        col_types: Dict[str, str] = {}
        seen_cols: Counter = Counter()
        for c in cols:
            if not isinstance(c, dict):
                f.error(where, f"table '{name}' has a non-object column")
                continue
            cname = c.get("name")
            ctype = c.get("data_type")
            if not isinstance(cname, str) or not cname:
                f.error(where, f"table '{name}' has an unnamed column")
                continue
            seen_cols[cname] += 1
            if not LOWER_SNAKE.fullmatch(cname):
                f.error(where, f"column '{name}.{cname}' is not lower_snake_case")
            if ctype not in DATA_TYPES:
                f.error(
                    where,
                    f"column '{name}.{cname}' has data_type {ctype!r}, "
                    f"not one of {sorted(DATA_TYPES)}",
                )
            col_types[cname] = ctype if isinstance(ctype, str) else ""
        for cname, n in seen_cols.items():
            if n > 1:
                f.error(where, f"table '{name}' declares column '{cname}' {n} times")

        pk = t.get("pk")
        if pk is None:
            f.error(where, f"table '{name}' has no pk")
        else:
            pk_cols = pk if isinstance(pk, list) else [pk]
            for p in pk_cols:
                if p not in col_types:
                    f.error(
                        where,
                        f"table '{name}' pk names '{p}', which is not one of its columns",
                    )
        resolved[name] = col_types

    for name, n in seen_tables.items():
        if n > 1:
            f.error(where, f"table name '{name}' is used {n} times")

    _check_relationships(where, schema, resolved, f)
    return resolved


def _check_relationships(
    where: str, schema: Dict[str, Any], tables: Dict[str, Dict[str, str]], f: Findings
) -> None:
    rels = schema.get("relationships")
    if rels is None:
        if len(tables) > 1:
            f.warn(where, f"{len(tables)} tables but no relationships declared")
        return
    if not isinstance(rels, list):
        f.error(where, "ground_truth_schema.relationships is not a list")
        return

    for r in rels:
        if not isinstance(r, dict):
            f.error(where, "a relationship entry is not an object")
            continue
        rt, rc = r.get("referencing_table"), r.get("referencing_column")
        dt, dc = r.get("referred_table"), r.get("referred_column")
        for label, tbl, col in (
            ("referencing", rt, rc),
            ("referred", dt, dc),
        ):
            if tbl not in tables:
                f.error(
                    where, f"FK {label}_table '{tbl}' is not a table in this schema"
                )
            elif col not in tables[tbl]:
                f.error(where, f"FK {label} column '{tbl}.{col}' does not exist")


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


def _is_nominal(weights: Any) -> bool:
    if not isinstance(weights, dict):
        return False
    for label in weights:
        try:
            float(label)
        except TypeError, ValueError:
            return True
    return False


def _check_distributions(
    where: str, dists: Any, tables: Dict[str, Dict[str, str]], f: Findings
) -> None:
    if dists is None:
        return
    if not isinstance(dists, dict):
        f.error(where, "ground_truth_distributions is not an object")
        return

    for key, spec in dists.items():
        if not isinstance(key, str) or key.count(".") != 1:
            f.error(where, f"distribution key {key!r} is not 'TABLE.column'")
            continue
        tname, cname = key.split(".", 1)
        if tname not in tables:
            f.error(where, f"distribution on '{key}': no such table")
            continue
        if cname not in tables[tname]:
            f.error(where, f"distribution on '{key}': no such column")
            continue

        if not isinstance(spec, dict):
            f.error(where, f"distribution on '{key}' is not an object")
            continue
        family = spec.get("distribution", spec.get("family"))
        if family not in DIST_PARAMS:
            f.error(
                where,
                f"distribution on '{key}' names family {family!r}, "
                f"not one of {sorted(DIST_PARAMS)}",
            )
            continue

        params = spec.get("params")
        if not isinstance(params, dict):
            f.error(where, f"distribution on '{key}' has no params object")
            continue
        required = DIST_PARAMS[family]
        missing = required - set(params)
        extra = set(params) - required
        if missing:
            f.error(
                where,
                f"distribution on '{key}' ({family}) is missing params {sorted(missing)}",
            )
        if extra:
            f.error(
                where,
                f"distribution on '{key}' ({family}) has unexpected params "
                f"{sorted(extra)}; {family} takes {sorted(required)}",
            )
        if missing or extra:
            continue

        _check_dist_values(where, key, family, params, tables[tname][cname], f)


def _check_dist_values(
    where: str,
    key: str,
    family: str,
    params: Dict[str, Any],
    col_type: str,
    f: Findings,
) -> None:
    def num(name: str) -> Optional[float]:
        try:
            return float(params[name])
        except TypeError, ValueError:
            f.error(where, f"distribution on '{key}': param '{name}' is not a number")
            return None

    if family == "categorical":
        weights = params.get("weights")
        if not isinstance(weights, dict) or not weights:
            f.error(where, f"distribution on '{key}': weights must be a non-empty map")
            return
        total = 0.0
        for label, w in weights.items():
            try:
                wf = float(w)
            except TypeError, ValueError:
                f.error(
                    where,
                    f"distribution on '{key}': weight for {label!r} is not a number",
                )
                return
            if wf < 0:
                f.error(
                    where, f"distribution on '{key}': negative weight for {label!r}"
                )
            total += wf
        if abs(total - 1.0) > 0.01:
            f.error(
                where,
                f"distribution on '{key}': weights sum to {total:.4f}, expected 1.0",
            )
        if _is_nominal(weights):
            f.warn(
                where,
                f"distribution on '{key}' is a NOMINAL categorical. Scored with "
                "total variation distance rather than KS, since nominal labels have "
                "no ordering to accumulate along. Informational only.",
            )
        return

    # Numeric families on a non-numeric column is almost always an authoring slip.
    if col_type and col_type not in NUMERIC_TYPES:
        f.warn(
            where,
            f"distribution on '{key}' is {family} but the column is {col_type}",
        )

    if family == "normal":
        std = num("std")
        if std is not None and std <= 0:
            f.error(where, f"distribution on '{key}': std must be > 0, got {std}")
    elif family == "lognormal":
        var = num("variance")
        if var is not None and var <= 0:
            f.error(where, f"distribution on '{key}': variance must be > 0, got {var}")
    elif family == "uniform":
        lo, hi = num("min"), num("max")
        if lo is not None and hi is not None and hi <= lo:
            f.error(
                where, f"distribution on '{key}': max ({hi}) must exceed min ({lo})"
            )
    elif family in ("poisson", "exponential"):
        lam = num("lambda")
        if lam is not None and lam <= 0:
            f.error(where, f"distribution on '{key}': lambda must be > 0, got {lam}")
    elif family == "zipf":
        a = num("a")
        if a is not None and a <= 1.0:
            f.error(
                where,
                f"distribution on '{key}': zipf 'a' must be > 1 for a finite mean, got {a}",
            )


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def _walk_condition(
    where: str,
    label: str,
    node: Any,
    home_table: str,
    tables: Dict[str, Dict[str, str]],
    f: Findings,
) -> None:
    """Resolve every column a condition tree mentions."""
    if not isinstance(node, dict):
        f.error(where, f"{label}: condition node is not an object")
        return

    ntype = node.get("type")
    if ntype in ("and", "or"):
        subs = node.get("conditions")
        if not isinstance(subs, list) or not subs:
            f.error(where, f"{label}: '{ntype}' node has no conditions list")
            return
        for sub in subs:
            _walk_condition(where, label, sub, home_table, tables, f)
        return
    if ntype == "not":
        inner = node.get("condition")
        if inner is None:
            f.error(where, f"{label}: 'not' node has no condition")
            return
        _walk_condition(where, label, inner, home_table, tables, f)
        return

    col = node.get("column")
    if col is None:
        f.error(where, f"{label}: leaf node of type {ntype!r} names no column")
        return
    tbl = node.get("table_ref") or home_table
    if tbl not in tables:
        f.error(where, f"{label}: references table '{tbl}', which is not in the schema")
        return
    if col not in tables[tbl]:
        f.error(
            where, f"{label}: references column '{tbl}.{col}', which does not exist"
        )

    _check_join(where, label, node.get("join"), "join", tables, f)

    # Right-hand side. Exactly one of value / rhs_column / rhs_expr, so a leaf
    # can never silently mean two things. rhs_column and rhs_expr are what make
    # a CROSS-COLUMN constraint expressible at all: the pipeline's own
    # RComparison takes a full expression on BOTH sides, so ground truth able
    # only to compare a column against a literal was strictly weaker than the
    # thing it is supposed to score.
    rhs_forms = [k for k in ("value", "rhs_column", "rhs_expr") if k in node]
    if not rhs_forms:
        f.error(
            where,
            f"{label}: leaf has no right-hand side; give one of value, "
            "rhs_column or rhs_expr",
        )
        return
    if len(rhs_forms) > 1:
        f.error(where, f"{label}: leaf carries several right-hand sides {rhs_forms}")
        return

    if "rhs_column" in node:
        rhs_tbl = node.get("rhs_table_ref") or tbl
        rhs_col = node.get("rhs_column")
        if rhs_tbl not in tables:
            f.error(where, f"{label}: rhs_table_ref '{rhs_tbl}' is not in the schema")
        elif rhs_col not in tables[rhs_tbl]:
            f.error(where, f"{label}: rhs column '{rhs_tbl}.{rhs_col}' does not exist")
        _check_join(where, label, node.get("rhs_join"), "rhs_join", tables, f)
        return

    if "rhs_expr" in node:
        expr = node.get("rhs_expr")
        if not isinstance(expr, dict):
            f.error(where, f"{label}: rhs_expr is not an object")
            return
        if expr.get("op") not in ("+", "-", "*", "/"):
            f.error(
                where, f"{label}: rhs_expr.op {expr.get('op')!r} is not one of + - * /"
            )
        e_tbl = expr.get("table_ref") or tbl
        e_col = expr.get("column")
        if e_tbl not in tables:
            f.error(
                where, f"{label}: rhs_expr.table_ref '{e_tbl}' is not in the schema"
            )
        elif e_col not in tables[e_tbl]:
            f.error(where, f"{label}: rhs_expr column '{e_tbl}.{e_col}' does not exist")
        operands = [k for k in ("value", "rhs_column") if k in expr]
        if len(operands) != 1:
            f.error(
                where,
                f"{label}: rhs_expr needs exactly one of value / rhs_column, "
                f"got {operands}",
            )
        elif "rhs_column" in expr:
            o_tbl = expr.get("rhs_table_ref") or e_tbl
            o_col = expr.get("rhs_column")
            if o_tbl not in tables:
                f.error(where, f"{label}: rhs_expr.rhs_table_ref '{o_tbl}' is unknown")
            elif o_col not in tables[o_tbl]:
                f.error(
                    where,
                    f"{label}: rhs_expr column '{o_tbl}.{o_col}' does not exist",
                )
        _check_join(where, label, expr.get("join"), "rhs_expr.join", tables, f)


def _check_join(
    where: str,
    label: str,
    join: Any,
    field_name: str,
    tables: Dict[str, Dict[str, str]],
    f: Findings,
) -> None:
    """Resolve both endpoints of a join, whichever field carried it."""
    if join is None:
        return
    if not isinstance(join, dict):
        f.error(where, f"{label}: {field_name} is not an object")
        return
    for side in ("from", "to"):
        ref = join.get(side)
        if not isinstance(ref, str) or ref.count(".") != 1:
            f.error(
                where, f"{label}: {field_name}.{side} {ref!r} is not 'TABLE.column'"
            )
            continue
        jt, jc = ref.split(".", 1)
        if jt not in tables:
            f.error(where, f"{label}: {field_name}.{side} names unknown table '{jt}'")
        elif jc not in tables[jt]:
            f.error(
                where,
                f"{label}: {field_name}.{side} names unknown column '{jt}.{jc}'",
            )


def _check_constraints(
    where: str, constraints: Any, tables: Dict[str, Dict[str, str]], f: Findings
) -> None:
    if constraints is None:
        return
    if not isinstance(constraints, list):
        f.error(where, "ground_truth_constraints is not a list")
        return

    for i, c in enumerate(constraints):
        label = f"constraint[{i}]"
        if not isinstance(c, dict):
            f.error(where, f"{label} is not an object")
            continue
        ctype = c.get("type")
        if ctype not in CONSTRAINT_TYPES:
            f.error(
                where,
                f"{label} has type {ctype!r}, not one of {sorted(CONSTRAINT_TYPES)}",
            )
            continue
        table = c.get("table")
        if table not in tables:
            f.error(where, f"{label} names table {table!r}, which is not in the schema")
            continue

        if ctype == "range":
            col = c.get("column")
            if col not in tables[table]:
                f.error(where, f"{label}: column '{table}.{col}' does not exist")
            lo, hi = c.get("min"), c.get("max")
            if lo is None and hi is None:
                f.error(where, f"{label}: a range needs at least one of min/max")
            if lo is not None and hi is not None:
                try:
                    if float(hi) < float(lo):
                        f.error(where, f"{label}: max ({hi}) is below min ({lo})")
                except TypeError, ValueError:
                    f.error(where, f"{label}: min/max are not numbers")
            if c.get("condition") is not None:
                _walk_condition(where, label, c["condition"], table, tables, f)
        else:  # ifthen
            cond, result = c.get("condition"), c.get("result")
            if cond is None:
                f.error(where, f"{label}: ifthen has no condition")
            else:
                _walk_condition(where, f"{label}.condition", cond, table, tables, f)
            if result is None:
                f.error(where, f"{label}: ifthen has no result")
            else:
                _walk_condition(where, f"{label}.result", result, table, tables, f)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def load_cases(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    cases: List[Dict[str, Any]] = []
    if not path.exists():
        return cases, [f"{path} does not exist"]
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {lineno}: invalid JSON: {exc}")
            continue
        if not isinstance(obj, dict):
            errors.append(f"line {lineno}: top-level value is not an object")
            continue
        cases.append(obj)
    return cases, errors


def validate(cases: Iterable[Dict[str, Any]]) -> Findings:
    f = Findings()
    ids: Counter = Counter()

    for case in cases:
        cid = case.get("id", "?")
        where = f"case {cid}"
        ids[cid] += 1

        for field in REQUIRED_CASE_FIELDS:
            if field not in case:
                f.error(where, f"missing required field '{field}'")

        nl = case.get("nl_description")
        if not isinstance(nl, str) or len(nl.strip()) < 80:
            f.error(where, "nl_description is missing or implausibly short")

        for field in ("domain", "profile"):
            if not isinstance(case.get(field), str) or not case.get(field):
                f.error(where, f"'{field}' must be a non-empty string")

        tables = _check_schema(where, case.get("ground_truth_schema"), f)
        _check_fds(where, case.get("functional_dependencies"), tables, f)
        if tables:
            _check_distributions(
                where, case.get("ground_truth_distributions"), tables, f
            )
            _check_constraints(where, case.get("ground_truth_constraints"), tables, f)
            _check_moments(
                where,
                case.get("ground_truth_moments"),
                tables,
                case.get("nl_description") or "",
                f,
            )

    for cid, n in ids.items():
        if n > 1:
            f.error("dataset", f"id {cid} appears {n} times")
    return f


def _check_fds(
    where: str, fds: Any, tables: Dict[str, Dict[str, str]], f: Findings
) -> None:
    """Ground-truth functional dependencies, if the case carries any.

    Optional, but when present they turn the KDC normalisation check from a
    circular self-comparison into a real measurement, so a malformed one is worth
    an error rather than a shrug.
    """
    if fds is None or not tables:
        return
    if not isinstance(fds, list):
        f.error(where, "functional_dependencies is not a list")
        return
    for i, fd in enumerate(fds):
        label = f"functional_dependencies[{i}]"
        if not isinstance(fd, dict):
            f.error(where, f"{label} is not an object")
            continue
        for side in ("determinant", "dependent"):
            refs = fd.get(side)
            if not isinstance(refs, list) or not refs:
                f.error(where, f"{label}.{side} must be a non-empty list")
                continue
            for ref in refs:
                if not isinstance(ref, str) or ref.count(".") != 1:
                    f.error(where, f"{label}.{side} {ref!r} is not 'TABLE.column'")
                    continue
                tbl, col = ref.split(".", 1)
                if tbl not in tables:
                    f.error(where, f"{label}.{side} names unknown table '{tbl}'")
                elif col not in tables[tbl]:
                    f.error(where, f"{label}.{side} names unknown column '{ref}'")


_MOMENT_AGGREGATES = {"avg", "sum", "count", "min", "max"}


def _check_moments(
    where: str,
    moments: Any,
    tables: Dict[str, Dict[str, str]],
    nl: str,
    f: Findings,
) -> None:
    """Ground-truth moment targets -- a stated average, total or count.

    Stage 3 emits seven constraint families and the benchmark could previously
    ground-truth two of them, so moment targets are the largest scoreable gap.
    They are also the easiest family to INVENT, which is what the `evidence`
    field is for: every moment must quote the phrase of the specification that
    states it, verbatim. That turns "did the author make this up" from a matter
    of trust into a deterministic check, and it is the same device that keeps the
    cross-column enrichment honest -- refuse to accept anything whose supporting
    text or whose named column is not really there.

    A moment the prose does not state is worse than a missing one: Stage 3 would
    be scored on extracting a fact that does not exist in its input.
    """
    if moments is None or not tables:
        return
    if not isinstance(moments, list):
        f.error(where, "ground_truth_moments is not a list")
        return

    normalised_nl = " ".join((nl or "").split()).lower()
    for i, m in enumerate(moments):
        label = f"ground_truth_moments[{i}]"
        if not isinstance(m, dict):
            f.error(where, f"{label} is not an object")
            continue

        table, column = m.get("table"), m.get("column")
        if table not in tables:
            f.error(where, f"{label} names unknown table {table!r}")
        elif column not in tables[table]:
            f.error(where, f"{label} names unknown column '{table}.{column}'")

        agg = m.get("aggregate")
        if agg not in _MOMENT_AGGREGATES:
            f.error(
                where,
                f"{label}.aggregate {agg!r} is not one of {sorted(_MOMENT_AGGREGATES)}",
            )

        if not isinstance(m.get("value"), (int, float)) or isinstance(
            m.get("value"), bool
        ):
            f.error(where, f"{label}.value must be a number, got {m.get('value')!r}")

        evidence = m.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            f.error(where, f"{label}.evidence must quote the phrase that states it")
        else:
            # Whitespace-normalised so a line break inside the quoted phrase is
            # not treated as invention; everything else must match the source.
            if " ".join(evidence.split()).lower() not in normalised_nl:
                f.error(
                    where,
                    f"{label}.evidence is not a verbatim phrase of nl_description: "
                    f"{evidence[:60]!r}",
                )


def coverage(cases: List[Dict[str, Any]]) -> str:
    families: Counter = Counter()
    ctypes: Counter = Counter()
    profiles: Counter = Counter()
    domains: Counter = Counter()
    tbl_sizes: List[int] = []
    nl_lens: List[int] = []
    per_case: List[Tuple[Any, int, int, int, int]] = []
    nominal = 0

    for c in cases:
        schema = c.get("ground_truth_schema") or {}
        tables = schema.get("tables") or []
        tbl_sizes.append(len(tables))
        nl_lens.append(len(c.get("nl_description") or ""))
        profiles[c.get("profile", "?")] += 1
        domains[c.get("domain", "?")] += 1
        dists = c.get("ground_truth_distributions") or {}
        for spec in dists.values():
            fam = (spec or {}).get("distribution", (spec or {}).get("family"))
            families[fam] += 1
            if fam == "categorical" and _is_nominal(
                (spec.get("params") or {}).get("weights")
            ):
                nominal += 1
        cons = c.get("ground_truth_constraints") or []
        for con in cons:
            ctypes[(con or {}).get("type")] += 1
        per_case.append(
            (
                c.get("id"),
                len(tables),
                len(schema.get("relationships") or []),
                len(dists),
                len(cons),
            )
        )

    def hist(counter: Counter, title: str) -> List[str]:
        out = [f"  {title}"]
        for k, v in counter.most_common():
            out.append(f"    {str(k):28} {v}")
        return out

    lines = ["", "=" * 70, "COVERAGE", "=" * 70, f"  cases: {len(cases)}"]
    if tbl_sizes:
        lines.append(
            f"  tables per case: min={min(tbl_sizes)} max={max(tbl_sizes)} "
            f"mean={sum(tbl_sizes) / len(tbl_sizes):.1f}"
        )
    if nl_lens:
        lines.append(
            f"  nl_description chars: min={min(nl_lens)} max={max(nl_lens)} "
            f"mean={sum(nl_lens) / len(nl_lens):.0f}"
        )
    lines += hist(families, "distribution families:")
    if nominal:
        lines.append(
            f"    (of which NOMINAL categorical, not data-scorable: {nominal})"
        )
    lines += hist(ctypes, "constraint types:")
    lines += hist(profiles, "profiles:")
    lines.append(f"  distinct domains: {len(domains)}")
    dupes = {d: n for d, n in domains.items() if n > 1}
    if dupes:
        lines.append(f"    repeated domains: {dupes}")
    return "\n".join(lines)


def cross_file_integrity(paths: List[Path]) -> List[str]:
    """Catch contamination BETWEEN batch files, which per-file validation cannot.

    Authoring is split across concurrent agents that share a scratchpad, and one
    run genuinely did overwrite another's staging file and briefly assemble the
    wrong batch's cases into batch_04.jsonl. It was caught and restored, but
    only because someone looked -- nothing in the gate would have said so, since
    each file was independently well-formed the whole time. That is the failure
    this checks for: valid files holding the wrong contents.

    Three signatures, cheapest first:
      * an id outside the range its filename implies (batch_04 owns 31-40)
      * the same id in more than one file
      * two cases sharing an nl_description -- the direct signature of a copy,
        since 150 independently authored specifications cannot collide
    """
    errors: List[str] = []
    by_id: Dict[int, List[str]] = {}
    by_nl: Dict[str, List[str]] = {}

    for p in sorted(paths):
        stem = p.stem
        expected: Optional[range] = None
        if stem.startswith("batch_"):
            try:
                n = int(stem.split("_")[1])
                expected = range((n - 1) * 10 + 1, n * 10 + 1)
            except IndexError, ValueError:
                expected = None

        cases, load_errs = load_cases(p)
        errors.extend(f"{p.name}: {e}" for e in load_errs)

        for c in cases:
            cid = c.get("id")
            if isinstance(cid, int):
                by_id.setdefault(cid, []).append(p.name)
                if expected is not None and cid not in expected:
                    errors.append(
                        f"{p.name}: case id {cid} is outside this file's range "
                        f"{expected.start}-{expected.stop - 1}. A file holding "
                        "another batch's cases is the signature of a clobbered "
                        "staging file, not a numbering slip."
                    )
            nl = (c.get("nl_description") or "").strip()
            if nl:
                digest = hashlib.sha256(nl.encode("utf-8")).hexdigest()
                by_nl.setdefault(digest, []).append(f"{p.name}#{cid}")

    for cid, where in sorted(by_id.items()):
        if len(where) > 1:
            errors.append(f"case id {cid} appears in {len(where)} files: {where}")
    for where in by_nl.values():
        if len(where) > 1:
            errors.append(
                "identical nl_description shared by "
                f"{len(where)} cases: {sorted(where)}. Independently authored "
                "specifications do not collide; this is a copy."
            )
    return errors


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--coverage", action="store_true", help="print the coverage report")
    ap.add_argument("--quiet", action="store_true", help="suppress warnings")
    ap.add_argument(
        "--cross-check",
        nargs="+",
        type=Path,
        metavar="FILE",
        help=(
            "check these files against EACH OTHER for contamination (ids out of "
            "range, duplicated ids, copied nl_description) instead of validating "
            "one file's contents. Per-file validation cannot see this class of "
            "fault, because every file stays independently well-formed."
        ),
    )
    args = ap.parse_args(argv)

    if args.cross_check:
        errs = cross_file_integrity(list(args.cross_check))
        print(f"cross-file integrity over {len(args.cross_check)} file(s)")
        for e in errs:
            print(f"  - {e}")
        if errs:
            print(f"\nFAILED: {len(errs)} integrity error(s)")
            return 1
        print("\nOK: no cross-file contamination detected")
        return 0

    cases, load_errors = load_cases(args.cases)
    f = validate(cases)
    f.errors = load_errors + f.errors

    print(f"{args.cases}: {len(cases)} case(s) loaded")
    if f.errors:
        print(f"\nERRORS ({len(f.errors)}):")
        for e in f.errors:
            print(f"  - {e}")
    if f.warnings and not args.quiet:
        print(f"\nWARNINGS ({len(f.warnings)}):")
        for w in f.warnings:
            print(f"  - {w}")
    if args.coverage:
        print(coverage(cases))

    if f.errors:
        print(f"\nFAILED: {len(f.errors)} error(s)")
        return 1
    print(f"\nOK: no errors ({len(f.warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
