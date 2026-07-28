"""Bidirectional SQL<->object conversion (Section 3.2), sqlglot-backed.

**Homogenization rule**: every level is a complete, valid SELECT statement
-- a base table is always rendered/parsed as `SELECT * FROM X`, never a
bare table name. `from_sql`'s top-level parse requires an `exp.Select`;
anything else (a bare table/column reference, DDL, etc.) is rejected.

**Scope, matching Section 1's closed algebra** -- out-of-scope SQL
features fail parsing with a specific reason, never silently produce a
partial object:
- `UNION`/`INTERSECT`/`EXCEPT` -- rejected (`exp.Union`/`Intersect`/`Except`
  aren't `exp.Select` at all, caught immediately).
- Window functions (`exp.Window`) -- rejected wherever they appear.
- `ORDER BY`/`LIMIT`/`DISTINCT` on the SELECT itself -- rejected; no
  Relation node models sorting or row-limiting.
- Composite join conditions (`AND`-ed `ON` clauses) -- rejected; matches
  nodes.py's own single-`JoinCondition` restriction.
- `RIGHT`/`FULL OUTER`/`CROSS JOIN` -- rejected; only plain/`INNER`/`LEFT`
  are accepted (`Join`'s own semantics are always effectively LEFT-JOIN-
  shaped for a nullable FK, per Section 4.4 -- derived from schema
  nullability, not dictated by the SQL author's join keyword).
- More than one aggregate function call in a single SELECT's expression
  list -- rejected; `Aggregate` (Section 3.1) models exactly one aggregate
  quantity per node, so a multi-aggregate SELECT has no single-node
  representation here (would need splitting into multiple Constraints
  upstream, same atomic-fact philosophy used everywhere else).
- Correlated subqueries -- NOT proactively detected (would need scope-
  chain analysis this module doesn't do); a correlated subquery that
  parses without error may still produce a nonsensical object, caught (if
  at all) by relation/validate.py's column-resolution checks downstream,
  not by this parser.
- `Fanout` -- **NOT parseable from SQL at all** (explicit, documented
  non-goal for this pass). Recognizing the specific
  `LEFT JOIN + COUNT(*) GROUP BY parent.pk` shape as a `Fanout` rather
  than an ordinary `Aggregate(source=Join(...))` needs a real, careful
  heuristic (checking the GROUP BY matches the parent's PK exactly, that
  no other columns are selected, etc.) genuinely out of scope for this
  pass -- such SQL parses fine, just as an `Aggregate`, not a `Fanout`.
  Serializing a `Fanout` TO SQL (the other direction) is fully supported.
- `RIfThen` serializes to `NOT (antecedent) OR (consequent)` (the standard
  logical-implication rewrite -- SQL has no native "IF P THEN Q" boolean
  predicate). The reverse direction is NOT attempted: parsing never
  recognizes an `OR` shape as a disguised `RIfThen` (that would be
  ambiguous with a genuine author-written `OR`), so an `RIfThen` doesn't
  round-trip -- it serializes one way only.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple, cast

import sqlglot
from sqlglot import exp

from src.util.constraint_model.condition.expressions import (
    RAggregateRef,
    RArithmetic,
    RColumnRef,
    RExprUnion,
    RLiteral,
)
from src.util.constraint_model.condition.predicates import (
    RAnd,
    RBetween,
    RComparison,
    RIfThen,
    RInSet,
    RNot,
    RNotInSet,
    ROr,
    RPredicateUnion,
)
from src.util.constraint_model.relation.nodes import (
    Aggregate,
    AggregateFn,
    BaseTable,
    Filter,
    Join,
    JoinCondition,
    Project,
    ProjectEntry,
    RelationUnion,
)

# sqlglot's own type stubs mark exp.Expression as a private export, which
# pyright flags on every reference -- alias it once here instead of
# suppressing the warning at each of the many usages below.
SqlExpr = exp.Expression  # pyright: ignore[reportPrivateImportUsage]

DEFAULT_DIALECT = "sqlite"

_ComparisonOp = Literal["<", "<=", "=", "!=", ">=", ">"]
_ArithmeticOp = Literal["+", "-", "*", "/"]

_SQL_TO_AGGREGATE_FN: Dict[type, AggregateFn] = {
    exp.Sum: "SUM",
    exp.Avg: "AVG",
    exp.Max: "MAX",
    exp.Min: "MIN",
    exp.Count: "COUNT",
    exp.Stddev: "STDDEV",
    exp.Variance: "VARIANCE",
}

_COMPARISON_OPS: Dict[type, _ComparisonOp] = {
    exp.EQ: "=",
    exp.NEQ: "!=",
    exp.GT: ">",
    exp.GTE: ">=",
    exp.LT: "<",
    exp.LTE: "<=",
}

_ARITHMETIC_OPS: Dict[type, _ArithmeticOp] = {
    exp.Add: "+",
    exp.Sub: "-",
    exp.Mul: "*",
    exp.Div: "/",
}


# ---------------------------------------------------------------------------
# Expression / predicate -> SQL text (serialization)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# sqlglot AST -> expression / predicate (parsing)
# ---------------------------------------------------------------------------


def _sql_expr_to_rexpr(
    node: SqlExpr,
) -> Tuple[Optional["RExprUnion"], List[str]]:
    if isinstance(node, exp.Column):
        return RColumnRef(name=node.name), []
    if isinstance(node, exp.Boolean):
        return RLiteral(value=bool(node.this)), []
    if isinstance(node, exp.Literal):
        if node.is_string:
            return RLiteral(value=node.this), []
        text = node.this
        try:
            value: float | int = int(text)
        except ValueError:
            value = float(text)
        return RLiteral(value=value), []
    if isinstance(node, exp.Paren):
        return _sql_expr_to_rexpr(node.this)
    if isinstance(node, (exp.Add, exp.Sub, exp.Mul, exp.Div)):
        op = _ARITHMETIC_OPS[type(node)]
        left, left_errs = _sql_expr_to_rexpr(node.this)
        right, right_errs = _sql_expr_to_rexpr(node.expression)
        errors = left_errs + right_errs
        if left is None or right is None:
            return None, errors
        return RArithmetic(op=op, left=left, right=right), errors
    if isinstance(node, exp.Window):
        return None, ["Window functions are out of scope (Section 1's non-goal)."]
    return None, [
        f"Unsupported expression shape: {type(node).__name__} ({node.sql()})."
    ]


def _sql_condition_to_rpredicate(
    node: SqlExpr,
) -> Tuple[Optional["RPredicateUnion"], List[str]]:
    if isinstance(node, exp.Paren):
        return _sql_condition_to_rpredicate(node.this)
    if isinstance(node, exp.And):
        flat = _flatten_binary(node, exp.And)
        operands: List["RPredicateUnion"] = []
        errors: List[str] = []
        for part in flat:
            parsed, part_errs = _sql_condition_to_rpredicate(part)
            errors.extend(part_errs)
            if parsed is not None:
                operands.append(parsed)
        if errors or len(operands) < 2:
            return None, errors or ["RAnd requires at least 2 resolvable operands."]
        return RAnd(operands=operands), []
    if isinstance(node, exp.Or):
        flat = _flatten_binary(node, exp.Or)
        operands = []
        errors = []
        for part in flat:
            parsed, part_errs = _sql_condition_to_rpredicate(part)
            errors.extend(part_errs)
            if parsed is not None:
                operands.append(parsed)
        if errors or len(operands) < 2:
            return None, errors or ["ROr requires at least 2 resolvable operands."]
        return ROr(operands=operands), []
    if isinstance(node, exp.Not):
        if isinstance(node.this, exp.In):
            return _in_to_rpredicate(node.this, negated=True)
        inner, errors = _sql_condition_to_rpredicate(node.this)
        if inner is None:
            return None, errors
        return RNot(operand=inner), []
    if isinstance(node, exp.Between):
        expr_node, e1 = _sql_expr_to_rexpr(node.this)
        low, e2 = _sql_expr_to_rexpr(node.args["low"])
        high, e3 = _sql_expr_to_rexpr(node.args["high"])
        errors = e1 + e2 + e3
        if expr_node is None or low is None or high is None:
            return None, errors
        return RBetween(expr=expr_node, low=low, high=high), []
    if isinstance(node, exp.In):
        return _in_to_rpredicate(node, negated=False)
    if type(node) in _COMPARISON_OPS:
        left, e1 = _sql_expr_to_rexpr(node.this)
        right, e2 = _sql_expr_to_rexpr(node.expression)
        errors = e1 + e2
        if left is None or right is None:
            return None, errors
        return RComparison(op=_COMPARISON_OPS[type(node)], left=left, right=right), []
    return None, [f"Unsupported predicate shape: {type(node).__name__} ({node.sql()})."]


def _in_to_rpredicate(
    node: exp.In, *, negated: bool
) -> Tuple[Optional["RPredicateUnion"], List[str]]:
    expr_node, errors = _sql_expr_to_rexpr(node.this)
    if expr_node is None:
        return None, errors
    values: List[str | int | float] = []
    for item in node.expressions:
        if not isinstance(item, exp.Literal):
            return None, ["IN/NOT IN values must all be literals."]
        values.append(item.this if item.is_string else _numeric_literal(item.this))
    cls = RNotInSet if negated else RInSet
    return cls(expr=expr_node, values=values), []


def _numeric_literal(text: str) -> float | int:
    try:
        return int(text)
    except ValueError:
        return float(text)


def _flatten_binary(node: SqlExpr, kind: type) -> List[SqlExpr]:
    if isinstance(node, kind):
        return _flatten_binary(node.this, kind) + _flatten_binary(node.expression, kind)
    return [node]


def _substitute_aggregate_ref(node: "RPredicateUnion", alias: str) -> "RPredicateUnion":
    """Recursively replaces any RColumnRef(name=alias) with
    RAggregateRef(alias=alias) inside an already-parsed predicate tree --
    see the WHERE-over-Aggregate-subquery handling in _select_to_relation
    for why this substitution is needed post-parse rather than during."""
    if isinstance(node, RComparison):
        return node.model_copy(
            update={
                "left": _substitute_aggregate_ref_expr(node.left, alias),
                "right": _substitute_aggregate_ref_expr(node.right, alias),
            }
        )
    if isinstance(node, (RAnd, ROr)):
        return node.model_copy(
            update={
                "operands": [_substitute_aggregate_ref(o, alias) for o in node.operands]
            }
        )
    if isinstance(node, RNot):
        return node.model_copy(
            update={"operand": _substitute_aggregate_ref(node.operand, alias)}
        )
    if isinstance(node, RBetween):
        return node.model_copy(
            update={
                "expr": _substitute_aggregate_ref_expr(node.expr, alias),
                "low": _substitute_aggregate_ref_expr(node.low, alias),
                "high": _substitute_aggregate_ref_expr(node.high, alias),
            }
        )
    if isinstance(node, (RInSet, RNotInSet)):
        return node.model_copy(
            update={"expr": _substitute_aggregate_ref_expr(node.expr, alias)}
        )
    if isinstance(node, RIfThen):
        return node.model_copy(
            update={
                "antecedent": _substitute_aggregate_ref(node.antecedent, alias),
                "consequent": _substitute_aggregate_ref(node.consequent, alias),
            }
        )
    return node


def _substitute_aggregate_ref_expr(node: "RExprUnion", alias: str) -> "RExprUnion":
    if isinstance(node, RColumnRef) and node.name == alias:
        return RAggregateRef(alias=alias)
    if isinstance(node, RArithmetic):
        return node.model_copy(
            update={
                "left": _substitute_aggregate_ref_expr(node.left, alias),
                "right": _substitute_aggregate_ref_expr(node.right, alias),
            }
        )
    return node


# ---------------------------------------------------------------------------
# Relation -> SQL (serialization)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SQL -> Relation (parsing)
# ---------------------------------------------------------------------------


def from_sql(
    sql: str, schema_pk_lookup: Optional[dict] = None, dialect: str = DEFAULT_DIALECT
) -> Tuple[Optional["RelationUnion"], List[str]]:
    """Public entry point: parses a complete SQL SELECT statement into a
    RelationUnion. Follows this package's non-raising convention -- a None
    result paired with a non-empty error list means the SQL couldn't be
    reduced to this closed algebra (Section 1), with a specific reason,
    never a silently partial object.

    `schema_pk_lookup` is unused by this pass (reserved -- Fanout
    recognition, were it ever added, would need it to confirm a GROUP BY
    matches a table's real PK)."""
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception as e:  # sqlglot raises its own ParseError subclasses
        return None, [f"SQL parse error: {e}"]
    return _select_to_relation(cast(SqlExpr, tree))


def _select_to_relation(
    tree: SqlExpr,
) -> Tuple[Optional["RelationUnion"], List[str]]:
    if not isinstance(tree, exp.Select):
        return None, [
            f"Only a single SELECT statement is supported (got {type(tree).__name__}) -- "
            "UNION/INTERSECT/EXCEPT and non-SELECT statements are out of scope."
        ]
    if tree.args.get("distinct"):
        return None, ["DISTINCT is not modeled by any Relation node -- out of scope."]
    if tree.args.get("order"):
        return None, ["ORDER BY is not modeled by any Relation node -- out of scope."]
    if tree.args.get("limit"):
        return None, ["LIMIT is not modeled by any Relation node -- out of scope."]

    from_clause = tree.args.get("from_") or tree.args.get("from")
    if from_clause is None:
        return None, ["SELECT with no FROM clause is not a Relation."]

    source, errors = _from_expr_to_relation(from_clause.this)
    if source is None:
        return None, errors

    for join in tree.args.get("joins") or []:
        source, join_errors = _apply_join(source, join)
        errors.extend(join_errors)
        if source is None:
            return None, errors

    where = tree.args.get("where")
    if where is not None:
        condition, cond_errors = _sql_condition_to_rpredicate(where.this)
        errors.extend(cond_errors)
        if condition is None:
            return None, errors
        if isinstance(source, Aggregate):
            # This module always serializes Filter(source=Aggregate(...))
            # as a subquery + WHERE referencing the aggregate's own OUTPUT
            # COLUMN by name (e.g. "WHERE total_paid > 1000"), never a
            # native re-invoked HAVING SUM(amount) > 1000 -- so the parser
            # above sees a bare Column, not an aggregate call, and builds
            # an ordinary RColumnRef. Substitute it back to RAggregateRef
            # wherever it names this Aggregate's own alias, or this would
            # be a real round-trip precision gap, not cosmetic.
            condition = _substitute_aggregate_ref(condition, source.alias)
        source = Filter(source=source, condition=condition)

    group = tree.args.get("group")
    select_exprs = tree.expressions
    has_aggregate_call = any(
        _is_aggregate_call(e.this if isinstance(e, exp.Alias) else e)
        for e in select_exprs
    )
    if group is not None or has_aggregate_call:
        # A whole-table aggregate (e.g. "SELECT COUNT(DISTINCT x) AS n FROM
        # T") has no GROUP BY clause at all but is still an Aggregate node
        # with group_by=None -- checked via has_aggregate_call, not just
        # group's presence, or this shape would be mis-parsed as a Project
        # trying (and failing) to treat the aggregate call as a plain column.
        source, errors2 = _build_aggregate(source, select_exprs, group)
        errors.extend(errors2)
        if source is None:
            return None, errors
        having = tree.args.get("having")
        if having is not None:
            assert isinstance(source, Aggregate)
            having_cond, having_errors = _having_to_rpredicate(having.this, source)
            errors.extend(having_errors)
            if having_cond is None:
                return None, errors
            source = Filter(source=source, condition=having_cond)
        return source, errors

    # No GROUP BY -- an explicit column list becomes a Project; a bare "*"
    # (or "*" plus nothing else) passes through unchanged.
    if len(select_exprs) == 1 and isinstance(select_exprs[0], exp.Star):
        return source, errors

    columns: List[ProjectEntry] = []
    for e in select_exprs:
        if isinstance(e, exp.Window):
            return None, ["Window functions are out of scope (Section 1's non-goal)."]
        alias = None
        target = e
        if isinstance(e, exp.Alias):
            alias = e.alias
            target = e.this
        expr_obj, expr_errors = _sql_expr_to_rexpr(target)
        errors.extend(expr_errors)
        if expr_obj is None:
            return None, errors
        columns.append(ProjectEntry(expr=expr_obj, alias=alias))
    return Project(source=source, columns=columns), errors


def _from_expr_to_relation(
    node: SqlExpr,
) -> Tuple[Optional["RelationUnion"], List[str]]:
    if isinstance(node, exp.Table):
        alias = node.alias or None
        return BaseTable(name=node.name, alias=alias), []
    if isinstance(node, exp.Subquery):
        inner, errors = _select_to_relation(node.this)
        if inner is None:
            return None, errors
        alias = node.alias or None
        if alias and getattr(inner, "alias", None) is None:
            inner = inner.model_copy(update={"alias": alias})
        return inner, errors
    return None, [
        f"Unsupported FROM-clause shape: {type(node).__name__} ({node.sql()})."
    ]


def _apply_join(
    source: "RelationUnion", join: exp.Join
) -> Tuple[Optional["RelationUnion"], List[str]]:
    side = (join.side or "").upper()
    if side not in ("", "INNER", "LEFT"):
        return None, [
            f"Join type '{side}' is out of scope -- only plain/INNER/LEFT are supported."
        ]
    on = join.args.get("on")
    if on is None:
        return None, ["A JOIN without an ON condition is out of scope."]
    if isinstance(on, exp.And):
        return None, [
            "Composite join conditions (AND-ed ON clauses) are not supported -- "
            "composite FKs are an explicit non-goal."
        ]
    if type(on) is not exp.EQ:
        return None, ["Only equi-join ON conditions (a = b) are supported."]

    left_ref = _column_ref_text(on.this)
    right_ref = _column_ref_text(on.expression)
    if left_ref is None or right_ref is None:
        return None, ["JOIN ON condition must be TABLE.column = TABLE.column."]

    right_relation, errors = _from_expr_to_relation(join.this)
    if right_relation is None:
        return None, errors

    return (
        Join(
            left=source,
            right=right_relation,
            on=[JoinCondition(left=left_ref, right=right_ref)],
        ),
        errors,
    )


def _column_ref_text(node: SqlExpr) -> Optional[str]:
    if not isinstance(node, exp.Column) or node.table is None:
        return None
    return f"{node.table}.{node.name}"


def _build_aggregate(
    source: "RelationUnion", select_exprs: List[SqlExpr], group: Optional[exp.Group]
) -> Tuple[Optional[Aggregate], List[str]]:
    group_by = (
        [e.name for e in group.expressions if isinstance(e, exp.Column)]
        if group is not None
        else []
    )

    agg_calls = []
    for e in select_exprs:
        target = e.this if isinstance(e, exp.Alias) else e
        if _is_aggregate_call(target):
            agg_calls.append(e)
    if len(agg_calls) == 0:
        return None, [
            "GROUP BY present but no recognizable single aggregate function call found."
        ]
    if len(agg_calls) > 1:
        return None, [
            "More than one aggregate function call in a single SELECT is out of scope -- "
            "Aggregate models exactly one aggregate quantity per node."
        ]

    entry = agg_calls[0]
    alias = entry.alias if isinstance(entry, exp.Alias) else None
    if not alias:
        return None, ["An aggregate SELECT expression must be aliased."]
    call = entry.this if isinstance(entry, exp.Alias) else entry

    fn, column, fn_param, errors = _aggregate_call_to_fn(call)
    if fn is None:
        return None, errors
    return (
        Aggregate(
            source=source,
            fn=fn,
            column=column,
            group_by=group_by or None,
            alias=alias,
            fn_param=fn_param,
        ),
        errors,
    )


def _is_aggregate_call(node: SqlExpr) -> bool:
    return type(node) in _SQL_TO_AGGREGATE_FN or isinstance(
        node, (exp.PercentileCont, exp.Mode)
    )


def _aggregate_call_to_fn(
    node: SqlExpr,
) -> Tuple[Optional[AggregateFn], str, Optional[float], List[str]]:
    if isinstance(node, exp.Count):
        inner = node.this
        if isinstance(inner, exp.Distinct):
            col = inner.expressions[0]
            return "COUNT_DISTINCT", _star_or_column_name(col), None, []
        return "COUNT", _star_or_column_name(inner), None, []
    if isinstance(node, exp.Mode):
        return "MODE", _star_or_column_name(node.this), None, []
    if isinstance(node, exp.PercentileCont):
        p = float(node.this.this) if isinstance(node.this, exp.Literal) else 0.5
        order_col = node.args.get("order")
        col_name = _percentile_order_column(order_col)
        if p == 0.5:
            return "MEDIAN", col_name, None, []
        return "PERCENTILE", col_name, p * 100.0, []
    fn = _SQL_TO_AGGREGATE_FN.get(type(node))
    if fn is None:
        return (
            None,
            "",
            None,
            [f"Unrecognized aggregate function: {type(node).__name__}."],
        )
    return fn, _star_or_column_name(node.this), None, []


def _star_or_column_name(node: SqlExpr) -> str:
    if isinstance(node, exp.Star):
        return "*"
    if isinstance(node, exp.Column):
        return node.name
    return node.sql()


def _percentile_order_column(order: Optional[SqlExpr]) -> str:
    if order is None:
        return "*"
    ordered = order.expressions[0] if order.expressions else None
    if isinstance(ordered, exp.Ordered) and isinstance(ordered.this, exp.Column):
        return ordered.this.name
    return "*"


def _having_to_rpredicate(
    node: SqlExpr, aggregate: Aggregate
) -> Tuple[Optional["RPredicateUnion"], List[str]]:
    """Parses a HAVING clause, substituting a reference to the SAME
    aggregate function the query already computes with an
    RAggregateRef(alias=...) rather than treating it as a second,
    unsupported aggregate call."""
    if isinstance(node, exp.Paren):
        return _having_to_rpredicate(node.this, aggregate)
    if isinstance(node, (exp.And, exp.Or)):
        kind = exp.And if isinstance(node, exp.And) else exp.Or
        flat = _flatten_binary(node, kind)
        operands: List["RPredicateUnion"] = []
        errors: List[str] = []
        for part in flat:
            parsed, part_errors = _having_to_rpredicate(part, aggregate)
            errors.extend(part_errors)
            if parsed is not None:
                operands.append(parsed)
        if errors or len(operands) < 2:
            return None, errors or [
                "HAVING AND/OR requires at least 2 resolvable operands."
            ]
        cls = RAnd if kind is exp.And else ROr
        return cls(operands=operands), []
    if type(node) in _COMPARISON_OPS:
        left = _having_operand_to_rexpr(node.this, aggregate)
        right = _having_operand_to_rexpr(node.expression, aggregate)
        if left is None or right is None:
            return None, [
                "HAVING operand does not resolve to the query's own aggregate or a literal."
            ]
        return RComparison(op=_COMPARISON_OPS[type(node)], left=left, right=right), []
    return None, [f"Unsupported HAVING shape: {type(node).__name__} ({node.sql()})."]


def _having_operand_to_rexpr(
    node: SqlExpr, aggregate: Aggregate
) -> Optional["RExprUnion"]:
    if _is_aggregate_call(node):
        fn, column, _fn_param, errs = _aggregate_call_to_fn(node)
        if not errs and fn == aggregate.fn and column == aggregate.column:
            return RAggregateRef(alias=aggregate.alias)
        return None  # a different aggregate than the query's own -- out of scope
    expr_obj, errs = _sql_expr_to_rexpr(node)
    return None if errs else expr_obj
