"""Same-population moment-fact conflict checks: plain aggregate-based
moment facts (Section 8.3 -- "the average is $150" decomposes to an
ordinary Aggregate + RComparison, never a dedicated node) cross-checked
against EACH OTHER and against a same-population Distributed fact's own
implied mean/variance (Section 8.1's per-family formulas, LOG_NORMAL
handled with care since its stated parameters describe the underlying
normal, not its own actual mean/variance).

Scope, deliberately: only AVG/VARIANCE/STDDEV aggregate facts participate
-- SUM/COUNT/COUNT_DISTINCT scale with row count (not a fixed "moment" of
the column's distribution the way mean/variance are) and MAX/MIN aren't
determined by mean/variance either, so cross-checking them against a
Distributed fact's implied moments wouldn't be meaningful. A `!=` moment
fact is also not cross-checked (a single excluded point rarely creates a
provable interval-emptiness against other facts, and handling it properly
needs a separate excluded-point mechanism this pass skips as a documented,
lower-priority simplification).

Implemented as an interval-intersection merge: every moment/Distributed-
implied observation about the same (MEAN or VARIANCE, column) at the same
population becomes an interval (an `=` fact is a single point; `>`, `>=`,
`<`, `<=` are half-open/open rays). If the merged intersection is empty,
that's a provable conflict -- this generalizes plain value-equality
mismatches to logical contradictions between inequalities too (e.g.
"AVG(x) > 100" and "AVG(x) < 50" together are already infeasible with no
Distributed fact involved at all).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.util.schema_model.schema import Schema
from src.util.constraint_model.condition.cohesive import Distributed
from src.util.constraint_model.condition.expressions import RAggregateRef, RLiteral
from src.util.constraint_model.condition.predicates import RComparison
from src.util.constraint_model.conflicts.models import Conflict
from src.util.constraint_model.constraint import Constraint, is_softenable
from src.util.constraint_model.population import compute_population
from src.util.constraint_model.relation.nodes import Aggregate, Join, RelationUnion


_INTERVAL_REL_TOL = 1e-4
_INTERVAL_ABS_TOL = 1e-6


@dataclass(frozen=True)
class _Interval:
    lo: float
    lo_closed: bool
    hi: float
    hi_closed: bool

    def is_empty(self) -> bool:
        # A small numeric tolerance, scaled to magnitude, absorbs rounding
        # noise between a stated (possibly rounded) fact and a fully-
        # precise implied value (e.g. LOG_NORMAL's exp(mu + sigma^2/2)) --
        # without it, "AVG(x) = 61.8678" (rounded to 4dp) would falsely
        # conflict with an implied mean of 61.867876488... down to the
        # last float digit.
        tol = max(
            _INTERVAL_ABS_TOL, _INTERVAL_REL_TOL * max(abs(self.lo), abs(self.hi), 1.0)
        )
        if self.lo > self.hi + tol:
            return True
        if abs(self.lo - self.hi) <= tol and not (self.lo_closed and self.hi_closed):
            return True
        return False

    def intersect(self, other: "_Interval") -> "_Interval":
        if self.lo > other.lo:
            lo, lo_closed = self.lo, self.lo_closed
        elif other.lo > self.lo:
            lo, lo_closed = other.lo, other.lo_closed
        else:
            lo, lo_closed = self.lo, self.lo_closed and other.lo_closed
        if self.hi < other.hi:
            hi, hi_closed = self.hi, self.hi_closed
        elif other.hi < self.hi:
            hi, hi_closed = other.hi, other.hi_closed
        else:
            hi, hi_closed = self.hi, self.hi_closed and other.hi_closed
        return _Interval(lo, lo_closed, hi, hi_closed)


_FULL = _Interval(-math.inf, False, math.inf, False)


def _interval_from_comparison(op: str, value: float) -> Optional[_Interval]:
    if op == "=":
        return _Interval(value, True, value, True)
    if op == ">":
        return _Interval(value, False, math.inf, False)
    if op == ">=":
        return _Interval(value, True, math.inf, False)
    if op == "<":
        return _Interval(-math.inf, False, value, False)
    if op == "<=":
        return _Interval(-math.inf, False, value, True)
    return None  # '!=' -- deliberately not handled, see module docstring


def _square_interval(interval: Optional[_Interval]) -> Optional[_Interval]:
    """Converts a std-dev interval into variance-space -- monotonic since
    std-dev's real domain is [0, inf); defensively clamps a sub-zero lower
    bound to 0 rather than assuming a well-formed input."""
    if interval is None:
        return None
    lo = max(interval.lo, 0.0)
    lo_closed = interval.lo_closed if interval.lo > 0 else True
    if interval.hi < 0:
        return _Interval(
            1.0, True, 0.0, True
        )  # collapses to empty; shouldn't occur for a real STDDEV fact
    hi = math.inf if interval.hi == math.inf else interval.hi * interval.hi
    return _Interval(lo * lo, lo_closed, hi, interval.hi_closed)


_FLIP_OP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "=": "=", "!=": "!="}


def _as_float(value: object) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) else None


def _find_aggregate(relation: "RelationUnion", alias: str) -> Optional[Aggregate]:
    if isinstance(relation, Aggregate):
        if relation.alias == alias:
            return relation
        return _find_aggregate(relation.source, alias)
    if isinstance(relation, Join):
        return _find_aggregate(relation.left, alias) or _find_aggregate(
            relation.right, alias
        )
    source = getattr(relation, "source", None)
    if source is not None:
        return _find_aggregate(source, alias)
    return None


def _split_aggregate_comparison(
    cond: RComparison,
) -> Tuple[Optional[RAggregateRef], Optional[str], Optional[RLiteral]]:
    """Returns (ref, effective_op, literal), flipping the operator if the
    aggregate ref is the right-hand operand (e.g. "100 < AVG(x)" means
    "AVG(x) > 100")."""
    if isinstance(cond.left, RAggregateRef) and isinstance(cond.right, RLiteral):
        return cond.left, cond.op, cond.right
    if isinstance(cond.right, RAggregateRef) and isinstance(cond.left, RLiteral):
        return cond.right, _FLIP_OP[cond.op], cond.left
    return None, None, None


def _moment_observation(constraint: Constraint) -> Optional[Tuple[str, str, _Interval]]:
    cond = constraint.condition
    if not isinstance(cond, RComparison):
        return None
    ref, op, literal = _split_aggregate_comparison(cond)
    if ref is None or op is None or literal is None:
        return None
    if not isinstance(literal.value, (int, float)) or isinstance(literal.value, bool):
        return None
    agg = _find_aggregate(constraint.relation, ref.alias)
    if agg is None or agg.fn not in ("AVG", "VARIANCE", "STDDEV"):
        return None
    value = float(literal.value)
    if agg.fn == "AVG":
        interval = _interval_from_comparison(op, value)
        kind = "MEAN"
    elif agg.fn == "VARIANCE":
        interval = _interval_from_comparison(op, value)
        kind = "VARIANCE"
    else:  # STDDEV
        interval = _square_interval(_interval_from_comparison(op, value))
        kind = "VARIANCE"
    if interval is None:
        return None
    return kind, agg.column, interval


def _implied_mean(cond: Distributed) -> Optional[float]:
    p = cond.parameters
    if cond.family == "GAUSSIAN":
        return _as_float(p.get("mean"))
    if cond.family == "POISSON":
        return _as_float(p.get("lam"))
    if cond.family == "UNIFORM":
        lo, hi = _as_float(p.get("min_value")), _as_float(p.get("max_value"))
        return (lo + hi) / 2 if lo is not None and hi is not None else None
    if cond.family == "BETA":
        a, b = _as_float(p.get("alpha")), _as_float(p.get("beta"))
        return a / (a + b) if a is not None and b is not None else None
    if cond.family == "LOG_NORMAL":
        mu, sigma = _as_float(p.get("mean")), _as_float(p.get("std_dev"))
        return (
            math.exp(mu + sigma**2 / 2)
            if mu is not None and sigma is not None
            else None
        )
    return None  # CATEGORICAL -- no scalar mean, per Section 8.1


def _implied_variance(cond: Distributed) -> Optional[float]:
    p = cond.parameters
    if cond.family == "GAUSSIAN":
        sd = _as_float(p.get("std_dev"))
        return sd * sd if sd is not None else None
    if cond.family == "POISSON":
        return _as_float(p.get("lam"))
    if cond.family == "UNIFORM":
        lo, hi = _as_float(p.get("min_value")), _as_float(p.get("max_value"))
        return (hi - lo) ** 2 / 12 if lo is not None and hi is not None else None
    if cond.family == "BETA":
        a, b = _as_float(p.get("alpha")), _as_float(p.get("beta"))
        if a is None or b is None:
            return None
        return (a * b) / ((a + b) ** 2 * (a + b + 1))
    if cond.family == "LOG_NORMAL":
        mu, sigma = _as_float(p.get("mean")), _as_float(p.get("std_dev"))
        if mu is None or sigma is None:
            return None
        return (math.exp(sigma * sigma) - 1) * math.exp(2 * mu + sigma * sigma)
    return None


def _distributed_observations(cond: Distributed) -> List[Tuple[str, str, _Interval]]:
    obs: List[Tuple[str, str, _Interval]] = []
    mean = _implied_mean(cond)
    if mean is not None:
        obs.append(("MEAN", cond.column, _Interval(mean, True, mean, True)))
    variance = _implied_variance(cond)
    if variance is not None:
        obs.append(("VARIANCE", cond.column, _Interval(variance, True, variance, True)))
    return obs


_PopKey = Tuple[str, frozenset, frozenset, bool, Tuple[str, ...]]
_GroupKey = Tuple[_PopKey, str, str]


def _moment_population_key(
    relation: "RelationUnion", schema: Schema
) -> Optional[_PopKey]:
    """The population-comparability key for cross-checking moment facts.
    Deliberately ignores an Aggregate's own alias -- a moment fact's real
    identity is its underlying quantity (source population + group_by),
    not whatever label an extractor happened to give it; two independently
    -extracted facts about "the average order total" could easily get
    different auto-generated aliases from an LLM. An ungrouped Aggregate
    (group_by=None/empty) and a plain BaseTable/Filter/Join relation (as a
    Distributed fact is rooted at) intentionally produce the SAME key when
    their underlying population matches -- both describe one statistic
    over the whole population. A non-empty group_by is part of the key,
    so per-group moments only compare against other facts sharing the
    identical group_by (a different, more granular kind of statistic)."""
    if isinstance(relation, Aggregate):
        base_relation = relation.source
        group_by = tuple(sorted(relation.group_by or ()))
    else:
        base_relation = relation
        group_by = ()
    pop, errs = compute_population(base_relation, schema)
    if pop is None:
        return None
    return (pop.table, pop.pk_columns, pop.edges, pop.narrowed, group_by)


def _fact_refs(*constraints: Constraint) -> List[int]:
    refs: set[int] = set()
    for c in constraints:
        refs.update(c.fact_references)
    return sorted(refs)


def check_moment_conflicts(
    constraints: List[Constraint], schema: Schema
) -> List[Conflict]:
    groups: Dict[_GroupKey, List[Tuple[_Interval, Constraint, bool]]] = {}
    for c in constraints:
        pop_key = _moment_population_key(c.relation, schema)
        if pop_key is None:
            continue
        obs = _moment_observation(c)
        if obs is not None:
            kind, column, interval = obs
            groups.setdefault((pop_key, kind, column), []).append((interval, c, False))
        if isinstance(c.condition, Distributed):
            for kind, column, interval in _distributed_observations(c.condition):
                groups.setdefault((pop_key, kind, column), []).append(
                    (interval, c, True)
                )

    conflicts: List[Conflict] = []
    for group_key, entries in groups.items():
        if len(entries) < 2:
            continue
        _, kind, column = group_key
        merged = _FULL
        involved: List[Constraint] = []
        any_distributed = False
        for interval, c, is_dist in entries:
            merged = merged.intersect(interval)
            involved.append(c)
            any_distributed = any_distributed or is_dist
        if merged.is_empty():
            conflicts.append(
                Conflict(
                    kind="moment_vs_distributed_mismatch"
                    if any_distributed
                    else "moment_value_mismatch",
                    summary=f"Column '{column}' {kind.lower()}: stated bounds have no common intersection.",
                    involved_fact_references=_fact_refs(*involved),
                    detail=(
                        f"{kind} of '{column}' is constrained by {len(entries)} facts whose stated "
                        f"bounds don't overlap at all -- no value could satisfy every fact simultaneously."
                    ),
                    softenable=all(is_softenable(c.condition) for c in involved),
                )
            )
    return conflicts
