"""Section 6.2's law-of-total-covariance/variance reconciliation for facts
stated over related-but-different populations sharing the same join
lineage (Section 5's Population, one Filter-narrowed relative to the
other). Scoped to EXACTLY one Filter layer of difference between facts --
composing multiple stacked narrowings is a real, documented non-goal for
this pass (Case A already needs a specific structural pattern to recognize
at all; stacking would multiply that combinatorially for little practical
gain given no NL fact in this codebase's own test corpus needs it yet).

Two cases:
- **Exhaustive partition** (Case A): a "whole" fact plus two Filter-
  narrowed facts whose conditions are direct structural negations of each
  other (P and NOT P). The subset/complement means uniquely DETERMINE the
  partition proportion p via the law of total expectation -- turning the
  variance check into an EQUALITY (not just a feasibility bound), the far
  stronger and more common real case (Section 8.1's "if separate facts
  state every branch of a categorical fork key" scenario).
- **Single subset** (Case B): a "whole" fact plus ONE Filter-narrowed
  subset fact, no known complement. The proportion p is genuinely free --
  checked via existence: is there ANY p in (0,1) for which the implied
  complement variance is non-negative? Only flagged infeasible if NONE is
  (Section 6.2's own conservative "if valid, or underdetermined, there is
  no conflict" framing). This deliberately fires rarely: p->0 trivially
  recovers the whole population's own (already-valid) variance in the
  limit, which is mathematically correct, not a bug -- a genuine conflict
  here means the subset/whole stats are irreconcilable at EVERY possible
  partition size, a strong claim.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from src.pipeline.stage2.models.schema import Schema
from src.util.constraint_model.condition.cohesive import Distributed
from src.util.constraint_model.condition.predicates import RComparison, RNot
from src.util.constraint_model.conflicts.grouping import ConstraintWithPopulation
from src.util.constraint_model.conflicts.models import Conflict
from src.util.constraint_model.conflicts.moments import (
    _distributed_observations,
    _moment_observation,
)
from src.util.constraint_model.constraint import Constraint, is_softenable
from src.util.constraint_model.population import Population, compute_population
from src.util.constraint_model.relation.nodes import Aggregate

_TOL = 1e-6

_ObsKey = Tuple[str, str]  # (kind, column)


def _fact_refs(*constraints: Constraint) -> List[int]:
    refs: set[int] = set()
    for c in constraints:
        refs.update(c.fact_references)
    return sorted(refs)


def _are_complementary(cond_a: object, cond_b: object) -> bool:
    """P and NOT P (direct RNot), or (col = v) and (col != v)."""
    if isinstance(cond_a, RNot) and cond_a.operand == cond_b:
        return True
    if isinstance(cond_b, RNot) and cond_b.operand == cond_a:
        return True
    if isinstance(cond_a, RComparison) and isinstance(cond_b, RComparison):
        if cond_a.left == cond_b.left and cond_a.right == cond_b.right:
            if {cond_a.op, cond_b.op} == {"=", "!="}:
                return True
    return False


def _point_observations(c: Constraint) -> List[Tuple[str, str, float]]:
    """(kind, column, value) POINT observations -- only exact '=' moment
    facts and Distributed-implied moments participate; an inequality fact
    doesn't pin a single number these equations can be solved with."""
    out: List[Tuple[str, str, float]] = []
    obs = _moment_observation(c)
    if obs is not None:
        kind, column, interval = obs
        if interval.lo == interval.hi:
            out.append((kind, column, interval.lo))
    if isinstance(c.condition, Distributed):
        for kind, column, interval in _distributed_observations(c.condition):
            out.append((kind, column, interval.lo))
    return out


def _by_key(
    items: List[ConstraintWithPopulation],
) -> Dict[_ObsKey, List[Tuple[float, Constraint]]]:
    result: Dict[_ObsKey, List[Tuple[float, Constraint]]] = {}
    for c, _ in items:
        for kind, column, value in _point_observations(c):
            result.setdefault((kind, column), []).append((value, c))
    return result


_LineageKey = Tuple[str, frozenset, frozenset, Tuple[str, ...]]


def _reconciliation_population(
    c: Constraint, schema: Schema
) -> Tuple[Optional[Population], Tuple[str, ...]]:
    """The population this reconciliation should key on: for a moment fact
    (relation = Aggregate(source=..., group_by=...)), that's the
    AGGREGATE'S SOURCE population (its own narrowed/filter_conditions
    describe exactly the population being averaged over) plus its
    group_by, not the Aggregate node's own agg_signature'd population
    (which grouping.py's shared group_by_base_lineage deliberately
    excludes entirely, per Section 5's "an Aggregate never shares
    identity with its ancestor" rule -- correct for THAT module's
    Distributed/Correlated-oriented purpose, wrong for this one, where
    moment facts are the whole point and are always Aggregate-wrapped).
    For any other Constraint (e.g. a bare Distributed fact), the
    relation's own population is used directly, group_by=()."""
    if isinstance(c.relation, Aggregate):
        pop, _errs = compute_population(c.relation.source, schema)
        group_by = tuple(sorted(c.relation.group_by or ()))
    else:
        pop, _errs = compute_population(c.relation, schema)
        group_by = ()
    return pop, group_by


def _lineage_key(pop: Population, group_by: Tuple[str, ...]) -> _LineageKey:
    return (pop.table, pop.pk_columns, pop.edges, group_by)


def check_population_reconciliation(
    constraints: List[Constraint], schema: Schema
) -> List[Conflict]:
    lineage_groups: Dict[_LineageKey, List[ConstraintWithPopulation]] = {}
    for c in constraints:
        pop, group_by = _reconciliation_population(c, schema)
        if pop is None:
            continue
        lineage_groups.setdefault(_lineage_key(pop, group_by), []).append((c, pop))

    conflicts: List[Conflict] = []
    for items in lineage_groups.values():
        whole = [(c, pop) for c, pop in items if len(pop.filter_conditions) == 0]
        filtered = [(c, pop) for c, pop in items if len(pop.filter_conditions) == 1]
        if not whole or not filtered:
            continue
        conflicts.extend(_check_lineage_group(whole, filtered))
    return conflicts


def _check_lineage_group(
    whole: List[ConstraintWithPopulation], filtered: List[ConstraintWithPopulation]
) -> List[Conflict]:
    """Groups `filtered` by their OWN filter condition first (predicates
    aren't hashable, so this is an equality-based O(n^2) grouping, fine
    for realistic fact counts) -- multiple separately-extracted facts
    (e.g. an atomic mean fact and a separate atomic variance fact) can and
    routinely will share the identical condition, describing the SAME
    population; merging them before cross-checking is what makes the
    reconciliation see mean+variance together instead of only whichever
    single fact happened to be inspected."""
    condition_groups: List[Tuple[object, List[ConstraintWithPopulation]]] = []
    for item in filtered:
        _, pop = item
        cond = pop.filter_conditions[0]
        for existing_cond, group in condition_groups:
            if existing_cond == cond:
                group.append(item)
                break
        else:
            condition_groups.append((cond, [item]))

    whole_by_key = _by_key(whole)
    conflicts: List[Conflict] = []
    matched: set[int] = set()
    for i in range(len(condition_groups)):
        cond_i, items_i = condition_groups[i]
        for j in range(i + 1, len(condition_groups)):
            cond_j, items_j = condition_groups[j]
            if _are_complementary(cond_i, cond_j):
                matched.add(i)
                matched.add(j)
                conflicts.extend(
                    _check_exhaustive_partition(whole_by_key, items_i, items_j)
                )

    for i, (_cond, items) in enumerate(condition_groups):
        if i not in matched:
            conflicts.extend(_check_single_subset(whole_by_key, items))

    return conflicts


def _check_exhaustive_partition(
    whole_by_key: Dict[_ObsKey, List[Tuple[float, Constraint]]],
    subset_items: List[ConstraintWithPopulation],
    complement_items: List[ConstraintWithPopulation],
) -> List[Conflict]:
    subset_by_key = _by_key(subset_items)
    complement_by_key = _by_key(complement_items)

    conflicts: List[Conflict] = []
    for key in sorted(set(subset_by_key) & set(complement_by_key)):
        kind, column = key
        if kind != "MEAN":
            continue
        for mu_s, c_s in subset_by_key[key]:
            for mu_c, c_c in complement_by_key[key]:
                if math.isclose(mu_s, mu_c, rel_tol=1e-9):
                    continue  # subset/complement means coincide -- p underdetermined

                for mu_w, c_w in whole_by_key.get(key, []):
                    p = (mu_w - mu_c) / (mu_s - mu_c)
                    if not (-_TOL <= p <= 1.0 + _TOL):
                        conflicts.append(
                            Conflict(
                                kind="population_reconciliation_infeasible",
                                summary=(
                                    f"Column '{column}' mean: whole-population mean is not a "
                                    "valid weighted average of its exhaustive partition."
                                ),
                                involved_fact_references=_fact_refs(c_w, c_s, c_c),
                                detail=(
                                    f"Solving mu_whole = p*mu_subset + (1-p)*mu_complement for "
                                    f"the partition proportion p gives p={p:.6f}, outside the "
                                    f"valid range [0,1] (mu_whole={mu_w}, mu_subset={mu_s}, "
                                    f"mu_complement={mu_c})."
                                ),
                                softenable=all(
                                    is_softenable(x.condition) for x in (c_w, c_s, c_c)
                                ),
                            )
                        )
                        continue

                    var_key = ("VARIANCE", column)
                    if var_key not in subset_by_key or var_key not in complement_by_key:
                        continue
                    for var_s, c_s2 in subset_by_key[var_key]:
                        for var_c_, c_c2 in complement_by_key[var_key]:
                            for var_w, c_w2 in whole_by_key.get(var_key, []):
                                implied_var_w = (
                                    p * var_s
                                    + (1 - p) * var_c_
                                    + p * (1 - p) * (mu_s - mu_c) ** 2
                                )
                                if not math.isclose(
                                    var_w, implied_var_w, rel_tol=1e-3, abs_tol=1e-6
                                ):
                                    conflicts.append(
                                        Conflict(
                                            kind="population_reconciliation_infeasible",
                                            summary=(
                                                f"Column '{column}' variance: law-of-total-"
                                                "variance mismatch across the exhaustive partition."
                                            ),
                                            involved_fact_references=_fact_refs(
                                                c_w2, c_s2, c_c2, c_w, c_s, c_c
                                            ),
                                            detail=(
                                                f"Given p={p:.6f} (determined from the means), "
                                                f"the law of total variance implies Var_whole="
                                                f"{implied_var_w:.6f}, but the whole-population "
                                                f"fact states Var_whole={var_w:.6f}."
                                            ),
                                            softenable=all(
                                                is_softenable(x.condition)
                                                for x in (
                                                    c_w2,
                                                    c_s2,
                                                    c_c2,
                                                    c_w,
                                                    c_s,
                                                    c_c,
                                                )
                                            ),
                                        )
                                    )
    return conflicts


def _check_single_subset(
    whole_by_key: Dict[_ObsKey, List[Tuple[float, Constraint]]],
    subset_items: List[ConstraintWithPopulation],
) -> List[Conflict]:
    subset_by_key = _by_key(subset_items)
    conflicts: List[Conflict] = []

    for key, whole_entries in whole_by_key.items():
        kind, column = key
        if kind != "VARIANCE" or key not in subset_by_key:
            continue
        mean_key = ("MEAN", column)
        if mean_key not in subset_by_key or mean_key not in whole_by_key:
            continue
        for var_s, c_s_var in subset_by_key[key]:
            for mu_s, c_s_mean in subset_by_key[mean_key]:
                for var_w, c_w_var in whole_entries:
                    for mu_w, c_w_mean in whole_by_key[mean_key]:
                        if not _feasible_for_some_p(mu_w, var_w, mu_s, var_s):
                            conflicts.append(
                                Conflict(
                                    kind="population_reconciliation_infeasible",
                                    summary=(
                                        f"Column '{column}' variance: no valid partition "
                                        "proportion reconciles subset and whole-population "
                                        "statistics."
                                    ),
                                    involved_fact_references=_fact_refs(
                                        c_w_var, c_w_mean, c_s_var, c_s_mean
                                    ),
                                    detail=(
                                        f"For EVERY p in (0,1), the law of total variance "
                                        f"implies a negative complement variance given "
                                        f"mu_whole={mu_w}, var_whole={var_w}, mu_subset={mu_s}, "
                                        f"var_subset={var_s} -- no valid database could satisfy "
                                        "both facts."
                                    ),
                                    softenable=all(
                                        is_softenable(x.condition)
                                        for x in (c_w_var, c_w_mean, c_s_var, c_s_mean)
                                    ),
                                )
                            )
    return conflicts


def _feasible_for_some_p(mu_w: float, var_w: float, mu_s: float, var_s: float) -> bool:
    """Analytic feasibility check for "does some p in (0,1) give a non-
    negative implied complement variance". An earlier version of this
    function used a fixed-resolution numeric grid search over p -- WRONG
    in general: for a large |mu_s - mu_w| gap, the only feasible p can be
    arbitrarily close to 0 (see the derivation below), far below any fixed
    grid's resolution, producing false "infeasible" verdicts on perfectly
    reconcilable inputs. Replaced with a closed-form derivation instead of
    a finer grid, since no fixed resolution is ever truly safe.

    Multiplying the law-of-total-variance identity through by (1-p)^2 > 0
    (valid for p in (0,1)) turns the question into: is
      f(p) = var_s*p^2 - B*p + var_w,  where B = var_w + var_s + (mu_s-mu_w)^2 >= 0,
    non-negative for some p in (0,1)? f(0) = var_w and f'(0) = -B <= 0, so:
    - var_w > 0: ALWAYS feasible -- continuity from a strictly positive
      f(0) guarantees f(p) > 0 for p in some neighborhood right of 0.
    - var_w == 0: feasible ONLY in the degenerate case where the subset is
      identical to the whole (var_s == 0 and mu_s == mu_w) -- otherwise
      f'(0) < 0 drives f(p) negative immediately past p=0, for every p.
    - var_w < 0: not a legitimate variance in the first place; correctly
      falls out as infeasible too.
    """
    if var_w > 0:
        return True
    if var_w == 0:
        return var_s == 0 and math.isclose(mu_s, mu_w, rel_tol=1e-9, abs_tol=1e-9)
    return False
