"""Groups Constraints by Population so only genuinely comparable facts are
ever cross-checked against each other (Section 5) -- this is the shared
first step every conflict-family check in this package builds on.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Tuple

from src.util.schema_model.schema import Schema
from src.util.constraint_model.constraint import Constraint
from src.util.constraint_model.population import FKEdge, Population, compute_population

ConstraintWithPopulation = Tuple[Constraint, Population]

_LineageKey = Tuple[str, FrozenSet[str], FrozenSet[Tuple[FKEdge, int]]]


def annotate_populations(
    constraints: List[Constraint], schema: Schema
) -> Tuple[List[ConstraintWithPopulation], List[str]]:
    """Computes each Constraint's Population, dropping (with a reported
    error) any whose Relation can't be reduced to one at all -- callers
    decide whether an unresolvable Constraint should block the whole
    evaluation or just be skipped; this function itself never raises."""
    annotated: List[ConstraintWithPopulation] = []
    errors: List[str] = []
    for c in constraints:
        pop, errs = compute_population(c.relation, schema)
        if pop is None:
            errors.extend(
                f"Constraint (fact_references={c.fact_references}): {e}" for e in errs
            )
            continue
        annotated.append((c, pop))
    return annotated, errors


def group_by_comparable_population(
    annotated: List[ConstraintWithPopulation],
) -> List[List[ConstraintWithPopulation]]:
    """Clusters Constraints whose Populations are population_sensitive-
    comparable (Section 5) -- i.e. genuinely describe the same population,
    the precondition for any same-population statistical cross-check."""
    clusters: List[List[ConstraintWithPopulation]] = []
    for item in annotated:
        _, pop = item
        placed = False
        for cluster in clusters:
            _, rep_pop = cluster[0]
            if pop.is_comparable_with(rep_pop, population_sensitive=True):
                cluster.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])
    return clusters


def group_by_base_lineage(
    annotated: List[ConstraintWithPopulation],
) -> Dict[_LineageKey, List[ConstraintWithPopulation]]:
    """Groups by (table, pk_columns, edges) -- the same join lineage,
    possibly at different Filter depths. This is Section 6.2's precondition
    for the law-of-total-covariance reconciliation: two facts belong here
    together only if they trace through the identical joins/aggregation-
    free history, differing solely in how much they've been Filtered down.
    Aggregate-signature'd populations are excluded -- Section 5's own rule
    that an Aggregate's re-rooted population never shares identity with
    its un-aggregated ancestor, so they can't be meaningfully compared this
    way at all."""
    groups: Dict[_LineageKey, List[ConstraintWithPopulation]] = {}
    for c, pop in annotated:
        if pop.agg_signature is not None:
            continue
        key = (pop.table, pop.pk_columns, pop.edges)
        groups.setdefault(key, []).append((c, pop))
    return groups
