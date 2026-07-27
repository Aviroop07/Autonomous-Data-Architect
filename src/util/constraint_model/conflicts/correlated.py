"""Correlated matrix merge + chordal/PD-completion feasibility (Section
2.1 and 8.2 of the consolidated design; MULTIVARIATE_CONSTRAINT_DESIGN.md's
Grone-Johnson-Sa-Wolkowicz 1984 theorem). Only GAUSSIAN/STUDENT_T facts
share the correlation-matrix parameterization this check applies to --
CLAYTON/GUMBEL/FRANK's scalar theta isn't comparable the same way and has
no feasibility test here; this is a real, documented scope boundary (the
copula catalog, Section 4), not an oversight.

Also implements Section 6.1's precondition conflict: Student-t correlation
is only defined for nu > 2, so a Correlated(STUDENT_T, nu<=2) sharing 2+
columns with another correlation-family fact over the same population is a
deterministic conflict independent of any specific numeric disagreement.

Non-chordal specified-entry patterns are an explicit, honest "cannot
determine" result (ConflictReport.unsupported), never silently treated as
either feasible or infeasible -- Grone et al.'s completion guarantee only
covers the chordal case (Section 7's own documented open question).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from src.util.constraint_model.condition.cohesive import Correlated
from src.util.constraint_model.conflicts.grouping import ConstraintWithPopulation
from src.util.constraint_model.conflicts.models import Conflict
from src.util.constraint_model.constraint import is_softenable

_MATRIX_FAMILIES = frozenset({"GAUSSIAN", "STUDENT_T"})
_TOLERANCE_REL = 1e-6
_TOLERANCE_ABS = 1e-9


def _fact_refs(*items: ConstraintWithPopulation) -> List[int]:
    refs: set[int] = set()
    for c, _ in items:
        refs.update(c.fact_references)
    return sorted(refs)


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    x, y = sorted((a, b))
    return x, y


def check_correlated_conflicts(
    cluster: List[ConstraintWithPopulation],
) -> Tuple[List[Conflict], List[str]]:
    """Runs every Correlated check over one already-population-comparable
    cluster. Returns (conflicts, unsupported-descriptions)."""
    matrix_facts: List[ConstraintWithPopulation] = []
    for c, pop in cluster:
        if (
            isinstance(c.condition, Correlated)
            and c.condition.family in _MATRIX_FAMILIES
        ):
            matrix_facts.append((c, pop))

    conflicts: List[Conflict] = []
    conflicts.extend(_check_value_mismatches(matrix_facts))
    conflicts.extend(_check_student_t_precondition(matrix_facts))

    feasibility_conflicts, unsupported = _check_matrix_feasibility(matrix_facts)
    conflicts.extend(feasibility_conflicts)
    return conflicts, unsupported


def _check_value_mismatches(facts: List[ConstraintWithPopulation]) -> List[Conflict]:
    conflicts: List[Conflict] = []
    seen: Dict[frozenset, Tuple[float, ConstraintWithPopulation]] = {}
    for item in facts:
        c, _ = item
        assert isinstance(c.condition, Correlated)
        for pw in c.condition.pairwise:
            key = frozenset({pw.left, pw.right})
            if key in seen:
                prior_value, prior_item = seen[key]
                if not math.isclose(
                    prior_value,
                    pw.value,
                    rel_tol=_TOLERANCE_REL,
                    abs_tol=_TOLERANCE_ABS,
                ):
                    prior_c, _ = prior_item
                    conflicts.append(
                        Conflict(
                            kind="correlated_value_mismatch",
                            summary=f"Correlation({pw.left}, {pw.right}) disagrees: {prior_value} vs {pw.value}.",
                            involved_fact_references=_fact_refs(prior_item, item),
                            detail=(
                                f"One fact states corr({pw.left}, {pw.right})={prior_value}, "
                                f"another states {pw.value}, over the same population."
                            ),
                            softenable=is_softenable(prior_c.condition)
                            and is_softenable(c.condition),
                        )
                    )
            else:
                seen[key] = (pw.value, item)
    return conflicts


def _check_student_t_precondition(
    facts: List[ConstraintWithPopulation],
) -> List[Conflict]:
    conflicts: List[Conflict] = []
    for st_item in facts:
        st_c, _ = st_item
        assert isinstance(st_c.condition, Correlated)
        if st_c.condition.family != "STUDENT_T":
            continue
        nu = st_c.condition.shared_parameters.get("nu")
        if nu is None or nu > 2:
            continue
        st_cols = set(st_c.condition.columns)
        for other_item in facts:
            other_c, _ = other_item
            if other_c is st_c:
                continue
            assert isinstance(other_c.condition, Correlated)
            overlap = st_cols & set(other_c.condition.columns)
            if len(overlap) < 2:
                continue
            conflicts.append(
                Conflict(
                    kind="correlated_precondition_violation",
                    summary=(
                        f"Student-t nu={nu} (<=2) shares columns {sorted(overlap)} with another "
                        "correlation fact -- correlation is undefined for nu<=2."
                    ),
                    involved_fact_references=_fact_refs(st_item, other_item),
                    detail=(
                        f"Correlated(STUDENT_T, nu={nu}) has an undefined correlation "
                        f"interpretation for nu<=2 (Section 6.1), but another fact states a "
                        f"correlation-family relationship over the overlapping columns "
                        f"{sorted(overlap)} at the same population."
                    ),
                    softenable=is_softenable(st_c.condition)
                    and is_softenable(other_c.condition),
                )
            )
    return conflicts


def _check_matrix_feasibility(
    facts: List[ConstraintWithPopulation],
) -> Tuple[List[Conflict], List[str]]:
    pairwise_values: Dict[Tuple[str, str], Tuple[float, ConstraintWithPopulation]] = {}
    all_columns: set[str] = set()
    for item in facts:
        c, _ = item
        assert isinstance(c.condition, Correlated)
        all_columns.update(c.condition.columns)
        for pw in c.condition.pairwise:
            key = _pair_key(pw.left, pw.right)
            if key not in pairwise_values:
                pairwise_values[key] = (pw.value, item)

    if not pairwise_values:
        return [], []

    graph = nx.Graph()
    graph.add_nodes_from(all_columns)
    graph.add_edges_from(pairwise_values.keys())

    if not nx.is_chordal(graph):
        return [], [
            f"Correlation graph over columns {sorted(all_columns)} is not chordal -- "
            "PD-completion feasibility cannot be determined by this pass (an explicit, "
            "documented boundary, not a silent 'assumed fine')."
        ]

    conflicts: List[Conflict] = []
    seen_cliques: set[frozenset] = set()
    for clique in nx.find_cliques(graph):
        if len(clique) < 3:
            continue  # a single edge or isolated node is trivially PD
        key = frozenset(clique)
        if key in seen_cliques:
            continue
        seen_cliques.add(key)
        conflict = _check_clique_pd(sorted(clique), pairwise_values)
        if conflict is not None:
            conflicts.append(conflict)
    return conflicts, []


def _check_clique_pd(
    clique: List[str],
    pairwise_values: Dict[Tuple[str, str], Tuple[float, ConstraintWithPopulation]],
) -> Conflict | None:
    n = len(clique)
    matrix = np.eye(n)
    involved: List[ConstraintWithPopulation] = []
    for i in range(n):
        for j in range(i + 1, n):
            key = _pair_key(clique[i], clique[j])
            value, item = pairwise_values[key]
            matrix[i, j] = matrix[j, i] = value
            involved.append(item)

    try:
        np.linalg.cholesky(matrix)
        return None
    except np.linalg.LinAlgError:
        return Conflict(
            kind="correlated_infeasible_matrix",
            summary=f"Correlation matrix over {clique} is not positive-definite -- infeasible.",
            involved_fact_references=_fact_refs(*involved),
            detail=(
                f"The stated pairwise correlations among {clique} cannot jointly hold in any "
                "real distribution (Cholesky factorization fails on the fully-specified clique "
                "submatrix, per Grone-Johnson-Sa-Wolkowicz 1984)."
            ),
            softenable=all(is_softenable(c.condition) for c, _ in involved),
        )
