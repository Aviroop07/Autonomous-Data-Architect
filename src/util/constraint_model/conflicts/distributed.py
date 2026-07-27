"""Same-population Distributed-vs-Distributed conflict checks (Section
8.1). Precondition conflicts involving Correlated (e.g. Student-t nu<=2
combined with a stated Pearson correlation for the same pair, Section 6.1)
live in correlated.py, not here -- this module is Distributed-only.

Per Section 8.1's own simplifying assumption (the source NL is self-
consistent, only possibly incomplete): "two facts assert different
families for the same column+grain" is expected to resolve as MISEXTRACTION
almost always -- this module still reports it as a Conflict (that
judgment is for the future reconciliation loop, not this evaluator).
"""

from __future__ import annotations

import math
from typing import Dict, List, Union

from src.util.constraint_model.condition.cohesive import Distributed
from src.util.constraint_model.conflicts.grouping import ConstraintWithPopulation
from src.util.constraint_model.conflicts.models import Conflict
from src.util.constraint_model.constraint import is_softenable

_NUMERIC_TOLERANCE_REL = 1e-6
_NUMERIC_TOLERANCE_ABS = 1e-9


def _numeric_close(a: float, b: float) -> bool:
    return math.isclose(
        a, b, rel_tol=_NUMERIC_TOLERANCE_REL, abs_tol=_NUMERIC_TOLERANCE_ABS
    )


def _fact_refs(*items: ConstraintWithPopulation) -> List[int]:
    refs: set[int] = set()
    for c, _ in items:
        refs.update(c.fact_references)
    return sorted(refs)


def check_distributed_conflicts(
    cluster: List[ConstraintWithPopulation],
) -> List[Conflict]:
    """Checks every pair of Distributed facts about the SAME column within
    one already-population-comparable cluster."""
    by_column: Dict[str, List[ConstraintWithPopulation]] = {}
    for item in cluster:
        constraint, _ = item
        if isinstance(constraint.condition, Distributed):
            by_column.setdefault(constraint.condition.column, []).append(item)

    conflicts: List[Conflict] = []
    for column, items in by_column.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                conflicts.extend(_check_pair(column, items[i], items[j]))
    return conflicts


def _check_pair(
    column: str, item_a: ConstraintWithPopulation, item_b: ConstraintWithPopulation
) -> List[Conflict]:
    a, _ = item_a
    b, _ = item_b
    assert isinstance(a.condition, Distributed) and isinstance(b.condition, Distributed)
    da, db = a.condition, b.condition
    refs = _fact_refs(item_a, item_b)
    softenable = is_softenable(da) and is_softenable(b.condition)

    if da.family != db.family:
        return [
            Conflict(
                kind="distributed_family_mismatch",
                summary=f"Column '{column}': conflicting distribution families {da.family} vs {db.family}.",
                involved_fact_references=refs,
                detail=(
                    f"One fact states Distributed(column='{column}', family='{da.family}') "
                    f"while another, over the same population, states family='{db.family}'."
                ),
                softenable=softenable,
            )
        ]

    conflicts: List[Conflict] = []
    common_keys = set(da.parameters) & set(db.parameters)
    for key in sorted(common_keys):
        va, vb = da.parameters[key], db.parameters[key]
        mismatch_detail = _compare_parameter(key, va, vb)
        if mismatch_detail is not None:
            conflicts.append(
                Conflict(
                    kind="distributed_parameter_mismatch",
                    summary=f"Column '{column}' ({da.family}): parameter '{key}' disagrees.",
                    involved_fact_references=refs,
                    detail=mismatch_detail,
                    softenable=softenable,
                )
            )
    return conflicts


def _compare_parameter(
    key: str,
    va: Union[float, List[str], List[float]],
    vb: Union[float, List[str], List[float]],
) -> str | None:
    if key == "categories":
        set_a, set_b = (
            set(va) if isinstance(va, list) else set(),
            set(vb) if isinstance(vb, list) else set(),
        )
        if set_a != set_b:
            return f"categories differ: {sorted(set_a)} vs {sorted(set_b)}."
        return None
    if key == "probabilities":
        # Only meaningfully comparable alongside a matching categories list;
        # a bare probabilities-list mismatch (different lengths/values) is
        # still worth flagging even without cross-referencing categories.
        if isinstance(va, list) and isinstance(vb, list):
            if len(va) == len(vb) and all(
                isinstance(x, (int, float))
                and isinstance(y, (int, float))
                and _numeric_close(x, y)
                for x, y in zip(va, vb)
            ):
                return None
            return f"probabilities differ: {va} vs {vb}."
        return None
    if (
        isinstance(va, (int, float))
        and isinstance(vb, (int, float))
        and not _numeric_close(va, vb)
    ):
        return f"{key}={va} vs {key}={vb}."
    return None
