"""StateSequence cross-fact transition-graph merge (Section 9.2).

Facts are grouped by sequence_column AND population comparability
(Section 5) -- two StateSequence facts naming the same sequence_column over
unrelated tables/populations must never be merged just because the column
name happens to coincide. Within one such group:

1. Union every fact's allowed/forbidden transitions into one merged graph.
2. **Direct contradiction**: an edge asserted both allowed (by some fact)
   and forbidden (by another) -- a syntactic, unconditional conflict.
3. **Cycle**: a cycle in the merged ALLOWED-transitions graph, but ONLY
   flagged when at least one contributing fact declares `strict=True`
   (cycles are allowed by default -- returns/reprocessing are legitimate
   real flows, so silence on a cycle is the correct default, not a gap).

Both conflict kinds are NEVER softenable (Section 11.2's explicit list:
"StateSequence transitions... never softenable" -- a binary integrity
property, not a numeric approximation).
"""

from __future__ import annotations

from typing import List, Tuple

import networkx as nx

from src.util.constraint_model.condition.cohesive import StateSequence
from src.util.constraint_model.conflicts.models import Conflict
from src.util.constraint_model.constraint import Constraint
from src.util.constraint_model.population import Population, compute_population
from src.pipeline.stage2.models.schema import Schema

_Edge = Tuple[str, str]


def _fact_refs(*constraints: Constraint) -> List[int]:
    refs: set[int] = set()
    for c in constraints:
        refs.update(c.fact_references)
    return sorted(refs)


def _column_matches(a: StateSequence, b: StateSequence) -> bool:
    return a.sequence_column == b.sequence_column


def check_state_sequence_conflicts(
    constraints: List[Constraint], schema: Schema
) -> List[Conflict]:
    state_seq_items: List[Tuple[Constraint, StateSequence, Population]] = []
    for c in constraints:
        if not isinstance(c.condition, StateSequence):
            continue
        pop, _errs = compute_population(c.relation, schema)
        if pop is None:
            continue
        state_seq_items.append((c, c.condition, pop))

    groups: List[List[Tuple[Constraint, StateSequence, Population]]] = []
    for item in state_seq_items:
        c, cond, pop = item
        placed = False
        for group in groups:
            _, rep_cond, rep_pop = group[0]
            if _column_matches(cond, rep_cond) and pop.is_comparable_with(
                rep_pop, population_sensitive=True
            ):
                group.append(item)
                placed = True
                break
        if not placed:
            groups.append([item])

    conflicts: List[Conflict] = []
    for group in groups:
        if len(group) < 2:
            continue
        conflicts.extend(_check_group(group))
    return conflicts


def _check_group(
    group: List[Tuple[Constraint, StateSequence, Population]],
) -> List[Conflict]:
    allowed_owner: dict[_Edge, Constraint] = {}
    forbidden_owner: dict[_Edge, Constraint] = {}
    any_strict = False
    strict_owners: List[Constraint] = []

    for c, cond, _pop in group:
        if cond.strict:
            any_strict = True
            strict_owners.append(c)
        for t in cond.allowed_transitions:
            allowed_owner.setdefault((t.from_state, t.to_state), c)
        for t in cond.forbidden_transitions:
            forbidden_owner.setdefault((t.from_state, t.to_state), c)

    conflicts: List[Conflict] = []
    for edge in sorted(set(allowed_owner) & set(forbidden_owner)):
        c_allow = allowed_owner[edge]
        c_forbid = forbidden_owner[edge]
        conflicts.append(
            Conflict(
                kind="state_sequence_direct_contradiction",
                summary=f"Transition {edge[0]} -> {edge[1]} is asserted both allowed and forbidden.",
                involved_fact_references=_fact_refs(c_allow, c_forbid),
                detail=(
                    f"One fact allows the transition {edge[0]} -> {edge[1]}, another fact "
                    "forbids the same transition, for the same sequence_column over a "
                    "comparable population."
                ),
                softenable=False,
            )
        )

    if any_strict:
        graph = nx.DiGraph()
        graph.add_edges_from(allowed_owner.keys())
        try:
            cycle = nx.find_cycle(graph)
        except nx.NetworkXNoCycle:
            cycle = None
        if cycle is not None:
            cycle_edges = [(u, v) for u, v, *_ in cycle]
            involved = [allowed_owner[e] for e in cycle_edges if e in allowed_owner]
            involved.extend(strict_owners)
            conflicts.append(
                Conflict(
                    kind="state_sequence_cycle",
                    summary=f"Cycle detected in the merged allowed-transitions graph: {cycle_edges}.",
                    involved_fact_references=_fact_refs(*involved),
                    detail=(
                        f"The union of allowed transitions across all facts for this "
                        f"sequence_column "
                        f"contains a cycle ({cycle_edges}), but at least one fact declares this "
                        "sequence strict/acyclic (strict=True) -- cycles are allowed by default, "
                        "only flagged when explicitly asserted acyclic."
                    ),
                    softenable=False,
                )
            )
    return conflicts
