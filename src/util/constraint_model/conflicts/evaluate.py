"""Top-level API: evaluate_constraints(constraints, schema) -> ConflictReport.

Combines every conflict-family check built in this package (Distributed,
Correlated, moment facts, law-of-total-covariance population
reconciliation, StateSequence) plus a structural check reusing the
EXISTING util/algorithms/dof_graph.py classifier via variables.py's bridge
(row-count/selectivity equations forming a structurally overconstrained
block -- Section 4.2's fact-independent structural equations, never tied
to any single NL fact, so never softenable and never carrying
fact_references).

This is a purely deterministic evaluation -- no LLM calls anywhere in this
package. It exposes conflicts for a FUTURE reconciliation loop to consume
and judge (MISEXTRACTION/FALSE_POSITIVE/GENUINE_CONTRADICTION/SOFTEN);
this function itself never makes that judgment.
"""

from __future__ import annotations

from typing import List

from src.pipeline.stage2.models.schema import Schema
from src.util.constraint_model.conflicts.correlated import check_correlated_conflicts
from src.util.constraint_model.conflicts.distributed import check_distributed_conflicts
from src.util.constraint_model.conflicts.grouping import (
    annotate_populations,
    group_by_comparable_population,
)
from src.util.constraint_model.conflicts.models import Conflict, ConflictReport
from src.util.constraint_model.conflicts.moments import check_moment_conflicts
from src.util.constraint_model.conflicts.population_reconcile import (
    check_population_reconciliation,
)
from src.util.constraint_model.conflicts.state_sequence import (
    check_state_sequence_conflicts,
)
from src.util.constraint_model.constraint import Constraint
from src.util.constraint_model.relation.schema import NodeNamer, RowCountVar
from src.util.constraint_model.variables import build_dof_graph, collect_row_count_vars


def evaluate_constraints(
    constraints: List[Constraint], schema: Schema
) -> ConflictReport:
    conflicts: List[Conflict] = []
    unsupported: List[str] = []

    annotated, pop_errors = annotate_populations(constraints, schema)
    unsupported.extend(pop_errors)

    for cluster in group_by_comparable_population(annotated):
        conflicts.extend(check_distributed_conflicts(cluster))
        correlated_conflicts, correlated_unsupported = check_correlated_conflicts(
            cluster
        )
        conflicts.extend(correlated_conflicts)
        unsupported.extend(correlated_unsupported)

    conflicts.extend(check_moment_conflicts(constraints, schema))
    conflicts.extend(check_population_reconciliation(constraints, schema))
    conflicts.extend(check_state_sequence_conflicts(constraints, schema))
    conflicts.extend(_check_structural_overconstrained(constraints, schema))

    return ConflictReport(conflicts=conflicts, unsupported=unsupported)


def _check_structural_overconstrained(
    constraints: List[Constraint], schema: Schema
) -> List[Conflict]:
    """Merges every Constraint's row-count/selectivity structural
    equations into ONE combined DOF graph and reports any overconstrained
    block. A single SHARED NodeNamer is used across all of them -- see
    relation/schema.py's synthesize_schema_tree docstring for why this is
    required, not optional: two DIFFERENT constraints' relations, each
    rooted at an anonymous (alias-less) node, would otherwise coincidentally
    mint the identical synthetic row-count-variable name, silently merging
    two unrelated quantities into one DOF variable."""
    namer = NodeNamer()
    all_row_counts: List[RowCountVar] = []
    seen_names: set[str] = set()
    for c in constraints:
        row_counts, _errs = collect_row_count_vars(c.relation, schema, namer=namer)
        for rc in row_counts:
            if rc.name not in seen_names:
                seen_names.add(rc.name)
                all_row_counts.append(rc)

    if not all_row_counts:
        return []

    graph = build_dof_graph(all_row_counts)
    classification = graph.classify()

    conflicts: List[Conflict] = []
    for block in classification.overconstrained_blocks:
        conflicts.append(
            Conflict(
                kind="structural_overconstrained",
                summary=(
                    f"Structural row-count/selectivity block is overconstrained: "
                    f"variables {block.variables}."
                ),
                involved_fact_references=[],
                detail=(
                    f"The auto-generated row-count/selectivity structural equations over "
                    f"variables {block.variables} (equations: {block.constraints}) form an "
                    "overconstrained block -- more equations than degrees of freedom. This is "
                    "not derived from any single NL fact (Section 4.2), so no fact_references "
                    "apply; a conflict here signals a structural/data inconsistency, not a "
                    "misread fact."
                ),
                softenable=False,
            )
        )
    return conflicts
