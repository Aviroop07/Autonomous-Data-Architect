"""Thin bridge from constraint_model's structural descriptors (row-count,
selectivity, latent-threshold variables) to the EXISTING util/algorithms/
dof_graph.py `Variable`/`Constraint` objects and `DOFGraph` classifier --
NOT a reimplementation (Section 10). Mirrors how `src/pipeline/stage3/
middleware/constraint_graph.py` bridges `cross_shard.py` shapes into
`dof_graph.py` today.

Row-count/selectivity equations are Section 4.2/4.3's fact-independent
structural equations -- auto-generated from Relation STRUCTURE, not from
any NL fact, so every `DOFConstraint` built here carries an empty
`fact_references` list by construction. `conflict_reconciler`'s handling of
a conflict among these (no fact to blame) is explicitly unresolved --
Section 4.2/13's own deferral, unchanged by this bridge.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.util.schema_model.schema import Schema
from src.util.algorithms.dof_graph import Constraint as DOFConstraint
from src.util.algorithms.dof_graph import DOFGraph
from src.util.algorithms.dof_graph import Variable as DOFVariable
from src.util.constraint_model.relation.nodes import RelationUnion
from src.util.constraint_model.relation.schema import (
    NodeNamer,
    RowCountVar,
    synthesize_schema_tree,
)


def row_count_variable(
    row_count: RowCountVar, fact_references: Optional[List[int]] = None
) -> DOFVariable:
    """The DOF Variable for a single RowCountVar's own quantity. Domain-
    bounded `[1, ...)` only for kind='grouped' (Section 4.2's "distinct
    group_by combinations present" quantity) -- the dynamic upper bound
    ("<= source row count") can't be expressed as `Variable.upper_bound`
    (a literal float, not a variable reference); left as a documented gap,
    same class as this module's other deferred structural-equation
    representation questions."""
    lower_bound = 1.0 if row_count.kind == "grouped" else None
    return DOFVariable(
        name=row_count.name,
        fact_references=fact_references or [],
        lower_bound=lower_bound,
    )


def selectivity_variable(
    row_count: RowCountVar, fact_references: Optional[List[int]] = None
) -> DOFVariable:
    """The `[0,1]`-bounded selectivity-factor variable a `Filter`'s
    RowCountVar mints (Section 4.3) -- also covers the nullable-FK
    presence-rate use case, since that's the same underlying mechanism,
    not a second one."""
    if row_count.kind != "filtered" or row_count.selectivity is None:
        raise ValueError(
            "selectivity_variable() requires a RowCountVar of kind='filtered'."
        )
    return DOFVariable(
        name=row_count.selectivity,
        fact_references=fact_references or [],
        lower_bound=0.0,
        upper_bound=1.0,
    )


def row_count_constraint(row_count: RowCountVar) -> Optional[DOFConstraint]:
    """The structural equation tying this RowCountVar to its operand(s),
    if any -- kind='free' (a BaseTable's own row count) has nothing to tie
    to; kind='grouped' only has the undocumented-as-a-Constraint upper
    bound (see row_count_variable), so it mints no equation either.
    `fact_references` is always empty -- see the module docstring."""
    if row_count.kind == "identity":
        assert row_count.equals is not None
        return DOFConstraint(
            name=f"{row_count.name}=identity",
            variables=[row_count.name, row_count.equals],
        )
    if row_count.kind == "filtered":
        assert row_count.source is not None and row_count.selectivity is not None
        return DOFConstraint(
            name=f"{row_count.name}=filter_eq",
            variables=[row_count.name, row_count.source, row_count.selectivity],
        )
    return None


def latent_threshold_variables(column: str, num_categories: int) -> List[DOFVariable]:
    """Section 8.2.1's per-column latent-threshold cut-point variables for
    a categorical column participating in a polychoric/polyserial
    `Correlated` term -- `num_categories - 1` ordered cut-points on the
    shared latent-continuous scale. Callers must supply the confirmed
    category count; this module has no way to discover it on its own (a
    `Correlated` term doesn't carry a category list -- that lives on a
    `Distributed(CATEGORICAL)` fact for the same column elsewhere)."""
    if num_categories < 2:
        raise ValueError("latent_threshold_variables requires at least 2 categories.")
    return [
        DOFVariable(name=f"{column}.threshold[{i}]") for i in range(num_categories - 1)
    ]


def collect_row_count_vars(
    node: "RelationUnion", schema: Schema, namer: Optional[NodeNamer] = None
) -> Tuple[List[RowCountVar], List[str]]:
    """Every RowCountVar along `node`'s Relation tree -- a thin wrapper
    over relation/schema.py's synthesize_schema_tree(), which is where the
    single-shared-namer recursion actually lives (needed for every
    'identity'/'filtered' equation's equals/source name to be consistent
    with the Variable this module builds for it). Pass an externally-
    shared `namer` when collecting across MULTIPLE separate relation trees
    that will be merged into one combined DOF graph -- see
    synthesize_schema_tree's own docstring for why this matters (distinct
    trees' anonymous nodes would otherwise coincidentally mint identical
    synthetic names)."""
    _, row_counts, errors = synthesize_schema_tree(node, schema, namer=namer)
    return row_counts, errors


def build_dof_graph(
    row_counts: List[RowCountVar],
    extra_variables: Optional[List[DOFVariable]] = None,
    extra_constraints: Optional[List[DOFConstraint]] = None,
) -> DOFGraph:
    """Assembles a DOFGraph from a batch of RowCountVar descriptors (one
    Variable, plus a selectivity Variable and/or a structural Constraint
    where applicable) plus any extra Variables/Constraints the caller
    already built elsewhere (e.g. latent-threshold or ordinary Distributed/
    Correlated parameter variables -- constructing those is out of this
    function's scope)."""
    variables: Dict[str, DOFVariable] = {}
    constraints: List[DOFConstraint] = []
    for rc in row_counts:
        var = row_count_variable(rc)
        variables[var.name] = var
        if rc.kind == "filtered":
            sel_var = selectivity_variable(rc)
            variables[sel_var.name] = sel_var
        con = row_count_constraint(rc)
        if con is not None:
            constraints.append(con)
    for v in extra_variables or []:
        variables[v.name] = v
    constraints.extend(extra_constraints or [])
    return DOFGraph(variables=list(variables.values()), constraints=constraints)
