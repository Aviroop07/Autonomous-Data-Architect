"""Converts extracted Stage 3 constraints into the generic DOF graph
(src/util/algorithms/dof_graph.py), per STAGE3_PHASE2_DESIGN.md's taxonomy v2.

Scope of this module: StatisticalManifest.distributions and moment_targets
(Q3's derivation-chain walk, section 4), and StructuralManifest's
cardinalities/fanouts. Explicitly NOT handled: UniqueConstraint and
FormatConstraint (neither has a numeric parameter to pin -- not a DOF
concept at all), and conditional CrossColumnLogic (if_condition is set --
needs Q4's fork-key registry, no design yet).
"""

from __future__ import annotations

import sqlglot
from sqlglot import expressions as exp

from src.pipeline.stage3.models.constraints import (
    AggregationConstraint,
    BetaDistribution,
    CategoricalDistribution,
    ConstraintManifest,
    CrossColumnLogic,
    DistributionConstraint,
    FanoutConstraint,
    GaussianDistribution,
    LogNormalDistribution,
    MomentTarget,
    PoissonDistribution,
    StatisticalManifest,
    StructuralManifest,
    TableCardinality,
    UniformDistribution,
)
from src.util.algorithms.dof_graph import Constraint, Variable


def _merge_variable(a: Variable, b: Variable) -> Variable:
    """Combine two Variables that turned out to name the same parameter
    (e.g. two facts about the same column or table). Re-constructs rather
    than mutating in place, so Variable's bounds-ordering validator still
    runs on the merged result instead of being silently bypassed."""
    lower_candidates = [x for x in (a.lower_bound, b.lower_bound) if x is not None]
    upper_candidates = [x for x in (a.upper_bound, b.upper_bound) if x is not None]
    return Variable(
        name=a.name,
        fact_references=sorted(set(a.fact_references) | set(b.fact_references)),
        lower_bound=max(lower_candidates) if lower_candidates else None,
        upper_bound=min(upper_candidates) if upper_candidates else None,
    )


def _accumulate(
    variables_by_name: dict[str, Variable],
    constraints: list[Constraint],
    new_variables: list[Variable],
    new_constraints: list[Constraint],
) -> None:
    for variable in new_variables:
        existing = variables_by_name.get(variable.name)
        variables_by_name[variable.name] = (
            variable if existing is None else _merge_variable(existing, variable)
        )
    constraints.extend(new_constraints)


def distribution_to_graph_nodes(
    dist: DistributionConstraint, disambiguator: int
) -> tuple[list[Variable], list[Constraint]]:
    """Split one distribution fact into per-parameter Variable/Constraint
    pairs -- one Constraint PER parameter, never one Constraint touching
    several. A single fact stating "mean 8, std 2" pins two independent
    unknowns; modeling it as one multi-variable equation is mathematically
    underdetermined (see STAGE3_PHASE2_DESIGN.md section 2).

    `disambiguator` only matters when two distinct facts pin the same
    parameter (a real, if rare, case this must not crash on -- it should
    surface as an overconstrained block once graphed, not a construction
    error) -- it keeps their Constraint names apart while sharing one
    Variable name.
    """
    qualified_column = f"{dist.table_name}.{dist.column_name}"
    refs = dist.fact_references

    if isinstance(dist, CategoricalDistribution):
        var_name = f"{qualified_column}.probabilities"
        variable = Variable(name=var_name, fact_references=refs)
        if dist.probabilities is None:
            return [variable], []
        constraint = Constraint(
            name=f"pin_{var_name}#{disambiguator}",
            variables=[var_name],
            fact_references=refs,
        )
        return [variable], [constraint]

    if isinstance(dist, (GaussianDistribution, LogNormalDistribution)):
        params = ["mean", "std_dev"]
    elif isinstance(dist, BetaDistribution):
        params = ["alpha", "beta"]
    elif isinstance(dist, PoissonDistribution):
        params = ["lam"]
    elif isinstance(dist, UniformDistribution):
        params = ["min_value", "max_value"]
    else:
        raise ValueError(f"Unhandled distribution family: {type(dist).__name__}")

    variables = []
    constraints = []
    for param in params:
        var_name = f"{qualified_column}.{param}"
        variables.append(Variable(name=var_name, fact_references=refs))
        constraints.append(
            Constraint(
                name=f"pin_{var_name}#{disambiguator}",
                variables=[var_name],
                fact_references=refs,
            )
        )
    return variables, constraints


def statistical_manifest_to_graph_nodes(
    manifest: StatisticalManifest,
) -> tuple[list[Variable], list[Constraint]]:
    """Flatten a whole StatisticalManifest into deduplicated Variables (one
    per unique parameter, even if multiple facts pin it) and Constraints
    (one per fact -- multiple facts pinning the same parameter stay as
    separate Constraints, so DOFGraph's classification can flag the
    conflict as an overconstrained block instead of this function silently
    picking a winner)."""
    variables_by_name: dict[str, Variable] = {}
    constraints: list[Constraint] = []

    for index, dist in enumerate(manifest.distributions):
        new_variables, new_constraints = distribution_to_graph_nodes(
            dist, disambiguator=index
        )
        _accumulate(variables_by_name, constraints, new_variables, new_constraints)

    return list(variables_by_name.values()), constraints


def table_cardinality_to_graph_nodes(
    cardinality: TableCardinality, disambiguator: int
) -> tuple[Variable, list[Constraint]]:
    """One row-count Variable per table. An exact target (min_rows ==
    max_rows) becomes a pinning Constraint; a genuine range is domain-bound
    metadata only (section 3's hybrid inequality handling) -- the variable
    stays loose, with its plausible range recorded for the later
    LLM-probe/materialization step rather than discarded."""
    var_name = f"{cardinality.table_name}.row_count"
    variable = Variable(
        name=var_name,
        fact_references=cardinality.fact_references,
        lower_bound=float(cardinality.min_rows),
        upper_bound=float(cardinality.max_rows),
    )
    if cardinality.min_rows == cardinality.max_rows:
        constraint = Constraint(
            name=f"pin_{var_name}#{disambiguator}",
            variables=[var_name],
            fact_references=cardinality.fact_references,
        )
        return variable, [constraint]
    return variable, []


def fanout_constraint_to_graph_nodes(
    fanout: FanoutConstraint, disambiguator: int
) -> tuple[Variable, list[Constraint]]:
    """One fan-out-mean Variable per FK relationship. Same exact-vs-range
    treatment as table_cardinality_to_graph_nodes, except max_fanout is
    optional (min_fanout always is given) -- an exact target requires both
    to be present and equal."""
    fk_suffix = "_".join(fanout.foreign_key_columns)
    var_name = f"{fanout.parent_table}->{fanout.child_table}.fanout_mean[{fk_suffix}]"
    variable = Variable(
        name=var_name,
        fact_references=fanout.fact_references,
        lower_bound=fanout.min_fanout,
        upper_bound=fanout.max_fanout,
    )
    if fanout.max_fanout is not None and fanout.min_fanout == fanout.max_fanout:
        constraint = Constraint(
            name=f"pin_{var_name}#{disambiguator}",
            variables=[var_name],
            fact_references=fanout.fact_references,
        )
        return variable, [constraint]
    return variable, []


def structural_manifest_to_graph_nodes(
    manifest: StructuralManifest,
) -> tuple[list[Variable], list[Constraint]]:
    """Cardinalities and fan-outs only -- UniqueConstraint has no numeric
    parameter to pin (not a DOF concept) and AggregationConstraint needs
    Q3's dispatcher, not built yet. Both are silently skipped here, not
    because they're unimportant, but because taxonomy v2 either doesn't
    route them through this graph at all (uniqueness) or doesn't have a
    design yet (aggregations)."""
    variables_by_name: dict[str, Variable] = {}
    constraints: list[Constraint] = []

    for index, cardinality in enumerate(manifest.cardinalities):
        variable, new_constraints = table_cardinality_to_graph_nodes(
            cardinality, disambiguator=index
        )
        _accumulate(variables_by_name, constraints, [variable], new_constraints)

    for index, fanout in enumerate(manifest.fanouts):
        variable, new_constraints = fanout_constraint_to_graph_nodes(
            fanout, disambiguator=index
        )
        _accumulate(variables_by_name, constraints, [variable], new_constraints)

    return list(variables_by_name.values()), constraints


class _BailOut(Exception):
    """Raised internally when the derivation walk hits an unsupported shape
    (section 4.4's trigger list) -- caught once, at the top of _resolve_mean,
    and turned into a None return so the whole MomentTarget stays
    unresolved rather than partially resolved."""


def _base_mean_variable(
    table_name: str, column_name: str, manifest: ConstraintManifest
) -> str | None:
    """The Variable name representing E[column], for a column that is NOT
    derived (no matching AggregationConstraint or unconditional
    CrossColumnLogic). Gaussian/LogNormal contribute their `.mean` parameter
    directly; Poisson contributes `.lam`. Beta/Uniform/Categorical don't have
    a single parameter equal to the mean (it's a nonlinear combination of
    several) -- unsupported, bail. A column with no distribution at all is a
    genuinely free quantity: mint the same `.mean`-shaped name so it merges
    cleanly with any distribution fact that might independently pin it."""
    qualified = f"{table_name}.{column_name}"
    for dist in manifest.statistical.distributions:
        if dist.table_name != table_name or dist.column_name != column_name:
            continue
        if isinstance(dist, (GaussianDistribution, LogNormalDistribution)):
            return f"{qualified}.mean"
        if isinstance(dist, PoissonDistribution):
            return f"{qualified}.lam"
        return None
    return f"{qualified}.mean"


def _find_unconditional_cross_column(
    table_name: str, column_name: str, manifest: ConstraintManifest
) -> tuple[CrossColumnLogic, tuple[exp.Column, exp.Column]] | None:
    """Find the unconditional CrossColumnLogic fact defining column_name as
    a product or sum of exactly two base columns. Returns (fact, (lhs, rhs))
    on a match, or None if no fact defines this column this way at all.
    Raises `_BailOut` if a fact DOES define it but the shape isn't one this
    pass supports (see design doc section 4.4)."""
    for cross in manifest.logic.cross_column_logic:
        if cross.table_context != table_name or cross.if_condition is not None:
            continue
        try:
            parsed = sqlglot.parse_one(cross.then_enforcement)
        except Exception:
            raise _BailOut
        if not isinstance(parsed, exp.EQ) or not isinstance(parsed.this, exp.Column):
            raise _BailOut
        if parsed.this.name != column_name:
            continue
        rhs = parsed.expression
        if not isinstance(rhs, (exp.Mul, exp.Add)):
            raise _BailOut
        operands = (rhs.this, rhs.expression)
        if not all(isinstance(operand, exp.Column) for operand in operands):
            raise _BailOut
        return cross, operands
    return None


def _resolve_mean(
    table_name: str,
    column_name: str,
    manifest: ConstraintManifest,
    visited: frozenset[tuple[str, str]],
) -> tuple[set[str], set[int]] | None:
    """Recursively resolve E[column_name] to the set of already-graphable
    Variable names it depends on, per STAGE3_PHASE2_DESIGN.md section 4.2.
    Returns None on any bail-out condition (section 4.4) -- the caller
    treats the whole MomentTarget as unresolved, never partially resolved."""
    key = (table_name, column_name)
    if key in visited:
        return None
    visited = visited | {key}

    aggregation = next(
        (
            agg
            for agg in manifest.structural.aggregations
            if agg.parent_table == table_name and agg.parent_column == column_name
        ),
        None,
    )
    if aggregation is not None:
        return _resolve_aggregation(aggregation, manifest, visited)

    try:
        cross_match = _find_unconditional_cross_column(
            table_name, column_name, manifest
        )
    except _BailOut:
        return None
    if cross_match is not None:
        cross, operands = cross_match
        resolved: list[tuple[set[str], set[int]]] = []
        for operand in operands:
            operand_result = _resolve_mean(table_name, operand.name, manifest, visited)
            if operand_result is None:
                return None
            resolved.append(operand_result)
        variables: set[str] = set().union(*(r[0] for r in resolved))
        refs: set[int] = set(cross.fact_references).union(*(r[1] for r in resolved))
        return variables, refs

    base_var = _base_mean_variable(table_name, column_name, manifest)
    if base_var is None:
        return None
    return {base_var}, set()


def _resolve_aggregation(
    aggregation: AggregationConstraint,
    manifest: ConstraintManifest,
    visited: frozenset[tuple[str, str]],
) -> tuple[set[str], set[int]] | None:
    """Wald's identity for SUM (E[parent] = E[N] * E[descendant], N from the
    matching fanout); AVG needs no fanout at all (the sample mean already
    estimates the population mean); MAX/MIN have no general closed form --
    bail, per section 4.4."""
    if aggregation.operation in ("MAX", "MIN"):
        return None

    descendant = _resolve_mean(
        aggregation.descendant_table, aggregation.descendant_column, manifest, visited
    )
    if descendant is None:
        return None
    descendant_vars, descendant_refs = descendant
    refs = descendant_refs | set(aggregation.fact_references)

    if aggregation.operation == "AVG":
        return descendant_vars, refs

    matching_fanouts = [
        fanout
        for fanout in manifest.structural.fanouts
        if fanout.parent_table == aggregation.parent_table
        and fanout.child_table == aggregation.descendant_table
    ]
    if len(matching_fanouts) != 1:
        return None
    fanout = matching_fanouts[0]
    fk_suffix = "_".join(fanout.foreign_key_columns)
    fanout_var = f"{fanout.parent_table}->{fanout.child_table}.fanout_mean[{fk_suffix}]"
    return descendant_vars | {fanout_var}, refs | set(fanout.fact_references)


def moment_target_to_graph_nodes(
    target: MomentTarget, manifest: ConstraintManifest, disambiguator: int
) -> tuple[list[Variable], list[Constraint]] | None:
    """Resolve one MomentTarget into graph nodes by walking its derivation
    chain (section 4.2). Returns None if the walk bails out anywhere --
    the target stays unresolved for this pass (section 4.4), not partial."""
    resolved = _resolve_mean(
        target.table_name, target.column_name, manifest, frozenset()
    )
    if resolved is None:
        return None
    variable_names, refs = resolved
    refs = sorted(refs | set(target.fact_references))
    variables = [
        Variable(name=name, fact_references=refs) for name in sorted(variable_names)
    ]
    constraint = Constraint(
        name=f"moment_target_{target.table_name}.{target.column_name}#{disambiguator}",
        variables=sorted(variable_names),
        fact_references=refs,
    )
    return variables, [constraint]


def constraint_manifest_to_graph_nodes(
    manifest: ConstraintManifest,
) -> tuple[list[Variable], list[Constraint]]:
    """Combine everything this module currently knows how to graph:
    distributions, moment targets (via the section 4.2 derivation walk),
    and cardinalities/fanouts. Explicitly NOT included: uniqueness and
    formats (not DOF concepts), and conditional cross-column logic (Q4,
    if_condition is set). A MomentTarget the derivation walk can't resolve
    is logged and dropped rather than raising -- it's a known, enumerated
    boundary (section 4.4), not an error."""
    stat_variables, stat_constraints = statistical_manifest_to_graph_nodes(
        manifest.statistical
    )
    struct_variables, struct_constraints = structural_manifest_to_graph_nodes(
        manifest.structural
    )

    variables_by_name = {v.name: v for v in stat_variables}
    constraints = list(stat_constraints)
    _accumulate(variables_by_name, constraints, struct_variables, struct_constraints)

    for index, target in enumerate(manifest.statistical.moment_targets):
        resolved = moment_target_to_graph_nodes(target, manifest, disambiguator=index)
        if resolved is None:
            print(
                f"[Stage3] moment target unresolved (bailed derivation walk): "
                f"{target.table_name}.{target.column_name}"
            )
            continue
        new_variables, new_constraints = resolved
        _accumulate(variables_by_name, constraints, new_variables, new_constraints)

    return list(variables_by_name.values()), constraints
