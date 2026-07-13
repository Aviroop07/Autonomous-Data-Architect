"""Converts extracted Stage 3 constraints into the generic DOF graph
(src/util/algorithms/dof_graph.py).

Two pathways, kept side by side because they cover genuinely
non-overlapping fact families -- this is NOT two versions of the same
thing:

1. `analyze_constraint_manifest` (ConstraintManifest -> DOF graph): the
   ONLY pathway that resolves Q3 MomentTarget facts (the derivation-chain
   walk in section 4 of STAGE3_PHASE2_DESIGN.md, `_resolve_mean`/
   `_resolve_aggregation`) and handles Q4's fork-key conditional
   expansion for CrossColumnLogic. `cross_shard.py` (the shape the new
   extraction agents emit) has no MomentTarget/AggregationConstraint/
   ColumnCorrelation equivalent at all yet -- extending it to express
   Q3's derivation chain and D7 correlation over real Grain-scoped
   quantities is real, separate design work, not a quick migration, and
   is an explicit open follow-up, not silently pretended-solved.
   Explicitly NOT handled here: UniqueConstraint, FormatConstraint, and
   ColumnCorrelation (D7) -- none has a numeric parameter to pin, so none
   is a DOF concept at all; correlation is a joint-distribution shape
   parameter, not a variable/equation, and flows straight to Stage 4
   generation instead (see constraints.py's ColumnCorrelation docstring).
2. `analyze_cross_shard_constraints` (cross_shard.py Constraint/
   DistributionConstraint/DerivedColumnConstraint -> DOF graph): the LIVE
   path for everything the 3 real extraction agents
   (statistical/structural/logic_extractor) actually produce today --
   distributions, cardinalities, ranges, derived columns -- routed
   through Grain canonicalization so ON-scope comparability is handled
   correctly (see grain.py).

Both call the same real DOFGraph/Dulmage-Mendelsohn classifier
(src/util/algorithms/dof_graph.py) -- neither reimplements it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import sqlglot
from sqlglot import expressions as exp

from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage3.middleware.fork_registry import (
    ForkKey,
    ForkKeyRegistry,
    Unresolved,
    parse_if_condition,
)
from src.pipeline.stage3.models import cross_shard
from src.pipeline.stage3.models.condition_nodes import (
    RColumnRef,
    RComparison,
    RLiteral,
    RPredicate,
)
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
from src.pipeline.stage3.models.grain import (
    CanonicalizationFailure,
    Grain,
    _SchemaView,
    canonicalize,
)
from src.pipeline.stage3.models.on_nodes import ONAggregate, ONBaseTable, ONNode
from src.pipeline.stage3.models.probe import (
    MomentTargetProbe,
    Stage3AnalysisReport,
    VariableProbe,
)
from src.util.algorithms.dof_graph import (
    Constraint,
    Constraint as DOFConstraint,
    DOFClassification,
    DOFGraph,
    OverconstrainedBlock,
    Variable,
    Variable as DOFVariable,
)

logger = logging.getLogger(__name__)


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
    dist: DistributionConstraint,
    disambiguator: int,
    branches: list[str] | None = None,
    fork_key_str: str | None = None,
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

    suffixes = []
    if branches and fork_key_str:
        suffixes = [f"|{fork_key_str}={b}" for b in branches]
    else:
        suffixes = [""]

    if isinstance(dist, CategoricalDistribution):
        variables = []
        constraints = []
        for suffix in suffixes:
            var_name = f"{qualified_column}.probabilities{suffix}"
            variable = Variable(name=var_name, fact_references=refs)
            variables.append(variable)
            if dist.probabilities is not None:
                constraint = Constraint(
                    name=f"pin_{var_name}#{disambiguator}",
                    variables=[var_name],
                    fact_references=refs,
                )
                constraints.append(constraint)
        return variables, constraints

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
        for suffix in suffixes:
            var_name = f"{qualified_column}.{param}{suffix}"
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
    manifest: StatisticalManifest, registry: ForkKeyRegistry | None = None
) -> tuple[list[Variable], list[Constraint]]:
    """Flatten a whole StatisticalManifest into deduplicated Variables."""
    variables_by_name: dict[str, Variable] = {}
    constraints: list[Constraint] = []

    for index, dist in enumerate(manifest.distributions):
        branches = None
        fork_key_str = None
        if registry and getattr(dist, "if_condition", None):
            cond = parse_if_condition(dist.if_condition)
            if cond:
                resolved = registry.get_branches_for_condition(cond)
                if not isinstance(resolved, Unresolved):
                    branches = resolved
                    fork_key_str = cond.fork_key.to_string()

        new_variables, new_constraints = distribution_to_graph_nodes(
            dist, disambiguator=index, branches=branches, fork_key_str=fork_key_str
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
    parameter to pin (not a DOF concept, permanently out of scope here).
    AggregationConstraint is deliberately absent too, but not because it's
    unhandled: it's consumed by Q3's moment-target derivation walk
    (moment_target_to_graph_nodes) rather than turned into its own graph
    nodes here."""
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
    cleanly with any distribution fact that might independently pin it.

    Conditional (Q4-forked, `if_condition` set) distributions are
    deliberately skipped here -- an unconditional E[column] isn't any single
    branch's mean, and this pass has no rule for combining branch means
    weighted by branch probability into one population mean. A column whose
    ONLY distribution facts are conditional therefore bails (section 4.4
    discipline: unresolved, not guessed) rather than silently picking
    whichever branch happened to be listed first."""
    qualified = f"{table_name}.{column_name}"
    matches = [
        dist
        for dist in manifest.statistical.distributions
        if dist.table_name == table_name and dist.column_name == column_name
    ]
    if not matches:
        return f"{qualified}.mean"

    unconditional = [d for d in matches if getattr(d, "if_condition", None) is None]
    if not unconditional:
        return None

    dist = unconditional[0]
    if isinstance(dist, (GaussianDistribution, LogNormalDistribution)):
        return f"{qualified}.mean"
    if isinstance(dist, PoissonDistribution):
        return f"{qualified}.lam"
    return None


def _find_all_unconditional_cross_columns(
    table_name: str, column_name: str, manifest: ConstraintManifest
) -> list[tuple[CrossColumnLogic, tuple[exp.Column, exp.Column]]]:
    """Find ALL unconditional CrossColumnLogic facts defining column_name
    as a product or sum of exactly two base columns. Returns a list of
    matches (possibly empty, possibly >1). Raises _BailOut if ANY matching
    fact has an unsupported shape."""
    matches: list[tuple[CrossColumnLogic, tuple[exp.Column, exp.Column]]] = []
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
        matches.append((cross, operands))
    return matches


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
        cross_matches = _find_all_unconditional_cross_columns(
            table_name, column_name, manifest
        )
    except _BailOut:
        return None
    if cross_matches:
        if len(cross_matches) > 1:
            # Multiple unconditional derivations for the same column -- check
            # if they agree. If they disagree (different operands/formulas),
            # this is a confirmed contradiction flagged for LLM reconciliation
            # (ISSUES.md #11 fix: never silently pick first-match-wins).
            first_operands = frozenset((op.name for op in cross_matches[0][1]))
            for _, other_operands in cross_matches[1:]:
                other_frozenset = frozenset((op.name for op in other_operands))
                if other_frozenset != first_operands:
                    raise _BailOut
        cross, operands = cross_matches[0]
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

    from collections import deque

    paths = []
    queue = deque([(aggregation.parent_table, [])])
    while queue:
        current, path = queue.popleft()
        if current == aggregation.descendant_table:
            paths.append(path)
            continue
        for fanout in manifest.structural.fanouts:
            if fanout.parent_table == current:
                if any(f.child_table == fanout.child_table for f in path):
                    continue
                queue.append((fanout.child_table, path + [fanout]))

    if len(paths) != 1:
        return None

    path = paths[0]
    fanout_vars = set()
    fanout_refs = set()
    for fanout in path:
        fk_suffix = "_".join(fanout.foreign_key_columns)
        var_name = (
            f"{fanout.parent_table}->{fanout.child_table}.fanout_mean[{fk_suffix}]"
        )
        fanout_vars.add(var_name)
        fanout_refs.update(fanout.fact_references)

    return descendant_vars | fanout_vars, refs | fanout_refs


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
) -> tuple[list[Variable], list[Constraint], list[MomentTarget]]:
    """Combine everything this module currently knows how to graph. The
    third return value is every MomentTarget whose derivation walk bailed
    -- Stage 3's job stops at reporting these (see analyze_constraint_manifest),
    never at guessing or calibrating a value for them."""
    registry = ForkKeyRegistry()

    # Discovery Pass helper -- union from EVERY matching fact, no first-match-wins
    def _discover_fork(if_condition: str | None):
        if not if_condition:
            return
        cond = parse_if_condition(if_condition)
        if not cond:
            return
        for cdist in manifest.statistical.distributions:
            if (
                isinstance(cdist, CategoricalDistribution)
                and cdist.table_name == cond.fork_key.table_name
                and cdist.column_name == cond.fork_key.column_name
            ):
                registry.register_fork(cond.fork_key, cdist.categories)

    for dist in manifest.statistical.distributions:
        _discover_fork(getattr(dist, "if_condition", None))
    for cross in manifest.logic.cross_column_logic:
        _discover_fork(cross.if_condition)

    stat_variables, stat_constraints = statistical_manifest_to_graph_nodes(
        manifest.statistical, registry
    )
    struct_variables, struct_constraints = structural_manifest_to_graph_nodes(
        manifest.structural
    )

    variables_by_name = {v.name: v for v in stat_variables}
    constraints = list(stat_constraints)
    _accumulate(variables_by_name, constraints, struct_variables, struct_constraints)

    # Handle conditional CrossColumnLogic
    for index, cross in enumerate(manifest.logic.cross_column_logic):
        if cross.if_condition:
            cond = parse_if_condition(cross.if_condition)
            if not cond:
                continue
            resolved = registry.get_branches_for_condition(cond)
            if isinstance(resolved, Unresolved):
                continue
            branches = resolved
            fork_key_str = cond.fork_key.to_string()

            try:
                parsed = sqlglot.parse_one(cross.then_enforcement)
                if not isinstance(parsed, exp.EQ) or not isinstance(
                    parsed.this, exp.Column
                ):
                    print(
                        f"[Stage3] conditional cross-logic wrong shape: {cross.then_enforcement}"
                    )
                    continue
                target_col = parsed.this.name
            except Exception:
                print(
                    f"[Stage3] conditional cross-logic unparseable: {cross.then_enforcement}"
                )
                continue

            for branch in branches:
                var_name = (
                    f"{cross.table_context}.{target_col}.mean|{fork_key_str}={branch}"
                )
                variable = Variable(
                    name=var_name, fact_references=cross.fact_references
                )
                constraint = Constraint(
                    name=f"pin_cross_{var_name}#{index}",
                    variables=[var_name],
                    fact_references=cross.fact_references,
                )
                _accumulate(variables_by_name, constraints, [variable], [constraint])

    unresolved_moment_targets: list[MomentTarget] = []
    for index, target in enumerate(manifest.statistical.moment_targets):
        resolved = moment_target_to_graph_nodes(target, manifest, disambiguator=index)
        if resolved is None:
            print(
                f"[Stage3] moment target unresolved (bailed derivation walk): "
                f"{target.table_name}.{target.column_name}"
            )
            unresolved_moment_targets.append(target)
            continue
        new_variables, new_constraints = resolved
        _accumulate(variables_by_name, constraints, new_variables, new_constraints)

    return list(variables_by_name.values()), constraints, unresolved_moment_targets


def analyze_constraint_manifest(manifest: ConstraintManifest) -> Stage3AnalysisReport:
    """Stage 3's complete output for a ConstraintManifest: what's
    determined (square), what's genuinely free (loose -> VariableProbe),
    what's contradictory (overconstrained_blocks), and which MomentTargets
    couldn't be resolved (-> MomentTargetProbe). This is the boundary of
    Stage 3's job -- it reports, it never fills a probe in itself (see
    the project memory stage3_stage4_division_of_labor)."""
    variables, constraints, unresolved_targets = constraint_manifest_to_graph_nodes(
        manifest
    )
    classification = DOFGraph(variables, constraints).classify()

    variables_by_name = {v.name: v for v in variables}
    loose_probes = [
        VariableProbe(
            variable_name=name,
            lower_bound=variables_by_name[name].lower_bound,
            upper_bound=variables_by_name[name].upper_bound,
            fact_references=variables_by_name[name].fact_references,
        )
        for name in classification.loose_variables
    ]
    moment_target_probes = [
        MomentTargetProbe(
            table_name=target.table_name,
            column_name=target.column_name,
            statistic=target.statistic,
            target_value=target.target_value,
            fact_references=target.fact_references,
        )
        for target in unresolved_targets
    ]

    return Stage3AnalysisReport(
        square_variables=classification.square_variables,
        loose_variable_probes=loose_probes,
        unresolved_moment_target_probes=moment_target_probes,
        overconstrained_blocks=classification.overconstrained_blocks,
    )


# =============================================================================
# cross_shard.py pathway -- the live path for the 3 real extraction agents
# =============================================================================


# ---------------------------------------------------------------------------
# Rich variable model (adapted from experiments/stage3_conflict_v2/dof_engine.py)
# ---------------------------------------------------------------------------


class VariableKind(str, Enum):
    COLUMN_DISTRIBUTION_PARAM = "column_distribution_param"
    COLUMN_RANGE = "column_range"
    TABLE_CARDINALITY = "table_cardinality"
    DERIVED_COLUMN = "derived_column"
    CORRELATION = "correlation"


@dataclass(frozen=True)
class BranchTag:
    """A Q4-style conditional fork: this variable applies only within this
    branch. Variables in different, mutually-exclusive branches of the SAME
    fork key must never be unified."""

    fork_table: str
    fork_column: str
    branch_value: str


@dataclass(frozen=True)
class RichVariable:
    grain: Grain
    kind: VariableKind
    name: str
    branch: Optional[BranchTag] = None
    fact_references: Tuple[int, ...] = ()
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    # Only set for CATEGORICAL distribution facts -- the stated category
    # set, compared via set-overlap rather than interval-overlap (see
    # pins_conflict below).
    categories: Optional[frozenset] = None

    def flat_name(self) -> str:
        """Collapse to the flat string identity DOFGraph operates on.
        Grain-scoped and branch-scoped so two variables with the same short
        name but different grain/branch are never accidentally unified."""
        grain_id = (
            f"{self.grain.table}"
            f"[{sorted((fk.child_table, fk.fk_column, fk.parent_table, occ) for fk, occ in self.grain.edges)}]"
        )
        narrowed_id = "|narrowed" if self.grain.narrowed else ""
        agg_id = f"|agg={self.grain.agg_signature}" if self.grain.agg_signature else ""
        branch_id = (
            f"|{self.branch.fork_table}.{self.branch.fork_column}={self.branch.branch_value}"
            if self.branch
            else ""
        )
        return f"{grain_id}{narrowed_id}{agg_id}::{self.kind.value}::{self.name}{branch_id}"


@dataclass(frozen=True)
class RichConstraint:
    name: str
    variables: Tuple[RichVariable, ...]
    fact_references: Tuple[int, ...] = ()


@dataclass
class RichClassification:
    square: List[RichVariable] = field(default_factory=list)
    loose: List[RichVariable] = field(default_factory=list)
    overconstrained_blocks: List[Tuple[List[RichVariable], List[str]]] = field(
        default_factory=list
    )
    # Flat names where merging two or more RichVariables under the SAME
    # identity produced a provable contradiction (an empty bound interval,
    # or disjoint category sets) -- a genuine value-level conflict, not
    # just a structural DOF degree-count. Distinguished from
    # overconstrained_blocks (which DOF flags purely from constraint-to-
    # variable ratio and cannot tell "two facts stating the same value" --
    # harmless redundancy -- apart from "two facts stating incompatible
    # values" -- a real bug).
    confirmed_conflicts: List[str] = field(default_factory=list)


def _merge_rich_bounds(
    a: RichVariable, b: RichVariable
) -> Tuple[Optional[float], Optional[float], Optional[frozenset], bool]:
    """Merge two RichVariables sharing a flat_name into one bound/category
    pair, and report whether the merge is genuinely valid (non-empty).

    Interval merge is a plain max-lower/min-upper intersection (same rule
    the pre-existing ConstraintManifest pathway's _merge_variable already
    uses, and the same rule the multi-shard range-tightening behavior in
    test_stage3_constraint_graph.py's
    test_two_range_facts_about_the_same_table_tighten_the_bound depends
    on) -- this function is what actually performs that merge for the
    cross_shard.py pathway; without it, build_and_classify previously kept
    only the FIRST-seen RichVariable's bounds and silently discarded every
    other fact's bounds entirely.

    Category sets merge via intersection too -- if a CATEGORICAL fact
    states {Bronze, Silver} and another states {Gold, Platinum} for the
    exact same variable, that IS a genuine contradiction (disjoint), not
    resolvable by picking either side.

    Returns (lower, upper, categories, is_valid). is_valid=False means a
    genuine value-level conflict was found (empty interval or disjoint
    categories) -- callers must not treat the returned bounds as usable."""
    if a.categories is not None and b.categories is not None:
        merged_categories = a.categories & b.categories
        return None, None, merged_categories, bool(merged_categories)

    lower_candidates = [x for x in (a.lower_bound, b.lower_bound) if x is not None]
    upper_candidates = [x for x in (a.upper_bound, b.upper_bound) if x is not None]
    lower = max(lower_candidates) if lower_candidates else None
    upper = min(upper_candidates) if upper_candidates else None
    is_valid = lower is None or upper is None or lower <= upper
    return lower, upper, None, is_valid


# ---------------------------------------------------------------------------
# Build and classify (bridge to real DOFGraph)
# ---------------------------------------------------------------------------


def build_and_classify(
    variables: List[RichVariable], constraints: List[RichConstraint]
) -> RichClassification:
    """Deduplicate RichVariables by flat_name (properly MERGING bounds/
    categories across every fact sharing that identity, not just keeping
    whichever RichVariable happened to be seen first), build the bipartite
    graph, classify via Dulmage-Mendelsohn, map results back to rich
    objects. Variables whose merge is genuinely invalid (an empty bound
    interval or disjoint category sets) are pulled out as confirmed_conflicts
    BEFORE reaching DOFGraph -- a provable value contradiction is real
    infeasibility regardless of what the structural degree-count would say."""
    by_flat: Dict[str, RichVariable] = {}
    confirmed_conflicts: List[str] = []
    for v in variables:
        existing = by_flat.get(v.flat_name())
        if existing is None:
            by_flat[v.flat_name()] = v
            continue
        lower, upper, categories, is_valid = _merge_rich_bounds(existing, v)
        if not is_valid:
            if v.flat_name() not in confirmed_conflicts:
                confirmed_conflicts.append(v.flat_name())
        merged_refs = tuple(
            sorted(set(existing.fact_references) | set(v.fact_references))
        )
        by_flat[v.flat_name()] = RichVariable(
            grain=existing.grain,
            kind=existing.kind,
            name=existing.name,
            branch=existing.branch,
            fact_references=merged_refs,
            lower_bound=lower,
            upper_bound=upper,
            categories=categories,
        )
    for c in constraints:
        for v in c.variables:
            by_flat.setdefault(v.flat_name(), v)

    raw_vars = [
        DOFVariable(
            name=flat,
            lower_bound=rv.lower_bound if flat not in confirmed_conflicts else None,
            upper_bound=rv.upper_bound if flat not in confirmed_conflicts else None,
        )
        for flat, rv in by_flat.items()
    ]
    raw_constraints = [
        DOFConstraint(
            name=c.name,
            variables=sorted({v.flat_name() for v in c.variables}),
            fact_references=list(c.fact_references),
        )
        for c in constraints
    ]

    graph = DOFGraph(raw_vars, raw_constraints)
    result: DOFClassification = graph.classify()

    out = RichClassification()
    out.confirmed_conflicts = confirmed_conflicts
    out.square = [
        by_flat[n] for n in result.square_variables if n not in confirmed_conflicts
    ]
    out.loose = [
        by_flat[n] for n in result.loose_variables if n not in confirmed_conflicts
    ]
    for block in result.overconstrained_blocks:
        out.overconstrained_blocks.append(
            ([by_flat[n] for n in block.variables if n in by_flat], block.constraints)
        )
    return out


# ---------------------------------------------------------------------------
# Fork/branch resolution
# ---------------------------------------------------------------------------


def _resolve_branch(
    condition: Optional[RPredicate],
    registry: ForkKeyRegistry,
) -> Optional[BranchTag]:
    """If condition is a simple EQ/IN against a known categorical fork,
    resolve it to a BranchTag. Returns None for non-fork conditions or
    unresolved forks."""
    if condition is None:
        return None
    if not isinstance(condition, RComparison):
        return None
    if condition.op not in ("=", "==", "in"):
        return None
    if not isinstance(condition.left, RColumnRef):
        return None
    if not isinstance(condition.right, RLiteral):
        return None

    col_name = condition.left.name
    val = condition.right.value
    if not isinstance(val, str):
        return None

    # Try to find the fork key across all tables (we don't know the table
    # from the RColumnRef alone -- it's deliberately unqualified).
    # Check all registered fork keys for a matching column.
    for fk, _cats in registry.forks.items():
        if fk.column_name == col_name:
            branch_vals = registry.get_branches_for_condition(
                __import__(
                    "src.pipeline.stage3.middleware.fork_registry",
                    fromlist=["BranchCondition"],
                ).BranchCondition(
                    fork_key=fk,
                    operator="EQ",
                    values=[val],
                )
            )
            if isinstance(branch_vals, Unresolved):
                return None
            return BranchTag(
                fork_table=fk.table_name,
                fork_column=fk.column_name,
                branch_value=val,
            )
    return None


# ---------------------------------------------------------------------------
# Conversion: cross_shard.py shapes -> RichVariable + RichConstraint
# ---------------------------------------------------------------------------


def _on_base_table(node: ONNode) -> Optional[str]:
    """Extract the base table name from an ON tree."""
    if isinstance(node, ONBaseTable):
        return node.name
    if isinstance(node, ONAggregate):
        return _on_base_table(node.source)
    return None


def _distribution_to_rich(
    dc: cross_shard.DistributionConstraint,
    grain: Grain,
    view: _SchemaView,
    registry: ForkKeyRegistry,
    disambiguator: int,
) -> Tuple[List[RichVariable], List[RichConstraint]]:
    """Convert one DistributionConstraint into RichVariable/RichConstraint
    pairs -- one Constraint PER parameter, never one touching several."""
    table = grain.table
    col = dc.column
    refs = tuple(dc.fact_references)
    branch = _resolve_branch(dc.if_condition, registry)

    family = dc.family
    if family in ("GAUSSIAN", "LOG_NORMAL"):
        params = ["mean", "std_dev"]
    elif family == "BETA":
        params = ["alpha", "beta"]
    elif family == "POISSON":
        params = ["lam"]
    elif family == "UNIFORM":
        params = ["min_value", "max_value"]
    elif family == "CATEGORICAL":
        # Categorical: one variable for probabilities. The stated category
        # NAMES (not the probabilities) are the variable's pin content for
        # value-conflict comparison -- two facts naming disjoint category
        # sets for the same column is a genuine contradiction (see
        # _merge_rich_bounds), independent of whether either states weights.
        variables = []
        constraints = []
        var_name = f"{col}.probabilities"
        categories = dc.parameters.get("categories", [])
        rv = RichVariable(
            grain=grain,
            kind=VariableKind.COLUMN_DISTRIBUTION_PARAM,
            name=var_name,
            branch=branch,
            fact_references=refs,
            categories=frozenset(categories) if isinstance(categories, list) else None,
        )
        variables.append(rv)
        probs = dc.parameters.get("probabilities")
        if probs is not None:
            constraints.append(
                RichConstraint(
                    name=f"pin_{table}.{var_name}#{disambiguator}",
                    variables=(rv,),
                    fact_references=refs,
                )
            )
        return variables, constraints
    else:
        logger.warning("Unknown distribution family: %s", family)
        return [], []

    variables = []
    constraints = []
    for param in params:
        var_name = f"{col}.{param}"
        # An exact value pin: lower_bound == upper_bound == the stated
        # value. Lets two facts agreeing on this exact value merge cleanly
        # (see _merge_rich_bounds) instead of the value being invisible to
        # the graph entirely -- previously nothing recorded WHAT value a
        # distribution parameter was pinned to, only that it was pinned.
        raw_value = dc.parameters.get(param)
        value = float(raw_value) if isinstance(raw_value, (int, float)) else None
        rv = RichVariable(
            grain=grain,
            kind=VariableKind.COLUMN_DISTRIBUTION_PARAM,
            name=var_name,
            branch=branch,
            fact_references=refs,
            lower_bound=value,
            upper_bound=value,
        )
        variables.append(rv)
        # Pinning constraint: one per parameter
        constraints.append(
            RichConstraint(
                name=f"pin_{table}.{var_name}#{disambiguator}",
                variables=(rv,),
                fact_references=refs,
            )
        )
    return variables, constraints


def _range_constraint_to_rich(
    c: cross_shard.Constraint,
    grain: Grain,
    view: _SchemaView,
    registry: ForkKeyRegistry,
    disambiguator: int,
) -> Tuple[List[RichVariable], List[RichConstraint]]:
    """Convert a range/bounds constraint (e.g. ORDER.total >= 5) into a
    RichVariable with bound metadata. No pinning constraint -- bounds are
    domain metadata, not DOF-consuming equations."""
    cols = _extract_columns_from_condition(c.condition)
    if len(cols) != 1:
        return [], []
    col = next(iter(cols))
    refs = tuple(c.fact_references)
    branch = _resolve_branch(c.condition, registry)

    lower_bound = None
    upper_bound = None
    if isinstance(c.condition, RComparison) and isinstance(c.condition.right, RLiteral):
        val = c.condition.right.value
        if isinstance(val, (int, float)):
            val = float(val)
            if c.condition.op in (">=", ">"):
                lower_bound = val if c.condition.op == ">=" else val + 1e-9
            elif c.condition.op in ("<=", "<"):
                upper_bound = val if c.condition.op == "<=" else val - 1e-9
            elif c.condition.op in ("=", "=="):
                lower_bound = val
                upper_bound = val

    rv = RichVariable(
        grain=grain,
        kind=VariableKind.COLUMN_RANGE,
        name=col,
        branch=branch,
        fact_references=refs,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    return [rv], []


def _cardinality_to_rich(
    c: cross_shard.Constraint,
    grain: Grain,
    disambiguator: int,
) -> Tuple[List[RichVariable], List[RichConstraint]]:
    """Convert a table cardinality constraint into a RichVariable."""
    table = grain.table
    refs = tuple(c.fact_references)

    lower_bound = None
    upper_bound = None
    pinned = False
    if isinstance(c.condition, RComparison) and isinstance(c.condition.right, RLiteral):
        val = c.condition.right.value
        if isinstance(val, (int, float)):
            val = float(val)
            if c.condition.op in (">=", ">"):
                lower_bound = val if c.condition.op == ">=" else val + 1e-9
            elif c.condition.op in ("<=", "<"):
                upper_bound = val if c.condition.op == "<=" else val - 1e-9
            elif c.condition.op in ("=", "=="):
                lower_bound = val
                upper_bound = val
                pinned = True

    rv = RichVariable(
        grain=grain,
        kind=VariableKind.TABLE_CARDINALITY,
        name="row_count",
        fact_references=refs,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    constraints = []
    if pinned:
        constraints.append(
            RichConstraint(
                name=f"pin_{table}.row_count#{disambiguator}",
                variables=(rv,),
                fact_references=refs,
            )
        )
    return [rv], constraints


def _derived_column_to_rich(
    dc: cross_shard.DerivedColumnConstraint,
    grain: Grain,
    disambiguator: int,
) -> Tuple[List[RichVariable], List[RichConstraint]]:
    """Convert a DerivedColumnConstraint into RichVariable + pinning
    constraint. The variable is the derived column itself; the constraint
    pins it to its formula."""
    refs = tuple(dc.fact_references)
    rv = RichVariable(
        grain=grain,
        kind=VariableKind.DERIVED_COLUMN,
        name=dc.target_column,
        fact_references=refs,
    )
    constraint = RichConstraint(
        name=f"derived_{dc.target_table}.{dc.target_column}#{disambiguator}",
        variables=(rv,),
        fact_references=refs,
    )
    return [rv], [constraint]


def _extract_columns_from_condition(pred: RPredicate) -> set[str]:
    """Extract column names from an R-predicate tree."""
    from src.pipeline.stage3.middleware.conflict_detection import extract_columns

    return extract_columns(pred)


# ---------------------------------------------------------------------------
# Main conversion: cross_shard.py -> RichVariable + RichConstraint lists
# ---------------------------------------------------------------------------


def _convert_cross_shard_constraints(
    distributions: List[cross_shard.DistributionConstraint],
    structural: List[cross_shard.Constraint],
    logic: List[cross_shard.Constraint],
    derived: List[cross_shard.DerivedColumnConstraint],
    schema: Schema,
    registry: ForkKeyRegistry,
) -> Tuple[List[RichVariable], List[RichConstraint]]:
    """Convert all cross_shard.py extraction outputs into RichVariable +
    RichConstraint lists, ready for build_and_classify."""
    view = _SchemaView.from_schema(schema)
    all_vars: List[RichVariable] = []
    all_cons: List[RichConstraint] = []

    # 1. Distribution constraints
    for i, dc in enumerate(distributions):
        grain_result = canonicalize(dc.on, schema)
        if isinstance(grain_result, CanonicalizationFailure):
            logger.warning(
                "Cannot canonicalize distribution ON tree: %s",
                grain_result.reason,
            )
            continue
        vars, cons = _distribution_to_rich(
            dc, grain_result, view, registry, disambiguator=i
        )
        all_vars.extend(vars)
        all_cons.extend(cons)

    # 2. Structural constraints (cardinality + range)
    for i, c in enumerate(structural):
        grain_result = canonicalize(c.on, schema)
        if isinstance(grain_result, CanonicalizationFailure):
            logger.warning(
                "Cannot canonicalize structural ON tree: %s",
                grain_result.reason,
            )
            continue

        on_tables = set()
        from src.pipeline.stage3.models.on_nodes import extract_tables

        on_tables = extract_tables(c.on)

        if len(on_tables) == 1:
            # Single-table constraint -> cardinality
            vars, cons = _cardinality_to_rich(c, grain_result, disambiguator=i)
        else:
            # Multi-table / range constraint
            vars, cons = _range_constraint_to_rich(
                c, grain_result, view, registry, disambiguator=i
            )
        all_vars.extend(vars)
        all_cons.extend(cons)

    # 3. Logic constraints (range / cross-column / format)
    for i, c in enumerate(logic):
        grain_result = canonicalize(c.on, schema)
        if isinstance(grain_result, CanonicalizationFailure):
            logger.warning(
                "Cannot canonicalize logic ON tree: %s",
                grain_result.reason,
            )
            continue
        vars, cons = _range_constraint_to_rich(
            c, grain_result, view, registry, disambiguator=i
        )
        all_vars.extend(vars)
        all_cons.extend(cons)

    # 4. Derived column constraints
    for i, dc in enumerate(derived):
        # Build a synthetic ON tree for the target table
        on = ONBaseTable(name=dc.target_table)
        grain_result = canonicalize(on, schema)
        if isinstance(grain_result, CanonicalizationFailure):
            logger.warning(
                "Cannot canonicalize derived column ON tree: %s",
                grain_result.reason,
            )
            continue
        vars, cons = _derived_column_to_rich(dc, grain_result, disambiguator=i)
        all_vars.extend(vars)
        all_cons.extend(cons)

    return all_vars, all_cons


# ---------------------------------------------------------------------------
# New entry point: analyze cross_shard.py shapes
# ---------------------------------------------------------------------------


def analyze_cross_shard_constraints(
    distributions: List[cross_shard.DistributionConstraint] | None = None,
    structural: List[cross_shard.Constraint] | None = None,
    logic: List[cross_shard.Constraint] | None = None,
    derived: List[cross_shard.DerivedColumnConstraint] | None = None,
    schema: Schema | None = None,
    registry: ForkKeyRegistry | None = None,
) -> Stage3AnalysisReport:
    """Stage 3's complete DOF analysis for cross_shard.py-shaped extraction
    outputs. This is the NEW entry point that consumes the real extraction
    agent output shapes (on: ONNode, condition: RPredicate) and routes
    through grain canonicalization + real DOFGraph.

    Returns a Stage3AnalysisReport with:
    - square_variables: determined parameters (pinned by facts)
    - loose_variable_probes: free parameters for Stage 4
    - overconstrained_blocks: genuine contradictions flagged for review
    """
    distributions = distributions or []
    structural = structural or []
    logic = logic or []
    derived = derived or []

    if schema is None:
        logger.warning("No schema provided; returning empty analysis.")
        return Stage3AnalysisReport()

    if registry is None:
        registry = ForkKeyRegistry()

    # Build fork registry: scan all categorical distributions for fork keys
    _build_fork_registry(distributions, registry)

    all_vars, all_cons = _convert_cross_shard_constraints(
        distributions, structural, logic, derived, schema, registry
    )

    if not all_vars:
        return Stage3AnalysisReport()

    classification = build_and_classify(all_vars, all_cons)

    # Build probes from classification results
    loose_probes = [
        VariableProbe(
            variable_name=v.flat_name(),
            lower_bound=v.lower_bound,
            upper_bound=v.upper_bound,
            fact_references=list(v.fact_references),
        )
        for v in classification.loose
    ]

    confirmed_conflict_set = set(classification.confirmed_conflicts)
    overconstrained = []
    for block_vars, block_cons in classification.overconstrained_blocks:
        # Variables already reported via confirmed_conflicts (below) are
        # excluded here to avoid double-listing the same flat_name under
        # both mechanisms.
        remaining = [
            v for v in block_vars if v.flat_name() not in confirmed_conflict_set
        ]
        if remaining:
            overconstrained.append(
                OverconstrainedBlock(
                    variables=[v.flat_name() for v in remaining],
                    constraints=block_cons,
                )
            )
    # Genuine value-level contradictions (empty merged interval, disjoint
    # category sets) -- these bypass DOF's structural degree-count entirely
    # (a provable value conflict is real infeasibility regardless of
    # constraint-to-variable ratio) and would otherwise never surface, since
    # a variable pinned by exactly one constraint per fact is structurally
    # square/loose to DOF even when the facts disagree.
    for flat_name in classification.confirmed_conflicts:
        overconstrained.append(
            OverconstrainedBlock(variables=[flat_name], constraints=[])
        )

    return Stage3AnalysisReport(
        square_variables=[v.flat_name() for v in classification.square],
        loose_variable_probes=loose_probes,
        unresolved_moment_target_probes=[],
        overconstrained_blocks=overconstrained,
    )


def _build_fork_registry(
    distributions: List[cross_shard.DistributionConstraint],
    registry: ForkKeyRegistry,
) -> None:
    """Scan categorical distributions to populate the fork registry with
    all known category lists. This replaces the old _discover_fork pattern
    with a proper scan-and-union over cross_shard.py shapes."""
    for dc in distributions:
        if dc.family != "CATEGORICAL":
            continue
        if dc.if_condition is None:
            continue
        cond = parse_if_condition_from_predicate(dc.if_condition)
        if cond is None:
            continue
        cats = dc.parameters.get("categories", [])
        if isinstance(cats, list) and cats:
            registry.register_fork(cond.fork_key, cats)


def parse_if_condition_from_predicate(
    pred: RPredicate,
) -> Optional:
    """Parse an R-predicate into a BranchCondition for fork registry lookup.
    Handles RComparison EQ/IN against literal values."""
    from src.pipeline.stage3.middleware.fork_registry import BranchCondition, Operator

    if isinstance(pred, RComparison):
        if not isinstance(pred.left, RColumnRef):
            return None
        if not isinstance(pred.right, RLiteral):
            return None
        col_name = pred.left.name
        val = pred.right.value
        if not isinstance(val, str):
            return None
        # We don't know the table from RColumnRef alone -- use a placeholder
        # and let the registry's column-based lookup find the right key
        op = Operator.EQ if pred.op in ("=", "==") else Operator.NEQ
        return BranchCondition(
            fork_key=ForkKey(table_name="", column_name=col_name),
            operator=op,
            values=[val],
        )
    return None
