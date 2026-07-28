"""Converts extracted Stage 3 constraints into the generic DOF graph
(src/util/algorithms/dof_graph.py).

`analyze_cross_shard_constraints` (cross_shard.py Constraint/
DistributionConstraint/DerivedColumnConstraint -> DOF graph): the LIVE
path for everything the 3 real extraction agents
(statistical/structural/logic_extractor) actually produce today --
distributions, cardinalities, ranges, derived columns -- routed
through Grain canonicalization so ON-scope comparability is handled
correctly (see grain.py).

Calls the real DOFGraph/Dulmage-Mendelsohn classifier
(src/util/algorithms/dof_graph.py) -- does not reimplement it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.util.schema_model.schema import Schema
from src.pipeline.stage3.middleware.cycles import detect_derived_cycles
from src.pipeline.stage3.middleware.fork_registry import (
    BranchCondition,
    ForkKey,
    ForkKeyRegistry,
    Operator,
    Unresolved,
)
from src.pipeline.stage3.models import cross_shard
from src.pipeline.stage3.models.grain import (
    CanonicalizationFailure,
    Grain,
    _SchemaView,
    canonicalize,
)
from src.util.constraint_model.relation.nodes import Aggregate, BaseTable, RelationUnion
from src.pipeline.stage3.models.probe import (
    Stage3AnalysisReport,
    VariableProbe,
)
from src.util.algorithms.dof_graph import (
    Constraint as DOFConstraint,
    DOFClassification,
    DOFGraph,
    OverconstrainedBlock,
    Variable as DOFVariable,
)
from src.util.constraint_model.condition.expressions import RColumnRef, RLiteral
from src.util.constraint_model.condition.predicates import (
    RComparison,
    extract_columns,
)
from src.util.constraint_model.condition.predicates import (
    RPredicateUnion as RPredicate,
)

logger = logging.getLogger(__name__)


# =============================================================================
# cross_shard.py pathway -- the live path for the 3 real extraction agents
# =============================================================================
# cross_shard.py pathway -- the live path for the 3 real extraction agents
# =============================================================================


# ---------------------------------------------------------------------------
# Rich variable model (adapted from the stage3_conflict_v2 prototype, which is
# no longer in the repo)
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
    # RComparison.op is Literal["<", "<=", "=", "!=", ">=", ">"] -- this used
    # to also test for "==" and "in", neither of which the type permits.
    if condition.op != "=":
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
                BranchCondition(fork_key=fk, operator=Operator.EQ, values=[val])
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


def _on_base_table(node: RelationUnion) -> Optional[str]:
    """Extract the base table name from an ON tree."""
    if isinstance(node, BaseTable):
        return node.name
    if isinstance(node, Aggregate):
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
    cols = extract_columns(c.condition)
    if len(cols) != 1:
        # A range/bounds variable is single-column by construction: the bound
        # metadata attaches to one column's domain. A cross-column rule is a
        # real constraint that this representation simply cannot carry, so it
        # is dropped here -- but it was dropped with no log at all, which made
        # a whole class of extracted constraint (any rule relating two columns)
        # vanish between extraction and the DOF graph with no trace anywhere.
        logger.info(
            "[ConstraintGraph] Constraint over %d columns %s (facts %s) has no "
            "single-column range representation and contributes no DOF variable. "
            "It remains in the constraint set; only the bounds view skips it.",
            len(cols),
            sorted(cols),
            list(c.fact_references),
        )
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
        from src.util.constraint_model.relation.nodes import extract_base_tables

        on_tables = extract_base_tables(c.on)

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
        on = BaseTable(name=dc.target_table)
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
) -> Tuple[Stage3AnalysisReport, Dict[str, List[int]]]:
    """Stage 3's complete DOF analysis for cross_shard.py-shaped extraction
    outputs. This is the NEW entry point that consumes the real extraction
    agent output shapes (on: RelationUnion, condition: RPredicate) and routes
    through grain canonicalization + real DOFGraph.

    Returns (Stage3AnalysisReport, variable_fact_map). The report has:
    - square_variables: determined parameters (pinned by facts)
    - loose_variable_probes: free parameters for Stage 4
    - overconstrained_blocks: genuine contradictions flagged for review

    variable_fact_map maps each flat variable name (as it appears in
    overconstrained_blocks/confirmed_conflicts) back to the fact IDs that
    produced it -- callers doing conflict reconciliation need this to find
    which NL facts to re-examine; the report itself only carries flat
    names (Stage 4's contract doesn't need fact-level detail).
    """
    distributions = distributions or []
    structural = structural or []
    logic = logic or []
    derived = derived or []

    # Independent of grain/DOF conversion below -- a circular derived-column
    # definition (e.g. x = x + 5) is a genuine contradiction regardless of
    # whether anything else in this batch produces a single DOF variable.
    cycle_issues = detect_derived_cycles(derived)

    if schema is None:
        logger.warning("No schema provided; returning empty analysis.")
        return Stage3AnalysisReport(derived_cycle_conflicts=cycle_issues), {}

    if registry is None:
        registry = ForkKeyRegistry()

    # Build fork registry: scan all categorical distributions for fork keys
    _build_fork_registry(distributions, registry)

    all_vars, all_cons = _convert_cross_shard_constraints(
        distributions, structural, logic, derived, schema, registry
    )

    variable_fact_map: Dict[str, List[int]] = {}
    for v in all_vars:
        existing = variable_fact_map.setdefault(v.flat_name(), [])
        for fid in v.fact_references:
            if fid not in existing:
                existing.append(fid)

    if not all_vars:
        return (
            Stage3AnalysisReport(derived_cycle_conflicts=cycle_issues),
            variable_fact_map,
        )

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

    return (
        Stage3AnalysisReport(
            square_variables=[v.flat_name() for v in classification.square],
            loose_variable_probes=loose_probes,
            overconstrained_blocks=overconstrained,
            derived_cycle_conflicts=cycle_issues,
        ),
        variable_fact_map,
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
) -> Optional[BranchCondition]:
    """Parse an R-predicate into a BranchCondition for fork registry lookup.
    Handles RComparison EQ/IN against literal values."""
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
