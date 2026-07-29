"""Unit tests for src/pipeline/stage3/middleware/constraint_graph.py.

This module is the SOLE producer of `square_variables` /
`loose_variable_probes` -- Stage 3's entire output contract to Stage 4 --
and had zero executed lines before this file existed (every reference to
`analyze_cross_shard_constraints` in test_stage3_orchestration.py replaces
it with an empty report).

Everything here is pure input/output over real objects: real `Schema`,
real `cross_shard` constraints, a real `ForkKeyRegistry`, the real
`DOFGraph`. Nothing is patched or stubbed -- the module is deterministic,
so nothing needs to be.

Tests marked `xfail(strict=True)` assert the CORRECT behavior of a
confirmed defect, so they flip to passing the moment the defect is fixed.
They deliberately do not lock in today's wrong answer.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import pytest

from src.pipeline.stage3.middleware import constraint_graph as cg
from src.pipeline.stage3.middleware.constraint_graph import (
    BranchTag,
    RichConstraint,
    RichVariable,
    VariableKind,
    analyze_cross_shard_constraints,
    build_and_classify,
)
from src.pipeline.stage3.middleware.fork_registry import ForkKey, ForkKeyRegistry
from src.pipeline.stage3.models.cross_shard import (
    Constraint,
    DerivedColumnConstraint,
    DistributionConstraint,
)
from src.pipeline.stage3.models.grain import Grain, canonicalize
from src.pipeline.stage3.models.probe import Stage3AnalysisReport
from src.util.constraint_model.condition.expressions import (
    RArithmetic,
    RColumnRef,
    RLiteral,
)
from src.util.constraint_model.condition.predicates import RAnd, RComparison
from src.util.constraint_model.relation.nodes import (
    Aggregate,
    BaseTable,
    Fanout,
    Join,
    JoinCondition,
)

# The fintech schema from src/tests/fixtures/sample_data.py is the schema
# under test throughout: USER(user_id pk, credit_score, earnings_information,
# is_institutional) 1--* CREDIT_PRODUCT(credit_product_id pk, user_id fk,
# maturity, yield_value, spread, haircut, ltv, drawdown_phase).
USER = "USER"
PRODUCT = "CREDIT_PRODUCT"


# =========================================================================
# Small builders -- real model objects, no factories-of-factories
# =========================================================================


_Family = Literal["GAUSSIAN", "LOG_NORMAL", "BETA", "POISSON", "CATEGORICAL", "UNIFORM"]
_Op = Literal["<", "<=", "=", "!=", ">=", ">"]
_Category = Literal["statistical", "structural", "logic", "temporal", "derived"]


def _dist(
    column: str,
    family: _Family,
    parameters: dict,
    *,
    table: str = PRODUCT,
    facts: Optional[List[int]] = None,
    if_condition: Optional[RComparison] = None,
) -> DistributionConstraint:
    return DistributionConstraint(
        fact_references=facts or [1],
        on=BaseTable(name=table),
        column=column,
        family=family,
        parameters=parameters,
        if_condition=if_condition,
    )


def _cmp(column: str, op: _Op, value) -> RComparison:
    return RComparison(op=op, left=RColumnRef(name=column), right=RLiteral(value=value))


def _constraint(
    condition,
    *,
    on=None,
    category: _Category = "logic",
    facts: Optional[List[int]] = None,
) -> Constraint:
    return Constraint(
        fact_references=facts or [1],
        on=on if on is not None else BaseTable(name=PRODUCT),
        condition=condition,
        category=category,
    )


def _count_star(table: str = PRODUCT) -> Aggregate:
    return Aggregate(
        source=BaseTable(name=table), fn="COUNT", column="*", alias="row_count"
    )


def _grain(schema, table: str = PRODUCT) -> Grain:
    result = canonicalize(BaseTable(name=table), schema)
    assert isinstance(result, Grain), result
    return result


def _rich(
    grain: Grain,
    name: str,
    *,
    kind: VariableKind = VariableKind.COLUMN_RANGE,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    categories: Optional[frozenset] = None,
    branch: Optional[BranchTag] = None,
    facts: Tuple[int, ...] = (),
) -> RichVariable:
    return RichVariable(
        grain=grain,
        kind=kind,
        name=name,
        branch=branch,
        fact_references=facts,
        lower_bound=lower,
        upper_bound=upper,
        categories=categories,
    )


def _flat(table: str, kind: str, name: str, suffix: str = "") -> str:
    """The flat identity RichVariable.flat_name() produces for a bare
    (edge-free, non-aggregate) grain. Spelled out so a change to the naming
    scheme is a visible test failure rather than a silent contract break."""
    return f"{table}[[]]::{kind}::{name}{suffix}"


def _loose(report: Stage3AnalysisReport) -> dict:
    return {
        p.variable_name: (p.lower_bound, p.upper_bound)
        for p in report.loose_variable_probes
    }


def _over_vars(report: Stage3AnalysisReport) -> set:
    return {v for block in report.overconstrained_blocks for v in block.variables}


# =========================================================================
# analyze_cross_shard_constraints -- end to end
# =========================================================================


class TestAnalyzeEndToEnd:
    @pytest.mark.parametrize(
        "family, parameters, expected_params",
        [
            ("GAUSSIAN", {"mean": 50.0, "std_dev": 5.0}, ["mean", "std_dev"]),
            ("LOG_NORMAL", {"mean": 3.0, "std_dev": 1.0}, ["mean", "std_dev"]),
            ("BETA", {"alpha": 2.0, "beta": 5.0}, ["alpha", "beta"]),
            ("POISSON", {"lam": 4.0}, ["lam"]),
            (
                "UNIFORM",
                {"min_value": 1.0, "max_value": 9.0},
                ["min_value", "max_value"],
            ),
        ],
    )
    def test_each_distribution_family_pins_exactly_its_own_parameters(
        self, fintech_schema, family, parameters, expected_params
    ):
        """One Constraint per parameter (dof_graph.py's rule 1) means every
        stated parameter is exactly determined -- square, never loose.

        Catches: collapsing a two-parameter family into one Constraint over
        both variables (both would become loose), and any mix-up of a
        family's parameter names.
        """
        report, fact_map = analyze_cross_shard_constraints(
            distributions=[_dist("spread", family, parameters, facts=[7])],
            schema=fintech_schema,
        )
        expected = sorted(
            _flat(PRODUCT, "column_distribution_param", f"spread.{p}")
            for p in expected_params
        )
        assert sorted(report.square_variables) == expected
        assert report.loose_variable_probes == []
        assert report.overconstrained_blocks == []
        assert {k: v for k, v in fact_map.items()} == {n: [7] for n in expected}

    def test_range_fact_is_a_loose_probe_carrying_its_bound(self, fintech_schema):
        """A bound is domain metadata, not a DOF-consuming equation: the
        variable stays free but Stage 4 gets told the interval.

        Catches: emitting a pinning RichConstraint for a range (the variable
        would become square), or dropping the bound from the probe.
        """
        report, fact_map = analyze_cross_shard_constraints(
            logic=[_constraint(_cmp("haircut", ">=", 5), facts=[12])],
            schema=fintech_schema,
        )
        name = _flat(PRODUCT, "column_range", "haircut")
        assert report.square_variables == []
        assert _loose(report) == {name: (5.0, None)}
        assert fact_map == {name: [12]}

    def test_pinned_row_count_is_square_and_scoped_to_its_aggregate(
        self, fintech_schema
    ):
        """`COUNT(*) = 1000` pins the table's cardinality. The flat name must
        carry the aggregate signature, so a row-count variable can never be
        unified with a same-named variable at the bare table grain.

        Catches: dropping agg_signature from flat_name(), and failing to emit
        the pinning constraint for `=` (the variable would go loose).
        """
        report, _ = analyze_cross_shard_constraints(
            structural=[
                _constraint(
                    _cmp("row_count", "=", 1000),
                    on=_count_star(),
                    category="structural",
                )
            ],
            schema=fintech_schema,
        )
        assert len(report.square_variables) == 1
        only = report.square_variables[0]
        assert "::table_cardinality::row_count" in only
        assert "agg=('COUNT', '*'" in only
        assert only != _flat(PRODUCT, "table_cardinality", "row_count")
        assert report.loose_variable_probes == []

    def test_derived_column_is_pinned_by_its_formula(self, fintech_schema):
        """A derived column is determined by its expression, so it must be
        square -- Stage 4 must not be asked to invent a value for it.

        Catches: dropping the `derived_...` pinning RichConstraint, which
        would hand the column to Stage 4 as a free parameter.
        """
        report, fact_map = analyze_cross_shard_constraints(
            derived=[
                DerivedColumnConstraint(
                    fact_references=[11],
                    target_table=PRODUCT,
                    target_column="ltv",
                    expression=RArithmetic(
                        op="/",
                        left=RColumnRef(name="spread"),
                        right=RColumnRef(name="haircut"),
                    ),
                    referenced_tables=[PRODUCT],
                )
            ],
            schema=fintech_schema,
        )
        name = _flat(PRODUCT, "derived_column", "ltv")
        assert report.square_variables == [name]
        assert report.loose_variable_probes == []
        assert fact_map == {name: [11]}

    def test_two_range_facts_about_one_column_intersect_to_one_probe(
        self, fintech_schema
    ):
        """Multi-shard bound tightening: two facts, one variable, the
        intersection of both intervals.

        Catches: keeping only the first-seen variable's bounds (the historical
        bug _merge_rich_bounds was written to fix) -- the upper bound would be
        None -- and emitting two separate probes for one column.
        """
        report, fact_map = analyze_cross_shard_constraints(
            logic=[
                _constraint(_cmp("maturity", ">=", 5), facts=[9]),
                _constraint(_cmp("maturity", "<=", 100), facts=[10]),
            ],
            schema=fintech_schema,
        )
        name = _flat(PRODUCT, "column_range", "maturity")
        assert _loose(report) == {name: (5.0, 100.0)}
        assert fact_map == {name: [9, 10]}

    def test_contradictory_pins_surface_as_a_confirmed_conflict(self, fintech_schema):
        """Two facts pinning one parameter to different values is a provable
        value contradiction. It must NOT reach Stage 4 as either a determined
        value or a free parameter, and the fact map must name both culprits.

        Catches: letting the merged (empty) interval through to square/loose,
        which would hand Stage 4 an infeasible variable with no signal that
        the input facts disagree.
        """
        report, fact_map = analyze_cross_shard_constraints(
            distributions=[
                _dist("spread", "POISSON", {"lam": 5.0}, facts=[40]),
                _dist("spread", "POISSON", {"lam": 12.0}, facts=[41]),
            ],
            schema=fintech_schema,
        )
        name = _flat(PRODUCT, "column_distribution_param", "spread.lam")
        assert report.square_variables == []
        assert report.loose_variable_probes == []
        assert _over_vars(report) == {name}
        assert fact_map == {name: [40, 41]}
        assert report.is_feasible is False

    def test_disjoint_category_sets_are_a_confirmed_conflict(self, fintech_schema):
        """Categories merge by intersection, so two facts naming disjoint
        sets for one column is unresolvable -- not a pick-either-side.

        Catches: replacing the category intersection with a union or a
        last-write-wins overwrite, either of which reports no conflict.
        """
        report, _ = analyze_cross_shard_constraints(
            distributions=[
                _dist(
                    "is_institutional",
                    "CATEGORICAL",
                    {"categories": ["retail", "sme"], "probabilities": [0.5, 0.5]},
                    table=USER,
                    facts=[2],
                ),
                _dist(
                    "is_institutional",
                    "CATEGORICAL",
                    {"categories": ["institutional"], "probabilities": [1.0]},
                    table=USER,
                    facts=[3],
                ),
            ],
            schema=fintech_schema,
        )
        name = _flat(
            USER, "column_distribution_param", "is_institutional.probabilities"
        )
        assert _over_vars(report) == {name}
        assert report.square_variables == []
        assert report.loose_variable_probes == []

    def test_overlapping_category_sets_are_not_a_conflict(self, fintech_schema):
        """The mirror image of the test above: a non-empty intersection is
        harmless, so the variable must survive into the probe contract.

        Catches: treating any repeated CATEGORICAL fact as a conflict (an
        over-eager conflict rule would drop this variable entirely).
        """
        report, fact_map = analyze_cross_shard_constraints(
            distributions=[
                _dist(
                    "is_institutional",
                    "CATEGORICAL",
                    {"categories": ["retail", "sme"]},
                    table=USER,
                    facts=[2],
                ),
                _dist(
                    "is_institutional",
                    "CATEGORICAL",
                    {"categories": ["sme", "institutional"]},
                    table=USER,
                    facts=[3],
                ),
            ],
            schema=fintech_schema,
        )
        name = _flat(
            USER, "column_distribution_param", "is_institutional.probabilities"
        )
        assert report.overconstrained_blocks == []
        assert set(_loose(report)) == {name}
        assert fact_map == {name: [2, 3]}

    def test_two_agreeing_pins_are_reported_as_redundant_not_dropped(
        self, fintech_schema
    ):
        """Two facts stating the SAME value are not a value conflict, but they
        are two equations for one unknown -- Dulmage-Mendelsohn flags the
        block. The variable must still be visible somewhere.

        Catches: silently swallowing the redundant second pin, which would
        hide genuinely duplicated extraction from the reconciliation loop.
        """
        report, _ = analyze_cross_shard_constraints(
            distributions=[
                _dist("spread", "POISSON", {"lam": 5.0}, facts=[40]),
                _dist("spread", "POISSON", {"lam": 5.0}, facts=[41]),
            ],
            schema=fintech_schema,
        )
        name = _flat(PRODUCT, "column_distribution_param", "spread.lam")
        assert _over_vars(report) == {name}
        # Distinguishing feature vs. the contradictory case: the constraint
        # names are carried, because DOF (not the value merge) found this.
        assert sorted(report.overconstrained_blocks[0].constraints) == [
            "pin_CREDIT_PRODUCT.spread.lam#0",
            "pin_CREDIT_PRODUCT.spread.lam#1",
        ]

    def test_join_grain_and_base_grain_are_never_unified(self, fintech_schema):
        """The same column name at two different grains is two different
        populations, hence two different variables.

        Catches: dropping the edge set from flat_name(), which would merge
        these into one variable and intersect bounds across incomparable
        populations.
        """
        joined = Join(
            left=BaseTable(name=PRODUCT),
            right=BaseTable(name=USER),
            on=[JoinCondition(left=f"{PRODUCT}.user_id", right=f"{USER}.user_id")],
        )
        report, _ = analyze_cross_shard_constraints(
            logic=[
                _constraint(_cmp("maturity", ">=", 7), on=joined, facts=[9]),
                _constraint(_cmp("maturity", ">=", 9), facts=[10]),
            ],
            schema=fintech_schema,
        )
        probes = _loose(report)
        assert len(probes) == 2
        assert probes[_flat(PRODUCT, "column_range", "maturity")] == (9.0, None)
        joined_name = next(
            n for n in probes if n != _flat(PRODUCT, "column_range", "maturity")
        )
        assert "('CREDIT_PRODUCT', 'user_id', 'USER', 1)" in joined_name
        assert probes[joined_name] == (7.0, None)

    def test_fanout_child_count_bound_is_scoped_to_the_fanout(self, fintech_schema):
        """A Fanout ON tree canonicalizes to a COUNT_CHILDREN_LEFT_JOIN grain
        rooted at the parent; a bound on `child_count` must inherit it.

        Catches: losing the Fanout signature (the variable would collide with
        a plain USER-grain column named child_count).
        """
        report, _ = analyze_cross_shard_constraints(
            structural=[
                _constraint(
                    _cmp("child_count", ">=", 3),
                    on=Fanout(
                        parent_table=USER, child_table=PRODUCT, fk_column="user_id"
                    ),
                    category="structural",
                    facts=[9],
                )
            ],
            schema=fintech_schema,
        )
        ((name, bounds),) = _loose(report).items()
        assert name.startswith(f"{USER}[[]]|agg=('COUNT_CHILDREN_LEFT_JOIN'")
        assert bounds == (3.0, None)

    def test_derived_column_cycle_with_no_fixed_point_is_reported(self, fintech_schema):
        """`x = x + 5` is unsatisfiable and must be flagged regardless of what
        the DOF pass makes of the same batch.

        Catches: gating the cycle check behind the DOF path (e.g. returning
        early before detect_derived_cycles when no variables are produced).
        """
        cyclic = DerivedColumnConstraint(
            fact_references=[11],
            target_table=PRODUCT,
            target_column="ltv",
            expression=RArithmetic(
                op="+", left=RColumnRef(name="ltv"), right=RLiteral(value=5)
            ),
            referenced_tables=[PRODUCT],
        )
        with_schema, _ = analyze_cross_shard_constraints(
            derived=[cyclic], schema=fintech_schema
        )
        assert len(with_schema.derived_cycle_conflicts) == 1
        assert with_schema.derived_cycle_conflicts[0].nodes == (f"{PRODUCT}.ltv",)
        assert with_schema.is_feasible is False

        # Also survives the no-schema early return.
        no_schema, fact_map = analyze_cross_shard_constraints(derived=[cyclic])
        assert len(no_schema.derived_cycle_conflicts) == 1
        assert fact_map == {}


# =========================================================================
# Conservation: every produced variable is classified exactly once
# =========================================================================


def _scenario_distribution(schema) -> Dict[str, Any]:
    return dict(
        distributions=[_dist("spread", "GAUSSIAN", {"mean": 4.0, "std_dev": 1.0})]
    )


def _scenario_range_only(schema) -> Dict[str, Any]:
    return dict(logic=[_constraint(_cmp("haircut", "<=", 3))])


def _scenario_conflict(schema) -> Dict[str, Any]:
    return dict(
        distributions=[
            _dist("spread", "POISSON", {"lam": 5.0}, facts=[1]),
            _dist("spread", "POISSON", {"lam": 12.0}, facts=[2]),
        ]
    )


def _scenario_redundant(schema) -> Dict[str, Any]:
    return dict(
        distributions=[
            _dist("spread", "POISSON", {"lam": 5.0}, facts=[1]),
            _dist("spread", "POISSON", {"lam": 5.0}, facts=[2]),
        ]
    )


def _scenario_mixed(schema) -> Dict[str, Any]:
    return dict(
        distributions=[
            _dist("spread", "GAUSSIAN", {"mean": 4.0, "std_dev": 1.0}, facts=[1]),
            _dist(
                "is_institutional",
                "CATEGORICAL",
                {"categories": ["retail", "institutional"]},
                table=USER,
                facts=[2],
            ),
        ],
        structural=[
            _constraint(
                _cmp("row_count", "=", 500),
                on=_count_star(),
                category="structural",
                facts=[3],
            )
        ],
        logic=[
            _constraint(_cmp("maturity", ">=", 6), facts=[4]),
            _constraint(_cmp("maturity", "<=", 60), facts=[5]),
        ],
        derived=[
            DerivedColumnConstraint(
                fact_references=[6],
                target_table=PRODUCT,
                target_column="ltv",
                expression=RArithmetic(
                    op="*",
                    left=RColumnRef(name="spread"),
                    right=RLiteral(value=2),
                ),
                referenced_tables=[PRODUCT],
            )
        ],
    )


def _scenario_empty(schema) -> Dict[str, Any]:
    return {}


def _scenario_unresolvable_table(schema) -> Dict[str, Any]:
    return dict(logic=[_constraint(_cmp("nope", ">=", 1), on=BaseTable(name="ABSENT"))])


def _scenario_cross_column(schema) -> Dict[str, Any]:
    return dict(
        logic=[
            _constraint(
                RComparison(
                    op="<=",
                    left=RColumnRef(name="haircut"),
                    right=RColumnRef(name="spread"),
                )
            )
        ]
    )


_SCENARIOS: List[Tuple[str, Callable[[Any], Dict[str, Any]]]] = [
    ("distribution", _scenario_distribution),
    ("range_only", _scenario_range_only),
    ("value_conflict", _scenario_conflict),
    ("redundant_pins", _scenario_redundant),
    ("mixed_batch", _scenario_mixed),
    ("empty_shard", _scenario_empty),
    ("unresolvable_table", _scenario_unresolvable_table),
    ("cross_column", _scenario_cross_column),
]


class TestConservation:
    """The invariant Stage 4 depends on: the report partitions every variable
    the conversion actually produced. Nothing may appear twice, and nothing
    may vanish."""

    @pytest.mark.parametrize(
        "build", [s[1] for s in _SCENARIOS], ids=[s[0] for s in _SCENARIOS]
    )
    def test_square_loose_and_overconstrained_partition_every_variable(
        self, fintech_schema, build
    ):
        """Catches: any code path that reports a variable in two buckets
        (double-counting a determined parameter as also free), and any path
        that computes a variable then silently drops it -- the failure mode
        that makes a probe contract quietly incomplete.
        """
        report, fact_map = analyze_cross_shard_constraints(
            schema=fintech_schema, **build(fintech_schema)
        )
        square = set(report.square_variables)
        loose = set(_loose(report))
        over = _over_vars(report)

        assert len(report.square_variables) == len(square), "duplicate square entries"
        assert len(report.loose_variable_probes) == len(loose), "duplicate probes"
        assert square & loose == set(), "variable is both determined and free"
        assert square & over == set()
        assert loose & over == set()
        assert square | loose | over == set(fact_map), (
            "report does not account for exactly the variables that were built"
        )

    def test_every_reported_variable_traces_back_to_at_least_one_fact(
        self, fintech_schema
    ):
        """Provenance is what the reconciliation agent needs to find which NL
        facts to re-examine.

        Catches: dropping fact_references while merging two RichVariables that
        share a flat name (the merge must union them, not overwrite).
        """
        report, fact_map = analyze_cross_shard_constraints(
            schema=fintech_schema, **_scenario_mixed(fintech_schema)
        )
        reported = (
            set(report.square_variables) | set(_loose(report)) | _over_vars(report)
        )
        assert reported
        for name in reported:
            assert fact_map[name], f"{name} has no originating fact"


# =========================================================================
# Degenerate inputs
# =========================================================================


class TestDegenerateInputs:
    def test_no_arguments_at_all_returns_an_empty_report(self):
        """Catches: raising (or returning None) instead of an empty report
        when a caller has nothing to analyze."""
        report, fact_map = analyze_cross_shard_constraints()
        assert report == Stage3AnalysisReport()
        assert fact_map == {}
        assert report.is_feasible is True

    def test_missing_schema_short_circuits_before_conversion(self):
        """Grain canonicalization is impossible without a schema, so nothing
        may be fabricated.

        Catches: proceeding with schema=None and emitting variables built from
        an unvalidated ON tree.
        """
        report, fact_map = analyze_cross_shard_constraints(
            distributions=[_dist("spread", "POISSON", {"lam": 3.0})],
            structural=[_constraint(_cmp("row_count", "=", 5), on=_count_star())],
        )
        assert report.square_variables == []
        assert report.loose_variable_probes == []
        assert fact_map == {}

    def test_empty_lists_with_a_schema_produce_an_empty_report(self, fintech_schema):
        report, fact_map = analyze_cross_shard_constraints(
            distributions=[], structural=[], logic=[], derived=[], schema=fintech_schema
        )
        assert report == Stage3AnalysisReport()
        assert fact_map == {}

    def test_constraint_on_a_table_outside_the_schema_is_skipped(self, fintech_schema):
        """Catches: falling back to the raw table name when canonicalize()
        fails, which would mint a variable at a grain that does not exist."""
        report, fact_map = analyze_cross_shard_constraints(
            logic=[
                _constraint(_cmp("spread", ">=", 1), on=BaseTable(name="ABSENT")),
                _constraint(_cmp("haircut", ">=", 2), facts=[12]),
            ],
            schema=fintech_schema,
        )
        assert set(fact_map) == {_flat(PRODUCT, "column_range", "haircut")}

    def test_cross_column_condition_yields_no_range_variable(self, fintech_schema):
        """A two-column rule has no single-column bound representation, so it
        contributes no DOF variable (it is logged, not silently vanished).

        Catches: picking an arbitrary column out of a multi-column predicate
        and attaching the literal-free bound to it.
        """
        report, fact_map = analyze_cross_shard_constraints(
            logic=[
                _constraint(
                    RComparison(
                        op="<=",
                        left=RColumnRef(name="haircut"),
                        right=RColumnRef(name="spread"),
                    )
                )
            ],
            schema=fintech_schema,
        )
        assert fact_map == {}
        assert report == Stage3AnalysisReport()

    def test_variable_touched_by_no_constraint_is_loose(self, fintech_schema):
        """Catches: defaulting unmatched variables to square, which would tell
        Stage 4 a free parameter is already determined."""
        grain = _grain(fintech_schema)
        result = build_and_classify([_rich(grain, "spread", lower=1.0)], [])
        assert [v.name for v in result.loose] == ["spread"]
        assert result.square == []

    def test_variable_present_only_inside_a_constraint_is_still_registered(
        self, fintech_schema
    ):
        """build_and_classify must backfill any variable a constraint
        references but the variable list omits -- otherwise DOFGraph raises on
        an undefined variable reference.

        Catches: removing the `by_flat.setdefault` backfill loop.
        """
        grain = _grain(fintech_schema)
        var = _rich(grain, "spread")
        result = build_and_classify([], [RichConstraint(name="pin", variables=(var,))])
        assert [v.name for v in result.square] == ["spread"]

    def test_no_variables_and_no_constraints_classifies_to_nothing(self):
        result = build_and_classify([], [])
        assert (result.square, result.loose) == ([], [])
        assert result.overconstrained_blocks == []
        assert result.confirmed_conflicts == []

    def test_one_constraint_over_two_variables_leaves_both_free(self, fintech_schema):
        """dof_graph.py rule 1: one equation cannot determine two unknowns.

        Catches: marking every variable a constraint touches as square, which
        is exactly the modeling mistake the DOF pass exists to catch.
        """
        grain = _grain(fintech_schema)
        a, b = _rich(grain, "spread"), _rich(grain, "haircut")
        result = build_and_classify(
            [a, b], [RichConstraint(name="c", variables=(a, b))]
        )
        assert sorted(v.name for v in result.loose) == ["haircut", "spread"]
        assert result.square == []

    @pytest.mark.parametrize("kind", list(VariableKind))
    def test_flat_name_is_scoped_by_kind(self, fintech_schema, kind):
        """Catches: dropping `kind` from flat_name(), which would unify e.g. a
        column's range variable with its derived-column variable."""
        grain = _grain(fintech_schema)
        assert _rich(grain, "spread", kind=kind).flat_name() == _flat(
            PRODUCT, kind.value, "spread"
        )


# =========================================================================
# _merge_rich_bounds
# =========================================================================


class TestMergeRichBounds:
    @pytest.mark.parametrize(
        "a_bounds, b_bounds, expected",
        [
            # Compatible intervals intersect to the tighter one.
            ((5.0, 20.0), (10.0, 15.0), (10.0, 15.0, True)),
            ((10.0, 15.0), (5.0, 20.0), (10.0, 15.0, True)),
            # Half-open on each side compose into a closed interval.
            ((5.0, None), (None, 20.0), (5.0, 20.0, True)),
            ((None, 20.0), (5.0, None), (5.0, 20.0, True)),
            # Wholly unbounded side contributes nothing.
            ((None, None), (3.0, 8.0), (3.0, 8.0, True)),
            ((3.0, 8.0), (None, None), (3.0, 8.0, True)),
            # Two facts agreeing on an exact pin stay valid.
            ((5.0, 5.0), (5.0, 5.0), (5.0, 5.0, True)),
            # Touching at a single point is still non-empty.
            ((5.0, 10.0), (10.0, 20.0), (10.0, 10.0, True)),
            # Contradictions: empty interval.
            ((5.0, 5.0), (12.0, 12.0), (12.0, 5.0, False)),
            ((None, 4.0), (9.0, None), (9.0, 4.0, False)),
            ((0.0, 3.0), (4.0, 7.0), (4.0, 3.0, False)),
        ],
    )
    def test_interval_merge(self, fintech_schema, a_bounds, b_bounds, expected):
        """Catches: swapping max/min in the intersection (an inverted merge
        widens instead of tightens), and an off-by-one `<` vs `<=` validity
        test that would call a single-point interval empty.
        """
        grain = _grain(fintech_schema)
        a = _rich(grain, "spread", lower=a_bounds[0], upper=a_bounds[1])
        b = _rich(grain, "spread", lower=b_bounds[0], upper=b_bounds[1])
        lower, upper, categories, is_valid = cg._merge_rich_bounds(a, b)
        assert (lower, upper, is_valid) == expected
        assert categories is None

    @pytest.mark.parametrize(
        "a_cats, b_cats, expected_cats, expected_valid",
        [
            ({"gold", "silver"}, {"silver", "bronze"}, {"silver"}, True),
            ({"gold"}, {"gold"}, {"gold"}, True),
            ({"gold", "silver"}, {"bronze"}, set(), False),
        ],
    )
    def test_category_merge_is_an_intersection(
        self, fintech_schema, a_cats, b_cats, expected_cats, expected_valid
    ):
        """Catches: unioning categories instead of intersecting them -- two
        facts naming disjoint sets would then look compatible."""
        grain = _grain(fintech_schema)
        a = _rich(grain, "tier", categories=frozenset(a_cats))
        b = _rich(grain, "tier", categories=frozenset(b_cats))
        lower, upper, categories, is_valid = cg._merge_rich_bounds(a, b)
        assert categories == frozenset(expected_cats)
        assert is_valid is expected_valid
        assert (lower, upper) == (None, None)

    def test_three_way_merge_detects_a_contradiction_introduced_last(
        self, fintech_schema
    ):
        """Catches: comparing only the first two variables sharing a flat name
        and never folding the third in."""
        grain = _grain(fintech_schema)
        variables = [
            _rich(grain, "spread", lower=5.0, upper=5.0, facts=(1,)),
            _rich(grain, "spread", lower=5.0, upper=5.0, facts=(2,)),
            _rich(grain, "spread", lower=12.0, upper=12.0, facts=(3,)),
        ]
        result = build_and_classify(variables, [])
        assert result.confirmed_conflicts == [variables[0].flat_name()]
        assert result.square == []
        assert result.loose == []

    def test_a_confirmed_conflict_is_reported_once_not_per_pair(self, fintech_schema):
        """Catches: appending to confirmed_conflicts without the membership
        check, which double-reports the same variable."""
        grain = _grain(fintech_schema)
        variables = [
            _rich(grain, "spread", lower=5.0, upper=5.0),
            _rich(grain, "spread", lower=12.0, upper=12.0),
            _rich(grain, "spread", lower=20.0, upper=20.0),
        ]
        result = build_and_classify(variables, [])
        assert result.confirmed_conflicts == [variables[0].flat_name()]

    def test_merged_variable_unions_fact_references(self, fintech_schema):
        """Catches: keeping only the first (or last) variable's provenance, so
        reconciliation can only see half the disagreeing facts."""
        grain = _grain(fintech_schema)
        result = build_and_classify(
            [
                _rich(grain, "spread", lower=1.0, facts=(9, 4)),
                _rich(grain, "spread", upper=8.0, facts=(4, 2)),
            ],
            [],
        )
        (merged,) = result.loose
        assert merged.fact_references == (2, 4, 9)
        assert (merged.lower_bound, merged.upper_bound) == (1.0, 8.0)


# =========================================================================
# Conversion paths
# =========================================================================


class TestConversionPaths:
    @pytest.mark.parametrize(
        "family, parameters, expected",
        [
            ("GAUSSIAN", {"mean": 8.0, "std_dev": 2.0}, {"mean": 8.0, "std_dev": 2.0}),
            ("BETA", {"alpha": 2.0, "beta": 3.0}, {"alpha": 2.0, "beta": 3.0}),
            ("POISSON", {"lam": 7.0}, {"lam": 7.0}),
            (
                "UNIFORM",
                {"min_value": 1.0, "max_value": 4.0},
                {"min_value": 1.0, "max_value": 4.0},
            ),
        ],
    )
    def test_distribution_parameter_is_pinned_to_its_stated_value(
        self, fintech_schema, family, parameters, expected
    ):
        """The stated value must land on the variable as lower==upper, or two
        facts agreeing on it can never be compared.

        Catches: recording only THAT a parameter was pinned without recording
        to what -- the historical gap that made value conflicts invisible.
        """
        grain = _grain(fintech_schema)
        variables, constraints = cg._distribution_to_rich(
            _dist("spread", family, parameters, facts=[7]),
            grain,
            cg._SchemaView.from_schema(fintech_schema),
            ForkKeyRegistry(),
            disambiguator=0,
        )
        assert {v.name: (v.lower_bound, v.upper_bound) for v in variables} == {
            f"spread.{p}": (val, val) for p, val in expected.items()
        }
        # Exactly one pinning constraint per parameter, never one for several.
        assert len(constraints) == len(expected)
        for c in constraints:
            assert len(c.variables) == 1
            assert c.fact_references == (7,)

    def test_categorical_without_probabilities_creates_no_pinning_constraint(
        self, fintech_schema
    ):
        """Naming the categories does not state the weights, so the
        probabilities vector stays a degree of freedom.

        Catches: always emitting the pin for CATEGORICAL, which would report
        an unstated probability vector as determined.
        """
        grain = _grain(fintech_schema, USER)
        variables, constraints = cg._distribution_to_rich(
            _dist(
                "is_institutional",
                "CATEGORICAL",
                {"categories": ["retail", "institutional"]},
                table=USER,
            ),
            grain,
            cg._SchemaView.from_schema(fintech_schema),
            ForkKeyRegistry(),
            disambiguator=0,
        )
        assert [v.name for v in variables] == ["is_institutional.probabilities"]
        assert variables[0].categories == frozenset({"retail", "institutional"})
        assert constraints == []

    def test_categorical_with_probabilities_is_pinned(self, fintech_schema):
        grain = _grain(fintech_schema, USER)
        _, constraints = cg._distribution_to_rich(
            _dist(
                "is_institutional",
                "CATEGORICAL",
                {
                    "categories": ["retail", "institutional"],
                    "probabilities": [0.4, 0.6],
                },
                table=USER,
            ),
            grain,
            cg._SchemaView.from_schema(fintech_schema),
            ForkKeyRegistry(),
            disambiguator=3,
        )
        assert [c.name for c in constraints] == [
            "pin_USER.is_institutional.probabilities#3"
        ]

    @pytest.mark.parametrize(
        "op, value, expected",
        [
            (">=", 5, (5.0, None)),
            ("<=", 5, (None, 5.0)),
            ("=", 5, (5.0, 5.0)),
        ],
    )
    def test_range_conversion_maps_operator_to_bound_side(
        self, fintech_schema, op, value, expected
    ):
        """Catches: swapping the >= and <= branches, which inverts every
        stated bound handed to Stage 4.
        """
        grain = _grain(fintech_schema)
        variables, constraints = cg._range_constraint_to_rich(
            _constraint(_cmp("haircut", op, value), facts=[12]),
            grain,
            cg._SchemaView.from_schema(fintech_schema),
            ForkKeyRegistry(),
            disambiguator=0,
        )
        (var,) = variables
        assert (var.lower_bound, var.upper_bound) == expected
        assert var.kind is VariableKind.COLUMN_RANGE
        assert var.name == "haircut"
        # A bound never consumes a degree of freedom.
        assert constraints == []

    def test_range_conversion_with_a_non_numeric_literal_states_no_bound(
        self, fintech_schema
    ):
        """Catches: coercing a string literal into a numeric bound (e.g. via
        float(str)), which would invent an interval from a categorical rule.
        """
        grain = _grain(fintech_schema)
        variables, _ = cg._range_constraint_to_rich(
            _constraint(_cmp("drawdown_phase", "=", "active")),
            grain,
            cg._SchemaView.from_schema(fintech_schema),
            ForkKeyRegistry(),
            disambiguator=0,
        )
        (var,) = variables
        assert (var.lower_bound, var.upper_bound) == (None, None)

    @pytest.mark.parametrize(
        "op, value, expected_bounds, expects_pin",
        [
            ("=", 1000, (1000.0, 1000.0), True),
            (">=", 10, (10.0, None), False),
            ("<=", 99, (None, 99.0), False),
        ],
    )
    def test_cardinality_conversion_pins_only_on_equality(
        self, fintech_schema, op, value, expected_bounds, expects_pin
    ):
        """Only `= n` determines a row count; an inequality merely bounds it.

        Catches: emitting the pinning constraint for an inequality, which
        would report a bounded-but-unknown row count as determined.
        """
        grain = _grain(fintech_schema)
        variables, constraints = cg._cardinality_to_rich(
            _constraint(_cmp("row_count", op, value), category="structural", facts=[3]),
            grain,
            disambiguator=2,
        )
        (var,) = variables
        assert var.kind is VariableKind.TABLE_CARDINALITY
        assert var.name == "row_count"
        assert (var.lower_bound, var.upper_bound) == expected_bounds
        assert [c.name for c in constraints] == (
            ["pin_CREDIT_PRODUCT.row_count#2"] if expects_pin else []
        )

    def test_derived_column_conversion_emits_one_variable_and_one_pin(
        self, fintech_schema
    ):
        """Catches: emitting the variable without the pinning constraint (the
        derived column would be probed to Stage 4 as free)."""
        grain = _grain(fintech_schema)
        variables, constraints = cg._derived_column_to_rich(
            DerivedColumnConstraint(
                fact_references=[11],
                target_table=PRODUCT,
                target_column="ltv",
                expression=RArithmetic(
                    op="/",
                    left=RColumnRef(name="spread"),
                    right=RColumnRef(name="haircut"),
                ),
                referenced_tables=[PRODUCT],
            ),
            grain,
            disambiguator=4,
        )
        (var,) = variables
        (constraint,) = constraints
        assert var.kind is VariableKind.DERIVED_COLUMN
        assert var.name == "ltv"
        assert constraint.name == "derived_CREDIT_PRODUCT.ltv#4"
        assert constraint.variables == (var,)
        assert constraint.fact_references == (11,)


# =========================================================================
# Fork / conditional distributions
# =========================================================================


def _forked_pair() -> List[DistributionConstraint]:
    """Two branches of ONE conditional distribution: spread is Poisson(5) for
    institutional customers and Poisson(12) for everyone else."""
    return [
        _dist(
            "spread",
            "POISSON",
            {"lam": 5.0},
            facts=[8],
            if_condition=_cmp("is_institutional", "=", "institutional"),
        ),
        _dist(
            "spread",
            "POISSON",
            {"lam": 12.0},
            facts=[9],
            if_condition=_cmp("is_institutional", "=", "retail"),
        ),
    ]


def _fork_defining_categorical() -> DistributionConstraint:
    """The fact that DEFINES the fork key: an unconditional categorical over
    USER.is_institutional. It has no if_condition -- it is what the branches
    condition ON."""
    return _dist(
        "is_institutional",
        "CATEGORICAL",
        {"categories": ["institutional", "retail"], "probabilities": [0.2, 0.8]},
        table=USER,
        facts=[14],
    )


class TestForkResolution:
    @pytest.mark.parametrize(
        "condition, expected_value",
        [
            (_cmp("is_institutional", "=", "institutional"), "institutional"),
            (_cmp("is_institutional", "=", "retail"), "retail"),
        ],
    )
    def test_equality_against_a_registered_fork_resolves_to_a_branch(
        self, condition, expected_value
    ):
        """Catches: matching the fork by table instead of by column name (the
        RColumnRef is deliberately unqualified, so a table match can never
        succeed) -- every branch would resolve to None."""
        registry = ForkKeyRegistry()
        registry.register_fork(
            ForkKey(table_name=USER, column_name="is_institutional"),
            ["institutional", "retail"],
        )
        tag = cg._resolve_branch(condition, registry)
        assert tag == BranchTag(
            fork_table=USER,
            fork_column="is_institutional",
            branch_value=expected_value,
        )

    @pytest.mark.parametrize(
        "condition, reason",
        [
            (None, "no condition at all"),
            (_cmp("is_institutional", "!=", "retail"), "negation is not a branch"),
            (_cmp("is_institutional", ">=", "retail"), "ordering is not a branch"),
            (_cmp("unrelated_column", "=", "retail"), "column is not a fork key"),
            (_cmp("is_institutional", "=", 3), "literal is not a category label"),
            (
                RComparison(
                    op="=",
                    left=RLiteral(value="retail"),
                    right=RColumnRef(name="is_institutional"),
                ),
                "left side is not a column reference",
            ),
            (
                RComparison(
                    op="=",
                    left=RColumnRef(name="is_institutional"),
                    right=RColumnRef(name="drawdown_phase"),
                ),
                "right side is a column, not a category label",
            ),
        ],
    )
    def test_non_branch_conditions_resolve_to_none(self, condition, reason):
        """Catches: accepting any RComparison as a branch, which would tag
        variables with fabricated branches and split one variable into many.
        """
        registry = ForkKeyRegistry()
        registry.register_fork(
            ForkKey(table_name=USER, column_name="is_institutional"),
            ["institutional", "retail"],
        )
        assert cg._resolve_branch(condition, registry) is None, reason

    def test_an_empty_registry_resolves_nothing(self):
        assert (
            cg._resolve_branch(
                _cmp("is_institutional", "=", "retail"), ForkKeyRegistry()
            )
            is None
        )

    def test_a_pre_registered_fork_keeps_the_two_branches_separate(
        self, fintech_schema
    ):
        """With the fork key known, the two branches of one conditional
        distribution are two distinct, independently determined variables.
        This is the behavior the auto-discovery path is supposed to reach.

        Catches: dropping the branch suffix from flat_name(), which merges the
        branches and turns lam=5 vs lam=12 into a bogus contradiction.
        """
        registry = ForkKeyRegistry()
        registry.register_fork(
            ForkKey(table_name=USER, column_name="is_institutional"),
            ["institutional", "retail"],
        )
        report, fact_map = analyze_cross_shard_constraints(
            distributions=_forked_pair(), schema=fintech_schema, registry=registry
        )
        base = _flat(PRODUCT, "column_distribution_param", "spread.lam")
        assert sorted(report.square_variables) == sorted(
            [
                f"{base}|{USER}.is_institutional=institutional",
                f"{base}|{USER}.is_institutional=retail",
            ]
        )
        assert report.overconstrained_blocks == []
        assert fact_map[f"{base}|{USER}.is_institutional=institutional"] == [8]
        assert fact_map[f"{base}|{USER}.is_institutional=retail"] == [9]

    def test_registry_is_populated_in_place_for_the_caller(self, fintech_schema):
        """The registry is an in/out parameter -- the orchestration layer
        reuses it across shards.

        Catches: building a private registry internally and discarding it, so
        later shards lose every fork discovered by earlier ones.
        """
        registry = ForkKeyRegistry()
        registry.register_fork(
            ForkKey(table_name=USER, column_name="is_institutional"), ["institutional"]
        )
        analyze_cross_shard_constraints(
            distributions=_forked_pair(), schema=fintech_schema, registry=registry
        )
        assert registry.forks[
            ForkKey(table_name=USER, column_name="is_institutional")
        ] == ["institutional"]

    # ----------------------------------------------------------------- BUG 1
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "KNOWN BUG 1a (constraint_graph.py:759-769): _build_fork_registry "
            "skips every distribution whose if_condition is None -- i.e. it "
            "skips exactly the fork-DEFINING categorical, the only fact that "
            "actually states (table, column, categories). It should register "
            "ForkKey(USER, is_institutional) from dc.on.name/dc.column."
        ),
    )
    def test_fork_defining_categorical_registers_its_own_column_as_the_fork_key(self):
        registry = ForkKeyRegistry()
        cg._build_fork_registry([_fork_defining_categorical()], registry)
        assert registry.forks == {
            ForkKey(table_name=USER, column_name="is_institutional"): [
                "institutional",
                "retail",
            ]
        }

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "KNOWN BUG 1b (constraint_graph.py:752-769): because the fork key "
            "is never registered from the defining categorical, _resolve_branch "
            "returns None for both branches, they collapse to one flat_name, "
            "_merge_rich_bounds intersects [5,5] with [12,12], and BOTH forked "
            "parameters are excluded from square AND loose -- they vanish from "
            "the probe contract entirely."
        ),
    )
    def test_auto_discovered_fork_keeps_the_two_branches_separate(self, fintech_schema):
        report, _ = analyze_cross_shard_constraints(
            distributions=[_fork_defining_categorical()] + _forked_pair(),
            schema=fintech_schema,
        )
        base = _flat(PRODUCT, "column_distribution_param", "spread.lam")
        assert {
            f"{base}|{USER}.is_institutional=institutional",
            f"{base}|{USER}.is_institutional=retail",
        } <= set(report.square_variables)
        assert _over_vars(report) == set()

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "KNOWN BUG 1c (constraint_graph.py:764-769 via "
            "parse_if_condition_from_predicate): a conditional CATEGORICAL is "
            "registered under the IF-CONDITION's column with table_name='' -- "
            "the key '.is_institutional' names neither the fork's table nor "
            "the branch distribution's own column, and its category list is "
            "the BRANCH's categories, not the fork's."
        ),
    )
    def test_a_conditional_categorical_does_not_corrupt_the_registry(self):
        registry = ForkKeyRegistry()
        cg._build_fork_registry(
            [
                _dist(
                    "drawdown_phase",
                    "CATEGORICAL",
                    {"categories": ["open", "closed"]},
                    if_condition=_cmp("is_institutional", "=", "retail"),
                )
            ],
            registry,
        )
        assert (
            ForkKey(table_name="", column_name="is_institutional") not in registry.forks
        )

    # ------------------------------------------------------- NEW FINDING (B)
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "NEW BUG B (constraint_graph.py:300-311): _resolve_branch ignores "
            "the registry's answer. get_branches_for_condition returns [] for "
            "a value outside the fork's known category set, and the code "
            "returns a BranchTag anyway -- minting a branch-scoped variable "
            "for a branch that provably has no rows, which Stage 4 is then "
            "asked to parameterize."
        ),
    )
    def test_a_value_outside_the_forks_category_set_resolves_to_none(self):
        registry = ForkKeyRegistry()
        registry.register_fork(
            ForkKey(table_name=USER, column_name="is_institutional"),
            ["institutional", "retail"],
        )
        assert (
            cg._resolve_branch(_cmp("is_institutional", "=", "sovereign"), registry)
            is None
        )


# =========================================================================
# Known bugs 2 and 3, plus one new finding
# =========================================================================


class TestKnownBugs:
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "KNOWN BUG 2 (constraint_graph.py:591-598): _convert_cross_shard_"
            "constraints routes ANY single-table `structural` constraint to "
            "_cardinality_to_rich, so a per-row column bound (spread <= 10 on "
            "CREDIT_PRODUCT) fabricates a table_cardinality::row_count "
            "variable bounded above by 10. Routing must key off the ON tree "
            "actually being a COUNT (Aggregate/Fanout), not off table count."
        ),
    )
    def test_a_single_table_column_bound_is_not_a_row_count(self, fintech_schema):
        report, _ = analyze_cross_shard_constraints(
            structural=[
                _constraint(_cmp("spread", "<=", 10), category="structural", facts=[8])
            ],
            schema=fintech_schema,
        )
        assert _loose(report) == {
            _flat(PRODUCT, "column_range", "spread"): (None, 10.0)
        }

    # FIXED: _cardinality_to_rich now steps a strict inequality by a whole unit.
    # A row count needs no type lookup to know it is integral -- a table cannot
    # hold 5.000000001 rows -- so `row_count > 5` means 6, unconditionally.
    @pytest.mark.parametrize(
        "op, value, expected",
        [(">", 5, (6.0, None)), ("<", 10, (None, 9.0))],
    )
    def test_strict_inequality_on_a_row_count_steps_a_whole_unit(
        self, fintech_schema, op, value, expected
    ):
        grain = _grain(fintech_schema)
        variables, _ = cg._cardinality_to_rich(
            _constraint(_cmp("row_count", op, value), category="structural"),
            grain,
            disambiguator=0,
        )
        (var,) = variables
        assert (var.lower_bound, var.upper_bound) == expected

    # FIXED: _range_constraint_to_rich now consults the column's declared type
    # via _SchemaView.is_column_integral, so a strict inequality on an INTEGER
    # column steps a whole unit (6) instead of emitting 5.000000001, which the
    # column cannot hold and no integer solver could use.
    def test_strict_inequality_on_an_integer_column_steps_a_whole_unit(
        self, fintech_schema
    ):
        report, _ = analyze_cross_shard_constraints(
            logic=[_constraint(_cmp("maturity", ">", 5), facts=[9])],
            schema=fintech_schema,
        )
        assert _loose(report) == {
            _flat(PRODUCT, "column_range", "maturity"): (6.0, None)
        }

    # FIXED: _column_relative_bound reads the literal from whichever side holds
    # it and flips the operator when that side is the left, so `5 <= spread` and
    # `spread >= 5` now state the same bound instead of the former being dropped.
    def test_a_literal_on_the_left_still_yields_the_bound(self, fintech_schema):
        report, _ = analyze_cross_shard_constraints(
            logic=[
                _constraint(
                    RComparison(
                        op="<=",
                        left=RLiteral(value=5),
                        right=RColumnRef(name="spread"),
                    ),
                    facts=[8],
                )
            ],
            schema=fintech_schema,
        )
        assert _loose(report) == {_flat(PRODUCT, "column_range", "spread"): (5.0, None)}


# =========================================================================
# Canonicalization failures and predicate shapes with no branch meaning
# =========================================================================


class TestUnconvertibleInputs:
    @pytest.mark.parametrize(
        "kwargs_name",
        ["distributions", "structural", "logic", "derived"],
        ids=["distribution", "structural", "logic", "derived"],
    )
    def test_an_on_tree_that_cannot_be_canonicalized_contributes_nothing(
        self, fintech_schema, kwargs_name
    ):
        """Every one of the four conversion loops must skip a constraint whose
        ON tree names a table outside the schema -- there is no grain to scope
        its variable to.

        Catches: any loop that forgets the CanonicalizationFailure guard and
        proceeds to build a variable at a nonexistent grain, which would put a
        phantom table into Stage 4's probe contract.
        """
        absent = "ABSENT_TABLE"
        payload: Dict[str, Any] = {
            "distributions": [
                _dist("spread", "POISSON", {"lam": 2.0}, table=absent, facts=[1])
            ],
            "structural": [
                _constraint(
                    _cmp("row_count", "=", 5),
                    on=_count_star(absent),
                    category="structural",
                    facts=[2],
                )
            ],
            "logic": [
                _constraint(
                    _cmp("spread", ">=", 1), on=BaseTable(name=absent), facts=[3]
                )
            ],
            "derived": [
                DerivedColumnConstraint(
                    fact_references=[4],
                    target_table=absent,
                    target_column="ltv",
                    expression=RLiteral(value=1),
                    referenced_tables=[absent],
                )
            ],
        }
        report, fact_map = analyze_cross_shard_constraints(
            schema=fintech_schema, **{kwargs_name: payload[kwargs_name]}
        )
        assert fact_map == {}
        assert report.square_variables == []
        assert report.loose_variable_probes == []
        assert report.overconstrained_blocks == []

    def test_a_compound_predicate_is_not_a_branch(self, fintech_schema):
        """Only a bare equality names one branch of a fork; a conjunction
        does not, and must not be silently reduced to one of its operands.

        Catches: reaching into an RAnd's first operand to guess a branch,
        which would tag a variable with a branch narrower than the fact.
        """
        registry = ForkKeyRegistry()
        registry.register_fork(
            ForkKey(table_name=USER, column_name="is_institutional"),
            ["institutional", "retail"],
        )
        compound = RAnd(
            operands=[
                _cmp("is_institutional", "=", "retail"),
                _cmp("drawdown_phase", "=", "open"),
            ]
        )
        assert cg._resolve_branch(compound, registry) is None
        assert cg.parse_if_condition_from_predicate(compound) is None

    @pytest.mark.parametrize(
        "predicate, expected_op",
        [
            (_cmp("is_institutional", "=", "retail"), cg.Operator.EQ),
            (_cmp("is_institutional", "!=", "retail"), cg.Operator.NEQ),
            (_cmp("is_institutional", ">=", "retail"), cg.Operator.NEQ),
        ],
    )
    def test_parse_if_condition_maps_the_operator(self, predicate, expected_op):
        """Catches: collapsing every operator to EQ, which would make a
        negated fork condition register (and later resolve) as its own
        opposite.

        Also documents that any non-'='/'==' operator falls through to NEQ --
        '>=' against a string is nonsense, and reading it as a negation is a
        silent misinterpretation rather than a rejection.
        """
        parsed = cg.parse_if_condition_from_predicate(predicate)
        assert parsed is not None
        assert parsed.operator is expected_op
        assert parsed.values == ["retail"]

    @pytest.mark.parametrize(
        "predicate",
        [
            RComparison(
                op="=",
                left=RLiteral(value="retail"),
                right=RColumnRef(name="is_institutional"),
            ),
            _cmp("is_institutional", "=", 7),
            RComparison(
                op="=",
                left=RColumnRef(name="is_institutional"),
                right=RColumnRef(name="drawdown_phase"),
            ),
        ],
        ids=["literal_on_left", "numeric_literal", "column_on_both_sides"],
    )
    def test_parse_if_condition_rejects_non_categorical_shapes(self, predicate):
        """Catches: accepting a numeric or column-to-column comparison as a
        fork condition, which would register a ForkKey whose 'categories' are
        not category labels at all.
        """
        assert cg.parse_if_condition_from_predicate(predicate) is None

    def test_a_categorical_with_an_unparseable_if_condition_is_skipped(self):
        """Catches: registering a fork under a None/garbage key when the
        if_condition has no BranchCondition reading."""
        registry = ForkKeyRegistry()
        cg._build_fork_registry(
            [
                _dist(
                    "drawdown_phase",
                    "CATEGORICAL",
                    {"categories": ["open", "closed"]},
                    if_condition=RComparison(
                        op="=",
                        left=RColumnRef(name="is_institutional"),
                        right=RColumnRef(name="drawdown_phase"),
                    ),
                )
            ],
            registry,
        )
        assert registry.forks == {}
