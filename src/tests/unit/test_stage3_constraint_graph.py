"""Unit tests for the taxonomy-v2-to-DOF-graph adapter
(src/pipeline/stage3/middleware/constraint_graph.py). Scope: distributions
(including Q3's moment-target derivation walk and Q4's conditional
fork-key expansion), cardinalities, and fanouts -- see the module
docstring for what's still deliberately not handled (UniqueConstraint,
FormatConstraint, conditional CrossColumnLogic outside the moment-target
walk).
"""

from __future__ import annotations

from src.pipeline.stage3.middleware.constraint_graph import (
    constraint_manifest_to_graph_nodes,
    distribution_to_graph_nodes,
    fanout_constraint_to_graph_nodes,
    statistical_manifest_to_graph_nodes,
    structural_manifest_to_graph_nodes,
    table_cardinality_to_graph_nodes,
)
from src.pipeline.stage3.models.constraints import (
    AggregationConstraint,
    BetaDistribution,
    CategoricalDistribution,
    ConstraintManifest,
    CrossColumnLogic,
    FanoutConstraint,
    GaussianDistribution,
    LogicManifest,
    MomentTarget,
    PoissonDistribution,
    StatisticalManifest,
    StructuralManifest,
    TableCardinality,
    UniformDistribution,
)
from src.util.algorithms.dof_graph import DOFGraph


class TestDistributionToGraphNodes:
    def test_gaussian_splits_into_two_independent_pins(self):
        """The core split rule: one fact stating two parameter values must
        become two Constraints, never one Constraint touching both."""
        dist = GaussianDistribution(
            table_name="ORDER",
            column_name="shipping_cost",
            mean=8,
            std_dev=2,
            fact_references=[42],
        )
        variables, constraints = distribution_to_graph_nodes(dist, disambiguator=0)

        assert {v.name for v in variables} == {
            "ORDER.shipping_cost.mean",
            "ORDER.shipping_cost.std_dev",
        }
        assert len(constraints) == 2
        assert all(len(c.variables) == 1 for c in constraints)
        assert all(c.fact_references == [42] for c in constraints)

    def test_poisson_pins_a_single_parameter(self):
        dist = PoissonDistribution(
            table_name="LINE_ITEM", column_name="quantity", lam=2, fact_references=[34]
        )
        variables, constraints = distribution_to_graph_nodes(dist, disambiguator=0)

        assert [v.name for v in variables] == ["LINE_ITEM.quantity.lam"]
        assert len(constraints) == 1

    def test_beta_and_uniform_use_their_own_parameter_names(self):
        beta = BetaDistribution(
            table_name="T", column_name="c", alpha=2, beta=5, fact_references=[1]
        )
        beta_vars, _ = distribution_to_graph_nodes(beta, disambiguator=0)
        assert {v.name for v in beta_vars} == {"T.c.alpha", "T.c.beta"}

        uniform = UniformDistribution(
            table_name="T",
            column_name="c",
            min_value=0,
            max_value=1,
            fact_references=[1],
        )
        uniform_vars, _ = distribution_to_graph_nodes(uniform, disambiguator=0)
        assert {v.name for v in uniform_vars} == {"T.c.min_value", "T.c.max_value"}

    def test_categorical_with_stated_probabilities_is_pinned(self):
        dist = CategoricalDistribution(
            table_name="CUSTOMER",
            column_name="loyalty_tier",
            categories=["Bronze", "Silver"],
            probabilities=[0.7, 0.3],
            fact_references=[8],
        )
        variables, constraints = distribution_to_graph_nodes(dist, disambiguator=0)

        assert [v.name for v in variables] == ["CUSTOMER.loyalty_tier.probabilities"]
        assert len(constraints) == 1

    def test_categorical_without_probabilities_is_unpinned(self):
        """Categories named but no weights given -- a real, common case in
        the brainstorm_report.md data (loyalty_tier, return reason)."""
        dist = CategoricalDistribution(
            table_name="CUSTOMER",
            column_name="loyalty_tier",
            categories=["Bronze", "Silver", "Gold", "Platinum"],
            fact_references=[8],
        )
        variables, constraints = distribution_to_graph_nodes(dist, disambiguator=0)

        assert [v.name for v in variables] == ["CUSTOMER.loyalty_tier.probabilities"]
        assert constraints == []


class TestStatisticalManifestToGraphNodes:
    def test_independent_facts_all_resolve_cleanly(self):
        manifest = StatisticalManifest(
            distributions=[
                GaussianDistribution(
                    table_name="ORDER",
                    column_name="shipping_cost",
                    mean=8,
                    std_dev=2,
                    fact_references=[42],
                ),
                PoissonDistribution(
                    table_name="LINE_ITEM",
                    column_name="quantity",
                    lam=2,
                    fact_references=[34],
                ),
                CategoricalDistribution(
                    table_name="CUSTOMER",
                    column_name="loyalty_tier",
                    categories=["Bronze"],
                    fact_references=[8],
                ),
            ]
        )
        variables, constraints = statistical_manifest_to_graph_nodes(manifest)
        result = DOFGraph(variables, constraints).classify()

        assert set(result.square_variables) == {
            "ORDER.shipping_cost.mean",
            "ORDER.shipping_cost.std_dev",
            "LINE_ITEM.quantity.lam",
        }
        assert result.loose_variables == ["CUSTOMER.loyalty_tier.probabilities"]
        assert result.overconstrained_blocks == []

    def test_two_facts_pinning_the_same_parameter_deduplicate_the_variable(self):
        """Real scenario: duplicate/conflicting facts about one column,
        e.g. from two shards' facts before a proper merge. Must not crash,
        must not silently pick a winner -- must surface as a structurally
        overconstrained block for a later numeric pass to adjudicate."""
        manifest = StatisticalManifest(
            distributions=[
                GaussianDistribution(
                    table_name="ORDER",
                    column_name="shipping_cost",
                    mean=8,
                    std_dev=2,
                    fact_references=[42],
                ),
                GaussianDistribution(
                    table_name="ORDER",
                    column_name="shipping_cost",
                    mean=10,
                    std_dev=2,
                    fact_references=[99],
                ),
            ]
        )
        variables, constraints = statistical_manifest_to_graph_nodes(manifest)

        # exactly one Variable per parameter, not one per fact
        assert len(variables) == 2
        mean_var = next(v for v in variables if v.name == "ORDER.shipping_cost.mean")
        assert mean_var.fact_references == [42, 99]

        # but both facts' Constraints survive independently
        assert len(constraints) == 4

        result = DOFGraph(variables, constraints).classify()
        flagged_variables = {
            v for b in result.overconstrained_blocks for v in b.variables
        }
        assert "ORDER.shipping_cost.mean" in flagged_variables
        assert "ORDER.shipping_cost.std_dev" in flagged_variables

    def test_empty_manifest_produces_empty_graph(self):
        variables, constraints = statistical_manifest_to_graph_nodes(
            StatisticalManifest()
        )
        assert variables == []
        assert constraints == []


class TestTableCardinalityToGraphNodes:
    def test_exact_target_is_pinned(self):
        cardinality = TableCardinality(
            table_name="CARRIER", min_rows=5, max_rows=5, fact_references=[999]
        )
        variable, constraints = table_cardinality_to_graph_nodes(
            cardinality, disambiguator=0
        )

        assert variable.name == "CARRIER.row_count"
        assert (variable.lower_bound, variable.upper_bound) == (5.0, 5.0)
        assert len(constraints) == 1

    def test_genuine_range_stays_unpinned_but_keeps_its_bounds(self):
        """Real fact 27: line items per order, [1,8] -- a range, not a
        target. Must stay loose but must not lose the bound information."""
        cardinality = TableCardinality(
            table_name="LINE_ITEM", min_rows=1, max_rows=8, fact_references=[27]
        )
        variable, constraints = table_cardinality_to_graph_nodes(
            cardinality, disambiguator=0
        )

        assert (variable.lower_bound, variable.upper_bound) == (1.0, 8.0)
        assert constraints == []


class TestFanoutConstraintToGraphNodes:
    def test_exact_target_is_pinned(self):
        fanout = FanoutConstraint(
            parent_table="WAREHOUSE",
            child_table="ORDER",
            foreign_key_columns=["warehouse_code"],
            min_fanout=10,
            max_fanout=10,
            fact_references=[1],
        )
        variable, constraints = fanout_constraint_to_graph_nodes(
            fanout, disambiguator=0
        )

        assert len(constraints) == 1

    def test_unbounded_max_stays_loose(self):
        """max_fanout is Optional -- min_fanout alone never pins an exact
        value."""
        fanout = FanoutConstraint(
            parent_table="ORDER",
            child_table="LINE_ITEM",
            foreign_key_columns=["order_number"],
            min_fanout=1,
            fact_references=[26],
        )
        variable, constraints = fanout_constraint_to_graph_nodes(
            fanout, disambiguator=0
        )

        assert variable.lower_bound == 1.0
        assert variable.upper_bound is None
        assert constraints == []

    def test_range_stays_loose(self):
        fanout = FanoutConstraint(
            parent_table="ORDER",
            child_table="LINE_ITEM",
            foreign_key_columns=["order_number"],
            min_fanout=1,
            max_fanout=8,
            fact_references=[27],
        )
        _, constraints = fanout_constraint_to_graph_nodes(fanout, disambiguator=0)
        assert constraints == []


class TestStructuralManifestToGraphNodes:
    def test_cardinalities_and_fanouts_combine_cleanly(self):
        manifest = StructuralManifest(
            cardinalities=[
                TableCardinality(
                    table_name="CARRIER", min_rows=5, max_rows=5, fact_references=[999]
                )
            ],
            fanouts=[
                FanoutConstraint(
                    parent_table="ORDER",
                    child_table="LINE_ITEM",
                    foreign_key_columns=["order_number"],
                    min_fanout=1,
                    max_fanout=8,
                    fact_references=[27],
                )
            ],
        )
        variables, constraints = structural_manifest_to_graph_nodes(manifest)
        result = DOFGraph(variables, constraints).classify()

        assert result.square_variables == ["CARRIER.row_count"]
        assert result.loose_variables == ["ORDER->LINE_ITEM.fanout_mean[order_number]"]

    def test_two_range_facts_about_the_same_table_tighten_the_bound(self):
        """Two facts giving overlapping ranges for the same table must
        intersect (tighten), not just pick one or crash."""
        manifest = StructuralManifest(
            cardinalities=[
                TableCardinality(
                    table_name="CARRIER", min_rows=5, max_rows=10, fact_references=[1]
                ),
                TableCardinality(
                    table_name="CARRIER", min_rows=8, max_rows=20, fact_references=[2]
                ),
            ]
        )
        variables, _ = structural_manifest_to_graph_nodes(manifest)

        assert len(variables) == 1
        variable = variables[0]
        assert (variable.lower_bound, variable.upper_bound) == (8.0, 10.0)
        assert variable.fact_references == [1, 2]


class TestConstraintManifestToGraphNodes:
    def test_combines_statistical_and_structural(self):
        manifest = ConstraintManifest(
            statistical=StatisticalManifest(
                distributions=[
                    GaussianDistribution(
                        table_name="ORDER",
                        column_name="shipping_cost",
                        mean=8,
                        std_dev=2,
                        fact_references=[42],
                    ),
                ]
            ),
            structural=StructuralManifest(
                cardinalities=[
                    TableCardinality(
                        table_name="CARRIER",
                        min_rows=5,
                        max_rows=5,
                        fact_references=[999],
                    )
                ],
            ),
        )
        variables, constraints = constraint_manifest_to_graph_nodes(manifest)
        result = DOFGraph(variables, constraints).classify()

        assert set(result.square_variables) == {
            "ORDER.shipping_cost.mean",
            "ORDER.shipping_cost.std_dev",
            "CARRIER.row_count",
        }


def test_q4_fork_registry_graph_expansion():
    cat_dist = CategoricalDistribution(
        table_name="CUSTOMER",
        column_name="loyalty_tier",
        categories=["Bronze", "Silver", "Gold", "Platinum"],
        fact_references=[1],
    )

    cross_logic = CrossColumnLogic(
        table_context="ORDER",
        if_condition="CUSTOMER.loyalty_tier = 'Platinum'",
        then_enforcement="shipping_cost = 0",
        fact_references=[41],
    )

    gauss_dist = GaussianDistribution(
        table_name="ORDER",
        column_name="shipping_cost",
        if_condition="CUSTOMER.loyalty_tier != 'Platinum'",
        mean=8.0,
        std_dev=2.0,
        fact_references=[42],
    )

    manifest = ConstraintManifest(
        statistical=StatisticalManifest(distributions=[cat_dist, gauss_dist]),
        logic=LogicManifest(cross_column_logic=[cross_logic]),
    )

    variables, constraints = constraint_manifest_to_graph_nodes(manifest)

    var_names = {v.name for v in variables}
    assert "ORDER.shipping_cost.mean|CUSTOMER.loyalty_tier=Platinum" in var_names
    assert "ORDER.shipping_cost.mean|CUSTOMER.loyalty_tier=Bronze" in var_names
    assert "ORDER.shipping_cost.mean|CUSTOMER.loyalty_tier=Silver" in var_names
    assert "ORDER.shipping_cost.mean|CUSTOMER.loyalty_tier=Gold" in var_names
    assert "ORDER.shipping_cost.std_dev|CUSTOMER.loyalty_tier=Bronze" in var_names

    cross_cons = [c for c in constraints if "pin_cross" in c.name]
    assert len(cross_cons) == 1
    assert (
        "ORDER.shipping_cost.mean|CUSTOMER.loyalty_tier=Platinum"
        in cross_cons[0].variables
    )


def test_q4_moment_target_bails_on_conditional_base():
    """A MomentTarget whose derivation chain bottoms out at a base column
    with ONLY conditional distribution facts must bail, not fabricate an
    unconditional mean from whichever branch happens to be listed first.

    Uses AVG deliberately, not SUM: AVG needs no FanoutConstraint match, so
    nothing else in the derivation walk can bail first for an unrelated
    reason -- this isolates the conditional-base-column check itself. (A
    prior version of this test used SUM with no FanoutConstraint in the
    manifest, which passed for the wrong reason: _resolve_aggregation bails
    on the missing-fanout check before ever reaching this column, so the
    test kept passing even when the conditional-base bug was still live.)
    """
    cat_dist = CategoricalDistribution(
        table_name="CUSTOMER",
        column_name="tier",
        categories=["A", "B"],
        fact_references=[1],
    )

    gauss_dist = GaussianDistribution(
        table_name="ORDER",
        column_name="shipping_cost",
        if_condition="CUSTOMER.tier = 'A'",
        mean=5.0,
        std_dev=1.0,
        fact_references=[2],
    )

    agg = AggregationConstraint(
        parent_table="CUSTOMER",
        parent_column="total_shipping",
        descendant_table="ORDER",
        descendant_column="shipping_cost",
        operation="AVG",
        fact_references=[3],
    )

    moment = MomentTarget(
        table_name="CUSTOMER",
        column_name="total_shipping",
        statistic="MEAN",
        target_value=100.0,
        fact_references=[4],
    )

    manifest = ConstraintManifest(
        statistical=StatisticalManifest(
            distributions=[cat_dist, gauss_dist], moment_targets=[moment]
        ),
        structural=StructuralManifest(aggregations=[agg]),
    )

    variables, constraints = constraint_manifest_to_graph_nodes(manifest)
    var_names = {v.name for v in variables}

    assert "ORDER.shipping_cost.mean" not in var_names
    assert "CUSTOMER.total_shipping" not in var_names
    # The moment_target constraint itself must never have been created --
    # the conditional distribution's own branch-suffixed variable is fine
    # (that's Q4's normal output), but nothing should reference the bare,
    # unconditional "ORDER.shipping_cost.mean" this bug used to fabricate.
    assert not any("moment_target" in c.name for c in constraints)
    assert "ORDER.shipping_cost.mean|CUSTOMER.tier=A" in var_names
