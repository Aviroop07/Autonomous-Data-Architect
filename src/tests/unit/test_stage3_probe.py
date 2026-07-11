"""Unit tests for Stage 3's final probe-reporting entry point,
analyze_constraint_manifest() -- the boundary where Stage 3 stops:
surfacing determined values, flagging real infeasibility, and reporting
(never resolving) loose variables and bailed moment targets. See the
project memory stage3_stage4_division_of_labor for the underlying
principle this enforces.
"""

from __future__ import annotations

from src.pipeline.stage3.middleware.constraint_graph import (
    analyze_constraint_manifest,
)
from src.pipeline.stage3.models.constraints import (
    AggregationConstraint,
    ConstraintManifest,
    FanoutConstraint,
    GaussianDistribution,
    MomentTarget,
    PoissonDistribution,
    StatisticalManifest,
    StructuralManifest,
    TableCardinality,
)


class TestAnalyzeConstraintManifest:
    def test_square_and_loose_are_reported_correctly(self):
        manifest = ConstraintManifest(
            statistical=StatisticalManifest(
                distributions=[
                    GaussianDistribution(
                        table_name="ORDER",
                        column_name="shipping_cost",
                        mean=8,
                        std_dev=2,
                        fact_references=[1],
                    ),
                ]
            ),
            structural=StructuralManifest(
                cardinalities=[
                    TableCardinality(
                        table_name="CARRIER",
                        min_rows=1,
                        max_rows=8,
                        fact_references=[2],
                    )
                ]
            ),
        )
        report = analyze_constraint_manifest(manifest)

        assert set(report.square_variables) == {
            "ORDER.shipping_cost.mean",
            "ORDER.shipping_cost.std_dev",
        }
        loose_names = {p.variable_name for p in report.loose_variable_probes}
        assert loose_names == {"CARRIER.row_count"}
        probe = report.loose_variable_probes[0]
        assert (probe.lower_bound, probe.upper_bound) == (1.0, 8.0)
        assert probe.fact_references == [2]
        assert report.unresolved_moment_target_probes == []
        assert report.is_feasible is True

    def test_overconstrained_block_marks_infeasible(self):
        manifest = ConstraintManifest(
            statistical=StatisticalManifest(
                distributions=[
                    GaussianDistribution(
                        table_name="ORDER",
                        column_name="shipping_cost",
                        mean=8,
                        std_dev=2,
                        fact_references=[1],
                    ),
                    GaussianDistribution(
                        table_name="ORDER",
                        column_name="shipping_cost",
                        mean=10,
                        std_dev=2,
                        fact_references=[2],
                    ),
                ]
            )
        )
        report = analyze_constraint_manifest(manifest)

        assert report.is_feasible is False
        # Both mean (genuinely conflicting, 8 vs 10) AND std_dev (numerically
        # identical, 2 vs 2) show up as their own overconstrained blocks --
        # structural analysis alone can't distinguish harmless redundancy
        # from a real contradiction (design doc section 3/Q7).
        assert len(report.overconstrained_blocks) == 2
        flagged_variables = {
            v for block in report.overconstrained_blocks for v in block.variables
        }
        assert flagged_variables == {
            "ORDER.shipping_cost.mean",
            "ORDER.shipping_cost.std_dev",
        }

    def test_bailed_moment_target_becomes_a_probe_not_a_guess(self):
        """MAX has no closed form (section 4.4) -- must surface as a
        MomentTargetProbe carrying the original stated target, not vanish
        or get a fabricated resolution."""
        manifest = ConstraintManifest(
            statistical=StatisticalManifest(
                distributions=[
                    PoissonDistribution(
                        table_name="LINE_ITEM",
                        column_name="quantity",
                        lam=2,
                        fact_references=[10],
                    )
                ],
                moment_targets=[
                    MomentTarget(
                        table_name="ORDER",
                        column_name="max_quantity",
                        statistic="MEAN",
                        target_value=42,
                        fact_references=[11],
                    )
                ],
            ),
            structural=StructuralManifest(
                aggregations=[
                    AggregationConstraint(
                        parent_table="ORDER",
                        parent_column="max_quantity",
                        descendant_table="LINE_ITEM",
                        descendant_column="quantity",
                        operation="MAX",
                        fact_references=[12],
                    )
                ]
            ),
        )
        report = analyze_constraint_manifest(manifest)

        assert len(report.unresolved_moment_target_probes) == 1
        probe = report.unresolved_moment_target_probes[0]
        assert (probe.table_name, probe.column_name) == ("ORDER", "max_quantity")
        assert probe.target_value == 42
        assert probe.fact_references == [11]
        assert not any("max_quantity" in v for v in report.square_variables)

    def test_real_fact_chain_resolves_cleanly_with_no_probes_left(self):
        """The Q3 real-data reproduction (28/30/34/38/39) -- when everything
        resolves, there should be nothing left to probe."""
        manifest = ConstraintManifest(
            statistical=StatisticalManifest(
                distributions=[
                    PoissonDistribution(
                        table_name="LINE_ITEM",
                        column_name="quantity",
                        lam=2,
                        fact_references=[34],
                    ),
                    GaussianDistribution(
                        table_name="LINE_ITEM",
                        column_name="unit_price",
                        mean=25,
                        std_dev=5,
                        fact_references=[38],
                    ),
                ],
                moment_targets=[
                    MomentTarget(
                        table_name="ORDER",
                        column_name="order_total",
                        statistic="MEAN",
                        target_value=150,
                        fact_references=[30],
                    )
                ],
            ),
            structural=StructuralManifest(
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
                aggregations=[
                    AggregationConstraint(
                        parent_table="ORDER",
                        parent_column="order_total",
                        descendant_table="LINE_ITEM",
                        descendant_column="subtotal",
                        operation="SUM",
                        fact_references=[28],
                    )
                ],
            ),
        )
        # subtotal = quantity * unit_price, needed for the aggregation's
        # descendant to resolve.
        from src.pipeline.stage3.models.constraints import (
            CrossColumnLogic,
            LogicManifest,
        )

        manifest.structural.aggregations[0].descendant_column = "subtotal"
        manifest.logic = LogicManifest(
            cross_column_logic=[
                CrossColumnLogic(
                    table_context="LINE_ITEM",
                    then_enforcement="subtotal = quantity * unit_price",
                    fact_references=[39],
                )
            ]
        )

        report = analyze_constraint_manifest(manifest)

        assert report.unresolved_moment_target_probes == []
        assert "ORDER->LINE_ITEM.fanout_mean[order_number]" in report.square_variables
        assert report.loose_variable_probes == []
        assert report.is_feasible is True
