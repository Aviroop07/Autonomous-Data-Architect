"""Unit tests for the Q3 derivation-chain walk (STAGE3_PHASE2_DESIGN.md
section 4): resolving a MomentTarget on a derived column down to the base
Variables it depends on. Covers the real fact chain (28/30/34/38/39) and
each of section 4.4's four enumerated bail-out conditions.
"""

from __future__ import annotations

from src.pipeline.stage3.middleware.constraint_graph import (
    constraint_manifest_to_graph_nodes,
    moment_target_to_graph_nodes,
)
from src.pipeline.stage3.models.constraints import (
    AggregationConstraint,
    ConstraintManifest,
    CrossColumnLogic,
    FanoutConstraint,
    GaussianDistribution,
    LogicManifest,
    MomentTarget,
    PoissonDistribution,
    StatisticalManifest,
    StructuralManifest,
)
from src.util.algorithms.dof_graph import DOFGraph


def _order_line_item_manifest(*, max_fanout: float | None = 8) -> ConstraintManifest:
    """The real fact chain: order_total = SUM(line_item.subtotal) over the
    ORDER->LINE_ITEM fanout; subtotal = quantity * unit_price; both base
    columns independently pinned. Mirrors facts 27/28/34/38/39."""
    return ConstraintManifest(
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
                    max_fanout=max_fanout,
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
        logic=LogicManifest(
            cross_column_logic=[
                CrossColumnLogic(
                    table_context="LINE_ITEM",
                    then_enforcement="subtotal = quantity * unit_price",
                    fact_references=[39],
                )
            ]
        ),
    )


class TestRealDataChain:
    def test_reproduces_the_hand_built_prototype_equation(self):
        """fact 30 -> AggregationConstraint (28, SUM over fanout 27) ->
        E[order_total] = E[N_LI] * E[subtotal] -> subtotal is unconditional
        CrossColumnLogic (39, product of two base columns) -> exactly the
        equation the real_data_prototype.py hand-built as C5_moment_target."""
        manifest = _order_line_item_manifest()
        target = manifest.statistical.moment_targets[0]

        resolved = moment_target_to_graph_nodes(target, manifest, disambiguator=0)
        assert resolved is not None
        variables, constraints = resolved

        assert {v.name for v in variables} == {
            "ORDER->LINE_ITEM.fanout_mean[order_number]",
            "LINE_ITEM.quantity.lam",
            "LINE_ITEM.unit_price.mean",
        }
        assert len(constraints) == 1
        assert set(constraints[0].variables) == {v.name for v in variables}
        # The derived constraint's own provenance is the facts that justify
        # ITS existence (fanout 27, aggregation 28, target 30, formula 39) --
        # not the base columns' own distribution facts (34, 38), which are
        # already carried by those Variables' own pinning constraints.
        assert constraints[0].fact_references == [27, 28, 30, 39]

    def test_full_manifest_graphs_cleanly_and_resolves_the_fanout(self):
        """End to end: the moment target supplies the third equation needed
        to pin the otherwise-loose fanout mean, exactly like the multi
        -variable-elimination case validated in test_util_dof_graph.py."""
        manifest = _order_line_item_manifest(max_fanout=8)  # range, not pinned directly
        variables, constraints = constraint_manifest_to_graph_nodes(manifest)
        result = DOFGraph(variables, constraints).classify()

        assert "ORDER->LINE_ITEM.fanout_mean[order_number]" in result.square_variables
        assert result.overconstrained_blocks == []


class TestBailOutConditions:
    """The four triggers enumerated in section 4.4 -- each must leave the
    MomentTarget unresolved (None), not partially resolved or crashing."""

    def test_max_operation_bails(self):
        manifest = _order_line_item_manifest()
        manifest.structural.aggregations[0].operation = "MAX"
        target = manifest.statistical.moment_targets[0]

        assert moment_target_to_graph_nodes(target, manifest, disambiguator=0) is None

    def test_min_operation_bails(self):
        manifest = _order_line_item_manifest()
        manifest.structural.aggregations[0].operation = "MIN"
        target = manifest.statistical.moment_targets[0]

        assert moment_target_to_graph_nodes(target, manifest, disambiguator=0) is None

    def test_ambiguous_fanout_bails(self):
        """Two FanoutConstraints for the same (parent_table, child_table) --
        the walk must refuse to guess which one applies."""
        manifest = _order_line_item_manifest()
        manifest.structural.fanouts.append(
            FanoutConstraint(
                parent_table="ORDER",
                child_table="LINE_ITEM",
                foreign_key_columns=["order_number"],
                min_fanout=2,
                max_fanout=6,
                fact_references=[27],
            )
        )
        target = manifest.statistical.moment_targets[0]

        assert moment_target_to_graph_nodes(target, manifest, disambiguator=0) is None

    def test_cross_column_rhs_with_more_than_two_operands_bails(self):
        manifest = _order_line_item_manifest()
        manifest.logic.cross_column_logic[
            0
        ].then_enforcement = "subtotal = quantity * unit_price * discount"
        target = manifest.statistical.moment_targets[0]

        assert moment_target_to_graph_nodes(target, manifest, disambiguator=0) is None

    def test_cross_column_rhs_division_bails(self):
        manifest = _order_line_item_manifest()
        manifest.logic.cross_column_logic[
            0
        ].then_enforcement = "subtotal = quantity / unit_price"
        target = manifest.statistical.moment_targets[0]

        assert moment_target_to_graph_nodes(target, manifest, disambiguator=0) is None

    def test_non_mean_statistic_is_rejected_at_the_model_level(self):
        """MEAN is the only Literal accepted by the model itself -- variance
        /median targets can't even be constructed, let alone resolved."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MomentTarget(
                table_name="ORDER",
                column_name="order_total",
                statistic="VARIANCE",
                target_value=10,
                fact_references=[30],
            )


class TestDerivationBuildingBlocks:
    def test_avg_operation_skips_the_fanout(self):
        """AVG needs no fanout at all -- the sample mean already estimates
        the population mean, unlike SUM's Wald's-identity scaling."""
        manifest = _order_line_item_manifest()
        manifest.structural.aggregations[0].operation = "AVG"
        target = manifest.statistical.moment_targets[0]

        resolved = moment_target_to_graph_nodes(target, manifest, disambiguator=0)
        assert resolved is not None
        variables, _ = resolved
        assert {v.name for v in variables} == {
            "LINE_ITEM.quantity.lam",
            "LINE_ITEM.unit_price.mean",
        }

    def test_unconstrained_base_column_becomes_a_free_variable(self):
        """A column with no distribution fact at all still resolves the
        WALK -- it contributes a `.mean`-named Variable rather than a bail.
        With `unit_price` unpinned, the moment-target equation alone can't
        determine both it and the loose fanout range (1 equation, 2
        unknowns) -- both correctly stay loose, since the walk adds
        structure but doesn't manufacture missing equations."""
        manifest = _order_line_item_manifest()
        manifest.statistical.distributions = [
            d
            for d in manifest.statistical.distributions
            if d.column_name != "unit_price"
        ]
        target = manifest.statistical.moment_targets[0]

        resolved = moment_target_to_graph_nodes(target, manifest, disambiguator=0)
        assert resolved is not None
        variables, _ = resolved
        assert "LINE_ITEM.unit_price.mean" in {v.name for v in variables}

        variables_full, constraints_full = constraint_manifest_to_graph_nodes(manifest)
        result = DOFGraph(variables_full, constraints_full).classify()
        assert set(result.loose_variables) == {
            "LINE_ITEM.unit_price.mean",
            "ORDER->LINE_ITEM.fanout_mean[order_number]",
        }
        assert result.overconstrained_blocks == []
