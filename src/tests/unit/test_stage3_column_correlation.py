"""Unit tests for ColumnCorrelation (D7): the model's own validation, and
confirmation that it is deliberately excluded from the DOF graph -- see
constraints.py's ColumnCorrelation docstring and constraint_graph.py's
module docstring for why (not a DOF concept, consumed by Stage 4 instead).
"""

from __future__ import annotations

from src.pipeline.stage3.middleware.constraint_graph import (
    constraint_manifest_to_graph_nodes,
)
from src.pipeline.stage3.models.constraints import (
    ColumnCorrelation,
    ConstraintManifest,
    GaussianDistribution,
    StatisticalManifest,
)


class TestColumnCorrelationValidation:
    def test_valid_cross_table_correlation_has_no_errors(self):
        correlation = ColumnCorrelation(
            table_name_a="CUSTOMER",
            column_name_a="age",
            table_name_b="CUSTOMER",
            column_name_b="income",
            direction="POSITIVE",
            strength="MODERATE",
            fact_references=[50],
        )
        assert correlation._validate() == []

    def test_rejects_a_column_correlated_with_itself(self):
        correlation = ColumnCorrelation(
            table_name_a="ORDER",
            column_name_a="shipping_cost",
            table_name_b="ORDER",
            column_name_b="shipping_cost",
            direction="POSITIVE",
            strength="STRONG",
            fact_references=[50],
        )
        errors = correlation._validate()
        assert len(errors) == 1
        assert "cannot be correlated with itself" in errors[0]


class TestColumnCorrelationExcludedFromGraph:
    def test_correlations_produce_no_graph_nodes(self):
        """Not a DOF concept -- present alongside real distributions, the
        graph must contain exactly the distributions' own nodes and nothing
        derived from the correlation fact."""
        manifest = ConstraintManifest(
            statistical=StatisticalManifest(
                distributions=[
                    GaussianDistribution(
                        table_name="CUSTOMER",
                        column_name="age",
                        mean=40,
                        std_dev=10,
                        fact_references=[1],
                    ),
                ],
                correlations=[
                    ColumnCorrelation(
                        table_name_a="CUSTOMER",
                        column_name_a="age",
                        table_name_b="CUSTOMER",
                        column_name_b="income",
                        direction="POSITIVE",
                        strength="MODERATE",
                        fact_references=[50],
                    )
                ],
            )
        )
        variables, constraints = constraint_manifest_to_graph_nodes(manifest)

        assert {v.name for v in variables} == {
            "CUSTOMER.age.mean",
            "CUSTOMER.age.std_dev",
        }
        assert len(constraints) == 2
        assert not any("income" in v.name for v in variables)
