"""Unit tests for cross-shard constraint models (src/pipeline/stage3/models/cross_shard.py).

Covers: Constraint, DistributionConstraint (family-specific parameter
validation), DerivedColumnConstraint, and all three extraction output
wrappers.
"""

from __future__ import annotations

import pytest

from src.pipeline.stage3.models.condition_nodes import (
    RArithmetic,
    RBetween,
    RColumnRef,
    RComparison,
    RLiteral,
)
from src.pipeline.stage3.models.cross_shard import (
    Constraint,
    DerivedColumnConstraint,
    DistributionConstraint,
    LogicExtractionOutput,
    StatisticalExtractionOutput,
    StructuralExtractionOutput,
)
from src.pipeline.stage3.models.on_nodes import (
    ONAggregate,
    ONBaseTable,
    ONJoin,
    JoinCondition,
)


# =========================================================================
# Constraint
# =========================================================================


class TestConstraint:
    def _make(self, **overrides):
        defaults = dict(
            fact_references=[1],
            on=ONBaseTable(name="ORDER"),
            condition=RComparison(
                op="=",
                left=RColumnRef(name="status"),
                right=RLiteral(value="active"),
            ),
            category="logic",
        )
        defaults.update(overrides)
        return Constraint(**defaults)

    def test_valid_construction(self):
        c = self._make()
        assert c.fact_references == [1]
        assert c.category == "logic"
        assert c.severity == "hard"

    def test_soft_severity(self):
        c = self._make(severity="soft")
        assert c.severity == "soft"

    def test_empty_fact_references_rejected(self):
        with pytest.raises(Exception):
            self._make(fact_references=[])

    def test_duplicate_fact_references_rejected(self):
        with pytest.raises(Exception):
            self._make(fact_references=[1, 1])

    def test_rename_optional(self):
        c = self._make(rename=None)
        assert c.rename is None

    def test_rename_dict(self):
        c = self._make(rename={"SUM(x)": "total_x"})
        assert c.rename["SUM(x)"] == "total_x"

    def test_validate_with_on_join(self):
        on = ONJoin(
            left=ONBaseTable(name="A"),
            right=ONBaseTable(name="B"),
            on=[JoinCondition(left="A.id", right="B.id")],
        )
        c = self._make(on=on)
        errors = c._validate()
        assert errors == []

    def test_all_categories(self):
        for cat in ["statistical", "structural", "logic", "temporal", "derived"]:
            c = self._make(category=cat)
            assert c.category == cat


# =========================================================================
# DistributionConstraint
# =========================================================================


class TestDistributionConstraint:
    def _make_gaussian(self, **overrides):
        defaults = dict(
            fact_references=[1],
            on=ONBaseTable(name="ORDER"),
            column="shipping_cost",
            family="GAUSSIAN",
            parameters={"mean": 8.0, "std_dev": 2.0},
        )
        defaults.update(overrides)
        return DistributionConstraint(**defaults)

    def test_valid_gaussian(self):
        dc = self._make_gaussian()
        assert dc.family == "GAUSSIAN"
        assert dc._validate() == []

    def test_gaussian_negative_std_dev_rejected(self):
        with pytest.raises(Exception):
            self._make_gaussian(parameters={"mean": 8, "std_dev": -1})

    def test_gaussian_zero_std_dev_rejected(self):
        with pytest.raises(Exception):
            self._make_gaussian(parameters={"mean": 8, "std_dev": 0})

    def test_gaussian_huge_std_dev_relative_to_mean_rejected(self):
        with pytest.raises(Exception):
            self._make_gaussian(parameters={"mean": 1, "std_dev": 20})

    def test_gaussian_missing_param_rejected(self):
        with pytest.raises(Exception):
            self._make_gaussian(parameters={"mean": 8})

    def test_valid_poisson(self):
        dc = DistributionConstraint(
            fact_references=[1],
            on=ONBaseTable(name="LINE_ITEM"),
            column="quantity",
            family="POISSON",
            parameters={"lam": 2.0},
        )
        assert dc._validate() == []

    def test_poisson_negative_lambda_rejected(self):
        with pytest.raises(Exception):
            DistributionConstraint(
                fact_references=[1],
                on=ONBaseTable(name="T"),
                column="x",
                family="POISSON",
                parameters={"lam": -1},
            )

    def test_valid_beta(self):
        dc = DistributionConstraint(
            fact_references=[1],
            on=ONBaseTable(name="T"),
            column="score",
            family="BETA",
            parameters={"alpha": 2.0, "beta": 5.0},
        )
        assert dc._validate() == []

    def test_beta_negative_params_rejected(self):
        with pytest.raises(Exception):
            DistributionConstraint(
                fact_references=[1],
                on=ONBaseTable(name="T"),
                column="x",
                family="BETA",
                parameters={"alpha": -1, "beta": 5},
            )

    def test_valid_categorical(self):
        dc = DistributionConstraint(
            fact_references=[1],
            on=ONBaseTable(name="T"),
            column="color",
            family="CATEGORICAL",
            parameters={"categories": ["R", "G", "B"]},
        )
        assert dc._validate() == []

    def test_categorical_with_probabilities(self):
        dc = DistributionConstraint(
            fact_references=[1],
            on=ONBaseTable(name="T"),
            column="color",
            family="CATEGORICAL",
            parameters={
                "categories": ["R", "G", "B"],
                "probabilities": [0.5, 0.3, 0.2],
            },
        )
        assert dc._validate() == []

    def test_categorical_probs_dont_sum_to_one_rejected(self):
        with pytest.raises(Exception):
            DistributionConstraint(
                fact_references=[1],
                on=ONBaseTable(name="T"),
                column="x",
                family="CATEGORICAL",
                parameters={
                    "categories": ["A", "B"],
                    "probabilities": [0.5, 0.6],
                },
            )

    def test_categorical_empty_categories_rejected(self):
        with pytest.raises(Exception):
            DistributionConstraint(
                fact_references=[1],
                on=ONBaseTable(name="T"),
                column="x",
                family="CATEGORICAL",
                parameters={"categories": []},
            )

    def test_valid_uniform(self):
        dc = DistributionConstraint(
            fact_references=[1],
            on=ONBaseTable(name="T"),
            column="x",
            family="UNIFORM",
            parameters={"min_value": 0.0, "max_value": 1.0},
        )
        assert dc._validate() == []

    def test_uniform_min_gt_max_rejected(self):
        with pytest.raises(Exception):
            DistributionConstraint(
                fact_references=[1],
                on=ONBaseTable(name="T"),
                column="x",
                family="UNIFORM",
                parameters={"min_value": 1.0, "max_value": 0.0},
            )

    def test_valid_log_normal(self):
        dc = DistributionConstraint(
            fact_references=[1],
            on=ONBaseTable(name="T"),
            column="income",
            family="LOG_NORMAL",
            parameters={"mean": 10.0, "std_dev": 1.5},
        )
        assert dc._validate() == []

    def test_log_normal_negative_std_dev_rejected(self):
        with pytest.raises(Exception):
            DistributionConstraint(
                fact_references=[1],
                on=ONBaseTable(name="T"),
                column="x",
                family="LOG_NORMAL",
                parameters={"mean": 10, "std_dev": -1},
            )

    def test_with_if_condition(self):
        cond = RComparison(
            op="=",
            left=RColumnRef(name="membership"),
            right=RLiteral(value="gold"),
        )
        dc = DistributionConstraint(
            fact_references=[1],
            on=ONBaseTable(name="T"),
            column="discount",
            family="GAUSSIAN",
            parameters={"mean": 0.1, "std_dev": 0.05},
            if_condition=cond,
        )
        assert dc.if_condition is not None
        assert dc._validate() == []

    def test_duplicate_fact_references_rejected(self):
        with pytest.raises(Exception):
            self._make_gaussian(fact_references=[1, 1])


# =========================================================================
# DerivedColumnConstraint
# =========================================================================


class TestDerivedColumnConstraint:
    def _make(self):
        return DerivedColumnConstraint(
            fact_references=[10],
            target_table="ORDER",
            target_column="line_total",
            expression=RArithmetic(
                op="*",
                left=RColumnRef(name="quantity"),
                right=RColumnRef(name="unit_price"),
            ),
            referenced_tables=["ORDER", "LINEITEM"],
        )

    def test_valid_construction(self):
        dc = self._make()
        assert dc.target_table == "ORDER"
        assert dc.target_column == "line_total"
        assert len(dc.referenced_tables) == 2
        assert dc._validate() == []

    def test_empty_fact_references_rejected(self):
        with pytest.raises(Exception):
            DerivedColumnConstraint(
                fact_references=[],
                target_table="T",
                target_column="x",
                expression=RColumnRef(name="y"),
                referenced_tables=["T"],
            )

    def test_empty_referenced_tables_rejected(self):
        with pytest.raises(Exception):
            DerivedColumnConstraint(
                fact_references=[1],
                target_table="T",
                target_column="x",
                expression=RColumnRef(name="y"),
                referenced_tables=[],
            )

    def test_nested_arithmetic(self):
        dc = DerivedColumnConstraint(
            fact_references=[5],
            target_table="ORDER",
            target_column="discounted_total",
            expression=RArithmetic(
                op="*",
                left=RArithmetic(
                    op="+",
                    left=RColumnRef(name="quantity"),
                    right=RColumnRef(name="tax"),
                ),
                right=RColumnRef(name="unit_price"),
            ),
            referenced_tables=["ORDER"],
        )
        assert dc._validate() == []


# =========================================================================
# Extraction output wrappers
# =========================================================================


class TestStatisticalExtractionOutput:
    def test_empty(self):
        out = StatisticalExtractionOutput()
        assert out.distributions == []
        assert out.moment_targets == []
        assert out.correlations == []

    def test_with_distribution(self):
        out = StatisticalExtractionOutput(
            distributions=[
                DistributionConstraint(
                    fact_references=[1],
                    on=ONBaseTable(name="T"),
                    column="x",
                    family="GAUSSIAN",
                    parameters={"mean": 50, "std_dev": 10},
                )
            ]
        )
        assert len(out.distributions) == 1
        assert out.distributions[0].family == "GAUSSIAN"


class TestStructuralExtractionOutput:
    def test_empty(self):
        out = StructuralExtractionOutput()
        assert out.constraints == []


class TestLogicExtractionOutput:
    def test_empty(self):
        out = LogicExtractionOutput()
        assert out.constraints == []
