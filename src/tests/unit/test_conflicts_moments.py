"""Tests for src/util/constraint_model/conflicts/moments.py."""

from __future__ import annotations

import math
from typing import Literal, Optional

from src.util.schema_model.data_types import DataType
from src.util.schema_model.schema import Column, Schema, Table
from src.util.constraint_model.condition.cohesive import Distributed, DistributionFamily
from src.util.constraint_model.condition.expressions import (
    RColumnRef,
    RAggregateRef,
    RLiteral,
)
from src.util.constraint_model.condition.predicates import RComparison
from src.util.constraint_model.conflicts.moments import check_moment_conflicts
from src.util.constraint_model.constraint import Constraint
from src.util.constraint_model.relation.nodes import (
    Aggregate,
    AggregateFn,
    BaseTable,
    Filter,
)

_ComparisonOp = Literal["<", "<=", "=", "!=", ">=", ">"]


def _schema() -> Schema:
    return Schema(
        tables=[
            Table(
                name="ORDER",
                columns=[
                    Column(name="id", data_type=DataType.INTEGER),
                    Column(name="total", data_type=DataType.FLOAT),
                ],
                primary_key=["id"],
            )
        ]
    )


def _moment(
    fn: AggregateFn,
    op: _ComparisonOp,
    value: float,
    alias: str,
    fid: int,
    group_by: Optional[list[str]] = None,
) -> Constraint:
    return Constraint(
        relation=Aggregate(
            source=BaseTable(name="ORDER"),
            fn=fn,
            column="total",
            alias=alias,
            group_by=group_by,
        ),
        condition=RComparison(
            op=op, left=RAggregateRef(alias=alias), right=RLiteral(value=value)
        ),
        fact_references=[fid],
    )


def _dist(family: DistributionFamily, params: dict, fid: int) -> Constraint:
    return Constraint(
        relation=BaseTable(name="ORDER"),
        condition=Distributed(column="total", family=family, parameters=params),
        fact_references=[fid],
    )


class TestValueEquality:
    def test_disagreeing_avg_values_conflict(self):
        c1 = _moment("AVG", "=", 100, "a1", 1)
        c2 = _moment("AVG", "=", 200, "a2", 2)
        conflicts = check_moment_conflicts([c1, c2], _schema())
        assert len(conflicts) == 1
        assert conflicts[0].kind == "moment_value_mismatch"

    def test_agreeing_avg_values_no_conflict(self):
        c1 = _moment("AVG", "=", 100, "a1", 1)
        c2 = _moment("AVG", "=", 100, "a2", 2)
        assert check_moment_conflicts([c1, c2], _schema()) == []

    def test_different_aliases_still_compare_correctly(self):
        # aliases are deliberately different (as two independent extractors
        # would produce) -- must still be recognized as the same quantity.
        c1 = _moment("AVG", "=", 100, "avg_order_total", 1)
        c2 = _moment("AVG", "=", 999, "average_value", 2)
        conflicts = check_moment_conflicts([c1, c2], _schema())
        assert len(conflicts) == 1


class TestInequalityLogic:
    def test_contradictory_inequalities(self):
        c1 = _moment("AVG", ">", 100, "a1", 1)
        c2 = _moment("AVG", "<", 50, "a2", 2)
        conflicts = check_moment_conflicts([c1, c2], _schema())
        assert len(conflicts) == 1

    def test_consistent_inequalities_no_conflict(self):
        c1 = _moment("AVG", ">", 100, "a1", 1)
        c2 = _moment("AVG", "<", 200, "a2", 2)
        assert check_moment_conflicts([c1, c2], _schema()) == []

    def test_three_way_narrowing_still_consistent(self):
        c1 = _moment("AVG", ">", 100, "a1", 1)
        c2 = _moment("AVG", "<", 200, "a2", 2)
        c3 = _moment("AVG", ">=", 150, "a3", 3)
        assert check_moment_conflicts([c1, c2, c3], _schema()) == []

    def test_three_way_narrowing_becomes_infeasible(self):
        c1 = _moment("AVG", ">", 100, "a1", 1)
        c2 = _moment("AVG", "<", 200, "a2", 2)
        c3 = _moment("AVG", ">", 250, "a3", 3)  # incompatible with < 200
        conflicts = check_moment_conflicts([c1, c2, c3], _schema())
        assert len(conflicts) == 1

    def test_boundary_touching_open_intervals_is_empty(self):
        # x > 100 and x < 100 share no point at all.
        c1 = _moment("AVG", ">", 100, "a1", 1)
        c2 = _moment("AVG", "<", 100, "a2", 2)
        conflicts = check_moment_conflicts([c1, c2], _schema())
        assert len(conflicts) == 1

    def test_closed_boundary_touching_is_feasible(self):
        # x >= 100 and x <= 100 share exactly the point 100.
        c1 = _moment("AVG", ">=", 100, "a1", 1)
        c2 = _moment("AVG", "<=", 100, "a2", 2)
        assert check_moment_conflicts([c1, c2], _schema()) == []


class TestScopeExclusions:
    def test_sum_facts_not_cross_checked(self):
        c1 = _moment("SUM", "=", 100, "s1", 1)
        c2 = _moment("SUM", "=", 999999, "s2", 2)
        assert check_moment_conflicts([c1, c2], _schema()) == []

    def test_max_facts_not_cross_checked(self):
        c1 = _moment("MAX", "=", 100, "m1", 1)
        c2 = _moment("MAX", "=", 999999, "m2", 2)
        assert check_moment_conflicts([c1, c2], _schema()) == []

    def test_grouped_vs_ungrouped_never_compared(self):
        c1 = _moment("AVG", "=", 100, "a1", 1)  # whole table
        c2 = _moment("AVG", "=", 999999, "a2", 2, group_by=["id"])  # per-group
        assert check_moment_conflicts([c1, c2], _schema()) == []

    def test_same_group_by_facts_still_compare(self):
        c1 = _moment("AVG", "=", 100, "a1", 1, group_by=["id"])
        c2 = _moment("AVG", "=", 200, "a2", 2, group_by=["id"])
        conflicts = check_moment_conflicts([c1, c2], _schema())
        assert len(conflicts) == 1


class TestStddevVarianceConversion:
    def test_stddev_and_variance_share_the_same_bucket(self):
        c1 = _moment("STDDEV", "=", 10, "sd1", 1)  # variance = 100
        c2 = _moment("VARIANCE", "=", 100, "v1", 2)
        assert check_moment_conflicts([c1, c2], _schema()) == []

    def test_stddev_and_variance_disagree(self):
        c1 = _moment("STDDEV", "=", 10, "sd1", 1)  # variance = 100
        c2 = _moment("VARIANCE", "=", 400, "v1", 2)  # would need stddev=20
        conflicts = check_moment_conflicts([c1, c2], _schema())
        assert len(conflicts) == 1


class TestDistributedCrossCheck:
    def test_gaussian_mean_mismatch(self):
        c1 = _dist("GAUSSIAN", {"mean": 100, "std_dev": 10}, 1)
        c2 = _moment("AVG", "=", 500, "a1", 2)
        conflicts = check_moment_conflicts([c1, c2], _schema())
        assert len(conflicts) == 1
        assert conflicts[0].kind == "moment_vs_distributed_mismatch"

    def test_gaussian_variance_mismatch(self):
        c1 = _dist("GAUSSIAN", {"mean": 100, "std_dev": 10}, 1)  # variance = 100
        c2 = _moment("VARIANCE", "=", 5000, "v1", 2)
        conflicts = check_moment_conflicts([c1, c2], _schema())
        assert len(conflicts) == 1
        assert conflicts[0].kind == "moment_vs_distributed_mismatch"

    def test_poisson_mean_equals_lambda(self):
        c1 = _dist("POISSON", {"lam": 7.5}, 1)
        c2 = _moment("AVG", "=", 7.5, "a1", 2)
        assert check_moment_conflicts([c1, c2], _schema()) == []

    def test_uniform_mean_and_variance(self):
        c1 = _dist(
            "UNIFORM", {"min_value": 0, "max_value": 10}, 1
        )  # mean=5, var=100/12
        c2 = _moment("AVG", "=", 5, "a1", 2)
        c3 = _moment("VARIANCE", "=", 100 / 12, "v1", 3)
        assert check_moment_conflicts([c1, c2, c3], _schema()) == []

    def test_beta_mean_and_variance(self):
        alpha, beta = 2.0, 3.0
        mean = alpha / (alpha + beta)
        variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        c1 = _dist("BETA", {"alpha": alpha, "beta": beta}, 1)
        c2 = _moment("AVG", "=", mean, "a1", 2)
        c3 = _moment("VARIANCE", "=", variance, "v1", 3)
        assert check_moment_conflicts([c1, c2, c3], _schema()) == []

    def test_log_normal_correct_actual_mean_no_conflict(self):
        mu, sigma = 4.0, 0.5
        actual_mean = math.exp(mu + sigma**2 / 2)
        c1 = _dist("LOG_NORMAL", {"mean": mu, "std_dev": sigma}, 1)
        c2 = _moment("AVG", "=", round(actual_mean, 4), "a1", 2)
        assert check_moment_conflicts([c1, c2], _schema()) == []

    def test_log_normal_naive_mu_as_mean_is_wrong(self):
        # a common misextraction: treating the underlying-normal mu as the
        # log-normal's own actual mean -- must be caught, not silently accepted.
        mu, sigma = 4.0, 0.5
        c1 = _dist("LOG_NORMAL", {"mean": mu, "std_dev": sigma}, 1)
        c2 = _moment("AVG", "=", mu, "a1", 2)
        conflicts = check_moment_conflicts([c1, c2], _schema())
        assert len(conflicts) == 1

    def test_log_normal_variance_formula(self):
        mu, sigma = 2.0, 0.3
        actual_variance = (math.exp(sigma**2) - 1) * math.exp(2 * mu + sigma**2)
        c1 = _dist("LOG_NORMAL", {"mean": mu, "std_dev": sigma}, 1)
        c2 = _moment("VARIANCE", "=", round(actual_variance, 3), "v1", 2)
        assert check_moment_conflicts([c1, c2], _schema()) == []

    def test_categorical_has_no_mean_never_cross_checked(self):
        c1 = _dist("CATEGORICAL", {"categories": ["a", "b"]}, 1)
        c2 = _moment("AVG", "=", 999999, "a1", 2)
        # CATEGORICAL contributes no MEAN observation at all -- nothing to compare.
        assert check_moment_conflicts([c1, c2], _schema()) == []

    def test_partial_distributed_parameters_contribute_no_observation(self):
        # GAUSSIAN with only mean stated (no std_dev) -- MEAN observation
        # still derivable; VARIANCE is not, and must not crash or fabricate one.
        c1 = _dist("GAUSSIAN", {"mean": 100}, 1)
        c2 = _moment("AVG", "=", 100, "a1", 2)
        c3 = _moment("VARIANCE", "=", 999999, "v1", 3)
        assert check_moment_conflicts([c1, c2, c3], _schema()) == []


class TestThreeOrMoreOverlappingFacts:
    def test_three_facts_two_moment_one_distributed_all_consistent(self):
        c1 = _dist("GAUSSIAN", {"mean": 150, "std_dev": 20}, 1)
        c2 = _moment("AVG", "=", 150, "a1", 2)
        c3 = _moment("AVG", ">", 100, "a2", 3)
        assert check_moment_conflicts([c1, c2, c3], _schema()) == []


class TestFilterConditions:
    def test_different_filters_on_same_base_do_not_conflict(self):
        """Two AVG facts on the same column but with mutually exclusive filter
        conditions describe different populations (disjoint row sets) and must
        NOT be reported as a conflict. Regression: _moment_population_key was
        dropping filter_conditions, so both produced the same grouping key."""
        c1 = Constraint(
            relation=Filter(
                source=Aggregate(
                    source=BaseTable(name="ORDER"),
                    fn="AVG",
                    column="total",
                    alias="a1",
                ),
                condition=RComparison(
                    op="=",
                    left=RColumnRef(name="tier"),
                    right=RLiteral(value="premium"),
                ),
            ),
            condition=RComparison(
                op="=", left=RAggregateRef(alias="a1"), right=RLiteral(value=100)
            ),
            fact_references=[1],
        )
        c2 = Constraint(
            relation=Filter(
                source=Aggregate(
                    source=BaseTable(name="ORDER"),
                    fn="AVG",
                    column="total",
                    alias="a2",
                ),
                condition=RComparison(
                    op="=",
                    left=RColumnRef(name="tier"),
                    right=RLiteral(value="standard"),
                ),
            ),
            condition=RComparison(
                op="=", left=RAggregateRef(alias="a2"), right=RLiteral(value=200)
            ),
            fact_references=[2],
        )
        assert check_moment_conflicts([c1, c2], _schema()) == []

    def test_same_filter_on_same_base_still_conflicts(self):
        """Two AVG facts at the SAME filtered population that disagree ARE
        a conflict -- the fix must not make all filtered facts incomparable."""
        c1 = Constraint(
            relation=Filter(
                source=Aggregate(
                    source=BaseTable(name="ORDER"),
                    fn="AVG",
                    column="total",
                    alias="a1",
                ),
                condition=RComparison(
                    op="=",
                    left=RColumnRef(name="tier"),
                    right=RLiteral(value="premium"),
                ),
            ),
            condition=RComparison(
                op="=", left=RAggregateRef(alias="a1"), right=RLiteral(value=100)
            ),
            fact_references=[1],
        )
        c2 = Constraint(
            relation=Filter(
                source=Aggregate(
                    source=BaseTable(name="ORDER"),
                    fn="AVG",
                    column="total",
                    alias="a2",
                ),
                condition=RComparison(
                    op="=",
                    left=RColumnRef(name="tier"),
                    right=RLiteral(value="premium"),
                ),
            ),
            condition=RComparison(
                op="=", left=RAggregateRef(alias="a2"), right=RLiteral(value=200)
            ),
            fact_references=[2],
        )
        conflicts = check_moment_conflicts([c1, c2], _schema())
        assert len(conflicts) == 1

    def test_three_facts_one_outlier_flagged(self):
        c1 = _dist("GAUSSIAN", {"mean": 150, "std_dev": 20}, 1)
        c2 = _moment("AVG", "=", 150, "a1", 2)
        c3 = _moment("AVG", "=", 999999, "a2", 3)
        conflicts = check_moment_conflicts([c1, c2, c3], _schema())
        # every pair including fact 3 disagrees -- one merged empty-interval conflict for the group.
        assert len(conflicts) == 1
        assert set(conflicts[0].involved_fact_references) == {1, 2, 3}
