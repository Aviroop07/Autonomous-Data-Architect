"""Tests for src/util/constraint_model/conflicts/population_reconcile.py."""

from __future__ import annotations

from src.pipeline.stage2.models.data_types import DataType
from src.pipeline.stage2.models.schema import Column, Schema, Table
from src.util.constraint_model.condition.expressions import (
    RAggregateRef,
    RColumnRef,
    RLiteral,
)
from src.util.constraint_model.condition.predicates import RComparison
from src.util.constraint_model.conflicts.population_reconcile import (
    _feasible_for_some_p,
    check_population_reconciliation,
)
from src.util.constraint_model.constraint import Constraint
from src.util.constraint_model.relation.nodes import (
    Aggregate,
    AggregateFn,
    BaseTable,
    Filter,
    RelationUnion,
)


def _schema() -> Schema:
    return Schema(
        tables=[
            Table(
                name="ORDER",
                columns=[
                    Column(name="id", data_type=DataType.INTEGER),
                    Column(name="total", data_type=DataType.FLOAT),
                    Column(name="quantity", data_type=DataType.FLOAT),
                    Column(name="region", data_type=DataType.VARCHAR),
                    Column(name="status", data_type=DataType.VARCHAR),
                ],
                primary_key=["id"],
            )
        ]
    )


def _moment(
    relation: "RelationUnion",
    alias: str,
    value: float,
    fid: int,
    fn: AggregateFn = "AVG",
    column: str = "total",
) -> Constraint:
    return Constraint(
        relation=Aggregate(source=relation, fn=fn, column=column, alias=alias),
        condition=RComparison(
            op="=", left=RAggregateRef(alias=alias), right=RLiteral(value=value)
        ),
        fact_references=[fid],
    )


def _us_filter():
    return Filter(
        source=BaseTable(name="ORDER"),
        condition=RComparison(
            op="=", left=RColumnRef(name="region"), right=RLiteral(value="US")
        ),
    )


def _non_us_filter():
    return Filter(
        source=BaseTable(name="ORDER"),
        condition=RComparison(
            op="!=", left=RColumnRef(name="region"), right=RLiteral(value="US")
        ),
    )


def _shipped_filter():
    return Filter(
        source=BaseTable(name="ORDER"),
        condition=RComparison(
            op="=", left=RColumnRef(name="status"), right=RLiteral(value="shipped")
        ),
    )


def _unshipped_filter():
    return Filter(
        source=BaseTable(name="ORDER"),
        condition=RComparison(
            op="!=", left=RColumnRef(name="status"), right=RLiteral(value="shipped")
        ),
    )


class TestExhaustivePartitionMean:
    def test_consistent_partition_no_conflict(self):
        whole = _moment(BaseTable(name="ORDER"), "w", 150, 1)
        us = _moment(_us_filter(), "us", 200, 2)
        non_us = _moment(_non_us_filter(), "nu", 100, 3)
        assert check_population_reconciliation([whole, us, non_us], _schema()) == []

    def test_p_out_of_range_conflicts(self):
        whole = _moment(BaseTable(name="ORDER"), "w", 300, 1)
        us = _moment(_us_filter(), "us", 200, 2)
        non_us = _moment(_non_us_filter(), "nu", 100, 3)
        conflicts = check_population_reconciliation([whole, us, non_us], _schema())
        assert len(conflicts) == 1
        assert conflicts[0].kind == "population_reconciliation_infeasible"
        assert set(conflicts[0].involved_fact_references) == {1, 2, 3}

    def test_p_negative_conflicts(self):
        # mu_whole below BOTH subset means -> p would need to be negative
        whole = _moment(BaseTable(name="ORDER"), "w", 50, 1)
        us = _moment(_us_filter(), "us", 200, 2)
        non_us = _moment(_non_us_filter(), "nu", 100, 3)
        conflicts = check_population_reconciliation([whole, us, non_us], _schema())
        assert len(conflicts) == 1

    def test_identical_subset_and_complement_means_are_underdetermined(self):
        # can't solve for p when the two groups have the same mean -- no
        # conflict should be manufactured from an unsolvable system.
        whole = _moment(BaseTable(name="ORDER"), "w", 999999, 1)
        us = _moment(_us_filter(), "us", 150, 2)
        non_us = _moment(_non_us_filter(), "nu", 150, 3)
        assert check_population_reconciliation([whole, us, non_us], _schema()) == []

    def test_mean_and_variance_facts_come_from_separate_atomic_constraints(self):
        # the realistic case: mean and variance for the SAME population are
        # always separately-extracted atomic facts, never one fact object.
        whole_mean = _moment(BaseTable(name="ORDER"), "w_m", 150, 1)
        whole_var = _moment(BaseTable(name="ORDER"), "w_v", 2600, 2, fn="VARIANCE")
        us_mean = _moment(_us_filter(), "us_m", 200, 3)
        us_var = _moment(_us_filter(), "us_v", 100, 4, fn="VARIANCE")
        nonus_mean = _moment(_non_us_filter(), "nu_m", 100, 5)
        nonus_var = _moment(_non_us_filter(), "nu_v", 100, 6, fn="VARIANCE")
        assert (
            check_population_reconciliation(
                [whole_mean, whole_var, us_mean, us_var, nonus_mean, nonus_var],
                _schema(),
            )
            == []
        )

    def test_variance_mismatch_given_determined_p(self):
        whole_mean = _moment(BaseTable(name="ORDER"), "w_m", 150, 1)
        whole_var = _moment(
            BaseTable(name="ORDER"), "w_v", 999, 2, fn="VARIANCE"
        )  # wrong, should be ~2600
        us_mean = _moment(_us_filter(), "us_m", 200, 3)
        us_var = _moment(_us_filter(), "us_v", 100, 4, fn="VARIANCE")
        nonus_mean = _moment(_non_us_filter(), "nu_m", 100, 5)
        nonus_var = _moment(_non_us_filter(), "nu_v", 100, 6, fn="VARIANCE")
        conflicts = check_population_reconciliation(
            [whole_mean, whole_var, us_mean, us_var, nonus_mean, nonus_var], _schema()
        )
        assert len(conflicts) == 1
        assert "2600" in conflicts[0].detail


class TestSingleSubsetAnalyticFeasibility:
    def test_positive_whole_variance_always_feasible(self):
        assert (
            _feasible_for_some_p(mu_w=150, var_w=500, mu_s=1_000_000, var_s=1) is True
        )

    def test_zero_whole_variance_identical_subset_feasible(self):
        assert _feasible_for_some_p(mu_w=150, var_w=0, mu_s=150, var_s=0) is True

    def test_zero_whole_variance_differing_subset_mean_infeasible(self):
        assert _feasible_for_some_p(mu_w=150, var_w=0, mu_s=200, var_s=0) is False

    def test_zero_whole_variance_differing_subset_variance_infeasible(self):
        assert _feasible_for_some_p(mu_w=150, var_w=0, mu_s=150, var_s=5) is False

    def test_negative_whole_variance_always_infeasible(self):
        assert _feasible_for_some_p(mu_w=150, var_w=-1, mu_s=150, var_s=0) is False

    def test_end_to_end_zero_variance_contradiction(self):
        whole_mean = _moment(BaseTable(name="ORDER"), "w_m", 150, 1)
        whole_var = _moment(BaseTable(name="ORDER"), "w_v", 0, 2, fn="VARIANCE")
        us_mean = _moment(_us_filter(), "us_m", 200, 3)
        us_var = _moment(_us_filter(), "us_v", 0, 4, fn="VARIANCE")
        conflicts = check_population_reconciliation(
            [whole_mean, whole_var, us_mean, us_var], _schema()
        )
        assert len(conflicts) == 1
        assert conflicts[0].kind == "population_reconciliation_infeasible"

    def test_end_to_end_extreme_gap_still_feasible(self):
        whole_mean = _moment(BaseTable(name="ORDER"), "w_m", 150, 1)
        whole_var = _moment(BaseTable(name="ORDER"), "w_v", 500, 2, fn="VARIANCE")
        us_mean = _moment(_us_filter(), "us_m", 1_000_000, 3)
        us_var = _moment(_us_filter(), "us_v", 1, 4, fn="VARIANCE")
        assert (
            check_population_reconciliation(
                [whole_mean, whole_var, us_mean, us_var], _schema()
            )
            == []
        )


class TestUnrelatedFactsDoNotInterfere:
    def test_facts_on_a_different_table_are_ignored(self):
        other_schema = Schema(
            tables=[
                Table(
                    name="ORDER",
                    columns=[
                        Column(name="id", data_type=DataType.INTEGER),
                        Column(name="total", data_type=DataType.FLOAT),
                        Column(name="region", data_type=DataType.VARCHAR),
                    ],
                    primary_key=["id"],
                ),
                Table(
                    name="CUSTOMER",
                    columns=[
                        Column(name="id", data_type=DataType.INTEGER),
                        Column(name="age", data_type=DataType.INTEGER),
                    ],
                    primary_key=["id"],
                ),
            ]
        )
        whole = _moment(BaseTable(name="ORDER"), "w", 150, 1)
        us = _moment(_us_filter(), "us", 200, 2)
        non_us = _moment(_non_us_filter(), "nu", 100, 3)
        unrelated = Constraint(
            relation=Aggregate(
                source=BaseTable(name="CUSTOMER"), fn="AVG", column="age", alias="a"
            ),
            condition=RComparison(
                op="=", left=RAggregateRef(alias="a"), right=RLiteral(value=999999)
            ),
            fact_references=[99],
        )
        assert (
            check_population_reconciliation(
                [whole, us, non_us, unrelated], other_schema
            )
            == []
        )

    def test_two_genuinely_independent_partitions_do_not_cross_contaminate(self):
        # region partition on ORDER.total (consistent) and an UNRELATED
        # status partition on ORDER.quantity (also consistent) -- different
        # column, different filter condition entirely. Confirms the two
        # lineage/condition groups are evaluated independently without one
        # leaking into the other's result.
        schema = _schema()
        whole1 = _moment(BaseTable(name="ORDER"), "w1", 150, 1, column="total")
        us1 = _moment(_us_filter(), "us1", 200, 2, column="total")
        nonus1 = _moment(
            _non_us_filter(), "nu1", 100, 3, column="total"
        )  # p=0.5, consistent

        whole2 = _moment(BaseTable(name="ORDER"), "w2", 10, 4, column="quantity")
        shipped2 = _moment(_shipped_filter(), "sh2", 12, 5, column="quantity")
        unshipped2 = _moment(
            _unshipped_filter(), "un2", 8, 6, column="quantity"
        )  # p=0.5, consistent

        assert (
            check_population_reconciliation(
                [whole1, us1, nonus1, whole2, shipped2, unshipped2], schema
            )
            == []
        )

    def test_two_independent_partitions_one_broken_isolates_the_conflict(self):
        schema = _schema()
        whole1 = _moment(BaseTable(name="ORDER"), "w1", 150, 1, column="total")
        us1 = _moment(_us_filter(), "us1", 200, 2, column="total")
        nonus1 = _moment(_non_us_filter(), "nu1", 100, 3, column="total")  # consistent

        whole2 = _moment(
            BaseTable(name="ORDER"), "w2", 30, 4, column="quantity"
        )  # inconsistent
        shipped2 = _moment(_shipped_filter(), "sh2", 12, 5, column="quantity")
        unshipped2 = _moment(_unshipped_filter(), "un2", 8, 6, column="quantity")

        conflicts = check_population_reconciliation(
            [whole1, us1, nonus1, whole2, shipped2, unshipped2], schema
        )
        assert len(conflicts) == 1
        assert set(conflicts[0].involved_fact_references) == {4, 5, 6}

    def test_multiple_disagreeing_whole_facts_cross_multiply_against_every_pairing(
        self,
    ):
        # A genuinely adversarial case, not a bug: TWO different "whole"
        # facts stating DIFFERENT means for the SAME population are
        # themselves mutually inconsistent (a separate, real problem
        # check_moment_conflicts is responsible for) -- but from THIS
        # check's own narrow perspective, each whole fact is independently
        # tested against the shared subset/complement pair, so an
        # internally-inconsistent set of "whole" facts correctly produces
        # one conflict per combination that doesn't reconcile, not just one
        # overall verdict. This documents that behavior deliberately.
        whole_ok = _moment(BaseTable(name="ORDER"), "w_ok", 150, 1)
        whole_bad = _moment(BaseTable(name="ORDER"), "w_bad", 300, 2)
        us = _moment(_us_filter(), "us", 200, 3)
        non_us = _moment(_non_us_filter(), "nu", 100, 4)
        conflicts = check_population_reconciliation(
            [whole_ok, whole_bad, us, non_us], _schema()
        )
        assert len(conflicts) == 1
        assert set(conflicts[0].involved_fact_references) == {2, 3, 4}
