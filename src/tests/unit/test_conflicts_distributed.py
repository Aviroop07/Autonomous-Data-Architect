"""Tests for src/util/constraint_model/conflicts/distributed.py."""

from __future__ import annotations

from src.pipeline.stage2.models.data_types import DataType
from src.pipeline.stage2.models.schema import Column, Schema, Table
from src.util.constraint_model.condition.cohesive import Distributed, DistributionFamily
from src.util.constraint_model.conflicts.distributed import check_distributed_conflicts
from src.util.constraint_model.conflicts.grouping import (
    annotate_populations,
    group_by_comparable_population,
)
from src.util.constraint_model.constraint import Constraint
from src.util.constraint_model.relation.nodes import BaseTable


def _schema() -> Schema:
    return Schema(
        tables=[
            Table(
                name="ORDER",
                columns=[
                    Column(name="id", data_type=DataType.INTEGER),
                    Column(name="total", data_type=DataType.FLOAT),
                    Column(name="status", data_type=DataType.VARCHAR),
                    Column(name="quantity", data_type=DataType.INTEGER),
                ],
                primary_key=["id"],
            )
        ]
    )


def _cluster(constraints):
    annotated, errors = annotate_populations(constraints, _schema())
    assert errors == []
    clusters = group_by_comparable_population(annotated)
    assert len(clusters) == 1
    return clusters[0]


def _dist(
    column: str, family: DistributionFamily, params: dict, fid: int
) -> Constraint:
    return Constraint(
        relation=BaseTable(name="ORDER"),
        condition=Distributed(column=column, family=family, parameters=params),
        fact_references=[fid],
    )


class TestFamilyMismatch:
    def test_two_families_conflict(self):
        c1 = _dist("total", "GAUSSIAN", {}, 1)
        c2 = _dist("total", "POISSON", {}, 2)
        conflicts = check_distributed_conflicts(_cluster([c1, c2]))
        assert len(conflicts) == 1
        assert conflicts[0].kind == "distributed_family_mismatch"
        assert conflicts[0].involved_fact_references == [1, 2]
        assert conflicts[0].softenable is True

    def test_three_way_family_mismatch_produces_three_pairwise_conflicts(self):
        c1 = _dist("total", "GAUSSIAN", {}, 1)
        c2 = _dist("total", "POISSON", {}, 2)
        c3 = _dist("total", "UNIFORM", {}, 3)
        conflicts = check_distributed_conflicts(_cluster([c1, c2, c3]))
        assert len(conflicts) == 3

    def test_different_columns_never_compared(self):
        c1 = _dist("total", "GAUSSIAN", {}, 1)
        c2 = _dist("quantity", "POISSON", {}, 2)
        assert check_distributed_conflicts(_cluster([c1, c2])) == []


class TestParameterMismatch:
    def test_disagreeing_mean(self):
        c1 = _dist("total", "GAUSSIAN", {"mean": 100, "std_dev": 10}, 1)
        c2 = _dist("total", "GAUSSIAN", {"mean": 200, "std_dev": 10}, 2)
        conflicts = check_distributed_conflicts(_cluster([c1, c2]))
        assert len(conflicts) == 1
        assert conflicts[0].kind == "distributed_parameter_mismatch"
        assert "mean" in conflicts[0].summary

    def test_agreeing_parameters_no_conflict(self):
        c1 = _dist("total", "GAUSSIAN", {"mean": 100, "std_dev": 10}, 1)
        c2 = _dist("total", "GAUSSIAN", {"mean": 100, "std_dev": 10}, 2)
        assert check_distributed_conflicts(_cluster([c1, c2])) == []

    def test_partial_parameters_no_conflict_when_no_overlap(self):
        # one fact states only mean, the other only std_dev -- nothing to compare
        c1 = _dist("total", "GAUSSIAN", {"mean": 100}, 1)
        c2 = _dist("total", "GAUSSIAN", {"std_dev": 10}, 2)
        assert check_distributed_conflicts(_cluster([c1, c2])) == []

    def test_categorical_category_set_mismatch(self):
        c1 = _dist("status", "CATEGORICAL", {"categories": ["shipped", "pending"]}, 1)
        c2 = _dist("status", "CATEGORICAL", {"categories": ["shipped", "cancelled"]}, 2)
        conflicts = check_distributed_conflicts(_cluster([c1, c2]))
        assert len(conflicts) == 1
        assert "categories differ" in conflicts[0].detail

    def test_categorical_same_categories_different_order_no_conflict(self):
        c1 = _dist("status", "CATEGORICAL", {"categories": ["a", "b", "c"]}, 1)
        c2 = _dist("status", "CATEGORICAL", {"categories": ["c", "b", "a"]}, 2)
        assert check_distributed_conflicts(_cluster([c1, c2])) == []

    def test_categorical_probabilities_mismatch(self):
        c1 = _dist(
            "status",
            "CATEGORICAL",
            {"categories": ["a", "b"], "probabilities": [0.5, 0.5]},
            1,
        )
        c2 = _dist(
            "status",
            "CATEGORICAL",
            {"categories": ["a", "b"], "probabilities": [0.9, 0.1]},
            2,
        )
        conflicts = check_distributed_conflicts(_cluster([c1, c2]))
        assert len(conflicts) == 1

    def test_numeric_tolerance_absorbs_rounding_noise(self):
        c1 = _dist("total", "GAUSSIAN", {"mean": 100.00000001, "std_dev": 10}, 1)
        c2 = _dist("total", "GAUSSIAN", {"mean": 100.0, "std_dev": 10}, 2)
        assert check_distributed_conflicts(_cluster([c1, c2])) == []

    def test_three_overlapping_facts_multiple_disagreements(self):
        c1 = _dist("total", "GAUSSIAN", {"mean": 100, "std_dev": 5}, 1)
        c2 = _dist("total", "GAUSSIAN", {"mean": 200, "std_dev": 5}, 2)
        c3 = _dist("total", "GAUSSIAN", {"mean": 100, "std_dev": 50}, 3)
        conflicts = check_distributed_conflicts(_cluster([c1, c2, c3]))
        kinds = {c.kind for c in conflicts}
        assert kinds == {"distributed_parameter_mismatch"}
        # (1 vs 2): mean differs. (1 vs 3): std_dev differs. (2 vs 3): both differ (2 conflicts).
        assert len(conflicts) == 4
