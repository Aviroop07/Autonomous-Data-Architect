"""Tests for src/util/constraint_model/conflicts/grouping.py."""

from __future__ import annotations

from src.util.schema_model.data_types import DataType
from src.util.schema_model.schema import Column, ForeignKey, Schema, Table
from src.util.constraint_model.condition.cohesive import Distributed
from src.util.constraint_model.condition.expressions import RColumnRef, RLiteral
from src.util.constraint_model.condition.predicates import RComparison
from src.util.constraint_model.conflicts.grouping import (
    annotate_populations,
    group_by_base_lineage,
    group_by_comparable_population,
)
from src.util.constraint_model.constraint import Constraint
from src.util.constraint_model.relation.nodes import (
    Aggregate,
    BaseTable,
    Filter,
    Join,
    JoinCondition,
)


def _schema() -> Schema:
    return Schema(
        tables=[
            Table(
                name="CUSTOMER",
                columns=[
                    Column(name="id", data_type=DataType.INTEGER),
                    Column(name="region", data_type=DataType.VARCHAR),
                ],
                primary_key=["id"],
            ),
            Table(
                name="ORDER",
                columns=[
                    Column(name="id", data_type=DataType.INTEGER),
                    Column(name="customer_id", data_type=DataType.INTEGER),
                    Column(name="total", data_type=DataType.FLOAT),
                ],
                primary_key=["id"],
            ),
        ],
        relationships=[
            ForeignKey(
                referencing_table="ORDER",
                referencing_column="customer_id",
                referred_table="CUSTOMER",
            )
        ],
    )


def _dist(column: str, fid: int, relation=None) -> Constraint:
    return Constraint(
        relation=relation or BaseTable(name="ORDER"),
        condition=Distributed(column=column, family="GAUSSIAN", parameters={}),
        fact_references=[fid],
    )


class TestAnnotatePopulations:
    def test_valid_constraints_all_annotated(self):
        annotated, errors = annotate_populations(
            [_dist("total", 1), _dist("total", 2)], _schema()
        )
        assert errors == []
        assert len(annotated) == 2

    def test_unresolvable_relation_reported_and_skipped(self):
        bad = _dist("total", 1, relation=BaseTable(name="NOPE"))
        annotated, errors = annotate_populations([bad, _dist("total", 2)], _schema())
        assert len(annotated) == 1
        assert len(errors) == 1
        assert "fact_references=[1]" in errors[0]


class TestGroupByComparablePopulation:
    def test_same_table_facts_cluster_together(self):
        annotated, _ = annotate_populations(
            [_dist("total", 1), _dist("total", 2)], _schema()
        )
        clusters = group_by_comparable_population(annotated)
        assert len(clusters) == 1
        assert len(clusters[0]) == 2

    def test_different_tables_do_not_cluster(self):
        annotated, _ = annotate_populations(
            [
                _dist("total", 1),
                _dist("region", 2, relation=BaseTable(name="CUSTOMER")),
            ],
            _schema(),
        )
        clusters = group_by_comparable_population(annotated)
        assert len(clusters) == 2

    def test_filtered_subset_does_not_cluster_with_whole(self):
        f = Filter(
            source=BaseTable(name="ORDER"),
            condition=RComparison(
                op=">", left=RColumnRef(name="total"), right=RLiteral(value=100)
            ),
        )
        annotated, _ = annotate_populations(
            [_dist("total", 1), _dist("total", 2, relation=f)], _schema()
        )
        clusters = group_by_comparable_population(annotated)
        assert len(clusters) == 2

    def test_three_way_cluster(self):
        annotated, _ = annotate_populations(
            [_dist("total", 1), _dist("total", 2), _dist("total", 3)], _schema()
        )
        clusters = group_by_comparable_population(annotated)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3


class TestGroupByBaseLineage:
    def test_whole_and_filtered_share_lineage(self):
        f = Filter(
            source=BaseTable(name="ORDER"),
            condition=RComparison(
                op=">", left=RColumnRef(name="total"), right=RLiteral(value=100)
            ),
        )
        annotated, _ = annotate_populations(
            [_dist("total", 1), _dist("total", 2, relation=f)], _schema()
        )
        groups = group_by_base_lineage(annotated)
        assert len(groups) == 1
        assert len(next(iter(groups.values()))) == 2

    def test_aggregate_rooted_constraints_are_excluded(self):
        agg = Aggregate(
            source=BaseTable(name="ORDER"), fn="COUNT", column="*", alias="n"
        )
        c = Constraint(
            relation=agg,
            condition=RComparison(
                op=">", left=RColumnRef(name="n"), right=RLiteral(value=0)
            ),
            fact_references=[1],
        )
        annotated, _ = annotate_populations([c], _schema())
        groups = group_by_base_lineage(annotated)
        assert groups == {}

    def test_different_join_lineage_yields_different_groups(self):
        j = Join(
            left=BaseTable(name="ORDER"),
            right=BaseTable(name="CUSTOMER"),
            on=[JoinCondition(left="ORDER.customer_id", right="CUSTOMER.id")],
        )
        annotated, _ = annotate_populations(
            [_dist("total", 1), _dist("total", 2, relation=j)], _schema()
        )
        groups = group_by_base_lineage(annotated)
        assert len(groups) == 2
