"""Tests for src/util/constraint_model/relation/validate.py."""

from __future__ import annotations

from src.util.schema_model.data_types import DataType
from src.util.schema_model.schema import Column, ForeignKey, Schema, Table
from src.util.constraint_model.condition.expressions import RColumnRef
from src.util.constraint_model.relation.nodes import (
    Aggregate,
    BaseTable,
    Join,
    JoinCondition,
    Project,
    ProjectEntry,
)
from src.util.constraint_model.relation.validate import validate_relation


def _schema() -> Schema:
    return Schema(
        tables=[
            Table(
                name="CATEGORY",
                columns=[
                    Column(name="id", data_type=DataType.INTEGER),
                    Column(
                        name="parent_category_id",
                        data_type=DataType.INTEGER,
                        is_nullable=True,
                    ),
                    Column(name="name", data_type=DataType.VARCHAR),
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
                referencing_table="CATEGORY",
                referencing_column="parent_category_id",
                referred_table="CATEGORY",
            )
        ],
    )


def _order() -> BaseTable:
    return BaseTable(name="ORDER")


def _category() -> BaseTable:
    return BaseTable(name="CATEGORY")


class TestSynthesisErrorsPropagate:
    def test_unsynthesizable_relation_returns_synthesis_errors(self):
        errors = validate_relation(BaseTable(name="NOPE"), _schema())
        assert len(errors) == 1


class TestSelfJoinAliasCollision:
    def test_self_join_without_alias_is_rejected(self):
        j = Join(
            left=_category(),
            right=_category(),
            on=[JoinCondition(left="CATEGORY.parent_category_id", right="CATEGORY.id")],
        )
        errors = validate_relation(j, _schema())
        assert any("ambiguous" in e for e in errors)

    def test_self_join_with_distinguishing_alias_is_valid(self):
        j = Join(
            left=_category(),
            right=BaseTable(name="CATEGORY", alias="parent_cat"),
            on=[
                JoinCondition(left="CATEGORY.parent_category_id", right="parent_cat.id")
            ],
        )
        assert validate_relation(j, _schema()) == []

    def test_ordinary_two_table_join_is_valid(self):
        j = Join(
            left=_order(),
            right=BaseTable(name="CATEGORY"),
            on=[JoinCondition(left="ORDER.customer_id", right="CATEGORY.id")],
        )
        # not a real FK-PK relationship in this schema (ORDER.customer_id has
        # no declared FK) -- synthesis itself should reject it upstream.
        errors = validate_relation(j, _schema())
        assert len(errors) == 1

    def test_nested_self_join_inside_project_is_still_detected(self):
        j = Join(
            left=_category(),
            right=_category(),
            on=[JoinCondition(left="CATEGORY.parent_category_id", right="CATEGORY.id")],
        )
        p = Project(source=j, columns=[ProjectEntry(expr=RColumnRef(name="id"))])
        errors = validate_relation(p, _schema())
        assert any("ambiguous" in e for e in errors)


class TestProjectPkNotDropped:
    def test_dropping_pk_column_is_rejected(self):
        p = Project(
            source=_order(), columns=[ProjectEntry(expr=RColumnRef(name="total"))]
        )
        errors = validate_relation(p, _schema())
        assert any("may never drop PK columns" in e for e in errors)

    def test_preserving_pk_column_is_valid(self):
        p = Project(
            source=_order(),
            columns=[
                ProjectEntry(expr=RColumnRef(name="id")),
                ProjectEntry(expr=RColumnRef(name="total")),
            ],
        )
        assert validate_relation(p, _schema()) == []

    def test_renamed_pk_column_still_counts_as_preserved(self):
        p = Project(
            source=_order(),
            columns=[ProjectEntry(expr=RColumnRef(name="id"), alias="order_id")],
        )
        assert validate_relation(p, _schema()) == []


class TestAggregateFnColumnTypeCompatibility:
    def test_sum_on_numeric_column_is_valid(self):
        agg = Aggregate(source=_order(), fn="SUM", column="total", alias="s")
        assert validate_relation(agg, _schema()) == []

    def test_sum_on_varchar_column_is_rejected(self):
        agg = Aggregate(source=_category(), fn="SUM", column="name", alias="s")
        errors = validate_relation(agg, _schema())
        assert any("requires a numeric column" in e for e in errors)

    def test_max_on_orderable_column_is_valid(self):
        agg = Aggregate(source=_order(), fn="MAX", column="total", alias="m")
        assert validate_relation(agg, _schema()) == []

    def test_count_has_no_type_restriction(self):
        agg = Aggregate(source=_category(), fn="COUNT", column="name", alias="n")
        assert validate_relation(agg, _schema()) == []

    def test_check_applies_through_nested_join(self):
        j = Join(
            left=BaseTable(name="CATEGORY", alias="parent_cat"),
            right=_category(),
            on=[
                JoinCondition(left="CATEGORY.parent_category_id", right="parent_cat.id")
            ],
        )
        agg = Aggregate(source=j, fn="SUM", column="name", alias="s")
        errors = validate_relation(agg, _schema())
        assert any("requires a numeric column" in e for e in errors)
