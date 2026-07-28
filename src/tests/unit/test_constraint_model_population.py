"""Tests for src/util/constraint_model/population.py."""

from __future__ import annotations

from src.util.schema_model.data_types import DataType
from src.util.schema_model.schema import Column, ForeignKey, Schema, Table
from src.util.constraint_model.condition.expressions import RColumnRef, RLiteral
from src.util.constraint_model.condition.predicates import RComparison
from src.util.constraint_model.population import compute_population
from src.util.constraint_model.relation.nodes import (
    Aggregate,
    BaseTable,
    Fanout,
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
                    Column(
                        name="customer_id", data_type=DataType.INTEGER, is_nullable=True
                    ),
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


def _order() -> BaseTable:
    return BaseTable(name="ORDER")


def _customer() -> BaseTable:
    return BaseTable(name="CUSTOMER")


class TestBaseTable:
    def test_valid_base_table(self):
        pop, errors = compute_population(_order(), _schema())
        assert errors == []
        assert pop is not None
        assert pop.table == "ORDER"
        assert pop.pk_columns == frozenset({"id"})
        assert pop.narrowed is False
        assert pop.edges == frozenset()

    def test_unknown_table_is_an_error(self):
        pop, errors = compute_population(BaseTable(name="NOPE"), _schema())
        assert pop is None
        assert len(errors) == 1

    def test_two_populations_of_same_base_table_are_comparable(self):
        p1, _ = compute_population(_order(), _schema())
        p2, _ = compute_population(_order(), _schema())
        assert p1 is not None and p2 is not None
        assert p1.is_comparable_with(p2, population_sensitive=True)


class TestJoin:
    def _join(self) -> Join:
        return Join(
            left=_order(),
            right=_customer(),
            on=[JoinCondition(left="ORDER.customer_id", right="CUSTOMER.id")],
        )

    def test_nullable_fk_join_is_narrowed(self):
        pop, errors = compute_population(self._join(), _schema())
        assert errors == []
        assert pop is not None
        assert pop.narrowed is True
        assert pop.table == "ORDER"

    def test_edge_records_the_fk_hop(self):
        pop, errors = compute_population(self._join(), _schema())
        assert errors == []
        assert pop is not None
        edge, occurrence = next(iter(pop.edges))
        assert edge.child_table == "ORDER"
        assert edge.fk_column == "customer_id"
        assert edge.parent_table == "CUSTOMER"
        assert occurrence == 1

    def test_non_fk_pk_join_is_an_error(self):
        j = Join(
            left=_order(),
            right=_customer(),
            on=[JoinCondition(left="ORDER.total", right="CUSTOMER.region")],
        )
        pop, errors = compute_population(j, _schema())
        assert pop is None
        assert len(errors) >= 1

    def test_propagates_nested_operand_errors(self):
        j = Join(
            left=BaseTable(name="NOPE"),
            right=_customer(),
            on=[JoinCondition(left="ORDER.customer_id", right="CUSTOMER.id")],
        )
        pop, errors = compute_population(j, _schema())
        assert pop is None
        assert len(errors) == 1


class TestFilter:
    def test_filter_marks_narrowed_and_records_condition(self):
        f = Filter(
            source=_order(),
            condition=RComparison(
                op=">", left=RColumnRef(name="total"), right=RLiteral(value=100)
            ),
        )
        pop, errors = compute_population(f, _schema())
        assert errors == []
        assert pop is not None
        assert pop.narrowed is True
        assert len(pop.filter_conditions) == 1

    def test_base_table_vs_filtered_same_table_not_population_sensitive_comparable(
        self,
    ):
        base_pop, _ = compute_population(_order(), _schema())
        f = Filter(
            source=_order(),
            condition=RComparison(
                op=">", left=RColumnRef(name="total"), right=RLiteral(value=100)
            ),
        )
        filtered_pop, _ = compute_population(f, _schema())
        assert base_pop is not None and filtered_pop is not None
        assert (
            base_pop.is_comparable_with(filtered_pop, population_sensitive=True)
            is False
        )

    def test_base_table_vs_filtered_same_table_is_comparable_when_not_population_sensitive(
        self,
    ):
        base_pop, _ = compute_population(_order(), _schema())
        f = Filter(
            source=_order(),
            condition=RComparison(
                op=">", left=RColumnRef(name="total"), right=RLiteral(value=100)
            ),
        )
        filtered_pop, _ = compute_population(f, _schema())
        assert base_pop is not None and filtered_pop is not None
        assert (
            base_pop.is_comparable_with(filtered_pop, population_sensitive=False)
            is True
        )

    def test_propagates_nested_source_errors(self):
        f = Filter(
            source=BaseTable(name="NOPE"),
            condition=RComparison(
                op=">", left=RColumnRef(name="total"), right=RLiteral(value=1)
            ),
        )
        pop, errors = compute_population(f, _schema())
        assert pop is None
        assert len(errors) == 1


class TestAggregate:
    def test_group_by_becomes_the_pk_columns(self):
        agg = Aggregate(
            source=_order(), fn="COUNT", column="*", group_by=["customer_id"], alias="n"
        )
        pop, errors = compute_population(agg, _schema())
        assert errors == []
        assert pop is not None
        assert pop.pk_columns == frozenset({"customer_id"})
        assert pop.agg_signature is not None

    def test_propagates_nested_source_errors(self):
        agg = Aggregate(
            source=BaseTable(name="NOPE"), fn="COUNT", column="*", alias="n"
        )
        pop, errors = compute_population(agg, _schema())
        assert pop is None
        assert len(errors) == 1


class TestFanout:
    def test_valid_fanout(self):
        fan = Fanout(
            parent_table="CUSTOMER", child_table="ORDER", fk_column="customer_id"
        )
        pop, errors = compute_population(fan, _schema())
        assert errors == []
        assert pop is not None
        assert pop.table == "CUSTOMER"
        assert pop.agg_signature is not None

    def test_unknown_parent_table_is_an_error(self):
        fan = Fanout(parent_table="NOPE", child_table="ORDER", fk_column="customer_id")
        pop, errors = compute_population(fan, _schema())
        assert pop is None
        assert len(errors) == 1


class TestSection5WorkedExamples:
    def test_aggregate_reroot_vs_fanout_not_comparable(self):
        agg = Aggregate(
            source=_order(), fn="COUNT", column="*", group_by=["customer_id"], alias="n"
        )
        agg_pop, agg_errs = compute_population(agg, _schema())
        fan = Fanout(
            parent_table="CUSTOMER", child_table="ORDER", fk_column="customer_id"
        )
        fan_pop, fan_errs = compute_population(fan, _schema())
        assert agg_errs == [] and fan_errs == []
        assert agg_pop is not None and fan_pop is not None
        assert agg_pop.is_comparable_with(fan_pop, population_sensitive=True) is False

    def test_base_table_vs_filter_same_schema_different_population(self):
        base_pop, _ = compute_population(_customer(), _schema())
        f = Filter(
            source=_customer(),
            condition=RComparison(
                op="=", left=RColumnRef(name="region"), right=RLiteral(value="US")
            ),
        )
        filtered_pop, errors = compute_population(f, _schema())
        assert errors == []
        assert base_pop is not None and filtered_pop is not None
        assert (
            base_pop.is_comparable_with(filtered_pop, population_sensitive=True)
            is False
        )
