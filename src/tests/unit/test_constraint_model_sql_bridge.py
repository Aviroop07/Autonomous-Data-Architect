"""Tests for src/util/constraint_model/relation/sql_bridge.py."""

from __future__ import annotations

from src.util.constraint_model.condition.expressions import (
    RAggregateRef,
    RArithmetic,
    RColumnRef,
    RLiteral,
)
from src.util.constraint_model.condition.predicates import (
    RAnd,
    RBetween,
    RComparison,
    RInSet,
    RNot,
    RNotInSet,
)
from src.util.constraint_model.relation.nodes import (
    Aggregate,
    BaseTable,
    Fanout,
    Filter,
    Join,
    JoinCondition,
    Project,
    ProjectEntry,
)
from src.util.constraint_model.relation.sql_bridge import (
    condition_to_sql,
    expr_to_sql,
    from_sql,
    to_sql,
)


def _order() -> BaseTable:
    return BaseTable(name="ORDER")


class TestExprToSql:
    def test_literal_numeric(self):
        assert expr_to_sql(RLiteral(value=5)) == "5"

    def test_literal_string_escapes_quotes(self):
        assert expr_to_sql(RLiteral(value="it's")) == "'it''s'"

    def test_literal_bool(self):
        assert expr_to_sql(RLiteral(value=True)) == "TRUE"

    def test_column_ref(self):
        assert expr_to_sql(RColumnRef(name="total")) == "total"

    def test_aggregate_ref(self):
        assert expr_to_sql(RAggregateRef(alias="total_sum")) == "total_sum"

    def test_arithmetic(self):
        expr = RArithmetic(
            op="+", left=RColumnRef(name="a"), right=RColumnRef(name="b")
        )
        assert expr_to_sql(expr) == "(a + b)"


class TestConditionToSql:
    def test_comparison(self):
        c = RComparison(
            op=">", left=RColumnRef(name="total"), right=RLiteral(value=100)
        )
        assert condition_to_sql(c) == "total > 100"

    def test_and(self):
        c = RAnd(
            operands=[
                RComparison(op=">", left=RColumnRef(name="a"), right=RLiteral(value=1)),
                RComparison(op="<", left=RColumnRef(name="b"), right=RLiteral(value=2)),
            ]
        )
        assert condition_to_sql(c) == "(a > 1) AND (b < 2)"


class TestBaseTableRoundTrip:
    def test_valid_base_table(self):
        sql = to_sql(_order())
        assert sql == "SELECT * FROM ORDER"
        obj, errors = from_sql(sql)
        assert errors == []
        assert isinstance(obj, BaseTable)
        assert obj.name == "ORDER"

    def test_homogenization_rejects_bare_table_name(self):
        obj, errors = from_sql("ORDER")
        assert obj is None
        assert len(errors) == 1


class TestJoinRoundTrip:
    def test_single_condition_join(self):
        j = Join(
            left=BaseTable(name="ORDER_ITEM"),
            right=_order(),
            on=[JoinCondition(left="ORDER_ITEM.order_id", right="ORDER.id")],
        )
        sql = to_sql(j)
        obj, errors = from_sql(sql)
        assert errors == []
        assert isinstance(obj, Join)
        assert obj.on[0].left == "ORDER_ITEM.order_id"
        assert obj.on[0].right == "ORDER.id"

    def test_composite_join_condition_is_rejected(self):
        obj, errors = from_sql("SELECT * FROM A JOIN B ON A.x = B.x AND A.y = B.y")
        assert obj is None
        assert len(errors) == 1

    def test_right_join_is_rejected(self):
        obj, errors = from_sql("SELECT * FROM A RIGHT JOIN B ON A.x = B.x")
        assert obj is None
        assert len(errors) == 1

    def test_left_join_is_accepted(self):
        obj, errors = from_sql("SELECT * FROM A LEFT JOIN B ON A.x = B.id")
        assert errors == []
        assert isinstance(obj, Join)


class TestFilterRoundTrip:
    def test_simple_comparison(self):
        f = Filter(
            source=_order(),
            condition=RComparison(
                op=">", left=RColumnRef(name="total"), right=RLiteral(value=100)
            ),
        )
        obj, errors = from_sql(to_sql(f))
        assert errors == []
        assert isinstance(obj, Filter)
        assert isinstance(obj.condition, RComparison)

    def test_between(self):
        f = Filter(
            source=_order(),
            condition=RBetween(
                expr=RColumnRef(name="total"),
                low=RLiteral(value=1),
                high=RLiteral(value=10),
            ),
        )
        obj, errors = from_sql(to_sql(f))
        assert errors == []
        assert isinstance(obj, Filter)
        assert isinstance(obj.condition, RBetween)

    def test_in_set(self):
        f = Filter(
            source=_order(),
            condition=RInSet(expr=RColumnRef(name="status"), values=["a", "b"]),
        )
        obj, errors = from_sql(to_sql(f))
        assert errors == []
        assert isinstance(obj, Filter)
        assert isinstance(obj.condition, RInSet)
        assert set(obj.condition.values) == {"a", "b"}

    def test_not_in_set(self):
        f = Filter(
            source=_order(),
            condition=RNotInSet(expr=RColumnRef(name="status"), values=["a", "b"]),
        )
        obj, errors = from_sql(to_sql(f))
        assert errors == []
        assert isinstance(obj, Filter)
        assert isinstance(obj.condition, RNotInSet)

    def test_not(self):
        f = Filter(
            source=_order(),
            condition=RNot(
                operand=RComparison(
                    op="=", left=RColumnRef(name="x"), right=RLiteral(value=1)
                )
            ),
        )
        obj, errors = from_sql(to_sql(f))
        assert errors == []
        assert isinstance(obj, Filter)
        assert isinstance(obj.condition, RNot)

    def test_compound_and(self):
        f = Filter(
            source=_order(),
            condition=RAnd(
                operands=[
                    RComparison(
                        op=">", left=RColumnRef(name="total"), right=RLiteral(value=1)
                    ),
                    RComparison(
                        op="=",
                        left=RColumnRef(name="status"),
                        right=RLiteral(value="shipped"),
                    ),
                ]
            ),
        )
        obj, errors = from_sql(to_sql(f))
        assert errors == []
        assert isinstance(obj, Filter)
        assert isinstance(obj.condition, RAnd)
        assert len(obj.condition.operands) == 2


class TestProjectRoundTrip:
    def test_passthrough_and_rename(self):
        p = Project(
            source=_order(),
            columns=[
                ProjectEntry(expr=RColumnRef(name="id")),
                ProjectEntry(expr=RColumnRef(name="total"), alias="amount"),
            ],
        )
        obj, errors = from_sql(to_sql(p))
        assert errors == []
        assert isinstance(obj, Project)
        assert [c.output_name() for c in obj.columns] == ["id", "amount"]

    def test_star_stays_base_table(self):
        obj, errors = from_sql("SELECT * FROM ORDER")
        assert errors == []
        assert isinstance(obj, BaseTable)


class TestAggregateAndHavingRoundTrip:
    def test_grouped_sum(self):
        agg = Aggregate(
            source=BaseTable(name="PAYMENT"),
            fn="SUM",
            column="amount",
            group_by=["order_id"],
            alias="total_paid",
        )
        obj, errors = from_sql(to_sql(agg))
        assert errors == []
        assert isinstance(obj, Aggregate)
        assert obj.fn == "SUM"
        assert obj.group_by == ["order_id"]
        assert obj.alias == "total_paid"

    def test_count_distinct(self):
        agg = Aggregate(
            source=_order(), fn="COUNT_DISTINCT", column="customer_id", alias="n"
        )
        obj, errors = from_sql(to_sql(agg))
        assert errors == []
        assert isinstance(obj, Aggregate)
        assert obj.fn == "COUNT_DISTINCT"
        assert obj.column == "customer_id"

    def test_having_over_aggregate_round_trips_to_aggregate_ref(self):
        agg = Aggregate(
            source=BaseTable(name="PAYMENT"),
            fn="SUM",
            column="amount",
            group_by=["order_id"],
            alias="total_paid",
        )
        having = Filter(
            source=agg,
            condition=RComparison(
                op=">",
                left=RAggregateRef(alias="total_paid"),
                right=RLiteral(value=1000),
            ),
        )
        obj, errors = from_sql(to_sql(having))
        assert errors == []
        assert isinstance(obj, Filter)
        assert isinstance(obj.source, Aggregate)
        assert isinstance(obj.condition, RComparison)
        assert isinstance(obj.condition.left, RAggregateRef)
        assert obj.condition.left.alias == "total_paid"

    def test_native_having_syntax_also_produces_aggregate_ref(self):
        sql = 'SELECT customer_id, SUM(total) AS total_sum FROM "ORDER" GROUP BY customer_id HAVING SUM(total) > 1000'
        obj, errors = from_sql(sql)
        assert errors == []
        assert isinstance(obj, Filter)
        assert isinstance(obj.condition, RComparison)
        assert isinstance(obj.condition.left, RAggregateRef)

    def test_multiple_aggregate_expressions_rejected(self):
        sql = 'SELECT customer_id, SUM(total) AS s, COUNT(*) AS c FROM "ORDER" GROUP BY customer_id'
        obj, errors = from_sql(sql)
        assert obj is None
        assert len(errors) == 1


class TestFanoutSerializationOnly:
    def test_fanout_serializes_to_left_join_count(self):
        fan = Fanout(
            parent_table="CUSTOMER", child_table="ORDER", fk_column="customer_id"
        )
        sql = to_sql(fan)
        assert "LEFT JOIN" in sql
        assert "COUNT(*)" in sql
        assert "GROUP BY" in sql


class TestOutOfScopeRejections:
    def test_union_rejected(self):
        obj, errors = from_sql("SELECT * FROM A UNION SELECT * FROM B")
        assert obj is None
        assert len(errors) == 1

    def test_window_function_rejected(self):
        obj, errors = from_sql("SELECT RANK() OVER (ORDER BY x) FROM T")
        assert obj is None
        assert len(errors) == 1

    def test_order_by_rejected(self):
        obj, errors = from_sql("SELECT * FROM T ORDER BY x")
        assert obj is None
        assert len(errors) == 1

    def test_limit_rejected(self):
        obj, errors = from_sql("SELECT * FROM T LIMIT 10")
        assert obj is None
        assert len(errors) == 1

    def test_distinct_rejected(self):
        obj, errors = from_sql("SELECT DISTINCT x FROM T")
        assert obj is None
        assert len(errors) == 1

    def test_malformed_sql_rejected(self):
        obj, errors = from_sql("SELEC * FORM T")
        assert obj is None
        assert len(errors) == 1
