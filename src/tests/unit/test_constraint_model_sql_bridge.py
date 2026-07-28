"""Tests for src/util/constraint_model/relation/sql_bridge.py."""

from __future__ import annotations

from src.util.constraint_model.condition.expressions import (
    RAggregateRef,
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
    Project,
)
from src.util.constraint_model.relation.sql_bridge import from_sql


def _order() -> BaseTable:
    return BaseTable(name="ORDER")


class TestBaseTableRoundTrip:
    def test_valid_base_table(self):
        obj, errors = from_sql('SELECT * FROM ORDER')
        assert errors == []
        assert isinstance(obj, BaseTable)
        assert obj.name == "ORDER"

    def test_homogenization_rejects_bare_table_name(self):
        obj, errors = from_sql("ORDER")
        assert obj is None
        assert len(errors) == 1


class TestJoinRoundTrip:
    def test_single_condition_join(self):
        obj, errors = from_sql('SELECT * FROM ORDER_ITEM INNER JOIN ORDER ON ORDER_ITEM.order_id = ORDER.id')
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
        obj, errors = from_sql('SELECT * FROM ORDER WHERE total > 100')
        assert errors == []
        assert isinstance(obj, Filter)
        assert isinstance(obj.condition, RComparison)

    def test_between(self):
        obj, errors = from_sql('SELECT * FROM ORDER WHERE total BETWEEN 1 AND 10')
        assert errors == []
        assert isinstance(obj, Filter)
        assert isinstance(obj.condition, RBetween)

    def test_in_set(self):
        obj, errors = from_sql("SELECT * FROM ORDER WHERE status IN ('a', 'b')")
        assert errors == []
        assert isinstance(obj, Filter)
        assert isinstance(obj.condition, RInSet)
        assert set(obj.condition.values) == {"a", "b"}

    def test_not_in_set(self):
        obj, errors = from_sql("SELECT * FROM ORDER WHERE NOT status IN ('a', 'b')")
        assert errors == []
        assert isinstance(obj, Filter)
        assert isinstance(obj.condition, RNotInSet)

    def test_not(self):
        obj, errors = from_sql('SELECT * FROM ORDER WHERE NOT (x = 1)')
        assert errors == []
        assert isinstance(obj, Filter)
        assert isinstance(obj.condition, RNot)

    def test_compound_and(self):
        obj, errors = from_sql("SELECT * FROM ORDER WHERE (total > 1) AND (status = 'shipped')")
        assert errors == []
        assert isinstance(obj, Filter)
        assert isinstance(obj.condition, RAnd)
        assert len(obj.condition.operands) == 2


class TestProjectRoundTrip:
    def test_passthrough_and_rename(self):
        obj, errors = from_sql('SELECT id, total AS amount FROM ORDER')
        assert errors == []
        assert isinstance(obj, Project)
        assert [c.output_name() for c in obj.columns] == ["id", "amount"]

    def test_star_stays_base_table(self):
        obj, errors = from_sql("SELECT * FROM ORDER")
        assert errors == []
        assert isinstance(obj, BaseTable)


class TestAggregateAndHavingRoundTrip:
    def test_grouped_sum(self):
        obj, errors = from_sql('SELECT order_id, SUM(amount) AS total_paid FROM PAYMENT GROUP BY order_id')
        assert errors == []
        assert isinstance(obj, Aggregate)
        assert obj.fn == "SUM"
        assert obj.group_by == ["order_id"]
        assert obj.alias == "total_paid"

    def test_count_distinct(self):
        obj, errors = from_sql('SELECT COUNT(DISTINCT customer_id) AS n FROM ORDER')
        assert errors == []
        assert isinstance(obj, Aggregate)
        assert obj.fn == "COUNT_DISTINCT"
        assert obj.column == "customer_id"

    def test_having_over_aggregate_round_trips_to_aggregate_ref(self):
        obj, errors = from_sql('SELECT * FROM (SELECT order_id, SUM(amount) AS total_paid FROM PAYMENT GROUP BY order_id) AS sub_1 WHERE total_paid > 1000')
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


class TestFanoutIsNotRecoverableFromSql:
    """The zero-preserving guarantee has no SQL spelling.

    A fanout serializes to a LEFT JOIN plus COUNT(*) GROUP BY, but parsing that
    shape back yields an ordinary Aggregate -- the "count parents with zero
    children" semantics is not recoverable from the SQL. This is exactly why the
    constraint_generator prompt tells the model to emit a `fanout` node directly
    rather than compose one, so it is worth pinning as a parser property rather
    than leaving it as prose in a prompt.
    """

    def test_left_join_count_parses_as_aggregate_not_fanout(self):
        sql = (
            "SELECT CUSTOMER.*, COUNT(*) AS child_count FROM CUSTOMER "
            "LEFT JOIN ORDER ON ORDER.customer_id = CUSTOMER.id GROUP BY CUSTOMER.id"
        )
        obj, errors = from_sql(sql)
        assert not isinstance(obj, Fanout), (
            "if the parser ever learns to recover a Fanout from this shape, the "
            "generator prompt's 'never compose a fanout' rule can be relaxed"
        )
