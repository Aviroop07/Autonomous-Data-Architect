"""Tests for src/pipeline/stage3/models/on_sql_normalize.py.

This file replaces a much larger one that tested `_relation_to_on()`, a
second SQL->node translation that existed only because Stage 3 carried its
own copy of the relation taxonomy. That function is gone: `from_sql()`
already returns the node types the ON tree is made of, so normalize_on()'s
whole job is now "recursively swap RawSQL subtrees for their parsed form".

Coverage that used to live here and has moved rather than disappeared:
  - SQL parsing itself           -> test_constraint_model_sql_bridge.py
  - node-shape validation        -> test_constraint_model_nodes.py
  - rejecting Filter/Project/bad
    aggregate fns in an ON tree  -> test_stage3_grain_gaps.py (canonicalize
                                    is the layer that enforces it now)
"""

from __future__ import annotations

from src.pipeline.stage3.models.on_sql_normalize import normalize_on
from src.util.constraint_model.relation.nodes import (
    Aggregate,
    BaseTable,
    Fanout,
    Join,
    JoinCondition,
    RawSQL,
)


class TestNoOpCases:
    def test_base_table_returns_the_same_object(self):
        node = BaseTable(name="ORDER_ROW")
        out, err = normalize_on(node)
        assert err is None
        assert out is node

    def test_fanout_returns_the_same_object(self):
        node = Fanout(
            parent_table="CUSTOMER", child_table="ORDER_ROW", fk_column="customer_id"
        )
        out, err = normalize_on(node)
        assert err is None
        assert out is node

    def test_structured_tree_with_no_raw_sql_is_returned_unchanged(self):
        node = Join(
            left=BaseTable(name="ORDER_ROW"),
            right=BaseTable(name="CUSTOMER"),
            on=[JoinCondition(left="ORDER_ROW.customer_id", right="CUSTOMER.id")],
        )
        out, err = normalize_on(node)
        assert err is None
        assert out is node


class TestRawSqlReplacement:
    def test_top_level_raw_sql_is_replaced_by_structured_nodes(self):
        out, err = normalize_on(RawSQL(sql="SELECT * FROM ORDER_ROW"))
        assert err is None
        assert isinstance(out, BaseTable)
        assert out.name == "ORDER_ROW"

    def test_raw_sql_join_is_replaced(self):
        out, err = normalize_on(
            RawSQL(
                sql="SELECT * FROM ORDER_ROW o JOIN CUSTOMER c ON o.customer_id = c.id"
            )
        )
        assert err is None
        assert isinstance(out, Join)

    def test_raw_sql_nested_inside_a_join_is_replaced_in_place(self):
        node = Join(
            left=BaseTable(name="CUSTOMER"),
            right=RawSQL(sql="SELECT * FROM ORDER_ROW"),
            on=[JoinCondition(left="ORDER_ROW.customer_id", right="CUSTOMER.id")],
        )
        out, err = normalize_on(node)
        assert err is None
        assert isinstance(out, Join)
        assert isinstance(out.right, BaseTable)
        # A new object, not a mutation of the input.
        assert out is not node
        assert isinstance(node.right, RawSQL)

    def test_raw_sql_nested_inside_an_aggregate_is_replaced(self):
        node = Aggregate(
            source=RawSQL(sql="SELECT * FROM ORDER_ROW"),
            fn="AVG",
            column="total",
            alias="avg_total",
        )
        out, err = normalize_on(node)
        assert err is None
        assert isinstance(out, Aggregate)
        assert isinstance(out.source, BaseTable)

    def test_unparseable_sql_reports_a_reason_and_no_tree(self):
        out, err = normalize_on(RawSQL(sql="SELECT FROM WHERE ;;"))
        assert out is None
        assert err is not None
        assert "could not be parsed" in err
