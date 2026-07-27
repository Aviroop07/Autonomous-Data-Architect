"""Unit tests for normalize_on() (src/pipeline/stage3/models/on_sql_normalize.py).

Covers: passthrough of already-structured nodes, ONSubquery translated via
constraint_model's from_sql() into ONBaseTable/ONJoin/ONAggregate, alias
resolution to real table names, recursion into ONJoin/ONAggregate, the
deliberate non-goal of inferring ONFanout from generic SQL, and rejection
of shapes with no ON-tree equivalent (WHERE/Filter, SELECT column list/
Project, unsupported aggregate functions, composite join conditions,
non-SELECT statements).
"""

from __future__ import annotations

from src.pipeline.stage2.models.data_types import DataType
from src.pipeline.stage2.models.schema import Column, ForeignKey, Schema, Table
from src.pipeline.stage3.models.grain import CanonicalizationFailure, canonicalize
from src.pipeline.stage3.models.on_nodes import (
    JoinCondition,
    ONAggregate,
    ONBaseTable,
    ONFanout,
    ONJoin,
    ONSubquery,
)
from src.pipeline.stage3.models.on_sql_normalize import normalize_on


class TestPassthrough:
    def test_base_table_returns_same_object(self):
        node = ONBaseTable(name="ORDER_ROW")
        result, err = normalize_on(node)
        assert err is None
        assert result is node

    def test_fanout_returns_same_object(self):
        node = ONFanout(
            parent_table="CUSTOMER", child_table="ORDER_ROW", fk_column="customer_id"
        )
        result, err = normalize_on(node)
        assert err is None
        assert result is node

    def test_join_with_no_subquery_returns_same_object(self):
        node = ONJoin(
            left=ONBaseTable(name="ORDER_ROW"),
            right=ONBaseTable(name="CUSTOMER"),
            on=[JoinCondition(left="ORDER_ROW.customer_id", right="CUSTOMER.id")],
        )
        result, err = normalize_on(node)
        assert err is None
        assert result is node


class TestSubqueryBaseTable:
    def test_single_table(self):
        node = ONSubquery(sql="SELECT * FROM ORDER_ROW")
        result, err = normalize_on(node)
        assert err is None
        assert isinstance(result, ONBaseTable)
        assert result.name == "ORDER_ROW"

    def test_single_table_with_alias(self):
        node = ONSubquery(sql="SELECT * FROM ORDER_ROW o")
        result, err = normalize_on(node)
        assert err is None
        assert isinstance(result, ONBaseTable)
        assert result.name == "ORDER_ROW"


class TestSubqueryJoin:
    def test_join_resolves_alias_to_real_table_name(self):
        node = ONSubquery(
            sql=("SELECT * FROM ORDER_ROW o JOIN CUSTOMER c ON o.customer_id = c.id")
        )
        result, err = normalize_on(node)
        assert err is None
        assert isinstance(result, ONJoin)
        assert isinstance(result.left, ONBaseTable) and result.left.name == "ORDER_ROW"
        assert isinstance(result.right, ONBaseTable) and result.right.name == "CUSTOMER"
        assert len(result.on) == 1
        assert result.on[0].left == "ORDER_ROW.customer_id"
        assert result.on[0].right == "CUSTOMER.id"

    def test_composite_join_condition_rejected(self):
        node = ONSubquery(
            sql=(
                "SELECT * FROM ORDER_ROW o JOIN CUSTOMER c "
                "ON o.customer_id = c.id AND o.region = c.region"
            )
        )
        result, err = normalize_on(node)
        assert result is None
        assert err is not None and "Composite join conditions" in err


class TestSubqueryAggregate:
    def test_group_by_aggregate(self):
        node = ONSubquery(
            sql=(
                "SELECT customer_id, AVG(total) AS avg_total FROM ORDER_ROW "
                "GROUP BY customer_id"
            )
        )
        result, err = normalize_on(node)
        assert err is None
        assert isinstance(result, ONAggregate)
        assert (
            isinstance(result.source, ONBaseTable) and result.source.name == "ORDER_ROW"
        )
        assert result.fn == "AVG"
        assert result.column == "total"
        assert result.group_by == ["customer_id"]
        assert result.alias == "avg_total"

    def test_whole_table_aggregate_no_group_by(self):
        node = ONSubquery(sql="SELECT COUNT(*) AS n FROM ORDER_ROW")
        result, err = normalize_on(node)
        assert err is None
        assert isinstance(result, ONAggregate)
        assert result.fn == "COUNT"
        assert result.column == "*"
        assert result.group_by is None
        assert result.alias == "n"

    def test_fanout_shaped_sql_stays_an_aggregate_not_a_fanout(self):
        """Documents the deliberate non-goal (matching sql_bridge.py's own
        scope): recognizing LEFT JOIN + COUNT + GROUP BY parent.pk as a
        zero-preserving Fanout needs a real heuristic this pass doesn't
        attempt. This SQL normalizes to an ordinary ONAggregate."""
        node = ONSubquery(
            sql=(
                "SELECT CUSTOMER.id, COUNT(*) AS n FROM CUSTOMER "
                "LEFT JOIN ORDER_ROW ON ORDER_ROW.customer_id = CUSTOMER.id "
                "GROUP BY CUSTOMER.id"
            )
        )
        result, err = normalize_on(node)
        assert err is None
        assert isinstance(result, ONAggregate)
        assert not isinstance(result, ONFanout)

    def test_unsupported_aggregate_function_rejected(self):
        node = ONSubquery(sql="SELECT COUNT(DISTINCT id) AS n FROM ORDER_ROW")
        result, err = normalize_on(node)
        assert result is None
        assert err is not None and "no ON-tree equivalent" in err


class TestSubqueryRejections:
    def test_where_clause_rejected(self):
        node = ONSubquery(sql="SELECT * FROM ORDER_ROW WHERE total > 100")
        result, err = normalize_on(node)
        assert result is None
        assert err is not None and "WHERE/HAVING filter" in err

    def test_explicit_column_list_rejected(self):
        node = ONSubquery(sql="SELECT id, total FROM ORDER_ROW")
        result, err = normalize_on(node)
        assert result is None
        assert err is not None and "SELECT column list" in err

    def test_non_select_statement_rejected(self):
        node = ONSubquery(sql="CREATE TABLE X (id INT)")
        result, err = normalize_on(node)
        assert result is None
        assert err is not None and "ONSubquery SQL could not be parsed" in err


class TestRecursion:
    def test_subquery_nested_inside_join_left(self):
        node = ONJoin(
            left=ONSubquery(sql="SELECT * FROM ORDER_ROW"),
            right=ONBaseTable(name="CUSTOMER"),
            on=[JoinCondition(left="ORDER_ROW.customer_id", right="CUSTOMER.id")],
        )
        result, err = normalize_on(node)
        assert err is None
        assert isinstance(result, ONJoin)
        assert isinstance(result.left, ONBaseTable) and result.left.name == "ORDER_ROW"
        assert result.right is node.right

    def test_subquery_nested_inside_aggregate_source(self):
        node = ONAggregate(
            source=ONSubquery(sql="SELECT * FROM ORDER_ROW"),
            fn="COUNT",
            column="*",
            group_by=None,
            alias="n",
        )
        result, err = normalize_on(node)
        assert err is None
        assert isinstance(result, ONAggregate)
        assert (
            isinstance(result.source, ONBaseTable) and result.source.name == "ORDER_ROW"
        )

    def test_error_from_deeply_nested_subquery_propagates(self):
        node = ONAggregate(
            source=ONJoin(
                left=ONSubquery(sql="SELECT * FROM ORDER_ROW WHERE total > 100"),
                right=ONBaseTable(name="CUSTOMER"),
                on=[JoinCondition(left="ORDER_ROW.customer_id", right="CUSTOMER.id")],
            ),
            fn="COUNT",
            column="*",
            group_by=None,
            alias="n",
        )
        result, err = normalize_on(node)
        assert result is None
        assert err is not None and "WHERE/HAVING filter" in err


def _self_join_schema() -> Schema:
    """CATEGORY is self-referential (parent_id -> its own id) -- the
    exact shape grain.py's own docstring calls out as a first-class,
    legitimately-traversable-more-than-once case."""
    return Schema(
        tables=[
            Table(
                name="CATEGORY",
                primary_key=["id"],
                columns=[
                    Column(name="id", data_type=DataType.INTEGER, is_nullable=False),
                    Column(
                        name="parent_id", data_type=DataType.INTEGER, is_nullable=True
                    ),
                ],
            ),
        ],
        relationships=[
            ForeignKey(
                referencing_table="CATEGORY",
                referencing_column="parent_id",
                referred_table="CATEGORY",
            )
        ],
    )


def _chain_schema() -> Schema:
    """A.b_id -> B.id -> C via B.c_id -> C.id, for 3-table join chains."""
    return Schema(
        tables=[
            Table(
                name="TABLE_A",
                primary_key=["id"],
                columns=[
                    Column(name="id", data_type=DataType.INTEGER, is_nullable=False),
                    Column(name="b_id", data_type=DataType.INTEGER, is_nullable=False),
                ],
            ),
            Table(
                name="TABLE_B",
                primary_key=["id"],
                columns=[
                    Column(name="id", data_type=DataType.INTEGER, is_nullable=False),
                    Column(name="c_id", data_type=DataType.INTEGER, is_nullable=False),
                ],
            ),
            Table(
                name="TABLE_C",
                primary_key=["id"],
                columns=[
                    Column(name="id", data_type=DataType.INTEGER, is_nullable=False),
                ],
            ),
        ],
        relationships=[
            ForeignKey(
                referencing_table="TABLE_A",
                referencing_column="b_id",
                referred_table="TABLE_B",
            ),
            ForeignKey(
                referencing_table="TABLE_B",
                referencing_column="c_id",
                referred_table="TABLE_C",
            ),
        ],
    )


class TestComplexJoinChains:
    def test_three_table_join_chain_normalizes_and_canonicalizes(self):
        node = ONSubquery(
            sql=(
                "SELECT * FROM TABLE_A a "
                "JOIN TABLE_B b ON a.b_id = b.id "
                "JOIN TABLE_C c ON b.c_id = c.id"
            )
        )
        result, err = normalize_on(node)
        assert err is None
        assert isinstance(result, ONJoin)
        # Outer join is (A join B) join C -- left side is itself a join.
        assert isinstance(result.left, ONJoin)
        assert result.on[0].left == "TABLE_B.c_id"
        assert result.on[0].right == "TABLE_C.id"
        assert result.left.on[0].left == "TABLE_A.b_id"
        assert result.left.on[0].right == "TABLE_B.id"
        canon = canonicalize(result, _chain_schema())
        assert not isinstance(canon, CanonicalizationFailure)

    def test_aggregate_over_three_table_join(self):
        node = ONSubquery(
            sql=(
                "SELECT a.id, COUNT(*) AS n FROM TABLE_A a "
                "JOIN TABLE_B b ON a.b_id = b.id "
                "JOIN TABLE_C c ON b.c_id = c.id "
                "GROUP BY a.id"
            )
        )
        result, err = normalize_on(node)
        assert err is None
        assert isinstance(result, ONAggregate)
        assert isinstance(result.source, ONJoin)
        assert result.fn == "COUNT"
        assert result.group_by == ["id"]

    def test_self_referential_join_normalizes_and_canonicalizes(self):
        node = ONSubquery(
            sql="SELECT * FROM CATEGORY c1 JOIN CATEGORY c2 ON c1.parent_id = c2.id"
        )
        result, err = normalize_on(node)
        assert err is None
        assert isinstance(result, ONJoin)
        assert isinstance(result.left, ONBaseTable) and result.left.name == "CATEGORY"
        assert isinstance(result.right, ONBaseTable) and result.right.name == "CATEGORY"
        assert result.on[0].left == "CATEGORY.parent_id"
        assert result.on[0].right == "CATEGORY.id"
        canon = canonicalize(result, _self_join_schema())
        assert not isinstance(canon, CanonicalizationFailure)

    def test_subquery_of_subquery_of_subquery_reduces_to_base_table(self):
        node = ONSubquery(
            sql="SELECT * FROM (SELECT * FROM (SELECT * FROM ORDER_ROW) x) y"
        )
        result, err = normalize_on(node)
        assert err is None
        assert isinstance(result, ONBaseTable)
        assert result.name == "ORDER_ROW"


class TestUnsupportedJoinTypes:
    def test_right_join_rejected(self):
        node = ONSubquery(
            sql="SELECT * FROM ORDER_ROW o RIGHT JOIN CUSTOMER c ON o.customer_id = c.id"
        )
        result, err = normalize_on(node)
        assert result is None
        assert err is not None and "out of scope" in err

    def test_full_outer_join_rejected(self):
        node = ONSubquery(
            sql=(
                "SELECT * FROM ORDER_ROW o FULL OUTER JOIN CUSTOMER c "
                "ON o.customer_id = c.id"
            )
        )
        result, err = normalize_on(node)
        assert result is None
        assert err is not None and "out of scope" in err


class TestCaseSensitivityMismatch:
    def test_lowercase_table_name_fails_canonicalization_not_silently(self):
        """normalize_on() is purely syntactic -- it doesn't know the real
        schema's casing, so a lowercase table name in the SQL structurally
        succeeds here. canonicalize() is what must catch the mismatch, and
        must do so as an explicit failure, never a silent false-pass."""
        node = ONSubquery(sql="SELECT * FROM order_row")
        result, err = normalize_on(node)
        assert err is None
        assert isinstance(result, ONBaseTable)
        assert result.name == "order_row"
        canon = canonicalize(result, _chain_schema())
        assert isinstance(canon, CanonicalizationFailure)
        assert "not found in schema" in canon.reason


def _reserved_word_schema() -> Schema:
    """ORDER is a SQL reserved word (ORDER BY) -- this codebase's own
    prompt examples use it as a table name regularly, so it must survive
    quoting through sqlglot's parser without special-casing."""
    return Schema(
        tables=[
            Table(
                name="ORDER",
                primary_key=["id"],
                columns=[
                    Column(name="id", data_type=DataType.INTEGER, is_nullable=False),
                    Column(
                        name="customer_id",
                        data_type=DataType.INTEGER,
                        is_nullable=False,
                    ),
                ],
            ),
            Table(
                name="GROUP",
                primary_key=["id"],
                columns=[
                    Column(name="id", data_type=DataType.INTEGER, is_nullable=False),
                ],
            ),
        ],
        relationships=[
            ForeignKey(
                referencing_table="ORDER",
                referencing_column="customer_id",
                referred_table="GROUP",
            )
        ],
    )


class TestQuotedReservedWordIdentifiers:
    def test_quoted_reserved_word_table_name(self):
        node = ONSubquery(sql='SELECT * FROM "ORDER"')
        result, err = normalize_on(node)
        assert err is None
        assert isinstance(result, ONBaseTable)
        assert result.name == "ORDER"

    def test_quoted_reserved_word_join_both_sides(self):
        node = ONSubquery(
            sql=('SELECT * FROM "ORDER" o JOIN "GROUP" g ON o.customer_id = g.id')
        )
        result, err = normalize_on(node)
        assert err is None
        assert isinstance(result, ONJoin)
        assert isinstance(result.left, ONBaseTable) and result.left.name == "ORDER"
        assert isinstance(result.right, ONBaseTable) and result.right.name == "GROUP"
        assert result.on[0].left == "ORDER.customer_id"
        assert result.on[0].right == "GROUP.id"
        canon = canonicalize(result, _reserved_word_schema())
        assert not isinstance(canon, CanonicalizationFailure)
