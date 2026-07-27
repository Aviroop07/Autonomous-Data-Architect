"""Unit tests for ON clause models (src/pipeline/stage3/models/on_nodes.py).

Comprehensive coverage: construction, validation, table extraction,
recursive trees, SQL validation, name conventions. Each test exercises
one property; edge cases are explicit test methods, not parameterized.
"""

from __future__ import annotations

import pytest

from src.pipeline.stage3.models.on_nodes import (
    JoinCondition,
    ONAggregate,
    ONBaseTable,
    ONJoin,
    ONSubquery,
    extract_tables,
)


# =========================================================================
# ONBaseTable
# =========================================================================


class TestONBaseTable:
    def test_valid_construction(self):
        t = ONBaseTable(name="ORDER")
        assert t.type == "table"
        assert t.name == "ORDER"

    def test_valid_lowercase_name_rejected(self):
        errors = ONBaseTable(name="order")._validate()
        assert any("UPPER_SNAKE_CASE" in e for e in errors)

    def test_empty_name_rejected(self):
        errors = ONBaseTable(name="")._validate()
        assert any("cannot be empty" in e for e in errors)

    def test_whitespace_name_rejected(self):
        errors = ONBaseTable(name="   ")._validate()
        assert any("cannot be empty" in e for e in errors)

    def test_name_with_numbers_valid(self):
        t = ONBaseTable(name="TABLE123")
        assert t._validate() == []

    def test_name_with_special_chars_rejected(self):
        errors = ONBaseTable(name="ORDER$")._validate()
        assert any("UPPER_SNAKE_CASE" in e for e in errors)

    def test_single_char_name_valid(self):
        t = ONBaseTable(name="X")
        assert t._validate() == []


# =========================================================================
# JoinCondition
# =========================================================================


class TestJoinCondition:
    def test_valid_construction(self):
        j = JoinCondition(left="ORDER.order_id", right="LINEITEM.order_id")
        assert j.op == "="
        assert j._validate() == []

    def test_empty_left_rejected(self):
        errors = JoinCondition(left="", right="LINEITEM.order_id")._validate()
        assert any("cannot be empty" in e for e in errors)

    def test_empty_right_rejected(self):
        errors = JoinCondition(left="ORDER.order_id", right="")._validate()
        assert any("cannot be empty" in e for e in errors)

    def test_identical_sides_rejected(self):
        errors = JoinCondition(
            left="ORDER.order_id", right="ORDER.order_id"
        )._validate()
        assert any("identical" in e for e in errors)

    def test_bare_column_names_valid(self):
        j = JoinCondition(left="order_id", right="order_id_bis")
        assert j._validate() == []


# =========================================================================
# ONJoin
# =========================================================================


class TestONJoin:
    def _make_join(self) -> ONJoin:
        left = ONBaseTable(name="ORDER")
        right = ONBaseTable(name="LINEITEM")
        return ONJoin(
            left=left,
            right=right,
            on=[JoinCondition(left="ORDER.order_id", right="LINEITEM.order_id")],
        )

    def test_valid_construction(self):
        j = self._make_join()
        assert j.type == "join"
        assert j.on[0].op == "="

    def test_alias_valid(self):
        j = self._make_join()
        j.alias = "ol"
        assert j._validate() == []

    def test_alias_invalid_format(self):
        j = self._make_join()
        j.alias = "MyJoin"
        errors = j._validate()
        assert any("lower_snake_case" in e for e in errors)

    def test_empty_on_list_rejected(self):
        left = ONBaseTable(name="A")
        right = ONBaseTable(name="B")
        with pytest.raises(Exception):
            ONJoin(left=left, right=right, on=[])

    def test_recursive_validation_catches_bad_left(self):
        bad_left = ONBaseTable(name="lower_case")
        right = ONBaseTable(name="B")
        j = ONJoin(
            left=bad_left,
            right=right,
            on=[JoinCondition(left="lower_case.x", right="B.x")],
        )
        errors = j._validate()
        assert any("ONJoin.left" in e and "UPPER_SNAKE_CASE" in e for e in errors)


# =========================================================================
# ONAggregate
# =========================================================================


class TestONAggregate:
    def _make_agg(self) -> ONAggregate:
        return ONAggregate(
            source=ONBaseTable(name="EMPLOYEE"),
            fn="SUM",
            column="salary",
            group_by=["department_id"],
            alias="total_salary",
        )

    def test_valid_construction(self):
        a = self._make_agg()
        assert a.type == "aggregate"
        assert a.fn == "SUM"
        assert a._validate() == []

    def test_valid_count_star(self):
        a = ONAggregate(
            source=ONBaseTable(name="EMPLOYEE"),
            fn="COUNT",
            column="*",
            alias="row_count",
        )
        assert a._validate() == []

    def test_empty_alias_rejected(self):
        a = ONAggregate(
            source=ONBaseTable(name="EMPLOYEE"),
            fn="COUNT",
            column="*",
            alias="",
        )
        errors = a._validate()
        assert any("alias cannot be empty" in e for e in errors)

    def test_invalid_fn_rejected_at_construction(self):
        with pytest.raises(Exception):
            ONAggregate(
                source=ONBaseTable(name="EMPLOYEE"),
                fn="VARIANCE",
                column="salary",
                alias="var",
            )

    def test_duplicate_group_by_rejected(self):
        a = ONAggregate(
            source=ONBaseTable(name="EMPLOYEE"),
            fn="COUNT",
            column="*",
            group_by=["dept", "dept"],
            alias="cnt",
        )
        errors = a._validate()
        assert any("duplicate" in e for e in errors)

    def test_group_by_invalid_format(self):
        a = ONAggregate(
            source=ONBaseTable(name="EMPLOYEE"),
            fn="COUNT",
            column="*",
            group_by=["DeptID"],
            alias="cnt",
        )
        errors = a._validate()
        assert any("lower_snake_case" in e for e in errors)

    def test_column_invalid_format(self):
        a = ONAggregate(
            source=ONBaseTable(name="EMPLOYEE"),
            fn="AVG",
            column="Salary",
            alias="avg_sal",
        )
        errors = a._validate()
        assert any("column must be lower_snake_case" in e for e in errors)

    def test_recursive_source_validation(self):
        a = ONAggregate(
            source=ONBaseTable(name="lower_case_table"),
            fn="SUM",
            column="x",
            alias="s",
        )
        errors = a._validate()
        assert any("ONAggregate.source" in e for e in errors)


# =========================================================================
# ONSubquery
# =========================================================================


class TestONSubquery:
    def test_valid_sql(self):
        s = ONSubquery(sql="SELECT * FROM EMPLOYEE")
        assert s.type == "subquery"
        assert s._validate() == []

    def test_invalid_sql_rejected(self):
        with pytest.raises(Exception):
            ONSubquery(sql="NOT VALID SQL AT ALL;;;")

    def test_empty_sql_rejected(self):
        with pytest.raises(Exception):
            ONSubquery(sql="")

    def test_whitespace_sql_rejected(self):
        with pytest.raises(Exception):
            ONSubquery(sql="   ")

    def test_complex_query_valid(self):
        s = ONSubquery(
            sql="SELECT department_id, SUM(salary) AS total FROM EMPLOYEE GROUP BY department_id"
        )
        assert s._validate() == []


# =========================================================================
# validate_on_tree (recursive)
# =========================================================================


# =========================================================================
# extract_tables
# =========================================================================


class TestExtractTables:
    def test_single_table(self):
        assert extract_tables(ONBaseTable(name="ORDER")) == {"ORDER"}

    def test_join(self):
        j = ONJoin(
            left=ONBaseTable(name="A"),
            right=ONBaseTable(name="B"),
            on=[JoinCondition(left="A.id", right="B.id")],
        )
        assert extract_tables(j) == {"A", "B"}

    def test_aggregate(self):
        a = ONAggregate(
            source=ONBaseTable(name="EMPLOYEE"),
            fn="SUM",
            column="salary",
            alias="total",
        )
        assert extract_tables(a) == {"EMPLOYEE"}

    def test_aggregate_over_join(self):
        a = ONAggregate(
            source=ONJoin(
                left=ONBaseTable(name="A"),
                right=ONBaseTable(name="B"),
                on=[JoinCondition(left="A.id", right="B.id")],
            ),
            fn="COUNT",
            column="*",
            alias="cnt",
        )
        assert extract_tables(a) == {"A", "B"}

    def test_subquery_extracts_tables(self):
        s = ONSubquery(
            sql="SELECT * FROM ORDERS JOIN ITEMS ON ORDERS.id = ITEMS.order_id"
        )
        tables = extract_tables(s)
        assert "ORDERS" in tables
        assert "ITEMS" in tables

    def test_subquery_invalid_sql_returns_empty(self):
        s = ONSubquery(sql="SELECT 1")
        # SELECT 1 has no tables
        assert extract_tables(s) == set()

    def test_three_level_nesting(self):
        deep = ONAggregate(
            source=ONBaseTable(name="DEEP"),
            fn="MAX",
            column="val",
            alias="max_val",
        )
        mid = ONJoin(
            left=ONBaseTable(name="MID"),
            right=deep,
            on=[JoinCondition(left="MID.id", right="DEEP.mid_id")],
        )
        top = ONJoin(
            left=ONBaseTable(name="TOP"),
            right=mid,
            on=[JoinCondition(left="TOP.id", right="MID.top_id")],
        )
        assert extract_tables(top) == {"TOP", "MID", "DEEP"}
