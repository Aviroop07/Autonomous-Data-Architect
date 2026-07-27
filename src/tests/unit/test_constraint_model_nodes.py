"""Tests for src/util/constraint_model/relation/nodes.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.util.constraint_model.condition.expressions import (
    RArithmetic,
    RColumnRef,
    RLiteral,
)
from src.util.constraint_model.condition.predicates import RAnd, RComparison
from src.util.constraint_model.relation.nodes import (
    Aggregate,
    BaseTable,
    Fanout,
    Filter,
    Join,
    JoinCondition,
    Project,
    ProjectEntry,
    RawSQL,
    _validate_relation,
    extract_base_tables,
    validate_relation_tree,
)


def _order() -> BaseTable:
    return BaseTable(name="ORDER")


def _customer() -> BaseTable:
    return BaseTable(name="CUSTOMER")


class TestBaseTable:
    def test_valid_table_name(self):
        assert validate_relation_tree(BaseTable(name="ORDER")) == []

    def test_lowercase_table_name_rejected(self):
        errors = validate_relation_tree(BaseTable(name="order"))
        assert len(errors) == 1

    def test_alias_must_be_lower_snake(self):
        errors = validate_relation_tree(BaseTable(name="ORDER", alias="Bad-Alias"))
        assert len(errors) == 1


class TestJoin:
    def test_valid_join(self):
        j = Join(
            left=_order(),
            right=_customer(),
            on=[JoinCondition(left="ORDER.customer_id", right="CUSTOMER.id")],
        )
        assert validate_relation_tree(j) == []
        assert extract_base_tables(j) == {"ORDER", "CUSTOMER"}

    def test_join_requires_at_least_one_condition(self):
        with pytest.raises(ValidationError):
            Join(left=_order(), right=_customer(), on=[])

    def test_join_condition_requires_table_qualification(self):
        cond = JoinCondition(left="customer_id", right="CUSTOMER.id")
        assert len(cond._validate()) == 1

    def test_join_condition_rejects_identical_sides(self):
        cond = JoinCondition(left="ORDER.id", right="ORDER.id")
        assert len(cond._validate()) == 1

    def test_join_alias_must_be_lower_snake(self):
        j = Join(
            left=_order(),
            right=_customer(),
            on=[JoinCondition(left="ORDER.customer_id", right="CUSTOMER.id")],
            alias="BadAlias",
        )
        errors = validate_relation_tree(j)
        assert any("Join.alias" in e for e in errors)

    def test_join_propagates_nested_relation_errors(self):
        j = Join(
            left=BaseTable(name="order"),
            right=_customer(),
            on=[JoinCondition(left="ORDER.customer_id", right="CUSTOMER.id")],
        )
        errors = validate_relation_tree(j)
        assert any(e.startswith("Join.left:") for e in errors)


class TestAggregate:
    def test_percentile_requires_fn_param(self):
        agg = Aggregate(source=_order(), fn="PERCENTILE", column="total", alias="p")
        errors = validate_relation_tree(agg)
        assert any("fn_param" in e for e in errors)

    def test_percentile_fn_param_out_of_range(self):
        agg = Aggregate(
            source=_order(), fn="PERCENTILE", column="total", fn_param=150, alias="p"
        )
        errors = validate_relation_tree(agg)
        assert any("[0, 100]" in e for e in errors)

    def test_fn_param_on_non_percentile_rejected(self):
        agg = Aggregate(
            source=_order(), fn="SUM", column="total", alias="s", fn_param=5
        )
        errors = validate_relation_tree(agg)
        assert any("only meaningful for PERCENTILE" in e for e in errors)

    def test_valid_percentile(self):
        agg = Aggregate(
            source=_order(), fn="PERCENTILE", column="total", fn_param=90, alias="p90"
        )
        assert validate_relation_tree(agg) == []

    def test_count_distinct_valid(self):
        agg = Aggregate(
            source=_order(), fn="COUNT_DISTINCT", column="customer_id", alias="n"
        )
        assert validate_relation_tree(agg) == []

    def test_star_column_only_valid_for_count(self):
        agg = Aggregate(source=_order(), fn="SUM", column="*", alias="s")
        errors = validate_relation_tree(agg)
        assert any("only valid for fn='COUNT'" in e for e in errors)

    def test_group_by_duplicate_columns_rejected(self):
        agg = Aggregate(
            source=_order(),
            fn="COUNT",
            column="*",
            group_by=["customer_id", "customer_id"],
            alias="n",
        )
        errors = validate_relation_tree(agg)
        assert any("duplicate" in e for e in errors)

    def test_alias_must_be_lower_snake(self):
        agg = Aggregate(source=_order(), fn="COUNT", column="*", alias="BadAlias")
        errors = validate_relation_tree(agg)
        assert any("Aggregate.alias" in e for e in errors)

    def test_column_must_be_lower_snake(self):
        agg = Aggregate(source=_order(), fn="SUM", column="Total", alias="s")
        errors = validate_relation_tree(agg)
        assert any("Aggregate.column" in e for e in errors)

    def test_group_by_entry_must_be_lower_snake(self):
        agg = Aggregate(
            source=_order(), fn="COUNT", column="*", group_by=["CustomerId"], alias="n"
        )
        errors = validate_relation_tree(agg)
        assert any("group_by[0]" in e for e in errors)

    def test_propagates_nested_source_errors(self):
        agg = Aggregate(
            source=BaseTable(name="order"), fn="COUNT", column="*", alias="n"
        )
        errors = validate_relation_tree(agg)
        assert any(e.startswith("Aggregate.source:") for e in errors)


class TestFilter:
    def test_valid_filter(self):
        f = Filter(
            source=_order(),
            condition=RComparison(
                op=">", left=RColumnRef(name="total"), right=RLiteral(value=100)
            ),
        )
        assert validate_relation_tree(f) == []

    def test_alias_must_be_lower_snake(self):
        f = Filter(
            source=_order(),
            condition=RComparison(
                op=">", left=RColumnRef(name="total"), right=RLiteral(value=100)
            ),
            alias="BadAlias",
        )
        errors = validate_relation_tree(f)
        assert any("Filter.alias" in e for e in errors)

    def test_valid_compound_condition_produces_no_errors(self):
        f = Filter(
            source=_order(),
            condition=RAnd(
                operands=[
                    RComparison(
                        op=">", left=RColumnRef(name="total"), right=RLiteral(value=1)
                    ),
                    RComparison(
                        op="<", left=RColumnRef(name="total"), right=RLiteral(value=2)
                    ),
                ],
            ),
        )
        assert validate_relation_tree(f) == []

    def test_propagates_nested_source_errors(self):
        f = Filter(
            source=BaseTable(name="order"),
            condition=RComparison(
                op=">", left=RColumnRef(name="total"), right=RLiteral(value=100)
            ),
        )
        errors = validate_relation_tree(f)
        assert any(e.startswith("Filter.source:") for e in errors)


class TestProject:
    def test_computed_entry_requires_alias(self):
        with pytest.raises(ValidationError):
            ProjectEntry(
                expr=RArithmetic(
                    op="+", left=RColumnRef(name="a"), right=RColumnRef(name="b")
                )
            )

    def test_passthrough_entry_no_alias_needed(self):
        entry = ProjectEntry(expr=RColumnRef(name="id"))
        assert entry.output_name() == "id"

    def test_renamed_passthrough(self):
        entry = ProjectEntry(expr=RColumnRef(name="customer_id"), alias="cust_id")
        assert entry.output_name() == "cust_id"

    def test_computed_entry_with_alias_is_valid(self):
        entry = ProjectEntry(
            expr=RArithmetic(
                op="+", left=RColumnRef(name="a"), right=RColumnRef(name="b")
            ),
            alias="a_plus_b",
        )
        assert entry.output_name() == "a_plus_b"

    def test_valid_project(self):
        p = Project(
            source=_order(),
            columns=[
                ProjectEntry(expr=RColumnRef(name="id")),
                ProjectEntry(expr=RColumnRef(name="total")),
            ],
        )
        assert validate_relation_tree(p) == []

    def test_duplicate_output_names_rejected(self):
        p = Project(
            source=_order(),
            columns=[
                ProjectEntry(expr=RColumnRef(name="id")),
                ProjectEntry(expr=RColumnRef(name="total"), alias="id"),
            ],
        )
        errors = validate_relation_tree(p)
        assert any("duplicate" in e.lower() for e in errors)

    def test_alias_must_be_lower_snake(self):
        p = Project(
            source=_order(),
            columns=[ProjectEntry(expr=RColumnRef(name="id"))],
            alias="BadAlias",
        )
        errors = validate_relation_tree(p)
        assert any("Project.alias" in e for e in errors)

    def test_propagates_nested_source_errors(self):
        p = Project(
            source=BaseTable(name="order"),
            columns=[ProjectEntry(expr=RColumnRef(name="id"))],
        )
        errors = validate_relation_tree(p)
        assert any(e.startswith("Project.source:") for e in errors)


class TestFanout:
    def test_valid_fanout(self):
        f = Fanout(
            parent_table="CUSTOMER", child_table="ORDER", fk_column="customer_id"
        )
        assert validate_relation_tree(f) == []
        assert extract_base_tables(f) == {"CUSTOMER", "ORDER"}

    def test_parent_table_must_be_upper_snake(self):
        f = Fanout(
            parent_table="customer", child_table="ORDER", fk_column="customer_id"
        )
        errors = validate_relation_tree(f)
        assert any("Fanout.parent_table" in e for e in errors)

    def test_child_table_must_be_upper_snake(self):
        f = Fanout(
            parent_table="CUSTOMER", child_table="order", fk_column="customer_id"
        )
        errors = validate_relation_tree(f)
        assert any("Fanout.child_table" in e for e in errors)

    def test_fk_column_must_be_lower_snake(self):
        f = Fanout(parent_table="CUSTOMER", child_table="ORDER", fk_column="CustomerId")
        errors = validate_relation_tree(f)
        assert any("Fanout.fk_column" in e for e in errors)

    def test_alias_must_be_lower_snake(self):
        f = Fanout(
            parent_table="CUSTOMER",
            child_table="ORDER",
            fk_column="customer_id",
            alias="BadAlias",
        )
        errors = validate_relation_tree(f)
        assert any("Fanout.alias" in e for e in errors)


class TestRawSQL:
    def test_bare_table_name_rejected(self):
        errors = validate_relation_tree(RawSQL(sql="ORDER"))
        assert len(errors) == 1

    def test_select_star_accepted_structurally(self):
        assert validate_relation_tree(RawSQL(sql="SELECT * FROM ORDER")) == []

    def test_empty_sql_rejected(self):
        errors = validate_relation_tree(RawSQL(sql="  "))
        assert len(errors) == 1

    def test_lowercase_select_accepted(self):
        assert validate_relation_tree(RawSQL(sql="select * from order")) == []


class TestExtractBaseTables:
    def test_project_over_filter_over_join(self):
        j = Join(
            left=_order(),
            right=_customer(),
            on=[JoinCondition(left="ORDER.customer_id", right="CUSTOMER.id")],
        )
        f = Filter(
            source=j,
            condition=RComparison(
                op=">", left=RColumnRef(name="total"), right=RLiteral(value=1)
            ),
        )
        p = Project(source=f, columns=[ProjectEntry(expr=RColumnRef(name="id"))])
        assert extract_base_tables(p) == {"ORDER", "CUSTOMER"}

    def test_raw_sql_returns_empty_set(self):
        assert extract_base_tables(RawSQL(sql="SELECT * FROM ORDER")) == set()


class TestUnknownNodeDispatch:
    def test_validate_relation_rejects_non_relation_object(self):
        errors = _validate_relation(object())  # type: ignore[arg-type]
        assert len(errors) == 1
        assert "Unknown Relation node type" in errors[0]
