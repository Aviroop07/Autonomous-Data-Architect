"""Tests for src/util/constraint_model/relation/schema.py."""

from __future__ import annotations

from src.util.schema_model.data_types import DataType
from src.util.schema_model.schema import Column, ForeignKey, Schema, Table
from src.util.constraint_model.condition.expressions import RColumnRef, RLiteral
from src.util.constraint_model.condition.predicates import RComparison
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
from src.util.constraint_model.relation.schema import (
    FANOUT_CHILD_COUNT_COLUMN,
    synthesize_schema,
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


def _order_customer_join() -> Join:
    return Join(
        left=_order(),
        right=_customer(),
        on=[JoinCondition(left="ORDER.customer_id", right="CUSTOMER.id")],
    )


class TestBaseTable:
    def test_valid_table_synthesizes_columns_and_pk(self):
        eff, errors = synthesize_schema(_order(), _schema())
        assert errors == []
        assert eff is not None
        assert set(eff.columns) == {"id", "customer_id", "total"}
        assert eff.primary_key == ["id"]
        id_provenance = eff.columns["id"].provenance
        customer_id_provenance = eff.columns["customer_id"].provenance
        assert id_provenance is not None
        assert customer_id_provenance is not None
        assert id_provenance.is_primary_key is True
        assert customer_id_provenance.referred_table == "CUSTOMER"
        assert eff.columns["total"].provenance is None
        assert eff.row_count.kind == "free"
        assert eff.row_count.name == "ORDER.row_count"

    def test_unknown_table_is_an_error(self):
        eff, errors = synthesize_schema(BaseTable(name="NOPE"), _schema())
        assert eff is None
        assert len(errors) == 1


class TestJoin:
    def test_valid_fk_pk_join(self):
        eff, errors = synthesize_schema(_order_customer_join(), _schema())
        assert errors == []
        assert eff is not None
        # ORDER is the child (holds the FK) -- its own PK survives, per Section 5.
        assert eff.primary_key == ["id"]
        assert eff.row_count.kind == "identity"
        assert eff.row_count.equals == "ORDER.row_count"

    def test_nullable_fk_makes_parent_columns_nullable(self):
        eff, errors = synthesize_schema(_order_customer_join(), _schema())
        assert errors == []
        assert eff is not None
        assert eff.columns["region"].nullable is True

    def test_colliding_column_name_marked_ambiguous_not_rejected(self):
        eff, errors = synthesize_schema(_order_customer_join(), _schema())
        assert errors == []
        assert eff is not None
        assert eff.columns["id"].ambiguous is True
        assert eff.columns["id"].provenance is None

    def test_swapped_condition_sides_still_resolve(self):
        j = Join(
            left=_order(),
            right=_customer(),
            on=[JoinCondition(left="CUSTOMER.id", right="ORDER.customer_id")],
        )
        eff, errors = synthesize_schema(j, _schema())
        assert errors == []
        assert eff is not None
        assert eff.primary_key == ["id"]

    def test_non_fk_pk_join_is_rejected(self):
        j = Join(
            left=_order(),
            right=_customer(),
            on=[JoinCondition(left="ORDER.total", right="CUSTOMER.region")],
        )
        eff, errors = synthesize_schema(j, _schema())
        assert eff is None
        assert any("genuine FK-PK relationship" in e for e in errors)

    def test_unresolvable_qualifier_is_rejected(self):
        j = Join(
            left=_order(),
            right=_customer(),
            on=[JoinCondition(left="NOPE.customer_id", right="CUSTOMER.id")],
        )
        eff, errors = synthesize_schema(j, _schema())
        assert eff is None
        assert any("do not unambiguously resolve" in e for e in errors)

    def test_missing_column_in_condition_is_rejected(self):
        j = Join(
            left=_order(),
            right=_customer(),
            on=[JoinCondition(left="ORDER.nonexistent", right="CUSTOMER.id")],
        )
        eff, errors = synthesize_schema(j, _schema())
        assert eff is None
        assert any("not found on Join.left" in e for e in errors)

    def test_composite_pk_fk_column_still_resolves_as_child(self):
        """Weak-entity / identifying-relationship regression: REVIEW's PK is
        the composite (customer_id, product_id) -- both columns are ALSO
        real FKs to their parent tables. A column being a PK member must
        not suppress its FK-ness (see ColumnProvenance's is_primary_key /
        referred_table split)."""
        schema = Schema(
            tables=[
                Table(
                    name="CUSTOMER",
                    columns=[Column(name="id", data_type=DataType.INTEGER)],
                    primary_key=["id"],
                ),
                Table(
                    name="PRODUCT",
                    columns=[Column(name="id", data_type=DataType.INTEGER)],
                    primary_key=["id"],
                ),
                Table(
                    name="REVIEW",
                    columns=[
                        Column(name="customer_id", data_type=DataType.INTEGER),
                        Column(name="product_id", data_type=DataType.INTEGER),
                        Column(name="rating", data_type=DataType.INTEGER),
                    ],
                    primary_key=["customer_id", "product_id"],
                ),
            ],
            relationships=[
                ForeignKey(
                    referencing_table="REVIEW",
                    referencing_column="customer_id",
                    referred_table="CUSTOMER",
                ),
                ForeignKey(
                    referencing_table="REVIEW",
                    referencing_column="product_id",
                    referred_table="PRODUCT",
                ),
            ],
        )
        j = Join(
            left=BaseTable(name="REVIEW"),
            right=BaseTable(name="CUSTOMER"),
            on=[JoinCondition(left="REVIEW.customer_id", right="CUSTOMER.id")],
        )
        eff, errors = synthesize_schema(j, schema)
        assert errors == []
        assert eff is not None
        # REVIEW is the child -- its own composite PK survives, per Section 5.
        assert eff.primary_key == ["customer_id", "product_id"]

    def test_propagates_nested_operand_errors(self):
        j = Join(
            left=BaseTable(name="NOPE"),
            right=_customer(),
            on=[JoinCondition(left="ORDER.customer_id", right="CUSTOMER.id")],
        )
        eff, errors = synthesize_schema(j, _schema())
        assert eff is None
        assert len(errors) == 1


class TestFilter:
    def test_valid_filter_preserves_columns_and_pk(self):
        f = Filter(
            source=_order(),
            condition=RComparison(
                op=">", left=RColumnRef(name="total"), right=RLiteral(value=100)
            ),
        )
        eff, errors = synthesize_schema(f, _schema())
        assert errors == []
        assert eff is not None
        assert set(eff.columns) == {"id", "customer_id", "total"}
        assert eff.primary_key == ["id"]
        assert eff.row_count.kind == "filtered"
        assert eff.row_count.selectivity is not None

    def test_narrows_nullable_column_referenced_by_comparison(self):
        f = Filter(
            source=_order(),
            condition=RComparison(
                op="=", left=RColumnRef(name="customer_id"), right=RLiteral(value=1)
            ),
        )
        eff, errors = synthesize_schema(f, _schema())
        assert errors == []
        assert eff is not None
        assert eff.columns["customer_id"].nullable is False

    def test_propagates_nested_source_errors(self):
        f = Filter(
            source=BaseTable(name="NOPE"),
            condition=RComparison(
                op=">", left=RColumnRef(name="total"), right=RLiteral(value=1)
            ),
        )
        eff, errors = synthesize_schema(f, _schema())
        assert eff is None
        assert len(errors) == 1


class TestProject:
    def test_valid_passthrough_and_rename(self):
        p = Project(
            source=_order(),
            columns=[
                ProjectEntry(expr=RColumnRef(name="id")),
                ProjectEntry(expr=RColumnRef(name="customer_id"), alias="cust_id"),
            ],
        )
        eff, errors = synthesize_schema(p, _schema())
        assert errors == []
        assert eff is not None
        assert set(eff.columns) == {"id", "cust_id"}
        assert eff.primary_key == ["id"]
        assert eff.row_count.kind == "identity"

    def test_dropping_pk_column_yields_empty_effective_pk(self):
        p = Project(
            source=_order(), columns=[ProjectEntry(expr=RColumnRef(name="total"))]
        )
        eff, errors = synthesize_schema(p, _schema())
        assert errors == []
        assert eff is not None
        assert eff.primary_key == []

    def test_unknown_source_column_is_an_error(self):
        p = Project(
            source=_order(), columns=[ProjectEntry(expr=RColumnRef(name="nonexistent"))]
        )
        eff, errors = synthesize_schema(p, _schema())
        assert eff is None
        assert len(errors) == 1

    def test_propagates_nested_source_errors(self):
        p = Project(
            source=BaseTable(name="NOPE"),
            columns=[ProjectEntry(expr=RColumnRef(name="id"))],
        )
        eff, errors = synthesize_schema(p, _schema())
        assert eff is None
        assert len(errors) == 1


class TestAggregate:
    def test_group_by_forms_effective_pk(self):
        agg = Aggregate(
            source=_order(), fn="COUNT", column="*", group_by=["customer_id"], alias="n"
        )
        eff, errors = synthesize_schema(agg, _schema())
        assert errors == []
        assert eff is not None
        assert eff.primary_key == ["customer_id"]
        assert eff.columns["n"].data_type == DataType.INTEGER
        assert eff.columns["n"].nullable is False
        assert eff.row_count.kind == "grouped"

    def test_no_group_by_yields_empty_effective_pk(self):
        agg = Aggregate(source=_order(), fn="SUM", column="total", alias="s")
        eff, errors = synthesize_schema(agg, _schema())
        assert errors == []
        assert eff is not None
        assert eff.primary_key == []
        assert eff.columns["s"].data_type == DataType.FLOAT

    def test_max_preserves_source_column_type(self):
        agg = Aggregate(source=_order(), fn="MAX", column="total", alias="max_total")
        eff, errors = synthesize_schema(agg, _schema())
        assert errors == []
        assert eff is not None
        assert eff.columns["max_total"].data_type == DataType.FLOAT
        assert eff.columns["max_total"].nullable is True

    def test_unknown_aggregate_column_is_an_error(self):
        agg = Aggregate(source=_order(), fn="SUM", column="nonexistent", alias="s")
        eff, errors = synthesize_schema(agg, _schema())
        assert eff is None
        assert len(errors) == 1

    def test_unknown_group_by_column_is_an_error(self):
        agg = Aggregate(
            source=_order(), fn="COUNT", column="*", group_by=["nonexistent"], alias="n"
        )
        eff, errors = synthesize_schema(agg, _schema())
        assert eff is None
        assert len(errors) == 1

    def test_alias_colliding_with_group_by_is_an_error(self):
        agg = Aggregate(
            source=_order(),
            fn="COUNT",
            column="*",
            group_by=["customer_id"],
            alias="customer_id",
        )
        eff, errors = synthesize_schema(agg, _schema())
        assert eff is None
        assert len(errors) == 1

    def test_propagates_nested_source_errors(self):
        agg = Aggregate(
            source=BaseTable(name="NOPE"), fn="COUNT", column="*", alias="n"
        )
        eff, errors = synthesize_schema(agg, _schema())
        assert eff is None
        assert len(errors) == 1


class TestFanout:
    def test_valid_fanout_synthesizes_parent_columns_plus_child_count(self):
        fan = Fanout(
            parent_table="CUSTOMER", child_table="ORDER", fk_column="customer_id"
        )
        eff, errors = synthesize_schema(fan, _schema())
        assert errors == []
        assert eff is not None
        assert set(eff.columns) == {"id", "region", FANOUT_CHILD_COUNT_COLUMN}
        assert eff.columns[FANOUT_CHILD_COUNT_COLUMN].data_type == DataType.INTEGER
        assert eff.primary_key == ["id"]
        assert eff.row_count.kind == "identity"
        assert eff.row_count.equals == "CUSTOMER.row_count"

    def test_unknown_parent_table_is_an_error(self):
        fan = Fanout(parent_table="NOPE", child_table="ORDER", fk_column="customer_id")
        eff, errors = synthesize_schema(fan, _schema())
        assert eff is None
        assert len(errors) == 1

    def test_unknown_child_table_is_an_error(self):
        fan = Fanout(
            parent_table="CUSTOMER", child_table="NOPE", fk_column="customer_id"
        )
        eff, errors = synthesize_schema(fan, _schema())
        assert eff is None
        assert len(errors) == 1

    def test_unknown_fk_column_is_an_error(self):
        fan = Fanout(
            parent_table="CUSTOMER", child_table="ORDER", fk_column="nonexistent"
        )
        eff, errors = synthesize_schema(fan, _schema())
        assert eff is None
        assert len(errors) == 1


class TestUnknownNodeType:
    def test_raw_sql_is_not_yet_synthesizable(self):
        from src.util.constraint_model.relation.nodes import RawSQL

        eff, errors = synthesize_schema(RawSQL(sql="SELECT * FROM ORDER"), _schema())
        assert eff is None
        assert len(errors) == 1
