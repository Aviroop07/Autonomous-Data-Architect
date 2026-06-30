"""Deterministic offline unit tests for Stage 2 schema models.

Targets: src/pipeline/stage2/models/schema.py
Covers Column._validate, Table._validate, Table.normalize, Schema._validate,
and Schema.detect_cycles (circular, acyclic, self-referential).

No LLM / no network. Source is NOT modified; suspected bugs are xfail'd.
"""

from __future__ import annotations

import pytest

from src.pipeline.stage2.models.schema import (
    Column,
    CompositeUnique,
    Table,
    ForeignKey,
    Schema,
    to_snake_case,
    is_upper_snake,
    is_lower_snake,
)
from src.pipeline.stage2.models.data_types import DataType
from src.tests.fixtures import sample_data


# --------------------------------------------------------------------------- #
# Column._validate
# --------------------------------------------------------------------------- #


def test_column_valid_name_yields_no_errors():
    assert Column(name="credit_score", data_type="INTEGER")._validate() == []


def test_column_lowercase_single_word_is_valid():
    assert Column(name="balance", data_type="FLOAT")._validate() == []


@pytest.mark.parametrize("bad", ["CreditScore", "Credit_Score", "UPPER", "1leading"])
def test_column_non_lower_snake_flagged(bad):
    errors = Column(name=bad, data_type=DataType.VARCHAR)._validate()
    assert any("lowercase snake_case" in e for e in errors)


# --------------------------------------------------------------------------- #
# Table._validate
# --------------------------------------------------------------------------- #


def test_valid_table_yields_no_errors():
    t = Table(
        name="CUSTOMER",
        pk="customer_id",
        columns=[
            Column(name="customer_id", data_type="INTEGER"),
            Column(name="full_name", data_type="VARCHAR"),
            Column(name="credit_score", data_type="INTEGER"),
        ],
    )
    assert t._validate() == []


def test_table_plural_name_is_advisory_not_hard_error():
    # Singular-noun is a non-blocking STYLE advisory: it must NOT be a _validate() error
    # (so it never crashes the mapper postcondition), but IS surfaced via _style_warnings().
    t = Table(
        name="CUSTOMERS",
        pk="customers_id",
        columns=[Column(name="customers_id", data_type="INTEGER")],
    )
    assert not any("singular" in e for e in t._validate())
    assert any("singular" in w for w in t._style_warnings())


@pytest.mark.parametrize(
    "allowed", ["STATUS", "ACCESS", "PROCESS", "DIAGNOSIS", "TV_SERIES"]
)
def test_table_allowed_s_ending_not_flagged_as_plural(allowed):
    t = Table(
        name=allowed,
        pk=f"{allowed.lower()}_id",
        columns=[Column(name=f"{allowed.lower()}_id", data_type="INTEGER")],
    )
    assert not any("singular" in e for e in t._validate())
    assert not any("singular" in w for w in t._style_warnings())


@pytest.mark.parametrize("suffix", ["FACT", "DIM", "ID", "ATTR", "TABLE"])
def test_table_forbidden_suffix_flagged(suffix):
    name = f"LOAN_{suffix}"
    t = Table(
        name=name,
        pk="loan_pk",
        columns=[Column(name="loan_pk", data_type="INTEGER")],
    )
    errors = t._validate()
    assert any("Forbidden" in e and suffix in e for e in errors)


def test_table_lowercase_name_flagged_as_not_upper_snake():
    t = Table(
        name="customer",
        pk="customer_id",
        columns=[Column(name="customer_id", data_type="INTEGER")],
    )
    errors = t._validate()
    assert any("UPPER_SNAKE_CASE" in e for e in errors)



def test_table_missing_pk_flagged():
    t = Table(
        name="WIDGET",
        pk="",
        columns=[Column(name="widget_id", data_type="INTEGER")],
    )
    errors = t._validate()
    assert any("must have a primary key" in e for e in errors)


def test_table_pk_not_in_columns_flagged():
    t = Table(
        name="WIDGET",
        pk="widget_id",
        columns=[Column(name="label", data_type="VARCHAR")],
    )
    errors = t._validate()
    assert any("not found in columns" in e for e in errors)


def test_table_bad_pk_type_flagged():
    # PK typed FLOAT is not in ALLOWED_PK_TYPES {INTEGER, VARCHAR}
    t = Table(
        name="WIDGET",
        pk="widget_id",
        columns=[Column(name="widget_id", data_type="FLOAT")],
    )
    errors = t._validate()
    assert any("must be of type" in e for e in errors)


def test_table_varchar_pk_is_allowed():
    t = Table(
        name="WIDGET",
        pk="widget_code",
        columns=[Column(name="widget_code", data_type="VARCHAR")],
    )
    assert not any("must be of type" in e for e in t._validate())


def test_table_duplicate_column_flagged():
    t = Table(
        name="WIDGET",
        pk="widget_id",
        columns=[
            Column(name="widget_id", data_type="INTEGER"),
            Column(name="label", data_type="VARCHAR"),
            Column(name="label", data_type="VARCHAR"),
        ],
    )
    errors = t._validate()
    assert any("Duplicate column" in e for e in errors)


def test_table_pk_in_unique_singleton_flagged_redundant():
    t = Table(
        name="WIDGET",
        pk="widget_id",
        columns=[
            Column(name="widget_id", data_type="INTEGER"),
            Column(name="label", data_type="VARCHAR"),
        ],
        unique=[CompositeUnique(columns=["widget_id"])],
    )
    errors = t._validate()
    assert any("Redundant unique constraint" in e for e in errors)


def test_table_unique_unknown_column_flagged():
    t = Table(
        name="WIDGET",
        pk="widget_id",
        columns=[Column(name="widget_id", data_type="INTEGER")],
        unique=[CompositeUnique(columns=["nonexistent"])],
    )
    errors = t._validate()
    assert any("unknown column" in e for e in errors)


# --------------------------------------------------------------------------- #
# helper functions
# --------------------------------------------------------------------------- #


def test_to_snake_case():
    assert to_snake_case("CreditScore") == "Credit_Score"
    assert to_snake_case("  Loan Amount ") == "Loan_Amount"
    assert to_snake_case("loan__amount") == "loan_amount"


def test_is_upper_snake_and_lower_snake():
    assert is_upper_snake("CUSTOMER_NAME")
    assert not is_upper_snake("customer")
    assert is_lower_snake("credit_score")
    assert not is_lower_snake("CreditScore")


# --------------------------------------------------------------------------- #
# Table.normalize
# --------------------------------------------------------------------------- #


def test_table_normalize_uppercases_table_and_lowercases_cols():
    t = Table(
        name="creditProduct",
        pk="creditProductId",
        columns=[
            Column(name="creditProductId", data_type="INTEGER"),
            Column(name="YieldValue", data_type="FLOAT"),
        ],
    )
    t.normalize()
    assert t.name == "CREDIT_PRODUCT"
    # columns snake_cased + lowercased
    col_names = {c.name for c in t.columns}
    assert col_names == {"credit_product_id", "yield_value"}
    # pk normalized and still points to an existing column
    assert t.pk == "credit_product_id"
    assert any(c.name == t.pk for c in t.columns)


def test_table_normalize_scrubs_pk_from_composite_unique():
    t = Table(
        name="ORDER_LINE",
        pk="order_line_id",
        columns=[
            Column(name="order_line_id", data_type="INTEGER"),
            Column(name="sku", data_type="VARCHAR"),
        ],
        unique=[CompositeUnique(columns=["order_line_id", "sku"])],
    )
    t.normalize()
    # PK removed from composite -> becomes UNIQUE(sku)
    assert t.unique is not None
    assert len(t.unique) == 1
    assert t.unique[0].columns == ["sku"]


# --------------------------------------------------------------------------- #
# Schema._validate
# --------------------------------------------------------------------------- #


def test_fintech_schema_fixture_is_valid():
    schema = sample_data.fintech_schema()
    assert schema._validate() == []


def test_simple_two_table_schema_fixture_is_valid():
    schema = sample_data.simple_two_table_schema()
    assert schema._validate() == []


def test_schema_duplicate_table_flagged():
    t1 = Table(
        name="ALPHA",
        pk="alpha_id",
        columns=[Column(name="alpha_id", data_type="INTEGER")],
    )
    t2 = Table(
        name="ALPHA",
        pk="alpha_id",
        columns=[Column(name="alpha_id", data_type="INTEGER")],
    )
    schema = Schema(tables=[t1, t2])
    errors = schema._validate()
    assert any("Duplicate table name" in e for e in errors)


def test_schema_fk_to_missing_table_flagged():
    schema = Schema(
        tables=[
            Table(
                name="BETA",
                pk="beta_id",
                columns=[
                    Column(name="beta_id", data_type="INTEGER"),
                    Column(name="alpha_id", data_type="INTEGER"),
                ],
            ),
        ],
        relationships=[
            ForeignKey(
                referencing_table="BETA",
                referencing_column="alpha_id",
                referred_table="ALPHA",
            ),
        ],
    )
    errors = schema._validate()
    assert any("Referred table 'ALPHA' does not exist" in e for e in errors)


# --------------------------------------------------------------------------- #
# Schema.detect_cycles
# --------------------------------------------------------------------------- #


def test_detect_cycles_acyclic_returns_empty():
    schema = sample_data.simple_two_table_schema()
    assert schema.detect_cycles() == []


def test_detect_cycles_no_relationships_returns_empty():
    schema = Schema(
        tables=[
            Table(
                name="ALPHA",
                pk="alpha_id",
                columns=[Column(name="alpha_id", data_type="INTEGER")],
            )
        ],
    )
    assert schema.detect_cycles() == []


def test_detect_cycles_self_referential_no_cycle():
    """EMPLOYEE.manager_id -> EMPLOYEE.

    The FK column (manager_id) differs from the target PK (employee_id), so the
    column-level graph node 'EMPLOYEE.manager_id' points to 'EMPLOYEE.employee_id'
    with no edge back. detect_cycles therefore finds NO cycle.
    """
    schema = Schema(
        tables=[
            Table(
                name="EMPLOYEE",
                pk="employee_id",
                columns=[
                    Column(name="employee_id", data_type="INTEGER"),
                    Column(name="manager_id", data_type="INTEGER"),
                ],
            ),
        ],
        relationships=[
            ForeignKey(
                referencing_table="EMPLOYEE",
                referencing_column="manager_id",
                referred_table="EMPLOYEE",
            ),
        ],
    )
    assert schema.detect_cycles() == []


def test_detect_cycles_circular_fk_detected():
    """Construct an actual column-level cycle.

    detect_cycles builds a graph where each FK edge goes from
    "referencing_table.referencing_column" -> "referred_table.<referred_table.pk>".
    A cycle requires that node X is both a source (referencing_column) and a
    target (pk of some table). We engineer this by using each table's own pk as
    the referencing column in an FK (which is invalid per _validate, but
    detect_cycles itself never calls _validate):

      A.pk = "a_id"; FK A.a_id -> B  => edge "A.a_id" -> "B.b_id" (B.pk=b_id)
      B.pk = "b_id"; FK B.b_id -> A  => edge "B.b_id" -> "A.a_id" (A.pk=a_id)

    This creates: A.a_id -> B.b_id -> A.a_id (a 2-node cycle).
    """
    schema = Schema(
        tables=[
            Table(
                name="A",
                pk="a_id",
                columns=[
                    Column(name="a_id", data_type="INTEGER"),
                    Column(name="label", data_type="VARCHAR"),
                ],
            ),
            Table(
                name="B",
                pk="b_id",
                columns=[
                    Column(name="b_id", data_type="INTEGER"),
                    Column(name="label", data_type="VARCHAR"),
                ],
            ),
        ],
        relationships=[
            # A.a_id (A's own PK) -> B  => source A.a_id, target B.b_id
            ForeignKey(
                referencing_table="A", referencing_column="a_id", referred_table="B"
            ),
            # B.b_id (B's own PK) -> A  => source B.b_id, target A.a_id
            ForeignKey(
                referencing_table="B", referencing_column="b_id", referred_table="A"
            ),
        ],
    )
    cycles = schema.detect_cycles()
    assert cycles, f"expected a column-level cycle, got {cycles}"
    # every reported cycle is a list of "table.column" node strings that closes
    for cycle in cycles:
        assert cycle[0] == cycle[-1]
        assert all("." in node for node in cycle)
