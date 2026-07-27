"""Tests for src/util/constraint_model/condition/expressions.py."""

from __future__ import annotations

import pytest

from src.pipeline.stage2.models.data_types import DataType
from src.util.constraint_model.condition.expressions import (
    RAggregateRef,
    RArithmetic,
    RColumnRef,
    RLiteral,
    TypeMismatch,
    infer_type,
    is_categorical_eligible,
    is_numeric,
    is_orderable,
    numeric_promote,
)


class TestRLiteralInferredType:
    def test_int_literal(self):
        assert RLiteral(value=5).inferred_type() == DataType.INTEGER

    def test_float_literal(self):
        assert RLiteral(value=5.5).inferred_type() == DataType.FLOAT

    def test_string_literal(self):
        assert RLiteral(value="x").inferred_type() == DataType.VARCHAR

    def test_bool_literal_is_boolean_not_integer(self):
        # bool is a subclass of int in Python -- must check bool first.
        assert RLiteral(value=True).inferred_type() == DataType.BOOLEAN


class TestNumericPromote:
    def test_same_type(self):
        assert numeric_promote(DataType.INTEGER, DataType.INTEGER) == DataType.INTEGER

    def test_int_float_promotes_to_float(self):
        assert numeric_promote(DataType.INTEGER, DataType.FLOAT) == DataType.FLOAT

    def test_decimal_wins_over_float(self):
        assert numeric_promote(DataType.FLOAT, DataType.DECIMAL) == DataType.DECIMAL


class TestInferType:
    def test_column_ref_resolves_from_context(self):
        ctx = {"total": DataType.FLOAT}
        assert infer_type(RColumnRef(name="total"), ctx) == DataType.FLOAT

    def test_unknown_column_ref_raises(self):
        with pytest.raises(TypeMismatch):
            infer_type(RColumnRef(name="missing"), {})

    def test_aggregate_ref_resolves_from_context(self):
        ctx = {"avg_total": DataType.FLOAT}
        assert infer_type(RAggregateRef(alias="avg_total"), ctx) == DataType.FLOAT

    def test_unknown_aggregate_ref_raises(self):
        with pytest.raises(TypeMismatch):
            infer_type(RAggregateRef(alias="missing"), {})

    def test_arithmetic_promotes_operand_types(self):
        expr = RArithmetic(
            op="*", left=RColumnRef(name="price"), right=RColumnRef(name="qty")
        )
        ctx = {"price": DataType.FLOAT, "qty": DataType.INTEGER}
        assert infer_type(expr, ctx) == DataType.FLOAT

    def test_arithmetic_on_non_numeric_operand_raises(self):
        expr = RArithmetic(
            op="+", left=RColumnRef(name="name"), right=RLiteral(value=5)
        )
        with pytest.raises(TypeMismatch):
            infer_type(expr, {"name": DataType.VARCHAR})

    def test_nested_arithmetic(self):
        inner = RArithmetic(
            op="*", left=RColumnRef(name="a"), right=RColumnRef(name="b")
        )
        outer = RArithmetic(op="+", left=inner, right=RLiteral(value=1))
        ctx = {"a": DataType.INTEGER, "b": DataType.INTEGER}
        assert infer_type(outer, ctx) == DataType.INTEGER


class TestTypeClassification:
    def test_numeric_types(self):
        assert is_numeric(DataType.INTEGER)
        assert is_numeric(DataType.FLOAT)
        assert is_numeric(DataType.DECIMAL)
        assert not is_numeric(DataType.VARCHAR)

    def test_orderable_types(self):
        assert is_orderable(DataType.INTEGER)
        assert is_orderable(DataType.DATE)
        assert is_orderable(DataType.VARCHAR)
        assert not is_orderable(DataType.UUID)
        assert not is_orderable(DataType.BOOLEAN)

    def test_categorical_eligible_types(self):
        assert is_categorical_eligible(DataType.VARCHAR)
        assert is_categorical_eligible(DataType.BOOLEAN)
        assert not is_categorical_eligible(DataType.INTEGER)
        assert not is_categorical_eligible(DataType.FLOAT)
