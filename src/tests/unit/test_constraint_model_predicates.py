"""Tests for src/util/constraint_model/condition/predicates.py."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import ValidationError

from src.pipeline.stage2.models.data_types import DataType
from src.util.constraint_model.condition.expressions import RColumnRef, RLiteral
from src.util.constraint_model.condition.predicates import (
    RAnd,
    RBetween,
    RComparison,
    RIfThen,
    RInSet,
    RNot,
    RNotInSet,
    ROr,
    extract_columns,
    validate_predicate_tree,
    validate_predicate_types,
)

_CmpOp = Literal["<", "<=", "=", "!=", ">=", ">"]


def _cmp(col: str, op: _CmpOp, val) -> RComparison:
    return RComparison(op=op, left=RColumnRef(name=col), right=RLiteral(value=val))


class TestStructuralValidation:
    def test_and_requires_at_least_two_operands(self):
        with pytest.raises(ValidationError):
            RAnd(operands=[_cmp("x", "=", 1)])

    def test_or_requires_at_least_two_operands(self):
        with pytest.raises(ValidationError):
            ROr(operands=[_cmp("x", "=", 1)])

    def test_valid_and_has_no_structural_errors(self):
        tree = RAnd(operands=[_cmp("x", ">", 1), _cmp("y", "<", 2)])
        assert validate_predicate_tree(tree) == []

    def test_in_set_rejects_duplicates(self):
        node = RInSet(expr=RColumnRef(name="status"), values=["a", "a"])
        assert any("duplicate" in e.lower() for e in node._validate())

    def test_not_in_set_rejects_duplicates(self):
        node = RNotInSet(expr=RColumnRef(name="status"), values=["a", "a"])
        assert any("duplicate" in e.lower() for e in node._validate())


class TestExtractColumns:
    def test_comparison(self):
        assert extract_columns(_cmp("total", ">", 5)) == {"total"}

    def test_and_or_recurse(self):
        tree = RAnd(
            operands=[
                _cmp("x", ">", 1),
                ROr(operands=[_cmp("y", "=", 2), _cmp("z", "=", 3)]),
            ]
        )
        assert extract_columns(tree) == {"x", "y", "z"}

    def test_not_recurses(self):
        assert extract_columns(RNot(operand=_cmp("x", "=", 1))) == {"x"}

    def test_between(self):
        node = RBetween(
            expr=RColumnRef(name="total"),
            low=RLiteral(value=1),
            high=RLiteral(value=10),
        )
        assert extract_columns(node) == {"total"}

    def test_if_then(self):
        node = RIfThen(antecedent=_cmp("a", "=", 1), consequent=_cmp("b", "=", 2))
        assert extract_columns(node) == {"a", "b"}

    def test_in_set_does_not_include_aggregate_ref_values(self):
        node = RInSet(expr=RColumnRef(name="status"), values=["shipped"])
        assert extract_columns(node) == {"status"}


class TestTypeValidation:
    def test_compatible_comparison_passes(self):
        ctx = {"total": DataType.FLOAT}
        assert validate_predicate_types(_cmp("total", ">", 5), ctx) == []

    def test_incompatible_comparison_flagged(self):
        ctx = {"name": DataType.VARCHAR}
        errors = validate_predicate_types(_cmp("name", "=", 5), ctx)
        assert len(errors) == 1

    def test_between_type_mismatch_flagged(self):
        node = RBetween(
            expr=RColumnRef(name="name"), low=RLiteral(value=1), high=RLiteral(value=10)
        )
        errors = validate_predicate_types(node, {"name": DataType.VARCHAR})
        assert len(errors) == 1

    def test_errors_propagate_through_and_or_not_ifthen(self):
        bad = _cmp("name", "=", 5)
        tree = RAnd(
            operands=[bad, RNot(operand=RIfThen(antecedent=bad, consequent=bad))]
        )
        errors = validate_predicate_types(tree, {"name": DataType.VARCHAR})
        assert len(errors) == 3
