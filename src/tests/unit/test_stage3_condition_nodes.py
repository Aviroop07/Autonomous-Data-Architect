"""Unit tests for R-AST condition models (src/pipeline/stage3/models/condition_nodes.py).

Comprehensive coverage: every node type's construction and validation,
recursive trees, JSON round-trip via discriminated unions, edge cases.
"""

from __future__ import annotations

import json

import pytest
from pydantic import TypeAdapter

from src.pipeline.stage3.models.condition_nodes import (
    RAggregateRef,
    RAnd,
    RArithmetic,
    RBetween,
    RColumnRef,
    RComparison,
    RExists,
    RIfThen,
    RInSet,
    RLiteral,
    RNot,
    RNotExists,
    RNotInSet,
    ROr,
    RPredicate,
    RExprUnion,
    SubqueryRef,
)


# =========================================================================
# RLiteral
# =========================================================================


class TestRLiteral:
    def test_float(self):
        node = RLiteral(value=3.14)
        assert node.node_type == "literal"
        assert node.value == 3.14
        assert node._validate() == []

    def test_int(self):
        node = RLiteral(value=42)
        assert node._validate() == []

    def test_string(self):
        node = RLiteral(value="gold")
        assert node._validate() == []

    def test_bool(self):
        node = RLiteral(value=True)
        assert node._validate() == []


# =========================================================================
# RColumnRef
# =========================================================================


class TestRColumnRef:
    def test_valid(self):
        node = RColumnRef(name="salary")
        assert node.node_type == "column_ref"
        assert node._validate() == []

    def test_empty_rejected(self):
        errors = RColumnRef(name="")._validate()
        assert any("cannot be empty" in e for e in errors)

    def test_whitespace_rejected(self):
        errors = RColumnRef(name="   ")._validate()
        assert any("cannot be empty" in e for e in errors)


# =========================================================================
# RArithmetic
# =========================================================================


class TestRArithmetic:
    def test_simple_addition(self):
        node = RArithmetic(
            op="+",
            left=RLiteral(value=1),
            right=RLiteral(value=2),
        )
        assert node.node_type == "arithmetic"
        assert node._validate() == []

    def test_nested(self):
        node = RArithmetic(
            op="*",
            left=RColumnRef(name="x"),
            right=RArithmetic(
                op="+",
                left=RLiteral(value=1),
                right=RColumnRef(name="y"),
            ),
        )
        assert node._validate() == []

    def test_all_ops_valid(self):
        for op in ["+", "-", "*", "/"]:
            node = RArithmetic(op=op, left=RLiteral(value=1), right=RLiteral(value=2))
            assert node._validate() == []


# =========================================================================
# RAggregateRef
# =========================================================================


class TestRAggregateRef:
    def test_valid(self):
        node = RAggregateRef(alias="total_salary")
        assert node.node_type == "aggregate_ref"
        assert node._validate() == []

    def test_empty_alias_rejected(self):
        errors = RAggregateRef(alias="")._validate()
        assert any("cannot be empty" in e for e in errors)


# =========================================================================
# RComparison
# =========================================================================


class TestRComparison:
    def test_simple_eq(self):
        node = RComparison(
            op="=",
            left=RColumnRef(name="status"),
            right=RLiteral(value="active"),
        )
        assert node.node_type == "comparison"
        assert node._validate() == []

    def test_all_comparison_ops(self):
        for op in ["<", "<=", "=", "!=", ">=", ">", "~"]:
            node = RComparison(
                op=op,
                left=RLiteral(value=1),
                right=RLiteral(value=2),
            )
            assert node._validate() == []

    def test_with_arithmetic_operand(self):
        node = RComparison(
            op="<=",
            left=RArithmetic(
                op="*", left=RColumnRef(name="qty"), right=RColumnRef(name="price")
            ),
            right=RLiteral(value=1000),
        )
        assert node._validate() == []


# =========================================================================
# RAnd / ROr
# =========================================================================


class TestRLogic:
    def test_and_valid(self):
        node = RAnd(
            operands=[
                RComparison(
                    op=">", left=RColumnRef(name="age"), right=RLiteral(value=18)
                ),
                RComparison(
                    op="<", left=RColumnRef(name="age"), right=RLiteral(value=65)
                ),
            ]
        )
        assert node.node_type == "and"
        assert node._validate() == []

    def test_and_single_operand_rejected_at_construction(self):
        with pytest.raises(Exception):
            RAnd(
                operands=[
                    RComparison(
                        op="=", left=RColumnRef(name="x"), right=RLiteral(value=1)
                    ),
                ]
            )

    def test_or_valid(self):
        node = ROr(
            operands=[
                RComparison(
                    op="=", left=RColumnRef(name="status"), right=RLiteral(value="A")
                ),
                RComparison(
                    op="=", left=RColumnRef(name="status"), right=RLiteral(value="B")
                ),
            ]
        )
        assert node._validate() == []

    def test_or_single_operand_rejected_at_construction(self):
        with pytest.raises(Exception):
            ROr(
                operands=[
                    RComparison(
                        op="=", left=RColumnRef(name="x"), right=RLiteral(value=1)
                    ),
                ]
            )


# =========================================================================
# RNot
# =========================================================================


class TestRNot:
    def test_valid(self):
        node = RNot(
            operand=RComparison(
                op="=",
                left=RColumnRef(name="deleted"),
                right=RLiteral(value=True),
            )
        )
        assert node.node_type == "not"
        assert node._validate() == []


# =========================================================================
# RBetween
# =========================================================================


class TestRBetween:
    def test_valid(self):
        node = RBetween(
            expr=RColumnRef(name="age"),
            low=RLiteral(value=0),
            high=RLiteral(value=120),
        )
        assert node.node_type == "between"
        assert node._validate() == []

    def test_with_arithmetic_bounds(self):
        node = RBetween(
            expr=RColumnRef(name="total"),
            low=RArithmetic(
                op="*", left=RColumnRef(name="base"), right=RLiteral(value=0.9)
            ),
            high=RArithmetic(
                op="*", left=RColumnRef(name="base"), right=RLiteral(value=1.1)
            ),
        )
        assert node._validate() == []


# =========================================================================
# RInSet / RNotInSet
# =========================================================================


class TestRSetMembership:
    def test_in_set_valid(self):
        node = RInSet(
            expr=RColumnRef(name="category"),
            values=["A", "B", "C"],
        )
        assert node.node_type == "in_set"
        assert node._validate() == []

    def test_in_set_empty_values_rejected_at_construction(self):
        with pytest.raises(Exception):
            RInSet(expr=RColumnRef(name="x"), values=[])

    def test_in_set_duplicates_rejected(self):
        node = RInSet(expr=RColumnRef(name="x"), values=[1, 2, 1])
        errors = node._validate()
        assert any("duplicates" in e for e in errors)

    def test_not_in_set_valid(self):
        node = RNotInSet(
            expr=RColumnRef(name="status"),
            values=["DELETED", "ARCHIVED"],
        )
        assert node.node_type == "not_in_set"
        assert node._validate() == []

    def test_not_in_set_duplicates_rejected(self):
        node = RNotInSet(expr=RColumnRef(name="x"), values=[1, 1])
        errors = node._validate()
        assert any("duplicates" in e for e in errors)


# =========================================================================
# RIfThen
# =========================================================================


class TestRIfThen:
    def test_valid_simple(self):
        node = RIfThen(
            antecedent=RComparison(
                op="=",
                left=RColumnRef(name="membership"),
                right=RLiteral(value="gold"),
            ),
            consequent=RComparison(
                op="<=",
                left=RColumnRef(name="discount"),
                right=RLiteral(value=0.2),
            ),
        )
        assert node.node_type == "if_then"
        assert node._validate() == []

    def test_recursive_nesting(self):
        node = RIfThen(
            antecedent=RComparison(
                op="=",
                left=RColumnRef(name="type"),
                right=RLiteral(value="premium"),
            ),
            consequent=RIfThen(
                antecedent=RComparison(
                    op=">",
                    left=RColumnRef(name="spend"),
                    right=RLiteral(value=1000),
                ),
                consequent=RComparison(
                    op="<=",
                    left=RColumnRef(name="discount"),
                    right=RLiteral(value=0.15),
                ),
            ),
        )
        errors = node._validate()
        assert errors == []

    def test_if_with_and_antecedent(self):
        node = RIfThen(
            antecedent=RAnd(
                operands=[
                    RComparison(
                        op="=",
                        left=RColumnRef(name="status"),
                        right=RLiteral(value="active"),
                    ),
                    RComparison(
                        op=">", left=RColumnRef(name="tenure"), right=RLiteral(value=12)
                    ),
                ]
            ),
            consequent=RComparison(
                op="<=",
                left=RColumnRef(name="rate"),
                right=RLiteral(value=0.05),
            ),
        )
        assert node._validate() == []


# =========================================================================
# SubqueryRef + RExists / RNotExists
# =========================================================================


class TestSubqueryRef:
    def test_valid(self):
        ref = SubqueryRef(
            from_table="PRESCRIPTION",
            where_left="PRESCRIPTION.diagnosis_id",
            where_right="DIAGNOSIS.diagnosis_id",
        )
        assert ref._validate() == []

    def test_empty_table_rejected(self):
        ref = SubqueryRef(from_table="", where_left="A.x", where_right="B.x")
        errors = ref._validate()
        assert any("from_table cannot be empty" in e for e in errors)

    def test_empty_where_left_rejected(self):
        ref = SubqueryRef(from_table="A", where_left="", where_right="B.x")
        errors = ref._validate()
        assert any("where_left cannot be empty" in e for e in errors)

    def test_identical_where_rejected(self):
        ref = SubqueryRef(
            from_table="A",
            where_left="A.x",
            where_right="A.x",
        )
        errors = ref._validate()
        assert any("identical" in e for e in errors)


class TestRExists:
    def test_valid(self):
        node = RExists(
            subquery=SubqueryRef(
                from_table="PRESCRIPTION",
                where_left="PRESCRIPTION.diagnosis_id",
                where_right="DIAGNOSIS.diagnosis_id",
            )
        )
        assert node.node_type == "exists"
        assert node._validate() == []


class TestRNotExists:
    def test_valid(self):
        node = RNotExists(
            subquery=SubqueryRef(
                from_table="LOG",
                where_left="LOG.user_id",
                where_right="USER.id",
            )
        )
        assert node.node_type == "not_exists"
        assert node._validate() == []


# =========================================================================
# Recursive validation
# =========================================================================


# =========================================================================
# Discriminated union round-trip (JSON serialization)
# =========================================================================


class TestDiscriminatedUnion:
    def test_expr_round_trip(self):
        original = RArithmetic(
            op="+",
            left=RColumnRef(name="x"),
            right=RLiteral(value=5),
        )
        data = original.model_dump()
        restored = TypeAdapter(RExprUnion).validate_python(data)
        assert isinstance(restored, RArithmetic)
        assert restored.op == "+"

    def test_predicate_round_trip(self):
        original = RAnd(
            operands=[
                RComparison(op=">", left=RColumnRef(name="a"), right=RLiteral(value=0)),
                RBetween(
                    expr=RColumnRef(name="b"),
                    low=RLiteral(value=0),
                    high=RLiteral(value=100),
                ),
            ]
        )
        data = original.model_dump()
        restored = TypeAdapter(RPredicate).validate_python(data)
        assert isinstance(restored, RAnd)
        assert len(restored.operands) == 2
        assert isinstance(restored.operands[1], RBetween)

    def test_exists_round_trip(self):
        original = RExists(
            subquery=SubqueryRef(
                from_table="PRESCRIPTION",
                where_left="PRESCRIPTION.diagnosis_id",
                where_right="DIAGNOSIS.diagnosis_id",
            )
        )
        data = original.model_dump()
        restored = TypeAdapter(RPredicate).validate_python(data)
        assert isinstance(restored, RExists)
        assert restored.subquery.from_table == "PRESCRIPTION"
