"""Tests for src/util/constraint_model/constraint.py."""

from __future__ import annotations

from src.util.schema_model.data_types import DataType
from src.util.schema_model.schema import Column, Schema, Table
from src.util.constraint_model.condition.cohesive import (
    Correlated,
    Distributed,
    StateSequence,
)
from src.util.constraint_model.condition.expressions import (
    RAggregateRef,
    RColumnRef,
    RLiteral,
)
from src.util.constraint_model.condition.predicates import RComparison
from src.util.constraint_model.constraint import (
    Constraint,
    is_softenable,
    validate_constraint,
)
from src.util.constraint_model.relation.nodes import Aggregate, BaseTable


def _schema() -> Schema:
    return Schema(
        tables=[
            Table(
                name="ORDER",
                columns=[
                    Column(name="id", data_type=DataType.INTEGER),
                    Column(name="total", data_type=DataType.FLOAT),
                    Column(name="status", data_type=DataType.VARCHAR),
                ],
                primary_key=["id"],
            ),
        ],
    )


def _order() -> BaseTable:
    return BaseTable(name="ORDER")


def _plain_comparison() -> RComparison:
    return RComparison(op=">", left=RColumnRef(name="total"), right=RLiteral(value=100))


class TestIsSoftenable:
    def test_distributed_is_softenable(self):
        assert (
            is_softenable(Distributed(column="total", family="GAUSSIAN", parameters={}))
            is True
        )

    def test_correlated_is_softenable(self):
        assert (
            is_softenable(Correlated(columns=["total", "id"], family="GAUSSIAN"))
            is True
        )

    def test_state_sequence_is_never_softenable(self):
        s = StateSequence(sequence_column="status")
        assert is_softenable(s) is False

    def test_moment_fact_via_aggregate_ref_is_softenable(self):
        cond = RComparison(
            op=">", left=RAggregateRef(alias="total_sum"), right=RLiteral(value=1000)
        )
        assert is_softenable(cond) is True

    def test_plain_per_row_comparison_is_not_softenable(self):
        assert is_softenable(_plain_comparison()) is False


class TestConstraintStructuralValidation:
    def test_valid_hard_constraint(self):
        c = Constraint(
            relation=_order(), condition=_plain_comparison(), fact_references=[1]
        )
        assert c._validate() == []

    def test_invalid_severity_value_rejected(self):
        c = Constraint(
            relation=_order(),
            condition=_plain_comparison(),
            fact_references=[1],
            severity="maybe",
        )
        errors = c._validate()
        assert any("must be 'hard' or 'soft'" in e for e in errors)

    def test_soft_on_non_softenable_condition_rejected(self):
        s = StateSequence(sequence_column="status")
        c = Constraint(
            relation=_order(), condition=s, fact_references=[1], severity="soft"
        )
        errors = c._validate()
        assert any("not allowed for a StateSequence condition" in e for e in errors)

    def test_soft_distributed_is_valid(self):
        d = Distributed(column="total", family="GAUSSIAN", parameters={})
        c = Constraint(
            relation=_order(), condition=d, fact_references=[1], severity="soft"
        )
        assert c._validate() == []

    def test_soft_moment_fact_is_valid(self):
        cond = RComparison(
            op=">", left=RAggregateRef(alias="total_sum"), right=RLiteral(value=1000)
        )
        c = Constraint(
            relation=Aggregate(
                source=_order(), fn="SUM", column="total", alias="total_sum"
            ),
            condition=cond,
            fact_references=[1],
            severity="soft",
        )
        assert c._validate() == []

    def test_propagates_invalid_relation_structural_errors(self):
        c = Constraint(
            relation=BaseTable(name="order"),
            condition=_plain_comparison(),
            fact_references=[1],
        )
        errors = c._validate()
        assert any("Constraint.relation" in e for e in errors)

    def test_propagates_invalid_condition_structural_errors(self):
        bad = Distributed(column="Total", family="GAUSSIAN", parameters={})
        c = Constraint(relation=_order(), condition=bad, fact_references=[1])
        errors = c._validate()
        assert any("Constraint.condition" in e for e in errors)


class TestValidateConstraintFullPipeline:
    def test_valid_constraint_against_real_schema(self):
        c = Constraint(
            relation=_order(), condition=_plain_comparison(), fact_references=[1]
        )
        assert validate_constraint(c, _schema()) == []

    def test_unknown_column_caught_by_condition_layer(self):
        cond = RComparison(
            op=">", left=RColumnRef(name="nonexistent"), right=RLiteral(value=100)
        )
        c = Constraint(relation=_order(), condition=cond, fact_references=[1])
        errors = validate_constraint(c, _schema())
        assert any("unknown column" in e for e in errors)

    def test_unknown_table_caught_by_relation_layer(self):
        c = Constraint(
            relation=BaseTable(name="NOPE"),
            condition=_plain_comparison(),
            fact_references=[1],
        )
        errors = validate_constraint(c, _schema())
        assert len(errors) == 1

    def test_structural_error_short_circuits_before_schema_checks(self):
        # invalid severity is a structural error -- should be returned without
        # ever attempting relation/condition schema validation.
        c = Constraint(
            relation=_order(),
            condition=_plain_comparison(),
            fact_references=[1],
            severity="maybe",
        )
        errors = validate_constraint(c, _schema())
        assert any("must be 'hard' or 'soft'" in e for e in errors)

    def test_type_mismatch_caught_by_condition_layer(self):
        cond = RComparison(
            op=">", left=RColumnRef(name="status"), right=RLiteral(value=100)
        )
        c = Constraint(relation=_order(), condition=cond, fact_references=[1])
        errors = validate_constraint(c, _schema())
        assert len(errors) == 1
