"""Tests for src/util/constraint_model/condition/validate.py."""

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
from src.util.constraint_model.condition.predicates import RAnd, RComparison
from src.util.constraint_model.condition.validate import (
    resolve_condition_columns,
    validate_condition,
)
from src.util.constraint_model.relation.nodes import Aggregate, BaseTable
from src.util.constraint_model.relation.schema import synthesize_schema


def _schema() -> Schema:
    return Schema(
        tables=[
            Table(
                name="ORDER",
                columns=[
                    Column(name="id", data_type=DataType.INTEGER),
                    Column(name="customer_id", data_type=DataType.INTEGER),
                    Column(name="total", data_type=DataType.FLOAT),
                    Column(name="status", data_type=DataType.VARCHAR),
                    Column(name="created_at", data_type=DataType.DATETIME),
                ],
                primary_key=["id"],
            ),
        ],
        relationships=[],
    )


def _order_effective_schema():
    eff, errors = synthesize_schema(BaseTable(name="ORDER"), _schema())
    assert errors == []
    assert eff is not None
    return eff


class TestResolveConditionColumnsOrdinaryPredicate:
    def test_known_column_resolves(self):
        cond = RComparison(
            op=">", left=RColumnRef(name="total"), right=RLiteral(value=100)
        )
        assert resolve_condition_columns(cond, _order_effective_schema()) == []

    def test_unknown_column_is_an_error(self):
        cond = RComparison(
            op=">", left=RColumnRef(name="nonexistent"), right=RLiteral(value=100)
        )
        errors = resolve_condition_columns(cond, _order_effective_schema())
        assert any("unknown column" in e for e in errors)

    def test_nested_and_resolves_all_operands(self):
        cond = RAnd(
            operands=[
                RComparison(
                    op=">", left=RColumnRef(name="total"), right=RLiteral(value=1)
                ),
                RComparison(
                    op="=", left=RColumnRef(name="nonexistent"), right=RLiteral(value=1)
                ),
            ]
        )
        errors = resolve_condition_columns(cond, _order_effective_schema())
        assert len(errors) == 1

    def test_aggregate_ref_alias_resolves_when_present_in_schema(self):
        eff, errors = synthesize_schema(
            Aggregate(
                source=BaseTable(name="ORDER"),
                fn="SUM",
                column="total",
                alias="total_sum",
            ),
            _schema(),
        )
        assert errors == []
        assert eff is not None
        cond = RComparison(
            op=">", left=RAggregateRef(alias="total_sum"), right=RLiteral(value=100)
        )
        assert resolve_condition_columns(cond, eff) == []

    def test_ambiguous_column_is_rejected(self):
        from src.util.constraint_model.relation.schema import (
            EffectiveColumn,
            EffectiveSchema,
            RowCountVar,
        )

        eff = EffectiveSchema(
            columns={
                "id": EffectiveColumn(
                    data_type=DataType.INTEGER, nullable=False, ambiguous=True
                ),
            },
            primary_key=[],
            row_count=RowCountVar(name="x.row_count", kind="free"),
        )
        cond = RComparison(op="=", left=RColumnRef(name="id"), right=RLiteral(value=1))
        errors = resolve_condition_columns(cond, eff)
        assert any("ambiguous" in e for e in errors)


class TestResolveConditionColumnsCohesive:
    def test_distributed_column_resolves(self):
        d = Distributed(column="total", family="GAUSSIAN", parameters={})
        assert resolve_condition_columns(d, _order_effective_schema()) == []

    def test_distributed_unknown_column_is_an_error(self):
        d = Distributed(column="nonexistent", family="GAUSSIAN", parameters={})
        errors = resolve_condition_columns(d, _order_effective_schema())
        assert any("unknown column" in e for e in errors)

    def test_correlated_all_columns_resolve(self):
        c = Correlated(columns=["total", "customer_id"], family="GAUSSIAN")
        assert resolve_condition_columns(c, _order_effective_schema()) == []

    def test_correlated_unknown_column_is_an_error(self):
        c = Correlated(columns=["total", "nonexistent"], family="GAUSSIAN")
        errors = resolve_condition_columns(c, _order_effective_schema())
        assert any("unknown column" in e for e in errors)

    def test_state_sequence_resolves_sequence_column(self):
        s = StateSequence(sequence_column="status")
        assert resolve_condition_columns(s, _order_effective_schema()) == []

    def test_state_sequence_unknown_sequence_column_is_an_error(self):
        s = StateSequence(sequence_column="nonexistent")
        errors = resolve_condition_columns(s, _order_effective_schema())
        assert any("unknown column" in e for e in errors)


class TestValidateConditionTypes:
    def test_valid_comparison(self):
        cond = RComparison(
            op=">", left=RColumnRef(name="total"), right=RLiteral(value=100)
        )
        assert validate_condition(cond, _order_effective_schema()) == []

    def test_type_mismatch_comparison(self):
        cond = RComparison(
            op=">", left=RColumnRef(name="status"), right=RLiteral(value=100)
        )
        errors = validate_condition(cond, _order_effective_schema())
        assert len(errors) == 1

    def test_distributed_gaussian_on_numeric_column_is_valid(self):
        d = Distributed(column="total", family="GAUSSIAN", parameters={})
        assert validate_condition(d, _order_effective_schema()) == []

    def test_distributed_gaussian_on_varchar_column_is_rejected(self):
        d = Distributed(column="status", family="GAUSSIAN", parameters={})
        errors = validate_condition(d, _order_effective_schema())
        assert any("must be numeric" in e for e in errors)

    def test_distributed_categorical_on_varchar_column_is_valid(self):
        d = Distributed(
            column="status", family="CATEGORICAL", parameters={"categories": ["a", "b"]}
        )
        assert validate_condition(d, _order_effective_schema()) == []

    def test_distributed_poisson_requires_integer(self):
        d = Distributed(column="total", family="POISSON", parameters={})
        errors = validate_condition(d, _order_effective_schema())
        assert any("must be INTEGER" in e for e in errors)

    def test_distributed_poisson_on_integer_is_valid(self):
        d = Distributed(column="customer_id", family="POISSON", parameters={})
        assert validate_condition(d, _order_effective_schema()) == []

    def test_correlated_mixed_types_is_always_valid(self):
        c = Correlated(columns=["total", "status"], family="GAUSSIAN")
        assert validate_condition(c, _order_effective_schema()) == []

    def test_state_sequence_categorical_column_is_valid(self):
        s = StateSequence(sequence_column="status")
        assert validate_condition(s, _order_effective_schema()) == []

    def test_state_sequence_non_categorical_sequence_column_is_rejected(self):
        s = StateSequence(sequence_column="total")
        errors = validate_condition(s, _order_effective_schema())
        assert any("must be discrete/categorical" in e for e in errors)

    def test_type_checking_is_skipped_when_resolution_fails(self):
        # unknown-column error takes priority; no attempt to type-check it too.
        d = Distributed(column="nonexistent", family="GAUSSIAN", parameters={})
        errors = validate_condition(d, _order_effective_schema())
        assert len(errors) == 1
        assert "unknown column" in errors[0]
