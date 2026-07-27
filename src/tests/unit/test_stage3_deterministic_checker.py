"""Tests for src/pipeline/stage3/middleware/deterministic_checker.py.

Focus: the normalize_on() -> canonicalize() wiring in _canonicalize_list()
(previously zero coverage on this module). A generic Constraint's `on` can
legally hold an ONSubquery -- these tests confirm normalize_on() replaces it
in place (so canonicalize() sees a structured tree, and the bridge later
never sees a raw ONSubquery) and that both normalization and
canonicalization failures surface as distinct, labeled error strings.
"""

from __future__ import annotations

from src.pipeline.stage2.models.data_types import DataType
from src.pipeline.stage2.models.schema import Column, ForeignKey, Schema, Table
from src.pipeline.stage3.agents.extraction_outputs import UnifiedOutput
from src.pipeline.stage3.middleware.deterministic_checker import (
    DeterministicCheckerLoopAgent,
)
from src.pipeline.stage3.models.condition_nodes import RColumnRef, RComparison, RLiteral
from src.pipeline.stage3.models.cross_shard import (
    Constraint,
    CorrelatedConstraint,
    DistributionConstraint,
    StateSequenceConstraint,
)
from src.pipeline.stage3.models.grain import _SchemaView
from src.pipeline.stage3.models.on_nodes import (
    JoinCondition,
    ONBaseTable,
    ONJoin,
    ONSubquery,
)


def _schema() -> Schema:
    return Schema(
        tables=[
            Table(
                name="ORDER_ROW",
                primary_key=["id"],
                columns=[
                    Column(name="id", data_type=DataType.INTEGER, is_nullable=False),
                    Column(name="total", data_type=DataType.FLOAT, is_nullable=False),
                    Column(
                        name="customer_id",
                        data_type=DataType.INTEGER,
                        is_nullable=False,
                    ),
                ],
            ),
            Table(
                name="CUSTOMER",
                primary_key=["id"],
                columns=[
                    Column(name="id", data_type=DataType.INTEGER, is_nullable=False),
                    Column(
                        name="loyalty_tier",
                        data_type=DataType.VARCHAR,
                        is_nullable=False,
                    ),
                ],
            ),
        ],
        relationships=[
            ForeignKey(
                referencing_table="ORDER_ROW",
                referencing_column="customer_id",
                referred_table="CUSTOMER",
            )
        ],
    )


def _view() -> _SchemaView:
    return _SchemaView.from_schema(_schema())


def _simple_condition() -> RComparison:
    return RComparison(op=">", left=RColumnRef(name="total"), right=RLiteral(value=0))


class TestCanonicalizeListNormalization:
    def test_subquery_join_is_normalized_and_canonicalizes_clean(self):
        c = Constraint(
            fact_references=[1],
            on=ONSubquery(
                sql=(
                    "SELECT * FROM ORDER_ROW o JOIN CUSTOMER c ON o.customer_id = c.id"
                )
            ),
            condition=_simple_condition(),
            category="logic",
            severity="hard",
        )
        agent = DeterministicCheckerLoopAgent()
        errors = agent._canonicalize_list([c], "Logic", _schema(), _view())
        assert errors == []
        # The constraint's own `on` was replaced in place -- no ONSubquery
        # survives past this node, so the bridge never has to see one.
        assert isinstance(c.on, ONJoin)
        assert isinstance(c.on.left, ONBaseTable)
        assert isinstance(c.on.right, ONBaseTable)

    def test_subquery_that_cannot_normalize_reports_normalization_error(self):
        c = Constraint(
            fact_references=[2],
            on=ONSubquery(sql="SELECT * FROM ORDER_ROW WHERE total > 100"),
            condition=_simple_condition(),
            category="logic",
            severity="hard",
        )
        agent = DeterministicCheckerLoopAgent()
        errors = agent._canonicalize_list([c], "Logic", _schema(), _view())
        assert len(errors) == 1
        assert "Logic[0] ON normalization failed" in errors[0]
        # Left untouched on failure -- nothing to canonicalize.
        assert isinstance(c.on, ONSubquery)

    def test_structurally_valid_join_with_no_real_fk_reports_canonicalization_error(
        self,
    ):
        c = Constraint(
            fact_references=[3],
            on=ONJoin(
                left=ONBaseTable(name="ORDER_ROW"),
                right=ONBaseTable(name="CUSTOMER"),
                on=[JoinCondition(left="ORDER_ROW.id", right="CUSTOMER.id")],
            ),
            condition=_simple_condition(),
            category="logic",
            severity="hard",
        )
        agent = DeterministicCheckerLoopAgent()
        errors = agent._canonicalize_list([c], "Logic", _schema(), _view())
        assert len(errors) == 1
        assert "Logic[0] ON canonicalization failed" in errors[0]

    def test_plain_base_table_needs_no_normalization(self):
        c = Constraint(
            fact_references=[4],
            on=ONBaseTable(name="ORDER_ROW"),
            condition=_simple_condition(),
            category="logic",
            severity="hard",
        )
        agent = DeterministicCheckerLoopAgent()
        errors = agent._canonicalize_list([c], "Logic", _schema(), _view())
        assert errors == []


def _self_join_schema() -> Schema:
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


class TestColumnAccessibilityWiring:
    """The new validate_column() wiring: once canonicalize() succeeds,
    every column a constraint's own fields reference must resolve
    unambiguously against the resolved Grain."""

    def test_condition_referencing_nonexistent_column_is_rejected(self):
        c = Constraint(
            fact_references=[10],
            on=ONBaseTable(name="ORDER_ROW"),
            condition=RComparison(
                op=">",
                left=RColumnRef(name="this_column_does_not_exist"),
                right=RLiteral(value=0),
            ),
            category="logic",
            severity="hard",
        )
        agent = DeterministicCheckerLoopAgent()
        errors = agent._canonicalize_list([c], "Logic", _schema(), _view())
        assert len(errors) == 1
        assert "Logic[0] column 'this_column_does_not_exist' invalid" in errors[0]

    def test_condition_referencing_ambiguous_self_join_column_is_rejected(self):
        c = Constraint(
            fact_references=[11],
            on=ONJoin(
                left=ONBaseTable(name="CATEGORY"),
                right=ONBaseTable(name="CATEGORY"),
                on=[JoinCondition(left="CATEGORY.parent_id", right="CATEGORY.id")],
            ),
            condition=RComparison(
                op=">", left=RColumnRef(name="id"), right=RLiteral(value=0)
            ),
            category="logic",
            severity="hard",
        )
        agent = DeterministicCheckerLoopAgent()
        schema = _self_join_schema()
        errors = agent._canonicalize_list(
            [c], "Logic", schema, _SchemaView.from_schema(schema)
        )
        assert len(errors) == 1
        assert "column 'id' invalid" in errors[0]
        assert "ambiguous" in errors[0]

    def test_condition_referencing_joined_table_column_is_accepted(self):
        c = Constraint(
            fact_references=[12],
            on=ONJoin(
                left=ONBaseTable(name="ORDER_ROW"),
                right=ONBaseTable(name="CUSTOMER"),
                on=[JoinCondition(left="ORDER_ROW.customer_id", right="CUSTOMER.id")],
            ),
            condition=RComparison(
                op="=",
                left=RColumnRef(name="loyalty_tier"),
                right=RLiteral(value="Gold"),
            ),
            category="logic",
            severity="hard",
        )
        agent = DeterministicCheckerLoopAgent()
        errors = agent._canonicalize_list([c], "Logic", _schema(), _view())
        assert errors == []

    def test_distribution_column_field_is_checked(self):
        d = DistributionConstraint(
            fact_references=[13],
            on=ONBaseTable(name="ORDER_ROW"),
            column="not_a_real_column",
            family="GAUSSIAN",
            parameters={"mean": 100, "std_dev": 10},
        )
        agent = DeterministicCheckerLoopAgent()
        errors = agent._canonicalize_list([d], "Distribution", _schema(), _view())
        assert len(errors) == 1
        assert "column 'not_a_real_column' invalid" in errors[0]

    def test_distribution_if_condition_columns_are_checked(self):
        d = DistributionConstraint(
            fact_references=[14],
            on=ONBaseTable(name="ORDER_ROW"),
            column="total",
            family="GAUSSIAN",
            parameters={"mean": 100, "std_dev": 10},
            if_condition=RComparison(
                op="=",
                left=RColumnRef(name="not_a_real_column"),
                right=RLiteral(value=1),
            ),
        )
        agent = DeterministicCheckerLoopAgent()
        errors = agent._canonicalize_list([d], "Distribution", _schema(), _view())
        assert len(errors) == 1
        assert "column 'not_a_real_column' invalid" in errors[0]

    def test_correlated_columns_are_checked(self):
        c = CorrelatedConstraint(
            fact_references=[15],
            on=ONBaseTable(name="ORDER_ROW"),
            columns=["total", "not_a_real_column"],
        )
        agent = DeterministicCheckerLoopAgent()
        errors = agent._canonicalize_list([c], "Correlation", _schema(), _view())
        assert len(errors) == 1
        assert "column 'not_a_real_column' invalid" in errors[0]

    def test_state_sequence_sequence_column_is_checked(self):
        s = StateSequenceConstraint(
            fact_references=[16],
            on=ONBaseTable(name="ORDER_ROW"),
            sequence_column="not_a_real_column",
        )
        agent = DeterministicCheckerLoopAgent()
        errors = agent._canonicalize_list([s], "StateSequence", _schema(), _view())
        assert len(errors) == 1
        assert "column 'not_a_real_column' invalid" in errors[0]

    def test_state_sequence_with_all_valid_columns_passes(self):
        s = StateSequenceConstraint(
            fact_references=[18],
            on=ONBaseTable(name="ORDER_ROW"),
            sequence_column="total",
        )
        agent = DeterministicCheckerLoopAgent()
        errors = agent._canonicalize_list([s], "StateSequence", _schema(), _view())
        assert errors == []


class TestCanonicalizeAllCoversEveryList:
    def test_empty_output_has_no_errors(self):
        agent = DeterministicCheckerLoopAgent()
        errors = agent._canonicalize_all(UnifiedOutput(), _schema())
        assert errors == []
