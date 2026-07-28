"""Tests for src/pipeline/stage3/middleware/deterministic_checker.py.

Focus: the normalize_on() -> canonicalize() wiring in _canonicalize_list()
(previously zero coverage on this module). A generic Constraint's `on` can
legally hold an RawSQL -- these tests confirm normalize_on() replaces it
in place (so canonicalize() sees a structured tree, and the bridge later
never sees a raw RawSQL) and that both normalization and
canonicalization failures surface as distinct, labeled error strings.
"""

from __future__ import annotations

from src.util.schema_model.data_types import DataType
from src.util.schema_model.schema import Column, ForeignKey, Schema, Table
from src.pipeline.stage3.agents.extraction_outputs import UnifiedOutput
from src.pipeline.stage3.middleware.deterministic_checker import (
    DeterministicCheckerLoopAgent,
)
from src.util.constraint_model.condition.expressions import RColumnRef, RLiteral
from src.util.constraint_model.condition.predicates import RAnd, RComparison
from src.pipeline.stage3.models.cross_shard import (
    Constraint,
    CorrelatedConstraint,
    DistributionConstraint,
    StateSequenceConstraint,
)
from src.pipeline.stage3.models.grain import _SchemaView
from src.util.constraint_model.relation.nodes import (
    Aggregate,
    BaseTable,
    Fanout,
    Join,
    JoinCondition,
    RawSQL,
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
            on=RawSQL(
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
        # The constraint's own `on` was replaced in place -- no RawSQL
        # survives past this node, so the bridge never has to see one.
        assert isinstance(c.on, Join)
        assert isinstance(c.on.left, BaseTable)
        assert isinstance(c.on.right, BaseTable)

    def test_where_clause_in_an_on_subquery_is_rejected(self):
        """A WHERE inside `on` has no ON-tree meaning -- row filtering belongs
        in the constraint's own condition.

        This used to fail at normalization, because the old translation layer
        had its own opinion about which node shapes were legal. Now from_sql()
        parses the WHERE into a Filter perfectly well and canonicalize() is the
        single place that decides a Filter is not a legal ON tree. Same
        rejection, one layer later, and only one layer knows the rule."""
        c = Constraint(
            fact_references=[2],
            on=RawSQL(sql="SELECT * FROM ORDER_ROW WHERE total > 100"),
            condition=_simple_condition(),
            category="logic",
            severity="hard",
        )
        agent = DeterministicCheckerLoopAgent()
        errors = agent._canonicalize_list([c], "Logic", _schema(), _view())
        assert len(errors) == 1
        assert "Logic[0] ON canonicalization failed" in errors[0]
        assert "Filter" in errors[0]

    def test_structurally_valid_join_with_no_real_fk_reports_canonicalization_error(
        self,
    ):
        c = Constraint(
            fact_references=[3],
            on=Join(
                left=BaseTable(name="ORDER_ROW"),
                right=BaseTable(name="CUSTOMER"),
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
            on=BaseTable(name="ORDER_ROW"),
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
            on=BaseTable(name="ORDER_ROW"),
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
            on=Join(
                left=BaseTable(name="CATEGORY"),
                right=BaseTable(name="CATEGORY"),
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
            on=Join(
                left=BaseTable(name="ORDER_ROW"),
                right=BaseTable(name="CUSTOMER"),
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
            on=BaseTable(name="ORDER_ROW"),
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
            on=BaseTable(name="ORDER_ROW"),
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
            on=BaseTable(name="ORDER_ROW"),
            columns=["total", "not_a_real_column"],
        )
        agent = DeterministicCheckerLoopAgent()
        errors = agent._canonicalize_list([c], "Correlation", _schema(), _view())
        assert len(errors) == 1
        assert "column 'not_a_real_column' invalid" in errors[0]

    def test_state_sequence_sequence_column_is_checked(self):
        s = StateSequenceConstraint(
            fact_references=[16],
            on=BaseTable(name="ORDER_ROW"),
            sequence_column="not_a_real_column",
        )
        agent = DeterministicCheckerLoopAgent()
        errors = agent._canonicalize_list([s], "StateSequence", _schema(), _view())
        assert len(errors) == 1
        assert "column 'not_a_real_column' invalid" in errors[0]

    def test_state_sequence_with_all_valid_columns_passes(self):
        s = StateSequenceConstraint(
            fact_references=[18],
            on=BaseTable(name="ORDER_ROW"),
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


class TestVacuousBounds:
    """A bound every possible value satisfies constrains nothing, but still
    canonicalizes cleanly and becomes a degree-of-freedom variable for Stage 4.

    A live Gemini run emitted exactly `child_count >= 0` on a fanout. A count
    is never negative, so it asserts nothing. This was previously only
    discouraged by a prompt rule -- the model policing itself -- which is the
    wrong layer for a purely mechanical check.
    """

    def _fanout_with(self, op, value):
        return Constraint(
            fact_references=[1],
            on=Fanout(
                parent_table="CUSTOMER",
                child_table="ORDER_ROW",
                fk_column="customer_id",
            ),
            condition=RComparison(
                op=op,
                left=RColumnRef(name="child_count"),
                right=RLiteral(value=value),
            ),
            category="structural",
        )

    def _check(self, item):
        agent = DeterministicCheckerLoopAgent()
        return agent._canonicalize_list([item], "Structural", _schema(), _view())

    def test_child_count_ge_zero_is_rejected(self):
        errors = self._check(self._fanout_with(">=", 0))
        assert len(errors) == 1
        assert "vacuous" in errors[0]
        assert "child_count" in errors[0]

    def test_child_count_gt_negative_is_rejected(self):
        assert self._check(self._fanout_with(">", -1)) != []

    def test_child_count_ne_negative_is_rejected(self):
        assert self._check(self._fanout_with("!=", -5)) != []

    def test_the_real_bounds_are_accepted(self):
        """'multiple' -> > 1 and 'at least one' -> >= 1 both say something."""
        assert self._check(self._fanout_with(">", 1)) == []
        assert self._check(self._fanout_with(">=", 1)) == []

    def test_upper_bounds_are_untouched(self):
        """<= 0 on a count is restrictive (means exactly zero), not vacuous."""
        assert self._check(self._fanout_with("<=", 0)) == []

    def test_ordinary_columns_are_not_assumed_non_negative(self):
        """The rule applies to COUNTS, not to arbitrary numeric columns -- a
        real column may legitimately be negative, so `total >= 0` is a genuine
        constraint and must not be flagged."""
        c = Constraint(
            fact_references=[2],
            on=BaseTable(name="ORDER_ROW"),
            condition=RComparison(
                op=">=", left=RColumnRef(name="total"), right=RLiteral(value=0)
            ),
            category="logic",
        )
        assert self._check(c) == []

    def test_count_aggregate_alias_is_also_protected(self):
        """A COUNT aggregate's own alias is non-negative for the same reason
        child_count is."""
        c = Constraint(
            fact_references=[3],
            on=Aggregate(
                source=BaseTable(name="ORDER_ROW"),
                fn="COUNT",
                column="*",
                group_by=["customer_id"],
                alias="n_orders",
            ),
            condition=RComparison(
                op=">=", left=RColumnRef(name="n_orders"), right=RLiteral(value=0)
            ),
            category="structural",
        )
        assert self._check(c) != []

    def test_nested_inside_a_compound_predicate_is_still_found(self):
        c = Constraint(
            fact_references=[4],
            on=Fanout(
                parent_table="CUSTOMER",
                child_table="ORDER_ROW",
                fk_column="customer_id",
            ),
            condition=RAnd(
                operands=[
                    RComparison(
                        op=">=",
                        left=RColumnRef(name="child_count"),
                        right=RLiteral(value=0),
                    ),
                    RComparison(
                        op="<",
                        left=RColumnRef(name="child_count"),
                        right=RLiteral(value=100),
                    ),
                ]
            ),
            category="structural",
        )
        assert self._check(c) != []
