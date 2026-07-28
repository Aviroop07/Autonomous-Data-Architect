"""Tests for src/pipeline/stage3/bridge/from_cross_shard.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.util.schema_model.data_types import DataType
from src.util.schema_model.schema import Column, Schema, Table
from src.util.constraint_model.condition.expressions import RColumnRef, RLiteral
from src.util.constraint_model.condition.predicates import (
    RBetween,
    RComparison,
    RIfThen,
)
from src.pipeline.stage3.models.cross_shard import (
    Constraint,
    CorrelatedConstraint,
    DistributionConstraint,
    PairwiseCorrelationSpec,
    StateSequenceConstraint,
    StateTransitionSpec,
)
from src.util.constraint_model.relation.nodes import (
    Aggregate,
    BaseTable,
    Fanout,
    Join,
    RawSQL,
)
from src.util.constraint_model.relation.nodes import JoinCondition as CSJoinCondition
from src.pipeline.stage3.bridge.from_cross_shard import bridge_constraints
from src.util.constraint_model.condition.cohesive import Correlated, Distributed
from src.util.constraint_model.condition.cohesive import StateSequence as MStateSequence
from src.util.constraint_model.relation.nodes import (
    Filter,
)


def _schema() -> Schema:
    return Schema(
        tables=[
            Table(
                name="ORDER",
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
    )


class TestBridgeDistribution:
    def test_unconditional_distribution_bridges_to_base_table_plus_distributed(self):
        d = DistributionConstraint(
            fact_references=[1],
            on=BaseTable(name="ORDER"),
            column="total",
            family="GAUSSIAN",
            parameters={"mean": 100, "std_dev": 10},
        )
        bridged, unsupported = bridge_constraints([d], [], [], [], [], _schema())
        assert unsupported == []
        assert len(bridged) == 1
        assert isinstance(bridged[0].relation, BaseTable)
        assert bridged[0].relation.name == "ORDER"
        assert isinstance(bridged[0].condition, Distributed)
        assert bridged[0].condition.family == "GAUSSIAN"

    def test_conditional_distribution_bridges_if_condition_to_a_filter_wrapper(self):
        # DistributionConstraint.on is typed BaseTable only (a single-table
        # constraint) -- if_condition restricts to a same-table categorical
        # column, not a cross-table join.
        d = DistributionConstraint(
            fact_references=[2],
            on=BaseTable(name="ORDER"),
            column="total",
            family="GAUSSIAN",
            parameters={"mean": 300, "std_dev": 50},
            if_condition=RComparison(
                op="=",
                left=RColumnRef(name="customer_id"),
                right=RLiteral(value=7),
            ),
        )
        bridged, unsupported = bridge_constraints([d], [], [], [], [], _schema())
        assert unsupported == []
        assert len(bridged) == 1
        assert isinstance(bridged[0].relation, Filter)
        assert isinstance(bridged[0].relation.source, BaseTable)


class TestBridgeGenericConstraint:
    def test_moment_target_over_aggregate_bridges_relation_and_condition(self):
        c = Constraint(
            fact_references=[3],
            on=Aggregate(
                source=BaseTable(name="ORDER"),
                fn="AVG",
                column="total",
                group_by=["customer_id"],
                alias="avg_total",
            ),
            condition=RComparison(
                op="=",
                left=RColumnRef(name="avg_total"),
                right=RLiteral(value=250),
            ),
            category="statistical",
        )
        bridged, unsupported = bridge_constraints([], [c], [], [], [], _schema())
        assert unsupported == []
        assert len(bridged) == 1
        assert isinstance(bridged[0].relation, Aggregate)
        assert bridged[0].relation.fn == "AVG"

    def test_fanout_constraint_bridges_to_fanout_relation(self):
        c = Constraint(
            fact_references=[4],
            on=Fanout(
                parent_table="CUSTOMER", child_table="ORDER", fk_column="customer_id"
            ),
            condition=RBetween(
                expr=RColumnRef(name="child_count"),
                low=RLiteral(value=1),
                high=RLiteral(value=20),
            ),
            category="structural",
        )
        bridged, unsupported = bridge_constraints([], [], [], [c], [], _schema())
        assert unsupported == []
        assert isinstance(bridged[0].relation, Fanout)

    def test_exists_predicates_are_unconstructible_not_merely_rejected(self):
        """These used to be two tests asserting that RExists and RComparison
        op='~' reached the bridge and were REPORTED as unsupported. Both node
        types are now simply absent from the predicate taxonomy the extraction
        models use, so the constraint can no longer be built at all -- a
        strictly stronger guarantee than catching it one layer later, and one
        that also removes them from the JSON schema the LLM is given."""
        import src.util.constraint_model.condition.predicates as predicates

        assert not hasattr(predicates, "RExists")
        assert not hasattr(predicates, "RNotExists")
        assert not hasattr(predicates, "SubqueryRef")

        with pytest.raises(ValidationError):
            RComparison(
                op="~",  # type: ignore[arg-type]
                left=RColumnRef(name="total"),
                right=RLiteral(value=1),
            )

    def test_if_then_wrapped_conditional_bridges_recursively(self):
        c = Constraint(
            fact_references=[7],
            on=BaseTable(name="ORDER"),
            condition=RIfThen(
                antecedent=RComparison(
                    op="=", left=RColumnRef(name="total"), right=RLiteral(value=1)
                ),
                consequent=RComparison(
                    op=">=", left=RColumnRef(name="total"), right=RLiteral(value=0)
                ),
            ),
            category="logic",
        )
        bridged, unsupported = bridge_constraints([], [], [], [], [c], _schema())
        assert unsupported == []
        assert len(bridged) == 1


class TestBridgeCorrelated:
    def test_same_table_correlation_with_pairwise_value_bridges_directly(self):
        c = CorrelatedConstraint(
            fact_references=[10],
            on=BaseTable(name="ORDER"),
            columns=["total", "customer_id"],
            family="GAUSSIAN",
            pairwise=[
                PairwiseCorrelationSpec(left="total", right="customer_id", value=0.4)
            ],
        )
        bridged, unsupported = bridge_constraints([], [], [c], [], [], _schema())
        assert unsupported == []
        assert len(bridged) == 1
        assert isinstance(bridged[0].relation, BaseTable)
        assert isinstance(bridged[0].condition, Correlated)
        assert bridged[0].condition.columns == ["total", "customer_id"]
        assert bridged[0].condition.pairwise[0].value == 0.4

    def test_qualitative_correlation_with_empty_pairwise_bridges_directly(self):
        c = CorrelatedConstraint(
            fact_references=[11],
            on=BaseTable(name="ORDER"),
            columns=["total", "customer_id"],
            pairwise=[],
        )
        bridged, unsupported = bridge_constraints([], [], [c], [], [], _schema())
        assert unsupported == []
        assert isinstance(bridged[0].condition, Correlated)
        assert bridged[0].condition.pairwise == []
        assert bridged[0].condition.family == "GAUSSIAN"  # default

    def test_cross_table_correlation_bridges_via_join(self):
        c = CorrelatedConstraint(
            fact_references=[12],
            on=Join(
                left=BaseTable(name="ORDER"),
                right=BaseTable(name="CUSTOMER"),
                on=[CSJoinCondition(left="ORDER.customer_id", right="CUSTOMER.id")],
            ),
            columns=["total", "loyalty_tier"],
        )
        bridged, unsupported = bridge_constraints([], [], [c], [], [], _schema())
        assert unsupported == []
        assert isinstance(bridged[0].relation, Join)


class TestBridgeStateSequence:
    def test_state_sequence_with_transitions_bridges_directly(self):
        c = StateSequenceConstraint(
            fact_references=[20],
            on=BaseTable(name="ORDER"),
            sequence_column="customer_id",
            allowed_transitions=[
                StateTransitionSpec(from_state="a", to_state="b"),
            ],
            forbidden_transitions=[
                StateTransitionSpec(from_state="a", to_state="c"),
            ],
            strict=True,
        )
        bridged, unsupported = bridge_constraints(
            [], [], [], [], [], _schema(), state_sequences=[c]
        )
        assert unsupported == []
        assert len(bridged) == 1
        assert isinstance(bridged[0].relation, BaseTable)
        assert isinstance(bridged[0].condition, MStateSequence)
        assert bridged[0].condition.strict is True
        assert bridged[0].condition.allowed_transitions[0].from_state == "a"
        assert bridged[0].condition.forbidden_transitions[0].to_state == "c"

    def test_no_state_sequences_is_a_no_op(self):
        bridged, unsupported = bridge_constraints([], [], [], [], [], _schema())
        assert bridged == []
        assert unsupported == []


class TestCorrelatedConstraintValidation:
    def test_single_column_is_rejected_at_construction(self):
        # Field(min_length=2) enforces this structurally -- _validate()'s
        # own len(columns) < 2 check is unreachable, since Pydantic never
        # lets such an object exist to call _validate() on in the first
        # place (matches constraint_model's own Correlated, which relies
        # on the same Field(min_length=2) rather than a redundant check).
        with pytest.raises(ValidationError):
            CorrelatedConstraint(
                fact_references=[30], on=BaseTable(name="ORDER"), columns=["total"]
            )

    def test_duplicate_columns_is_rejected(self):
        c = CorrelatedConstraint(
            fact_references=[30],
            on=BaseTable(name="ORDER"),
            columns=["total", "total"],
        )
        errors = c._validate()
        assert any("duplicate" in e for e in errors)

    def test_pairwise_value_out_of_range_is_rejected(self):
        c = CorrelatedConstraint(
            fact_references=[31],
            on=BaseTable(name="ORDER"),
            columns=["total", "customer_id"],
            pairwise=[
                PairwiseCorrelationSpec(left="total", right="customer_id", value=1.5)
            ],
        )
        errors = c._validate()
        assert any("[-1, 1]" in e for e in errors)


class TestStateSequenceConstraintValidation:
    def test_self_loop_transition_is_rejected(self):
        t = StateTransitionSpec(from_state="a", to_state="a")
        errors = t._validate()
        assert len(errors) == 1

    def test_same_edge_allowed_and_forbidden_is_rejected(self):
        c = StateSequenceConstraint(
            fact_references=[32],
            on=BaseTable(name="ORDER"),
            sequence_column="customer_id",
            allowed_transitions=[StateTransitionSpec(from_state="a", to_state="b")],
            forbidden_transitions=[StateTransitionSpec(from_state="a", to_state="b")],
        )
        errors = c._validate()
        assert any("both allowed and forbidden" in e for e in errors)


class TestBridgeNeverDropsAConstraint:
    def test_every_constraint_in_a_mixed_batch_is_wrapped(self):
        """This used to assert that a bad constraint came back in
        `unsupported`. Nothing can be unsupported any more: `on` is already a
        RelationUnion and `condition` already an RPredicateUnion, so wrapping
        is total. The invariant worth keeping is the one the class is named
        for -- every input appears in the output, none is silently dropped.

        Constraints whose ON tree is nonetheless meaningless (an unnormalized
        RawSQL, a Filter) are rejected by canonicalize() in the checker node
        BEFORE reaching the bridge -- see test_stage3_deterministic_checker.py.
        """
        made = [
            Constraint(
                fact_references=[8],
                on=BaseTable(name="ORDER"),
                condition=RComparison(
                    op=">", left=RColumnRef(name="total"), right=RLiteral(value=0)
                ),
                category="logic",
            ),
            Constraint(
                fact_references=[9],
                on=RawSQL(sql="SELECT * FROM ORDER"),
                category="logic",
                condition=RComparison(
                    op=">", left=RColumnRef(name="total"), right=RLiteral(value=0)
                ),
            ),
        ]
        bridged, unsupported = bridge_constraints([], [], [], [], made, _schema())
        assert len(bridged) == len(made)
        assert unsupported == []
        assert [c.fact_references for c in bridged] == [[8], [9]]
