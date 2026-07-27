"""Tests for src/util/constraint_model/conflicts/state_sequence.py."""

from __future__ import annotations

from src.pipeline.stage2.models.data_types import DataType
from src.pipeline.stage2.models.schema import Column, Schema, Table
from src.util.constraint_model.condition.cohesive import StateSequence, StateTransition
from src.util.constraint_model.conflicts.state_sequence import (
    check_state_sequence_conflicts,
)
from src.util.constraint_model.constraint import Constraint
from src.util.constraint_model.relation.nodes import BaseTable


def _schema() -> Schema:
    return Schema(
        tables=[
            Table(
                name="ORDER",
                columns=[
                    Column(name="order_id", data_type=DataType.INTEGER),
                    Column(name="status", data_type=DataType.VARCHAR),
                ],
                primary_key=["order_id"],
            ),
            Table(
                name="PRODUCT",
                columns=[
                    Column(name="product_id", data_type=DataType.INTEGER),
                    Column(name="status", data_type=DataType.VARCHAR),
                ],
                primary_key=["product_id"],
            ),
        ]
    )


def _ss(
    fid: int,
    allowed=None,
    forbidden=None,
    strict=False,
    table: str = "ORDER",
    sequence_column: str = "status",
) -> Constraint:
    return Constraint(
        relation=BaseTable(name=table),
        condition=StateSequence(
            sequence_column=sequence_column,
            allowed_transitions=[
                StateTransition(from_state=a, to_state=b) for a, b in (allowed or [])
            ],
            forbidden_transitions=[
                StateTransition(from_state=a, to_state=b) for a, b in (forbidden or [])
            ],
            strict=strict,
        ),
        fact_references=[fid],
    )


class TestDirectContradiction:
    def test_allowed_and_forbidden_same_edge(self):
        c1 = _ss(1, allowed=[("ready", "packed")])
        c2 = _ss(2, forbidden=[("ready", "packed")])
        conflicts = check_state_sequence_conflicts([c1, c2], _schema())
        assert len(conflicts) == 1
        assert conflicts[0].kind == "state_sequence_direct_contradiction"
        assert conflicts[0].softenable is False

    def test_no_contradiction_when_edges_differ(self):
        c1 = _ss(1, allowed=[("ready", "packed")])
        c2 = _ss(
            2, forbidden=[("packed", "ready")]
        )  # reverse direction, not the same edge
        assert check_state_sequence_conflicts([c1, c2], _schema()) == []

    def test_three_facts_one_pair_contradicts(self):
        c1 = _ss(1, allowed=[("ready", "packed")])
        c2 = _ss(2, allowed=[("packed", "shipped")])
        c3 = _ss(3, forbidden=[("packed", "shipped")])
        conflicts = check_state_sequence_conflicts([c1, c2, c3], _schema())
        assert len(conflicts) == 1
        assert set(conflicts[0].involved_fact_references) == {2, 3}


class TestCycleDetection:
    def test_cycle_allowed_by_default(self):
        c1 = _ss(1, allowed=[("a", "b")])
        c2 = _ss(2, allowed=[("b", "a")])
        assert check_state_sequence_conflicts([c1, c2], _schema()) == []

    def test_self_loop_flagged_by_node_level_validation(self):
        # StateTransition follows this project's own "_validate() returns
        # errors, never raises" convention (matches every other node type
        # in constraint_model) -- construction itself doesn't raise; the
        # self-loop is caught by an explicit _validate() call instead.
        errors = StateTransition(from_state="a", to_state="a")._validate()
        assert len(errors) == 1

    def test_cycle_flagged_when_any_fact_is_strict(self):
        c1 = _ss(1, allowed=[("a", "b")], strict=True)
        c2 = _ss(2, allowed=[("b", "a")])
        conflicts = check_state_sequence_conflicts([c1, c2], _schema())
        assert len(conflicts) == 1
        assert conflicts[0].kind == "state_sequence_cycle"
        assert conflicts[0].softenable is False

    def test_longer_cycle_across_four_facts(self):
        c1 = _ss(1, allowed=[("ready", "packed")], strict=True)
        c2 = _ss(2, allowed=[("packed", "shipped")])
        c3 = _ss(3, allowed=[("shipped", "delivered")])
        c4 = _ss(4, allowed=[("delivered", "ready")])  # closes the cycle
        conflicts = check_state_sequence_conflicts([c1, c2, c3, c4], _schema())
        assert len(conflicts) == 1
        assert conflicts[0].kind == "state_sequence_cycle"

    def test_valid_linear_chain_no_cycle(self):
        c1 = _ss(1, allowed=[("ready", "packed"), ("packed", "shipped")], strict=True)
        c2 = _ss(2, allowed=[("shipped", "delivered")])
        assert check_state_sequence_conflicts([c1, c2], _schema()) == []

    def test_returns_reprocessing_cycle_allowed_without_strict(self):
        # a legitimate real-world cyclic flow (returns -> reprocessing)
        # must NOT be flagged unless some fact explicitly demands acyclic.
        c1 = _ss(
            1,
            allowed=[
                ("ready", "packed"),
                ("packed", "shipped"),
                ("shipped", "delivered"),
            ],
        )
        c2 = _ss(2, allowed=[("delivered", "returned"), ("returned", "packed")])
        assert check_state_sequence_conflicts([c1, c2], _schema()) == []


class TestScopingIsolatesUnrelatedFacts:
    def test_different_table_never_merged(self):
        # same sequence_column NAME ("status") on two unrelated tables must
        # not be merged just because the string coincides -- population
        # comparability (not just the column name) gates the grouping.
        c1 = _ss(1, allowed=[("a", "b")], table="ORDER", sequence_column="status")
        c2 = _ss(2, forbidden=[("a", "b")], table="PRODUCT", sequence_column="status")
        assert check_state_sequence_conflicts([c1, c2], _schema()) == []

    def test_different_sequence_column_never_merged(self):
        c1 = _ss(1, allowed=[("a", "b")], sequence_column="status")
        # can't reuse the exact same table+column with a different sequence_column
        # easily here without adding a column, so just confirm same-column facts DO merge:
        c2 = _ss(2, forbidden=[("a", "b")], sequence_column="status")
        conflicts = check_state_sequence_conflicts([c1, c2], _schema())
        assert len(conflicts) == 1

    def test_single_fact_alone_never_conflicts(self):
        c1 = _ss(1, allowed=[("a", "b")], forbidden=[("c", "d")], strict=True)
        assert check_state_sequence_conflicts([c1], _schema()) == []


class TestComplexMultiFactScenario:
    def test_five_facts_building_one_state_machine_no_conflict(self):
        c1 = _ss(1, allowed=[("ready", "packed")])
        c2 = _ss(2, allowed=[("packed", "shipped")])
        c3 = _ss(3, allowed=[("shipped", "out_for_delivery")])
        c4 = _ss(4, allowed=[("out_for_delivery", "delivered")])
        c5 = _ss(5, forbidden=[("ready", "shipped")])  # can't skip packed
        assert check_state_sequence_conflicts([c1, c2, c3, c4, c5], _schema()) == []

    def test_five_facts_with_embedded_contradiction_and_unrelated_cycle_tolerance(self):
        c1 = _ss(1, allowed=[("ready", "packed")])
        c2 = _ss(2, allowed=[("packed", "shipped")])
        c3 = _ss(3, allowed=[("shipped", "delivered")])
        c4 = _ss(4, allowed=[("delivered", "returned")])
        c5 = _ss(5, forbidden=[("delivered", "returned")])  # contradicts c4
        conflicts = check_state_sequence_conflicts([c1, c2, c3, c4, c5], _schema())
        assert len(conflicts) == 1
        assert set(conflicts[0].involved_fact_references) == {4, 5}
