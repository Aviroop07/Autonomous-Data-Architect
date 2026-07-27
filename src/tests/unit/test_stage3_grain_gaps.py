"""Characterization tests for two real, PRE-EXISTING gaps in grain.py
found while stress-testing on_sql_normalize.py -- neither is caused by,
or specific to, the RawSQL/SQL path; both apply equally to any
directly-authored ON tree.

1. Grain.pk_columns for a multi-column GROUP BY aggregate stays the
   SOURCE table's original PK, not the composite group_by set --
   unlike constraint_model/population.py's Population, which explicitly
   computes pk_columns=frozenset(group_by) for exactly this case (see
   its own module comment). Confirmed inert today: accessible_columns()
   derives from agg_signature, never from pk_columns, so nothing reads
   the stale value -- but it IS stale if anyone ever does. Still an
   open gap, left as a characterization test rather than fixed.

2. Grain.validate_column()/accessible_columns() were built to detect
   ambiguous/nonexistent column references (and are correct: see the
   self-join case below) but were, at first, never actually CALLED
   anywhere in the live pipeline -- canonicalize() alone still has no
   notion of a constraint's own condition/if_condition, only its ON
   tree's structure. THIS GAP IS NOW CLOSED: deterministic_checker.py's
   _canonicalize_list() calls validate_column() against the resolved
   Grain for every column each constraint type references (condition,
   if_condition, column, columns, partition_by, sequence_column,
   order_by), once canonicalize() itself succeeds. See
   test_stage3_deterministic_checker.py for the wiring's own tests --
   the last test below now documents that the wiring IS present, the
   opposite of what it originally asserted.
"""

from __future__ import annotations

from src.pipeline.stage2.models.data_types import DataType
from src.pipeline.stage2.models.schema import Column, ForeignKey, Schema, Table
from src.pipeline.stage3.models.grain import CanonicalizationFailure, canonicalize
from src.util.constraint_model.relation.nodes import (
    JoinCondition,
    Aggregate,
    BaseTable,
    Join,
)


def _order_schema() -> Schema:
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
                    Column(
                        name="region", data_type=DataType.VARCHAR, is_nullable=False
                    ),
                ],
            ),
        ],
    )


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


class TestMultiColumnGroupByPkColumnsStaysStale:
    def test_pk_columns_is_not_the_composite_group_by_set(self):
        """Documents current behavior -- NOT the semantically ideal
        answer. Contrast with constraint_model/population.py's
        Population, which correctly sets pk_columns=frozenset(group_by)
        for the same shape."""
        agg = Aggregate(
            source=BaseTable(name="ORDER_ROW"),
            fn="AVG",
            column="total",
            group_by=["customer_id", "region"],
            alias="avg_total",
        )
        grain = canonicalize(agg, _order_schema())
        assert not isinstance(grain, CanonicalizationFailure)
        assert grain.pk_columns == frozenset({"id"})  # stale: ORDER_ROW's own PK
        assert grain.pk_columns != frozenset({"customer_id", "region"})

    def test_staleness_is_inert_for_accessibility_purposes(self):
        """accessible_columns() derives from agg_signature, not
        pk_columns -- so the staleness above doesn't currently produce
        a wrong accessible-column answer."""
        from src.pipeline.stage3.models.grain import _SchemaView

        agg = Aggregate(
            source=BaseTable(name="ORDER_ROW"),
            fn="AVG",
            column="total",
            group_by=["customer_id", "region"],
            alias="avg_total",
        )
        schema = _order_schema()
        grain = canonicalize(agg, schema)
        assert not isinstance(grain, CanonicalizationFailure)
        accessible, ambiguous = grain.accessible_columns(
            _SchemaView.from_schema(schema)
        )
        assert accessible == frozenset({"customer_id", "region", "avg_total"})
        assert ambiguous == frozenset()


class TestColumnAccessibilityValidation:
    def test_aggregate_of_aggregate_with_nonexistent_column_is_silently_accepted(self):
        """canonicalize() has no notion of the CONSTRAINT's own condition
        at all -- it only validates the ON tree's structure. An outer
        aggregate's `column` is never checked against what its `source`
        actually exposes, aggregate-of-aggregate or not."""
        inner = Aggregate(
            source=BaseTable(name="ORDER_ROW"),
            fn="SUM",
            column="total",
            group_by=["customer_id"],
            alias="sum_total",
        )
        outer = Aggregate(
            source=inner,
            fn="AVG",
            column="this_column_does_not_exist_anywhere",
            group_by=None,
            alias="avg_of_sums",
        )
        result = canonicalize(outer, _order_schema())
        assert not isinstance(result, CanonicalizationFailure)

    def test_validate_column_would_catch_self_join_ambiguity_if_called(self):
        """Proves the detection logic ITSELF is correct -- the gap is
        purely that nothing in the live pipeline calls it. A
        self-referential join (both sides literally "CATEGORY", no
        alias survives in the ON-tree's own shape) makes both id and
        parent_id genuinely ambiguous without table-qualification."""
        from src.pipeline.stage3.models.grain import _SchemaView

        join = Join(
            left=BaseTable(name="CATEGORY"),
            right=BaseTable(name="CATEGORY"),
            on=[JoinCondition(left="CATEGORY.parent_id", right="CATEGORY.id")],
        )
        schema = _self_join_schema()
        grain = canonicalize(join, schema)
        assert not isinstance(grain, CanonicalizationFailure)  # no ambiguity check here

        view = _SchemaView.from_schema(schema)
        err = grain.validate_column("id", view)
        assert err is not None and "ambiguous" in err

    def test_deterministic_checker_now_calls_validate_column(self):
        """Was originally written asserting the OPPOSITE (the wiring gap)
        -- inverted once the gap was closed, so this file's premise can't
        silently go stale in either direction. If this ever starts
        failing again, the wiring was removed and the gap re-opened."""
        import inspect

        from src.pipeline.stage3.middleware import deterministic_checker

        source = inspect.getsource(deterministic_checker)
        assert "validate_column" in source
