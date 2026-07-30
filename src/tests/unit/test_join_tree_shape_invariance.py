"""A Join's TREE SHAPE must not change the Population it derives.

Regression cover for a defect found by probing Stage 3 on a 41-table spec:
population.py identified a join's parent table by walking the tree
(`_root_table_name`), which returned None for a nested Join, so a right-deep
three-table join was rejected outright with "Join: could not determine the
parent side's own table name" while the semantically IDENTICAL left-deep join
was accepted. The generator picks the nesting freely, so this made
supportability depend on an arbitrary choice.

Two further things were wrong in the shape it DID accept, and both are pinned
here because fixing only the rejection would have made them worse rather than
visible: the parent side's own population was never computed, so every edge
internal to that subtree was silently dropped; and the edge recorded the
CHILD'S GRAIN as the FK-holding table, so a two-hop join claimed the FK column
sat on a table that does not have it.
"""

from __future__ import annotations

from src.util.constraint_model.population import compute_population
from src.util.constraint_model.relation.nodes import BaseTable, Join, JoinCondition
from src.util.schema_model.data_types import DataType
from src.util.schema_model.schema import Column, ForeignKey, Schema, Table

RIDE_RIDER = JoinCondition(left="RIDE.rider_id", right="RIDER.rider_id")
RIDER_CLUB = JoinCondition(left="RIDER.club_id", right="CLUB.club_id")


def _schema() -> Schema:
    """CLUB <- RIDER <- RIDE: a plain two-hop parent chain."""
    return Schema(
        tables=[
            Table(
                name="CLUB",
                columns=[
                    Column(name="club_id", data_type=DataType.INTEGER),
                    Column(name="club_name", data_type=DataType.VARCHAR),
                ],
                primary_key=["club_id"],
            ),
            Table(
                name="RIDER",
                columns=[
                    Column(name="rider_id", data_type=DataType.INTEGER),
                    Column(name="club_id", data_type=DataType.INTEGER),
                ],
                primary_key=["rider_id"],
            ),
            Table(
                name="RIDE",
                columns=[
                    Column(name="ride_id", data_type=DataType.INTEGER),
                    Column(name="rider_id", data_type=DataType.INTEGER),
                ],
                primary_key=["ride_id"],
            ),
        ],
        relationships=[
            ForeignKey(
                referencing_table="RIDER",
                referencing_column="club_id",
                referred_table="CLUB",
            ),
            ForeignKey(
                referencing_table="RIDE",
                referencing_column="rider_id",
                referred_table="RIDER",
            ),
        ],
    )


def _left_deep() -> Join:
    """((RIDE join RIDER) join CLUB) -- parent side is a BaseTable."""
    return Join(
        left=Join(
            left=BaseTable(name="RIDE"), right=BaseTable(name="RIDER"), on=[RIDE_RIDER]
        ),
        right=BaseTable(name="CLUB"),
        on=[RIDER_CLUB],
    )


def _right_deep() -> Join:
    """(RIDE join (RIDER join CLUB)) -- parent side is itself a Join."""
    return Join(
        left=BaseTable(name="RIDE"),
        right=Join(
            left=BaseTable(name="RIDER"), right=BaseTable(name="CLUB"), on=[RIDER_CLUB]
        ),
        on=[RIDE_RIDER],
    )


class TestJoinTreeShapeInvariance:
    def test_right_deep_join_is_supported(self):
        """The shape that used to be rejected outright."""
        pop, errors = compute_population(_right_deep(), _schema())
        assert errors == []
        assert pop is not None

    def test_both_nestings_derive_the_same_population(self):
        """The invariant. Nesting is the generator's arbitrary choice, so it
        must not change population identity -- two constraints written over the
        same join must compare as the same population."""
        schema = _schema()
        left, left_errs = compute_population(_left_deep(), schema)
        right, right_errs = compute_population(_right_deep(), schema)
        assert left_errs == [] and right_errs == []
        assert left is not None and right is not None
        assert left == right

    def test_grain_is_the_child_table(self):
        for tree in (_left_deep(), _right_deep()):
            pop, _ = compute_population(tree, _schema())
            assert pop is not None
            assert pop.table == "RIDE"

    def test_every_traversed_fk_is_recorded_on_the_table_that_holds_it(self):
        """Both hops must appear, each attributed to the table declaring the FK.

        The parent subtree's internal RIDER->CLUB hop used to vanish in the
        right-deep case, and in the left-deep case it was recorded against RIDE,
        which has no club_id column at all.
        """
        for tree in (_left_deep(), _right_deep()):
            pop, _ = compute_population(tree, _schema())
            assert pop is not None
            edges = {(e.child_table, e.fk_column, e.parent_table) for e, _ in pop.edges}
            assert edges == {
                ("RIDE", "rider_id", "RIDER"),
                ("RIDER", "club_id", "CLUB"),
            }
