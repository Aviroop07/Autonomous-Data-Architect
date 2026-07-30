"""Two mapper defects found by probing Stage 2 on a 41-table spec.

1. A GENERIC NATURAL KEY POISONED FOREIGN-KEY INFERENCE. The extractor gave
   Club the identifier `name`, so CLUB's primary key became `name`; the
   natural-key rule in wire_orphan_fk_columns then read EVERY `name` column in
   the schema as a reference to CLUB and declared ROUTE.name and SEGMENT.name --
   each row's OWN name -- as foreign keys to CLUB, which requires a route to be
   named after a club. Neither the mapper nor Schema._validate() objected.

   The fix is at key SELECTION, not at FK inference, and that placement was
   measured rather than assumed: refusing the inference whenever the name also
   appears elsewhere as a non-key column ALSO refuses the correct
   RIDE.email -> RIDER and CLUB.region_code -> REGION, because a legitimate FK
   column is itself a non-key column on another table.

2. M:N JUNCTION ATTRIBUTES SILENTLY LOST NULLABILITY. Relationship attributes
   were mapped to Columns without passing is_nullable, which defaults to False,
   so an optional `date_left` became NOT NULL and the schema could no longer
   represent a CURRENT member. The entity-attribute path always passed it.
"""

from __future__ import annotations

import logging

from src.pipeline.stage2.mapper.conceptual_model import (
    CMAttribute,
    ConceptualModel,
    Entity,
    Participant,
    Relationship,
)
from src.pipeline.stage2.mapper.relational_mapper import (
    _non_distinctive_identifier_names,
    map_conceptual_to_relational,
)
from src.util.schema_model.data_types import DataType


def _attr(name: str, type_: DataType = DataType.VARCHAR, nullable: bool = False):
    return CMAttribute(name=name, type=type_, is_nullable=nullable)


def _model_with_generic_key() -> ConceptualModel:
    """Club is identified by `name`; Route and Segment each have their own
    ordinary `name`. This is the live shape, reduced."""
    return ConceptualModel(
        entities=[
            Entity(
                name="Club",
                attributes=[_attr("name"), _attr("founding_year", DataType.INTEGER)],
                identifier_attributes=["name"],
            ),
            Entity(
                name="Route",
                attributes=[
                    _attr("route_id", DataType.INTEGER),
                    _attr("name"),
                    _attr("distance", DataType.FLOAT),
                ],
                identifier_attributes=["route_id"],
            ),
            Entity(
                name="Segment",
                attributes=[
                    _attr("segment_id", DataType.INTEGER),
                    _attr("name"),
                ],
                identifier_attributes=["segment_id"],
            ),
        ],
        relationships=[],
        functional_dependencies=[],
    )


def _model_with_distinctive_key() -> ConceptualModel:
    """Region is identified by `region_code`, which no entity uses as an
    ordinary attribute -- a genuine natural key that must survive."""
    return ConceptualModel(
        entities=[
            Entity(
                name="Region",
                attributes=[_attr("region_code"), _attr("region_name")],
                identifier_attributes=["region_code"],
            ),
            Entity(
                name="Club",
                attributes=[_attr("club_id", DataType.INTEGER), _attr("name")],
                identifier_attributes=["club_id"],
            ),
        ],
        relationships=[],
        functional_dependencies=[],
    )


class TestNaturalKeyDistinctiveness:
    def test_name_used_as_an_ordinary_attribute_elsewhere_is_not_distinctive(self):
        plain = _non_distinctive_identifier_names(_model_with_generic_key())
        assert "name" in plain

    def test_a_key_name_used_nowhere_else_is_distinctive(self):
        plain = _non_distinctive_identifier_names(_model_with_distinctive_key())
        assert "region_code" not in plain

    def test_an_entitys_own_identifier_does_not_disqualify_itself(self):
        """Club.name is Club's identifier, so Club's own use of it must not be
        what marks it generic -- otherwise every natural key disqualifies
        itself and the rule degenerates to 'always use a surrogate'."""
        model = ConceptualModel(
            entities=[
                Entity(
                    name="Club",
                    attributes=[_attr("name")],
                    identifier_attributes=["name"],
                )
            ],
            relationships=[],
            functional_dependencies=[],
        )
        assert "name" not in _non_distinctive_identifier_names(model)

    def test_generic_key_falls_back_to_a_surrogate(self):
        schema = map_conceptual_to_relational(_model_with_generic_key())
        club = next(t for t in schema.tables if t.name == "CLUB")
        assert club.primary_key == ["club_id"]

    def test_distinctive_key_is_kept(self):
        schema = map_conceptual_to_relational(_model_with_distinctive_key())
        region = next(t for t in schema.tables if t.name == "REGION")
        assert region.primary_key == ["region_code"]

    def test_the_route_and_segment_names_are_not_wired_to_club(self):
        """The actual harm. With CLUB keyed on a surrogate, no table has a
        single-column PK named `name`, so the natural-key rule cannot fire."""
        schema = map_conceptual_to_relational(_model_with_generic_key())
        schema.wire_orphan_fk_columns()
        wired = {
            (r.referencing_table, r.referencing_column, r.referred_table)
            for r in schema.relationships or []
        }
        assert ("ROUTE", "name", "CLUB") not in wired
        assert ("SEGMENT", "name", "CLUB") not in wired

    def test_the_substitution_is_logged_with_its_reason(self, caplog):
        """A silently different primary key is exactly the kind of change that
        cost a day of investigation here, so it must announce itself."""
        with caplog.at_level(logging.INFO):
            map_conceptual_to_relational(_model_with_generic_key())
        assert "Club" in caplog.text
        assert "name" in caplog.text


class TestJunctionAttributeNullability:
    def _membership_model(self) -> ConceptualModel:
        return ConceptualModel(
            entities=[
                Entity(
                    name="Club",
                    attributes=[_attr("club_id", DataType.INTEGER), _attr("title")],
                    identifier_attributes=["club_id"],
                ),
                Entity(
                    name="Rider",
                    attributes=[_attr("rider_id", DataType.INTEGER), _attr("email")],
                    identifier_attributes=["rider_id"],
                ),
            ],
            relationships=[
                Relationship(
                    name="ClubMembership",
                    participants=[
                        Participant(entity="Club", cardinality_min=0),
                        Participant(entity="Rider", cardinality_min=0),
                    ],
                    degree="binary",
                    kind="M:N",
                    attributes=[
                        _attr("date_joined", DataType.DATE, nullable=False),
                        _attr("date_left", DataType.DATE, nullable=True),
                    ],
                )
            ],
            functional_dependencies=[],
        )

    def test_nullable_relationship_attribute_stays_nullable(self):
        schema = map_conceptual_to_relational(self._membership_model())
        junction = next(t for t in schema.tables if "MEMBERSHIP" in t.name)
        date_left = next(c for c in junction.columns if c.name == "date_left")
        assert date_left.is_nullable is True

    def test_non_nullable_relationship_attribute_stays_non_nullable(self):
        schema = map_conceptual_to_relational(self._membership_model())
        junction = next(t for t in schema.tables if "MEMBERSHIP" in t.name)
        date_joined = next(c for c in junction.columns if c.name == "date_joined")
        assert date_joined.is_nullable is False

    def test_the_junctions_key_columns_are_still_not_null(self):
        """The participant FKs form the PK, so they must stay NOT NULL -- the
        fix must not blanket-nullify the junction."""
        schema = map_conceptual_to_relational(self._membership_model())
        junction = next(t for t in schema.tables if "MEMBERSHIP" in t.name)
        for col in junction.columns:
            if col.name in junction.pk_set:
                assert col.is_nullable is False
