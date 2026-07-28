"""A self-referential M:N relationship must survive mapping.

Both ends resolve to the same table, so with no participant roles both produced
the IDENTICAL foreign-key column name, the dedup check collapsed them into one
FK, and the resulting single-FK junction was then classified "hollow" and
dropped -- deleting the relationship outright with no error. Verified: a
self-referencing M:N produced zero foreign keys and no junction table.

These relationships are common (prerequisite, supersedes, manager-of, part-of),
so losing them silently is expensive.
"""

from __future__ import annotations

from src.pipeline.stage2.mapper.conceptual_model import (
    CMAttribute,
    ConceptualModel,
    Entity,
    Participant,
    Relationship,
)
from src.pipeline.stage2.mapper.relational_mapper import (
    _derive_junction_name,
    _is_new_token,
    map_conceptual_to_relational,
)
from src.util.schema_model import DataType, Table


def _entity(name: str, pk: str) -> Entity:
    return Entity(
        name=name,
        attributes=[
            CMAttribute(name=pk, type=DataType.INTEGER),
            CMAttribute(name="label", type=DataType.VARCHAR),
        ],
        identifier_attributes=[pk],
    )


def _self_ref(role_a=None, role_b=None, rel_name="PREREQ") -> ConceptualModel:
    return ConceptualModel(
        entities=[_entity("COURSE", "course_id")],
        relationships=[
            Relationship(
                name=rel_name,
                kind="M:N",
                degree="binary",
                participants=[
                    Participant(entity="COURSE", role=role_a),
                    Participant(entity="COURSE", role=role_b),
                ],
            )
        ],
    )


class TestSelfReferentialSurvives:
    def test_junction_table_is_created_without_roles(self):
        schema = map_conceptual_to_relational(_self_ref())
        assert "PREREQ" in {t.name for t in schema.tables}

    def test_both_ends_get_their_own_foreign_key(self):
        schema = map_conceptual_to_relational(_self_ref())
        fks = [f for f in (schema.relationships or []) if f.referencing_table == "PREREQ"]
        assert len(fks) == 2, f"expected two FK ends, got {fks}"
        assert len({f.referencing_column for f in fks}) == 2, "the two ends collided"
        assert all(f.referred_table == "COURSE" for f in fks)

    def test_roles_are_preferred_over_positional_names(self):
        schema = map_conceptual_to_relational(
            _self_ref(role_a="requires", role_b="required_by")
        )
        cols = {
            f.referencing_column
            for f in (schema.relationships or [])
            if f.referencing_table == "PREREQ"
        }
        assert cols == {"requires_course_id", "required_by_course_id"}

    def test_positional_fallback_is_deterministic(self):
        first = map_conceptual_to_relational(_self_ref())
        second = map_conceptual_to_relational(_self_ref())
        assert [f.referencing_column for f in (first.relationships or [])] == [
            f.referencing_column for f in (second.relationships or [])
        ]


class TestOrdinaryManyToManyUnaffected:
    def test_two_distinct_entities_still_map_normally(self):
        cm = ConceptualModel(
            entities=[_entity("COURSE", "course_id"), _entity("STUDENT", "student_id")],
            relationships=[
                Relationship(
                    name="ENROLLMENT",
                    kind="M:N",
                    degree="binary",
                    participants=[
                        Participant(entity="COURSE"),
                        Participant(entity="STUDENT"),
                    ],
                )
            ],
        )
        schema = map_conceptual_to_relational(cm)
        fks = {
            (f.referencing_column, f.referred_table)
            for f in (schema.relationships or [])
            if f.referencing_table == "ENROLLMENT"
        }
        assert fks == {("course_id", "COURSE"), ("student_id", "STUDENT")}


class TestJunctionNaming:
    def test_a_suffix_never_repeats_a_token_already_in_the_name(self):
        """A live run produced a junction named <TABLE>_<TABLE>: the composed name
        collided with the real table, and the relationship-name suffix repeated
        the same word. Such a name carries no information."""
        used = {"PRESCRIPTION"}
        name = _derive_junction_name(
            Relationship(
                name="prescription",
                kind="M:N",
                degree="binary",
                participants=[
                    Participant(entity="PRESCRIPTION"),
                    Participant(entity="PRESCRIPTION"),
                ],
            ),
            [Table(name="PRESCRIPTION", columns=[], primary_key=["id"])],
            used,
        )
        assert name != "PRESCRIPTION_PRESCRIPTION"
        assert name not in used

    def test_roles_disambiguate_a_colliding_junction_name(self):
        name = _derive_junction_name(
            Relationship(
                name="link",
                kind="M:N",
                degree="binary",
                participants=[
                    Participant(entity="NODE", role="parent"),
                    Participant(entity="NODE", role="child"),
                ],
            ),
            [Table(name="NODE", columns=[], primary_key=["id"])],
            {"NODE", "LINK"},
        )
        assert "PARENT" in name or "CHILD" in name

    def test_is_new_token(self):
        assert _is_new_token("BETA", "ALPHA")
        assert not _is_new_token("ALPHA", "ALPHA")
        assert not _is_new_token("ALPHA", "ALPHA_BETA")
        assert _is_new_token("GAMMA", "ALPHA_BETA")
