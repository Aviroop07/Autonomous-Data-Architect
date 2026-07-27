"""Malformed adjudicator resolutions are skipped, not applied.

ResolutionAction._validate() already existed but was never called from
orchestration. A MERGE_ENTITIES action omitting the typing-optional but
in-practice-required `new_name` therefore assigned None straight onto
entity.name, and surfaced much later as an AttributeError inside the relational
mapper's to_snake_case -- with nothing left pointing at the adjudicator.

Omitting an optional field is routine model behaviour, and this agent is
single-shot with no retry loop, so it has to degrade rather than fail.
"""

from __future__ import annotations

from src.orchestration.stage2.entry import apply_adjudicator_patches
from src.pipeline.stage2.mapper.conceptual_model import (
    CMAttribute,
    ConceptualModel,
    Entity,
    Participant,
    Relationship,
)
from src.pipeline.stage2.models.conflicts import (
    LEGAL_CARDINALITIES,
    ActionType,
    ResolutionAction,
)
from src.pipeline.stage2.models.data_types import DataType


def _model() -> ConceptualModel:
    return ConceptualModel(
        entities=[
            Entity(
                name="ALPHA",
                attributes=[CMAttribute(name="alpha_id", type=DataType.INTEGER)],
                identifier_attributes=["alpha_id"],
            ),
            Entity(
                name="BETA",
                attributes=[CMAttribute(name="beta_id", type=DataType.INTEGER)],
                identifier_attributes=["beta_id"],
            ),
        ],
        relationships=[
            Relationship(
                name="LINK",
                kind="1:N",
                degree="binary",
                participants=[
                    Participant(entity="ALPHA"),
                    Participant(entity="BETA"),
                ],
            )
        ],
    )


class TestMalformedPatchesAreSkipped:
    def test_merge_without_new_name_does_not_null_the_entity_name(self):
        patch = ResolutionAction(
            action_type=ActionType.MERGE_ENTITIES,
            entity_a="ALPHA",
            entity_b="BETA",
            new_name=None,
            rationale="omitted the new name",
        )
        result = apply_adjudicator_patches(_model(), [patch])
        assert {e.name for e in result.entities} == {"ALPHA", "BETA"}
        assert all(e.name is not None for e in result.entities)

    def test_rename_without_attribute_old_is_skipped(self):
        patch = ResolutionAction(
            action_type=ActionType.RENAME_ATTRIBUTE,
            entity_a="ALPHA",
            attribute_old=None,
            new_name="renamed",
            rationale="omitted the old attribute",
        )
        result = apply_adjudicator_patches(_model(), [patch])
        alpha = next(e for e in result.entities if e.name == "ALPHA")
        assert [a.name for a in alpha.attributes] == ["alpha_id"]

    def test_a_valid_patch_alongside_a_malformed_one_still_applies(self):
        """One bad action must not discard the adjudicator's good work."""
        bad = ResolutionAction(
            action_type=ActionType.MERGE_ENTITIES,
            entity_a="ALPHA",
            entity_b="BETA",
            rationale="bad",
        )
        good = ResolutionAction(
            action_type=ActionType.RESOLVE_IDENTIFIER,
            entity_a="ALPHA",
            new_identifier_attributes=["alpha_id"],
            rationale="good",
        )
        result = apply_adjudicator_patches(_model(), [bad, good])
        alpha = next(e for e in result.entities if e.name == "ALPHA")
        assert alpha.identifier_attributes == ["alpha_id"]
        assert len(result.entities) == 2


class TestCardinalityIsCheckedAgainstTheLiteral:
    def test_legal_cardinalities_match_the_relationship_literal(self):
        """LEGAL_CARDINALITIES mirrors Relationship.kind; assert they agree so
        the two cannot drift."""
        import typing

        from src.pipeline.stage2.mapper.conceptual_model import Relationship as R

        literal = typing.get_args(R.model_fields["kind"].annotation)
        assert set(literal) == set(LEGAL_CARDINALITIES)

    def test_free_form_cardinality_is_rejected(self):
        action = ResolutionAction(
            action_type=ActionType.RESOLVE_CARDINALITY,
            relationship_name="LINK",
            new_cardinality="one-to-many",
            rationale="prose instead of the literal",
        )
        assert action._validate()

    def test_free_form_cardinality_never_reaches_the_model(self):
        """Relationship.kind has no validate_assignment, so an unchecked value
        is stored happily and then falls through the mapper's if/elif chain,
        silently dropping the relationship."""
        patch = ResolutionAction(
            action_type=ActionType.RESOLVE_CARDINALITY,
            relationship_name="LINK",
            new_cardinality="one-to-many",
            rationale="prose",
        )
        result = apply_adjudicator_patches(_model(), [patch])
        assert result.relationships[0].kind in LEGAL_CARDINALITIES

    def test_a_legal_cardinality_is_applied(self):
        patch = ResolutionAction(
            action_type=ActionType.RESOLVE_CARDINALITY,
            relationship_name="LINK",
            new_cardinality="M:N",
            rationale="genuinely many-to-many",
        )
        result = apply_adjudicator_patches(_model(), [patch])
        assert result.relationships[0].kind == "M:N"
