"""Duplicate names in a conceptual model silently destroy one of the duplicates.

Measured before this check existed: two entities named Student in one shard gave
get_errors() == [], the merger refused to merge them (it blocks same-shard pairs
by construction), the mapper reported "Duplicate table name in schema: STUDENT",
and the final schema validated CLEAN having lost one entity's attributes and the
facts only it carried. Rejecting at the model is the right layer -- the extractor
has the context to decide whether the two are really one concept.
"""

from __future__ import annotations

from src.pipeline.stage2.mapper.conceptual_model import (
    CMAttribute,
    ConceptualModel,
    Entity,
)
from src.util.schema_model.data_types import DataType


def _entity(name: str, attrs: list[str], fact_id: int) -> Entity:
    return Entity(
        name=name,
        attributes=[
            CMAttribute(name=a, type=DataType.VARCHAR, source_fact_ids=[fact_id])
            for a in attrs
        ],
        source_fact_ids=[fact_id],
    )


def _model(*entities: Entity) -> ConceptualModel:
    return ConceptualModel(
        entities=list(entities), relationships=[], functional_dependencies=[]
    )


def test_duplicate_entity_names_are_rejected() -> None:
    errors = _model(
        _entity("Student", ["full_name"], 1),
        _entity("Student", ["enrolled_on"], 2),
    ).get_errors()
    assert any("used by 2 entities" in e for e in errors), errors


def test_duplicate_entity_names_are_compared_case_insensitively() -> None:
    """Table names are upper-cased downstream, so two spellings collide too."""
    errors = _model(
        _entity("Student", ["full_name"], 1),
        _entity("student", ["enrolled_on"], 2),
    ).get_errors()
    assert any("used by 2 entities" in e for e in errors), errors


def test_duplicate_attributes_within_an_entity_are_rejected() -> None:
    errors = _model(_entity("Student", ["email", "EMAIL"], 1)).get_errors()
    assert any("declares the attribute" in e for e in errors), errors


def test_attributes_may_repeat_across_different_entities() -> None:
    """Only a collision WITHIN one entity collapses a column."""
    errors = _model(
        _entity("Student", ["name"], 1),
        _entity("Course", ["name"], 2),
    ).get_errors()
    assert errors == []


def test_a_clean_model_still_reports_no_errors() -> None:
    errors = _model(
        _entity("Student", ["full_name", "enrolled_on"], 1),
        _entity("Course", ["title"], 2),
    ).get_errors()
    assert errors == []
