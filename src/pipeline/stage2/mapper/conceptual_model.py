from typing import List, Literal, Optional
from pydantic import BaseModel
from src.util.orchestration.loop_types import LoopOutputModel

from src.pipeline.stage2.models.data_types import DataType


class CMAttribute(BaseModel):
    name: str
    type: DataType
    is_multivalued: bool = False
    is_derived: bool = False
    # Whether this attribute may be absent/unknown for an entity instance
    # (e.g. "middle name is optional"). Propagated verbatim into the mapped
    # Column.is_nullable for ordinary attributes -- see relational_mapper.py's
    # entity-to-table pass. Separate from an FK column's own nullability,
    # which instead comes from the owning Relationship's Participant
    # cardinality (this field only covers an entity's own attributes, not
    # relationship participation).
    is_nullable: bool = False
    source_fact_ids: List[int] = []


class Entity(BaseModel):
    name: str
    attributes: List[CMAttribute] = []
    identifier_attributes: List[
        str
    ] = []  # ordered natural-key member names (may be empty)
    is_weak: bool = False
    owner: Optional[str] = None  # identifying owner entity (weak entities)
    source_fact_ids: List[int] = []


class Participant(BaseModel):
    entity: str
    role: Optional[str] = None  # e.g. "captain", "first_officer"
    # cardinality_min == 0 means an instance of THIS entity need not
    # participate in the relationship at all (e.g. a PATIENT need not have
    # INSURANCE); cardinality_min == 1 means every instance must. This is the
    # source of truth the relational mapper reads to decide whether the FK
    # column it synthesizes on the child/FK-holding side is nullable -- see
    # relational_mapper.py's 1:N and 1:1 branches. Not the same axis as
    # CMAttribute.is_nullable, which covers an entity's own plain attributes.
    cardinality_min: Optional[int] = None  # 0 / 1
    cardinality_max: Optional[int] = None  # 1 / None (= many)


class Relationship(BaseModel):
    name: str
    participants: List[Participant]
    degree: Literal["binary", "n-ary"]
    kind: Literal["1:1", "1:N", "M:N"]  # binary; n-ary always -> junction
    attributes: List[CMAttribute] = []
    source_fact_ids: List[int] = []


class FunctionalDependency(BaseModel):
    determinant: List[str]  # qualified "ENTITY.attr"
    dependent: List[str]


class ConceptualModel(LoopOutputModel):  # participates in the self-correction loop
    entities: List[Entity]
    relationships: List[Relationship] = []
    functional_dependencies: List[FunctionalDependency] = []

    def get_errors(self) -> list[str]:
        errors = []
        entity_names = {e.name.lower() for e in self.entities}

        for e in self.entities:
            if e.is_weak and e.owner and e.owner.lower() not in entity_names:
                errors.append(f"Weak entity '{e.name}' has unknown owner '{e.owner}'.")

        for r in self.relationships:
            for p in r.participants:
                if p.entity.lower() not in entity_names:
                    errors.append(
                        f"Relationship '{r.name}' references unknown entity '{p.entity}'."
                    )

        return errors
