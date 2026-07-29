from typing import List, Literal, Optional
from pydantic import BaseModel
from src.util.orchestration.loop_types import LoopOutputModel

from src.util.schema_model.data_types import DataType


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

        # Two entities sharing a name silently destroy one of them. Nothing else
        # in the pipeline catches it: the merger refuses to merge entities from
        # the SAME shard by construction, so both survive to the mapper, which
        # maps them onto one table name; the duplicate is discarded and the
        # final schema validates clean, having lost an entity's attributes and
        # every fact only it carried. Compared case-insensitively because table
        # names are upper-cased downstream, so two spellings collide as well.
        seen_entities: dict[str, int] = {}
        for e in self.entities:
            key = e.name.lower()
            seen_entities[key] = seen_entities.get(key, 0) + 1
        for name, count in seen_entities.items():
            if count > 1:
                errors.append(
                    f"Entity name '{name}' is used by {count} entities. Each concept "
                    "must appear exactly once: merge them into a single entity holding "
                    "the union of their attributes and source_fact_ids, or rename them "
                    "to the distinct concepts the facts actually describe."
                )

        # Same failure one level down: duplicate attributes collapse into one
        # column, so the survivor's type and nullability silently win.
        for e in self.entities:
            seen_attrs: dict[str, int] = {}
            for a in e.attributes:
                seen_attrs[a.name.lower()] = seen_attrs.get(a.name.lower(), 0) + 1
            for attr_name, count in seen_attrs.items():
                if count > 1:
                    errors.append(
                        f"Entity '{e.name}' declares the attribute '{attr_name}' "
                        f"{count} times. Keep one, carrying the union of its "
                        "source_fact_ids."
                    )

        for e in self.entities:
            if e.is_weak and e.owner and e.owner.lower() not in entity_names:
                errors.append(f"Weak entity '{e.name}' has unknown owner '{e.owner}'.")

        for r in self.relationships:
            for p in r.participants:
                if p.entity.lower() not in entity_names:
                    errors.append(
                        f"Relationship '{r.name}' references unknown entity '{p.entity}'."
                    )

        errors.extend(self._cardinality_errors())

        return errors

    def _cardinality_errors(self) -> list[str]:
        """Enforce the cardinality contract the mapper silently depends on.

        Until now the invariant lived only in two prompts (er_extractor and
        er_auditor both ask for it) and in a comment on `Participant`. Nothing
        checked it, yet `relational_mapper.py`'s 1:N branch decides FK DIRECTION
        from `cardinality_max` alone -- so a violation does not fail, it produces
        a structurally valid schema with a reversed foreign key, which no
        downstream validator can detect because a reversed edge is still a
        legal edge.

        Two distinct violations, both observed or reachable:

        * OUT-OF-DOMAIN values. `cardinality_max` is structural -- 1 for the
          "one" side, null for the "many" side -- but a population count can
          leak in instead (a saved run carried `min=50, max=5000`, meaning "each
          warehouse handles 50-5000 orders"). That is the LOOK-HERE reading of
          participation, whereas the mapper implements LOOK-ACROSS, so its
          presence is evidence the sides may also be inverted. The pair is
          symmetric under inversion and therefore undetectable on its own; an
          out-of-domain value is the one syntactic tell there is.

        * WRONG SHAPE for the declared `kind`. A 1:N with both participants at
          null is the live hazard: the mapper's `p1.cardinality_max != 1` test
          then makes p1 the child, so the foreign key lands wherever participant
          ORDER happens to put it. Zero of 464 saved binary relationships hit
          this, so it is a guard against an unhit path rather than a fix for a
          current failure -- but the cost of being wrong is a silently reversed
          key, which is the most damaging schema error there is.
        """
        errors: list[str] = []

        for r in self.relationships:
            for p in r.participants:
                if p.cardinality_max not in (1, None):
                    errors.append(
                        f"Relationship '{r.name}' participant '{p.entity}' has "
                        f"cardinality_max={p.cardinality_max}. This field is "
                        "STRUCTURAL, not a population estimate: use 1 on the side of "
                        "which at most one instance is associated with each instance "
                        "of the other participant, and null on the side whose "
                        "instances repeat. Expected counts belong in the facts, not "
                        "here."
                    )
                if p.cardinality_min not in (0, 1, None):
                    errors.append(
                        f"Relationship '{r.name}' participant '{p.entity}' has "
                        f"cardinality_min={p.cardinality_min}. Use 0 for optional "
                        "participation or 1 for mandatory participation -- it records "
                        "WHETHER an instance must participate, not how many times."
                    )

            if r.degree != "binary" or len(r.participants) != 2:
                continue

            maxes = [p.cardinality_max for p in r.participants]
            names = [p.entity for p in r.participants]
            if r.kind == "1:N" and maxes.count(1) != 1:
                if maxes.count(1) == 2:
                    detail = (
                        "both sides are 1, which describes a 1:1 relationship. Set the "
                        "repeating side to null, or change kind to '1:1'"
                    )
                else:
                    detail = (
                        "neither side is 1, so there is no 'one' side to point at and "
                        "the foreign key would land on whichever participant happens "
                        "to be listed first. Set cardinality_max=1 on the parent, or "
                        "change kind to 'M:N'"
                    )
                errors.append(
                    f"Relationship '{r.name}' is declared 1:N between "
                    f"'{names[0]}' and '{names[1]}' but {detail}."
                )
            elif r.kind == "1:1" and maxes != [1, 1]:
                errors.append(
                    f"Relationship '{r.name}' is declared 1:1 but its participants "
                    f"carry cardinality_max {maxes}. Both sides of a 1:1 must be 1."
                )
            elif r.kind == "M:N" and 1 in maxes:
                errors.append(
                    f"Relationship '{r.name}' is declared M:N but participant "
                    f"'{names[maxes.index(1)]}' carries cardinality_max=1. Both sides "
                    "of an M:N repeat, so both must be null; if one side really is "
                    "single-valued, the relationship is 1:N."
                )

        return errors
