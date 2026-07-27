from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


# Mirrors Relationship.kind's Literal in mapper/conceptual_model.py. Kept as a
# named constant rather than a repeated inline tuple so the two cannot drift; a
# test asserts they agree.
LEGAL_CARDINALITIES = frozenset({"1:1", "1:N", "M:N"})


class ActionType(str, Enum):
    MERGE_ENTITIES = "MERGE_ENTITIES"
    RENAME_ATTRIBUTE = "RENAME_ATTRIBUTE"
    RESOLVE_CARDINALITY = "RESOLVE_CARDINALITY"
    RESOLVE_CROSS_CATEGORY = "RESOLVE_CROSS_CATEGORY"
    RESOLVE_IDENTIFIER = "RESOLVE_IDENTIFIER"
    NO_ACTION = "NO_ACTION"


class ConflictFlag(BaseModel):
    flag_type: str = Field(
        description="Type of conflict flag (e.g., VETOED_MERGE, FORCED_MERGE, IDENTIFIER_DISAGREEMENT, CARDINALITY_CONTRADICTION, POSSIBLE_ATTR_SYNONYM, CROSS_CATEGORY_COLLISION)"
    )
    entities: List[str] = Field(
        default_factory=list, description="Entity names involved in this flag"
    )
    relationship: Optional[str] = Field(
        default=None, description="Relationship name if applicable"
    )
    posterior: Optional[float] = Field(
        default=None, description="Posterior probability from the Beta mixture"
    )
    message: str = Field(description="Human-readable explanation of the conflict")


class ResolutionAction(BaseModel):
    action_type: ActionType
    entity_a: Optional[str] = Field(
        default=None, description="First entity involved (for merge/rename)"
    )
    entity_b: Optional[str] = Field(
        default=None, description="Second entity involved (for merge or cross-category)"
    )
    attribute_old: Optional[str] = Field(
        default=None, description="Old attribute name (for rename)"
    )
    new_name: Optional[str] = Field(
        default=None, description="New unified name (for merge or rename)"
    )
    relationship_name: Optional[str] = Field(
        default=None, description="Relationship name (for cardinality)"
    )
    new_cardinality: Optional[str] = Field(
        default=None, description="New cardinality (for cardinality, e.g. '1:N')"
    )
    new_identifier_attributes: Optional[List[str]] = Field(
        default=None, description="New identifier attribute names (for RESOLVE_IDENTIFIER)"
    )
    rationale: str = Field(
        description="Brief explanation for why this action was taken."
    )

    def _validate(self) -> list[str]:
        errors = []
        if self.action_type == ActionType.MERGE_ENTITIES:
            if not self.entity_a or not self.entity_b or not self.new_name:
                errors.append(
                    "MERGE_ENTITIES requires entity_a, entity_b, and new_name"
                )
        elif self.action_type == ActionType.RENAME_ATTRIBUTE:
            if not self.entity_a or not self.attribute_old or not self.new_name:
                errors.append(
                    "RENAME_ATTRIBUTE requires entity_a, attribute_old, and new_name"
                )
        elif self.action_type == ActionType.RESOLVE_CARDINALITY:
            if not self.relationship_name or not self.new_cardinality:
                errors.append(
                    "RESOLVE_CARDINALITY requires relationship_name and new_cardinality"
                )
            elif self.new_cardinality not in LEGAL_CARDINALITIES:
                # Relationship.kind is a closed Literal, but this field is a
                # free-form Optional[str] and is assigned straight onto it with
                # no validate_assignment. An unchecked value like "one-to-many"
                # is stored happily and then falls through the relational
                # mapper's if/elif chain, silently dropping the relationship.
                errors.append(
                    f"RESOLVE_CARDINALITY new_cardinality must be one of "
                    f"{sorted(LEGAL_CARDINALITIES)}, got {self.new_cardinality!r}"
                )
        elif self.action_type == ActionType.RESOLVE_CROSS_CATEGORY:
            if not self.entity_a or not self.relationship_name:
                errors.append(
                    "RESOLVE_CROSS_CATEGORY requires entity_a and relationship_name"
                )
        elif self.action_type == ActionType.RESOLVE_IDENTIFIER:
            if not self.entity_a or not self.new_identifier_attributes:
                errors.append(
                    "RESOLVE_IDENTIFIER requires entity_a and new_identifier_attributes"
                )
        return errors


class AdjudicatorResponse(BaseModel):
    resolutions: List[ResolutionAction]

    def _validate(self) -> list[str]:
        errors = []
        for r in self.resolutions:
            errors.extend(r._validate())
        return errors
