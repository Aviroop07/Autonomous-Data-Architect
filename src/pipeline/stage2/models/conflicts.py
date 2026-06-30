from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ConflictType(str, Enum):
    UNCERTAIN_MERGE = "UNCERTAIN_MERGE"
    ORPHAN_COLLISION = "ORPHAN_COLLISION"
    CROSS_CATEGORY_COLLISION = "CROSS_CATEGORY_COLLISION"
    STRUCTURAL_CONTRADICTION = "STRUCTURAL_CONTRADICTION"


class ResolutionAction(str, Enum):
    MERGE = "MERGE"
    KEEP_SEPARATE = "KEEP_SEPARATE"
    CONVERT_TO_ENTITY = "CONVERT_TO_ENTITY"
    CONVERT_TO_RELATIONSHIP = "CONVERT_TO_RELATIONSHIP"


class RelationshipKind(str, Enum):
    ONE_TO_ONE = "1:1"
    ONE_TO_MANY = "1:N"
    MANY_TO_ONE = "M:1"
    MANY_TO_MANY = "M:N"


class ConflictResolution(BaseModel):
    conflict_id: str = Field(description="The unique identifier for the conflict being resolved.")
    action: ResolutionAction = Field(description="The deterministic action to take to resolve the conflict.")
    merged_name: Optional[str] = Field(
        default=None, 
        description="Required if action is MERGE, CONVERT_TO_ENTITY, or CONVERT_TO_RELATIONSHIP. The unified name to use."
    )
    resolved_kind: Optional[RelationshipKind] = Field(
        default=None, 
        description="Required if resolving a STRUCTURAL_CONTRADICTION. The correct cardinality."
    )
    rationale: str = Field(description="A brief explanation for the decision.")

    def _validate(self) -> list[str]:
        errors = []
        if self.action in (ResolutionAction.MERGE, ResolutionAction.CONVERT_TO_ENTITY, ResolutionAction.CONVERT_TO_RELATIONSHIP):
            if not self.merged_name:
                errors.append(f"merged_name is required when action is {self.action.value}")
        return errors


class ConflictResolutionPlan(BaseModel):
    resolutions: list[ConflictResolution]

    def _validate(self) -> list[str]:
        errors = []
        for res in self.resolutions:
            errors.extend(res._validate())
        return errors
