from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from enum import Enum
from src.pipeline.stage2.models.schema import Schema

class CorrectionStatus(str, Enum):
    FIXED = "fixed"
    NOT_FIXED = "not_fixed"
    DEFERRED = "deferred"

class Correction(BaseModel):
    error_message: str
    status: CorrectionStatus
    description: Optional[str]

class SchemaResolve(BaseModel):
    corrections: List[Correction]
    corrected_schema: Schema

    def get_errors_by_status(self, status: CorrectionStatus) -> List[str]:
        return [c.error_message for c in self.corrections if c.status == status]

    def all_deferred(self) -> bool:
        """Returns True if all corrections are deferred."""
        if not self.corrections:
            return False
        return all(c.status == CorrectionStatus.DEFERRED for c in self.corrections)

    def has_not_fixed(self) -> bool:
        """Returns True if there are any errors that the agent tried to fix but failed."""
        return any(c.status == CorrectionStatus.NOT_FIXED for c in self.corrections)

class FixHistoryStep(BaseModel):
    attempt: int
    errors: List[str]
    corrections: List[Correction]
    fixed_schema: str
