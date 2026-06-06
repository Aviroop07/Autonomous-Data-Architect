from pydantic import BaseModel
from typing import List, Optional
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

    def __str__(self) -> str:
        msg = f"{self.status.upper()}: {self.error_message}"
        if self.description:
            msg += f" (Note: {self.description})"
        return msg

class FixHistoryStep(BaseModel):
    attempt: int
    errors: List[str]
    corrections: List[Correction]
    fixed_schema: str
    schema_state: Optional[Schema] = None
