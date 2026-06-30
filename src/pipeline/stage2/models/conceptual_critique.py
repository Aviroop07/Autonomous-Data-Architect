from typing import List
from pydantic import BaseModel
from src.util.orchestration.loop_types import LoopOutputModel

class SuggestedFix(BaseModel):
    description: str
    rationale: str

class ConceptualCritiqueReport(LoopOutputModel):
    is_valid: bool
    fixes: List[SuggestedFix] = []
    
    def get_errors(self) -> list[str]:
        if not self.is_valid and not self.fixes:
            return ["Critique report is invalid but provides no suggested fixes."]
        if self.is_valid and self.fixes:
            return ["Critique report is marked valid but contains suggested fixes."]
        return []
