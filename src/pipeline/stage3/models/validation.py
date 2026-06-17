from typing import List, Literal

from pydantic import BaseModel, Field

from src.pipeline.stage3.models.sql_models import LLMResponse


IssueSeverity = Literal["low", "medium", "high", "critical"]


class Stage3Issue(BaseModel):
    code: str = Field(description="Stable machine-readable issue code.")
    severity: IssueSeverity = Field(description="Issue severity.")
    target: str = Field(description="Path to the affected Stage 3 output element.")
    message: str = Field(description="Human-readable issue description.")
    suggested_action: str = Field(default="", description="Recommended correction.")
    fact_references: List[int] = Field(default_factory=list, description="Related source fact IDs.")


class MathematicsValidationReport(BaseModel):
    is_valid: bool = Field(description="True if no mathematical or feasibility issues remain.")
    issues: List[Stage3Issue] = Field(default_factory=list, description="Detected math/feasibility issues.")
    reasoning: str = Field(default="", description="Brief rationale for the validation decision.")


class Stage3PatchPlan(BaseModel):
    patched_response: LLMResponse = Field(description="Corrected Stage 3 SQL-scoped metadata output.")
    addressed_issue_codes: List[str] = Field(default_factory=list, description="Issue codes addressed by this patch.")
    rationale: str = Field(description="Explanation of the patch choices.")
