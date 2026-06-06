from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Issue(BaseModel):
    fact_id: Optional[int] = Field(default=None, description="The ID of the fact related to this issue.")
    description: str = Field(description="A clear explanation of why this issue exists.")
    severity: Severity = Field(description="The risk level of this issue.")

class IntegrityReport(BaseModel):
    is_safe: bool = Field(description="True if no actual information loss or hallucination found.")
    missing_information: List[Issue] = Field(description="Missing details from the original.")
    introduced_information: List[Issue] = Field(description="Undesired inferences or invented details.")
    changed_constraints: List[Issue] = Field(description="Statistical or numeric deviations.")

    def __str__(self) -> str:
        lines = []
        status = "SAFE" if self.is_safe else "ISSUES DETECTED"
        lines.append(f"Integrity Status: {status}")

        def format_issues(title, issues):
            if not issues: return
            lines.append(f"\n{title}:")
            for iss in issues:
                f_id = f" (Fact {iss.fact_id})" if iss.fact_id else ""
                lines.append(f"  - [{iss.severity.upper()}] {iss.description}{f_id}")

        format_issues("Missing Information", self.missing_information)
        format_issues("Introduced Information", self.introduced_information)
        format_issues("Changed Constraints", self.changed_constraints)
        return "\n".join(lines)
