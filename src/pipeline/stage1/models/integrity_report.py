from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional
from src.util.orchestration.loop_types import LoopOutputModel


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Issue(BaseModel):
    fact_id: Optional[int] = Field(
        default=None, description="The ID of the fact related to this issue."
    )
    description: str = Field(
        description="A clear explanation of why this issue exists."
    )
    severity: Severity = Field(
        default=Severity.MEDIUM, description="The risk level of this issue."
    )


class IntegrityReport(LoopOutputModel):
    is_safe: bool = Field(
        description="True if no actual information loss or hallucination found."
    )
    missing_information: List[Issue] = Field(
        default_factory=list, description="Missing details from the original."
    )
    introduced_information: List[Issue] = Field(
        default_factory=list, description="Undesired inferences or invented details."
    )
    changed_constraints: List[Issue] = Field(
        default_factory=list, description="Statistical or numeric deviations."
    )
    unresolved_ambiguities: List[Issue] = Field(
        default_factory=list,
        description="Ambiguities in the extracted facts that could not be resolved.",
    )
    search_suggestions: List[str] = Field(
        default_factory=list,
        description="Suggested web searches to gather missing domain context or resolve ambiguities.",
    )

    def get_errors(self) -> list[str]:
        all_issues = (
            self.missing_information
            + self.introduced_information
            + self.changed_constraints
            + self.unresolved_ambiguities
        )
        errors = []
        for issue in all_issues:
            if issue.severity in (Severity.HIGH, Severity.CRITICAL):
                f_id = f" (Fact {issue.fact_id})" if issue.fact_id else ""
                errors.append(f"[{issue.severity.upper()}] {issue.description}{f_id}")
        return errors

    def __str__(self) -> str:
        lines = []
        status = "SAFE" if self.is_safe else "ISSUES DETECTED"
        lines.append(f"Integrity Status: {status}")

        def format_issues(title, issues):
            if not issues:
                return
            lines.append(f"\n{title}:")
            for iss in issues:
                f_id = f" (Fact {iss.fact_id})" if iss.fact_id else ""
                lines.append(f"  - [{iss.severity.upper()}] {iss.description}{f_id}")

        format_issues("Missing Information", self.missing_information)
        format_issues("Introduced Information", self.introduced_information)
        format_issues("Changed Constraints", self.changed_constraints)
        format_issues("Unresolved Ambiguities", self.unresolved_ambiguities)
        if self.search_suggestions:
            lines.append("\nSearch Suggestions:")
            for s in self.search_suggestions:
                lines.append(f"  - {s}")
        return "\n".join(lines)
