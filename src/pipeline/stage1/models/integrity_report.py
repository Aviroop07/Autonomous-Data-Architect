import logging
from enum import Enum
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional
from src.util.orchestration.loop_types import LoopOutputModel

logger = logging.getLogger(__name__)


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

    @model_validator(mode="after")
    def _is_safe_must_agree_with_the_issues(self) -> "IntegrityReport":
        """Recompute is_safe from the issue lists instead of trusting the field.

        is_safe drives the retry edge out of the verifier, and get_errors()
        surfaces only HIGH and CRITICAL issues as feedback text. Those two facts
        have to agree, or the loop misbehaves in one of two ways:

          is_safe=False with no HIGH/CRITICAL issue -> the extractor is re-run
            with an EMPTY error list, told to fix something it cannot see, and
            the loop can burn its whole budget that way.
          is_safe=True alongside a HIGH issue -> a real defect is reported and
            then routed straight past.

        The prompt states the invariant, but a prompt cannot enforce one. This
        makes it structural: the field becomes derived rather than asserted, so
        the loop's routing and the feedback it carries can never disagree.
        """
        should_be_safe = not self._blocking_issues()
        if self.is_safe != should_be_safe:
            logger.warning(
                "[Verifier] is_safe=%s contradicts the reported issues "
                "(%d blocking); correcting to %s.",
                self.is_safe,
                len(self._blocking_issues()),
                should_be_safe,
            )
            self.is_safe = should_be_safe
        return self

    def _blocking_issues(self) -> list[Issue]:
        """Issues severe enough to justify spending another extraction round.
        This is the single definition of "blocking" -- get_errors() formats
        exactly these, and is_safe is exactly their absence."""
        return [
            issue
            for issue in (
                self.missing_information
                + self.introduced_information
                + self.changed_constraints
                + self.unresolved_ambiguities
            )
            if issue.severity in (Severity.HIGH, Severity.CRITICAL)
        ]

    def get_errors(self) -> list[str]:
        errors = []
        for issue in self._blocking_issues():
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
