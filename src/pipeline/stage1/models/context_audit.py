from enum import Enum
from typing import List

from pydantic import BaseModel, Field
from src.util.orchestration.loop_types import LoopOutputModel


class ContextRejectionCode(str, Enum):
    GENERIC_DATABASE_ADVICE = "GENERIC_DATABASE_ADVICE"
    RESTATES_INPUT = "RESTATES_INPUT"
    UNRELATED_TO_SOURCE = "UNRELATED_TO_SOURCE"
    TOO_SPECULATIVE = "TOO_SPECULATIVE"
    LOW_VALUE = "LOW_VALUE"
    INVALID_REFERENCE = "INVALID_REFERENCE"


class ContextRejectedFact(BaseModel):
    fact_id: int = Field(description="ID of the rejected proposed external fact.")
    reason_code: ContextRejectionCode = Field(
        description="Why the fact should be rejected."
    )
    explanation: str = Field(description="Concise audit explanation.")


class ContextAuditReport(LoopOutputModel):
    is_acceptable: bool = Field(
        description="True if the proposed context can be used without retry."
    )
    accepted_fact_ids: List[int] = Field(
        default_factory=list,
        description="IDs of proposed external facts accepted by the auditor.",
    )
    rejected_facts: List[ContextRejectedFact] = Field(
        default_factory=list, description="Rejected proposed facts with reasons."
    )
    missing_recommended_context: List[str] = Field(
        default_factory=list, description="Optional high-value context still missing."
    )
    retry_instructions: str = Field(
        default="",
        description="Instructions for the context enricher if another attempt is needed.",
    )

    def get_errors(self) -> list[str]:
        if self.is_acceptable:
            return []
        return (
            [self.retry_instructions]
            if self.retry_instructions
            else ["Context enrichment retry required."]
        )


class ContextAuditAttempt(BaseModel):
    attempt: int = Field(description="1-based audit attempt number.")
    proposed_fact_count: int = Field(
        description="Number of external facts proposed by the enricher."
    )
    report: ContextAuditReport = Field(description="Audit report for this attempt.")
