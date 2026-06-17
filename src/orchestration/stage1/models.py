from pydantic import BaseModel, Field
from typing import List
from src.pipeline.stage1.models.atomic_fact import AtomicFact
from src.pipeline.stage1.models.context_audit import ContextAuditAttempt
from src.pipeline.stage1.models.rephrased_nl import EnrichedNL
from src.pipeline.stage1.middleware.external_context_filter import ExternalFactFilterResult

class Output(BaseModel):
    final_facts: List[AtomicFact] = Field(description="The finalized list of atomic facts.")
    domain: str = Field(description="The identified industry or technical sector.")
    analytical_goal: str = Field(description="The primary analytical purpose.")
    iterations: List[EnrichedNL] = Field(description="The full history of enrichment attempts.")
    original_nl: str = Field(description="The original natural language description.")
    enrichment_filter_report: ExternalFactFilterResult = Field(default_factory=ExternalFactFilterResult)
    context_audit_trail: List[ContextAuditAttempt] = Field(default_factory=list)
    token_usage: int = 0

    def __str__(self) -> str:
        return f"Stage 1 Output: {len(self.final_facts)} facts, {self.token_usage} tokens."
