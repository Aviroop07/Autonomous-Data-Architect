from pydantic import BaseModel, Field
from typing import List, Optional
from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage2.models.chunk import ChunkedPlan
from src.util.schema_ops.schema_patch import CritiqueReport
from src.pipeline.stage2.models.corrections import FixHistoryStep

class PatchRepairStep(BaseModel):
    attempt: int
    errors: List[str]
    critique: CritiqueReport
    fixed_schema: str
    schema_state: Optional[Schema] = None

    def __str__(self) -> str:
        errs = "\n".join([f"      - {e}" for e in self.errors])
        patches = "\n".join([f"      - {p}" for p in self.critique.patches])
        return (
            f"   - **Attempt {self.attempt}**\n"
            f"     - *Errors Found*:\n{errs}\n"
            f"     - *Patches Applied*:\n{patches}"
        )

class RefinementIteration(BaseModel):
    iteration: int
    critique: CritiqueReport
    repair_history: List[PatchRepairStep] = Field(default_factory=list)
    fix_history: List[FixHistoryStep] = Field(default_factory=list)
    schema_state: Schema

    def __str__(self) -> str:
        return f"   - **Iteration {self.iteration}**: {len(self.critique.patches)} patches proposed."

class Output(BaseModel):
    segments: List[Schema] = Field(description="The initial generated schema shards.")
    plan: ChunkedPlan = Field(description="The structural chunking plan used.")
    fix_history: List[List[FixHistoryStep]] = Field(default_factory=list, description="History of corrections for each initial segment.")
    merged_schema: Optional[Schema] = Field(default=None, description="The unified schema resulting from initial merging.")
    final_global_schema: Optional[Schema] = Field(None, description="The finalized refined global schema.")
    final_fix_history: List[FixHistoryStep] = Field(default_factory=list, description="History of corrections for the final refined global schema.")
    domain_iterations: List[RefinementIteration] = Field(default_factory=list, description="History of global domain-level refinements.")
    token_usage: int = 0
    cycles: List[List[str]] = Field(default_factory=list, description="Relational cycles detected (if any).")

    def __str__(self) -> str:
        refine_status = "Refined" if self.final_global_schema else "Initial Only"
        return f"Stage 2 Output: {len(self.segments)} segments, {refine_status}, {self.token_usage} tokens."
