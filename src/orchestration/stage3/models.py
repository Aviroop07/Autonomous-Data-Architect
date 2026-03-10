from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage3.models.patch import CritiqueReport, PatchValidationError

class PatchRepairStep(BaseModel):
    original_errors: List[PatchValidationError]
    repaired_patches: List[Dict[str, Any]] # model_dump() output

class ShardStep(BaseModel):
    chunk_index: int
    pass_a_schema: Schema
    pass_a_report: CritiqueReport
    pass_a_image_uri: Optional[str] = None
    pass_b_schema: Schema
    pass_b_report: CritiqueReport
    pass_b_image_uri: Optional[str] = None

class Output(BaseModel):
    global_schema: Schema = Field(description="The finalized global schema.")
    style_guide: str = Field(description="The domain-specific style guide generated.")
    shard_steps: List[ShardStep] = Field(default_factory=list, description="Detailed tracking of Pass A and Pass B for each shard.")
    merge_image_uri: Optional[str] = None
    final_image_uri: Optional[str] = None
    reports: List[CritiqueReport] = Field(default_factory=list, description="Legacy field for all reports.")
    cycles: List[List[str]] = Field(default_factory=list, description="Relational cycles detected (if any).")
    repair_history: List[PatchRepairStep] = Field(default_factory=list, description="History of patch repairs performed.")
