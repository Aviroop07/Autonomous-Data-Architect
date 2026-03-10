from pydantic import BaseModel, Field
from typing import List
from src.pipeline.stage1.models.rephrased_nl import AtomicFact

class ChunkedPlan(BaseModel):
    core_modeling_facts: List[AtomicFact] = Field(description="The filtered list of all atomic facts relevant for schema modeling.")
    chunks: List[List[AtomicFact]] = Field(description="A list of chunks, where each chunk is a curated list of AtomicFact objects.")
