from pydantic import BaseModel, Field, field_validator
from typing import List, Any
from src.pipeline.stage1.models.rephrased_nl import AtomicFact

class ChunkedPlan(BaseModel):
    core_modeling_facts: List[AtomicFact] = Field(description="The filtered list of all atomic facts relevant for schema modeling.")
    chunks: List[List[AtomicFact]] = Field(description="A list of chunks, where each chunk is a curated list of AtomicFact objects.")

    @field_validator('chunks', mode='before')
    @classmethod
    def ensure_list_of_lists(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return v
        new_v = []
        for item in v:
            if isinstance(item, dict):
                # If the LLM returned a single fact instead of a list of facts for a chunk
                new_v.append([item])
            elif isinstance(item, list):
                new_v.append(item)
            else:
                new_v.append(item)
        return new_v
