from pydantic import BaseModel, Field
from typing import List
from src.pipeline.stage1.models.rephrased_nl import AtomicFact, EnrichedNL

class Output(BaseModel):
    final_facts: List[AtomicFact] = Field(description="The finalized list of atomic facts.")
    domain: str = Field(description="The identified industry or technical sector.")
    analytical_goal: str = Field(description="The primary analytical purpose.")
    iterations: List[EnrichedNL] = Field(description="The full history of enrichment attempts.")
