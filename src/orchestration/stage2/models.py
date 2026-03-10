from pydantic import BaseModel, Field
from typing import List
from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage2.models.chunk import ChunkedPlan

class Output(BaseModel):
    segments: List[Schema] = Field(description="The generated schema shards.")
    plan: ChunkedPlan = Field(description="The structural chunking plan used.")
