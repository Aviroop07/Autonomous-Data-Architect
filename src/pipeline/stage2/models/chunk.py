from pydantic import BaseModel, Field
from typing import List

class ChunkedPlan(BaseModel):
    chunks: List[str] = Field(description="A list of smaller natural language descriptions (each a string) that collectively cover the original structural description.")
