from pydantic import BaseModel, Field
from typing import List, Optional

class RawFact(BaseModel):
    id: int = Field(description="Unique identifier for the fact.")
    fact: str = Field(description="A rich, standalone declarative sentence expressing exactly one fact.")
    origin: str = Field(default="", description="The exact verbatim substring from the source text. Empty for generated facts.")
    referenced_fact_ids: List[int] = Field(default_factory=list, description="IDs of facts this fact references (for external/context facts).")
    is_external: bool = Field(default=False, description="True if this is a generated external fact (definition, context).")

    def __str__(self) -> str:
        origin_str = f' | Origin: "{self.origin}"' if self.origin else ""
        return f"{self.id}. {self.fact}{origin_str}"

    def __repr__(self) -> str:
        return f"RawFact(id={self.id}, external={self.is_external})"
