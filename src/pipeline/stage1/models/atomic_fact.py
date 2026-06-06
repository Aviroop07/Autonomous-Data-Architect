from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional
from .raw_fact import RawFact

class FactTag(str, Enum):
    STRUCTURAL = "STRUCTURAL"
    LOGICAL = "LOGICAL"
    STATISTICAL = "STATISTICAL"
    METADATA = "METADATA"

class AtomicFact(RawFact):
    tags: List[FactTag] = Field(default_factory=list, description="Categorical tags for the fact.")

    @staticmethod
    def from_raw(raw: RawFact, tags: List[FactTag] = None) -> "AtomicFact":
        return AtomicFact(
            id=raw.id,
            fact=raw.fact,
            origin=raw.origin,
            referenced_fact_ids=raw.referenced_fact_ids,
            is_external=raw.is_external,
            tags=tags or []
        )

    def __str__(self) -> str:
        snippet = f' | Origin: "{self.origin}"' if self.origin else ""
        tags_str = ", ".join([t.value for t in self.tags])
        return f"{self.id}. [{tags_str}] {self.fact}{snippet}"

    def __repr__(self) -> str:
        return f"AtomicFact(id={self.id}, tags={[t.value for t in self.tags]})"
