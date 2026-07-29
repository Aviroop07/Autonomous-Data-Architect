from enum import Enum
from pydantic import Field
from typing import List, Optional
from .raw_fact import RawFact


class FactTag(str, Enum):
    STRUCTURAL = "STRUCTURAL"
    LOGICAL = "LOGICAL"
    STATISTICAL = "STATISTICAL"
    METADATA = "METADATA"


class AtomicFact(RawFact):
    tags: List[FactTag] = Field(
        default_factory=list, description="Categorical tags for the fact."
    )
    segment_text: str = Field(
        default="", description="The text of the segment this fact belongs to."
    )
    start_char: int = Field(default=-1, description="Start offset.")
    end_char: int = Field(default=-1, description="End offset.")

    @staticmethod
    def from_raw(
        raw: RawFact,
        tags: Optional[List[FactTag]] = None,
        segment_text: str = "",
        start_char: int = -1,
        end_char: int = -1,
    ) -> "AtomicFact":
        # Spread the base model rather than enumerating its fields: an explicit
        # field list silently dropped addresses_gap and evidence_refs here, so
        # every enrichment fact lost its gap attribution and evidence grounding
        # at the Stage 1 output boundary. Spreading means a new RawFact field
        # can never go missing again without a type error.
        return AtomicFact(
            **raw.model_dump(),
            tags=tags or [],
            segment_text=segment_text,
            start_char=start_char,
            end_char=end_char,
        )

    def __str__(self) -> str:
        snippet = (
            f' | Segment: "{self.segment_text[:30]}..."' if self.segment_text else ""
        )
        tags_str = ", ".join([t.value for t in self.tags])
        return f"{self.id}. [{tags_str}] {self.fact}{snippet}"

    def __repr__(self) -> str:
        return f"AtomicFact(id={self.id}, tags={[t.value for t in self.tags]})"
