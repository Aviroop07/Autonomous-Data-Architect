from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Tuple, Union
from .raw_fact import RawFact, Segment
from .atomic_fact import AtomicFact, FactTag
from .integrity_report import IntegrityReport
from src.util.orchestration.loop_types import LoopOutputModel


class RephrasedOutput(LoopOutputModel):
    domain: Optional[str] = Field(
        default="Unknown", description="The identified industry or technical sector."
    )
    analytical_goal: Optional[str] = Field(
        default="General Purpose",
        description="The primary analytical purpose of the dataset.",
    )
    segments: List[Segment] = Field(
        description="An exhaustive list of text segments and their extracted facts."
    )

    @field_validator("domain", "analytical_goal", mode="before")
    @classmethod
    def ensure_string(cls, v: object) -> object:
        if v is None:
            return "Unknown"
        return v

    def get_errors(self) -> list[str]:
        return []

    @property
    def flat_facts(self) -> List[RawFact]:
        return [f for s in self.segments for f in s.facts]


class EnrichedNL(BaseModel):
    extracted_output: RephrasedOutput = Field(
        description="The structured facts extracted by the agent."
    )
    integrity_report: Optional[Union[str, IntegrityReport]] = Field(
        default=None, description="Detailed validation of the extraction accuracy."
    )

    def __str__(self) -> str:
        report_str = (
            str(self.integrity_report)
            if self.integrity_report
            else "No integrity report."
        )
        facts_count = sum(len(s.facts) for s in self.extracted_output.segments)
        return f"EnrichedNL(Report: {report_str}, Facts: {facts_count})"


class FactList(LoopOutputModel):
    facts: List[RawFact] = Field(
        description="A sequential list of extracted raw facts."
    )

    def get_errors(self) -> list[str]:
        return []


class TaggedFact(BaseModel):
    id: int = Field(description="Original fact identifier.")
    tags: List[str] = Field(
        description="List of applied semantic tags (e.g., RELATIONAL, LOGICAL)."
    )


class TaggerOutput(BaseModel):
    facts: List[TaggedFact] = Field(
        description="List of all fact mappings to their newly identified tags."
    )


def convert_to_atomic(
    facts: List[RawFact],
    tag_results: List[TaggedFact],
    segment_lookup: Optional[Dict[int, Tuple[str, int, int]]] = None,
) -> List[AtomicFact]:
    """Converts RawFacts to AtomicFacts using tagger output for tag assignment.

    segment_lookup maps fact id -> (segment_text, start_char, end_char) so each atomic
    fact carries the source segment it was extracted from. Without it, the graph chunker
    sees no segments and degrades to a single chunk. External/enrichment facts have no
    segment and are left with the default empty segment (handled as standalone downstream).
    """
    segment_lookup = segment_lookup or {}
    result = []
    for raw in facts:
        tag_result = next((t for t in tag_results if str(t.id) == str(raw.id)), None)
        tags = []
        if tag_result:
            for t_str in tag_result.tags:
                try:
                    tags.append(FactTag(t_str.upper()))
                except ValueError:
                    continue
        if not tags:
            tags = [FactTag.STRUCTURAL]
        seg_text, start_char, end_char = segment_lookup.get(raw.id, ("", -1, -1))
        result.append(
            AtomicFact.from_raw(
                raw,
                tags,
                segment_text=seg_text,
                start_char=start_char,
                end_char=end_char,
            )
        )
    return result
