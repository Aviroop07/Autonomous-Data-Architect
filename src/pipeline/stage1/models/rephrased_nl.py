from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union
from .raw_fact import RawFact
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
    facts: List[RawFact] = Field(
        description="An exhaustive list of extracted raw facts."
    )

    @field_validator("domain", "analytical_goal", mode="before")
    @classmethod
    def ensure_string(cls, v: object) -> object:
        if v is None:
            return "Unknown"
        return v

    def get_errors(self) -> list[str]:
        return []


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
        facts_count = len(self.extracted_output.facts)
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
    facts: List[RawFact], tag_results: List[TaggedFact]
) -> List[AtomicFact]:
    """Converts RawFacts to AtomicFacts using tagger output for tag assignment."""
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
        result.append(AtomicFact.from_raw(raw, tags))
    return result
