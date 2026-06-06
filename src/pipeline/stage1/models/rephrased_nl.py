from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any, Union
from .raw_fact import RawFact
from .atomic_fact import AtomicFact, FactTag
from .technical_term import TechnicalTerm
from .integrity_report import IntegrityReport

class Interpretation(BaseModel):
    meaning: str = Field(description="One specific possible meaning of the ambiguous text.")
    schema_impact: str = Field(description="How this meaning would change the tables or columns.")

class Ambiguity(BaseModel):
    original_text: str = Field(description="The exact snippet in the original text that is ambiguous.")
    potential_interpretations: List[Interpretation] = Field(description="List of possible interpretations and their impacts.")
    status: str = Field(description="Reported as 'Unresolved' to preserve accuracy.")

class RephrasedOutput(BaseModel):
    domain: Optional[str] = Field(default="Unknown", description="The identified industry or technical sector.")
    analytical_goal: Optional[str] = Field(default="General Purpose", description="The primary analytical purpose of the dataset.")
    facts: List[RawFact] = Field(description="An exhaustive list of extracted raw facts.")
    definitions: List[TechnicalTerm] = Field(description="List of technical terms defined in the facts.")
    ambiguities: List[Ambiguity] = Field(default_factory=list, description="List of ambiguous terms or phrases found in the text.")

    @field_validator('domain', 'analytical_goal', mode='before')
    @classmethod
    def ensure_string(cls, v: Any) -> Any:
        if v is None:
            return "Unknown"
        return v

class EnrichedNL(BaseModel):
    extracted_output: RephrasedOutput = Field(description="The structured facts extracted by the agent.")
    integrity_report: Optional[Union[str, IntegrityReport]] = Field(default=None, description="Detailed validation of the extraction accuracy.")

    def __str__(self) -> str:
        report_str = str(self.integrity_report) if self.integrity_report else "No integrity report."
        facts_count = len(self.extracted_output.facts)
        return f"EnrichedNL(Report: {report_str}, Facts: {facts_count})"

class FactList(BaseModel):
    facts: List[RawFact] = Field(description="A sequential list of extracted raw facts.")

class TaggedFact(BaseModel):
    id: int = Field(description="Original fact identifier.")
    tags: List[str] = Field(description="List of applied semantic tags (e.g., RELATIONAL, LOGICAL).")

class TaggerOutput(BaseModel):
    facts: List[TaggedFact] = Field(description="List of all fact mappings to their newly identified tags.")

def convert_to_atomic(facts: List[RawFact], tag_results: List[TaggedFact]) -> List[AtomicFact]:
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
