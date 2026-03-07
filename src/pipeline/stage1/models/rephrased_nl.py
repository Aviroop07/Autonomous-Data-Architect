from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any, Union

class TechnicalTerm(BaseModel):
    term: str = Field(description="The technical or domain-specific term identified.")
    definition: str = Field(description="The precise definition or context found via research.")
    citation: Optional[str] = Field(default=None, description="A URL link to the source of the definition, if available.")

class ModelingConstraint(BaseModel):
    name: str = Field(description="Short name for the constraint or distribution rule.")
    description: str = Field(description="Detailed formal description of the constraint.")

class Ambiguity(BaseModel):
    original_text: str = Field(description="The exact snippet in the original text that is ambiguous.")
    potential_interpretations: str = Field(description="List or description of possible interpretations.")
    status: str = Field(description="Reported as 'Unresolved' to preserve accuracy.")

class AtomicFact(BaseModel):
    id: int
    fact: str = Field(description="A single declarative sentence expressing exactly one fact.")

class IntegrityReport(BaseModel):
    missing_information: List[Union[str, dict]] = Field(default=[], description="Facts present in original but lost in rephrased.")
    introduced_information: List[Union[str, dict]] = Field(default=[], description="New information inferred by the model but not in original.")
    changed_constraints: List[Union[str, dict]] = Field(default=[], description="Numeric or logic changes identified.")
    is_safe: bool = Field(description="True if no information loss or hallucination detected.")

    @field_validator("missing_information", "introduced_information", "changed_constraints", mode="after")
    @classmethod
    def convert_dicts_to_strings(cls, v: List[Any]) -> List[str]:
        result = []
        for item in v:
            if isinstance(item, dict):
                # Try to extract 'item', 'fact', or similar keys, or just use the first value
                val = item.get("item") or item.get("fact") or list(item.values())[0] if item else ""
                result.append(str(val))
            else:
                result.append(str(item))
        return result

class RephrasedOutput(BaseModel):
    domain: str = Field(description="The identified industry or technical sector (e.g., 'Autonomous Vehicle Telemetry').")
    analytical_goal: str = Field(description="The primary analytical purpose of the dataset (e.g., 'Sensor fusion for safety analytics').")
    rephrased_text: str = Field(description="The clarified, structured, one-fact-per-sentence description.")
    research_notes: List[TechnicalTerm] = Field(description="Formal definitions of domain terms.")
    modeling_constraints: List[ModelingConstraint] = Field(description="Explicitly identified data constraints/laws/distributions.")
    ambiguities: List[Ambiguity] = Field(description="Unresolved ambiguities found in original text.")

class EnrichedNL(BaseModel):
    rephrased_output: RephrasedOutput = Field(description="The structured output from the technical editor.")
    original_facts: List[AtomicFact] = Field(description="Atomic facts extracted from original text.")
    rephrased_facts: List[AtomicFact] = Field(description="Atomic facts extracted from rephrased text.")
    integrity_report: Optional[IntegrityReport] = None
