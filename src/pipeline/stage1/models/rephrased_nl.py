from pydantic import BaseModel, Field
from typing import List, Optional, Any, Union, Literal

class TechnicalTerm(BaseModel):
    term: str = Field(description="The technical or domain-specific term identified.")
    definition: str = Field(description="The precise definition or context found via research.")
    citation: Optional[str] = Field(default=None, description="A URL link to the source of the definition, if available.")

class ModelingConstraint(BaseModel):
    name: str = Field(description="Short name for the constraint or distribution rule.")
    description: str = Field(description="Detailed formal description of the constraint.")

class Interpretation(BaseModel):
    meaning: str = Field(description="One specific possible meaning of the ambiguous text.")
    schema_impact: str = Field(description="How this meaning would change the tables or columns.")

class Ambiguity(BaseModel):
    original_text: str = Field(description="The exact snippet in the original text that is ambiguous.")
    potential_interpretations: List[Interpretation] = Field(description="List of possible interpretations and their impacts.")
    status: str = Field(description="Reported as 'Unresolved' to preserve accuracy.")

class AtomicFact(BaseModel):
    id: int
    fact: str = Field(description="A single declarative sentence expressing exactly one fact.")
    tag: Literal["SCHEMATIC", "DISTRIBUTIONAL", "CONSTRAINT", "ANALYTICAL"] = Field(description="Categorical tag for the fact: SCHEMATIC (tables/columns), DISTRIBUTIONAL (stats/patterns), CONSTRAINT (logic/rules), ANALYTICAL (queries/goals).")

class RephrasedOutput(BaseModel):
    domain: str = Field(description="The identified industry or technical sector (e.g., 'Autonomous Vehicle Telemetry').")
    analytical_goal: str = Field(description="The primary analytical purpose of the dataset (e.g., 'Sensor fusion for safety analytics').")
    rephrased_text: List[AtomicFact] = Field(description="An exhaustive list of clarifying, structured atomic facts.")
    research_notes: List[TechnicalTerm] = Field(description="Formal definitions of domain terms.")
    modeling_constraints: List[ModelingConstraint] = Field(description="Explicitly identified data constraints/laws/distributions.")
    ambiguities: List[Ambiguity] = Field(description="Unresolved ambiguities found in original text.")

class Issue(BaseModel):
    fact_id: Optional[int] = None
    description: str
    severity: Literal["low", "medium", "high"]

class IntegrityReport(BaseModel):
    is_safe: bool = Field(description="True if no actual information loss or hallucination found.")
    missing_information: List[Issue] = Field(description="Facts present in original but missing/altered in rephrased.")
    introduced_information: List[Issue] = Field(description="New information/inferences introduced in rephrased.")
    changed_constraints: List[Issue] = Field(description="Numeric values or statistical constraints that changed.")

class EnrichedNL(BaseModel):
    rephrased_output: RephrasedOutput = Field(description="The structured output from the technical editor.")
    original_facts: List[AtomicFact] = Field(description="Atomic facts extracted from original text.")
    integrity_report: Optional[IntegrityReport] = None
