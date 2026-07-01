from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class GapDimension(str, Enum):
    ENTITY = "entity"
    ATTRIBUTE = "attribute"
    RELATIONSHIP = "relationship"
    CARDINALITY = "cardinality"
    CONSTRAINT = "constraint"
    VALUE_DOMAIN = "value_domain"


class GapSeverity(str, Enum):
    BLOCKING = "blocking"
    MAJOR = "major"
    MINOR = "minor"


class SpecGap(BaseModel):
    id: int = Field(description="Unique identifier stable within one CoverageReport")
    dimension: GapDimension = Field(description="The dimension of the missing information")
    description: str = Field(description="Concrete description of WHAT is missing")
    severity: GapSeverity = Field(description="Severity of the gap")
    search_query: str = Field(description="A targeted DDG query to fill THIS gap")


class CoverageReport(BaseModel):
    detected_entities: List[str] = Field(description="Entities detected in the domain")
    detected_relationships: List[str] = Field(description="Relationships detected in the domain")
    gaps: List[SpecGap] = Field(default_factory=list, description="List of coverage gaps")



    @property
    def open_severities(self) -> set:
        return {GapSeverity.BLOCKING, GapSeverity.MAJOR}

    @property
    def is_underspecified(self) -> bool:
        return any(g.severity in self.open_severities for g in self.gaps)

    def gaps_for_enrichment(self) -> List[SpecGap]:
        if not self.is_underspecified:
            return []
        order = {GapSeverity.BLOCKING: 0, GapSeverity.MAJOR: 1, GapSeverity.MINOR: 2}
        return sorted(self.gaps, key=lambda g: order[g.severity])
