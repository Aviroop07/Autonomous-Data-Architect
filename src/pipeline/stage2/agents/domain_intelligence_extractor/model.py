from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class EntityStandard(BaseModel):
    standard_name: str = Field(description="The industry-standard name for this entity (e.g., 'STUDENT' instead of 'FRESHMAN').")
    common_attributes: List[str] = Field(description="Typical columns/attributes found in this entity.")
    primary_key_preference: str = Field(description="Whether industry standards prefer a Natural PK (e.g., SKU, Passport#) or a Surrogate ID.")

class RelationshipCardinality(BaseModel):
    entity_a: str = Field(description="The first entity in the relationship (e.g. 'Orders').")
    entity_b: str = Field(description="The second entity in the relationship (e.g. 'Customers').")
    typical_cardinality: str = Field(description="The standard relationship type: '1:1', '1:M', or 'M:N'.")
    context: str = Field(description="Brief explanation of why this is the standard (e.g., 'A student can enroll in multiple courses').")

class DomainIntelligenceReport(BaseModel):
    domain: str = Field(description="The industry or sector studied (e.g. 'Fintech', 'Retail').")
    entities: List[EntityStandard] = Field(description="List of core entities and their industry-standard definitions.")
    cardinalities: List[RelationshipCardinality] = Field(description="List of typical relationship types between key entities.")
    common_hierarchies: List[str] = Field(description="Description of common subtype/supertype patterns (e.g., 'Staff can be Professors or Admins').")
    research_summary: str = Field(description="Concise summary of the domain modeling research.")
