from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional


class ExternalFactKind(str, Enum):
    TECHNICAL_DEFINITION = "TECHNICAL_DEFINITION"
    DOMAIN_MODELING_HINT = "DOMAIN_MODELING_HINT"
    DOMAIN_CONSTRAINT_HINT = "DOMAIN_CONSTRAINT_HINT"
    ARCHITECTURE_PATTERN = "ARCHITECTURE_PATTERN"
    DOMAIN_PATTERN = "DOMAIN_PATTERN"


class RawFact(BaseModel):
    id: int = Field(description="Unique identifier for the fact.")
    fact: str = Field(
        description="A rich, standalone declarative sentence expressing exactly one fact."
    )
    origin: str = Field(
        default="",
        description="The exact verbatim substring from the source text. Empty for generated facts.",
    )
    referenced_fact_ids: List[int] = Field(
        default_factory=list,
        description="IDs of facts this fact references (for external/context facts).",
    )
    is_external: bool = Field(
        default=False,
        description="True if this is a generated external fact (definition, context).",
    )
    external_kind: Optional[ExternalFactKind] = Field(
        default=None,
        description="Quality category for accepted external context facts.",
    )
    novelty_reason: Optional[str] = Field(
        default=None,
        description="Why this external fact adds non-redundant domain-specific context.",
    )

    def __str__(self) -> str:
        origin_str = f' | Origin: "{self.origin}"' if self.origin else ""
        return f"{self.id}. {self.fact}{origin_str}"

    def __repr__(self) -> str:
        return f"RawFact(id={self.id}, external={self.is_external})"
