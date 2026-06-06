from pydantic import BaseModel, Field
from typing import Optional

class TechnicalTerm(BaseModel):
    term: str = Field(description="The technical or domain-specific term identified.")
    definition: str = Field(description="The precise definition or context found via research.")
    citation: Optional[str] = Field(default=None, description="A URL link to the source of the definition, if available.")

    def __str__(self) -> str:
        cit = f" (Source: {self.citation})" if self.citation else ""
        return f"{self.term}: {self.definition}{cit}"

    def __repr__(self) -> str:
        return f"TechnicalTerm(term={self.term})"
