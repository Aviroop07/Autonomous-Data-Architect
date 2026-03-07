from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from src.pipeline.stage2.models.schema import Table, ForeignKey

class GlobalSchema(BaseModel):
    version: str = "1.0"
    tables: List[Table] = Field(default_factory=list)
    relationships: List[ForeignKey] = Field(default_factory=list)

    def get_table_map(self) -> Dict[str, Table]:
        return {t.name: t for t in self.tables}

    def __str__(self) -> str:
        lines = ["=== GLOBAL SCHEMA ==="]
        for table in self.tables:
            lines.append(str(table))
        if self.relationships:
            lines.append("\nGLOBAL RELATIONSHIPS:")
            for rel in self.relationships:
                lines.append(f"    {rel}")
        return "\n".join(lines)
