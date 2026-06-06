from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Tuple, Any

class TableParameters(BaseModel):
    table_name: str = Field(description="The formal table name.")
    n_seeds: Optional[int] = Field(None, description="Absolute count of seed rows (for anchor tables).")
    avg_fanout: Optional[float] = Field(None, description="Average child-to-parent ratio (for dependent tables).")
    sparsity: Dict[str, float] = Field(default_factory=dict, description="Null probability (0.0 to 1.0) per column.")

    @field_validator('n_seeds', 'avg_fanout')
    @classmethod
    def validate_positive(cls, v: Any) -> Any:
        # Pydantic may pass None here if optional
        if v is not None and v <= 0:
            raise ValueError("Scale values must be positive.")
        return v

class ParameterManifest(BaseModel):
    parameters: List[TableParameters] = Field(description="List of parameter definitions per table.")
    reasoning: str = Field(description="Explanation for the derived parameters based on atomic facts.")

class SynthesisResult(BaseModel):
    generated_code: str = Field(description="The complete Python synthesis script.")
    token_usage: int = Field(default=0, description="Total tokens consumed across all agent calls.")
    success: bool = Field(default=True, description="Overall success status of the pipeline.")
    error_message: Optional[str] = Field(default=None, description="Detailed error trace if the pipeline failed.")
    verification_status: Optional[str] = Field(default=None, description="The final audit result: PASSED, FAILED, or WARNING.")
    verification_logs: List[str] = Field(default_factory=list, description="The reasoning logs from the deterministic validator.")
    column_coverage: float = Field(default=0.0, description="The percentage of requested schema columns successfully materialized.")
