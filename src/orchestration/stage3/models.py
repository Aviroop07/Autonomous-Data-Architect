from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from src.pipeline.stage3.models import TableConstraintManifest, AlgebraicManifest

class RetryStep(BaseModel):
    attempt: int
    feedback: Optional[str] = None
    error: Optional[str] = None
    token_usage: int = 0

class RawSQLRule(BaseModel):
    on: str
    condition: str
    fact_references: List[int] = Field(default_factory=list)

class HealingAttempt(BaseModel):
    attempt: int
    success: bool
    errors: List[str] = Field(default_factory=list)

class ShardMetadata(BaseModel):
    shard_index: int
    table_names: List[str]
    allocated_fact_ids: List[int] = Field(default_factory=list)
    manifests: Dict[str, TableConstraintManifest] = Field(default_factory=dict)
    raw_sql_rules: List[RawSQLRule] = Field(default_factory=list, description="Raw LLM outputs (on/condition) before algebraic parsing.")
    retry_history: List[RetryStep] = Field(default_factory=list)
    token_usage: int = 0
    validation_logs: List[str] = Field(default_factory=list)

class Output(BaseModel):
    global_manifest: AlgebraicManifest
    shard_results: List[ShardMetadata]
    total_tokens: int = 0
    execution_success: bool = True
    healing_history: List[HealingAttempt] = Field(default_factory=list, description="History of the global healing loop attempts.")
