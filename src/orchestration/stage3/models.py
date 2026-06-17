from pydantic import BaseModel, Field
from typing import List, Optional
from src.pipeline.stage3.models import TableConstraintManifest, AlgebraicManifest
from src.pipeline.stage3.models.sql_models import BinaryOperand

class RetryStep(BaseModel):
    attempt: int
    feedback: Optional[str] = None
    error: Optional[str] = None
    token_usage: int = 0

class RawSQLRule(BaseModel):
    state_query: str
    left_operand: str
    operator: str
    right_operand: Optional[BinaryOperand] = None
    fact_references: List[int] = Field(default_factory=list)

class HealingAttempt(BaseModel):
    attempt: int
    success: bool
    errors: List[str] = Field(default_factory=list)

class ShardMetadata(BaseModel):
    shard_index: int
    table_names: List[str]
    allocated_fact_ids: List[int] = Field(default_factory=list)
    manifests: List[TableConstraintManifest] = Field(default_factory=list)
    raw_sql_rules: List[RawSQLRule] = Field(default_factory=list, description="Raw LLM state-table predicates before global validation.")
    retry_history: List[RetryStep] = Field(default_factory=list)
    token_usage: int = 0
    validation_logs: List[str] = Field(default_factory=list)

class Output(BaseModel):
    global_manifest: AlgebraicManifest
    shard_results: List[ShardMetadata]
    total_tokens: int = 0
    execution_success: bool = True
    healing_history: List[HealingAttempt] = Field(default_factory=list, description="History of the global healing loop attempts.")
